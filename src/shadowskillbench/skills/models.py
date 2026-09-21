from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, ClassVar, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator
from pydantic.config import ExtraValues
from pydantic_core import core_schema

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

_ORDERED_STEP_FIELDS = (
    "step_id",
    "action_intent",
    "tool_name",
    "argument_bindings",
    "preconditions",
    "optional",
    "evidence_refs",
)
_SKILL_IR_FIELDS = (
    "skill_id",
    "schema_version",
    "domain",
    "objective",
    "applicability",
    "required_inputs",
    "preconditions",
    "ordered_steps",
    "decision_hints",
    "verification_steps",
    "stop_conditions",
    "escalation_hints",
    "source_trace_ids",
    "instruction_provenance",
    "compiler_manifest_ref",
)
_SKILL_ID = re.compile(r"^skill_[a-z0-9_-]+$")
_TRACE_ID = re.compile(r"^trace_[a-z0-9_-]+$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOMAINS = frozenset({"access_provisioning", "financial_adjustments"})
_FORBIDDEN_KEYS = frozenset(
    {
        "worker_policy_id",
        "semantic_class",
        "contamination_ratio",
        "hidden_benchmark_metadata_ref",
        "hidden_class_manifest_hash",
        "ordered_trace_hashes",
        "source_manifest",
        "hidden_class_manifest",
    }
)
_FORBIDDEN_VALUES = frozenset(
    {
        "access_compliant_v1",
        "access_workaround_v1",
        "finance_compliant_v1",
        "finance_workaround_v1",
        "compliant",
        "prohibited_workaround",
    }
)
_MAX_IDENTIFIER_LENGTH = 128
_MAX_TEXT_CODEPOINTS = 4_096
_MAX_OTHER_ARRAY_ITEMS = 1_024
_MAX_STEPS = 128
_MAX_TRACE_IDS = 12
_MAX_JSON_DEPTH = 32
_MAX_JSON_VALUES = 100_000
_MAX_JSON_TEXT_BYTES = 1_000_000
_MAX_INT_BITS = 256


def _exact_dict(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact built-in object")
    return cast(dict[str, object], value)


def _exact_list(value: object, *, field: str, maximum: int) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{field} must be an exact built-in JSON array")
    raw = cast(list[object], value)
    if len(raw) > maximum:
        raise ValueError(f"{field} exceeds its item budget")
    return raw


def _require_exact_fields(
    value: object, *, fields: tuple[str, ...], field: str
) -> dict[str, object]:
    raw = _exact_dict(value, field=field)
    if len(raw) != len(fields):
        raise ValueError(f"{field} has missing or extra fields")
    if any(type(key) is not str for key in raw):
        raise ValueError(f"{field} keys must be exact strings")
    if any(name not in raw for name in fields):
        raise ValueError(f"{field} has missing or extra fields")
    return {name: raw[name] for name in fields}


def _model_data(
    value: object, *, expected: type[BaseModel], fields: tuple[str, ...]
) -> dict[str, object]:
    if type(value) is not expected:
        raise ValueError(f"value must be an exact {expected.__name__}")
    try:
        extra = object.__getattribute__(value, "__pydantic_extra__")
        private = object.__getattribute__(value, "__pydantic_private__")
        raw = object.__getattribute__(value, "__dict__")
        fields_set = object.__getattribute__(value, "__pydantic_fields_set__")
    except AttributeError as error:
        raise ValueError(f"{expected.__name__} has missing Pydantic state") from error
    if extra is not None or private is not None:
        raise ValueError(f"{expected.__name__} has forbidden private or extra state")
    if type(raw) is not dict or len(raw) != len(fields):
        raise ValueError(f"{expected.__name__} has an invalid key tree")
    if any(type(key) is not str for key in raw) or any(name not in raw for name in fields):
        raise ValueError(f"{expected.__name__} has an invalid key tree")
    if (
        type(fields_set) is not set
        or len(fields_set) != len(fields)
        or any(type(name) is not str for name in fields_set)
        or any(name not in fields_set for name in fields)
    ):
        raise ValueError(f"{expected.__name__} has an invalid field set")
    return {name: raw[name] for name in fields}


def _text(value: object, *, field: str, markdown: bool = True) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact string")
    text = cast(str, value)
    if len(text) > _MAX_TEXT_CODEPOINTS:
        raise ValueError(f"{field} must be nonblank, NUL-free, and bounded")
    if not text.strip() or "\x00" in text:
        raise ValueError(f"{field} must be nonblank, NUL-free, and bounded")
    if markdown and ("\r" in text or "\n" in text):
        raise ValueError(f"{field} must be a single Markdown line")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field} must be UTF-8 encodable") from error
    if text in _FORBIDDEN_VALUES:
        raise ValueError(f"{field} contains a forbidden treatment marker")
    return text


def _identifier(value: object, *, field: str, pattern: re.Pattern[str] = _IDENTIFIER) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact string")
    text = cast(str, value)
    if (
        len(text) > _MAX_IDENTIFIER_LENGTH
        or pattern.fullmatch(text) is None
        or text in _FORBIDDEN_VALUES
    ):
        raise ValueError(f"{field} must be a bounded identifier")
    return text


def _array_of_text(
    value: object, *, field: str, maximum: int = _MAX_OTHER_ARRAY_ITEMS
) -> tuple[str, ...]:
    raw = _exact_list(value, field=field, maximum=maximum)
    return tuple(_text(item, field=f"{field} item") for item in raw)


class _RawContainerState:
    def __init__(self, root: dict[str, object]) -> None:
        self.seen = {id(root)}
        self.keepalive: list[object] = [root]
        self.json_values = 0
        self.json_text_bytes = 0

    def add(self, value: object, *, field: str) -> None:
        if type(value) not in {dict, list}:
            raise ValueError(f"{field} must be an exact built-in JSON container")
        container_id = id(value)
        if container_id in self.seen:
            raise ValueError(f"{field} containers must not be cyclic or aliased")
        self.seen.add(container_id)
        self.keepalive.append(value)


def _scan_json_root(value: object, *, field: str, state: _RawContainerState) -> None:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact built-in JSON object")
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        state.json_values += 1
        current_type = type(current)
        if current is None or current_type is bool:
            continue
        if current_type is int:
            if abs(cast(int, current)).bit_length() > _MAX_INT_BITS:
                raise ValueError(f"{field} integer exceeds its magnitude budget")
            continue
        if current_type is float:
            if not math.isfinite(cast(float, current)):
                raise ValueError(f"{field} floats must be finite")
            continue
        if current_type is str:
            text = cast(str, current)
            if len(text) > _MAX_TEXT_CODEPOINTS or "\x00" in text or text in _FORBIDDEN_VALUES:
                raise ValueError(f"{field} JSON string is invalid")
            try:
                state.json_text_bytes += len(text.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError(f"{field} JSON strings must be UTF-8 encodable") from error
            if state.json_text_bytes > _MAX_JSON_TEXT_BYTES:
                raise ValueError("SkillIR JSON text exceeds the aggregate budget")
            continue
        if current_type not in {dict, list}:
            raise ValueError(f"{field} must contain exact built-in JSON values")
        if depth > _MAX_JSON_DEPTH:
            raise ValueError(f"{field} exceeds its JSON depth budget")
        state.add(current, field=field)
        if current_type is list:
            children = cast(list[object], current)
            if len(children) > _MAX_JSON_VALUES - state.json_values - len(stack):
                raise ValueError("SkillIR JSON values exceed the aggregate budget")
            stack.extend((item, depth + 1) for item in reversed(children))
            continue
        children = cast(dict[object, object], current)
        if len(children) > _MAX_JSON_VALUES - state.json_values - len(stack):
            raise ValueError("SkillIR JSON values exceed the aggregate budget")
        for key, item in children.items():
            if type(key) is not str or len(key) > _MAX_IDENTIFIER_LENGTH or key in _FORBIDDEN_KEYS:
                raise ValueError(f"{field} JSON key is forbidden or too long")
            try:
                state.json_text_bytes += len(key.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError(f"{field} JSON keys must be UTF-8 encodable") from error
            if state.json_text_bytes > _MAX_JSON_TEXT_BYTES:
                raise ValueError("SkillIR JSON text exceeds the aggregate budget")
            stack.append((item, depth + 1))


def _preflight_step(value: object, *, state: _RawContainerState | None = None) -> dict[str, object]:
    raw = _require_exact_fields(value, fields=_ORDERED_STEP_FIELDS, field="OrderedStep")
    owned_state = state or _RawContainerState(raw)
    if state is not None:
        state.add(raw, field="OrderedStep")
    for field in ("preconditions", "evidence_refs"):
        raw_array = _exact_list(raw[field], field=field, maximum=_MAX_OTHER_ARRAY_ITEMS)
        owned_state.add(raw_array, field=field)
    _scan_json_root(raw["argument_bindings"], field="argument_bindings", state=owned_state)
    return raw


def _preflight_skill(value: object) -> dict[str, object]:
    raw = _require_exact_fields(value, fields=_SKILL_IR_FIELDS, field="SkillIR")
    state = _RawContainerState(raw)
    for field in (
        "applicability",
        "required_inputs",
        "preconditions",
        "decision_hints",
        "verification_steps",
        "stop_conditions",
        "escalation_hints",
        "source_trace_ids",
    ):
        raw_array = _exact_list(raw[field], field=field, maximum=_MAX_OTHER_ARRAY_ITEMS)
        state.add(raw_array, field=field)
    steps = _exact_list(raw["ordered_steps"], field="ordered_steps", maximum=_MAX_STEPS)
    if not steps:
        raise ValueError("ordered_steps must not be empty")
    state.add(steps, field="ordered_steps")
    for step in steps:
        _preflight_step(step, state=state)
    _scan_json_root(raw["instruction_provenance"], field="instruction_provenance", state=state)
    return raw


def _preflight_model_step(value: OrderedStep) -> None:
    raw = _model_data(value, expected=OrderedStep, fields=_ORDERED_STEP_FIELDS)
    state = _RawContainerState(raw)
    _scan_json_root(raw["argument_bindings"], field="argument_bindings", state=state)


def _preflight_model_skill(value: SkillIR) -> None:
    raw = _model_data(value, expected=SkillIR, fields=_SKILL_IR_FIELDS)
    steps = raw["ordered_steps"]
    if type(steps) is not tuple:
        raise ValueError("SkillIR ordered_steps must be an exact built-in tuple")
    state = _RawContainerState(raw)
    for step in steps:
        step_raw = _model_data(step, expected=OrderedStep, fields=_ORDERED_STEP_FIELDS)
        _scan_json_root(step_raw["argument_bindings"], field="argument_bindings", state=state)
    _scan_json_root(raw["instruction_provenance"], field="instruction_provenance", state=state)
    return None


def _owned_json_object(value: object, *, field: str) -> JsonObject:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact built-in JSON object")
    stack: list[tuple[object, int, object | None, str | int | None]] = [(value, 0, None, None)]
    seen: set[int] = set()
    values = 0
    text_bytes = 0
    root: JsonObject | None = None
    while stack:
        current, depth, parent, location = stack.pop()
        values += 1
        if values > _MAX_JSON_VALUES:
            raise ValueError(f"{field} exceeds its JSON value budget")
        current_type = type(current)
        copied: JsonValue
        if current is None or current_type is bool:
            copied = cast(JsonValue, current)
        elif current_type is int:
            integer = cast(int, current)
            if abs(integer).bit_length() > _MAX_INT_BITS:
                raise ValueError(f"{field} integer exceeds its magnitude budget")
            copied = integer
        elif current_type is float:
            number = cast(float, current)
            if not math.isfinite(number):
                raise ValueError(f"{field} floats must be finite")
            copied = number
        elif current_type is str:
            text = cast(str, current)
            if len(text) > _MAX_TEXT_CODEPOINTS or "\x00" in text:
                raise ValueError(f"{field} JSON strings are invalid")
            try:
                size = len(text.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError(f"{field} JSON strings must be UTF-8 encodable") from error
            text_bytes += size
            if text_bytes > _MAX_JSON_TEXT_BYTES:
                raise ValueError(f"{field} exceeds its JSON text budget")
            copied = text
        elif current_type in {dict, list}:
            if depth > _MAX_JSON_DEPTH:
                raise ValueError(f"{field} exceeds its JSON depth budget")
            container = cast(dict[object, object] | list[object], current)
            container_id = id(container)
            if container_id in seen:
                raise ValueError(f"{field} JSON containers must not be cyclic or aliased")
            seen.add(container_id)
            if current_type is dict:
                raw_object = cast(dict[object, object], container)
                copied_object: dict[str, JsonValue] = {}
                copied = copied_object
                if len(raw_object) > _MAX_JSON_VALUES - values - len(stack):
                    raise ValueError(f"{field} exceeds its JSON value budget")
                children: list[tuple[object, str]] = []
                for key, item in raw_object.items():
                    if type(key) is not str:
                        raise ValueError(f"{field} JSON keys must be exact strings")
                    if len(key) > _MAX_IDENTIFIER_LENGTH:
                        raise ValueError(f"{field} JSON key is forbidden or too long")
                    try:
                        key_size = len(key.encode("utf-8"))
                    except UnicodeEncodeError as error:
                        raise ValueError(f"{field} JSON keys must be UTF-8 encodable") from error
                    text_bytes += key_size
                    if text_bytes > _MAX_JSON_TEXT_BYTES:
                        raise ValueError(f"{field} exceeds its JSON text budget")
                    children.append((item, key))
                for item, key in reversed(children):
                    stack.append((item, depth + 1, copied_object, key))
            else:
                raw_array = cast(list[object], container)
                copied_array: list[JsonValue] = [None] * len(raw_array)
                copied = copied_array
                if len(raw_array) > _MAX_JSON_VALUES - values - len(stack):
                    raise ValueError(f"{field} exceeds its JSON value budget")
                for index in range(len(raw_array) - 1, -1, -1):
                    stack.append((raw_array[index], depth + 1, copied_array, index))
        else:
            raise ValueError(f"{field} must contain exact built-in JSON values")
        if parent is None:
            if type(copied) is not dict:
                raise ValueError(f"{field} must be a JSON object")
            root = cast(JsonObject, copied)
        elif type(parent) is dict:
            cast(dict[str, JsonValue], parent)[cast(str, location)] = copied
        else:
            cast(list[JsonValue], parent)[cast(int, location)] = copied
    if root is None:
        raise ValueError(f"{field} must be a JSON object")
    try:
        canonical_json_bytes(root)
    except (RecursionError, ValueError) as error:
        raise ValueError(f"{field} is not strict SSB-CJ1 JSON") from error
    return root


class _SkillModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )
    _fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def _require_exact_model_class(cls) -> None:
        if cls.__bases__ != (_SkillModel,):
            raise ValueError("SkillIR models do not admit subclasses")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[BaseModel], handler: object
    ) -> core_schema.CoreSchema:
        schema = cast(Any, handler)(source)
        return core_schema.with_info_before_validator_function(cls._schema_guard, schema)

    @classmethod
    def _schema_guard(cls, value: object, info: ValidationInfo) -> dict[str, object]:
        cls._require_exact_model_class()
        if info.mode != "python":
            raise ValueError(f"{cls.__name__} requires an exact built-in object")
        if type(value) is cls:
            if cls is OrderedStep:
                _preflight_model_step(cast(OrderedStep, value))
                return _step_to_raw(cast(OrderedStep, value))
            if cls is SkillIR:
                _preflight_model_skill(cast(SkillIR, value))
                return _preflight_skill(_model_to_raw(cast(SkillIR, value)))
            raise ValueError(f"{cls.__name__} is not an admitted SkillIR model")
        if cls is OrderedStep:
            return _preflight_step(value)
        if cls is SkillIR:
            return _preflight_skill(value)
        raise ValueError(f"{cls.__name__} is not an admitted SkillIR model")

    @model_validator(mode="before")
    @classmethod
    def _before(cls, value: object, info: ValidationInfo) -> dict[str, object]:
        return cls._schema_guard(value, info)

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        cls._require_exact_model_class()
        if strict is False or extra not in {None, "forbid"} or from_attributes is True:
            raise ValueError(f"{cls.__name__} requires strict exact mapping ingress")
        if type(obj) is cls:
            if cls is OrderedStep:
                _preflight_model_step(cast(OrderedStep, obj))
                obj = _step_to_raw(cast(OrderedStep, obj))
            elif cls is SkillIR:
                _preflight_model_skill(cast(SkillIR, obj))
                obj = _model_to_raw(cast(SkillIR, obj))
            else:
                raise ValueError(f"{cls.__name__} is not an admitted SkillIR model")
        if cls is OrderedStep:
            obj = _preflight_step(obj)
        elif cls is SkillIR:
            obj = _preflight_skill(obj)
        else:
            raise ValueError(f"{cls.__name__} is not an admitted SkillIR model")
        return super().model_validate(
            _require_exact_fields(obj, fields=cls._fields, field=cls.__name__),
            strict=True,
            extra="forbid",
            from_attributes=False,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        del json_data, kwargs
        raise ValueError(f"{cls.__name__} requires an exact built-in object")

    @classmethod
    def model_validate_strings(cls, obj: Any, **kwargs: Any) -> Self:
        del obj, kwargs
        raise ValueError(f"{cls.__name__} requires an exact built-in object")

    @classmethod
    def parse_raw(cls, b: Any, *args: Any, **kwargs: Any) -> Self:
        del b, args, kwargs
        raise ValueError("legacy JSON/string ingress is forbidden")

    @classmethod
    def parse_file(cls, path: Any, *args: Any, **kwargs: Any) -> Self:
        del path, args, kwargs
        raise ValueError("legacy filesystem ingress is forbidden")

    @classmethod
    def parse_obj(cls, obj: Any) -> Self:
        return cls.model_validate(obj)

    @classmethod
    def validate(cls, value: Any) -> Self:
        return cls.model_validate(value)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        if type(deep) is not bool or update is not None and type(update) is not dict:
            raise ValueError("model_copy requires exact built-in inputs")
        if type(self) is OrderedStep:
            _preflight_model_step(cast(OrderedStep, self))
            raw = _step_to_raw(cast(OrderedStep, self))
        elif type(self) is SkillIR:
            _preflight_model_skill(cast(SkillIR, self))
            raw = _model_to_raw(cast(SkillIR, self))
        else:
            raise ValueError("model_copy requires an exact SkillIR model")
        if update is not None:
            copied = _exact_dict(update, field="model_copy update")
            if any(type(key) is not str for key in copied):
                raise ValueError("model_copy update keys must be exact strings")
            if any(name not in self._fields for name in copied):
                raise ValueError("model_copy update has invalid fields")
            raw.update(copied)
        return cast(Self, type(self).model_validate(raw))

    def copy(
        self,
        *,
        include: object = None,
        exclude: object = None,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if include is not None or exclude is not None:
            raise ValueError("legacy copy does not admit include or exclude")
        return self.model_copy(update=update, deep=deep)

    def __copy__(self) -> Self:
        return self.model_copy()

    def __deepcopy__(self, memo: object | None = None) -> Self:
        if memo is not None and type(memo) is not dict:
            raise ValueError("deepcopy memo must be an exact built-in object")
        return self.model_copy(deep=True)


class OrderedStep(_SkillModel):
    _fields: ClassVar[tuple[str, ...]] = _ORDERED_STEP_FIELDS
    step_id: str
    action_intent: str
    tool_name: str
    argument_bindings: JsonObject
    preconditions: tuple[str, ...]
    optional: bool
    evidence_refs: tuple[str, ...]

    @field_validator("step_id", "tool_name", mode="before")
    @classmethod
    def _identifiers(cls, value: object, info: ValidationInfo) -> str:
        return _identifier(value, field=cast(str, info.field_name))

    @field_validator("action_intent", mode="before")
    @classmethod
    def _intent(cls, value: object) -> str:
        return _text(value, field="action_intent")

    @field_validator("argument_bindings", mode="before")
    @classmethod
    def _arguments(cls, value: object) -> JsonObject:
        return _owned_json_object(value, field="argument_bindings")

    @field_validator("preconditions", "evidence_refs", mode="before")
    @classmethod
    def _text_arrays(cls, value: object, info: ValidationInfo) -> tuple[str, ...]:
        return _array_of_text(value, field=cast(str, info.field_name))

    @field_validator("optional", mode="before")
    @classmethod
    def _optional(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("optional must be an exact boolean")
        return cast(bool, value)


class SkillIR(_SkillModel):
    _fields: ClassVar[tuple[str, ...]] = _SKILL_IR_FIELDS
    skill_id: str
    schema_version: str
    domain: str
    objective: str
    applicability: tuple[str, ...]
    required_inputs: tuple[str, ...]
    preconditions: tuple[str, ...]
    ordered_steps: tuple[OrderedStep, ...]
    decision_hints: tuple[str, ...]
    verification_steps: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    escalation_hints: tuple[str, ...]
    source_trace_ids: tuple[str, ...]
    instruction_provenance: JsonObject
    compiler_manifest_ref: str

    @field_validator("skill_id", mode="before")
    @classmethod
    def _skill_id(cls, value: object) -> str:
        return _identifier(value, field="skill_id", pattern=_SKILL_ID)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> str:
        if type(value) is not str or value != "1.0":
            raise ValueError("schema_version must be exactly 1.0")
        return "1.0"

    @field_validator("domain", mode="before")
    @classmethod
    def _domain(cls, value: object) -> str:
        if type(value) is not str or len(value) > _MAX_IDENTIFIER_LENGTH or value not in _DOMAINS:
            raise ValueError("domain is invalid")
        return cast(str, value)

    @field_validator("objective", mode="before")
    @classmethod
    def _objective(cls, value: object) -> str:
        return _text(value, field="objective")

    @field_validator(
        "applicability",
        "required_inputs",
        "preconditions",
        "decision_hints",
        "verification_steps",
        "stop_conditions",
        "escalation_hints",
        mode="before",
    )
    @classmethod
    def _arrays(cls, value: object, info: ValidationInfo) -> tuple[str, ...]:
        return _array_of_text(value, field=cast(str, info.field_name))

    @field_validator("ordered_steps", mode="before")
    @classmethod
    def _steps(cls, value: object) -> tuple[object, ...]:
        raw = _exact_list(value, field="ordered_steps", maximum=_MAX_STEPS)
        if not raw:
            raise ValueError("ordered_steps must not be empty")
        return tuple(_step_to_raw(OrderedStep.model_validate(item)) for item in raw)

    @field_validator("source_trace_ids", mode="before")
    @classmethod
    def _traces(cls, value: object) -> tuple[str, ...]:
        raw = _exact_list(value, field="source_trace_ids", maximum=_MAX_TRACE_IDS)
        if not raw:
            raise ValueError("source_trace_ids must contain 1 through 12 items")
        traces = tuple(
            _identifier(item, field="source_trace_ids item", pattern=_TRACE_ID) for item in raw
        )
        if len(set(traces)) != len(traces):
            raise ValueError("source_trace_ids must be unique")
        return traces

    @field_validator("instruction_provenance", mode="before")
    @classmethod
    def _provenance(cls, value: object) -> JsonObject:
        return _owned_json_object(value, field="instruction_provenance")

    @field_validator("compiler_manifest_ref", mode="before")
    @classmethod
    def _manifest(cls, value: object) -> str:
        if type(value) is not str or len(value) != 71 or _SHA256_REF.fullmatch(value) is None:
            raise ValueError("compiler_manifest_ref must be a lowercase sha256 reference")
        return cast(str, value)

    @model_validator(mode="after")
    def _unique_step_ids(self) -> Self:
        if len({step.step_id for step in self.ordered_steps}) != len(self.ordered_steps):
            raise ValueError("step IDs must be unique")
        return self


def _step_to_raw(value: OrderedStep) -> dict[str, object]:
    raw = _model_data(value, expected=OrderedStep, fields=_ORDERED_STEP_FIELDS)
    for field in ("preconditions", "evidence_refs"):
        if type(raw[field]) is not tuple:
            raise ValueError(f"OrderedStep {field} must be an exact built-in tuple")
    return {
        "step_id": raw["step_id"],
        "action_intent": raw["action_intent"],
        "tool_name": raw["tool_name"],
        "argument_bindings": _owned_json_object(
            raw["argument_bindings"], field="argument_bindings"
        ),
        "preconditions": list(cast(tuple[str, ...], raw["preconditions"])),
        "optional": raw["optional"],
        "evidence_refs": list(cast(tuple[str, ...], raw["evidence_refs"])),
    }


def _model_to_raw(value: SkillIR) -> dict[str, object]:
    raw = _model_data(value, expected=SkillIR, fields=_SKILL_IR_FIELDS)
    steps = raw["ordered_steps"]
    if type(steps) is not tuple:
        raise ValueError("SkillIR ordered_steps must be an exact built-in tuple")
    detached_steps: list[dict[str, object]] = []
    for step in steps:
        detached_steps.append(_step_to_raw(cast(OrderedStep, step)))
    for field in (
        "applicability",
        "required_inputs",
        "preconditions",
        "decision_hints",
        "verification_steps",
        "stop_conditions",
        "escalation_hints",
        "source_trace_ids",
    ):
        if type(raw[field]) is not tuple:
            raise ValueError(f"SkillIR {field} must be an exact built-in tuple")
    return {
        "skill_id": raw["skill_id"],
        "schema_version": raw["schema_version"],
        "domain": raw["domain"],
        "objective": raw["objective"],
        "applicability": list(cast(tuple[str, ...], raw["applicability"])),
        "required_inputs": list(cast(tuple[str, ...], raw["required_inputs"])),
        "preconditions": list(cast(tuple[str, ...], raw["preconditions"])),
        "ordered_steps": detached_steps,
        "decision_hints": list(cast(tuple[str, ...], raw["decision_hints"])),
        "verification_steps": list(cast(tuple[str, ...], raw["verification_steps"])),
        "stop_conditions": list(cast(tuple[str, ...], raw["stop_conditions"])),
        "escalation_hints": list(cast(tuple[str, ...], raw["escalation_hints"])),
        "source_trace_ids": list(cast(tuple[str, ...], raw["source_trace_ids"])),
        "instruction_provenance": _owned_json_object(
            raw["instruction_provenance"], field="instruction_provenance"
        ),
        "compiler_manifest_ref": raw["compiler_manifest_ref"],
    }


def parse_skill_ir(value: object) -> SkillIR:
    if type(value) is SkillIR:
        return SkillIR.model_validate(value)
    return SkillIR.model_validate(value)


def skill_ir_projection(value: object) -> JsonObject:
    parsed = parse_skill_ir(value)
    raw = _model_to_raw(parsed)
    projection = cast(JsonObject, _owned_json_object(raw, field="skill_ir_projection"))
    return projection


def hash_skill_ir(value: object) -> str:
    return sha256_ref(skill_ir_projection(value))


__all__ = [
    "JsonObject",
    "JsonValue",
    "OrderedStep",
    "SkillIR",
    "hash_skill_ir",
    "parse_skill_ir",
    "skill_ir_projection",
]
