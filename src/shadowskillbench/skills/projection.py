from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator
from pydantic.config import ExtraValues
from pydantic_core import core_schema

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.traces.bundles import DemonstrationBundle, SourceBundleManifest
from shadowskillbench.traces.models import ActionTrace, JsonObject, JsonValue, TraceEvent

type CompilerDomain = Literal["access_provisioning", "financial_adjustments"]
type CompilerKind = Literal["observation", "action", "tool_result", "state_delta"]

_PROFILE = "SSB-COMPILER-VIEW1"
_EVENT_FIELDS = ("event_id", "index", "kind", "payload")
_TRACE_FIELDS = (
    "trace_id",
    "domain",
    "task_template_id",
    "worker_role",
    "world_hash",
    "events",
    "terminal_state_hash",
    "local_task_outcome",
)
_INPUT_FIELDS = ("projection_profile", "domain", "traces")
_SOURCE_MANIFEST_FIELDS = (
    "schema_version",
    "profile",
    "bundle_id",
    "domain",
    "contamination_ratio",
    "count",
    "seed",
    "ordered_trace_hashes",
    "hidden_class_manifest_hash",
)
_SOURCE_TRACE_FIELDS = (
    "trace_id",
    "schema_version",
    "domain",
    "task_template_id",
    "worker_policy_id",
    "worker_role",
    "world_hash",
    "narration_mode",
    "events",
    "terminal_state_hash",
    "local_task_outcome",
    "hidden_benchmark_metadata_ref",
)
_FORBIDDEN_KEYS = frozenset(
    {
        "worker_policy_id",
        "semantic_class",
        "contamination_ratio",
        "hidden_benchmark_metadata_ref",
        "source_manifest",
        "hidden_class_manifest",
        "bundle_id",
        "ordered_trace_hashes",
        "hidden_class_manifest_hash",
        "class_slot_order",
        "final_trace_order",
        "bindings",
        "source_slot",
        "fixture_seed",
        "fixture_variant",
        "trace_hash",
        "compliance_label",
        "policy",
        "policy_id",
        "policy_text",
        "authority",
        "authority_record",
        "authority_decision",
        "heldout",
        "held_out",
        "heldout_case",
        "held_out_case",
        "verifier",
        "verifier_outcome",
        "final_verifier",
        "final_verifier_outcome",
        "cup",
        "cup_outcome",
        "scorer",
        "score",
        "scorer_outcome",
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
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRACE_ID = re.compile(r"^trace_[a-z0-9_-]+$")
_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_EVENT_ID = re.compile(r"^event_trace_[a-z0-9_-]+_[0-9]{6}$")
_MAX_ID_LENGTH = 128
_MAX_EVENT_ID_LENGTH = 141
_MAX_EVENTS = 10_000
_MAX_DEPTH = 64
_MAX_VALUES = 100_000
_MAX_TEXT_BYTES = 1_000_000
_MAX_KEY_LENGTH = 128
_MAX_INT_BITS = 256


def _exact_dict(value: object, *, field: str) -> dict[object, object]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact built-in object")
    return cast(dict[object, object], value)


def _exact_fields(
    value: object, *, expected: type[BaseModel], fields: tuple[str, ...]
) -> dict[str, object]:
    if type(value) is not expected:
        raise ValueError(f"value must be an exact {expected.__name__}")
    try:
        data = object.__getattribute__(value, "__dict__")
        field_set = object.__getattribute__(value, "__pydantic_fields_set__")
        extra = object.__getattribute__(value, "__pydantic_extra__")
        private = object.__getattribute__(value, "__pydantic_private__")
    except AttributeError as error:
        raise ValueError(f"{expected.__name__} is incomplete or forged") from error
    if (
        type(data) is not dict
        or type(field_set) is not set
        or len(field_set) != len(fields)
        or any(type(name) is not str for name in field_set)
        or any(name not in fields for name in field_set)
        or extra is not None
        or private is not None
        or len(data) != len(fields)
        or any(type(key) is not str for key in data)
        or any(field not in data for field in fields)
    ):
        raise ValueError(f"{expected.__name__} is incomplete or forged")
    return {field: data[field] for field in fields}


def _raw_fields(value: object, *, fields: tuple[str, ...], field: str) -> dict[str, object]:
    raw = _exact_dict(value, field=field)
    if (
        len(raw) != len(fields)
        or any(type(key) is not str for key in raw)
        or any(name not in raw for name in fields)
    ):
        raise ValueError(f"{field} has missing or extra fields")
    return {name: raw[name] for name in fields}


def _string(value: object, *, field: str, maximum: int | None = None) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact string")
    text = cast(str, value)
    if text in _FORBIDDEN_VALUES:
        raise ValueError(f"{field} contains a forbidden leakage value")
    if maximum is not None and len(text) > maximum:
        raise ValueError(f"{field} exceeds its maximum length")
    return text


def _identifier(
    value: object,
    *,
    field: str,
    pattern: re.Pattern[str] = _PATH_SAFE_ID,
    maximum: int = _MAX_ID_LENGTH,
) -> str:
    text = _string(value, field=field, maximum=maximum)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{field} has invalid grammar")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _string(value, field=field)
    if _SHA256_REF.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase sha256 reference")
    return text


def _index(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("index must be an exact nonnegative integer")
    return cast(int, value)


def _domain(value: object) -> CompilerDomain:
    text = _string(value, field="domain")
    if text not in {"access_provisioning", "financial_adjustments"}:
        raise ValueError("domain is invalid")
    return cast(CompilerDomain, text)


def _event_id(trace_id: str, index: int) -> str:
    return f"event_{trace_id}_{index:06d}"


class _JsonState:
    def __init__(self) -> None:
        self.containers: set[int] = set()
        self.values = 0
        self.text_bytes = 0

    def reset_trace_budget(self) -> None:
        self.values = 0
        self.text_bytes = 0


def _owned_payload(value: object, *, field: str, state: _JsonState | None = None) -> JsonObject:
    owned = state if state is not None else _JsonState()

    def account(text: str) -> None:
        try:
            size = len(text.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValueError(f"{field} contains non-UTF-8 text") from error
        owned.text_bytes += size
        if owned.text_bytes > _MAX_TEXT_BYTES:
            raise ValueError(f"{field} exceeds its UTF-8 budget")

    def copy(item: object, depth: int) -> JsonValue:
        owned.values += 1
        if owned.values > _MAX_VALUES:
            raise ValueError(f"{field} exceeds its JSON value budget")
        item_type = type(item)
        if item is None or item_type is bool:
            return cast(JsonValue, item)
        if item_type is int:
            if abs(cast(int, item)).bit_length() > _MAX_INT_BITS:
                raise ValueError(f"{field} integer exceeds its magnitude budget")
            return cast(JsonValue, item)
        if item_type is float:
            if not math.isfinite(cast(float, item)):
                raise ValueError(f"{field} floats must be finite")
            return cast(JsonValue, item)
        if item_type is str:
            text = cast(str, item)
            account(text)
            if text in _FORBIDDEN_VALUES:
                raise ValueError(f"{field} contains a forbidden leakage value")
            return text
        if item_type not in {dict, list}:
            raise ValueError(f"{field} must contain exact built-in JSON values")
        if depth > _MAX_DEPTH:
            raise ValueError(f"{field} exceeds its JSON depth budget")
        container = cast(dict[object, object] | list[object], item)
        if id(container) in owned.containers:
            raise ValueError(f"{field} JSON containers must not be cyclic or aliased")
        owned.containers.add(id(container))
        if item_type is list:
            return [copy(child, depth + 1) for child in cast(list[object], container)]
        result: dict[str, JsonValue] = {}
        for key, child in cast(dict[object, object], container).items():
            if type(key) is not str:
                raise ValueError(f"{field} JSON keys must be exact strings")
            text_key = cast(str, key)
            if len(text_key) > _MAX_KEY_LENGTH:
                raise ValueError(f"{field} JSON key exceeds its maximum length")
            account(text_key)
            if text_key in _FORBIDDEN_KEYS:
                raise ValueError(f"{field} contains a forbidden leakage key")
            result[text_key] = copy(child, depth + 1)
        return result

    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact built-in JSON object")
    result = copy(value, 0)
    if type(result) is not dict:
        raise ValueError(f"{field} must be a JSON object")
    try:
        canonical_json_bytes(result)
    except (RecursionError, ValueError) as error:
        raise ValueError(f"{field} is not strict SSB-CJ1 JSON") from error
    return cast(JsonObject, result)


def _source_preflight(bundle: object) -> DemonstrationBundle:
    """Reject forged source state before D-021 reconstruction can detach it."""
    if type(bundle) is not DemonstrationBundle:
        raise ValueError("compiler_view requires an exact DemonstrationBundle")
    raw_bundle = _exact_fields(
        bundle, expected=DemonstrationBundle, fields=("source_manifest", "traces")
    )
    source = raw_bundle["source_manifest"]
    _exact_fields(source, expected=SourceBundleManifest, fields=_SOURCE_MANIFEST_FIELDS)
    traces = raw_bundle["traces"]
    if type(traces) is not tuple or len(traces) != 12:
        raise ValueError("DemonstrationBundle traces must be an exact twelve-item tuple")
    state = _JsonState()
    for trace in traces:
        state.reset_trace_budget()
        trace_data = _exact_fields(trace, expected=ActionTrace, fields=_SOURCE_TRACE_FIELDS)
        events = trace_data["events"]
        if type(events) is not tuple:
            raise ValueError("ActionTrace events must be an exact built-in tuple")
        for event in events:
            event_data = _exact_fields(event, expected=TraceEvent, fields=_EVENT_FIELDS)
            _owned_payload(event_data["payload"], field="source event payload", state=state)
    # D-021 owns all source-side integrity links. Invoke it only after the original
    # identity tree has been checked, so a reconstruction cannot erase hostile aliases.
    return DemonstrationBundle.model_validate(bundle)


def _event_raw(value: object, *, state: _JsonState | None = None) -> dict[str, object]:
    if type(value) is CompilerEvent:
        raw = _exact_fields(value, expected=CompilerEvent, fields=_EVENT_FIELDS)
    else:
        raw = _raw_fields(value, fields=_EVENT_FIELDS, field="CompilerEvent")
    return {
        "event_id": _identifier(
            raw["event_id"],
            field="event_id",
            pattern=_EVENT_ID,
            maximum=_MAX_EVENT_ID_LENGTH,
        ),
        "index": _index(raw["index"]),
        "kind": _kind(raw["kind"]),
        "payload": _owned_payload(raw["payload"], field="event payload", state=state),
    }


def _kind(value: object) -> CompilerKind:
    text = _string(value, field="kind")
    if text not in {"observation", "action", "tool_result", "state_delta"}:
        raise ValueError("CompilerEvent kind is invalid")
    return cast(CompilerKind, text)


def _events_raw(value: object, *, state: _JsonState | None = None) -> tuple[dict[str, object], ...]:
    if type(value) not in {list, tuple}:
        raise ValueError("events must be an exact built-in list or tuple")
    events = cast(list[object] | tuple[object, ...], value)
    if not events or len(events) > _MAX_EVENTS:
        raise ValueError("events must be nonempty and bounded")
    owned_state = state if state is not None else _JsonState()
    return tuple(_event_raw(event, state=owned_state) for event in events)


def _trace_raw(value: object, *, state: _JsonState | None = None) -> dict[str, object]:
    if type(value) is CompilerTrace:
        raw = _exact_fields(value, expected=CompilerTrace, fields=_TRACE_FIELDS)
    else:
        raw = _raw_fields(value, fields=_TRACE_FIELDS, field="CompilerTrace")
    owned_state = state if state is not None else _JsonState()
    owned_state.reset_trace_budget()
    events = _events_raw(raw["events"], state=owned_state)
    trace_id = _identifier(raw["trace_id"], field="trace_id", pattern=_TRACE_ID)
    if len(trace_id) > _MAX_ID_LENGTH:
        raise ValueError("trace_id exceeds its maximum length")
    for index, event in enumerate(events):
        if event["index"] != index or event["event_id"] != _event_id(trace_id, index):
            raise ValueError("CompilerEvent identities must be contiguous and derived")
    kinds = [cast(str, event["kind"]) for event in events]
    if kinds[0] != "observation" or kinds[1:] != [
        kind
        for _ in range((len(kinds) - 1) // 3)
        for kind in ("action", "tool_result", "state_delta")
    ]:
        raise ValueError("CompilerTrace events must be observation plus complete triples")
    outcome = _string(raw["local_task_outcome"], field="local_task_outcome")
    if outcome != "completed":
        raise ValueError("CompilerTrace local_task_outcome must be completed")
    return {
        "trace_id": trace_id,
        "domain": _domain(raw["domain"]),
        "task_template_id": _identifier(raw["task_template_id"], field="task_template_id"),
        "worker_role": _identifier(raw["worker_role"], field="worker_role"),
        "world_hash": _sha256(raw["world_hash"], field="world_hash"),
        "events": events,
        "terminal_state_hash": _sha256(raw["terminal_state_hash"], field="terminal_state_hash"),
        "local_task_outcome": "completed",
    }


def _traces_raw(value: object) -> tuple[dict[str, object], ...]:
    if type(value) not in {list, tuple}:
        raise ValueError("traces must be an exact built-in list or tuple")
    traces = cast(list[object] | tuple[object, ...], value)
    if len(traces) != 12:
        raise ValueError("CompilerInput traces must contain exactly twelve values")
    state = _JsonState()
    raw = tuple(_trace_raw(trace, state=state) for trace in traces)
    if len({cast(str, trace["trace_id"]) for trace in raw}) != 12:
        raise ValueError("CompilerInput trace IDs must be unique")
    domain = raw[0]["domain"]
    if any(trace["domain"] != domain for trace in raw):
        raise ValueError("CompilerInput traces must share a domain")
    return raw


def _input_raw(value: object) -> dict[str, object]:
    if type(value) is CompilerInput:
        raw = _exact_fields(value, expected=CompilerInput, fields=_INPUT_FIELDS)
    else:
        raw = _raw_fields(value, fields=_INPUT_FIELDS, field="CompilerInput")
    profile = _string(raw["projection_profile"], field="projection_profile")
    if profile != _PROFILE:
        raise ValueError("projection_profile is invalid")
    traces = _traces_raw(raw["traces"])
    domain = _domain(raw["domain"])
    if any(trace["domain"] != domain for trace in traces):
        raise ValueError("CompilerInput domain does not match traces")
    return {"projection_profile": _PROFILE, "domain": domain, "traces": traces}


class _CompilerModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )
    _fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def _require_exact_model_class(cls) -> None:
        if cls.__bases__ != (_CompilerModel,):
            raise ValueError("Compiler projection models do not admit subclasses")

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
        return _model_raw(cls, value)

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
        raw = _model_raw(cls, obj)
        return super().model_validate(
            raw,
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
        if type(self) not in {CompilerEvent, CompilerTrace, CompilerInput}:
            raise ValueError("model_copy requires an exact compiler projection model")
        if type(deep) is not bool or update is not None and type(update) is not dict:
            raise ValueError("model_copy requires exact built-in inputs")
        raw = _model_raw(type(self), self)
        if update is not None:
            copied = _exact_dict(update, field="model_copy update")
            if any(type(key) is not str for key in copied) or any(
                key not in self._fields for key in copied
            ):
                raise ValueError("model_copy update has invalid fields")
            raw.update({cast(str, key): item for key, item in copied.items()})
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


def _model_raw(cls: type[_CompilerModel], value: object) -> dict[str, object]:
    if cls is CompilerEvent:
        return _event_raw(value)
    if cls is CompilerTrace:
        return _trace_raw(value)
    if cls is CompilerInput:
        return _input_raw(value)
    raise ValueError("unknown compiler projection model")


class CompilerEvent(_CompilerModel):
    _fields: ClassVar[tuple[str, ...]] = _EVENT_FIELDS
    event_id: str
    index: int
    kind: CompilerKind
    payload: JsonObject

    @field_validator("event_id", mode="before")
    @classmethod
    def _event_identifier(cls, value: object) -> str:
        return _identifier(
            value,
            field="event_id",
            pattern=_EVENT_ID,
            maximum=_MAX_EVENT_ID_LENGTH,
        )

    @field_validator("index", mode="before")
    @classmethod
    def _event_index(cls, value: object) -> int:
        return _index(value)

    @field_validator("kind", mode="before")
    @classmethod
    def _event_kind(cls, value: object) -> CompilerKind:
        return _kind(value)

    @field_validator("payload", mode="before")
    @classmethod
    def _event_payload(cls, value: object) -> JsonObject:
        return _owned_payload(value, field="event payload")


class CompilerTrace(_CompilerModel):
    _fields: ClassVar[tuple[str, ...]] = _TRACE_FIELDS
    trace_id: str
    domain: CompilerDomain
    task_template_id: str
    worker_role: str
    world_hash: str
    events: tuple[CompilerEvent, ...]
    terminal_state_hash: str
    local_task_outcome: Literal["completed"]

    @field_validator("trace_id", mode="before")
    @classmethod
    def _trace_identifier(cls, value: object) -> str:
        return _identifier(value, field="trace_id", pattern=_TRACE_ID)

    _trace_domain = field_validator("domain", mode="before")(_domain)
    _template_id = field_validator("task_template_id", mode="before")(
        lambda value: _identifier(value, field="task_template_id")
    )
    _role = field_validator("worker_role", mode="before")(
        lambda value: _identifier(value, field="worker_role")
    )
    _world_hash = field_validator("world_hash", mode="before")(
        lambda value: _sha256(value, field="world_hash")
    )
    _terminal_hash = field_validator("terminal_state_hash", mode="before")(
        lambda value: _sha256(value, field="terminal_state_hash")
    )

    @field_validator("events", mode="before")
    @classmethod
    def _trace_events(cls, value: object) -> tuple[dict[str, object], ...]:
        return _events_raw(value)

    @field_validator("local_task_outcome", mode="before")
    @classmethod
    def _outcome(cls, value: object) -> Literal["completed"]:
        if _string(value, field="local_task_outcome") != "completed":
            raise ValueError("local_task_outcome must be completed")
        return "completed"

    @model_validator(mode="after")
    def _event_links(self) -> Self:
        raw = _trace_raw(self)
        del raw
        return self


class CompilerInput(_CompilerModel):
    _fields: ClassVar[tuple[str, ...]] = _INPUT_FIELDS
    projection_profile: Literal["SSB-COMPILER-VIEW1"]
    domain: CompilerDomain
    traces: tuple[CompilerTrace, ...]

    @field_validator("projection_profile", mode="before")
    @classmethod
    def _profile(cls, value: object) -> Literal["SSB-COMPILER-VIEW1"]:
        if _string(value, field="projection_profile") != _PROFILE:
            raise ValueError("projection_profile is invalid")
        return _PROFILE

    _input_domain = field_validator("domain", mode="before")(_domain)

    @field_validator("traces", mode="before")
    @classmethod
    def _input_traces(cls, value: object) -> tuple[dict[str, object], ...]:
        return _traces_raw(value)

    @model_validator(mode="after")
    def _input_links(self) -> Self:
        raw = _input_raw(self)
        del raw
        return self


def compiler_view(bundle: DemonstrationBundle) -> CompilerInput:
    validated = _source_preflight(bundle)
    traces = tuple(
        {
            "trace_id": trace.trace_id,
            "domain": trace.domain,
            "task_template_id": trace.task_template_id,
            "worker_role": trace.worker_role,
            "world_hash": trace.world_hash,
            "events": tuple(
                {
                    "event_id": event.event_id,
                    "index": event.index,
                    "kind": event.kind,
                    "payload": event.payload,
                }
                for event in trace.events
            ),
            "terminal_state_hash": trace.terminal_state_hash,
            "local_task_outcome": trace.local_task_outcome,
        }
        for trace in validated.traces
    )
    return CompilerInput.model_validate(
        {
            "projection_profile": _PROFILE,
            "domain": validated.source_manifest.domain,
            "traces": traces,
        }
    )


def compiler_input_projection(value: object) -> JsonObject:
    parsed = CompilerInput.model_validate(value)
    raw = _input_raw(parsed)
    return cast(
        JsonObject,
        {
            "projection_profile": raw["projection_profile"],
            "domain": raw["domain"],
            "traces": [
                {
                    "trace_id": trace["trace_id"],
                    "domain": trace["domain"],
                    "task_template_id": trace["task_template_id"],
                    "worker_role": trace["worker_role"],
                    "world_hash": trace["world_hash"],
                    "events": [
                        {
                            "event_id": event["event_id"],
                            "index": event["index"],
                            "kind": event["kind"],
                            "payload": event["payload"],
                        }
                        for event in cast(tuple[dict[str, object], ...], trace["events"])
                    ],
                    "terminal_state_hash": trace["terminal_state_hash"],
                    "local_task_outcome": trace["local_task_outcome"],
                }
                for trace in cast(tuple[dict[str, object], ...], raw["traces"])
            ],
        },
    )


def hash_compiler_input(value: object) -> str:
    return sha256_ref(compiler_input_projection(value))


__all__ = [
    "CompilerEvent",
    "CompilerInput",
    "CompilerTrace",
    "compiler_input_projection",
    "compiler_view",
    "hash_compiler_input",
]
