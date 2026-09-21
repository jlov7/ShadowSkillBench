from __future__ import annotations

# ruff: noqa: E501, UP046, UP047
import math
import re
import weakref
from collections.abc import Mapping
from inspect import isabstract
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic_core import SchemaValidator, core_schema

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref

T = TypeVar("T", bound=BaseModel)

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_TEXT = 1_000_000
_MAX_BODY = 4_000_000
_MAX_I64 = 2**63 - 1


def _provenance_registry() -> tuple[Any, Any]:
    entries: dict[int, weakref.ReferenceType[BaseModel]] = {}

    def post_validation(schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
        def mark(value: BaseModel) -> BaseModel:
            identifier = id(value)

            def remove(reference: weakref.ReferenceType[BaseModel]) -> None:
                if entries.get(identifier) is reference:
                    entries.pop(identifier, None)

            entries[identifier] = weakref.ref(value, remove)
            return value

        return core_schema.no_info_after_validator_function(mark, schema)

    def owns(value: object) -> bool:
        reference = entries.get(id(value))
        return reference is not None and reference() is value

    return post_validation, owns


_owned_post_validation, _is_owned = _provenance_registry()


def _is_concrete_schema_class(value: object) -> bool:
    return (
        type(value) is type(BaseModel)
        and value is not BaseModel
        and not isabstract(cast(type[BaseModel], value))
        and not getattr(value, "__parameters__", ())
    )


def _response_output_class(value_class: type[BaseModel]) -> type[BaseModel] | None:
    response_class = globals().get("ModelResponse")
    metadata = getattr(value_class, "__pydantic_generic_metadata__", None)
    if (
        response_class is None
        or type(metadata) is not dict
        or metadata.get("origin") is not response_class
    ):
        return None
    arguments = metadata.get("args")
    if type(arguments) is not tuple or len(arguments) != 1:
        return None
    candidate = arguments[0]
    if not _is_concrete_schema_class(candidate):
        return None
    return cast(type[BaseModel], candidate)


def _is_admitted_value_class(value_class: type[BaseModel]) -> bool:
    if type(value_class) is not type(BaseModel):
        return False
    public_names = {
        "Message",
        "ProviderCapabilities",
        "ModelRequest",
        "TokenUsage",
        "TokenPricing",
        "TokenCost",
        "TransportResponse",
    }
    if value_class.__name__ in public_names and globals().get(value_class.__name__) is value_class:
        return True
    return _response_output_class(value_class) is not None


def _text(value: object, *, field: str, maximum: int = _MAX_TEXT, nonblank: bool = True) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact string")
    text = cast(str, value)
    if len(text) > maximum or "\x00" in text:
        raise ValueError(f"{field} is invalid")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field} must be UTF-8 encodable") from error
    if nonblank and not text.strip():
        raise ValueError(f"{field} is invalid")
    return text


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0 or value > _MAX_I64:
        raise ValueError(f"{field} must be a bounded exact nonnegative integer")
    return cast(int, value)


def _model_data(value: object, expected: type[BaseModel]) -> dict[str, object]:
    if type(value) is not expected or not _is_admitted_value_class(expected):
        raise ValueError(f"value must be an exact {expected.__name__}")
    fields = tuple(expected.model_fields)
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
        or len(data) != len(fields)
        or len(field_set) != len(fields)
        or any(type(name) is not str or name not in fields for name in data)
        or any(type(name) is not str or name not in fields for name in field_set)
        or extra is not None
        or private is not None
        or not _is_owned(value)
    ):
        raise ValueError(f"{expected.__name__} is incomplete or forged")
    return {field: data[field] for field in fields}


class _Value(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[BaseModel], handler: Any
    ) -> core_schema.CoreSchema:
        schema = handler(source)

        def before(value: object) -> object:
            if not _is_admitted_value_class(cls):
                raise ValueError(f"{cls.__name__} is not an admitted value class")
            if type(value) is dict:
                return value
            if type(value) is cls and _is_owned(value):
                return _model_data(value, cls)
            raise ValueError(f"{cls.__name__} requires exact admitted ingress")

        def reject_json(_: object) -> object:
            raise ValueError("JSON/string ingress is forbidden")

        python_schema = core_schema.no_info_before_validator_function(before, schema)
        python_schema = _owned_post_validation(python_schema)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_plain_validator_function(reject_json),
            python_schema=python_schema,
        )

    @model_validator(mode="before")
    @classmethod
    def _exact_mapping(cls, value: object) -> object:
        if type(value) is not dict:
            raise ValueError(f"{cls.__name__} requires an exact built-in object")
        fields = tuple(cls.model_fields)
        if len(value) != len(fields) or any(
            type(key) is not str or key not in fields for key in value
        ):
            raise ValueError(f"{cls.__name__} mapping fields are invalid")
        return value

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Any:
        if type(obj) is not dict and not (type(obj) is cls and _is_owned(obj)):
            raise ValueError(f"{cls.__name__} requires an exact built-in object")
        if kwargs.get("strict") is False or kwargs.get("from_attributes") is True:
            raise ValueError(f"{cls.__name__} requires strict mapping ingress")
        return super().model_validate(obj, strict=True, extra="forbid", from_attributes=False)

    @classmethod
    def model_validate_json(cls, json_data: Any, **kwargs: Any) -> Any:
        del json_data, kwargs
        raise ValueError("JSON/string ingress is forbidden")

    @classmethod
    def model_validate_strings(cls, obj: Any, **kwargs: Any) -> Any:
        del obj, kwargs
        raise ValueError("coercive ingress is forbidden")

    @classmethod
    def parse_raw(cls, b: Any, *args: Any, **kwargs: Any) -> Any:
        del b, args, kwargs
        raise ValueError("JSON/string ingress is forbidden")

    @classmethod
    def parse_file(cls, path: Any, *args: Any, **kwargs: Any) -> Any:
        del path, args, kwargs
        raise ValueError("JSON/string ingress is forbidden")

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Any:
        if type(deep) is not bool or update is not None and type(update) is not dict:
            raise ValueError("model_copy requires exact built-in inputs")
        raw = _model_data(self, type(self))
        if update is not None:
            if any(type(key) is not str or key not in raw for key in update):
                raise ValueError("model_copy update is invalid")
            raw.update(update)
        return type(self)(**raw)

    def copy(
        self,
        *,
        include: object = None,
        exclude: object = None,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Any:
        if include is not None or exclude is not None:
            raise ValueError("legacy copy selection is forbidden")
        return self.model_copy(update=update, deep=deep)

    def __copy__(self) -> Any:
        return self.model_copy()

    def __deepcopy__(self, memo: object | None = None) -> Any:
        if memo is not None and type(memo) is not dict:
            raise ValueError("deepcopy memo must be exact")
        return self.model_copy(deep=True)

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Any:
        del _fields_set, values
        raise ValueError("unvalidated construction is forbidden")


class Message(_Value):
    role: Literal["system", "developer", "user", "assistant"]
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def _role(cls, value: object) -> str:
        if type(value) is not str or value not in {"system", "developer", "user", "assistant"}:
            raise ValueError("role is invalid")
        return cast(str, value)

    @field_validator("content", mode="before")
    @classmethod
    def _content(cls, value: object) -> str:
        return _text(value, field="content", nonblank=False)


class ProviderCapabilities(_Value):
    provider: str
    model: str
    model_version: str
    supports_system_role: bool
    supports_developer_role: bool
    supports_seed: bool
    supports_structured_output: bool

    @field_validator("provider", "model", "model_version", mode="before")
    @classmethod
    def _identity(cls, value: object, info: Any) -> str:
        return _text(value, field=cast(str, info.field_name), maximum=128)

    @field_validator(
        "supports_system_role",
        "supports_developer_role",
        "supports_seed",
        "supports_structured_output",
        mode="before",
    )
    @classmethod
    def _bool(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("capability flags must be exact booleans")
        return cast(bool, value)


class ModelRequest(_Value):
    messages: tuple[Message, ...]
    temperature: float
    seed: int | None
    max_tokens: int

    @field_validator("messages", mode="before")
    @classmethod
    def _messages(cls, value: object) -> tuple[Message, ...]:
        if type(value) not in {tuple, list}:
            raise ValueError("messages must be an exact built-in sequence")
        items = cast(tuple[object, ...] | list[object], value)
        if not items or len(items) > 256:
            raise ValueError("messages must contain 1 through 256 values")
        parsed: list[Message] = []
        aggregate_bytes = 0
        for item in items:
            message = (
                Message.model_validate(_model_data(item, Message))
                if type(item) is Message
                else Message.model_validate(item)
            )
            aggregate_bytes += len(message.content.encode("utf-8"))
            if aggregate_bytes > _MAX_TEXT:
                raise ValueError("messages exceed the UTF-8 budget")
            parsed.append(message)
        return tuple(parsed)

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature(cls, value: object) -> float:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
            or value > 2.0
            or value == 0.0
            and math.copysign(1.0, value) < 0
        ):
            raise ValueError(
                "temperature must be a finite exact non-negative non-zero-signed float"
            )
        return cast(float, value)

    @field_validator("seed", mode="before")
    @classmethod
    def _seed(cls, value: object) -> int | None:
        if value is None:
            return None
        if type(value) is not int or not -(2**63) <= value <= _MAX_I64:
            raise ValueError("seed must be an exact signed 64-bit integer")
        return cast(int, value)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _max_tokens(cls, value: object) -> int:
        if type(value) is not int or not 1 <= value <= 1_000_000:
            raise ValueError("max_tokens is invalid")
        return cast(int, value)


class TokenUsage(_Value):
    input_tokens: int
    output_tokens: int
    total_tokens: int

    _input = field_validator("input_tokens", mode="before")(
        lambda value: _nonnegative_int(value, field="input_tokens")
    )
    _output = field_validator("output_tokens", mode="before")(
        lambda value: _nonnegative_int(value, field="output_tokens")
    )
    _total = field_validator("total_tokens", mode="before")(
        lambda value: _nonnegative_int(value, field="total_tokens")
    )

    @model_validator(mode="after")
    def _sum(self) -> TokenUsage:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must be the exact sum")
        return self


class TokenPricing(_Value):
    currency: Literal["USD"]
    input_nanos_per_token: int
    output_nanos_per_token: int

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, value: object) -> str:
        if type(value) is not str or value != "USD":
            raise ValueError("currency must be USD")
        return "USD"

    _input = field_validator("input_nanos_per_token", mode="before")(
        lambda value: _nonnegative_int(value, field="input_nanos_per_token")
    )
    _output = field_validator("output_nanos_per_token", mode="before")(
        lambda value: _nonnegative_int(value, field="output_nanos_per_token")
    )


class TokenCost(_Value):
    currency: Literal["USD"]
    input_nanos_per_token: int
    output_nanos_per_token: int
    input_nanos: int
    output_nanos: int
    total_nanos: int

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, value: object) -> str:
        if type(value) is not str or value != "USD":
            raise ValueError("currency must be USD")
        return "USD"

    _input_rate = field_validator("input_nanos_per_token", mode="before")(
        lambda value: _nonnegative_int(value, field="input_nanos_per_token")
    )
    _output_rate = field_validator("output_nanos_per_token", mode="before")(
        lambda value: _nonnegative_int(value, field="output_nanos_per_token")
    )
    _input = field_validator("input_nanos", mode="before")(
        lambda value: _nonnegative_int(value, field="input_nanos")
    )
    _output = field_validator("output_nanos", mode="before")(
        lambda value: _nonnegative_int(value, field="output_nanos")
    )
    _total = field_validator("total_nanos", mode="before")(
        lambda value: _nonnegative_int(value, field="total_nanos")
    )

    @model_validator(mode="after")
    def _sum(self) -> TokenCost:
        if self.total_nanos != self.input_nanos + self.output_nanos:
            raise ValueError("total_nanos must be the exact sum")
        return self


class TransportResponse(_Value):
    status_code: int
    body: bytes

    @field_validator("status_code", mode="before")
    @classmethod
    def _status(cls, value: object) -> int:
        if type(value) is not int or not 100 <= value <= 599:
            raise ValueError("status_code is invalid")
        return cast(int, value)

    @field_validator("body", mode="before")
    @classmethod
    def _body(cls, value: object) -> bytes:
        if type(value) is not bytes or len(value) > _MAX_BODY:
            raise ValueError("body must be bounded exact immutable bytes")
        return bytes(value)


def _caller_output_data(value: object, schema: type[BaseModel]) -> dict[str, object]:
    if type(value) is not schema:
        raise ValueError("output must have the exact requested schema class")
    fields = tuple(schema.model_fields)
    try:
        data = object.__getattribute__(value, "__dict__")
        field_set = object.__getattribute__(value, "__pydantic_fields_set__")
        extra = object.__getattribute__(value, "__pydantic_extra__")
        _private = object.__getattribute__(value, "__pydantic_private__")
    except AttributeError as error:
        raise ValueError("output is incomplete or forged") from error
    if (
        type(data) is not dict
        or type(field_set) is not set
        or len(data) != len(fields)
        or any(type(name) is not str or name not in fields for name in data)
        or any(type(name) is not str or name not in fields for name in field_set)
        or (extra is not None and (type(extra) is not dict or bool(extra)))
    ):
        raise ValueError("output is incomplete or forged")
    return {field: data[field] for field in fields}


def _validate_schema_output(schema: type[BaseModel], raw: object) -> BaseModel:
    if not _is_concrete_schema_class(schema) or type(raw) is not dict:
        raise ValueError("output schema or raw object is invalid")
    try:
        validator = type.__getattribute__(schema, "__pydantic_validator__")
    except AttributeError as error:
        raise ValueError("output schema has no compiled validator") from error
    if type(validator) is not SchemaValidator:
        raise ValueError("output schema validator is invalid")
    output = validator.validate_python(raw, strict=True, extra="forbid", from_attributes=False, by_alias=False, by_name=True)  # fmt: skip
    if type(output) is not schema:
        raise ValueError("output must retain the exact requested schema class")
    _caller_output_data(output, schema)
    return output


def _revalidate_caller_output(value: object, schema: type[BaseModel]) -> BaseModel:
    raw = _caller_output_data(value, schema)
    try:
        return _validate_schema_output(schema, raw)
    except (TypeError, ValueError):
        pass
    try:
        validator = type.__getattribute__(schema, "__pydantic_validator__")
    except AttributeError as error:
        raise ValueError("output schema has no compiled validator") from error
    if type(validator) is not SchemaValidator:
        raise ValueError("output schema validator is invalid")
    output = validator.validate_python(value, strict=True, extra="forbid", from_attributes=False, by_alias=False, by_name=True)  # fmt: skip
    if output is value or type(output) is not schema:
        raise ValueError("output must retain a distinct exact requested schema class")
    _caller_output_data(output, schema)
    return output


class ModelResponse(_Value, Generic[T]):
    capabilities: ProviderCapabilities
    output: T
    raw_request_hash: str
    raw_response_hash: str
    structured_output_schema_hash: str
    usage: TokenUsage
    cost: TokenCost | None
    attempts: int

    @field_validator("capabilities", mode="before")
    @classmethod
    def _capabilities(cls, value: object) -> ProviderCapabilities:
        raw = (
            _model_data(value, ProviderCapabilities)
            if type(value) is ProviderCapabilities
            else value
        )
        return ProviderCapabilities.model_validate(raw)

    @field_validator("output", mode="before")
    @classmethod
    def _output(cls, value: object) -> BaseModel:
        schema = _response_output_class(cls)
        if schema is None:
            raise ValueError("ModelResponse requires one concrete output schema")
        return _revalidate_caller_output(value, schema)

    @field_validator("usage", mode="before")
    @classmethod
    def _usage(cls, value: object) -> TokenUsage:
        raw = _model_data(value, TokenUsage) if type(value) is TokenUsage else value
        return TokenUsage.model_validate(raw)

    @field_validator("cost", mode="before")
    @classmethod
    def _cost(cls, value: object) -> TokenCost | None:
        if value is None:
            return None
        raw = _model_data(value, TokenCost) if type(value) is TokenCost else value
        return TokenCost.model_validate(raw)

    @model_validator(mode="after")
    def _valid(self) -> ModelResponse[T]:
        if _response_output_class(type(self)) is None:
            raise ValueError("ModelResponse requires one concrete output schema")
        if any(
            type(value) is not str or _HASH.fullmatch(value) is None
            for value in (
                self.raw_request_hash,
                self.raw_response_hash,
                self.structured_output_schema_hash,
            )
        ):
            raise ValueError("response hashes are invalid")
        if type(self.attempts) is not int or self.attempts < 1 or self.attempts > 5:
            raise ValueError("attempts are invalid")
        _model_data(self.capabilities, ProviderCapabilities)
        _model_data(self.usage, TokenUsage)
        if self.cost is not None:
            cost = self.cost
            _model_data(cost, TokenCost)
            if (
                cost.input_nanos != self.usage.input_tokens * cost.input_nanos_per_token
                or cost.output_nanos != self.usage.output_tokens * cost.output_nanos_per_token
                or cost.total_nanos != cost.input_nanos + cost.output_nanos
            ):
                raise ValueError("cost does not bind usage and rates")
        return self


class TransportTransientError(Exception):
    pass


class TransportTerminalError(Exception):
    pass


class ModelAdapterError(RuntimeError):
    __slots__ = (
        "code",
        "attempts",
        "before_meaningful_behavior",
        "raw_request_hash",
        "raw_response_hash",
        "finish_reason",
        "reported_usage",
        "provider_done",
        "valid_tool_name",
        "valid_arguments_hash_match",
        "_sealed",
    )

    def __init__(
        self,
        *,
        code: str,
        attempts: int,
        before_meaningful_behavior: bool,
        raw_request_hash: str | None,
        raw_response_hash: str | None,
        finish_reason: str | None = None,
        reported_usage: TokenUsage | None = None,
        provider_done: bool | None = None,
        valid_tool_name: bool | None = None,
        valid_arguments_hash_match: bool | None = None,
    ) -> None:
        if type(self) is not ModelAdapterError:
            raise ValueError("adapter errors must have the exact public class")
        if type(code) is not str or code not in {
            "CONFIGURATION_ERROR",
            "SCHEMA_ERROR",
            "MODEL_PROVIDER_TRANSIENT",
            "MODEL_PROVIDER_TERMINAL",
            "MODEL_OUTPUT_INVALID",
        }:
            raise ValueError("invalid adapter error code")
        if type(attempts) is not int or attempts < 0 or attempts > 5:
            raise ValueError("adapter error attempts are invalid")
        if type(before_meaningful_behavior) is not bool:
            raise ValueError("adapter error behavior flag is invalid")
        if attempts == 0:
            if raw_request_hash is not None or raw_response_hash is not None:
                raise ValueError("zero attempts require absent hashes")
        elif raw_request_hash is None:
            raise ValueError("positive attempts require a request hash")
        if raw_response_hash is not None and raw_request_hash is None:
            raise ValueError("response hashes require a request hash")
        for digest in (raw_request_hash, raw_response_hash):
            if digest is not None and (type(digest) is not str or _HASH.fullmatch(digest) is None):
                raise ValueError("adapter error hash is invalid")
        if finish_reason is not None and (
            type(finish_reason) is not str
            or not finish_reason.strip()
            or len(finish_reason) > 32
            or not finish_reason.isprintable()
        ):
            raise ValueError("adapter error finish reason is invalid")
        if reported_usage is not None and type(reported_usage) is not TokenUsage:
            raise ValueError("adapter error usage is invalid")
        if raw_response_hash is None and (finish_reason is not None or reported_usage is not None):
            raise ValueError("response diagnostics require a response hash")
        if any(
            value is not None and type(value) is not bool
            for value in (provider_done, valid_tool_name, valid_arguments_hash_match)
        ):
            raise ValueError("native response diagnostics are invalid")
        if raw_response_hash is None and any(
            value is not None
            for value in (provider_done, valid_tool_name, valid_arguments_hash_match)
        ):
            raise ValueError("native response diagnostics require a response hash")
        super().__init__(code)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "before_meaningful_behavior", before_meaningful_behavior)
        object.__setattr__(self, "raw_request_hash", raw_request_hash)
        object.__setattr__(self, "raw_response_hash", raw_response_hash)
        object.__setattr__(self, "finish_reason", finish_reason)
        object.__setattr__(self, "reported_usage", reported_usage)
        object.__setattr__(self, "provider_done", provider_done)
        object.__setattr__(self, "valid_tool_name", valid_tool_name)
        object.__setattr__(self, "valid_arguments_hash_match", valid_arguments_hash_match)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "__traceback__":
            BaseException.__setattr__(self, name, value)
            return
        if name in {"__cause__", "__context__", "__suppress_context__"}:
            if value is not None:
                raise AttributeError("adapter errors do not retain exception context")
            BaseException.__setattr__(self, name, value)
            return
        if object.__getattribute__(self, "_sealed"):
            raise AttributeError("adapter errors are immutable")
        object.__setattr__(self, name, value)

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return f"ModelAdapterError(code={self.code!r})"


class AsyncTransport(Protocol):
    async def __call__(self, body: bytes) -> TransportResponse: ...


class ModelClient(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...

    async def structured(self, request: ModelRequest, schema: type[T]) -> ModelResponse[T]: ...


def _calculate_cost(usage: TokenUsage, pricing: TokenPricing) -> TokenCost:
    input_nanos = usage.input_tokens * pricing.input_nanos_per_token
    output_nanos = usage.output_tokens * pricing.output_nanos_per_token
    total = input_nanos + output_nanos
    if any(value > _MAX_I64 for value in (input_nanos, output_nanos, total)):
        raise OverflowError("cost overflow")
    return TokenCost(
        currency="USD",
        input_nanos_per_token=pricing.input_nanos_per_token,
        output_nanos_per_token=pricing.output_nanos_per_token,
        input_nanos=input_nanos,
        output_nanos=output_nanos,
        total_nanos=total,
    )


def _schema_hash(schema: type[T]) -> tuple[dict[str, object], str]:
    if (
        type(schema) is not type(BaseModel)
        or schema is BaseModel
        or isabstract(schema)
        or getattr(schema, "__parameters__", ())
    ):
        raise ValueError("schema must be an exact concrete Pydantic model class")
    raw = schema.model_json_schema()
    if type(raw) is not dict:
        raise ValueError("schema must be a JSON object")
    encoded = canonical_json_bytes(raw)
    if len(encoded) > _MAX_TEXT:
        raise ValueError("schema exceeds byte budget")
    return cast(dict[str, object], raw), sha256_ref(raw)


__all__ = [
    "AsyncTransport",
    "Message",
    "ModelAdapterError",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "ProviderCapabilities",
    "TokenCost",
    "TokenPricing",
    "TokenUsage",
    "TransportResponse",
    "TransportTerminalError",
    "TransportTransientError",
]
