from __future__ import annotations

import math
import re
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from shadowskillbench.core.hashing import sha256_ref

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type TraceKind = Literal["observation", "action", "tool_result", "state_delta", "narration"]
type NarrationMode = Literal["none", "neutral", "rationalized"]
type Domain = Literal["access_provisioning", "financial_adjustments"]
type LocalTaskOutcome = Literal["completed", "failed", "escalated"]

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
_MAX_NARRATION_CODEPOINTS = 4_096


class _TraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _exact_dict(value: object, *, field: str) -> dict[object, object]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact built-in object")
    return cast(dict[object, object], value)


def _required_fields(
    value: object, *, model_type: type[BaseModel], fields: tuple[str, ...]
) -> dict[str, object]:
    if type(value) is not model_type:
        raise ValueError(f"value must be an exact {model_type.__name__}")
    data = object.__getattribute__(value, "__dict__")
    if type(data) is not dict or len(data) != len(fields):
        raise ValueError(f"{model_type.__name__} is incomplete or forged")
    if any(type(key) is not str for key in data) or any(field not in data for field in fields):
        raise ValueError(f"{model_type.__name__} is incomplete or forged")
    return {field: data[field] for field in fields}


def _event_input(value: object) -> dict[str, object]:
    fields = ("event_id", "index", "kind", "payload")
    if type(value) is TraceEvent:
        return _required_fields(value, model_type=TraceEvent, fields=fields)
    raw = _exact_dict(value, field="event")
    if type(raw) is not dict or len(raw) != len(fields):
        raise ValueError("event has missing or extra fields")
    if any(type(key) is not str for key in raw) or any(field not in raw for field in fields):
        raise ValueError("event has missing or extra fields")
    return {field: raw[field] for field in fields}


def _trace_input(value: object) -> dict[str, object]:
    fields = (
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
    if type(value) is ActionTrace:
        return _required_fields(value, model_type=ActionTrace, fields=fields)
    raw = _exact_dict(value, field="action trace")
    required = fields[:-1]
    if len(raw) not in {len(required), len(fields)}:
        raise ValueError("action trace has missing or extra fields")
    if any(type(key) is not str for key in raw) or any(field not in raw for field in required):
        raise ValueError("action trace has missing or extra fields")
    if len(raw) == len(fields) and "hidden_benchmark_metadata_ref" not in raw:
        raise ValueError("action trace has missing or extra fields")
    return {field: raw.get(field) for field in fields}


def _exact_string(value: object, *, field: str, maximum: int | None = None) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact string")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{field} exceeds its maximum length")
    return cast(str, value)


def _path_safe_id(value: object, *, field: str) -> str:
    text = _exact_string(value, field=field, maximum=_MAX_ID_LENGTH)
    if _PATH_SAFE_ID.fullmatch(text) is None:
        raise ValueError(f"{field} must be a path-safe ASCII identifier")
    return text


def _trace_id(value: object) -> str:
    text = _exact_string(value, field="trace_id", maximum=_MAX_ID_LENGTH)
    if _TRACE_ID.fullmatch(text) is None:
        raise ValueError("trace_id has invalid grammar")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _exact_string(value, field=field)
    if _SHA256_REF.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 reference")
    return text


def _exact_index(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an exact nonnegative integer")
    return cast(int, value)


def _literal(value: object, *, field: str, values: set[str]) -> str:
    text = _exact_string(value, field=field)
    if text not in values:
        raise ValueError(f"{field} has an invalid value")
    return text


def _event_id(trace_id: str, index: int) -> str:
    return f"event_{trace_id}_{index:06d}"


def _detach_payloads(payloads: list[object]) -> list[JsonObject]:
    seen_containers: set[int] = set()
    value_count = 0
    text_bytes = 0

    def account_text(value: str, *, field: str) -> None:
        nonlocal text_bytes
        remaining = _MAX_TEXT_BYTES - text_bytes
        if len(value) > remaining:
            raise ValueError(f"{field} exceeds the remaining UTF-8 budget")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"{field} is not UTF-8 encodable") from error
        if len(encoded) > remaining:
            raise ValueError(f"{field} exceeds the UTF-8 budget")
        text_bytes += len(encoded)

    def copy(value: object, *, depth: int) -> JsonValue:
        nonlocal value_count
        value_count += 1
        if value_count > _MAX_VALUES:
            raise ValueError("payload values exceed the trace budget")
        value_type = type(value)
        if value_type is type(None) or value_type is bool:
            return cast(JsonValue, value)
        if value_type is int:
            if abs(cast(int, value)).bit_length() > _MAX_INT_BITS:
                raise ValueError("payload integer exceeds the bit-length budget")
            return cast(JsonValue, value)
        if value_type is float:
            if not math.isfinite(cast(float, value)):
                raise ValueError("payload floats must be finite")
            return cast(JsonValue, value)
        if value_type is str:
            text = cast(str, value)
            account_text(text, field="payload string")
            return text
        if value_type not in {dict, list}:
            raise ValueError("payloads must contain exact built-in JSON values")
        if depth > _MAX_DEPTH:
            raise ValueError("payload depth exceeds the trace budget")
        container = cast(dict[object, object] | list[object], value)
        container_id = id(container)
        if container_id in seen_containers:
            raise ValueError("payload containers must not be cyclic or aliased")
        seen_containers.add(container_id)
        if value_type is list:
            return [copy(item, depth=depth + 1) for item in cast(list[object], container)]
        copied: dict[str, JsonValue] = {}
        for key, item in cast(dict[object, object], container).items():
            if type(key) is not str:
                raise ValueError("payload object keys must be exact strings")
            key_text = cast(str, key)
            if len(key_text) > _MAX_KEY_LENGTH:
                raise ValueError("payload object key exceeds the maximum length")
            account_text(key_text, field="payload object key")
            copied[key_text] = copy(item, depth=depth + 1)
        return copied

    detached: list[JsonObject] = []
    for payload in payloads:
        if type(payload) is not dict:
            raise ValueError("event payload must be an exact built-in JSON object")
        copied = copy(payload, depth=0)
        detached.append(cast(JsonObject, copied))
    return detached


class TraceEvent(_TraceModel):
    event_id: str
    index: int
    kind: TraceKind
    payload: JsonObject

    @model_validator(mode="before")
    @classmethod
    def _require_exact_input(cls, value: object) -> object:
        return _event_input(value)

    @field_validator("event_id", mode="before")
    @classmethod
    def _validate_event_id(cls, value: object) -> str:
        text = _exact_string(value, field="event_id", maximum=_MAX_EVENT_ID_LENGTH)
        if _EVENT_ID.fullmatch(text) is None:
            raise ValueError("event_id has invalid grammar")
        return text

    @field_validator("index", mode="before")
    @classmethod
    def _validate_index(cls, value: object) -> int:
        return _exact_index(value, field="index")

    @field_validator("kind", mode="before")
    @classmethod
    def _validate_kind(cls, value: object) -> TraceKind:
        return cast(
            TraceKind,
            _literal(
                value,
                field="kind",
                values={"observation", "action", "tool_result", "state_delta", "narration"},
            ),
        )

    @field_validator("payload", mode="before")
    @classmethod
    def _validate_payload(cls, value: object) -> JsonObject:
        return _detach_payloads([value])[0]

    @model_validator(mode="after")
    def _validate_narration(self) -> Self:
        if self.kind != "narration":
            return self
        if set(self.payload) != {"text"}:
            raise ValueError("narration payload must contain exactly text")
        text = self.payload["text"]
        if (
            type(text) is not str
            or not cast(str, text).strip()
            or "\x00" in cast(str, text)
            or len(cast(str, text)) > _MAX_NARRATION_CODEPOINTS
        ):
            raise ValueError("narration text must be nonblank, NUL-free, and bounded")
        return self


def _normalise_events(value: object) -> tuple[dict[str, object], ...]:
    if type(value) not in {list, tuple}:
        raise ValueError("events must be an exact built-in list or tuple")
    raw_events = cast(list[object] | tuple[object, ...], value)
    if len(raw_events) > _MAX_EVENTS:
        raise ValueError("events exceed the trace budget")
    normalised = [_event_input(event) for event in raw_events]
    payloads = _detach_payloads([event["payload"] for event in normalised])
    return tuple(
        {**event, "payload": payload} for event, payload in zip(normalised, payloads, strict=True)
    )


class ActionTrace(_TraceModel):
    trace_id: str
    schema_version: Literal["1.0"]
    domain: Domain
    task_template_id: str
    worker_policy_id: str
    worker_role: str
    world_hash: str
    narration_mode: NarrationMode
    events: tuple[TraceEvent, ...]
    terminal_state_hash: str
    local_task_outcome: LocalTaskOutcome
    hidden_benchmark_metadata_ref: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _require_exact_input(cls, value: object) -> object:
        return _trace_input(value)

    @field_validator("trace_id", mode="before")
    @classmethod
    def _validate_trace_id(cls, value: object) -> str:
        return _trace_id(value)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> Literal["1.0"]:
        if _exact_string(value, field="schema_version") != "1.0":
            raise ValueError("schema_version must be 1.0")
        return "1.0"

    @field_validator("domain", mode="before")
    @classmethod
    def _validate_domain(cls, value: object) -> Domain:
        return cast(
            Domain,
            _literal(
                value,
                field="domain",
                values={"access_provisioning", "financial_adjustments"},
            ),
        )

    @field_validator("task_template_id", "worker_policy_id", "worker_role", mode="before")
    @classmethod
    def _validate_ids(cls, value: object, info: object) -> str:
        return _path_safe_id(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("world_hash", "terminal_state_hash", mode="before")
    @classmethod
    def _validate_hashes(cls, value: object, info: object) -> str:
        return _sha256(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("narration_mode", mode="before")
    @classmethod
    def _validate_narration_mode(cls, value: object) -> NarrationMode:
        return cast(
            NarrationMode,
            _literal(value, field="narration_mode", values={"none", "neutral", "rationalized"}),
        )

    @field_validator("events", mode="before")
    @classmethod
    def _validate_events(cls, value: object) -> tuple[dict[str, object], ...]:
        return _normalise_events(value)

    @field_validator("local_task_outcome", mode="before")
    @classmethod
    def _validate_outcome(cls, value: object) -> LocalTaskOutcome:
        return cast(
            LocalTaskOutcome,
            _literal(
                value, field="local_task_outcome", values={"completed", "failed", "escalated"}
            ),
        )

    @field_validator("hidden_benchmark_metadata_ref", mode="before")
    @classmethod
    def _validate_hidden_reference(cls, value: object) -> str | None:
        if value is None:
            return None
        return _sha256(value, field="hidden_benchmark_metadata_ref")

    @model_validator(mode="after")
    def _validate_trace_contract(self) -> Self:
        if not self.events:
            raise ValueError("events must be nonempty")
        if len(self.events) > _MAX_EVENTS:
            raise ValueError("events exceed the trace budget")
        if self.events[0].kind != "observation":
            raise ValueError("the first event must be an observation")
        expected_kinds: list[str] = []
        for index, event in enumerate(self.events):
            if event.index != index or event.event_id != _event_id(self.trace_id, index):
                raise ValueError("event indices and identities must be contiguous and derived")
            if event.kind != "narration":
                expected_kinds.append(event.kind)
        if not expected_kinds or expected_kinds[0] != "observation":
            raise ValueError("the first normative event must be an observation")
        if expected_kinds[1:] != [
            kind
            for _ in range((len(expected_kinds) - 1) // 3)
            for kind in ("action", "tool_result", "state_delta")
        ]:
            raise ValueError("normative events must form complete action/result/delta triples")
        narration_count = sum(event.kind == "narration" for event in self.events)
        if (self.narration_mode == "none" and narration_count) or (
            self.narration_mode != "none" and not narration_count
        ):
            raise ValueError("narration events do not match narration_mode")
        return self


def _validated_trace(value: object) -> ActionTrace:
    return ActionTrace.model_validate(_trace_input(value))


def make_trace_event(
    *, trace_id: object, index: object, kind: object, payload: object
) -> TraceEvent:
    owned_trace_id = _trace_id(trace_id)
    owned_index = _exact_index(index, field="index")
    return TraceEvent.model_validate(
        {
            "event_id": _event_id(owned_trace_id, owned_index),
            "index": owned_index,
            "kind": kind,
            "payload": payload,
        }
    )


def make_action_trace(
    *,
    trace_id: object,
    schema_version: object,
    domain: object,
    task_template_id: object,
    worker_policy_id: object,
    worker_role: object,
    world_hash: object,
    narration_mode: object,
    events: object,
    terminal_state_hash: object,
    local_task_outcome: object,
    hidden_benchmark_metadata_ref: object = None,
) -> ActionTrace:
    return ActionTrace.model_validate(
        {
            "trace_id": trace_id,
            "schema_version": schema_version,
            "domain": domain,
            "task_template_id": task_template_id,
            "worker_policy_id": worker_policy_id,
            "worker_role": worker_role,
            "world_hash": world_hash,
            "narration_mode": narration_mode,
            "events": events,
            "terminal_state_hash": terminal_state_hash,
            "local_task_outcome": local_task_outcome,
            "hidden_benchmark_metadata_ref": hidden_benchmark_metadata_ref,
        }
    )


def parse_action_trace(value: object) -> ActionTrace:
    return _validated_trace(value)


def _event_projection_trusted(event: TraceEvent) -> JsonObject:
    return {
        "event_id": event.event_id,
        "index": event.index,
        "kind": event.kind,
        "payload": event.payload,
    }


def _trace_projection_trusted(trace: ActionTrace) -> JsonObject:
    projection: JsonObject = {
        "trace_id": trace.trace_id,
        "schema_version": trace.schema_version,
        "domain": trace.domain,
        "task_template_id": trace.task_template_id,
        "worker_policy_id": trace.worker_policy_id,
        "worker_role": trace.worker_role,
        "world_hash": trace.world_hash,
        "narration_mode": trace.narration_mode,
        "events": [_event_projection_trusted(event) for event in trace.events],
        "terminal_state_hash": trace.terminal_state_hash,
        "local_task_outcome": trace.local_task_outcome,
    }
    if trace.hidden_benchmark_metadata_ref is not None:
        projection["hidden_benchmark_metadata_ref"] = trace.hidden_benchmark_metadata_ref
    return projection


def action_trace_projection(trace: object) -> JsonObject:
    validated = _validated_trace(trace)
    projection = _trace_projection_trusted(validated)
    events = cast(list[JsonObject], projection["events"])
    payloads = _detach_payloads([event["payload"] for event in events])
    projection["events"] = [
        cast(JsonValue, {**event, "payload": payload})
        for event, payload in zip(events, payloads, strict=True)
    ]
    return projection


def hash_action_trace(trace: object) -> str:
    return sha256_ref(action_trace_projection(trace))


__all__ = [
    "TraceEvent",
    "ActionTrace",
    "make_trace_event",
    "make_action_trace",
    "parse_action_trace",
    "action_trace_projection",
    "hash_action_trace",
]
