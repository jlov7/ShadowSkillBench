from __future__ import annotations

from typing import cast

from shadowskillbench.traces.models import (
    ActionTrace,
    TraceEvent,
    make_action_trace,
    make_trace_event,
)

_TRACE_FIELDS = (
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
_EVENT_FIELDS = ("event_id", "index", "kind", "payload")
_MAX_EVENTS = 10_000
_MAX_DEPTH = 64
_MAX_VALUES = 100_000


def _exact_model_data(
    value: object, expected: type[ActionTrace] | type[TraceEvent], fields: tuple[str, ...]
) -> dict[str, object]:
    if type(value) is not expected:
        raise ValueError(f"trace ingress requires an exact {expected.__name__}")
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
    if any(type(key) is not str for key in raw) or any(field not in raw for field in fields):
        raise ValueError(f"{expected.__name__} has an invalid key tree")
    if (
        type(fields_set) is not set
        or len(fields_set) != len(fields)
        or any(type(field) is not str for field in fields_set)
        or any(field not in fields_set for field in fields)
    ):
        raise ValueError(f"{expected.__name__} has an invalid field set")
    return {field: raw[field] for field in fields}


def _require_exact_payload_tree(
    value: object, *, seen_containers: set[int], value_count: list[int]
) -> None:
    if type(value) is not dict:
        raise ValueError("event payload must be an exact built-in JSON object")
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        value_count[0] += 1
        if value_count[0] > _MAX_VALUES:
            raise ValueError("payload values exceed the trace budget")
        current_type = type(current)
        if current_type in {type(None), bool, int, float, str}:
            continue
        if current_type not in {dict, list}:
            raise ValueError("payloads must contain exact built-in JSON values")
        if depth > _MAX_DEPTH:
            raise ValueError("payload depth exceeds the trace budget")
        container = cast(dict[object, object] | list[object], current)
        container_id = id(container)
        if container_id in seen_containers:
            raise ValueError("payload containers must not be cyclic or aliased")
        seen_containers.add(container_id)
        remaining_unqueued = _MAX_VALUES - value_count[0] - len(stack)
        if len(container) > remaining_unqueued:
            raise ValueError("payload children exceed the remaining trace budget")
        if current_type is list:
            stack.extend((item, depth + 1) for item in cast(list[object], container))
            continue
        for key, item in cast(dict[object, object], container).items():
            if type(key) is not str:
                raise ValueError("payload object keys must be exact strings")
            stack.append((item, depth + 1))


def _source_is_silent(mode: str, kinds: list[str]) -> bool:
    return mode == "none" and all(kind != "narration" for kind in kinds)


def _revalidated_silent_source(trace: object) -> ActionTrace:
    raw = _exact_model_data(trace, ActionTrace, _TRACE_FIELDS)
    events = raw["events"]
    if type(events) is not tuple or len(events) > _MAX_EVENTS:
        raise ValueError("ActionTrace events must be an exact bounded built-in tuple")
    seen_events: set[int] = set()
    seen_containers: set[int] = set()
    value_count = [0]
    kinds: list[str] = []
    for event in events:
        if id(event) in seen_events:
            raise ValueError("trace events must not be aliased")
        seen_events.add(id(event))
        event_data = _exact_model_data(event, TraceEvent, _EVENT_FIELDS)
        _require_exact_payload_tree(
            event_data["payload"], seen_containers=seen_containers, value_count=value_count
        )
        if type(event_data["kind"]) is not str:
            raise ValueError("event kind must be an exact string")
        kinds.append(cast(str, event_data["kind"]))
    if type(raw["narration_mode"]) is not str:
        raise ValueError("source narration_mode must be an exact string")
    if not _source_is_silent(cast(str, raw["narration_mode"]), kinds):
        raise ValueError("source trace must be silent")
    return make_action_trace(
        trace_id=raw["trace_id"],
        schema_version=raw["schema_version"],
        domain=raw["domain"],
        task_template_id=raw["task_template_id"],
        worker_policy_id=raw["worker_policy_id"],
        worker_role=raw["worker_role"],
        world_hash=raw["world_hash"],
        narration_mode=raw["narration_mode"],
        events=events,
        terminal_state_hash=raw["terminal_state_hash"],
        local_task_outcome=raw["local_task_outcome"],
        hidden_benchmark_metadata_ref=raw["hidden_benchmark_metadata_ref"],
    )


def _narration_text(mode: object, text: object | None) -> tuple[str, str | None]:
    if type(mode) is not str or mode not in {"none", "neutral", "rationalized"}:
        raise ValueError("mode must be an exact declared narration mode")
    owned_mode = cast(str, mode)
    if owned_mode == "none":
        if text is not None:
            raise ValueError("text must be None when mode is none")
        return owned_mode, None
    if (
        type(text) is not str
        or not cast(str, text).strip()
        or "\x00" in cast(str, text)
        or len(cast(str, text)) > 4_096
    ):
        raise ValueError("narration text must be an exact nonblank bounded string")
    owned_text = cast(str, text)
    try:
        owned_text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("narration text must be UTF-8 encodable") from error
    return owned_mode, owned_text


def narrate_trace(
    trace: object,
    *,
    mode: object = "none",
    text: object | None = None,
) -> ActionTrace:
    """Return a detached exploratory narration variant of one silent source trace."""
    owned_mode, owned_text = _narration_text(mode, text)
    source = _revalidated_silent_source(trace)
    if owned_mode == "none":
        return source
    events: list[TraceEvent] = [
        make_trace_event(
            trace_id=source.trace_id,
            index=0,
            kind=source.events[0].kind,
            payload=source.events[0].payload,
        )
    ]
    events.append(
        make_trace_event(
            trace_id=source.trace_id,
            index=1,
            kind="narration",
            payload={"text": owned_text},
        )
    )
    events.extend(
        make_trace_event(
            trace_id=source.trace_id,
            index=index,
            kind=event.kind,
            payload=event.payload,
        )
        for index, event in enumerate(source.events[1:], start=2)
    )
    return make_action_trace(
        trace_id=source.trace_id,
        schema_version=source.schema_version,
        domain=source.domain,
        task_template_id=source.task_template_id,
        worker_policy_id=source.worker_policy_id,
        worker_role=source.worker_role,
        world_hash=source.world_hash,
        narration_mode=owned_mode,
        events=tuple(events),
        terminal_state_hash=source.terminal_state_hash,
        local_task_outcome=source.local_task_outcome,
        hidden_benchmark_metadata_ref=source.hidden_benchmark_metadata_ref,
    )


__all__ = ["narrate_trace"]
