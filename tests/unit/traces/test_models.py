from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from shadowskillbench.traces.models import (
    ActionTrace,
    TraceEvent,
    action_trace_projection,
    hash_action_trace,
    make_action_trace,
    make_trace_event,
    parse_action_trace,
)

HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64


def _event(index: int, kind: str, payload: dict[str, Any]) -> TraceEvent:
    return make_trace_event(trace_id="trace_access_01", index=index, kind=kind, payload=payload)


def _events(
    *, narration: bool = False, payload: dict[str, Any] | None = None
) -> tuple[TraceEvent, ...]:
    evidence = payload if payload is not None else {"observation": "request"}
    result = [_event(0, "observation", evidence)]
    if narration:
        result.append(_event(1, "narration", {"text": "The request needs review."}))
    offset = len(result)
    result.extend(
        (
            _event(offset, "action", {"tool": "submit"}),
            _event(offset + 1, "tool_result", {"status": "ok"}),
            _event(offset + 2, "state_delta", {"path": "/status", "after": "submitted"}),
        )
    )
    return tuple(result)


def _trace(**overrides: object) -> ActionTrace:
    values: dict[str, object] = {
        "trace_id": "trace_access_01",
        "schema_version": "1.0",
        "domain": "access_provisioning",
        "task_template_id": "access_request",
        "worker_policy_id": "policy_follow_authority",
        "worker_role": "access_analyst",
        "world_hash": HASH,
        "narration_mode": "none",
        "events": _events(),
        "terminal_state_hash": OTHER_HASH,
        "local_task_outcome": "completed",
        "hidden_benchmark_metadata_ref": None,
    }
    values.update(overrides)
    return make_action_trace(**values)


def test_round_trip_projection_and_fixed_hash_vector() -> None:
    trace = _trace(hidden_benchmark_metadata_ref=HASH)

    projection = action_trace_projection(trace)

    assert projection == {
        "trace_id": "trace_access_01",
        "schema_version": "1.0",
        "domain": "access_provisioning",
        "task_template_id": "access_request",
        "worker_policy_id": "policy_follow_authority",
        "worker_role": "access_analyst",
        "world_hash": HASH,
        "narration_mode": "none",
        "events": [
            {
                "event_id": "event_trace_access_01_000000",
                "index": 0,
                "kind": "observation",
                "payload": {"observation": "request"},
            },
            {
                "event_id": "event_trace_access_01_000001",
                "index": 1,
                "kind": "action",
                "payload": {"tool": "submit"},
            },
            {
                "event_id": "event_trace_access_01_000002",
                "index": 2,
                "kind": "tool_result",
                "payload": {"status": "ok"},
            },
            {
                "event_id": "event_trace_access_01_000003",
                "index": 3,
                "kind": "state_delta",
                "payload": {"path": "/status", "after": "submitted"},
            },
        ],
        "terminal_state_hash": OTHER_HASH,
        "local_task_outcome": "completed",
        "hidden_benchmark_metadata_ref": HASH,
    }
    assert (
        hash_action_trace(trace)
        == "sha256:ae106cc71d00de086579a6af5bd953304a87a7cc3b20dde4af6e7b3ae12c131d"
    )
    assert parse_action_trace(projection) == trace


def test_schema_compatibility_and_timestamp_absence() -> None:
    trace = _trace()
    projection = action_trace_projection(trace)
    schema = json.loads(
        (Path(__file__).parents[3] / "schemas" / "action_trace.schema.json").read_text()
    )

    Draft202012Validator(schema).validate(projection)
    assert "hidden_benchmark_metadata_ref" not in projection
    events = cast(list[dict[str, object]], projection["events"])
    assert all("timestamp_ms" not in event for event in events)

    timestamped = cast(dict[str, object], json.loads(json.dumps(projection)))
    cast(list[dict[str, object]], timestamped["events"])[0]["timestamp_ms"] = 0
    with pytest.raises((ValidationError, ValueError)):
        parse_action_trace(timestamped)


def test_object_key_order_is_invariant_and_event_order_is_sensitive() -> None:
    left = _trace(events=(_event(0, "observation", {"a": 1, "b": [2, 3]}),))
    right = _trace(events=(_event(0, "observation", {"b": [2, 3], "a": 1}),))

    assert hash_action_trace(left) == hash_action_trace(right)

    first_order = _trace(
        events=(
            _event(0, "observation", {"observation": "request"}),
            _event(1, "action", {"tool": "first"}),
            _event(2, "tool_result", {"status": "first"}),
            _event(3, "state_delta", {"path": "/first", "after": 1}),
            _event(4, "action", {"tool": "second"}),
            _event(5, "tool_result", {"status": "second"}),
            _event(6, "state_delta", {"path": "/second", "after": 2}),
        ),
    )
    second_order = _trace(
        events=(
            _event(0, "observation", {"observation": "request"}),
            _event(1, "action", {"tool": "second"}),
            _event(2, "tool_result", {"status": "second"}),
            _event(3, "state_delta", {"path": "/second", "after": 2}),
            _event(4, "action", {"tool": "first"}),
            _event(5, "tool_result", {"status": "first"}),
            _event(6, "state_delta", {"path": "/first", "after": 1}),
        ),
    )
    assert hash_action_trace(first_order) != hash_action_trace(second_order)


def test_models_are_frozen_strict_and_detached() -> None:
    payload: dict[str, Any] = {"nested": {"before": "value"}}
    event = _event(0, "observation", payload)
    payload["nested"]["before"] = "changed"

    assert event.payload == {"nested": {"before": "value"}}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(
            {
                "event_id": event.event_id,
                "index": 0,
                "kind": "observation",
                "payload": {},
                "extra": True,
            }
        )
    with pytest.raises(ValidationError):
        event.index = 1


@pytest.mark.parametrize(
    "events",
    [
        (_event(0, "action", {"tool": "submit"}),),
        (_event(0, "observation", {}), _event(1, "action", {})),
        (_event(0, "observation", {}), _event(1, "tool_result", {}), _event(2, "state_delta", {})),
    ],
)
def test_sequence_grammar_is_not_repaired(events: tuple[TraceEvent, ...]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=events)


def test_narration_interleaves_but_cannot_substitute_for_evidence() -> None:
    trace = _trace(events=_events(narration=True), narration_mode="neutral")
    assert [event.kind for event in trace.events] == [
        "observation",
        "narration",
        "action",
        "tool_result",
        "state_delta",
    ]
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=(_event(0, "observation", {}), _event(1, "narration", {"text": "note"})))
    with pytest.raises((ValidationError, ValueError)):
        _trace(
            events=(
                _event(0, "narration", {"text": "preface"}),
                _event(1, "observation", {}),
                _event(2, "action", {}),
                _event(3, "tool_result", {}),
                _event(4, "state_delta", {}),
            ),
            narration_mode="neutral",
        )
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=_events(narration=True), narration_mode="none")
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=_events(), narration_mode="neutral")
    with pytest.raises((ValidationError, ValueError)):
        make_trace_event(
            trace_id="trace_access_01", index=1, kind="narration", payload={"text": "", "x": 1}
        )


def test_required_mode_id_and_key_limits() -> None:
    raw = action_trace_projection(_trace())
    raw.pop("narration_mode")
    with pytest.raises((ValidationError, ValueError)):
        parse_action_trace(raw)
    with pytest.raises((ValidationError, ValueError)):
        _trace(trace_id="trace_" + "a" * 123)
    with pytest.raises((ValidationError, ValueError)):
        _trace(task_template_id="a" * 129)
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=(_event(0, "observation", {"k" * 129: 1}),))


def _nested_payload(depth: int) -> dict[str, object]:
    value: object = "leaf"
    for _ in range(depth):
        value = [value]
    return {"value": value}


def test_exact_json_budget_boundaries() -> None:
    _trace(events=(_event(0, "observation", _nested_payload(64)),))
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=(_event(0, "observation", _nested_payload(65)),))

    _trace(events=(_event(0, "observation", {"n": (1 << 256) - 1}),))
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=(_event(0, "observation", {"n": 1 << 256}),))

    _trace(events=(_event(0, "observation", {"x": "x" * 999_999}),))
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=(_event(0, "observation", {"x": "x" * 1_000_000}),))

    under_nodes = {str(index): 0 for index in range(99_999)}
    _trace(events=(_event(0, "observation", under_nodes),))
    over_nodes = {str(index): 0 for index in range(100_000)}
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=(_event(0, "observation", over_nodes),))


class _Mapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError("custom mapping hook was called")

    def __iter__(self):
        raise AssertionError("custom mapping hook was called")

    def __len__(self) -> int:
        raise AssertionError("custom mapping hook was called")


def test_hostile_cyclic_aliased_and_forged_inputs_are_rejected_without_hooks() -> None:
    with pytest.raises((ValidationError, ValueError)):
        _event(0, "observation", cast(dict[str, Any], _Mapping()))
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises((ValidationError, ValueError)):
        _event(0, "observation", cyclic)
    child: list[object] = []
    with pytest.raises((ValidationError, ValueError)):
        _event(0, "observation", {"a": child, "b": child})
    with pytest.raises((ValidationError, ValueError)):
        _event(0, "observation", {"nan": math.nan})

    forged = TraceEvent.model_construct(
        event_id="event_trace_access_01_000000",
        index=0,
        kind="observation",
        payload={"bad": object()},
    )
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=(forged,))
    valid_projection = cast(dict[str, Any], action_trace_projection(_trace()))
    forged_trace = ActionTrace.model_construct(**valid_projection)
    object.__setattr__(forged_trace, "events", (forged,))
    with pytest.raises((ValidationError, ValueError)):
        hash_action_trace(forged_trace)


def test_huge_forged_key_trees_fail_before_key_set_construction() -> None:
    extra_fields = {f"extra_{index}": 0 for index in range(100_001)}

    raw = cast(dict[str, object], action_trace_projection(_trace()))
    raw.update(extra_fields)
    with pytest.raises((ValidationError, ValueError)):
        parse_action_trace(raw)

    forged_event = TraceEvent.model_construct(
        event_id="event_trace_access_01_000000",
        index=0,
        kind="observation",
        payload={},
    )
    object.__getattribute__(forged_event, "__dict__").update(extra_fields)
    with pytest.raises((ValidationError, ValueError)):
        _trace(events=(forged_event,))

    forged_trace = ActionTrace.model_construct(
        **cast(dict[str, Any], action_trace_projection(_trace()))
    )
    object.__getattribute__(forged_trace, "__dict__").update(extra_fields)
    with pytest.raises((ValidationError, ValueError)):
        hash_action_trace(forged_trace)
