from __future__ import annotations

import ast
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

import shadowskillbench.traces.narration as narration_module
from shadowskillbench.traces.bundles import DemonstrationBundle, generate_bundle
from shadowskillbench.traces.models import (
    ActionTrace,
    TraceEvent,
    action_trace_projection,
    hash_action_trace,
    make_action_trace,
    make_trace_event,
)
from shadowskillbench.traces.narration import narrate_trace

HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64


def _event(index: int, kind: str, payload: dict[str, Any]) -> TraceEvent:
    return make_trace_event(trace_id="trace_narration_01", index=index, kind=kind, payload=payload)


def _trace(**overrides: object) -> ActionTrace:
    values: dict[str, object] = {
        "trace_id": "trace_narration_01",
        "schema_version": "1.0",
        "domain": "access_provisioning",
        "task_template_id": "access_request",
        "worker_policy_id": "access_compliant_v1",
        "worker_role": "access_operator",
        "world_hash": HASH,
        "narration_mode": "none",
        "events": (
            _event(0, "observation", {"request": {"subject": "alice"}}),
            _event(1, "action", {"tool": "inspect"}),
            _event(2, "tool_result", {"status": "ok"}),
            _event(3, "state_delta", {"after": "reviewed"}),
        ),
        "terminal_state_hash": OTHER_HASH,
        "local_task_outcome": "completed",
        "hidden_benchmark_metadata_ref": HASH,
    }
    values.update(overrides)
    return make_action_trace(**values)


def _raw(value: object) -> dict[str, object]:
    return object.__getattribute__(value, "__dict__")


def test_default_none_is_a_distinct_detached_projection_and_hash_noop() -> None:
    source = _trace()
    result = narrate_trace(source)

    assert result is not source
    assert result.events[0] is not source.events[0]
    assert action_trace_projection(result) == action_trace_projection(source)
    assert hash_action_trace(result) == hash_action_trace(source)

    source.events[0].payload["request"] = {"subject": "mutated"}
    assert result.events[0].payload == {"request": {"subject": "alice"}}


@pytest.mark.parametrize("mode", ["neutral", "rationalized"])
def test_non_silent_narration_is_exactly_after_observation_and_rebuilds_identities(
    mode: str,
) -> None:
    source = _trace()
    source_projection = action_trace_projection(source)
    result = narrate_trace(source, mode=mode, text="Review the visible evidence first.")
    result_projection = action_trace_projection(result)

    assert result.narration_mode == mode
    assert result.trace_id == source.trace_id
    assert result.schema_version == source.schema_version
    assert result.domain == source.domain
    assert result.task_template_id == source.task_template_id
    assert result.worker_policy_id == source.worker_policy_id
    assert result.worker_role == source.worker_role
    assert result.world_hash == source.world_hash
    assert result.terminal_state_hash == source.terminal_state_hash
    assert result.local_task_outcome == source.local_task_outcome
    assert result.hidden_benchmark_metadata_ref == source.hidden_benchmark_metadata_ref
    assert [event["kind"] for event in result_projection["events"]] == [
        "observation",
        "narration",
        "action",
        "tool_result",
        "state_delta",
    ]
    inserted = result_projection["events"][1]
    assert inserted == {
        "event_id": "event_trace_narration_01_000001",
        "index": 1,
        "kind": "narration",
        "payload": {"text": "Review the visible evidence first."},
    }
    assert [event["index"] for event in result_projection["events"]] == list(range(5))
    assert [event["event_id"] for event in result_projection["events"]] == [
        f"event_trace_narration_01_{index:06d}" for index in range(5)
    ]
    result_normative = [
        event for event in result_projection["events"] if event["kind"] != "narration"
    ]
    assert [event["kind"] for event in result_normative] == [
        event["kind"] for event in source_projection["events"]
    ]
    assert [event["payload"] for event in result_normative] == [
        event["payload"] for event in source_projection["events"]
    ]
    assert hash_action_trace(result) != hash_action_trace(source)
    assert source_projection == action_trace_projection(source)


def test_non_silent_narration_rebuilds_every_event_through_the_d019_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _trace()
    factory = narration_module.make_trace_event
    calls: list[dict[str, object]] = []

    def recording_factory(**kwargs: object) -> TraceEvent:
        calls.append(kwargs)
        return factory(**kwargs)

    monkeypatch.setattr(narration_module, "make_trace_event", recording_factory)
    result = narration_module.narrate_trace(source, mode="neutral", text="factory path")

    assert result.events[0] is not source.events[0]
    assert [call["index"] for call in calls] == [0, 1, 2, 3, 4]
    assert [call["kind"] for call in calls] == [
        "observation",
        "narration",
        "action",
        "tool_result",
        "state_delta",
    ]


def test_non_silent_output_is_deterministic_schema_valid_and_has_no_timestamp() -> None:
    first = narrate_trace(_trace(), mode="neutral", text="same stimulus")
    second = narrate_trace(_trace(), mode="neutral", text="same stimulus")
    projection = action_trace_projection(first)
    schema = json.loads(
        (Path(__file__).parents[3] / "schemas" / "action_trace.schema.json").read_text()
    )

    assert projection == action_trace_projection(second)
    assert hash_action_trace(first) == hash_action_trace(second)
    Draft202012Validator(schema).validate(projection)
    assert all("timestamp_ms" not in event for event in projection["events"])


def test_non_silent_mode_and_text_change_the_complete_trace_hash() -> None:
    neutral = narrate_trace(_trace(), mode="neutral", text="first stimulus")
    different_text = narrate_trace(_trace(), mode="neutral", text="second stimulus")
    rationalized = narrate_trace(_trace(), mode="rationalized", text="first stimulus")

    assert (
        len(
            {
                hash_action_trace(neutral),
                hash_action_trace(different_text),
                hash_action_trace(rationalized),
            }
        )
        == 3
    )


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        ("none", "unexpected"),
        ("neutral", None),
        ("neutral", ""),
        ("neutral", " \t\n"),
        ("neutral", "nul\x00text"),
        ("neutral", "x" * 4097),
        ("neutral", "\ud800"),
        ("other", "valid text"),
        (None, None),
        (True, None),
    ],
)
def test_mode_and_text_are_exact_and_bounded(mode: object, text: object) -> None:
    with pytest.raises(ValueError):
        narrate_trace(_trace(), mode=mode, text=text)


def test_subclassed_mode_and_text_are_rejected_without_coercion() -> None:
    class TextSubclass(str):
        pass

    with pytest.raises(ValueError):
        narrate_trace(_trace(), mode=TextSubclass("neutral"), text="text")
    with pytest.raises(ValueError):
        narrate_trace(_trace(), mode="neutral", text=TextSubclass("text"))


def test_source_must_be_exact_silent_trace_and_does_not_stack() -> None:
    with pytest.raises(ValueError):
        narrate_trace(action_trace_projection(_trace()))

    narrated = narrate_trace(_trace(), mode="neutral", text="already narrated")
    with pytest.raises(ValueError):
        narrate_trace(narrated, mode="rationalized", text="stacking is forbidden")

    forged = ActionTrace.model_construct(**_raw(_trace()))
    _raw(forged)["narration_mode"] = "none"
    _raw(forged)["events"] = (
        _event(0, "observation", {"request": "alice"}),
        _event(1, "narration", {"text": "forged"}),
        _event(2, "action", {"tool": "inspect"}),
        _event(3, "tool_result", {"status": "ok"}),
        _event(4, "state_delta", {"after": "reviewed"}),
    )
    with pytest.raises(ValueError):
        narrate_trace(forged)


def test_exact_trace_and_event_subclasses_are_rejected() -> None:
    class TraceSubclass(ActionTrace):
        pass

    class EventSubclass(TraceEvent):
        pass

    source = _trace()
    trace_subclass = TraceSubclass.model_construct(**_raw(source))
    with pytest.raises(ValueError):
        narrate_trace(trace_subclass)

    event_subclass = EventSubclass.model_construct(**_raw(source.events[0]))
    forged = ActionTrace.model_construct(**_raw(source))
    _raw(forged)["events"] = (event_subclass,) + source.events[1:]
    with pytest.raises(ValueError):
        narrate_trace(forged)


@pytest.mark.parametrize(
    "attribute",
    ["__pydantic_extra__", "__pydantic_private__", "__pydantic_fields_set__", "__dict__"],
)
def test_missing_internal_pydantic_state_is_a_value_error(attribute: str) -> None:
    forged = ActionTrace.model_construct(**_raw(_trace()))
    object.__delattr__(forged, attribute)
    with pytest.raises(ValueError):
        narrate_trace(forged)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda value: object.__setattr__(value, "__pydantic_private__", {"forged": True}),
        lambda value: object.__setattr__(value, "__pydantic_extra__", {"forged": True}),
        lambda value: object.__setattr__(value, "__pydantic_fields_set__", {"trace_id"}),
        lambda value: _raw(value).update({"unknown": 1}),
        lambda value: _raw(value).update({"events": list(value.events)}),
        lambda value: _raw(value).update({"events": (value.events[0],) * len(value.events)}),
    ],
)
def test_trace_ingress_rejects_pydantic_forgery_and_aliases(tamper: object) -> None:
    forged = ActionTrace.model_construct(**_raw(_trace()))
    cast(Any, tamper)(forged)
    with pytest.raises(ValueError):
        narrate_trace(forged)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda value: object.__setattr__(value, "__pydantic_private__", {"forged": True}),
        lambda value: object.__setattr__(value, "__pydantic_extra__", {"forged": True}),
        lambda value: object.__setattr__(value, "__pydantic_fields_set__", {"event_id"}),
        lambda value: _raw(value).update({"unknown": 1}),
        lambda value: _raw(value).update({"payload": {"forged": object()}}),
    ],
)
def test_nested_event_ingress_is_fully_revalidated(tamper: object) -> None:
    source = _trace()
    forged_event = TraceEvent.model_construct(**_raw(source.events[0]))
    cast(Any, tamper)(forged_event)
    forged = ActionTrace.model_construct(**_raw(source))
    _raw(forged)["events"] = (forged_event,) + source.events[1:]
    with pytest.raises(ValueError):
        narrate_trace(forged)


def test_payload_ingress_rejects_custom_containers_and_aliases_without_hooks() -> None:
    class HostileMapping(dict[str, object]):
        def items(self):
            raise AssertionError("caller hook invoked")

    source = _trace()
    forged_event = TraceEvent.model_construct(**_raw(source.events[0]))
    _raw(forged_event)["payload"] = HostileMapping()
    forged = ActionTrace.model_construct(**_raw(source))
    _raw(forged)["events"] = (forged_event,) + source.events[1:]
    with pytest.raises(ValueError):
        narrate_trace(forged)

    child: list[object] = []
    forged_event = TraceEvent.model_construct(**_raw(source.events[0]))
    _raw(forged_event)["payload"] = {"left": child, "right": child}
    _raw(forged)["events"] = (forged_event,) + source.events[1:]
    with pytest.raises(ValueError):
        narrate_trace(forged)


def test_payload_child_count_accounts_for_pending_work_before_stack_growth() -> None:
    source = _trace()
    forged_event = TraceEvent.model_construct(**_raw(source.events[0]))
    _raw(forged_event)["payload"] = {"sibling": 0, "wide": [0] * 99_998}
    forged = ActionTrace.model_construct(**_raw(source))
    _raw(forged)["events"] = (forged_event,) + source.events[1:]

    with pytest.raises(ValueError):
        narrate_trace(forged)


def test_narrated_trace_is_rejected_by_d021_demonstration_bundle() -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.5"), 12, 7)
    narrated = narrate_trace(generated.bundle.traces[0], mode="neutral", text="exploratory cue")
    hashes = list(generated.bundle.source_manifest.ordered_trace_hashes)
    hashes[0] = hash_action_trace(narrated)
    source_manifest = generated.bundle.source_manifest.model_copy(
        update={"ordered_trace_hashes": tuple(hashes)}
    )
    with pytest.raises((ValidationError, ValueError)):
        DemonstrationBundle.model_validate(
            {
                "source_manifest": source_manifest,
                "traces": (narrated,) + generated.bundle.traces[1:],
            }
        )


def test_narration_module_has_only_deterministic_source_imports() -> None:
    source = Path(__file__).parents[3] / "src" / "shadowskillbench" / "traces" / "narration.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(module) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all(
        isinstance(node, ast.ImportFrom)
        and node.module in {"__future__", "typing", "shadowskillbench.traces.models"}
        for node in imports
    )
