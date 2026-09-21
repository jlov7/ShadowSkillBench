from __future__ import annotations

from decimal import Decimal

import pytest

from shadowskillbench.skills import compiler_input_projection, compiler_view, hash_compiler_input
from shadowskillbench.traces.bundles import generate_bundle


def _raw_with_payload_array() -> dict[str, object]:
    traces: list[dict[str, object]] = []
    for number in range(12):
        trace_id = f"trace_property_{number}"
        events = [
            {
                "event_id": f"event_{trace_id}_000000",
                "index": 0,
                "kind": "observation",
                "payload": {"items": ["first", "second"]},
            }
        ]
        for index, kind in enumerate(("action", "tool_result", "state_delta"), start=1):
            events.append(
                {
                    "event_id": f"event_{trace_id}_{index:06d}",
                    "index": index,
                    "kind": kind,
                    "payload": {"visible": index},
                }
            )
        traces.append(
            {
                "trace_id": trace_id,
                "domain": "access_provisioning",
                "task_template_id": "task_property",
                "worker_role": "worker",
                "world_hash": "sha256:" + "a" * 64,
                "events": events,
                "terminal_state_hash": "sha256:" + "b" * 64,
                "local_task_outcome": "completed",
            }
        )
    return {
        "projection_profile": "SSB-COMPILER-VIEW1",
        "domain": "access_provisioning",
        "traces": traces,
    }


@pytest.mark.property
@pytest.mark.parametrize("domain", ["access_provisioning", "financial_adjustments"])
@pytest.mark.parametrize(
    "ratio", [Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1")]
)
def test_hash_preserves_trace_and_event_array_order(domain: str, ratio: Decimal) -> None:
    view = compiler_view(generate_bundle(domain, ratio, 12, 311).bundle)
    raw = compiler_input_projection(view)
    baseline = hash_compiler_input(raw)
    traces = raw["traces"]
    assert type(traces) is list
    reordered = dict(raw)
    reordered["traces"] = list(reversed(traces))
    assert hash_compiler_input(reordered) != baseline
    changed_event_order = compiler_input_projection(view)
    changed_traces = changed_event_order["traces"]
    assert type(changed_traces) is list and type(changed_traces[0]) is dict
    events = changed_traces[0]["events"]
    assert type(events) is list
    events[1], events[2] = events[2], events[1]
    with pytest.raises(ValueError):
        hash_compiler_input(changed_event_order)


@pytest.mark.property
def test_json_object_order_is_hash_invariant_but_payload_arrays_are_not() -> None:
    view = compiler_view(generate_bundle("access_provisioning", Decimal("0.5"), 12, 312).bundle)
    raw = compiler_input_projection(view)
    first = dict(raw)
    second = {
        "traces": raw["traces"],
        "domain": raw["domain"],
        "projection_profile": raw["projection_profile"],
    }
    assert hash_compiler_input(first) == hash_compiler_input(second)

    payload_array = _raw_with_payload_array()
    baseline = hash_compiler_input(payload_array)
    trace = payload_array["traces"]
    assert type(trace) is list and type(trace[0]) is dict
    events = trace[0]["events"]
    assert type(events) is list and type(events[0]) is dict
    payload = events[0]["payload"]
    assert type(payload) is dict and type(payload["items"]) is list
    payload["items"].reverse()
    assert hash_compiler_input(payload_array) != baseline
