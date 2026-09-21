from __future__ import annotations

from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.traces.bundles import generate_bundle


@given(
    domain=st.sampled_from(["access_provisioning", "financial_adjustments"]),
    seed=st.integers(min_value=-(2**63), max_value=2**63 - 1),
)
@settings(max_examples=12, deadline=None)
def test_all_ratio_arms_preserve_paired_visible_positions(domain: str, seed: int) -> None:
    arms = [
        generate_bundle(domain, ratio, 12, seed)
        for ratio in map(Decimal, ["0", "0.25", "0.5", "0.75", "1"])
    ]  # type: ignore[arg-type]
    first = arms[0]
    assert len({binding.fixture_seed for binding in first.hidden_class_manifest.bindings}) == 12
    for arm in arms[1:]:
        assert (
            arm.hidden_class_manifest.class_slot_order
            == first.hidden_class_manifest.class_slot_order
        )
        assert (
            arm.hidden_class_manifest.final_trace_order
            == first.hidden_class_manifest.final_trace_order
        )
        assert [trace.trace_id for trace in arm.bundle.traces] == [
            trace.trace_id for trace in first.bundle.traces
        ]
        assert [trace.task_template_id for trace in arm.bundle.traces] == [
            trace.task_template_id for trace in first.bundle.traces
        ]
        assert [trace.world_hash for trace in arm.bundle.traces] == [
            trace.world_hash for trace in first.bundle.traces
        ]
        assert [trace.events[0].payload for trace in arm.bundle.traces] == [
            trace.events[0].payload for trace in first.bundle.traces
        ]
        assert [binding.fixture_seed for binding in arm.hidden_class_manifest.bindings] == [
            binding.fixture_seed for binding in first.hidden_class_manifest.bindings
        ]
