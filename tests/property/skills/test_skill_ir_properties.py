from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.skills import hash_skill_ir, parse_skill_ir, skill_ir_projection


@st.composite
def skill_values(draw: st.DrawFn) -> dict[str, object]:
    domain = draw(st.sampled_from(["access_provisioning", "financial_adjustments"]))
    generated = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=24))
    text = f"sample {generated.strip()}"
    traces = draw(
        st.lists(
            st.from_regex(r"trace_[a-z0-9]{1,8}", fullmatch=True),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    return {
        "skill_id": "skill_property_test",
        "schema_version": "1.0",
        "domain": domain,
        "objective": text,
        "applicability": [text],
        "required_inputs": [],
        "preconditions": [],
        "ordered_steps": [
            {
                "step_id": "step_one",
                "action_intent": text,
                "tool_name": "inspect",
                "argument_bindings": {"text": text, "items": [1, True, None]},
                "preconditions": [],
                "optional": False,
                "evidence_refs": [],
            }
        ],
        "decision_hints": [],
        "verification_steps": [],
        "stop_conditions": [],
        "escalation_hints": [],
        "source_trace_ids": traces,
        "instruction_provenance": {"text": text},
        "compiler_manifest_ref": "sha256:" + "a" * 64,
    }


@given(value=skill_values())
@settings(max_examples=24, deadline=None)
@pytest.mark.property
def test_parse_projection_and_hash_are_deterministic_and_detached(value: dict[str, object]) -> None:
    first = parse_skill_ir(value)
    second = parse_skill_ir(value)
    projection = skill_ir_projection(first)

    assert first == second
    assert projection == skill_ir_projection(second)
    assert hash_skill_ir(first) == hash_skill_ir(second)
    projection["objective"] = "changed"
    assert skill_ir_projection(first)["objective"] != "changed"


@given(value=skill_values())
@settings(max_examples=20, deadline=None)
@pytest.mark.property
def test_array_order_is_hash_sensitive_while_json_object_order_is_not(
    value: dict[str, object],
) -> None:
    reordered_object = dict(value)
    reordered_object["instruction_provenance"] = {"second": 2, "first": 1}
    object_reordered_again = dict(value)
    object_reordered_again["instruction_provenance"] = {"first": 1, "second": 2}
    assert hash_skill_ir(reordered_object) == hash_skill_ir(object_reordered_again)

    changed_array = dict(value)
    changed_array["applicability"] = ["second", "first"]
    assert hash_skill_ir(value) != hash_skill_ir(changed_array)
