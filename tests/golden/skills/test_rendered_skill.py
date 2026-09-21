from __future__ import annotations

import pytest

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.skills import hash_rendered_skill, render_skill, rendered_skill_bytes

HASH = "sha256:" + "b" * 64


def access_skill() -> dict[str, object]:
    return {
        "skill_id": "skill_access_review",
        "schema_version": "1.0",
        "domain": "access_provisioning",
        "objective": "Review *access* safely.",
        "applicability": ["Use for [visible] requests."],
        "required_inputs": [],
        "preconditions": ["Request `REQ-1` is available."],
        "ordered_steps": [
            {
                "step_id": "review_request",
                "action_intent": "Review [requested] access.",
                "tool_name": "inspect_request",
                "argument_bindings": {"scope": ["read"], "request_id": "REQ-1"},
                "preconditions": [],
                "optional": False,
                "evidence_refs": ["not rendered"],
            }
        ],
        "decision_hints": [],
        "verification_steps": ["Confirm status."],
        "stop_conditions": [],
        "escalation_hints": ["Escalate <unclear> scope."],
        "source_trace_ids": ["trace_access_01"],
        "instruction_provenance": {"source": "not rendered"},
        "compiler_manifest_ref": HASH,
    }


def finance_skill() -> dict[str, object]:
    return {
        "skill_id": "skill_finance_adjustment",
        "schema_version": "1.0",
        "domain": "financial_adjustments",
        "objective": "Verify an adjustment.",
        "applicability": [],
        "required_inputs": ["Adjustment identifier."],
        "preconditions": [],
        "ordered_steps": [
            {
                "step_id": "inspect_adjustment",
                "action_intent": "Inspect the adjustment.",
                "tool_name": "inspect_adjustment",
                "argument_bindings": {"adjustment_id": "ADJ-7", "amount": 12},
                "preconditions": ["The record exists."],
                "optional": True,
                "evidence_refs": [],
            }
        ],
        "decision_hints": ["Do not infer approval."],
        "verification_steps": [],
        "stop_conditions": ["Stop if amount differs."],
        "escalation_hints": [],
        "source_trace_ids": ["trace_finance_01"],
        "instruction_provenance": {"source": "not rendered"},
        "compiler_manifest_ref": HASH,
    }


def test_access_render_is_exact_golden_bytes() -> None:
    expected = """# skill\\_access\\_review

Domain: access\\_provisioning

Review \\*access\\* safely.

## When to use

- Use for \\[visible\\] requests.

## Required inputs

- None.

## Preconditions

- Request \\`REQ-1\\` is available.

## Steps

### 1. Review \\[requested\\] access.

Tool: inspect\\_request
Optional: no
Preconditions:
- None.
Arguments:
    {"request_id":"REQ-1","scope":["read"]}

## Decision hints

- None.

## Verification

- Confirm status.

## Stop conditions

- None.

## Escalation

- Escalate \\<unclear\\> scope.
"""
    assert render_skill(access_skill()) == expected
    assert rendered_skill_bytes(access_skill()) == expected.encode("utf-8")
    assert (
        hash_rendered_skill(access_skill())
        == "sha256:cae0c0f6fe031d47b3db7199c422cf0bfc5b379a3062ef123da59f08e5eb6e66"
    )
    assert expected.endswith("\n") and not expected.endswith("\n\n") and "\r" not in expected


def test_finance_render_is_exact_golden_bytes() -> None:
    expected = """# skill\\_finance\\_adjustment

Domain: financial\\_adjustments

Verify an adjustment.

## When to use

- None.

## Required inputs

- Adjustment identifier.

## Preconditions

- None.

## Steps

### 1. Inspect the adjustment.

Tool: inspect\\_adjustment
Optional: yes
Preconditions:
- The record exists.
Arguments:
    {"adjustment_id":"ADJ-7","amount":12}

## Decision hints

- Do not infer approval.

## Verification

- None.

## Stop conditions

- Stop if amount differs.

## Escalation

- None.
"""
    assert render_skill(finance_skill()) == expected
    assert rendered_skill_bytes(finance_skill()) == expected.encode("utf-8")
    assert (
        hash_rendered_skill(finance_skill())
        == "sha256:243c37495d26381b6fca3924d330649884714b5139c4a39908fda231de5bb50c"
    )


def test_render_excludes_references_and_hash_uses_exact_profile() -> None:
    value = access_skill()
    rendered = render_skill(value)
    assert "trace_access_01" not in rendered
    assert "not rendered" not in rendered
    assert HASH not in rendered
    assert hash_rendered_skill(value) == sha256_ref(
        {"profile": "SSB-SKILLMD1", "rendered_text": rendered}
    )


def test_render_changes_for_visible_fields_but_not_excluded_reference_fields() -> None:
    baseline = render_skill(access_skill())
    visible_changes = {
        "skill_id": "skill_access_changed",
        "domain": "financial_adjustments",
        "objective": "Different objective.",
        "applicability": ["Different use."],
        "required_inputs": ["Different input."],
        "preconditions": ["Different precondition."],
        "decision_hints": ["Different decision."],
        "verification_steps": ["Different verification."],
        "stop_conditions": ["Different stop."],
        "escalation_hints": ["Different escalation."],
    }
    for field, changed_value in visible_changes.items():
        changed = access_skill()
        changed[field] = changed_value
        assert render_skill(changed) != baseline, field
    for step_field, changed_value in {
        "action_intent": "Different intent.",
        "tool_name": "different_tool",
        "argument_bindings": {"different": True},
        "preconditions": ["Different step precondition."],
        "optional": True,
    }.items():
        changed = access_skill()
        changed_steps = changed["ordered_steps"]
        assert type(changed_steps) is list and type(changed_steps[0]) is dict
        changed_steps[0][step_field] = changed_value
        assert render_skill(changed) != baseline, step_field
    hidden = access_skill()
    hidden["source_trace_ids"] = ["trace_access_99"]
    hidden["instruction_provenance"] = {"source": "changed"}
    hidden["compiler_manifest_ref"] = "sha256:" + "c" * 64
    hidden_step = hidden["ordered_steps"]
    assert type(hidden_step) is list
    assert type(hidden_step[0]) is dict
    hidden_step[0]["evidence_refs"] = ["changed"]
    assert render_skill(hidden) == baseline


@pytest.mark.parametrize("objective", ["line one\nline two", "nul\x00text"])
def test_render_rejects_unsafe_dynamic_markdown_text(objective: str) -> None:
    value = access_skill()
    value["objective"] = objective
    with pytest.raises(ValueError):
        render_skill(value)
