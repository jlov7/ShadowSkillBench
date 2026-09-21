from __future__ import annotations

import copy
import hashlib
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.skills.contamination import (
    SkillSemanticRules,
    analyze_contamination,
    contamination_report_projection,
    hash_contamination_report,
    hash_semantic_rules,
    semantic_rules_projection,
)
from shadowskillbench.skills.models import SkillIR, parse_skill_ir


def _rules(domain: str) -> SkillSemanticRules:
    return SkillSemanticRules.model_validate(
        {
            "profile": "SSB-SKILL-SEMANTICS1",
            "domain": domain,
            "intents": [
                {
                    "intent_id": "safe_a",
                    "action_intent": "Inspect A",
                    "tool_name": "inspect",
                    "required_arguments": [{"key": "flag", "value": True}],
                    "forbidden_argument_keys": [],
                },
                {
                    "intent_id": "safe_b",
                    "action_intent": "Inspect B",
                    "tool_name": "inspect",
                    "required_arguments": [],
                    "forbidden_argument_keys": [],
                },
            ],
            "prohibited_direct_intent_ids": [],
            "approval_requirements": [],
            "prohibited_workaround_signatures": [],
            "conditionality_requirements": [],
            "verification_requirements": [],
            "safe_variant_intent_ids": ["safe_a", "safe_b"],
            "escalation_requirements": [],
        }
    )


def _skill(domain: str, steps: list[dict[str, object]]) -> SkillIR:
    return parse_skill_ir(
        {
            "skill_id": "skill_property",
            "schema_version": "1.0",
            "domain": domain,
            "objective": "Property test skill.",
            "applicability": [],
            "required_inputs": [],
            "preconditions": [],
            "ordered_steps": steps,
            "decision_hints": [],
            "verification_steps": [],
            "stop_conditions": [],
            "escalation_hints": [],
            "source_trace_ids": ["trace_property"],
            "instruction_provenance": {},
            "compiler_manifest_ref": "sha256:" + "b" * 64,
        }
    )


def _step(step_id: str, action: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "step_id": step_id,
        "action_intent": action,
        "tool_name": "inspect",
        "argument_bindings": arguments,
        "preconditions": [],
        "optional": False,
        "evidence_refs": [],
    }


def _order_matrix_rules() -> dict[str, object]:
    def intent(
        intent_id: str,
        action_intent: str,
        tool_name: str,
        *,
        required_arguments: list[dict[str, object]] | None = None,
        forbidden_argument_keys: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "intent_id": intent_id,
            "action_intent": action_intent,
            "tool_name": tool_name,
            "required_arguments": required_arguments or [],
            "forbidden_argument_keys": forbidden_argument_keys or [],
        }

    return {
        "profile": "SSB-SKILL-SEMANTICS1",
        "domain": "access_provisioning",
        "intents": [
            intent(
                "direct_1",
                "Direct one",
                "apply_one",
                required_arguments=[
                    {"key": "kind", "value": "one"},
                    {"key": "scope", "value": "standard"},
                ],
            ),
            intent("direct_2", "Direct two", "apply_two"),
            intent("work_1", "Work one", "work_one", forbidden_argument_keys=["deny_a", "deny_b"]),
            intent("work_2", "Work two", "work_two"),
            intent("condition_1", "Condition one", "condition_one"),
            intent("condition_2", "Condition two", "condition_two"),
            intent("approval_1", "Approval one", "approval_one"),
            intent("approval_2", "Approval two", "approval_two"),
            intent("guard_1", "Guard one", "guard_one"),
            intent("guard_2", "Guard two", "guard_two"),
            intent("auth_1", "Auth one", "auth_one"),
            intent("auth_2", "Auth two", "auth_two"),
            intent("local_1", "Local one", "local_one"),
            intent("local_2", "Local two", "local_two"),
            intent("escalate_1", "Escalate one", "escalate_one"),
            intent("escalate_2", "Escalate two", "escalate_two"),
        ],
        "prohibited_direct_intent_ids": ["direct_1", "direct_2"],
        "approval_requirements": [
            {
                "requirement_id": "approval_requirement_1",
                "trigger_intent_id": "direct_1",
                "acceptable_prior_intent_ids": ["approval_1", "approval_2"],
            },
            {
                "requirement_id": "approval_requirement_2",
                "trigger_intent_id": "direct_2",
                "acceptable_prior_intent_ids": ["approval_1", "approval_2"],
            },
        ],
        "prohibited_workaround_signatures": [
            {"signature_id": "signature_1", "ordered_intent_ids": ["work_1", "direct_1"]},
            {"signature_id": "signature_2", "ordered_intent_ids": ["work_2", "direct_2"]},
        ],
        "conditionality_requirements": [
            {
                "requirement_id": "conditional_requirement_1",
                "trigger_intent_id": "direct_1",
                "acceptable_prior_intent_ids": ["guard_1", "guard_2"],
            },
            {
                "requirement_id": "conditional_requirement_2",
                "trigger_intent_id": "direct_2",
                "acceptable_prior_intent_ids": ["guard_1", "guard_2"],
            },
        ],
        "verification_requirements": [
            {
                "requirement_id": "verification_requirement_1",
                "trigger_intent_id": "direct_1",
                "acceptable_following_intent_ids": ["auth_1", "auth_2"],
                "local_success_only_intent_ids": ["local_1", "local_2"],
            },
            {
                "requirement_id": "verification_requirement_2",
                "trigger_intent_id": "direct_2",
                "acceptable_following_intent_ids": ["auth_1", "auth_2"],
                "local_success_only_intent_ids": ["local_1", "local_2"],
            },
        ],
        "safe_variant_intent_ids": [
            "approval_1",
            "approval_2",
            "guard_1",
            "guard_2",
            "auth_1",
            "auth_2",
            "local_1",
            "local_2",
            "escalate_1",
            "escalate_2",
        ],
        "escalation_requirements": [
            {
                "requirement_id": "escalation_requirement_1",
                "condition_intent_id": "condition_1",
                "acceptable_following_intent_ids": ["escalate_1", "escalate_2"],
            },
            {
                "requirement_id": "escalation_requirement_2",
                "condition_intent_id": "condition_2",
                "acceptable_following_intent_ids": ["escalate_1", "escalate_2"],
            },
        ],
    }


def _list_value(parent: dict[str, object], key: str) -> list[object]:
    value = parent[key]
    assert type(value) is list
    return value


def _object_value(items: list[object], index: int) -> dict[str, object]:
    value = items[index]
    assert type(value) is dict
    return value


def _reverse_order_case(value: dict[str, object], case: str) -> None:
    if case in {
        "intents",
        "prohibited_direct_intent_ids",
        "approval_requirements",
        "prohibited_workaround_signatures",
        "conditionality_requirements",
        "verification_requirements",
        "safe_variant_intent_ids",
        "escalation_requirements",
    }:
        value[case] = list(reversed(_list_value(value, case)))
        return
    intents = _list_value(value, "intents")
    if case == "required_arguments":
        direct = _object_value(intents, 0)
        direct[case] = list(reversed(_list_value(direct, case)))
        return
    if case == "forbidden_argument_keys":
        workaround = _object_value(intents, 2)
        workaround[case] = list(reversed(_list_value(workaround, case)))
        return
    if case == "approval_acceptable_prior":
        approval = _object_value(_list_value(value, "approval_requirements"), 0)
        approval["acceptable_prior_intent_ids"] = list(
            reversed(_list_value(approval, "acceptable_prior_intent_ids"))
        )
        return
    if case == "conditional_acceptable_prior":
        conditional = _object_value(_list_value(value, "conditionality_requirements"), 0)
        conditional["acceptable_prior_intent_ids"] = list(
            reversed(_list_value(conditional, "acceptable_prior_intent_ids"))
        )
        return
    if case == "signature_ordered_intent_ids":
        signature = _object_value(_list_value(value, "prohibited_workaround_signatures"), 0)
        signature["ordered_intent_ids"] = list(
            reversed(_list_value(signature, "ordered_intent_ids"))
        )
        return
    if case in {"verification_acceptable_following", "verification_local_success_only"}:
        verification = _object_value(_list_value(value, "verification_requirements"), 0)
        key = (
            "acceptable_following_intent_ids"
            if case == "verification_acceptable_following"
            else "local_success_only_intent_ids"
        )
        verification[key] = list(reversed(_list_value(verification, key)))
        return
    if case == "escalation_acceptable_following":
        escalation = _object_value(_list_value(value, "escalation_requirements"), 0)
        escalation["acceptable_following_intent_ids"] = list(
            reversed(_list_value(escalation, "acceptable_following_intent_ids"))
        )
        return
    raise AssertionError(f"unhandled order case: {case}")


@given(domain=st.sampled_from(["access_provisioning", "financial_adjustments"]), flag=st.booleans())
@settings(max_examples=16, deadline=None)
@pytest.mark.property
def test_rules_hash_matches_independent_stdlib_canonical_json(domain: str, flag: bool) -> None:
    rules = _rules(domain)
    projection = semantic_rules_projection(rules)
    independent = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert hash_semantic_rules(rules) == independent
    report = analyze_contamination(
        _skill(domain, [_step("step", "Inspect A", {"flag": flag})]), rules
    )
    report_projection = contamination_report_projection(report)
    report_independent = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                report_projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert hash_contamination_report(report) == report_independent
    assert report.status == ("CLEAN" if flag else "UNCLASSIFIABLE")


@given(order=st.permutations(["safe_a", "safe_b"]))
@settings(max_examples=2, deadline=None)
@pytest.mark.property
def test_safe_variant_order_is_hash_sensitive(order: tuple[str, str]) -> None:
    first = semantic_rules_projection(_rules("access_provisioning"))
    second = semantic_rules_projection(_rules("access_provisioning"))
    first["safe_variant_intent_ids"] = list(order)
    second["safe_variant_intent_ids"] = list(reversed(order))
    assert hash_semantic_rules(first) != hash_semantic_rules(second)


@pytest.mark.parametrize("path", ["intents", "required_arguments"])
def test_top_level_and_nested_rule_array_order_are_hash_sensitive(path: str) -> None:
    first = semantic_rules_projection(_rules("access_provisioning"))
    second = semantic_rules_projection(_rules("access_provisioning"))
    if path == "intents":
        intents = second["intents"]
        assert type(intents) is list
        second["intents"] = list(reversed(intents))
    else:
        for projection in (first, second):
            intents = projection["intents"]
            assert type(intents) is list and type(intents[0]) is dict
            intents[0]["required_arguments"] = [
                {"key": "flag", "value": True},
                {"key": "other", "value": "value"},
            ]
        intents = second["intents"]
        assert type(intents) is list and type(intents[0]) is dict
        required = intents[0]["required_arguments"]
        assert type(required) is list
        intents[0]["required_arguments"] = list(reversed(required))
    assert hash_semantic_rules(first) != hash_semantic_rules(second)


@pytest.mark.parametrize(
    "case",
    [
        "intents",
        "prohibited_direct_intent_ids",
        "approval_requirements",
        "prohibited_workaround_signatures",
        "conditionality_requirements",
        "verification_requirements",
        "safe_variant_intent_ids",
        "escalation_requirements",
        "required_arguments",
        "forbidden_argument_keys",
        "approval_acceptable_prior",
        "conditional_acceptable_prior",
        "signature_ordered_intent_ids",
        "verification_acceptable_following",
        "verification_local_success_only",
        "escalation_acceptable_following",
    ],
)
def test_every_reorderable_rules_array_is_hash_sensitive_when_both_orders_are_valid(
    case: str,
) -> None:
    first = _order_matrix_rules()
    second = copy.deepcopy(first)
    _reverse_order_case(second, case)
    assert SkillSemanticRules.model_validate(first)
    assert SkillSemanticRules.model_validate(second)
    assert hash_semantic_rules(first) != hash_semantic_rules(second)


@given(domain=st.sampled_from(["access_provisioning", "financial_adjustments"]))
@settings(max_examples=8, deadline=None)
@pytest.mark.property
def test_analysis_is_repeat_run_deterministic(domain: str) -> None:
    rules = _rules(domain)
    value = _skill(domain, [_step("step", "Inspect A", {"flag": True})])
    first = analyze_contamination(value, rules)
    second = analyze_contamination(value, rules)
    assert contamination_report_projection(first) == contamination_report_projection(second)
    assert hash_contamination_report(first) == hash_contamination_report(second)
