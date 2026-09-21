from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.domains.semantics import domain_skill_semantics
from shadowskillbench.skills.contamination import hash_semantic_rules, semantic_rules_projection


@settings(max_examples=12, deadline=None)
@given(st.sampled_from(["access_provisioning", "financial_adjustments"]))
def test_lookup_is_deterministic_and_hash_stable(domain: str) -> None:
    first = domain_skill_semantics(domain)  # type: ignore[arg-type]
    second = domain_skill_semantics(domain)  # type: ignore[arg-type]

    assert first == second
    assert first is not second
    assert semantic_rules_projection(first) == semantic_rules_projection(second)
    assert hash_semantic_rules(first) == hash_semantic_rules(second)


@settings(max_examples=12, deadline=None)
@given(st.sampled_from(["access_provisioning", "financial_adjustments"]))
def test_every_canonical_rule_has_unique_intents_and_no_dynamic_wildcards(domain: str) -> None:
    rules = domain_skill_semantics(domain)  # type: ignore[arg-type]
    intent_ids = tuple(intent.intent_id for intent in rules.intents)

    assert len(intent_ids) == len(set(intent_ids))
    assert all("*" not in intent_id for intent_id in intent_ids)
    assert all("*" not in intent.tool_name for intent in rules.intents)
    assert all("*" not in intent.action_intent for intent in rules.intents)
