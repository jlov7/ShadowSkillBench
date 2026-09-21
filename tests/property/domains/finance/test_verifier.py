from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.authority.models import (
    create_finance_authority_query,
    create_issuer_registry,
)
from shadowskillbench.authority.resolver import resolve_authority
from shadowskillbench.domains.finance.fixtures import FinanceFixtureVariant, build_finance_fixture
from shadowskillbench.domains.finance.models import parse_finance_world
from shadowskillbench.domains.finance.verifier import verify_finance_case


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_admitted_escalation_is_deterministic_without_target_state_inspection(seed: int) -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.CREATE_PENDING, seed=seed)
    world = parse_finance_world(fixture.initial_state)
    fact = world.control_facts[0]
    query = create_finance_authority_query(
        subject_id=fact.adjustment_id,
        resource_id=fact.portco_id,
        action_type="post_adjustment",
        at_time="2026-01-15T00:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[],
        issuer_registry=create_issuer_registry(authorizations=[]),
        category_id=fact.original_economic_category,
        period_id=fact.period_id,
        amount_minor=fact.signed_amount_minor,
        currency=fact.currency,
        unit=fact.unit,
        currency_exponent=fact.currency_exponent,
    )
    decision = resolve_authority([], query)
    first = verify_finance_case(world, fixture.case, authority_decision=decision)
    second = verify_finance_case(world, fixture.case, authority_decision=decision)
    assert first == second
    assert first.status == "ESCALATE"
    assert first.reason_code == "AUTHORITY_ESCALATION"
