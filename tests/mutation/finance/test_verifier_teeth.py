from __future__ import annotations

import tracemalloc
from typing import cast

import pytest

from shadowskillbench.authority.models import (
    AuthorityDecision,
    AuthoritySourceType,
    NormativeStatus,
    create_authority_record,
    create_finance_authority_query,
    create_issuer_registry,
)
from shadowskillbench.authority.resolver import resolve_authority
from shadowskillbench.domains.finance import verifier as finance_verifier
from shadowskillbench.domains.finance.adapter import FinanceAdapter
from shadowskillbench.domains.finance.fixtures import FinanceFixtureVariant, build_finance_fixture
from shadowskillbench.domains.finance.models import FinanceWorldState, parse_finance_world
from shadowskillbench.domains.finance.verifier import FinanceVerifierInputError, verify_finance_case
from shadowskillbench.engine.models import ActionCall, JsonObject, WorldState


def _escalation(state: object) -> AuthorityDecision:
    world = parse_finance_world(state)
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
    return resolve_authority([], query)


def _proceed(state: WorldState) -> AuthorityDecision:
    world = parse_finance_world(state)
    fact = world.control_facts[0]
    registry = create_issuer_registry(
        authorizations=[
            {
                "issuer_id": "issuer_finance",
                "issuer_role": "policy_owner",
                "authority_rank": 10,
                "allowed_domains": ["financial_adjustments"],
                "allowed_source_types": ["POLICY"],
                "can_supersede": False,
                "can_issue_scoped_evidence": False,
            }
        ]
    )
    query = create_finance_authority_query(
        subject_id=fact.adjustment_id,
        resource_id=fact.portco_id,
        action_type="post_adjustment",
        at_time="2026-01-15T00:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[],
        issuer_registry=registry,
        category_id=fact.original_economic_category,
        period_id=fact.period_id,
        amount_minor=fact.signed_amount_minor,
        currency=fact.currency,
        unit=fact.unit,
        currency_exponent=fact.currency_exponent,
    )
    rule = create_authority_record(
        authority_id="rule_finance",
        schema_version="1.0",
        source_type=AuthoritySourceType.POLICY,
        title="Finance rule",
        issuer_id="issuer_finance",
        issuer_role="policy_owner",
        authority_rank=10,
        action_type="post_adjustment",
        scope={
            "domain": "financial_adjustments",
            "scope_kind": "rule",
            "subject_id": fact.adjustment_id,
            "resource_id": fact.portco_id,
            "organization_id": None,
            "geography_id": None,
            "category_id": fact.original_economic_category,
            "period_id": fact.period_id,
            "minimum_amount_minor": None,
            "minimum_inclusive": False,
            "maximum_amount_minor": None,
            "maximum_inclusive": False,
            "currency": fact.currency,
            "unit": fact.unit,
            "currency_exponent": fact.currency_exponent,
            "rule_disposition": "PROCEED",
            "threshold_minor": 50_000_000,
            "restricted_categories": [],
        },
        effective_at="2026-01-01T00:00:00Z",
        expires_at=None,
        supersedes=[],
        exception_to=[],
        provenance_locator="authority/rule_finance",
        normative_status=NormativeStatus.ACTIVE_AUTHORITY,
    )
    return resolve_authority([rule], query)


def _apply(state: WorldState, tool_name: str, arguments: dict[str, object]) -> WorldState:
    proposal = FinanceAdapter().apply(
        state,
        ActionCall(
            action_id=f"mutation_{tool_name}",
            tool_name=tool_name,
            arguments=cast(JsonObject, arguments),
        ),
    )
    assert proposal.local_status == "success"
    assert proposal.next_state is not None
    return proposal.next_state


@pytest.mark.mutation
def test_malformed_sibling_cannot_be_hidden_by_named_currency_drift() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=88)
    world = parse_finance_world(fixture.initial_state)
    drifted_snapshot = world.ledger_snapshots[0].model_copy(update={"currency": "EUR"})
    malformed = FinanceWorldState.model_construct(
        domain_schema_version="1.0",
        case_id=world.case_id,
        reporting_periods=world.reporting_periods,
        ledger_snapshots=(drifted_snapshot,),
        adjustments=world.adjustments,
        monthly_reports=("malformed",),
        approvals=world.approvals,
        control_facts=world.control_facts,
        finished=False,
        finish_summary=None,
    )
    with pytest.raises(FinanceVerifierInputError):
        verify_finance_case(
            malformed, fixture.case, authority_decision=_escalation(fixture.initial_state)
        )


@pytest.mark.mutation
@pytest.mark.parametrize("value", [True, 2.5, "2"])
def test_malformed_money_scalar_is_input_error_even_for_escalation(value: object) -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=89)
    world = parse_finance_world(fixture.initial_state)
    snapshot = world.ledger_snapshots[0].model_copy(update={"currency_exponent": value})
    malformed = world.model_copy(update={"ledger_snapshots": (snapshot,)})
    with pytest.raises(FinanceVerifierInputError):
        verify_finance_case(
            malformed, fixture.case, authority_decision=_escalation(fixture.initial_state)
        )


@pytest.mark.mutation
def test_oversized_sequence_is_rejected_before_detached_preallocation() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=90)
    world = parse_finance_world(fixture.initial_state)
    oversized: list[object] = [None] * 100_000
    malformed = world.model_copy(update={"adjustments": oversized})
    with pytest.raises(FinanceVerifierInputError):
        verify_finance_case(
            malformed, fixture.case, authority_decision=_escalation(fixture.initial_state)
        )


@pytest.mark.mutation
def test_oversized_hostile_model_key_tree_is_rejected_before_key_traversal() -> None:
    class HostileKey:
        calls = 0

        def __hash__(self) -> int:
            type(self).calls += 1
            return 0

        def __eq__(self, other: object) -> bool:
            type(self).calls += 1
            return self is other

    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=901)
    world = parse_finance_world(fixture.initial_state)
    values = object.__getattribute__(world, "__dict__")
    hostile_key = HostileKey()
    values[hostile_key] = None
    HostileKey.calls = 0
    values.update({f"hostile_extra_{index}": None for index in range(100_001)})

    tracemalloc.start()
    try:
        with pytest.raises(FinanceVerifierInputError):
            verify_finance_case(
                world, fixture.case, authority_decision=_escalation(fixture.initial_state)
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000
    assert HostileKey.calls == 0


@pytest.mark.mutation
def test_verification_does_not_rescan_facts_with_next(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=902)
    completed = _apply(
        _apply(
            fixture.initial_state,
            "publish_report",
            {"report_id": fixture.case.target_report_id},
        ),
        "finish_task",
        {"summary": "Completed deterministically"},
    )

    def forbidden_next(*args: object, **kwargs: object) -> object:
        raise AssertionError("finance verifier must not rescan facts with next")

    monkeypatch.setattr(finance_verifier, "next", forbidden_next, raising=False)
    verdict = verify_finance_case(
        parse_finance_world(completed),
        fixture.case,
        authority_decision=_proceed(completed),
    )

    assert (verdict.status, verdict.reason_code) == ("PASS", "FINANCE_PUBLISHED")


@pytest.mark.mutation
def test_hostile_subclass_and_cycle_are_rejected_without_hook_execution() -> None:
    class HostileList(list[object]):
        calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).calls += 1
            raise AssertionError("verifier must not invoke hostile hooks")

    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=91)
    world = parse_finance_world(fixture.initial_state)
    hostile = world.model_copy(update={"adjustments": HostileList()})
    cycle: list[object] = []
    cycle.append(cycle)
    cyclic = world.model_copy(update={"adjustments": cycle})
    for candidate in (hostile, cyclic):
        with pytest.raises(FinanceVerifierInputError):
            verify_finance_case(
                candidate, fixture.case, authority_decision=_escalation(fixture.initial_state)
            )
    assert HostileList.calls == 0


@pytest.mark.mutation
def test_depth_limit_and_false_authority_marker_are_input_errors() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=92)
    world = parse_finance_world(fixture.initial_state)
    nested: list[object] = []
    cursor = nested
    for _ in range(65):
        child: list[object] = []
        cursor.append(child)
        cursor = child
    too_deep = world.model_copy(update={"adjustments": nested})
    with pytest.raises(FinanceVerifierInputError):
        verify_finance_case(
            too_deep, fixture.case, authority_decision=_escalation(fixture.initial_state)
        )
    decision = _escalation(fixture.initial_state)
    object.__setattr__(decision, "_validated", False)
    with pytest.raises(FinanceVerifierInputError):
        verify_finance_case(world, fixture.case, authority_decision=decision)


@pytest.mark.mutation
def test_subclassed_state_is_not_admitted() -> None:
    class FinanceWorldSubclass(FinanceWorldState):
        pass

    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=93)
    world = parse_finance_world(fixture.initial_state)
    subclassed = FinanceWorldSubclass.model_construct(
        domain_schema_version=world.domain_schema_version,
        case_id=world.case_id,
        reporting_periods=world.reporting_periods,
        ledger_snapshots=world.ledger_snapshots,
        adjustments=world.adjustments,
        monthly_reports=world.monthly_reports,
        approvals=world.approvals,
        control_facts=world.control_facts,
        finished=world.finished,
        finish_summary=world.finish_summary,
    )
    with pytest.raises(FinanceVerifierInputError):
        verify_finance_case(
            subclassed, fixture.case, authority_decision=_escalation(fixture.initial_state)
        )


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("finished", "finish_summary"),
    [
        (False, "Completed deterministically"),
        (True, ""),
        (True, "first line\nsecond line"),
        (True, "x" * 241),
    ],
)
def test_finish_gate_rejects_a_published_proceed_state(finished: bool, finish_summary: str) -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=94)
    published = _apply(
        fixture.initial_state, "publish_report", {"report_id": fixture.case.target_report_id}
    )
    completed = _apply(published, "finish_task", {"summary": "Completed deterministically"})
    malformed = parse_finance_world(completed).model_copy(
        update={"finished": finished, "finish_summary": finish_summary}
    )
    verdict = verify_finance_case(malformed, fixture.case, authority_decision=_proceed(completed))
    assert verdict.status == "FAIL"
    assert verdict.reason_code == "TASK_NOT_FINISHED"


@pytest.mark.mutation
def test_finish_gate_precedes_target_source_classification() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=95)
    world = parse_finance_world(fixture.initial_state)
    drifted_snapshot = world.ledger_snapshots[0].model_copy(
        update={"source_version": "other_source"}
    )
    unfinished_with_source_drift = world.model_copy(
        update={"ledger_snapshots": (drifted_snapshot,)}
    )
    verdict = verify_finance_case(
        unfinished_with_source_drift,
        fixture.case,
        authority_decision=_proceed(fixture.initial_state),
    )
    assert verdict.status == "FAIL"
    assert verdict.reason_code == "TASK_NOT_FINISHED"
