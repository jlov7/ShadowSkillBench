from __future__ import annotations

import inspect

import pytest

from shadowskillbench.authority import (
    GateDecision,
    GateInputError,
    GateOutcome,
    RuleDisposition,
    create_access_authority_query,
    create_authority_record,
    create_finance_authority_query,
    create_issuer_registry,
    gate_action,
)
from shadowskillbench.domains.access.fixtures import build_access_fixture
from shadowskillbench.domains.finance.fixtures import FinanceFixtureVariant, build_finance_fixture
from shadowskillbench.domains.finance.models import parse_finance_world
from shadowskillbench.engine import ActionCall, TaskCase, WorldState


def _task(state, *, objective: str = "Complete the synthetic task.") -> TaskCase:
    return TaskCase(
        case_id=state.data["case_id"],
        domain=state.domain,
        world_id=state.world_id,
        seed=state.seed,
        objective=objective,
        inputs={"task_id": state.data["case_id"]},
    )


def _access_registry():
    return create_issuer_registry(
        authorizations=[
            {
                "issuer_id": "issuer_access",
                "issuer_role": "policy_owner",
                "authority_rank": 10,
                "allowed_domains": ["access_provisioning"],
                "allowed_source_types": ["POLICY"],
                "can_supersede": False,
                "can_issue_scoped_evidence": False,
            }
        ]
    )


def _access_query(state, *, role_id: str | None = None):
    request = state.data["access_requests"][0]
    application = next(
        value
        for value in state.data["applications"]
        if value["application_id"] == request["application_id"]
    )
    return create_access_authority_query(
        subject_id=request["employee_id"],
        resource_id=request["application_id"],
        action_type="grant_access",
        at_time="2026-08-01T12:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[],
        issuer_registry=_access_registry(),
        role_id=request["requested_role"] if role_id is None else role_id,
        role_derived_access=application["supports_role_derived_access"],
    )


def _access_record(state, *, authority_id: str, disposition: RuleDisposition):
    request = state.data["access_requests"][0]
    return create_authority_record(
        authority_id=authority_id,
        schema_version="1.0",
        source_type="POLICY",
        title=authority_id,
        issuer_id="issuer_access",
        issuer_role="policy_owner",
        authority_rank=10,
        action_type="grant_access",
        scope={
            "domain": "access_provisioning",
            "scope_kind": "rule",
            "subject_id": request["employee_id"],
            "resource_id": request["application_id"],
            "organization_id": None,
            "geography_id": None,
            "role_id": request["requested_role"],
            "rule_disposition": disposition,
            "allowed_action": "grant_access",
            "requires_security_approval": False,
            "role_derived_without_approval": True,
        },
        effective_at="2026-08-01T00:00:00Z",
        expires_at=None,
        supersedes=[],
        exception_to=[],
        provenance_locator=f"authority/{authority_id}",
        normative_status="ACTIVE_AUTHORITY",
    )


def _access_call(state, *, action_id: str = "grant") -> ActionCall:
    return ActionCall(
        action_id=action_id,
        tool_name="grant_access",
        arguments={"request_id": state.data["access_requests"][0]["request_id"]},
    )


def _access_request_id(state):
    return _access_call(state).arguments["request_id"]


def _finance_registry():
    return create_issuer_registry(
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


def _finance_fact(state):
    return parse_finance_world(state).control_facts[0]


def _finance_query(state, *, amount_minor: int | None = None):
    fact = _finance_fact(state)
    return create_finance_authority_query(
        subject_id=fact.adjustment_id,
        resource_id=fact.portco_id,
        action_type="post_adjustment",
        at_time="2026-01-15T00:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[],
        issuer_registry=_finance_registry(),
        category_id=fact.original_economic_category,
        period_id=fact.period_id,
        amount_minor=fact.signed_amount_minor if amount_minor is None else amount_minor,
        currency=fact.currency,
        unit=fact.unit,
        currency_exponent=fact.currency_exponent,
    )


def _finance_record(state):
    fact = _finance_fact(state)
    return create_authority_record(
        authority_id="rule_finance",
        schema_version="1.0",
        source_type="POLICY",
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
        normative_status="ACTIVE_AUTHORITY",
    )


def test_access_gate_allows_and_is_deterministic() -> None:
    fixture = build_access_fixture("role_derived", seed=41)
    state = fixture.initial_world
    case = _task(state)
    call = _access_call(state)
    authority = [
        _access_record(state, authority_id="rule_access", disposition=RuleDisposition.PROCEED)
    ]

    first = gate_action(state, call, case, authority, query=_access_query(state))
    second = gate_action(state, call, case, authority, query=_access_query(state))

    assert first.outcome is GateOutcome.ALLOW
    assert first == second
    assert first.gate_hash == second.gate_hash
    assert first.authority_decision_hash is not None


def test_gate_signature_requires_keyword_only_query() -> None:
    parameter = inspect.signature(gate_action).parameters["query"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_access_gate_blocks_and_escalates() -> None:
    fixture = build_access_fixture("role_derived", seed=42)
    state = fixture.initial_world
    case = _task(state)
    call = _access_call(state)
    query = _access_query(state)

    blocked = gate_action(
        state,
        call,
        case,
        [_access_record(state, authority_id="rule_block", disposition=RuleDisposition.BLOCK)],
        query=query,
    )
    escalated = gate_action(
        state,
        call,
        case,
        [
            _access_record(state, authority_id="rule_allow", disposition=RuleDisposition.PROCEED),
            _access_record(state, authority_id="rule_block", disposition=RuleDisposition.BLOCK),
        ],
        query=query,
    )

    assert blocked.outcome is GateOutcome.BLOCK
    assert escalated.outcome is GateOutcome.ESCALATE


def test_access_gate_rejects_mismatched_query_and_unknown_tool() -> None:
    fixture = build_access_fixture("role_derived", seed=43)
    state = fixture.initial_world
    case = _task(state)
    authority = [
        _access_record(state, authority_id="rule_access", disposition=RuleDisposition.PROCEED)
    ]

    with pytest.raises(GateInputError, match="access query"):
        gate_action(
            state,
            _access_call(state),
            case,
            authority,
            query=_access_query(state, role_id="role_other"),
        )
    with pytest.raises(GateInputError, match="unknown access tool"):
        gate_action(
            state,
            ActionCall(action_id="unknown", tool_name="unknown_tool", arguments={}),
            case,
            authority,
            query=_access_query(state),
        )


def test_gate_rejects_state_case_mismatch_and_rehashed_decision_tampering() -> None:
    fixture = build_access_fixture("role_derived", seed=47)
    state = fixture.initial_world
    case = _task(state)
    call = _access_call(state)
    authority = [
        _access_record(state, authority_id="rule_access", disposition=RuleDisposition.PROCEED)
    ]
    decision = gate_action(state, call, case, authority, query=_access_query(state))
    changed_state = WorldState(
        schema_version=state.schema_version,
        world_id=state.world_id,
        domain=state.domain,
        seed=state.seed + 1,
        data=state.data,
    )
    tampered = decision.model_dump(mode="json")
    tampered["gate_hash"] = "sha256:" + "0" * 64

    with pytest.raises(GateInputError, match="state must bind case"):
        gate_action(changed_state, call, case, authority, query=_access_query(state))
    with pytest.raises(ValueError, match="gate_hash"):
        GateDecision.model_validate(tampered)


def test_nonconsequential_access_call_is_not_applicable_and_action_hash_binds() -> None:
    fixture = build_access_fixture("role_derived", seed=44)
    state = fixture.initial_world
    case = _task(state)
    query = _access_query(state)
    read = ActionCall(
        action_id="read_request",
        tool_name="get_access_request",
        arguments={"request_id": _access_request_id(state)},
    )
    changed_read = ActionCall(
        action_id="read_request_2",
        tool_name="get_access_request",
        arguments={"request_id": _access_request_id(state)},
    )

    first = gate_action(state, read, case, [], query=query)
    second = gate_action(state, changed_read, case, [], query=query)

    assert first.outcome is GateOutcome.NOT_APPLICABLE
    assert first.authority_query_hash is None
    assert first.action_hash != second.action_hash
    assert first.gate_hash != second.gate_hash


def test_finance_gate_binds_immutable_control_fact_for_mutation_and_publish() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=45)
    state = fixture.initial_state
    case = _task(state)
    world = parse_finance_world(state)
    fact = world.control_facts[0]
    report = world.monthly_reports[0]
    adjustment = world.adjustments[0]
    authority = [_finance_record(state)]
    query = _finance_query(state)
    classify = ActionCall(
        action_id="classify",
        tool_name="classify_adjustment",
        arguments={"adjustment_id": fact.adjustment_id, "reported_category": "reported_other"},
    )
    create = ActionCall(
        action_id="create",
        tool_name="create_adjustment",
        arguments={
            "report_id": report.report_id,
            "control_fact_id": fact.control_fact_id,
            "reported_category": adjustment.reported_category,
            "signed_amount_minor": fact.signed_amount_minor,
            "currency": fact.currency,
            "unit": fact.unit,
            "currency_exponent": fact.currency_exponent,
            "period_id": fact.period_id,
            "portco_id": fact.portco_id,
            "rationale": adjustment.rationale,
        },
    )
    publish = ActionCall(
        action_id="publish",
        tool_name="publish_report",
        arguments={"report_id": report.report_id},
    )

    assert gate_action(state, classify, case, authority, query=query).outcome is GateOutcome.ALLOW
    assert gate_action(state, create, case, authority, query=query).outcome is GateOutcome.ALLOW
    assert gate_action(state, publish, case, authority, query=query).outcome is GateOutcome.ALLOW


def test_finance_gate_rejects_control_fact_query_mismatch() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=46)
    state = fixture.initial_state
    case = _task(state)
    fact = _finance_fact(state)
    call = ActionCall(
        action_id="classify",
        tool_name="classify_adjustment",
        arguments={"adjustment_id": fact.adjustment_id, "reported_category": "reported_other"},
    )

    with pytest.raises(GateInputError, match="immutable control facts"):
        gate_action(
            state,
            call,
            case,
            [_finance_record(state)],
            query=_finance_query(state, amount_minor=fact.signed_amount_minor + 1),
        )


@pytest.mark.parametrize("mode", ["zero", "multiple"])
def test_finance_publish_requires_exactly_one_target_adjustment(mode: str) -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=48)
    state = fixture.initial_state
    case = _task(state)
    world = parse_finance_world(state)
    report = world.monthly_reports[0]
    payload = state.model_dump(mode="json")
    data = payload["data"]
    if mode == "zero":
        data["monthly_reports"][0]["adjustment_ids"] = []
    else:
        data["monthly_reports"][0]["adjustment_ids"] = [
            data["adjustments"][0]["adjustment_id"],
            "adjustment_other",
        ]
        adjustment = dict(data["adjustments"][0])
        adjustment["adjustment_id"] = "adjustment_other"
        data["adjustments"].append(adjustment)
        second_fact = dict(data["control_facts"][0])
        second_fact["control_fact_id"] = "fact_other"
        second_fact["adjustment_id"] = "adjustment_other"
        data["control_facts"].append(second_fact)
        data["adjustments"].sort(key=lambda item: item["adjustment_id"])
        data["control_facts"].sort(key=lambda item: item["control_fact_id"])
    malformed_report = WorldState.model_validate(payload)
    call = ActionCall(
        action_id="publish",
        tool_name="publish_report",
        arguments={"report_id": report.report_id},
    )

    with pytest.raises(GateInputError, match="exactly one adjustment"):
        gate_action(
            malformed_report,
            call,
            case,
            [_finance_record(state)],
            query=_finance_query(state),
        )
