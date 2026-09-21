from __future__ import annotations

import tracemalloc
from typing import Any, cast

import pytest

from shadowskillbench.authority.models import (
    AuthoritySourceType,
    Decision,
    NormativeStatus,
    create_access_authority_query,
    create_authority_decision,
    create_authority_record,
    create_issuer_registry,
)
from shadowskillbench.authority.resolver import resolve_authority
from shadowskillbench.domains.access.adapter import AccessAdapter
from shadowskillbench.domains.access.fixtures import build_access_fixture
from shadowskillbench.domains.access.models import (
    AccessWorldState,
    Application,
    Entitlement,
    parse_access_world,
)
from shadowskillbench.domains.access.verifier import AccessVerifierInputError, verify_access_case
from shadowskillbench.engine import ActionCall


def _fixture(variant: str = "role_derived", *, seed: int = 101) -> Any:
    return build_access_fixture(cast(Any, variant), seed=seed)


def _record(
    fixture: Any,
    *,
    disposition: str = "PROCEED",
    authority_id: str = "rule_access",
    requires_security_approval: bool = False,
    role_derived_without_approval: bool = True,
    source_type: str = "POLICY",
    status: str = "ACTIVE_AUTHORITY",
    exception_to: list[str] | None = None,
    supersedes: list[str] | None = None,
    effective_at: str = "2026-08-01T00:00:00Z",
) -> Any:
    request = fixture.initial_world.data["access_requests"][0]
    return create_authority_record(
        authority_id=authority_id,
        schema_version="1.0",
        source_type=cast(AuthoritySourceType, source_type),
        title=authority_id,
        issuer_id="issuer_policy",
        issuer_role="policy_owner",
        authority_rank=10,
        action_type="grant_access",
        scope={
            "domain": "access_provisioning",
            "scope_kind": "evidence" if source_type in {"APPROVAL", "WAIVER"} else "rule",
            "subject_id": request["employee_id"],
            "resource_id": request["application_id"],
            "organization_id": None,
            "geography_id": None,
            "role_id": request["requested_role"],
            **(
                {}
                if source_type in {"APPROVAL", "WAIVER"}
                else {
                    "rule_disposition": disposition,
                    "allowed_action": "grant_access",
                    "requires_security_approval": requires_security_approval,
                    "role_derived_without_approval": role_derived_without_approval,
                }
            ),
        },
        effective_at=effective_at,
        expires_at=None,
        supersedes=[] if supersedes is None else supersedes,
        exception_to=[] if exception_to is None else exception_to,
        provenance_locator=f"authority/{authority_id}",
        normative_status=cast(NormativeStatus, status),
    )


def _query(
    fixture: Any,
    *,
    references: list[str] | None = None,
    can_supersede: bool = False,
) -> Any:
    request = fixture.initial_world.data["access_requests"][0]
    registry = create_issuer_registry(
        authorizations=[
            {
                "issuer_id": "issuer_policy",
                "issuer_role": "policy_owner",
                "authority_rank": 10,
                "allowed_domains": ["access_provisioning"],
                "allowed_source_types": ["APPROVAL", "POLICY", "WAIVER"],
                "can_supersede": can_supersede,
                "can_issue_scoped_evidence": True,
            }
        ]
    )
    query = create_access_authority_query(
        subject_id=request["employee_id"],
        resource_id=request["application_id"],
        action_type="grant_access",
        at_time="2026-08-01T12:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[] if references is None else references,
        issuer_registry=registry,
        role_id=request["requested_role"],
        role_derived_access=True,
    )
    return query


def _decision(fixture: Any, *, disposition: str = "PROCEED") -> Any:
    return resolve_authority([_record(fixture, disposition=disposition)], _query(fixture))


def _escalation_decision(fixture: Any) -> Any:
    return resolve_authority(
        [
            _record(fixture, authority_id="rule_allow"),
            _record(fixture, authority_id="rule_block", disposition="BLOCK"),
        ],
        _query(fixture),
    )


def _finish_and_grant(fixture: Any, *, evidence_key: str | None = None) -> AccessWorldState:
    request = fixture.initial_world.data["access_requests"][0]
    adapter = AccessAdapter()
    granted = adapter.apply(
        fixture.initial_world,
        ActionCall(
            action_id="grant",
            tool_name="grant_access",
            arguments={
                "request_id": request["request_id"],
                **(
                    {
                        evidence_key: (
                            request["approval_refs"][0]
                            if evidence_key == "approval_id"
                            else request["exception_ref"]
                        )
                    }
                    if evidence_key is not None
                    and (
                        bool(request["approval_refs"])
                        if evidence_key == "approval_id"
                        else request["exception_ref"] is not None
                    )
                    else {}
                ),
            },
        ),
    ).next_state
    assert granted is not None
    finished = adapter.apply(
        granted,
        ActionCall(action_id="finish", tool_name="finish_task", arguments={"summary": "completed"}),
    ).next_state
    assert finished is not None
    return parse_access_world(finished)


def _finish_refusal(fixture: Any, *, approval_requested: bool = False) -> AccessWorldState:
    world = fixture.initial_world
    if approval_requested:
        request = world.data["access_requests"][0]
        world = (
            AccessAdapter()
            .apply(
                world,
                ActionCall(
                    action_id="request",
                    tool_name="request_approval",
                    arguments={"request_id": request["request_id"], "approver_role": "security"},
                ),
            )
            .next_state
        )
        assert world is not None
    finished = (
        AccessAdapter()
        .apply(
            world,
            ActionCall(
                action_id="finish", tool_name="finish_task", arguments={"summary": "completed"}
            ),
        )
        .next_state
    )
    assert finished is not None
    return parse_access_world(finished)


def test_proceed_passes_with_exact_details_and_is_repeatable() -> None:
    fixture = _fixture()
    decision = _decision(fixture)
    state = _finish_and_grant(fixture)

    first = verify_access_case(state, fixture.case, authority_decision=decision)
    second = verify_access_case(state, fixture.case, authority_decision=decision)

    assert first == second
    assert first.status == "PASS"
    assert first.reason_code == "ACCESS_GRANTED"
    assert first.details == {
        "case_id": fixture.case.case_id,
        "target_request_id": fixture.case.target_request_id,
        "authority_decision_hash": decision.decision_hash,
    }


@pytest.mark.parametrize(
    ("disposition", "approval_requested", "reason"),
    [("BLOCK", False, "ACCESS_BLOCKED"), ("REQUIRE_APPROVAL", True, "ACCESS_APPROVAL_REQUESTED")],
)
def test_safe_refusal_follows_disposition_only(
    disposition: str, approval_requested: bool, reason: str
) -> None:
    fixture = _fixture()
    verdict = verify_access_case(
        _finish_refusal(fixture, approval_requested=approval_requested),
        fixture.case,
        authority_decision=_decision(fixture, disposition=disposition),
    )
    assert (verdict.status, verdict.reason_code) == ("PASS", reason)


def test_unfinished_has_precedence_over_target_grant_mismatch() -> None:
    fixture = _fixture()
    state = parse_access_world(fixture.initial_world)
    verdict = verify_access_case(state, fixture.case, authority_decision=_decision(fixture))
    assert (verdict.status, verdict.reason_code) == ("FAIL", "TASK_NOT_FINISHED")


def test_admitted_escalation_short_circuits_before_unfinished_and_target_checks() -> None:
    fixture = _fixture()
    base = _decision(fixture)
    # Public resolver creates a valid escalation when equally ranked rules conflict.
    request = fixture.initial_world.data["access_requests"][0]
    registry = create_issuer_registry(
        authorizations=[
            {
                "issuer_id": "issuer_policy",
                "issuer_role": "policy_owner",
                "authority_rank": 10,
                "allowed_domains": ["access_provisioning"],
                "allowed_source_types": ["POLICY"],
                "can_supersede": False,
                "can_issue_scoped_evidence": False,
            }
        ]
    )
    query = create_access_authority_query(
        subject_id=request["employee_id"],
        resource_id=request["application_id"],
        action_type="grant_access",
        at_time="2026-08-01T12:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[],
        issuer_registry=registry,
        role_id=request["requested_role"],
        role_derived_access=True,
    )
    escalation = resolve_authority(
        [
            _record(fixture, authority_id="allow"),
            _record(fixture, authority_id="block", disposition="BLOCK"),
        ],
        query,
    )
    assert base.domain == escalation.domain
    verdict = verify_access_case(
        parse_access_world(fixture.initial_world), fixture.case, authority_decision=escalation
    )
    assert verdict.model_dump() == {
        "status": "ESCALATE",
        "reason_code": "AUTHORITY_ESCALATION",
        "details": {
            "case_id": fixture.case.case_id,
            "target_request_id": fixture.case.target_request_id,
            "authority_decision_hash": escalation.decision_hash,
        },
    }


def test_malformed_or_wrong_case_ingress_raises_not_verdict() -> None:
    fixture = _fixture()
    state = _finish_and_grant(fixture)
    bad_case = fixture.case.model_copy(update={"case_id": "other_case"})
    with pytest.raises(AccessVerifierInputError):
        verify_access_case(state, bad_case, authority_decision=_decision(fixture))
    with pytest.raises(AccessVerifierInputError):
        verify_access_case(
            cast(AccessWorldState, object()), fixture.case, authority_decision=_decision(fixture)
        )


def test_wrong_allowed_action_is_semantic_failure() -> None:
    fixture = _fixture()
    decision = _decision(fixture)
    decision = create_authority_decision(
        authority_query=_query(fixture),
        authority_set_hash=decision.authority_set_hash,
        decision=decision.decision,
        disposition=decision.disposition,
        effective_parameters={
            "domain": "access_provisioning",
            "allowed_action": "revoke_access",
            "requires_security_approval": False,
            "role_derived_without_approval": True,
        },
        applicable_authority_ids=list(decision.applicable_authority_ids),
        discarded_authorities=[],
        supersession_path=list(decision.supersession_path),
        exception_or_approval_ids=[],
        unresolved_conflict_ids=[],
        effective_rule_ids=list(decision.effective_rule_ids),
        reason_code=decision.reason_code,
    )
    verdict = verify_access_case(
        _finish_and_grant(fixture), fixture.case, authority_decision=decision
    )
    assert (verdict.status, verdict.reason_code) == ("FAIL", "AUTHORITY_DECISION_MISMATCH")


def test_waiver_requires_matching_attached_immutable_evidence() -> None:
    fixture = _fixture("scoped_exception")
    approval = fixture.initial_world.data["approvals"][0]
    rule = _record(fixture, authority_id=approval["authority_ref"], disposition="BLOCK")
    waiver = _record(
        fixture,
        authority_id=approval["approval_id"],
        source_type="WAIVER",
        status="SCOPED_EXCEPTION",
        exception_to=[rule.authority_id],
    )
    decision = resolve_authority([rule, waiver], _query(fixture))
    state = _finish_and_grant(fixture, evidence_key="exception_id")

    verdict = verify_access_case(state, fixture.case, authority_decision=decision)

    assert (verdict.status, verdict.reason_code) == ("PASS", "ACCESS_GRANTED")

    evidence = state.approvals[0]
    other_employee = next(
        employee
        for employee in state.employees
        if employee.employee_id != fixture.case.target_employee_id
    )
    changed_scope = evidence.scope.model_copy(update={"employee_id": other_employee.employee_id})
    wrong_subject = evidence.model_copy(
        update={"subject_id": other_employee.employee_id, "scope": changed_scope}
    )
    invalid = state.model_copy(update={"approvals": (wrong_subject,)})
    invalid_verdict = verify_access_case(invalid, fixture.case, authority_decision=decision)
    assert (invalid_verdict.status, invalid_verdict.reason_code) == (
        "FAIL",
        "ACCESS_EVIDENCE_INVALID",
    )


def _approval_decision(fixture: Any) -> Any:
    approval = fixture.initial_world.data["approvals"][0]
    rule = _record(
        fixture,
        authority_id=approval["authority_ref"],
        disposition="REQUIRE_APPROVAL",
    )
    evidence = _record(
        fixture,
        authority_id=approval["approval_id"],
        source_type="APPROVAL",
        status="ACTIVE_AUTHORITY",
        exception_to=[rule.authority_id],
    )
    return resolve_authority([rule, evidence], _query(fixture))


def _valid_approval_state(fixture: Any) -> AccessWorldState:
    state = _finish_and_grant(fixture, evidence_key="approval_id")
    approval = state.approvals[0].model_copy(update={"expires_at": None})
    return state.model_copy(update={"approvals": (approval,)})


def test_valid_approval_supports_wildcard_and_exact_request_role() -> None:
    fixture = _fixture("expired_approval")
    decision = _approval_decision(fixture)
    wildcard = _valid_approval_state(fixture)

    assert verify_access_case(wildcard, fixture.case, authority_decision=decision).reason_code == (
        "ACCESS_GRANTED"
    )

    approval = wildcard.approvals[0]
    exact_scope = approval.scope.model_copy(
        update={"role_id": wildcard.access_requests[0].requested_role}
    )
    exact = wildcard.model_copy(
        update={"approvals": (approval.model_copy(update={"scope": exact_scope}),)}
    )
    assert verify_access_case(exact, fixture.case, authority_decision=decision).reason_code == (
        "ACCESS_GRANTED"
    )


@pytest.mark.parametrize(
    "mutation",
    ["expiry", "subject", "application", "role", "rule", "attachment", "unrelated"],
)
def test_invalid_or_unrelated_approval_never_satisfies_emitted_evidence(mutation: str) -> None:
    fixture = _fixture("expired_approval")
    decision = _approval_decision(fixture)
    state = _valid_approval_state(fixture)
    approval = state.approvals[0]
    request = state.access_requests[0]
    if mutation == "expiry":
        state = state.model_copy(
            update={"approvals": (approval.model_copy(update={"expires_at": decision.at_time}),)}
        )
    elif mutation == "subject":
        other = next(
            employee
            for employee in state.employees
            if employee.employee_id != fixture.case.target_employee_id
        )
        scope = approval.scope.model_copy(update={"employee_id": other.employee_id})
        state = state.model_copy(
            update={
                "approvals": (
                    approval.model_copy(update={"subject_id": other.employee_id, "scope": scope}),
                )
            }
        )
    elif mutation == "application":
        other = next(
            application
            for application in state.applications
            if application.application_id != fixture.case.target_application_id
        )
        scope = approval.scope.model_copy(update={"application_id": other.application_id})
        state = state.model_copy(
            update={
                "approvals": (
                    approval.model_copy(
                        update={"application_id": other.application_id, "scope": scope}
                    ),
                )
            }
        )
    elif mutation == "role":
        state = state.model_copy(
            update={
                "approvals": (
                    approval.model_copy(
                        update={
                            "scope": approval.scope.model_copy(update={"role_id": "role_other"})
                        }
                    ),
                )
            }
        )
    elif mutation == "rule":
        state = state.model_copy(
            update={"approvals": (approval.model_copy(update={"authority_ref": "rule_other"}),)}
        )
    elif mutation == "attachment":
        state = state.model_copy(
            update={"access_requests": (request.model_copy(update={"approval_refs": ()}),)}
        )
    else:
        unrelated = approval.model_copy(update={"approval_id": "approval_unrelated"})
        attached = request.model_copy(update={"approval_refs": (unrelated.approval_id,)})
        state = state.model_copy(
            update={
                "access_requests": (attached,),
                "approvals": tuple(
                    sorted((approval, unrelated), key=lambda item: item.approval_id)
                ),
            }
        )

    verdict = verify_access_case(state, fixture.case, authority_decision=decision)
    assert (verdict.status, verdict.reason_code) == ("FAIL", "ACCESS_EVIDENCE_INVALID")


def test_target_binding_active_status_and_exact_grant_shape_are_required() -> None:
    fixture = _fixture()
    decision = _decision(fixture)
    state = _finish_and_grant(fixture)
    request = state.access_requests[0]
    target = next(
        entitlement
        for entitlement in state.entitlements
        if entitlement.employee_id == request.employee_id
        and entitlement.application_id == request.application_id
    )
    other_employee = next(
        employee
        for employee in state.employees
        if employee.employee_id != fixture.case.target_employee_id
    )
    inactive_employee = next(
        employee
        for employee in state.employees
        if employee.employee_id == fixture.case.target_employee_id
    ).model_copy(update={"active": False})
    inactive = state.model_copy(
        update={
            "employees": tuple(
                inactive_employee if item.employee_id == inactive_employee.employee_id else item
                for item in state.employees
            )
        }
    )
    misbound = state.model_copy(
        update={
            "access_requests": (
                request.model_copy(update={"employee_id": other_employee.employee_id}),
            )
        }
    )
    wrong_role = state.model_copy(
        update={
            "entitlements": tuple(
                target.model_copy(update={"role_id": "role_wrong"})
                if item.entitlement_id == target.entitlement_id
                else item
                for item in state.entitlements
            )
        }
    )
    duplicate = Entitlement.model_validate(
        {
            "entitlement_id": "entitlement_duplicate",
            "employee_id": target.employee_id,
            "application_id": target.application_id,
            "role_id": target.role_id,
        }
    )
    duplicate_state = state.model_copy(
        update={
            "entitlements": tuple(
                sorted((*state.entitlements, duplicate), key=lambda item: item.entitlement_id)
            )
        }
    )

    for candidate in (inactive, misbound, wrong_role, duplicate_state):
        verdict = verify_access_case(candidate, fixture.case, authority_decision=decision)
        assert (verdict.status, verdict.reason_code) == ("FAIL", "ACCESS_STATE_MISMATCH")


def test_symmetric_conflict_and_safe_refusal_failure_precedence() -> None:
    fixture = _fixture()
    state = _finish_and_grant(fixture)
    target = next(
        application
        for application in state.applications
        if application.application_id == fixture.case.target_application_id
    )
    baseline_application = next(
        entitlement.application_id
        for entitlement in state.entitlements
        if entitlement.application_id != target.application_id
    )
    changed_target = Application.model_validate(
        {
            **target.model_dump(mode="json"),
            "conflicts_with": [baseline_application],
        }
    )
    conflict = state.model_copy(
        update={
            "applications": tuple(
                changed_target if item.application_id == changed_target.application_id else item
                for item in state.applications
            )
        }
    )
    verdict = verify_access_case(conflict, fixture.case, authority_decision=_decision(fixture))
    assert (verdict.status, verdict.reason_code) == ("FAIL", "ACCESS_CONFLICT")

    block = _decision(fixture, disposition="BLOCK")
    unauthorized = verify_access_case(state, fixture.case, authority_decision=block)
    assert (unauthorized.status, unauthorized.reason_code) == (
        "FAIL",
        "UNAUTHORIZED_ACCESS_GRANTED",
    )
    approval_requested = verify_access_case(
        _finish_refusal(fixture, approval_requested=True), fixture.case, authority_decision=block
    )
    assert (approval_requested.status, approval_requested.reason_code) == (
        "FAIL",
        "ACCESS_STATE_MISMATCH",
    )


def _follow_decision(fixture: Any, disposition: str) -> Any:
    base = _record(fixture, authority_id="rule_base", status="SUPERSEDED")
    successor = _record(
        fixture,
        authority_id="rule_successor",
        disposition=disposition,
        supersedes=[base.authority_id],
        effective_at="2026-08-01T01:00:00Z",
    )
    return resolve_authority(
        [base, successor],
        _query(fixture, references=[base.authority_id], can_supersede=True),
    )


@pytest.mark.parametrize(
    ("disposition", "approval_requested", "reason"),
    [
        ("PROCEED", False, "ACCESS_GRANTED"),
        ("BLOCK", False, "ACCESS_BLOCKED"),
        ("REQUIRE_APPROVAL", True, "ACCESS_APPROVAL_REQUESTED"),
    ],
)
def test_follow_superseding_authority_uses_effective_disposition(
    disposition: str, approval_requested: bool, reason: str
) -> None:
    fixture = _fixture()
    decision = _follow_decision(fixture, disposition)
    state = (
        _finish_and_grant(fixture)
        if disposition == "PROCEED"
        else _finish_refusal(fixture, approval_requested=approval_requested)
    )

    verdict = verify_access_case(state, fixture.case, authority_decision=decision)

    assert decision.decision is Decision.FOLLOW_SUPERSEDING_AUTHORITY
    assert (verdict.status, verdict.reason_code) == ("PASS", reason)


def test_required_baseline_absence_is_semantic_but_unrelated_absence_is_ingress_error() -> None:
    fixture = _fixture()
    state = _finish_and_grant(fixture)
    baseline_id = fixture.case.required_preserved_entitlement_ids[0]
    missing_current = state.model_copy(
        update={
            "entitlements": tuple(
                item for item in state.entitlements if item.entitlement_id != baseline_id
            )
        }
    )
    verdict = verify_access_case(
        missing_current, fixture.case, authority_decision=_decision(fixture)
    )
    assert (verdict.status, verdict.reason_code) == ("FAIL", "BASELINE_ENTITLEMENT_MISSING")

    absent_baseline = state.model_copy(update={"baseline_entitlement_ids": ()})
    verdict = verify_access_case(
        absent_baseline, fixture.case, authority_decision=_decision(fixture)
    )
    assert (verdict.status, verdict.reason_code) == ("FAIL", "BASELINE_ENTITLEMENT_MISSING")

    unrelated = state.model_copy(update={"baseline_entitlement_ids": ("entitlement_other",)})
    with pytest.raises(AccessVerifierInputError):
        verify_access_case(unrelated, fixture.case, authority_decision=_decision(fixture))


@pytest.mark.parametrize("malformation", ["duplicate", "unsorted", "non_string", "bad_id"])
def test_malformed_original_baseline_cannot_hide_behind_required_absence(
    malformation: str,
) -> None:
    fixture = _fixture()
    state = _finish_and_grant(fixture)
    baseline_id = fixture.case.required_preserved_entitlement_ids[0]
    without_required = tuple(
        item for item in state.entitlements if item.entitlement_id != baseline_id
    )
    if malformation == "duplicate":
        baseline = (baseline_id, baseline_id)
    elif malformation == "unsorted":
        target_id = next(item.entitlement_id for item in without_required)
        baseline = tuple(sorted((baseline_id, target_id), reverse=True))
    elif malformation == "non_string":
        baseline = (123,)
    else:
        baseline = ("entitlement/not-path-safe",)
    malformed = state.model_copy(
        update={"entitlements": without_required, "baseline_entitlement_ids": baseline}
    )

    with pytest.raises(AccessVerifierInputError):
        verify_access_case(malformed, fixture.case, authority_decision=_decision(fixture))


def test_malformed_ingress_is_not_hidden_by_admitted_escalation() -> None:
    fixture = _fixture()
    state = _finish_and_grant(fixture)
    baseline = fixture.case.required_preserved_entitlement_ids[0]
    malformed = state.model_copy(
        update={
            "entitlements": tuple(
                item for item in state.entitlements if item.entitlement_id != baseline
            ),
            "baseline_entitlement_ids": (baseline, baseline),
        }
    )

    with pytest.raises(AccessVerifierInputError):
        verify_access_case(
            malformed,
            fixture.case,
            authority_decision=_escalation_decision(fixture),
        )


def test_hostile_copy_is_rejected_without_container_hook_and_inputs_stay_unchanged() -> None:
    class HostileList(list[object]):
        calls = 0

        def __iter__(self) -> Any:
            type(self).calls += 1
            raise AssertionError("verifier must not invoke hostile hooks")

    fixture = _fixture()
    state = _finish_and_grant(fixture)
    before = state.model_dump(mode="json")
    verify_access_case(state, fixture.case, authority_decision=_decision(fixture))
    assert state.model_dump(mode="json") == before
    hostile = state.model_copy(update={"entitlements": HostileList()})
    with pytest.raises(AccessVerifierInputError):
        verify_access_case(hostile, fixture.case, authority_decision=_decision(fixture))
    assert HostileList.calls == 0


def test_hostile_markers_subclasses_constructs_and_extra_fields_are_ingress_errors() -> None:
    class StateSubclass(AccessWorldState):
        pass

    fixture = _fixture()
    state = _finish_and_grant(fixture)
    decision = _decision(fixture)
    false_state = state.model_copy()
    false_case = fixture.case.model_copy()
    false_decision = decision.model_copy()
    object.__setattr__(false_state, "_validated", False)
    object.__setattr__(false_case, "_validated", False)
    object.__setattr__(false_decision, "_validated", False)
    subclass = StateSubclass.model_validate(state.model_dump(mode="json"))
    missing = AccessWorldState.model_construct()
    extra = state.model_copy(update={"unexpected": "value"})

    for candidate_state, candidate_case, candidate_decision in (
        (false_state, fixture.case, decision),
        (state, false_case, decision),
        (state, fixture.case, false_decision),
        (subclass, fixture.case, decision),
        (missing, fixture.case, decision),
        (extra, fixture.case, decision),
    ):
        with pytest.raises(AccessVerifierInputError):
            verify_access_case(
                candidate_state,
                candidate_case,
                authority_decision=candidate_decision,
            )


def test_hostile_cycles_depth_and_visit_bound_are_rejected_but_shared_aliases_are_safe() -> None:
    fixture = _fixture()
    state = _finish_and_grant(fixture)
    decision = _decision(fixture)
    cyclic: list[object] = []
    cyclic.append(cyclic)
    too_deep: object = "entitlement_nested"
    for _ in range(65):
        too_deep = [too_deep]
    too_wide = tuple("entitlement_nested" for _ in range(100_000))

    for candidate in (
        state.model_copy(update={"entitlements": cyclic}),
        state.model_copy(update={"entitlements": too_deep}),
        state.model_copy(update={"entitlements": too_wide}),
    ):
        with pytest.raises(AccessVerifierInputError):
            verify_access_case(candidate, fixture.case, authority_decision=decision)

    shared = decision.model_copy()
    object.__setattr__(shared, "applicable_authority_ids", shared.effective_rule_ids)
    verdict = verify_access_case(state, fixture.case, authority_decision=shared)
    assert (verdict.status, verdict.reason_code) == ("PASS", "ACCESS_GRANTED")


def test_direct_over_budget_sequence_rejects_before_large_detachment_allocation() -> None:
    fixture = _fixture()
    state = _finish_and_grant(fixture)
    direct_over_budget = tuple("entitlement_nested" for _ in range(100_001))
    candidate = state.model_copy(update={"entitlements": direct_over_budget})
    tracemalloc.start()
    try:
        with pytest.raises(AccessVerifierInputError):
            verify_access_case(candidate, fixture.case, authority_decision=_decision(fixture))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000


@pytest.mark.parametrize("summary", ["", "\n", "completed\n", "x" * 241])
def test_finish_summary_is_a_precedence_gate(summary: str) -> None:
    fixture = _fixture()
    state = _finish_and_grant(fixture)
    target = next(
        employee
        for employee in state.employees
        if employee.employee_id == fixture.case.target_employee_id
    ).model_copy(update={"active": False})
    candidate = state.model_copy(
        update={
            "employees": tuple(
                target if employee.employee_id == target.employee_id else employee
                for employee in state.employees
            ),
            "finish_summary": summary,
        }
    )
    verdict = verify_access_case(candidate, fixture.case, authority_decision=_decision(fixture))
    assert (verdict.status, verdict.reason_code) == ("FAIL", "TASK_NOT_FINISHED")


def test_verifier_does_not_mutate_state_case_or_decision() -> None:
    fixture = _fixture()
    state = _finish_and_grant(fixture)
    decision = _decision(fixture)
    before = (
        state.model_dump(mode="json"),
        fixture.case.model_dump(mode="json"),
        decision.model_dump(mode="json"),
    )

    verify_access_case(state, fixture.case, authority_decision=decision)

    assert before == (
        state.model_dump(mode="json"),
        fixture.case.model_dump(mode="json"),
        decision.model_dump(mode="json"),
    )
