from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from shadowskillbench.authority.models import (
    AccessAuthorityQuery,
    AuthorityDecision,
    AuthorityQuery,
    AuthorityRecord,
    DecisionDisposition,
    FinanceAuthorityQuery,
    authority_decision_hash,
    authority_query_hash,
    create_access_authority_query,
    create_finance_authority_query,
    create_issuer_registry,
)
from shadowskillbench.authority.resolver import resolve_authority
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.domains.access.models import parse_access_world
from shadowskillbench.domains.finance.models import FinanceControlFact, parse_finance_world
from shadowskillbench.engine.models import (
    ActionCall,
    Sha256Ref,
    TaskCase,
    WorldState,
    hash_action,
    hash_state,
)


class GateInputError(ValueError):
    """Signals an input that cannot be safely admitted to the action gate."""


class GateOutcome(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    outcome: GateOutcome
    state_hash: Sha256Ref
    action_hash: Sha256Ref
    task_hash: Sha256Ref
    authority_query_hash: Sha256Ref | None
    authority_set_hash: Sha256Ref | None
    authority_decision_hash: Sha256Ref | None
    gate_hash: Sha256Ref

    @field_validator("outcome", mode="before")
    @classmethod
    def _validate_outcome(cls, value: object) -> GateOutcome:
        if type(value) is GateOutcome:
            return cast(GateOutcome, value)
        if type(value) is str:
            return GateOutcome(value)
        raise ValueError("outcome must be an explicit GateOutcome")

    @field_validator(
        "state_hash",
        "action_hash",
        "task_hash",
        "authority_query_hash",
        "authority_set_hash",
        "authority_decision_hash",
        "gate_hash",
        mode="before",
    )
    @classmethod
    def _validate_hashes(cls, value: object) -> Sha256Ref | None:
        if value is None:
            return None
        if type(value) is not str or not value.startswith("sha256:") or len(value) != 71:
            raise ValueError("hash fields must be lowercase sha256 references")
        try:
            int(value.removeprefix("sha256:"), 16)
        except ValueError as error:
            raise ValueError("hash fields must be lowercase sha256 references") from error
        if value != value.lower():
            raise ValueError("hash fields must be lowercase sha256 references")
        return cast(Sha256Ref, value)

    @model_validator(mode="after")
    def _bind_gate_hash(self) -> GateDecision:
        authority_hashes = (
            self.authority_query_hash,
            self.authority_set_hash,
            self.authority_decision_hash,
        )
        if self.outcome is GateOutcome.NOT_APPLICABLE:
            if any(value is not None for value in authority_hashes):
                raise ValueError("not-applicable gate decisions cannot carry authority hashes")
        elif any(value is None for value in authority_hashes):
            raise ValueError("applicable gate decisions require authority hashes")
        if self.gate_hash != _gate_hash(self):
            raise ValueError("gate_hash does not bind gate decision")
        return self


_ACCESS_CONSEQUENTIAL = frozenset({"grant_access"})
_FINANCE_CONSEQUENTIAL = frozenset({"create_adjustment", "classify_adjustment", "publish_report"})
_FINANCE_NONCONSEQUENTIAL = frozenset(
    {
        "get_period",
        "get_ledger_snapshot",
        "get_adjustment",
        "get_approval",
        "reconcile_report",
        "attach_approval",
        "finish_task",
    }
)
_ACCESS_NONCONSEQUENTIAL = frozenset(
    {
        "get_employee",
        "get_application",
        "get_access_request",
        "list_current_access",
        "get_approval",
        "request_approval",
        "revoke_access",
        "finish_task",
    }
)


def _gate_hash_payload(
    *,
    outcome: GateOutcome,
    state_hash: Sha256Ref,
    action_hash: Sha256Ref,
    task_hash: Sha256Ref,
    authority_query_hash: Sha256Ref | None,
    authority_set_hash: Sha256Ref | None,
    authority_decision_hash: Sha256Ref | None,
) -> Sha256Ref:
    return cast(
        Sha256Ref,
        sha256_ref(
            {
                "outcome": outcome.value,
                "state_hash": state_hash,
                "action_hash": action_hash,
                "task_hash": task_hash,
                "authority_query_hash": authority_query_hash,
                "authority_set_hash": authority_set_hash,
                "authority_decision_hash": authority_decision_hash,
            }
        ),
    )


def _gate_hash(decision: GateDecision) -> Sha256Ref:
    return _gate_hash_payload(
        outcome=decision.outcome,
        state_hash=decision.state_hash,
        action_hash=decision.action_hash,
        task_hash=decision.task_hash,
        authority_query_hash=decision.authority_query_hash,
        authority_set_hash=decision.authority_set_hash,
        authority_decision_hash=decision.authority_decision_hash,
    )


def _make_decision(
    *,
    outcome: GateOutcome,
    state_hash: Sha256Ref,
    action_hash: Sha256Ref,
    task_hash: Sha256Ref,
    authority_decision: AuthorityDecision | None = None,
) -> GateDecision:
    query_hash = None
    set_hash = None
    decision_hash = None
    if authority_decision is not None:
        query_hash = authority_decision.authority_query_hash
        set_hash = authority_decision.authority_set_hash
        decision_hash = authority_decision_hash(authority_decision)
    return GateDecision.model_validate(
        {
            "outcome": outcome.value,
            "state_hash": state_hash,
            "action_hash": action_hash,
            "task_hash": task_hash,
            "authority_query_hash": query_hash,
            "authority_set_hash": set_hash,
            "authority_decision_hash": decision_hash,
            "gate_hash": _gate_hash_payload(
                outcome=outcome,
                state_hash=state_hash,
                action_hash=action_hash,
                task_hash=task_hash,
                authority_query_hash=query_hash,
                authority_set_hash=set_hash,
                authority_decision_hash=decision_hash,
            ),
        }
    )


def _exact_state(value: object) -> WorldState:
    if type(value) is not WorldState:
        raise GateInputError("state must be an exact WorldState")
    try:
        return WorldState.model_validate(
            {
                "schema_version": value.schema_version,
                "world_id": value.world_id,
                "domain": value.domain,
                "seed": value.seed,
                "data": value.data,
            }
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise GateInputError("state failed detached revalidation") from error


def _exact_call(value: object) -> ActionCall:
    if type(value) is not ActionCall:
        raise GateInputError("call must be an exact ActionCall")
    try:
        return ActionCall.model_validate(
            {
                "action_id": value.action_id,
                "tool_name": value.tool_name,
                "arguments": value.arguments,
            }
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise GateInputError("call failed detached revalidation") from error


def _exact_case(value: object) -> TaskCase:
    if type(value) is not TaskCase:
        raise GateInputError("case must be an exact TaskCase")
    try:
        return TaskCase.model_validate(
            {
                "case_id": value.case_id,
                "domain": value.domain,
                "world_id": value.world_id,
                "seed": value.seed,
                "objective": value.objective,
                "inputs": value.inputs,
            }
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise GateInputError("case failed detached revalidation") from error


def _exact_query(value: object) -> AccessAuthorityQuery | FinanceAuthorityQuery:
    try:
        authority_query_hash(value)
        if type(value) is AccessAuthorityQuery:
            registry = create_issuer_registry(
                authorizations=[
                    item.model_dump(mode="json") for item in value.issuer_registry.authorizations
                ]
            )
            return create_access_authority_query(
                subject_id=value.subject_id,
                resource_id=value.resource_id,
                action_type=value.action_type,
                at_time=value.at_time,
                organization_id=value.organization_id,
                geography_id=value.geography_id,
                reference_authority_ids=list(value.reference_authority_ids),
                issuer_registry=registry,
                role_id=value.role_id,
                role_derived_access=value.role_derived_access,
            )
        if type(value) is FinanceAuthorityQuery:
            registry = create_issuer_registry(
                authorizations=[
                    item.model_dump(mode="json") for item in value.issuer_registry.authorizations
                ]
            )
            return create_finance_authority_query(
                subject_id=value.subject_id,
                resource_id=value.resource_id,
                action_type=value.action_type,
                at_time=value.at_time,
                organization_id=value.organization_id,
                geography_id=value.geography_id,
                reference_authority_ids=list(value.reference_authority_ids),
                issuer_registry=registry,
                category_id=value.category_id,
                period_id=value.period_id,
                amount_minor=value.amount_minor,
                currency=value.currency,
                unit=value.unit,
                currency_exponent=value.currency_exponent,
            )
    except (AttributeError, TypeError, ValueError) as error:
        raise GateInputError("query failed detached revalidation") from error
    raise GateInputError("query must be an exact authority query")


def _validate_common_bindings(
    state: WorldState, case: TaskCase, query: AccessAuthorityQuery | FinanceAuthorityQuery
) -> None:
    if state.world_id != case.world_id or state.domain != case.domain or state.seed != case.seed:
        raise GateInputError("state must bind case world, domain, and seed")
    if query.domain != state.domain:
        raise GateInputError("query domain must match state and case")


def _string_argument(call: ActionCall, name: str) -> str:
    value = call.arguments.get(name)
    if type(value) is not str:
        raise GateInputError(f"{call.tool_name} requires exact {name}")
    return value


def _access_binds(
    state: WorldState, case: TaskCase, call: ActionCall, query: AccessAuthorityQuery
) -> None:
    try:
        world = parse_access_world(state)
    except ValueError as error:
        raise GateInputError("state is not an admitted access world") from error
    if world.case_id != case.case_id:
        raise GateInputError("access world case_id must match case")
    request_id = _string_argument(call, "request_id")
    request = next((item for item in world.access_requests if item.request_id == request_id), None)
    application = (
        None
        if request is None
        else next(
            (item for item in world.applications if item.application_id == request.application_id),
            None,
        )
    )
    if request is None or application is None:
        raise GateInputError("grant_access target must exist in the admitted access world")
    if (
        query.action_type != "grant_access"
        or query.subject_id != request.employee_id
        or query.resource_id != request.application_id
        or query.role_id != request.requested_role
        or query.organization_id is not None
        or query.geography_id is not None
        or query.role_derived_access is not application.supports_role_derived_access
    ):
        raise GateInputError("access query must bind the grant_access target and immutable facts")


def _control_fact_for_adjustment(
    facts: tuple[FinanceControlFact, ...], adjustment_id: str
) -> FinanceControlFact:
    matches = tuple(item for item in facts if item.adjustment_id == adjustment_id)
    if len(matches) != 1:
        raise GateInputError("finance action must bind exactly one immutable control fact")
    return matches[0]


def _finance_fact(state: WorldState, case: TaskCase, call: ActionCall) -> FinanceControlFact:
    try:
        world = parse_finance_world(state)
    except ValueError as error:
        raise GateInputError("state is not an admitted finance world") from error
    if world.case_id != case.case_id:
        raise GateInputError("finance world case_id must match case")
    if call.tool_name == "create_adjustment":
        fact_id = _string_argument(call, "control_fact_id")
        fact = next((item for item in world.control_facts if item.control_fact_id == fact_id), None)
        if fact is None:
            raise GateInputError("create_adjustment control fact must exist")
        return fact
    if call.tool_name in {"attach_approval", "classify_adjustment"}:
        return _control_fact_for_adjustment(
            world.control_facts, _string_argument(call, "adjustment_id")
        )
    report_id = _string_argument(call, "report_id")
    report = next((item for item in world.monthly_reports if item.report_id == report_id), None)
    if report is None or len(report.adjustment_ids) != 1:
        raise GateInputError("publish_report must contain exactly one adjustment in v1")
    return _control_fact_for_adjustment(world.control_facts, report.adjustment_ids[0])


def _finance_binds(
    state: WorldState, case: TaskCase, call: ActionCall, query: FinanceAuthorityQuery
) -> None:
    fact = _finance_fact(state, case, call)
    if (
        query.action_type != "post_adjustment"
        or query.subject_id != fact.adjustment_id
        or query.resource_id != fact.portco_id
        or query.category_id != fact.original_economic_category
        or query.period_id != fact.period_id
        or query.amount_minor != fact.signed_amount_minor
        or query.currency != fact.currency
        or query.unit != fact.unit
        or query.currency_exponent != fact.currency_exponent
        or query.organization_id is not None
        or query.geography_id is not None
    ):
        raise GateInputError("finance query must bind immutable control facts")


def _resolved_outcome(decision: AuthorityDecision) -> GateOutcome:
    if decision.disposition is DecisionDisposition.PROCEED:
        return GateOutcome.ALLOW
    if decision.disposition in {DecisionDisposition.BLOCK, DecisionDisposition.REQUIRE_APPROVAL}:
        return GateOutcome.BLOCK
    return GateOutcome.ESCALATE


def gate_action(
    state: WorldState,
    call: ActionCall,
    case: TaskCase,
    authority: Sequence[AuthorityRecord],
    *,
    query: AuthorityQuery,
) -> GateDecision:
    """Resolve and intercept a closed set of consequential synthetic actions."""
    admitted_state = _exact_state(state)
    admitted_call = _exact_call(call)
    admitted_case = _exact_case(case)
    admitted_query = _exact_query(query)
    _validate_common_bindings(admitted_state, admitted_case, admitted_query)
    state_hash = hash_state(admitted_state)
    action_hash = hash_action(admitted_call)
    task_hash = cast(Sha256Ref, sha256_ref(admitted_case.model_dump(mode="json")))

    if type(authority) not in {list, tuple}:
        raise GateInputError("authority must be an exact built-in list or tuple")
    try:
        resolved = resolve_authority(authority, admitted_query)
    except (TypeError, ValueError) as error:
        raise GateInputError("authority resolution failed") from error

    if admitted_state.domain == "access_provisioning":
        if type(admitted_query) is not AccessAuthorityQuery:
            raise GateInputError("access state requires an access authority query")
        if admitted_call.tool_name in _ACCESS_NONCONSEQUENTIAL:
            return _make_decision(
                outcome=GateOutcome.NOT_APPLICABLE,
                state_hash=state_hash,
                action_hash=action_hash,
                task_hash=task_hash,
            )
        if admitted_call.tool_name not in _ACCESS_CONSEQUENTIAL:
            raise GateInputError("unknown access tool cannot bypass the gate")
        _access_binds(admitted_state, admitted_case, admitted_call, admitted_query)
    elif admitted_state.domain == "financial_adjustments":
        if type(admitted_query) is not FinanceAuthorityQuery:
            raise GateInputError("finance state requires a finance authority query")
        if admitted_call.tool_name in _FINANCE_NONCONSEQUENTIAL:
            return _make_decision(
                outcome=GateOutcome.NOT_APPLICABLE,
                state_hash=state_hash,
                action_hash=action_hash,
                task_hash=task_hash,
            )
        if admitted_call.tool_name not in _FINANCE_CONSEQUENTIAL:
            raise GateInputError("unknown finance tool cannot bypass the gate")
        _finance_binds(admitted_state, admitted_case, admitted_call, admitted_query)
    else:
        raise GateInputError("unsupported gate domain")

    return _make_decision(
        outcome=_resolved_outcome(resolved),
        state_hash=state_hash,
        action_hash=action_hash,
        task_hash=task_hash,
        authority_decision=resolved,
    )


__all__ = ["GateDecision", "GateInputError", "GateOutcome", "gate_action"]
