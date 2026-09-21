from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from shadowskillbench.authority.models import (
    AccessAuthorityQuery,
    AccessDescriptiveScope,
    AccessEffectiveParameters,
    AccessEvidenceScope,
    AccessRuleScope,
    AuthorityAdmissionError,
    AuthorityDecision,
    AuthorityQuery,
    AuthorityRecord,
    AuthoritySourceType,
    Decision,
    DecisionDisposition,
    DecisionReasonCode,
    DiscardedAuthority,
    DiscardReasonCode,
    EffectiveParameters,
    FinanceAuthorityQuery,
    FinanceDescriptiveScope,
    FinanceEffectiveParameters,
    FinanceEvidenceScope,
    FinanceRuleScope,
    IssuerAuthorization,
    IssuerRegistry,
    NormativeStatus,
    RuleDisposition,
    ScopeKind,
    authority_set_hash,
    create_authority_decision,
)

_DESCRIPTIVE_SOURCES = frozenset(
    {
        AuthoritySourceType.BEHAVIOR_TRACE,
        AuthoritySourceType.PROCEDURE_GUIDE,
        AuthoritySourceType.VERIFIER_RESULT,
    }
)
_RULE_SOURCES = frozenset(
    {
        AuthoritySourceType.POLICY,
        AuthoritySourceType.SIGNED_DIRECTIVE,
        AuthoritySourceType.CHANGE_RECORD,
        AuthoritySourceType.SYSTEM_CONFIGURATION,
    }
)
_EVIDENCE_SOURCES = frozenset({AuthoritySourceType.APPROVAL, AuthoritySourceType.WAIVER})
_CONFLICTING_STATUSES = frozenset({NormativeStatus.CONFLICTING, NormativeStatus.UNRESOLVED})
_EXACT_MODEL_FIELDS: dict[type[object], tuple[str, ...]] = {
    model_type: tuple(model_type.model_fields)
    for model_type in (
        AccessAuthorityQuery,
        AccessDescriptiveScope,
        AccessEffectiveParameters,
        AccessEvidenceScope,
        AccessRuleScope,
        AuthorityRecord,
        FinanceAuthorityQuery,
        FinanceDescriptiveScope,
        FinanceEffectiveParameters,
        FinanceEvidenceScope,
        FinanceRuleScope,
        IssuerAuthorization,
        IssuerRegistry,
    )
}
_EXACT_ENUM_TYPES = frozenset(
    {
        AuthoritySourceType,
        NormativeStatus,
        RuleDisposition,
        ScopeKind,
    }
)


def _admission_error(message: str, error: Exception | None = None) -> AuthorityAdmissionError:
    if error is None:
        return AuthorityAdmissionError(message)
    return AuthorityAdmissionError(f"{message}: {error}")


def _has_exact_nested_structure(value: object) -> bool:
    active_path: set[int] = set()
    stack: list[tuple[bool, object]] = [(True, value)]

    while stack:
        entering, current = stack.pop()
        if not entering:
            active_path.remove(cast(int, current))
            continue

        current_type = type(current)
        if current is None or current_type in {str, int, bool} or current_type in _EXACT_ENUM_TYPES:
            continue

        if current_type in {list, tuple}:
            node_id = id(current)
            if node_id in active_path:
                return False
            active_path.add(node_id)
            stack.append((False, node_id))
            sequence = cast(list[object] | tuple[object, ...], current)
            for index in range(len(sequence) - 1, -1, -1):
                stack.append((True, sequence[index]))
            continue

        fields = _EXACT_MODEL_FIELDS.get(current_type)
        if fields is None:
            return False

        node_id = id(current)
        if node_id in active_path:
            return False
        values = object.__getattribute__(current, "__dict__")
        if type(values) is not dict:
            return False
        if len(values) != len(fields) + 1:
            return False
        if "_validated" not in values or values["_validated"] is not True:
            return False
        if any(field_name not in values for field_name in fields):
            return False

        active_path.add(node_id)
        stack.append((False, node_id))
        for field_name in reversed(fields):
            stack.append((True, values[field_name]))

    return True


def _revalidate_query(query: object) -> AccessAuthorityQuery | FinanceAuthorityQuery:
    try:
        if type(query) is AccessAuthorityQuery:
            candidate = cast(AccessAuthorityQuery, query)
            if not _has_exact_nested_structure(candidate):
                raise ValueError("authority_query must have exact validated nested structure")
            return AccessAuthorityQuery.model_validate(candidate.model_dump(mode="json"))
        if type(query) is FinanceAuthorityQuery:
            candidate = cast(FinanceAuthorityQuery, query)
            if not _has_exact_nested_structure(candidate):
                raise ValueError("authority_query must have exact validated nested structure")
            return FinanceAuthorityQuery.model_validate(candidate.model_dump(mode="json"))
    except (RecursionError, TypeError, ValueError) as error:
        raise _admission_error("authority_query failed detached revalidation", error)
    raise _admission_error("authority_query must be an exact validated authority query")


def _admit_records(
    records: Sequence[AuthorityRecord], query: AccessAuthorityQuery | FinanceAuthorityQuery
) -> tuple[tuple[AuthorityRecord, ...], dict[str, AuthorityRecord], dict[str, str]]:
    if type(records) not in {list, tuple}:
        raise _admission_error("authority_records must be an exact built-in list or tuple")

    detached: list[AuthorityRecord] = []
    try:
        for item in records:
            if type(item) is not AuthorityRecord:
                raise ValueError("authority_records must contain exact AuthorityRecord models")
            record = cast(AuthorityRecord, item)
            if not _has_exact_nested_structure(record):
                raise ValueError("authority_records must have exact validated nested structure")
            detached.append(AuthorityRecord.model_validate(record.model_dump(mode="json")))
    except (RecursionError, TypeError, ValueError) as error:
        raise _admission_error("authority_records failed detached revalidation", error)

    ordered = tuple(sorted(detached, key=lambda record: record.authority_id))
    records_by_id = {record.authority_id: record for record in ordered}
    if len(records_by_id) != len(ordered):
        raise _admission_error("authority_records must have unique authority IDs")

    _validate_issuer_authorization(ordered, query)
    incoming = _validate_supersession_edges(ordered, records_by_id)
    _validate_evidence_edges(ordered, records_by_id)

    for record in ordered:
        if (
            record.source_type in _RULE_SOURCES
            and record.normative_status is NormativeStatus.SUPERSEDED
            and record.authority_id not in incoming
        ):
            raise _admission_error("superseded rule must have an admitted incoming supersession")

    _validate_acyclic_supersession(ordered)
    rule_ids = {record.authority_id for record in ordered if record.source_type in _RULE_SOURCES}
    for reference_id in query.reference_authority_ids:
        if reference_id not in rule_ids:
            raise _admission_error("reference_authority_ids must identify admitted rules")
    return ordered, records_by_id, incoming


def _validate_issuer_authorization(
    records: tuple[AuthorityRecord, ...], query: AccessAuthorityQuery | FinanceAuthorityQuery
) -> None:
    registry = query.issuer_registry
    authorizations: dict[tuple[str, str], IssuerAuthorization] = {
        (authorization.issuer_id, authorization.issuer_role): authorization
        for authorization in registry.authorizations
    }
    for record in records:
        authorization = authorizations.get((record.issuer_id, record.issuer_role))
        if authorization is None:
            raise _admission_error("record issuer and role are not registered")
        if record.authority_rank != authorization.authority_rank:
            raise _admission_error("record authority rank must exactly match issuer registry")
        if record.scope.domain not in authorization.allowed_domains:
            raise _admission_error("record domain is not authorized for issuer")
        if record.source_type not in authorization.allowed_source_types:
            raise _admission_error("record source type is not authorized for issuer")
        if record.supersedes and not authorization.can_supersede:
            raise _admission_error("issuer is not authorized to supersede")
        if record.source_type in _EVIDENCE_SOURCES and not authorization.can_issue_scoped_evidence:
            raise _admission_error("issuer is not authorized to issue scoped evidence")


def _validate_supersession_edges(
    records: tuple[AuthorityRecord, ...], records_by_id: dict[str, AuthorityRecord]
) -> dict[str, str]:
    incoming: dict[str, str] = {}
    for source in records:
        if len(source.supersedes) > 1:
            raise _admission_error("v1 supersession sources may name at most one target")
        if not source.supersedes:
            continue
        if source.source_type not in _RULE_SOURCES:
            raise _admission_error("only rule records may supersede")
        target_id = source.supersedes[0]
        target = records_by_id.get(target_id)
        if target is None or target.source_type not in _RULE_SOURCES:
            raise _admission_error("supersession target must be an admitted rule")
        if source.authority_id == target.authority_id:
            raise _admission_error("supersession self-edge is invalid")
        if target.authority_id in incoming:
            raise _admission_error("v1 supersession targets may have one incoming edge")
        if source.authority_rank < target.authority_rank:
            raise _admission_error("superseder rank must be at least target rank")
        if source.effective_at < target.effective_at:
            raise _admission_error("superseder cannot predate its target")
        if source.scope.domain != target.scope.domain or source.action_type != target.action_type:
            raise _admission_error("supersession requires matching domain and action")
        if not _scope_contains(target, source):
            raise _admission_error("supersession scope cannot be broader than its target")
        if not _time_contains(target, source):
            raise _admission_error("supersession time cannot be broader than its target")
        incoming[target.authority_id] = source.authority_id
    return incoming


def _validate_evidence_edges(
    records: tuple[AuthorityRecord, ...], records_by_id: dict[str, AuthorityRecord]
) -> None:
    for evidence in records:
        if evidence.source_type not in _EVIDENCE_SOURCES:
            continue
        for target_id in evidence.exception_to:
            target = records_by_id.get(target_id)
            if target is None or target.source_type not in _RULE_SOURCES:
                raise _admission_error("evidence target must be an admitted rule")
            if (
                evidence.scope.domain != target.scope.domain
                or evidence.action_type != target.action_type
            ):
                raise _admission_error("evidence requires matching domain and action")
            if not _scope_contains(target, evidence):
                raise _admission_error("evidence scope must be contained by every target rule")
            if not _time_contains(target, evidence):
                raise _admission_error("evidence time must be contained by every target rule")


def _validate_acyclic_supersession(records: tuple[AuthorityRecord, ...]) -> None:
    targets = {record.authority_id: record.supersedes[0] for record in records if record.supersedes}
    visited: set[str] = set()
    for start in targets:
        path: set[str] = set()
        current: str | None = start
        while current is not None and current in targets:
            if current in path:
                raise _admission_error("supersession graph must be acyclic")
            if current in visited:
                break
            path.add(current)
            current = targets[current]
        visited.update(path)


def _scope_contains(outer_record: AuthorityRecord, inner_record: AuthorityRecord) -> bool:
    outer = outer_record.scope
    inner = inner_record.scope
    if outer.domain != inner.domain:
        return False
    common_fields = (
        "subject_id",
        "resource_id",
        "organization_id",
        "geography_id",
    )
    for field in common_fields:
        outer_value = getattr(outer, field)
        inner_value = getattr(inner, field)
        if outer_value is not None and outer_value != inner_value:
            return False

    if outer.domain == "access_provisioning":
        access_outer = cast(AccessRuleScope, outer)
        access_inner = cast(AccessRuleScope, inner)
        return access_outer.role_id is None or access_outer.role_id == access_inner.role_id

    finance_outer = cast(FinanceRuleScope, outer)
    finance_inner = cast(FinanceRuleScope, inner)
    for field in ("category_id", "period_id"):
        outer_value = getattr(finance_outer, field)
        inner_value = getattr(finance_inner, field)
        if outer_value is not None and outer_value != inner_value:
            return False
    if (
        finance_outer.currency != finance_inner.currency
        or finance_outer.unit != finance_inner.unit
        or finance_outer.currency_exponent != finance_inner.currency_exponent
    ):
        return False
    return _interval_contains(
        finance_outer.minimum_amount_minor,
        finance_outer.minimum_inclusive,
        finance_outer.maximum_amount_minor,
        finance_outer.maximum_inclusive,
        finance_inner.minimum_amount_minor,
        finance_inner.minimum_inclusive,
        finance_inner.maximum_amount_minor,
        finance_inner.maximum_inclusive,
    )


def _interval_contains(
    outer_minimum: int | None,
    outer_minimum_inclusive: bool,
    outer_maximum: int | None,
    outer_maximum_inclusive: bool,
    inner_minimum: int | None,
    inner_minimum_inclusive: bool,
    inner_maximum: int | None,
    inner_maximum_inclusive: bool,
) -> bool:
    if outer_minimum is not None:
        if inner_minimum is None or inner_minimum < outer_minimum:
            return False
        if (
            inner_minimum == outer_minimum
            and inner_minimum_inclusive
            and not outer_minimum_inclusive
        ):
            return False
    if outer_maximum is not None:
        if inner_maximum is None or inner_maximum > outer_maximum:
            return False
        if (
            inner_maximum == outer_maximum
            and inner_maximum_inclusive
            and not outer_maximum_inclusive
        ):
            return False
    return True


def _time_contains(outer: AuthorityRecord, inner: AuthorityRecord) -> bool:
    if inner.effective_at < outer.effective_at:
        return False
    if outer.expires_at is None:
        return True
    return inner.expires_at is not None and inner.expires_at <= outer.expires_at


def _matches_query(
    record: AuthorityRecord, query: AccessAuthorityQuery | FinanceAuthorityQuery
) -> bool:
    scope = record.scope
    if scope.domain != query.domain:
        return False
    for field in ("subject_id", "resource_id", "organization_id", "geography_id"):
        scope_value = getattr(scope, field)
        query_value = getattr(query, field)
        if scope_value is not None and scope_value != query_value:
            return False

    if type(query) is AccessAuthorityQuery:
        access_scope = cast(AccessRuleScope, scope)
        return access_scope.role_id is None or access_scope.role_id == query.role_id

    finance_scope = cast(FinanceRuleScope, scope)
    finance_query = cast(FinanceAuthorityQuery, query)
    for field in ("category_id", "period_id"):
        scope_value = getattr(finance_scope, field)
        query_value = getattr(finance_query, field)
        if scope_value is not None and scope_value != query_value:
            return False
    if (
        finance_scope.currency != finance_query.currency
        or finance_scope.unit != finance_query.unit
        or finance_scope.currency_exponent != finance_query.currency_exponent
    ):
        return False
    amount = abs(finance_query.amount_minor)
    if finance_scope.minimum_amount_minor is not None and (
        amount < finance_scope.minimum_amount_minor
        or (amount == finance_scope.minimum_amount_minor and not finance_scope.minimum_inclusive)
    ):
        return False
    return finance_scope.maximum_amount_minor is None or (
        amount < finance_scope.maximum_amount_minor
        or (amount == finance_scope.maximum_amount_minor and finance_scope.maximum_inclusive)
    )


def _rule_disposition_and_parameters(
    record: AuthorityRecord, query: AccessAuthorityQuery | FinanceAuthorityQuery
) -> tuple[RuleDisposition, EffectiveParameters]:
    if type(query) is AccessAuthorityQuery:
        scope = cast(AccessRuleScope, record.scope)
        if scope.rule_disposition is RuleDisposition.BLOCK:
            disposition = RuleDisposition.BLOCK
        elif scope.rule_disposition is RuleDisposition.REQUIRE_APPROVAL:
            disposition = RuleDisposition.REQUIRE_APPROVAL
        elif scope.requires_security_approval or (
            query.role_derived_access and not scope.role_derived_without_approval
        ):
            disposition = RuleDisposition.REQUIRE_APPROVAL
        else:
            disposition = RuleDisposition.PROCEED
        return (
            disposition,
            AccessEffectiveParameters(
                domain="access_provisioning",
                allowed_action=scope.allowed_action,
                requires_security_approval=scope.requires_security_approval,
                role_derived_without_approval=scope.role_derived_without_approval,
            ),
        )

    finance_query = cast(FinanceAuthorityQuery, query)
    scope = cast(FinanceRuleScope, record.scope)
    if scope.rule_disposition is RuleDisposition.BLOCK:
        disposition = RuleDisposition.BLOCK
    elif scope.rule_disposition is RuleDisposition.REQUIRE_APPROVAL:
        disposition = RuleDisposition.REQUIRE_APPROVAL
    else:
        amount = abs(finance_query.amount_minor)
        if (
            finance_query.category_id in scope.restricted_categories
            or amount > scope.threshold_minor
        ):
            disposition = RuleDisposition.REQUIRE_APPROVAL
        else:
            disposition = RuleDisposition.PROCEED
    return (
        disposition,
        FinanceEffectiveParameters.model_validate(
            {
                "domain": "financial_adjustments",
                "threshold_minor": scope.threshold_minor,
                "currency": scope.currency,
                "currency_exponent": scope.currency_exponent,
                "restricted_categories": list(scope.restricted_categories),
            }
        ),
    )


def _path_for_effective_rule(
    record: AuthorityRecord,
    records_by_id: dict[str, AuthorityRecord],
    candidate_rule_ids: set[str],
) -> tuple[str, ...]:
    reverse_path = [record.authority_id]
    current = record
    while current.supersedes:
        target_id = current.supersedes[0]
        if target_id not in candidate_rule_ids:
            break
        target = records_by_id[target_id]
        reverse_path.append(target.authority_id)
        current = target
    return tuple(reversed(reverse_path))


def _canonical_path(
    effective_records: tuple[AuthorityRecord, ...],
    records_by_id: dict[str, AuthorityRecord],
    candidate_rule_ids: set[str],
    references: tuple[str, ...],
) -> tuple[str, ...]:
    paths = [
        (
            record.authority_id,
            _path_for_effective_rule(record, records_by_id, candidate_rule_ids),
        )
        for record in effective_records
    ]
    reference_ids = set(references)
    referenced = [item for item in paths if set(item[1]).intersection(reference_ids)]
    selected = min(referenced or paths, key=lambda item: (item[0], item[1]))
    return selected[1]


def _decision_disposition(disposition: RuleDisposition) -> DecisionDisposition:
    return DecisionDisposition(disposition.value)


def _ordinary_decision(disposition: RuleDisposition) -> Decision:
    return Decision(disposition.value)


def _discarded_records(discarded: dict[str, DiscardReasonCode]) -> list[DiscardedAuthority]:
    return [
        DiscardedAuthority(authority_id=authority_id, reason_code=reason_code)
        for authority_id, reason_code in sorted(discarded.items())
    ]


def _build_decision(
    *,
    query: AccessAuthorityQuery | FinanceAuthorityQuery,
    set_hash: str,
    decision: Decision,
    disposition: DecisionDisposition,
    parameters: EffectiveParameters | None,
    applicable_ids: tuple[str, ...],
    discarded: dict[str, DiscardReasonCode],
    path: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    conflict_ids: tuple[str, ...],
    effective_rule_ids: tuple[str, ...],
    reason_code: DecisionReasonCode,
):
    return create_authority_decision(
        authority_query=query,
        authority_set_hash=set_hash,
        decision=decision,
        disposition=disposition,
        effective_parameters=parameters,
        applicable_authority_ids=list(applicable_ids),
        discarded_authorities=_discarded_records(discarded),
        supersession_path=list(path),
        exception_or_approval_ids=list(evidence_ids),
        unresolved_conflict_ids=list(conflict_ids),
        effective_rule_ids=list(effective_rule_ids),
        reason_code=reason_code,
    )


def resolve_authority(
    records: Sequence[AuthorityRecord], query: AuthorityQuery
) -> AuthorityDecision:
    """Resolve one detached, fully admitted authority corpus for one explicit query."""

    admitted_query = _revalidate_query(query)
    admitted_records, records_by_id, incoming = _admit_records(records, admitted_query)
    try:
        set_hash = authority_set_hash(list(admitted_records))
    except (TypeError, ValueError) as error:
        raise _admission_error("authority set hash could not be bound", error)

    discarded: dict[str, DiscardReasonCode] = {}

    def discard(record: AuthorityRecord, reason: DiscardReasonCode) -> None:
        discarded.setdefault(record.authority_id, reason)

    rule_candidates: list[AuthorityRecord] = []
    evidence_candidates: list[AuthorityRecord] = []
    for record in admitted_records:
        if record.source_type in _DESCRIPTIVE_SOURCES:
            discard(record, DiscardReasonCode.DESCRIPTIVE_ONLY)
            continue
        if admitted_query.at_time < record.effective_at:
            discard(record, DiscardReasonCode.NOT_YET_EFFECTIVE)
            continue
        if record.expires_at is not None and admitted_query.at_time >= record.expires_at:
            discard(record, DiscardReasonCode.EXPIRED_AT_QUERY)
            continue
        if record.action_type != admitted_query.action_type:
            discard(record, DiscardReasonCode.ACTION_MISMATCH)
            continue
        if not _matches_query(record, admitted_query):
            discard(record, DiscardReasonCode.SCOPE_MISMATCH)
            continue
        if record.source_type in _RULE_SOURCES:
            rule_candidates.append(record)
        elif record.source_type in _EVIDENCE_SOURCES:
            evidence_candidates.append(record)
        else:
            raise _admission_error("record source type is outside the closed authority contract")

    candidate_rule_ids = {record.authority_id for record in rule_candidates}
    superseded_ids = {
        target_id
        for target_id, source_id in incoming.items()
        if target_id in candidate_rule_ids and source_id in candidate_rule_ids
    }
    for authority_id in superseded_ids:
        discard(records_by_id[authority_id], DiscardReasonCode.SUPERSEDED)
    surviving_rules = tuple(
        record for record in rule_candidates if record.authority_id not in superseded_ids
    )

    if surviving_rules:
        highest_rank = max(record.authority_rank for record in surviving_rules)
        top_rules = tuple(
            record for record in surviving_rules if record.authority_rank == highest_rank
        )
        for record in surviving_rules:
            if record.authority_rank != highest_rank:
                discard(record, DiscardReasonCode.LOWER_AUTHORITY_RANK)
    else:
        top_rules = ()

    applicable_ids = tuple(sorted(record.authority_id for record in top_rules))
    if not top_rules:
        for evidence in evidence_candidates:
            discard(evidence, DiscardReasonCode.EVIDENCE_TARGET_NOT_EFFECTIVE)
        return _build_decision(
            query=admitted_query,
            set_hash=set_hash,
            decision=Decision.ESCALATE,
            disposition=DecisionDisposition.ESCALATE,
            parameters=None,
            applicable_ids=(),
            discarded=discarded,
            path=(),
            evidence_ids=(),
            conflict_ids=(),
            effective_rule_ids=(),
            reason_code=DecisionReasonCode.NO_APPLICABLE_AUTHORITY,
        )

    status_conflict_ids = tuple(
        sorted(
            record.authority_id
            for record in top_rules
            if record.normative_status in _CONFLICTING_STATUSES
        )
    )
    computations = [
        _rule_disposition_and_parameters(record, admitted_query) for record in top_rules
    ]
    first_disposition, first_parameters = computations[0]
    is_equivalent = all(
        disposition is first_disposition and parameters == first_parameters
        for disposition, parameters in computations[1:]
    )
    if status_conflict_ids or not is_equivalent:
        for evidence in evidence_candidates:
            discard(evidence, DiscardReasonCode.EVIDENCE_TARGET_NOT_EFFECTIVE)
        conflict_ids = status_conflict_ids or applicable_ids
        return _build_decision(
            query=admitted_query,
            set_hash=set_hash,
            decision=Decision.ESCALATE,
            disposition=DecisionDisposition.ESCALATE,
            parameters=None,
            applicable_ids=applicable_ids,
            discarded=discarded,
            path=(),
            evidence_ids=(),
            conflict_ids=conflict_ids,
            effective_rule_ids=(),
            reason_code=DecisionReasonCode.UNRESOLVED_TOP_RANK_CONFLICT,
        )

    effective_rule_ids = applicable_ids
    full_path = _canonical_path(
        top_rules,
        records_by_id,
        candidate_rule_ids,
        admitted_query.reference_authority_ids,
    )
    reporting_path = full_path
    target_ids = set(effective_rule_ids)
    eligible_evidence: list[AuthorityRecord] = []
    for evidence in evidence_candidates:
        if target_ids.intersection(evidence.exception_to):
            eligible_evidence.append(evidence)
        else:
            discard(evidence, DiscardReasonCode.EVIDENCE_TARGET_NOT_EFFECTIVE)

    waivers = [
        evidence
        for evidence in eligible_evidence
        if evidence.source_type is AuthoritySourceType.WAIVER
        and first_disposition in {RuleDisposition.BLOCK, RuleDisposition.REQUIRE_APPROVAL}
    ]
    approvals = [
        evidence
        for evidence in eligible_evidence
        if evidence.source_type is AuthoritySourceType.APPROVAL
        and first_disposition is RuleDisposition.REQUIRE_APPROVAL
    ]
    if waivers:
        waiver_ids = tuple(sorted(record.authority_id for record in waivers))
        for evidence in eligible_evidence:
            if evidence.authority_id not in waiver_ids:
                discard(evidence, DiscardReasonCode.EVIDENCE_NOT_NEEDED)
        return _build_decision(
            query=admitted_query,
            set_hash=set_hash,
            decision=Decision.PROCEED_UNDER_EXCEPTION,
            disposition=DecisionDisposition.PROCEED,
            parameters=first_parameters,
            applicable_ids=applicable_ids,
            discarded=discarded,
            path=reporting_path,
            evidence_ids=waiver_ids,
            conflict_ids=(),
            effective_rule_ids=effective_rule_ids,
            reason_code=DecisionReasonCode.VALID_WAIVER,
        )
    if approvals:
        approval_ids = tuple(sorted(record.authority_id for record in approvals))
        for evidence in eligible_evidence:
            if evidence.authority_id not in approval_ids:
                discard(evidence, DiscardReasonCode.EVIDENCE_NOT_NEEDED)
        return _build_decision(
            query=admitted_query,
            set_hash=set_hash,
            decision=Decision.PROCEED,
            disposition=DecisionDisposition.PROCEED,
            parameters=first_parameters,
            applicable_ids=applicable_ids,
            discarded=discarded,
            path=reporting_path,
            evidence_ids=approval_ids,
            conflict_ids=(),
            effective_rule_ids=effective_rule_ids,
            reason_code=DecisionReasonCode.VALID_APPROVAL,
        )
    for evidence in eligible_evidence:
        discard(evidence, DiscardReasonCode.EVIDENCE_NOT_NEEDED)

    predecessor_ids = set(full_path[:-1])
    follows_reference = bool(predecessor_ids.intersection(admitted_query.reference_authority_ids))
    decision = (
        Decision.FOLLOW_SUPERSEDING_AUTHORITY
        if len(full_path) > 1 and follows_reference
        else _ordinary_decision(first_disposition)
    )
    reason_code = (
        DecisionReasonCode.EFFECTIVE_SUPERSEDING_RULE
        if decision is Decision.FOLLOW_SUPERSEDING_AUTHORITY
        else DecisionReasonCode.EFFECTIVE_RULE
    )
    return _build_decision(
        query=admitted_query,
        set_hash=set_hash,
        decision=decision,
        disposition=_decision_disposition(first_disposition),
        parameters=first_parameters,
        applicable_ids=applicable_ids,
        discarded=discarded,
        path=reporting_path,
        evidence_ids=(),
        conflict_ids=(),
        effective_rule_ids=effective_rule_ids,
        reason_code=reason_code,
    )


__all__ = ["resolve_authority"]
