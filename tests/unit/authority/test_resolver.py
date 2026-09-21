from __future__ import annotations

import pytest

from shadowskillbench.authority.models import (
    AuthorityAdmissionError,
    AuthorityRecord,
    AuthoritySourceType,
    Decision,
    DecisionDisposition,
    DecisionReasonCode,
    NormativeStatus,
    RuleDisposition,
    create_access_authority_query,
    create_authority_record,
    create_finance_authority_query,
    create_issuer_registry,
)
from shadowskillbench.authority.resolver import _has_exact_nested_structure, resolve_authority


def _authorizations() -> list[dict[str, object]]:
    return [
        {
            "issuer_id": "issuer_primary",
            "issuer_role": "policy_owner",
            "authority_rank": 10,
            "allowed_domains": ["access_provisioning", "financial_adjustments"],
            "allowed_source_types": [
                "APPROVAL",
                "BEHAVIOR_TRACE",
                "CHANGE_RECORD",
                "POLICY",
                "SIGNED_DIRECTIVE",
                "SYSTEM_CONFIGURATION",
                "WAIVER",
            ],
            "can_supersede": True,
            "can_issue_scoped_evidence": True,
        },
        {
            "issuer_id": "issuer_senior",
            "issuer_role": "policy_owner",
            "authority_rank": 20,
            "allowed_domains": ["access_provisioning", "financial_adjustments"],
            "allowed_source_types": ["POLICY"],
            "can_supersede": True,
            "can_issue_scoped_evidence": False,
        },
    ]


def _registry(
    *,
    primary_can_evidence: bool = True,
    primary_can_supersede: bool = True,
    primary_domains: list[str] | None = None,
    primary_sources: list[str] | None = None,
) -> object:
    authorizations = _authorizations()
    authorizations[0]["can_issue_scoped_evidence"] = primary_can_evidence
    authorizations[0]["can_supersede"] = primary_can_supersede
    if primary_domains is not None:
        authorizations[0]["allowed_domains"] = primary_domains
    if primary_sources is not None:
        authorizations[0]["allowed_source_types"] = primary_sources
    return create_issuer_registry(authorizations=authorizations)


def _access_scope(
    *,
    subject_id: str | None = "employee_a",
    role_id: str | None = "role_a",
    organization_id: str | None = None,
    disposition: RuleDisposition | str = RuleDisposition.PROCEED,
    requires_security_approval: bool = False,
    role_derived_without_approval: bool = True,
) -> dict[str, object]:
    return {
        "domain": "access_provisioning",
        "scope_kind": "rule",
        "subject_id": subject_id,
        "resource_id": "application_a",
        "organization_id": organization_id,
        "geography_id": None,
        "role_id": role_id,
        "rule_disposition": disposition,
        "allowed_action": "grant_access",
        "requires_security_approval": requires_security_approval,
        "role_derived_without_approval": role_derived_without_approval,
    }


def _access_evidence_scope(*, organization_id: str | None = None) -> dict[str, object]:
    return {
        "domain": "access_provisioning",
        "scope_kind": "evidence",
        "subject_id": "employee_a",
        "resource_id": "application_a",
        "organization_id": organization_id,
        "geography_id": None,
        "role_id": "role_a",
    }


def _finance_scope(
    *,
    threshold_minor: int = 500,
    disposition: RuleDisposition | str = RuleDisposition.PROCEED,
    minimum_amount_minor: int | None = None,
    minimum_inclusive: bool = False,
    maximum_amount_minor: int | None = None,
    maximum_inclusive: bool = False,
) -> dict[str, object]:
    return {
        "domain": "financial_adjustments",
        "scope_kind": "rule",
        "subject_id": "adjustment_a",
        "resource_id": "portco_a",
        "organization_id": None,
        "geography_id": None,
        "category_id": "category_a",
        "period_id": "period_a",
        "minimum_amount_minor": minimum_amount_minor,
        "minimum_inclusive": minimum_inclusive,
        "maximum_amount_minor": maximum_amount_minor,
        "maximum_inclusive": maximum_inclusive,
        "currency": "USD",
        "unit": "minor_units",
        "currency_exponent": 2,
        "rule_disposition": disposition,
        "threshold_minor": threshold_minor,
        "restricted_categories": [],
    }


def _record(
    authority_id: str,
    *,
    scope: dict[str, object] | None = None,
    source_type: AuthoritySourceType | str = AuthoritySourceType.POLICY,
    status: NormativeStatus | str = NormativeStatus.ACTIVE_AUTHORITY,
    issuer_id: str = "issuer_primary",
    authority_rank: int = 10,
    action_type: str = "grant_access",
    effective_at: str = "2026-01-01T00:00:00Z",
    expires_at: str | None = None,
    supersedes: list[str] | None = None,
    exception_to: list[str] | None = None,
) -> AuthorityRecord:
    return create_authority_record(
        authority_id=authority_id,
        schema_version="1.0",
        source_type=source_type,
        title=f"Authority {authority_id}",
        issuer_id=issuer_id,
        issuer_role="policy_owner",
        authority_rank=authority_rank,
        action_type=action_type,
        scope=_access_scope() if scope is None else scope,
        effective_at=effective_at,
        expires_at=expires_at,
        supersedes=[] if supersedes is None else supersedes,
        exception_to=[] if exception_to is None else exception_to,
        provenance_locator=f"authority/{authority_id}",
        normative_status=status,
    )


def _access_query(
    *,
    at_time: str = "2026-02-01T00:00:00Z",
    role_id: str = "role_a",
    references: list[str] | None = None,
    registry: object | None = None,
    organization_id: str | None = None,
) -> object:
    return create_access_authority_query(
        subject_id="employee_a",
        resource_id="application_a",
        action_type="grant_access",
        at_time=at_time,
        organization_id=organization_id,
        geography_id=None,
        reference_authority_ids=[] if references is None else references,
        issuer_registry=_registry() if registry is None else registry,
        role_id=role_id,
        role_derived_access=True,
    )


def _finance_query(amount_minor: int) -> object:
    return create_finance_authority_query(
        subject_id="adjustment_a",
        resource_id="portco_a",
        action_type="post_adjustment",
        at_time="2026-02-01T00:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[],
        issuer_registry=_registry(),
        category_id="category_a",
        period_id="period_a",
        amount_minor=amount_minor,
        currency="USD",
        unit="minor_units",
        currency_exponent=2,
    )


def test_resolves_exact_rule_and_binds_full_admitted_set() -> None:
    governing = _record("rule_governing")
    descriptive = _record(
        "trace_a",
        source_type="BEHAVIOR_TRACE",
        status="DESCRIPTIVE_ONLY",
        scope={
            "domain": "access_provisioning",
            "scope_kind": "descriptive",
            "subject_id": None,
            "resource_id": None,
            "organization_id": None,
            "geography_id": None,
            "role_id": None,
        },
    )

    decision = resolve_authority([descriptive, governing], _access_query())

    assert decision.decision is Decision.PROCEED
    assert decision.disposition is DecisionDisposition.PROCEED
    assert decision.reason_code is DecisionReasonCode.EFFECTIVE_RULE
    assert decision.applicable_authority_ids == ("rule_governing",)
    assert decision.effective_rule_ids == ("rule_governing",)
    assert decision.supersession_path == ("rule_governing",)
    discarded = [
        (item.authority_id, item.reason_code.value) for item in decision.discarded_authorities
    ]
    assert discarded == [("trace_a", "DESCRIPTIVE_ONLY")]
    assert decision.authority_set_hash != ""


def test_rejects_invalid_or_stale_input_before_ordinary_resolution() -> None:
    record = _record("rule_a")
    with pytest.raises(AuthorityAdmissionError):
        resolve_authority({record}, _access_query())  # type: ignore[arg-type]

    object.__setattr__(record, "content_hash", "sha256:" + "0" * 64)
    with pytest.raises(AuthorityAdmissionError):
        resolve_authority([record], _access_query())

    unknown = _record("rule_unknown", issuer_id="issuer_unknown")
    with pytest.raises(AuthorityAdmissionError):
        resolve_authority([unknown], _access_query())


def test_whole_corpus_rejects_an_unused_lower_rank_supersession_edge() -> None:
    target = _record(
        "target",
        issuer_id="issuer_senior",
        authority_rank=20,
        status="SUPERSEDED",
    )
    lower = _record(
        "unused_lower",
        authority_rank=10,
        effective_at="2026-01-02T00:00:00Z",
        supersedes=["target"],
    )

    with pytest.raises(AuthorityAdmissionError):
        resolve_authority([target, lower], _access_query())


def test_supersession_is_query_relative_and_follow_requires_explicit_predecessor_reference() -> (
    None
):
    base = _record("rule_base", status="SUPERSEDED", scope=_access_scope(role_id=None))
    narrow = _record(
        "rule_narrow",
        issuer_id="issuer_senior",
        authority_rank=20,
        scope=_access_scope(role_id="role_a", requires_security_approval=True),
        effective_at="2026-01-15T00:00:00Z",
        supersedes=["rule_base"],
    )

    historical = resolve_authority([narrow, base], _access_query(at_time="2026-01-10T00:00:00Z"))
    outside_scope = resolve_authority([base, narrow], _access_query(role_id="role_other"))
    followed = resolve_authority([base, narrow], _access_query(references=["rule_base"]))

    assert historical.decision is Decision.PROCEED
    assert outside_scope.decision is Decision.PROCEED
    assert outside_scope.effective_rule_ids == ("rule_base",)
    assert followed.decision is Decision.FOLLOW_SUPERSEDING_AUTHORITY
    assert followed.disposition is DecisionDisposition.REQUIRE_APPROVAL
    assert followed.supersession_path == ("rule_base", "rule_narrow")
    assert followed.reason_code is DecisionReasonCode.EFFECTIVE_SUPERSEDING_RULE


def test_same_rank_conflict_never_uses_input_order_or_recency_as_a_tiebreak() -> None:
    allow = _record("allow", scope=_access_scope())
    block = _record(
        "block",
        scope=_access_scope(disposition="BLOCK"),
        effective_at="2026-01-02T00:00:00Z",
    )

    first = resolve_authority([allow, block], _access_query())
    second = resolve_authority([block, allow], _access_query())

    assert first.decision is Decision.ESCALATE
    assert first.disposition is DecisionDisposition.ESCALATE
    assert first.effective_parameters is None
    assert first.unresolved_conflict_ids == ("allow", "block")
    assert first.decision_hash == second.decision_hash


def test_approval_only_satisfies_require_approval_and_waiver_precedes_approval() -> None:
    required = _record("rule_required", scope=_access_scope(disposition="REQUIRE_APPROVAL"))
    approval = _record(
        "approval_a",
        source_type="APPROVAL",
        status="ACTIVE_AUTHORITY",
        scope=_access_evidence_scope(),
        exception_to=["rule_required"],
    )
    blocked = _record("rule_blocked", scope=_access_scope(disposition="BLOCK"))
    approval_for_block = _record(
        "approval_block",
        source_type="APPROVAL",
        status="ACTIVE_AUTHORITY",
        scope=_access_evidence_scope(),
        exception_to=["rule_blocked"],
    )
    waiver = _record(
        "waiver_a",
        source_type="WAIVER",
        status="SCOPED_EXCEPTION",
        scope=_access_evidence_scope(),
        exception_to=["rule_blocked"],
    )

    approved = resolve_authority([required, approval], _access_query())
    waived = resolve_authority([blocked, approval_for_block, waiver], _access_query())
    approval_cannot_unblock = resolve_authority([blocked, approval_for_block], _access_query())

    assert approved.decision is Decision.PROCEED
    assert approved.reason_code is DecisionReasonCode.VALID_APPROVAL
    assert approved.exception_or_approval_ids == ("approval_a",)
    assert waived.decision is Decision.PROCEED_UNDER_EXCEPTION
    assert waived.reason_code is DecisionReasonCode.VALID_WAIVER
    assert waived.exception_or_approval_ids == ("waiver_a",)
    assert approval_cannot_unblock.decision is Decision.BLOCK
    assert approval_cannot_unblock.exception_or_approval_ids == ()
    assert [item.reason_code.value for item in approval_cannot_unblock.discarded_authorities] == [
        "EVIDENCE_NOT_NEEDED"
    ]


def test_finance_uses_absolute_amount_and_strict_threshold_boundary() -> None:
    rule = _record(
        "finance_rule",
        action_type="post_adjustment",
        scope=_finance_scope(threshold_minor=500),
    )

    equality = resolve_authority([rule], _finance_query(-500))
    over = resolve_authority([rule], _finance_query(-501))

    assert equality.decision is Decision.PROCEED
    assert equality.disposition is DecisionDisposition.PROCEED
    assert over.decision is Decision.REQUIRE_APPROVAL
    assert over.disposition is DecisionDisposition.REQUIRE_APPROVAL


def test_unmatched_rules_and_evidence_have_first_stable_discard_codes() -> None:
    future = _record("future", effective_at="2026-03-01T00:00:00Z")
    wrong_action = _record(
        "wrong_action",
        action_type="revoke_access",
        scope={**_access_scope(), "allowed_action": "revoke_access"},
    )
    old_rule = _record(
        "old_rule",
        status="SUPERSEDED",
        scope=_access_scope(role_id=None, disposition="REQUIRE_APPROVAL"),
    )
    rule = _record(
        "rule_active",
        issuer_id="issuer_senior",
        authority_rank=20,
        effective_at="2026-01-02T00:00:00Z",
        scope=_access_scope(disposition="REQUIRE_APPROVAL"),
        supersedes=["old_rule"],
    )
    stale_evidence = _record(
        "approval_old_target",
        source_type="APPROVAL",
        status="ACTIVE_AUTHORITY",
        scope=_access_evidence_scope(),
        exception_to=["old_rule"],
    )

    decision = resolve_authority(
        [stale_evidence, future, wrong_action, old_rule, rule], _access_query()
    )

    assert decision.decision is Decision.REQUIRE_APPROVAL
    discarded = [
        (item.authority_id, item.reason_code.value) for item in decision.discarded_authorities
    ]
    assert discarded == [
        ("approval_old_target", "EVIDENCE_TARGET_NOT_EFFECTIVE"),
        ("future", "NOT_YET_EFFECTIVE"),
        ("old_rule", "SUPERSEDED"),
        ("wrong_action", "ACTION_MISMATCH"),
    ]


def test_hostile_model_copy_containers_are_rejected_before_hooks_run() -> None:
    class HostileList(list[object]):
        calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).calls += 1
            raise AssertionError("resolver must not invoke hostile collection hooks")

    record = _record("rule").model_copy(update={"supersedes": HostileList()})
    query = _access_query().model_copy(update={"reference_authority_ids": HostileList()})
    registry = _access_query().issuer_registry.model_copy(update={"authorizations": HostileList()})
    registry_query = _access_query().model_copy(update={"issuer_registry": registry})
    finance = _record(
        "finance",
        action_type="post_adjustment",
        scope=_finance_scope(),
    )
    hostile_scope = finance.scope.model_copy(update={"restricted_categories": HostileList()})
    scoped_record = finance.model_copy(update={"scope": hostile_scope})

    for records, candidate in (
        ([record], _access_query()),
        ([_record("rule_query")], query),
        ([_record("rule_registry")], registry_query),
        ([scoped_record], _finance_query(1)),
    ):
        with pytest.raises(AuthorityAdmissionError):
            resolve_authority(records, candidate)
    assert HostileList.calls == 0


def test_hostile_model_copy_validation_markers_are_rejected_before_truthiness() -> None:
    class HostileValidatedMarker:
        calls = 0

        def __bool__(self) -> bool:
            type(self).calls += 1
            raise RuntimeError("resolver must not invoke hostile marker truthiness")

    record = _record("record_marker").model_copy(update={"_validated": HostileValidatedMarker()})
    query = _access_query().model_copy(update={"_validated": HostileValidatedMarker()})
    registry = _access_query().issuer_registry.model_copy(
        update={"_validated": HostileValidatedMarker()}
    )
    registry_query = _access_query().model_copy(update={"issuer_registry": registry})
    finance = _record(
        "scope_marker",
        action_type="post_adjustment",
        scope=_finance_scope(),
    )
    hostile_scope = finance.scope.model_copy(update={"_validated": HostileValidatedMarker()})
    scoped_record = finance.model_copy(update={"scope": hostile_scope})

    for records, candidate in (
        ([record], _access_query()),
        ([_record("rule_query_marker")], query),
        ([_record("rule_registry_marker")], registry_query),
        ([scoped_record], _finance_query(1)),
    ):
        with pytest.raises(AuthorityAdmissionError):
            resolve_authority(records, candidate)
    assert HostileValidatedMarker.calls == 0


def test_model_copy_extra_keys_are_rejected_before_hooks_run() -> None:
    class HostileExtra:
        calls = 0

        def __bool__(self) -> bool:
            type(self).calls += 1
            raise AssertionError("resolver must not invoke hostile extra truthiness")

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).calls += 1
            raise AssertionError("resolver must not traverse hostile extra values")

    record = _record("record_extra").model_copy(update={"evil": HostileExtra()})
    query = _access_query().model_copy(update={"evil": HostileExtra()})
    registry = _access_query().issuer_registry.model_copy(update={"evil": HostileExtra()})
    registry_query = _access_query().model_copy(update={"issuer_registry": registry})
    access = _record("access_scope_extra")
    access_scope = access.scope.model_copy(update={"evil": HostileExtra()})
    scoped_access_record = access.model_copy(update={"scope": access_scope})
    finance = _record(
        "finance_scope_extra",
        action_type="post_adjustment",
        scope=_finance_scope(),
    )
    finance_scope = finance.scope.model_copy(update={"evil": HostileExtra()})
    scoped_finance_record = finance.model_copy(update={"scope": finance_scope})

    for records, candidate in (
        ([record], _access_query()),
        ([_record("rule_query_extra")], query),
        ([_record("rule_registry_extra")], registry_query),
        ([scoped_access_record], _access_query()),
        ([scoped_finance_record], _finance_query(1)),
    ):
        with pytest.raises(AuthorityAdmissionError):
            resolve_authority(records, candidate)
    assert HostileExtra.calls == 0


def test_model_copy_missing_fields_are_rejected_before_revalidation() -> None:
    record = _record("record_missing").model_copy()
    object.__getattribute__(record, "__dict__").pop("title")
    query = _access_query().model_copy()
    object.__getattribute__(query, "__dict__").pop("role_id")
    registry = _access_query().issuer_registry.model_copy()
    object.__getattribute__(registry, "__dict__").pop("authorizations")
    registry_query = _access_query().model_copy(update={"issuer_registry": registry})
    access = _record("access_scope_missing")
    access_scope = access.scope.model_copy()
    object.__getattribute__(access_scope, "__dict__").pop("role_id")
    scoped_access_record = access.model_copy(update={"scope": access_scope})
    finance = _record(
        "finance_scope_missing",
        action_type="post_adjustment",
        scope=_finance_scope(),
    )
    finance_scope = finance.scope.model_copy()
    object.__getattribute__(finance_scope, "__dict__").pop("restricted_categories")
    scoped_finance_record = finance.model_copy(update={"scope": finance_scope})

    for records, candidate in (
        ([record], _access_query()),
        ([_record("rule_query_missing")], query),
        ([_record("rule_registry_missing")], registry_query),
        ([scoped_access_record], _access_query()),
        ([scoped_finance_record], _finance_query(1)),
    ):
        with pytest.raises(AuthorityAdmissionError):
            resolve_authority(records, candidate)


def test_model_copy_cycles_are_rejected_without_recursion_errors() -> None:
    record_cycle: list[object] = []
    record_cycle.append(record_cycle)
    record = _record("record_cycle").model_copy(update={"supersedes": record_cycle})

    query_cycle: list[object] = []
    query_cycle.append(query_cycle)
    query = _access_query().model_copy(update={"reference_authority_ids": query_cycle})

    registry_cycle: list[object] = []
    registry = _access_query().issuer_registry.model_copy(update={"authorizations": registry_cycle})
    registry_cycle.append(registry)
    registry_query = _access_query().model_copy(update={"issuer_registry": registry})

    finance = _record(
        "scope_cycle",
        action_type="post_adjustment",
        scope=_finance_scope(),
    )
    scope_cycle: list[object] = []
    finance_scope = finance.scope.model_copy(update={"restricted_categories": scope_cycle})
    scope_cycle.append(finance_scope)
    scoped_record = finance.model_copy(update={"scope": finance_scope})

    for records, candidate in (
        ([record], _access_query()),
        ([_record("query_cycle_rule")], query),
        ([_record("registry_cycle_rule")], registry_query),
        ([scoped_record], _finance_query(1)),
    ):
        with pytest.raises(AuthorityAdmissionError):
            resolve_authority(records, candidate)


def test_exact_structure_permits_noncyclic_shared_collection_aliases() -> None:
    shared: list[object] = ["shared"]

    assert _has_exact_nested_structure([shared, shared])


def test_deep_acyclic_model_copy_list_is_rejected_without_recursion_errors() -> None:
    nested: object = "leaf"
    for _ in range(2_000):
        nested = [nested]
    record = _record("deep_nesting").model_copy(update={"supersedes": nested})

    with pytest.raises(AuthorityAdmissionError):
        resolve_authority([record], _access_query())


@pytest.mark.parametrize(
    "records, query",
    [
        ([_record("wrong_rank", authority_rank=9)], _access_query()),
        (
            [
                _record(
                    "behavior",
                    source_type="BEHAVIOR_TRACE",
                    status="DESCRIPTIVE_ONLY",
                    scope={
                        "domain": "access_provisioning",
                        "scope_kind": "descriptive",
                        "subject_id": None,
                        "resource_id": None,
                        "organization_id": None,
                        "geography_id": None,
                        "role_id": None,
                    },
                )
            ],
            _access_query(registry=_registry(primary_sources=["POLICY"])),
        ),
        (
            [_record("finance", action_type="post_adjustment", scope=_finance_scope())],
            _access_query(registry=_registry(primary_domains=["access_provisioning"])),
        ),
        (
            [
                _record("target", status="SUPERSEDED"),
                _record(
                    "source",
                    effective_at="2026-01-02T00:00:00Z",
                    supersedes=["target"],
                ),
            ],
            _access_query(registry=_registry(primary_can_supersede=False)),
        ),
        (
            [
                _record("target"),
                _record(
                    "approval",
                    source_type="APPROVAL",
                    status="ACTIVE_AUTHORITY",
                    scope=_access_evidence_scope(),
                    exception_to=["target"],
                ),
            ],
            _access_query(registry=_registry(primary_can_evidence=False)),
        ),
        (
            [
                _record(
                    "dangling_supersession",
                    effective_at="2026-01-02T00:00:00Z",
                    supersedes=["missing"],
                )
            ],
            _access_query(),
        ),
        (
            [
                _record(
                    "dangling_evidence",
                    source_type="APPROVAL",
                    status="ACTIVE_AUTHORITY",
                    scope=_access_evidence_scope(),
                    exception_to=["missing"],
                )
            ],
            _access_query(),
        ),
    ],
)
def test_whole_corpus_rejects_irrelevant_invalid_records(
    records: list[AuthorityRecord], query: object
) -> None:
    with pytest.raises(AuthorityAdmissionError):
        resolve_authority(records, query)


def test_supersession_graph_rejects_two_otherwise_valid_incoming_edges() -> None:
    target = _record("target", status="SUPERSEDED", scope=_access_scope(role_id=None))
    source_a = _record(
        "source_a",
        issuer_id="issuer_senior",
        authority_rank=20,
        effective_at="2026-01-02T00:00:00Z",
        supersedes=["target"],
    )
    source_b = _record(
        "source_b",
        issuer_id="issuer_senior",
        authority_rank=20,
        effective_at="2026-01-03T00:00:00Z",
        supersedes=["target"],
    )

    with pytest.raises(AuthorityAdmissionError):
        resolve_authority([target, source_a, source_b], _access_query())


def test_supersession_rejects_open_ended_source_outside_finite_target_time() -> None:
    target = _record(
        "target",
        status="SUPERSEDED",
        expires_at="2026-12-31T00:00:00Z",
    )
    source = _record(
        "source",
        issuer_id="issuer_senior",
        authority_rank=20,
        effective_at="2026-01-02T00:00:00Z",
        supersedes=["target"],
    )

    with pytest.raises(AuthorityAdmissionError):
        resolve_authority([target, source], _access_query())


def test_evidence_cannot_broaden_concrete_organization_target_scope() -> None:
    target = _record(
        "target",
        scope=_access_scope(organization_id="organization_a", disposition="REQUIRE_APPROVAL"),
    )
    evidence = _record(
        "approval",
        source_type="APPROVAL",
        status="ACTIVE_AUTHORITY",
        scope=_access_evidence_scope(organization_id=None),
        exception_to=["target"],
    )

    with pytest.raises(AuthorityAdmissionError):
        resolve_authority([target, evidence], _access_query(organization_id="organization_a"))


def test_finance_interval_containment_respects_endpoint_orientation() -> None:
    outer_open = _record(
        "outer_open",
        action_type="post_adjustment",
        status="SUPERSEDED",
        scope=_finance_scope(
            minimum_amount_minor=0,
            minimum_inclusive=False,
            maximum_amount_minor=100,
            maximum_inclusive=True,
        ),
    )
    inner_closed = _record(
        "inner_closed",
        issuer_id="issuer_senior",
        authority_rank=20,
        action_type="post_adjustment",
        effective_at="2026-01-02T00:00:00Z",
        scope=_finance_scope(
            minimum_amount_minor=0,
            minimum_inclusive=True,
            maximum_amount_minor=100,
            maximum_inclusive=True,
        ),
        supersedes=["outer_open"],
    )
    outer_closed = _record(
        "outer_closed",
        action_type="post_adjustment",
        status="SUPERSEDED",
        scope=_finance_scope(
            minimum_amount_minor=0,
            minimum_inclusive=True,
            maximum_amount_minor=100,
            maximum_inclusive=True,
        ),
    )
    inner_open = _record(
        "inner_open",
        issuer_id="issuer_senior",
        authority_rank=20,
        action_type="post_adjustment",
        effective_at="2026-01-02T00:00:00Z",
        scope=_finance_scope(
            minimum_amount_minor=0,
            minimum_inclusive=False,
            maximum_amount_minor=100,
            maximum_inclusive=True,
        ),
        supersedes=["outer_closed"],
    )

    with pytest.raises(AuthorityAdmissionError):
        resolve_authority([outer_open, inner_closed], _finance_query(50))
    accepted = resolve_authority([outer_closed, inner_open], _finance_query(50))
    assert accepted.decision is Decision.PROCEED


@pytest.mark.parametrize("status", [NormativeStatus.CONFLICTING, NormativeStatus.UNRESOLVED])
def test_explicit_top_status_conflict_escalates(status: NormativeStatus) -> None:
    conflict = _record("conflict", status=status)

    decision = resolve_authority([conflict], _access_query())

    assert decision.decision is Decision.ESCALATE
    assert decision.unresolved_conflict_ids == ("conflict",)


def test_waiver_beats_approval_for_require_approval_and_discards_approval() -> None:
    required = _record("required", scope=_access_scope(disposition="REQUIRE_APPROVAL"))
    approval = _record(
        "approval",
        source_type="APPROVAL",
        status="ACTIVE_AUTHORITY",
        scope=_access_evidence_scope(),
        exception_to=["required"],
    )
    waiver = _record(
        "waiver",
        source_type="WAIVER",
        status="SCOPED_EXCEPTION",
        scope=_access_evidence_scope(),
        exception_to=["required"],
    )

    decision = resolve_authority([required, approval, waiver], _access_query())

    assert decision.decision is Decision.PROCEED_UNDER_EXCEPTION
    assert decision.exception_or_approval_ids == ("waiver",)
    discarded = [
        (item.authority_id, item.reason_code.value) for item in decision.discarded_authorities
    ]
    assert discarded == [("approval", "EVIDENCE_NOT_NEEDED")]


def test_coalesced_follow_prefers_path_with_referenced_predecessor() -> None:
    base_a = _record("base_a", status="SUPERSEDED", scope=_access_scope(role_id=None))
    base_b = _record("base_b", status="SUPERSEDED", scope=_access_scope(role_id=None))
    effective_a = _record(
        "effective_a",
        issuer_id="issuer_senior",
        authority_rank=20,
        effective_at="2026-01-02T00:00:00Z",
        supersedes=["base_a"],
    )
    effective_b = _record(
        "effective_b",
        issuer_id="issuer_senior",
        authority_rank=20,
        effective_at="2026-01-02T00:00:00Z",
        supersedes=["base_b"],
    )

    decision = resolve_authority(
        [base_a, effective_a, base_b, effective_b], _access_query(references=["base_b"])
    )

    assert decision.decision is Decision.FOLLOW_SUPERSEDING_AUTHORITY
    assert decision.supersession_path == ("base_b", "effective_b")


def test_authority_set_hash_binds_discarded_descriptive_records() -> None:
    rule = _record("rule")
    trace = _record(
        "trace",
        source_type="BEHAVIOR_TRACE",
        status="DESCRIPTIVE_ONLY",
        scope={
            "domain": "access_provisioning",
            "scope_kind": "descriptive",
            "subject_id": None,
            "resource_id": None,
            "organization_id": None,
            "geography_id": None,
            "role_id": None,
        },
    )

    without_trace = resolve_authority([rule], _access_query())
    with_trace = resolve_authority([rule, trace], _access_query())

    assert without_trace.authority_set_hash != with_trace.authority_set_hash
