from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from shadowskillbench.authority import (
    AccessScope as ExportedAccessScope,
)
from shadowskillbench.authority import (
    AuthorityQuery as ExportedAuthorityQuery,
)
from shadowskillbench.authority import (
    AuthorityScope as ExportedAuthorityScope,
)
from shadowskillbench.authority import (
    EffectiveParameters as ExportedEffectiveParameters,
)
from shadowskillbench.authority import (
    FinanceScope as ExportedFinanceScope,
)
from shadowskillbench.authority.models import (
    AccessAuthorityQuery,
    AccessEffectiveParameters,
    AccessRuleScope,
    AccessScope,
    AuthorityDecision,
    AuthorityQuery,
    AuthorityRecord,
    AuthorityScope,
    AuthoritySourceType,
    Decision,
    DecisionDisposition,
    DecisionReasonCode,
    DiscardedAuthority,
    DiscardReasonCode,
    EffectiveParameters,
    FinanceAuthorityQuery,
    FinanceEffectiveParameters,
    FinanceEvidenceScope,
    FinanceRuleScope,
    FinanceScope,
    IssuerAuthorization,
    NormativeStatus,
    authority_decision_hash,
    authority_query_hash,
    authority_record_hash,
    authority_set_hash,
    create_access_authority_query,
    create_authority_decision,
    create_authority_record,
    create_finance_authority_query,
    create_issuer_registry,
)
from shadowskillbench.core.hashing import sha256_ref


def _authorization() -> IssuerAuthorization:
    return IssuerAuthorization(
        issuer_id="issuer_primary",
        issuer_role="policy_owner",
        authority_rank=10,
        allowed_domains=["access_provisioning", "financial_adjustments"],
        allowed_source_types=["POLICY", "SIGNED_DIRECTIVE"],
        can_supersede=True,
        can_issue_scoped_evidence=True,
    )


def _registry() -> Any:
    return create_issuer_registry(authorizations=[_authorization()])


def _access_rule_scope() -> dict[str, object]:
    return {
        "domain": "access_provisioning",
        "scope_kind": "rule",
        "subject_id": "employee_a",
        "resource_id": "application_a",
        "organization_id": None,
        "geography_id": None,
        "role_id": "role_a",
        "rule_disposition": "PROCEED",
        "allowed_action": "grant_access",
        "requires_security_approval": False,
        "role_derived_without_approval": True,
    }


def _finance_rule_scope() -> dict[str, object]:
    return {
        "domain": "financial_adjustments",
        "scope_kind": "rule",
        "subject_id": "adjustment_a",
        "resource_id": "portco_a",
        "organization_id": None,
        "geography_id": None,
        "category_id": "category_a",
        "period_id": "period_a",
        "minimum_amount_minor": None,
        "minimum_inclusive": False,
        "maximum_amount_minor": None,
        "maximum_inclusive": False,
        "currency": "USD",
        "unit": "minor_units",
        "currency_exponent": 2,
        "rule_disposition": "PROCEED",
        "threshold_minor": 5_000_000,
        "restricted_categories": [],
    }


def _finance_evidence_scope(maximum_amount_minor: object = 5_000_000) -> dict[str, object]:
    return {
        "domain": "financial_adjustments",
        "scope_kind": "evidence",
        "subject_id": "adjustment_a",
        "resource_id": "portco_a",
        "organization_id": None,
        "geography_id": None,
        "category_id": "category_a",
        "period_id": "period_a",
        "minimum_amount_minor": 0,
        "minimum_inclusive": True,
        "maximum_amount_minor": maximum_amount_minor,
        "maximum_inclusive": True,
        "currency": "USD",
        "unit": "minor_units",
        "currency_exponent": 2,
    }


def _record_input(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "authority_id": "authority_a",
        "schema_version": "1.0",
        "source_type": "POLICY",
        "title": "Access policy",
        "issuer_id": "issuer_primary",
        "issuer_role": "policy_owner",
        "authority_rank": 10,
        "action_type": "grant_access",
        "scope": _access_rule_scope(),
        "effective_at": "2026-01-01T00:00:00Z",
        "expires_at": None,
        "supersedes": [],
        "exception_to": [],
        "provenance_locator": "policy/access-v1",
        "normative_status": "ACTIVE_AUTHORITY",
    }
    base.update(overrides)
    return base


def _record(**overrides: object) -> AuthorityRecord:
    return create_authority_record(**_record_input(**overrides))


def _access_query() -> Any:
    return create_access_authority_query(
        subject_id="employee_a",
        resource_id="application_a",
        action_type="grant_access",
        at_time="2026-02-01T00:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[],
        issuer_registry=_registry(),
        role_id="role_a",
        role_derived_access=True,
    )


def _finance_query() -> Any:
    return create_finance_authority_query(
        subject_id="adjustment_a",
        resource_id="portco_a",
        action_type="publish_adjustment",
        at_time="2026-02-01T00:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[],
        issuer_registry=_registry(),
        category_id="category_a",
        period_id="period_a",
        amount_minor=-5_000_000,
        currency="USD",
        unit="minor_units",
        currency_exponent=2,
    )


def _decision(**overrides: object) -> AuthorityDecision:
    record = _record()
    base: dict[str, object] = {
        "authority_query": _access_query(),
        "authority_set_hash": sha256_ref([record.model_dump(mode="json")]),
        "decision": Decision.PROCEED,
        "disposition": DecisionDisposition.PROCEED,
        "effective_parameters": AccessEffectiveParameters(
            domain="access_provisioning",
            allowed_action="grant_access",
            requires_security_approval=False,
            role_derived_without_approval=True,
        ),
        "applicable_authority_ids": ["authority_a"],
        "discarded_authorities": [],
        "supersession_path": [],
        "exception_or_approval_ids": [],
        "unresolved_conflict_ids": [],
        "effective_rule_ids": ["authority_a"],
        "reason_code": DecisionReasonCode.EFFECTIVE_RULE,
    }
    base.update(overrides)
    return create_authority_decision(**base)


def test_access_and_finance_models_have_explicit_detached_happy_paths() -> None:
    access = _access_query()
    finance = _finance_query()
    access_scope = AccessRuleScope.model_validate(_access_rule_scope())
    finance_scope = FinanceRuleScope.model_validate(_finance_rule_scope())
    record = _record()

    assert access.domain == "access_provisioning"
    assert finance.amount_minor == -5_000_000
    assert access_scope.allowed_action == "grant_access"
    assert finance_scope.threshold_minor == 5_000_000
    assert record.content_hash == sha256_ref(
        {
            key: value
            for key, value in record.model_dump(mode="json").items()
            if key != "content_hash"
        }
    )
    assert _decision().decision is Decision.PROCEED

    mutable_scope = _access_rule_scope()
    record = _record(scope=mutable_scope)
    mutable_scope["role_id"] = "role_other"
    assert cast(AccessRuleScope, record.scope).role_id == "role_a"


def test_authority_package_exports_all_declared_union_aliases() -> None:
    assert ExportedAuthorityQuery is AuthorityQuery
    assert ExportedAuthorityScope is AuthorityScope
    assert ExportedAccessScope is AccessScope
    assert ExportedFinanceScope is FinanceScope
    assert ExportedEffectiveParameters is EffectiveParameters


@pytest.mark.parametrize(
    "field",
    sorted(AuthorityRecord.model_fields),
)
def test_runtime_record_rejects_omitted_wire_properties(field: str) -> None:
    raw = _record().model_dump(mode="json")
    raw.pop(field)
    with pytest.raises(ValidationError):
        AuthorityRecord(**raw)  # type: ignore[arg-type]


def test_records_queries_and_registry_forbid_extras_and_are_frozen() -> None:
    with pytest.raises(ValidationError):
        AuthorityRecord(**_record_input(content_hash="sha256:" + "0" * 64, extra=True))
    with pytest.raises(ValidationError):
        AccessAuthorityQuery(**{**_access_query().model_dump(mode="json"), "extra": True})
    with pytest.raises(ValidationError):
        _record().title = "changed"  # type: ignore[misc]


def test_validation_entry_points_cannot_weaken_strict_extra_or_exact_ingress() -> None:
    authorization = _authorization().model_dump(mode="json")
    with_extra = {**authorization, "unexpected": True}

    parsed_authorization = IssuerAuthorization.model_validate_json(json.dumps(authorization))
    assert parsed_authorization.issuer_id == "issuer_primary"

    for extra in ("ignore", "allow"):
        with pytest.raises((ValidationError, ValueError)):
            IssuerAuthorization.model_validate(with_extra, extra=extra)
    with pytest.raises((ValidationError, ValueError)):
        IssuerAuthorization.model_validate(authorization, strict=False)
    with pytest.raises((ValidationError, ValueError)):
        IssuerAuthorization.model_validate(authorization, from_attributes=True)
    with pytest.raises((ValidationError, ValueError)):
        IssuerAuthorization.model_validate_json(json.dumps(with_extra), extra="ignore")
    with pytest.raises((ValidationError, ValueError)):
        IssuerAuthorization.model_validate_json(json.dumps(authorization), strict=False)
    with pytest.raises((ValidationError, ValueError)):
        IssuerAuthorization.model_validate_strings(authorization)


@pytest.mark.parametrize(
    "bad_value",
    [True, 1.0, "10"],
)
def test_integer_fields_reject_bool_and_coercion(bad_value: object) -> None:
    with pytest.raises(ValidationError):
        _record(authority_rank=bad_value)
    with pytest.raises(ValidationError):
        create_finance_authority_query(
            subject_id="adjustment_a",
            resource_id="portco_a",
            action_type="publish_adjustment",
            at_time="2026-02-01T00:00:00Z",
            organization_id=None,
            geography_id=None,
            reference_authority_ids=[],
            issuer_registry=_registry(),
            category_id="category_a",
            period_id="period_a",
            amount_minor=bad_value,
            currency="USD",
            unit="minor_units",
            currency_exponent=2,
        )


@pytest.mark.parametrize(
    "timestamp",
    ["2026-01-01T00:00:00", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00.001Z"],
)
def test_times_must_be_canonical_utc_seconds(timestamp: str) -> None:
    with pytest.raises(ValidationError):
        _record(effective_at=timestamp)
    with pytest.raises(ValidationError):
        AccessAuthorityQuery(**{**_access_query().model_dump(mode="json"), "at_time": timestamp})


def test_ids_hashes_collections_and_self_excluding_hashes_are_exact() -> None:
    with pytest.raises(ValidationError):
        _record(authority_id="../authority")
    with pytest.raises(ValidationError):
        AuthorityRecord(
            **{**_record().model_dump(mode="json"), "content_hash": "sha256:" + "A" * 64}
        )
    with pytest.raises(ValidationError):
        _record(supersedes=["authority_b", "authority_a"])
    with pytest.raises(ValidationError):
        _record(supersedes=["authority_a", "authority_a"])

    record = _record()
    with pytest.raises(ValidationError, match="content_hash"):
        AuthorityRecord(**{**record.model_dump(mode="json"), "title": "tampered"})
    registry = _registry()
    with pytest.raises(ValidationError, match="registry_hash"):
        type(registry)(
            **{**registry.model_dump(mode="json"), "registry_hash": "sha256:" + "0" * 64}
        )
    query = _access_query()
    with pytest.raises(ValidationError, match="authority_query_hash"):
        type(query)(
            **{**query.model_dump(mode="json"), "authority_query_hash": "sha256:" + "0" * 64}
        )
    decision = _decision()
    with pytest.raises(ValidationError, match="decision_hash"):
        AuthorityDecision(
            **{**decision.model_dump(mode="json"), "decision_hash": "sha256:" + "0" * 64}
        )


def test_hash_helpers_reject_constructed_or_incomplete_models_without_attribute_errors() -> None:
    for helper, constructed in [
        (authority_record_hash, AuthorityRecord.model_construct()),
        (authority_query_hash, AccessAuthorityQuery.model_construct()),
        (authority_decision_hash, AuthorityDecision.model_construct()),
    ]:
        with pytest.raises((ValidationError, ValueError)):
            helper(constructed)

    with pytest.raises(ValueError):
        authority_record_hash(_access_query())
    with pytest.raises(ValueError):
        authority_query_hash(_record())
    with pytest.raises(ValueError):
        authority_decision_hash(_record())

    class RecordSubclass(AuthorityRecord):
        pass

    with pytest.raises(ValueError):
        authority_record_hash(RecordSubclass.model_construct())


def test_hash_validation_has_no_mutable_skip_switches() -> None:
    import shadowskillbench.authority.models as authority_models

    source = inspect.getsource(authority_models)
    assert "ContextVar" not in source
    assert "_SKIP_" not in source

    stale = {**_record().model_dump(mode="json"), "title": "tampered"}
    with pytest.raises(ValidationError, match="content_hash"):
        AuthorityRecord.model_validate(stale)


def test_model_construct_and_hostile_collections_are_revalidated_without_hooks() -> None:
    constructed = AccessRuleScope.model_construct(**_access_rule_scope())
    with pytest.raises(ValidationError):
        _record(scope=constructed)

    class HostileList(list[object]):
        calls = 0

        def __iter__(self) -> Any:
            type(self).calls += 1
            raise AssertionError("hostile collection hook must not run")

    with pytest.raises((ValidationError, ValueError)):
        _record(supersedes=HostileList())
    assert HostileList.calls == 0


def test_source_status_and_scope_capabilities_are_structurally_closed() -> None:
    with pytest.raises(ValidationError):
        _record(source_type="BEHAVIOR_TRACE")
    with pytest.raises(ValidationError):
        _record(source_type="APPROVAL")
    with pytest.raises(ValidationError):
        _record(
            source_type="WAIVER",
            normative_status="SCOPED_EXCEPTION",
            supersedes=["authority_old"],
            exception_to=["authority_old"],
            scope={
                "domain": "access_provisioning",
                "scope_kind": "evidence",
                "subject_id": "employee_a",
                "resource_id": "application_a",
                "organization_id": None,
                "geography_id": None,
                "role_id": "role_a",
            },
        )
    descriptive = _record(
        source_type="PROCEDURE_GUIDE",
        normative_status="DESCRIPTIVE_ONLY",
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
    assert descriptive.source_type is AuthoritySourceType.PROCEDURE_GUIDE
    assert descriptive.normative_status is NormativeStatus.DESCRIPTIVE_ONLY
    with pytest.raises(ValidationError, match="expires_at"):
        _record(normative_status="EXPIRED")


def test_scope_discriminators_intervals_and_money_invariants_are_enforced() -> None:
    with pytest.raises(ValidationError):
        AccessRuleScope.model_validate({**_access_rule_scope(), "scope_kind": "evidence"})
    with pytest.raises(ValidationError):
        FinanceRuleScope.model_validate(
            {**_finance_rule_scope(), "minimum_amount_minor": None, "minimum_inclusive": True}
        )

    one_sided = FinanceRuleScope.model_validate(
        {**_finance_rule_scope(), "minimum_amount_minor": 1, "minimum_inclusive": True}
    )
    assert one_sided.maximum_amount_minor is None
    assert one_sided.minimum_inclusive is True
    with pytest.raises(ValidationError):
        FinanceRuleScope.model_validate(
            {
                **_finance_rule_scope(),
                "minimum_amount_minor": 2,
                "maximum_amount_minor": 1,
                "minimum_inclusive": True,
                "maximum_inclusive": True,
            }
        )
    with pytest.raises(ValidationError):
        FinanceRuleScope.model_validate(
            {
                **_finance_rule_scope(),
                "minimum_amount_minor": 1,
                "maximum_amount_minor": 1,
                "minimum_inclusive": True,
                "maximum_inclusive": False,
            }
        )
    with pytest.raises(ValidationError):
        FinanceEvidenceScope.model_validate(_finance_evidence_scope(maximum_amount_minor=-1))


def test_record_interval_and_rule_effect_invariants_are_local() -> None:
    with pytest.raises(ValidationError):
        _record(expires_at="2026-01-01T00:00:00Z")
    with pytest.raises(ValidationError):
        _record(
            action_type="revoke_access",
            scope={**_access_rule_scope(), "allowed_action": "grant_access"},
        )
    with pytest.raises(ValidationError):
        _record(
            source_type="APPROVAL",
            normative_status="ACTIVE_AUTHORITY",
            exception_to=[],
            scope={
                "domain": "access_provisioning",
                "scope_kind": "evidence",
                "subject_id": None,
                "resource_id": "application_a",
                "organization_id": None,
                "geography_id": None,
                "role_id": "role_a",
            },
        )


def test_query_discriminators_explicit_dimensions_and_finance_money_are_enforced() -> None:
    with pytest.raises(ValidationError):
        AccessAuthorityQuery(
            **{
                key: value
                for key, value in _access_query().model_dump(mode="json").items()
                if key != "organization_id"
            }
        )
    with pytest.raises(ValidationError):
        FinanceAuthorityQuery(**{**_finance_query().model_dump(mode="json"), "currency": "usd"})
    with pytest.raises(ValidationError):
        FinanceAuthorityQuery(
            **{**_finance_query().model_dump(mode="json"), "currency_exponent": 3}
        )


@pytest.mark.parametrize("invalid_exponent", [True, 2.0, "2"])
def test_currency_exponent_requires_an_exact_integer_two_everywhere(
    invalid_exponent: object,
) -> None:
    with pytest.raises(ValidationError, match="currency_exponent"):
        FinanceRuleScope.model_validate(
            {**_finance_rule_scope(), "currency_exponent": invalid_exponent}
        )
    with pytest.raises(ValidationError, match="currency_exponent"):
        create_finance_authority_query(
            subject_id="adjustment_a",
            resource_id="portco_a",
            action_type="publish_adjustment",
            at_time="2026-02-01T00:00:00Z",
            organization_id=None,
            geography_id=None,
            reference_authority_ids=[],
            issuer_registry=_registry(),
            category_id="category_a",
            period_id="period_a",
            amount_minor=-5_000_000,
            currency="USD",
            unit="minor_units",
            currency_exponent=invalid_exponent,
        )
    with pytest.raises(ValidationError, match="currency_exponent"):
        FinanceEffectiveParameters(
            domain="financial_adjustments",
            threshold_minor=1,
            currency="USD",
            currency_exponent=invalid_exponent,
            restricted_categories=[],
        )
    with pytest.raises(ValidationError, match="currency_exponent"):
        _decision(
            effective_parameters={
                "domain": "financial_adjustments",
                "threshold_minor": 1,
                "currency": "USD",
                "currency_exponent": invalid_exponent,
                "restricted_categories": [],
            }
        )


def test_decision_cross_fields_and_factory_only_hash_creation_are_enforced() -> None:
    with pytest.raises(ValidationError):
        _decision(decision=Decision.BLOCK, disposition=DecisionDisposition.PROCEED)
    with pytest.raises(ValidationError):
        _decision(
            decision=Decision.ESCALATE,
            disposition=DecisionDisposition.ESCALATE,
            effective_parameters=AccessEffectiveParameters(
                domain="access_provisioning",
                allowed_action="grant_access",
                requires_security_approval=False,
                role_derived_without_approval=True,
            ),
        )
    with pytest.raises(ValidationError):
        _decision(
            reason_code=DecisionReasonCode.VALID_APPROVAL,
            exception_or_approval_ids=[],
        )
    with pytest.raises(ValidationError):
        _decision(
            reason_code=DecisionReasonCode.VALID_WAIVER,
            decision=Decision.PROCEED,
            disposition=DecisionDisposition.PROCEED,
        )
    with pytest.raises(ValidationError):
        _decision(
            effective_parameters={
                "domain": "financial_adjustments",
                "threshold_minor": 1,
                "currency": "USD",
                "currency_exponent": 2,
                "restricted_categories": [],
            }
        )

    with pytest.raises(ValidationError):
        AuthorityDecision(
            **{
                key: value
                for key, value in _decision().model_dump(mode="json").items()
                if key != "decision_hash"
            }
        )


def test_hash_factories_sort_records_and_context_cannot_bypass_validation() -> None:
    first = _record(authority_id="authority_a")
    second = _record(authority_id="authority_b")
    assert authority_set_hash([second, first]) == authority_set_hash([first, second])

    tampered = {**first.model_dump(mode="json"), "title": "tampered"}
    with pytest.raises(ValidationError, match="content_hash"):
        AuthorityRecord.model_validate(tampered, context={"skip_content_hash": True})

    with pytest.raises(ValidationError):
        create_issuer_registry(
            authorizations=[
                {
                    **_authorization().model_dump(mode="json"),
                    "allowed_domains": ["financial_adjustments", "access_provisioning"],
                }
            ]
        )


def test_discarded_authority_and_wire_record_serialization_are_exact() -> None:
    discarded = DiscardedAuthority(
        authority_id="authority_b", reason_code=DiscardReasonCode.SUPERSEDED
    )
    assert discarded.reason_code is DiscardReasonCode.SUPERSEDED
    with pytest.raises(ValidationError):
        DiscardedAuthority(authority_id="authority_b", reason_code="made_up")

    wire = _record().model_dump(mode="json")
    schema = json.loads(Path("schemas/authority_record.schema.json").read_text(encoding="utf-8"))
    assert set(wire) == set(schema["properties"])
    assert set(schema["required"]).issubset(wire)
    assert wire["expires_at"] is None
    assert isinstance(wire["supersedes"], list)
    assert isinstance(wire["exception_to"], list)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(wire)
