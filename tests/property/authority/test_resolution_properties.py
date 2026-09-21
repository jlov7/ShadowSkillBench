from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.authority.models import (
    create_access_authority_query,
    create_authority_record,
    create_issuer_registry,
)
from shadowskillbench.authority.resolver import resolve_authority


def _registry() -> object:
    return create_issuer_registry(
        authorizations=[
            {
                "issuer_id": "issuer_primary",
                "issuer_role": "policy_owner",
                "authority_rank": 10,
                "allowed_domains": ["access_provisioning", "financial_adjustments"],
                "allowed_source_types": ["BEHAVIOR_TRACE", "POLICY"],
                "can_supersede": True,
                "can_issue_scoped_evidence": False,
            }
        ]
    )


def _query() -> object:
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


def _record(authority_id: str) -> object:
    return create_authority_record(
        authority_id=authority_id,
        schema_version="1.0",
        source_type="POLICY",
        title=authority_id,
        issuer_id="issuer_primary",
        issuer_role="policy_owner",
        authority_rank=10,
        action_type="grant_access",
        scope={
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
        },
        effective_at="2026-01-01T00:00:00Z",
        expires_at=None,
        supersedes=[],
        exception_to=[],
        provenance_locator=f"authority/{authority_id}",
        normative_status="ACTIVE_AUTHORITY",
    )


def _descriptive_record(authority_id: str) -> object:
    return create_authority_record(
        authority_id=authority_id,
        schema_version="1.0",
        source_type="BEHAVIOR_TRACE",
        title=authority_id,
        issuer_id="issuer_primary",
        issuer_role="policy_owner",
        authority_rank=10,
        action_type="grant_access",
        scope={
            "domain": "access_provisioning",
            "scope_kind": "descriptive",
            "subject_id": None,
            "resource_id": None,
            "organization_id": None,
            "geography_id": None,
            "role_id": None,
        },
        effective_at="2026-01-01T00:00:00Z",
        expires_at=None,
        supersedes=[],
        exception_to=[],
        provenance_locator=f"authority/{authority_id}",
        normative_status="DESCRIPTIVE_ONLY",
    )


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(st.permutations((0, 1, 2)))
def test_input_permutation_never_changes_equivalent_coalesced_decision(
    order: tuple[int, ...],
) -> None:
    records = [_record("rule_a"), _record("rule_b"), _record("rule_c")]
    baseline = resolve_authority(records, _query())
    permuted = resolve_authority([records[index] for index in order], _query())

    assert permuted.decision_hash == baseline.decision_hash
    assert permuted.authority_set_hash == baseline.authority_set_hash


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(st.permutations((0, 1, 2)), st.booleans())
def test_optional_discarded_records_have_stable_order_and_bound_set_hash(
    order: tuple[int, ...], include_trace: bool
) -> None:
    rules = [_record("rule_a"), _record("rule_b"), _record("rule_c")]
    records = [rules[index] for index in order]
    if include_trace:
        records.append(_descriptive_record("trace"))

    result = resolve_authority(records, _query())
    reordered = resolve_authority(list(reversed(records)), _query())
    baseline = resolve_authority(rules, _query())

    assert result.authority_set_hash == reordered.authority_set_hash
    assert result.decision_hash == reordered.decision_hash
    assert result.effective_rule_ids == baseline.effective_rule_ids
    if include_trace:
        assert result.authority_set_hash != baseline.authority_set_hash
        assert result.discarded_authorities[0].authority_id == "trace"
    else:
        assert result.authority_set_hash == baseline.authority_set_hash
