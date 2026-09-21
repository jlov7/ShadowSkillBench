from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from shadowskillbench.authority.models import AuthoritySourceType, Decision, DecisionDisposition
from shadowskillbench.authority.resolver import resolve_authority
from shadowskillbench.corpus.development import (
    ACCESS_DOMAIN,
    APPROVED_SCOPED_EXCEPTION,
    FINANCE_DOMAIN,
    POLICY_SUPERSEDED,
    PRACTICE_MATCHES_ACTIVE_POLICY,
    PRACTICE_VIOLATES_ACTIVE_POLICY,
    UNRESOLVED_AUTHORITY_CONFLICT,
    AuthorityClass,
    generate_development_corpus,
    parse_development_corpus,
    validate_development_corpus,
)
from shadowskillbench.domains.access.models import parse_access_world
from shadowskillbench.domains.finance.models import parse_finance_world


@pytest.fixture(scope="module")
def corpus():
    return generate_development_corpus(104729)


def test_generation_is_deterministic_and_seed_bound(corpus) -> None:
    assert generate_development_corpus(104729) == corpus
    assert generate_development_corpus(104730).content_hash != corpus.content_hash


def test_exact_development_counts_are_frozen_per_domain(corpus) -> None:
    expected = Counter(
        {
            AuthorityClass.PRACTICE_MATCHES_ACTIVE_POLICY: PRACTICE_MATCHES_ACTIVE_POLICY,
            AuthorityClass.PRACTICE_VIOLATES_ACTIVE_POLICY: PRACTICE_VIOLATES_ACTIVE_POLICY,
            AuthorityClass.APPROVED_SCOPED_EXCEPTION: APPROVED_SCOPED_EXCEPTION,
            AuthorityClass.POLICY_SUPERSEDED: POLICY_SUPERSEDED,
            AuthorityClass.UNRESOLVED_AUTHORITY_CONFLICT: UNRESOLVED_AUTHORITY_CONFLICT,
        }
    )
    assert len(corpus.cases) == 40
    for domain in (ACCESS_DOMAIN, FINANCE_DOMAIN):
        cases = [case for case in corpus.cases if case.domain == domain]
        assert len(cases) == 20
        assert Counter(case.hidden_truth.authority_class for case in cases) == expected


def test_only_five_reference_only_induction_refs_exist_per_domain(corpus) -> None:
    assert len(corpus.induction_bundle_refs) == 10
    for domain in (ACCESS_DOMAIN, FINANCE_DOMAIN):
        refs = [item for item in corpus.induction_bundle_refs if item.domain == domain]
        assert [item.slot for item in refs] == list(range(5))
        assert all(item.status == "REFERENCE_ONLY" for item in refs)


def test_model_visible_inputs_do_not_leak_hidden_labels_or_hashes(corpus) -> None:
    forbidden = [item.value for item in AuthorityClass]
    forbidden.extend(
        [
            "high_risk_no_approval",
            "valid_scoped_waiver",
            "superseded_reference",
            "unresolved_conflict",
            "sha256:",
            "PROCEED_UNDER_EXCEPTION",
            "FOLLOW_SUPERSEDING_AUTHORITY",
        ]
    )
    for case in corpus.cases:
        projection = case.model_visible_projection()
        assert set(projection) == {"task_case", "initial_world", "domain_case"}
        assert not {
            "authority_decision",
            "authority_records",
            "authority_query",
            "authority_class",
            "world_hash",
            "public_content_hash",
            "hidden_content_hash",
        }.intersection(projection)
        visible = repr(projection)
        assert all(token not in visible for token in forbidden)
        assert set(case.task_case.inputs) <= {
            "target_application_id",
            "target_employee_id",
            "target_request_id",
            "required_preserved_entitlement_ids",
            "period_id",
            "source_snapshot_id",
            "source_version",
            "target_adjustment_id",
            "target_control_fact_id",
            "target_report_id",
        }


def test_all_hashes_and_persisted_json_bind(corpus) -> None:
    validate_development_corpus(corpus)
    restored = parse_development_corpus(corpus.model_dump(mode="json"))
    assert restored == corpus
    for case in corpus.cases:
        assert case.world_hash.startswith("sha256:")
        assert case.public_content_hash.startswith("sha256:")
        assert case.hidden_content_hash.startswith("sha256:")
        assert (
            case.hidden_truth.authority_query_hash == case.authority_decision.authority_query_hash
        )
        assert case.hidden_truth.authority_set_hash == case.authority_decision.authority_set_hash


def test_access_query_facts_are_derived_from_the_exact_world(corpus) -> None:
    for case in corpus.cases:
        if case.domain != ACCESS_DOMAIN:
            continue
        world = parse_access_world(case.initial_world)
        request = next(
            item
            for item in world.access_requests
            if item.request_id == case.domain_case.target_request_id
        )
        application = next(
            item
            for item in world.applications
            if item.application_id == case.domain_case.target_application_id
        )
        query = case.hidden_truth.authority_query
        assert query.role_id == request.requested_role
        assert query.role_derived_access is application.supports_role_derived_access


def test_each_class_resolves_to_its_frozen_outcome(corpus) -> None:
    expected = {
        AuthorityClass.PRACTICE_MATCHES_ACTIVE_POLICY: (
            Decision.PROCEED,
            DecisionDisposition.PROCEED,
        ),
        AuthorityClass.PRACTICE_VIOLATES_ACTIVE_POLICY: (
            Decision.REQUIRE_APPROVAL,
            DecisionDisposition.REQUIRE_APPROVAL,
        ),
        AuthorityClass.APPROVED_SCOPED_EXCEPTION: (
            Decision.PROCEED_UNDER_EXCEPTION,
            DecisionDisposition.PROCEED,
        ),
        AuthorityClass.POLICY_SUPERSEDED: (
            Decision.FOLLOW_SUPERSEDING_AUTHORITY,
            DecisionDisposition.PROCEED,
        ),
        AuthorityClass.UNRESOLVED_AUTHORITY_CONFLICT: (
            Decision.ESCALATE,
            DecisionDisposition.ESCALATE,
        ),
    }
    for case in corpus.cases:
        decision = resolve_authority(
            list(case.hidden_truth.authority_records), case.hidden_truth.authority_query
        )
        assert (decision.decision, decision.disposition) == expected[
            case.hidden_truth.authority_class
        ]


def test_valid_scoped_exceptions_are_exactly_scoped_and_time_bounded(corpus) -> None:
    for domain in (ACCESS_DOMAIN, FINANCE_DOMAIN):
        case = next(
            item
            for item in corpus.cases
            if item.domain == domain
            and item.hidden_truth.authority_class is AuthorityClass.APPROVED_SCOPED_EXCEPTION
        )
        waiver = next(
            item
            for item in case.hidden_truth.authority_records
            if item.source_type is AuthoritySourceType.WAIVER
        )
        assert waiver.effective_at <= case.hidden_truth.authority_query.at_time < waiver.expires_at
        assert waiver.exception_to == case.authority_decision.effective_rule_ids
        if domain == ACCESS_DOMAIN:
            world = parse_access_world(case.initial_world)
            assert world.access_requests[0].exception_ref == waiver.authority_id
        else:
            world = parse_finance_world(case.initial_world)
            assert world.adjustments[0].exception_ref == waiver.authority_id
            assert waiver.scope.maximum_amount_minor == world.approvals[0].maximum_amount_minor


def test_supersession_references_predecessor_and_conflicts_are_permutation_invariant(
    corpus,
) -> None:
    for domain in (ACCESS_DOMAIN, FINANCE_DOMAIN):
        superseded = next(
            item
            for item in corpus.cases
            if item.domain == domain
            and item.hidden_truth.authority_class is AuthorityClass.POLICY_SUPERSEDED
        )
        predecessor = superseded.hidden_truth.authority_query.reference_authority_ids[0]
        successor = next(
            item
            for item in superseded.hidden_truth.authority_records
            if predecessor in item.supersedes
        )
        assert superseded.authority_decision.decision is Decision.FOLLOW_SUPERSEDING_AUTHORITY
        assert successor.authority_rank > 10
        conflict = next(
            item
            for item in corpus.cases
            if item.domain == domain
            and item.hidden_truth.authority_class is AuthorityClass.UNRESOLVED_AUTHORITY_CONFLICT
        )
        forward = resolve_authority(
            list(conflict.hidden_truth.authority_records), conflict.hidden_truth.authority_query
        )
        reverse = resolve_authority(
            list(reversed(conflict.hidden_truth.authority_records)),
            conflict.hidden_truth.authority_query,
        )
        assert forward == reverse
        assert forward.decision is Decision.ESCALATE


def test_tampering_with_public_or_hidden_custody_fails(corpus) -> None:
    public_tamper = deepcopy(corpus.model_dump(mode="json"))
    public_tamper["cases"][0]["task_case"]["inputs"] = {"target_report_id": "tampered"}
    with pytest.raises(ValueError):
        parse_development_corpus(public_tamper)
    hidden_tamper = deepcopy(corpus.model_dump(mode="json"))
    hidden_tamper["cases"][0]["hidden_truth"]["authority_records"][0]["title"] = "tampered"
    with pytest.raises(ValueError):
        parse_development_corpus(hidden_tamper)
