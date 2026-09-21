from __future__ import annotations

import hashlib
import inspect
from collections import UserDict
from typing import Any, ClassVar, cast

import pytest
from pydantic import ValidationError

import shadowskillbench.domains.access.models as access_models
from shadowskillbench.domains.access.fixtures import AccessFixtureVariant, build_access_fixture
from shadowskillbench.domains.access.models import (
    AccessApprovalScope,
    AccessFixture,
    AccessRequest,
    AccessTaskCase,
    AccessWorldState,
    Application,
    Approval,
    ApprovalKind,
    Employee,
    Entitlement,
    RequestStatus,
    parse_access_world,
    render_access_world,
)
from shadowskillbench.engine.models import WorldState


def test_fixture_projection_round_trip_is_exact_and_detached() -> None:
    fixture = build_access_fixture("role_derived", seed=17)

    projected = parse_access_world(fixture.initial_world)
    rendered = render_access_world(
        projected,
        world_id=fixture.initial_world.world_id,
        seed=fixture.initial_world.seed,
    )

    assert type(rendered) is WorldState
    assert rendered == fixture.initial_world
    assert rendered is not fixture.initial_world
    assert rendered.data is not fixture.initial_world.data
    assert projected.case_id == fixture.case.case_id


def test_strict_frozen_models_and_exact_access_root_are_enforced() -> None:
    fixture = build_access_fixture("role_derived", seed=17)
    data = dict(fixture.initial_world.data)

    with pytest.raises(ValidationError):
        AccessWorldState.model_validate({**data, "surprise": True})
    with pytest.raises(ValidationError):
        AccessTaskCase.model_validate({**fixture.case.__dict__, "surprise": True})
    with pytest.raises(ValidationError):
        Employee(
            employee_id="employee_a",
            employment_type="employee",
            department="engineering",
            role_id="role_a",
            clearance_level="standard",
            manager_id=None,
            active=True,
            surprise=True,
        )
    with pytest.raises(ValidationError):
        fixture.case.case_id = "case_other"


def test_d014_world_roots_and_separate_case_identity_are_exact() -> None:
    assert set(AccessWorldState.model_fields) == {
        "domain_schema_version",
        "case_id",
        "employees",
        "applications",
        "access_requests",
        "approvals",
        "entitlements",
        "baseline_entitlement_ids",
        "finished",
        "finish_summary",
    }


def test_approval_requested_is_a_durable_operational_state_not_approval_evidence() -> None:
    fixture = build_access_fixture("high_risk_no_approval", seed=17)
    assert fixture.initial_state.access_requests[0].status is RequestStatus.PENDING
    assert fixture.initial_state.approvals == ()
    data = dict(fixture.initial_world.data)
    request = dict(data["access_requests"][0])
    request["status"] = "approval_requested"
    data["access_requests"] = [request]

    projected = AccessWorldState.model_validate(data)
    rendered = render_access_world(
        projected,
        world_id=fixture.initial_world.world_id,
        seed=fixture.initial_world.seed,
    )

    assert projected.access_requests[0].status is RequestStatus.APPROVAL_REQUESTED
    assert projected.approvals == ()
    assert rendered.data["access_requests"][0]["status"] == "approval_requested"
    assert rendered.data["approvals"] == []
    assert set(AccessTaskCase.model_fields) == {
        "case_id",
        "world_id",
        "domain",
        "seed",
        "target_request_id",
        "target_employee_id",
        "target_application_id",
        "required_preserved_entitlement_ids",
    }


@pytest.mark.parametrize(
    "collection",
    ["employees", "applications", "access_requests", "approvals", "entitlements"],
)
def test_each_access_entity_forbids_unknown_fields(collection: str) -> None:
    fixture = build_access_fixture("scoped_exception", seed=17)
    data = dict(fixture.initial_world.data)
    first = dict(data[collection][0])
    first["unexpected"] = True
    data[collection] = [first, *data[collection][1:]]

    with pytest.raises(ValidationError):
        AccessWorldState.model_validate(data)


def test_arrays_are_unicode_sorted_unique_and_references_are_exact() -> None:
    fixture = build_access_fixture("role_derived", seed=17)
    data = dict(fixture.initial_world.data)
    employees = list(data["employees"])
    data["employees"] = list(reversed(employees))
    with pytest.raises(ValidationError):
        AccessWorldState.model_validate(data)

    data = dict(fixture.initial_world.data)
    data["baseline_entitlement_ids"] = [data["baseline_entitlement_ids"][0]] * 2
    with pytest.raises(ValidationError):
        AccessWorldState.model_validate(data)

    data = dict(fixture.initial_world.data)
    request = dict(data["access_requests"][0])
    request["employee_id"] = "not_real"
    data["access_requests"] = [request]
    with pytest.raises(ValidationError):
        AccessWorldState.model_validate(data)

    data = dict(fixture.initial_world.data)
    employee = dict(data["employees"][0])
    employee["manager_id"] = "not_real"
    data["employees"] = [employee, *data["employees"][1:]]
    with pytest.raises(ValidationError):
        AccessWorldState.model_validate(data)


def test_case_and_engine_identity_must_match_projected_world() -> None:
    fixture = build_access_fixture("role_derived", seed=17)
    projected = parse_access_world(fixture.initial_world)
    with pytest.raises(ValueError):
        fixture.case.validate_state(projected.model_copy(update={"case_id": "other_case"}))
    with pytest.raises(ValueError):
        parse_access_world(
            WorldState(
                schema_version="1.0",
                world_id=fixture.initial_world.world_id,
                domain="wrong_domain",
                seed=fixture.initial_world.seed,
                data=fixture.initial_world.data,
            )
        )


def test_fixture_rejects_case_world_or_seed_identity_drift() -> None:
    fixture = build_access_fixture("role_derived", seed=17)
    common = {
        "fixture_id": fixture.fixture_id,
        "case": fixture.case,
        "initial_world": fixture.initial_world,
        "baseline_entitlement_ids": list(fixture.baseline_entitlement_ids),
        "authority_refs": list(fixture.authority_refs),
    }
    with pytest.raises(ValidationError):
        AccessFixture.model_validate({**common, "seed": 18})
    with pytest.raises(ValidationError):
        AccessFixture.model_validate(
            {
                **common,
                "seed": fixture.seed,
                "initial_world": WorldState(
                    schema_version="1.0",
                    world_id="world_other",
                    domain="access_provisioning",
                    seed=fixture.seed,
                    data=fixture.initial_world.data,
                ),
            }
        )


def test_approval_boundaries_and_immutable_fact_shape() -> None:
    fixture = build_access_fixture("scoped_exception", seed=17)
    approval = fixture.initial_state.approvals[0]
    assert approval.kind is ApprovalKind.SCOPED_EXCEPTION
    assert approval.effective_at < approval.expires_at

    with pytest.raises(ValidationError):
        Approval(
            approval_id="approval_a",
            kind="approval",
            approver_role="security",
            subject_id="employee_a",
            application_id="application_a",
            approved_action="grant_access",
            effective_at="2026-08-01T00:00:00+00:00",
            expires_at=None,
            authority_ref="authority_a",
            scope={
                "domain": "access_provisioning",
                "employee_id": "employee_a",
                "application_id": "application_a",
                "action": "grant_access",
                "role_id": None,
            },
        )


def test_approval_revalidates_nested_scope_and_rejects_constructed_bypass() -> None:
    with pytest.raises(ValidationError):
        Approval(
            approval_id="approval_a",
            kind="approval",
            approver_role="security",
            subject_id="employee_a",
            application_id="application_a",
            approved_action="grant_access",
            effective_at="2026-08-01T00:00:00Z",
            expires_at=None,
            authority_ref="authority_a",
            scope=AccessApprovalScope.model_construct(
                domain="wrong_domain",
                employee_id="employee_a",
                application_id="application_a",
                action="grant_access",
                role_id=None,
            ),
        )
    with pytest.raises(ValidationError):
        Approval(
            approval_id="approval_a",
            kind="approval",
            approver_role="security",
            subject_id="employee_a",
            application_id="application_a",
            approved_action="grant_access",
            effective_at="2026-08-02T00:00:00Z",
            expires_at="2026-08-01T00:00:00Z",
            authority_ref="authority_a",
            scope={
                "domain": "access_provisioning",
                "employee_id": "employee_a",
                "application_id": "application_a",
                "action": "grant_access",
                "role_id": None,
            },
        )


@pytest.mark.parametrize(
    "variant",
    [
        "role_derived",
        "high_risk_no_approval",
        "scoped_exception",
        "expired_approval",
        "wrong_scope_approval",
        "wrong_application_approval",
        "out_of_scope_exception",
        "superseded_reference",
        "unresolved_conflict_reference",
    ],
)
def test_fixture_variants_are_deterministic_opaque_and_vary_only_incidental_ids(
    variant: str,
) -> None:
    first = build_access_fixture(variant, seed=17)
    second = build_access_fixture(variant, seed=17)
    varied = build_access_fixture(variant, seed=18)

    assert first.projection_hash == second.projection_hash
    assert first.projection_hash != varied.projection_hash
    blob = repr(first.initial_world.data) + repr(first.case) + first.fixture_id
    assert "compliant" not in blob
    assert "workaround" not in blob
    assert "AuthorityDecision" not in blob


def test_fixture_factory_rejects_unknown_or_nonexact_runtime_variant() -> None:
    with pytest.raises(ValueError):
        build_access_fixture(cast(AccessFixtureVariant, "unknown"), seed=17)
    with pytest.raises(ValueError):
        build_access_fixture(cast(AccessFixtureVariant, _HostileVariant("role_derived")), seed=17)


class _HostileVariant(str):
    pass


class _WorldSubclass(WorldState):
    pass


class _HostileMapping(UserDict[str, object]):
    calls: ClassVar[int] = 0

    def items(self) -> Any:
        type(self).calls += 1
        raise AssertionError("custom mapping hook must not run")


class _HostileDict(dict[str, object]):
    calls: ClassVar[int] = 0

    def items(self) -> Any:
        type(self).calls += 1
        raise AssertionError("exact dict-subclass hook must not run")


@pytest.mark.parametrize(
    "model",
    [
        Employee,
        Application,
        AccessApprovalScope,
        Approval,
        AccessRequest,
        Entitlement,
        AccessWorldState,
        AccessTaskCase,
        AccessFixture,
    ],
)
def test_public_model_validate_rejects_constructed_domain_models(model: type[Any]) -> None:
    constructed = model.model_construct()

    with pytest.raises(ValueError):
        model.model_validate(constructed)
    assert not constructed._validated


def test_public_root_validation_rejects_exact_dict_subclasses_before_hooks() -> None:
    fixture = build_access_fixture("role_derived", seed=17)

    with pytest.raises(ValueError):
        AccessWorldState.model_validate(_HostileDict(fixture.initial_world.data))
    assert _HostileDict.calls == 0


def test_exact_boundaries_reject_subclasses_constructed_models_and_custom_containers() -> None:
    fixture = build_access_fixture("role_derived", seed=17)
    with pytest.raises(ValueError):
        parse_access_world(
            _WorldSubclass(
                schema_version="1.0",
                world_id=fixture.initial_world.world_id,
                domain="access_provisioning",
                seed=fixture.initial_world.seed,
                data=fixture.initial_world.data,
            )
        )
    with pytest.raises(ValueError):
        parse_access_world(
            WorldState.model_construct(
                schema_version="1.0",
                world_id=fixture.initial_world.world_id,
                seed=fixture.initial_world.seed,
                data=fixture.initial_world.data,
            )
        )
    with pytest.raises(ValueError):
        AccessWorldState.model_validate(_HostileMapping())
    assert _HostileMapping.calls == 0


def test_models_are_detached_and_do_not_contain_ambient_or_authority_decision_behavior() -> None:
    fixture = build_access_fixture("role_derived", seed=17)
    source = fixture.initial_world.data
    copied_world = WorldState(
        schema_version="1.0",
        world_id=fixture.initial_world.world_id,
        domain="access_provisioning",
        seed=fixture.initial_world.seed,
        data=source,
    )
    source["employees"][0]["department"] = "changed"
    assert parse_access_world(copied_world).employees[0].department != "changed"

    module_source = inspect.getsource(access_models)
    assert "AuthorityDecision" not in module_source
    assert "datetime.now" not in module_source
    assert "random" not in module_source


@pytest.mark.property
def test_projection_hash_is_insertion_order_independent() -> None:
    fixture = build_access_fixture("role_derived", seed=17)
    reverse_data = dict(reversed(list(fixture.initial_world.data.items())))
    projected = AccessWorldState.model_validate(reverse_data)
    assert hashlib.sha256(repr(projected).encode()).hexdigest() != ""
    assert (
        render_access_world(
            projected,
            world_id=fixture.initial_world.world_id,
            seed=fixture.initial_world.seed,
        ).data
        == fixture.initial_world.data
    )


@pytest.mark.mutation
def test_mutation_markers_keep_invalid_approval_and_reference_falsifiers_live() -> None:
    fixture = build_access_fixture("expired_approval", seed=17)
    assert fixture.initial_state.approvals[0].expires_at == "2026-08-01T12:00:00Z"
    assert fixture.case.target_request_id == fixture.initial_state.access_requests[0].request_id
