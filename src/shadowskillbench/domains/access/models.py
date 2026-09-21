from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from pydantic.config import ExtraValues

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.engine.models import JsonObject, PathSafeId, WorldState

ACCESS_DOMAIN = "access_provisioning"
DOMAIN_SCHEMA_VERSION = "1.0"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class EmploymentType(StrEnum):
    EMPLOYEE = "employee"
    CONTRACTOR = "contractor"


class ClearanceLevel(StrEnum):
    STANDARD = "standard"
    ELEVATED = "elevated"


class RiskTier(StrEnum):
    LOW = "low"
    HIGH = "high"


class RequestStatus(StrEnum):
    PENDING = "pending"
    APPROVAL_REQUESTED = "approval_requested"
    GRANTED = "granted"
    DENIED = "denied"


class ApprovalKind(StrEnum):
    APPROVAL = "approval"
    SCOPED_EXCEPTION = "scoped_exception"


def _id(value: object, *, field: str) -> PathSafeId:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact path-safe identifier")
    return cast(PathSafeId, value)


def _sorted_unique_ids(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError(f"{field} must be Unicode-sorted and unique")
    return values


def _utc(value: str, *, field: str) -> str:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise ValueError(f"{field} must use canonical UTC seconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"{field} must be a valid UTC timestamp") from error
    return value


def _exact_dict(value: object, *, model_name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{model_name} input must be an exact built-in object")
    return cast(dict[str, object], value)


class _StrictFrozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    _validated: bool = PrivateAttr(default=False)

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        if type(obj) is not dict:
            raise ValueError(f"{cls.__name__} input must be an exact built-in object")
        return super().model_validate(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @model_validator(mode="after")
    def mark_validated(self) -> Self:
        object.__setattr__(self, "_validated", True)
        return self


class Employee(_StrictFrozen):
    employee_id: PathSafeId
    employment_type: EmploymentType
    department: PathSafeId
    role_id: PathSafeId
    clearance_level: ClearanceLevel
    manager_id: PathSafeId | None
    active: bool

    @field_validator("employee_id", "department", "role_id", "manager_id", mode="before")
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId | None:
        if value is None:
            return None
        return _id(value, field=getattr(info, "field_name", "identifier"))

    @field_validator("employment_type", mode="before")
    @classmethod
    def validate_employment_type(cls, value: object) -> EmploymentType:
        if type(value) is str:
            return EmploymentType(value)
        if type(value) is EmploymentType:
            return value
        raise ValueError("employment_type must be an explicit enum")

    @field_validator("clearance_level", mode="before")
    @classmethod
    def validate_clearance(cls, value: object) -> ClearanceLevel:
        if type(value) is str:
            return ClearanceLevel(value)
        if type(value) is ClearanceLevel:
            return value
        raise ValueError("clearance_level must be an explicit enum")


class Application(_StrictFrozen):
    application_id: PathSafeId
    risk_tier: RiskTier
    required_clearance: ClearanceLevel
    conflicts_with: tuple[PathSafeId, ...]
    supports_role_derived_access: bool

    @field_validator("application_id", mode="before")
    @classmethod
    def validate_application_id(cls, value: object) -> PathSafeId:
        return _id(value, field="application_id")

    @field_validator("risk_tier", mode="before")
    @classmethod
    def validate_risk_tier(cls, value: object) -> RiskTier:
        if type(value) is str:
            return RiskTier(value)
        if type(value) is RiskTier:
            return value
        raise ValueError("risk_tier must be an explicit enum")

    @field_validator("required_clearance", mode="before")
    @classmethod
    def validate_required_clearance(cls, value: object) -> ClearanceLevel:
        if type(value) is str:
            return ClearanceLevel(value)
        if type(value) is ClearanceLevel:
            return value
        raise ValueError("required_clearance must be an explicit enum")

    @field_validator("conflicts_with", mode="before")
    @classmethod
    def validate_conflicts(cls, value: object) -> tuple[PathSafeId, ...]:
        if type(value) is not list:
            raise ValueError("conflicts_with must be an exact JSON array")
        return tuple(_id(item, field="conflicts_with") for item in cast(list[object], value))

    @field_validator("conflicts_with")
    @classmethod
    def validate_sorted_conflicts(cls, value: tuple[PathSafeId, ...]) -> tuple[PathSafeId, ...]:
        return cast(tuple[PathSafeId, ...], _sorted_unique_ids(value, field="conflicts_with"))


class AccessApprovalScope(_StrictFrozen):
    domain: Literal["access_provisioning"]
    employee_id: PathSafeId
    application_id: PathSafeId
    action: Literal["grant_access"]
    role_id: PathSafeId | None

    @field_validator("employee_id", "application_id", "role_id", mode="before")
    @classmethod
    def validate_scope_ids(cls, value: object, info: object) -> PathSafeId | None:
        if value is None:
            return None
        return _id(value, field=getattr(info, "field_name", "scope identifier"))


class Approval(_StrictFrozen):
    approval_id: PathSafeId
    kind: ApprovalKind
    approver_role: PathSafeId
    subject_id: PathSafeId
    application_id: PathSafeId
    approved_action: Literal["grant_access"]
    effective_at: str
    expires_at: str | None
    authority_ref: PathSafeId
    scope: AccessApprovalScope

    @field_validator(
        "approval_id",
        "approver_role",
        "subject_id",
        "application_id",
        "authority_ref",
        mode="before",
    )
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _id(value, field=getattr(info, "field_name", "identifier"))

    @field_validator("kind", mode="before")
    @classmethod
    def validate_kind(cls, value: object) -> ApprovalKind:
        if type(value) is str:
            return ApprovalKind(value)
        if type(value) is ApprovalKind:
            return value
        raise ValueError("kind must be an explicit enum")

    @field_validator("scope", mode="before")
    @classmethod
    def validate_scope(cls, value: object) -> dict[str, object]:
        if isinstance(value, AccessApprovalScope):
            if type(value) is not AccessApprovalScope or not value._validated:
                raise ValueError("scope must be an exact validated AccessApprovalScope")
            return {
                "domain": value.domain,
                "employee_id": value.employee_id,
                "application_id": value.application_id,
                "action": value.action,
                "role_id": value.role_id,
            }
        return _exact_dict(value, model_name="AccessApprovalScope")

    @field_validator("effective_at", "expires_at")
    @classmethod
    def validate_times(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _utc(value, field=getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def validate_scope_and_times(self) -> Self:
        if self.expires_at is not None and self.effective_at >= self.expires_at:
            raise ValueError("expires_at must be later than effective_at")
        if (
            self.scope.employee_id != self.subject_id
            or self.scope.application_id != self.application_id
            or self.scope.action != self.approved_action
        ):
            raise ValueError("approval scope must exactly bind its immutable facts")
        return self


class AccessRequest(_StrictFrozen):
    request_id: PathSafeId
    employee_id: PathSafeId
    application_id: PathSafeId
    requested_role: PathSafeId
    status: RequestStatus
    approval_refs: tuple[PathSafeId, ...]
    exception_ref: PathSafeId | None

    @field_validator(
        "request_id",
        "employee_id",
        "application_id",
        "requested_role",
        "exception_ref",
        mode="before",
    )
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId | None:
        if value is None:
            return None
        return _id(value, field=getattr(info, "field_name", "identifier"))

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, value: object) -> RequestStatus:
        if type(value) is str:
            return RequestStatus(value)
        if type(value) is RequestStatus:
            return value
        raise ValueError("status must be an explicit enum")

    @field_validator("approval_refs", mode="before")
    @classmethod
    def validate_approval_refs(cls, value: object) -> tuple[PathSafeId, ...]:
        if type(value) is not list:
            raise ValueError("approval_refs must be an exact JSON array")
        return tuple(_id(item, field="approval_refs") for item in cast(list[object], value))

    @field_validator("approval_refs")
    @classmethod
    def validate_sorted_refs(cls, value: tuple[PathSafeId, ...]) -> tuple[PathSafeId, ...]:
        return cast(tuple[PathSafeId, ...], _sorted_unique_ids(value, field="approval_refs"))


class Entitlement(_StrictFrozen):
    entitlement_id: PathSafeId
    employee_id: PathSafeId
    application_id: PathSafeId
    role_id: PathSafeId

    @field_validator("entitlement_id", "employee_id", "application_id", "role_id", mode="before")
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _id(value, field=getattr(info, "field_name", "identifier"))


T = type[Employee] | type[Application] | type[AccessRequest] | type[Approval] | type[Entitlement]


def _validated_item(value: object, *, model: T, fields: tuple[str, ...]) -> object:
    if isinstance(value, model):
        if type(value) is not model or not value._validated:
            raise ValueError(f"{model.__name__} must be an exact validated model")
        return {field: getattr(value, field) for field in fields}
    return _exact_dict(value, model_name=model.__name__)


def _validated_items(value: object, *, model: T, fields: tuple[str, ...]) -> tuple[object, ...]:
    if type(value) is not list:
        raise ValueError(f"{model.__name__} collection must be an exact JSON array")
    return tuple(
        _validated_item(item, model=model, fields=fields) for item in cast(list[object], value)
    )


class AccessWorldState(_StrictFrozen):
    domain_schema_version: Literal["1.0"]
    case_id: PathSafeId
    employees: tuple[Employee, ...]
    applications: tuple[Application, ...]
    access_requests: tuple[AccessRequest, ...]
    approvals: tuple[Approval, ...]
    entitlements: tuple[Entitlement, ...]
    baseline_entitlement_ids: tuple[PathSafeId, ...]
    finished: bool
    finish_summary: str | None

    @field_validator("case_id", mode="before")
    @classmethod
    def validate_case_id(cls, value: object) -> PathSafeId:
        return _id(value, field="case_id")

    @field_validator("employees", mode="before")
    @classmethod
    def validate_employees(cls, value: object) -> tuple[object, ...]:
        return _validated_items(
            value,
            model=Employee,
            fields=(
                "employee_id",
                "employment_type",
                "department",
                "role_id",
                "clearance_level",
                "manager_id",
                "active",
            ),
        )

    @field_validator("applications", mode="before")
    @classmethod
    def validate_applications(cls, value: object) -> tuple[object, ...]:
        return _validated_items(
            value,
            model=Application,
            fields=(
                "application_id",
                "risk_tier",
                "required_clearance",
                "conflicts_with",
                "supports_role_derived_access",
            ),
        )

    @field_validator("access_requests", mode="before")
    @classmethod
    def validate_requests(cls, value: object) -> tuple[object, ...]:
        return _validated_items(
            value,
            model=AccessRequest,
            fields=(
                "request_id",
                "employee_id",
                "application_id",
                "requested_role",
                "status",
                "approval_refs",
                "exception_ref",
            ),
        )

    @field_validator("approvals", mode="before")
    @classmethod
    def validate_approvals(cls, value: object) -> tuple[object, ...]:
        return _validated_items(
            value,
            model=Approval,
            fields=(
                "approval_id",
                "kind",
                "approver_role",
                "subject_id",
                "application_id",
                "approved_action",
                "effective_at",
                "expires_at",
                "authority_ref",
                "scope",
            ),
        )

    @field_validator("entitlements", mode="before")
    @classmethod
    def validate_entitlements(cls, value: object) -> tuple[object, ...]:
        return _validated_items(
            value,
            model=Entitlement,
            fields=("entitlement_id", "employee_id", "application_id", "role_id"),
        )

    @field_validator("baseline_entitlement_ids", mode="before")
    @classmethod
    def validate_baselines(cls, value: object) -> tuple[PathSafeId, ...]:
        if type(value) is not list:
            raise ValueError("baseline_entitlement_ids must be an exact JSON array")
        return tuple(
            _id(item, field="baseline_entitlement_ids") for item in cast(list[object], value)
        )

    @field_validator("baseline_entitlement_ids")
    @classmethod
    def validate_sorted_baselines(cls, value: tuple[PathSafeId, ...]) -> tuple[PathSafeId, ...]:
        return cast(
            tuple[PathSafeId, ...], _sorted_unique_ids(value, field="baseline_entitlement_ids")
        )

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        _validate_sorted_entities(self.employees, "employee_id", "employees")
        _validate_sorted_entities(self.applications, "application_id", "applications")
        _validate_sorted_entities(self.access_requests, "request_id", "access_requests")
        _validate_sorted_entities(self.approvals, "approval_id", "approvals")
        _validate_sorted_entities(self.entitlements, "entitlement_id", "entitlements")
        employee_ids = {employee.employee_id for employee in self.employees}
        application_ids = {application.application_id for application in self.applications}
        approval_by_id = {approval.approval_id: approval for approval in self.approvals}
        entitlement_ids = {entitlement.entitlement_id for entitlement in self.entitlements}
        for application in self.applications:
            if application.application_id in application.conflicts_with:
                raise ValueError("application cannot conflict with itself")
            if not set(application.conflicts_with) <= application_ids:
                raise ValueError("application conflict reference does not exist")
        for employee in self.employees:
            if employee.manager_id is not None and employee.manager_id not in employee_ids:
                raise ValueError("employee manager reference does not exist")
        for request in self.access_requests:
            if (
                request.employee_id not in employee_ids
                or request.application_id not in application_ids
            ):
                raise ValueError("request reference does not exist")
            if any(
                reference not in approval_by_id
                or approval_by_id[reference].kind is not ApprovalKind.APPROVAL
                for reference in request.approval_refs
            ):
                raise ValueError("request approval reference does not exist or is not an approval")
            if request.exception_ref is not None and (
                request.exception_ref not in approval_by_id
                or approval_by_id[request.exception_ref].kind is not ApprovalKind.SCOPED_EXCEPTION
            ):
                raise ValueError("request exception reference does not exist or is not scoped")
        for approval in self.approvals:
            if (
                approval.subject_id not in employee_ids
                or approval.application_id not in application_ids
            ):
                raise ValueError("approval reference does not exist")
        for entitlement in self.entitlements:
            if (
                entitlement.employee_id not in employee_ids
                or entitlement.application_id not in application_ids
            ):
                raise ValueError("entitlement reference does not exist")
        if not set(self.baseline_entitlement_ids) <= entitlement_ids:
            raise ValueError("baseline entitlement reference does not exist")
        return self


def _validate_sorted_entities(values: tuple[object, ...], field: str, collection: str) -> None:
    identifiers = tuple(cast(str, getattr(value, field)) for value in values)
    if tuple(sorted(identifiers)) != identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{collection} must be Unicode ID-sorted and unique")


class AccessTaskCase(_StrictFrozen):
    case_id: PathSafeId
    world_id: PathSafeId
    domain: Literal["access_provisioning"]
    seed: int
    target_request_id: PathSafeId
    target_employee_id: PathSafeId
    target_application_id: PathSafeId
    required_preserved_entitlement_ids: tuple[PathSafeId, ...]

    @field_validator(
        "case_id",
        "world_id",
        "target_request_id",
        "target_employee_id",
        "target_application_id",
        mode="before",
    )
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _id(value, field=getattr(info, "field_name", "identifier"))

    @field_validator("required_preserved_entitlement_ids", mode="before")
    @classmethod
    def validate_preserved(cls, value: object) -> tuple[PathSafeId, ...]:
        if type(value) is not list:
            raise ValueError("required_preserved_entitlement_ids must be an exact JSON array")
        return tuple(
            _id(item, field="required_preserved_entitlement_ids")
            for item in cast(list[object], value)
        )

    @field_validator("required_preserved_entitlement_ids")
    @classmethod
    def validate_sorted_preserved(cls, value: tuple[PathSafeId, ...]) -> tuple[PathSafeId, ...]:
        return cast(
            tuple[PathSafeId, ...],
            _sorted_unique_ids(value, field="required_preserved_entitlement_ids"),
        )

    def validate_state(self, state: AccessWorldState) -> None:
        checked = _validated_access_state(state)
        if self.case_id != checked.case_id:
            raise ValueError("case_id does not match projected world")
        requests = {request.request_id: request for request in checked.access_requests}
        employees = {employee.employee_id: employee for employee in checked.employees}
        applications = {application.application_id for application in checked.applications}
        if self.target_request_id not in requests:
            raise ValueError("case target request does not exist")
        request = requests[self.target_request_id]
        if (
            request.employee_id != self.target_employee_id
            or request.application_id != self.target_application_id
            or self.target_employee_id not in employees
            or self.target_application_id not in applications
        ):
            raise ValueError("case target references do not match request")
        if not employees[self.target_employee_id].active:
            raise ValueError("case target employee must be active")
        if not set(self.required_preserved_entitlement_ids) <= set(
            checked.baseline_entitlement_ids
        ):
            raise ValueError("case preserved entitlements must be baseline facts")


def _access_state_fields(state: AccessWorldState) -> dict[str, object]:
    return {
        "domain_schema_version": state.domain_schema_version,
        "case_id": state.case_id,
        "employees": [_employee_projection(value) for value in state.employees],
        "applications": [_application_projection(value) for value in state.applications],
        "access_requests": [_request_projection(value) for value in state.access_requests],
        "approvals": [_approval_projection(value) for value in state.approvals],
        "entitlements": [_entitlement_projection(value) for value in state.entitlements],
        "baseline_entitlement_ids": list(state.baseline_entitlement_ids),
        "finished": state.finished,
        "finish_summary": state.finish_summary,
    }


def _validated_access_state(value: object) -> AccessWorldState:
    if type(value) is not AccessWorldState or not value._validated:
        raise ValueError("AccessWorldState must be an exact validated model")
    return AccessWorldState.model_validate(_access_state_fields(value))


def _employee_projection(value: Employee) -> dict[str, object]:
    return {
        "employee_id": value.employee_id,
        "employment_type": value.employment_type.value,
        "department": value.department,
        "role_id": value.role_id,
        "clearance_level": value.clearance_level.value,
        "manager_id": value.manager_id,
        "active": value.active,
    }


def _application_projection(value: Application) -> dict[str, object]:
    return {
        "application_id": value.application_id,
        "risk_tier": value.risk_tier.value,
        "required_clearance": value.required_clearance.value,
        "conflicts_with": list(value.conflicts_with),
        "supports_role_derived_access": value.supports_role_derived_access,
    }


def _request_projection(value: AccessRequest) -> dict[str, object]:
    return {
        "request_id": value.request_id,
        "employee_id": value.employee_id,
        "application_id": value.application_id,
        "requested_role": value.requested_role,
        "status": value.status.value,
        "approval_refs": list(value.approval_refs),
        "exception_ref": value.exception_ref,
    }


def _approval_projection(value: Approval) -> dict[str, object]:
    return {
        "approval_id": value.approval_id,
        "kind": value.kind.value,
        "approver_role": value.approver_role,
        "subject_id": value.subject_id,
        "application_id": value.application_id,
        "approved_action": value.approved_action,
        "effective_at": value.effective_at,
        "expires_at": value.expires_at,
        "authority_ref": value.authority_ref,
        "scope": {
            "domain": value.scope.domain,
            "employee_id": value.scope.employee_id,
            "application_id": value.scope.application_id,
            "action": value.scope.action,
            "role_id": value.scope.role_id,
        },
    }


def _entitlement_projection(value: Entitlement) -> dict[str, object]:
    return {
        "entitlement_id": value.entitlement_id,
        "employee_id": value.employee_id,
        "application_id": value.application_id,
        "role_id": value.role_id,
    }


def parse_access_world(state: WorldState) -> AccessWorldState:
    if type(state) is not WorldState:
        raise ValueError("state must be an exact WorldState")
    fields = ("schema_version", "world_id", "domain", "seed", "data")
    try:
        world = WorldState.model_validate({field: getattr(state, field) for field in fields})
    except Exception as error:
        raise ValueError("state must be complete and revalidatable") from error
    if world.schema_version != DOMAIN_SCHEMA_VERSION or world.domain != ACCESS_DOMAIN:
        raise ValueError("state identity is not access_provisioning")
    if type(world.data) is not dict:
        raise ValueError("state data must be an exact object")
    return AccessWorldState.model_validate(world.data)


def render_access_world(state: AccessWorldState, *, world_id: PathSafeId, seed: int) -> WorldState:
    checked = _validated_access_state(state)
    return WorldState(
        schema_version=DOMAIN_SCHEMA_VERSION,
        world_id=_id(world_id, field="world_id"),
        domain=ACCESS_DOMAIN,
        seed=seed,
        data=cast(JsonObject, _access_state_fields(checked)),
    )


class AccessFixture(_StrictFrozen):
    fixture_id: PathSafeId
    seed: int
    case: AccessTaskCase
    initial_world: WorldState
    baseline_entitlement_ids: tuple[PathSafeId, ...]
    authority_refs: tuple[PathSafeId, ...]

    @field_validator("fixture_id", mode="before")
    @classmethod
    def validate_fixture_id(cls, value: object) -> PathSafeId:
        return _id(value, field="fixture_id")

    @field_validator("case", mode="before")
    @classmethod
    def validate_case(cls, value: object) -> dict[str, object]:
        if type(value) is not AccessTaskCase or not value._validated:
            raise ValueError("case must be an exact validated AccessTaskCase")
        return {
            "case_id": value.case_id,
            "world_id": value.world_id,
            "domain": value.domain,
            "seed": value.seed,
            "target_request_id": value.target_request_id,
            "target_employee_id": value.target_employee_id,
            "target_application_id": value.target_application_id,
            "required_preserved_entitlement_ids": list(value.required_preserved_entitlement_ids),
        }

    @field_validator("initial_world", mode="before")
    @classmethod
    def validate_world(cls, value: object) -> dict[str, object]:
        if type(value) is not WorldState:
            raise ValueError("initial_world must be an exact WorldState")
        fields = ("schema_version", "world_id", "domain", "seed", "data")
        return {field: getattr(value, field) for field in fields}

    @field_validator("baseline_entitlement_ids", "authority_refs", mode="before")
    @classmethod
    def validate_id_collections(cls, value: object, info: object) -> tuple[PathSafeId, ...]:
        if type(value) is not list:
            raise ValueError(f"{getattr(info, 'field_name', 'IDs')} must be an exact JSON array")
        field = getattr(info, "field_name", "IDs")
        return tuple(_id(item, field=field) for item in cast(list[object], value))

    @field_validator("baseline_entitlement_ids", "authority_refs")
    @classmethod
    def validate_sorted_id_collections(
        cls, value: tuple[PathSafeId, ...], info: object
    ) -> tuple[PathSafeId, ...]:
        return cast(
            tuple[PathSafeId, ...],
            _sorted_unique_ids(value, field=getattr(info, "field_name", "IDs")),
        )

    @model_validator(mode="after")
    def bind_fixture(self) -> Self:
        projected = parse_access_world(self.initial_world)
        if (
            self.case.world_id != self.initial_world.world_id
            or self.case.seed != self.seed
            or self.initial_world.seed != self.seed
            or self.case.domain != ACCESS_DOMAIN
        ):
            raise ValueError("fixture case and world identity must match")
        self.case.validate_state(projected)
        if self.baseline_entitlement_ids != projected.baseline_entitlement_ids:
            raise ValueError("fixture baseline entitlements must match projected state")
        if not {approval.authority_ref for approval in projected.approvals} <= set(
            self.authority_refs
        ):
            raise ValueError("fixture approval authority reference is not declared")
        return self

    @property
    def initial_state(self) -> AccessWorldState:
        return parse_access_world(self.initial_world)

    @property
    def projection_hash(self) -> str:
        return sha256_ref(
            {
                "fixture_id": self.fixture_id,
                "seed": self.seed,
                "case": {
                    "case_id": self.case.case_id,
                    "world_id": self.case.world_id,
                    "domain": self.case.domain,
                    "seed": self.case.seed,
                    "target_request_id": self.case.target_request_id,
                    "target_employee_id": self.case.target_employee_id,
                    "target_application_id": self.case.target_application_id,
                    "required_preserved_entitlement_ids": list(
                        self.case.required_preserved_entitlement_ids
                    ),
                },
                "initial_world": {
                    "schema_version": self.initial_world.schema_version,
                    "world_id": self.initial_world.world_id,
                    "domain": self.initial_world.domain,
                    "seed": self.initial_world.seed,
                    "data": _access_state_fields(self.initial_state),
                },
                "baseline_entitlement_ids": list(self.baseline_entitlement_ids),
                "authority_refs": list(self.authority_refs),
            }
        )
