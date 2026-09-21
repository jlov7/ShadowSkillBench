from __future__ import annotations

import hashlib
import re
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.domains.access.models import (
    ACCESS_DOMAIN,
    AccessRequest,
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
from shadowskillbench.engine.models import (
    ActionCall,
    JsonObject,
    JsonValue,
    TransitionProposal,
    WorldState,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_READ_TOOLS = frozenset(
    {
        "get_employee",
        "get_application",
        "get_access_request",
        "list_current_access",
        "get_approval",
    }
)
_EFFECT_TOOLS = frozenset({"request_approval", "grant_access", "revoke_access", "finish_task"})
_ALL_TOOLS = _READ_TOOLS | _EFFECT_TOOLS


class AccessAdapter:
    def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
        world = self._world(state)
        if world is None:
            return _failure("STATE_INVALID")
        action = self._action(call)
        if action is None:
            return _failure("ACTION_INVALID")
        if action.tool_name not in _ALL_TOOLS:
            return _failure("UNKNOWN_TOOL")

        parsed = parse_access_world(world)
        baseline = _immutable_facts(world)
        if parsed.finished and action.tool_name in _EFFECT_TOOLS:
            if action.tool_name != "finish_task":
                return _failure("TASK_FINISHED")
            summary = _summary(action.arguments)
            if summary is None:
                return _failure("ARGUMENTS_INVALID")
            if summary != parsed.finish_summary:
                return _failure("TASK_FINISHED")
            return self._success(parsed, world, baseline, {"finished": True, "summary": summary})

        if action.tool_name == "get_employee":
            employee_id = _ids(action.arguments, required=("employee_id",))
            if employee_id is None:
                return _failure("ARGUMENTS_INVALID")
            employee = _by_id(parsed.employees, "employee_id", employee_id["employee_id"])
            if employee is None:
                return _failure("ENTITY_NOT_FOUND")
            return self._success(parsed, world, baseline, {"employee": _employee(employee)})
        if action.tool_name == "get_application":
            application_id = _ids(action.arguments, required=("application_id",))
            if application_id is None:
                return _failure("ARGUMENTS_INVALID")
            application = _by_id(
                parsed.applications, "application_id", application_id["application_id"]
            )
            if application is None:
                return _failure("ENTITY_NOT_FOUND")
            return self._success(
                parsed, world, baseline, {"application": _application(application)}
            )
        if action.tool_name == "get_access_request":
            request_id = _ids(action.arguments, required=("request_id",))
            if request_id is None:
                return _failure("ARGUMENTS_INVALID")
            request = _by_id(parsed.access_requests, "request_id", request_id["request_id"])
            if request is None:
                return _failure("ENTITY_NOT_FOUND")
            return self._success(parsed, world, baseline, {"access_request": _request(request)})
        if action.tool_name == "list_current_access":
            employee_id = _ids(action.arguments, required=("employee_id",))
            if employee_id is None:
                return _failure("ARGUMENTS_INVALID")
            if _by_id(parsed.employees, "employee_id", employee_id["employee_id"]) is None:
                return _failure("ENTITY_NOT_FOUND")
            entitlements = [
                _entitlement(value)
                for value in parsed.entitlements
                if value.employee_id == employee_id["employee_id"]
            ]
            return self._success(
                parsed,
                world,
                baseline,
                {"employee_id": employee_id["employee_id"], "entitlements": entitlements},
            )
        if action.tool_name == "get_approval":
            approval_id = _ids(action.arguments, required=("approval_id",))
            if approval_id is None:
                return _failure("ARGUMENTS_INVALID")
            approval = _by_id(parsed.approvals, "approval_id", approval_id["approval_id"])
            if approval is None:
                return _failure("ENTITY_NOT_FOUND")
            return self._success(parsed, world, baseline, {"approval": _approval(approval)})
        if action.tool_name == "request_approval":
            return self._request_approval(parsed, world, baseline, action.arguments)
        if action.tool_name == "grant_access":
            return self._grant_access(parsed, world, baseline, action.arguments)
        if action.tool_name == "revoke_access":
            return self._revoke_access(parsed, world, baseline, action.arguments)
        return self._finish_task(parsed, world, baseline, action.arguments)

    @staticmethod
    def _world(value: object) -> WorldState | None:
        if type(value) is not WorldState:
            return None
        try:
            fields = ("schema_version", "world_id", "domain", "seed", "data")
            world = WorldState.model_validate({field: getattr(value, field) for field in fields})
            if world.domain != ACCESS_DOMAIN:
                return None
            parse_access_world(world)
            return world
        except Exception:
            return None

    @staticmethod
    def _action(value: object) -> ActionCall | None:
        if type(value) is not ActionCall:
            return None
        try:
            fields = ("action_id", "tool_name", "arguments")
            return ActionCall.model_validate({field: getattr(value, field) for field in fields})
        except Exception:
            return None

    @staticmethod
    def _success(
        state: AccessWorldState,
        world: WorldState,
        immutable_facts: bytes,
        observation: object,
    ) -> TransitionProposal:
        next_state = render_access_world(state, world_id=world.world_id, seed=world.seed)
        if _immutable_facts(next_state) != immutable_facts:
            raise ValueError("access immutable facts changed during local transition")
        return TransitionProposal(
            local_status="success", next_state=next_state, observation=cast(JsonObject, observation)
        )

    def _request_approval(
        self,
        state: AccessWorldState,
        world: WorldState,
        immutable_facts: bytes,
        arguments: JsonObject,
    ) -> TransitionProposal:
        values = _ids(arguments, required=("request_id", "approver_role"))
        if values is None:
            return _failure("ARGUMENTS_INVALID")
        if values["approver_role"] != "security":
            return _failure("APPROVER_ROLE_UNSUPPORTED")
        request = _by_id(state.access_requests, "request_id", values["request_id"])
        if request is None:
            return _failure("ENTITY_NOT_FOUND")
        if request.status not in {RequestStatus.PENDING, RequestStatus.APPROVAL_REQUESTED}:
            return _failure("REQUEST_STATE_CONFLICT")
        data = _data(state, world)
        requests = cast(list[dict[str, JsonValue]], data["access_requests"])
        for candidate in requests:
            if candidate["request_id"] == values["request_id"]:
                candidate["status"] = RequestStatus.APPROVAL_REQUESTED.value
                break
        next_access = AccessWorldState.model_validate(data)
        return self._success(
            next_access,
            world,
            immutable_facts,
            {
                "request_id": values["request_id"],
                "approver_role": values["approver_role"],
                "status": RequestStatus.APPROVAL_REQUESTED.value,
            },
        )

    def _grant_access(
        self,
        state: AccessWorldState,
        world: WorldState,
        immutable_facts: bytes,
        arguments: JsonObject,
    ) -> TransitionProposal:
        values = _ids(arguments, required=("request_id",), optional=("approval_id", "exception_id"))
        if values is None:
            return _failure("ARGUMENTS_INVALID")
        request = _by_id(state.access_requests, "request_id", values["request_id"])
        if request is None:
            return _failure("ENTITY_NOT_FOUND")
        employee = _by_id(state.employees, "employee_id", request.employee_id)
        application = _by_id(state.applications, "application_id", request.application_id)
        if employee is None or application is None:
            return _failure("ENTITY_NOT_FOUND")
        if not employee.active:
            return _failure("EMPLOYEE_INACTIVE")
        approval_id = values.get("approval_id")
        exception_id = values.get("exception_id")
        if approval_id is not None:
            approval = _by_id(state.approvals, "approval_id", approval_id)
            if approval is None:
                return _failure("ENTITY_NOT_FOUND")
            if approval.kind is not ApprovalKind.APPROVAL:
                return _failure("EVIDENCE_KIND_INVALID")
        if exception_id is not None:
            exception = _by_id(state.approvals, "approval_id", exception_id)
            if exception is None:
                return _failure("ENTITY_NOT_FOUND")
            if exception.kind is not ApprovalKind.SCOPED_EXCEPTION:
                return _failure("EVIDENCE_KIND_INVALID")

        entitlement_id = _entitlement_id(
            request.request_id, request.employee_id, request.application_id
        )
        present = _by_id(state.entitlements, "entitlement_id", entitlement_id)
        if present is not None and (
            present.employee_id != request.employee_id
            or present.application_id != request.application_id
            or present.role_id != request.requested_role
        ):
            return _failure("ENTITLEMENT_CONFLICT")
        if request.status is RequestStatus.GRANTED:
            if (
                present is None
                or (approval_id is not None and approval_id not in request.approval_refs)
                or (exception_id is not None and exception_id != request.exception_ref)
            ):
                return _failure("REQUEST_CONFLICT")
            return self._success(
                state,
                world,
                immutable_facts,
                {
                    "request_id": request.request_id,
                    "entitlement_id": entitlement_id,
                    "status": "granted",
                },
            )
        if request.status not in {RequestStatus.PENDING, RequestStatus.APPROVAL_REQUESTED}:
            return _failure("REQUEST_STATE_CONFLICT")
        if request.exception_ref is not None and exception_id not in {None, request.exception_ref}:
            return _failure("REQUEST_CONFLICT")

        data = _data(state, world)
        requests = cast(list[dict[str, JsonValue]], data["access_requests"])
        for candidate in requests:
            if candidate["request_id"] == request.request_id:
                refs = cast(list[str], candidate["approval_refs"])
                if approval_id is not None and approval_id not in refs:
                    refs.append(approval_id)
                    refs.sort()
                candidate["status"] = RequestStatus.GRANTED.value
                if exception_id is not None:
                    candidate["exception_ref"] = exception_id
                break
        entitlements = cast(list[dict[str, JsonValue]], data["entitlements"])
        if present is None:
            entitlements.append(
                {
                    "entitlement_id": entitlement_id,
                    "employee_id": request.employee_id,
                    "application_id": request.application_id,
                    "role_id": request.requested_role,
                }
            )
            entitlements.sort(key=lambda value: cast(str, value["entitlement_id"]))
        next_access = AccessWorldState.model_validate(data)
        return self._success(
            next_access,
            world,
            immutable_facts,
            {
                "request_id": request.request_id,
                "entitlement_id": entitlement_id,
                "status": "granted",
            },
        )

    def _revoke_access(
        self,
        state: AccessWorldState,
        world: WorldState,
        immutable_facts: bytes,
        arguments: JsonObject,
    ) -> TransitionProposal:
        values = _ids(arguments, required=("employee_id", "application_id"))
        if values is None:
            return _failure("ARGUMENTS_INVALID")
        if (
            _by_id(state.employees, "employee_id", values["employee_id"]) is None
            or _by_id(state.applications, "application_id", values["application_id"]) is None
        ):
            return _failure("ENTITY_NOT_FOUND")
        selected = [
            value
            for value in state.entitlements
            if value.employee_id == values["employee_id"]
            and value.application_id == values["application_id"]
        ]
        if any(value.entitlement_id in state.baseline_entitlement_ids for value in selected):
            return _failure("BASELINE_ENTITLEMENT_IMMUTABLE")
        removed_ids = [value.entitlement_id for value in selected]
        data = _data(state, world)
        data["entitlements"] = [
            value
            for value in cast(list[dict[str, JsonValue]], data["entitlements"])
            if value["entitlement_id"] not in removed_ids
        ]
        next_access = AccessWorldState.model_validate(data)
        return self._success(
            next_access,
            world,
            immutable_facts,
            {
                "employee_id": values["employee_id"],
                "application_id": values["application_id"],
                "revoked_entitlement_ids": removed_ids,
            },
        )

    def _finish_task(
        self,
        state: AccessWorldState,
        world: WorldState,
        immutable_facts: bytes,
        arguments: JsonObject,
    ) -> TransitionProposal:
        summary = _summary(arguments)
        if summary is None:
            return _failure("ARGUMENTS_INVALID")
        data = _data(state, world)
        data["finished"] = True
        data["finish_summary"] = summary
        next_access = AccessWorldState.model_validate(data)
        return self._success(
            next_access, world, immutable_facts, {"finished": True, "summary": summary}
        )


def _failure(error_code: str) -> TransitionProposal:
    return TransitionProposal(local_status="failure", observation={}, error_code=error_code)


def _ids(
    arguments: JsonObject, *, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> dict[str, str] | None:
    allowed = set(required) | set(optional)
    if set(arguments) != allowed and not (set(required) <= set(arguments) <= allowed):
        return None
    values: dict[str, str] = {}
    for name in (*required, *optional):
        if name not in arguments:
            continue
        value = arguments[name]
        if type(value) is not str or _ID.fullmatch(value) is None:
            return None
        values[name] = value
    return values


def _summary(arguments: JsonObject) -> str | None:
    if set(arguments) != {"summary"}:
        return None
    summary = arguments["summary"]
    if (
        type(summary) is not str
        or not summary.strip()
        or len(summary) > 240
        or "\n" in summary
        or "\r" in summary
    ):
        return None
    return summary


def _by_id[T](values: tuple[T, ...], field: str, identifier: str) -> T | None:
    for value in values:
        if getattr(value, field) == identifier:
            return value
    return None


def _data(state: AccessWorldState, world: WorldState) -> dict[str, JsonValue]:
    rendered = render_access_world(state, world_id=world.world_id, seed=world.seed)
    return cast(dict[str, JsonValue], rendered.data)


def _immutable_facts(world: WorldState) -> bytes:
    data = world.data
    approvals = data["approvals"]
    baseline_ids = set(cast(list[str], data["baseline_entitlement_ids"]))
    entitlements = [
        value
        for value in cast(list[dict[str, JsonValue]], data["entitlements"])
        if value["entitlement_id"] in baseline_ids
    ]
    return canonical_json_bytes({"approvals": approvals, "baseline_entitlements": entitlements})


def _entitlement_id(request_id: str, employee_id: str, application_id: str) -> str:
    digest = hashlib.sha256(f"{request_id}:{employee_id}:{application_id}".encode()).hexdigest()
    return f"entitlement_{digest[:32]}"


def _employee(value: Employee) -> JsonObject:
    return {
        "employee_id": value.employee_id,
        "employment_type": value.employment_type.value,
        "department": value.department,
        "role_id": value.role_id,
        "clearance_level": value.clearance_level.value,
        "manager_id": value.manager_id,
        "active": value.active,
    }


def _application(value: Application) -> JsonObject:
    return {
        "application_id": value.application_id,
        "risk_tier": value.risk_tier.value,
        "required_clearance": value.required_clearance.value,
        "conflicts_with": list(value.conflicts_with),
        "supports_role_derived_access": value.supports_role_derived_access,
    }


def _request(value: AccessRequest) -> JsonObject:
    return {
        "request_id": value.request_id,
        "employee_id": value.employee_id,
        "application_id": value.application_id,
        "requested_role": value.requested_role,
        "status": value.status.value,
        "approval_refs": list(value.approval_refs),
        "exception_ref": value.exception_ref,
    }


def _approval(value: Approval) -> JsonObject:
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


def _entitlement(value: Entitlement) -> JsonObject:
    return {
        "entitlement_id": value.entitlement_id,
        "employee_id": value.employee_id,
        "application_id": value.application_id,
        "role_id": value.role_id,
    }
