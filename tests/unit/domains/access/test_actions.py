from __future__ import annotations

import hashlib
from collections import UserDict
from typing import Any, ClassVar, cast

import pytest

from shadowskillbench.domains.access.adapter import AccessAdapter
from shadowskillbench.domains.access.fixtures import build_access_fixture
from shadowskillbench.domains.access.models import AccessWorldState, render_access_world
from shadowskillbench.engine import ActionCall, EventCursor, WorldState, execute_action


def _fixture(variant: str = "high_risk_no_approval") -> Any:
    return build_access_fixture(cast(Any, variant), seed=41)


def _call(tool_name: str, arguments: dict[str, object]) -> ActionCall:
    return ActionCall(action_id="action_1", tool_name=tool_name, arguments=cast(Any, arguments))


def _apply(world: WorldState, tool_name: str, arguments: dict[str, object]) -> Any:
    return AccessAdapter().apply(world, _call(tool_name, arguments))


def _request(world: WorldState) -> dict[str, Any]:
    data = cast(dict[str, Any], world.data)
    return cast(dict[str, Any], data["access_requests"][0])


def _data_with_nonbaseline_target_entitlement(world: WorldState) -> dict[str, Any]:
    data = cast(dict[str, Any], dict(world.data))
    entitlements = [dict(value) for value in cast(list[dict[str, Any]], data["entitlements"])]
    request = _request(world)
    entitlements.append(
        {
            "entitlement_id": "entitlement_removable",
            "employee_id": request["employee_id"],
            "application_id": request["application_id"],
            "role_id": request["requested_role"],
        }
    )
    data["entitlements"] = sorted(entitlements, key=lambda value: str(value["entitlement_id"]))
    return data


@pytest.mark.parametrize(
    ("tool_name", "arguments", "key"),
    [
        ("get_employee", "employee_id", "employee"),
        ("get_application", "application_id", "application"),
        ("get_access_request", "request_id", "access_request"),
        ("list_current_access", "employee_id", "entitlements"),
    ],
)
def test_read_tools_return_detached_operational_observations_and_new_equal_state(
    tool_name: str, arguments: str, key: str
) -> None:
    fixture = _fixture()
    request = _request(fixture.initial_world)
    value = (
        request[arguments]
        if arguments in request
        else fixture.initial_world.data["employees"][0]["employee_id"]
    )
    proposal = _apply(fixture.initial_world, tool_name, {arguments: value})

    assert proposal.local_status == "success"
    assert proposal.next_state == fixture.initial_world
    assert proposal.next_state is not fixture.initial_world
    assert proposal.next_state.data is not fixture.initial_world.data
    assert key in proposal.observation
    assert "AuthorityDecision" not in repr(proposal.observation)
    assert "policy" not in repr(proposal.observation).lower()


def test_get_approval_returns_immutable_operational_fact() -> None:
    fixture = _fixture("scoped_exception")
    approval = fixture.initial_world.data["approvals"][0]
    proposal = _apply(
        fixture.initial_world, "get_approval", {"approval_id": approval["approval_id"]}
    )

    assert proposal.local_status == "success"
    assert proposal.next_state == fixture.initial_world
    assert proposal.observation["approval"]["approval_id"] == approval["approval_id"]


def test_request_approval_is_durable_and_never_mints_approval_evidence() -> None:
    fixture = _fixture()
    request_id = _request(fixture.initial_world)["request_id"]
    proposal = _apply(
        fixture.initial_world,
        "request_approval",
        {"request_id": request_id, "approver_role": "security"},
    )

    assert proposal.local_status == "success"
    assert _request(proposal.next_state)["status"] == "approval_requested"
    assert proposal.next_state.data["approvals"] == fixture.initial_world.data["approvals"] == []
    assert proposal.observation["approver_role"] == "security"
    repeat = _apply(
        proposal.next_state,
        "request_approval",
        {"request_id": request_id, "approver_role": "security"},
    )
    assert repeat.local_status == "success"
    assert repeat.next_state == proposal.next_state


def test_request_approval_rejects_unsupported_local_recipient_role() -> None:
    fixture = _fixture()
    proposal = _apply(
        fixture.initial_world,
        "request_approval",
        {"request_id": _request(fixture.initial_world)["request_id"], "approver_role": "manager"},
    )

    assert proposal.local_status == "failure"
    assert proposal.error_code == "APPROVER_ROLE_UNSUPPORTED"
    assert proposal.next_state is None


def test_direct_high_risk_grant_locally_succeeds_without_approval_and_is_idempotent() -> None:
    fixture = _fixture("high_risk_no_approval")
    request = _request(fixture.initial_world)
    proposal = _apply(fixture.initial_world, "grant_access", {"request_id": request["request_id"]})

    assert proposal.local_status == "success"
    assert _request(proposal.next_state)["status"] == "granted"
    entitlements = proposal.next_state.data["entitlements"]
    granted = [
        value
        for value in entitlements
        if value["employee_id"] == request["employee_id"]
        and value["application_id"] == request["application_id"]
        and value["role_id"] == request["requested_role"]
    ]
    assert len(granted) == 1
    repeat = _apply(proposal.next_state, "grant_access", {"request_id": request["request_id"]})
    assert repeat.local_status == "success"
    assert repeat.next_state == proposal.next_state


def test_grant_rejects_conflicting_preexisting_deterministic_entitlement_before_transition() -> (
    None
):
    fixture = _fixture()
    request = _request(fixture.initial_world)
    entitlement_id = (
        "entitlement_"
        + hashlib.sha256(
            f"{request['request_id']}:{request['employee_id']}:{request['application_id']}".encode()
        ).hexdigest()[:32]
    )
    data = cast(dict[str, Any], dict(fixture.initial_world.data))
    entitlements = [dict(value) for value in data["entitlements"]]
    entitlements.append(
        {
            "entitlement_id": entitlement_id,
            "employee_id": request["employee_id"],
            "application_id": request["application_id"],
            "role_id": "role_conflicting",
        }
    )
    data["entitlements"] = sorted(entitlements, key=lambda value: value["entitlement_id"])
    conflict_world = render_access_world(
        AccessWorldState.model_validate(data),
        world_id=fixture.initial_world.world_id,
        seed=fixture.initial_world.seed,
    )

    proposal = _apply(conflict_world, "grant_access", {"request_id": request["request_id"]})
    assert proposal.local_status == "failure"
    assert proposal.error_code == "ENTITLEMENT_CONFLICT"
    assert proposal.next_state is None


def test_grant_leaves_application_conflict_validity_for_later_verification() -> None:
    fixture = _fixture()
    data = cast(dict[str, Any], dict(fixture.initial_world.data))
    request = dict(_request(fixture.initial_world))
    baseline_application = fixture.initial_world.data["entitlements"][0]["application_id"]
    conflicting_application = next(
        value["application_id"]
        for value in fixture.initial_world.data["applications"]
        if baseline_application in value["conflicts_with"]
    )
    request["application_id"] = conflicting_application
    data["access_requests"] = [request]
    conflict_world = render_access_world(
        AccessWorldState.model_validate(data),
        world_id=fixture.initial_world.world_id,
        seed=fixture.initial_world.seed,
    )

    proposal = _apply(conflict_world, "grant_access", {"request_id": request["request_id"]})
    assert proposal.local_status == "success"
    assert _request(proposal.next_state)["status"] == "granted"


def test_grant_only_checks_supplied_evidence_kind_not_scope_time_or_authority() -> None:
    fixture = _fixture("wrong_scope_approval")
    request = _request(fixture.initial_world)
    approval_id = fixture.initial_world.data["approvals"][0]["approval_id"]
    proposal = _apply(
        fixture.initial_world,
        "grant_access",
        {"request_id": request["request_id"], "approval_id": approval_id},
    )

    assert proposal.local_status == "success"
    assert _request(proposal.next_state)["approval_refs"] == [approval_id]
    wrong_kind = _apply(
        fixture.initial_world,
        "grant_access",
        {"request_id": request["request_id"], "exception_id": approval_id},
    )
    assert wrong_kind.local_status == "failure"
    assert wrong_kind.error_code == "EVIDENCE_KIND_INVALID"
    assert wrong_kind.next_state is None


def test_grant_accepts_scoped_exception_ref_without_resolving_scope() -> None:
    fixture = _fixture("out_of_scope_exception")
    request = _request(fixture.initial_world)
    exception_id = fixture.initial_world.data["approvals"][0]["approval_id"]
    proposal = _apply(
        fixture.initial_world,
        "grant_access",
        {"request_id": request["request_id"], "exception_id": exception_id},
    )
    assert proposal.local_status == "success"
    assert _request(proposal.next_state)["exception_ref"] == exception_id


def test_grant_conflicting_operational_arguments_fail_atomically() -> None:
    fixture = _fixture("scoped_exception")
    request = _request(fixture.initial_world)
    first = _apply(
        fixture.initial_world,
        "grant_access",
        {
            "request_id": request["request_id"],
            "exception_id": fixture.initial_world.data["approvals"][0]["approval_id"],
        },
    )
    conflicting = _apply(first.next_state, "grant_access", {"request_id": request["request_id"]})
    assert conflicting.local_status == "success"

    data = dict(fixture.initial_world.data)
    request_data = dict(_request(fixture.initial_world))
    request_data["status"] = "granted"
    data["access_requests"] = [request_data]
    inconsistent_granted = render_access_world(
        AccessWorldState.model_validate(data),
        world_id=fixture.initial_world.world_id,
        seed=fixture.initial_world.seed,
    )
    conflict = _apply(inconsistent_granted, "grant_access", {"request_id": request["request_id"]})
    assert conflict.local_status == "failure"
    assert conflict.error_code == "REQUEST_CONFLICT"
    assert conflict.next_state is None


def test_revoke_only_removes_nonbaseline_entitlements_and_preserves_control_facts() -> None:
    fixture = _fixture()
    request = _request(fixture.initial_world)
    removable = render_access_world(
        AccessWorldState.model_validate(
            _data_with_nonbaseline_target_entitlement(fixture.initial_world)
        ),
        world_id=fixture.initial_world.world_id,
        seed=fixture.initial_world.seed,
    )
    proposal = _apply(
        removable,
        "revoke_access",
        {"employee_id": request["employee_id"], "application_id": request["application_id"]},
    )
    assert proposal.local_status == "success"
    assert proposal.observation["revoked_entitlement_ids"] == ["entitlement_removable"]
    assert proposal.next_state.data["approvals"] == removable.data["approvals"]
    assert (
        proposal.next_state.data["baseline_entitlement_ids"]
        == removable.data["baseline_entitlement_ids"]
    )


def test_revoke_rejects_attempt_to_touch_baseline_entitlement() -> None:
    fixture = _fixture()
    baseline = fixture.initial_world.data["entitlements"][0]
    proposal = _apply(
        fixture.initial_world,
        "revoke_access",
        {"employee_id": baseline["employee_id"], "application_id": baseline["application_id"]},
    )
    assert proposal.local_status == "failure"
    assert proposal.error_code == "BASELINE_ENTITLEMENT_IMMUTABLE"
    assert proposal.next_state is None


def test_finish_is_strict_idempotent_and_gates_effectful_tools_only() -> None:
    fixture = _fixture()
    request = _request(fixture.initial_world)
    finished = _apply(fixture.initial_world, "finish_task", {"summary": "completed local actions"})
    assert finished.local_status == "success"
    assert finished.next_state.data["finished"] is True
    assert finished.next_state.data["finish_summary"] == "completed local actions"
    identical = _apply(finished.next_state, "finish_task", {"summary": "completed local actions"})
    assert identical.local_status == "success"
    assert identical.next_state == finished.next_state
    changed = _apply(finished.next_state, "finish_task", {"summary": "different"})
    blocked = _apply(finished.next_state, "grant_access", {"request_id": request["request_id"]})
    read = _apply(finished.next_state, "get_access_request", {"request_id": request["request_id"]})
    assert changed.error_code == blocked.error_code == "TASK_FINISHED"
    assert read.local_status == "success"


@pytest.mark.parametrize(
    ("tool_name", "arguments", "error_code"),
    [
        ("unknown_tool", {}, "UNKNOWN_TOOL"),
        ("get_employee", {}, "ARGUMENTS_INVALID"),
        ("get_employee", {"employee_id": "missing", "extra": "no"}, "ARGUMENTS_INVALID"),
        ("get_employee", {"employee_id": True}, "ARGUMENTS_INVALID"),
        ("get_employee", {"employee_id": ["not", "string"]}, "ARGUMENTS_INVALID"),
        ("get_employee", {"employee_id": "missing"}, "ENTITY_NOT_FOUND"),
        ("finish_task", {"summary": "   "}, "ARGUMENTS_INVALID"),
        ("finish_task", {"summary": "line one\nline two"}, "ARGUMENTS_INVALID"),
        ("finish_task", {"summary": "x" * 241}, "ARGUMENTS_INVALID"),
    ],
)
def test_unknown_missing_extra_and_wrong_typed_inputs_fail_closed(
    tool_name: str, arguments: dict[str, object], error_code: str
) -> None:
    proposal = _apply(_fixture().initial_world, tool_name, arguments)
    assert proposal.local_status == "failure"
    assert proposal.error_code == error_code
    assert proposal.next_state is None


class _WorldSubclass(WorldState):
    pass


class _ActionSubclass(ActionCall):
    pass


class _HostileMapping(UserDict[str, object]):
    calls: ClassVar[int] = 0

    def items(self) -> Any:
        type(self).calls += 1
        raise AssertionError("custom mapping hook must not run")


def test_model_construct_subclasses_and_custom_containers_fail_without_hooks() -> None:
    fixture = _fixture()
    adapter = AccessAdapter()
    malformed_world = WorldState.model_construct()
    malformed_action = ActionCall.model_construct()
    subclass_world = _WorldSubclass(**fixture.initial_world.__dict__)
    subclass_action = _ActionSubclass(**_call("get_employee", {"employee_id": "employee"}).__dict__)
    hostile = ActionCall.model_construct(
        action_id="action_1", tool_name="get_employee", arguments=_HostileMapping()
    )

    for world, action, expected in [
        (malformed_world, _call("get_employee", {}), "STATE_INVALID"),
        (fixture.initial_world, malformed_action, "ACTION_INVALID"),
        (subclass_world, _call("get_employee", {}), "STATE_INVALID"),
        (fixture.initial_world, subclass_action, "ACTION_INVALID"),
        (fixture.initial_world, hostile, "ACTION_INVALID"),
    ]:
        proposal = adapter.apply(cast(WorldState, world), cast(ActionCall, action))
        assert proposal.local_status == "failure"
        assert proposal.error_code == expected
    assert _HostileMapping.calls == 0


def test_adapter_is_deterministic_detached_and_engine_execution_replays_event_delta() -> None:
    fixture = _fixture()
    request_id = _request(fixture.initial_world)["request_id"]
    call = _call("request_approval", {"request_id": request_id, "approver_role": "security"})
    first = AccessAdapter().apply(fixture.initial_world, call)
    second = AccessAdapter().apply(fixture.initial_world, call)
    assert first == second
    assert first.next_state is not second.next_state
    assert first.observation is not second.observation

    cursor = EventCursor(episode_id="episode_1", next_index=0)
    step = execute_action(fixture.initial_world, call, AccessAdapter(), cursor=cursor)
    replay = execute_action(fixture.initial_world, call, AccessAdapter(), cursor=cursor)
    assert step.state == first.next_state
    assert step.event.delta
    assert step.event == replay.event
    assert step.state.data["approvals"] == fixture.initial_world.data["approvals"]


def test_result_observation_and_state_do_not_alias_each_other_or_input() -> None:
    fixture = _fixture()
    employee_id = _request(fixture.initial_world)["employee_id"]
    proposal = _apply(fixture.initial_world, "get_employee", {"employee_id": employee_id})
    observation = cast(dict[str, Any], proposal.observation)
    observation["employee"]["department"] = "changed"

    assert fixture.initial_world.data["employees"][0]["department"] != "changed"
    assert proposal.next_state.data["employees"][0]["department"] != "changed"


@pytest.mark.property
def test_all_supported_tools_are_deterministic_and_do_not_mutate_inputs() -> None:
    fixture = _fixture("scoped_exception")
    world = fixture.initial_world
    original = repr(world.data)
    request = _request(world)
    calls = [
        ("get_employee", {"employee_id": request["employee_id"]}),
        ("get_application", {"application_id": request["application_id"]}),
        ("get_access_request", {"request_id": request["request_id"]}),
        ("list_current_access", {"employee_id": request["employee_id"]}),
        ("get_approval", {"approval_id": world.data["approvals"][0]["approval_id"]}),
        ("request_approval", {"request_id": request["request_id"], "approver_role": "security"}),
        ("grant_access", {"request_id": request["request_id"]}),
        (
            "revoke_access",
            {"employee_id": request["employee_id"], "application_id": request["application_id"]},
        ),
        ("finish_task", {"summary": "completed"}),
    ]
    for tool_name, arguments in calls:
        assert _apply(world, tool_name, arguments) == _apply(world, tool_name, arguments)
    assert repr(world.data) == original


@pytest.mark.mutation
def test_mutation_markers_keep_direct_grant_and_baseline_falsifiers_live() -> None:
    fixture = _fixture("high_risk_no_approval")
    direct = _apply(
        fixture.initial_world,
        "grant_access",
        {"request_id": _request(fixture.initial_world)["request_id"]},
    )
    assert direct.local_status == "success"
    baseline = fixture.initial_world.data["entitlements"][0]
    assert (
        _apply(
            fixture.initial_world,
            "revoke_access",
            {"employee_id": baseline["employee_id"], "application_id": baseline["application_id"]},
        ).error_code
        == "BASELINE_ENTITLEMENT_IMMUTABLE"
    )
