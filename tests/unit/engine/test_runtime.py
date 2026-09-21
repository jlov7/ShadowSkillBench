from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal, cast

import pytest

import shadowskillbench.engine.runtime as runtime
from shadowskillbench.engine import make_state_event_from_cursor
from shadowskillbench.engine.models import (
    ActionCall,
    ActionResult,
    EventCursor,
    JsonObject,
    StatePatch,
    TransitionProposal,
    WorldState,
    hash_action,
    hash_state,
)
from shadowskillbench.engine.runtime import (
    EnvironmentInvariantFailure,
    PatchApplicationError,
    apply_state_data_patches,
    apply_world_state_patches,
    diff_state_data,
    diff_world_state,
    execute_action,
)


def state(data: JsonObject | None = None, *, seed: int = 7) -> WorldState:
    return WorldState(
        schema_version="1.0",
        world_id="world_1",
        domain="domain_1",
        seed=seed,
        data={"items": [1, 2], "stable": True} if data is None else data,
    )


def call(arguments: JsonObject | None = None) -> ActionCall:
    return ActionCall(
        action_id="action_1",
        tool_name="tool_1",
        arguments={"value": 1} if arguments is None else arguments,
    )


def cursor(
    index: int = 0, parent_id: str | None = None, parent_hash: str | None = None
) -> EventCursor:
    return EventCursor(
        episode_id="episode_1",
        next_index=index,
        parent_event_id=parent_id,
        parent_event_hash=parent_hash,
    )


class Adapter:
    def __init__(self, apply: Callable[[WorldState, ActionCall], TransitionProposal]) -> None:
        self.calls = 0
        self._apply = apply
        self.received_state: WorldState | None = None
        self.received_call: ActionCall | None = None

    def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
        self.calls += 1
        self.received_state = state
        self.received_call = call
        return self._apply(state, call)


def success(next_state: WorldState, observation: JsonObject | None = None) -> TransitionProposal:
    return TransitionProposal(
        local_status="success",
        next_state=next_state,
        observation={"seen": True} if observation is None else observation,
    )


def test_diff_apply_is_exact_sorted_escaped_and_non_aliasing() -> None:
    before: JsonObject = {
        "z": 1,
        "a": {"~/": 1, "gone": None},
        "equal_length": [1, True],
        "length_change": [1],
    }
    after: JsonObject = {
        "z": 1.0,
        "a": {"~/": 2, "added": None},
        "equal_length": [1, 2],
        "length_change": [1, 2],
    }

    delta = diff_state_data(before, after)
    rebuilt = apply_state_data_patches(before, delta)

    assert rebuilt == after
    assert before["a"] == {"~/": 1, "gone": None}
    assert tuple(patch.path for patch in delta) == tuple(sorted(patch.path for patch in delta))
    assert any(patch.path == "/a/~0~1" for patch in delta)
    assert any(patch.path == "/length_change" and patch.op == "replace" for patch in delta)
    assert any(patch.path == "/z" and patch.op == "replace" for patch in delta)
    assert diff_state_data({"value": -0.0}, {"value": 0.0}) == ()


def test_apply_rejects_stale_overlap_bad_arrays_and_invalid_root() -> None:
    before: JsonObject = {"array": [1, 2], "value": 1}
    stale = StatePatch(
        op="replace",
        path="/value",
        before_present=True,
        before=2,
        after_present=True,
        after=3,
    )
    overlap = StatePatch(
        op="replace",
        path="/array",
        before_present=True,
        before=[1, 2],
        after_present=True,
        after=[1, 2, 3],
    )
    child = StatePatch(
        op="replace",
        path="/array/0",
        before_present=True,
        before=1,
        after_present=True,
        after=2,
    )
    array_add = StatePatch(
        op="add",
        path="/array/2",
        before_present=False,
        after_present=True,
        after=3,
    )
    root_add = StatePatch(
        op="add",
        path="",
        before_present=False,
        after_present=True,
        after={},
    )
    noncanonical_index = StatePatch(
        op="replace",
        path="/array/01",
        before_present=True,
        before=2,
        after_present=True,
        after=3,
    )
    append_index = StatePatch(
        op="replace",
        path="/array/-",
        before_present=True,
        before=2,
        after_present=True,
        after=3,
    )

    for patches in (
        (stale,),
        (overlap, child),
        (array_add,),
        (root_add,),
        (noncanonical_index,),
        (append_index,),
    ):
        with pytest.raises(PatchApplicationError):
            apply_state_data_patches(before, patches)

    root_replace = StatePatch(
        op="replace",
        path="",
        before_present=True,
        before=before,
        after_present=True,
        after={"root": True},
    )
    assert apply_state_data_patches(before, (root_replace,)) == {"root": True}

    remove_null = StatePatch(
        op="remove",
        path="/nullable",
        before_present=True,
        before=None,
        after_present=False,
    )
    assert apply_state_data_patches({"nullable": None}, (remove_null,)) == {}


def test_apply_rejects_oversized_canonical_array_index_with_typed_error() -> None:
    oversized_index = "9" * 5000
    patch = StatePatch(
        op="replace",
        path=f"/array/{oversized_index}",
        before_present=True,
        before=1,
        after_present=True,
        after=2,
    )

    with pytest.raises(PatchApplicationError) as error:
        apply_state_data_patches({"array": [1]}, (patch,))

    assert error.value.code == "PATCH_PATH_INVALID"


def test_diff_rejects_deep_nested_json_with_typed_error() -> None:
    nested: JsonObject = {}
    current = nested
    for _ in range(2000):
        child: JsonObject = {}
        current["next"] = child
        current = child

    with pytest.raises(PatchApplicationError) as error:
        diff_state_data(nested, nested)

    assert error.value.code == "PATCH_INVALID"


def test_world_state_deltas_require_identity_and_round_trip() -> None:
    before = state({"a": 1})
    after = state({"a": 2})
    delta = diff_world_state(before, after)

    assert apply_world_state_patches(before, delta) == after
    with pytest.raises(PatchApplicationError):
        diff_world_state(before, state({"a": 2}, seed=8))


def test_success_change_and_noop_emit_one_cursor_bound_event() -> None:
    change_adapter = Adapter(
        lambda _state, _call: success(state({"items": [1, 2, 3], "stable": True}))
    )
    changed = execute_action(state(), call(), change_adapter, cursor=cursor())

    assert change_adapter.calls == 1
    assert changed.event.event_id == "event_episode_1_000000"
    assert changed.event.parent_event_id is None
    assert changed.event.delta
    assert changed.cursor.next_index == 1
    assert changed.cursor.parent_event_id == changed.event.event_id
    assert changed.cursor.parent_event_hash == changed.event.event_hash
    assert changed.state.data == {"items": [1, 2, 3], "stable": True}

    noop_adapter = Adapter(lambda received_state, _call: success(received_state))
    noop = execute_action(changed.state, call(), noop_adapter, cursor=changed.cursor)
    assert noop.event.event_id == "event_episode_1_000001"
    assert noop.event.before_hash == noop.event.after_hash
    assert noop.event.delta == ()


def test_cursor_event_factory_consumes_parent_without_a_parent_event() -> None:
    parent_hash = "sha256:" + "1" * 64
    owned_cursor = cursor(1, "event_episode_1_000000", parent_hash)
    before = state()
    action = call()
    result = ActionResult(local_status="success", observation={})

    event = make_state_event_from_cursor(
        cursor=owned_cursor,
        action=action,
        result=result,
        before_state=before,
        after_state=before,
        delta=(),
    )

    assert event.parent_event_id == owned_cursor.parent_event_id
    assert event.parent_event_hash == owned_cursor.parent_event_hash
    assert event.action_hash == hash_action(action)
    with pytest.raises(ValueError, match="EventCursor is incomplete"):
        make_state_event_from_cursor(
            cursor=EventCursor.model_construct(),
            action=action,
            result=result,
            before_state=before,
            after_state=before,
            delta=(),
        )


@pytest.mark.parametrize("status", ["failure", "clarify"])
def test_failure_and_clarify_preserve_state_and_emit_empty_event(
    status: Literal["failure", "clarify"],
) -> None:
    adapter = Adapter(
        lambda _state, _call: TransitionProposal(
            local_status=status,
            next_state=None,
            observation={"reason": status},
            error_code="INVALID_ACTION" if status == "failure" else "CLARIFICATION_REQUIRED",
        )
    )
    before = state()
    step = execute_action(before, call(), adapter, cursor=cursor())

    assert step.state == before
    assert step.event.before_hash == step.event.after_hash == hash_state(before)
    assert step.event.delta == ()
    assert step.result.local_status == status


def test_runtime_uses_private_callback_inputs_and_rejects_persistent_mutation() -> None:
    before = state()
    original_call = call()

    def mutate(received_state: WorldState, received_call: ActionCall) -> TransitionProposal:
        cast(list[object], received_state.data["items"]).append(9)
        received_call.arguments["value"] = 2
        return success(state())

    adapter = Adapter(mutate)
    with pytest.raises(EnvironmentInvariantFailure) as error:
        execute_action(before, original_call, adapter, cursor=cursor())

    assert error.value.code == "CALLBACK_INPUT_MUTATION"
    assert adapter.calls == 1
    assert before.data == {"items": [1, 2], "stable": True}
    assert original_call.arguments == {"value": 1}
    assert adapter.received_state is not before
    assert adapter.received_call is not original_call

    action_only = Adapter(
        lambda _state, received_call: (
            received_call.arguments.__setitem__("value", 2) or success(state())
        )
    )
    with pytest.raises(EnvironmentInvariantFailure) as action_error:
        execute_action(before, original_call, action_only, cursor=cursor())
    assert action_error.value.code == "CALLBACK_INPUT_MUTATION"
    assert original_call.arguments == {"value": 1}


def test_runtime_rejects_bad_proposals_identity_drift_and_wraps_exceptions() -> None:
    wrong_return = Adapter(lambda _state, _call: cast(TransitionProposal, object()))
    with pytest.raises(EnvironmentInvariantFailure) as wrong:
        execute_action(state(), call(), wrong_return, cursor=cursor())
    assert wrong.value.code == "PROPOSAL_INVALID"

    invalid_constructed = Adapter(lambda _state, _call: TransitionProposal.model_construct())
    with pytest.raises(EnvironmentInvariantFailure) as invalid:
        execute_action(state(), call(), invalid_constructed, cursor=cursor())
    assert invalid.value.code == "PROPOSAL_INVALID"

    class ProposalSubclass(TransitionProposal):
        pass

    subclassed = Adapter(
        lambda _state, _call: ProposalSubclass(
            local_status="success", next_state=state(), observation={}
        )
    )
    with pytest.raises(EnvironmentInvariantFailure) as invalid_subclass:
        execute_action(state(), call(), subclassed, cursor=cursor())
    assert invalid_subclass.value.code == "PROPOSAL_INVALID"

    drift = Adapter(lambda _state, _call: success(state(seed=8)))
    with pytest.raises(EnvironmentInvariantFailure) as changed_identity:
        execute_action(state(), call(), drift, cursor=cursor())
    assert changed_identity.value.code == "STATE_IDENTITY_DRIFT"

    exploding = Adapter(lambda _state, _call: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(EnvironmentInvariantFailure) as wrapped:
        execute_action(state(), call(), exploding, cursor=cursor())
    assert wrapped.value.code == "ADAPTER_EXCEPTION"
    assert isinstance(wrapped.value.__cause__, RuntimeError)

    interrupting = Adapter(lambda _state, _call: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        execute_action(state(), call(), interrupting, cursor=cursor())

    assert wrong_return.calls == invalid_constructed.calls == drift.calls == exploding.calls == 1


def test_runtime_refuses_raw_or_subclass_models_and_has_no_ambient_behavior() -> None:
    adapter = Adapter(lambda received_state, _call: success(received_state))
    with pytest.raises(EnvironmentInvariantFailure):
        execute_action(cast(WorldState, {}), call(), adapter, cursor=cursor())
    with pytest.raises(EnvironmentInvariantFailure):
        execute_action(state(), cast(ActionCall, {}), adapter, cursor=cursor())
    with pytest.raises(EnvironmentInvariantFailure):
        execute_action(state(), call(), adapter, cursor=cast(EventCursor, {}))

    source = inspect.getsource(runtime)
    assert all(
        forbidden not in source
        for forbidden in ("datetime", "time.", "uuid", "random", "provider", "pathlib", "open(")
    )
