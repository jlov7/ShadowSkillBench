from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar, Self, cast

import pytest
from pydantic import BaseModel, ValidationError

import shadowskillbench.engine.models as engine_models
from shadowskillbench.core.hashing import CanonicalizationError, sha256_ref
from shadowskillbench.engine import JsonValue as ExportedJsonValue
from shadowskillbench.engine.models import (
    ActionCall,
    ActionResult,
    EventCursor,
    JsonValue,
    OutcomeVerdict,
    StateEvent,
    StatePatch,
    StateSnapshot,
    TaskCase,
    TerminalRecord,
    TransitionProposal,
    WorldState,
    action_projection,
    event_projection,
    hash_action,
    hash_event,
    hash_observation,
    hash_result,
    hash_state,
    hash_terminal,
    make_snapshot,
    make_state_event,
    make_terminal_record,
    observation_projection,
    result_projection,
    state_projection,
    terminal_projection,
)

HASH = sha256_ref({"fixture": "hash"})
OTHER_HASH = sha256_ref({"fixture": "other"})


class _DeepCopyReturnsOne:
    calls = 0

    def __deepcopy__(self, memo: object) -> int:
        type(self).calls += 1
        return 1


class _DeepCopyRaises:
    calls = 0

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        type(self).calls += 1
        raise RuntimeError("hostile deepcopy hook")


class _DeepCopyPydanticModel(BaseModel):
    calls: ClassVar[int] = 0
    value: int = 1

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        type(self).calls += 1
        raise RuntimeError("hostile Pydantic deepcopy hook")


class _WorldStateSubclass(WorldState):
    pass


class _ActionCallSubclass(ActionCall):
    pass


class _ActionResultSubclass(ActionResult):
    pass


class _StatePatchSubclass(StatePatch):
    pass


class _OutcomeVerdictSubclass(OutcomeVerdict):
    pass


class _HostileId(str):
    format_calls = 0
    equality_calls = 0

    def __format__(self, format_spec: str) -> str:
        type(self).format_calls += 1
        return super().__format__(format_spec)

    def __eq__(self, value: object) -> bool:
        type(self).equality_calls += 1
        return super().__eq__(value)


class _HostileIndex(int):
    format_calls = 0
    equality_calls = 0

    def __format__(self, format_spec: str) -> str:
        type(self).format_calls += 1
        return super().__format__(format_spec)

    def __eq__(self, value: object) -> bool:
        type(self).equality_calls += 1
        return super().__eq__(value)


class _HostileHashModel(BaseModel):
    calls: ClassVar[int] = 0
    value: str = HASH

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        type(self).calls += 1
        raise RuntimeError("hostile hash model_dump hook")


class _WorldStateDumpSubclass(WorldState):
    calls: ClassVar[int] = 0

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        type(self).calls += 1
        raise RuntimeError("hostile WorldState model_dump hook")


class _ActionCallDumpSubclass(ActionCall):
    calls: ClassVar[int] = 0

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        type(self).calls += 1
        raise RuntimeError("hostile ActionCall model_dump hook")


class _ActionResultDumpSubclass(ActionResult):
    calls: ClassVar[int] = 0

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        type(self).calls += 1
        raise RuntimeError("hostile ActionResult model_dump hook")


class _StateEventDumpSubclass(StateEvent):
    calls: ClassVar[int] = 0

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        type(self).calls += 1
        raise RuntimeError("hostile StateEvent model_dump hook")


class _TerminalRecordDumpSubclass(TerminalRecord):
    calls: ClassVar[int] = 0

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        type(self).calls += 1
        raise RuntimeError("hostile TerminalRecord model_dump hook")


def world(data: dict[str, JsonValue] | None = None, *, seed: int = 7) -> WorldState:
    return WorldState(
        schema_version="1.0",
        world_id="world_1",
        domain="domain_1",
        seed=seed,
        data={"items": [1, 2]} if data is None else data,
    )


def action(
    arguments: dict[str, JsonValue] | None = None, *, action_id: str = "action_1"
) -> ActionCall:
    return ActionCall(
        action_id=action_id,
        tool_name="tool_1",
        arguments={"value": 1} if arguments is None else arguments,
    )


def success_result(observation: dict[str, JsonValue] | None = None) -> ActionResult:
    return ActionResult(
        local_status="success", observation={"seen": 1} if observation is None else observation
    )


def patch(path: str = "/items", *, after: JsonValue | None = [1, 2, 3]) -> StatePatch:
    return StatePatch(
        op="replace",
        path=path,
        before_present=True,
        before=[1, 2],
        after_present=True,
        after=after,
    )


def event_zero() -> StateEvent:
    return make_state_event(
        episode_id="episode_1",
        index=0,
        action=action(),
        result=success_result(),
        before_state=world(),
        after_state=world({"items": [1, 2, 3]}),
        delta=(patch(),),
    )


def test_recursive_json_alias_is_importable_and_models_are_strict_and_frozen() -> None:
    case = TaskCase(
        case_id="case_1",
        domain="domain_1",
        world_id="world_1",
        seed=7,
        objective="complete the case",
        inputs={"nested": [True, None, {"value": "text"}]},
    )

    assert JsonValue is ExportedJsonValue
    assert case.inputs["nested"] == [True, None, {"value": "text"}]
    with pytest.raises(ValidationError):
        TaskCase(**{**case.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        WorldState(**{**world().model_dump(), "schema_version": "2.0"})
    with pytest.raises(ValidationError):
        world().world_id = "replacement"


@pytest.mark.parametrize("unsafe", ["", "with/slash", "with space", "dot.name", "é"])
def test_path_safe_identifiers_are_executable_validators(unsafe: str) -> None:
    with pytest.raises(ValidationError):
        TaskCase(
            case_id=unsafe,
            domain="domain_1",
            world_id="world_1",
            seed=0,
            objective="fixture",
            inputs={},
        )


def test_hash_error_code_and_nonnegative_index_aliases_are_validated() -> None:
    snapshot = make_snapshot(
        world(), episode_id="episode_1", snapshot_id="snapshot_1", observation={}
    )

    with pytest.raises(ValidationError):
        StateSnapshot(**{**snapshot.model_dump(), "state_hash": "sha256:ABC"})
    with pytest.raises(ValidationError):
        ActionResult(local_status="failure", observation={}, error_code="bad-code")
    with pytest.raises(ValidationError):
        EventCursor(episode_id="episode_1", next_index=-1)


def test_json_ingress_is_owned_but_nested_containers_remain_mutable() -> None:
    supplied: dict[str, JsonValue] = {
        "nested": cast(JsonValue, {"items": [1, 2]}),
    }
    state = world(supplied)
    original_hash = hash_state(state)
    supplied_nested = cast(dict[str, JsonValue], supplied["nested"])
    supplied_items = cast(list[JsonValue], supplied_nested["items"])
    supplied_items.append(3)

    assert state.data == {"nested": {"items": [1, 2]}}
    with pytest.raises(ValidationError):
        state.seed = 8
    state_nested = cast(dict[str, JsonValue], state.data["nested"])
    state_items = cast(list[JsonValue], state_nested["items"])
    state_items.append(4)
    assert hash_state(state) != original_hash


def test_canonical_json_rejects_invalid_values_at_model_ingress() -> None:
    cyclic: dict[str, JsonValue] = {}
    cyclic["self"] = cyclic

    for value in (
        {"nonfinite": float("nan")},
        {"surrogate": "\ud800"},
        {1: "non-string key"},
        cyclic,
    ):
        with pytest.raises(ValidationError):
            world(value)  # type: ignore[arg-type]


def test_json_ingress_rejects_hostile_deepcopy_hooks_without_executing_them() -> None:
    _DeepCopyReturnsOne.calls = 0
    _DeepCopyRaises.calls = 0
    _DeepCopyPydanticModel.calls = 0

    for value in (_DeepCopyReturnsOne(), _DeepCopyRaises(), _DeepCopyPydanticModel()):
        with pytest.raises(ValidationError):
            WorldState(
                schema_version="1.0",
                world_id="world_1",
                domain="domain_1",
                seed=7,
                data=cast(dict[str, JsonValue], {"hostile": value}),
            )

    assert _DeepCopyReturnsOne.calls == 0
    assert _DeepCopyRaises.calls == 0
    assert _DeepCopyPydanticModel.calls == 0


def test_nested_model_ingress_revalidates_post_construction_mutation_safely() -> None:
    _DeepCopyRaises.calls = 0
    clean_snapshot = make_snapshot(
        world(), episode_id="episode_1", snapshot_id="snapshot_1", observation={}
    )
    mutated_state = world()
    mutated_state.data["hostile"] = cast(JsonValue, _DeepCopyRaises())
    with pytest.raises(ValidationError):
        StateSnapshot(**{**clean_snapshot.model_dump(), "state": mutated_state})

    mutated_next_state = world()
    mutated_next_state.data["hostile"] = cast(JsonValue, _DeepCopyRaises())
    with pytest.raises(ValidationError):
        TransitionProposal(local_status="success", next_state=mutated_next_state, observation={})

    clean_event = event_zero()
    mutated_result = success_result()
    mutated_result.observation["hostile"] = cast(JsonValue, _DeepCopyRaises())
    with pytest.raises(ValidationError):
        StateEvent(**{**clean_event.model_dump(), "result": mutated_result})

    mutated_patch = patch()
    patch_after = cast(list[JsonValue], mutated_patch.after)
    patch_after.append(cast(JsonValue, _DeepCopyRaises()))
    with pytest.raises(ValidationError):
        StateEvent(**{**clean_event.model_dump(), "delta": (mutated_patch,)})

    clean_terminal = make_terminal_record(
        episode_id="episode_1",
        initial_state_hash=hash_state(world()),
        terminal_state_hash=hash_state(world()),
        event_root_hash=None,
        final_observation={},
        verdict=OutcomeVerdict(status="PASS", reason_code="COMPLETE", details={}),
    )
    mutated_verdict = OutcomeVerdict(status="PASS", reason_code="COMPLETE", details={})
    mutated_verdict.details["hostile"] = cast(JsonValue, _DeepCopyRaises())
    with pytest.raises(ValidationError):
        TerminalRecord(**{**clean_terminal.model_dump(), "verdict": mutated_verdict})

    assert _DeepCopyRaises.calls == 0


def test_factories_revalidate_nested_models_before_hashing_or_drafts() -> None:
    _DeepCopyRaises.calls = 0

    mutated_state = world()
    mutated_state.data["hostile"] = cast(JsonValue, _DeepCopyRaises())
    with pytest.raises((ValidationError, CanonicalizationError)):
        make_snapshot(
            mutated_state,
            episode_id="episode_1",
            snapshot_id="snapshot_1",
            observation={},
        )

    mutated_result = success_result()
    mutated_result.observation["hostile"] = cast(JsonValue, _DeepCopyRaises())
    with pytest.raises((ValidationError, CanonicalizationError)):
        make_state_event(
            episode_id="episode_1",
            index=0,
            action=action(),
            result=mutated_result,
            before_state=world(),
            after_state=world({"items": [1, 2, 3]}),
            delta=(patch(),),
        )

    mutated_patch = patch()
    patch_after = cast(list[JsonValue], mutated_patch.after)
    patch_after.append(cast(JsonValue, _DeepCopyRaises()))
    with pytest.raises((ValidationError, CanonicalizationError)):
        make_state_event(
            episode_id="episode_1",
            index=0,
            action=action(),
            result=success_result(),
            before_state=world(),
            after_state=world({"items": [1, 2, 3]}),
            delta=(mutated_patch,),
        )

    mutated_verdict = OutcomeVerdict(status="PASS", reason_code="COMPLETE", details={})
    mutated_verdict.details["hostile"] = cast(JsonValue, _DeepCopyRaises())
    with pytest.raises((ValidationError, CanonicalizationError)):
        make_terminal_record(
            episode_id="episode_1",
            initial_state_hash=hash_state(world()),
            terminal_state_hash=hash_state(world()),
            event_root_hash=None,
            final_observation={},
            verdict=mutated_verdict,
        )

    assert _DeepCopyRaises.calls == 0


def test_nested_model_subclasses_are_rejected_instead_of_trusted() -> None:
    clean_snapshot = make_snapshot(
        world(), episode_id="episode_1", snapshot_id="snapshot_1", observation={}
    )
    derived_state = _WorldStateSubclass(**world().model_dump())
    with pytest.raises(ValidationError):
        StateSnapshot(**{**clean_snapshot.model_dump(), "state": derived_state})
    with pytest.raises(ValidationError):
        TransitionProposal(local_status="success", next_state=derived_state, observation={})
    with pytest.raises(ValueError):
        make_snapshot(
            derived_state,
            episode_id="episode_1",
            snapshot_id="snapshot_1",
            observation={},
        )

    clean_event = event_zero()
    derived_action = _ActionCallSubclass(**action().model_dump())
    with pytest.raises(ValueError):
        make_state_event(
            episode_id="episode_1",
            index=0,
            action=derived_action,
            result=success_result(),
            before_state=world(),
            after_state=world({"items": [1, 2, 3]}),
            delta=(patch(),),
        )
    with pytest.raises(ValidationError):
        StateEvent(
            **{
                **clean_event.model_dump(),
                "result": _ActionResultSubclass(**success_result().model_dump()),
            }
        )
    with pytest.raises(ValidationError):
        StateEvent(
            **{
                **clean_event.model_dump(),
                "delta": (_StatePatchSubclass(**patch().model_dump()),),
            }
        )

    terminal = make_terminal_record(
        episode_id="episode_1",
        initial_state_hash=hash_state(world()),
        terminal_state_hash=hash_state(world()),
        event_root_hash=None,
        final_observation={},
        verdict=OutcomeVerdict(status="PASS", reason_code="COMPLETE", details={}),
    )
    with pytest.raises(ValidationError):
        TerminalRecord(
            **{
                **terminal.model_dump(),
                "verdict": _OutcomeVerdictSubclass(
                    status="PASS", reason_code="COMPLETE", details={}
                ),
            }
        )


def test_factories_reject_hostile_scalar_aliases_before_formatting_or_hashing() -> None:
    _HostileId.format_calls = 0
    _HostileId.equality_calls = 0
    _HostileIndex.format_calls = 0
    _HostileIndex.equality_calls = 0
    _HostileHashModel.calls = 0

    with pytest.raises(ValueError):
        make_snapshot(
            world(),
            episode_id=_HostileId("episode_1"),
            snapshot_id="snapshot_1",
            observation={},
        )
    with pytest.raises(ValueError):
        make_state_event(
            episode_id=_HostileId("episode_1"),
            index=_HostileIndex(0),
            action=action(),
            result=success_result(),
            before_state=world(),
            after_state=world({"items": [1, 2, 3]}),
            delta=(patch(),),
        )
    with pytest.raises(ValueError):
        make_terminal_record(
            episode_id="episode_1",
            initial_state_hash=_HostileHashModel(),  # type: ignore[arg-type]
            terminal_state_hash=HASH,
            event_root_hash=None,
            final_observation={},
            verdict=OutcomeVerdict(status="PASS", reason_code="COMPLETE", details={}),
        )
    with pytest.raises(ValueError):
        hash_observation(_HostileHashModel(), {})  # type: ignore[arg-type]

    assert _HostileId.format_calls == 0
    assert _HostileId.equality_calls == 0
    assert _HostileIndex.format_calls == 0
    assert _HostileIndex.equality_calls == 0
    assert _HostileHashModel.calls == 0


def test_public_helpers_revalidate_models_without_model_dump_hooks() -> None:
    _WorldStateDumpSubclass.calls = 0
    _ActionCallDumpSubclass.calls = 0
    _ActionResultDumpSubclass.calls = 0
    _StateEventDumpSubclass.calls = 0
    _TerminalRecordDumpSubclass.calls = 0

    clean_event = event_zero()
    clean_terminal = make_terminal_record(
        episode_id="episode_1",
        initial_state_hash=hash_state(world()),
        terminal_state_hash=hash_state(world({"items": [1, 2, 3]})),
        event_root_hash=clean_event.event_hash,
        final_observation={"seen": 1},
        verdict=OutcomeVerdict(status="PASS", reason_code="COMPLETE", details={}),
    )
    hostile_values = (
        (state_projection, hash_state, _WorldStateDumpSubclass(**world().model_dump())),
        (action_projection, hash_action, _ActionCallDumpSubclass(**action().model_dump())),
        (
            result_projection,
            hash_result,
            _ActionResultDumpSubclass(**success_result().model_dump()),
        ),
        (event_projection, hash_event, _StateEventDumpSubclass(**clean_event.model_dump())),
        (
            terminal_projection,
            hash_terminal,
            _TerminalRecordDumpSubclass(**clean_terminal.model_dump()),
        ),
    )
    for projection, hasher, hostile in hostile_values:
        with pytest.raises(ValueError):
            cast(Callable[[object], object], projection)(hostile)
        with pytest.raises(ValueError):
            cast(Callable[[object], object], hasher)(hostile)

    assert _WorldStateDumpSubclass.calls == 0
    assert _ActionCallDumpSubclass.calls == 0
    assert _ActionResultDumpSubclass.calls == 0
    assert _StateEventDumpSubclass.calls == 0
    assert _TerminalRecordDumpSubclass.calls == 0


def test_public_helpers_reject_invalid_model_construct_instances_without_recursion() -> None:
    hostile = _DeepCopyRaises()
    invalid_state = WorldState.model_construct(
        schema_version="1.0", world_id="world_1", domain="domain_1", seed=7, data={"x": hostile}
    )
    invalid_action = ActionCall.model_construct(
        action_id="action_1", tool_name="tool_1", arguments={"x": hostile}
    )
    invalid_result = ActionResult.model_construct(
        local_status="success", observation={"x": hostile}, error_code=None, error_detail=None
    )
    invalid_event = StateEvent.model_construct(
        schema_version="1.0",
        episode_id="episode_1",
        event_id="event_episode_1_000000",
        index=0,
        parent_event_id=None,
        parent_event_hash=None,
        action_id="action_1",
        action_hash=HASH,
        result=invalid_result,
        before_hash=HASH,
        after_hash=HASH,
        delta=(),
        event_hash=HASH,
    )
    invalid_terminal = TerminalRecord.model_construct(
        episode_id="episode_1",
        initial_state_hash=HASH,
        terminal_state_hash=HASH,
        event_root_hash=None,
        final_observation_hash=HASH,
        verdict=OutcomeVerdict.model_construct(
            status="PASS", reason_code="COMPLETE", details={"x": hostile}
        ),
        verdict_hash=HASH,
    )
    for projection, hasher, invalid in (
        (state_projection, hash_state, invalid_state),
        (action_projection, hash_action, invalid_action),
        (result_projection, hash_result, invalid_result),
        (event_projection, hash_event, invalid_event),
        (terminal_projection, hash_terminal, invalid_terminal),
    ):
        with pytest.raises(ValidationError):
            cast(Callable[[object], object], projection)(invalid)
        with pytest.raises(ValidationError):
            cast(Callable[[object], object], hasher)(invalid)


def test_incomplete_model_construct_instances_fail_closed_without_attribute_errors() -> None:
    incomplete_state = WorldState.model_construct()
    incomplete_action = ActionCall.model_construct()
    incomplete_result = ActionResult.model_construct()
    incomplete_event = StateEvent.model_construct()
    incomplete_terminal = TerminalRecord.model_construct()

    for helper, incomplete in (
        (state_projection, incomplete_state),
        (hash_state, incomplete_state),
        (action_projection, incomplete_action),
        (hash_action, incomplete_action),
        (result_projection, incomplete_result),
        (hash_result, incomplete_result),
        (event_projection, incomplete_event),
        (hash_event, incomplete_event),
        (terminal_projection, incomplete_terminal),
        (hash_terminal, incomplete_terminal),
    ):
        with pytest.raises((ValueError, ValidationError)) as error:
            cast(Callable[[object], object], helper)(incomplete)
        assert not isinstance(error.value, AttributeError)

    with pytest.raises(ValueError):
        make_state_event(
            episode_id="episode_1",
            index=1,
            action=action(),
            result=success_result(),
            before_state=world({"items": [1, 2, 3]}),
            after_state=world({"items": [1, 2, 3]}),
            delta=(),
            parent_event=StateEvent.model_construct(),
        )
    with pytest.raises(ValueError):
        make_state_event(
            episode_id="episode_1",
            index=0,
            action=action(),
            result=success_result(),
            before_state=world(),
            after_state=world({"items": [1, 2, 3]}),
            delta=(StatePatch.model_construct(),),
        )
    with pytest.raises(ValueError):
        make_terminal_record(
            episode_id="episode_1",
            initial_state_hash=HASH,
            terminal_state_hash=HASH,
            event_root_hash=None,
            final_observation={},
            verdict=OutcomeVerdict.model_construct(),
        )


def test_public_projection_and_hash_helpers_preserve_clean_bindings() -> None:
    state = world()
    action_call = action()
    result = success_result()
    event = event_zero()
    terminal = make_terminal_record(
        episode_id="episode_1",
        initial_state_hash=hash_state(state),
        terminal_state_hash=hash_state(state),
        event_root_hash=event.event_hash,
        final_observation={"seen": 1},
        verdict=OutcomeVerdict(status="PASS", reason_code="COMPLETE", details={}),
    )

    assert state_projection(state) == state.model_dump(mode="json")
    assert action_projection(action_call) == action_call.model_dump(mode="json")
    assert result_projection(result) == result.model_dump(mode="json")
    assert observation_projection(hash_state(state), {"seen": 1}) == {
        "state_hash": hash_state(state),
        "observation": {"seen": 1},
    }
    assert hash_event(event) == event.event_hash
    assert hash_terminal(terminal) == terminal.verdict_hash


def test_plain_text_fields_also_reject_noncanonical_unicode() -> None:
    with pytest.raises(ValidationError):
        TaskCase(
            case_id="case_1",
            domain="domain_1",
            world_id="world_1",
            seed=0,
            objective="bad\ud800objective",
            inputs={},
        )
    with pytest.raises(ValidationError):
        StatePatch(
            op="add",
            path="/bad\ud800token",
            before_present=False,
            after_present=True,
            after=1,
        )


def test_state_hash_uses_full_ssb_cj1_projection() -> None:
    first = world({"z": [1, 2], "a": True})
    reordered = world({"a": True, "z": [1, 2]})

    assert hash_state(first) == hash_state(reordered)
    assert hash_state(first) != hash_state(world({"z": [2, 1], "a": True}))
    assert hash_state(first) != hash_state(world({"z": [1, 2], "a": True}, seed=8))


def test_snapshot_binds_full_state_and_observation_hashes() -> None:
    state = world()
    snapshot = make_snapshot(
        state,
        episode_id="episode_1",
        snapshot_id="snapshot_1",
        observation={"visible": True},
    )

    assert snapshot.state_hash == hash_state(state)
    assert snapshot.observation_hash == hash_observation(snapshot.state_hash, snapshot.observation)
    with pytest.raises(ValidationError):
        StateSnapshot(**{**snapshot.model_dump(), "state_hash": HASH})
    with pytest.raises(ValidationError):
        StateSnapshot(**{**snapshot.model_dump(), "observation_hash": HASH})


def test_action_result_and_transition_status_rules_fail_closed() -> None:
    assert success_result().error_code is None
    assert (
        TransitionProposal(local_status="success", next_state=world(), observation={}).next_state
        is not None
    )
    assert (
        ActionResult(
            local_status="failure", observation={}, error_code="INVALID_ACTION"
        ).local_status
        == "failure"
    )

    with pytest.raises(ValidationError):
        ActionResult(local_status="success", observation={}, error_code="INVALID_ACTION")
    with pytest.raises(ValidationError):
        ActionResult(local_status="clarify", observation={})
    with pytest.raises(ValidationError):
        TransitionProposal(local_status="success", observation={})
    with pytest.raises(ValidationError):
        TransitionProposal(
            local_status="failure", next_state=world(), observation={}, error_code="INVALID_ACTION"
        )


@pytest.mark.parametrize("path", ["items", "/bad~2escape", "/bad~", "/also~3bad"])
def test_patch_pointer_and_presence_contracts(path: str) -> None:
    with pytest.raises(ValidationError):
        patch(path)
    with pytest.raises(ValidationError):
        StatePatch(
            op="add",
            path="/items",
            before_present=True,
            before=None,
            after_present=True,
            after=1,
        )
    with pytest.raises(ValidationError):
        StatePatch(
            op="remove",
            path="/items",
            before_present=True,
            before=1,
            after_present=False,
            after=1,
        )
    assert (
        StatePatch(op="add", path="", before_present=False, after_present=True, after=None).path
        == ""
    )


def test_event_cursor_and_factories_bind_identity_parent_and_hashes() -> None:
    first = event_zero()
    second = make_state_event(
        episode_id="episode_1",
        index=1,
        action=action(action_id="action_2"),
        result=success_result({"seen": 2}),
        before_state=world({"items": [1, 2, 3]}),
        after_state=world({"items": [1, 2, 3]}),
        delta=(),
        parent_event=first,
    )

    assert first.event_id == "event_episode_1_000000"
    assert first.parent_event_id is None and first.parent_event_hash is None
    assert second.event_id == "event_episode_1_000001"
    assert second.parent_event_id == first.event_id
    assert second.parent_event_hash == first.event_hash
    assert hash_action(action()) == first.action_hash
    assert hash_event(first) == first.event_hash
    assert EventCursor(
        episode_id="episode_1",
        next_index=1,
        parent_event_id=first.event_id,
        parent_event_hash=first.event_hash,
    )

    with pytest.raises(ValidationError):
        EventCursor(episode_id="episode_1", next_index=1)
    with pytest.raises(ValidationError):
        StateEvent(**{**first.model_dump(), "parent_event_id": "event_episode_1_000000"})


def test_event_hash_projection_has_chain_binding_teeth() -> None:
    first = event_zero()
    second = make_state_event(
        episode_id="episode_1",
        index=1,
        action=action(action_id="action_2"),
        result=success_result(),
        before_state=world({"items": [1, 2, 3]}),
        after_state=world({"items": [1, 2, 3]}),
        delta=(),
        parent_event=first,
    )

    def rebind_event(**changes: Any) -> StateEvent:
        fields: dict[str, Any] = {
            "schema_version": second.schema_version,
            "episode_id": second.episode_id,
            "event_id": second.event_id,
            "index": second.index,
            "parent_event_id": second.parent_event_id,
            "parent_event_hash": second.parent_event_hash,
            "action_id": second.action_id,
            "action_hash": second.action_hash,
            "result": second.result,
            "before_hash": second.before_hash,
            "after_hash": second.after_hash,
            "delta": second.delta,
        }
        fields.update(changes)
        body = {
            **fields,
            "result": fields["result"].model_dump(mode="json"),
            "delta": [item.model_dump(mode="json") for item in fields["delta"]],
        }
        return StateEvent(**fields, event_hash=sha256_ref(body))

    variants = (
        rebind_event(
            index=2,
            event_id="event_episode_1_000002",
            parent_event_id="event_episode_1_000001",
            parent_event_hash=second.event_hash,
        ),
        rebind_event(parent_event_hash=OTHER_HASH),
        rebind_event(action_id="action_99"),
        rebind_event(action_hash=OTHER_HASH),
        rebind_event(result=success_result({"changed": True})),
        rebind_event(before_hash=OTHER_HASH, delta=(patch("/different"),)),
        rebind_event(after_hash=OTHER_HASH, delta=(patch("/different"),)),
        rebind_event(after_hash=OTHER_HASH, delta=(patch("/another"),)),
    )

    assert all(hash_event(variant) != second.event_hash for variant in variants)


def test_event_status_delta_and_patch_overlap_rules_fail_closed() -> None:
    with pytest.raises(ValidationError):
        make_state_event(
            episode_id="episode_1",
            index=0,
            action=action(),
            result=ActionResult(
                local_status="failure", observation={}, error_code="INVALID_ACTION"
            ),
            before_state=world(),
            after_state=world({"items": [1, 2, 3]}),
            delta=(),
        )
    with pytest.raises(ValidationError):
        make_state_event(
            episode_id="episode_1",
            index=0,
            action=action(),
            result=success_result(),
            before_state=world(),
            after_state=world({"items": [1, 2, 3]}),
            delta=(patch("/items"), patch("/items/0")),
        )
    with pytest.raises(ValidationError):
        make_state_event(
            episode_id="episode_1",
            index=0,
            action=action(),
            result=success_result(),
            before_state=world(),
            after_state=world(),
            delta=(patch("/items", after=[1, 2]),),
        )


def test_terminal_projection_binds_all_generic_verdict_inputs() -> None:
    event = event_zero()
    verdict = OutcomeVerdict(status="PASS", reason_code="COMPLETE", details={"checked": True})
    terminal = make_terminal_record(
        episode_id="episode_1",
        initial_state_hash=hash_state(world()),
        terminal_state_hash=hash_state(world({"items": [1, 2, 3]})),
        event_root_hash=event.event_hash,
        final_observation={"seen": 1},
        verdict=verdict,
    )

    def rebind_terminal(**changes: Any) -> TerminalRecord:
        fields: dict[str, Any] = {
            "episode_id": terminal.episode_id,
            "initial_state_hash": terminal.initial_state_hash,
            "terminal_state_hash": terminal.terminal_state_hash,
            "event_root_hash": terminal.event_root_hash,
            "final_observation_hash": terminal.final_observation_hash,
            "verdict": terminal.verdict,
        }
        fields.update(changes)
        body = {**fields, "verdict": fields["verdict"].model_dump(mode="json")}
        return TerminalRecord(**fields, verdict_hash=sha256_ref(body))

    variants = (
        rebind_terminal(episode_id="episode_2"),
        rebind_terminal(initial_state_hash=OTHER_HASH),
        rebind_terminal(terminal_state_hash=OTHER_HASH),
        rebind_terminal(event_root_hash=OTHER_HASH),
        rebind_terminal(final_observation_hash=OTHER_HASH),
        rebind_terminal(verdict=OutcomeVerdict(status="FAIL", reason_code="COMPLETE", details={})),
        rebind_terminal(verdict=OutcomeVerdict(status="PASS", reason_code="OTHER", details={})),
        rebind_terminal(
            verdict=OutcomeVerdict(status="PASS", reason_code="COMPLETE", details={"x": 1})
        ),
    )

    assert isinstance(terminal, TerminalRecord)
    assert hash_terminal(terminal) == terminal.verdict_hash
    assert all(hash_terminal(variant) != terminal.verdict_hash for variant in variants)


def test_factories_are_deterministic_and_have_no_ambient_behavior() -> None:
    first = event_zero()
    second = event_zero()
    source = inspect.getsource(engine_models)

    assert first == second
    assert all(
        forbidden not in source
        for forbidden in ("datetime", "time.", "uuid", "random", "provider", "pathlib", "open(")
    )
