from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, cast, overload

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import shadowskillbench.engine.replay as replay_module
from shadowskillbench.engine.models import (
    ActionCall,
    JsonObject,
    OutcomeVerdict,
    StateEvent,
    StateSnapshot,
    TerminalRecord,
    TransitionProposal,
    WorldState,
    event_projection,
    make_terminal_record,
    terminal_projection,
)
from shadowskillbench.engine.replay import (
    ReplayIntegrityFailure,
    ReplayResult,
    replay,
    restore,
    snapshot,
)
from shadowskillbench.engine.runtime import EnvironmentInvariantFailure

REPLAY_SOURCE = Path(__file__).parents[2] / "src" / "shadowskillbench" / "engine" / "replay.py"


def _run_replay_mutant(
    tmp_path: Path, mutated_source: str, invariant: str
) -> subprocess.CompletedProcess[str]:
    mutant = tmp_path / "replay_mutant.py"
    invariant_script = tmp_path / "replay_invariant.py"
    mutant.write_text(mutated_source)
    invariant_script.write_text(invariant)
    return subprocess.run(
        [sys.executable, str(invariant_script), str(mutant)],
        capture_output=True,
        text=True,
        check=False,
    )


def state(value: int | float = 0) -> WorldState:
    return WorldState(
        schema_version="1.0",
        world_id="world",
        domain="domain",
        seed=1,
        data={"value": value, "nested": {"values": [value]}},
    )


make_state = state


def initial(value: int = 0) -> StateSnapshot:
    return snapshot(
        state(value),
        episode_id="episode",
        snapshot_id="snapshot",
        observation={"initial": value},
    )


def action(
    action_id: str,
    amount: int | float = 1,
    status: Literal["success", "failure", "clarify"] = "success",
) -> ActionCall:
    return ActionCall(
        action_id=action_id,
        tool_name="adjust",
        arguments={"amount": amount, "status": status},
    )


def verdict(status: Literal["PASS", "FAIL", "ESCALATE"] = "PASS") -> OutcomeVerdict:
    return OutcomeVerdict(status=status, reason_code="VERIFIED", details={"checked": True})


class Adapter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.received: list[WorldState] = []

    def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
        self.calls.append(call.action_id)
        self.received.append(state)
        status = cast(Literal["success", "failure", "clarify"], call.arguments["status"])
        if status != "success":
            return TransitionProposal(
                local_status=status,
                observation={"status": status},
                error_code="INVALID_ACTION",
            )
        next_value = cast(int | float, state.data["value"]) + cast(
            int | float, call.arguments["amount"]
        )
        return TransitionProposal(
            local_status="success",
            next_state=make_state(next_value),
            observation={"value": next_value},
        )


class Verifier:
    def __init__(self, callback: Callable[[WorldState], OutcomeVerdict] | None = None) -> None:
        self.calls = 0
        self.received: list[WorldState] = []
        self._callback = callback if callback is not None else lambda _state: verdict()

    def __call__(self, state: WorldState) -> OutcomeVerdict:
        self.calls += 1
        self.received.append(state)
        return self._callback(state)


def test_snapshot_delegates_restore_detaches_and_rejects_stale_nested_data() -> None:
    source = inspect.getsource(replay_module.snapshot)
    assert "make_snapshot" in source
    original = initial(3)

    restored = restore(original)
    restored_nested = cast(JsonObject, restored.state.data["nested"])
    restored_nested["values"] = [99]
    assert original.state.data["nested"] == {"values": [3]}

    original_nested = cast(JsonObject, original.state.data["nested"])
    original_nested["values"] = [4]
    with pytest.raises(ReplayIntegrityFailure):
        restore(original)
    with pytest.raises(ReplayIntegrityFailure):
        restore(StateSnapshot.model_construct())


def test_restore_and_preflight_reject_constructed_or_hostile_models_without_hooks() -> None:
    class HostileSnapshot(StateSnapshot):
        hook_calls: ClassVar[int] = 0

        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            type(self).hook_calls += 1
            return {}

        def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
            type(self).hook_calls += 1
            return self

    source = initial()
    hostile = HostileSnapshot(
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        state=source.state,
        state_hash=source.state_hash,
        observation=source.observation,
        observation_hash=source.observation_hash,
    )
    with pytest.raises(ReplayIntegrityFailure):
        restore(hostile)
    with pytest.raises(ReplayIntegrityFailure):
        replay(
            source,
            (ActionCall.model_construct(),),
            Adapter(),
            verifier=Verifier(),
        )
    assert HostileSnapshot.hook_calls == 0


@pytest.mark.parametrize("status", ["success", "failure", "clarify"])
def test_replay_emits_exact_chain_for_all_local_statuses(
    status: Literal["success", "failure", "clarify"],
) -> None:
    adapter = Adapter()
    verifier = Verifier()
    result = replay(initial(), (action("a1", status=status),), adapter, verifier=verifier)

    assert tuple(item.action_id for item in result.actions) == ("a1",)
    assert len(result.events) == 1
    assert result.events[0].index == 0
    assert result.events[0].parent_event_hash is None
    assert result.events[0].action_id == "a1"
    assert result.terminal.event_root_hash == result.events[0].event_hash
    assert verifier.calls == adapter.calls.__len__() == 1
    if status == "success":
        assert result.terminal_state.data["value"] == 1
    else:
        assert result.terminal_state.data["value"] == 0
        assert result.events[0].delta == ()


def test_zero_action_replay_uses_initial_observation_and_calls_verifier_once() -> None:
    adapter = Adapter()
    verifier = Verifier()
    source = initial(4)

    result = replay(source, (), adapter, verifier=verifier)

    assert result.actions == result.events == ()
    assert result.terminal.event_root_hash is None
    assert result.terminal.final_observation_hash == source.observation_hash
    assert result.terminal_state == source.state
    assert adapter.calls == []
    assert verifier.calls == 1
    assert verifier.received[0] is not source.state


def test_replay_retains_input_order_rejects_duplicate_ids_before_callbacks() -> None:
    adapter = Adapter()
    verifier = Verifier()
    ordered = (action("second", 2), action("first", 1))
    result = replay(initial(), ordered, adapter, verifier=verifier)
    assert tuple(item.action_id for item in result.actions) == ("second", "first")
    assert adapter.calls == ["second", "first"]

    duplicate = (action("same"), action("same", 2))
    with pytest.raises(ReplayIntegrityFailure):
        replay(initial(), duplicate, adapter, verifier=verifier)
    assert verifier.calls == 1
    assert adapter.calls == ["second", "first"]


def test_custom_sequence_is_materialized_once_before_callbacks() -> None:
    class OneShotActions(Sequence[ActionCall]):
        def __init__(self) -> None:
            self.iterations = 0
            self.items = (action("second", 2), action("first", 1))

        @overload
        def __getitem__(self, index: int) -> ActionCall: ...

        @overload
        def __getitem__(self, index: slice) -> Sequence[ActionCall]: ...

        def __getitem__(self, index: int | slice) -> ActionCall | Sequence[ActionCall]:
            return self.items[index]

        def __len__(self) -> int:
            return len(self.items)

        def __iter__(self) -> Iterator[ActionCall]:
            self.iterations += 1
            yield from self.items

    supplied = OneShotActions()
    adapter = Adapter()
    result = replay(initial(), supplied, adapter, verifier=Verifier())

    assert supplied.iterations == 1
    assert tuple(item.action_id for item in result.actions) == ("second", "first")
    assert adapter.calls == ["second", "first"]


def test_custom_sequence_failure_and_invalid_actions_stop_before_callbacks() -> None:
    class ExplodingActions(Sequence[ActionCall]):
        @overload
        def __getitem__(self, index: int) -> ActionCall: ...

        @overload
        def __getitem__(self, index: slice) -> Sequence[ActionCall]: ...

        def __getitem__(self, index: int | slice) -> ActionCall | Sequence[ActionCall]:
            raise RuntimeError("unexpected indexing")

        def __len__(self) -> int:
            return 1

        def __iter__(self) -> Iterator[ActionCall]:
            raise RuntimeError("sequence exploded")
            yield action("unreachable")

    adapter = Adapter()
    verifier = Verifier()
    with pytest.raises(ReplayIntegrityFailure) as sequence_error:
        replay(initial(), ExplodingActions(), adapter, verifier=verifier)
    assert sequence_error.value.code == "ACTIONS_INVALID"

    cyclic: dict[str, object] = {}
    cyclic["loop"] = cyclic
    malformed = ActionCall.model_construct(
        action_id="bad", tool_name="adjust", arguments=cast(JsonObject, cyclic)
    )
    with pytest.raises(ReplayIntegrityFailure) as action_error:
        replay(initial(), (malformed,), adapter, verifier=verifier)
    assert action_error.value.code == "ACTIONS_INVALID"
    assert adapter.calls == []
    assert verifier.calls == 0


def test_expected_pair_is_verified_without_driving_execution() -> None:
    actions = (action("a1", 2), action("a2", 3))
    generated = replay(initial(), actions, Adapter(), verifier=Verifier())
    adapter = Adapter()
    verifier = Verifier()

    checked = replay(
        initial(),
        actions,
        adapter,
        verifier=verifier,
        expected_events=generated.events,
        expected_terminal=generated.terminal,
    )

    assert checked == generated
    assert adapter.calls == ["a1", "a2"]
    assert verifier.calls == 1


@pytest.mark.parametrize(
    ("actions", "mismatch"),
    [
        ((action("a1"),), "state_hash"),
        ((action("a1"),), "observation"),
        ((), "state_hash"),
        ((), "observation"),
    ],
)
def test_expected_terminal_static_bindings_fail_before_callbacks(
    actions: tuple[ActionCall, ...], mismatch: Literal["state_hash", "observation"]
) -> None:
    generated = replay(initial(), actions, Adapter(), verifier=Verifier())
    expected_state_hash = (
        generated.events[-1].after_hash if generated.events else generated.initial.state_hash
    )
    expected_observation = (
        generated.events[-1].result.observation
        if generated.events
        else generated.initial.observation
    )
    terminal = make_terminal_record(
        episode_id=generated.terminal.episode_id,
        initial_state_hash=generated.initial.state_hash,
        terminal_state_hash=(
            "sha256:" + "f" * 64 if mismatch == "state_hash" else expected_state_hash
        ),
        event_root_hash=generated.terminal.event_root_hash,
        final_observation=({"wrong": True} if mismatch == "observation" else expected_observation),
        verdict=generated.terminal.verdict,
    )
    adapter = Adapter()
    verifier = Verifier()

    with pytest.raises(ReplayIntegrityFailure) as error:
        replay(
            initial(),
            actions,
            adapter,
            verifier=verifier,
            expected_events=generated.events,
            expected_terminal=terminal,
        )

    assert error.value.code == "EXPECTED_ARTIFACT_INVALID"
    assert adapter.calls == []
    assert verifier.calls == 0


def test_malformed_or_subclass_expected_artifacts_fail_without_hooks() -> None:
    generated = replay(initial(), (action("a1"),), Adapter(), verifier=Verifier())

    class HostileEvent(StateEvent):
        hook_calls: ClassVar[int] = 0

        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            type(self).hook_calls += 1
            return {}

    class HostileTerminal(TerminalRecord):
        hook_calls: ClassVar[int] = 0

        def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
            type(self).hook_calls += 1
            return self

    event = generated.events[0]
    hostile_event = HostileEvent(**event.model_dump())
    terminal = generated.terminal
    hostile_terminal = HostileTerminal(**terminal.model_dump())
    adapter = Adapter()
    verifier = Verifier()

    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            (action("a1"),),
            adapter,
            verifier=verifier,
            expected_events=(hostile_event,),
            expected_terminal=generated.terminal,
        )
    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            (action("a1"),),
            adapter,
            verifier=verifier,
            expected_events=(StateEvent.model_construct(),),
            expected_terminal=hostile_terminal,
        )
    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            (action("a1"),),
            adapter,
            verifier=verifier,
            expected_events=generated.events,
            expected_terminal=TerminalRecord.model_construct(),
        )
    assert HostileEvent.hook_calls == 0
    assert HostileTerminal.hook_calls == 0
    assert adapter.calls == []
    assert verifier.calls == 0


def test_self_consistent_result_mismatch_stops_before_later_action_or_verifier() -> None:
    class AlternateObservationAdapter(Adapter):
        def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
            proposal = super().apply(state, call)
            if proposal.local_status != "success":
                return proposal
            return TransitionProposal(
                local_status="success",
                next_state=proposal.next_state,
                observation={"alternate": call.action_id},
            )

    actions = (action("a1"), action("a2"))
    expected = replay(initial(), actions, AlternateObservationAdapter(), verifier=Verifier())
    adapter = Adapter()
    verifier = Verifier()

    with pytest.raises(ReplayIntegrityFailure) as error:
        replay(
            initial(),
            actions,
            adapter,
            verifier=verifier,
            expected_events=expected.events,
            expected_terminal=expected.terminal,
        )
    assert error.value.code == "EXPECTED_EVENT_MISMATCH"
    assert adapter.calls == ["a1"]
    assert verifier.calls == 0


def test_partial_or_stale_expected_artifacts_fail_before_callbacks() -> None:
    generated = replay(initial(), (action("a1"),), Adapter(), verifier=Verifier())
    adapter = Adapter()
    verifier = Verifier()
    stale_event = generated.events[0].model_copy(update={"event_hash": "sha256:" + "0" * 64})

    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(), (action("a1"),), adapter, verifier=verifier, expected_events=generated.events
        )
    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            (action("a1"),),
            adapter,
            verifier=verifier,
            expected_terminal=generated.terminal,
        )
    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            (action("a1"),),
            adapter,
            verifier=verifier,
            expected_events=(),
            expected_terminal=generated.terminal,
        )
    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            (action("a1"),),
            adapter,
            verifier=verifier,
            expected_events=(generated.events[0], generated.events[0]),
            expected_terminal=generated.terminal,
        )
    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            (action("a1"),),
            adapter,
            verifier=verifier,
            expected_events=(stale_event,),
            expected_terminal=generated.terminal,
        )
    assert adapter.calls == []
    assert verifier.calls == 0


def test_expected_structure_and_canonical_action_comparison_fail_before_callbacks() -> None:
    actions = (action("a1", 0), action("a2", 0))
    generated = replay(initial(), actions, Adapter(), verifier=Verifier())
    adapter = Adapter()
    verifier = Verifier()

    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            actions,
            adapter,
            verifier=verifier,
            expected_events=(generated.events[1], generated.events[0]),
            expected_terminal=generated.terminal,
        )

    bool_action = ActionCall(
        action_id="flag",
        tool_name="adjust",
        arguments={"amount": 0, "status": "success", "flag": True},
    )
    bool_generated = replay(initial(), (bool_action,), Adapter(), verifier=Verifier())
    integer_action = ActionCall(
        action_id="flag",
        tool_name="adjust",
        arguments={"amount": 0, "status": "success", "flag": 1},
    )
    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            (integer_action,),
            adapter,
            verifier=verifier,
            expected_events=bool_generated.events,
            expected_terminal=bool_generated.terminal,
        )

    signed_zero = action("zero", -0.0)
    zero_generated = replay(initial(), (signed_zero,), Adapter(), verifier=Verifier())
    zero_action = action("zero", 0.0)
    accepted = replay(
        initial(),
        (zero_action,),
        Adapter(),
        verifier=Verifier(),
        expected_events=zero_generated.events,
        expected_terminal=zero_generated.terminal,
    )
    assert accepted.events == zero_generated.events
    assert adapter.calls == []
    assert verifier.calls == 0


def test_replay_rejects_expected_action_event_and_terminal_mismatches() -> None:
    actions = (action("a1", 1),)
    generated = replay(initial(), actions, Adapter(), verifier=Verifier())

    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            (action("a1", 2),),
            Adapter(),
            verifier=Verifier(),
            expected_events=generated.events,
            expected_terminal=generated.terminal,
        )

    different_terminal = make_terminal_record(
        episode_id="episode",
        initial_state_hash=generated.initial.state_hash,
        terminal_state_hash=generated.terminal.terminal_state_hash,
        event_root_hash=generated.terminal.event_root_hash,
        final_observation={"value": 1},
        verdict=verdict("FAIL"),
    )
    with pytest.raises(ReplayIntegrityFailure):
        replay(
            initial(),
            actions,
            Adapter(),
            verifier=Verifier(),
            expected_events=generated.events,
            expected_terminal=different_terminal,
        )


def test_replay_propagates_adapter_failure_and_rejects_verifier_contract_breaks() -> None:
    class ExplodingAdapter(Adapter):
        def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
            raise RuntimeError("adapter failed")

    verifier = Verifier()
    with pytest.raises(EnvironmentInvariantFailure) as adapter_error:
        replay(initial(), (action("a1"),), ExplodingAdapter(), verifier=verifier)
    assert adapter_error.value.code == "ADAPTER_EXCEPTION"
    assert verifier.calls == 0

    direct = EnvironmentInvariantFailure("DIRECT", "preserved identity")

    class DirectFailureAdapter(Adapter):
        def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
            raise direct

    with pytest.raises(EnvironmentInvariantFailure) as direct_error:
        replay(initial(), (action("a1"), action("a2")), DirectFailureAdapter(), verifier=verifier)
    assert direct_error.value.code == "ADAPTER_EXCEPTION"
    assert direct_error.value.__cause__ is direct
    assert verifier.calls == 0

    def mutate(received_state: WorldState) -> OutcomeVerdict:
        received_state.data["value"] = 99
        return verdict()

    with pytest.raises(EnvironmentInvariantFailure) as mutation_error:
        replay(initial(), (), Adapter(), verifier=Verifier(mutate))
    assert mutation_error.value.code == "VERIFIER_INPUT_MUTATION"

    with pytest.raises(EnvironmentInvariantFailure) as verifier_error:
        replay(
            initial(),
            (),
            Adapter(),
            verifier=Verifier(lambda _state: (_ for _ in ()).throw(RuntimeError("verify failed"))),
        )
    assert verifier_error.value.code == "VERIFIER_EXCEPTION"

    def make_cyclic_verdict(received_state: WorldState) -> OutcomeVerdict:
        cycle: dict[str, object] = {}
        cycle["self"] = cycle
        received_state.data["bad"] = cast(JsonObject, cycle)
        return verdict()

    with pytest.raises(EnvironmentInvariantFailure) as cyclic_error:
        replay(initial(), (), Adapter(), verifier=Verifier(make_cyclic_verdict))
    assert cyclic_error.value.code == "VERIFIER_INPUT_MUTATION"

    with pytest.raises(EnvironmentInvariantFailure):
        replay(
            initial(),
            (),
            Adapter(),
            verifier=Verifier(lambda _state: cast(OutcomeVerdict, object())),
        )
    with pytest.raises(EnvironmentInvariantFailure):
        replay(
            initial(),
            (),
            Adapter(),
            verifier=Verifier(lambda _state: OutcomeVerdict.model_construct()),
        )

    class VerdictSubclass(OutcomeVerdict):
        pass

    with pytest.raises(EnvironmentInvariantFailure):
        replay(
            initial(),
            (),
            Adapter(),
            verifier=Verifier(
                lambda _state: VerdictSubclass(status="PASS", reason_code="OK", details={})
            ),
        )
    with pytest.raises(KeyboardInterrupt):
        replay(
            initial(),
            (),
            Adapter(),
            verifier=Verifier(lambda _state: (_ for _ in ()).throw(KeyboardInterrupt())),
        )

    class VerifierBaseException(BaseException):
        pass

    with pytest.raises(VerifierBaseException):
        replay(
            initial(),
            (),
            Adapter(),
            verifier=Verifier(lambda _state: (_ for _ in ()).throw(VerifierBaseException())),
        )


def test_replay_propagates_task6_invariant_identity_without_later_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = EnvironmentInvariantFailure("TASK6", "preconstructed invariant")

    def fail_step(*args: object, **kwargs: object) -> object:
        raise direct

    monkeypatch.setattr(replay_module, "execute_action", fail_step)
    adapter = Adapter()
    verifier = Verifier()
    with pytest.raises(EnvironmentInvariantFailure) as captured:
        replay(initial(), (action("a1"), action("a2")), adapter, verifier=verifier)
    assert captured.value is direct
    assert adapter.calls == []
    assert verifier.calls == 0


def test_replay_result_revalidates_cross_links_and_caller_storage_is_owned() -> None:
    source = initial()
    actions = [action("a1")]
    result = replay(source, actions, Adapter(), verifier=Verifier())
    result.terminal_state.data["value"] = 8
    assert source.state.data["value"] == 0
    assert actions[0].arguments == {"amount": 1, "status": "success"}

    expected = replay(initial(), (action("a1"),), Adapter(), verifier=Verifier())
    checked = replay(
        initial(),
        (action("a1"),),
        Adapter(),
        verifier=Verifier(),
        expected_events=expected.events,
        expected_terminal=expected.terminal,
    )
    expected.events[0].result.observation["caller"] = True
    expected.terminal.verdict.details["caller"] = True
    assert checked.events[0].result.observation == {"value": 1}
    assert checked.terminal.verdict.details == {"checked": True}

    with pytest.raises(Exception):
        ReplayResult(
            initial=result.initial,
            actions=result.actions,
            events=(),
            terminal_state=result.terminal_state,
            terminal=result.terminal,
        )

    with pytest.raises(Exception):
        ReplayResult.model_validate(
            {
                "initial": result.initial,
                "actions": result.actions,
                "events": result.events,
                "terminal_state": result.terminal_state,
                "terminal": result.terminal,
                "extra": True,
            }
        )


@pytest.mark.property
@settings(max_examples=20, deadline=None)
@given(st.lists(st.integers(min_value=-3, max_value=3), max_size=4))
def test_replay_property_retains_sequence_and_exact_chain(amounts: list[int]) -> None:
    actions = tuple(action(f"a{index}", amount) for index, amount in enumerate(amounts))
    result = replay(initial(), actions, Adapter(), verifier=Verifier())

    assert result.actions == actions
    assert len(result.events) == len(actions)
    assert tuple(event.index for event in result.events) == tuple(range(len(actions)))
    assert result.terminal.event_root_hash == (
        result.events[-1].event_hash if result.events else None
    )


@pytest.mark.property
def test_replay_is_deterministic_for_one_hundred_runs() -> None:
    actions = (action("a1", 1), action("a2", 2))
    results = [replay(initial(), actions, Adapter(), verifier=Verifier()) for _ in range(100)]
    first = results[0]
    assert all(result == first for result in results[1:])
    assert all(
        tuple(event_projection(event) for event in result.events)
        == tuple(event_projection(event) for event in first.events)
        and terminal_projection(result.terminal) == terminal_projection(first.terminal)
        for result in results[1:]
    )


def test_replay_has_no_ambient_or_domain_behavior() -> None:
    source = inspect.getsource(replay_module)
    assert all(
        forbidden not in source
        for forbidden in (
            "datetime",
            "time.",
            "uuid",
            "random",
            "provider",
            "pathlib",
            "open(",
            "TaskCase",
            "authority",
        )
    )


@pytest.mark.mutation
def test_replay_mutation_kills_expected_pair_gate_bypass(tmp_path: Path) -> None:
    source = REPLAY_SOURCE.read_text()
    needle = "if (expected_events is None) != (expected_terminal is None):"
    assert needle in source
    mutant = source.replace(needle, "if False:", 1)
    invariant = """\\
import importlib.util
import sys

from shadowskillbench.engine.models import (
    ActionCall, OutcomeVerdict, StateSnapshot, TransitionProposal, WorldState,
    hash_observation, hash_state,
)

spec = importlib.util.spec_from_file_location("replay_mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
state = WorldState(
    schema_version="1.0", world_id="world", domain="domain", seed=1, data={}
)
state_hash = hash_state(state)
initial = StateSnapshot(
    snapshot_id="snapshot", episode_id="episode", state=state, state_hash=state_hash,
    observation={}, observation_hash=hash_observation(state_hash, {}),
)
action = ActionCall(action_id="action", tool_name="tool", arguments={})
class Adapter:
    def apply(self, state, call):
        return TransitionProposal(local_status="success", next_state=state, observation={})
try:
    module.replay(
        initial, (action,), Adapter(),
        verifier=lambda state: OutcomeVerdict(status="PASS", reason_code="OK", details={}),
        expected_events=(),
    )
except module.ReplayIntegrityFailure:
    pass
else:
    raise AssertionError("partial expected artifacts were accepted")
"""

    result = _run_replay_mutant(tmp_path, mutant, invariant)

    assert result.returncode == 1
    assert "AssertionError" in result.stderr


@pytest.mark.mutation
def test_replay_mutation_kills_canonical_event_comparison_bypass(tmp_path: Path) -> None:
    source = REPLAY_SOURCE.read_text()
    needle = "return hash_event(left) == hash_event(right) and canonical_json_bytes("
    assert needle in source
    mutant = source.replace(
        "return hash_event(left) == hash_event(right) and canonical_json_bytes(\n"
        "        event_projection(left)\n"
        "    ) == canonical_json_bytes(event_projection(right))",
        "return left == right",
        1,
    )
    invariant = """\\
import importlib.util
import sys

from shadowskillbench.engine.models import (
    ActionCall, ActionResult, EventCursor, WorldState, make_state_event_from_cursor,
)

spec = importlib.util.spec_from_file_location("replay_mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
state = WorldState(
    schema_version="1.0", world_id="world", domain="domain", seed=1, data={}
)
action = ActionCall(action_id="action", tool_name="tool", arguments={})
cursor = EventCursor(episode_id="episode", next_index=0)
base = make_state_event_from_cursor(
    cursor=cursor, action=action, result=ActionResult(local_status="success", observation={}),
    before_state=state, after_state=state, delta=(),
)
left = base.model_copy(update={
    "result": ActionResult(local_status="success", observation={"value": True}),
    "event_hash": "sha256:" + "0" * 64,
})
right = base.model_copy(update={
    "result": ActionResult(local_status="success", observation={"value": 1}),
    "event_hash": "sha256:" + "0" * 64,
})
assert not module._same_event(left, right)
"""

    result = _run_replay_mutant(tmp_path, mutant, invariant)

    assert result.returncode == 1
    assert "AssertionError" in result.stderr
