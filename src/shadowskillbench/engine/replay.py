from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn, Protocol

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.engine.models import (
    ActionCall,
    ActionResult,
    EventCursor,
    JsonObject,
    OutcomeVerdict,
    PathSafeId,
    Sha256Ref,
    StateEvent,
    StatePatch,
    StateSnapshot,
    TerminalRecord,
    WorldState,
    event_projection,
    hash_action,
    hash_event,
    hash_observation,
    hash_state,
    hash_terminal,
    make_snapshot,
    make_terminal_record,
    terminal_projection,
)
from shadowskillbench.engine.runtime import (
    DomainAdapter,
    EnvironmentInvariantFailure,
    execute_action,
)


class ReplayIntegrityFailure(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class ReplayVerifier(Protocol):
    def __call__(self, state: WorldState) -> OutcomeVerdict: ...


def _integrity(code: str, detail: str, cause: Exception | None = None) -> NoReturn:
    error = ReplayIntegrityFailure(code, detail)
    if cause is None:
        raise error
    raise error from cause


def _environment(code: str, detail: str, cause: Exception | None = None) -> NoReturn:
    error = EnvironmentInvariantFailure(code, detail)
    if cause is None:
        raise error
    raise error from cause


def _fields(
    value: object,
    model_type: type[BaseModel],
    model_name: str,
    names: tuple[str, ...],
) -> dict[str, object]:
    if type(value) is not model_type:
        raise ValueError(f"{model_name} must be an exact validated model")
    try:
        return {name: getattr(value, name) for name in names}
    except AttributeError as error:
        raise ValueError(f"{model_name} is incomplete") from error


def _world_state(value: object) -> WorldState:
    fields = _fields(
        value,
        WorldState,
        "WorldState",
        ("schema_version", "world_id", "domain", "seed", "data"),
    )
    return WorldState.model_validate(fields)


def _action(value: object) -> ActionCall:
    fields = _fields(value, ActionCall, "ActionCall", ("action_id", "tool_name", "arguments"))
    return ActionCall.model_validate(fields)


def _result(value: object) -> ActionResult:
    fields = _fields(
        value,
        ActionResult,
        "ActionResult",
        ("local_status", "observation", "error_code", "error_detail"),
    )
    return ActionResult.model_validate(fields)


def _patch(value: object) -> StatePatch:
    fields = _fields(
        value,
        StatePatch,
        "StatePatch",
        ("op", "path", "before_present", "before", "after_present", "after"),
    )
    return StatePatch.model_validate(fields)


def _event(value: object) -> StateEvent:
    fields = _fields(
        value,
        StateEvent,
        "StateEvent",
        (
            "schema_version",
            "episode_id",
            "event_id",
            "index",
            "parent_event_id",
            "parent_event_hash",
            "action_id",
            "action_hash",
            "result",
            "before_hash",
            "after_hash",
            "delta",
            "event_hash",
        ),
    )
    delta = fields["delta"]
    if type(delta) is not tuple:
        raise ValueError("StateEvent delta must be a tuple")
    return StateEvent.model_validate(
        {
            **fields,
            "result": _result(fields["result"]),
            "delta": tuple(_patch(item) for item in delta),
        }
    )


def _verdict(value: object) -> OutcomeVerdict:
    fields = _fields(value, OutcomeVerdict, "OutcomeVerdict", ("status", "reason_code", "details"))
    return OutcomeVerdict.model_validate(fields)


def _terminal(value: object) -> TerminalRecord:
    fields = _fields(
        value,
        TerminalRecord,
        "TerminalRecord",
        (
            "episode_id",
            "initial_state_hash",
            "terminal_state_hash",
            "event_root_hash",
            "final_observation_hash",
            "verdict",
            "verdict_hash",
        ),
    )
    return TerminalRecord.model_validate({**fields, "verdict": _verdict(fields["verdict"])})


def _snapshot(value: object) -> StateSnapshot:
    fields = _fields(
        value,
        StateSnapshot,
        "StateSnapshot",
        ("snapshot_id", "episode_id", "state", "state_hash", "observation", "observation_hash"),
    )
    return StateSnapshot.model_validate({**fields, "state": _world_state(fields["state"])})


def _cursor(value: object) -> EventCursor:
    fields = _fields(
        value,
        EventCursor,
        "EventCursor",
        ("episode_id", "next_index", "parent_event_id", "parent_event_hash"),
    )
    return EventCursor.model_validate(fields)


def _actions(value: Sequence[ActionCall]) -> tuple[ActionCall, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("actions must be a sequence of ActionCall values")
    owned = tuple(_action(item) for item in tuple(value))
    if len({item.action_id for item in owned}) != len(owned):
        raise ValueError("action ids must be unique")
    return owned


def _events(value: Sequence[StateEvent]) -> tuple[StateEvent, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("events must be a sequence of StateEvent values")
    return tuple(_event(item) for item in tuple(value))


class ReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    initial: StateSnapshot
    actions: tuple[ActionCall, ...]
    events: tuple[StateEvent, ...]
    terminal_state: WorldState
    terminal: TerminalRecord

    @field_validator("initial", mode="before")
    @classmethod
    def own_initial(cls, value: object) -> StateSnapshot:
        return _snapshot(value)

    @field_validator("actions", mode="before")
    @classmethod
    def own_actions(cls, value: object) -> tuple[ActionCall, ...]:
        if type(value) is not tuple:
            raise ValueError("ReplayResult actions must be a tuple")
        return _actions(value)

    @field_validator("events", mode="before")
    @classmethod
    def own_events(cls, value: object) -> tuple[StateEvent, ...]:
        if type(value) is not tuple:
            raise ValueError("ReplayResult events must be a tuple")
        return _events(value)

    @field_validator("terminal_state", mode="before")
    @classmethod
    def own_terminal_state(cls, value: object) -> WorldState:
        return _world_state(value)

    @field_validator("terminal", mode="before")
    @classmethod
    def own_terminal(cls, value: object) -> TerminalRecord:
        return _terminal(value)

    @model_validator(mode="after")
    def validate_links(self) -> ReplayResult:
        _validate_replay_links(
            self.initial,
            self.actions,
            self.events,
            self.terminal_state,
            self.terminal,
        )
        return self

    @property
    def episode_id(self) -> PathSafeId:
        return self.initial.episode_id

    @property
    def initial_state_hash(self) -> Sha256Ref:
        return self.initial.state_hash

    @property
    def terminal_state_hash(self) -> Sha256Ref:
        return self.terminal.terminal_state_hash

    @property
    def event_root_hash(self) -> Sha256Ref | None:
        return self.terminal.event_root_hash


def _validate_event_chain(
    initial: StateSnapshot, actions: tuple[ActionCall, ...], events: tuple[StateEvent, ...]
) -> None:
    if len(actions) != len(events):
        raise ValueError("action and event counts differ")
    if len({action.action_id for action in actions}) != len(actions):
        raise ValueError("action ids must be unique")
    previous_hash = initial.state_hash
    previous_event: StateEvent | None = None
    for index, (action, event) in enumerate(zip(actions, events, strict=True)):
        if event.episode_id != initial.episode_id or event.index != index:
            raise ValueError("event episode or index differs from replay sequence")
        if event.action_id != action.action_id or event.action_hash != hash_action(action):
            raise ValueError("event does not bind its retained action")
        if event.before_hash != previous_hash:
            raise ValueError("event before hash breaks state chain")
        if previous_event is None:
            if event.parent_event_id is not None or event.parent_event_hash is not None:
                raise ValueError("first event has a parent")
        elif (
            event.parent_event_id != previous_event.event_id
            or event.parent_event_hash != previous_event.event_hash
        ):
            raise ValueError("event parent chain is broken")
        if event.event_hash != hash_event(event):
            raise ValueError("event hash is stale")
        previous_hash = event.after_hash
        previous_event = event


def _validate_replay_links(
    initial: StateSnapshot,
    actions: tuple[ActionCall, ...],
    events: tuple[StateEvent, ...],
    terminal_state: WorldState,
    terminal: TerminalRecord,
) -> None:
    _validate_event_chain(initial, actions, events)
    if terminal.episode_id != initial.episode_id:
        raise ValueError("terminal episode differs from initial snapshot")
    if terminal.initial_state_hash != initial.state_hash:
        raise ValueError("terminal does not bind initial snapshot")
    terminal_state_hash = hash_state(terminal_state)
    if terminal.terminal_state_hash != terminal_state_hash:
        raise ValueError("terminal does not bind terminal state")
    if events:
        if events[-1].after_hash != terminal_state_hash:
            raise ValueError("terminal state does not continue event chain")
        if terminal.event_root_hash != events[-1].event_hash:
            raise ValueError("terminal root does not bind last event")
        final_observation = events[-1].result.observation
    else:
        if terminal_state_hash != initial.state_hash:
            raise ValueError("zero-action terminal state differs from initial state")
        if terminal.event_root_hash is not None:
            raise ValueError("zero-action terminal has an event root")
        final_observation = initial.observation
    if terminal.final_observation_hash != hash_observation(terminal_state_hash, final_observation):
        raise ValueError("terminal does not bind final observation")
    if terminal.verdict_hash != hash_terminal(terminal):
        raise ValueError("terminal hash is stale")


def _validate_expected_static(
    initial: StateSnapshot,
    actions: tuple[ActionCall, ...],
    events: tuple[StateEvent, ...],
    terminal: TerminalRecord,
) -> None:
    _validate_event_chain(initial, actions, events)
    if terminal.episode_id != initial.episode_id:
        raise ValueError("expected terminal episode differs from initial snapshot")
    if terminal.initial_state_hash != initial.state_hash:
        raise ValueError("expected terminal does not bind initial snapshot")
    if events:
        if terminal.event_root_hash != events[-1].event_hash:
            raise ValueError("expected terminal root does not bind last event")
        expected_state_hash = events[-1].after_hash
        expected_observation = events[-1].result.observation
    elif terminal.event_root_hash is not None:
        raise ValueError("zero-action expected terminal has an event root")
    else:
        expected_state_hash = initial.state_hash
        expected_observation = initial.observation
    if terminal.terminal_state_hash != expected_state_hash:
        raise ValueError("expected terminal state does not continue event chain")
    if terminal.final_observation_hash != hash_observation(
        expected_state_hash, expected_observation
    ):
        raise ValueError("expected terminal does not bind final observation")
    if terminal.verdict_hash != hash_terminal(terminal):
        raise ValueError("expected terminal hash is stale")


def _same_event(left: StateEvent, right: StateEvent) -> bool:
    return hash_event(left) == hash_event(right) and canonical_json_bytes(
        event_projection(left)
    ) == canonical_json_bytes(event_projection(right))


def _same_terminal(left: TerminalRecord, right: TerminalRecord) -> bool:
    return hash_terminal(left) == hash_terminal(right) and canonical_json_bytes(
        terminal_projection(left)
    ) == canonical_json_bytes(terminal_projection(right))


def snapshot(
    state: WorldState,
    *,
    episode_id: PathSafeId,
    snapshot_id: PathSafeId,
    observation: JsonObject,
) -> StateSnapshot:
    return make_snapshot(
        state,
        episode_id=episode_id,
        snapshot_id=snapshot_id,
        observation=observation,
    )


def restore(snapshot: StateSnapshot) -> StateSnapshot:
    try:
        return _snapshot(snapshot)
    except Exception as error:
        _integrity("SNAPSHOT_INVALID", "snapshot binding or structure is invalid", error)


def _prepared_expected(
    initial: StateSnapshot,
    actions: tuple[ActionCall, ...],
    expected_events: Sequence[StateEvent] | None,
    expected_terminal: TerminalRecord | None,
) -> tuple[tuple[StateEvent, ...] | None, TerminalRecord | None]:
    if (expected_events is None) != (expected_terminal is None):
        _integrity(
            "EXPECTED_PAIR_REQUIRED", "expected events and terminal must be supplied together"
        )
    if expected_events is None or expected_terminal is None:
        return None, None
    try:
        events = _events(expected_events)
        terminal = _terminal(expected_terminal)
        _validate_expected_static(initial, actions, events, terminal)
        return events, terminal
    except Exception as error:
        _integrity("EXPECTED_ARTIFACT_INVALID", "expected artifacts fail static validation", error)


def _verified_verdict(verifier: ReplayVerifier, final_state: WorldState) -> OutcomeVerdict:
    verifier_state = _world_state(final_state)
    before_hash = hash_state(verifier_state)
    try:
        value = verifier(verifier_state)
    except Exception as error:
        _environment("VERIFIER_EXCEPTION", "verifier raised", error)
    try:
        after_hash = hash_state(verifier_state)
    except Exception as error:
        _environment("VERIFIER_INPUT_MUTATION", "verifier input became invalid", error)
    if before_hash != after_hash:
        _environment("VERIFIER_INPUT_MUTATION", "verifier changed its input")
    try:
        return _verdict(value)
    except Exception as error:
        _environment("VERIFIER_INVALID", "verifier returned an invalid verdict", error)


def replay(
    initial: StateSnapshot,
    actions: Sequence[ActionCall],
    adapter: DomainAdapter,
    *,
    verifier: ReplayVerifier,
    expected_events: Sequence[StateEvent] | None = None,
    expected_terminal: TerminalRecord | None = None,
) -> ReplayResult:
    restored_initial = restore(initial)
    try:
        owned_actions = _actions(actions)
    except Exception as error:
        _integrity("ACTIONS_INVALID", "actions are not an exact ordered sequence", error)
    expected = _prepared_expected(
        restored_initial,
        owned_actions,
        expected_events,
        expected_terminal,
    )
    current_state = _world_state(restored_initial.state)
    cursor = EventCursor(episode_id=restored_initial.episode_id, next_index=0)
    events: list[StateEvent] = []
    for index, action in enumerate(owned_actions):
        step = execute_action(current_state, action, adapter, cursor=cursor)
        try:
            generated_event = _event(step.event)
            current_state = _world_state(step.state)
            cursor = _cursor(step.cursor)
            if (
                cursor.episode_id != generated_event.episode_id
                or cursor.next_index != generated_event.index + 1
                or cursor.parent_event_id != generated_event.event_id
                or cursor.parent_event_hash != generated_event.event_hash
            ):
                raise ValueError("returned cursor does not bind generated event")
        except Exception as error:
            _environment("REPLAY_STEP_INVALID", "atomic step violates replay contract", error)
        expected_events_owned, _ = expected
        if expected_events_owned is not None and not _same_event(
            generated_event, expected_events_owned[index]
        ):
            _integrity("EXPECTED_EVENT_MISMATCH", "generated event differs from expected artifact")
        events.append(generated_event)
    verdict = _verified_verdict(verifier, current_state)
    final_observation = events[-1].result.observation if events else restored_initial.observation
    try:
        terminal = make_terminal_record(
            episode_id=restored_initial.episode_id,
            initial_state_hash=restored_initial.state_hash,
            terminal_state_hash=hash_state(current_state),
            event_root_hash=events[-1].event_hash if events else None,
            final_observation=final_observation,
            verdict=verdict,
        )
    except Exception as error:
        _environment("TERMINAL_INVALID", "terminal construction failed", error)
    _, expected_terminal_owned = expected
    if expected_terminal_owned is not None and not _same_terminal(
        terminal, expected_terminal_owned
    ):
        _integrity(
            "EXPECTED_TERMINAL_MISMATCH", "generated terminal differs from expected artifact"
        )
    try:
        return ReplayResult(
            initial=restored_initial,
            actions=owned_actions,
            events=tuple(events),
            terminal_state=current_state,
            terminal=terminal,
        )
    except Exception as error:
        _environment("REPLAY_CROSS_LINK_INVALID", "generated replay links are invalid", error)
