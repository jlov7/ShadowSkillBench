from __future__ import annotations

import re
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from shadowskillbench.core.hashing import CanonicalizationError, canonical_json_bytes, sha256_ref

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type PathSafeId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"),
]
type Sha256Ref = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
type ErrorCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$")]
type NonNegativeIndex = Annotated[int, Field(ge=0)]

_POINTER_ESCAPE = re.compile(r"~(?:[^01]|$)")
_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")


class _EngineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _required_model_fields(
    value: BaseModel, *, model_name: str, fields: tuple[str, ...]
) -> dict[str, object]:
    try:
        return {field: getattr(value, field) for field in fields}
    except AttributeError as error:
        raise ValueError(f"{model_name} is incomplete") from error


def _owned_json_value(value: object) -> JsonValue:
    try:
        owned = _copy_exact_json(value, active_container_ids=set())
    except RecursionError as error:
        raise ValueError("JSON value exceeds supported nesting") from error
    try:
        canonical_json_bytes(owned)
    except CanonicalizationError as error:
        raise ValueError("value is not strict SSB-CJ1 JSON") from error
    return owned


def _copy_exact_json(value: object, *, active_container_ids: set[int]) -> JsonValue:
    value_type = type(value)
    if value_type in {type(None), bool, int, float, str}:
        return cast(JsonValue, value)
    if value_type is list:
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError("JSON value cannot be cyclic")
        active_container_ids.add(container_id)
        try:
            return [
                _copy_exact_json(item, active_container_ids=active_container_ids)
                for item in cast(list[object], value)
            ]
        finally:
            active_container_ids.remove(container_id)
    if value_type is dict:
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError("JSON value cannot be cyclic")
        active_container_ids.add(container_id)
        try:
            copied: dict[str, JsonValue] = {}
            for key, item in cast(dict[object, object], value).items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                copied[key] = _copy_exact_json(item, active_container_ids=active_container_ids)
            return copied
        finally:
            active_container_ids.remove(container_id)
    raise ValueError("value must contain exact built-in JSON types")


def _owned_json_object(value: object) -> JsonObject:
    if type(value) is not dict:
        raise ValueError("value must be a JSON object")
    return cast(JsonObject, _owned_json_value(value))


def _validated_path_safe_id(value: object, *, field: str) -> PathSafeId:
    if type(value) is not str or _PATH_SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact path-safe identifier")
    return cast(PathSafeId, value)


def _validated_nonnegative_index(value: object, *, field: str) -> NonNegativeIndex:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an exact nonnegative integer")
    return cast(NonNegativeIndex, value)


def _validated_sha256_ref(value: object, *, field: str) -> Sha256Ref:
    if type(value) is not str or _SHA256_REF.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact sha256 reference")
    return cast(Sha256Ref, value)


def _validated_optional_sha256_ref(value: object, *, field: str) -> Sha256Ref | None:
    if value is None:
        return None
    return _validated_sha256_ref(value, field=field)


def _validate_detail(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not value
        or len(value) > 240
        or any(ord(character) < 32 or ord(character) > 126 for character in value)
    ):
        raise ValueError("error_detail must be bounded single-line printable metadata")
    return value


def _validate_json_string(value: str, *, field: str) -> str:
    try:
        canonical_json_bytes(value)
    except CanonicalizationError as error:
        raise ValueError(f"{field} must be strict SSB-CJ1 text") from error
    return value


def _pointer_tokens(path: str) -> tuple[str, ...]:
    if path == "":
        return ()
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/"))


def _overlap(left: str, right: str) -> bool:
    left_tokens = _pointer_tokens(left)
    right_tokens = _pointer_tokens(right)
    return (
        left_tokens[: len(right_tokens)] == right_tokens
        or right_tokens[: len(left_tokens)] == left_tokens
    )


class TaskCase(_EngineModel):
    case_id: PathSafeId
    domain: PathSafeId
    world_id: PathSafeId
    seed: int
    objective: str
    inputs: JsonObject

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("objective must be nonblank")
        return _validate_json_string(value, field="objective")

    @field_validator("inputs", mode="before")
    @classmethod
    def own_inputs(cls, value: object) -> JsonObject:
        return _owned_json_object(value)


class WorldState(_EngineModel):
    schema_version: Literal["1.0"]
    world_id: PathSafeId
    domain: PathSafeId
    seed: int
    data: JsonObject

    @field_validator("data", mode="before")
    @classmethod
    def own_data(cls, value: object) -> JsonObject:
        return _owned_json_object(value)


def _world_state_input(value: object) -> object:
    if isinstance(value, WorldState):
        if type(value) is not WorldState:
            raise ValueError("state must be an exact WorldState instance")
        return _required_model_fields(
            value,
            model_name="WorldState",
            fields=("schema_version", "world_id", "domain", "seed", "data"),
        )
    return value


def _validated_world_state(value: object) -> WorldState:
    return WorldState.model_validate(_world_state_input(value))


class StateSnapshot(_EngineModel):
    snapshot_id: PathSafeId
    episode_id: PathSafeId
    state: WorldState
    state_hash: Sha256Ref
    observation: JsonObject
    observation_hash: Sha256Ref

    @field_validator("state", mode="before")
    @classmethod
    def own_state(cls, value: object) -> object:
        return _world_state_input(value)

    @field_validator("observation", mode="before")
    @classmethod
    def own_observation(cls, value: object) -> JsonObject:
        return _owned_json_object(value)

    @model_validator(mode="after")
    def bind_hashes(self) -> StateSnapshot:
        if self.state_hash != _hash_state_trusted(self.state):
            raise ValueError("state_hash does not bind state")
        if self.observation_hash != _hash_observation_trusted(self.state_hash, self.observation):
            raise ValueError("observation_hash does not bind state and observation")
        return self


class ActionCall(_EngineModel):
    action_id: PathSafeId
    tool_name: PathSafeId
    arguments: JsonObject

    @field_validator("arguments", mode="before")
    @classmethod
    def own_arguments(cls, value: object) -> JsonObject:
        return _owned_json_object(value)


def _action_call_input(value: object) -> object:
    if isinstance(value, ActionCall):
        if type(value) is not ActionCall:
            raise ValueError("action must be an exact ActionCall instance")
        return _required_model_fields(
            value,
            model_name="ActionCall",
            fields=("action_id", "tool_name", "arguments"),
        )
    return value


def _validated_action_call(value: object) -> ActionCall:
    return ActionCall.model_validate(_action_call_input(value))


class ActionResult(_EngineModel):
    local_status: Literal["success", "failure", "clarify"]
    observation: JsonObject
    error_code: ErrorCode | None = None
    error_detail: str | None = None

    @field_validator("observation", mode="before")
    @classmethod
    def own_observation(cls, value: object) -> JsonObject:
        return _owned_json_object(value)

    @field_validator("error_detail")
    @classmethod
    def validate_error_detail(cls, value: str | None) -> str | None:
        return _validate_detail(value)

    @model_validator(mode="after")
    def validate_status(self) -> ActionResult:
        if self.local_status == "success":
            if self.error_code is not None or self.error_detail is not None:
                raise ValueError("successful result cannot carry error metadata")
        elif self.error_code is None:
            raise ValueError("failure and clarification results require error_code")
        return self


def _action_result_input(value: object) -> object:
    if isinstance(value, ActionResult):
        if type(value) is not ActionResult:
            raise ValueError("result must be an exact ActionResult instance")
        return _required_model_fields(
            value,
            model_name="ActionResult",
            fields=("local_status", "observation", "error_code", "error_detail"),
        )
    return value


def _validated_action_result(value: object) -> ActionResult:
    return ActionResult.model_validate(_action_result_input(value))


class StatePatch(_EngineModel):
    op: Literal["add", "replace", "remove"]
    path: str
    before_present: bool
    before: JsonValue | None = None
    after_present: bool
    after: JsonValue | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        _validate_json_string(value, field="path")
        if value != "" and not value.startswith("/"):
            raise ValueError("path must be an RFC 6901 pointer")
        if _POINTER_ESCAPE.search(value):
            raise ValueError("path contains an invalid RFC 6901 escape")
        return value

    @field_validator("before", "after", mode="before")
    @classmethod
    def own_value(cls, value: object) -> JsonValue | None:
        return None if value is None else _owned_json_value(value)

    @model_validator(mode="after")
    def validate_operation(self) -> StatePatch:
        if not self.before_present and self.before is not None:
            raise ValueError("absent before value must be null")
        if not self.after_present and self.after is not None:
            raise ValueError("absent after value must be null")
        expected = {
            "add": (False, True),
            "replace": (True, True),
            "remove": (True, False),
        }[self.op]
        if (self.before_present, self.after_present) != expected:
            raise ValueError("patch presence flags contradict operation")
        return self


def _state_patch_input(value: object) -> object:
    if isinstance(value, StatePatch):
        if type(value) is not StatePatch:
            raise ValueError("patch must be an exact StatePatch instance")
        return _required_model_fields(
            value,
            model_name="StatePatch",
            fields=("op", "path", "before_present", "before", "after_present", "after"),
        )
    return value


def _validated_state_patches(value: object) -> tuple[StatePatch, ...]:
    if type(value) is not tuple:
        raise ValueError("delta must be a tuple")
    return tuple(StatePatch.model_validate(_state_patch_input(item)) for item in value)


class TransitionProposal(_EngineModel):
    local_status: Literal["success", "failure", "clarify"]
    next_state: WorldState | None = None
    observation: JsonObject
    error_code: ErrorCode | None = None
    error_detail: str | None = None

    @field_validator("next_state", mode="before")
    @classmethod
    def own_next_state(cls, value: object) -> object:
        return _world_state_input(value)

    @field_validator("observation", mode="before")
    @classmethod
    def own_observation(cls, value: object) -> JsonObject:
        return _owned_json_object(value)

    @field_validator("error_detail")
    @classmethod
    def validate_error_detail(cls, value: str | None) -> str | None:
        return _validate_detail(value)

    @model_validator(mode="after")
    def validate_status(self) -> TransitionProposal:
        if self.local_status == "success":
            if self.next_state is None:
                raise ValueError("successful proposal requires next_state")
            if self.error_code is not None or self.error_detail is not None:
                raise ValueError("successful proposal cannot carry error metadata")
        elif self.next_state is not None or self.error_code is None:
            raise ValueError(
                "failure and clarification proposals require error_code and no next_state"
            )
        return self


class EventCursor(_EngineModel):
    episode_id: PathSafeId
    next_index: NonNegativeIndex
    parent_event_id: PathSafeId | None = None
    parent_event_hash: Sha256Ref | None = None

    @model_validator(mode="after")
    def validate_parent(self) -> EventCursor:
        if self.next_index == 0:
            if self.parent_event_id is not None or self.parent_event_hash is not None:
                raise ValueError("first cursor cannot have a parent")
        elif self.parent_event_id is None or self.parent_event_hash is None:
            raise ValueError("non-first cursor requires parent identity and hash")
        elif self.parent_event_id != _event_id(self.episode_id, self.next_index - 1):
            raise ValueError("cursor parent event id does not match episode sequence")
        return self


def _event_cursor_input(value: object) -> object:
    if isinstance(value, EventCursor):
        if type(value) is not EventCursor:
            raise ValueError("cursor must be an exact EventCursor instance")
        return _required_model_fields(
            value,
            model_name="EventCursor",
            fields=("episode_id", "next_index", "parent_event_id", "parent_event_hash"),
        )
    return value


def _validated_event_cursor(value: object) -> EventCursor:
    return EventCursor.model_validate(_event_cursor_input(value))


class StateEvent(_EngineModel):
    schema_version: Literal["1.0"]
    episode_id: PathSafeId
    event_id: PathSafeId
    index: NonNegativeIndex
    parent_event_id: PathSafeId | None
    parent_event_hash: Sha256Ref | None
    action_id: PathSafeId
    action_hash: Sha256Ref
    result: ActionResult
    before_hash: Sha256Ref
    after_hash: Sha256Ref
    delta: tuple[StatePatch, ...]
    event_hash: Sha256Ref

    @field_validator("result", mode="before")
    @classmethod
    def own_result(cls, value: object) -> object:
        return _action_result_input(value)

    @field_validator("delta", mode="before")
    @classmethod
    def own_delta(cls, value: object) -> object:
        if type(value) is tuple:
            return tuple(_state_patch_input(item) for item in value)
        return value

    @model_validator(mode="after")
    def validate_event(self) -> StateEvent:
        if self.event_id != _event_id(self.episode_id, self.index):
            raise ValueError("event_id does not match episode and index")
        if self.index == 0:
            if self.parent_event_id is not None or self.parent_event_hash is not None:
                raise ValueError("first event cannot have a parent")
        elif self.parent_event_id is None or self.parent_event_hash is None:
            raise ValueError("non-first event requires parent identity and hash")
        elif self.parent_event_id != _event_id(self.episode_id, self.index - 1):
            raise ValueError("parent event id does not match episode sequence")
        if self.result.local_status in {"failure", "clarify"}:
            if self.before_hash != self.after_hash or self.delta:
                raise ValueError("failure and clarification events cannot change state")
        elif (self.before_hash != self.after_hash) != bool(self.delta):
            raise ValueError("success event delta must exactly match state change")
        paths = [patch.path for patch in self.delta]
        if any(
            _overlap(path, other)
            for position, path in enumerate(paths)
            for other in paths[position + 1 :]
        ):
            raise ValueError("event delta contains duplicate or overlapping paths")
        if self.event_hash != _hash_event_trusted(self):
            raise ValueError("event_hash does not bind event body")
        return self


def _state_event_input(value: object) -> object:
    if isinstance(value, StateEvent):
        if type(value) is not StateEvent:
            raise ValueError("event must be an exact StateEvent instance")
        fields = _required_model_fields(
            value,
            model_name="StateEvent",
            fields=(
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
        return {
            **fields,
            "result": _action_result_input(fields["result"]),
            "delta": tuple(_state_patch_input(item) for item in delta)
            if type(delta) is tuple
            else delta,
        }
    return value


def _validated_state_event(value: object) -> StateEvent:
    return StateEvent.model_validate(_state_event_input(value))


class OutcomeVerdict(_EngineModel):
    status: Literal["PASS", "FAIL", "ESCALATE"]
    reason_code: ErrorCode
    details: JsonObject

    @field_validator("details", mode="before")
    @classmethod
    def own_details(cls, value: object) -> JsonObject:
        return _owned_json_object(value)


def _outcome_verdict_input(value: object) -> object:
    if isinstance(value, OutcomeVerdict):
        if type(value) is not OutcomeVerdict:
            raise ValueError("verdict must be an exact OutcomeVerdict instance")
        return _required_model_fields(
            value,
            model_name="OutcomeVerdict",
            fields=("status", "reason_code", "details"),
        )
    return value


def _validated_outcome_verdict(value: object) -> OutcomeVerdict:
    return OutcomeVerdict.model_validate(_outcome_verdict_input(value))


class TerminalRecord(_EngineModel):
    episode_id: PathSafeId
    initial_state_hash: Sha256Ref
    terminal_state_hash: Sha256Ref
    event_root_hash: Sha256Ref | None = None
    final_observation_hash: Sha256Ref
    verdict: OutcomeVerdict
    verdict_hash: Sha256Ref

    @field_validator("verdict", mode="before")
    @classmethod
    def own_verdict(cls, value: object) -> object:
        return _outcome_verdict_input(value)

    @model_validator(mode="after")
    def bind_hash(self) -> TerminalRecord:
        if self.verdict_hash != _hash_terminal_trusted(self):
            raise ValueError("verdict_hash does not bind terminal body")
        return self


def _terminal_record_input(value: object) -> object:
    if isinstance(value, TerminalRecord):
        if type(value) is not TerminalRecord:
            raise ValueError("terminal must be an exact TerminalRecord instance")
        fields = _required_model_fields(
            value,
            model_name="TerminalRecord",
            fields=(
                "episode_id",
                "initial_state_hash",
                "terminal_state_hash",
                "event_root_hash",
                "final_observation_hash",
                "verdict",
                "verdict_hash",
            ),
        )
        return {
            **fields,
            "verdict": _outcome_verdict_input(fields["verdict"]),
        }
    return value


def _validated_terminal_record(value: object) -> TerminalRecord:
    return TerminalRecord.model_validate(_terminal_record_input(value))


def _state_projection_trusted(state: WorldState) -> JsonObject:
    return {
        "schema_version": state.schema_version,
        "world_id": state.world_id,
        "domain": state.domain,
        "seed": state.seed,
        "data": state.data,
    }


def _hash_state_trusted(state: WorldState) -> Sha256Ref:
    return sha256_ref(_state_projection_trusted(state))


def state_projection(state: WorldState) -> JsonObject:
    return _state_projection_trusted(_validated_world_state(state))


def hash_state(state: WorldState) -> Sha256Ref:
    return _hash_state_trusted(_validated_world_state(state))


def _observation_projection_trusted(state_hash: Sha256Ref, observation: JsonObject) -> JsonObject:
    return {"state_hash": state_hash, "observation": observation}


def _hash_observation_trusted(state_hash: Sha256Ref, observation: JsonObject) -> Sha256Ref:
    return sha256_ref(_observation_projection_trusted(state_hash, observation))


def observation_projection(state_hash: Sha256Ref, observation: JsonObject) -> JsonObject:
    owned_state_hash = _validated_sha256_ref(state_hash, field="state_hash")
    owned_observation = _owned_json_object(observation)
    return _observation_projection_trusted(owned_state_hash, owned_observation)


def hash_observation(state_hash: Sha256Ref, observation: JsonObject) -> Sha256Ref:
    owned_state_hash = _validated_sha256_ref(state_hash, field="state_hash")
    owned_observation = _owned_json_object(observation)
    return _hash_observation_trusted(owned_state_hash, owned_observation)


def _action_projection_trusted(action: ActionCall) -> JsonObject:
    return {
        "action_id": action.action_id,
        "tool_name": action.tool_name,
        "arguments": action.arguments,
    }


def _hash_action_trusted(action: ActionCall) -> Sha256Ref:
    return sha256_ref(_action_projection_trusted(action))


def action_projection(action: ActionCall) -> JsonObject:
    return _action_projection_trusted(_validated_action_call(action))


def hash_action(action: ActionCall) -> Sha256Ref:
    return _hash_action_trusted(_validated_action_call(action))


def _result_projection_trusted(result: ActionResult) -> JsonObject:
    return {
        "local_status": result.local_status,
        "observation": result.observation,
        "error_code": result.error_code,
        "error_detail": result.error_detail,
    }


def _hash_result_trusted(result: ActionResult) -> Sha256Ref:
    return sha256_ref(_result_projection_trusted(result))


def result_projection(result: ActionResult) -> JsonObject:
    return _result_projection_trusted(_validated_action_result(result))


def hash_result(result: ActionResult) -> Sha256Ref:
    return _hash_result_trusted(_validated_action_result(result))


def _state_patch_projection_trusted(patch: StatePatch) -> JsonObject:
    return {
        "op": patch.op,
        "path": patch.path,
        "before_present": patch.before_present,
        "before": patch.before,
        "after_present": patch.after_present,
        "after": patch.after,
    }


def _event_projection_trusted(event: StateEvent) -> JsonObject:
    return {
        "schema_version": event.schema_version,
        "episode_id": event.episode_id,
        "event_id": event.event_id,
        "index": event.index,
        "parent_event_id": event.parent_event_id,
        "parent_event_hash": event.parent_event_hash,
        "action_id": event.action_id,
        "action_hash": event.action_hash,
        "result": _result_projection_trusted(event.result),
        "before_hash": event.before_hash,
        "after_hash": event.after_hash,
        "delta": [_state_patch_projection_trusted(patch) for patch in event.delta],
    }


def _hash_event_trusted(event: StateEvent) -> Sha256Ref:
    return sha256_ref(_event_projection_trusted(event))


def event_projection(event: StateEvent) -> JsonObject:
    return _event_projection_trusted(_validated_state_event(event))


def hash_event(event: StateEvent) -> Sha256Ref:
    return _hash_event_trusted(_validated_state_event(event))


def _outcome_verdict_projection_trusted(verdict: OutcomeVerdict) -> JsonObject:
    return {
        "status": verdict.status,
        "reason_code": verdict.reason_code,
        "details": verdict.details,
    }


def _terminal_projection_trusted(terminal: TerminalRecord) -> JsonObject:
    return {
        "episode_id": terminal.episode_id,
        "initial_state_hash": terminal.initial_state_hash,
        "terminal_state_hash": terminal.terminal_state_hash,
        "event_root_hash": terminal.event_root_hash,
        "final_observation_hash": terminal.final_observation_hash,
        "verdict": _outcome_verdict_projection_trusted(terminal.verdict),
    }


def _hash_terminal_trusted(terminal: TerminalRecord) -> Sha256Ref:
    return sha256_ref(_terminal_projection_trusted(terminal))


def terminal_projection(terminal: TerminalRecord) -> JsonObject:
    return _terminal_projection_trusted(_validated_terminal_record(terminal))


def hash_terminal(terminal: TerminalRecord) -> Sha256Ref:
    return _hash_terminal_trusted(_validated_terminal_record(terminal))


def make_snapshot(
    state: WorldState,
    *,
    episode_id: PathSafeId,
    snapshot_id: PathSafeId,
    observation: JsonObject,
) -> StateSnapshot:
    owned_episode_id = _validated_path_safe_id(episode_id, field="episode_id")
    owned_snapshot_id = _validated_path_safe_id(snapshot_id, field="snapshot_id")
    owned_state = _validated_world_state(state)
    owned_observation = _owned_json_object(observation)
    state_hash = _hash_state_trusted(owned_state)
    return StateSnapshot(
        snapshot_id=owned_snapshot_id,
        episode_id=owned_episode_id,
        state=owned_state,
        state_hash=state_hash,
        observation=owned_observation,
        observation_hash=_hash_observation_trusted(state_hash, owned_observation),
    )


def _make_state_event_with_owned_referents(
    cursor: EventCursor,
    action: ActionCall,
    result: ActionResult,
    before_state: WorldState,
    after_state: WorldState,
    delta: tuple[StatePatch, ...],
) -> StateEvent:
    event_id = _event_id(cursor.episode_id, cursor.next_index)
    parent_event_id = cursor.parent_event_id
    parent_event_hash = cursor.parent_event_hash
    action_hash = _hash_action_trusted(action)
    before_hash = _hash_state_trusted(before_state)
    after_hash = _hash_state_trusted(after_state)
    event_body: JsonObject = {
        "schema_version": "1.0",
        "episode_id": cursor.episode_id,
        "event_id": event_id,
        "index": cursor.next_index,
        "parent_event_id": parent_event_id,
        "parent_event_hash": parent_event_hash,
        "action_id": action.action_id,
        "action_hash": action_hash,
        "result": _result_projection_trusted(result),
        "before_hash": before_hash,
        "after_hash": after_hash,
        "delta": [_state_patch_projection_trusted(patch) for patch in delta],
    }
    return StateEvent(
        schema_version="1.0",
        episode_id=cursor.episode_id,
        event_id=event_id,
        index=cursor.next_index,
        parent_event_id=parent_event_id,
        parent_event_hash=parent_event_hash,
        action_id=action.action_id,
        action_hash=action_hash,
        result=result,
        before_hash=before_hash,
        after_hash=after_hash,
        delta=delta,
        event_hash=sha256_ref(event_body),
    )


def make_state_event_from_cursor(
    *,
    cursor: EventCursor,
    action: ActionCall,
    result: ActionResult,
    before_state: WorldState,
    after_state: WorldState,
    delta: tuple[StatePatch, ...],
) -> StateEvent:
    return _make_state_event_with_owned_referents(
        _validated_event_cursor(cursor),
        _validated_action_call(action),
        _validated_action_result(result),
        _validated_world_state(before_state),
        _validated_world_state(after_state),
        _validated_state_patches(delta),
    )


def make_state_event(
    *,
    episode_id: PathSafeId,
    index: NonNegativeIndex,
    action: ActionCall,
    result: ActionResult,
    before_state: WorldState,
    after_state: WorldState,
    delta: tuple[StatePatch, ...],
    parent_event: StateEvent | None = None,
) -> StateEvent:
    owned_episode_id = _validated_path_safe_id(episode_id, field="episode_id")
    owned_index = _validated_nonnegative_index(index, field="index")
    if owned_index == 0:
        if parent_event is not None:
            raise ValueError("first event cannot receive a parent")
        cursor = EventCursor(episode_id=owned_episode_id, next_index=owned_index)
    else:
        if parent_event is None:
            raise ValueError("non-first event requires a parent")
        owned_parent = _validated_state_event(parent_event)
        if owned_parent.episode_id != owned_episode_id or owned_parent.index != owned_index - 1:
            raise ValueError("event parent does not match episode sequence")
        cursor = EventCursor(
            episode_id=owned_episode_id,
            next_index=owned_index,
            parent_event_id=owned_parent.event_id,
            parent_event_hash=owned_parent.event_hash,
        )
    return make_state_event_from_cursor(
        cursor=cursor,
        action=action,
        result=result,
        before_state=before_state,
        after_state=after_state,
        delta=delta,
    )


def make_terminal_record(
    *,
    episode_id: PathSafeId,
    initial_state_hash: Sha256Ref,
    terminal_state_hash: Sha256Ref,
    event_root_hash: Sha256Ref | None,
    final_observation: JsonObject,
    verdict: OutcomeVerdict,
) -> TerminalRecord:
    owned_episode_id = _validated_path_safe_id(episode_id, field="episode_id")
    owned_initial_state_hash = _validated_sha256_ref(initial_state_hash, field="initial_state_hash")
    owned_terminal_state_hash = _validated_sha256_ref(
        terminal_state_hash, field="terminal_state_hash"
    )
    owned_event_root_hash = _validated_optional_sha256_ref(event_root_hash, field="event_root_hash")
    owned_observation = _owned_json_object(final_observation)
    owned_verdict = _validated_outcome_verdict(verdict)
    final_observation_hash = _hash_observation_trusted(owned_terminal_state_hash, owned_observation)
    terminal_body: JsonObject = {
        "episode_id": owned_episode_id,
        "initial_state_hash": owned_initial_state_hash,
        "terminal_state_hash": owned_terminal_state_hash,
        "event_root_hash": owned_event_root_hash,
        "final_observation_hash": final_observation_hash,
        "verdict": _outcome_verdict_projection_trusted(owned_verdict),
    }
    return TerminalRecord(
        episode_id=owned_episode_id,
        initial_state_hash=owned_initial_state_hash,
        terminal_state_hash=owned_terminal_state_hash,
        event_root_hash=owned_event_root_hash,
        final_observation_hash=final_observation_hash,
        verdict=owned_verdict,
        verdict_hash=sha256_ref(terminal_body),
    )


def _event_id(episode_id: PathSafeId, index: NonNegativeIndex) -> str:
    return f"event_{episode_id}_{index:06d}"
