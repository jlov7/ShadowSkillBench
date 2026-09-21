from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NamedTuple, Protocol, cast

from shadowskillbench.core.hashing import CanonicalizationError, canonical_json_bytes
from shadowskillbench.engine.models import (
    ActionCall,
    ActionResult,
    EventCursor,
    JsonObject,
    JsonValue,
    StateEvent,
    StatePatch,
    TransitionProposal,
    WorldState,
    hash_action,
    hash_state,
    make_state_event_from_cursor,
)

_ARRAY_INDEX = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class EnvironmentInvariantFailure(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class PatchApplicationError(EnvironmentInvariantFailure):
    pass


class DomainAdapter(Protocol):
    def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal: ...


class ExecutionStep(NamedTuple):
    state: WorldState
    result: ActionResult
    event: StateEvent
    cursor: EventCursor


def _invariant(
    code: str, detail: str, cause: Exception | None = None
) -> EnvironmentInvariantFailure:
    error = EnvironmentInvariantFailure(code, detail)
    if cause is not None:
        raise error from cause
    return error


def _patch_error(code: str, detail: str, cause: Exception | None = None) -> PatchApplicationError:
    error = PatchApplicationError(code, detail)
    if cause is not None:
        raise error from cause
    return error


def _model_fields(
    value: object,
    model_type: type[object],
    model_name: str,
    fields: tuple[str, ...],
    *,
    code: str,
) -> dict[str, object]:
    if type(value) is not model_type:
        raise _invariant(code, f"{model_name} must be an exact validated model")
    try:
        return {field: getattr(value, field) for field in fields}
    except AttributeError as error:
        raise _invariant(code, f"{model_name} is incomplete", error)


def _validated_world_state(value: object, *, code: str = "STATE_INVALID") -> WorldState:
    fields = _model_fields(
        value,
        WorldState,
        "WorldState",
        ("schema_version", "world_id", "domain", "seed", "data"),
        code=code,
    )
    try:
        return WorldState.model_validate(fields)
    except Exception as error:
        raise _invariant(code, "WorldState failed revalidation", error)


def _validated_action_call(value: object, *, code: str = "ACTION_INVALID") -> ActionCall:
    fields = _model_fields(
        value,
        ActionCall,
        "ActionCall",
        ("action_id", "tool_name", "arguments"),
        code=code,
    )
    try:
        return ActionCall.model_validate(fields)
    except Exception as error:
        raise _invariant(code, "ActionCall failed revalidation", error)


def _validated_cursor(value: object) -> EventCursor:
    fields = _model_fields(
        value,
        EventCursor,
        "EventCursor",
        ("episode_id", "next_index", "parent_event_id", "parent_event_hash"),
        code="CURSOR_INVALID",
    )
    try:
        return EventCursor.model_validate(fields)
    except Exception as error:
        raise _invariant("CURSOR_INVALID", "EventCursor failed revalidation", error)


def _validated_proposal(value: object) -> TransitionProposal:
    fields = _model_fields(
        value,
        TransitionProposal,
        "TransitionProposal",
        ("local_status", "next_state", "observation", "error_code", "error_detail"),
        code="PROPOSAL_INVALID",
    )
    try:
        return TransitionProposal.model_validate(fields)
    except Exception as error:
        raise _invariant("PROPOSAL_INVALID", "TransitionProposal failed revalidation", error)


def _copy_json_value(value: object, active: set[int]) -> JsonValue:
    value_type = type(value)
    if value_type in {type(None), bool, int, float, str}:
        return cast(JsonValue, value)
    if value_type is list:
        value_id = id(value)
        if value_id in active:
            raise _patch_error("PATCH_INVALID", "JSON input is cyclic")
        active.add(value_id)
        try:
            return [_copy_json_value(item, active) for item in cast(list[object], value)]
        finally:
            active.remove(value_id)
    if value_type is dict:
        value_id = id(value)
        if value_id in active:
            raise _patch_error("PATCH_INVALID", "JSON input is cyclic")
        active.add(value_id)
        try:
            copied: dict[str, JsonValue] = {}
            for key, item in cast(dict[object, object], value).items():
                if type(key) is not str:
                    raise _patch_error("PATCH_INVALID", "JSON object keys must be strings")
                copied[key] = _copy_json_value(item, active)
            return copied
        finally:
            active.remove(value_id)
    raise _patch_error("PATCH_INVALID", "JSON input must use exact built-in types")


def _owned_json_object(value: object) -> JsonObject:
    if type(value) is not dict:
        raise _patch_error("PATCH_INVALID", "state data must be a JSON object")
    try:
        copied = cast(JsonObject, _copy_json_value(value, set()))
        canonical_json_bytes(copied)
        return copied
    except (CanonicalizationError, RecursionError) as error:
        raise _patch_error("PATCH_INVALID", "state data is not strict SSB-CJ1 JSON", error)


def _json_equal(left: JsonValue, right: JsonValue) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (CanonicalizationError, RecursionError) as error:
        raise _patch_error("PATCH_INVALID", "patch comparison is not strict JSON", error)


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _join(path: str, token: str) -> str:
    return f"{path}/{_escape(token)}"


def _diff_values(before: JsonValue, after: JsonValue, path: str, out: list[StatePatch]) -> None:
    if _json_equal(before, after):
        return
    if type(before) is dict and type(after) is dict:
        before_object = cast(JsonObject, before)
        after_object = cast(JsonObject, after)
        for key in sorted(set(before_object) | set(after_object)):
            child_path = _join(path, key)
            if key not in before_object:
                out.append(
                    StatePatch(
                        op="add",
                        path=child_path,
                        before_present=False,
                        after_present=True,
                        after=after_object[key],
                    )
                )
            elif key not in after_object:
                out.append(
                    StatePatch(
                        op="remove",
                        path=child_path,
                        before_present=True,
                        before=before_object[key],
                        after_present=False,
                    )
                )
            else:
                _diff_values(before_object[key], after_object[key], child_path, out)
        return
    if type(before) is list and type(after) is list:
        before_array = cast(list[JsonValue], before)
        after_array = cast(list[JsonValue], after)
        if len(before_array) != len(after_array):
            out.append(
                StatePatch(
                    op="replace",
                    path=path,
                    before_present=True,
                    before=before_array,
                    after_present=True,
                    after=after_array,
                )
            )
            return
        for index, (before_item, after_item) in enumerate(
            zip(before_array, after_array, strict=True)
        ):
            _diff_values(before_item, after_item, _join(path, str(index)), out)
        return
    out.append(
        StatePatch(
            op="replace",
            path=path,
            before_present=True,
            before=before,
            after_present=True,
            after=after,
        )
    )


def diff_state_data(before: JsonObject, after: JsonObject) -> tuple[StatePatch, ...]:
    try:
        owned_before = _owned_json_object(before)
        owned_after = _owned_json_object(after)
        patches: list[StatePatch] = []
        _diff_values(owned_before, owned_after, "", patches)
        return tuple(patches)
    except RecursionError as error:
        raise _patch_error("PATCH_INVALID", "state data exceeds supported nesting", error)


def _decoded_pointer(path: object) -> tuple[str, ...]:
    if type(path) is not str:
        raise _patch_error("PATCH_PATH_INVALID", "patch path must be an exact string")
    if path == "":
        return ()
    if not path.startswith("/"):
        raise _patch_error("PATCH_PATH_INVALID", "patch path must start with slash")
    decoded: list[str] = []
    for token in path[1:].split("/"):
        characters: list[str] = []
        position = 0
        while position < len(token):
            character = token[position]
            if character != "~":
                characters.append(character)
                position += 1
                continue
            if position + 1 == len(token) or token[position + 1] not in {"0", "1"}:
                raise _patch_error("PATCH_PATH_INVALID", "patch path has an invalid escape")
            characters.append("~" if token[position + 1] == "0" else "/")
            position += 2
        decoded.append("".join(characters))
    return tuple(decoded)


def _revalidated_patch(value: object) -> StatePatch:
    fields = _model_fields(
        value,
        StatePatch,
        "StatePatch",
        ("op", "path", "before_present", "before", "after_present", "after"),
        code="PATCH_INVALID",
    )
    try:
        return StatePatch.model_validate(fields)
    except Exception as error:
        raise _patch_error("PATCH_INVALID", "StatePatch failed revalidation", error)


def _validate_patch_paths(patches: tuple[StatePatch, ...]) -> tuple[tuple[str, ...], ...]:
    decoded = tuple(_decoded_pointer(patch.path) for patch in patches)
    for position, current in enumerate(decoded):
        for other in decoded[position + 1 :]:
            if (
                current == other
                or current[: len(other)] == other
                or other[: len(current)] == current
            ):
                raise _patch_error("PATCH_PATH_INVALID", "patch paths overlap or duplicate")
    if () in decoded:
        if len(patches) != 1:
            raise _patch_error("PATCH_PATH_INVALID", "root patch must be standalone")
        root_patch = patches[0]
        if (
            root_patch.op != "replace"
            or not root_patch.before_present
            or not root_patch.after_present
        ):
            raise _patch_error("PATCH_PATH_INVALID", "root patch must be a replace")
        if type(root_patch.after) is not dict:
            raise _patch_error("PATCH_PATH_INVALID", "root replacement must be a JSON object")
    return decoded


def _array_index(token: str, *, length: int) -> int:
    if _ARRAY_INDEX.fullmatch(token) is None:
        raise _patch_error("PATCH_PATH_INVALID", "array index is noncanonical")
    if len(token) > len(str(length)):
        raise _patch_error("PATCH_PATH_INVALID", "array index is out of range")
    try:
        index = int(token)
    except ValueError as error:
        raise _patch_error("PATCH_PATH_INVALID", "array index is invalid", error)
    if index >= length:
        raise _patch_error("PATCH_PATH_INVALID", "array index is out of range")
    return index


def _parent_at(
    root: JsonObject, tokens: tuple[str, ...]
) -> tuple[JsonObject | list[JsonValue], str]:
    if not tokens:
        raise _patch_error("PATCH_PATH_INVALID", "root has no parent")
    current: JsonObject | list[JsonValue] | JsonValue = root
    for token in tokens[:-1]:
        if type(current) is dict:
            current_object = cast(JsonObject, current)
            if token not in current_object:
                raise _patch_error("PATCH_PATH_INVALID", "patch parent is missing")
            current = current_object[token]
        elif type(current) is list:
            current_array = cast(list[JsonValue], current)
            current = current_array[_array_index(token, length=len(current_array))]
        else:
            raise _patch_error("PATCH_PATH_INVALID", "patch parent is not a container")
    if type(current) not in {dict, list}:
        raise _patch_error("PATCH_PATH_INVALID", "patch parent is not a container")
    return cast(JsonObject | list[JsonValue], current), tokens[-1]


def apply_state_data_patches(before: JsonObject, delta: Sequence[StatePatch]) -> JsonObject:
    data = _owned_json_object(before)
    patches = tuple(_revalidated_patch(patch) for patch in delta)
    decoded = _validate_patch_paths(patches)
    if decoded and decoded[0] == ():
        root_patch = patches[0]
        if not _json_equal(data, cast(JsonValue, root_patch.before)):
            raise _patch_error("PATCH_STALE", "root patch before value is stale")
        return _owned_json_object(root_patch.after)
    for patch, tokens in zip(patches, decoded, strict=True):
        parent, token = _parent_at(data, tokens)
        if type(parent) is dict:
            parent_object = cast(JsonObject, parent)
            if patch.op == "add":
                if token in parent_object:
                    raise _patch_error("PATCH_STALE", "add target already exists")
                parent_object[token] = _copy_json_value(patch.after, set())
                continue
            if token not in parent_object:
                raise _patch_error("PATCH_STALE", "patch target is missing")
            if not _json_equal(parent_object[token], cast(JsonValue, patch.before)):
                raise _patch_error("PATCH_STALE", "patch before value is stale")
            if patch.op == "remove":
                del parent_object[token]
            else:
                parent_object[token] = _copy_json_value(patch.after, set())
            continue
        parent_array = cast(list[JsonValue], parent)
        if patch.op != "replace":
            raise _patch_error("PATCH_PATH_INVALID", "array patches must be replacements")
        index = _array_index(token, length=len(parent_array))
        if not _json_equal(parent_array[index], cast(JsonValue, patch.before)):
            raise _patch_error("PATCH_STALE", "patch before value is stale")
        parent_array[index] = _copy_json_value(patch.after, set())
    return data


def _patch_world_state(value: object) -> WorldState:
    try:
        return _validated_world_state(value, code="WORLD_STATE_INVALID")
    except EnvironmentInvariantFailure as error:
        raise _patch_error("WORLD_STATE_INVALID", error.detail, error)


def diff_world_state(before: WorldState, after: WorldState) -> tuple[StatePatch, ...]:
    owned_before = _patch_world_state(before)
    owned_after = _patch_world_state(after)
    if (
        owned_before.schema_version,
        owned_before.world_id,
        owned_before.domain,
        owned_before.seed,
    ) != (
        owned_after.schema_version,
        owned_after.world_id,
        owned_after.domain,
        owned_after.seed,
    ):
        raise _patch_error("WORLD_IDENTITY_DRIFT", "world identity changed")
    return diff_state_data(owned_before.data, owned_after.data)


def apply_world_state_patches(before: WorldState, delta: Sequence[StatePatch]) -> WorldState:
    owned_before = _patch_world_state(before)
    return WorldState(
        schema_version=owned_before.schema_version,
        world_id=owned_before.world_id,
        domain=owned_before.domain,
        seed=owned_before.seed,
        data=apply_state_data_patches(owned_before.data, delta),
    )


def execute_action(
    state: WorldState,
    call: ActionCall,
    adapter: DomainAdapter,
    *,
    cursor: EventCursor,
) -> ExecutionStep:
    before_state = _validated_world_state(state)
    before_call = _validated_action_call(call)
    before_cursor = _validated_cursor(cursor)
    callback_state = _validated_world_state(before_state)
    callback_call = _validated_action_call(before_call)
    before_hash = hash_state(before_state)
    before_action_hash = hash_action(before_call)
    try:
        proposal_value = adapter.apply(callback_state, callback_call)
    except Exception as error:
        raise _invariant("ADAPTER_EXCEPTION", "adapter apply raised", error)
    try:
        callback_state_hash = hash_state(callback_state)
        callback_action_hash = hash_action(callback_call)
    except Exception as error:
        raise _invariant("CALLBACK_INPUT_MUTATION", "callback input became invalid", error)
    if callback_state_hash != before_hash or callback_action_hash != before_action_hash:
        raise _invariant("CALLBACK_INPUT_MUTATION", "callback changed its input")
    proposal = _validated_proposal(proposal_value)
    if proposal.local_status == "success":
        if proposal.next_state is None:
            raise _invariant("PROPOSAL_INVALID", "success proposal lacks next state")
        after_state = _validated_world_state(proposal.next_state, code="PROPOSAL_INVALID")
        if (
            before_state.schema_version,
            before_state.world_id,
            before_state.domain,
            before_state.seed,
        ) != (
            after_state.schema_version,
            after_state.world_id,
            after_state.domain,
            after_state.seed,
        ):
            raise _invariant("STATE_IDENTITY_DRIFT", "success proposal changed world identity")
    else:
        after_state = _validated_world_state(before_state)
    result = ActionResult(
        local_status=proposal.local_status,
        observation=proposal.observation,
        error_code=proposal.error_code,
        error_detail=proposal.error_detail,
    )
    try:
        delta = diff_world_state(before_state, after_state)
        reconstructed = apply_world_state_patches(before_state, delta)
        if hash_state(reconstructed) != hash_state(after_state):
            raise _patch_error("PATCH_INCONSISTENT", "delta does not reconstruct proposed state")
    except PatchApplicationError as error:
        raise _invariant("PATCH_INCONSISTENT", "delta verification failed", error)
    try:
        event = make_state_event_from_cursor(
            cursor=before_cursor,
            action=before_call,
            result=result,
            before_state=before_state,
            after_state=after_state,
            delta=delta,
        )
        next_cursor = EventCursor(
            episode_id=event.episode_id,
            next_index=event.index + 1,
            parent_event_id=event.event_id,
            parent_event_hash=event.event_hash,
        )
    except Exception as error:
        raise _invariant("EVENT_INVARIANT_FAILURE", "event construction failed", error)
    return ExecutionStep(after_state, result, event, next_cursor)
