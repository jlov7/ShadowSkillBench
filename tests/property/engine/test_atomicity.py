from __future__ import annotations

from typing import Literal, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.engine.models import (
    ActionCall,
    EventCursor,
    JsonObject,
    JsonValue,
    StatePatch,
    TransitionProposal,
    WorldState,
)
from shadowskillbench.engine.runtime import (
    apply_state_data_patches,
    diff_state_data,
    execute_action,
)

SAFE_TEXT = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=8)
FINITE_FLOATS = st.floats(allow_nan=False, allow_infinity=False, width=64)
JSON_VALUES = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), FINITE_FLOATS, SAFE_TEXT),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(SAFE_TEXT, children, max_size=4),
    ),
    max_leaves=15,
)
JSON_OBJECTS = st.dictionaries(SAFE_TEXT, JSON_VALUES, max_size=6)


@pytest.mark.property
@settings(max_examples=35, deadline=None)
@given(JSON_OBJECTS, JSON_OBJECTS)
def test_diff_apply_round_trip_for_strict_json_objects(
    before: dict[str, object], after: dict[str, object]
) -> None:
    owned_before = cast(JsonObject, before)
    owned_after = cast(JsonObject, after)
    assert (
        apply_state_data_patches(owned_before, diff_state_data(owned_before, owned_after))
        == owned_after
    )


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(JSON_OBJECTS, JSON_OBJECTS)
def test_mapping_insertion_order_does_not_change_delta(
    before: dict[str, object], after: dict[str, object]
) -> None:
    first = diff_state_data(cast(JsonObject, before), cast(JsonObject, after))
    second = diff_state_data(
        cast(JsonObject, dict(reversed(list(before.items())))),
        cast(JsonObject, dict(reversed(list(after.items())))),
    )
    assert first == second


@pytest.mark.property
@settings(max_examples=25, deadline=None)
@given(st.lists(JSON_VALUES, max_size=4), st.lists(JSON_VALUES, max_size=4))
def test_array_length_changes_are_single_replace(
    before: list[JsonValue], after: list[JsonValue]
) -> None:
    if len(before) == len(after):
        return
    delta = diff_state_data({"array": before}, {"array": after})
    assert delta == (
        StatePatch(
            op="replace",
            path="/array",
            before_present=True,
            before=before,
            after_present=True,
            after=after,
        ),
    )


@pytest.mark.property
@settings(max_examples=20, deadline=None)
@given(st.integers(), st.integers())
def test_repeated_execution_is_deterministic(before_value: int, after_value: int) -> None:
    initial = WorldState(
        schema_version="1.0",
        world_id="world",
        domain="domain",
        seed=1,
        data={"value": before_value},
    )
    call = ActionCall(action_id="action", tool_name="tool", arguments={})
    cursor = EventCursor(episode_id="episode", next_index=0)

    class Adapter:
        def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
            return TransitionProposal(
                local_status="success",
                next_state=WorldState(
                    schema_version="1.0",
                    world_id=state.world_id,
                    domain=state.domain,
                    seed=state.seed,
                    data={"value": after_value},
                ),
                observation={"tool": call.tool_name},
            )

    assert execute_action(initial, call, Adapter(), cursor=cursor) == execute_action(
        initial, call, Adapter(), cursor=cursor
    )


@pytest.mark.property
@settings(max_examples=15, deadline=None)
@given(st.sampled_from(["failure", "clarify"]), st.integers())
def test_non_success_proposals_always_preserve_state(
    status: Literal["failure", "clarify"], seed: int
) -> None:
    initial = WorldState(
        schema_version="1.0",
        world_id="world",
        domain="domain",
        seed=seed,
        data={"value": seed},
    )
    call = ActionCall(action_id="action", tool_name="tool", arguments={})
    cursor = EventCursor(episode_id="episode", next_index=0)

    class Adapter:
        def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
            return TransitionProposal(
                local_status=status,
                observation={"tool": call.tool_name},
                error_code="INVALID_ACTION",
            )

    step = execute_action(initial, call, Adapter(), cursor=cursor)
    assert step.state == initial
    assert step.event.delta == ()
    assert step.event.before_hash == step.event.after_hash
