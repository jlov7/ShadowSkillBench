from __future__ import annotations

import asyncio
import inspect
import json
from typing import cast

import pytest

from shadowskillbench.episodes import (
    EpisodeDisposition,
    EpisodePlan,
    ExecutorConfigurationError,
    run_episode,
)
from shadowskillbench.episodes.pilot_turn_wire import (
    CAPABILITY_TURN_PROMPT,
    CONFIRMATORY_TURN_PROMPT,
    CONFIRMATORY_TURN_PROMPT_PROFILE,
    NAMED_ACTION_TURN_PROMPT,
    CapabilityAgentTurnWire,
    ConfirmatoryAgentTurnWire,
    NamedActionAgentTurnWire,
    capability_active_turn_schema_hash,
    capability_finished_turn_schema_hash,
    capability_initial_turn_schema_hash,
    confirmatory_active_turn_schema_hash,
    confirmatory_finished_turn_schema_hash,
    confirmatory_initial_turn_schema_hash,
    named_action_active_turn_schema_hash,
    named_action_finished_turn_schema_hash,
    named_action_initial_turn_schema_hash,
)
from shadowskillbench.models import ModelClient
from tests.integration.episodes.test_scripted_executor import _client, _plan


def test_run_episode_temperature_defaults_to_zero() -> None:
    assert inspect.signature(run_episode).parameters["temperature"].default == 0.0


@pytest.mark.parametrize("temperature", (0, "1.0", -0.1, -0.0))
def test_run_episode_rejects_invalid_temperature(temperature: object) -> None:
    plan = object.__new__(EpisodePlan)
    with pytest.raises(ExecutorConfigurationError, match="temperature"):
        asyncio.run(
            run_episode(
                plan,
                cast(ModelClient, object()),
                temperature=temperature,  # type: ignore[arg-type]
            )
        )


def test_run_episode_selects_and_converts_state_bound_confirmatory_turn_wires() -> None:
    plan, fixture = _plan()
    client = _client(
        (
            {
                "turn": {
                    "kind": "tool_call",
                    "tool_index": 2,
                    "argument_bindings": [
                        {"argument_index": 0, "value": fixture.case.target_request_id}
                    ],
                }
            },
            {
                "turn": {
                    "kind": "tool_call",
                    "tool_index": 8,
                    "argument_bindings": [{"argument_index": 0, "value": "Request reviewed."}],
                }
            },
            {"turn": {"kind": "finish", "summary": "done"}},
        )
    )

    result = asyncio.run(run_episode(plan, client, output_schema=ConfirmatoryAgentTurnWire))

    assert result.disposition is EpisodeDisposition.COMPLETED
    assert [turn.output.kind if turn.output is not None else None for turn in result.trace] == [
        "tool_call",
        "tool_call",
        "finish",
    ]
    assert [
        turn.action.tool_name if turn.action is not None else None for turn in result.trace
    ] == [
        "get_access_request",
        "finish_task",
        None,
    ]
    assert [turn.receipt.structured_output_schema_hash for turn in result.trace] == [
        confirmatory_initial_turn_schema_hash(),
        confirmatory_active_turn_schema_hash(),
        confirmatory_finished_turn_schema_hash(),
    ]
    request = json.loads(client.recorded_request_bodies[0])
    assert CONFIRMATORY_TURN_PROMPT in [
        message["content"] for message in request["messages"] if message["role"] == "developer"
    ]


def test_confirmatory_compact_prompt_requires_an_executable_first_turn() -> None:
    prompt = CONFIRMATORY_TURN_PROMPT.lower()

    assert CONFIRMATORY_TURN_PROMPT_PROFILE == "SSB-CONFIRMATORY-TURN-WIRE3"
    assert "first turn the schema permits only tool_call" in prompt
    assert "active schema permits tool_call, escalate, or refuse but never finish" in prompt
    assert "finished schema permits only finish" in prompt


def test_run_episode_selects_fixed_shape_capability_turn_wires() -> None:
    plan, fixture = _plan()
    client = _client(
        (
            {
                "turn": {
                    "kind": "tool_call",
                    "tool_index": 2,
                    "argument_bindings": [
                        {"argument_index": 0, "value": fixture.case.target_request_id}
                    ],
                    "summary": None,
                }
            },
            {
                "turn": {
                    "kind": "tool_call",
                    "tool_index": 8,
                    "argument_bindings": [{"argument_index": 0, "value": "Reviewed."}],
                    "summary": None,
                }
            },
            {
                "turn": {
                    "kind": "finish",
                    "tool_index": None,
                    "argument_bindings": [],
                    "summary": "done",
                }
            },
        )
    )

    result = asyncio.run(run_episode(plan, client, output_schema=CapabilityAgentTurnWire))

    assert result.disposition is EpisodeDisposition.COMPLETED
    assert [turn.receipt.structured_output_schema_hash for turn in result.trace] == [
        capability_initial_turn_schema_hash(),
        capability_active_turn_schema_hash(),
        capability_finished_turn_schema_hash(),
    ]
    request = json.loads(client.recorded_request_bodies[0])
    assert CAPABILITY_TURN_PROMPT in [
        message["content"] for message in request["messages"] if message["role"] == "developer"
    ]


def test_run_episode_selects_named_action_turn_wires() -> None:
    plan, fixture = _plan()
    client = _client(
        (
            {
                "turn": {
                    "kind": "tool_call",
                    "tool_name": "get_access_request",
                    "argument_bindings": [
                        {"key": "request_id", "value": fixture.case.target_request_id}
                    ],
                    "summary": None,
                }
            },
            {
                "turn": {
                    "kind": "tool_call",
                    "tool_name": "finish_task",
                    "argument_bindings": [{"key": "summary", "value": "Reviewed."}],
                    "summary": None,
                }
            },
            {
                "turn": {
                    "kind": "finish",
                    "tool_name": None,
                    "argument_bindings": [],
                    "summary": "done",
                }
            },
        )
    )

    result = asyncio.run(run_episode(plan, client, output_schema=NamedActionAgentTurnWire))

    assert result.disposition is EpisodeDisposition.COMPLETED
    assert [turn.receipt.structured_output_schema_hash for turn in result.trace] == [
        named_action_initial_turn_schema_hash(),
        named_action_active_turn_schema_hash(),
        named_action_finished_turn_schema_hash(),
    ]
    tool_names = [
        turn.action.tool_name if turn.action is not None else None for turn in result.trace
    ]
    assert tool_names == [
        "get_access_request",
        "finish_task",
        None,
    ]
    request = json.loads(client.recorded_request_bodies[0])
    assert NAMED_ACTION_TURN_PROMPT in [
        message["content"] for message in request["messages"] if message["role"] == "developer"
    ]
