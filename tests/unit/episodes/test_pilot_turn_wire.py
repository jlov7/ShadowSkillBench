from __future__ import annotations

from dataclasses import replace

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.engine import ActionCall, JsonValue
from shadowskillbench.episodes.executor import AgentTurn
from shadowskillbench.episodes.models import ExperimentCondition
from shadowskillbench.episodes.pilot_turn_wire import (
    CAPABILITY_TURN_PROMPT,
    CAPABILITY_TURN_PROMPT_PROFILE,
    CONFIRMATORY_TURN_PROMPT,
    CONFIRMATORY_TURN_PROMPT_PROFILE,
    NAMED_ACTION_TURN_PROMPT,
    NAMED_ACTION_TURN_PROMPT_PROFILE,
    PILOT_TURN_PROMPT,
    CapabilityActiveTurnWire,
    CapabilityFinishedTurnWire,
    CapabilityInitialTurnWire,
    ConfirmatoryActiveTurnWire,
    ConfirmatoryFinishedTurnWire,
    ConfirmatoryInitialToolTurnWire,
    NamedActionInitialTurnWire,
    PilotAgentTurnWire,
    capability_active_turn_schema_hash,
    capability_finished_turn_schema_hash,
    capability_initial_turn_schema_hash,
    confirmatory_active_turn_schema_hash,
    confirmatory_finished_turn_schema_hash,
    confirmatory_initial_turn_schema_hash,
    named_action_initial_turn_schema_hash,
    named_action_turn_to_agent_turn_data,
    pilot_turn_from_agent_turn_data,
    pilot_turn_layout_hash,
    pilot_turn_schema_hash,
    pilot_turn_to_agent_turn_data,
    pilot_turn_tool_layout,
)
from shadowskillbench.episodes.tools import ToolRegistry


def _layout(domain: str):
    return pilot_turn_tool_layout(ToolRegistry.for_domain(domain), ExperimentCondition.A0_BARE)


def _round_trip(turn: AgentTurn, domain: str, turn_index: int) -> AgentTurn:
    layout = _layout(domain)
    wire = pilot_turn_from_agent_turn_data(turn.model_dump(mode="json"), layout, turn_index)
    return AgentTurn.model_validate(
        pilot_turn_to_agent_turn_data(wire, layout, turn_index), strict=True
    )


def test_pilot_turn_wire_schema_is_closed_and_compact() -> None:
    schema = PilotAgentTurnWire.model_json_schema()
    encoded = canonical_json_bytes(schema)

    assert pilot_turn_schema_hash().startswith("sha256:")
    assert b'"additionalProperties":{"' not in encoded
    assert b'"tool_index"' in encoded
    assert b'"argument_index"' in encoded
    assert b'"action_id"' not in encoded
    assert b'"tool_name"' not in encoded
    assert b'"arguments"' not in encoded
    assert b'"oneOf"' in encoded
    assert "tool_index is zero-based" in PILOT_TURN_PROMPT
    assert "argument_index is zero-based" in PILOT_TURN_PROMPT
    assert "binding for every required key" in PILOT_TURN_PROMPT
    assert "Do not invoke, address, or emit a Harmony tool or function call" in PILOT_TURN_PROMPT
    assert "switch to the final channel" in PILOT_TURN_PROMPT
    assert "never end the response from the analysis channel" in PILOT_TURN_PROMPT


def test_confirmatory_turn_wire_schemas_are_state_bound() -> None:
    initial = ConfirmatoryInitialToolTurnWire.model_validate(
        {"turn": {"kind": "tool_call", "tool_index": 0, "argument_bindings": []}}
    )
    active_tool = ConfirmatoryActiveTurnWire.model_validate(
        {"turn": {"kind": "tool_call", "tool_index": 0, "argument_bindings": []}}
    )
    active_escalate = ConfirmatoryActiveTurnWire.model_validate({"turn": {"kind": "escalate"}})
    active_refuse = ConfirmatoryActiveTurnWire.model_validate(
        {"turn": {"kind": "refuse", "summary": "Cannot proceed."}}
    )
    finished = ConfirmatoryFinishedTurnWire.model_validate(
        {"turn": {"kind": "finish", "summary": "Done."}}
    )

    assert initial.turn.kind == "tool_call"
    assert active_tool.turn.kind == "tool_call"
    assert active_escalate.turn.kind == "escalate"
    assert active_refuse.turn.kind == "refuse"
    assert finished.turn.kind == "finish"
    assert CONFIRMATORY_TURN_PROMPT_PROFILE == "SSB-CONFIRMATORY-TURN-WIRE3"
    assert "first turn the schema permits only tool_call" in CONFIRMATORY_TURN_PROMPT
    assert "active schema permits tool_call, escalate, or refuse but never finish" in (
        CONFIRMATORY_TURN_PROMPT
    )
    assert "finished schema permits only finish" in CONFIRMATORY_TURN_PROMPT
    assert confirmatory_initial_turn_schema_hash() == sha256_ref(
        ConfirmatoryInitialToolTurnWire.model_json_schema()
    )
    assert confirmatory_active_turn_schema_hash() == sha256_ref(
        ConfirmatoryActiveTurnWire.model_json_schema()
    )
    assert confirmatory_finished_turn_schema_hash() == sha256_ref(
        ConfirmatoryFinishedTurnWire.model_json_schema()
    )

    for schema, value in (
        (ConfirmatoryInitialToolTurnWire, {"turn": {"kind": "finish", "summary": "Done."}}),
        (ConfirmatoryActiveTurnWire, {"turn": {"kind": "finish", "summary": "Done."}}),
        (
            ConfirmatoryFinishedTurnWire,
            {"turn": {"kind": "tool_call", "tool_index": 0, "argument_bindings": []}},
        ),
    ):
        try:
            schema.model_validate(value)
        except ValueError:
            pass
        else:
            raise AssertionError("state-ineligible confirmatory turn was admitted")


def test_capability_turn_wire_is_fixed_shape_and_state_bound() -> None:
    tool = {
        "turn": {
            "kind": "tool_call",
            "tool_index": 2,
            "argument_bindings": [{"argument_index": 0, "value": "request_123"}],
            "summary": None,
        }
    }
    initial = CapabilityInitialTurnWire.model_validate(tool)
    active = CapabilityActiveTurnWire.model_validate(tool)
    finished = CapabilityFinishedTurnWire.model_validate(
        {
            "turn": {
                "kind": "finish",
                "tool_index": None,
                "argument_bindings": [],
                "summary": "Done.",
            }
        }
    )

    assert CAPABILITY_TURN_PROMPT_PROFILE == "SSB-CONFIRMATORY-TURN-WIRE4"
    assert "always contains exactly kind, tool_index, argument_bindings, and summary" in (
        CAPABILITY_TURN_PROMPT
    )
    assert initial.turn.summary is None
    assert active.turn.kind == "tool_call"
    assert finished.turn.kind == "finish"
    assert capability_initial_turn_schema_hash() == sha256_ref(
        CapabilityInitialTurnWire.model_json_schema()
    )
    assert capability_active_turn_schema_hash() == sha256_ref(
        CapabilityActiveTurnWire.model_json_schema()
    )
    assert capability_finished_turn_schema_hash() == sha256_ref(
        CapabilityFinishedTurnWire.model_json_schema()
    )
    assert (
        pilot_turn_to_agent_turn_data(initial, _layout("access_provisioning"), 0)["kind"]
        == "tool_call"
    )

    for schema, value in (
        (CapabilityInitialTurnWire, {"turn": {"kind": "finish"}}),
        (CapabilityActiveTurnWire, {"turn": {"kind": "finish"}}),
        (CapabilityFinishedTurnWire, tool),
    ):
        try:
            schema.model_validate(value)
        except ValueError:
            pass
        else:
            raise AssertionError("state-ineligible capability turn was admitted")


def test_named_action_turn_wire_uses_visible_names_and_host_validates_bindings() -> None:
    registry = ToolRegistry.for_domain("access_provisioning")
    valid = NamedActionInitialTurnWire.model_validate(
        {
            "turn": {
                "kind": "tool_call",
                "tool_name": "get_access_request",
                "argument_bindings": [{"key": "request_id", "value": "request_123"}],
                "summary": None,
            }
        }
    )

    converted = named_action_turn_to_agent_turn_data(
        valid, registry, ExperimentCondition.A0_BARE, 0
    )
    assert converted == {
        "kind": "tool_call",
        "action": {
            "action_id": "action_1",
            "tool_name": "get_access_request",
            "arguments": {"request_id": "request_123"},
        },
    }
    assert NAMED_ACTION_TURN_PROMPT_PROFILE == "SSB-CONFIRMATORY-TURN-WIRE5"
    assert "exactly kind, tool_name, argument_bindings, and summary" in NAMED_ACTION_TURN_PROMPT
    assert named_action_initial_turn_schema_hash() == sha256_ref(
        NamedActionInitialTurnWire.model_json_schema()
    )

    for value in (
        {
            "turn": {
                "kind": "tool_call",
                "tool_name": "hidden_tool",
                "argument_bindings": [],
                "summary": None,
            }
        },
        {
            "turn": {
                "kind": "tool_call",
                "tool_name": "get_access_request",
                "argument_bindings": [],
                "summary": None,
            }
        },
        {
            "turn": {
                "kind": "tool_call",
                "tool_name": "get_access_request",
                "argument_bindings": [
                    {"key": "request_id", "value": "request_123"},
                    {"key": "request_id", "value": "request_456"},
                ],
                "summary": None,
            }
        },
    ):
        try:
            wire = NamedActionInitialTurnWire.model_validate(value)
            named_action_turn_to_agent_turn_data(wire, registry, ExperimentCondition.A0_BARE, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid named action was admitted")


def test_pilot_turn_wire_rejects_invalid_combinations_duplicates_and_ranges() -> None:
    invalid = {
        "turn": {
            "kind": "finish",
            "summary": "done",
            "tool_index": 0,
        }
    }
    try:
        PilotAgentTurnWire.model_validate(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("terminal/tool combination was admitted")

    duplicate = {
        "turn": {
            "kind": "tool_call",
            "tool_index": 0,
            "argument_bindings": [
                {"argument_index": 0, "value": "one"},
                {"argument_index": 0, "value": "two"},
            ],
        }
    }
    try:
        PilotAgentTurnWire.model_validate(duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate argument indexes were admitted")

    out_of_range = {
        "turn": {
            "kind": "tool_call",
            "tool_index": 16,
            "argument_bindings": [],
        }
    }
    try:
        PilotAgentTurnWire.model_validate(out_of_range)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range tool index was admitted")


def test_pilot_turn_wire_terminal_summaries_match_canonical_nonblank_requirement() -> None:
    for kind in ("finish", "refuse", "escalate"):
        for summary in ("", "   "):
            try:
                PilotAgentTurnWire.model_validate({"turn": {"kind": kind, "summary": summary}})
            except ValueError:
                pass
            else:
                raise AssertionError(f"{kind} admitted a blank summary")

    assert PilotAgentTurnWire.model_validate({"turn": {"kind": "escalate"}}).turn.kind == "escalate"


def test_pilot_turn_wire_hydrates_host_owned_tool_fields_deterministically() -> None:
    layout = _layout("access_provisioning")
    wire = PilotAgentTurnWire.model_validate(
        {
            "turn": {
                "kind": "tool_call",
                "tool_index": 2,
                "argument_bindings": [{"argument_index": 0, "value": "request_123"}],
            }
        }
    )

    assert pilot_turn_to_agent_turn_data(wire, layout, 3) == {
        "kind": "tool_call",
        "action": {
            "action_id": "action_4",
            "tool_name": "get_access_request",
            "arguments": {"request_id": "request_123"},
        },
    }


def test_pilot_turn_wire_round_trips_canonical_scripted_executor_witnesses() -> None:
    access = (
        AgentTurn(
            kind="tool_call",
            action=ActionCall(
                action_id="action_1",
                tool_name="get_access_request",
                arguments={"request_id": "request_123"},
            ),
        ),
        AgentTurn(
            kind="tool_call",
            action=ActionCall(
                action_id="action_1",
                tool_name="request_approval",
                arguments={"request_id": "request_123", "approver_role": "security"},
            ),
        ),
        AgentTurn(
            kind="tool_call",
            action=ActionCall(
                action_id="action_1",
                tool_name="finish_task",
                arguments={"summary": "Access granted and reviewed."},
            ),
        ),
        AgentTurn(kind="finish", summary="Done."),
        AgentTurn(kind="escalate", summary=None),
        AgentTurn(kind="refuse", summary="I cannot safely proceed."),
    )
    finance = (
        AgentTurn(
            kind="tool_call",
            action=ActionCall(
                action_id="action_1",
                tool_name="get_ledger_snapshot",
                arguments={"snapshot_id": "snapshot_123"},
            ),
        ),
        AgentTurn(
            kind="tool_call",
            action=ActionCall(
                action_id="action_1",
                tool_name="create_adjustment",
                arguments={
                    "report_id": "report_123",
                    "control_fact_id": "control_123",
                    "reported_category": "ordinary",
                    "signed_amount_minor": -1200,
                    "currency": "USD",
                    "unit": "minor",
                    "currency_exponent": 2,
                    "period_id": "period_123",
                    "portco_id": "portco_123",
                    "rationale": "Reconciled source evidence.",
                },
            ),
        ),
        AgentTurn(
            kind="tool_call",
            action=ActionCall(
                action_id="action_1",
                tool_name="publish_report",
                arguments={"report_id": "report_123"},
            ),
        ),
    )

    for turn in access:
        assert _round_trip(turn, "access_provisioning", 0) == turn
    for turn in finance:
        assert _round_trip(turn, "financial_adjustments", 0) == turn


def _argument_value(key: str) -> JsonValue:
    values: dict[str, JsonValue] = {
        "approver_role": "security",
        "reported_category": "ordinary",
        "signed_amount_minor": -1200,
        "currency": "USD",
        "unit": "minor",
        "currency_exponent": 2,
        "rationale": "Reconciled source evidence.",
        "summary": "Reviewed visible evidence.",
    }
    return values.get(key, f"{key}_123")


def test_pilot_turn_wire_round_trips_and_validates_every_visible_tool() -> None:
    for domain in ("access_provisioning", "financial_adjustments"):
        registry = ToolRegistry.for_domain(domain)
        layout = pilot_turn_tool_layout(registry, ExperimentCondition.A0_BARE)
        for tool_index, (tool_name, required_keys, optional_keys) in enumerate(layout.tools):
            argument_keys = required_keys + optional_keys
            arguments = {key: _argument_value(key) for key in argument_keys}
            turn = AgentTurn(
                kind="tool_call",
                action=ActionCall(action_id="action_1", tool_name=tool_name, arguments=arguments),
            )

            wire = pilot_turn_from_agent_turn_data(turn.model_dump(mode="json"), layout, 0)
            hydrated = AgentTurn.model_validate(
                pilot_turn_to_agent_turn_data(wire, layout, 0), strict=True
            )

            assert hydrated == turn
            assert hydrated.action is not None
            assert (
                registry.validate_call(hydrated.action, condition=ExperimentCondition.A0_BARE)
                == hydrated.action
            )


def test_pilot_turn_wire_maximum_schema_valid_output_is_bounded_and_ascii_only() -> None:
    value = {
        "turn": {
            "kind": "tool_call",
            "tool_index": 15,
            "argument_bindings": [
                {"argument_index": index, "value": ["x" * 96, "x" * 96]} for index in range(12)
            ],
        }
    }
    admitted = PilotAgentTurnWire.model_validate(value)
    assert len(canonical_json_bytes(admitted.model_dump(mode="python"))) < 8192

    invalid_text = {"turn": {"kind": "finish", "summary": "done \U0001f600"}}
    try:
        PilotAgentTurnWire.model_validate(invalid_text)
    except ValueError:
        pass
    else:
        raise AssertionError("non-ASCII text was admitted")


def test_pilot_turn_layout_hash_binds_visible_tools() -> None:
    access = _layout("access_provisioning")
    finance = _layout("financial_adjustments")

    assert pilot_turn_layout_hash(access).startswith("sha256:")
    assert pilot_turn_layout_hash(access) != pilot_turn_layout_hash(finance)


def test_pilot_turn_layout_hash_binds_required_optional_classification() -> None:
    registry = ToolRegistry.for_domain("access_provisioning")
    original = pilot_turn_tool_layout(registry, ExperimentCondition.A0_BARE)
    grant_access = next(spec for spec in registry.specs if spec.name == "grant_access")
    mutated_grant_access = replace(
        grant_access,
        argument_schema={
            "required": ["request_id", "approval_id"],
            "optional": ["exception_id"],
        },
    )
    mutated_registry = replace(
        registry,
        specs=tuple(
            mutated_grant_access if spec.name == "grant_access" else spec for spec in registry.specs
        ),
    )
    mutated = pilot_turn_tool_layout(mutated_registry, ExperimentCondition.A0_BARE)

    assert pilot_turn_layout_hash(original) != pilot_turn_layout_hash(mutated)
    original_grant = next(tool for tool in original.tools if tool[0] == "grant_access")
    mutated_grant = next(tool for tool in mutated.tools if tool[0] == "grant_access")
    assert original_grant[1:] == (("request_id",), ("approval_id", "exception_id"))
    assert mutated_grant[1:] == (("request_id", "approval_id"), ("exception_id",))
