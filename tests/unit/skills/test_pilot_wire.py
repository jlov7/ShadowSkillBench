from __future__ import annotations

import json
from decimal import Decimal

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.skills import gate2
from shadowskillbench.skills.models import SkillIR, skill_ir_projection
from shadowskillbench.skills.pilot_wire import (
    PilotSkillIRWire,
    pilot_wire_from_skill_ir,
    pilot_wire_schema_hash,
    pilot_wire_to_skill_ir,
)
from shadowskillbench.skills.projection import compiler_view
from shadowskillbench.traces.bundles import generate_bundle


def _wire_value() -> dict[str, object]:
    return {
        "objective": "Handle the visible request.",
        "objective_evidence_indices": [0],
        "ordered_steps": [
            {
                "step_id": "inspect_visible_state",
                "action_intent": "Inspect visible state.",
                "tool_name": "inspect_state",
                "argument_bindings": [{"key": "scope", "value": "visible"}],
                "optional": False,
                "evidence_indices": [0],
            }
        ],
    }


def test_pilot_wire_schema_is_compact_and_hashable() -> None:
    schema = PilotSkillIRWire.model_json_schema()
    encoded = canonical_json_bytes(schema)
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    ordered_step = definitions["PilotWireOrderedStep"]
    assert isinstance(ordered_step, dict)
    ordered_properties = ordered_step["properties"]
    assert isinstance(ordered_properties, dict)
    argument_bindings = ordered_properties["argument_bindings"]
    assert isinstance(argument_bindings, dict)

    assert pilot_wire_schema_hash().startswith("sha256:")
    assert "JsonValue" not in definitions
    assert b'"additionalProperties":{"' not in encoded
    assert b'"maxItems":7' in encoded
    assert b'"maxItems":12' in encoded
    assert b'"maxLength":192' in encoded
    assert b'"additionalProperties":false' in encoded
    assert argument_bindings["maxItems"] == 2
    assert argument_bindings["items"] == {"$ref": "#/$defs/PilotWireArgumentBinding"}
    root_properties = schema["properties"]
    assert isinstance(root_properties, dict)
    bounded_text = definitions["BoundedText"]
    assert isinstance(bounded_text, dict)
    assert bounded_text["pattern"] == r"^[\x20-\x21\x23-\x5B\x5D-\x7E]*$"
    assert root_properties["objective_evidence_indices"]["maxItems"] == 12
    assert ordered_properties["evidence_indices"]["maxItems"] == 12


def test_pilot_wire_rejects_unbounded_provider_shapes() -> None:
    value = _wire_value()
    PilotSkillIRWire.model_validate(value)

    value["objective"] = "x" * 193
    with pytest.raises(ValueError):
        PilotSkillIRWire.model_validate(value)


def test_pilot_wire_maximum_serialized_output_is_below_the_8192_cap() -> None:
    identifier = "a" * 64
    value = {
        "objective": "x" * 192,
        "objective_evidence_indices": list(range(4084, 4096)),
        "ordered_steps": [
            {
                "step_id": identifier,
                "action_intent": "x" * 192,
                "tool_name": identifier,
                "argument_bindings": [
                    {"key": identifier, "value": ["x" * 96, "x" * 96]},
                    {"key": f"b{identifier[1:]}", "value": ["x" * 96, "x" * 96]},
                ],
                "optional": False,
                "evidence_indices": list(range(4084, 4096)),
            }
            for _ in range(7)
        ],
    }

    admitted = PilotSkillIRWire.model_validate(value)
    with pytest.raises(ValueError, match="event indices"):
        pilot_wire_to_skill_ir(
            admitted,
            compiler_view(generate_bundle("access_provisioning", Decimal("0"), 12, 4242).bundle),
            "sha256:" + "b" * 64,
        )
    assert len(canonical_json_bytes(admitted.model_dump(mode="python"))) == 7702

    value = _wire_value()
    step = value["ordered_steps"]
    assert isinstance(step, list)
    bindings = step[0]["argument_bindings"]
    assert isinstance(bindings, list)
    bindings.append({"key": "scope", "value": "other"})
    with pytest.raises(ValueError, match="binding keys"):
        pilot_wire_to_skill_ir(
            PilotSkillIRWire.model_validate(value),
            compiler_view(generate_bundle("access_provisioning", Decimal("0"), 12, 4242).bundle),
            "sha256:" + "b" * 64,
        )

    value = _wire_value()
    step = value["ordered_steps"]
    assert isinstance(step, list)
    step[0]["argument_bindings"] = [{"key": f"key_{index}", "value": "value"} for index in range(9)]
    with pytest.raises(ValueError):
        PilotSkillIRWire.model_validate(value)

    value = _wire_value()
    value["objective_evidence_indices"] = [0] * 13
    with pytest.raises(ValueError):
        PilotSkillIRWire.model_validate(value)


def test_pilot_wire_evidence_indices_are_unique_and_within_schema_range() -> None:
    value = _wire_value()
    value["objective_evidence_indices"] = [0, 0]
    with pytest.raises(ValueError, match="unique"):
        PilotSkillIRWire.model_validate(value)

    value = _wire_value()
    step = value["ordered_steps"]
    assert isinstance(step, list)
    step[0]["evidence_indices"] = [0, 0]
    with pytest.raises(ValueError, match="unique"):
        PilotSkillIRWire.model_validate(value)

    value = _wire_value()
    value["objective_evidence_indices"] = [4095]
    PilotSkillIRWire.model_validate(value)
    value["objective_evidence_indices"] = [4096]
    with pytest.raises(ValueError):
        PilotSkillIRWire.model_validate(value)


def test_pilot_wire_argument_values_are_bounded() -> None:
    value = _wire_value()
    step = value["ordered_steps"]
    assert isinstance(step, list)
    bindings = step[0]["argument_bindings"]
    assert isinstance(bindings, list)
    bindings[0]["value"] = 1_000_001
    with pytest.raises(ValueError):
        PilotSkillIRWire.model_validate(value)

    value = _wire_value()
    step = value["ordered_steps"]
    assert isinstance(step, list)
    bindings = step[0]["argument_bindings"]
    assert isinstance(bindings, list)
    bindings[0]["value"] = float("inf")
    with pytest.raises(ValueError):
        PilotSkillIRWire.model_validate(value)


def test_pilot_wire_rejects_non_ascii_text_before_provider_use() -> None:
    value = _wire_value()
    value["objective"] = "visible \U0001f600"
    with pytest.raises(ValueError):
        PilotSkillIRWire.model_validate(value)

    value = _wire_value()
    step = value["ordered_steps"]
    assert isinstance(step, list)
    bindings = step[0]["argument_bindings"]
    assert isinstance(bindings, list)
    bindings[0]["value"] = "visible \U0001f600"
    with pytest.raises(ValueError):
        PilotSkillIRWire.model_validate(value)


def test_pilot_wire_accepts_all_canonical_gate2_witness_shapes() -> None:
    witness_sizes: list[int] = []
    for domain in ("access_provisioning", "financial_adjustments"):
        for ratio in gate2.RATIOS:
            compiler_input = compiler_view(generate_bundle(domain, Decimal(ratio), 12, 4242).bundle)
            witness = SkillIR.model_validate(
                json.loads(
                    canonical_json_bytes(gate2._witness_skill(compiler_input, "sha256:" + "c" * 64))
                )
            )
            manifest_ref = "sha256:" + "c" * 64
            admitted = pilot_wire_from_skill_ir(witness, compiler_input, manifest_ref)
            assert skill_ir_projection(
                pilot_wire_to_skill_ir(admitted, compiler_input, manifest_ref)
            ) == skill_ir_projection(witness)
            witness_sizes.append(len(admitted.ordered_steps))

    assert len(witness_sizes) == 10
    assert max(witness_sizes) == 7
