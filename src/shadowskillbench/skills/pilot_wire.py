"""Compact provider-facing output schema for the exploratory live pilot."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.skills.models import SkillIR
from shadowskillbench.skills.projection import CompilerInput, hash_compiler_input

_MAX_TEXT_CODEPOINTS = 192
_MAX_ARGUMENT_TEXT_CODEPOINTS = 96
_MAX_ORDERED_STEPS = 7
_MAX_ARGUMENT_BINDINGS = 2
_MAX_EVIDENCE_INDICES = 12
_MAX_IDENTIFIER_LENGTH = 64
_MAX_ARGUMENT_INTEGER_ABS = 1_000_000
_MAX_ARGUMENT_FLOAT_ABS = 1_000_000.0
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
_PRINTABLE_ASCII_NO_JSON_ESCAPES = r"^[\x20-\x21\x23-\x5B\x5D-\x7E]*$"

type BoundedText = Annotated[
    str,
    Field(pattern=_PRINTABLE_ASCII_NO_JSON_ESCAPES, max_length=_MAX_TEXT_CODEPOINTS),
]
type Identifier = Annotated[str, Field(pattern=_IDENTIFIER, max_length=_MAX_IDENTIFIER_LENGTH)]
type ArgumentText = Annotated[
    str,
    Field(
        pattern=_PRINTABLE_ASCII_NO_JSON_ESCAPES,
        max_length=_MAX_ARGUMENT_TEXT_CODEPOINTS,
    ),
]
type ArgumentInteger = Annotated[
    int, Field(ge=-_MAX_ARGUMENT_INTEGER_ABS, le=_MAX_ARGUMENT_INTEGER_ABS)
]
type ArgumentFloat = Annotated[
    float, Field(ge=-_MAX_ARGUMENT_FLOAT_ABS, le=_MAX_ARGUMENT_FLOAT_ABS)
]
type JsonScalar = None | bool | ArgumentInteger | ArgumentFloat | ArgumentText
type ArgumentValue = JsonScalar | Annotated[list[JsonScalar], Field(max_length=2)]
type EvidenceIndex = Annotated[int, Field(ge=0, le=4095)]


class _PilotWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PilotWireArgumentBinding(_PilotWireModel):
    key: Identifier
    value: ArgumentValue


class PilotWireOrderedStep(_PilotWireModel):
    step_id: Identifier
    action_intent: BoundedText
    tool_name: Identifier
    argument_bindings: Annotated[
        list[PilotWireArgumentBinding], Field(max_length=_MAX_ARGUMENT_BINDINGS)
    ]
    optional: bool
    evidence_indices: Annotated[
        list[EvidenceIndex], Field(min_length=1, max_length=_MAX_EVIDENCE_INDICES)
    ]

    @field_validator("evidence_indices")
    @classmethod
    def _evidence_indices_are_unique(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_indices must be unique")
        return value


class PilotSkillIRWire(_PilotWireModel):
    objective: BoundedText
    objective_evidence_indices: Annotated[
        list[EvidenceIndex], Field(min_length=1, max_length=_MAX_EVIDENCE_INDICES)
    ]
    ordered_steps: Annotated[
        list[PilotWireOrderedStep], Field(min_length=1, max_length=_MAX_ORDERED_STEPS)
    ]

    @field_validator("objective_evidence_indices")
    @classmethod
    def _objective_evidence_indices_are_unique(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("objective_evidence_indices must be unique")
        return value


def pilot_wire_schema_hash() -> str:
    return sha256_ref(PilotSkillIRWire.model_json_schema())


def _event_ids(compiler_input: CompilerInput) -> tuple[str, ...]:
    if type(compiler_input) is not CompilerInput:
        raise ValueError("pilot wire compiler input is invalid")
    return tuple(event.event_id for trace in compiler_input.traces for event in trace.events)


def _event_references(indices: list[int], event_ids: tuple[str, ...]) -> list[str]:
    if len(indices) != len(set(indices)) or any(index >= len(event_ids) for index in indices):
        raise ValueError("pilot wire event indices are invalid")
    return [event_ids[index] for index in indices]


def _argument_bindings(value: list[PilotWireArgumentBinding]) -> dict[str, object]:
    bindings: dict[str, object] = {}
    for record in value:
        if record.key in bindings:
            raise ValueError("pilot wire argument binding keys must be unique")
        bindings[record.key] = record.value
    return bindings


def _indices_for_references(value: object, event_indices: dict[str, int]) -> list[int]:
    if type(value) is not list or any(type(reference) is not str for reference in value):
        raise ValueError("canonical SkillIR evidence is invalid")
    try:
        return [event_indices[reference] for reference in value]
    except KeyError as error:
        raise ValueError("canonical SkillIR evidence is invalid") from error


def pilot_wire_to_skill_ir(
    value: PilotSkillIRWire, compiler_input: CompilerInput, compiler_manifest_ref: str
) -> SkillIR:
    if type(value) is not PilotSkillIRWire or type(compiler_manifest_ref) is not str:
        raise ValueError("pilot wire value is invalid")
    event_ids = _event_ids(compiler_input)
    objective_references = _event_references(value.objective_evidence_indices, event_ids)
    ordered_steps = [
        {
            "step_id": step.step_id,
            "action_intent": step.action_intent,
            "tool_name": step.tool_name,
            "argument_bindings": _argument_bindings(step.argument_bindings),
            "preconditions": [],
            "optional": step.optional,
            "evidence_refs": _event_references(step.evidence_indices, event_ids),
        }
        for step in value.ordered_steps
    ]
    return SkillIR.model_validate(
        {
            "skill_id": f"skill_{compiler_input.domain}",
            "schema_version": "1.0",
            "domain": compiler_input.domain,
            "objective": value.objective,
            "applicability": [],
            "required_inputs": [],
            "preconditions": [],
            "ordered_steps": ordered_steps,
            "decision_hints": [],
            "verification_steps": [],
            "stop_conditions": [],
            "escalation_hints": [],
            "source_trace_ids": [trace.trace_id for trace in compiler_input.traces],
            "instruction_provenance": {
                "profile": "SSB-INSTRUCTION-PROVENANCE1",
                "compiler_input_hash": hash_compiler_input(compiler_input),
                "instruction_evidence": {
                    "/objective": objective_references,
                    **{
                        f"/ordered_steps/{index}": list(step["evidence_refs"])
                        for index, step in enumerate(ordered_steps)
                    },
                },
            },
            "compiler_manifest_ref": compiler_manifest_ref,
        }
    )


def pilot_wire_from_skill_ir(
    value: SkillIR, compiler_input: CompilerInput, compiler_manifest_ref: str
) -> PilotSkillIRWire:
    if type(value) is not SkillIR or value.compiler_manifest_ref != compiler_manifest_ref:
        raise ValueError("canonical SkillIR value is invalid")
    event_ids = _event_ids(compiler_input)
    event_indices = {event_id: index for index, event_id in enumerate(event_ids)}
    evidence = value.instruction_provenance.get("instruction_evidence")
    if type(evidence) is not dict:
        raise ValueError("canonical SkillIR provenance is invalid")
    return PilotSkillIRWire.model_validate(
        {
            "objective": value.objective,
            "objective_evidence_indices": _indices_for_references(
                evidence.get("/objective"), event_indices
            ),
            "ordered_steps": [
                {
                    "step_id": step.step_id,
                    "action_intent": step.action_intent,
                    "tool_name": step.tool_name,
                    "argument_bindings": [
                        {"key": key, "value": binding}
                        for key, binding in step.argument_bindings.items()
                    ],
                    "optional": step.optional,
                    "evidence_indices": _indices_for_references(
                        list(step.evidence_refs), event_indices
                    ),
                }
                for step in value.ordered_steps
            ],
        }
    )


__all__ = [
    "PilotSkillIRWire",
    "pilot_wire_from_skill_ir",
    "pilot_wire_schema_hash",
    "pilot_wire_to_skill_ir",
]
