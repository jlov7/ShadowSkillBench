from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.models import ProviderCapabilities, ScriptedModelClient, TransportResponse
from shadowskillbench.skills import SkillIR, compiler_view, hash_compiler_input
from shadowskillbench.skills.compiler import (
    CompilerConfig,
    CompilerManifest,
    compile_skill,
    compiled_skill_artifact_hash,
    compiler_manifest_hash,
)
from shadowskillbench.traces.bundles import DemonstrationBundle, generate_bundle

_PROMPT = Path(__file__).resolve().parents[3] / "prompts" / "skill_compiler.md"


def _caps() -> ProviderCapabilities:
    return ProviderCapabilities(
        provider="scripted",
        model="property-compiler",
        model_version="v1",
        supports_system_role=True,
        supports_developer_role=False,
        supports_seed=True,
        supports_structured_output=True,
    )


def _config(temperature: float, seed: int | None, max_tokens: int) -> CompilerConfig:
    return CompilerConfig(
        prompt_profile="SSB-SKILL-COMPILER1",
        prompt_bytes=_PROMPT.read_bytes(),
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )


def _manifest(
    bundle: DemonstrationBundle,
    config: CompilerConfig,
    caps: ProviderCapabilities,
) -> tuple[CompilerManifest, object]:
    view = compiler_view(bundle)
    manifest = CompilerManifest(
        manifest_profile="SSB-COMPILER-MANIFEST1",
        compiler_input_hash=hash_compiler_input(view),
        prompt_profile=config.prompt_profile,
        system_prompt_raw_hash="sha256:" + hashlib.sha256(config.prompt_bytes).hexdigest(),
        request_profile="SSB-SKILL-COMPILER-REQUEST1",
        instruction_provenance_profile="SSB-INSTRUCTION-PROVENANCE1",
        structured_output_schema_hash=sha256_ref(SkillIR.model_json_schema()),
        declared_capabilities=caps,
        temperature=config.temperature,
        seed=config.seed,
        max_tokens=config.max_tokens,
    )
    return manifest, view


def _skill(
    bundle: DemonstrationBundle,
    config: CompilerConfig,
    caps: ProviderCapabilities,
    *,
    objective: str,
    evidence_order: tuple[int, int],
) -> dict[str, object]:
    manifest, view = _manifest(bundle, config, caps)
    events = [event.event_id for trace in view.traces for event in trace.events]
    evidence = [events[index] for index in evidence_order]
    return {
        "skill_id": f"skill_{view.domain}",
        "schema_version": "1.0",
        "domain": view.domain,
        "objective": objective,
        "applicability": ["Use for the observed visible task."],
        "required_inputs": ["Visible task state."],
        "preconditions": ["A visible task record is available."],
        "ordered_steps": [
            {
                "step_id": "inspect_visible_state",
                "action_intent": "Inspect visible state.",
                "tool_name": "inspect_state",
                "argument_bindings": {"scope": "visible"},
                "preconditions": [],
                "optional": False,
                "evidence_refs": evidence,
            }
        ],
        "decision_hints": ["Use recorded visible information."],
        "verification_steps": ["Confirm a tool result is recorded."],
        "stop_conditions": ["Stop if visible input is absent."],
        "escalation_hints": ["Escalate unresolved visible state."],
        "source_trace_ids": [trace.trace_id for trace in view.traces],
        "instruction_provenance": {
            "profile": "SSB-INSTRUCTION-PROVENANCE1",
            "compiler_input_hash": hash_compiler_input(view),
            "instruction_evidence": {
                "/objective": evidence,
                "/applicability/0": [events[2]],
                "/required_inputs/0": [events[3]],
                "/preconditions/0": [events[4]],
                "/ordered_steps/0": evidence,
                "/decision_hints/0": [events[5]],
                "/verification_steps/0": [events[6]],
                "/stop_conditions/0": [events[7]],
                "/escalation_hints/0": [events[8]],
            },
        },
        "compiler_manifest_ref": compiler_manifest_hash(manifest),
    }


def _client(caps: ProviderCapabilities, value: dict[str, object]) -> ScriptedModelClient:
    response = {
        "model": caps.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": canonical_json_bytes(value).decode()},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }
    return ScriptedModelClient(
        capabilities=caps,
        script=(
            TransportResponse(
                status_code=200,
                body=json.dumps(response, separators=(",", ":")).encode(),
            ),
        ),
        max_attempts=1,
    )


def _run(bundle: DemonstrationBundle, config: CompilerConfig, value: dict[str, object]):
    caps = _caps()
    client = _client(caps, value)
    return asyncio.run(compile_skill(bundle, client, config)), client.recorded_request_bodies


@pytest.mark.property
@given(
    domain=st.sampled_from(["access_provisioning", "financial_adjustments"]),
    temperature=st.sampled_from([0.0, 0.25, 1.0]),
    seed=st.one_of(st.none(), st.integers(min_value=-9, max_value=9)),
    max_tokens=st.sampled_from([32, 256, 1024]),
)
@settings(max_examples=12, deadline=None)
def test_admissible_bundle_and_config_are_repeatably_compiled(
    domain: str, temperature: float, seed: int | None, max_tokens: int
) -> None:
    bundle = generate_bundle(domain, Decimal("0.5"), 12, 808).bundle
    caps = _caps()
    config = _config(temperature, seed, max_tokens)
    value = _skill(bundle, config, caps, objective="Use visible evidence.", evidence_order=(0, 1))
    first, first_request = _run(bundle, config, value)
    second, second_request = _run(bundle, config, value)
    assert first == second
    assert first_request == second_request

    alternative_temperature = 1.0 if temperature != 1.0 else 0.25
    changed_config = _config(alternative_temperature, seed, max_tokens)
    changed_value = _skill(
        bundle, changed_config, caps, objective="Use visible evidence.", evidence_order=(0, 1)
    )
    changed, changed_request = _run(bundle, changed_config, changed_value)
    assert changed.compiler_manifest_hash != first.compiler_manifest_hash
    assert changed_request != first_request
    assert compiled_skill_artifact_hash(changed) != compiled_skill_artifact_hash(first)


@pytest.mark.property
@given(
    objective_tail=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=32),
    reverse_evidence=st.booleans(),
)
@settings(max_examples=12, deadline=None)
def test_admissible_output_values_and_evidence_order_are_not_normalized(
    objective_tail: str, reverse_evidence: bool
) -> None:
    bundle = generate_bundle("access_provisioning", Decimal("0.5"), 12, 809).bundle
    caps = _caps()
    config = _config(0.25, 7, 256)
    clean_tail = objective_tail.strip() or "visible"
    baseline_value = _skill(
        bundle, config, caps, objective=f"Use {clean_tail} evidence.", evidence_order=(0, 1)
    )
    ordered_value = _skill(
        bundle,
        config,
        caps,
        objective=f"Use {clean_tail} evidence.",
        evidence_order=(1, 0) if reverse_evidence else (0, 1),
    )
    baseline, baseline_request = _run(bundle, config, baseline_value)
    ordered, ordered_request = _run(bundle, config, ordered_value)
    assert baseline_request == ordered_request
    if reverse_evidence:
        assert ordered.skill_ir.instruction_provenance != baseline.skill_ir.instruction_provenance
        assert compiled_skill_artifact_hash(ordered) != compiled_skill_artifact_hash(baseline)
    else:
        assert ordered == baseline
