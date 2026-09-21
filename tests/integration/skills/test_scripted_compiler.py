from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.models import (
    ModelAdapterError,
    ModelResponse,
    ProviderCapabilities,
    ScriptedModelClient,
    TokenPricing,
    TransportResponse,
)
from shadowskillbench.skills import (
    CompilerInput,
    SkillIR,
    compiler_input_projection,
    compiler_view,
    hash_compiler_input,
    hash_rendered_skill,
    hash_skill_ir,
    render_skill,
    skill_ir_projection,
)
from shadowskillbench.skills.compiler import (
    CONFIRMATORY_WIRE_PROMPT_PROFILE,
    PILOT_WIRE_PROMPT_PROFILE,
    CompiledSkillArtifact,
    CompilerConfig,
    CompilerContractError,
    CompilerManifest,
    compile_skill,
    compiled_skill_artifact_hash,
    compiled_skill_artifact_projection,
    compiler_manifest_hash,
    compiler_manifest_projection,
)
from shadowskillbench.skills.confirmatory_wire import (
    ConfirmatorySkillIRWire,
    confirmatory_wire_from_skill_ir,
)
from shadowskillbench.skills.pilot_wire import PilotSkillIRWire
from shadowskillbench.traces.bundles import DemonstrationBundle, generate_bundle

_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "skill_compiler.md"
_PROMPT_PROFILE = "SSB-SKILL-COMPILER1"
_REQUEST_PROFILE = "SSB-SKILL-COMPILER-REQUEST1"
_PROVENANCE_PROFILE = "SSB-INSTRUCTION-PROVENANCE1"
_FIXED_MANIFEST_HASH = "sha256:5d1d8e92a3e6738fe1110173fedb3d78015acf180512ee45ce9f0d6bfcc27618"
_FIXED_ARTIFACT_HASH = "sha256:d40624392b812dced2a6e0a11eba0fc8e316397ee7e2533e4ba526f120a82ad9"


def _prompt() -> bytes:
    return _PROMPT_PATH.read_bytes()


def _raw_hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _independent_hash_ref(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _caps(**changes: object) -> ProviderCapabilities:
    values: dict[str, object] = {
        "provider": "scripted",
        "model": "skill-compiler-test",
        "model_version": "2026-08-24",
        "supports_system_role": True,
        "supports_developer_role": False,
        "supports_seed": True,
        "supports_structured_output": True,
    }
    values.update(changes)
    return ProviderCapabilities(**values)


def _config(**changes: object) -> CompilerConfig:
    values: dict[str, object] = {
        "prompt_profile": _PROMPT_PROFILE,
        "prompt_bytes": _prompt(),
        "temperature": 0.25,
        "seed": 314,
        "max_tokens": 777,
    }
    values.update(changes)
    return CompilerConfig(**values)


def _manifest(
    bundle: DemonstrationBundle,
    config: CompilerConfig,
    caps: ProviderCapabilities,
) -> tuple[CompilerManifest, object]:
    view = compiler_view(bundle)
    input_hash = hash_compiler_input(view)
    manifest = CompilerManifest(
        manifest_profile="SSB-COMPILER-MANIFEST1",
        compiler_input_hash=input_hash,
        prompt_profile=config.prompt_profile,
        system_prompt_raw_hash=_raw_hash(config.prompt_bytes),
        request_profile=_REQUEST_PROFILE,
        instruction_provenance_profile=_PROVENANCE_PROFILE,
        structured_output_schema_hash=sha256_ref(SkillIR.model_json_schema()),
        declared_capabilities=caps,
        temperature=config.temperature,
        seed=config.seed,
        max_tokens=config.max_tokens,
    )
    return manifest, view


def _skill_value(
    bundle: DemonstrationBundle,
    config: CompilerConfig,
    caps: ProviderCapabilities,
    *,
    objective: str = "Handle the visible request using the demonstrated tools.",
    evidence_order: tuple[int, int] = (0, 1),
    mutate: Callable[[dict[str, object]], None] | None = None,
) -> tuple[dict[str, object], CompilerManifest, object]:
    manifest, view = _manifest(bundle, config, caps)
    manifest_hash = compiler_manifest_hash(manifest)
    input_hash = hash_compiler_input(view)
    event_ids = [event.event_id for trace in view.traces for event in trace.events]
    evidence = [event_ids[index] for index in evidence_order]
    instruction_evidence = {
        "/objective": list(evidence),
        "/applicability/0": [event_ids[2]],
        "/required_inputs/0": [event_ids[3]],
        "/preconditions/0": [event_ids[4]],
        "/ordered_steps/0": list(evidence),
        "/decision_hints/0": [event_ids[5]],
        "/verification_steps/0": [event_ids[6]],
        "/stop_conditions/0": [event_ids[7]],
        "/escalation_hints/0": [event_ids[8]],
    }
    raw: dict[str, object] = {
        "skill_id": f"skill_{view.domain}",
        "schema_version": "1.0",
        "domain": view.domain,
        "objective": objective,
        "applicability": ["Use when the visible task matches the demonstrated workflow."],
        "required_inputs": ["Visible task record."],
        "preconditions": ["The visible task state is available."],
        "ordered_steps": [
            {
                "step_id": "inspect_visible_state",
                "action_intent": "Inspect the visible task state.",
                "tool_name": "inspect_state",
                "argument_bindings": {"scope": "visible"},
                "preconditions": ["The task record is available."],
                "optional": False,
                "evidence_refs": list(evidence),
            }
        ],
        "decision_hints": ["Use the observed task state."],
        "verification_steps": ["Confirm the tool result is recorded."],
        "stop_conditions": ["Stop when required visible input is absent."],
        "escalation_hints": ["Escalate unresolved visible state."],
        "source_trace_ids": [trace.trace_id for trace in view.traces],
        "instruction_provenance": {
            "profile": _PROVENANCE_PROFILE,
            "compiler_input_hash": input_hash,
            "instruction_evidence": instruction_evidence,
        },
        "compiler_manifest_ref": manifest_hash,
    }
    if mutate is not None:
        mutate(raw)
    return raw, manifest, view


def _response_body(caps: ProviderCapabilities, skill: dict[str, object]) -> bytes:
    payload = {
        "model": caps.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": canonical_json_bytes(skill).decode("utf-8"),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 23, "total_tokens": 40},
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _client(
    caps: ProviderCapabilities,
    skill: dict[str, object],
    *,
    pricing: TokenPricing | None = None,
) -> ScriptedModelClient:
    return ScriptedModelClient(
        capabilities=caps,
        script=(TransportResponse(status_code=200, body=_response_body(caps, skill)),),
        max_attempts=1,
        pricing=pricing,
    )


def _bundle(domain: str) -> DemonstrationBundle:
    return generate_bundle(domain, Decimal("0.5"), 12, 4242).bundle


def _run(
    bundle: DemonstrationBundle, client: ScriptedModelClient, config: CompilerConfig
) -> CompiledSkillArtifact:
    return asyncio.run(compile_skill(bundle, client, config))


class _UnadmittedOutput(BaseModel):
    value: str


class _SkillIRSubclass(SkillIR):
    pass


@pytest.mark.parametrize("output_schema", [_UnadmittedOutput, _SkillIRSubclass])
def test_compiler_rejects_unadmitted_output_schema_before_provider_call(
    output_schema: type[BaseModel],
) -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    skill, _, _ = _skill_value(bundle, config, caps)
    client = _client(caps, skill)

    with pytest.raises(CompilerContractError, match="COMPILER_REQUEST_INVALID"):
        asyncio.run(compile_skill(bundle, client, config, output_schema=output_schema))

    assert client.recorded_request_bodies == ()


@pytest.mark.parametrize(
    ("prompt_profile", "output_schema"),
    [
        ("SSB-SKILL-COMPILER1", PilotSkillIRWire),
        (PILOT_WIRE_PROMPT_PROFILE, SkillIR),
        (CONFIRMATORY_WIRE_PROMPT_PROFILE, SkillIR),
    ],
)
def test_compiler_rejects_mismatched_prompt_profile_and_output_schema_before_call(
    prompt_profile: str, output_schema: type[BaseModel]
) -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config(prompt_profile=prompt_profile)
    client = ScriptedModelClient(capabilities=caps, script=(), max_attempts=1)

    with pytest.raises(CompilerContractError, match="COMPILER_REQUEST_INVALID"):
        asyncio.run(compile_skill(bundle, client, config, output_schema=output_schema))

    assert client.recorded_request_bodies == ()


@pytest.mark.parametrize("domain", ["access_provisioning", "financial_adjustments"])
def test_scripted_compiler_binds_the_complete_request_and_artifact(domain: str) -> None:
    bundle = _bundle(domain)
    caps = _caps()
    config = _config()
    skill, manifest, view = _skill_value(bundle, config, caps)
    pricing = TokenPricing(currency="USD", input_nanos_per_token=3, output_nanos_per_token=5)
    client = _client(caps, skill, pricing=pricing)

    artifact = _run(bundle, client, config)
    expected_input = compiler_input_projection(view)
    expected_manifest_hash = compiler_manifest_hash(manifest)
    request_body = client.recorded_request_bodies
    assert len(request_body) == 1

    wire = json.loads(request_body[0])
    assert wire == {
        "model": caps.model,
        "messages": [
            {"role": "system", "content": config.prompt_bytes.decode("utf-8")},
            {
                "role": "user",
                "content": canonical_json_bytes(
                    {
                        "request_profile": _REQUEST_PROFILE,
                        "compiler_input": expected_input,
                        "required_output_bindings": {
                            "compiler_manifest_ref": expected_manifest_hash,
                            "compiler_input_hash": hash_compiler_input(view),
                            "instruction_provenance_profile": _PROVENANCE_PROFILE,
                        },
                    }
                ).decode("utf-8"),
            },
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
        "seed": config.seed,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "shadowskillbench_output",
                "strict": True,
                "schema": SkillIR.model_json_schema(),
            },
        },
    }
    user_envelope = json.loads(wire["messages"][1]["content"])
    assert user_envelope["compiler_input"] == expected_input
    assert tuple(user_envelope) == (
        "compiler_input",
        "request_profile",
        "required_output_bindings",
    )
    assert "worker_policy_id" not in wire["messages"][1]["content"]
    for canary in (
        "semantic_class",
        "contamination_ratio",
        "hidden_benchmark_metadata_ref",
        "source_manifest",
        "hidden_class_manifest",
        "ordered_trace_hashes",
        "hidden_class_manifest_hash",
        "compliance_label",
        "policy_id",
        "authority_record",
        "heldout_case",
        "verifier_outcome",
        "cup_outcome",
        "scorer_outcome",
    ):
        assert canary not in wire["messages"][1]["content"]

    assert tuple(artifact.__dict__) == (
        "artifact_profile",
        "compiler_manifest",
        "compiler_manifest_hash",
        "request_envelope_hash",
        "skill_ir",
        "skill_ir_hash",
        "rendered_skill",
        "rendered_skill_hash",
        "raw_request_hash",
        "raw_response_hash",
        "structured_output_schema_hash",
        "usage",
        "cost",
        "attempts",
    )
    assert artifact.compiler_manifest == manifest
    assert artifact.compiler_manifest_hash == expected_manifest_hash
    assert artifact.request_envelope_hash == sha256_ref(user_envelope)
    assert artifact.request_envelope_hash != artifact.compiler_manifest_hash
    assert artifact.skill_ir == SkillIR.model_validate(skill)
    assert artifact.skill_ir_hash == hash_skill_ir(skill)
    assert artifact.rendered_skill == render_skill(skill)
    assert artifact.rendered_skill_hash == hash_rendered_skill(skill)
    assert artifact.structured_output_schema_hash == sha256_ref(SkillIR.model_json_schema())
    assert artifact.raw_request_hash == _raw_hash(request_body[0])
    raw_response = _response_body(caps, skill)
    assert artifact.raw_response_hash == _raw_hash(raw_response)
    assert artifact.usage.input_tokens == 17
    assert artifact.usage.output_tokens == 23
    assert artifact.usage.total_tokens == 40
    assert artifact.cost is not None and artifact.cost.total_nanos == 166
    assert artifact.attempts == 1
    assert compiler_manifest_projection(manifest)["compiler_input_hash"] == hash_compiler_input(
        view
    )
    assert compiled_skill_artifact_hash(artifact) == sha256_ref(
        compiled_skill_artifact_projection(artifact)
    )

    copied = compiled_skill_artifact_projection(artifact)
    copied["rendered_skill"] = "tampered"
    assert compiled_skill_artifact_projection(artifact)["rendered_skill"] == artifact.rendered_skill
    assert client.recorded_request_bodies == (bytes(request_body[0]),)


def test_confirmatory_wire_is_an_admitted_compiler_output_profile() -> None:
    bundle = _bundle("access_provisioning")
    capabilities = _caps()
    config = _config(prompt_profile=CONFIRMATORY_WIRE_PROMPT_PROFILE)
    skill, manifest, compiler_input = _skill_value(bundle, config, capabilities)
    wire = confirmatory_wire_from_skill_ir(
        SkillIR.model_validate(skill),
        cast(CompilerInput, compiler_input),
        compiler_manifest_hash(manifest),
    )
    client = ScriptedModelClient(
        capabilities=capabilities,
        script=(
            TransportResponse(
                status_code=200,
                body=_response_body(capabilities, wire.model_dump(mode="json")),
            ),
        ),
        max_attempts=1,
    )

    artifact = asyncio.run(
        compile_skill(bundle, client, config, output_schema=ConfirmatorySkillIRWire)
    )

    assert artifact.compiler_manifest.prompt_profile == CONFIRMATORY_WIRE_PROMPT_PROFILE
    assert artifact.skill_ir.compiler_manifest_ref == artifact.compiler_manifest_hash
    assert artifact.skill_ir.instruction_provenance["compiler_input_hash"] == hash_compiler_input(
        compiler_input
    )


def test_scripted_compiler_is_stable_and_prompt_bytes_are_hash_sensitive() -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    skill, _, _ = _skill_value(bundle, config, caps)
    first_client = _client(caps, skill)
    second_client = _client(caps, skill)
    first = _run(bundle, first_client, config)
    second = _run(bundle, second_client, config)
    assert first == second
    assert first_client.recorded_request_bodies == second_client.recorded_request_bodies

    changed_prompt = bytes(config.prompt_bytes).replace(b"skill", b"Skill", 1)
    assert changed_prompt != config.prompt_bytes
    changed_config = _config(prompt_bytes=changed_prompt)
    changed_skill, _, _ = _skill_value(bundle, changed_config, caps)
    changed_client = _client(caps, changed_skill)
    changed = _run(bundle, changed_client, changed_config)
    assert (
        changed.compiler_manifest.system_prompt_raw_hash
        != first.compiler_manifest.system_prompt_raw_hash
    )
    assert changed.compiler_manifest_hash != first.compiler_manifest_hash
    assert changed.request_envelope_hash != first.request_envelope_hash
    assert changed_client.recorded_request_bodies != first_client.recorded_request_bodies


def test_fixed_scripted_fixture_has_independently_hashed_manifest_and_artifact() -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    skill, manifest, _ = _skill_value(bundle, config, caps)
    artifact = _run(bundle, _client(caps, skill), config)
    manifest_projection = compiler_manifest_projection(manifest)
    artifact_projection = compiled_skill_artifact_projection(artifact)
    independent_manifest_hash = _independent_hash_ref(manifest_projection)
    independent_artifact_hash = _independent_hash_ref(artifact_projection)

    assert set(manifest_projection) == {
        "manifest_profile",
        "compiler_input_hash",
        "prompt_profile",
        "system_prompt_raw_hash",
        "request_profile",
        "instruction_provenance_profile",
        "structured_output_schema_hash",
        "declared_capabilities",
        "temperature",
        "seed",
        "max_tokens",
    }
    assert set(artifact_projection) == {
        "artifact_profile",
        "compiler_manifest",
        "compiler_manifest_hash",
        "request_envelope_hash",
        "skill_ir",
        "skill_ir_hash",
        "rendered_skill",
        "rendered_skill_hash",
        "raw_request_hash",
        "raw_response_hash",
        "structured_output_schema_hash",
        "usage",
        "cost",
        "attempts",
    }
    for field in manifest_projection:
        changed = dict(manifest_projection)
        changed[field] = f"changed-{field}"
        assert _independent_hash_ref(changed) != independent_manifest_hash
    for field in artifact_projection:
        changed = dict(artifact_projection)
        changed[field] = f"changed-{field}"
        assert _independent_hash_ref(changed) != independent_artifact_hash

    assert independent_manifest_hash == _FIXED_MANIFEST_HASH
    assert independent_artifact_hash == _FIXED_ARTIFACT_HASH
    assert compiler_manifest_hash(manifest) == independent_manifest_hash
    assert compiled_skill_artifact_hash(artifact) == independent_artifact_hash


def test_compiler_calls_the_frozen_projection_sequence_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    skill, _, _ = _skill_value(bundle, config, caps)
    calls: list[str] = []

    def tracked_view(value: DemonstrationBundle) -> CompilerInput:
        calls.append("view")
        return compiler_view(value)

    def tracked_projection(value: CompilerInput) -> dict[str, object]:
        calls.append("projection")
        return cast(dict[str, object], compiler_input_projection(value))

    def tracked_hash(value: CompilerInput) -> str:
        calls.append("hash")
        return hash_compiler_input(value)

    monkeypatch.setattr("shadowskillbench.skills.compiler.compiler_view", tracked_view)
    monkeypatch.setattr(
        "shadowskillbench.skills.compiler.compiler_input_projection", tracked_projection
    )
    monkeypatch.setattr("shadowskillbench.skills.compiler.hash_compiler_input", tracked_hash)

    _run(bundle, _client(caps, skill), config)
    assert calls == ["view", "projection", "hash"]


@pytest.mark.parametrize(
    "binding",
    (
        "schema-hash",
        "manifest-ref",
        "provenance-profile",
        "provenance-input-hash",
        "provenance-extra-key",
        "provenance-evidence-not-object",
    ),
)
def test_direct_artifact_construction_rejects_cross_binding_drift(
    binding: str,
) -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    skill, _, _ = _skill_value(bundle, config, caps)
    artifact = _run(bundle, _client(caps, skill), config)
    values = dict(artifact.__dict__)
    raw_skill = skill_ir_projection(artifact.skill_ir)
    if binding == "schema-hash":
        values["structured_output_schema_hash"] = "sha256:" + "0" * 64
    elif binding == "manifest-ref":
        raw_skill["compiler_manifest_ref"] = "sha256:" + "1" * 64
    elif binding == "provenance-profile":
        provenance = cast(dict[str, object], raw_skill["instruction_provenance"])
        provenance["profile"] = "SSB-WRONG1"
    elif binding == "provenance-input-hash":
        provenance = cast(dict[str, object], raw_skill["instruction_provenance"])
        provenance["compiler_input_hash"] = "sha256:" + "2" * 64
    elif binding == "provenance-extra-key":
        provenance = cast(dict[str, object], raw_skill["instruction_provenance"])
        provenance["unexpected"] = "not admitted"
    else:
        provenance = cast(dict[str, object], raw_skill["instruction_provenance"])
        provenance["instruction_evidence"] = []
    values["skill_ir"] = SkillIR.model_validate(raw_skill)

    with pytest.raises(ValueError, match="invalid|bindings"):
        CompiledSkillArtifact(**values)


def test_direct_artifact_construction_rejects_cost_usage_drift() -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    skill, _, _ = _skill_value(bundle, config, caps)
    pricing = TokenPricing(currency="USD", input_nanos_per_token=3, output_nanos_per_token=5)
    artifact = _run(bundle, _client(caps, skill, pricing=pricing), config)
    assert artifact.cost is not None
    values = dict(artifact.__dict__)
    values["cost"] = artifact.cost.model_copy(
        update={
            "input_nanos": artifact.cost.input_nanos + 1,
            "total_nanos": artifact.cost.total_nanos + 1,
        }
    )

    with pytest.raises(ValueError, match="cost does not bind usage and rates"):
        CompiledSkillArtifact(**values)


def _assert_sanitized_error(error: CompilerContractError, code: str, secret_canary: str) -> None:
    assert error.code == code
    assert error.args == (code,)
    assert error.__cause__ is None
    assert error.__context__ is None
    printable = "\n".join((str(error), repr(error), repr(error.args)))
    assert secret_canary not in printable


def _precontext_contract_error(secret_canary: str) -> CompilerContractError:
    try:
        raise RuntimeError(secret_canary)
    except RuntimeError as cause:
        try:
            raise CompilerContractError("COMPILER_OUTPUT_INVALID") from cause
        except CompilerContractError as error:
            return error


def test_compiler_contract_errors_detach_caught_exceptions_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    skill, _, _ = _skill_value(bundle, config, caps)

    secret_canary = "compiler-secret-canary"

    class CapabilityFailureClient:
        @property
        def capabilities(self) -> ProviderCapabilities:
            raise _precontext_contract_error(secret_canary)

        async def structured(self, request: object, schema: object) -> object:
            raise AssertionError("must not be called")

    with pytest.raises(CompilerContractError) as caught:
        asyncio.run(compile_skill(bundle, CapabilityFailureClient(), config))
    _assert_sanitized_error(caught.value, "COMPILER_CONFIGURATION_ERROR", secret_canary)

    monkeypatch.setattr(
        "shadowskillbench.skills.compiler.compiler_view",
        lambda _: (_ for _ in ()).throw(_precontext_contract_error(secret_canary)),
    )
    with pytest.raises(CompilerContractError) as caught:
        _run(bundle, _client(caps, skill), config)
    _assert_sanitized_error(caught.value, "COMPILER_REQUEST_INVALID", secret_canary)

    monkeypatch.setattr("shadowskillbench.skills.compiler.compiler_view", compiler_view)
    monkeypatch.setattr(
        "shadowskillbench.skills.compiler._validate_output_bindings",
        lambda *_: (_ for _ in ()).throw(_precontext_contract_error(secret_canary)),
    )
    with pytest.raises(CompilerContractError) as caught:
        _run(bundle, _client(caps, skill), config)
    _assert_sanitized_error(caught.value, "COMPILER_OUTPUT_INVALID", secret_canary)

    class StructuredFailureClient:
        capabilities = caps

        async def structured(self, request: object, schema: object) -> object:
            raise RuntimeError(secret_canary)

    with pytest.raises(CompilerContractError) as caught:
        asyncio.run(compile_skill(bundle, StructuredFailureClient(), config))
    _assert_sanitized_error(caught.value, "COMPILER_OUTPUT_INVALID", secret_canary)


def test_pilot_wire_semantic_failure_retains_only_safe_response_custody() -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config(prompt_profile=PILOT_WIRE_PROMPT_PROFILE)
    wire = {
        "objective": "Handle the visible request.",
        "objective_evidence_indices": [4095],
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
    client = _client(caps, wire)

    with pytest.raises(CompilerContractError) as caught:
        asyncio.run(compile_skill(bundle, client, config, output_schema=PilotSkillIRWire))

    error = caught.value
    _assert_sanitized_error(error, "COMPILER_OUTPUT_INVALID", "Handle the visible request")
    assert error.stage == "pilot_wire_conversion"
    assert error.validation_code == "PILOT_WIRE_EVENT_INDEX_OUT_OF_RANGE"
    assert error.validation_paths == ()
    assert error.finish_reason == "stop"
    assert error.finish_reason_source == "adapter_success_contract"
    assert error.reported_usage is not None
    assert error.reported_usage.model_dump(mode="json") == {
        "input_tokens": 17,
        "output_tokens": 23,
        "total_tokens": 40,
    }
    assert error.attempts == 1
    assert error.raw_request_hash is not None
    assert error.raw_response_hash is not None
    with pytest.raises(AttributeError, match="immutable"):
        error.stage = "rendering"  # type: ignore[misc]

    error.__traceback__ = error.__traceback__


def test_compiler_rejects_an_unowned_exact_model_response() -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    skill, _, _ = _skill_value(bundle, config, caps)
    response = ModelResponse[SkillIR](
        capabilities=caps,
        output=SkillIR.model_validate(skill),
        raw_request_hash="sha256:" + "a" * 64,
        raw_response_hash="sha256:" + "b" * 64,
        structured_output_schema_hash=sha256_ref(SkillIR.model_json_schema()),
        usage={"input_tokens": 17, "output_tokens": 23, "total_tokens": 40},
        cost=None,
        attempts=1,
    )
    forged = object.__new__(type(response))
    for attribute in (
        "__dict__",
        "__pydantic_fields_set__",
        "__pydantic_extra__",
        "__pydantic_private__",
    ):
        value = object.__getattribute__(response, attribute)
        copied = value.copy() if type(value) in {dict, set} else value
        object.__setattr__(forged, attribute, copied)

    class ForgedResponseClient:
        capabilities = caps

        async def structured(self, request: object, schema: object) -> object:
            del request, schema
            return forged

    artifact: CompiledSkillArtifact | None = None
    with pytest.raises(CompilerContractError) as caught:
        artifact = asyncio.run(compile_skill(bundle, ForgedResponseClient(), config))
    _assert_sanitized_error(caught.value, "COMPILER_OUTPUT_INVALID", "forged-response")
    assert artifact is None


def test_compiler_propagates_cancellation() -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()

    class CancelledClient:
        capabilities = caps

        async def structured(self, request: object, schema: object) -> object:
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(compile_skill(bundle, CancelledClient(), config))


def _changed_domain(value: dict[str, object]) -> None:
    value["domain"] = "financial_adjustments"


def _changed_skill_id(value: dict[str, object]) -> None:
    value["skill_id"] = "skill_other"


def _reordered_sources(value: dict[str, object]) -> None:
    sources = cast(list[str], value["source_trace_ids"])
    value["source_trace_ids"] = list(reversed(sources))


def _wrong_manifest_ref(value: dict[str, object]) -> None:
    value["compiler_manifest_ref"] = "sha256:" + "0" * 64


def _wrong_provenance_profile(value: dict[str, object]) -> None:
    provenance = cast(dict[str, object], value["instruction_provenance"])
    provenance["profile"] = "SSB-WRONG1"


def _wrong_input_hash(value: dict[str, object]) -> None:
    provenance = cast(dict[str, object], value["instruction_provenance"])
    provenance["compiler_input_hash"] = "sha256:" + "1" * 64


def _wrong_pointer_set(value: dict[str, object]) -> None:
    provenance = cast(dict[str, object], value["instruction_provenance"])
    evidence = cast(dict[str, object], provenance["instruction_evidence"])
    evidence.pop("/objective")


def _unresolved_reference(value: dict[str, object]) -> None:
    provenance = cast(dict[str, object], value["instruction_provenance"])
    evidence = cast(dict[str, list[str]], provenance["instruction_evidence"])
    evidence["/objective"] = ["event_missing_000000"]


def _duplicate_reference(value: dict[str, object]) -> None:
    provenance = cast(dict[str, object], value["instruction_provenance"])
    evidence = cast(dict[str, list[str]], provenance["instruction_evidence"])
    evidence["/objective"] = [evidence["/objective"][0]] * 2


def _step_mismatch(value: dict[str, object]) -> None:
    steps = cast(list[dict[str, object]], value["ordered_steps"])
    steps[0]["evidence_refs"] = ["event_missing_000000"]


@pytest.mark.parametrize(
    "mutate",
    [
        _changed_domain,
        _changed_skill_id,
        _reordered_sources,
        _wrong_manifest_ref,
        _wrong_provenance_profile,
        _wrong_input_hash,
        _wrong_pointer_set,
        _unresolved_reference,
        _duplicate_reference,
        _step_mismatch,
    ],
    ids=[
        "domain",
        "skill-id",
        "source-order",
        "manifest-ref",
        "provenance-profile",
        "input-hash",
        "pointer-set",
        "unresolved-reference",
        "duplicate-reference",
        "step-evidence-mismatch",
    ],
)
def test_scripted_compiler_rejects_output_binding_drift(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    skill, _, _ = _skill_value(bundle, config, caps, mutate=mutate)
    client = _client(caps, skill)
    with pytest.raises(CompilerContractError) as caught:
        _run(bundle, client, config)
    assert caught.value.code == "COMPILER_OUTPUT_INVALID"
    assert len(client.recorded_request_bodies) == 1


def test_adapter_output_and_transport_errors_propagate_without_repair_or_retry() -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    malformed = {
        "model": caps.model,
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    bad_client = ScriptedModelClient(
        capabilities=caps,
        script=(
            TransportResponse(
                status_code=200,
                body=json.dumps(malformed, separators=(",", ":")).encode("utf-8"),
            ),
        ),
        max_attempts=1,
    )
    with pytest.raises(ModelAdapterError) as malformed_error:
        _run(bundle, bad_client, config)
    assert malformed_error.value.code == "MODEL_OUTPUT_INVALID"
    assert len(bad_client.recorded_request_bodies) == 1

    terminal_client = ScriptedModelClient(capabilities=caps, script=("terminal",), max_attempts=1)
    with pytest.raises(ModelAdapterError) as terminal_error:
        _run(bundle, terminal_client, config)
    assert terminal_error.value.code == "MODEL_PROVIDER_TERMINAL"
    assert len(terminal_client.recorded_request_bodies) == 1


@pytest.mark.parametrize(
    "capability_change",
    [
        {"supports_system_role": False},
        {"supports_structured_output": False},
        {"supports_seed": False},
    ],
)
def test_unsupported_compiler_capabilities_fail_before_the_adapter_call(
    capability_change: dict[str, object],
) -> None:
    bundle = _bundle("financial_adjustments")
    caps = _caps(**capability_change)
    config = _config()
    skill, _, _ = _skill_value(bundle, config, _caps())
    client = _client(caps, skill)
    with pytest.raises(CompilerContractError) as caught:
        _run(bundle, client, config)
    assert caught.value.code == "COMPILER_CONFIGURATION_ERROR"
    assert client.recorded_request_bodies == ()


def _oversized_view() -> CompilerInput:
    traces: list[dict[str, object]] = []
    for trace_number in range(12):
        trace_id = f"trace_large_{trace_number}"
        events: list[dict[str, object]] = []
        for index, kind in enumerate(("observation", "action", "tool_result", "state_delta")):
            events.append(
                {
                    "event_id": f"event_{trace_id}_{index:06d}",
                    "index": index,
                    "kind": kind,
                    "payload": {"visible": "x" * 100_000},
                }
            )
        traces.append(
            {
                "trace_id": trace_id,
                "domain": "access_provisioning",
                "task_template_id": "visible_task",
                "worker_role": "visible_worker",
                "world_hash": "sha256:" + "a" * 64,
                "events": events,
                "terminal_state_hash": "sha256:" + "b" * 64,
                "local_task_outcome": "completed",
            }
        )
    return CompilerInput.model_validate(
        {
            "projection_profile": "SSB-COMPILER-VIEW1",
            "domain": "access_provisioning",
            "traces": traces,
        }
    )


def test_over_budget_compiler_request_is_rejected_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("access_provisioning")
    caps = _caps()
    config = _config()
    oversized = _oversized_view()
    monkeypatch.setattr("shadowskillbench.skills.compiler.compiler_view", lambda _: oversized)
    client = ScriptedModelClient(capabilities=caps, script=(), max_attempts=1)
    with pytest.raises(CompilerContractError) as caught:
        _run(bundle, client, config)
    assert caught.value.code == "COMPILER_REQUEST_INVALID"
    assert client.recorded_request_bodies == ()
