from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.development import generate_development_corpus
from shadowskillbench.experiments import live_pilot
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
    OLLAMA_GPT_OSS_CONTEXT_LENGTH,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
    OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
    OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_MODEL,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
    OLLAMA_GPT_OSS_PROFILE,
    OLLAMA_GPT_OSS_PROVIDER,
    OLLAMA_GPT_OSS_REASONING_EFFORT,
    OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_SERVER_VERSION,
    canonical_role_profile_receipt,
    role_conformance_probe_projection,
)
from shadowskillbench.models import (
    ModelAdapterError,
    OllamaNativeClient,
    ProviderCapabilities,
    ScriptedModelClient,
    TokenUsage,
    TransportResponse,
)
from shadowskillbench.skills import gate2
from shadowskillbench.skills.compiler import (
    CompilerConfig,
    CompilerContractError,
    compiler_manifest_hash,
)
from shadowskillbench.skills.models import SkillIR
from shadowskillbench.skills.pilot_wire import pilot_wire_from_skill_ir
from shadowskillbench.skills.projection import compiler_view
from shadowskillbench.traces.bundles import generate_bundle

ROOT = Path(__file__).resolve().parents[3]
PROMPT = ROOT / "prompts" / "skill_compiler.md"
PROTOCOL = ROOT / "protocol" / "EXPLORATORY_LIVE_PILOT.md"


def _global_episode_artifact_snapshot() -> tuple[tuple[str, int, str], ...]:
    root = ROOT / "artifacts" / "episode_result"
    if not root.is_dir():
        return ()
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _descriptor(model: str = "pilot-model") -> live_pilot.LivePilotDescriptor:
    return live_pilot.LivePilotDescriptor(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        api_key_environment="SSB_TEST_KEY",
        provider="scripted-pilot",
        model=model,
        model_version="test-v1",
    )


def _gpt_receipt() -> dict[str, object]:
    probe = role_conformance_probe_projection()
    return json.loads(
        canonical_role_profile_receipt(
            "sha256:" + "1" * 64,
            baseline={
                "raw_request_hash": "sha256:" + "a" * 64,
                "raw_response_hash": "sha256:" + "b" * 64,
                "prompt_tokens": 100,
                "response_marker": probe["baseline_expected_marker"],
            },
            developer={
                "raw_request_hash": "sha256:" + "c" * 64,
                "raw_response_hash": "sha256:" + "d" * 64,
                "prompt_tokens": 101,
                "response_marker": probe["developer_expected_marker"],
            },
            executor_model_digest=OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
        )
    )


def _gpt_descriptor() -> live_pilot.LivePilotDescriptor:
    receipt = _gpt_receipt()
    return live_pilot.LivePilotDescriptor(
        endpoint="http://127.0.0.1:11435/v1/chat/completions",
        api_key_environment="SSB_TEST_KEY",
        provider=OLLAMA_GPT_OSS_PROVIDER,
        model=OLLAMA_GPT_OSS_MODEL,
        model_version="sha256:" + "1" * 64,
        ollama_server_version=OLLAMA_GPT_OSS_SERVER_VERSION,
        base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
        reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
        executor_reasoning_effort=OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
        sampling_temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
        executor_sampling_temperature=OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
        executor_endpoint="http://127.0.0.1:11435/api/generate",
        executor_transport=OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
        executor_request_profile=OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
        executor_request_profile_hash=OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
        executor_think=False,
        executor_model_name="ssb-gpt-oss-20b-harmony-final:v2",
        executor_model_version=OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
        executor_profile=OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
        executor_modelfile_sha256=live_pilot.ollama_gpt_oss_executor_modelfile_hash(),
        executor_template_sha256=live_pilot.ollama_gpt_oss_executor_template_hash(),
        compiler_max_tokens=live_pilot.OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
        declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
        provider_profile=OLLAMA_GPT_OSS_PROFILE,
        role_profile_receipt_hash=sha256_ref(receipt),
    )


def _response(capabilities: ProviderCapabilities, output: dict[str, object]) -> TransportResponse:
    return TransportResponse(
        status_code=200,
        body=canonical_json_bytes(
            {
                "model": capabilities.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": canonical_json_bytes(output).decode("utf-8"),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ),
    )


def _compiler_client(descriptor: live_pilot.LivePilotDescriptor) -> ScriptedModelClient:
    config = CompilerConfig(
        prompt_profile=live_pilot.PILOT_COMPILER_PROMPT_PROFILE,
        prompt_bytes=live_pilot._pilot_prompt_bytes(PROMPT.read_bytes()),
        temperature=descriptor.sampling_temperature,
        seed=4242,
        max_tokens=descriptor.compiler_max_tokens,
    )
    responses: list[TransportResponse] = []
    for domain in ("access_provisioning", "financial_adjustments"):
        for ratio in (Decimal("0"), Decimal("1")):
            compiler_input = compiler_view(generate_bundle(domain, ratio, 12, 4242).bundle)
            manifest = gate2._manifest(compiler_input, config, descriptor.capabilities).model_copy(
                update={"structured_output_schema_hash": live_pilot.PILOT_WIRE_SCHEMA_HASH}
            )
            skill = gate2._witness_skill(compiler_input, compiler_manifest_hash(manifest))
            wire = pilot_wire_from_skill_ir(
                SkillIR.model_validate(json.loads(canonical_json_bytes(skill))),
                compiler_input,
                compiler_manifest_hash(manifest),
            ).model_dump(mode="python")
            responses.append(_response(descriptor.capabilities, wire))
    return ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=tuple(responses),
        max_attempts=1,
        reasoning_effort=descriptor.reasoning_effort,
    )


class _NativeTransport:
    def __init__(self, responses: tuple[TransportResponse, ...]) -> None:
        self.recorded_request_bodies: list[bytes] = []
        self._responses = list(responses)

    async def __call__(self, body: bytes) -> TransportResponse:
        self.recorded_request_bodies.append(body)
        return self._responses.pop(0)


def _native_response(
    capabilities: ProviderCapabilities, output: dict[str, object]
) -> TransportResponse:
    return TransportResponse(
        status_code=200,
        body=canonical_json_bytes(
            {
                "model": capabilities.model,
                "response": canonical_json_bytes(output).decode("utf-8"),
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 2,
                "eval_count": 3,
            }
        ),
    )


def _executor_client(descriptor: live_pilot.LivePilotDescriptor) -> OllamaNativeClient:
    transport = _NativeTransport(
        (
            _native_response(
                descriptor.executor_capabilities, {"turn": {"kind": "finish", "summary": "done"}}
            ),
        )
    )
    return OllamaNativeClient(
        capabilities=descriptor.executor_capabilities,
        transport=transport,
        max_attempts=1,
    )


def _pilot_client_with_one_finish(
    descriptor: live_pilot.LivePilotDescriptor,
) -> ScriptedModelClient:
    compiler_client = _compiler_client(descriptor)
    return ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=(
            *compiler_client._script,
            _response(descriptor.capabilities, {"turn": {"kind": "finish", "summary": "done"}}),
        ),
        max_attempts=1,
        reasoning_effort=descriptor.executor_reasoning_effort,
    )


def test_live_pilot_plan_is_exactly_the_fixed_60_cell_design() -> None:
    plan = live_pilot.build_live_pilot_plan(generate_development_corpus(4242))

    assert len(plan) == 60
    assert sum(cell.condition.value == "A0_BARE" for cell in plan) == 20
    assert sum(cell.condition.value == "A2_SKILL_ONLY" for cell in plan) == 40
    assert {(cell.domain, cell.ratio_label) for cell in plan if cell.ratio_label} == {
        ("access_provisioning", "R0"),
        ("access_provisioning", "R100"),
        ("financial_adjustments", "R0"),
        ("financial_adjustments", "R100"),
    }
    assert [(cell.domain, cell.condition.value, cell.ratio_label) for cell in plan[:6]] == [
        ("access_provisioning", "A0_BARE", None),
        ("access_provisioning", "A2_SKILL_ONLY", "R0"),
        ("access_provisioning", "A2_SKILL_ONLY", "R100"),
        ("financial_adjustments", "A0_BARE", None),
        ("financial_adjustments", "A2_SKILL_ONLY", "R0"),
        ("financial_adjustments", "A2_SKILL_ONLY", "R100"),
    ]


def test_live_pilot_protocol_declares_fixed_runtime_capacity_and_operator_check() -> None:
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "fixed 900-second timeout" in protocol
    assert "fixed 16,384-token compiler output cap" in protocol
    assert "74,526 + 16,384 = 90,910" in protocol
    assert "No further\n  compiler-cap increase is permitted." in protocol
    assert "131,072-token" in protocol
    assert "OLLAMA_ROLE_CONFORMANCE_RECEIPT1" in protocol
    assert "ssb-gpt-oss-20b-harmony-roles:v1" in protocol
    assert "ollama-gpt-oss-20b-harmony-roles-v4" in protocol
    assert "ollama-gpt-oss-20b-harmony-roles-v5" in protocol
    assert "Frozen gpt-oss sampling profile (V6)" in protocol
    assert "65,735 input tokens" in protocol
    assert "16,384 output tokens, equal to the cap" in protocol
    assert "zero episode results and zero research comparisons" in protocol
    assert "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES2" in protocol
    assert "temperature=1.0" in protocol
    assert "reasoning profile or compiler\nprojection, never the cap" in protocol
    assert "no\nreproducibility claim beyond hash custody" in protocol
    assert "ollama-gpt-oss-20b-harmony-roles-v6" in protocol
    assert "Frozen Ollama runtime profile (V7)" in protocol
    assert "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES3" in protocol
    assert 'ollama_server_version="0.33.1"' in protocol
    assert "only V7 profile change" in protocol
    assert "ollama-gpt-oss-20b-harmony-roles-v7" in protocol
    assert "Frozen executor final-channel profile (V8)" in protocol
    assert "SSB-PILOT-TURN-WIRE2" in protocol
    assert "four valid compiled skills, zero" in protocol
    assert "ollama-gpt-oss-20b-harmony-roles-v8" in protocol
    assert "PILOT_COMPILER_CONTRACT_FAILURE_RECEIPT1" in protocol
    assert "Frozen split-reasoning profile (V9)" in protocol
    assert "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES4" in protocol
    assert "707 input tokens and 66 output" in protocol
    assert "output_cap_exhausted=false" in protocol
    assert "ollama-gpt-oss-20b-harmony-roles-v9" in protocol
    assert "702 input tokens, 33 output" in protocol
    assert "`thinking = 0` and `truncated = 0`" in protocol
    assert "V9 therefore has four preserved" in protocol
    assert "Frozen executor final-channel template (V11)" in protocol
    assert "Frozen raw native generate transport (V12)" in protocol
    assert "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES7" in protocol
    assert "V12 completed all six preflight cells" in protocol
    assert "## Full exploratory prefix (V13)" in protocol
    assert "complete fixed 60-cell prefix" in protocol
    assert "ollama-gpt-oss-20b-harmony-roles-v13" in protocol
    assert "V13 terminated after six completed cells" in protocol
    assert "episode_f13ea326c103653787fd518d7b08ef59" in protocol
    assert "must reconstruct the seventh cell" in protocol
    assert "raw exploratory counts only; not confirmatory evidence" in protocol
    assert "ssb-gpt-oss-20b-harmony-final:v2" in protocol
    assert "ollama-gpt-oss-20b-harmony-roles-v11/role-conformance-receipt.json" in protocol
    assert "thinking_length=82" in protocol
    assert "content_length=0" in protocol
    assert "zero completed episode results, and zero research" in protocol
    assert "semantic compiler-contract failure" in protocol
    assert "`2026-08-26`" in protocol
    assert "--role-profile-receipt" in protocol
    assert "operator must verify the provider" in protocol
    assert "configuration before launch" in protocol
    assert "## Frozen split-sampling profile (V14)" in protocol
    assert "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES8" in protocol
    assert "executor_sampling_temperature" in protocol
    assert "706 input tokens and 16 output tokens" in protocol
    assert "sampling-dependent executor behavior" in protocol
    assert "ollama-gpt-oss-20b-harmony-roles-v14" in protocol
    assert "V14 completed the full prefix on its first fresh run" in protocol
    assert "completed=60" in protocol
    assert "sha256:c91936790f6144499ca4fcca6e00360c6494653e63d32670c003497f50565190" in protocol
    assert "No further compiler-cap increase is permitted." in protocol
    assert "## Frozen Ollama 0.33.2 runtime profile (V15)" in protocol
    assert "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES9" in protocol
    assert "only runtime change is the Ollama\nserver version" in protocol
    with pytest.raises(live_pilot.LivePilotError, match="explicit loopback"):
        live_pilot.LivePilotDescriptor(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            api_key_environment="SSB_TEST_KEY",
            provider="scripted-pilot",
            model="pilot-model",
            model_version="test-v1",
            request_timeout_seconds=60,  # type: ignore[arg-type]
        )
    with pytest.raises(live_pilot.LivePilotError, match="explicit loopback"):
        live_pilot.LivePilotDescriptor(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            api_key_environment="SSB_TEST_KEY",
            provider="scripted-pilot",
            model="pilot-model",
            model_version="test-v1",
            compiler_max_tokens=4096,  # type: ignore[arg-type]
        )
    with pytest.raises(live_pilot.LivePilotError, match="explicit loopback"):
        live_pilot.LivePilotDescriptor(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            api_key_environment="SSB_TEST_KEY",
            provider="scripted-pilot",
            model="pilot-model",
            model_version="test-v1",
            compiler_wire_schema_hash="sha256:" + "0" * 64,
        )
    with pytest.raises(live_pilot.LivePilotError, match="explicit loopback"):
        live_pilot.LivePilotDescriptor(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            api_key_environment="SSB_TEST_KEY",
            provider="scripted-pilot",
            model="pilot-model",
            model_version="test-v1",
            declared_server_context_length=32_768,  # type: ignore[arg-type]
        )


def test_compiler_uses_an_injected_scripted_live_client_and_persists_four_skills(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor()
    client = _compiler_client(descriptor)

    skills = asyncio.run(
        live_pilot._compiled_skills(tmp_path, descriptor, PROMPT.read_bytes(), client)
    )

    assert len(skills) == 4
    assert len(client.recorded_request_bodies) == 4
    assert all(b'"reasoning_effort":"none"' in body for body in client.recorded_request_bodies)
    assert all(b'"max_tokens":8192' in body for body in client.recorded_request_bodies)
    assert all(b'"maxItems":12' in body for body in client.recorded_request_bodies)
    assert all(b'"maxLength":192' in body for body in client.recorded_request_bodies)
    assert all(b'"objective_evidence_indices"' in body for body in client.recorded_request_bodies)
    assert all("confirmatory" not in path.name for path in tmp_path.rglob("*"))
    assert all(
        artifact.compiler_manifest.prompt_profile == live_pilot.PILOT_COMPILER_PROMPT_PROFILE
        for artifact in skills.values()
    )
    resumed_client = ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=(),
        max_attempts=1,
        reasoning_effort="none",
    )
    resumed = asyncio.run(
        live_pilot._compiled_skills(tmp_path, descriptor, PROMPT.read_bytes(), resumed_client)
    )
    assert resumed.keys() == skills.keys()
    assert resumed_client.recorded_request_bodies == ()
    with pytest.raises(live_pilot.LivePilotError, match="injected live client"):
        asyncio.run(
            live_pilot._compiled_skills(
                tmp_path,
                descriptor,
                PROMPT.read_bytes(),
                ScriptedModelClient(
                    capabilities=descriptor.capabilities, script=(), max_attempts=1
                ),
            )
        )


def test_gpt_oss_sampling_binds_compiler_executor_resume_and_audit(
    tmp_path: Path,
) -> None:
    descriptor = _gpt_descriptor()
    receipt = _gpt_receipt()
    compiler_client = _compiler_client(descriptor)
    executor_client = _executor_client(descriptor)

    summary = asyncio.run(
        live_pilot.run_live_pilot(
            output_dir=tmp_path,
            protocol_path=PROTOCOL,
            prompt_path=PROMPT,
            descriptor=descriptor,
            compiler_client=compiler_client,
            executor_client=executor_client,
            limit=1,
            role_profile_receipt=receipt,
        )
    )

    assert summary["completed"] == 1
    assert len(compiler_client.recorded_request_bodies) == 4
    assert len(executor_client._transport.recorded_request_bodies) == 1
    assert all(b'"temperature":1.0' in body for body in compiler_client.recorded_request_bodies)
    assert all(
        b'"temperature":1.0' in body for body in executor_client._transport.recorded_request_bodies
    )
    assert all(
        b'"reasoning_effort":"low"' in body for body in compiler_client.recorded_request_bodies
    )
    assert all(b'"response_format"' in body for body in compiler_client.recorded_request_bodies)
    assert all(
        b'"raw":true' in body
        and b'"think"' not in body
        and b'"messages"' not in body
        and b'"reasoning_effort"' not in body
        for body in executor_client._transport.recorded_request_bodies
    )
    assert all(
        b'"format"' in body
        and b'"model":"ssb-gpt-oss-20b-harmony-final:v2"' in body
        and b'"num_ctx":131072' in body
        and b'"top_p":1.0' in body
        for body in executor_client._transport.recorded_request_bodies
    )
    runtime_path = tmp_path / "runtime.json"
    runtime = json.loads(runtime_path.read_bytes())
    assert runtime["sampling_temperature"] == 1.0
    assert runtime["executor_sampling_temperature"] == 1.0
    assert runtime["reasoning_effort"] == "low"
    assert runtime["executor_reasoning_effort"] == "none"
    assert runtime["executor_endpoint_hash"].startswith("sha256:")
    assert runtime["executor_transport"] == OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT
    assert runtime["executor_think"] is False
    assert runtime["executor_model"] == "ssb-gpt-oss-20b-harmony-final:v2"
    assert runtime["executor_model_version"] == OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST
    result = json.loads(next((tmp_path / "results").glob("*.json")).read_bytes())
    assert result["executor_sampling_temperature"] == 1.0
    assert result["executor_reasoning_effort"] == "none"
    assert live_pilot.audit_live_pilot(tmp_path)["classification"] == live_pilot.PILOT_LABEL

    runtime["executor_think"] = True
    runtime_body = {key: value for key, value in runtime.items() if key != "runtime_hash"}
    runtime["runtime_hash"] = sha256_ref(runtime_body)
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(live_pilot.LivePilotError, match="runtime descriptor is invalid"):
        live_pilot.audit_live_pilot(tmp_path)
    runtime["executor_think"] = False
    runtime_body = {key: value for key, value in runtime.items() if key != "runtime_hash"}
    runtime["runtime_hash"] = sha256_ref(runtime_body)
    runtime_path.write_bytes(canonical_json_bytes(runtime))

    runtime["executor_sampling_temperature"] = 0.0
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(live_pilot.LivePilotError, match="runtime descriptor is invalid"):
        live_pilot.audit_live_pilot(tmp_path)
    runtime["executor_sampling_temperature"] = 1.0
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))

    result_path = next((tmp_path / "results").glob("*.json"))
    result_envelope = json.loads(result_path.read_bytes())
    result_envelope["executor_reasoning_effort"] = "low"
    result_path.write_bytes(canonical_json_bytes(result_envelope))
    with pytest.raises(live_pilot.LivePilotError, match="not bound to this plan"):
        live_pilot.audit_live_pilot(tmp_path)
    result_envelope["executor_reasoning_effort"] = "none"
    result_path.write_bytes(canonical_json_bytes(result_envelope))

    stored_path = tmp_path / "compiled-skills" / "access_provisioning-r0.json"
    stored = json.loads(stored_path.read_bytes())
    stored["compiler_manifest"]["temperature"] = 0.0
    stored_path.write_bytes(canonical_json_bytes(stored))
    with pytest.raises(live_pilot.LivePilotError, match="temperature does not match"):
        asyncio.run(
            live_pilot._compiled_skills(
                tmp_path,
                descriptor,
                PROMPT.read_bytes(),
                ScriptedModelClient(
                    capabilities=descriptor.capabilities,
                    script=(),
                    max_attempts=1,
                    reasoning_effort="low",
                ),
            )
        )

    runtime["ollama_server_version"] = "0.33.0"
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(live_pilot.LivePilotError, match="runtime descriptor"):
        live_pilot.audit_live_pilot(tmp_path)

    runtime["ollama_server_version"] = OLLAMA_GPT_OSS_SERVER_VERSION
    runtime["sampling_temperature"] = 0.0
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(live_pilot.LivePilotError, match="runtime descriptor"):
        live_pilot.audit_live_pilot(tmp_path)


def test_live_pilot_stops_on_model_failure_without_persisting_a_completed_cell(
    tmp_path: Path,
) -> None:
    before = _global_episode_artifact_snapshot()
    descriptor = _descriptor()
    compiler_client = _compiler_client(descriptor)
    client = ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=(*compiler_client._script, "terminal"),
        max_attempts=1,
        reasoning_effort="none",
    )

    with pytest.raises(live_pilot.LivePilotError, match="infrastructure failure"):
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=descriptor,
                client=client,
                limit=1,
            )
        )

    assert not (tmp_path / "results").exists()
    assert _global_episode_artifact_snapshot() == before


def test_live_pilot_stops_on_malformed_model_output_without_persisting_a_completed_cell(
    tmp_path: Path,
) -> None:
    before = _global_episode_artifact_snapshot()
    descriptor = _descriptor()
    compiler_client = _compiler_client(descriptor)
    client = ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=(*compiler_client._script, _response(descriptor.capabilities, {})),
        max_attempts=1,
        reasoning_effort="none",
    )

    with pytest.raises(live_pilot.LivePilotError, match="MODEL_OUTPUT_INVALID") as caught:
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=descriptor,
                client=client,
                limit=1,
            )
        )

    assert "finish_reason=stop" in str(caught.value)
    assert "input_tokens=1 output_tokens=1" in str(caught.value)
    assert "output_cap_exhausted=false" in str(caught.value)
    assert not (tmp_path / "results").exists()
    assert _global_episode_artifact_snapshot() == before


def test_live_pilot_halts_on_schema_valid_but_layout_invalid_turn_without_completed_cell(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor()
    compiler_client = _compiler_client(descriptor)
    client = ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=(
            *compiler_client._script,
            _response(
                descriptor.capabilities,
                {
                    "turn": {
                        "kind": "tool_call",
                        "tool_index": 15,
                        "argument_bindings": [],
                    }
                },
            ),
        ),
        max_attempts=1,
        reasoning_effort="none",
    )

    with pytest.raises(live_pilot.LivePilotError, match="MODEL_OUTPUT_INVALID"):
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=descriptor,
                client=client,
                limit=1,
            )
        )

    assert not (tmp_path / "results").exists()
    artifacts = list((tmp_path / "artifacts" / "episode_result").rglob("*.json"))
    assert len(artifacts) == 1
    engine_artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert engine_artifact["payload"]["error_code"] == "MODEL_OUTPUT_INVALID"
    assert engine_artifact["payload"]["overhead"]["model_calls"] == 1


def test_live_pilot_rejects_a_retried_compiler_response_before_episode_dispatch(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor()
    compiler_client = _compiler_client(descriptor)
    client = ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=("transient", compiler_client._script[0]),
        max_attempts=2,
        reasoning_effort="none",
    )

    with pytest.raises(live_pilot.LivePilotError, match="compiled skill does not bind"):
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=descriptor,
                client=client,
                limit=1,
            )
        )

    assert not (tmp_path / "results").exists()


def test_audit_rejects_missing_engine_artifact_after_validating_a_real_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _descriptor()
    client = _pilot_client_with_one_finish(descriptor)
    asyncio.run(
        live_pilot.run_live_pilot(
            output_dir=tmp_path,
            protocol_path=PROTOCOL,
            prompt_path=PROMPT,
            descriptor=descriptor,
            client=client,
            limit=1,
        )
    )
    assert live_pilot.audit_live_pilot(tmp_path)["classification"] == live_pilot.PILOT_LABEL
    assert (
        json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))["reasoning_effort"]
        == "none"
    )
    runtime_path = tmp_path / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["request_timeout_seconds"] == 900
    assert runtime["compiler_max_tokens"] == 8192
    assert runtime["compiler_wire_schema_hash"] == live_pilot.PILOT_WIRE_SCHEMA_HASH
    assert runtime["compiler_prompt_profile"] == live_pilot.PILOT_COMPILER_PROMPT_PROFILE
    assert runtime["executor_turn_prompt_profile"] == live_pilot.PILOT_TURN_PROMPT_PROFILE
    assert runtime["executor_turn_prompt_hash"] == live_pilot.PILOT_TURN_PROMPT_HASH
    assert runtime["executor_turn_schema_hash"] == live_pilot.PILOT_TURN_SCHEMA_HASH
    assert runtime["declared_server_context_length"] == 262144
    plan = json.loads((tmp_path / "pilot-plan.json").read_text(encoding="utf-8"))
    assert plan["compiler_prompt_profile"] == live_pilot.PILOT_COMPILER_PROMPT_PROFILE
    assert plan["executor_turn_prompt_profile"] == live_pilot.PILOT_TURN_PROMPT_PROFILE
    assert plan["executor_turn_prompt_hash"] == live_pilot.PILOT_TURN_PROMPT_HASH
    assert plan["executor_turn_schema_hash"] == live_pilot.PILOT_TURN_SCHEMA_HASH
    assert plan["executor_turn_tool_layout_hashes"] == runtime["executor_turn_tool_layout_hashes"]
    assert b'"tool_index"' in client.recorded_request_bodies[-1]
    assert b"SSB-PILOT-TURN-WIRE2" in client.recorded_request_bodies[-1]
    assert b"never end the response from the analysis channel" in client.recorded_request_bodies[-1]
    runtime["compiler_prompt_profile"] = "SSB-SKILL-COMPILER1"
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(live_pilot.LivePilotError, match="runtime descriptor"):
        live_pilot.audit_live_pilot(tmp_path)
    runtime["compiler_prompt_profile"] = live_pilot.PILOT_COMPILER_PROMPT_PROFILE
    runtime["request_timeout_seconds"] = 60
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(live_pilot.LivePilotError, match="runtime descriptor"):
        live_pilot.audit_live_pilot(tmp_path)
    runtime["request_timeout_seconds"] = 900
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    runtime["compiler_max_tokens"] = 4096
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(live_pilot.LivePilotError, match="runtime descriptor"):
        live_pilot.audit_live_pilot(tmp_path)
    runtime["compiler_max_tokens"] = 8192
    runtime["compiler_wire_schema_hash"] = "sha256:" + "0" * 64
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(live_pilot.LivePilotError, match="runtime descriptor"):
        live_pilot.audit_live_pilot(tmp_path)
    runtime["compiler_wire_schema_hash"] = live_pilot.PILOT_WIRE_SCHEMA_HASH
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    result_path = next((tmp_path / "results").glob("*.json"))
    result_envelope = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_envelope["executor_turn_prompt_profile"] == live_pilot.PILOT_TURN_PROMPT_PROFILE
    assert result_envelope["executor_turn_schema_hash"] == live_pilot.PILOT_TURN_SCHEMA_HASH
    assert result_envelope["executor_turn_tool_layout_hash"] in set(
        runtime["executor_turn_tool_layout_hashes"].values()
    )
    runtime["executor_turn_schema_hash"] = "sha256:" + "0" * 64
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(live_pilot.LivePilotError, match="runtime descriptor"):
        live_pilot.audit_live_pilot(tmp_path)
    runtime["executor_turn_schema_hash"] = live_pilot.PILOT_TURN_SCHEMA_HASH
    runtime["runtime_hash"] = sha256_ref(
        {key: value for key, value in runtime.items() if key != "runtime_hash"}
    )
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    monkeypatch.setattr(
        live_pilot,
        "_pilot_turn_layout_hash",
        lambda domain, condition: "sha256:" + "0" * 64,
    )
    with pytest.raises(live_pilot.LivePilotError, match="pilot plan"):
        live_pilot.audit_live_pilot(tmp_path)
    with pytest.raises(live_pilot.LivePilotError, match="pilot-plan.json differs"):
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=descriptor,
                client=_executor_client(descriptor),
                limit=1,
            )
        )
    monkeypatch.undo()
    artifact_ref = json.loads(result_path.read_text(encoding="utf-8"))["result"]["artifact_ref"]
    assert isinstance(artifact_ref, str)
    (tmp_path / artifact_ref).unlink()

    with pytest.raises(live_pilot.LivePilotError, match="episode artifact"):
        live_pilot.audit_live_pilot(tmp_path)


def test_resume_rejects_a_tampered_result_manifest(tmp_path: Path) -> None:
    descriptor = _descriptor()
    asyncio.run(
        live_pilot.run_live_pilot(
            output_dir=tmp_path,
            protocol_path=PROTOCOL,
            prompt_path=PROMPT,
            descriptor=descriptor,
            client=_pilot_client_with_one_finish(descriptor),
            limit=1,
        )
    )
    result_path = next((tmp_path / "results").glob("*.json"))
    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["execution_manifest_hash"] = "sha256:" + "0" * 64
    result_path.write_bytes(canonical_json_bytes(tampered))

    with pytest.raises(live_pilot.LivePilotError, match="stored pilot result is invalid"):
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=descriptor,
                client=ScriptedModelClient(
                    capabilities=descriptor.capabilities,
                    script=(),
                    max_attempts=1,
                    reasoning_effort="none",
                ),
                limit=1,
            )
        )


def test_live_pilot_resume_descriptor_mismatch_and_confirmatory_isolation(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor()
    compiler_client = _compiler_client(descriptor)
    client = ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=(
            *compiler_client._script,
            _response(descriptor.capabilities, {"turn": {"kind": "finish", "summary": "done"}}),
        ),
        max_attempts=1,
        reasoning_effort="none",
    )
    confirmatory_inputs = {
        path: path.read_bytes() for path in (ROOT / "protocol").glob("freeze_*") if path.is_file()
    }
    first = asyncio.run(
        live_pilot.run_live_pilot(
            output_dir=tmp_path,
            protocol_path=PROTOCOL,
            prompt_path=PROMPT,
            descriptor=descriptor,
            client=client,
            limit=1,
        )
    )
    increased = asyncio.run(
        live_pilot.run_live_pilot(
            output_dir=tmp_path,
            protocol_path=PROTOCOL,
            prompt_path=PROMPT,
            descriptor=descriptor,
            client=ScriptedModelClient(
                capabilities=descriptor.capabilities,
                script=(
                    _response(
                        descriptor.capabilities,
                        {"turn": {"kind": "finish", "summary": "done"}},
                    ),
                ),
                max_attempts=1,
                reasoning_effort="none",
            ),
            limit=2,
        )
    )
    resumed = asyncio.run(
        live_pilot.run_live_pilot(
            output_dir=tmp_path,
            protocol_path=PROTOCOL,
            prompt_path=PROMPT,
            descriptor=descriptor,
            client=ScriptedModelClient(
                capabilities=descriptor.capabilities,
                script=(),
                max_attempts=1,
                reasoning_effort="none",
            ),
            limit=2,
        )
    )

    assert first["completed"] == 1
    assert increased["completed"] == 1 and increased["skipped"] == 1
    assert resumed["completed"] == 0 and resumed["skipped"] == 2
    result_path = next((tmp_path / "results").glob("*.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))["result"]
    artifact_ref = result["artifact_ref"]
    assert isinstance(artifact_ref, str)
    assert (tmp_path / artifact_ref).is_file()
    assert not (ROOT / artifact_ref).exists()
    assert all(path.read_bytes() == content for path, content in confirmatory_inputs.items())
    with pytest.raises(live_pilot.LivePilotError, match="runtime.json differs"):
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=_descriptor("different-model"),
                client=_executor_client(_descriptor("different-model")),
                limit=1,
            )
        )


def _truncated_response(capabilities: ProviderCapabilities, cap: int) -> TransportResponse:
    return TransportResponse(
        status_code=200,
        body=canonical_json_bytes(
            {
                "model": capabilities.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 65735,
                    "completion_tokens": cap,
                    "total_tokens": 65735 + cap,
                },
            }
        ),
    )


def test_failed_compiler_call_persists_a_secret_free_failure_receipt_and_holds(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor()
    compiler_client = _compiler_client(descriptor)
    cap = descriptor.compiler_max_tokens
    client = ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=(*compiler_client._script[:2], _truncated_response(descriptor.capabilities, cap)),
        max_attempts=1,
        reasoning_effort="none",
    )

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=descriptor,
                client=client,
                limit=6,
            )
        )

    error = caught.value
    assert error.code == "MODEL_OUTPUT_INVALID"
    assert error.finish_reason == "length"
    assert error.reported_usage == TokenUsage(
        input_tokens=65735, output_tokens=cap, total_tokens=65735 + cap
    )
    compiled = tmp_path / "compiled-skills"
    assert (compiled / "access_provisioning-r0.json").exists()
    assert (compiled / "access_provisioning-r100.json").exists()
    assert not (compiled / "financial_adjustments-r0.json").exists()
    assert not (tmp_path / "results").exists()
    receipts = sorted(compiled.glob("financial_adjustments-r0.failure-*.json"))
    assert len(receipts) == 1
    raw = receipts[0].read_bytes()
    receipt = json.loads(raw)
    assert canonical_json_bytes(receipt) == raw
    runtime = json.loads((tmp_path / "runtime.json").read_bytes())
    plan = json.loads((tmp_path / "pilot-plan.json").read_bytes())
    assert receipt == {
        "record_kind": "PILOT_COMPILER_FAILURE_RECEIPT1",
        "profile": live_pilot.PILOT_PROFILE,
        "classification": live_pilot.PILOT_LABEL,
        "plan_hash": plan["plan_hash"],
        "runtime_hash": runtime["runtime_hash"],
        "domain": "financial_adjustments",
        "ratio": "R0",
        "contamination_ratio": "0",
        "compiler_input_hash": live_pilot._compiler_input_hash(
            "financial_adjustments", Decimal("0")
        ),
        "compiler_prompt_profile": live_pilot.PILOT_COMPILER_PROMPT_PROFILE,
        "compiler_wire_schema_hash": live_pilot.PILOT_WIRE_SCHEMA_HASH,
        "compiler_max_tokens": cap,
        "provider": descriptor.provider,
        "model": descriptor.model,
        "model_version": descriptor.model_version,
        "error_code": "MODEL_OUTPUT_INVALID",
        "attempts": 1,
        "before_meaningful_behavior": True,
        "raw_request_hash": error.raw_request_hash,
        "raw_response_hash": error.raw_response_hash,
        "finish_reason": "length",
        "reported_usage": {
            "input_tokens": 65735,
            "output_tokens": cap,
            "total_tokens": 65735 + cap,
        },
        "output_cap_exhausted": True,
        "raw_provider_trace_retained": False,
    }
    assert receipts[0].name.endswith(f"{sha256_ref(receipt)[7:19]}.json")
    assert b"content" not in raw and b"choices" not in raw

    # A resumed run reuses the two valid artifacts and records a second, distinct failure.
    resumed = ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=(_response(descriptor.capabilities, {}),),
        max_attempts=1,
        reasoning_effort="none",
    )
    with pytest.raises(ModelAdapterError) as second:
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=descriptor,
                client=resumed,
                limit=6,
            )
        )
    assert second.value.finish_reason == "stop" and second.value.before_meaningful_behavior is False
    assert len(resumed.recorded_request_bodies) == 1
    assert len(sorted(compiled.glob("financial_adjustments-r0.failure-*.json"))) == 2
    assert not (tmp_path / "results").exists()


def test_semantic_compiler_failure_persists_safe_contract_receipt_and_holds(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor()
    wire = {
        "objective": "secret model output must not be retained",
        "objective_evidence_indices": [4095],
        "ordered_steps": [
            {
                "step_id": "inspect_visible_state",
                "action_intent": "Inspect visible state.",
                "tool_name": "inspect_state",
                "argument_bindings": [],
                "optional": False,
                "evidence_indices": [0],
            }
        ],
    }
    client = ScriptedModelClient(
        capabilities=descriptor.capabilities,
        script=(_response(descriptor.capabilities, wire),),
        max_attempts=1,
        reasoning_effort="none",
    )

    with pytest.raises(CompilerContractError) as caught:
        asyncio.run(
            live_pilot.run_live_pilot(
                output_dir=tmp_path,
                protocol_path=PROTOCOL,
                prompt_path=PROMPT,
                descriptor=descriptor,
                client=client,
                limit=6,
            )
        )

    error = caught.value
    assert error.stage == "pilot_wire_conversion"
    assert error.validation_code == "PILOT_WIRE_EVENT_INDEX_OUT_OF_RANGE"
    receipts = sorted(
        (tmp_path / "compiled-skills").glob("access_provisioning-r0.contract-failure-*.json")
    )
    assert len(receipts) == 1
    raw = receipts[0].read_bytes()
    receipt = json.loads(raw)
    assert canonical_json_bytes(receipt) == raw
    assert receipt["record_kind"] == "PILOT_COMPILER_CONTRACT_FAILURE_RECEIPT1"
    assert receipt["stage"] == "pilot_wire_conversion"
    assert receipt["validation_code"] == "PILOT_WIRE_EVENT_INDEX_OUT_OF_RANGE"
    assert receipt["validation_paths"] == []
    assert receipt["finish_reason"] == "stop"
    assert receipt["finish_reason_source"] == "adapter_success_contract"
    assert receipt["reported_usage"] == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }
    assert receipt["output_cap_exhausted"] is False
    assert receipt["raw_provider_trace_retained"] is False
    assert not (tmp_path / "results").exists()
    assert b"secret model output" not in raw
    assert b'"content"' not in raw and b'"choices"' not in raw
