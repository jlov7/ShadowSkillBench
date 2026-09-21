from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.corpus.confirmatory import (
    ConfirmatoryAuthorization,
    _current_generator_hash,
    authorize_confirmatory_generation,
)
from shadowskillbench.experiments import confirmatory_compilation as compilation
from shadowskillbench.experiments.audit import FrozenExclusionRule
from shadowskillbench.experiments.confirmatory_compilation import (
    ConfirmatoryCompilationHold,
    compile_confirmatory_skills,
)
from shadowskillbench.experiments.confirmatory_preparation import write_design_commitment
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    OLLAMA_GPT_OSS_MODEL,
    OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH,
    OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT,
    OLLAMA_GPT_OSS_PROVIDER,
    canonical_role_profile_receipt,
    ollama_gpt_oss_native_compiler_projection,
    role_conformance_probe_projection,
)
from shadowskillbench.models import (
    ModelAdapterError,
    ProviderCapabilities,
    ScriptedModelClient,
    TokenUsage,
)
from shadowskillbench.protocol.freeze import REQUIRED_INPUTS
from shadowskillbench.skills import gate2
from shadowskillbench.skills.compiler import (
    CompiledSkillArtifact,
    CompilerManifest,
    compiler_manifest_hash,
)
from shadowskillbench.skills.models import SkillIR, hash_skill_ir
from shadowskillbench.skills.projection import compiler_view, hash_compiler_input
from shadowskillbench.skills.render import hash_rendered_skill, render_skill
from shadowskillbench.traces.bundles import generate_bundle
from tests.custody import write_local_custody_receipt

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIGEST = "sha256:a939fca6d222577c1a6534f6be3c9f01f69ef646081b3d44e924257d3eb04289"


def test_compiler_frozen_input_roles_match_the_real_freeze_contract() -> None:
    freeze_roles = {item.path: item.role for item in REQUIRED_INPUTS}

    assert {
        path: freeze_roles.get(path) for path in compilation._FROZEN_INPUTS
    } == compilation._FROZEN_INPUTS
    assert compilation._FROZEN_INPUTS["src/shadowskillbench/models/runtime.py"] == "model_runtime"
    assert (
        compilation._FROZEN_INPUTS["src/shadowskillbench/models/ollama_native.py"]
        == "native_model_adapter"
    )


def test_compiler6_config_binds_the_native_request_profile() -> None:
    config = compilation.load_confirmatory_compiler_config(ROOT)

    assert config.config_hash.startswith("sha256:")
    assert compilation._EXPECTED_CONFIG["transport"] == OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT
    assert (
        compilation._EXPECTED_CONFIG["request_profile"]
        == OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE
    )
    assert (
        compilation._EXPECTED_CONFIG["request_profile_hash"]
        == OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH
    )
    assert compilation._EXPECTED_CONFIG["top_p"] == "omitted"
    projection = ollama_gpt_oss_native_compiler_projection()
    assert projection["api_path"] == "/api/generate"
    assert projection["prompt"] == "rendered_harmony_final_v2_reasoning_low"
    assert projection["options"]["top_p"] == {"wire": "omitted"}
    assert compilation.sha256_ref(projection) == OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH


def _rule() -> FrozenExclusionRule:
    return FrozenExclusionRule(
        allowed_error_codes=("MODEL_PROVIDER_TERMINAL", "MODEL_PROVIDER_TRANSIENT"),
        max_retries_per_episode=1,
        max_total_exclusions=10,
        max_exclusions_per_primary_cell=1,
    )


def _receipt() -> dict[str, object]:
    probe = role_conformance_probe_projection()
    return json.loads(
        canonical_role_profile_receipt(
            MODEL_DIGEST,
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


def _repository_root(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative_path in compilation._FROZEN_INPUTS:
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative_path == "protocol/commitments/role_conformance_receipt.json":
            target.write_bytes(canonical_json_bytes(_receipt()))
        else:
            target.write_bytes((ROOT / relative_path).read_bytes())
    generator = root / "src" / "shadowskillbench" / "corpus" / "confirmatory.py"
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator.write_bytes(
        (ROOT / "src" / "shadowskillbench" / "corpus" / "confirmatory.py").read_bytes()
    )
    return root


def _authorization(root: Path) -> ConfirmatoryAuthorization:
    entries = {
        "src/shadowskillbench/corpus/confirmatory.py": {
            "role": "confirmatory_generator",
            "sha256": _current_generator_hash(),
        }
    }
    for relative_path, role in compilation._FROZEN_INPUTS.items():
        entries[relative_path] = {
            "role": role,
            "sha256": "sha256:" + hashlib.sha256((root / relative_path).read_bytes()).hexdigest(),
        }
    manifest = (
        canonical_json_bytes(
            {
                "anchor_status": "PENDING_HUMAN_ANCHOR",
                "inputs": [
                    {"path": path, **entry}
                    for path, entry in sorted(entries.items(), key=lambda item: item[0])
                ],
                "preregistration_core": {
                    "path": "protocol/preregistration.md",
                    "sha256": "sha256:" + "a" * 64,
                },
                "schema_version": "1.0",
            }
        )
        + b"\n"
    )
    digest = hashlib.sha256(manifest).hexdigest()
    protocol = root / "protocol"
    protocol.joinpath("freeze_manifest.json").write_bytes(manifest)
    protocol.joinpath("freeze_manifest.sha256").write_bytes(
        f"{digest}  freeze_manifest.json\n".encode()
    )
    anchor = write_local_custody_receipt(root, manifest)
    return authorize_confirmatory_generation(
        manifest,
        f"{digest}  freeze_manifest.json\n".encode(),
        anchor,
        repository_root=root,
    )


def _client() -> ScriptedModelClient:
    return ScriptedModelClient(
        capabilities=ProviderCapabilities(
            provider=OLLAMA_GPT_OSS_PROVIDER,
            model=OLLAMA_GPT_OSS_MODEL,
            model_version=MODEL_DIGEST,
            supports_system_role=True,
            supports_developer_role=True,
            supports_seed=True,
            supports_structured_output=True,
        ),
        script=(),
        max_attempts=1,
        reasoning_effort="low",
    )


def _fake_compile(compiler_input):
    async def compile(_bundle, client, config, *, output_schema):
        assert output_schema.__name__ == "ConfirmatorySkillIRWire"
        manifest = CompilerManifest(
            manifest_profile="SSB-COMPILER-MANIFEST1",
            compiler_input_hash=hash_compiler_input(compiler_input),
            prompt_profile=config.prompt_profile,
            system_prompt_raw_hash="sha256:" + hashlib.sha256(config.prompt_bytes).hexdigest(),
            request_profile="SSB-SKILL-COMPILER-REQUEST1",
            instruction_provenance_profile="SSB-INSTRUCTION-PROVENANCE1",
            structured_output_schema_hash=compilation.sha256_ref(output_schema.model_json_schema()),
            declared_capabilities=client.capabilities,
            temperature=config.temperature,
            seed=config.seed,
            max_tokens=config.max_tokens,
        )
        manifest_hash = compiler_manifest_hash(manifest)
        skill = SkillIR.model_validate(
            json.loads(canonical_json_bytes(gate2._witness_skill(compiler_input, manifest_hash)))
        )
        return CompiledSkillArtifact(
            artifact_profile="SSB-COMPILED-SKILL1",
            compiler_manifest=manifest,
            compiler_manifest_hash=manifest_hash,
            request_envelope_hash="sha256:" + "0" * 64,
            skill_ir=skill,
            skill_ir_hash=hash_skill_ir(skill),
            rendered_skill=render_skill(skill),
            rendered_skill_hash=hash_rendered_skill(skill),
            raw_request_hash="sha256:" + "1" * 64,
            raw_response_hash="sha256:" + "2" * 64,
            structured_output_schema_hash=compilation.sha256_ref(output_schema.model_json_schema()),
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=None,
            attempts=1,
        )

    return compile


def _commitment(tmp_path: Path) -> Path:
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    return write_design_commitment(protocol, exclusion_rule=_rule())


def test_compilation_requires_exact_post_anchor_authorization(tmp_path: Path) -> None:
    root = _repository_root(tmp_path)
    with pytest.raises(ConfirmatoryCompilationHold, match="post-anchor authorization"):
        asyncio.run(
            compile_confirmatory_skills(
                repository_root=root,
                output_dir=tmp_path / "compiled",
                commitment_path=_commitment(tmp_path),
                exclusion_rule=_rule(),
                authorization=cast(ConfirmatoryAuthorization, object()),
                role_profile_receipt_path=root
                / "protocol/commitments/role_conformance_receipt.json",
                client=_client(),
            )
        )


def test_compilation_refuses_to_overwrite_output_directory(tmp_path: Path) -> None:
    root = _repository_root(tmp_path)
    output = tmp_path / "compiled"
    output.mkdir()
    with pytest.raises(ConfirmatoryCompilationHold, match="output directory must be fresh"):
        asyncio.run(
            compile_confirmatory_skills(
                repository_root=root,
                output_dir=output,
                commitment_path=_commitment(tmp_path),
                exclusion_rule=_rule(),
                authorization=_authorization(root),
                role_profile_receipt_path=root
                / "protocol/commitments/role_conformance_receipt.json",
                client=_client(),
            )
        )


def test_compilation_writes_exactly_thirty_committed_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository_root(tmp_path)
    commitment_path = _commitment(tmp_path)
    bundle_records = json.loads(commitment_path.read_bytes())["bundles"]
    source_hashes = {entry["bundle_id"]: entry["source_manifest_hash"] for entry in bundle_records}
    bundle_ids = {
        (entry["domain"], entry["contamination_ratio"], entry["seed"]): entry["bundle_id"]
        for entry in bundle_records
    }
    base_input = compiler_view(generate_bundle("access_provisioning", Decimal("0"), 12, 99).bundle)

    def fast_bundle(domain, ratio, count, seed):
        bundle_id = bundle_ids[(domain, str(ratio), seed)]
        return SimpleNamespace(
            bundle=SimpleNamespace(source_manifest=SimpleNamespace(bundle_id=bundle_id))
        )

    monkeypatch.setattr(compilation, "compile_skill", _fake_compile(base_input))
    monkeypatch.setattr(compilation, "generate_bundle", fast_bundle)
    monkeypatch.setattr(compilation, "compiler_view", lambda _bundle: base_input)
    monkeypatch.setattr(
        compilation,
        "hash_source_bundle_manifest",
        lambda manifest: source_hashes[manifest.bundle_id],
    )
    compiled = asyncio.run(
        compile_confirmatory_skills(
            repository_root=root,
            output_dir=tmp_path / "compiled",
            commitment_path=commitment_path,
            exclusion_rule=_rule(),
            authorization=_authorization(root),
            role_profile_receipt_path=root / "protocol/commitments/role_conformance_receipt.json",
            client=_client(),
        )
    )
    assert len(list(compiled.glob("*.json"))) == 30
    manifest = json.loads((compiled.parent / "compiler-run-manifest.json").read_bytes())
    assert len(manifest["artifacts"]) == 30
    assert manifest["compiler_transport"] == OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT
    assert manifest["compiler_request_profile"] == OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE
    assert (
        manifest["compiler_request_profile_hash"]
        == OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH
    )
    assert manifest["compiler_top_p"] == "omitted"


def test_compilation_writes_sanitized_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository_root(tmp_path)

    async def fail(*_args, **_kwargs):
        raise ModelAdapterError(
            code="MODEL_OUTPUT_INVALID",
            attempts=1,
            before_meaningful_behavior=False,
            raw_request_hash="sha256:" + "a" * 64,
            raw_response_hash="sha256:" + "b" * 64,
            finish_reason="length",
        )

    monkeypatch.setattr(compilation, "compile_skill", fail)
    with pytest.raises(ConfirmatoryCompilationHold, match="failure receipt was written"):
        asyncio.run(
            compile_confirmatory_skills(
                repository_root=root,
                output_dir=tmp_path / "compiled",
                commitment_path=_commitment(tmp_path),
                exclusion_rule=_rule(),
                authorization=_authorization(root),
                role_profile_receipt_path=root
                / "protocol/commitments/role_conformance_receipt.json",
                client=_client(),
            )
        )
    receipts = list((tmp_path / "compiled" / "compiled-skills").glob("*.failure-*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_bytes())
    assert payload["finish_reason"] == "length"
    assert payload["raw_provider_trace_retained"] is False
    assert "content" not in payload
    assert payload["compiler_transport"] == OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT
    assert payload["compiler_request_profile"] == OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE
    assert payload["compiler_top_p"] == "omitted"


def test_confirmatory_compilation_cli_delegates_to_the_post_anchor_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository_root(tmp_path)
    receipt_path = root / "protocol/commitments/role_conformance_receipt.json"
    for name in ("manifest.json", "manifest.sha256", "anchor.json", "rule.json"):
        (tmp_path / name).write_text("{}")
    captured: dict[str, object] = {}

    async def fake_producer(**kwargs):
        captured.update(kwargs)
        return cast(Path, kwargs["output_dir"]) / "compiled-skills"

    monkeypatch.setattr(
        "shadowskillbench.experiments.confirmatory_preparation_cli.authorize_confirmatory_generation",
        lambda *_args, **_kwargs: _authorization(root),
    )
    monkeypatch.setattr(
        "shadowskillbench.experiments.confirmatory_preparation_cli._rule", lambda _path: _rule()
    )

    def native_client(*args: object, **kwargs: object):
        captured["native_client_args"] = args
        captured["native_client_kwargs"] = kwargs
        return _client()

    monkeypatch.setattr(
        "shadowskillbench.experiments.confirmatory_preparation_cli.ollama_native_client_from_run_descriptor",
        native_client,
    )
    monkeypatch.setattr(
        "shadowskillbench.experiments.confirmatory_preparation_cli.compile_confirmatory_skills",
        fake_producer,
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "episodes",
            "compile-confirmatory-skills",
            "--endpoint",
            "http://127.0.0.1:11435/api/generate",
            "--api-key-environment",
            "SSB_TEST_KEY",
            "--role-profile-receipt",
            str(receipt_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--exclusion-rule",
            str(tmp_path / "rule.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--manifest-sha256",
            str(tmp_path / "manifest.sha256"),
            "--anchor-receipt",
            str(tmp_path / "anchor.json"),
            "--repository-root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PASS_CONFIRMATORY_COMPILATION" in result.output
    assert captured["client"] is not None
    descriptor = cast(tuple[object, ...], captured["native_client_args"])[0]
    assert getattr(descriptor, "endpoint") == "http://127.0.0.1:11435/api/generate"
    assert captured["native_client_kwargs"] == {
        "trust_env": False,
        "timeout_seconds": 900.0,
        "reasoning_effort": "low",
        "include_top_p": False,
    }
