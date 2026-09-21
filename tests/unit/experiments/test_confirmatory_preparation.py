from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.confirmatory import (
    SplitInventory,
    _current_generator_hash,
    authorize_confirmatory_generation,
)
from shadowskillbench.episodes.pilot_turn_wire import (
    CONFIRMATORY_TURN_PROMPT_PROFILE,
    confirmatory_active_turn_schema_hash,
    confirmatory_finished_turn_schema_hash,
    confirmatory_initial_turn_schema_hash,
)
from shadowskillbench.experiments.audit import FrozenExclusionRule
from shadowskillbench.experiments.confirmatory_package import (
    load_confirmatory_execution_package_inputs,
)
from shadowskillbench.experiments.confirmatory_preparation import (
    CORPUS_SEED,
    build_design_commitment,
    stage_confirmatory_execution_source,
    verify_design_commitment,
    write_design_commitment,
)
from shadowskillbench.models import ProviderCapabilities, ScriptedModelClient, TransportResponse
from shadowskillbench.models.runtime import ModelRunDescriptor
from shadowskillbench.skills import gate2
from shadowskillbench.skills.compiler import (
    CompiledSkillArtifact,
    CompilerConfig,
    CompilerManifest,
    compile_skill,
    compiler_manifest_hash,
)
from shadowskillbench.skills.models import SkillIR
from shadowskillbench.skills.projection import compiler_view, hash_compiler_input
from shadowskillbench.traces.bundles import generate_bundle
from tests.custody import write_local_custody_receipt


def _rule() -> FrozenExclusionRule:
    return FrozenExclusionRule(
        allowed_error_codes=("MODEL_PROVIDER_TERMINAL", "MODEL_PROVIDER_TRANSIENT"),
        max_retries_per_episode=1,
        max_total_exclusions=10,
        max_exclusions_per_primary_cell=1,
    )


def _authorization(repository_root: Path):
    generator_bytes = (
        Path(__file__)
        .parents[3]
        .joinpath("src", "shadowskillbench", "corpus", "confirmatory.py")
        .read_bytes()
    )
    generator_path = repository_root / "src" / "shadowskillbench" / "corpus"
    generator_path.mkdir(parents=True, exist_ok=True)
    (generator_path / "confirmatory.py").write_bytes(generator_bytes)
    manifest = (
        canonical_json_bytes(
            {
                "anchor_status": "PENDING_HUMAN_ANCHOR",
                "inputs": [
                    {
                        "path": "src/shadowskillbench/corpus/confirmatory.py",
                        "role": "confirmatory_generator",
                        "sha256": _current_generator_hash(),
                    }
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
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    protocol = repository_root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    protocol.joinpath("freeze_manifest.json").write_bytes(manifest)
    protocol.joinpath("freeze_manifest.sha256").write_bytes(
        f"{manifest_hash}  freeze_manifest.json\n".encode()
    )
    receipt = write_local_custody_receipt(repository_root, manifest)
    return authorize_confirmatory_generation(
        manifest,
        f"{manifest_hash}  freeze_manifest.json\n".encode(),
        receipt,
        repository_root=repository_root,
    )


def _compiled_skill(bundle) -> CompiledSkillArtifact:
    config = CompilerConfig(
        prompt_profile="SSB-SKILL-COMPILER1",
        prompt_bytes=b"Compile the supplied demonstrations into a skill.\n",
        temperature=0.0,
        seed=CORPUS_SEED,
        max_tokens=4096,
    )
    capabilities = gate2._capabilities()
    compiler_input = compiler_view(bundle)
    manifest = CompilerManifest(
        manifest_profile="SSB-COMPILER-MANIFEST1",
        compiler_input_hash=hash_compiler_input(compiler_input),
        prompt_profile=config.prompt_profile,
        system_prompt_raw_hash="sha256:" + hashlib.sha256(config.prompt_bytes).hexdigest(),
        request_profile="SSB-SKILL-COMPILER-REQUEST1",
        instruction_provenance_profile="SSB-INSTRUCTION-PROVENANCE1",
        structured_output_schema_hash=sha256_ref(SkillIR.model_json_schema()),
        declared_capabilities=capabilities,
        temperature=config.temperature,
        seed=config.seed,
        max_tokens=config.max_tokens,
    )
    response = gate2._scripted_response(
        capabilities, gate2._witness_skill(compiler_input, compiler_manifest_hash(manifest))
    )
    return asyncio.run(
        compile_skill(
            bundle,
            ScriptedModelClient(
                capabilities=cast(ProviderCapabilities, capabilities),
                script=(TransportResponse(status_code=200, body=response),),
                max_attempts=1,
            ),
            config,
        )
    )


def test_commitment_is_deterministic_and_contains_only_hash_commitments(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    rule = _rule()

    path = write_design_commitment(protocol, exclusion_rule=rule)
    verified = verify_design_commitment(path, exclusion_rule=rule)

    assert verified == build_design_commitment(exclusion_rule=rule)
    payload = json.loads(path.read_bytes())
    assert payload["corpus_seed"] == CORPUS_SEED
    assert len(payload["bundles"]) == 30
    assert len(payload["stage_a_cases"]) == 40
    assert len(payload["stage_b_cases"]) == 50
    assert len(payload["condition_definitions"]) == 10
    assert len(payload["structural_proofs"]) == 4
    assert "authority_records" not in path.read_text()
    assert "expected_decision" not in path.read_text()


def test_commitment_preserves_compiler_owned_commitment_inputs(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol"
    commitments = protocol / "commitments"
    commitments.mkdir(parents=True)
    runtime_manifest = commitments / "runtime_prompt_manifest.json"
    runtime_manifest.write_bytes(canonical_json_bytes({"profile": "compiler-owned"}))

    write_design_commitment(protocol, exclusion_rule=_rule())

    assert runtime_manifest.read_bytes() == canonical_json_bytes({"profile": "compiler-owned"})


def test_post_anchor_staging_writes_the_complete_package_source(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    rule = _rule()
    commitment = write_design_commitment(protocol, exclusion_rule=rule)
    design = build_design_commitment(exclusion_rule=rule)
    skills: dict[str, CompiledSkillArtifact] = {}
    for entry in cast(list[object], design.payload["bundles"]):
        value = cast(dict[str, object], entry)
        bundle = generate_bundle(
            cast(Literal["access_provisioning", "financial_adjustments"], value["domain"]),
            Decimal(cast(str, value["contamination_ratio"])),
            12,
            cast(int, value["seed"]),
        )
        skills[cast(str, value["bundle_id"])] = _compiled_skill(bundle.bundle)
    source = stage_confirmatory_execution_source(
        commitment_path=commitment,
        authorization=_authorization(tmp_path / "repository"),
        development_inventory=SplitInventory(
            frozenset({-1}),
            frozenset({"development_sentinel"}),
            frozenset({"sha256:" + "0" * 64}),
        ),
        compiled_skills=skills,
        run_descriptor=ModelRunDescriptor(
            provider="fixture",
            model="scripted-executor",
            model_version="2026-08-27",
            endpoint="http://localhost:9000/v1/chat/completions",
            api_key_environment="SSB_TEST_KEY",
            max_attempts=1,
        ),
        exclusion_rule=rule,
        source_dir=tmp_path / "package-source",
    )

    inputs = load_confirmatory_execution_package_inputs(
        source,
        authorization=_authorization(tmp_path / "repository"),
    )
    assert len(inputs.plan.episodes) == 9_240
    assert len(inputs.execution_plans) == 9_240


def test_run_descriptor_cli_writes_the_frozen_native_executor_profile(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    receipt = root / "protocol/commitments/role_conformance_receipt.json"
    output = tmp_path / "run-descriptor.json"

    result = CliRunner().invoke(
        cli.app,
        [
            "protocol",
            "write-confirmatory-run-descriptor",
            "--role-profile-receipt",
            str(receipt),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_bytes())
    assert payload["endpoint"] == "http://127.0.0.1:11435/api/generate"
    profile = payload["executor_runtime_profile"]
    assert profile["profile"] == "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR6"
    assert profile["temperature"] == 1.0
    assert profile["transport"] == "ollama_native_api_generate_raw"
    assert profile["prompt_profile"] == CONFIRMATORY_TURN_PROMPT_PROFILE
    assert profile["initial_output_schema_hash"] == confirmatory_initial_turn_schema_hash()
    assert profile["active_output_schema_hash"] == confirmatory_active_turn_schema_hash()
    assert profile["finished_output_schema_hash"] == confirmatory_finished_turn_schema_hash()
    assert profile["output_schema_hash"] == sha256_ref(
        {
            "initial": profile["initial_output_schema_hash"],
            "active": profile["active_output_schema_hash"],
            "finished": profile["finished_output_schema_hash"],
        }
    )
    assert profile["max_turns"] == profile["max_tool_calls"] == 12
    assert profile["max_aggregate_tokens"] == 8192


def test_run_descriptor_cli_rejects_an_unfrozen_endpoint(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    result = CliRunner().invoke(
        cli.app,
        [
            "protocol",
            "write-confirmatory-run-descriptor",
            "--endpoint",
            "http://127.0.0.1:9999/api/generate",
            "--role-profile-receipt",
            str(root / "protocol/commitments/role_conformance_receipt.json"),
            "--output",
            str(tmp_path / "run-descriptor.json"),
        ],
    )

    assert result.exit_code == 1
    assert "HOLD_CONFIRMATORY_RUN_DESCRIPTOR" in result.output
