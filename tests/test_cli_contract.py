from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.corpus import confirmatory
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    canonical_role_profile_receipt,
    role_conformance_probe_projection,
)
from shadowskillbench.models import ModelAdapterError
from shadowskillbench.release import ReproductionResult
from shadowskillbench.skills.compiler import CompilerContractError
from tests.custody import write_local_custody_receipt

CLI_LEAF_CONTRACTS = (
    (["world", "validate", "--all"], 0, "PASS_WORLD_VALIDATION"),
    (["traces", "generate", "--split", "development"], 0, "PRACTICE_NOT_EVIDENCE"),
    (["skills", "compile", "--split", "confirmatory"], 1, "HOLD_SKILL_COMPILATION_SCOPE"),
    (
        ["episodes", "run", "--stage-a", "--split", "development"],
        1,
        "HOLD_DEVELOPMENT_EPISODE_COMPOSITION",
    ),
    (["authority", "validate"], 0, "PASS_AUTHORITY_VALIDATION"),
    (
        ["episodes", "run", "--stage-b", "--split", "development"],
        1,
        "HOLD_DEVELOPMENT_EPISODE_COMPOSITION",
    ),
    (["corpus", "generate-confirmatory"], 1, "HOLD_CONFIRMATORY_CORPUS_GENERATION"),
    (
        ["episodes", "run-confirmatory", "--stage-a"],
        1,
        "HOLD_CONFIRMATORY_EXECUTION_AUTHORIZATION",
    ),
    (
        ["episodes", "run-confirmatory", "--stage-b"],
        1,
        "HOLD_CONFIRMATORY_EXECUTION_AUTHORIZATION",
    ),
    (
        ["experiments", "audit", "--split", "development"],
        1,
        "HOLD_DEVELOPMENT_EXPERIMENT_AUDIT",
    ),
    (
        ["experiments", "audit", "--stage-a"],
        1,
        "HOLD_MISSING_EXPERIMENT_ARTIFACTS",
    ),
    (
        ["experiments", "audit", "--stage-b"],
        1,
        "HOLD_MISSING_EXPERIMENT_ARTIFACTS",
    ),
    (["corpus", "audit", "--split", "confirmatory"], 1, "HOLD_CONFIRMATORY_CORPUS_AUDIT"),
)


def _role_profile_receipt() -> bytes:
    probe = role_conformance_probe_projection()
    return canonical_role_profile_receipt(
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


def run_command(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "shadowskillbench.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )


@pytest.mark.parametrize(
    ("args", "expected_code", "status"),
    CLI_LEAF_CONTRACTS,
)
def test_cli_leaves_are_operational_or_name_their_missing_composition(
    args: list[str], expected_code: int, status: str, tmp_path: Path
) -> None:
    result = run_command(args, cwd=tmp_path)

    assert result.returncode == expected_code
    assert status in result.stdout + result.stderr
    assert "not implemented" not in result.stdout.lower() + result.stderr.lower()


def test_help_is_available() -> None:
    result = run_command(["--help"])

    assert result.returncode == 0
    assert "world" in result.stdout
    assert "reproduce" in result.stdout


def test_confirmatory_corpus_commands_require_and_accept_local_anchor_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    generator_path = Path(confirmatory.__file__)
    generator_hash = "sha256:" + hashlib.sha256(generator_path.read_bytes()).hexdigest()
    manifest = (
        json.dumps(
            {
                "anchor_status": "PENDING_HUMAN_ANCHOR",
                "inputs": [
                    {
                        "path": "src/shadowskillbench/corpus/confirmatory.py",
                        "role": "confirmatory_generator",
                        "sha256": generator_hash,
                    }
                ],
                "preregistration_core": {
                    "path": "protocol/preregistration.md",
                    "sha256": "sha256:" + "a" * 64,
                },
                "schema_version": "1.0",
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    (protocol / "freeze_manifest.json").write_bytes(manifest)
    (protocol / "freeze_manifest.sha256").write_text(f"{manifest_hash}  freeze_manifest.json\n")
    checked_out_generator = tmp_path / "src/shadowskillbench/corpus/confirmatory.py"
    checked_out_generator.parent.mkdir(parents=True)
    checked_out_generator.write_bytes(generator_path.read_bytes())
    write_local_custody_receipt(tmp_path, manifest)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    generated = runner.invoke(cli.app, ["corpus", "generate-confirmatory"])
    audited = runner.invoke(cli.app, ["corpus", "audit", "--split", "confirmatory"])

    assert generated.exit_code == 0, generated.output
    assert "PASS_CONFIRMATORY_CORPUS_GENERATION" in generated.output
    assert audited.exit_code == 0, audited.output
    assert "PASS_MODEL_VISIBLE_CORPUS_BOUNDARY" in audited.output


def test_confirmatory_corpus_rejects_seed_overrides_before_anchor_io() -> None:
    result = CliRunner().invoke(
        cli.app,
        ["corpus", "generate-confirmatory", "--corpus-seed", "104731"],
    )

    assert result.exit_code == 1
    assert "corpus seed must be frozen value 104730" in result.output


def test_development_skill_compilation_persists_a_scripted_gate2_receipt(tmp_path: Path) -> None:
    output = tmp_path / "gate2.json"
    prompt = Path(__file__).parents[1] / "prompts" / "skill_compiler.md"

    result = CliRunner().invoke(
        cli.app,
        [
            "skills",
            "compile",
            "--split",
            "development",
            "--output",
            str(output),
            "--prompt",
            str(prompt),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "PASS_DEVELOPMENT_SKILL_COMPILATION" in result.output
    assert output.is_file()


def test_live_pilot_adapter_error_has_a_clean_hold_without_a_typer_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_pilot(**_kwargs: object) -> dict[str, object]:
        raise ModelAdapterError(
            code="MODEL_PROVIDER_TRANSIENT",
            attempts=1,
            before_meaningful_behavior=True,
            raw_request_hash="sha256:" + "a" * 64,
            raw_response_hash=None,
        )

    monkeypatch.setattr(cli, "client_from_run_descriptor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli, "ollama_native_client_from_run_descriptor", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli, "run_live_pilot", fail_pilot)
    receipt = tmp_path / "ollama-role-profile-receipt.json"
    receipt.write_bytes(_role_profile_receipt())

    result = CliRunner().invoke(
        cli.app,
        [
            "pilot",
            "live-run",
            "--endpoint",
            "http://127.0.0.1:8080/v1/chat/completions",
            "--api-key-environment",
            "SSB_TEST_KEY",
            "--role-profile-receipt",
            str(receipt),
            "--limit",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "HOLD_EXPLORATORY_LIVE_PILOT: MODEL_PROVIDER_TRANSIENT" in result.output
    assert "Traceback" not in result.output
    assert tuple(tmp_path.iterdir()) == (receipt,)


def test_live_pilot_compiler_contract_error_has_a_clean_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_pilot(**_kwargs: object) -> dict[str, object]:
        raise CompilerContractError("COMPILER_OUTPUT_INVALID")

    monkeypatch.setattr(cli, "client_from_run_descriptor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli, "ollama_native_client_from_run_descriptor", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(cli, "run_live_pilot", fail_pilot)
    receipt = tmp_path / "ollama-role-profile-receipt.json"
    receipt.write_bytes(_role_profile_receipt())

    result = CliRunner().invoke(
        cli.app,
        [
            "pilot",
            "live-run",
            "--endpoint",
            "http://127.0.0.1:8080/v1/chat/completions",
            "--api-key-environment",
            "SSB_TEST_KEY",
            "--role-profile-receipt",
            str(receipt),
            "--limit",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "HOLD_EXPLORATORY_LIVE_PILOT: COMPILER_OUTPUT_INVALID" in result.output
    assert "Traceback" not in result.output
    assert "immutable" not in result.output


def test_live_pilot_cli_pins_the_extended_request_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def no_op_pilot(**kwargs: object) -> dict[str, object]:
        captured["descriptor"] = kwargs["descriptor"]
        return {
            "completed": 0,
            "skipped": 0,
            "limit": 1,
            "total_cells": 60,
        }

    compiler_configs: list[dict[str, object]] = []
    executor_configs: list[dict[str, object]] = []

    def compiler_factory(*_args: object, **kwargs: object) -> object:
        compiler_configs.append(kwargs)
        return object()

    def executor_factory(*_args: object, **kwargs: object) -> object:
        executor_configs.append(kwargs)
        return object()

    monkeypatch.setattr(cli, "client_from_run_descriptor", compiler_factory)
    monkeypatch.setattr(cli, "ollama_native_client_from_run_descriptor", executor_factory)
    monkeypatch.setattr(cli, "run_live_pilot", no_op_pilot)
    receipt = tmp_path / "ollama-role-profile-receipt.json"
    receipt.write_bytes(_role_profile_receipt())

    result = CliRunner().invoke(
        cli.app,
        [
            "pilot",
            "live-run",
            "--endpoint",
            "http://127.0.0.1:8080/v1/chat/completions",
            "--api-key-environment",
            "SSB_TEST_KEY",
            "--role-profile-receipt",
            str(receipt),
            "--limit",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [config["reasoning_effort"] for config in compiler_configs] == ["low"]
    assert [config["timeout_seconds"] for config in compiler_configs + executor_configs] == [
        900.0,
        900.0,
    ]
    assert all(config["trust_env"] is False for config in compiler_configs + executor_configs)
    assert executor_configs[0]["top_p"] == 1.0
    assert executor_configs[0]["context_length"] == 131_072
    assert isinstance(captured["descriptor"], cli.LivePilotDescriptor)
    assert captured["descriptor"].sampling_temperature == 1.0
    assert captured["descriptor"].executor_sampling_temperature == 1.0
    assert captured["descriptor"].reasoning_effort == "low"
    assert captured["descriptor"].executor_reasoning_effort == "none"
    assert captured["descriptor"].ollama_server_version == "0.33.2"
    assert captured["descriptor"].executor_endpoint == "http://127.0.0.1:8080/api/generate"
    assert captured["descriptor"].executor_think is False
    assert captured["descriptor"].executor_model_name == "ssb-gpt-oss-20b-harmony-final:v2"


def test_reproduce_has_its_own_truthful_hold_instead_of_not_implemented() -> None:
    result = run_command(["reproduce", "--protocol", "protocol/freeze_manifest.json"])

    assert result.returncode == 1
    assert "HOLD_MISSING_SEALED_INPUT" in result.stderr
    assert "not implemented" not in result.stderr.lower()


def test_reproduce_names_the_confirmatory_claim_ceiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "reproduce_sealed_artifacts",
        lambda *args: ReproductionResult(
            classification="CONFIRMATORY",
            report_sha256="sha256:" + "a" * 64,
            workbench_sha256="sha256:" + "b" * 64,
            output_paths=(),
        ),
    )

    result = CliRunner().invoke(cli.app, ["reproduce", "--protocol", str(tmp_path / "freeze.json")])

    assert result.exit_code == 0, result.output
    assert "PASS_CONFIRMATORY_REPRODUCTION" in result.output
    assert "claim_ceiling=confirmatory" in result.output


@pytest.mark.parametrize("args", (["analyze"], ["report", "build"]))
def test_sealed_analysis_commands_hold_for_invalid_custody_before_reading_inputs(
    args: list[str],
) -> None:
    result = run_command(args)

    assert result.returncode == 1
    assert "HOLD_INVALID_CONFIRMATORY_AUTHORIZATION" in result.stderr
    assert "not implemented" not in result.stderr.lower()


def test_report_build_passes_one_evidence_object_to_both_artifact_publisher_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "sha256:" + "a" * 64
    sealed = SimpleNamespace(
        dataset=object(),
        stage_a=object(),
        stage_b=object(),
        dataset_sha256=digest,
        audit=SimpleNamespace(
            report_hash=digest,
            plan_hash=digest,
            runtime_binding_hash=digest,
        ),
        runtime_binding=SimpleNamespace(
            freeze_manifest_hash=digest,
            anchor_receipt_hash=digest,
            plan_hash=digest,
        ),
        exclusion_policy=None,
    )
    protocol_source = SimpleNamespace(
        protocol_tag="v1.0",
        freeze_manifest_sha256=digest,
        prompt_hashes=(digest,),
        code_commit="abcdef0",
        reproduce_command="uv run reproduce",
        external_anchor_locator="anchor",
        custody_locator="anchor",
        custody_mode="EXTERNAL_SIGNED_ANCHOR",
        anchor_receipt_sha256=digest,
        anchor_verified=True,
    )
    evidence = object()
    captured: list[tuple[object, Path, Path]] = []
    captured_claims_ledgers: list[Path] = []
    monkeypatch.setattr(cli, "_scientific_freeze_binding", lambda _: object())
    monkeypatch.setattr(cli, "analyze_sealed_confirmatory_artifacts", lambda _, **__: sealed)
    monkeypatch.setattr(cli, "_read_analysis_receipt", lambda _: {"receipt": True})
    monkeypatch.setattr(cli, "sealed_analysis_receipt_matches", lambda *_: True)
    monkeypatch.setattr(
        cli,
        "validate_preregistration",
        lambda _: SimpleNamespace(valid=True, core=object()),
    )
    monkeypatch.setattr(
        cli,
        "validate_claims_ledger",
        lambda path, _root: captured_claims_ledgers.append(path) or SimpleNamespace(valid=True),
    )
    monkeypatch.setattr(cli, "_authorized_protocol_evidence", lambda _: protocol_source)
    monkeypatch.setattr(cli, "ReportEvidence", lambda **_: evidence)
    monkeypatch.setattr(cli, "report_status", lambda _: SimpleNamespace(value="confirmatory"))
    monkeypatch.setattr(
        cli, "validate_release_claims", lambda *_args, **_kwargs: SimpleNamespace(passed=True)
    )
    monkeypatch.setattr(
        cli,
        "write_report_artifacts",
        lambda item, html, workbench: captured.append((item, html, workbench)),
    )

    output = tmp_path / "report.html"
    claims_ledger = tmp_path / "claims.yaml"
    result = CliRunner().invoke(
        cli.app,
        [
            "report",
            "build",
            "--output",
            str(output),
            "--claims-ledger",
            str(claims_ledger),
            "--limitation",
            "test limitation",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == [(evidence, output, Path("artifacts/reports/workbench.json"))]
    assert captured_claims_ledgers == [claims_ledger]


@pytest.mark.parametrize("args", (["protocol", "validate"], ["protocol", "freeze"]))
def test_protocol_commands_hold_for_missing_preregistration_instead_of_stubbing(
    args: list[str], tmp_path: Path
) -> None:
    result = run_command(args, cwd=tmp_path)

    assert result.returncode == 1
    assert "HOLD_PREREGISTRATION_INVALID" in result.stderr
    assert "protocol/preregistration.md" in result.stderr
    assert "not implemented" not in result.stderr.lower()
