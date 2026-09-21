"""Offline falsifiers for the prospective Gemma 4 recovery route.

These tests use the real recovery runner and native executor.  The only fake
boundary is a deterministic native client; it returns the same typed response
or adapter error that the provider adapter exposes.
"""

from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

import pytest
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.development import generate_development_corpus
from shadowskillbench.episodes.executor import episode_result_projection
from shadowskillbench.episodes.models import ExperimentCondition
from shadowskillbench.experiments import gemma4_recovery
from shadowskillbench.experiments.ollama_gemma4_recovery_profile import (
    canonical_recovery_role_failure,
    recovery_candidate_spec,
)
from shadowskillbench.models import ModelAdapterError, ProviderCapabilities, TokenUsage
from shadowskillbench.models.native_tools import (
    NativeToolDefinition,
    NativeToolProviderCapabilities,
    NativeToolResponse,
    native_tool_declaration_hash,
)

Candidate = Literal["primary-26b", "fallback-12b"]


class _ScriptedNativeClient:
    """A native client boundary that produces real executor inputs."""

    def __init__(self, candidate: Candidate, outcome: str = "MODEL_OUTPUT_INVALID") -> None:
        spec = recovery_candidate_spec(candidate)
        self.capabilities = ProviderCapabilities(
            provider="ollama",
            model=spec.model,
            model_version=spec.model_digest,
            supports_system_role=True,
            supports_developer_role=True,
            supports_seed=True,
            supports_structured_output=False,
        )
        self.native_capabilities = NativeToolProviderCapabilities(
            base=self.capabilities,
            endpoint_path="/api/chat",
            supports_native_tool_calls=True,
            request_profile="SSB-OLLAMA-NATIVE-TOOLS3",
        )
        self.outcome = outcome
        self.calls = 0

    async def call_tools(
        self,
        _request: object,
        declarations: tuple[NativeToolDefinition, ...],
        *,
        history: tuple[object, ...] = (),
    ) -> NativeToolResponse:
        del history
        self.calls += 1
        if self.outcome != "success":
            raise ModelAdapterError(
                code=self.outcome,
                attempts=1,
                before_meaningful_behavior=False,
                raw_request_hash="sha256:" + "a" * 64,
                raw_response_hash="sha256:" + "b" * 64,
                finish_reason="stop",
                reported_usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        return NativeToolResponse(
            tool_name="finish_task",
            arguments={"summary": "offline scripted completion"},
            raw_request_hash="sha256:" + "a" * 64,
            raw_response_hash="sha256:" + "b" * 64,
            tool_declaration_hash=native_tool_declaration_hash(declarations),
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=None,
            attempts=1,
            finish_reason="tool_calls",
            provider_done=True,
            provider_finish_reason="stop",
        )


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _fake_receipts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, Path, Path]:
    """Keep executor tests independent of live Ollama identity captures."""

    identity = {"fixture": "identity"}
    role = {"fixture": "role"}
    monkeypatch.setattr(gemma4_recovery, "load_recovery_identity", lambda *_args: identity)
    monkeypatch.setattr(gemma4_recovery, "load_recovery_role_receipt", lambda *_args: role)
    monkeypatch.setattr(
        gemma4_recovery,
        "canonical_recovery_identity",
        lambda **_kwargs: canonical_json_bytes(identity),
    )
    identity_path = tmp_path / "identity.json"
    role_path = tmp_path / "role.json"
    show_path = tmp_path / "show.json"
    tags_path = tmp_path / "tags.json"
    version_path = tmp_path / "version.json"
    for path in (identity_path, role_path, show_path, tags_path, version_path):
        _write(path, {})
    return identity_path, role_path, show_path, tags_path, version_path


def _commitment(tmp_path: Path, candidate: Candidate) -> Path:
    path = tmp_path / f"{candidate}-commitment.json"
    gemma4_recovery.write_recovery_commitment(path, candidate)
    return path


def _run_screen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    candidate: Candidate = "primary-26b",
    outcome: str = "MODEL_OUTPUT_INVALID",
    output_name: str = "screen",
    commitment_path: Path | None = None,
    primary_terminal_audit_path: Path | None = None,
) -> tuple[Path, Path, _ScriptedNativeClient]:
    commitment = commitment_path or _commitment(tmp_path, candidate)
    identity, role, show, tags, version = _fake_receipts(monkeypatch, tmp_path / output_name)
    client = _ScriptedNativeClient(candidate, outcome)
    monkeypatch.setattr(gemma4_recovery, "build_recovery_client", lambda *_args: client)
    output = tmp_path / output_name / "output"
    asyncio.run(
        gemma4_recovery.run_recovery(
            output_dir=output,
            commitment_path=commitment,
            role_receipt_path=role,
            identity_receipt_path=identity,
            endpoint="http://127.0.0.1:11434/api/chat",
            api_key_environment="SSB_OFFLINE_TEST_KEY",
            phase="screen",
            requested_candidate=candidate,
            identity_show_path=show,
            identity_tags_path=tags,
            identity_version_path=version,
            primary_terminal_audit_path=primary_terminal_audit_path,
        )
    )
    return output, commitment, client


def _write_audit(root: Path, commitment: Path) -> dict[str, object]:
    return gemma4_recovery.write_recovery_audit(root, commitment_path=commitment, phase="screen")


def _as_sibling_screen(root: Path) -> Path:
    screen = root.parent / "screen"
    root.rename(screen)
    return screen


def _rehash_result(result: dict[str, object]) -> None:
    result.pop("content_hash", None)
    result.pop("artifact_ref", None)
    content_hash = sha256_ref(episode_result_projection(result))
    result["content_hash"] = content_hash
    result["artifact_ref"] = (
        f"artifacts/episode_result/{content_hash.removeprefix('sha256:')[:2]}/"
        f"{content_hash.removeprefix('sha256:')}.json"
    )


def _replace_envelope_result(path: Path, result: dict[str, object]) -> None:
    envelope = json.loads(path.read_bytes())
    envelope["result"] = result
    envelope["result_content_hash"] = result["content_hash"]
    _write(path, envelope)


def test_all_recovery_cells_materialize_with_their_condition_stage() -> None:
    for candidate in ("primary-26b", "fallback-12b"):
        typed_candidate = cast(Candidate, candidate)
        corpus = generate_development_corpus(recovery_candidate_spec(typed_candidate).corpus_seed)
        commitment = gemma4_recovery.recovery_commitment(corpus, typed_candidate)
        screen = gemma4_recovery.select_recovery_cells(commitment, corpus, "screen")
        validation = gemma4_recovery.select_recovery_cells(commitment, corpus, "validation")

        for phase, cells, expected_total in (("screen", screen, 8), ("validation", validation, 40)):
            assert len(cells) == expected_total
            assert sum(cell.condition is ExperimentCondition.A0_BARE for cell in cells) == 4
            assert len({tuple(cell.projection().items()) for cell in cells}) == expected_total
            for cell in cells:
                episode = gemma4_recovery.materialize_development_episode(
                    corpus,
                    gemma4_recovery.DevelopmentPlannedEpisode(
                        cell.stage, cell.condition, cell.case_id, cell.domain, cell.bundle_id
                    ),
                    executor_model_override=gemma4_recovery.ExecutorModel(
                        provider="ollama",
                        model=recovery_candidate_spec(typed_candidate).model,
                        model_version_date=recovery_candidate_spec(typed_candidate).model_digest,
                    ),
                    max_tokens=gemma4_recovery.OLLAMA_GEMMA4_RECOVERY_MAX_TOKENS,
                    corpus_seed=recovery_candidate_spec(typed_candidate).corpus_seed,
                )
                assert episode.manifest.condition.value == cell.condition.value
                expected_stage = "stage_a" if cell.condition.value.startswith("A") else "stage_b"
                assert cell.stage == expected_stage
                assert cell.phase == phase
        assert {cell.case_id for cell in screen}.isdisjoint({cell.case_id for cell in validation})


@pytest.mark.parametrize("technical_code", ["CONFIGURATION_ERROR", "MODEL_PROVIDER_TERMINAL"])
def test_technical_adapter_failures_override_baseline_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, technical_code: str
) -> None:
    root, commitment, client = _run_screen(
        monkeypatch, tmp_path, outcome=technical_code, output_name=technical_code
    )

    receipt = _write_audit(root, commitment)

    assert client.calls == 8
    assert receipt["status"] == "HOLD_TECHNICAL_INTEGRITY"
    assert any(
        cast(str, finding).startswith("PROVIDER_OR_CONFIGURATION_FAILURE:")
        for finding in cast(list[object], receipt["technical_findings"])
    )


def test_model_invalid_output_is_diagnostic_but_a0_failure_is_baseline_hold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, commitment, client = _run_screen(monkeypatch, tmp_path)

    receipt = _write_audit(root, commitment)

    assert client.calls == 8
    assert receipt["status"] == "HOLD_BASELINE_ADMISSION"
    assert receipt["technical_findings"] == []
    assert receipt["counts"] == {
        "accounted_results": 8,
        "total_cells": 8,
        "baseline_failures": 4,
    }
    diagnostics = cast(list[dict[str, object]], receipt["diagnostics"])
    assert {item["error_code"] for item in diagnostics} == {"MODEL_OUTPUT_INVALID"}


def test_rehashed_native_tool_and_world_case_substitutions_are_technical_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, commitment, _client = _run_screen(
        monkeypatch, tmp_path, outcome="success", output_name="semantic"
    )
    paths = sorted((root / "results").glob("*.json"))
    envelopes = [json.loads(path.read_bytes()) for path in paths]
    access = [item for item in envelopes if item["cell"]["domain"] == "access_provisioning"]
    assert len(access) >= 2

    native_path = paths[envelopes.index(access[0])]
    original_native_bytes = native_path.read_bytes()
    for receipt_hash in ("raw_request_hash", "raw_response_hash"):
        missing_hash = copy.deepcopy(access[0]["result"])
        missing_hash["trace"][0]["receipt"][receipt_hash] = None
        _rehash_result(missing_hash)
        _replace_envelope_result(native_path, missing_hash)
        missing_hash_audit = gemma4_recovery.audit_recovery(
            root, commitment_path=commitment, phase="screen"
        )
        assert missing_hash_audit["status"] == "HOLD_TECHNICAL_INTEGRITY"
        native_path.write_bytes(original_native_bytes)

    native_tampered = copy.deepcopy(access[0]["result"])
    native_tampered["trace"][0]["receipt"]["action_interface_hash"] = "sha256:" + "f" * 64
    _rehash_result(native_tampered)
    _replace_envelope_result(native_path, native_tampered)
    native_audit = gemma4_recovery.audit_recovery(root, commitment_path=commitment, phase="screen")
    assert native_audit["status"] == "HOLD_TECHNICAL_INTEGRITY"

    # Substitute a separately valid result from another access case, but bind its
    # outer episode and manifest identifiers to the target envelope and rehash it.
    # Internal result validation still succeeds; only the recovery audit can catch
    # the mismatched world, task case, and authority decision.
    original = access[1]
    forged = copy.deepcopy(access[0]["result"])
    forged["episode_id"] = original["result"]["episode_id"]
    forged["manifest_hash"] = original["result"]["manifest_hash"]
    forged["terminal_record"]["episode_id"] = original["result"]["episode_id"]
    _rehash_result(forged)
    target_path = paths[envelopes.index(original)]
    _replace_envelope_result(target_path, forged)
    substituted_audit = gemma4_recovery.audit_recovery(
        root, commitment_path=commitment, phase="screen"
    )
    assert substituted_audit["status"] == "HOLD_TECHNICAL_INTEGRITY"


def test_forged_or_missing_primary_audit_never_calls_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commitment = _commitment(tmp_path, "fallback-12b")
    client = _ScriptedNativeClient("fallback-12b")
    monkeypatch.setattr(gemma4_recovery, "build_recovery_client", lambda *_args: client)
    forged = tmp_path / "forged-primary-audit.json"
    _write(
        forged,
        {
            "candidate": "primary-26b",
            "phase": "screen",
            "status": "HOLD_BASELINE_ADMISSION",
            "technical_findings": [],
        },
    )

    with pytest.raises(gemma4_recovery.Gemma4RecoveryHold):
        asyncio.run(
            gemma4_recovery.run_recovery(
                output_dir=tmp_path / "fallback-forged",
                commitment_path=commitment,
                role_receipt_path=tmp_path / "missing-role.json",
                identity_receipt_path=tmp_path / "missing-identity.json",
                endpoint="http://127.0.0.1:11434/api/chat",
                api_key_environment="SSB_OFFLINE_TEST_KEY",
                phase="screen",
                requested_candidate="fallback-12b",
                identity_show_path=tmp_path / "show.json",
                identity_tags_path=tmp_path / "tags.json",
                identity_version_path=tmp_path / "version.json",
                primary_terminal_audit_path=forged,
            )
        )
    assert client.calls == 0
    with pytest.raises(gemma4_recovery.Gemma4RecoveryHold):
        asyncio.run(
            gemma4_recovery.run_recovery(
                output_dir=tmp_path / "fallback-missing",
                commitment_path=commitment,
                role_receipt_path=tmp_path / "missing-role.json",
                identity_receipt_path=tmp_path / "missing-identity.json",
                endpoint="http://127.0.0.1:11434/api/chat",
                api_key_environment="SSB_OFFLINE_TEST_KEY",
                phase="screen",
                requested_candidate="fallback-12b",
                identity_show_path=tmp_path / "show.json",
                identity_tags_path=tmp_path / "tags.json",
                identity_version_path=tmp_path / "version.json",
            )
        )
    assert client.calls == 0


@pytest.mark.parametrize(
    ("failure_code", "expected_calls"),
    [("MODEL_OUTPUT_INVALID", 8), ("CONFIGURATION_ERROR", 0)],
)
def test_only_model_origin_role_failure_can_admit_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_code: str,
    expected_calls: int,
) -> None:
    commitment = _commitment(tmp_path, "fallback-12b")
    identity, role, show, tags, version = _fake_receipts(monkeypatch, tmp_path / failure_code)
    failure_path = tmp_path / f"{failure_code}.json"
    failure_path.write_bytes(
        canonical_recovery_role_failure(
            candidate="primary-26b", failure_code=failure_code, completed_calls=1
        )
    )
    client = _ScriptedNativeClient("fallback-12b")
    monkeypatch.setattr(gemma4_recovery, "build_recovery_client", lambda *_args: client)

    if expected_calls:
        asyncio.run(
            gemma4_recovery.run_recovery(
                output_dir=tmp_path / "admitted",
                commitment_path=commitment,
                role_receipt_path=role,
                identity_receipt_path=identity,
                endpoint="http://127.0.0.1:11434/api/chat",
                api_key_environment="SSB_OFFLINE_TEST_KEY",
                phase="screen",
                requested_candidate="fallback-12b",
                identity_show_path=show,
                identity_tags_path=tags,
                identity_version_path=version,
                primary_role_failure_path=failure_path,
            )
        )
    else:
        with pytest.raises(gemma4_recovery.Gemma4RecoveryHold):
            asyncio.run(
                gemma4_recovery.run_recovery(
                    output_dir=tmp_path / "rejected",
                    commitment_path=commitment,
                    role_receipt_path=role,
                    identity_receipt_path=identity,
                    endpoint="http://127.0.0.1:11434/api/chat",
                    api_key_environment="SSB_OFFLINE_TEST_KEY",
                    phase="screen",
                    requested_candidate="fallback-12b",
                    identity_show_path=show,
                    identity_tags_path=tags,
                    identity_version_path=version,
                    primary_role_failure_path=failure_path,
                )
            )
    assert client.calls == expected_calls


def test_cold_process_validation_audit_never_calls_asyncio_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the async validation antecedent audit with no process cache."""

    root, _commitment_path, _client = _run_screen(monkeypatch, tmp_path, output_name="cold-source")
    screen_root = _as_sibling_screen(root)
    forged_screen = screen_root / "screen-audit.json"
    _write(forged_screen, {"candidate": "primary-26b", "phase": "screen", "status": "PASS_SCREEN"})
    identity, role, show, tags, version = _fake_receipts(monkeypatch, tmp_path / "cold-child")
    arguments = {
        "output": str(screen_root.parent / "validation"),
        "commitment": str(screen_root / "commitment.json"),
        "role": str(role),
        "identity": str(identity),
        "show": str(show),
        "tags": str(tags),
        "version": str(version),
        "screen": str(forged_screen),
    }
    program = """
import asyncio
import json
from pathlib import Path
from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.experiments import gemma4_recovery as recovery

args = json.loads(__ARGS__)
identity = {\"fixture\": \"identity\"}
role = {\"fixture\": \"role\"}
recovery.load_recovery_identity = lambda *_args: identity
recovery.load_recovery_role_receipt = lambda *_args: role
recovery.canonical_recovery_identity = lambda **_kwargs: canonical_json_bytes(identity)
try:
    asyncio.run(recovery.run_recovery(
        output_dir=Path(args[\"output\"]), commitment_path=Path(args[\"commitment\"]),
        role_receipt_path=Path(args[\"role\"]), identity_receipt_path=Path(args[\"identity\"]),
        endpoint=\"http://127.0.0.1:11434/api/chat\", api_key_environment=\"SSB_OFFLINE_TEST_KEY\",
        phase=\"validation\", requested_candidate=\"primary-26b\",
        identity_show_path=Path(args[\"show\"]), identity_tags_path=Path(args[\"tags\"]),
        identity_version_path=Path(args[\"version\"]), screen_audit_path=Path(args[\"screen\"]),
    ))
except recovery.Gemma4RecoveryHold as error:
    print(error)
else:
    raise SystemExit(2)
""".replace("__ARGS__", repr(json.dumps(arguments)))
    child = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert child.returncode == 0, child.stderr
    assert "HOLD_RECOVERY_SCREEN_AUDIT: invalid" in child.stdout


def test_direct_validation_audit_rejects_forged_embedded_screen_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The auditor must recompute the fixed sibling screen receipt itself."""

    candidate: Candidate = "primary-26b"
    commitment_path = _commitment(tmp_path, candidate)
    identity_path, role_path, show_path, tags_path, version_path = _fake_receipts(
        monkeypatch, tmp_path / "validation-inputs"
    )
    del identity_path, role_path
    root = tmp_path / "experiment" / "validation"
    screen_root = root.parent / "screen"
    screen_root.mkdir(parents=True)
    (screen_root / "commitment.json").write_bytes(commitment_path.read_bytes())
    root.mkdir()
    (root / "results").mkdir()
    commitment = gemma4_recovery.load_recovery_commitment(commitment_path, candidate)
    corpus = generate_development_corpus(recovery_candidate_spec(candidate).corpus_seed)
    cells = gemma4_recovery.select_recovery_cells(commitment, corpus, "validation")
    forged_screen: dict[str, object] = {
        "candidate": candidate,
        "phase": "screen",
        "status": "PASS_SCREEN",
    }
    identity: dict[str, object] = {"fixture": "identity"}
    role: dict[str, object] = {"fixture": "role"}
    plan = gemma4_recovery._plan(commitment, cells, "validation", forged_screen)
    runtime = gemma4_recovery._runtime(
        "http://127.0.0.1:11434/api/chat", "SSB_OFFLINE_TEST_KEY", candidate, identity, role
    )
    _write(root / "capability-plan.json", {**plan, "plan_hash": sha256_ref(plan)})
    _write(root / "runtime.json", {**runtime, "runtime_hash": sha256_ref(runtime)})
    _write(root / "screen-audit.json", forged_screen)
    _write(root / "model-identity-receipt.json", identity)
    _write(root / "role-profile-receipt.json", role)
    for target, source in (
        (root / "identity-show.raw.json", show_path),
        (root / "identity-tags.raw.json", tags_path),
        (root / "identity-version.raw.json", version_path),
    ):
        target.write_bytes(source.read_bytes())

    receipt = gemma4_recovery.audit_recovery(
        root, commitment_path=commitment_path, phase="validation"
    )

    assert receipt["status"] == "HOLD_TECHNICAL_INTEGRITY"
    assert "SCREEN_AUDIT_BINDING" in cast(list[str], receipt["technical_findings"])


def test_repeated_run_and_forged_screen_pass_reject_before_native_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, commitment, client = _run_screen(monkeypatch, tmp_path, output_name="first")
    assert client.calls == 8
    calls_before_repeat = client.calls
    identity, role, show, tags, version = _fake_receipts(monkeypatch, tmp_path / "repeat")
    with pytest.raises(gemma4_recovery.Gemma4RecoveryHold):
        asyncio.run(
            gemma4_recovery.run_recovery(
                output_dir=root,
                commitment_path=commitment,
                role_receipt_path=role,
                identity_receipt_path=identity,
                endpoint="http://127.0.0.1:11434/api/chat",
                api_key_environment="SSB_OFFLINE_TEST_KEY",
                phase="screen",
                requested_candidate="primary-26b",
                identity_show_path=show,
                identity_tags_path=tags,
                identity_version_path=version,
            )
        )
    assert client.calls == calls_before_repeat

    screen_root = _as_sibling_screen(root)
    forged_screen = screen_root / "screen-audit.json"
    _write(forged_screen, {"candidate": "primary-26b", "phase": "screen", "status": "PASS_SCREEN"})
    validation_client = _ScriptedNativeClient("primary-26b")
    monkeypatch.setattr(gemma4_recovery, "build_recovery_client", lambda *_args: validation_client)
    with pytest.raises(gemma4_recovery.Gemma4RecoveryHold):
        asyncio.run(
            gemma4_recovery.run_recovery(
                output_dir=screen_root.parent / "validation",
                commitment_path=screen_root / "commitment.json",
                role_receipt_path=role,
                identity_receipt_path=identity,
                endpoint="http://127.0.0.1:11434/api/chat",
                api_key_environment="SSB_OFFLINE_TEST_KEY",
                phase="validation",
                requested_candidate="primary-26b",
                identity_show_path=show,
                identity_tags_path=tags,
                identity_version_path=version,
                screen_audit_path=forged_screen,
            )
        )
    assert validation_client.calls == 0


def test_invalid_cli_phase_rejects_before_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(cli, "run_recovery", lambda **_kwargs: calls.append(_kwargs))

    result = CliRunner().invoke(
        cli.app,
        [
            "pilot",
            "recovery-run",
            "--candidate",
            "primary-26b",
            "--phase",
            "not-a-phase",
            "--endpoint",
            "http://127.0.0.1:11434/api/chat",
            "--api-key-environment",
            "SSB_OFFLINE_TEST_KEY",
            "--commitment",
            str(tmp_path / "commitment.json"),
            "--role-profile-receipt",
            str(tmp_path / "role.json"),
            "--model-identity-receipt",
            str(tmp_path / "identity.json"),
            "--identity-show",
            str(tmp_path / "show.json"),
            "--identity-tags",
            str(tmp_path / "tags.json"),
            "--identity-version",
            str(tmp_path / "version.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 1
    assert "invalid candidate or phase" in result.output
    assert calls == []
