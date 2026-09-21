from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_gemma4_profile_v3 import (
    OLLAMA_GEMMA4_CONTEXT_LENGTH,
    OLLAMA_GEMMA4_MODEFILE_SHA256,
    OLLAMA_GEMMA4_MODEL,
    OLLAMA_GEMMA4_MODEL_DIGEST,
    OLLAMA_GEMMA4_PROFILE,
    OLLAMA_GEMMA4_PROVIDER,
    OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS,
    OLLAMA_GEMMA4_SEED,
    OLLAMA_GEMMA4_SERVER_VERSION,
    OLLAMA_GEMMA4_TEMPERATURE,
    Gemma4RoleMarker,
    Gemma4RoleProfileError,
    canonical_gemma4_role_profile_receipt,
    load_gemma4_role_profile_receipt,
    ollama_gemma4_profile_projection,
    validate_gemma4_role_profile_receipt,
)


def _call(marker: str, request: str, response: str, input_tokens: int) -> dict[str, object]:
    projection = ollama_gemma4_profile_projection()
    return {
        "attempts": 1,
        "raw_request_hash": "sha256:" + request * 64,
        "raw_response_hash": "sha256:" + response * 64,
        "response_marker": marker,
        "structured_output_schema_hash": projection["structured_output_schema_hash"],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": 2,
            "total_tokens": input_tokens + 2,
        },
    }


def _receipt() -> dict[str, object]:
    return json.loads(
        canonical_gemma4_role_profile_receipt(
            baseline=_call("SSB_SYSTEM_ROLE_MARKER_7fa38c", "a", "b", 10),
            developer=_call("SSB_DEVELOPER_ROLE_MARKER_81c2ad", "c", "d", 11),
        )
    )


def _load_verifier(monkeypatch: pytest.MonkeyPatch) -> Any:
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "test_ollama_gemma4_role_profile_verifier",
        root / "scripts" / "verify_ollama_gemma4_role_profile.py",
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, verifier)
    spec.loader.exec_module(verifier)
    return verifier


def test_gemma4_projection_binds_the_frozen_commitment() -> None:
    root = Path(__file__).resolve().parents[3]
    commitment_path = root / "protocol/development_gemma4_executor_bundle_v3.json"
    base_path = root / "protocol/development_gemma4_executor_bundle_v1.json"
    commitment = json.loads(commitment_path.read_text())
    base = json.loads(base_path.read_text())
    projection = ollama_gemma4_profile_projection()

    assert projection["role_profile"] == OLLAMA_GEMMA4_PROFILE
    assert projection["provider"] == OLLAMA_GEMMA4_PROVIDER
    assert projection["model"] == OLLAMA_GEMMA4_MODEL
    assert projection["model_digest"] == OLLAMA_GEMMA4_MODEL_DIGEST
    assert projection["ollama_server_version"] == OLLAMA_GEMMA4_SERVER_VERSION
    assert projection["modelfile_sha256"] == OLLAMA_GEMMA4_MODEFILE_SHA256
    assert projection["declared_server_context_length"] == OLLAMA_GEMMA4_CONTEXT_LENGTH
    assert projection["server_context_length_binding"] == "declared_operator_bound"
    assert projection["server_context_length_per_response_observable"] is False
    assert projection["sampling_temperature"] == OLLAMA_GEMMA4_TEMPERATURE
    assert projection["seed"] == OLLAMA_GEMMA4_SEED
    assert projection["max_tokens"] == OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS == 8192
    assert commitment["role_profile"] == OLLAMA_GEMMA4_PROFILE
    assert commitment["delta"]["field"] == (
        "role_structured_output_conformance.pass.prompt_token_difference"
    )
    assert commitment["delta"]["from"] == "developer_prompt_tokens_strictly_exceed_baseline"
    assert commitment["delta"]["to"] == (
        "prompt_token_counts_must_differ_direction_not_interpreted"
    )
    assert projection["prompt_token_difference_rule"] == "nonzero_direction_not_interpreted"
    assert projection["reasoning_effort_wire"] == "omitted"
    assert projection["top_p_wire"] == "omitted"
    assert projection["top_p_effective_observed_default"] == 0.95
    assert projection["top_p_effective_per_response_observable"] is False
    assert (
        Gemma4RoleMarker.model_json_schema() == base["role_structured_output_conformance"]["schema"]
    )
    assert (
        projection["structured_output_schema"]
        == base["role_structured_output_conformance"]["schema"]
    )
    assert projection["structured_output_schema_hash"] == sha256_ref(
        base["role_structured_output_conformance"]["schema"]
    )


def test_gemma4_receipt_is_canonical_and_tamper_evident(tmp_path: Path) -> None:
    receipt = _receipt()
    raw = canonical_json_bytes(receipt)
    path = tmp_path / "receipt.json"
    path.write_bytes(raw)

    assert load_gemma4_role_profile_receipt(path) == receipt
    receipt["developer_prompt_token_delta"] = 0
    with pytest.raises(Gemma4RoleProfileError):
        validate_gemma4_role_profile_receipt(receipt)


def test_gemma4_rejects_an_old_role_receipt_and_noncanonical_bytes(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt["role_profile"] = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES11"
    with pytest.raises(Gemma4RoleProfileError):
        validate_gemma4_role_profile_receipt(receipt)

    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_receipt(), indent=2))
    with pytest.raises(Gemma4RoleProfileError, match="canonical"):
        load_gemma4_role_profile_receipt(path)


def test_verifier_writes_a_canonical_success_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    verifier = _load_verifier(monkeypatch)
    output = tmp_path / "receipt.json"
    calls = iter(
        (
            _call("SSB_SYSTEM_ROLE_MARKER_7fa38c", "a", "b", 10),
            _call("SSB_DEVELOPER_ROLE_MARKER_81c2ad", "c", "d", 11),
        )
    )
    monkeypatch.setattr(verifier, "_preflight", lambda *_args: None)
    monkeypatch.setattr(verifier, "_completion", lambda *_args: next(calls))
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["verify-gemma4", "--output", str(output)])

    assert verifier.main() == 0
    assert load_gemma4_role_profile_receipt(output)["record_kind"] == (
        "OLLAMA_GEMMA4_ROLE_STRUCTURED_OUTPUT_RECEIPT3"
    )
    assert capsys.readouterr().out == (
        "PASS_OLLAMA_GEMMA4_ROLE_PROFILE_V3: baseline_marker=SSB_SYSTEM_ROLE_MARKER_7fa38c "
        "developer_marker=SSB_DEVELOPER_ROLE_MARKER_81c2ad prompt_delta=1 temperature=1.0\n"
    )


def test_verifier_writes_secret_free_failure_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = _load_verifier(monkeypatch)
    output = tmp_path / "receipt.json"
    failure = {
        "code": "MODEL_OUTPUT_INVALID",
        "attempts": 1,
        "raw_request_hash": "sha256:" + "a" * 64,
        "raw_response_hash": "sha256:" + "b" * 64,
        "finish_reason": "length",
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    }
    monkeypatch.setattr(verifier, "_preflight", lambda *_args: None)
    monkeypatch.setattr(
        verifier,
        "_completion",
        lambda *_args: (_ for _ in ()).throw(verifier.ProbeFailure(failure)),
    )
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["verify-gemma4", "--output", str(output)])

    with pytest.raises(RuntimeError, match="failure receipt was written"):
        verifier.main()

    failed = output.with_name("receipt.failed.json")
    payload = json.loads(failed.read_bytes())
    assert set(payload["failure"]) == set(failure)
    assert "test-key" not in failed.read_text()
    assert not output.exists()


def test_verifier_preflight_binds_stock_modelfile_to_the_selected_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_verifier(monkeypatch)
    calls: list[tuple[tuple[str, ...], str | None]] = []

    class Digest:
        def hexdigest(self) -> str:
            return OLLAMA_GEMMA4_MODEFILE_SHA256.removeprefix("sha256:")

    def command(*args: str, host: str | None = None) -> str:
        calls.append((args, host))
        return "FROM gemma4:12b-it-q4_K_M\n"

    monkeypatch.setattr(verifier, "_command", command)
    monkeypatch.setattr(verifier, "_server_version", lambda _host: OLLAMA_GEMMA4_SERVER_VERSION)
    monkeypatch.setattr(verifier, "_model_digest", lambda _host, _model: OLLAMA_GEMMA4_MODEL_DIGEST)
    monkeypatch.setattr(verifier, "sha256", lambda _value: Digest())

    verifier._preflight("http://127.0.0.1:11435", "ollama")

    assert calls == [
        (("ollama", "show", OLLAMA_GEMMA4_MODEL, "--modelfile"), "http://127.0.0.1:11435")
    ]


def test_post_call_failure_preserves_both_secret_free_call_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_verifier(monkeypatch)
    baseline = _call("wrong", "a", "b", 10)
    developer = _call("wrong", "c", "d", 10)

    failure = verifier._post_call_failure(baseline, developer)

    assert failure is not None
    assert failure["reasons"] == [
        "BASELINE_MARKER_MISMATCH",
        "DEVELOPER_MARKER_MISMATCH",
        "PROMPT_TOKEN_DIFFERENCE_ZERO",
    ]
    assert failure["baseline"] == {
        "attempts": 1,
        "raw_request_hash": "sha256:" + "a" * 64,
        "raw_response_hash": "sha256:" + "b" * 64,
        "finish_reason": "stop",
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    }
