from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_nemotron_profile import (
    OLLAMA_NEMOTRON_CONFIG_BLOB_SHA256,
    OLLAMA_NEMOTRON_CONTEXT_LENGTH,
    OLLAMA_NEMOTRON_MODEFILE_SHA256,
    OLLAMA_NEMOTRON_MODEL,
    OLLAMA_NEMOTRON_MODEL_DIGEST,
    OLLAMA_NEMOTRON_PROFILE,
    OLLAMA_NEMOTRON_PROVIDER,
    OLLAMA_NEMOTRON_ROLE_PROBE_MAX_TOKENS,
    OLLAMA_NEMOTRON_SEED,
    OLLAMA_NEMOTRON_SERVER_VERSION,
    OLLAMA_NEMOTRON_TEMPERATURE,
    NemotronRoleProfileError,
    canonical_nemotron_live_show_identity_receipt,
    canonical_nemotron_role_profile_receipt,
    load_nemotron_role_profile_receipt,
    nemotron_live_show_identity_projection,
    ollama_nemotron_profile_projection,
    validate_nemotron_live_show_identity,
    validate_nemotron_role_profile_receipt,
)


def _call(marker: str, request: str, response: str, input_tokens: int) -> dict[str, object]:
    projection = ollama_nemotron_profile_projection()
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


def _receipt(*, baseline_tokens: int = 20, developer_tokens: int = 18) -> dict[str, object]:
    return json.loads(
        canonical_nemotron_role_profile_receipt(
            baseline=_call("SSB_SYSTEM_ROLE_MARKER_7fa38c", "a", "b", baseline_tokens),
            developer=_call("SSB_DEVELOPER_ROLE_MARKER_81c2ad", "c", "d", developer_tokens),
        )
    )


def test_nemotron_projection_binds_role_separation_and_structured_output() -> None:
    projection = ollama_nemotron_profile_projection()

    assert projection["role_profile"] == OLLAMA_NEMOTRON_PROFILE
    assert projection["provider"] == OLLAMA_NEMOTRON_PROVIDER
    assert projection["model"] == OLLAMA_NEMOTRON_MODEL
    assert projection["model_digest"] == OLLAMA_NEMOTRON_MODEL_DIGEST
    assert projection["ollama_server_version"] == OLLAMA_NEMOTRON_SERVER_VERSION
    assert projection["config_blob_sha256"] == OLLAMA_NEMOTRON_CONFIG_BLOB_SHA256
    assert projection["modelfile_sha256"] == OLLAMA_NEMOTRON_MODEFILE_SHA256
    assert projection["declared_server_context_length"] == OLLAMA_NEMOTRON_CONTEXT_LENGTH
    assert projection["sampling_temperature"] == OLLAMA_NEMOTRON_TEMPERATURE == 0.15
    assert projection["seed"] == OLLAMA_NEMOTRON_SEED == 4242
    assert projection["max_tokens"] == OLLAMA_NEMOTRON_ROLE_PROBE_MAX_TOKENS == 8192
    assert projection["endpoint_path"] == "/v1/chat/completions"
    assert projection["request_profile"] == "SSB-OAI-CHAT1"
    assert projection["top_p_wire"] == "omitted"
    assert projection["reasoning_effort_wire"] == "omitted"
    assert "native" not in json.dumps(projection["role_structured_output_probe"]).lower()
    assert "tool_call" not in json.dumps(projection["role_structured_output_probe"])


def test_nemotron_receipt_proves_distinct_role_markers_without_raw_content(tmp_path: Path) -> None:
    receipt = _receipt()
    path = tmp_path / "receipt.json"
    path.write_bytes(canonical_json_bytes(receipt))

    loaded = load_nemotron_role_profile_receipt(path)
    assert loaded["developer_prompt_token_delta"] == -2
    assert '"raw_request":' not in json.dumps(loaded)
    assert '"raw_response":' not in json.dumps(loaded)

    receipt["developer_prompt_token_delta"] = 0
    with pytest.raises(NemotronRoleProfileError):
        validate_nemotron_role_profile_receipt(receipt)


def test_nemotron_commitment_binds_fresh_corpus_selection_and_pending_live_identity() -> None:
    root = Path(__file__).resolve().parents[3]
    commitment = json.loads(
        (root / "protocol/development_nemotron35_lightning_mlx_executor_bundle_v1.json").read_text()
    )
    selection = commitment["selection"]

    assert sha256_ref(selection) == commitment["selection_hash"]
    assert selection["corpus"] == {
        "content_hash": "sha256:39af9311f72df3dc174e7c3c4a6a244cd8649c1e62e6e97145fa54689ebde6c6",
        "generation_seed": 4243,
        "profile": "SSB-DEVELOPMENT-CORPUS-NEMOTRON35-MLX1",
    }
    assert commitment["preflight"]["model_identity_live_show"]["status"] == (
        "PENDING_LIVE_SHOW_RESULT"
    )


def test_nemotron_live_show_validator_accepts_only_hashed_identity_custody() -> None:
    body = {
        "record_kind": "OLLAMA_NEMOTRON_LIVE_SHOW_IDENTITY_RECEIPT1",
        **nemotron_live_show_identity_projection(),
        "show_response_hash": "sha256:" + "0" * 64,
    }
    receipt = {**body, "receipt_hash": sha256_ref(body)}

    assert validate_nemotron_live_show_identity(receipt) == receipt
    receipt["show_response"] = {"raw": "not permitted"}
    with pytest.raises(NemotronRoleProfileError):
        validate_nemotron_live_show_identity(receipt)


def test_nemotron_live_show_identity_binds_the_frozen_modelfile() -> None:
    assert (
        nemotron_live_show_identity_projection()["modelfile_sha256"]
        == OLLAMA_NEMOTRON_MODEFILE_SHA256
    )


def test_nemotron_verifier_uses_the_modelfile_hash_not_the_config_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "test_ollama_nemotron_role_profile_verifier",
        root / "scripts" / "verify_ollama_nemotron_role_profile.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    verifier = module.verifier
    assert verifier.OLLAMA_GEMMA4_MODEFILE_SHA256 == OLLAMA_NEMOTRON_MODEFILE_SHA256
    assert verifier.OLLAMA_GEMMA4_MODEFILE_SHA256 != OLLAMA_NEMOTRON_CONFIG_BLOB_SHA256


def test_nemotron_live_show_recorder_is_hash_only_and_fails_closed() -> None:
    show = {"capabilities": ["completion", "tools", "thinking"], "unretained": "provider"}
    receipt = json.loads(
        canonical_nemotron_live_show_identity_receipt(
            show_response=canonical_json_bytes(show),
            ollama_version_text="ollama version is 0.33.2\n",
        )
    )

    assert (
        receipt["show_response_hash"] == "sha256:" + sha256(canonical_json_bytes(show)).hexdigest()
    )
    assert "unretained" not in receipt
    with pytest.raises(NemotronRoleProfileError, match="capabilities"):
        canonical_nemotron_live_show_identity_receipt(
            show_response=canonical_json_bytes({"capabilities": ["completion", "tools"]}),
            ollama_version_text="ollama version is 0.33.2\n",
        )
