from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_mistral_profile import (
    OLLAMA_MISTRAL_CONTEXT_LENGTH,
    OLLAMA_MISTRAL_MODEFILE_SHA256,
    OLLAMA_MISTRAL_MODEL,
    OLLAMA_MISTRAL_MODEL_DIGEST,
    OLLAMA_MISTRAL_PROFILE,
    OLLAMA_MISTRAL_PROVIDER,
    OLLAMA_MISTRAL_ROLE_PROBE_MAX_TOKENS,
    OLLAMA_MISTRAL_SEED,
    OLLAMA_MISTRAL_SERVER_VERSION,
    OLLAMA_MISTRAL_TEMPERATURE,
    MistralRoleProfileError,
    canonical_mistral_role_profile_receipt,
    load_mistral_role_profile_receipt,
    ollama_mistral_profile_projection,
    validate_mistral_role_profile_receipt,
)


def _call(marker: str, request: str, response: str, input_tokens: int) -> dict[str, object]:
    projection = ollama_mistral_profile_projection()
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
        canonical_mistral_role_profile_receipt(
            baseline=_call("SSB_SYSTEM_ROLE_MARKER_7fa38c", "a", "b", baseline_tokens),
            developer=_call("SSB_DEVELOPER_ROLE_MARKER_81c2ad", "c", "d", developer_tokens),
        )
    )


def test_mistral_projection_binds_the_frozen_candidate() -> None:
    projection = ollama_mistral_profile_projection()

    assert projection["role_profile"] == OLLAMA_MISTRAL_PROFILE
    assert projection["provider"] == OLLAMA_MISTRAL_PROVIDER
    assert projection["model"] == OLLAMA_MISTRAL_MODEL
    assert projection["model_digest"] == OLLAMA_MISTRAL_MODEL_DIGEST
    assert projection["ollama_server_version"] == OLLAMA_MISTRAL_SERVER_VERSION
    assert projection["modelfile_sha256"] == OLLAMA_MISTRAL_MODEFILE_SHA256
    assert projection["declared_server_context_length"] == OLLAMA_MISTRAL_CONTEXT_LENGTH
    assert projection["sampling_temperature"] == OLLAMA_MISTRAL_TEMPERATURE == 0.15
    assert projection["seed"] == OLLAMA_MISTRAL_SEED == 4242
    assert projection["max_tokens"] == OLLAMA_MISTRAL_ROLE_PROBE_MAX_TOKENS == 1024
    assert projection["prompt_token_difference_rule"] == "nonzero_direction_not_interpreted"
    assert projection["top_p_wire"] == "omitted"
    assert projection["top_p_effective_per_response_observable"] is False


def test_mistral_receipt_accepts_a_signed_nonzero_delta_and_rejects_zero(tmp_path: Path) -> None:
    receipt = _receipt()
    path = tmp_path / "receipt.json"
    path.write_bytes(canonical_json_bytes(receipt))

    assert load_mistral_role_profile_receipt(path)["developer_prompt_token_delta"] == -2
    receipt["developer_prompt_token_delta"] = 0
    with pytest.raises(MistralRoleProfileError):
        validate_mistral_role_profile_receipt(receipt)


def test_mistral_receipt_rejects_cross_profile_and_noncanonical_bytes(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt["role_profile"] = "SSB-OLLAMA-GEMMA4-12B-OAI-CHAT-ROLES3"
    with pytest.raises(MistralRoleProfileError):
        validate_mistral_role_profile_receipt(receipt)

    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_receipt(), indent=2))
    with pytest.raises(MistralRoleProfileError, match="canonical"):
        load_mistral_role_profile_receipt(path)


def test_mistral_commitment_binds_fresh_screen_and_reserved_validation() -> None:
    root = Path(__file__).resolve().parents[3]
    commitment = json.loads(
        (root / "protocol/development_mistral_executor_bundle_v1.json").read_text()
    )
    gemma = json.loads((root / "protocol/development_gemma4_executor_bundle_v1.json").read_text())
    effective_selection = {
        "screen": commitment["selection"]["screen"],
        "validation": gemma["selection"]["validation"],
    }

    assert sha256_ref(commitment) == (
        "sha256:639bddff22aa8ef2ee97793f5f9e0dc2292cd666388e4b37d88551386fc20b64"
    )
    assert sha256_ref(effective_selection) == commitment["selection_hash"]
    assert commitment["selection"]["validation"]["reservation_status"] == (
        "UNEXECUTED_AT_MISTRAL_FREEZE"
    )
    assert commitment["predecessor_evidence"]["gemma4_screen_status"] == "HOLD"
    assert commitment["predecessor_evidence"]["gemma4_task_completion"] == "0/8"
