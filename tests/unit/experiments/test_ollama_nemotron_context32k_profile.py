from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_nemotron_context32k_profile import (
    OLLAMA_NEMOTRON_CONTEXT32K_CONTEXT_LENGTH,
    OLLAMA_NEMOTRON_CONTEXT32K_PROFILE,
    NemotronContext32kRoleProfileError,
    canonical_nemotron_context32k_role_profile_receipt,
    load_nemotron_context32k_role_profile_receipt,
    ollama_nemotron_context32k_profile_projection,
    validate_nemotron_context32k_live_show_identity,
    validate_nemotron_context32k_role_profile_receipt,
)
from shadowskillbench.experiments.ollama_nemotron_profile import (
    canonical_nemotron_role_profile_receipt,
)


def _call(marker: str, request: str, response: str, input_tokens: int) -> dict[str, object]:
    projection = ollama_nemotron_context32k_profile_projection()
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
        canonical_nemotron_context32k_role_profile_receipt(
            baseline=_call("SSB_SYSTEM_ROLE_MARKER_7fa38c", "a", "b", 20),
            developer=_call("SSB_DEVELOPER_ROLE_MARKER_81c2ad", "c", "d", 18),
        )
    )


def test_context32k_projection_and_receipts_are_distinct_from_v1(tmp_path: Path) -> None:
    projection = ollama_nemotron_context32k_profile_projection()
    receipt = _receipt()
    path = tmp_path / "receipt.json"
    path.write_bytes(canonical_json_bytes(receipt))

    assert projection["role_profile"] == OLLAMA_NEMOTRON_CONTEXT32K_PROFILE
    assert (
        projection["declared_server_context_length"]
        == OLLAMA_NEMOTRON_CONTEXT32K_CONTEXT_LENGTH
        == 32768
    )
    assert receipt["record_kind"] == "OLLAMA_NEMOTRON_CONTEXT32K_ROLE_STRUCTURED_OUTPUT_RECEIPT2"
    assert load_nemotron_context32k_role_profile_receipt(path) == receipt

    old_receipt = json.loads(
        canonical_nemotron_role_profile_receipt(
            baseline=_call("SSB_SYSTEM_ROLE_MARKER_7fa38c", "a", "b", 20),
            developer=_call("SSB_DEVELOPER_ROLE_MARKER_81c2ad", "c", "d", 18),
        )
    )
    with pytest.raises(NemotronContext32kRoleProfileError):
        validate_nemotron_context32k_role_profile_receipt(old_receipt)


def test_context32k_identity_rejects_v1_record_kind_and_context() -> None:
    body = {
        "record_kind": "OLLAMA_NEMOTRON_LIVE_SHOW_IDENTITY_RECEIPT1",
        "context_length": 131072,
        "show_response_hash": "sha256:" + "0" * 64,
    }
    with pytest.raises(NemotronContext32kRoleProfileError):
        validate_nemotron_context32k_live_show_identity({**body, "receipt_hash": sha256_ref(body)})
