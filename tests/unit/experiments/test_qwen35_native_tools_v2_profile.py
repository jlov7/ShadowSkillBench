from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments import ollama_qwen35_native_tools_v2_profile as profile
from shadowskillbench.experiments.ollama_qwen35_native_tools_v2_profile import (
    Qwen35NativeToolsV2ProfileError,
    canonical_qwen35_native_tools_v2_identity,
    canonical_qwen35_native_tools_v2_receipt,
    validate_qwen35_native_tools_v2_identity,
    validate_qwen35_native_tools_v2_receipt,
)

_EXPECTED_MODEL_DIGEST = "sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
_EXPECTED_CONFIG_DIGEST = "sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
_EXPECTED_MODEFILE_DIGEST = (
    "sha256:f3d0707d33423bdec8d4144d22c342e00ef19a1755d7ecfa7786f3f30292ca95"
)
_CAPABILITIES = ["completion", "vision", "tools", "thinking"]
_SYNTHETIC_MODEFILE = (
    "qwen3.5 test fixture\n"
    "FROM sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
)


def _synthetic_show(modelfile: str = _SYNTHETIC_MODEFILE) -> bytes:
    return canonical_json_bytes({"modelfile": modelfile, "capabilities": _CAPABILITIES})


def _patch_synthetic_modelfile_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    # This exercises validation logic; it does not validate the unavailable historical show.
    monkeypatch.setattr(
        profile,
        "OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEFILE_SHA256",
        "sha256:" + hashlib.sha256(_SYNTHETIC_MODEFILE.encode("utf-8")).hexdigest(),
    )


def _call(marker: str, request: str, response: str, input_tokens: int) -> dict[str, object]:
    return {
        "raw_request_hash": "sha256:" + request * 64,
        "raw_response_hash": "sha256:" + response * 64,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": 1,
            "total_tokens": input_tokens + 1,
        },
        "attempts": 1,
        "finish_reason": "tool_calls",
        "provider_finish_reason": "stop",
        "provider_done": True,
        "role_marker": marker,
    }


def test_qwen_receipt_and_identity_are_pinned_and_cross_profile_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = {
        **_call("SSB_NATIVE_TOOL_CALL_MARKER", "5", "6", 3),
        "tool_name": "finish_task",
        "arguments_hash": sha256_ref({"summary": "native tool conformance"}),
        "tool_declaration_hash": "sha256:" + "7" * 64,
    }
    receipt = json.loads(
        canonical_qwen35_native_tools_v2_receipt(
            baseline=_call("SSB_NATIVE_SYSTEM_ROLE_MARKER", "1", "2", 1),
            developer=_call("SSB_NATIVE_DEVELOPER_ROLE_MARKER", "3", "4", 2),
            tool_call=tool,
        )
    )
    assert validate_qwen35_native_tools_v2_receipt(receipt) == receipt
    receipt["record_kind"] = "OLLAMA_QWEN35_NATIVE_TOOLS_CONTEXT32K_ROLE_TOOL_CONFORMANCE_RECEIPT1"
    with pytest.raises(Qwen35NativeToolsV2ProfileError):
        validate_qwen35_native_tools_v2_receipt(receipt)

    assert profile.OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONFIG_BLOB_SHA256 == _EXPECTED_CONFIG_DIGEST
    assert profile.OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEFILE_SHA256 == _EXPECTED_MODEFILE_DIGEST
    with pytest.raises(Qwen35NativeToolsV2ProfileError):
        canonical_qwen35_native_tools_v2_identity(
            show_response=_synthetic_show(), ollama_version_text="0.33.2"
        )

    _patch_synthetic_modelfile_hash(monkeypatch)
    identity = json.loads(
        canonical_qwen35_native_tools_v2_identity(
            show_response=_synthetic_show(), ollama_version_text="0.33.2"
        )
    )
    assert validate_qwen35_native_tools_v2_identity(identity) == identity
    assert identity["model_digest"] == _EXPECTED_MODEL_DIGEST
    assert identity["capabilities"] == _CAPABILITIES
    with pytest.raises(Qwen35NativeToolsV2ProfileError):
        canonical_qwen35_native_tools_v2_identity(
            show_response=_synthetic_show(_SYNTHETIC_MODEFILE + " mismatch"),
            ollama_version_text="0.33.2",
        )


def test_qwen_conformance_manifest_hash_is_frozen() -> None:
    manifest = json.loads(
        Path(
            "protocol/development_qwen35_9b_native_tools_conformance_candidate_v2.json"
        ).read_bytes()
    )
    assert (
        sha256_ref(manifest)
        == "sha256:7976815fe06c2bb06fc3d3df8f180ebd60a50ebf054d45c64a6af70abaa992b7"
    )


def test_roles2_recorder_discloses_exact_summary_in_every_probe_request() -> None:
    source = Path("scripts/record_ollama_qwen35_native_tools_v2_profile.py").read_text()
    assert (
        source.count(
            'Call exactly one finish_task tool with summary exactly "native tool conformance".'
        )
        == 1
    )
    assert source.count("content=prompt") == 2
