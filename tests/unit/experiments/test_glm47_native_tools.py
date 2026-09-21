from __future__ import annotations

import json

import pytest

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.corpus.development import generate_development_corpus
from shadowskillbench.experiments import capability_pilot
from shadowskillbench.experiments.ollama_glm47_native_tools_profile import (
    Glm47NativeToolsRoleProfileError,
    canonical_glm47_native_tools_role_profile_receipt,
    glm47_native_tools_live_show_identity_projection,
    ollama_glm47_native_tools_profile_projection,
    validate_glm47_native_tools_live_show_identity,
    validate_glm47_native_tools_role_profile_receipt,
)
from shadowskillbench.experiments.ollama_nemotron_native_tools_profile import (
    canonical_nemotron_native_tools_role_profile_failure_receipt,
)


def _call(request: str, response: str, marker: str, tokens: int) -> dict[str, object]:
    return {
        "raw_request_hash": "sha256:" + request * 64,
        "raw_response_hash": "sha256:" + response * 64,
        "usage": {"input_tokens": tokens, "output_tokens": 2, "total_tokens": tokens + 2},
        "attempts": 1,
        "finish_reason": "tool_calls",
        "role_marker": marker,
    }


def _receipt() -> dict[str, object]:
    return json.loads(
        canonical_glm47_native_tools_role_profile_receipt(
            baseline=_call("1", "2", "SSB_NATIVE_SYSTEM_ROLE_MARKER", 10),
            developer=_call("3", "4", "SSB_NATIVE_DEVELOPER_ROLE_MARKER", 11),
            tool_call={
                **_call("5", "6", "SSB_NATIVE_TOOL_CALL_MARKER", 12),
                "tool_name": "finish_task",
                "arguments_hash": "sha256:" + "7" * 64,
                "tool_declaration_hash": "sha256:" + "8" * 64,
            },
        )
    )


def test_glm_profile_pins_identity_and_rejects_cross_profile() -> None:
    receipt = _receipt()
    assert validate_glm47_native_tools_role_profile_receipt(receipt) == receipt
    assert receipt["model"] == "glm-4.7-flash:q4_K_M"
    assert receipt["context_length"] == 32768
    identity_body = {
        "record_kind": "OLLAMA_GLM47_CONTEXT32K_LIVE_SHOW_IDENTITY_RECEIPT1",
        **glm47_native_tools_live_show_identity_projection(),
        "show_response_hash": "sha256:" + "9" * 64,
    }
    identity = {**identity_body, "receipt_hash": sha256_ref(identity_body)}
    assert validate_glm47_native_tools_live_show_identity(identity) == identity
    with pytest.raises(Glm47NativeToolsRoleProfileError):
        validate_glm47_native_tools_role_profile_receipt(
            json.loads(
                canonical_nemotron_native_tools_role_profile_failure_receipt(
                    failure_code="MODEL_OUTPUT_INVALID", completed_calls=0, failed_call=None
                )
            )
        )


def test_glm_selection_is_fresh_and_commitment_bound() -> None:
    corpus = generate_development_corpus(4247)
    screen = capability_pilot.select_capability_cells(
        corpus, phase="screen", bundle="glm47-native-tools-context32k"
    )
    validation = capability_pilot.select_capability_cells(
        corpus, phase="validation", bundle="glm47-native-tools-context32k"
    )
    commitment = capability_pilot._grounding_commitment("glm47-native-tools-context32k")
    assert len(screen) == 8
    assert len(validation) == 40
    assert commitment["selection_hash"] == sha256_ref(commitment["selection"])
    assert {cell.case_id for cell in screen}.isdisjoint(
        {"development_d6b479ccd34e7720", "development_ca0556a5125691b8"}
    )
    assert ollama_glm47_native_tools_profile_projection()["modelfile_sha256"].startswith("sha256:")
