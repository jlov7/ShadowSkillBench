from __future__ import annotations

import json

import pytest

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.experiments.ollama_glm47_native_tools_response_v2_profile import (
    Glm47NativeToolsResponseV2Error,
    canonical_glm47_native_tools_response_v2_receipt,
    validate_glm47_native_tools_response_v2_receipt,
)


def _call(marker: str) -> dict[str, object]:
    return {
        "raw_request_hash": "sha256:" + "1" * 64,
        "raw_response_hash": "sha256:" + "2" * 64,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "attempts": 1,
        "finish_reason": "tool_calls",
        "provider_finish_reason": None,
        "provider_done": True,
        "role_marker": marker,
    }


def _developer_call() -> dict[str, object]:
    value = _call("SSB_NATIVE_DEVELOPER_ROLE_MARKER")
    value["usage"] = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    return value


def test_roles2_receipt_binds_new_contract_and_rejects_roles1() -> None:
    tool = {
        **_call("SSB_NATIVE_TOOL_CALL_MARKER"),
        "tool_name": "finish_task",
        "arguments_hash": sha256_ref({"summary": "native tool conformance"}),
        "tool_declaration_hash": "sha256:" + "3" * 64,
    }
    receipt = json.loads(
        canonical_glm47_native_tools_response_v2_receipt(
            baseline=_call("SSB_NATIVE_SYSTEM_ROLE_MARKER"),
            developer=_developer_call(),
            tool_call=tool,
        )
    )
    assert validate_glm47_native_tools_response_v2_receipt(receipt) == receipt
    receipt["record_kind"] = "OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT32K_ROLE_TOOL_CONFORMANCE_RECEIPT1"
    with pytest.raises(Glm47NativeToolsResponseV2Error):
        validate_glm47_native_tools_response_v2_receipt(receipt)
