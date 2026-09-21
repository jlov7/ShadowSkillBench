from __future__ import annotations

import json
from typing import cast

import pytest

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.experiments.native_tools_conformance_matrix import (
    Candidate,
    MatrixConformanceError,
    canonical_receipt,
    validate_receipt,
)


def _call(marker: str, count: int) -> dict[str, object]:
    return {
        "raw_request_hash": "sha256:" + "1" * 64,
        "raw_response_hash": "sha256:" + "2" * 64,
        "usage": {"input_tokens": count, "output_tokens": 1, "total_tokens": count + 1},
        "attempts": 1,
        "finish_reason": "tool_calls",
        "provider_finish_reason": "stop",
        "provider_done": True,
        "role_marker": marker,
        "tool_name": "finish_task",
        "arguments_hash": sha256_ref({"summary": "native tool conformance"}),
        "tool_declaration_hash": "sha256:" + "3" * 64,
    }


@pytest.mark.parametrize("candidate", ["gpt-oss-20b", "gemma4-12b-it-q4-k-m", "llama3.1-8b"])
def test_matrix_receipt_is_candidate_bound(candidate: str) -> None:
    value = json.loads(
        canonical_receipt(
            cast(Candidate, candidate),
            calls=[_call("baseline", 1), _call("developer", 2), _call("tool_call", 3)],
        )
    )
    assert validate_receipt(value) == value
    value["candidate"] = "llama3.1-8b" if candidate != "llama3.1-8b" else "gpt-oss-20b"
    with pytest.raises(MatrixConformanceError):
        validate_receipt(value)


def test_matrix_zero_delta_is_terminal_invalid() -> None:
    with pytest.raises(MatrixConformanceError):
        canonical_receipt(
            "gpt-oss-20b",
            calls=[_call("baseline", 1), _call("developer", 1), _call("tool_call", 2)],
        )
