from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.experiments.ollama_gemma4_profile_v3 import (
    canonical_gemma4_role_profile_receipt as canonical_v3_receipt,
)
from shadowskillbench.experiments.ollama_gemma4_profile_v3 import (
    ollama_gemma4_profile_projection as v3_projection,
)
from shadowskillbench.experiments.ollama_gemma4_profile_v4 import (
    Gemma4RoleProfileError,
    canonical_gemma4_role_profile_receipt,
    load_gemma4_role_profile_receipt,
    ollama_gemma4_profile_projection,
    validate_gemma4_role_profile_receipt,
)


def _call(marker: str, request: str, response: str, input_tokens: int) -> dict[str, object]:
    return {
        "attempts": 1,
        "raw_request_hash": "sha256:" + request * 64,
        "raw_response_hash": "sha256:" + response * 64,
        "response_marker": marker,
        "structured_output_schema_hash": ollama_gemma4_profile_projection()[
            "structured_output_schema_hash"
        ],
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


def test_v4_changes_only_the_temperature_experiment_and_receipt_identity() -> None:
    previous = v3_projection()
    projection = ollama_gemma4_profile_projection()

    assert projection["sampling_temperature"] == 0.0
    assert projection["seed"] == 4242
    assert projection["max_tokens"] == 8192
    assert projection["prompt_token_difference_rule"] == "nonzero_direction_not_interpreted"
    assert projection["top_p_wire"] == "omitted"
    assert projection["role_structured_output_probe"] == previous["role_structured_output_probe"]
    assert projection["structured_output_schema"] == previous["structured_output_schema"]
    assert projection["predecessor_role_profile_receipt_hash"] == (
        "sha256:700273d82a1cda18c06a1608f285c1fad00690ce8b44a39eb5736d166150ad03"
    )
    assert projection["predecessor_screen_audit_receipt_hash"] == (
        "sha256:2e028ae2f74c510641c176c6b3ee80cdb22059b8ff8dee2bfe45630ffb280b19"
    )
    assert "predecessor_failure_receipt_hash" not in projection
    assert projection["single_variable_delta"] == {
        "field": "sampling_temperature",
        "from": 1.0,
        "to": 0.0,
    }


def test_v4_receipts_are_canonical_and_reject_v3_receipts(tmp_path: Path) -> None:
    receipt = _receipt()
    path = tmp_path / "receipt.json"
    path.write_bytes(canonical_json_bytes(receipt))

    assert load_gemma4_role_profile_receipt(path) == receipt
    v3 = json.loads(
        canonical_v3_receipt(
            baseline=_call("SSB_SYSTEM_ROLE_MARKER_7fa38c", "a", "b", 10),
            developer=_call("SSB_DEVELOPER_ROLE_MARKER_81c2ad", "c", "d", 11),
        )
    )
    with pytest.raises(Gemma4RoleProfileError):
        validate_gemma4_role_profile_receipt(v3)

    receipt["developer_prompt_token_delta"] = 0
    with pytest.raises(Gemma4RoleProfileError):
        validate_gemma4_role_profile_receipt(receipt)


def test_greedy_verifier_rebinds_only_the_v4_temperature_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "test_ollama_gemma4_greedy_role_profile_verifier",
        root / "scripts" / "verify_ollama_gemma4_greedy_role_profile.py",
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, verifier)
    spec.loader.exec_module(verifier)

    assert verifier._PASS_PREFIX == "PASS_OLLAMA_GEMMA4_ROLE_PROFILE_V4"
    assert verifier._FAILURE_RECORD_KIND.endswith("RECEIPT4")
    assert verifier.OLLAMA_GEMMA4_TEMPERATURE == 0.0
