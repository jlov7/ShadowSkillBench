"""Frozen Gemma4 role/structured-output conformance profile V3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.experiments import ollama_gemma4_profile_v2 as _v2
from shadowskillbench.experiments.ollama_gemma4_profile import (
    Gemma4RoleMarker,
    Gemma4RoleProfileError,
    _call,
)

OLLAMA_GEMMA4_PROFILE = "SSB-OLLAMA-GEMMA4-12B-OAI-CHAT-ROLES3"
OLLAMA_GEMMA4_CONTEXT_LENGTH = _v2.OLLAMA_GEMMA4_CONTEXT_LENGTH
OLLAMA_GEMMA4_MODEFILE_SHA256 = _v2.OLLAMA_GEMMA4_MODEFILE_SHA256
OLLAMA_GEMMA4_MODEL = _v2.OLLAMA_GEMMA4_MODEL
OLLAMA_GEMMA4_MODEL_DIGEST = _v2.OLLAMA_GEMMA4_MODEL_DIGEST
OLLAMA_GEMMA4_PROVIDER = _v2.OLLAMA_GEMMA4_PROVIDER
OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS = _v2.OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS
OLLAMA_GEMMA4_SEED = _v2.OLLAMA_GEMMA4_SEED
OLLAMA_GEMMA4_SERVER_VERSION = _v2.OLLAMA_GEMMA4_SERVER_VERSION
OLLAMA_GEMMA4_TEMPERATURE = _v2.OLLAMA_GEMMA4_TEMPERATURE

_SYSTEM_MARKER = "SSB_SYSTEM_ROLE_MARKER_7fa38c"
_DEVELOPER_MARKER = "SSB_DEVELOPER_ROLE_MARKER_81c2ad"


def ollama_gemma4_profile_projection() -> dict[str, object]:
    """Return the complete frozen V3 projection."""

    projection = _v2.ollama_gemma4_profile_projection()
    projection.pop("predecessor_role_profile")
    projection.pop("predecessor_failure_receipt_hash")
    projection.pop("single_variable_delta")
    projection.update(
        {
            "role_profile": OLLAMA_GEMMA4_PROFILE,
            "prompt_token_difference_rule": "nonzero_direction_not_interpreted",
            "predecessor_role_profile": "SSB-OLLAMA-GEMMA4-12B-OAI-CHAT-ROLES2",
            "predecessor_failure_receipt_hash": (
                "sha256:0a32dc9d83b87fa086c2f35a4a60ed58ff3a18edc7117a1c12b0ef1be883eb52"
            ),
            "single_variable_delta": {
                "field": "prompt_token_difference_rule",
                "from": "strictly_positive",
                "to": "nonzero_direction_not_interpreted",
            },
        }
    )
    return projection


def validate_gemma4_role_profile_receipt(value: object) -> dict[str, object]:
    """Reject stale, tampered, noncanonical, or nonconformant V3 receipts."""

    if type(value) is not dict:
        raise Gemma4RoleProfileError("Gemma4 V3 role-profile receipt is invalid")
    projection = ollama_gemma4_profile_projection()
    baseline = value.get("baseline")
    developer = value.get("developer")
    delta = value.get("developer_prompt_token_delta")
    expected = {
        **projection,
        "record_kind": "OLLAMA_GEMMA4_ROLE_STRUCTURED_OUTPUT_RECEIPT3",
        "baseline": baseline,
        "developer": developer,
        "developer_prompt_token_delta": delta,
    }
    schema_hash = cast(str, projection["structured_output_schema_hash"])
    if (
        set(value) != set(expected)
        or value != expected
        or not _call(baseline, _SYSTEM_MARKER, schema_hash)
        or not _call(developer, _DEVELOPER_MARKER, schema_hash)
        or type(delta) is not int
        or delta == 0
        or cast(dict[str, object], baseline)["raw_request_hash"]
        == cast(dict[str, object], developer)["raw_request_hash"]
        or cast(dict[str, object], baseline)["raw_response_hash"]
        == cast(dict[str, object], developer)["raw_response_hash"]
        or cast(
            int,
            cast(dict[str, object], cast(dict[str, object], developer)["usage"])["input_tokens"],
        )
        != cast(
            int,
            cast(dict[str, object], cast(dict[str, object], baseline)["usage"])["input_tokens"],
        )
        + delta
    ):
        raise Gemma4RoleProfileError("Gemma4 V3 role-profile receipt is not bound to this profile")
    return cast(dict[str, object], value)


def canonical_gemma4_role_profile_receipt(
    *, baseline: dict[str, object], developer: dict[str, object]
) -> bytes:
    """Create a canonical V3 receipt after exactly two successful calls."""

    baseline_usage = baseline.get("usage") if type(baseline) is dict else None
    developer_usage = developer.get("usage") if type(developer) is dict else None
    value = {
        **ollama_gemma4_profile_projection(),
        "record_kind": "OLLAMA_GEMMA4_ROLE_STRUCTURED_OUTPUT_RECEIPT3",
        "baseline": baseline,
        "developer": developer,
        "developer_prompt_token_delta": (
            cast(int, developer_usage["input_tokens"]) - cast(int, baseline_usage["input_tokens"])
            if type(baseline_usage) is dict and type(developer_usage) is dict
            else None
        ),
    }
    validate_gemma4_role_profile_receipt(value)
    return canonical_json_bytes(value)


def load_gemma4_role_profile_receipt(path: Path) -> dict[str, object]:
    """Load only an exact canonical V3 receipt from a regular path."""

    if not isinstance(path, Path) or path.is_symlink():
        raise Gemma4RoleProfileError("Gemma4 V3 role-profile receipt is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma4RoleProfileError("Gemma4 V3 role-profile receipt is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise Gemma4RoleProfileError("Gemma4 V3 role-profile receipt is not canonical")
    return validate_gemma4_role_profile_receipt(value)


__all__ = [
    "OLLAMA_GEMMA4_PROFILE",
    "OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS",
    "Gemma4RoleMarker",
    "Gemma4RoleProfileError",
    "ollama_gemma4_profile_projection",
    "canonical_gemma4_role_profile_receipt",
    "validate_gemma4_role_profile_receipt",
    "load_gemma4_role_profile_receipt",
]
