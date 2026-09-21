"""Frozen Gemma4 role/structured-output conformance profile V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_gemma4_profile import (
    OLLAMA_GEMMA4_CONTEXT_LENGTH,
    OLLAMA_GEMMA4_MODEFILE_SHA256,
    OLLAMA_GEMMA4_MODEL,
    OLLAMA_GEMMA4_MODEL_DIGEST,
    OLLAMA_GEMMA4_PROVIDER,
    OLLAMA_GEMMA4_SEED,
    OLLAMA_GEMMA4_SERVER_VERSION,
    OLLAMA_GEMMA4_TEMPERATURE,
    Gemma4RoleMarker,
    Gemma4RoleProfileError,
    _call,
    _probe,
    _schema,
)

OLLAMA_GEMMA4_PROFILE = "SSB-OLLAMA-GEMMA4-12B-OAI-CHAT-ROLES2"
OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS = 8192

_REQUEST_PROFILE = "SSB-OAI-CHAT1"
_ENDPOINT_PATH = "/v1/chat/completions"
_SYSTEM_MARKER = "SSB_SYSTEM_ROLE_MARKER_7fa38c"
_DEVELOPER_MARKER = "SSB_DEVELOPER_ROLE_MARKER_81c2ad"


def ollama_gemma4_profile_projection() -> dict[str, object]:
    """Return the complete frozen V2 projection."""

    schema = _schema()
    return {
        "role_profile": OLLAMA_GEMMA4_PROFILE,
        "provider": OLLAMA_GEMMA4_PROVIDER,
        "model": OLLAMA_GEMMA4_MODEL,
        "model_digest": OLLAMA_GEMMA4_MODEL_DIGEST,
        "ollama_server_version": OLLAMA_GEMMA4_SERVER_VERSION,
        "modelfile_sha256": OLLAMA_GEMMA4_MODEFILE_SHA256,
        "declared_server_context_length": OLLAMA_GEMMA4_CONTEXT_LENGTH,
        "server_context_length_binding": "declared_operator_bound",
        "server_context_length_per_response_observable": False,
        "request_profile": _REQUEST_PROFILE,
        "endpoint_path": _ENDPOINT_PATH,
        "sampling_temperature": OLLAMA_GEMMA4_TEMPERATURE,
        "seed": OLLAMA_GEMMA4_SEED,
        "max_tokens": OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS,
        "reasoning_effort_wire": "omitted",
        "top_p_wire": "omitted",
        "top_p_effective_observed_default": 0.95,
        "top_p_effective_per_response_observable": False,
        "structured_output_schema": schema,
        "structured_output_schema_hash": sha256_ref(schema),
        "role_structured_output_probe": _probe(),
        "predecessor_role_profile": "SSB-OLLAMA-GEMMA4-12B-OAI-CHAT-ROLES1",
        "predecessor_failure_receipt_hash": (
            "sha256:5c5ed140d44d1b2a6ae28ca8d5f708708b4fe0af74a3517fb116c73deb3d6d12"
        ),
        "single_variable_delta": {
            "field": "max_tokens",
            "from": 128,
            "to": OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS,
        },
    }


def validate_gemma4_role_profile_receipt(value: object) -> dict[str, object]:
    """Reject stale, tampered, noncanonical, or nonconformant V2 receipts."""

    if type(value) is not dict:
        raise Gemma4RoleProfileError("Gemma4 V2 role-profile receipt is invalid")
    projection = ollama_gemma4_profile_projection()
    baseline = value.get("baseline")
    developer = value.get("developer")
    delta = value.get("developer_prompt_token_delta")
    expected = {
        **projection,
        "record_kind": "OLLAMA_GEMMA4_ROLE_STRUCTURED_OUTPUT_RECEIPT2",
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
        or delta <= 0
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
        raise Gemma4RoleProfileError("Gemma4 V2 role-profile receipt is not bound to this profile")
    return cast(dict[str, object], value)


def canonical_gemma4_role_profile_receipt(
    *, baseline: dict[str, object], developer: dict[str, object]
) -> bytes:
    """Create a canonical V2 receipt after exactly two successful calls."""

    baseline_usage = baseline.get("usage") if type(baseline) is dict else None
    developer_usage = developer.get("usage") if type(developer) is dict else None
    value = {
        **ollama_gemma4_profile_projection(),
        "record_kind": "OLLAMA_GEMMA4_ROLE_STRUCTURED_OUTPUT_RECEIPT2",
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
    """Load only an exact canonical V2 receipt from a regular path."""

    if not isinstance(path, Path) or path.is_symlink():
        raise Gemma4RoleProfileError("Gemma4 V2 role-profile receipt is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma4RoleProfileError("Gemma4 V2 role-profile receipt is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise Gemma4RoleProfileError("Gemma4 V2 role-profile receipt is not canonical")
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
