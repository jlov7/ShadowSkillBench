"""Frozen Mistral Small 3.2 role/structured-output conformance profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_gemma4_profile import (
    Gemma4RoleMarker as MistralRoleMarker,
)
from shadowskillbench.experiments.ollama_gemma4_profile import (
    _call,
    _probe,
    _schema,
)

OLLAMA_MISTRAL_PROFILE = "SSB-OLLAMA-MISTRAL-SMALL3.2-24B-OAI-CHAT-ROLES1"
OLLAMA_MISTRAL_PROVIDER = "ollama-mistral-small3.2-24b"
OLLAMA_MISTRAL_MODEL = "mistral-small3.2:24b-instruct-2506-q4_K_M"
OLLAMA_MISTRAL_MODEL_DIGEST = (
    "sha256:5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b"
)
OLLAMA_MISTRAL_SERVER_VERSION = "0.33.2"
OLLAMA_MISTRAL_MODEFILE_SHA256 = (
    "sha256:93f13e1eed5e0b5b7709796da2ea55db4aeaf0cdfc0e290f45e922a50c4a1301"
)
OLLAMA_MISTRAL_CONTEXT_LENGTH = 131_072
OLLAMA_MISTRAL_TEMPERATURE = 0.15
OLLAMA_MISTRAL_SEED = 4242
OLLAMA_MISTRAL_ROLE_PROBE_MAX_TOKENS = 1024

_SYSTEM_MARKER = "SSB_SYSTEM_ROLE_MARKER_7fa38c"
_DEVELOPER_MARKER = "SSB_DEVELOPER_ROLE_MARKER_81c2ad"


class MistralRoleProfileError(ValueError):
    """The Mistral role-profile receipt is unavailable or not pinned."""


def ollama_mistral_profile_projection() -> dict[str, object]:
    """Return the complete public, frozen Mistral configuration."""

    schema = _schema()
    return {
        "role_profile": OLLAMA_MISTRAL_PROFILE,
        "provider": OLLAMA_MISTRAL_PROVIDER,
        "model": OLLAMA_MISTRAL_MODEL,
        "model_digest": OLLAMA_MISTRAL_MODEL_DIGEST,
        "ollama_server_version": OLLAMA_MISTRAL_SERVER_VERSION,
        "modelfile_sha256": OLLAMA_MISTRAL_MODEFILE_SHA256,
        "declared_server_context_length": OLLAMA_MISTRAL_CONTEXT_LENGTH,
        "server_context_length_binding": "declared_operator_bound",
        "server_context_length_per_response_observable": False,
        "request_profile": "SSB-OAI-CHAT1",
        "endpoint_path": "/v1/chat/completions",
        "sampling_temperature": OLLAMA_MISTRAL_TEMPERATURE,
        "seed": OLLAMA_MISTRAL_SEED,
        "max_tokens": OLLAMA_MISTRAL_ROLE_PROBE_MAX_TOKENS,
        "reasoning_effort_wire": "omitted",
        "top_p_wire": "omitted",
        "top_p_effective_per_response_observable": False,
        "structured_output_schema": schema,
        "structured_output_schema_hash": sha256_ref(schema),
        "role_structured_output_probe": _probe(),
        "prompt_token_difference_rule": "nonzero_direction_not_interpreted",
    }


def validate_mistral_role_profile_receipt(value: object) -> dict[str, object]:
    """Reject stale, tampered, noncanonical, or nonconformant Mistral receipts."""

    if type(value) is not dict:
        raise MistralRoleProfileError("Mistral role-profile receipt is invalid")
    projection = ollama_mistral_profile_projection()
    baseline = value.get("baseline")
    developer = value.get("developer")
    delta = value.get("developer_prompt_token_delta")
    expected = {
        **projection,
        "record_kind": "OLLAMA_MISTRAL_ROLE_STRUCTURED_OUTPUT_RECEIPT1",
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
        raise MistralRoleProfileError("Mistral role-profile receipt is not bound to this profile")
    return cast(dict[str, object], value)


def canonical_mistral_role_profile_receipt(
    *, baseline: dict[str, object], developer: dict[str, object]
) -> bytes:
    """Create a canonical receipt after exactly two successful calls."""

    baseline_usage = baseline.get("usage") if type(baseline) is dict else None
    developer_usage = developer.get("usage") if type(developer) is dict else None
    value = {
        **ollama_mistral_profile_projection(),
        "record_kind": "OLLAMA_MISTRAL_ROLE_STRUCTURED_OUTPUT_RECEIPT1",
        "baseline": baseline,
        "developer": developer,
        "developer_prompt_token_delta": (
            cast(int, developer_usage["input_tokens"]) - cast(int, baseline_usage["input_tokens"])
            if type(baseline_usage) is dict and type(developer_usage) is dict
            else None
        ),
    }
    validate_mistral_role_profile_receipt(value)
    return canonical_json_bytes(value)


def load_mistral_role_profile_receipt(path: Path) -> dict[str, object]:
    """Load only an exact canonical receipt from a regular path."""

    if not isinstance(path, Path) or path.is_symlink():
        raise MistralRoleProfileError("Mistral role-profile receipt is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MistralRoleProfileError("Mistral role-profile receipt is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise MistralRoleProfileError("Mistral role-profile receipt is not canonical")
    return validate_mistral_role_profile_receipt(value)


__all__ = [
    "OLLAMA_MISTRAL_PROFILE",
    "OLLAMA_MISTRAL_PROVIDER",
    "OLLAMA_MISTRAL_MODEL",
    "OLLAMA_MISTRAL_MODEL_DIGEST",
    "OLLAMA_MISTRAL_SERVER_VERSION",
    "OLLAMA_MISTRAL_MODEFILE_SHA256",
    "OLLAMA_MISTRAL_CONTEXT_LENGTH",
    "OLLAMA_MISTRAL_TEMPERATURE",
    "OLLAMA_MISTRAL_SEED",
    "OLLAMA_MISTRAL_ROLE_PROBE_MAX_TOKENS",
    "MistralRoleMarker",
    "MistralRoleProfileError",
    "ollama_mistral_profile_projection",
    "canonical_mistral_role_profile_receipt",
    "validate_mistral_role_profile_receipt",
    "load_mistral_role_profile_receipt",
]
