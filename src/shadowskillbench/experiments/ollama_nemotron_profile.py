"""Frozen Nemotron role-separation and structured-output conformance profile."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_gemma4_profile import (
    Gemma4RoleMarker as NemotronRoleMarker,
)
from shadowskillbench.experiments.ollama_gemma4_profile import _call, _probe, _schema

OLLAMA_NEMOTRON_PROFILE = "SSB-OLLAMA-NEMOTRON35-LIGHTNING-30B-MLX-OAI-CHAT-ROLES1"
OLLAMA_NEMOTRON_PROVIDER = "ollama-nemotron35-lightning-30b-mlx"
OLLAMA_NEMOTRON_MODEL = "nemotron-3.5-lightning:30b-mlx"
OLLAMA_NEMOTRON_MODEL_DIGEST = (
    "sha256:8b1474be6e54dc19eb7aa08bebfb9bda147c4b9ef9796a726131ad29ad15645a"
)
OLLAMA_NEMOTRON_SERVER_VERSION = "0.33.2"
OLLAMA_NEMOTRON_CONFIG_BLOB_SHA256 = (
    "sha256:2070f1ef6337efdcec267ce890140ee11da89e53d808eb673b4128be166ef188"
)
OLLAMA_NEMOTRON_MODEFILE_SHA256 = (
    "sha256:1fe8708c629c26e16c6e05d46ac07dad37b7c2124416eaf77e077295c5c64663"
)
OLLAMA_NEMOTRON_CONTEXT_LENGTH = 131_072
OLLAMA_NEMOTRON_TEMPERATURE = 0.15
OLLAMA_NEMOTRON_SEED = 4242
OLLAMA_NEMOTRON_ROLE_PROBE_MAX_TOKENS = 8192

_SYSTEM_MARKER = "SSB_SYSTEM_ROLE_MARKER_7fa38c"
_DEVELOPER_MARKER = "SSB_DEVELOPER_ROLE_MARKER_81c2ad"


class NemotronRoleProfileError(ValueError):
    """The Nemotron role-profile receipt is unavailable or not pinned."""


def ollama_nemotron_profile_projection() -> dict[str, object]:
    """Return the frozen OpenAI-chat role and structured-output configuration."""

    schema = _schema()
    return {
        "role_profile": OLLAMA_NEMOTRON_PROFILE,
        "provider": OLLAMA_NEMOTRON_PROVIDER,
        "model": OLLAMA_NEMOTRON_MODEL,
        "model_digest": OLLAMA_NEMOTRON_MODEL_DIGEST,
        "ollama_server_version": OLLAMA_NEMOTRON_SERVER_VERSION,
        "config_blob_sha256": OLLAMA_NEMOTRON_CONFIG_BLOB_SHA256,
        "modelfile_sha256": OLLAMA_NEMOTRON_MODEFILE_SHA256,
        "declared_server_context_length": OLLAMA_NEMOTRON_CONTEXT_LENGTH,
        "server_context_length_binding": "declared_operator_bound",
        "server_context_length_per_response_observable": False,
        "request_profile": "SSB-OAI-CHAT1",
        "endpoint_path": "/v1/chat/completions",
        "sampling_temperature": OLLAMA_NEMOTRON_TEMPERATURE,
        "seed": OLLAMA_NEMOTRON_SEED,
        "max_tokens": OLLAMA_NEMOTRON_ROLE_PROBE_MAX_TOKENS,
        "reasoning_effort_wire": "omitted",
        "top_p_wire": "omitted",
        "top_p_effective_per_response_observable": False,
        "structured_output_schema": schema,
        "structured_output_schema_hash": sha256_ref(schema),
        "role_structured_output_probe": _probe(),
        "prompt_token_difference_rule": "nonzero_direction_not_interpreted",
    }


def nemotron_live_show_identity_projection() -> dict[str, object]:
    """Return the later-live-show fields required before a real inference run."""

    return {
        "model": OLLAMA_NEMOTRON_MODEL,
        "model_digest": OLLAMA_NEMOTRON_MODEL_DIGEST,
        "config_blob_sha256": OLLAMA_NEMOTRON_CONFIG_BLOB_SHA256,
        "modelfile_sha256": OLLAMA_NEMOTRON_MODEFILE_SHA256,
        "ollama_server_version": OLLAMA_NEMOTRON_SERVER_VERSION,
        "context_length": OLLAMA_NEMOTRON_CONTEXT_LENGTH,
        "renderer": "nemotron-3.5-nano",
        "parser": "nemotron-3.5-nano",
        "capabilities": ["completion", "tools", "thinking"],
    }


def validate_nemotron_live_show_identity(value: object) -> dict[str, object]:
    """Validate a hashed, no-raw-content live identity receipt before inference."""

    if type(value) is not dict:
        raise NemotronRoleProfileError("Nemotron live-show identity receipt is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": "OLLAMA_NEMOTRON_LIVE_SHOW_IDENTITY_RECEIPT1",
        **nemotron_live_show_identity_projection(),
        "show_response_hash": value.get("show_response_hash"),
    }
    if (
        body != expected
        or type(value.get("show_response_hash")) is not str
        or not cast(str, value["show_response_hash"]).startswith("sha256:")
        or value.get("receipt_hash") != sha256_ref(body)
    ):
        raise NemotronRoleProfileError("Nemotron live-show identity receipt is not bound")
    return cast(dict[str, object], value)


def canonical_nemotron_live_show_identity_receipt(
    *, show_response: bytes, ollama_version_text: str
) -> bytes:
    """Build a hash-only receipt from operator-captured ``ollama show --json`` bytes."""

    try:
        show = json.loads(show_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NemotronRoleProfileError("Nemotron live-show response is not JSON") from error
    if type(show) is not dict:
        raise NemotronRoleProfileError("Nemotron live-show response has an invalid shape")
    if show.get("capabilities") != ["completion", "tools", "thinking"]:
        raise NemotronRoleProfileError("Nemotron live-show capabilities are not pinned")
    version = re.search(r"\b0\.33\.2\b", ollama_version_text)
    if version is None:
        raise NemotronRoleProfileError("Nemotron Ollama version is not pinned")
    body = {
        "record_kind": "OLLAMA_NEMOTRON_LIVE_SHOW_IDENTITY_RECEIPT1",
        **nemotron_live_show_identity_projection(),
        "show_response_hash": "sha256:" + sha256(show_response).hexdigest(),
    }
    value = {**body, "receipt_hash": sha256_ref(body)}
    validate_nemotron_live_show_identity(value)
    return canonical_json_bytes(value)


def load_nemotron_live_show_identity(path: Path) -> dict[str, object]:
    """Load a canonical hashed live-show identity receipt without provider content."""

    if not isinstance(path, Path) or path.is_symlink():
        raise NemotronRoleProfileError("Nemotron live-show identity receipt is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NemotronRoleProfileError(
            "Nemotron live-show identity receipt is unreadable"
        ) from error
    if canonical_json_bytes(value) != raw:
        raise NemotronRoleProfileError("Nemotron live-show identity receipt is not canonical")
    return validate_nemotron_live_show_identity(value)


def validate_nemotron_role_profile_receipt(value: object) -> dict[str, object]:
    """Reject stale, tampered, noncanonical, or nonconformant receipts."""

    if type(value) is not dict:
        raise NemotronRoleProfileError("Nemotron role-profile receipt is invalid")
    projection = ollama_nemotron_profile_projection()
    baseline = value.get("baseline")
    developer = value.get("developer")
    delta = value.get("developer_prompt_token_delta")
    expected = {
        **projection,
        "record_kind": "OLLAMA_NEMOTRON_ROLE_STRUCTURED_OUTPUT_RECEIPT1",
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
        raise NemotronRoleProfileError("Nemotron role-profile receipt is not bound to this profile")
    return cast(dict[str, object], value)


def canonical_nemotron_role_profile_receipt(
    *, baseline: dict[str, object], developer: dict[str, object]
) -> bytes:
    """Create a canonical receipt after exactly two successful calls."""

    baseline_usage = baseline.get("usage") if type(baseline) is dict else None
    developer_usage = developer.get("usage") if type(developer) is dict else None
    value = {
        **ollama_nemotron_profile_projection(),
        "record_kind": "OLLAMA_NEMOTRON_ROLE_STRUCTURED_OUTPUT_RECEIPT1",
        "baseline": baseline,
        "developer": developer,
        "developer_prompt_token_delta": (
            cast(int, developer_usage["input_tokens"]) - cast(int, baseline_usage["input_tokens"])
            if type(baseline_usage) is dict and type(developer_usage) is dict
            else None
        ),
    }
    validate_nemotron_role_profile_receipt(value)
    return canonical_json_bytes(value)


def load_nemotron_role_profile_receipt(path: Path) -> dict[str, object]:
    """Load only an exact canonical receipt from a regular path."""

    if not isinstance(path, Path) or path.is_symlink():
        raise NemotronRoleProfileError("Nemotron role-profile receipt is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NemotronRoleProfileError("Nemotron role-profile receipt is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise NemotronRoleProfileError("Nemotron role-profile receipt is not canonical")
    return validate_nemotron_role_profile_receipt(value)


__all__ = [
    "OLLAMA_NEMOTRON_CONFIG_BLOB_SHA256",
    "OLLAMA_NEMOTRON_CONTEXT_LENGTH",
    "OLLAMA_NEMOTRON_MODEFILE_SHA256",
    "OLLAMA_NEMOTRON_MODEL",
    "OLLAMA_NEMOTRON_MODEL_DIGEST",
    "OLLAMA_NEMOTRON_PROFILE",
    "OLLAMA_NEMOTRON_PROVIDER",
    "OLLAMA_NEMOTRON_ROLE_PROBE_MAX_TOKENS",
    "OLLAMA_NEMOTRON_SEED",
    "OLLAMA_NEMOTRON_SERVER_VERSION",
    "OLLAMA_NEMOTRON_TEMPERATURE",
    "NemotronRoleMarker",
    "NemotronRoleProfileError",
    "canonical_nemotron_role_profile_receipt",
    "canonical_nemotron_live_show_identity_receipt",
    "load_nemotron_role_profile_receipt",
    "load_nemotron_live_show_identity",
    "nemotron_live_show_identity_projection",
    "ollama_nemotron_profile_projection",
    "validate_nemotron_live_show_identity",
    "validate_nemotron_role_profile_receipt",
]
