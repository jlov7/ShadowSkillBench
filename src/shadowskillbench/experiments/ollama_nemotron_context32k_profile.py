"""Frozen 32k Nemotron role and identity profile for the prospective successor."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_gemma4_profile import (
    Gemma4RoleMarker as NemotronContext32kRoleMarker,
)
from shadowskillbench.experiments.ollama_gemma4_profile import _call, _probe, _schema

OLLAMA_NEMOTRON_CONTEXT32K_PROFILE = (
    "SSB-OLLAMA-NEMOTRON35-LIGHTNING-30B-MLX-OAI-CHAT-CONTEXT32768-ROLES2"
)
OLLAMA_NEMOTRON_CONTEXT32K_PROVIDER = "ollama-nemotron35-lightning-30b-mlx"
OLLAMA_NEMOTRON_CONTEXT32K_MODEL = "nemotron-3.5-lightning:30b-mlx"
OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST = (
    "sha256:8b1474be6e54dc19eb7aa08bebfb9bda147c4b9ef9796a726131ad29ad15645a"
)
OLLAMA_NEMOTRON_CONTEXT32K_SERVER_VERSION = "0.33.2"
OLLAMA_NEMOTRON_CONTEXT32K_CONFIG_BLOB_SHA256 = (
    "sha256:2070f1ef6337efdcec267ce890140ee11da89e53d808eb673b4128be166ef188"
)
OLLAMA_NEMOTRON_CONTEXT32K_MODEFILE_SHA256 = (
    "sha256:1fe8708c629c26e16c6e05d46ac07dad37b7c2124416eaf77e077295c5c64663"
)
OLLAMA_NEMOTRON_CONTEXT32K_CONTEXT_LENGTH = 32_768
OLLAMA_NEMOTRON_CONTEXT32K_TEMPERATURE = 0.15
OLLAMA_NEMOTRON_CONTEXT32K_SEED = 4242
OLLAMA_NEMOTRON_CONTEXT32K_ROLE_PROBE_MAX_TOKENS = 8192

_SYSTEM_MARKER = "SSB_SYSTEM_ROLE_MARKER_7fa38c"
_DEVELOPER_MARKER = "SSB_DEVELOPER_ROLE_MARKER_81c2ad"


class NemotronContext32kRoleProfileError(ValueError):
    """The context-32k Nemotron profile receipt is unavailable or not pinned."""


def ollama_nemotron_context32k_profile_projection() -> dict[str, object]:
    """Return the frozen OpenAI-chat profile, including its 32k context binding."""

    schema = _schema()
    return {
        "role_profile": OLLAMA_NEMOTRON_CONTEXT32K_PROFILE,
        "provider": OLLAMA_NEMOTRON_CONTEXT32K_PROVIDER,
        "model": OLLAMA_NEMOTRON_CONTEXT32K_MODEL,
        "model_digest": OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST,
        "ollama_server_version": OLLAMA_NEMOTRON_CONTEXT32K_SERVER_VERSION,
        "config_blob_sha256": OLLAMA_NEMOTRON_CONTEXT32K_CONFIG_BLOB_SHA256,
        "modelfile_sha256": OLLAMA_NEMOTRON_CONTEXT32K_MODEFILE_SHA256,
        "declared_server_context_length": OLLAMA_NEMOTRON_CONTEXT32K_CONTEXT_LENGTH,
        "server_context_length_binding": "declared_operator_bound",
        "server_context_length_per_response_observable": False,
        "request_profile": "SSB-OAI-CHAT1",
        "endpoint_path": "/v1/chat/completions",
        "sampling_temperature": OLLAMA_NEMOTRON_CONTEXT32K_TEMPERATURE,
        "seed": OLLAMA_NEMOTRON_CONTEXT32K_SEED,
        "max_tokens": OLLAMA_NEMOTRON_CONTEXT32K_ROLE_PROBE_MAX_TOKENS,
        "reasoning_effort_wire": "omitted",
        "top_p_wire": "omitted",
        "top_p_effective_per_response_observable": False,
        "structured_output_schema": schema,
        "structured_output_schema_hash": sha256_ref(schema),
        "role_structured_output_probe": _probe(),
        "prompt_token_difference_rule": "nonzero_direction_not_interpreted",
    }


def nemotron_context32k_live_show_identity_projection() -> dict[str, object]:
    """Return the hash-only live identity fields bound before future inference."""

    return {
        "model": OLLAMA_NEMOTRON_CONTEXT32K_MODEL,
        "model_digest": OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST,
        "config_blob_sha256": OLLAMA_NEMOTRON_CONTEXT32K_CONFIG_BLOB_SHA256,
        "modelfile_sha256": OLLAMA_NEMOTRON_CONTEXT32K_MODEFILE_SHA256,
        "ollama_server_version": OLLAMA_NEMOTRON_CONTEXT32K_SERVER_VERSION,
        "context_length": OLLAMA_NEMOTRON_CONTEXT32K_CONTEXT_LENGTH,
        "renderer": "nemotron-3.5-nano",
        "parser": "nemotron-3.5-nano",
        "capabilities": ["completion", "tools", "thinking"],
    }


def validate_nemotron_context32k_live_show_identity(value: object) -> dict[str, object]:
    """Validate only a V2 identity receipt bound to the declared 32k server context."""

    if type(value) is not dict:
        raise NemotronContext32kRoleProfileError("Nemotron context-32k identity receipt is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": "OLLAMA_NEMOTRON_CONTEXT32K_LIVE_SHOW_IDENTITY_RECEIPT2",
        **nemotron_context32k_live_show_identity_projection(),
        "show_response_hash": value.get("show_response_hash"),
    }
    if (
        body != expected
        or type(value.get("show_response_hash")) is not str
        or not cast(str, value["show_response_hash"]).startswith("sha256:")
        or value.get("receipt_hash") != sha256_ref(body)
    ):
        raise NemotronContext32kRoleProfileError(
            "Nemotron context-32k identity receipt is not bound"
        )
    return cast(dict[str, object], value)


def canonical_nemotron_context32k_live_show_identity_receipt(
    *, show_response: bytes, ollama_version_text: str
) -> bytes:
    """Build the V2 hash-only identity receipt from operator-captured show bytes."""

    try:
        show = json.loads(show_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NemotronContext32kRoleProfileError(
            "Nemotron context-32k show response is not JSON"
        ) from error
    if type(show) is not dict or show.get("capabilities") != ["completion", "tools", "thinking"]:
        raise NemotronContext32kRoleProfileError("Nemotron context-32k show response is not pinned")
    if re.search(r"\b0\.33\.2\b", ollama_version_text) is None:
        raise NemotronContext32kRoleProfileError(
            "Nemotron context-32k Ollama version is not pinned"
        )
    body = {
        "record_kind": "OLLAMA_NEMOTRON_CONTEXT32K_LIVE_SHOW_IDENTITY_RECEIPT2",
        **nemotron_context32k_live_show_identity_projection(),
        "show_response_hash": "sha256:" + sha256(show_response).hexdigest(),
    }
    value = {**body, "receipt_hash": sha256_ref(body)}
    validate_nemotron_context32k_live_show_identity(value)
    return canonical_json_bytes(value)


def load_nemotron_context32k_live_show_identity(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise NemotronContext32kRoleProfileError(
            "Nemotron context-32k identity receipt is unavailable"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NemotronContext32kRoleProfileError(
            "Nemotron context-32k identity receipt is unreadable"
        ) from error
    if canonical_json_bytes(value) != raw:
        raise NemotronContext32kRoleProfileError(
            "Nemotron context-32k identity receipt is not canonical"
        )
    return validate_nemotron_context32k_live_show_identity(value)


def validate_nemotron_context32k_role_profile_receipt(value: object) -> dict[str, object]:
    """Reject V1/131072-shaped or otherwise nonconformant role receipts."""

    if type(value) is not dict:
        raise NemotronContext32kRoleProfileError("Nemotron context-32k role receipt is invalid")
    projection = ollama_nemotron_context32k_profile_projection()
    baseline = value.get("baseline")
    developer = value.get("developer")
    delta = value.get("developer_prompt_token_delta")
    expected = {
        **projection,
        "record_kind": "OLLAMA_NEMOTRON_CONTEXT32K_ROLE_STRUCTURED_OUTPUT_RECEIPT2",
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
            int, cast(dict[str, object], cast(dict[str, object], baseline)["usage"])["input_tokens"]
        )
        + delta
    ):
        raise NemotronContext32kRoleProfileError("Nemotron context-32k role receipt is not bound")
    return cast(dict[str, object], value)


def canonical_nemotron_context32k_role_profile_receipt(
    *, baseline: dict[str, object], developer: dict[str, object]
) -> bytes:
    baseline_usage = baseline.get("usage") if type(baseline) is dict else None
    developer_usage = developer.get("usage") if type(developer) is dict else None
    value = {
        **ollama_nemotron_context32k_profile_projection(),
        "record_kind": "OLLAMA_NEMOTRON_CONTEXT32K_ROLE_STRUCTURED_OUTPUT_RECEIPT2",
        "baseline": baseline,
        "developer": developer,
        "developer_prompt_token_delta": (
            cast(int, developer_usage["input_tokens"]) - cast(int, baseline_usage["input_tokens"])
            if type(baseline_usage) is dict and type(developer_usage) is dict
            else None
        ),
    }
    validate_nemotron_context32k_role_profile_receipt(value)
    return canonical_json_bytes(value)


def load_nemotron_context32k_role_profile_receipt(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise NemotronContext32kRoleProfileError("Nemotron context-32k role receipt is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NemotronContext32kRoleProfileError(
            "Nemotron context-32k role receipt is unreadable"
        ) from error
    if canonical_json_bytes(value) != raw:
        raise NemotronContext32kRoleProfileError(
            "Nemotron context-32k role receipt is not canonical"
        )
    return validate_nemotron_context32k_role_profile_receipt(value)


__all__ = [
    "OLLAMA_NEMOTRON_CONTEXT32K_CONFIG_BLOB_SHA256",
    "OLLAMA_NEMOTRON_CONTEXT32K_CONTEXT_LENGTH",
    "OLLAMA_NEMOTRON_CONTEXT32K_MODEFILE_SHA256",
    "OLLAMA_NEMOTRON_CONTEXT32K_MODEL",
    "OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST",
    "OLLAMA_NEMOTRON_CONTEXT32K_PROFILE",
    "OLLAMA_NEMOTRON_CONTEXT32K_PROVIDER",
    "OLLAMA_NEMOTRON_CONTEXT32K_ROLE_PROBE_MAX_TOKENS",
    "OLLAMA_NEMOTRON_CONTEXT32K_SEED",
    "OLLAMA_NEMOTRON_CONTEXT32K_SERVER_VERSION",
    "OLLAMA_NEMOTRON_CONTEXT32K_TEMPERATURE",
    "NemotronContext32kRoleMarker",
    "NemotronContext32kRoleProfileError",
    "canonical_nemotron_context32k_live_show_identity_receipt",
    "canonical_nemotron_context32k_role_profile_receipt",
    "load_nemotron_context32k_live_show_identity",
    "load_nemotron_context32k_role_profile_receipt",
    "nemotron_context32k_live_show_identity_projection",
    "ollama_nemotron_context32k_profile_projection",
    "validate_nemotron_context32k_live_show_identity",
    "validate_nemotron_context32k_role_profile_receipt",
]
