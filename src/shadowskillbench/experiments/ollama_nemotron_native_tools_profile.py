"""Frozen native-tools role/conformance receipt for the 32k Nemotron successor."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_nemotron_context32k_profile import (
    OLLAMA_NEMOTRON_CONTEXT32K_CONFIG_BLOB_SHA256,
    OLLAMA_NEMOTRON_CONTEXT32K_CONTEXT_LENGTH,
    OLLAMA_NEMOTRON_CONTEXT32K_MODEL,
    OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST,
    OLLAMA_NEMOTRON_CONTEXT32K_PROVIDER,
    OLLAMA_NEMOTRON_CONTEXT32K_SEED,
    OLLAMA_NEMOTRON_CONTEXT32K_SERVER_VERSION,
    OLLAMA_NEMOTRON_CONTEXT32K_TEMPERATURE,
)

OLLAMA_NEMOTRON_NATIVE_TOOLS_PROFILE = (
    "SSB-OLLAMA-NEMOTRON35-LIGHTNING-30B-MLX-NATIVE-TOOLS-CONTEXT32768-ROLES1"
)
OLLAMA_NEMOTRON_NATIVE_TOOLS_RECEIPT_KIND = (
    "OLLAMA_NEMOTRON_NATIVE_TOOLS_CONTEXT32K_ROLE_TOOL_CONFORMANCE_RECEIPT1"
)
OLLAMA_NEMOTRON_NATIVE_TOOLS_FAILURE_RECEIPT_KIND = (
    "OLLAMA_NEMOTRON_NATIVE_TOOLS_CONTEXT32K_ROLE_TOOL_CONFORMANCE_FAILURE_RECEIPT1"
)
OLLAMA_NEMOTRON_NATIVE_TOOLS_IDENTITY_KIND = (
    "OLLAMA_NEMOTRON_CONTEXT32K_LIVE_SHOW_IDENTITY_RECEIPT2"
)
OLLAMA_NEMOTRON_NATIVE_TOOLS_PROBE_TOOL = "finish_task"
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class NemotronNativeToolsRoleProfileError(ValueError):
    """Native tools conformance evidence is unavailable or does not bind the route."""


def _usage(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"input_tokens", "output_tokens", "total_tokens"}
        and all(type(value[key]) is int and value[key] >= 0 for key in value)
        and value["total_tokens"] == value["input_tokens"] + value["output_tokens"]
    )


def _call(value: object, marker: str | None = None) -> bool:
    if type(value) is not dict or set(value) != {
        "raw_request_hash",
        "raw_response_hash",
        "usage",
        "attempts",
        "finish_reason",
        "role_marker",
    }:
        return False
    usage = value.get("usage")
    return (
        type(value.get("raw_request_hash")) is str
        and _HASH.fullmatch(cast(str, value["raw_request_hash"])) is not None
        and type(value.get("raw_response_hash")) is str
        and _HASH.fullmatch(cast(str, value["raw_response_hash"])) is not None
        and _usage(usage)
        and value.get("attempts") == 1
        and value.get("finish_reason") == "tool_calls"
        and (marker is None or value.get("role_marker") == marker)
    )


def _tool_call(value: object) -> bool:
    if type(value) is not dict or set(value) != {
        "raw_request_hash",
        "raw_response_hash",
        "usage",
        "attempts",
        "finish_reason",
        "role_marker",
        "tool_name",
        "arguments_hash",
        "tool_declaration_hash",
    }:
        return False
    parsed = cast(dict[str, object], value)
    return (
        _call(
            {
                key: parsed[key]
                for key in {
                    "raw_request_hash",
                    "raw_response_hash",
                    "usage",
                    "attempts",
                    "finish_reason",
                    "role_marker",
                }
            }
        )
        and parsed.get("tool_name") == OLLAMA_NEMOTRON_NATIVE_TOOLS_PROBE_TOOL
        and all(
            type(parsed.get(key)) is str and _HASH.fullmatch(cast(str, parsed[key])) is not None
            for key in ("arguments_hash", "tool_declaration_hash")
        )
    )


def ollama_nemotron_native_tools_profile_projection() -> dict[str, object]:
    return {
        "role_profile": OLLAMA_NEMOTRON_NATIVE_TOOLS_PROFILE,
        "provider": OLLAMA_NEMOTRON_CONTEXT32K_PROVIDER,
        "model": OLLAMA_NEMOTRON_CONTEXT32K_MODEL,
        "model_digest": OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST,
        "ollama_server_version": OLLAMA_NEMOTRON_CONTEXT32K_SERVER_VERSION,
        "config_blob_sha256": OLLAMA_NEMOTRON_CONTEXT32K_CONFIG_BLOB_SHA256,
        "context_length": OLLAMA_NEMOTRON_CONTEXT32K_CONTEXT_LENGTH,
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
        "endpoint_path": "/api/chat",
        "stream": False,
        "temperature": OLLAMA_NEMOTRON_CONTEXT32K_TEMPERATURE,
        "seed": OLLAMA_NEMOTRON_CONTEXT32K_SEED,
        "max_output_tokens": 8192,
        "top_p_wire": "omitted",
        "reasoning_effort_wire": "omitted",
        "tool_conformance": {
            "exactly_one_call": True,
            "tool_name": OLLAMA_NEMOTRON_NATIVE_TOOLS_PROBE_TOOL,
            "arguments": {"summary": "native tool conformance"},
        },
    }


def validate_nemotron_native_tools_role_profile_receipt(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise NemotronNativeToolsRoleProfileError("native tools receipt is invalid")
    baseline = value.get("baseline")
    developer = value.get("developer")
    tool_call = value.get("tool_call")
    tool_declaration_hash = value.get("tool_declaration_hash")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_NEMOTRON_NATIVE_TOOLS_RECEIPT_KIND,
        **ollama_nemotron_native_tools_profile_projection(),
        "baseline": baseline,
        "developer": developer,
        "tool_call": tool_call,
        "tool_declaration_hash": tool_declaration_hash,
        "developer_prompt_token_delta": value.get("developer_prompt_token_delta"),
    }
    if (
        body != expected
        or value.get("receipt_hash") != sha256_ref(body)
        or not _call(baseline, "SSB_NATIVE_SYSTEM_ROLE_MARKER")
        or not _call(developer, "SSB_NATIVE_DEVELOPER_ROLE_MARKER")
        or not _tool_call(tool_call)
        or cast(dict[str, object], tool_call).get("role_marker") != "SSB_NATIVE_TOOL_CALL_MARKER"
        or type(tool_declaration_hash) is not str
        or _HASH.fullmatch(tool_declaration_hash) is None
        or cast(dict[str, object], tool_call).get("tool_declaration_hash") != tool_declaration_hash
        or type(value.get("developer_prompt_token_delta")) is not int
        or value["developer_prompt_token_delta"] == 0
        or cast(dict[str, object], baseline)["raw_request_hash"]
        == cast(dict[str, object], developer)["raw_request_hash"]
        or cast(dict[str, object], baseline)["raw_response_hash"]
        == cast(dict[str, object], developer)["raw_response_hash"]
    ):
        raise NemotronNativeToolsRoleProfileError("native tools receipt is not bound")
    return cast(dict[str, object], value)


def canonical_nemotron_native_tools_role_profile_receipt(
    *, baseline: dict[str, object], developer: dict[str, object], tool_call: dict[str, object]
) -> bytes:
    baseline_usage = baseline.get("usage")
    developer_usage = developer.get("usage")
    value = {
        "record_kind": OLLAMA_NEMOTRON_NATIVE_TOOLS_RECEIPT_KIND,
        **ollama_nemotron_native_tools_profile_projection(),
        "baseline": baseline,
        "developer": developer,
        "tool_call": tool_call,
        "tool_declaration_hash": tool_call.get("tool_declaration_hash"),
        "developer_prompt_token_delta": (
            cast(int, developer_usage["input_tokens"]) - cast(int, baseline_usage["input_tokens"])
            if type(baseline_usage) is dict and type(developer_usage) is dict
            else None
        ),
    }
    body = {**value, "receipt_hash": sha256_ref(value)}
    validate_nemotron_native_tools_role_profile_receipt(body)
    return canonical_json_bytes(body)


def canonical_nemotron_native_tools_role_profile_failure_receipt(
    *,
    failure_code: str,
    completed_calls: int,
    failed_call: dict[str, object] | None,
) -> bytes:
    """Build a hash-only, non-admissible recorder failure receipt.

    It preserves custody hashes and usage where the provider returned them, but
    deliberately has a different record kind so it can never admit this route.
    """

    failure_fields = {
        "raw_request_hash",
        "raw_response_hash",
        "usage",
        "attempts",
        "finish_reason",
        "output_cap_exhausted",
    }
    if (
        failure_code
        not in {
            "CONFIGURATION_ERROR",
            "MODEL_OUTPUT_INVALID",
            "MODEL_PROVIDER_TERMINAL",
            "MODEL_PROVIDER_TRANSIENT",
        }
        or type(completed_calls) is not int
        or not 0 <= completed_calls <= 2
        or failed_call is not None
        and (
            type(failed_call) is not dict
            or set(failed_call) != failure_fields
            or failed_call.get("raw_request_hash") is not None
            and (
                type(failed_call["raw_request_hash"]) is not str
                or _HASH.fullmatch(cast(str, failed_call["raw_request_hash"])) is None
            )
            or failed_call.get("raw_response_hash") is not None
            and (
                type(failed_call["raw_response_hash"]) is not str
                or _HASH.fullmatch(cast(str, failed_call["raw_response_hash"])) is None
            )
            or type(failed_call.get("attempts")) is not int
            or not 0 <= cast(int, failed_call["attempts"]) <= 1
            or failed_call.get("usage") is not None
            and not _usage(failed_call["usage"])
            or failed_call.get("finish_reason") is not None
            and type(failed_call["finish_reason"]) is not str
            or failed_call.get("output_cap_exhausted") is not None
            and type(failed_call["output_cap_exhausted"]) is not bool
        )
    ):
        raise NemotronNativeToolsRoleProfileError("native tools failure receipt is invalid")
    body = {
        "record_kind": OLLAMA_NEMOTRON_NATIVE_TOOLS_FAILURE_RECEIPT_KIND,
        **ollama_nemotron_native_tools_profile_projection(),
        "failure_code": failure_code,
        "completed_calls": completed_calls,
        "failed_call": failed_call,
    }
    return canonical_json_bytes({**body, "receipt_hash": sha256_ref(body)})


def load_nemotron_native_tools_role_profile_receipt(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise NemotronNativeToolsRoleProfileError("native tools receipt is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NemotronNativeToolsRoleProfileError("native tools receipt is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise NemotronNativeToolsRoleProfileError("native tools receipt is not canonical")
    return validate_nemotron_native_tools_role_profile_receipt(value)


__all__ = [
    "NemotronNativeToolsRoleProfileError",
    "OLLAMA_NEMOTRON_NATIVE_TOOLS_FAILURE_RECEIPT_KIND",
    "OLLAMA_NEMOTRON_NATIVE_TOOLS_IDENTITY_KIND",
    "OLLAMA_NEMOTRON_NATIVE_TOOLS_PROFILE",
    "OLLAMA_NEMOTRON_NATIVE_TOOLS_RECEIPT_KIND",
    "canonical_nemotron_native_tools_role_profile_receipt",
    "canonical_nemotron_native_tools_role_profile_failure_receipt",
    "load_nemotron_native_tools_role_profile_receipt",
    "ollama_nemotron_native_tools_profile_projection",
    "validate_nemotron_native_tools_role_profile_receipt",
]
