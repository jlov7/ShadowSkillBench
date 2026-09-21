"""Frozen GLM native-tool response-contract successor (ROLES2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_glm47_native_tools_profile import (
    _HASH,
    OLLAMA_GLM47_NATIVE_TOOLS_CONFIG_BLOB_SHA256,
    OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT_LENGTH,
    OLLAMA_GLM47_NATIVE_TOOLS_MODEFILE_SHA256,
    OLLAMA_GLM47_NATIVE_TOOLS_MODEL,
    OLLAMA_GLM47_NATIVE_TOOLS_MODEL_DIGEST,
    OLLAMA_GLM47_NATIVE_TOOLS_PROBE_TOOL,
    OLLAMA_GLM47_NATIVE_TOOLS_PROVIDER,
    OLLAMA_GLM47_NATIVE_TOOLS_ROLE_PROBE_MAX_TOKENS,
    OLLAMA_GLM47_NATIVE_TOOLS_SEED,
    OLLAMA_GLM47_NATIVE_TOOLS_SERVER_VERSION,
    OLLAMA_GLM47_NATIVE_TOOLS_TEMPERATURE,
    Glm47NativeToolsRoleProfileError,
    _usage,
    glm47_native_tools_live_show_identity_projection,
)
from shadowskillbench.experiments.ollama_glm47_native_tools_profile import (
    canonical_glm47_native_tools_live_show_identity_receipt as _v1_identity,
)

OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_PROFILE = (
    "SSB-OLLAMA-GLM47-FLASH-NATIVE-TOOLS-CONTEXT32768-ROLES2"
)
OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_RECEIPT_KIND = (
    "OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT32K_ROLE_TOOL_CONFORMANCE_RECEIPT2"
)
OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_FAILURE_RECEIPT_KIND = (
    "OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT32K_ROLE_TOOL_CONFORMANCE_FAILURE_RECEIPT2"
)
OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_IDENTITY_KIND = (
    "OLLAMA_GLM47_CONTEXT32K_LIVE_SHOW_IDENTITY_RECEIPT2"
)
OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_CONTRACT = "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2"


class Glm47NativeToolsResponseV2Error(Glm47NativeToolsRoleProfileError):
    """The GLM ROLES2 response-contract receipt is unavailable or invalid."""


def ollama_glm47_native_tools_response_v2_projection() -> dict[str, object]:
    return {
        "role_profile": OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_PROFILE,
        "provider": OLLAMA_GLM47_NATIVE_TOOLS_PROVIDER,
        "model": OLLAMA_GLM47_NATIVE_TOOLS_MODEL,
        "model_digest": OLLAMA_GLM47_NATIVE_TOOLS_MODEL_DIGEST,
        "ollama_server_version": OLLAMA_GLM47_NATIVE_TOOLS_SERVER_VERSION,
        "config_blob_sha256": OLLAMA_GLM47_NATIVE_TOOLS_CONFIG_BLOB_SHA256,
        "modelfile_sha256": OLLAMA_GLM47_NATIVE_TOOLS_MODEFILE_SHA256,
        "context_length": OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT_LENGTH,
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
        "response_contract": OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_CONTRACT,
        "endpoint_path": "/api/chat",
        "stream": False,
        "temperature": OLLAMA_GLM47_NATIVE_TOOLS_TEMPERATURE,
        "seed": OLLAMA_GLM47_NATIVE_TOOLS_SEED,
        "max_output_tokens": OLLAMA_GLM47_NATIVE_TOOLS_ROLE_PROBE_MAX_TOKENS,
        "top_p_wire": "omitted",
        "reasoning_effort_wire": "omitted",
        "tool_conformance": {
            "exactly_one_call": True,
            "tool_name": OLLAMA_GLM47_NATIVE_TOOLS_PROBE_TOOL,
            "arguments": {"summary": "native tool conformance"},
        },
    }


def _call(value: object, marker: str) -> bool:
    if type(value) is not dict or set(value) != {
        "raw_request_hash",
        "raw_response_hash",
        "usage",
        "attempts",
        "finish_reason",
        "provider_finish_reason",
        "provider_done",
        "role_marker",
    }:
        return False
    reason = value.get("provider_finish_reason")
    return (
        type(value.get("raw_request_hash")) is str
        and _HASH.fullmatch(cast(str, value["raw_request_hash"])) is not None
        and type(value.get("raw_response_hash")) is str
        and _HASH.fullmatch(cast(str, value["raw_response_hash"])) is not None
        and _usage(value.get("usage"))
        and value.get("attempts") == 1
        and value.get("finish_reason") == "tool_calls"
        and value.get("provider_done") is True
        and (reason is None or type(reason) is str)
        and value.get("role_marker") == marker
    )


def _tool_call(value: object) -> bool:
    if type(value) is not dict or set(value) != {
        "raw_request_hash",
        "raw_response_hash",
        "usage",
        "attempts",
        "finish_reason",
        "provider_finish_reason",
        "provider_done",
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
                    "provider_finish_reason",
                    "provider_done",
                    "role_marker",
                }
            },
            "SSB_NATIVE_TOOL_CALL_MARKER",
        )
        and parsed.get("tool_name") == OLLAMA_GLM47_NATIVE_TOOLS_PROBE_TOOL
        and all(
            type(parsed.get(key)) is str and _HASH.fullmatch(cast(str, parsed[key])) is not None
            for key in ("arguments_hash", "tool_declaration_hash")
        )
    )


def validate_glm47_native_tools_response_v2_receipt(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise Glm47NativeToolsResponseV2Error("GLM ROLES2 receipt is invalid")
    baseline = value.get("baseline")
    developer = value.get("developer")
    tool_call = value.get("tool_call")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_RECEIPT_KIND,
        **ollama_glm47_native_tools_response_v2_projection(),
        "baseline": baseline,
        "developer": developer,
        "tool_call": tool_call,
        "tool_declaration_hash": value.get("tool_declaration_hash"),
        "developer_prompt_token_delta": value.get("developer_prompt_token_delta"),
    }
    if (
        body != expected
        or value.get("receipt_hash") != sha256_ref(body)
        or not _call(baseline, "SSB_NATIVE_SYSTEM_ROLE_MARKER")
        or not _call(developer, "SSB_NATIVE_DEVELOPER_ROLE_MARKER")
        or not _tool_call(tool_call)
        or cast(dict[str, object], tool_call).get("tool_declaration_hash")
        != value.get("tool_declaration_hash")
        or type(value.get("developer_prompt_token_delta")) is not int
        or value["developer_prompt_token_delta"] == 0
    ):
        raise Glm47NativeToolsResponseV2Error("GLM ROLES2 receipt is not bound")
    return cast(dict[str, object], value)


def canonical_glm47_native_tools_response_v2_receipt(
    *, baseline: dict[str, object], developer: dict[str, object], tool_call: dict[str, object]
) -> bytes:
    baseline_usage, developer_usage = baseline.get("usage"), developer.get("usage")
    body = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_RECEIPT_KIND,
        **ollama_glm47_native_tools_response_v2_projection(),
        "baseline": baseline,
        "developer": developer,
        "tool_call": tool_call,
        "tool_declaration_hash": tool_call.get("tool_declaration_hash"),
        "developer_prompt_token_delta": cast(int, developer_usage["input_tokens"])
        - cast(int, baseline_usage["input_tokens"])
        if type(baseline_usage) is dict and type(developer_usage) is dict
        else None,
    }
    value = {**body, "receipt_hash": sha256_ref(body)}
    validate_glm47_native_tools_response_v2_receipt(value)
    return canonical_json_bytes(value)


def canonical_glm47_native_tools_response_v2_failure_receipt(
    *, failure_code: str, completed_calls: int, failed_call: dict[str, object] | None
) -> bytes:
    fields = {
        "raw_request_hash",
        "raw_response_hash",
        "usage",
        "attempts",
        "provider_finish_reason",
        "provider_done",
        "valid_tool_name",
        "valid_arguments_hash_match",
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
        and (type(failed_call) is not dict or set(failed_call) != fields)
        or failed_call is not None
        and (
            any(
                type(failed_call.get(key)) is not str
                or _HASH.fullmatch(cast(str, failed_call[key])) is None
                for key in ("raw_request_hash", "raw_response_hash")
            )
            or not _usage(failed_call.get("usage"))
            or failed_call.get("attempts") != 1
            or failed_call.get("provider_finish_reason") is not None
            and type(failed_call.get("provider_finish_reason")) is not str
            or failed_call.get("provider_done") is not None
            and type(failed_call.get("provider_done")) is not bool
            or failed_call.get("valid_tool_name") is not None
            and type(failed_call.get("valid_tool_name")) is not bool
            or failed_call.get("valid_arguments_hash_match") is not None
            and type(failed_call.get("valid_arguments_hash_match")) is not bool
            or type(failed_call.get("output_cap_exhausted")) is not bool
        )
    ):
        raise Glm47NativeToolsResponseV2Error("GLM ROLES2 failure receipt is invalid")
    body = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_FAILURE_RECEIPT_KIND,
        **ollama_glm47_native_tools_response_v2_projection(),
        "failure_code": failure_code,
        "completed_calls": completed_calls,
        "failed_call": failed_call,
    }
    return canonical_json_bytes({**body, "receipt_hash": sha256_ref(body)})


def validate_glm47_native_tools_response_v2_identity(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise Glm47NativeToolsResponseV2Error("GLM ROLES2 identity is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_IDENTITY_KIND,
        **glm47_native_tools_live_show_identity_projection(),
        "show_response_hash": value.get("show_response_hash"),
    }
    if body != expected or value.get("receipt_hash") != sha256_ref(body):
        raise Glm47NativeToolsResponseV2Error("GLM ROLES2 identity is not bound")
    return cast(dict[str, object], value)


def canonical_glm47_native_tools_response_v2_identity(
    *, show_response: bytes, ollama_version_text: str
) -> bytes:
    v1 = json.loads(
        _v1_identity(show_response=show_response, ollama_version_text=ollama_version_text)
    )
    body = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_RESPONSE_V2_IDENTITY_KIND,
        **glm47_native_tools_live_show_identity_projection(),
        "show_response_hash": v1["show_response_hash"],
    }
    value = {**body, "receipt_hash": sha256_ref(body)}
    validate_glm47_native_tools_response_v2_identity(value)
    return canonical_json_bytes(value)


def _load(path: Path, validator, label: str) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise Glm47NativeToolsResponseV2Error(f"GLM ROLES2 {label} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Glm47NativeToolsResponseV2Error(f"GLM ROLES2 {label} is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise Glm47NativeToolsResponseV2Error(f"GLM ROLES2 {label} is not canonical")
    return validator(value)


def load_glm47_native_tools_response_v2_receipt(path: Path) -> dict[str, object]:
    return _load(path, validate_glm47_native_tools_response_v2_receipt, "receipt")


def load_glm47_native_tools_response_v2_identity(path: Path) -> dict[str, object]:
    return _load(path, validate_glm47_native_tools_response_v2_identity, "identity")
