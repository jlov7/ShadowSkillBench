"""Frozen Qwen native-tool response-contract successor (ROLES2)."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.ollama_glm47_native_tools_profile import _HASH, _usage

OLLAMA_QWEN35_NATIVE_TOOLS_V2_PROVIDER = "ollama-qwen35-9b"
OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL = "qwen3.5:9b"
OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL_DIGEST = (
    "sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
)
OLLAMA_QWEN35_NATIVE_TOOLS_V2_SERVER_VERSION = "0.33.2"
OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONFIG_BLOB_SHA256 = (
    "sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
)
OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEFILE_SHA256 = (
    "sha256:f3d0707d33423bdec8d4144d22c342e00ef19a1755d7ecfa7786f3f30292ca95"
)
OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONTEXT_LENGTH = 32768
OLLAMA_QWEN35_NATIVE_TOOLS_V2_TEMPERATURE = 0.15
OLLAMA_QWEN35_NATIVE_TOOLS_V2_SEED = 4242
OLLAMA_QWEN35_NATIVE_TOOLS_V2_ROLE_PROBE_MAX_TOKENS = 8192
OLLAMA_QWEN35_NATIVE_TOOLS_V2_PROBE_TOOL = "finish_task"

OLLAMA_QWEN35_NATIVE_TOOLS_V2_PROFILE = "SSB-OLLAMA-QWEN35-9B-NATIVE-TOOLS-CONTEXT32768-ROLES2"
OLLAMA_QWEN35_NATIVE_TOOLS_V2_RECEIPT_KIND = (
    "OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONTEXT32K_ROLE_TOOL_CONFORMANCE_RECEIPT2"
)
OLLAMA_QWEN35_NATIVE_TOOLS_V2_FAILURE_RECEIPT_KIND = (
    "OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONTEXT32K_ROLE_TOOL_CONFORMANCE_FAILURE_RECEIPT2"
)
OLLAMA_QWEN35_NATIVE_TOOLS_V2_IDENTITY_KIND = "OLLAMA_QWEN35_CONTEXT32K_LIVE_SHOW_IDENTITY_RECEIPT2"
OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONTRACT = "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2"


class Qwen35NativeToolsV2ProfileError(ValueError):
    """The Qwen 3.5 ROLES2 response-contract receipt is unavailable or invalid."""


def ollama_qwen35_native_tools_v2_projection() -> dict[str, object]:
    return {
        "role_profile": OLLAMA_QWEN35_NATIVE_TOOLS_V2_PROFILE,
        "provider": OLLAMA_QWEN35_NATIVE_TOOLS_V2_PROVIDER,
        "model": OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL,
        "model_digest": OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL_DIGEST,
        "ollama_server_version": OLLAMA_QWEN35_NATIVE_TOOLS_V2_SERVER_VERSION,
        "config_blob_sha256": OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONFIG_BLOB_SHA256,
        "modelfile_sha256": OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEFILE_SHA256,
        "context_length": OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONTEXT_LENGTH,
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
        "response_contract": OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONTRACT,
        "endpoint_path": "/api/chat",
        "stream": False,
        "temperature": OLLAMA_QWEN35_NATIVE_TOOLS_V2_TEMPERATURE,
        "seed": OLLAMA_QWEN35_NATIVE_TOOLS_V2_SEED,
        "max_output_tokens": OLLAMA_QWEN35_NATIVE_TOOLS_V2_ROLE_PROBE_MAX_TOKENS,
        "top_p_wire": "omitted",
        "reasoning_effort_wire": "omitted",
        "tool_conformance": {
            "exactly_one_call": True,
            "tool_name": OLLAMA_QWEN35_NATIVE_TOOLS_V2_PROBE_TOOL,
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
        and parsed.get("tool_name") == OLLAMA_QWEN35_NATIVE_TOOLS_V2_PROBE_TOOL
        and all(
            type(parsed.get(key)) is str and _HASH.fullmatch(cast(str, parsed[key])) is not None
            for key in ("arguments_hash", "tool_declaration_hash")
        )
    )


def validate_qwen35_native_tools_v2_receipt(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise Qwen35NativeToolsV2ProfileError("Qwen 3.5 ROLES2 receipt is invalid")
    baseline = value.get("baseline")
    developer = value.get("developer")
    tool_call = value.get("tool_call")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_QWEN35_NATIVE_TOOLS_V2_RECEIPT_KIND,
        **ollama_qwen35_native_tools_v2_projection(),
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
        raise Qwen35NativeToolsV2ProfileError("Qwen 3.5 ROLES2 receipt is not bound")
    return cast(dict[str, object], value)


def canonical_qwen35_native_tools_v2_receipt(
    *, baseline: dict[str, object], developer: dict[str, object], tool_call: dict[str, object]
) -> bytes:
    baseline_usage, developer_usage = baseline.get("usage"), developer.get("usage")
    body = {
        "record_kind": OLLAMA_QWEN35_NATIVE_TOOLS_V2_RECEIPT_KIND,
        **ollama_qwen35_native_tools_v2_projection(),
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
    validate_qwen35_native_tools_v2_receipt(value)
    return canonical_json_bytes(value)


def canonical_qwen35_native_tools_v2_failure_receipt(
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
            "ROLE_CONFORMANCE_INVALID",
        }
        or type(completed_calls) is not int
        or not 0 <= completed_calls <= 3
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
        raise Qwen35NativeToolsV2ProfileError("Qwen 3.5 ROLES2 failure receipt is invalid")
    body = {
        "record_kind": OLLAMA_QWEN35_NATIVE_TOOLS_V2_FAILURE_RECEIPT_KIND,
        **ollama_qwen35_native_tools_v2_projection(),
        "failure_code": failure_code,
        "completed_calls": completed_calls,
        "failed_call": failed_call,
    }
    return canonical_json_bytes({**body, "receipt_hash": sha256_ref(body)})


def validate_qwen35_native_tools_v2_identity(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise Qwen35NativeToolsV2ProfileError("Qwen 3.5 ROLES2 identity is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_QWEN35_NATIVE_TOOLS_V2_IDENTITY_KIND,
        **qwen35_native_tools_v2_identity_projection(),
        "show_response_hash": value.get("show_response_hash"),
    }
    if body != expected or value.get("receipt_hash") != sha256_ref(body):
        raise Qwen35NativeToolsV2ProfileError("Qwen 3.5 ROLES2 identity is not bound")
    return cast(dict[str, object], value)


def canonical_qwen35_native_tools_v2_identity(
    *, show_response: bytes, ollama_version_text: str
) -> bytes:
    try:
        show = json.loads(show_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen35NativeToolsV2ProfileError("Qwen show is not JSON") from error
    modelfile = show.get("modelfile") if type(show) is dict else None
    if (
        type(modelfile) is not str
        or "sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
        not in modelfile
        or show.get("capabilities") != ["completion", "vision", "tools", "thinking"]
        or "qwen3.5" not in modelfile
        or "sha256:" + sha256(modelfile.encode("utf-8")).hexdigest()
        != OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEFILE_SHA256
        or "0.33.2" not in ollama_version_text
    ):
        raise Qwen35NativeToolsV2ProfileError("Qwen show identity is not pinned")
    body = {
        "record_kind": OLLAMA_QWEN35_NATIVE_TOOLS_V2_IDENTITY_KIND,
        **qwen35_native_tools_v2_identity_projection(),
        "show_response_hash": "sha256:" + sha256(show_response).hexdigest(),
    }
    value = {**body, "receipt_hash": sha256_ref(body)}
    validate_qwen35_native_tools_v2_identity(value)
    return canonical_json_bytes(value)


def qwen35_native_tools_v2_identity_projection() -> dict[str, object]:
    return {
        "model": OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL,
        "model_digest": OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL_DIGEST,
        "config_blob_sha256": OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONFIG_BLOB_SHA256,
        "modelfile_sha256": OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEFILE_SHA256,
        "ollama_server_version": OLLAMA_QWEN35_NATIVE_TOOLS_V2_SERVER_VERSION,
        "context_length": OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONTEXT_LENGTH,
        "renderer": "qwen3.5",
        "parser": "qwen3.5",
        "capabilities": ["completion", "vision", "tools", "thinking"],
    }


def _load(path: Path, validator, label: str) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise Qwen35NativeToolsV2ProfileError(f"Qwen 3.5 ROLES2 {label} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen35NativeToolsV2ProfileError(f"Qwen 3.5 ROLES2 {label} is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise Qwen35NativeToolsV2ProfileError(f"Qwen 3.5 ROLES2 {label} is not canonical")
    return validator(value)


def load_qwen35_native_tools_v2_receipt(path: Path) -> dict[str, object]:
    return _load(path, validate_qwen35_native_tools_v2_receipt, "receipt")


def load_qwen35_native_tools_v2_identity(path: Path) -> dict[str, object]:
    return _load(path, validate_qwen35_native_tools_v2_identity, "identity")
