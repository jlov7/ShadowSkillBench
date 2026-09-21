"""Frozen GLM 4.7 Flash native-tools role and identity profile."""
# ruff: noqa: E501

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref

OLLAMA_GLM47_NATIVE_TOOLS_PROFILE = "SSB-OLLAMA-GLM47-FLASH-NATIVE-TOOLS-CONTEXT32768-ROLES1"
OLLAMA_GLM47_NATIVE_TOOLS_RECEIPT_KIND = (
    "OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT32K_ROLE_TOOL_CONFORMANCE_RECEIPT1"
)
OLLAMA_GLM47_NATIVE_TOOLS_FAILURE_RECEIPT_KIND = (
    "OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT32K_ROLE_TOOL_CONFORMANCE_FAILURE_RECEIPT1"
)
OLLAMA_GLM47_NATIVE_TOOLS_IDENTITY_KIND = "OLLAMA_GLM47_CONTEXT32K_LIVE_SHOW_IDENTITY_RECEIPT1"
OLLAMA_GLM47_NATIVE_TOOLS_PROVIDER = "ollama-glm47-flash-q4-k-m"
OLLAMA_GLM47_NATIVE_TOOLS_MODEL = "glm-4.7-flash:q4_K_M"
OLLAMA_GLM47_NATIVE_TOOLS_MODEL_DIGEST = (
    "sha256:4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6"
)
OLLAMA_GLM47_NATIVE_TOOLS_SERVER_VERSION = "0.33.2"
OLLAMA_GLM47_NATIVE_TOOLS_CONFIG_BLOB_SHA256 = (
    "sha256:9eba2761cf0b88b8bc11a065a7b5b47f1b13ce820e8e492cb1010b450f9ec950"
)
OLLAMA_GLM47_NATIVE_TOOLS_MODEFILE_SHA256 = (
    "sha256:0d2c0e19ed615b5710421403ae7d824b4ade2e0a20cef20dd69e6a4b14efb5f9"
)
OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT_LENGTH = 32_768
OLLAMA_GLM47_NATIVE_TOOLS_TEMPERATURE = 0.15
OLLAMA_GLM47_NATIVE_TOOLS_SEED = 4242
OLLAMA_GLM47_NATIVE_TOOLS_ROLE_PROBE_MAX_TOKENS = 8192
OLLAMA_GLM47_NATIVE_TOOLS_PROBE_TOOL = "finish_task"
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class Glm47NativeToolsRoleProfileError(ValueError):
    """GLM native-tool evidence is unavailable or does not bind the frozen route."""


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
    return (
        type(value.get("raw_request_hash")) is str
        and _HASH.fullmatch(cast(str, value["raw_request_hash"])) is not None
        and type(value.get("raw_response_hash")) is str
        and _HASH.fullmatch(cast(str, value["raw_response_hash"])) is not None
        and _usage(value.get("usage"))
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
        and parsed.get("tool_name") == OLLAMA_GLM47_NATIVE_TOOLS_PROBE_TOOL
        and all(
            type(parsed.get(key)) is str and _HASH.fullmatch(cast(str, parsed[key])) is not None
            for key in ("arguments_hash", "tool_declaration_hash")
        )
    )


def ollama_glm47_native_tools_profile_projection() -> dict[str, object]:
    return {
        "role_profile": OLLAMA_GLM47_NATIVE_TOOLS_PROFILE,
        "provider": OLLAMA_GLM47_NATIVE_TOOLS_PROVIDER,
        "model": OLLAMA_GLM47_NATIVE_TOOLS_MODEL,
        "model_digest": OLLAMA_GLM47_NATIVE_TOOLS_MODEL_DIGEST,
        "ollama_server_version": OLLAMA_GLM47_NATIVE_TOOLS_SERVER_VERSION,
        "config_blob_sha256": OLLAMA_GLM47_NATIVE_TOOLS_CONFIG_BLOB_SHA256,
        "modelfile_sha256": OLLAMA_GLM47_NATIVE_TOOLS_MODEFILE_SHA256,
        "context_length": OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT_LENGTH,
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
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


def glm47_native_tools_live_show_identity_projection() -> dict[str, object]:
    return {
        "model": OLLAMA_GLM47_NATIVE_TOOLS_MODEL,
        "model_digest": OLLAMA_GLM47_NATIVE_TOOLS_MODEL_DIGEST,
        "config_blob_sha256": OLLAMA_GLM47_NATIVE_TOOLS_CONFIG_BLOB_SHA256,
        "modelfile_sha256": OLLAMA_GLM47_NATIVE_TOOLS_MODEFILE_SHA256,
        "ollama_server_version": OLLAMA_GLM47_NATIVE_TOOLS_SERVER_VERSION,
        "context_length": OLLAMA_GLM47_NATIVE_TOOLS_CONTEXT_LENGTH,
        "renderer": "glm-4.7",
        "parser": "glm-4.7",
        "capabilities": ["completion", "tools", "thinking"],
    }


def validate_glm47_native_tools_live_show_identity(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise Glm47NativeToolsRoleProfileError("GLM identity receipt is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_IDENTITY_KIND,
        **glm47_native_tools_live_show_identity_projection(),
        "show_response_hash": value.get("show_response_hash"),
    }
    if (
        body != expected
        or not isinstance(value.get("show_response_hash"), str)
        or _HASH.fullmatch(cast(str, value["show_response_hash"])) is None
        or value.get("receipt_hash") != sha256_ref(body)
    ):
        raise Glm47NativeToolsRoleProfileError("GLM identity receipt is not bound")
    return cast(dict[str, object], value)


def canonical_glm47_native_tools_live_show_identity_receipt(
    *, show_response: bytes, ollama_version_text: str
) -> bytes:
    try:
        show = json.loads(show_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Glm47NativeToolsRoleProfileError("GLM show response is not JSON") from error
    if type(show) is not dict:
        raise Glm47NativeToolsRoleProfileError("GLM show identity is not pinned")
    modelfile = show.get("modelfile")
    config_match = (
        re.search(r"FROM .*/(sha256-[0-9a-f]{64})", modelfile) if type(modelfile) is str else None
    )
    if (
        show.get("capabilities") != ["completion", "tools", "thinking"]
        or "glm-4.7" not in cast(str, modelfile)
        or config_match is None
        or config_match.group(1).replace("-", ":", 1)
        != OLLAMA_GLM47_NATIVE_TOOLS_CONFIG_BLOB_SHA256
        or "sha256:" + sha256(cast(str, modelfile).encode("utf-8")).hexdigest()
        != OLLAMA_GLM47_NATIVE_TOOLS_MODEFILE_SHA256
        or re.search(r"\b0\.33\.2\b", ollama_version_text) is None
    ):
        raise Glm47NativeToolsRoleProfileError("GLM show identity is not pinned")
    body = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_IDENTITY_KIND,
        **glm47_native_tools_live_show_identity_projection(),
        "show_response_hash": "sha256:" + sha256(show_response).hexdigest(),
    }
    value = {**body, "receipt_hash": sha256_ref(body)}
    validate_glm47_native_tools_live_show_identity(value)
    return canonical_json_bytes(value)


def validate_glm47_native_tools_role_profile_receipt(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise Glm47NativeToolsRoleProfileError("GLM native tools receipt is invalid")
    baseline, developer, tool_call = (
        value.get("baseline"),
        value.get("developer"),
        value.get("tool_call"),
    )
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_RECEIPT_KIND,
        **ollama_glm47_native_tools_profile_projection(),
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
        or cast(dict[str, object], tool_call).get("role_marker") != "SSB_NATIVE_TOOL_CALL_MARKER"
        or cast(dict[str, object], tool_call).get("tool_declaration_hash")
        != value.get("tool_declaration_hash")
        or type(value.get("tool_declaration_hash")) is not str
        or _HASH.fullmatch(cast(str, value["tool_declaration_hash"])) is None
        or type(value.get("developer_prompt_token_delta")) is not int
        or value["developer_prompt_token_delta"] == 0
        or cast(dict[str, object], baseline)["raw_request_hash"]
        == cast(dict[str, object], developer)["raw_request_hash"]
        or cast(dict[str, object], baseline)["raw_response_hash"]
        == cast(dict[str, object], developer)["raw_response_hash"]
    ):
        raise Glm47NativeToolsRoleProfileError("GLM native tools receipt is not bound")
    return cast(dict[str, object], value)


def canonical_glm47_native_tools_role_profile_receipt(
    *, baseline: dict[str, object], developer: dict[str, object], tool_call: dict[str, object]
) -> bytes:
    baseline_usage, developer_usage = baseline.get("usage"), developer.get("usage")
    body = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_RECEIPT_KIND,
        **ollama_glm47_native_tools_profile_projection(),
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
    validate_glm47_native_tools_role_profile_receipt(value)
    return canonical_json_bytes(value)


def canonical_glm47_native_tools_role_profile_failure_receipt(
    *, failure_code: str, completed_calls: int, failed_call: dict[str, object] | None
) -> bytes:
    fields = {
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
        or (
            failed_call is not None
            and (type(failed_call) is not dict or set(failed_call) != fields)
        )
    ):
        raise Glm47NativeToolsRoleProfileError("GLM native tools failure receipt is invalid")
    body = {
        "record_kind": OLLAMA_GLM47_NATIVE_TOOLS_FAILURE_RECEIPT_KIND,
        **ollama_glm47_native_tools_profile_projection(),
        "failure_code": failure_code,
        "completed_calls": completed_calls,
        "failed_call": failed_call,
    }
    return canonical_json_bytes({**body, "receipt_hash": sha256_ref(body)})


def load_glm47_native_tools_role_profile_receipt(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise Glm47NativeToolsRoleProfileError("GLM native tools receipt is unavailable")
    try:
        raw, value = path.read_bytes(), json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Glm47NativeToolsRoleProfileError("GLM native tools receipt is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise Glm47NativeToolsRoleProfileError("GLM native tools receipt is not canonical")
    return validate_glm47_native_tools_role_profile_receipt(value)


def load_glm47_native_tools_live_show_identity(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise Glm47NativeToolsRoleProfileError("GLM identity receipt is unavailable")
    try:
        raw, value = path.read_bytes(), json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Glm47NativeToolsRoleProfileError("GLM identity receipt is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise Glm47NativeToolsRoleProfileError("GLM identity receipt is not canonical")
    return validate_glm47_native_tools_live_show_identity(value)
