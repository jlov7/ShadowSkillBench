"""Frozen Gemma 4 native-tools identity and role/tool conformance receipts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref

OLLAMA_GEMMA4_NATIVE_TOOLS_PROFILE = (
    "SSB-OLLAMA-GEMMA4-12B-IT-Q4-K-M-NATIVE-TOOLS-CONTEXT32768-ROLES1"
)
OLLAMA_GEMMA4_NATIVE_TOOLS_RECEIPT_KIND = (
    "OLLAMA_NATIVE_TOOLS_MATRIX_ROLE_TOOL_CONFORMANCE_RECEIPT1"
)
OLLAMA_GEMMA4_NATIVE_TOOLS_IDENTITY_KIND = "OLLAMA_NATIVE_TOOLS_MATRIX_IDENTITY_RECEIPT1"
OLLAMA_GEMMA4_NATIVE_TOOLS_PROVIDER = "ollama"
OLLAMA_GEMMA4_NATIVE_TOOLS_MODEL = "gemma4:12b-it-q4_K_M"
OLLAMA_GEMMA4_NATIVE_TOOLS_MODEL_DIGEST = (
    "sha256:4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
)
OLLAMA_GEMMA4_NATIVE_TOOLS_SERVER_VERSION = "0.33.2"
OLLAMA_GEMMA4_NATIVE_TOOLS_CONFIG_BLOB_SHA256 = (
    "sha256:1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606"
)
OLLAMA_GEMMA4_NATIVE_TOOLS_MODEFILE_SHA256 = (
    "sha256:c6749ea4267cda459b08c1a04c95cf733f148fba470f8aa07070bbfba1599fbf"
)
OLLAMA_GEMMA4_NATIVE_TOOLS_CONTEXT_LENGTH = 32_768
OLLAMA_GEMMA4_NATIVE_TOOLS_TEMPERATURE = 1.0
OLLAMA_GEMMA4_NATIVE_TOOLS_SEED = 4242
OLLAMA_GEMMA4_NATIVE_TOOLS_MAX_OUTPUT_TOKENS = 8192
OLLAMA_GEMMA4_NATIVE_TOOLS_REQUEST_PROFILE = "SSB-OLLAMA-NATIVE-TOOLS1"
OLLAMA_GEMMA4_NATIVE_TOOLS_RESPONSE_CONTRACT = "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2"
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class Gemma4NativeToolsProfileError(ValueError):
    """Gemma native-tools evidence is unavailable or does not bind the route."""


def _usage(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"input_tokens", "output_tokens", "total_tokens"}
        and all(type(value[key]) is int and value[key] >= 0 for key in value)
        and value["total_tokens"] == value["input_tokens"] + value["output_tokens"]
    )


def _tool_call(value: object, marker: str) -> bool:
    fields = {
        "arguments_hash",
        "attempts",
        "finish_reason",
        "provider_done",
        "provider_finish_reason",
        "raw_request_hash",
        "raw_response_hash",
        "role_marker",
        "tool_declaration_hash",
        "tool_name",
        "usage",
    }
    if type(value) is not dict or set(value) != fields:
        return False
    return (
        all(
            type(value.get(key)) is str and _HASH.fullmatch(cast(str, value[key]))
            for key in (
                "arguments_hash",
                "raw_request_hash",
                "raw_response_hash",
                "tool_declaration_hash",
            )
        )
        and value.get("attempts") == 1
        and value.get("finish_reason") == "tool_calls"
        and value.get("provider_done") is True
        and value.get("provider_finish_reason") == "stop"
        and value.get("role_marker") == marker
        and value.get("tool_name") == "finish_task"
        and _usage(value.get("usage"))
    )


def ollama_gemma4_native_tools_projection() -> dict[str, object]:
    return {
        "candidate": "gemma4-12b-it-q4-k-m",
        "config_blob_sha256": OLLAMA_GEMMA4_NATIVE_TOOLS_CONFIG_BLOB_SHA256,
        "context_length": OLLAMA_GEMMA4_NATIVE_TOOLS_CONTEXT_LENGTH,
        "max_output_tokens": OLLAMA_GEMMA4_NATIVE_TOOLS_MAX_OUTPUT_TOKENS,
        "model": OLLAMA_GEMMA4_NATIVE_TOOLS_MODEL,
        "model_digest": OLLAMA_GEMMA4_NATIVE_TOOLS_MODEL_DIGEST,
        "modelfile_sha256": OLLAMA_GEMMA4_NATIVE_TOOLS_MODEFILE_SHA256,
        "ollama_server_version": OLLAMA_GEMMA4_NATIVE_TOOLS_SERVER_VERSION,
        "provider": OLLAMA_GEMMA4_NATIVE_TOOLS_PROVIDER,
        "reasoning_effort_wire": "omitted",
        "request_profile": OLLAMA_GEMMA4_NATIVE_TOOLS_REQUEST_PROFILE,
        "response_contract": OLLAMA_GEMMA4_NATIVE_TOOLS_RESPONSE_CONTRACT,
        "role_profile": OLLAMA_GEMMA4_NATIVE_TOOLS_PROFILE,
        "seed": OLLAMA_GEMMA4_NATIVE_TOOLS_SEED,
        "temperature": OLLAMA_GEMMA4_NATIVE_TOOLS_TEMPERATURE,
        "top_p_wire": "omitted",
    }


def validate_gemma4_native_tools_role_profile_receipt(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise Gemma4NativeToolsProfileError("Gemma native-tools receipt is invalid")
    baseline = value.get("baseline")
    developer = value.get("developer")
    tool_call = value.get("tool_call")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_GEMMA4_NATIVE_TOOLS_RECEIPT_KIND,
        **ollama_gemma4_native_tools_projection(),
        "baseline": baseline,
        "developer": developer,
        "developer_prompt_token_delta": value.get("developer_prompt_token_delta"),
        "tool_call": tool_call,
    }
    if (
        body != expected
        or value.get("receipt_hash") != sha256_ref(body)
        or not _tool_call(baseline, "baseline")
        or not _tool_call(developer, "developer")
        or not _tool_call(tool_call, "tool_call")
        or cast(dict[str, object], baseline).get("arguments_hash")
        != cast(dict[str, object], developer).get("arguments_hash")
        or cast(dict[str, object], baseline).get("arguments_hash")
        != cast(dict[str, object], tool_call).get("arguments_hash")
        or cast(dict[str, object], baseline).get("tool_declaration_hash")
        != cast(dict[str, object], developer).get("tool_declaration_hash")
        or cast(dict[str, object], developer).get("tool_declaration_hash")
        != cast(dict[str, object], tool_call).get("tool_declaration_hash")
        or type(value.get("developer_prompt_token_delta")) is not int
        or value["developer_prompt_token_delta"]
        != cast(dict[str, int], cast(dict[str, object], developer)["usage"])["input_tokens"]
        - cast(dict[str, int], cast(dict[str, object], baseline)["usage"])["input_tokens"]
        or value["developer_prompt_token_delta"] == 0
    ):
        raise Gemma4NativeToolsProfileError("Gemma native-tools receipt is not bound")
    return cast(dict[str, object], value)


def validate_gemma4_native_tools_identity(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise Gemma4NativeToolsProfileError("Gemma native-tools identity is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_GEMMA4_NATIVE_TOOLS_IDENTITY_KIND,
        **ollama_gemma4_native_tools_projection(),
        "capabilities": ["completion", "vision", "audio", "tools", "thinking"],
        "model_context": 262144,
        "show_response_hash": value.get("show_response_hash"),
    }
    if (
        body != expected
        or value.get("receipt_hash") != sha256_ref(body)
        or type(value.get("show_response_hash")) is not str
        or _HASH.fullmatch(cast(str, value["show_response_hash"])) is None
    ):
        raise Gemma4NativeToolsProfileError("Gemma native-tools identity is not bound")
    return cast(dict[str, object], value)


def _load(path: Path, validator, label: str) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise Gemma4NativeToolsProfileError(f"Gemma native-tools {label} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma4NativeToolsProfileError(f"Gemma native-tools {label} is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise Gemma4NativeToolsProfileError(f"Gemma native-tools {label} is not canonical")
    return validator(value)


def load_gemma4_native_tools_role_profile_receipt(path: Path) -> dict[str, object]:
    return _load(path, validate_gemma4_native_tools_role_profile_receipt, "role receipt")


def load_gemma4_native_tools_identity(path: Path) -> dict[str, object]:
    return _load(path, validate_gemma4_native_tools_identity, "identity receipt")


__all__ = [
    "Gemma4NativeToolsProfileError",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_CONFIG_BLOB_SHA256",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_CONTEXT_LENGTH",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_IDENTITY_KIND",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_MODEL",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_MODEL_DIGEST",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_PROFILE",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_PROVIDER",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_RECEIPT_KIND",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_SEED",
    "OLLAMA_GEMMA4_NATIVE_TOOLS_TEMPERATURE",
    "load_gemma4_native_tools_identity",
    "load_gemma4_native_tools_role_profile_receipt",
    "ollama_gemma4_native_tools_projection",
    "validate_gemma4_native_tools_identity",
    "validate_gemma4_native_tools_role_profile_receipt",
]
