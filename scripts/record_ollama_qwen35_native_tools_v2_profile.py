"""Record one bounded, hash-only Qwen native-tool role/tool receipt."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shadowskillbench.core.hashing import sha256_ref  # noqa: E402
from shadowskillbench.experiments.ollama_qwen35_native_tools_v2_profile import (  # noqa: E402
    OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL,
    OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL_DIGEST,
    OLLAMA_QWEN35_NATIVE_TOOLS_V2_PROVIDER,
    Qwen35NativeToolsV2ProfileError,
    canonical_qwen35_native_tools_v2_failure_receipt,
    canonical_qwen35_native_tools_v2_receipt,
)
from shadowskillbench.models import Message, ModelAdapterError, ModelRequest  # noqa: E402
from shadowskillbench.models.native_tools import (  # noqa: E402
    NativeToolDefinition,
    NativeToolResponse,
)
from shadowskillbench.models.runtime import (  # noqa: E402
    ModelRunDescriptor,
    RunDescriptorError,
    ollama_native_tool_client_from_run_descriptor,
)

_MAX_TOKENS = 8192
_SUMMARY = "native tool conformance"


def _definitions() -> tuple[NativeToolDefinition, ...]:
    return (
        NativeToolDefinition(
            name="finish_task",
            description="Finish the task with a concise summary.",
            parameters={
                "type": "object",
                "properties": {"summary": {}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
    )


def _request(*messages: Message) -> ModelRequest:
    return ModelRequest(messages=messages, temperature=0.15, seed=4242, max_tokens=_MAX_TOKENS)


def _endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint) if type(endpoint) is str else None
    if (
        parsed is None
        or parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.path != "/api/chat"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise Qwen35NativeToolsV2ProfileError("endpoint is not pinned loopback /api/chat")
    return endpoint


def _failure(response: NativeToolResponse | ModelAdapterError) -> dict[str, object]:
    usage = response.usage if isinstance(response, NativeToolResponse) else response.reported_usage
    provider_done = (
        response.provider_done
        if isinstance(response, NativeToolResponse)
        else response.provider_done
    )
    provider_reason = (
        response.provider_finish_reason
        if isinstance(response, NativeToolResponse)
        else response.finish_reason
    )
    valid_name = (
        response.tool_name == "finish_task" if isinstance(response, NativeToolResponse) else None
    )
    valid_arguments = (
        sha256_ref(response.arguments) == sha256_ref({"summary": _SUMMARY})
        if isinstance(response, NativeToolResponse)
        else None
    )
    return {
        "raw_request_hash": response.raw_request_hash,
        "raw_response_hash": response.raw_response_hash,
        "usage": None if usage is None else usage.model_dump(mode="json"),
        "attempts": response.attempts,
        "provider_finish_reason": provider_reason,
        "provider_done": provider_done,
        "valid_tool_name": valid_name,
        "valid_arguments_hash_match": valid_arguments,
        "output_cap_exhausted": False if usage is None else usage.output_tokens >= _MAX_TOKENS,
    }


def _role_call(response: NativeToolResponse, marker: str) -> dict[str, object]:
    return {
        "raw_request_hash": response.raw_request_hash,
        "raw_response_hash": response.raw_response_hash,
        "usage": response.usage.model_dump(mode="json"),
        "attempts": response.attempts,
        "finish_reason": response.finish_reason,
        "provider_finish_reason": response.provider_finish_reason,
        "provider_done": response.provider_done,
        "role_marker": marker,
    }


async def _record(endpoint: str, api_key_environment: str) -> bytes:
    client = ollama_native_tool_client_from_run_descriptor(
        ModelRunDescriptor(
            provider=OLLAMA_QWEN35_NATIVE_TOOLS_V2_PROVIDER,
            model=OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL,
            model_version=OLLAMA_QWEN35_NATIVE_TOOLS_V2_MODEL_DIGEST,
            endpoint=_endpoint(endpoint),
            api_key_environment=api_key_environment,
            max_attempts=1,
            supports_structured_output=False,
            executor_runtime_profile={
                "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
                "endpoint_path": "/api/chat",
                "context_length": 32768,
                "top_p_wire": "omitted",
                "reasoning_effort_wire": "omitted",
                "response_contract": "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
            },
        ),
        trust_env=False,
        timeout_seconds=900.0,
    )
    system = Message(role="system", content="SSB_NATIVE_SYSTEM_ROLE_MARKER")
    developer = Message(role="developer", content="SSB_NATIVE_DEVELOPER_ROLE_MARKER")
    prompt = 'Call exactly one finish_task tool with summary exactly "native tool conformance".'
    user = Message(role="user", content=prompt)
    tool_user = Message(role="user", content=prompt)
    calls: list[NativeToolResponse] = []
    for request in (
        _request(system, user),
        _request(system, developer, user),
        _request(system, developer, tool_user),
    ):
        try:
            response = await client.call_tools(request, _definitions())
        except ModelAdapterError as error:
            return canonical_qwen35_native_tools_v2_failure_receipt(
                failure_code=error.code, completed_calls=len(calls), failed_call=_failure(error)
            )
        if (
            response.tool_name != "finish_task"
            or response.arguments != {"summary": _SUMMARY}
            or response.finish_reason != "tool_calls"
            or response.provider_done is not True
            or response.attempts != 1
        ):
            return canonical_qwen35_native_tools_v2_failure_receipt(
                failure_code="MODEL_OUTPUT_INVALID",
                completed_calls=len(calls),
                failed_call=_failure(response),
            )
        calls.append(response)
    if calls[1].usage.input_tokens == calls[0].usage.input_tokens:
        return canonical_qwen35_native_tools_v2_failure_receipt(
            failure_code="ROLE_CONFORMANCE_INVALID",
            completed_calls=3,
            failed_call=_failure(calls[2]),
        )
    return canonical_qwen35_native_tools_v2_receipt(
        baseline=_role_call(calls[0], "SSB_NATIVE_SYSTEM_ROLE_MARKER"),
        developer=_role_call(calls[1], "SSB_NATIVE_DEVELOPER_ROLE_MARKER"),
        tool_call={
            **_role_call(calls[2], "SSB_NATIVE_TOOL_CALL_MARKER"),
            "tool_name": calls[2].tool_name,
            "arguments_hash": sha256_ref(calls[2].arguments),
            "tool_declaration_hash": calls[2].tool_declaration_hash,
        },
    )


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise Qwen35NativeToolsV2ProfileError("unsafe receipt output")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise Qwen35NativeToolsV2ProfileError("receipt output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key-environment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        payload = asyncio.run(_record(arguments.endpoint, arguments.api_key_environment))
        _write_once(arguments.output, payload)
    except (Qwen35NativeToolsV2ProfileError, RunDescriptorError, ValueError):
        _write_once(
            arguments.output,
            canonical_qwen35_native_tools_v2_failure_receipt(
                failure_code="CONFIGURATION_ERROR", completed_calls=0, failed_call=None
            ),
        )
        print("HOLD_QWEN35_NATIVE_TOOLS_V2_PROFILE: CONFIGURATION_ERROR")
        return 1
    if (
        b'"record_kind":"OLLAMA_QWEN35_NATIVE_TOOLS_V2_CONTEXT32K_ROLE_TOOL_CONFORMANCE_RECEIPT2"'
        not in payload
    ):
        print("HOLD_QWEN35_NATIVE_TOOLS_V2_PROFILE: failure receipt written")
        return 1
    print(f"PASS_QWEN35_NATIVE_TOOLS_V2_PROFILE: output={arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
