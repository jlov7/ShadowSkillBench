"""Record one bounded, hash-only native-tool role/conformance receipt."""

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
from shadowskillbench.experiments.ollama_nemotron_context32k_profile import (  # noqa: E402
    OLLAMA_NEMOTRON_CONTEXT32K_MODEL,
    OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST,
    OLLAMA_NEMOTRON_CONTEXT32K_PROVIDER,
)
from shadowskillbench.experiments.ollama_nemotron_native_tools_profile import (  # noqa: E402
    NemotronNativeToolsRoleProfileError,
    canonical_nemotron_native_tools_role_profile_failure_receipt,
    canonical_nemotron_native_tools_role_profile_receipt,
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
_SEED = 4242
_TEMPERATURE = 0.15
_SUMMARY = "native tool conformance"
_FAILURE_CODES = frozenset(
    {
        "CONFIGURATION_ERROR",
        "MODEL_OUTPUT_INVALID",
        "MODEL_PROVIDER_TERMINAL",
        "MODEL_PROVIDER_TRANSIENT",
    }
)


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
    return ModelRequest(
        messages=messages,
        temperature=_TEMPERATURE,
        seed=_SEED,
        max_tokens=_MAX_TOKENS,
    )


def _loopback_chat_endpoint(endpoint: str) -> str:
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
        raise NemotronNativeToolsRoleProfileError(
            "native tools endpoint is not pinned loopback /api/chat"
        )
    return endpoint


def _safe_call(
    *,
    raw_request_hash: str | None,
    raw_response_hash: str | None,
    usage: dict[str, int] | None,
    attempts: int,
    finish_reason: str | None,
) -> dict[str, object]:
    return {
        "raw_request_hash": raw_request_hash,
        "raw_response_hash": raw_response_hash,
        "usage": usage,
        "attempts": attempts,
        "finish_reason": finish_reason,
        "output_cap_exhausted": (None if usage is None else usage["output_tokens"] >= _MAX_TOKENS),
    }


def _role_call(response: NativeToolResponse, marker: str) -> dict[str, object]:
    return {
        "raw_request_hash": response.raw_request_hash,
        "raw_response_hash": response.raw_response_hash,
        "usage": response.usage.model_dump(mode="json"),
        "attempts": response.attempts,
        "finish_reason": response.finish_reason,
        "role_marker": marker,
    }


def _tool_call(response: NativeToolResponse) -> dict[str, object]:
    return {
        **_role_call(response, "SSB_NATIVE_TOOL_CALL_MARKER"),
        "tool_name": response.tool_name,
        "arguments_hash": sha256_ref(response.arguments),
        "tool_declaration_hash": response.tool_declaration_hash,
    }


def _valid(response: NativeToolResponse) -> bool:
    return (
        response.tool_name == "finish_task"
        and response.arguments == {"summary": _SUMMARY}
        and response.finish_reason == "tool_calls"
        and response.attempts == 1
    )


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise NemotronNativeToolsRoleProfileError("native tools receipt output path is unsafe")
    if path.exists():
        try:
            if path.read_bytes() == payload:
                return
        except OSError as error:
            raise NemotronNativeToolsRoleProfileError(
                "native tools receipt output is unreadable"
            ) from error
        raise NemotronNativeToolsRoleProfileError("native tools receipt output already exists")
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except OSError as error:
        raise NemotronNativeToolsRoleProfileError("native tools receipt write failed") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def _record(endpoint: str, api_key_environment: str) -> bytes:
    endpoint = _loopback_chat_endpoint(endpoint)
    client = ollama_native_tool_client_from_run_descriptor(
        ModelRunDescriptor(
            provider=OLLAMA_NEMOTRON_CONTEXT32K_PROVIDER,
            model=OLLAMA_NEMOTRON_CONTEXT32K_MODEL,
            model_version=OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST,
            endpoint=endpoint,
            api_key_environment=api_key_environment,
            max_attempts=1,
            supports_structured_output=False,
            executor_runtime_profile={
                "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
                "endpoint_path": "/api/chat",
                "context_length": 32768,
                "top_p_wire": "omitted",
                "reasoning_effort_wire": "omitted",
            },
        ),
        trust_env=False,
        timeout_seconds=900.0,
    )
    calls: list[NativeToolResponse] = []
    requests = (
        _request(
            Message(role="system", content="SSB_NATIVE_SYSTEM_ROLE_MARKER"),
            Message(role="user", content="Call finish_task with the required conformance summary."),
        ),
        _request(
            Message(role="system", content="SSB_NATIVE_SYSTEM_ROLE_MARKER"),
            Message(role="developer", content="SSB_NATIVE_DEVELOPER_ROLE_MARKER"),
            Message(role="user", content="Call finish_task with the required conformance summary."),
        ),
        _request(
            Message(role="system", content="SSB_NATIVE_SYSTEM_ROLE_MARKER"),
            Message(role="developer", content="SSB_NATIVE_DEVELOPER_ROLE_MARKER"),
            Message(
                role="user",
                content="Call exactly one finish_task tool with the required conformance summary.",
            ),
        ),
    )
    try:
        for request in requests:
            response = await client.call_tools(request, _definitions())
            if not _valid(response):
                return canonical_nemotron_native_tools_role_profile_failure_receipt(
                    failure_code="MODEL_OUTPUT_INVALID",
                    completed_calls=len(calls),
                    failed_call=_safe_call(
                        raw_request_hash=response.raw_request_hash,
                        raw_response_hash=response.raw_response_hash,
                        usage=response.usage.model_dump(mode="json"),
                        attempts=response.attempts,
                        finish_reason=response.finish_reason,
                    ),
                )
            calls.append(response)
    except ModelAdapterError as error:
        return canonical_nemotron_native_tools_role_profile_failure_receipt(
            failure_code=error.code,
            completed_calls=len(calls),
            failed_call=_safe_call(
                raw_request_hash=error.raw_request_hash,
                raw_response_hash=error.raw_response_hash,
                usage=(
                    None
                    if error.reported_usage is None
                    else error.reported_usage.model_dump(mode="json")
                ),
                attempts=error.attempts,
                finish_reason=error.finish_reason,
            ),
        )
    return canonical_nemotron_native_tools_role_profile_receipt(
        baseline=_role_call(calls[0], "SSB_NATIVE_SYSTEM_ROLE_MARKER"),
        developer=_role_call(calls[1], "SSB_NATIVE_DEVELOPER_ROLE_MARKER"),
        tool_call=_tool_call(calls[2]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key-environment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    failure_code = "CONFIGURATION_ERROR"
    try:
        payload = asyncio.run(_record(arguments.endpoint, arguments.api_key_environment))
        _write_once(arguments.output, payload)
    except (NemotronNativeToolsRoleProfileError, RunDescriptorError, ValueError):
        try:
            _write_once(
                arguments.output,
                canonical_nemotron_native_tools_role_profile_failure_receipt(
                    failure_code=failure_code,
                    completed_calls=0,
                    failed_call=None,
                ),
            )
        except NemotronNativeToolsRoleProfileError:
            pass
        print("HOLD_NEMOTRON_NATIVE_TOOLS_ROLE_PROFILE: CONFIGURATION_ERROR")
        return 1
    if (
        b'"record_kind":"OLLAMA_NEMOTRON_NATIVE_TOOLS_CONTEXT32K_ROLE_TOOL_CONFORMANCE_RECEIPT1"'
        not in payload
    ):
        print("HOLD_NEMOTRON_NATIVE_TOOLS_ROLE_PROFILE: failure receipt written")
        return 1
    print(f"PASS_NEMOTRON_NATIVE_TOOLS_ROLE_PROFILE: output={arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
