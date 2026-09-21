"""Run one frozen, hash-only native-tools matrix candidate."""

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
from shadowskillbench.experiments.native_tools_conformance_matrix import (  # noqa: E402
    CANDIDATES,
    Candidate,
    MatrixConformanceError,
    canonical_failure,
    canonical_receipt,
    load_candidate,
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

_SUMMARY = "native tool conformance"
_PROMPT = 'Call exactly one finish_task tool with summary exactly "native tool conformance".'


def _tools() -> tuple[NativeToolDefinition, ...]:
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


def _endpoint(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.path != "/api/chat"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise MatrixConformanceError("endpoint")
    return value


def _call(response: NativeToolResponse, marker: str) -> dict[str, object]:
    return {
        "raw_request_hash": response.raw_request_hash,
        "raw_response_hash": response.raw_response_hash,
        "usage": response.usage.model_dump(mode="json"),
        "attempts": response.attempts,
        "finish_reason": response.finish_reason,
        "provider_finish_reason": response.provider_finish_reason,
        "provider_done": response.provider_done,
        "role_marker": marker,
        "tool_name": response.tool_name,
        "arguments_hash": sha256_ref(response.arguments),
        "tool_declaration_hash": response.tool_declaration_hash,
    }


def _error(error: ModelAdapterError) -> dict[str, object]:
    return {
        "code": error.code,
        "raw_request_hash": error.raw_request_hash,
        "raw_response_hash": error.raw_response_hash,
        "usage": None
        if error.reported_usage is None
        else error.reported_usage.model_dump(mode="json"),
        "attempts": error.attempts,
        "provider_finish_reason": error.finish_reason,
        "provider_done": error.provider_done,
    }


async def _record(candidate: Candidate, endpoint: str, api_key_environment: str) -> bytes:
    item = CANDIDATES[candidate]
    temperature = item["temperature"]
    if type(temperature) is not float:
        raise MatrixConformanceError("temperature")
    client = ollama_native_tool_client_from_run_descriptor(
        ModelRunDescriptor(
            provider="ollama",
            model=str(item["model"]),
            model_version=str(item["digest"]),
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
    user = Message(role="user", content=_PROMPT)
    requests = (
        ModelRequest(
            messages=(system, user),
            temperature=temperature,
            seed=4242,
            max_tokens=8192,
        ),
        ModelRequest(
            messages=(system, developer, user),
            temperature=temperature,
            seed=4242,
            max_tokens=8192,
        ),
        ModelRequest(
            messages=(system, developer, user),
            temperature=temperature,
            seed=4242,
            max_tokens=8192,
        ),
    )
    markers = ("baseline", "developer", "tool_call")
    calls: list[dict[str, object]] = []
    for request, marker in zip(requests, markers, strict=True):
        try:
            response = await client.call_tools(request, _tools())
        except ModelAdapterError as error:
            return canonical_failure(
                candidate,
                failure_type="MODEL_OUTPUT_INVALID",
                stage=marker,
                completed_calls=len(calls),
                calls=[*calls, _error(error)],
            )
        call = _call(response, marker)
        if (
            response.tool_name != "finish_task"
            or response.arguments != {"summary": _SUMMARY}
            or response.finish_reason != "tool_calls"
            or response.provider_done is not True
        ):
            return canonical_failure(
                candidate,
                failure_type="MODEL_OUTPUT_INVALID",
                stage=marker,
                completed_calls=len(calls),
                calls=[*calls, call],
            )
        calls.append(call)
    try:
        return canonical_receipt(candidate, calls=calls)
    except MatrixConformanceError:
        return canonical_failure(
            candidate,
            failure_type="ROLE_CONFORMANCE_INVALID",
            stage="receipt",
            completed_calls=3,
            calls=calls,
        )


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise MatrixConformanceError("output")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise MatrixConformanceError("output exists")
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
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key-environment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        candidate = load_candidate(args.candidate)
        payload = asyncio.run(_record(candidate, args.endpoint, args.api_key_environment))
        _write_once(args.output, payload)
    except (MatrixConformanceError, RunDescriptorError, OSError, ValueError):
        return 1
    if b'"record_kind":"OLLAMA_NATIVE_TOOLS_MATRIX_ROLE_TOOL_CONFORMANCE_RECEIPT1"' in payload:
        print(f"PASS_NATIVE_TOOLS_MATRIX: candidate={candidate} output={args.output}")
        return 0
    print(f"HOLD_NATIVE_TOOLS_MATRIX: candidate={candidate} output={args.output}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
