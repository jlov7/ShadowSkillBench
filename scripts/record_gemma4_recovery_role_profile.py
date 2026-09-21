"""Record one bounded prospective Gemma native-tool role conformance receipt."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shadowskillbench.core.hashing import sha256_ref  # noqa: E402
from shadowskillbench.experiments.ollama_gemma4_recovery_profile import (  # noqa: E402
    Gemma4RecoveryProfileError,
    RecoveryCandidate,
    canonical_recovery_role_failure,
    canonical_recovery_role_receipt,
    load_recovery_identity,
    recovery_candidate_spec,
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

_NATIVE_TURN_PROMPT_HASH = "sha256:f8d870e824bee81ef715dcd88210a03c34d0a747f7f7ec29e26a30754aa0172e"


def _endpoint(value: str) -> str:
    if value != "http://127.0.0.1:11434/api/chat":
        raise Gemma4RecoveryProfileError("endpoint is not the pinned loopback /api/chat")
    return value


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


def _call(response: NativeToolResponse, marker: str) -> dict[str, object]:
    return {
        "arguments_hash": sha256_ref(response.arguments),
        "attempts": response.attempts,
        "finish_reason": response.finish_reason,
        "provider_done": response.provider_done,
        "provider_finish_reason": response.provider_finish_reason,
        "raw_request_hash": response.raw_request_hash,
        "raw_response_hash": response.raw_response_hash,
        "role_marker": marker,
        "tool_declaration_hash": response.tool_declaration_hash,
        "tool_name": response.tool_name,
        "usage": response.usage.model_dump(mode="json"),
    }


async def _record(candidate: RecoveryCandidate, endpoint: str, api_key_environment: str) -> bytes:
    spec = recovery_candidate_spec(candidate)
    client = ollama_native_tool_client_from_run_descriptor(
        ModelRunDescriptor(
            provider="ollama",
            model=spec.model,
            model_version=spec.model_digest,
            endpoint=_endpoint(endpoint),
            api_key_environment=api_key_environment,
            max_attempts=1,
            supports_structured_output=False,
            executor_runtime_profile={
                "request_profile": "SSB-OLLAMA-NATIVE-TOOLS3",
                "response_contract": "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
                "endpoint_path": "/api/chat",
                "context_length": 32768,
                "top_p_wire": "omitted",
                "reasoning_effort_wire": "omitted",
                "history_projection": "SSB-OLLAMA-NATIVE-TOOLS-HISTORY1",
                "native_turn_prompt_profile": "SSB-OLLAMA-NATIVE-TOOLS-TURN-PROMPT1",
                "native_turn_prompt_hash": _NATIVE_TURN_PROMPT_HASH,
            },
        ),
        trust_env=False,
        timeout_seconds=900.0,
    )
    system = Message(role="system", content="SSB_NATIVE_SYSTEM_ROLE_MARKER")
    developer = Message(role="developer", content="SSB_NATIVE_DEVELOPER_ROLE_MARKER")
    user = Message(
        role="user",
        content='Call exactly one finish_task tool with summary exactly "native tool conformance".',
    )
    responses: list[NativeToolResponse] = []
    for marker, request in (
        (
            "baseline",
            ModelRequest(messages=(system, user), temperature=1.0, seed=4242, max_tokens=16384),
        ),
        (
            "developer",
            ModelRequest(
                messages=(system, developer, user), temperature=1.0, seed=4242, max_tokens=16384
            ),
        ),
        (
            "tool_call",
            ModelRequest(
                messages=(system, developer, user), temperature=1.0, seed=4242, max_tokens=16384
            ),
        ),
    ):
        try:
            response = await client.call_tools(request, _tools())
        except ModelAdapterError as error:
            return canonical_recovery_role_failure(
                candidate=candidate, failure_code=error.code, completed_calls=len(responses)
            )
        if (
            response.tool_name != "finish_task"
            or response.arguments != {"summary": "native tool conformance"}
            or response.finish_reason != "tool_calls"
            or response.provider_done is not True
            or response.attempts != 1
        ):
            return canonical_recovery_role_failure(
                candidate=candidate,
                failure_code="MODEL_OUTPUT_INVALID",
                completed_calls=len(responses),
            )
        responses.append(response)
    return canonical_recovery_role_receipt(
        candidate=candidate,
        baseline=_call(responses[0], "baseline"),
        developer=_call(responses[1], "developer"),
        tool_call=_call(responses[2], "tool_call"),
    )


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise Gemma4RecoveryProfileError("role output is unsafe")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise Gemma4RecoveryProfileError("role output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _reserve_once(path: Path, candidate: str) -> None:
    if path.is_symlink() or path.exists():
        raise Gemma4RecoveryProfileError("role invocation already reserved")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(f"candidate={candidate}\n".encode())
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("primary-26b", "fallback-12b"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key-environment", required=True)
    parser.add_argument("--identity-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        print("HOLD_GEMMA4_RECOVERY_ROLE_PROFILE: output already reserved")
        return 1
    try:
        load_recovery_identity(arguments.identity_receipt, arguments.candidate)
        _reserve_once(
            arguments.output.with_suffix(arguments.output.suffix + ".invocation"),
            arguments.candidate,
        )
        payload = asyncio.run(
            _record(arguments.candidate, arguments.endpoint, arguments.api_key_environment)
        )
        _write_once(arguments.output, payload)
    except (Gemma4RecoveryProfileError, RunDescriptorError, ValueError):
        print(
            "HOLD_GEMMA4_RECOVERY_ROLE_PROFILE: configuration "
            f"wall_seconds={time.monotonic() - started:.6f}"
        )
        return 1
    if b'"record_kind":"OLLAMA_GEMMA4_RECOVERY_ROLE_TOOL_CONFORMANCE_RECEIPT1"' not in payload:
        print(
            "HOLD_GEMMA4_RECOVERY_ROLE_PROFILE: failure receipt written "
            f"wall_seconds={time.monotonic() - started:.6f}"
        )
        return 1
    print(
        "PASS_GEMMA4_RECOVERY_ROLE_PROFILE: "
        f"candidate={arguments.candidate} output={arguments.output} "
        f"wall_seconds={time.monotonic() - started:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
