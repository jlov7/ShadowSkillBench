from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from shadowskillbench.episodes.native_tool_turn_wire import NATIVE_TOOL_TURN_PROMPT
from shadowskillbench.models import (
    Message,
    ModelAdapterError,
    ModelRequest,
    OllamaNativeToolClient,
    ProviderCapabilities,
    TransportResponse,
)
from shadowskillbench.models.native_tools import (
    NativeToolDefinition,
    NativeToolHistoryTurn,
    native_tool_declaration_hash,
)


def _capabilities() -> ProviderCapabilities:
    return ProviderCapabilities(
        provider="ollama",
        model="model-test",
        model_version="digest-test",
        supports_system_role=True,
        supports_developer_role=True,
        supports_seed=True,
        supports_structured_output=False,
    )


def _request() -> ModelRequest:
    return ModelRequest(
        messages=(
            Message(role="system", content="system marker"),
            Message(role="developer", content="developer marker"),
            Message(role="user", content="call one tool"),
        ),
        temperature=0.15,
        seed=4242,
        max_tokens=8192,
    )


def _tools() -> tuple[NativeToolDefinition, ...]:
    return (
        NativeToolDefinition(
            name="finish_task",
            description="Finish the task.",
            parameters={
                "type": "object",
                "properties": {"summary": {}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
    )


def _body(*, calls: object) -> bytes:
    return json.dumps(
        {
            "message": {"role": "assistant", "content": "untrusted", "tool_calls": calls},
            "done": True,
            "done_reason": "tool_calls",
            "prompt_eval_count": 2,
            "eval_count": 3,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_native_tools_request_is_exact_and_omits_structured_wire() -> None:
    bodies: list[bytes] = []

    async def transport(body: bytes) -> TransportResponse:
        bodies.append(body)
        return TransportResponse(
            status_code=200,
            body=_body(
                calls=[{"function": {"name": "finish_task", "arguments": {"summary": "ok"}}}]
            ),
        )

    response = asyncio.run(
        OllamaNativeToolClient(
            capabilities=_capabilities(), transport=transport, max_attempts=1, context_length=32768
        ).call_tools(_request(), _tools())
    )

    expected = json.dumps(
        {
            "messages": [
                {"content": "system marker", "role": "system"},
                {"content": "developer marker", "role": "developer"},
                {"content": "call one tool", "role": "user"},
            ],
            "model": "model-test",
            "options": {"num_ctx": 32768, "num_predict": 8192, "seed": 4242, "temperature": 0.15},
            "stream": False,
            "tools": [tool.projection() for tool in _tools()],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert bodies == [expected]
    assert b'"response_format"' not in expected
    assert b'"top_p"' not in expected
    assert b'"reasoning_effort"' not in expected
    assert response.raw_request_hash == "sha256:" + hashlib.sha256(expected).hexdigest()
    assert response.tool_declaration_hash == native_tool_declaration_hash(_tools())
    assert response.tool_name == "finish_task"


def test_native_tools2_projects_validated_history_before_final_observation() -> None:
    bodies: list[bytes] = []

    async def transport(body: bytes) -> TransportResponse:
        bodies.append(body)
        return TransportResponse(
            status_code=200,
            body=_body(
                calls=[{"function": {"name": "finish_task", "arguments": {"summary": "ok"}}}]
            ),
        )

    history = (
        NativeToolHistoryTurn(
            tool_name="lookup",
            arguments={"id": "42"},
            result={"local_status": "success", "value": "answer"},
        ),
        NativeToolHistoryTurn(
            tool_name="finish_task",
            arguments={"summary": "prior"},
            result={"local_status": "success"},
        ),
    )
    response = asyncio.run(
        OllamaNativeToolClient(
            capabilities=_capabilities(),
            transport=transport,
            max_attempts=1,
            context_length=32768,
            request_profile="SSB-OLLAMA-NATIVE-TOOLS2",
            response_contract="SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
        ).call_tools(_request(), _tools(), history=history)
    )

    expected = json.dumps(
        {
            "messages": [
                {"content": "system marker", "role": "system"},
                {"content": "developer marker", "role": "developer"},
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"arguments": {"id": "42"}, "name": "lookup"}}],
                },
                {
                    "content": '{"local_status":"success","value":"answer"}',
                    "role": "tool",
                    "tool_name": "lookup",
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"arguments": {"summary": "prior"}, "name": "finish_task"}}
                    ],
                },
                {
                    "content": '{"local_status":"success"}',
                    "role": "tool",
                    "tool_name": "finish_task",
                },
                {"content": "call one tool", "role": "user"},
            ],
            "model": "model-test",
            "options": {"num_ctx": 32768, "num_predict": 8192, "seed": 4242, "temperature": 0.15},
            "stream": False,
            "tools": [tool.projection() for tool in _tools()],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert bodies == [expected]
    assert response.raw_request_hash == "sha256:" + hashlib.sha256(expected).hexdigest()
    assert (
        response.raw_request_hash
        == "sha256:0dbfdfee3e79373266b1508aa77e339f0da881836c49755fc0a63b125dbd98ad"
    )
    assert response.provider_done is True
    assert b"TOOL_RESULT" not in expected


def test_native_tools3_projects_turn_prompt_and_history_with_exact_bytes() -> None:
    bodies: list[bytes] = []

    async def transport(body: bytes) -> TransportResponse:
        bodies.append(body)
        return TransportResponse(
            status_code=200,
            body=_body(
                calls=[{"function": {"name": "finish_task", "arguments": {"summary": "ok"}}}]
            ),
        )

    history = (
        NativeToolHistoryTurn(
            tool_name="lookup",
            arguments={"id": "42"},
            result={"local_status": "success", "value": "answer"},
        ),
    )
    request = ModelRequest(
        messages=_request().messages[:-1]
        + (Message(role="developer", content=NATIVE_TOOL_TURN_PROMPT),)
        + _request().messages[-1:],
        temperature=0.15,
        seed=4242,
        max_tokens=8192,
    )
    response = asyncio.run(
        OllamaNativeToolClient(
            capabilities=_capabilities(),
            transport=transport,
            max_attempts=1,
            context_length=32768,
            request_profile="SSB-OLLAMA-NATIVE-TOOLS3",
            response_contract="SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
        ).call_tools(request, _tools(), history=history)
    )

    expected = json.dumps(
        {
            "messages": [
                {"content": "system marker", "role": "system"},
                {"content": "developer marker", "role": "developer"},
                {"content": NATIVE_TOOL_TURN_PROMPT, "role": "developer"},
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"arguments": {"id": "42"}, "name": "lookup"}}],
                },
                {
                    "content": '{"local_status":"success","value":"answer"}',
                    "role": "tool",
                    "tool_name": "lookup",
                },
                {"content": "call one tool", "role": "user"},
            ],
            "model": "model-test",
            "options": {"num_ctx": 32768, "num_predict": 8192, "seed": 4242, "temperature": 0.15},
            "stream": False,
            "tools": [tool.projection() for tool in _tools()],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert bodies == [expected]
    assert response.raw_request_hash == "sha256:" + hashlib.sha256(expected).hexdigest()
    assert (
        response.raw_request_hash
        == "sha256:be78f63583e70a2fe0c46732806920315bea9527c667effdf21be6ddf15d731d"
    )


def test_native_tools_history_rejects_profile_mismatch_and_malformed_turns() -> None:
    async def transport(body: bytes) -> TransportResponse:
        del body
        raise AssertionError("configuration rejection must happen before transport")

    turn = NativeToolHistoryTurn(
        tool_name="lookup", arguments={"id": "42"}, result={"local_status": "success"}
    )
    with pytest.raises(ModelAdapterError) as mismatch:
        asyncio.run(
            OllamaNativeToolClient(
                capabilities=_capabilities(),
                transport=transport,
                max_attempts=1,
                context_length=32768,
            ).call_tools(_request(), _tools(), history=(turn,))
        )
    assert mismatch.value.code == "CONFIGURATION_ERROR"

    with pytest.raises(ModelAdapterError) as malformed:
        asyncio.run(
            OllamaNativeToolClient(
                capabilities=_capabilities(),
                transport=transport,
                max_attempts=1,
                context_length=32768,
                request_profile="SSB-OLLAMA-NATIVE-TOOLS2",
            ).call_tools(_request(), _tools(), history=(object(),))  # type: ignore[arg-type]
        )
    assert malformed.value.code == "CONFIGURATION_ERROR"


@pytest.mark.parametrize("provider_reason", [None, "stop", "tool_calls"])
def test_roles2_accepts_done_tool_call_with_nullable_provider_reason(
    provider_reason: str | None,
) -> None:
    requests: list[bytes] = []

    async def transport(body: bytes) -> TransportResponse:
        requests.append(body)
        payload = json.loads(
            _body(calls=[{"function": {"name": "finish_task", "arguments": {"summary": "ok"}}}])
        )
        payload["done_reason"] = provider_reason
        return TransportResponse(
            status_code=200,
            body=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        )

    response = asyncio.run(
        OllamaNativeToolClient(
            capabilities=_capabilities(),
            transport=transport,
            max_attempts=1,
            context_length=32768,
            response_contract="SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
        ).call_tools(_request(), _tools())
    )
    assert response.finish_reason == "tool_calls"
    assert response.provider_finish_reason == provider_reason
    assert response.provider_done is True
    assert requests == [
        json.dumps(
            {
                "messages": [
                    {"content": "system marker", "role": "system"},
                    {"content": "developer marker", "role": "developer"},
                    {"content": "call one tool", "role": "user"},
                ],
                "model": "model-test",
                "options": {
                    "num_ctx": 32768,
                    "num_predict": 8192,
                    "seed": 4242,
                    "temperature": 0.15,
                },
                "stream": False,
                "tools": [tool.projection() for tool in _tools()],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ]


@pytest.mark.parametrize("done", [False, None])
def test_roles2_requires_done_true(done: bool | None) -> None:
    async def transport(body: bytes) -> TransportResponse:
        del body
        payload = json.loads(
            _body(calls=[{"function": {"name": "finish_task", "arguments": {"summary": "ok"}}}])
        )
        if done is None:
            del payload["done"]
        else:
            payload["done"] = done
        return TransportResponse(
            status_code=200,
            body=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        )

    with pytest.raises(ModelAdapterError) as captured:
        asyncio.run(
            OllamaNativeToolClient(
                capabilities=_capabilities(),
                transport=transport,
                max_attempts=1,
                context_length=32768,
                response_contract="SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
            ).call_tools(_request(), _tools())
        )
    assert captured.value.code == "MODEL_OUTPUT_INVALID"
    assert captured.value.provider_done is done


@pytest.mark.parametrize(
    "calls",
    [
        None,
        [],
        [{"function": {"name": "finish_task", "arguments": {}}}] * 2,
        [{"function": {"name": "finish_task", "arguments": "{"}}],
    ],
)
def test_native_tools_rejects_non_single_or_malformed_calls_without_raw_leakage(
    calls: object,
) -> None:
    secret = "provider-content-must-not-leak"

    async def transport(body: bytes) -> TransportResponse:
        del body
        body = _body(calls=calls).replace(b"untrusted", secret.encode())
        return TransportResponse(status_code=200, body=body)

    with pytest.raises(ModelAdapterError) as captured:
        asyncio.run(
            OllamaNativeToolClient(
                capabilities=_capabilities(),
                transport=transport,
                max_attempts=1,
                context_length=32768,
            ).call_tools(_request(), _tools())
        )
    error = captured.value
    assert error.code == "MODEL_OUTPUT_INVALID"
    assert error.raw_request_hash is not None
    assert error.raw_response_hash is not None
    assert secret not in str(error)
    assert secret not in repr(error)
