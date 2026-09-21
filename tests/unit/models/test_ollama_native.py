from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from pydantic import BaseModel

from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    render_harmony_native_generate_prompt,
)
from shadowskillbench.models import (
    Message,
    ModelAdapterError,
    ModelRequest,
    OllamaNativeClient,
    ProviderCapabilities,
    TransportResponse,
)


class Output(BaseModel):
    value: str


def _capabilities() -> ProviderCapabilities:
    return ProviderCapabilities(
        provider="ollama",
        model="model-test",
        model_version="v1",
        supports_system_role=True,
        supports_developer_role=True,
        supports_seed=True,
        supports_structured_output=True,
    )


def _request() -> ModelRequest:
    return ModelRequest(
        messages=(
            Message(role="system", content="follow the pilot contract"),
            Message(role="user", content="return JSON"),
        ),
        temperature=1.0,
        seed=7,
        max_tokens=33,
    )


def _body(**overrides: object) -> bytes:
    value: dict[str, object] = {
        "model": "model-test",
        "response": '{"value":"ok"}',
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 2,
        "eval_count": 3,
    }
    value.update(overrides)
    return json.dumps(value, separators=(",", ":")).encode()


def test_native_request_bytes_and_hash_are_pinned() -> None:
    bodies: list[bytes] = []

    async def transport(body: bytes) -> TransportResponse:
        bodies.append(body)
        return TransportResponse(status_code=200, body=_body())

    response = asyncio.run(
        OllamaNativeClient(
            capabilities=_capabilities(), transport=transport, max_attempts=1
        ).structured(_request(), Output)
    )

    expected = json.dumps(
        {
            "format": Output.model_json_schema(),
            "model": "model-test",
            "options": {
                "num_ctx": 131072,
                "num_predict": 33,
                "seed": 7,
                "temperature": 1.0,
                "top_p": 1.0,
            },
            "prompt": render_harmony_native_generate_prompt(messages=_request().messages),
            "raw": True,
            "stream": False,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert bodies == [expected]
    assert response.raw_request_hash == (
        "sha256:a3f5a7d7540d8e9df239ced92d20eef2eb75dc1ee182bd97b4858906daf9a109"
    )
    assert response.raw_request_hash == "sha256:" + hashlib.sha256(expected).hexdigest()
    assert response.usage.model_dump() == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    assert response.output == Output(value="ok")


def test_native_request_omits_seed_only_when_unspecified_and_never_uses_openai_fields() -> None:
    bodies: list[bytes] = []

    async def transport(body: bytes) -> TransportResponse:
        bodies.append(body)
        return TransportResponse(status_code=200, body=_body())

    request = _request().model_copy(update={"seed": None})
    asyncio.run(
        OllamaNativeClient(
            capabilities=_capabilities(), transport=transport, max_attempts=1
        ).structured(request, Output)
    )

    assert b'"seed"' not in bodies[0]
    assert b'"messages"' not in bodies[0]
    assert b'"max_tokens"' not in bodies[0]
    assert b'"reasoning_effort"' not in bodies[0]
    assert b'"raw":true' in bodies[0]


@pytest.mark.parametrize(
    "condition",
    (
        "A1_POLICY_ONLY_SYSTEM",
        "A4_SKILL_POLICY_SYSTEM_TIER",
        "B1_FLAT_POLICY_SYSTEM",
        "B2_AUTHORITY_RESOLVER",
        "B3_DETERMINISTIC_GATE",
    ),
)
def test_native_context_requests_with_two_leading_system_messages_reach_transport(
    condition: str,
) -> None:
    bodies: list[bytes] = []

    async def transport(body: bytes) -> TransportResponse:
        bodies.append(body)
        return TransportResponse(status_code=200, body=_body())

    request = _request().model_copy(
        update={
            "messages": (
                Message(role="system", content="runtime contract"),
                Message(role="system", content=f"policy for {condition}"),
                Message(role="developer", content=f"skill for {condition}"),
                Message(role="user", content="return JSON"),
            )
        }
    )
    response = asyncio.run(
        OllamaNativeClient(
            capabilities=_capabilities(), transport=transport, max_attempts=1
        ).structured(request, Output)
    )

    assert response.output == Output(value="ok")
    assert len(bodies) == 1
    assert response.raw_request_hash is not None
    assert b"SSB SYSTEM MESSAGE BOUNDARY" in bodies[0]


def test_compiler_native_request_uses_low_reasoning_and_omits_top_p() -> None:
    bodies: list[bytes] = []

    async def transport(body: bytes) -> TransportResponse:
        bodies.append(body)
        return TransportResponse(status_code=200, body=_body())

    response = asyncio.run(
        OllamaNativeClient(
            capabilities=_capabilities(),
            transport=transport,
            max_attempts=1,
            reasoning_effort="low",
            include_top_p=False,
        ).structured(_request(), Output)
    )

    expected = json.dumps(
        {
            "format": Output.model_json_schema(),
            "model": "model-test",
            "options": {
                "num_ctx": 131072,
                "num_predict": 33,
                "seed": 7,
                "temperature": 1.0,
            },
            "prompt": render_harmony_native_generate_prompt(
                messages=_request().messages,
                reasoning_effort="low",
            ),
            "raw": True,
            "stream": False,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert bodies == [expected]
    assert response.raw_request_hash == (
        "sha256:24bc90c330c9f79d7cfa51da21f3088b69dc0f483efd52add468b73786890559"
    )
    assert response.raw_request_hash == "sha256:" + hashlib.sha256(expected).hexdigest()
    assert b'"top_p"' not in expected
    assert b'"messages"' not in expected
    assert b'"max_tokens"' not in expected
    assert b'"reasoning_effort"' not in expected
    assert b"Reasoning: low\\n\\n" in expected
    assert expected.endswith(
        b'<|start|>assistant<|channel|>final<|message|>","raw":true,"stream":false}'
    )


@pytest.mark.parametrize(
    "body",
    (
        _body(model="other-model"),
        _body(response={"not": "a string"}),
        _body(done=False),
        _body(done_reason="length"),
        _body(prompt_eval_count=True),
        _body(eval_count="3"),
        _body(response='{"value":1}'),
        b'{"model":"model-test","model":"other"}',
        b"not-json",
    ),
)
def test_native_rejects_wrong_envelope_counts_and_invalid_content(body: bytes) -> None:
    async def transport(_: bytes) -> TransportResponse:
        return TransportResponse(status_code=200, body=body)

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(
            OllamaNativeClient(
                capabilities=_capabilities(), transport=transport, max_attempts=1
            ).structured(_request(), Output)
        )

    assert caught.value.code == "MODEL_OUTPUT_INVALID"
    assert caught.value.raw_response_hash == "sha256:" + hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize(
    "response",
    ("",),
)
def test_native_invalid_response_preserves_hash_only_diagnostics(
    response: str,
) -> None:
    async def transport(_: bytes) -> TransportResponse:
        return TransportResponse(status_code=200, body=_body(response=response))

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(
            OllamaNativeClient(
                capabilities=_capabilities(), transport=transport, max_attempts=1
            ).structured(_request(), Output)
        )

    error = caught.value
    assert error.code == "MODEL_OUTPUT_INVALID"
    assert error.attempts == 1
    assert error.raw_request_hash is not None
    assert (
        error.raw_response_hash == "sha256:" + hashlib.sha256(_body(response=response)).hexdigest()
    )
    assert error.finish_reason == "stop"
    assert error.reported_usage is not None
    assert error.reported_usage.total_tokens == 5
