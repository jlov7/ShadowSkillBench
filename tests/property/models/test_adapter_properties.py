from __future__ import annotations

# ruff: noqa: E501
import asyncio
import hashlib
import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from shadowskillbench.models import (
    Message,
    ModelRequest,
    ProviderCapabilities,
    ScriptedModelClient,
    TransportResponse,
)
from shadowskillbench.models.openai_compatible import OpenAICompatibleClient


class Output(BaseModel):
    value: str


class OtherOutput(BaseModel):
    other: str


async def _transport(_: bytes) -> object:
    raise AssertionError("property preflight must not transport")


def _capabilities() -> ProviderCapabilities:
    return ProviderCapabilities(
        provider="local",
        model="m",
        model_version="v",
        supports_system_role=True,
        supports_developer_role=True,
        supports_seed=True,
        supports_structured_output=True,
    )


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        capabilities=_capabilities(), transport=_transport, max_attempts=1
    )


def _request(
    *,
    messages: tuple[Message, ...] = (Message(role="user", content="value"),),
    temperature: float = 0.0,
    seed: int | None = 7,
    max_tokens: int = 9,
) -> ModelRequest:
    return ModelRequest(
        messages=messages, temperature=temperature, seed=seed, max_tokens=max_tokens
    )


def _wire_hash(request: ModelRequest, schema: type[BaseModel] = Output) -> tuple[bytes, str]:
    _, _, body = _client()._preflight(request, schema)
    return body, "sha256:" + hashlib.sha256(body).hexdigest()


def _response_body(value: str, *, reordered: bool) -> bytes:
    envelope: dict[str, object] = {
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps({"value": value})},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    if reordered:
        envelope = {"usage": envelope["usage"], "choices": envelope["choices"], "model": "m"}
        return json.dumps(envelope, indent=1).encode("utf-8")
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


TEXT = st.text(alphabet="abcXYZ012", min_size=1, max_size=12)


@pytest.mark.property
@settings(max_examples=1, deadline=None)
@given(st.just(-0.0))
def test_negative_zero_temperature_is_explicitly_generated_and_rejected(temperature: float) -> None:
    assert temperature == 0.0
    assert math.copysign(1.0, temperature) < 0.0
    with pytest.raises(ValueError, match="temperature"):
        _request(temperature=temperature)


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(TEXT, st.sampled_from((0.0, 0.25, 1.0, 2.0)), st.sampled_from((-7, 0, 9)))
def test_same_request_and_schema_have_byte_and_hash_determinism(
    content: str, temperature: float, seed: int
) -> None:
    request = _request(
        messages=(Message(role="user", content=content),), temperature=temperature, seed=seed
    )

    first = _wire_hash(request)
    second = _wire_hash(request.model_copy())

    assert first == second


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(TEXT, TEXT.filter(lambda value: value != "changed"), st.sampled_from((0.0, 0.5, 1.5)))
def test_each_request_or_schema_dimension_binds_the_raw_request_hash(
    content: str, distinct_content: str, temperature: float
) -> None:
    first = Message(role="user", content=content)
    second = Message(role="assistant", content="tail")
    base = _request(messages=(first, second), temperature=temperature)
    baseline = _wire_hash(base)[1]
    changed_content = distinct_content if distinct_content != content else "changed"
    alternatives = (
        _request(
            messages=(Message(role="system", content=content), second), temperature=temperature
        ),
        _request(
            messages=(Message(role="user", content=changed_content), second),
            temperature=temperature,
        ),
        _request(messages=(second, first), temperature=temperature),
        _request(messages=(first, second), temperature=temperature, seed=None),
        _request(messages=(first, second), temperature=temperature, seed=8),
        _request(
            messages=(first, second),
            temperature=0.0 if temperature != 0.0 else 0.5,
        ),
        _request(messages=(first, second), temperature=temperature, max_tokens=10),
    )

    assert all(_wire_hash(candidate)[1] != baseline for candidate in alternatives)
    assert _wire_hash(base, OtherOutput)[1] != baseline


@pytest.mark.property
@settings(max_examples=24, deadline=None)
@given(TEXT)
def test_raw_response_encoding_changes_raw_hash_without_changing_typed_semantics(
    value: str,
) -> None:
    compact = _response_body(value, reordered=False)
    spaced_reordered = _response_body(value, reordered=True)
    first = ScriptedModelClient(
        capabilities=_capabilities(),
        script=(TransportResponse(status_code=200, body=compact),),
        max_attempts=1,
    )
    second = ScriptedModelClient(
        capabilities=_capabilities(),
        script=(TransportResponse(status_code=200, body=spaced_reordered),),
        max_attempts=1,
    )

    first_response = asyncio.run(first.structured(_request(), Output))
    second_response = asyncio.run(second.structured(_request(), Output))

    assert compact != spaced_reordered
    assert first_response.output == second_response.output == Output(value=value)
    assert first_response.raw_response_hash != second_response.raw_response_hash
    assert first_response.raw_request_hash == second_response.raw_request_hash


@pytest.mark.property
@settings(max_examples=16, deadline=None)
@given(st.integers(min_value=1, max_value=3), TEXT)
def test_retries_copy_identical_wire_bytes_and_scripts_are_deterministic(
    transient_count: int, value: str
) -> None:
    response = TransportResponse(status_code=200, body=_response_body(value, reordered=False))
    script = ("transient",) * transient_count + (response,)
    first = ScriptedModelClient(
        capabilities=_capabilities(), script=script, max_attempts=transient_count + 1
    )
    second = ScriptedModelClient(
        capabilities=_capabilities(), script=script, max_attempts=transient_count + 1
    )

    first_response = asyncio.run(first.structured(_request(), Output))
    second_response = asyncio.run(second.structured(_request(), Output))

    assert first.recorded_request_bodies == (first.recorded_request_bodies[0],) * (
        transient_count + 1
    )
    assert first.recorded_request_bodies == second.recorded_request_bodies
    assert first_response == second_response
    assert first_response.attempts == transient_count + 1
