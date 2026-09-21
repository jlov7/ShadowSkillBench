from __future__ import annotations

import asyncio

import httpx
import pytest

from shadowskillbench.models.runtime import (
    ModelRunDescriptor,
    OpenAICompatibleHttpTransport,
    RunDescriptorError,
    client_from_run_descriptor,
    ollama_native_client_from_run_descriptor,
)


def _descriptor(**changes: object) -> ModelRunDescriptor:
    values: dict[str, object] = {
        "provider": "test-provider",
        "model": "test-model",
        "model_version": "2026-08-24",
        "endpoint": "https://models.example.invalid/v1/chat/completions",
        "api_key_environment": "SSB_TEST_API_KEY",
        "max_attempts": 2,
    }
    values.update(changes)
    return ModelRunDescriptor(**values)  # type: ignore[arg-type]


def test_descriptor_constructs_a_concrete_credential_separate_client() -> None:
    descriptor = _descriptor()

    client = client_from_run_descriptor(
        descriptor, environment={"SSB_TEST_API_KEY": "local-test-key"}
    )

    assert client.capabilities == descriptor.capabilities
    assert isinstance(client._transport, OpenAICompatibleHttpTransport)
    assert client._transport.timeout_seconds == 60.0


def test_descriptor_client_passes_an_explicit_validated_timeout_to_the_transport() -> None:
    descriptor = _descriptor()
    client = client_from_run_descriptor(
        descriptor,
        environment={"SSB_TEST_API_KEY": "local-test-key"},
        timeout_seconds=900.0,
    )

    assert isinstance(client._transport, OpenAICompatibleHttpTransport)
    assert client._transport.timeout_seconds == 900.0
    with pytest.raises(RunDescriptorError, match="transport configuration"):
        client_from_run_descriptor(
            descriptor,
            environment={"SSB_TEST_API_KEY": "local-test-key"},
            timeout_seconds=0.0,
        )
    with pytest.raises(RunDescriptorError, match="transport configuration"):
        client_from_run_descriptor(
            descriptor,
            environment={"SSB_TEST_API_KEY": "local-test-key"},
            timeout_seconds=901.0,
        )


def test_descriptor_client_can_explicitly_bind_reasoning_suppression() -> None:
    client = client_from_run_descriptor(
        _descriptor(),
        environment={"SSB_TEST_API_KEY": "local-test-key"},
        reasoning_effort="none",
    )

    assert client.reasoning_effort == "none"


def test_native_factory_pins_native_context_sampling_and_semantic_no_reasoning() -> None:
    client = ollama_native_client_from_run_descriptor(
        _descriptor(endpoint="http://127.0.0.1:11434/api/generate", max_attempts=1),
        environment={"SSB_TEST_API_KEY": "local-test-key"},
        top_p=1.0,
        context_length=131_072,
    )

    assert client.reasoning_effort == "none"
    assert client.transport_profile == "ollama_native_api_generate_raw"
    assert client._top_p == 1.0
    assert client._context_length == 131_072
    with pytest.raises(ValueError, match="client configuration"):
        ollama_native_client_from_run_descriptor(
            _descriptor(endpoint="http://127.0.0.1:11434/api/generate", max_attempts=1),
            environment={"SSB_TEST_API_KEY": "local-test-key"},
            top_p=0.9,
        )


def test_native_factory_allows_only_exact_compiler_reasoning_configuration() -> None:
    descriptor = _descriptor(endpoint="http://127.0.0.1:11434/api/generate", max_attempts=1)
    client = ollama_native_client_from_run_descriptor(
        descriptor,
        environment={"SSB_TEST_API_KEY": "local-test-key"},
        reasoning_effort="low",
        include_top_p=False,
    )

    assert client.reasoning_effort == "low"
    assert client._include_top_p is False

    class StringSubclass(str):
        pass

    with pytest.raises(RunDescriptorError, match="native reasoning_effort"):
        ollama_native_client_from_run_descriptor(
            descriptor,
            environment={"SSB_TEST_API_KEY": "local-test-key"},
            reasoning_effort=StringSubclass("low"),  # type: ignore[arg-type]
            include_top_p=False,
        )


def test_descriptor_refuses_missing_credentials_without_a_transport_attempt() -> None:
    with pytest.raises(RunDescriptorError, match="API key is unavailable"):
        client_from_run_descriptor(_descriptor(), environment={})


def test_http_transport_disables_redirects_and_bounds_response_bytes() -> None:
    redirected = OpenAICompatibleHttpTransport(
        "https://models.example.invalid/v1",
        "key",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(302, headers={"location": "https://elsewhere.invalid"})
        ),
    )
    oversized = OpenAICompatibleHttpTransport(
        "https://models.example.invalid/v1",
        "key",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-length": "4000001"})
        ),
    )

    assert asyncio.run(redirected(b"{}")).status_code == 302
    with pytest.raises(Exception) as error:
        asyncio.run(oversized(b"{}"))
    assert type(error.value).__name__ == "TransportTerminalError"


@pytest.mark.parametrize("field", ("endpoint", "api_key_environment", "max_attempts"))
def test_descriptor_rejects_invalid_nonsecret_configuration(field: str) -> None:
    values = {
        "endpoint": "not-a-url",
        "api_key_environment": "not valid",
        "max_attempts": 0,
    }
    with pytest.raises(RunDescriptorError):
        _descriptor(**{field: values[field]})


def test_transport_allows_http_only_for_explicit_loopback_development() -> None:
    assert OpenAICompatibleHttpTransport("http://127.0.0.1:8080/v1", "key")
    with pytest.raises(RunDescriptorError):
        OpenAICompatibleHttpTransport("http://models.example.invalid/v1", "key")


def test_loopback_transport_never_uses_ambient_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        status_code = 200
        headers: dict[str, str] = {}

        async def aiter_bytes(self):
            yield b"{}"

    class Stream:
        async def __aenter__(self) -> Response:
            return Response()

        async def __aexit__(self, *_: object) -> None:
            return None

    class Client:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        def stream(self, *_: object, **__: object) -> Stream:
            return Stream()

    monkeypatch.setattr(httpx, "AsyncClient", Client)

    asyncio.run(OpenAICompatibleHttpTransport("http://127.0.0.1:8080/v1", "key")(b"{}"))

    assert captured["trust_env"] is False
