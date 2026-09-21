"""Concrete, credential-separate composition for an OpenAI-compatible executor."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast
from urllib.parse import urlparse

import httpx

from shadowskillbench.episodes.native_tool_turn_wire import (
    NATIVE_TOOL_TURN_PROMPT_HASH,
    NATIVE_TOOL_TURN_PROMPT_PROFILE,
)
from shadowskillbench.models.ollama_native import OllamaNativeClient
from shadowskillbench.models.ollama_native_tools import OllamaNativeToolClient
from shadowskillbench.models.openai_compatible import OpenAICompatibleClient
from shadowskillbench.models.protocol import (
    ProviderCapabilities,
    TokenPricing,
    TransportResponse,
    TransportTerminalError,
    TransportTransientError,
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_MAX_RESPONSE_BYTES = 4_000_000


class RunDescriptorError(ValueError):
    """The configured run cannot be admitted without an explicit, local descriptor."""


@dataclass(frozen=True, slots=True)
class ModelRunDescriptor:
    """Frozen non-secret details required to construct one executor client."""

    provider: str
    model: str
    model_version: str
    endpoint: str
    api_key_environment: str
    max_attempts: int
    supports_system_role: bool = True
    supports_developer_role: bool = True
    supports_seed: bool = True
    supports_structured_output: bool = True
    pricing: TokenPricing | None = None
    executor_runtime_profile: dict[str, object] | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint) if type(self.endpoint) is str else None
        loopback = parsed is not None and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        try:
            ProviderCapabilities(
                provider=self.provider,
                model=self.model,
                model_version=self.model_version,
                supports_system_role=self.supports_system_role,
                supports_developer_role=self.supports_developer_role,
                supports_seed=self.supports_seed,
                supports_structured_output=self.supports_structured_output,
            )
        except (TypeError, ValueError) as error:
            raise RunDescriptorError("provider capabilities are invalid") from error
        if (
            type(self.endpoint) is not str
            or parsed is None
            or parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or (parsed.scheme == "http" and not loopback)
            or any(character.isspace() for character in self.endpoint)
        ):
            raise RunDescriptorError("endpoint must be an absolute HTTP(S) URL")
        if (
            type(self.api_key_environment) is not str
            or _ENVIRONMENT_NAME.fullmatch(self.api_key_environment) is None
        ):
            raise RunDescriptorError("api_key_environment is invalid")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 5:
            raise RunDescriptorError("max_attempts is invalid")
        if self.pricing is not None and type(self.pricing) is not TokenPricing:
            raise RunDescriptorError("pricing is invalid")
        if self.executor_runtime_profile is not None and (
            type(self.executor_runtime_profile) is not dict
            or any(type(key) is not str for key in self.executor_runtime_profile)
        ):
            raise RunDescriptorError("executor_runtime_profile is invalid")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.provider,
            model=self.model,
            model_version=self.model_version,
            supports_system_role=self.supports_system_role,
            supports_developer_role=self.supports_developer_role,
            supports_seed=self.supports_seed,
            supports_structured_output=self.supports_structured_output,
        )


@dataclass(frozen=True, slots=True)
class OpenAICompatibleHttpTransport:
    """A small async transport that never persists or reports credential material."""

    endpoint: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 60.0
    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False, compare=False)
    trust_env: bool = True

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint) if type(self.endpoint) is str else None
        loopback = parsed is not None and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        if (
            type(self.endpoint) is not str
            or parsed is None
            or parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or (parsed.scheme == "http" and not loopback)
            or any(character.isspace() for character in self.endpoint)
            or type(self.api_key) is not str
            or not self.api_key
            or "\x00" in self.api_key
            or type(self.timeout_seconds) not in {int, float}
            or not 0 < self.timeout_seconds <= 900
            or self.transport is not None
            and not isinstance(self.transport, httpx.AsyncBaseTransport)
            or type(self.trust_env) is not bool
        ):
            raise RunDescriptorError("HTTP transport configuration is invalid")

    async def __call__(self, body: bytes) -> TransportResponse:
        if type(body) is not bytes:
            raise TransportTerminalError()
        parsed = urlparse(self.endpoint)
        loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False if loopback else self.trust_env,
                timeout=httpx.Timeout(float(self.timeout_seconds)),
                transport=self.transport,
            ) as client:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    content=body,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                ) as response:
                    content_length = response.headers.get("content-length")
                    if content_length is not None and (
                        not content_length.isdecimal() or int(content_length) > _MAX_RESPONSE_BYTES
                    ):
                        raise TransportTerminalError()
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > _MAX_RESPONSE_BYTES:
                            raise TransportTerminalError()
                    return TransportResponse(status_code=response.status_code, body=bytes(content))
        except (httpx.TimeoutException, httpx.NetworkError, ConnectionError) as error:
            raise TransportTransientError() from error
        except TransportTerminalError:
            raise
        except Exception as error:
            raise TransportTerminalError() from error


def client_from_run_descriptor(
    descriptor: ModelRunDescriptor,
    *,
    environment: Mapping[str, str] | None = None,
    trust_env: bool = True,
    reasoning_effort: str | None = None,
    timeout_seconds: float = 60.0,
) -> OpenAICompatibleClient:
    """Construct a client from frozen metadata and one separately supplied secret."""

    if type(descriptor) is not ModelRunDescriptor:
        raise RunDescriptorError("descriptor must be exact")
    source = os.environ if environment is None else environment
    api_key = source.get(descriptor.api_key_environment)
    if type(api_key) is not str or not api_key:
        raise RunDescriptorError("configured API key is unavailable")
    if type(trust_env) is not bool:
        raise RunDescriptorError("trust_env is invalid")
    return OpenAICompatibleClient(
        capabilities=descriptor.capabilities,
        transport=OpenAICompatibleHttpTransport(
            descriptor.endpoint,
            api_key,
            timeout_seconds=timeout_seconds,
            trust_env=trust_env,
        ),
        max_attempts=descriptor.max_attempts,
        pricing=descriptor.pricing,
        reasoning_effort=reasoning_effort,
    )


def ollama_native_client_from_run_descriptor(
    descriptor: ModelRunDescriptor,
    *,
    environment: Mapping[str, str] | None = None,
    trust_env: bool = True,
    timeout_seconds: float = 60.0,
    top_p: float = 1.0,
    context_length: int = 131_072,
    reasoning_effort: Literal["none", "low"] = "none",
    include_top_p: bool = True,
) -> OllamaNativeClient:
    """Construct a native Ollama raw-generate client from frozen metadata and one secret."""

    if type(descriptor) is not ModelRunDescriptor:
        raise RunDescriptorError("descriptor must be exact")
    source = os.environ if environment is None else environment
    api_key = source.get(descriptor.api_key_environment)
    if type(api_key) is not str or not api_key:
        raise RunDescriptorError("configured API key is unavailable")
    if type(trust_env) is not bool:
        raise RunDescriptorError("trust_env is invalid")
    if type(reasoning_effort) is not str or reasoning_effort not in {"none", "low"}:
        raise RunDescriptorError("native reasoning_effort is invalid")
    return OllamaNativeClient(
        capabilities=descriptor.capabilities,
        transport=OpenAICompatibleHttpTransport(
            descriptor.endpoint,
            api_key,
            timeout_seconds=timeout_seconds,
            trust_env=trust_env,
        ),
        max_attempts=descriptor.max_attempts,
        pricing=descriptor.pricing,
        top_p=top_p,
        context_length=context_length,
        reasoning_effort=reasoning_effort,
        include_top_p=include_top_p,
    )


def ollama_native_tool_client_from_run_descriptor(
    descriptor: ModelRunDescriptor,
    *,
    environment: Mapping[str, str] | None = None,
    trust_env: bool = True,
    timeout_seconds: float = 60.0,
) -> OllamaNativeToolClient:
    """Construct the separately bound ``/api/chat`` native-tool client."""

    if type(descriptor) is not ModelRunDescriptor:
        raise RunDescriptorError("descriptor must be exact")
    parsed = urlparse(descriptor.endpoint)
    v1_profile = {
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
        "endpoint_path": "/api/chat",
        "context_length": 32_768,
        "top_p_wire": "omitted",
        "reasoning_effort_wire": "omitted",
    }
    v2_profile = {**v1_profile, "response_contract": "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2"}
    v3_profile = {
        **v2_profile,
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS2",
        "history_projection": "SSB-OLLAMA-NATIVE-TOOLS-HISTORY1",
    }
    v4_profile = {
        **v3_profile,
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS3",
        "native_turn_prompt_profile": NATIVE_TOOL_TURN_PROMPT_PROFILE,
        "native_turn_prompt_hash": NATIVE_TOOL_TURN_PROMPT_HASH,
    }
    if parsed.path != "/api/chat" or descriptor.executor_runtime_profile not in [
        v1_profile,
        v2_profile,
        v3_profile,
        v4_profile,
    ]:
        raise RunDescriptorError("native tool descriptor profile is invalid")
    source = os.environ if environment is None else environment
    api_key = source.get(descriptor.api_key_environment)
    if type(api_key) is not str or not api_key:
        raise RunDescriptorError("configured API key is unavailable")
    if type(trust_env) is not bool:
        raise RunDescriptorError("trust_env is invalid")
    return OllamaNativeToolClient(
        capabilities=descriptor.capabilities,
        transport=OpenAICompatibleHttpTransport(
            descriptor.endpoint,
            api_key,
            timeout_seconds=timeout_seconds,
            trust_env=trust_env,
        ),
        max_attempts=descriptor.max_attempts,
        pricing=descriptor.pricing,
        context_length=32_768,
        request_profile=cast(str, descriptor.executor_runtime_profile["request_profile"]),
        response_contract=(
            "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2"
            if descriptor.executor_runtime_profile in [v2_profile, v3_profile, v4_profile]
            else "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE1"
        ),
    )


__all__ = [
    "ModelRunDescriptor",
    "OpenAICompatibleHttpTransport",
    "RunDescriptorError",
    "client_from_run_descriptor",
    "ollama_native_client_from_run_descriptor",
    "ollama_native_tool_client_from_run_descriptor",
]
