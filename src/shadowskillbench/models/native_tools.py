"""Protocol types for native provider tool calling.

This module deliberately does not extend ``ModelClient.structured``: native
tool calls are a different provider contract with a different custody surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.engine.models import JsonObject, JsonValue
from shadowskillbench.models.protocol import (
    ModelRequest,
    ProviderCapabilities,
    TokenCost,
    TokenUsage,
)

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class NativeToolDefinition:
    """One canonical provider-neutral function declaration."""

    name: str
    description: str
    parameters: JsonObject

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or _NAME.fullmatch(self.name) is None
            or type(self.description) is not str
            or not self.description
            or type(self.parameters) is not dict
        ):
            raise ValueError("native tool declaration is invalid")
        canonical_json_bytes(self.parameters)

    def projection(self) -> dict[str, JsonValue]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def native_tool_declaration_projection(
    declarations: tuple[NativeToolDefinition, ...],
) -> list[dict[str, JsonValue]]:
    if (
        type(declarations) is not tuple
        or not declarations
        or any(type(item) is not NativeToolDefinition for item in declarations)
        or len({item.name for item in declarations}) != len(declarations)
    ):
        raise ValueError("native tool declarations are invalid")
    return [item.projection() for item in declarations]


def native_tool_declaration_hash(declarations: tuple[NativeToolDefinition, ...]) -> str:
    return sha256_ref(
        {
            "profile": "SSB-OLLAMA-NATIVE-TOOLS1",
            "tools": native_tool_declaration_projection(declarations),
        }
    )


@dataclass(frozen=True, slots=True)
class NativeToolProviderCapabilities:
    """Capabilities specific to the native-tools route, bound beside base identity."""

    base: ProviderCapabilities
    endpoint_path: str
    supports_native_tool_calls: bool
    request_profile: str

    def __post_init__(self) -> None:
        if (
            type(self.base) is not ProviderCapabilities
            or self.endpoint_path != "/api/chat"
            or self.supports_native_tool_calls is not True
            or self.request_profile
            not in {
                "SSB-OLLAMA-NATIVE-TOOLS1",
                "SSB-OLLAMA-NATIVE-TOOLS2",
                "SSB-OLLAMA-NATIVE-TOOLS3",
            }
        ):
            raise ValueError("native provider capabilities are invalid")


@dataclass(frozen=True, slots=True)
class NativeToolResponse:
    """Sanitized native response: provider bytes remain hash-only."""

    tool_name: str
    arguments: JsonObject
    raw_request_hash: str
    raw_response_hash: str
    tool_declaration_hash: str
    usage: TokenUsage
    cost: TokenCost | None
    attempts: int
    finish_reason: str | None
    provider_finish_reason: str | None = None
    provider_done: bool | None = None

    def __post_init__(self) -> None:
        if type(self.tool_name) is not str or _NAME.fullmatch(self.tool_name) is None:
            raise ValueError("native tool response name is invalid")
        if type(self.arguments) is not dict:
            raise ValueError("native tool response arguments are invalid")
        canonical_json_bytes(self.arguments)
        if any(
            type(value) is not str or _HASH.fullmatch(value) is None
            for value in (self.raw_request_hash, self.raw_response_hash, self.tool_declaration_hash)
        ):
            raise ValueError("native tool response hashes are invalid")
        if (
            type(self.usage) is not TokenUsage
            or self.cost is not None
            and type(self.cost) is not TokenCost
        ):
            raise ValueError("native tool response usage is invalid")
        if type(self.attempts) is not int or not 1 <= self.attempts <= 5:
            raise ValueError("native tool response attempts are invalid")
        if self.finish_reason is not None and (
            type(self.finish_reason) is not str
            or not self.finish_reason
            or len(self.finish_reason) > 32
            or not self.finish_reason.isprintable()
        ):
            raise ValueError("native tool response finish reason is invalid")
        if self.provider_finish_reason is not None and (
            type(self.provider_finish_reason) is not str
            or not self.provider_finish_reason
            or len(self.provider_finish_reason) > 32
            or not self.provider_finish_reason.isprintable()
        ):
            raise ValueError("native tool provider finish reason is invalid")
        if self.provider_done is not None and type(self.provider_done) is not bool:
            raise ValueError("native tool provider done is invalid")


@dataclass(frozen=True, slots=True)
class NativeToolHistoryTurn:
    """Validated, provider-neutral custody for one completed native tool turn."""

    tool_name: str
    arguments: JsonObject
    result: JsonObject

    def __post_init__(self) -> None:
        if (
            type(self.tool_name) is not str
            or _NAME.fullmatch(self.tool_name) is None
            or type(self.arguments) is not dict
            or type(self.result) is not dict
        ):
            raise ValueError("native tool history turn is invalid")
        canonical_json_bytes(self.arguments)
        canonical_json_bytes(self.result)


class NativeToolClient(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...

    @property
    def native_capabilities(self) -> NativeToolProviderCapabilities: ...

    async def call_tools(
        self,
        request: ModelRequest,
        declarations: tuple[NativeToolDefinition, ...],
        *,
        history: tuple[NativeToolHistoryTurn, ...] = (),
    ) -> NativeToolResponse: ...


__all__ = [
    "NativeToolClient",
    "NativeToolDefinition",
    "NativeToolHistoryTurn",
    "NativeToolProviderCapabilities",
    "NativeToolResponse",
    "native_tool_declaration_hash",
    "native_tool_declaration_projection",
]
