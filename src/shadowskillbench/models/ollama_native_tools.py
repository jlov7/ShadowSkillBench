"""Strict hash-only client for Ollama's native ``/api/chat`` tool interface."""

from __future__ import annotations

import hashlib
import json
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.engine import JsonObject
from shadowskillbench.models.native_tools import (
    NativeToolDefinition,
    NativeToolHistoryTurn,
    NativeToolProviderCapabilities,
    NativeToolResponse,
    native_tool_declaration_hash,
    native_tool_declaration_projection,
)
from shadowskillbench.models.protocol import (
    AsyncTransport,
    ModelAdapterError,
    ModelRequest,
    ProviderCapabilities,
    TokenPricing,
    TokenUsage,
    TransportResponse,
    TransportTransientError,
    _calculate_cost,
    _model_data,
)


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if type(key) is not str or key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def _strict_json(body: bytes) -> object:
    try:
        return json.loads(
            body.decode("utf-8", "strict"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            object_pairs_hook=_no_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON") from error


def _usage(payload: dict[str, object]) -> TokenUsage:
    input_tokens = payload.get("prompt_eval_count", 0)
    output_tokens = payload.get("eval_count", 0)
    if type(input_tokens) is not int or type(output_tokens) is not int:
        raise ValueError("usage is invalid")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


class OllamaNativeToolClient:
    """One-attempt native tool client; it never retains provider content or errors."""

    def __init__(
        self,
        *,
        capabilities: ProviderCapabilities,
        transport: AsyncTransport,
        max_attempts: int,
        context_length: int,
        request_profile: str = "SSB-OLLAMA-NATIVE-TOOLS1",
        response_contract: str = "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE1",
        pricing: TokenPricing | None = None,
    ) -> None:
        if (
            type(capabilities) is not ProviderCapabilities
            or not callable(transport)
            or type(max_attempts) is not int
            or max_attempts != 1
            or type(context_length) is not int
            or context_length != 32_768
            or request_profile
            not in {
                "SSB-OLLAMA-NATIVE-TOOLS1",
                "SSB-OLLAMA-NATIVE-TOOLS2",
                "SSB-OLLAMA-NATIVE-TOOLS3",
            }
            or response_contract
            not in {
                "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE1",
                "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
            }
            or pricing is not None
            and type(pricing) is not TokenPricing
        ):
            raise ValueError("native tool client configuration is invalid")
        self._capabilities = ProviderCapabilities.model_validate(
            _model_data(capabilities, ProviderCapabilities)
        )
        self._native_capabilities = NativeToolProviderCapabilities(
            base=self._capabilities,
            endpoint_path="/api/chat",
            supports_native_tool_calls=True,
            request_profile=request_profile,
        )
        self._transport = transport
        self._max_attempts = max_attempts
        self._context_length = context_length
        self._request_profile = request_profile
        self._response_contract = response_contract
        self._pricing = (
            TokenPricing.model_validate(_model_data(pricing, TokenPricing))
            if pricing is not None
            else None
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def native_capabilities(self) -> NativeToolProviderCapabilities:
        return self._native_capabilities

    @property
    def transport_profile(self) -> str:
        return "ollama_native_api_chat_tools"

    @property
    def response_contract(self) -> str:
        return self._response_contract

    def _error(
        self,
        code: str,
        *,
        attempts: int,
        before_meaningful_behavior: bool,
        raw_request_hash: str | None,
        raw_response_hash: str | None,
        finish_reason: str | None = None,
        usage: TokenUsage | None = None,
        provider_done: bool | None = None,
    ) -> ModelAdapterError:
        return ModelAdapterError(
            code=code,
            attempts=attempts,
            before_meaningful_behavior=before_meaningful_behavior,
            raw_request_hash=raw_request_hash,
            raw_response_hash=raw_response_hash,
            finish_reason=finish_reason,
            reported_usage=usage,
            provider_done=provider_done,
        )

    def _request_bytes(
        self,
        request: ModelRequest,
        declarations: tuple[NativeToolDefinition, ...],
        history: tuple[NativeToolHistoryTurn, ...] = (),
    ) -> tuple[bytes, str]:
        request = ModelRequest.model_validate(_model_data(request, ModelRequest))
        caps = ProviderCapabilities.model_validate(
            _model_data(self._capabilities, ProviderCapabilities)
        )
        if (
            not caps.supports_system_role
            and any(message.role == "system" for message in request.messages)
            or not caps.supports_developer_role
            and any(message.role == "developer" for message in request.messages)
            or not caps.supports_seed
            and request.seed is not None
        ):
            raise ValueError("native capability mismatch")
        tools = native_tool_declaration_projection(declarations)
        tool_hash = native_tool_declaration_hash(declarations)
        if (
            type(history) is not tuple
            or any(type(turn) is not NativeToolHistoryTurn for turn in history)
            or self._request_profile == "SSB-OLLAMA-NATIVE-TOOLS1"
            and history
        ):
            raise ValueError("native tool history profile is invalid")
        messages: list[dict[str, object]] = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]
        if (
            self._request_profile in {"SSB-OLLAMA-NATIVE-TOOLS2", "SSB-OLLAMA-NATIVE-TOOLS3"}
            and history
        ):
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("native tool history requires final observation")
            native_history: list[dict[str, object]] = []
            for turn in history:
                native_history.extend(
                    [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": turn.tool_name,
                                        "arguments": turn.arguments,
                                    }
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_name": turn.tool_name,
                            "content": canonical_json_bytes(turn.result).decode("utf-8"),
                        },
                    ]
                )
            messages[-1:-1] = native_history
        options: dict[str, object] = {
            "num_ctx": self._context_length,
            "num_predict": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        return (
            canonical_json_bytes(
                {
                    "model": caps.model,
                    "messages": messages,
                    "stream": False,
                    "tools": tools,
                    "options": options,
                }
            ),
            tool_hash,
        )

    def _parse(self, body: bytes, *, request_hash: str, tool_hash: str) -> NativeToolResponse:
        response_hash = "sha256:" + hashlib.sha256(body).hexdigest()
        payload: dict[str, object] | None = None
        provider_done: bool | None = None
        provider_finish_reason: str | None = None
        try:
            raw_payload = _strict_json(body)
            if type(raw_payload) is not dict:
                raise ValueError("response shape")
            payload = raw_payload
            done = payload.get("done")
            provider_done = done if type(done) is bool else None
            raw_finish_reason = payload.get("done_reason")
            if raw_finish_reason is not None and type(raw_finish_reason) is not str:
                raise ValueError("finish reason")
            provider_finish_reason = cast(str | None, raw_finish_reason)
            if self._response_contract == "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2" and done is not True:
                raise ValueError("response done")
            message = payload.get("message")
            if type(message) is not dict or message.get("role") != "assistant":
                raise ValueError("message shape")
            calls = message.get("tool_calls")
            if type(calls) is not list or len(calls) != 1 or type(calls[0]) is not dict:
                raise ValueError("tool calls shape")
            function = calls[0].get("function")
            if type(function) is not dict:
                raise ValueError("function shape")
            name = function.get("name")
            arguments = function.get("arguments")
            if type(arguments) is str:
                arguments = _strict_json(arguments.encode("utf-8"))
            if type(name) is not str or type(arguments) is not dict:
                raise ValueError("function arguments")
            usage = _usage(payload)
            cost = None if self._pricing is None else _calculate_cost(usage, self._pricing)
            return NativeToolResponse(
                tool_name=name,
                arguments=cast(JsonObject, arguments),
                raw_request_hash=request_hash,
                raw_response_hash=response_hash,
                tool_declaration_hash=tool_hash,
                usage=usage,
                cost=cost,
                attempts=1,
                finish_reason=(
                    "tool_calls"
                    if self._response_contract == "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2"
                    else provider_finish_reason
                ),
                provider_finish_reason=provider_finish_reason,
                provider_done=provider_done,
            )
        except (TypeError, ValueError, OverflowError) as error:
            del error
            try:
                parsed = payload if payload is not None else _strict_json(body)
                usage = _usage(cast(dict[str, object], parsed))
            except (TypeError, ValueError):
                usage = None
            raise self._error(
                "MODEL_OUTPUT_INVALID",
                attempts=1,
                before_meaningful_behavior=False,
                raw_request_hash=request_hash,
                raw_response_hash=response_hash,
                finish_reason=provider_finish_reason,
                usage=usage,
                provider_done=provider_done,
            ) from None

    async def call_tools(
        self,
        request: ModelRequest,
        declarations: tuple[NativeToolDefinition, ...],
        *,
        history: tuple[NativeToolHistoryTurn, ...] = (),
    ) -> NativeToolResponse:
        try:
            body, tool_hash = self._request_bytes(request, declarations, history)
        except (TypeError, ValueError):
            raise self._error(
                "CONFIGURATION_ERROR",
                attempts=0,
                before_meaningful_behavior=True,
                raw_request_hash=None,
                raw_response_hash=None,
            ) from None
        request_hash = "sha256:" + hashlib.sha256(body).hexdigest()
        try:
            transport_response = await self._transport(body)
        except TransportTransientError:
            raise self._error(
                "MODEL_PROVIDER_TRANSIENT",
                attempts=1,
                before_meaningful_behavior=True,
                raw_request_hash=request_hash,
                raw_response_hash=None,
            ) from None
        except Exception:
            raise self._error(
                "MODEL_PROVIDER_TERMINAL",
                attempts=1,
                before_meaningful_behavior=True,
                raw_request_hash=request_hash,
                raw_response_hash=None,
            ) from None
        if type(transport_response) is not TransportResponse:
            raise self._error(
                "MODEL_PROVIDER_TERMINAL",
                attempts=1,
                before_meaningful_behavior=True,
                raw_request_hash=request_hash,
                raw_response_hash=None,
            )
        response_hash = "sha256:" + hashlib.sha256(transport_response.body).hexdigest()
        if not 200 <= transport_response.status_code < 300:
            raise self._error(
                "MODEL_PROVIDER_TERMINAL",
                attempts=1,
                before_meaningful_behavior=True,
                raw_request_hash=request_hash,
                raw_response_hash=response_hash,
            )
        return self._parse(transport_response.body, request_hash=request_hash, tool_hash=tool_hash)


__all__ = ["OllamaNativeToolClient"]
