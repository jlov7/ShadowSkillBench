from __future__ import annotations

# ruff: noqa: E501
import asyncio
import hashlib
import json
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.models.protocol import (
    AsyncTransport,
    ModelAdapterError,
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    TokenPricing,
    TokenUsage,
    TransportResponse,
    TransportTransientError,
    _calculate_cost,
    _model_data,
    _revalidate_caller_output,
    _schema_hash,
)

T = TypeVar("T", bound=BaseModel)
type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high"})


class _ConfigurationFailure(Exception):
    pass


class _SchemaFailure(Exception):
    pass


def _strict_json(body: bytes) -> object:
    try:
        text = body.decode("utf-8", "strict")
        return json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite {value}")),
            object_pairs_hook=_no_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON") from error


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        capabilities: ProviderCapabilities,
        transport: AsyncTransport,
        max_attempts: int,
        pricing: TokenPricing | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if (
            type(capabilities) is not ProviderCapabilities
            or not callable(transport)
            or type(max_attempts) is not int
            or not 1 <= max_attempts <= 5
            or pricing is not None
            and type(pricing) is not TokenPricing
            or reasoning_effort is not None
            and (type(reasoning_effort) is not str or reasoning_effort not in _REASONING_EFFORTS)
        ):
            raise ValueError("client configuration is invalid")
        self._capabilities = ProviderCapabilities.model_validate(
            _model_data(capabilities, ProviderCapabilities)
        )
        self._transport = transport
        self._max_attempts = max_attempts
        self._pricing = (
            TokenPricing.model_validate(_model_data(pricing, TokenPricing))
            if pricing is not None
            else None
        )
        self._reasoning_effort = cast(ReasoningEffort | None, reasoning_effort)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def reasoning_effort(self) -> ReasoningEffort | None:
        return cast(ReasoningEffort | None, self._reasoning_effort)

    def _error(
        self,
        code: str,
        attempts: int,
        before: bool,
        request_hash: str | None,
        response_hash: str | None,
        finish_reason: str | None = None,
        reported_usage: TokenUsage | None = None,
    ) -> ModelAdapterError:
        return ModelAdapterError(
            code=code,
            attempts=attempts,
            before_meaningful_behavior=before,
            raw_request_hash=request_hash,
            raw_response_hash=response_hash,
            finish_reason=finish_reason,
            reported_usage=reported_usage,
        )

    def _preflight(
        self, request: ModelRequest, schema: type[T]
    ) -> tuple[dict[str, object], str, bytes]:
        request = ModelRequest.model_validate(_model_data(request, ModelRequest))
        caps = ProviderCapabilities.model_validate(
            _model_data(self._capabilities, ProviderCapabilities)
        )
        if (
            not caps.supports_structured_output
            or any(message.role == "system" for message in request.messages)
            and not caps.supports_system_role
            or any(message.role == "developer" for message in request.messages)
            and not caps.supports_developer_role
            or request.seed is not None
            and not caps.supports_seed
        ):
            raise _ConfigurationFailure()
        try:
            schema_object, schema_digest = _schema_hash(schema)
        except Exception as error:
            raise _SchemaFailure() from error
        body: dict[str, object] = {
            "model": caps.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "shadowskillbench_output",
                    "strict": True,
                    "schema": schema_object,
                },
            },
        }
        if request.seed is not None:
            body["seed"] = request.seed
        if self._reasoning_effort is not None:
            body["reasoning_effort"] = self._reasoning_effort
        return schema_object, schema_digest, canonical_json_bytes(body)

    async def structured(self, request: ModelRequest, schema: type[T]) -> ModelResponse[T]:
        preflight_code: str | None = None
        preflight: tuple[dict[str, object], str, bytes] | None = None
        try:
            preflight = self._preflight(request, schema)
        except (TypeError, ValueError):
            preflight_code = "CONFIGURATION_ERROR"
        except _ConfigurationFailure:
            preflight_code = "CONFIGURATION_ERROR"
        except _SchemaFailure:
            preflight_code = "SCHEMA_ERROR"
        if preflight_code is not None:
            raise self._error(preflight_code, 0, True, None, None)
        if preflight is None:
            raise RuntimeError("unreachable preflight state")
        _, schema_digest, body = preflight
        request_hash = "sha256:" + hashlib.sha256(body).hexdigest()
        attempts = 0
        while attempts < self._max_attempts:
            attempts += 1
            transport_code: str | None = None
            raw_response: object | None = None
            try:
                raw_response = await self._transport(body)
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError) or not isinstance(error, Exception):
                    raise
                transient = type(error) is TransportTransientError or isinstance(
                    error, (TimeoutError, ConnectionError)
                )
                if not transient:
                    transport_code = "MODEL_PROVIDER_TERMINAL"
                elif attempts == self._max_attempts:
                    transport_code = "MODEL_PROVIDER_TRANSIENT"
                if transport_code is None:
                    continue
            if transport_code is not None:
                raise self._error(transport_code, attempts, True, request_hash, None)
            returned: TransportResponse | None = None
            try:
                returned = TransportResponse.model_validate(
                    _model_data(raw_response, TransportResponse)
                )
            except ValueError:
                response_code = "MODEL_PROVIDER_TERMINAL"
            else:
                response_code = None
            if response_code is not None:
                raise self._error(response_code, attempts, True, request_hash, None)
            if returned is None:
                raise RuntimeError("unreachable response state")
            response_hash = "sha256:" + hashlib.sha256(returned.body).hexdigest()
            if returned.status_code in {408, 409, 425, 429} or returned.status_code >= 500:
                if attempts == self._max_attempts:
                    raise self._error(
                        "MODEL_PROVIDER_TRANSIENT", attempts, True, request_hash, response_hash
                    )
                continue
            if returned.status_code != 200:
                raise self._error(
                    "MODEL_PROVIDER_TERMINAL", attempts, True, request_hash, response_hash
                )
            parsed_result: tuple[T, TokenUsage] | None = None
            before = True
            try:
                parsed_result = self._parse_success(returned.body, schema)
            except Exception:
                before = not self._has_meaningful_behavior(returned.body)
            if parsed_result is None:
                finish_reason, reported_usage = self._response_diagnostics(returned.body)
                raise self._error(
                    "MODEL_OUTPUT_INVALID",
                    attempts,
                    before,
                    request_hash,
                    response_hash,
                    finish_reason,
                    reported_usage,
                )
            output, usage = parsed_result
            cost = None
            try:
                cost = _calculate_cost(usage, self._pricing) if self._pricing is not None else None
            except OverflowError:
                cost_failure = True
            else:
                cost_failure = False
            if cost_failure:
                raise self._error(
                    "CONFIGURATION_ERROR", attempts, False, request_hash, response_hash
                )
            response: ModelResponse[T] | None = None
            try:
                response_type = ModelResponse[schema]
                response = response_type(
                    capabilities=self._capabilities,
                    output=output,
                    raw_request_hash=request_hash,
                    raw_response_hash=response_hash,
                    structured_output_schema_hash=schema_digest,
                    usage=usage,
                    cost=cost,
                    attempts=attempts,
                )
            except Exception:
                response_failure = True
            else:
                response_failure = False
            if response_failure or response is None:
                raise self._error(
                    "MODEL_OUTPUT_INVALID", attempts, False, request_hash, response_hash
                )
            return response
        raise self._error("MODEL_PROVIDER_TRANSIENT", attempts, True, request_hash, None)

    def _has_meaningful_behavior(self, body: bytes) -> bool:
        try:
            payload = _strict_json(body)
        except Exception:
            return False
        if type(payload) is not dict:
            return False
        choices = payload.get("choices")
        if type(choices) is not list:
            return False
        for choice in choices:
            if type(choice) is not dict:
                continue
            message = choice.get("message")
            if type(message) is not dict:
                continue
            content = message.get("content")
            refusal = message.get("refusal")
            if (type(content) is str and bool(content.strip())) or (
                type(refusal) is str and bool(refusal.strip())
            ):
                return True
        return False

    def _response_diagnostics(self, body: bytes) -> tuple[str | None, TokenUsage | None]:
        """Best-effort finish reason and usage from an invalid 200 body; never raises."""

        try:
            payload = _strict_json(body)
        except Exception:
            return None, None
        if type(payload) is not dict:
            return None, None
        finish_reason: str | None = None
        choices = payload.get("choices")
        if type(choices) is list and len(choices) == 1 and type(choices[0]) is dict:
            candidate = choices[0].get("finish_reason")
            if (
                type(candidate) is str
                and candidate.strip()
                and len(candidate) <= 32
                and candidate.isprintable()
            ):
                finish_reason = candidate
        reported_usage: TokenUsage | None = None
        usage_raw = payload.get("usage")
        if type(usage_raw) is dict:
            try:
                reported_usage = TokenUsage(
                    input_tokens=cast(Any, usage_raw.get("prompt_tokens")),
                    output_tokens=cast(Any, usage_raw.get("completion_tokens")),
                    total_tokens=cast(Any, usage_raw.get("total_tokens")),
                )
            except Exception:
                reported_usage = None
        return finish_reason, reported_usage

    def _parse_success(self, body: bytes, schema: type[T]) -> tuple[T, TokenUsage]:
        payload = _strict_json(body)
        if type(payload) is not dict:
            raise ValueError("envelope")
        model = payload.get("model")
        choices = payload.get("choices")
        usage_raw = payload.get("usage")
        if (
            model != self._capabilities.model
            or type(choices) is not list
            or len(choices) != 1
            or type(usage_raw) is not dict
        ):
            raise ValueError("envelope")
        choice = choices[0]
        if (
            type(choice) is not dict
            or type(choice.get("index")) is not int
            or choice.get("index") != 0
            or choice.get("finish_reason") != "stop"
        ):
            raise ValueError("choice")
        message = choice.get("message")
        if type(message) is not dict or message.get("role") != "assistant":
            raise ValueError("message")
        content = message.get("content")
        refusal = message.get("refusal")
        if refusal is not None or type(content) is not str:
            raise ValueError("refusal/content")
        usage_object = cast(dict[str, Any], usage_raw)
        usage = TokenUsage(
            input_tokens=cast(int, usage_object.get("prompt_tokens")),
            output_tokens=cast(int, usage_object.get("completion_tokens")),
            total_tokens=cast(int, usage_object.get("total_tokens")),
        )
        parsed = _strict_json(content.encode("utf-8"))
        if type(parsed) is not dict:
            raise ValueError("content")
        output = schema.model_validate(parsed, strict=True, extra="forbid", from_attributes=False)
        if type(output) is not schema:
            raise ValueError("output type")
        return cast(T, _revalidate_caller_output(output, schema)), usage


__all__ = ["OpenAICompatibleClient"]
