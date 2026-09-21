"""Strict Ollama ``/api/generate`` raw structured-output client."""

from __future__ import annotations

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


class _ConfigurationFailure(Exception):
    pass


class _SchemaFailure(Exception):
    pass


def _strict_json(body: bytes) -> object:
    try:
        return json.loads(
            body.decode("utf-8", "strict"),
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


class OllamaNativeClient:
    """One-attempt, hash-only adapter for Ollama's raw native generate API."""

    def __init__(
        self,
        *,
        capabilities: ProviderCapabilities,
        transport: AsyncTransport,
        max_attempts: int,
        pricing: TokenPricing | None = None,
        top_p: float = 1.0,
        context_length: int = 131_072,
        reasoning_effort: Literal["none", "low"] = "none",
        include_top_p: bool = True,
    ) -> None:
        if (
            type(capabilities) is not ProviderCapabilities
            or not callable(transport)
            or type(max_attempts) is not int
            or not 1 <= max_attempts <= 5
            or pricing is not None
            and type(pricing) is not TokenPricing
            or type(top_p) is not float
            or top_p != 1.0
            or type(context_length) is not int
            or context_length != 131_072
            or type(reasoning_effort) is not str
            or reasoning_effort not in {"none", "low"}
            or type(include_top_p) is not bool
            or (reasoning_effort, include_top_p) not in {("none", True), ("low", False)}
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
        self._top_p = top_p
        self._context_length = context_length
        self._reasoning_effort: Literal["none", "low"] = reasoning_effort
        self._include_top_p = include_top_p

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def reasoning_effort(self) -> str:
        """The explicit Harmony reasoning binding for this native request profile."""

        return self._reasoning_effort

    @property
    def transport_profile(self) -> str:
        return "ollama_native_api_generate_raw"

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

    def _preflight(self, request: ModelRequest, schema: type[T]) -> tuple[str, bytes]:
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
        options: dict[str, object] = {
            "num_ctx": self._context_length,
            "num_predict": request.max_tokens,
            "temperature": request.temperature,
        }
        if self._include_top_p:
            options["top_p"] = self._top_p
        if request.seed is not None:
            options["seed"] = request.seed
        from shadowskillbench.experiments.ollama_gpt_oss_profile import (
            OllamaRoleProfileError,
            render_harmony_native_generate_prompt,
        )

        try:
            prompt = render_harmony_native_generate_prompt(
                messages=request.messages,
                reasoning_effort=self._reasoning_effort,
            )
        except OllamaRoleProfileError as error:
            raise _ConfigurationFailure() from error
        return schema_digest, canonical_json_bytes(
            {
                "format": schema_object,
                "model": caps.model,
                "options": options,
                "prompt": prompt,
                "raw": True,
                "stream": False,
            }
        )

    async def structured(self, request: ModelRequest, schema: type[T]) -> ModelResponse[T]:
        try:
            schema_digest, body = self._preflight(request, schema)
        except _SchemaFailure:
            raise self._error("SCHEMA_ERROR", 0, True, None, None) from None
        except (TypeError, ValueError, _ConfigurationFailure):
            raise self._error("CONFIGURATION_ERROR", 0, True, None, None) from None
        request_hash = "sha256:" + hashlib.sha256(body).hexdigest()
        attempts = 0
        while attempts < self._max_attempts:
            attempts += 1
            try:
                returned = await self._transport(body)
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                if not isinstance(error, Exception):
                    raise
                if isinstance(error, (TransportTransientError, TimeoutError, ConnectionError)):
                    if attempts < self._max_attempts:
                        continue
                    code = "MODEL_PROVIDER_TRANSIENT"
                else:
                    code = "MODEL_PROVIDER_TERMINAL"
                raise self._error(code, attempts, True, request_hash, None) from None
            try:
                returned = TransportResponse.model_validate(
                    _model_data(returned, TransportResponse)
                )
            except ValueError:
                raise self._error(
                    "MODEL_PROVIDER_TERMINAL", attempts, True, request_hash, None
                ) from None
            response_hash = "sha256:" + hashlib.sha256(returned.body).hexdigest()
            if returned.status_code in {408, 409, 425, 429} or returned.status_code >= 500:
                if attempts < self._max_attempts:
                    continue
                raise self._error(
                    "MODEL_PROVIDER_TRANSIENT", attempts, True, request_hash, response_hash
                )
            if returned.status_code != 200:
                raise self._error(
                    "MODEL_PROVIDER_TERMINAL", attempts, True, request_hash, response_hash
                )
            try:
                output, usage = self._parse_success(returned.body, schema)
            except Exception:
                finish_reason, reported_usage = self._response_diagnostics(returned.body)
                raise self._error(
                    "MODEL_OUTPUT_INVALID",
                    attempts,
                    not self._has_meaningful_behavior(returned.body),
                    request_hash,
                    response_hash,
                    finish_reason,
                    reported_usage,
                ) from None
            try:
                cost = _calculate_cost(usage, self._pricing) if self._pricing is not None else None
                response = ModelResponse[schema](
                    capabilities=self._capabilities,
                    output=output,
                    raw_request_hash=request_hash,
                    raw_response_hash=response_hash,
                    structured_output_schema_hash=schema_digest,
                    usage=usage,
                    cost=cost,
                    attempts=attempts,
                )
            except OverflowError:
                raise self._error(
                    "CONFIGURATION_ERROR", attempts, False, request_hash, response_hash
                ) from None
            except Exception:
                raise self._error(
                    "MODEL_OUTPUT_INVALID", attempts, False, request_hash, response_hash
                ) from None
            return response
        raise self._error("MODEL_PROVIDER_TRANSIENT", attempts, True, request_hash, None)

    def _has_meaningful_behavior(self, body: bytes) -> bool:
        try:
            payload = _strict_json(body)
        except ValueError:
            return False
        if type(payload) is not dict:
            return False
        content = payload.get("response")
        return type(content) is str and bool(content.strip())

    def _response_diagnostics(self, body: bytes) -> tuple[str | None, TokenUsage | None]:
        try:
            payload = _strict_json(body)
        except ValueError:
            return None, None
        if type(payload) is not dict:
            return None, None
        finish_reason = payload.get("done_reason")
        if not (
            type(finish_reason) is str
            and finish_reason.strip()
            and len(finish_reason) <= 32
            and finish_reason.isprintable()
        ):
            finish_reason = None
        try:
            input_tokens = payload["prompt_eval_count"]
            output_tokens = payload["eval_count"]
            usage = TokenUsage(
                input_tokens=cast(Any, input_tokens),
                output_tokens=cast(Any, output_tokens),
                total_tokens=cast(Any, input_tokens) + cast(Any, output_tokens),
            )
        except Exception:
            usage = None
        return cast(str | None, finish_reason), usage

    def _parse_success(self, body: bytes, schema: type[T]) -> tuple[T, TokenUsage]:
        payload = _strict_json(body)
        if type(payload) is not dict:
            raise ValueError("envelope")
        if (
            payload.get("model") != self._capabilities.model
            or payload.get("done") is not True
            or payload.get("done_reason") != "stop"
            or type(payload.get("response")) is not str
        ):
            raise ValueError("envelope")
        input_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        usage = TokenUsage(
            input_tokens=cast(Any, input_tokens),
            output_tokens=cast(Any, output_tokens),
            total_tokens=cast(Any, input_tokens) + cast(Any, output_tokens),
        )
        parsed = _strict_json(cast(str, payload["response"]).encode("utf-8"))
        if type(parsed) is not dict:
            raise ValueError("content")
        output = schema.model_validate(parsed, strict=True, extra="forbid", from_attributes=False)
        if type(output) is not schema:
            raise ValueError("output type")
        return cast(T, _revalidate_caller_output(output, schema)), usage


__all__ = ["OllamaNativeClient"]
