from __future__ import annotations

# ruff: noqa: E501
from typing import Literal, cast

from shadowskillbench.models.openai_compatible import OpenAICompatibleClient
from shadowskillbench.models.protocol import (
    ProviderCapabilities,
    TokenPricing,
    TransportResponse,
    TransportTerminalError,
    TransportTransientError,
    _model_data,
)


class ScriptedModelClient(OpenAICompatibleClient):
    def __init__(
        self,
        *,
        capabilities: ProviderCapabilities,
        script: tuple[TransportResponse | Literal["transient", "terminal"], ...],
        max_attempts: int,
        pricing: TokenPricing | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if type(script) is not tuple:
            raise ValueError("script must be an exact tuple")
        admitted: list[TransportResponse | Literal["transient", "terminal"]] = []
        for step in script:
            if type(step) is TransportResponse:
                admitted.append(
                    TransportResponse.model_validate(_model_data(step, TransportResponse))
                )
            elif type(step) is str and step in {"transient", "terminal"}:
                admitted.append(cast(Literal["transient", "terminal"], step))
            else:
                raise ValueError("script entry is invalid")
        self._script = tuple(admitted)
        self._cursor = 0
        self._recorded: list[bytes] = []

        async def transport(body: bytes) -> TransportResponse:
            self._recorded.append(bytes(body))
            if self._cursor >= len(self._script):
                raise TransportTerminalError()
            step = self._script[self._cursor]
            self._cursor += 1
            if step == "transient":
                raise TransportTransientError()
            if step == "terminal":
                raise TransportTerminalError()
            return step

        super().__init__(
            capabilities=capabilities,
            transport=transport,
            max_attempts=max_attempts,
            pricing=pricing,
            reasoning_effort=reasoning_effort,
        )

    @property
    def recorded_request_bodies(self) -> tuple[bytes, ...]:
        return tuple(bytes(body) for body in self._recorded)


__all__ = ["ScriptedModelClient"]
