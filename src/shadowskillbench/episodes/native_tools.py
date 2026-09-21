"""Canonical projection of ``ToolRegistry`` specs into native tool declarations."""

from __future__ import annotations

from typing import cast

from shadowskillbench.engine import JsonObject
from shadowskillbench.episodes.tools import ToolSpec
from shadowskillbench.models.native_tools import NativeToolDefinition


def native_tool_definitions(specs: tuple[ToolSpec, ...]) -> tuple[NativeToolDefinition, ...]:
    """Compile only the visible registry surface; no provider schema is authoritative."""

    if type(specs) is not tuple or not specs:
        raise ValueError("visible tool specifications are invalid")
    definitions: list[NativeToolDefinition] = []
    for spec in specs:
        if type(spec) is not ToolSpec:
            raise ValueError("visible tool specifications are invalid")
        required = spec.argument_schema.get("required")
        optional = spec.argument_schema.get("optional", [])
        if (
            type(required) is not list
            or type(optional) is not list
            or any(type(item) is not str for item in required + optional)
            or len(set(required + optional)) != len(required + optional)
        ):
            raise ValueError("tool specification arguments are invalid")
        parameters = cast(
            JsonObject,
            {
                "type": "object",
                "properties": {name: {} for name in required + optional},
                "required": list(required),
                "additionalProperties": False,
            },
        )
        definitions.append(
            NativeToolDefinition(
                name=spec.name,
                description=spec.description,
                parameters=parameters,
            )
        )
    return tuple(definitions)


__all__ = ["native_tool_definitions"]
