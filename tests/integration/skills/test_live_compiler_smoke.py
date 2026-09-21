from __future__ import annotations

import asyncio
from typing import cast

import pytest

from shadowskillbench.models import ModelClient
from shadowskillbench.skills.compiler import CompiledSkillArtifact, CompilerConfig, compile_skill
from shadowskillbench.traces.bundles import DemonstrationBundle

_PLUGIN_NAME = "shadowskillbench_live_compiler"
_CASE_NAME = "live_compiler_case"

LiveCompilerCase = tuple[DemonstrationBundle, ModelClient, CompilerConfig]


def _live_compiler_case(pytestconfig: pytest.Config) -> LiveCompilerCase:
    plugin = pytestconfig.pluginmanager.get_plugin(_PLUGIN_NAME)
    if plugin is None:
        pytest.skip(
            "live compiler plugin is not registered; run with -p shadowskillbench_live_compiler"
        )

    candidate = getattr(plugin, _CASE_NAME, None)
    if candidate is None:
        pytest.fail(
            f"live compiler plugin {_PLUGIN_NAME!r} must provide {_CASE_NAME!r} as a case or "
            "zero-argument factory"
        )
    factory_failed = False
    try:
        case = candidate() if callable(candidate) else candidate
    except Exception:
        factory_failed = True
        case = None
    if factory_failed:
        pytest.fail(f"live compiler plugin {_PLUGIN_NAME!r} {_CASE_NAME!r} factory failed")

    if type(case) is not tuple or len(case) != 3:
        pytest.fail(
            f"live compiler plugin {_PLUGIN_NAME!r} {_CASE_NAME!r} must return an exact "
            "(DemonstrationBundle, ModelClient, CompilerConfig) tuple"
        )
    bundle, client, config = case
    if type(bundle) is not DemonstrationBundle:
        pytest.fail("live compiler case bundle must be an exact DemonstrationBundle")
    if not hasattr(client, "capabilities") or not callable(getattr(client, "structured", None)):
        pytest.fail("live compiler case client must satisfy the ModelClient protocol")
    if type(config) is not CompilerConfig:
        pytest.fail("live compiler case config must be an exact CompilerConfig")
    return bundle, cast(ModelClient, client), config


@pytest.mark.live
def test_live_compiler_smoke_uses_an_explicit_injected_client(pytestconfig: pytest.Config) -> None:
    bundle, client, config = _live_compiler_case(pytestconfig)

    artifact = asyncio.run(compile_skill(bundle, client, config))

    assert type(artifact) is CompiledSkillArtifact
