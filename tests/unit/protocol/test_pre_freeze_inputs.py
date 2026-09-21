from __future__ import annotations

import json
from pathlib import Path

import yaml

from shadowskillbench.episodes.context import _RUNTIME_TEXT
from shadowskillbench.experiments.confirmatory_preparation import (
    CORPUS_SEED,
    DEVELOPMENT_CORPUS_SEED,
)
from shadowskillbench.experiments.live_pilot import OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_BASE_MODEL,
    OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
    OLLAMA_GPT_OSS_CONTEXT_LENGTH,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_MODEL,
    OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_SERVER_VERSION,
)
from shadowskillbench.protocol.freeze import REQUIRED_INPUTS
from shadowskillbench.protocol.runtime_prompt_manifest import runtime_prompt_manifest_bytes

ROOT = Path(__file__).resolve().parents[3]


def _yaml(path: str) -> dict[str, object]:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def test_frozen_model_and_sampling_configs_bind_the_local_gpt_oss_profile() -> None:
    models = _yaml("config/models.yaml")
    compiler = _yaml("config/compiler.yaml")
    executor = _yaml("config/executor.yaml")

    assert models["ollama_server_version"] == OLLAMA_GPT_OSS_SERVER_VERSION
    assert models["base_model"] == {
        "name": OLLAMA_GPT_OSS_BASE_MODEL,
        "digest": OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
    }
    assert models["compiler"] == {
        "model": OLLAMA_GPT_OSS_MODEL,
        "derived_model_digest": (
            "sha256:a939fca6d222577c1a6534f6be3c9f01f69ef646081b3d44e924257d3eb04289"
        ),
    }
    executor_model = models["executor"]
    server = models["server"]
    assert type(executor_model) is dict
    assert type(server) is dict
    assert executor_model["model"] == OLLAMA_GPT_OSS_EXECUTOR_MODEL
    assert executor_model["derived_model_digest"] == OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST
    assert server["context_length"] == OLLAMA_GPT_OSS_CONTEXT_LENGTH
    assert compiler["temperature"] == OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE
    assert compiler["max_tokens"] == OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS
    assert executor["temperature"] == OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE
    assert executor["temperature"] == 1.0
    assert executor["reasoning_effort"] == "none"
    assert executor["profile"] == "SSB-CONFIRMATORY-EXECUTOR6"
    assert executor["max_turns"] == executor["max_tool_calls"] == 12
    assert executor["max_tokens"] == 8192

    runtime_manifest = json.loads(runtime_prompt_manifest_bytes())
    assert runtime_manifest["profile"] == "SSB-CONFIRMATORY-RUNTIME-PROMPT-MANIFEST6"


def test_confirmatory_seed_is_frozen_and_disjoint_from_the_exploratory_preflight() -> None:
    corpora = _yaml("config/corpora.yaml")
    preflight = (ROOT / "scripts/preflight_confirmatory_profile.py").read_text(encoding="utf-8")

    assert corpora["corpus_seed"] == CORPUS_SEED == 104730
    assert corpora["development_seed"] == DEVELOPMENT_CORPUS_SEED == 4242
    assert "_PREFLIGHT_CORPUS_SEED = 104729" in preflight
    assert CORPUS_SEED != 104729


def test_preflight_operator_path_is_pinned_to_compiler5_native_generate() -> None:
    preflight = (ROOT / "scripts/preflight_confirmatory_profile.py").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/17_PROTOCOL_FREEZE_RUNBOOK.md").read_text(encoding="utf-8")

    assert '"http://127.0.0.1:11435/api/generate"' in preflight
    assert "    client_from_run_descriptor," not in preflight
    assert "compiler_client = ollama_native_client_from_run_descriptor(" in preflight
    assert "timeout_seconds=900.0" in preflight
    assert "reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT" in preflight
    assert "include_top_p=False" in preflight
    assert '"SSB-CONFIRMATORY-COMPILER6"' in preflight
    assert "SSB-CONFIRMATORY-COMPILER2" not in preflight
    assert "--compiler-endpoint http://127.0.0.1:11435/api/generate" in runbook
    assert "--compiler-endpoint http://127.0.0.1:11435/v1/chat/completions" not in runbook
    assert "SSB-CONFIRMATORY-COMPILER2" not in runbook


def test_executor_system_input_matches_the_runtime_base_system_message() -> None:
    assert (ROOT / "prompts/executor_system.md").read_text(encoding="utf-8").rstrip(
        "\n"
    ) == _RUNTIME_TEXT


def test_freeze_surface_uses_the_real_executor_system_prompt_only() -> None:
    paths_by_role = {item.role: item.path for item in REQUIRED_INPUTS}

    assert paths_by_role["executor_system_prompt"] == "prompts/executor_system.md"
    assert "executor_developer_prompt" not in paths_by_role
    assert {
        "protocol/analysis_plan.v3.json",
        "protocol/power_precision_plan.v3.json",
        "protocol/power_precision_evidence.v3.json",
        "src/shadowskillbench/protocol/power_precision.py",
        "src/shadowskillbench/protocol/power_precision_design.py",
        "config/models.yaml",
        "config/compiler.yaml",
        "config/executor.yaml",
        "config/corpora.yaml",
        "config/statistics.yaml",
        "src/shadowskillbench/protocol/claim_estimands.py",
        "src/shadowskillbench/protocol/claims.py",
        "src/shadowskillbench/protocol/scientific_freeze.py",
        "src/shadowskillbench/analysis/profile.py",
        "src/shadowskillbench/analysis/sealed.py",
        "src/shadowskillbench/analysis/stage_a.py",
        "src/shadowskillbench/analysis/stage_b.py",
    } <= {item.path for item in REQUIRED_INPUTS}
