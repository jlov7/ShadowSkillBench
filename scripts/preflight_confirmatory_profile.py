"""Run a secret-free exploratory preflight of the frozen confirmatory model profiles."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.confirmatory import AUTHORITY_CLASSES
from shadowskillbench.episodes.executor import AgentTurn, run_episode
from shadowskillbench.episodes.models import EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.confirmatory_compilation import (
    load_confirmatory_compiler_config,
)
from shadowskillbench.experiments.confirmatory_package import _parse_descriptor
from shadowskillbench.experiments.confirmatory_preparation import (
    _build_case,
    _bundle,
    _case_binding,
    _condition_bindings,
    _materialize_episode,
)
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_MODEL,
    OLLAMA_GPT_OSS_PROVIDER,
    OLLAMA_GPT_OSS_REASONING_EFFORT,
    load_ollama_role_profile_receipt,
)
from shadowskillbench.experiments.planner import PlannedEpisode, SkillBundleBinding
from shadowskillbench.models.protocol import ModelAdapterError
from shadowskillbench.models.runtime import (
    ModelRunDescriptor,
    ollama_native_client_from_run_descriptor,
)
from shadowskillbench.skills.compiler import (
    CompilerContractError,
    compile_skill,
    compiled_skill_artifact_hash,
)
from shadowskillbench.skills.confirmatory_wire import ConfirmatorySkillIRWire
from shadowskillbench.traces.bundles import generate_bundle

_LABEL = "EXPLORATORY_CONFIRMATORY_PROFILE_PREFLIGHT_NOT_CONFIRMATORY"
_PREFLIGHT_CORPUS_SEED = 104729
_COMPILER_ENDPOINT = "http://127.0.0.1:11435/api/generate"
_COMPILER_PROFILE = "SSB-CONFIRMATORY-COMPILER6"


def _usage(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "input_tokens": cast(int, getattr(value, "input_tokens")),
        "output_tokens": cast(int, getattr(value, "output_tokens")),
        "total_tokens": cast(int, getattr(value, "total_tokens")),
    }


async def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.compiler_endpoint != _COMPILER_ENDPOINT:
        raise ValueError("compiler endpoint must be the pinned native generate route")
    root = Path.cwd()
    output = cast(Path, args.output).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=False)
    receipt = load_ollama_role_profile_receipt(cast(Path, args.role_profile_receipt))
    compiler_config = load_confirmatory_compiler_config(root)
    compiler_descriptor = ModelRunDescriptor(
        provider=OLLAMA_GPT_OSS_PROVIDER,
        model=OLLAMA_GPT_OSS_MODEL,
        model_version=cast(str, receipt["profile_model_digest"]),
        endpoint=cast(str, args.compiler_endpoint),
        api_key_environment=cast(str, args.api_key_environment),
        max_attempts=1,
    )
    executor_descriptor = _parse_descriptor(
        json.loads(cast(Path, args.run_descriptor).read_bytes())
    )
    profile = cast(dict[str, object], executor_descriptor.executor_runtime_profile)
    compiler_client = ollama_native_client_from_run_descriptor(
        compiler_descriptor,
        trust_env=False,
        timeout_seconds=900.0,
        reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
        include_top_p=False,
    )
    executor_client = ollama_native_client_from_run_descriptor(
        executor_descriptor,
        trust_env=False,
        timeout_seconds=float(cast(int, profile["timeout_seconds"])),
        top_p=cast(float, profile["top_p"]),
        context_length=cast(int, profile["context_length"]),
    )
    bundles = tuple(
        _bundle(_PREFLIGHT_CORPUS_SEED, domain, Decimal("0"), 0)
        for domain in ("access_provisioning", "financial_adjustments")
    )
    stage_a_cases = tuple(
        _build_case("stage_a", _PREFLIGHT_CORPUS_SEED, domain, AUTHORITY_CLASSES[0], 0)
        for domain in ("access_provisioning", "financial_adjustments")
    )
    condition_bindings = _condition_bindings()
    cells: list[dict[str, object]] = []
    for domain in ("access_provisioning", "financial_adjustments"):
        bundle_seed = next(
            item
            for item in bundles
            if item.domain == domain and item.contamination_ratio == Decimal("0")
        )
        generated = generate_bundle(domain, Decimal("0"), 12, bundle_seed.seed)
        started = time.perf_counter()
        try:
            artifact = await compile_skill(
                generated.bundle,
                compiler_client,
                compiler_config.compiler_config,
                output_schema=ConfirmatorySkillIRWire,
            )
        except (ModelAdapterError, CompilerContractError) as error:
            failure = {
                "label": _LABEL,
                "domain": domain,
                "phase": "compiler",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error_code": error.code,
                "finish_reason": error.finish_reason,
                "reported_usage": _usage(error.reported_usage),
                "raw_request_hash": error.raw_request_hash,
                "raw_response_hash": error.raw_response_hash,
                "raw_provider_trace_retained": False,
            }
            (output / "failure-receipt.json").write_bytes(canonical_json_bytes(failure))
            raise RuntimeError(f"{domain} compiler HOLD") from None
        compiler_elapsed = time.perf_counter() - started
        case = next(item for item in stage_a_cases if item.domain == domain)
        skill_binding = SkillBundleBinding(
            domain=bundle_seed.domain,
            bundle_id=bundle_seed.bundle_id,
            contamination_ratio=bundle_seed.contamination_ratio,
            source_manifest_hash=bundle_seed.source_manifest_hash,
            compiler_manifest_hash=artifact.compiler_manifest_hash,
            compiled_skill_artifact_hash=compiled_skill_artifact_hash(artifact),
            rendered_skill_hash=artifact.rendered_skill_hash,
        )
        condition_binding = next(
            item
            for item in condition_bindings
            if item.domain == domain and item.condition is ExperimentCondition.A2_SKILL_ONLY
        )
        planned = PlannedEpisode(
            stage=EpisodeStage.CONFIRMATORY_A,
            condition=ExperimentCondition.A2_SKILL_ONLY,
            case=_case_binding(case, stage_b=False),
            condition_binding=condition_binding,
            repeat_index=1,
            skill=skill_binding,
        )
        plan = _materialize_episode(
            planned,
            {case.case_id: case},
            {bundle_seed.bundle_id: artifact},
            executor_descriptor,
        )
        started = time.perf_counter()
        previous_cwd = Path.cwd()
        try:
            os.chdir(output)
            result = await run_episode(plan, executor_client, temperature=0.0)
        finally:
            os.chdir(previous_cwd)
        executor_elapsed = time.perf_counter() - started
        cells.append(
            {
                "domain": domain,
                "ratio": "R0",
                "bundle_id": bundle_seed.bundle_id,
                "compiler_elapsed_seconds": round(compiler_elapsed, 3),
                "compiler_usage": _usage(artifact.usage),
                "compiler_artifact_hash": compiled_skill_artifact_hash(artifact),
                "executor_elapsed_seconds": round(executor_elapsed, 3),
                "executor_turns": result.overhead.turns,
                "executor_tool_calls": result.overhead.tool_calls,
                "executor_usage": {
                    "input_tokens": result.overhead.input_tokens,
                    "output_tokens": result.overhead.output_tokens,
                    "total_tokens": result.overhead.total_tokens,
                },
                "disposition": result.disposition.value,
                "claim": result.claim.value,
                "result_hash": result.content_hash,
            }
        )
    summary = {
        "label": _LABEL,
        "preflight_corpus_seed": _PREFLIGHT_CORPUS_SEED,
        "compiler_profile": _COMPILER_PROFILE,
        "executor_output_schema_hash": sha256_ref(AgentTurn.model_json_schema()),
        "raw_provider_trace_retained": False,
        "cells": cells,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-endpoint", required=True)
    parser.add_argument("--api-key-environment", required=True)
    parser.add_argument("--role-profile-receipt", type=Path, required=True)
    parser.add_argument("--run-descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = asyncio.run(_run(args))
    print(
        f"PASS_{_LABEL}: cells={len(cast(list[object], summary['cells']))} "
        f"summary={args.output / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
