"""Versioned V4 confirmatory execution and write-once result custody."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.episodes.executor import EpisodeResult, run_episode
from shadowskillbench.episodes.models import EpisodePlan, hash_episode_manifest
from shadowskillbench.experiments.confirmatory_package_v4 import (
    V4_RESULT_NAMESPACE,
    ConfirmatoryExecutionPackageV4,
    ConfirmatoryPackageV4Hold,
)
from shadowskillbench.experiments.planner import PlannedEpisode
from shadowskillbench.models.protocol import ModelClient
from shadowskillbench.skills.compiler import (
    compiled_skill_artifact_hash,
    compiler_manifest_hash,
)

V4_RUNTIME_BINDING_PROFILE = "SSB-CONFIRMATORY-RUNTIME-BINDING4"
V4_RESULT_PROFILE = "SSB-CONFIRMATORY-RUN-BUNDLE-RESULT4"


@dataclass(frozen=True, slots=True)
class ConfirmatoryExecutionResultV4:
    result: EpisodeResult
    result_ref: str
    runtime_binding: dict[str, str]


def _hold(message: str) -> ConfirmatoryPackageV4Hold:
    return ConfirmatoryPackageV4Hold(f"HOLD_CONFIRMATORY_V4_EXECUTION: {message}")


def _safe_result_directory(path: Path) -> Path:
    if not isinstance(path, Path) or path.is_symlink():
        raise _hold("result directory is unsafe")
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise _hold("result path contains a symbolic link")
        absolute.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _hold("result directory is unavailable") from error
    if absolute.is_symlink() or not absolute.is_dir():
        raise _hold("result directory is unsafe")
    return absolute


def _validate_execution_inputs(
    package: ConfirmatoryExecutionPackageV4,
    planned_episode: PlannedEpisode,
    execution_plan: EpisodePlan,
    client: ModelClient,
) -> None:
    if type(package) is not ConfirmatoryExecutionPackageV4:
        raise _hold("exact V4 package is required")
    if type(planned_episode) is not PlannedEpisode or type(execution_plan) is not EpisodePlan:
        raise _hold("exact planned and materialized episode inputs are required")
    package_manifest = package.package_manifest
    if (
        type(package_manifest) is not dict
        or package_manifest.get("profile") != "SSB-CONFIRMATORY-EXECUTION-PACKAGE4"
        or package_manifest.get("runtime_namespace") != "artifacts/experiments/confirmatory-v4"
        or package_manifest.get("result_namespace") != V4_RESULT_NAMESPACE
        or package_manifest.get("package_hash") != package.package_hash
        or type(package_manifest.get("run_descriptor_hash")) is not str
    ):
        raise _hold("package manifest is not an exact V4 runtime binding")
    package.require_episode(
        episode_id=planned_episode.episode_id,
        manifest_hash=planned_episode.manifest_hash,
    )
    manifest = execution_plan.manifest
    expected = planned_episode.manifest_projection()
    if (
        manifest.episode_id != planned_episode.episode_id
        or manifest.stage is not planned_episode.stage
        or manifest.condition is not planned_episode.condition
        or manifest.domain != planned_episode.case.domain
        or manifest.case_id != planned_episode.case.case_id
        or manifest.world_hash != planned_episode.case.world_hash
        or manifest.authority_graph_hash != planned_episode.case.authority_graph_hash
        or manifest.policy_hash != planned_episode.condition_binding.policy_hash
        or manifest.skill_bundle_id != expected["skill_bundle_id"]
        or manifest.contamination_ratio
        != (
            None
            if planned_episode.skill is None
            else float(planned_episode.skill.contamination_ratio)
        )
        or manifest.skill_hash != expected["rendered_skill_hash"]
        or execution_plan.order_assignment is not planned_episode.order_assignment
    ):
        raise _hold("materialized execution plan does not bind the V4 episode")
    if planned_episode.skill is None:
        if execution_plan.skill is not None:
            raise _hold("materialized execution plan has an unplanned skill")
    elif (
        execution_plan.skill is None
        or compiler_manifest_hash(execution_plan.skill.compiler_manifest)
        != planned_episode.skill.compiler_manifest_hash
        or compiled_skill_artifact_hash(execution_plan.skill)
        != planned_episode.skill.compiled_skill_artifact_hash
        or execution_plan.skill.rendered_skill_hash != planned_episode.skill.rendered_skill_hash
    ):
        raise _hold("materialized execution plan does not bind the planned skill")
    if execution_plan.policy is None:
        if planned_episode.condition_binding.policy_hash is not None:
            raise _hold("materialized execution plan is missing the planned policy")
    elif execution_plan.policy.rendered_hash != planned_episode.condition_binding.policy_hash:
        raise _hold("materialized execution plan does not bind the planned policy")
    descriptor = package.run_descriptor
    model = manifest.executor_model
    if (model.provider, model.model, model.model_version_date) != (
        descriptor.provider,
        descriptor.model,
        descriptor.model_version,
    ):
        raise _hold("execution plan does not bind the V4 run descriptor")
    if client.capabilities != descriptor.capabilities:
        raise _hold("executor client capabilities do not bind the V4 run descriptor")


def _result_envelope(
    *,
    package: ConfirmatoryExecutionPackageV4,
    planned_episode: PlannedEpisode,
    result: EpisodeResult,
    runtime_binding: dict[str, str],
    result_ref: str,
) -> bytes:
    value = {
        "profile": V4_RESULT_PROFILE,
        "schema_version": "4.0",
        "result_namespace": V4_RESULT_NAMESPACE,
        "result_ref": result_ref,
        "package_hash": package.package_hash,
        "plan_hash": package.plan_hash,
        "episode_id": planned_episode.episode_id,
        "planned_manifest_hash": planned_episode.manifest_hash,
        "execution_manifest_hash": result.manifest_hash,
        "expected_execution_manifest_hash": result.manifest_hash,
        "result_content_hash": result.content_hash,
        "runtime_binding": runtime_binding,
        "episode_result": result.model_dump(mode="json"),
    }
    return canonical_json_bytes({**value, "content_hash": sha256_ref(value)})


def parse_confirmatory_run_bundle_result_v4(payload: object) -> EpisodeResult:
    """Parse the single V4 result schema used by execution, audit, and analysis."""

    if type(payload) is not dict:
        raise _hold("V4 run-bundle result is invalid")
    required = {
        "profile",
        "schema_version",
        "result_namespace",
        "result_ref",
        "package_hash",
        "plan_hash",
        "episode_id",
        "planned_manifest_hash",
        "execution_manifest_hash",
        "expected_execution_manifest_hash",
        "result_content_hash",
        "runtime_binding",
        "episode_result",
        "content_hash",
    }
    value = payload
    body = (
        {key: value[key] for key in required - {"content_hash"}} if set(value) == required else {}
    )
    if (
        not body
        or value.get("profile") != V4_RESULT_PROFILE
        or value.get("schema_version") != "4.0"
        or value.get("result_namespace") != V4_RESULT_NAMESPACE
        or value.get("content_hash") != sha256_ref(body)
    ):
        raise _hold("V4 run-bundle result custody is invalid")
    try:
        from shadowskillbench.episodes.executor import parse_episode_result

        result = parse_episode_result(value["episode_result"])
    except (TypeError, ValueError) as error:
        raise _hold("V4 run-bundle result is semantically invalid") from error
    runtime = value.get("runtime_binding")
    if (
        value.get("episode_id") != result.episode_id
        or value.get("execution_manifest_hash") != result.manifest_hash
        or value.get("expected_execution_manifest_hash") != result.manifest_hash
        or value.get("result_content_hash") != result.content_hash
        or type(runtime) is not dict
        or runtime.get("profile") != V4_RUNTIME_BINDING_PROFILE
        or runtime.get("package_hash") != value.get("package_hash")
        or runtime.get("plan_hash") != value.get("plan_hash")
        or runtime.get("planned_manifest_hash") != value.get("planned_manifest_hash")
        or runtime.get("result_hash") != result.content_hash
        or runtime.get("result_ref") != value.get("result_ref")
    ):
        raise _hold("V4 run-bundle result does not bind the executed cell")
    return result


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise _hold("result path is unsafe")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        try:
            if path.is_symlink() or path.read_bytes() != payload:
                raise _hold("existing result does not bind the V4 execution")
        except OSError as error:
            raise _hold("existing V4 result is unavailable") from error
    except OSError as error:
        raise _hold("V4 result write failed") from error


async def run_confirmatory_episode_v4(
    *,
    package: ConfirmatoryExecutionPackageV4,
    planned_episode: PlannedEpisode,
    execution_plan: EpisodePlan,
    client: ModelClient,
    result_dir: Path,
) -> ConfirmatoryExecutionResultV4:
    """Execute a descriptor-pinned V4 cell and persist its V4 result envelope once."""

    _validate_execution_inputs(package, planned_episode, execution_plan, client)
    expected_execution_manifest_hash = hash_episode_manifest(execution_plan.manifest)
    result = await run_episode(execution_plan, client, persist=False)
    if result.manifest_hash != expected_execution_manifest_hash:
        raise _hold("executor result does not bind the materialized V4 execution plan")
    result_ref = f"{V4_RESULT_NAMESPACE}/{result.content_hash.removeprefix('sha256:')}.v4.json"
    run_descriptor_hash = package.package_manifest.get("run_descriptor_hash")
    if type(run_descriptor_hash) is not str:
        raise _hold("package V4 run descriptor hash is unavailable")
    runtime_binding = {
        "profile": V4_RUNTIME_BINDING_PROFILE,
        "package_hash": package.package_hash,
        "plan_hash": package.plan_hash,
        "planned_manifest_hash": planned_episode.manifest_hash,
        "run_descriptor_hash": run_descriptor_hash,
        "result_hash": result.content_hash,
        "result_ref": result_ref,
    }
    destination = _safe_result_directory(result_dir)
    _write_once(
        destination / Path(result_ref).name,
        _result_envelope(
            package=package,
            planned_episode=planned_episode,
            result=result,
            runtime_binding=runtime_binding,
            result_ref=result_ref,
        ),
    )
    return ConfirmatoryExecutionResultV4(result, result_ref, runtime_binding)


__all__ = [
    "ConfirmatoryExecutionResultV4",
    "V4_RESULT_PROFILE",
    "V4_RUNTIME_BINDING_PROFILE",
    "parse_confirmatory_run_bundle_result_v4",
    "run_confirmatory_episode_v4",
]
