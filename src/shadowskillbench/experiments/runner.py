from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.episodes.executor import (
    AgentClaim,
    EpisodeDisposition,
    EpisodeResult,
    episode_result_projection,
    parse_episode_result,
    run_episode,
)
from shadowskillbench.episodes.models import EpisodePlan, hash_episode_manifest
from shadowskillbench.experiments.audit import FrozenArtifactCatalog, FrozenExclusionRule
from shadowskillbench.experiments.development_execution import (
    development_episode_id,
    development_episode_projection,
    scripted_development_client,
    scripted_development_response,
    visible_execution_input,
)
from shadowskillbench.experiments.development_plan import DevelopmentPlannedEpisode
from shadowskillbench.experiments.planner import (
    ConfirmatoryEpisodePlan,
    PlannedEpisode,
    episode_manifest_projection,
)
from shadowskillbench.models.protocol import ModelAdapterError, ModelClient
from shadowskillbench.skills.compiler import compiled_skill_artifact_hash, compiler_manifest_hash

_PROFILE = "SSB-RUN1"
_AUDIT_BINDING_PROFILE = "SSB-AUDIT-BINDINGS1"
_LEDGER_EVENTS = frozenset({"RUN_RESUMED", "EPISODE_RETRY", "EPISODE_COMPLETED"})
_DEVELOPMENT_PROFILE = "SSB-DEVELOPMENT-RUN1"
_DEVELOPMENT_TELEMETRY_PROFILE = "SSB-DEVELOPMENT-TELEMETRY1"


class EpisodeMaterializer(Protocol):
    def __call__(self, episode: PlannedEpisode) -> EpisodePlan: ...


class ModelClientFactory(Protocol):
    def __call__(self, episode: PlannedEpisode) -> ModelClient: ...


type EpisodeExecutor = Callable[[EpisodePlan, ModelClient], Awaitable[EpisodeResult]]


class RunnerError(RuntimeError):
    pass


class RunCustodyError(RunnerError):
    pass


@dataclass(frozen=True, slots=True)
class RunSummary:
    plan_hash: str
    completed_episode_ids: tuple[str, ...]
    skipped_episode_ids: tuple[str, ...]
    retried_episode_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DevelopmentRunSummary:
    plan_hash: str
    completed_episode_ids: tuple[str, ...]
    skipped_episode_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PersistedEpisode:
    content_hash: str
    execution_manifest_hash: str


def _json_object(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise RunCustodyError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise RunCustodyError(f"{label} is a symlink")
    try:
        return _json_object(json.loads(path.read_bytes()), label=label)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunCustodyError(f"{label} is unreadable") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_directory(path: Path) -> None:
    """Create a run directory only when every existing path component is direct."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise RunCustodyError("run path contains a symlink")
        absolute.mkdir(parents=True, exist_ok=True)
        if absolute.is_symlink() or not absolute.is_dir():
            raise RunCustodyError("run directory is unsafe")
    except RunCustodyError:
        raise
    except OSError as error:
        raise RunCustodyError("run directory is unavailable") from error


def _write_exclusive_json(path: Path, value: dict[str, object]) -> bool:
    _safe_directory(path.parent)
    if path.exists() and path.is_symlink():
        raise RunCustodyError("run artifact path is a symlink")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        payload = canonical_json_bytes(value)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_manifest(plan: ConfirmatoryEpisodePlan) -> dict[str, object]:
    return {
        "profile": _PROFILE,
        "plan_hash": plan.plan_hash,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "manifest_hash": episode.manifest_hash,
                "manifest": episode_manifest_projection(episode),
            }
            for episode in plan.episodes
        ],
    }


def _plan_artifact(plan: ConfirmatoryEpisodePlan) -> dict[str, object]:
    """Return the canonical planned-matrix artifact consumed by audit and analysis."""

    return {
        "profile": "SSB-PLAN1",
        "plan_hash": plan.plan_hash,
        "episodes": [
            {
                "manifest": episode_manifest_projection(episode),
                "manifest_hash": episode.manifest_hash,
            }
            for episode in plan.episodes
        ],
    }


def _catalog_artifact(catalog: FrozenArtifactCatalog) -> dict[str, object]:
    """Return the closed audit referent catalog without inventing missing custody."""
    return catalog.projection()


def _exclusion_rule_artifact(rule: FrozenExclusionRule) -> dict[str, object]:
    return {
        "profile": "SSB-EXCLUSION-RULE2",
        "allowed_error_codes": list(rule.allowed_error_codes),
        "max_retries_per_episode": rule.max_retries_per_episode,
        "max_total_exclusions": rule.max_total_exclusions,
        "max_exclusions_per_primary_cell": rule.max_exclusions_per_primary_cell,
        "rule_hash": rule.rule_hash,
    }


def _write_audit_artifacts(
    run_directory: Path,
    plan: ConfirmatoryEpisodePlan,
    catalog: FrozenArtifactCatalog,
    exclusion_rule: FrozenExclusionRule | None,
    package_manifest: dict[str, object] | None,
) -> None:
    """Publish exact audit inputs before dispatch; existing bytes must remain identical."""

    artifacts: tuple[tuple[str, dict[str, object]], ...] = (
        (
            ("plan.json", _plan_artifact(plan)),
            ("catalog.json", _catalog_artifact(catalog)),
        )
        + (
            ()
            if exclusion_rule is None
            else (("exclusion-rule.json", _exclusion_rule_artifact(exclusion_rule)),)
        )
        + (() if package_manifest is None else (("package-manifest.json", package_manifest),))
    )
    for name, value in artifacts:
        path = run_directory / name
        if not _write_exclusive_json(path, value) and _read_json(path, label=name) != value:
            raise RunCustodyError(f"{name} does not match the supplied frozen input")


def _validate_run_manifest(path: Path, plan: ConfirmatoryEpisodePlan) -> None:
    expected = _run_manifest(plan)
    actual = _read_json(path, label="run manifest")
    if actual != expected:
        raise RunCustodyError("run manifest does not match the exact planned matrix")


def _ledger_projection(
    *,
    sequence: int,
    previous_hash: str | None,
    plan_hash: str,
    event: str,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "profile": _PROFILE,
        "sequence": sequence,
        "previous_hash": previous_hash,
        "plan_hash": plan_hash,
        "event": event,
        "payload": payload,
    }


def _read_ledger(path: Path, plan_hash: str) -> tuple[int, str | None]:
    if not path.exists():
        return 0, None
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise RunCustodyError("run ledger is unreadable") from error
    previous_hash: str | None = None
    for sequence, line in enumerate(lines, start=1):
        try:
            entry = _json_object(json.loads(line), label="run ledger entry")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RunCustodyError("run ledger has an invalid entry") from error
        entry_hash = entry.pop("entry_hash", None)
        event = entry.get("event")
        payload = entry.get("payload")
        if (
            type(entry_hash) is not str
            or type(entry.get("sequence")) is not int
            or type(entry.get("previous_hash")) not in {str, type(None)}
            or type(entry.get("plan_hash")) is not str
            or type(event) is not str
            or event not in _LEDGER_EVENTS
            or type(payload) is not dict
            or entry
            != _ledger_projection(
                sequence=sequence,
                previous_hash=previous_hash,
                plan_hash=plan_hash,
                event=event,
                payload=payload,
            )
            or sha256_ref(entry) != entry_hash
        ):
            raise RunCustodyError("run ledger custody chain is invalid")
        previous_hash = entry_hash
    return len(lines), previous_hash


def _append_ledger(
    path: Path,
    *,
    sequence: int,
    previous_hash: str | None,
    plan_hash: str,
    event: str,
    payload: dict[str, object],
) -> str:
    projection = _ledger_projection(
        sequence=sequence,
        previous_hash=previous_hash,
        plan_hash=plan_hash,
        event=event,
        payload=payload,
    )
    entry = {**projection, "entry_hash": sha256_ref(projection)}
    encoded = canonical_json_bytes(entry) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("short ledger write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return cast(str, entry["entry_hash"])


def _materialized_plan_binds(episode: PlannedEpisode, plan: EpisodePlan) -> None:
    manifest = plan.manifest
    skill = episode.skill
    materialized_skill = plan.skill
    if (
        manifest.episode_id != episode.episode_id
        or manifest.stage is not episode.stage
        or manifest.condition is not episode.condition
        or manifest.domain != episode.case.domain
        or manifest.case_id != episode.case.case_id
        or manifest.world_hash != episode.case.world_hash
        or manifest.authority_graph_hash != episode.case.authority_graph_hash
        or manifest.policy_hash != episode.condition_binding.policy_hash
        or manifest.skill_bundle_id != (None if skill is None else skill.bundle_id)
        or manifest.skill_hash != (None if skill is None else skill.rendered_skill_hash)
        or (
            manifest.contamination_ratio is not None
            and Decimal(str(manifest.contamination_ratio))
            != (None if skill is None else skill.contamination_ratio)
        )
        or (manifest.contamination_ratio is None) != (skill is None)
        or plan.order_assignment is not episode.order_assignment
        or (materialized_skill is None) != (skill is None)
        or (
            skill is not None
            and (
                materialized_skill is None
                or compiled_skill_artifact_hash(materialized_skill)
                != skill.compiled_skill_artifact_hash
                or compiler_manifest_hash(materialized_skill.compiler_manifest)
                != skill.compiler_manifest_hash
            )
        )
    ):
        raise RunnerError("materialized EpisodePlan does not bind the planned episode")


def _result_envelope(
    *,
    plan_hash: str,
    episode: PlannedEpisode,
    execution_plan: EpisodePlan,
    result: EpisodeResult,
) -> dict[str, object]:
    if result.episode_id != episode.episode_id:
        raise RunnerError("episode result does not bind the planned episode id")
    return {
        "profile": _PROFILE,
        "plan_hash": plan_hash,
        "episode_id": episode.episode_id,
        "planned_manifest_hash": episode.manifest_hash,
        "execution_manifest_hash": result.manifest_hash,
        "expected_execution_manifest_hash": hash_episode_manifest(execution_plan.manifest),
        "result_content_hash": result.content_hash,
        "result": result.model_dump(mode="json"),
    }


def _audit_binding(episode: PlannedEpisode, execution_plan: EpisodePlan) -> dict[str, object]:
    planned_skill = episode.skill
    skill = execution_plan.skill
    policy = execution_plan.policy
    package_plan = {
        "profile": "SSB-EXECUTION-PLAN1",
        "episode_id": episode.episode_id,
        "planned_manifest_hash": episode.manifest_hash,
        "execution_plan": execution_plan.model_dump(mode="json"),
    }
    return {
        "profile": _AUDIT_BINDING_PROFILE,
        "episode_id": episode.episode_id,
        "execution_manifest_hash": hash_episode_manifest(execution_plan.manifest),
        "condition": execution_plan.manifest.condition.value,
        "context_contract_hash": episode.condition_binding.context_contract_hash,
        "policy_hash": None if policy is None else policy.rendered_hash,
        "policy_text": None if policy is None else policy.rendered_text,
        "source_manifest_hash": None
        if planned_skill is None
        else planned_skill.source_manifest_hash,
        "compiler_manifest_hash": None
        if skill is None
        else compiler_manifest_hash(skill.compiler_manifest),
        "compiled_skill_artifact_hash": None
        if skill is None
        else compiled_skill_artifact_hash(skill),
        "rendered_skill_hash": None if skill is None else skill.rendered_skill_hash,
        "provider": execution_plan.manifest.executor_model.provider,
        "model": execution_plan.manifest.executor_model.model,
        "model_version": execution_plan.manifest.executor_model.model_version_date,
        "package_execution_plan_hash": sha256_ref(package_plan),
    }


def _persist_audit_binding(
    run_directory: Path, episode: PlannedEpisode, execution_plan: EpisodePlan
) -> None:
    path = run_directory / "execution-bindings" / f"{episode.episode_id}.json"
    expected = _audit_binding(episode, execution_plan)
    if (
        not _write_exclusive_json(path, expected)
        and _read_json(path, label="execution binding") != expected
    ):
        raise RunCustodyError("execution binding does not match materialized episode")


def _result_content_hash(value: object) -> str:
    result = _json_object(value, label="episode result payload")
    content_hash = result.get("content_hash")
    artifact_ref = result.get("artifact_ref")
    projection = episode_result_projection(result)
    digest = content_hash.removeprefix("sha256:") if type(content_hash) is str else ""
    if (
        type(content_hash) is not str
        or sha256_ref(projection) != content_hash
        or artifact_ref != f"artifacts/episode_result/{digest[:2]}/{digest}.json"
    ):
        raise RunCustodyError("episode result payload does not bind its content hash")
    return content_hash


def _load_completed(
    run_directory: Path, plan: ConfirmatoryEpisodePlan
) -> dict[str, _PersistedEpisode]:
    expected = {episode.episode_id: episode for episode in plan.episodes}
    results_directory = run_directory / "results"
    if not results_directory.exists():
        return {}
    completed: dict[str, _PersistedEpisode] = {}
    for path in results_directory.glob("*.json"):
        envelope = _read_json(path, label="episode result")
        episode_id = envelope.get("episode_id")
        episode = expected.get(episode_id) if type(episode_id) is str else None
        if (
            episode is None
            or path.name != f"{episode.episode_id}.json"
            or envelope.get("profile") != _PROFILE
            or envelope.get("plan_hash") != plan.plan_hash
            or envelope.get("planned_manifest_hash") != episode.manifest_hash
        ):
            raise RunCustodyError("episode result is not part of the exact planned matrix")
        result = _json_object(envelope.get("result"), label="episode result payload")
        result_content_hash = _result_content_hash(result)
        try:
            parsed_result = parse_episode_result(result)
        except (TypeError, ValueError) as error:
            raise RunCustodyError("episode result payload is semantically invalid") from error
        if (
            parsed_result.episode_id != episode.episode_id
            or parsed_result.content_hash != result_content_hash
            or envelope.get("result_content_hash") != parsed_result.content_hash
            or envelope.get("execution_manifest_hash") != parsed_result.manifest_hash
            or envelope.get("expected_execution_manifest_hash") != parsed_result.manifest_hash
        ):
            raise RunCustodyError("episode result does not bind its manifest and content")
        execution_manifest_hash = result.get("manifest_hash")
        if type(execution_manifest_hash) is not str:
            raise RunCustodyError("episode result execution manifest hash is invalid")
        completed[episode.episode_id] = _PersistedEpisode(
            content_hash=result_content_hash,
            execution_manifest_hash=execution_manifest_hash,
        )
    return completed


def _retryable(error: ModelAdapterError) -> bool:
    return (
        error.code in {"MODEL_PROVIDER_TRANSIENT", "MODEL_PROVIDER_TERMINAL"}
        and error.before_meaningful_behavior
    )


def _retryable_result(result: EpisodeResult) -> bool:
    return (
        result.disposition is EpisodeDisposition.MODEL_FAILURE
        and result.error_code in {"MODEL_PROVIDER_TRANSIENT", "MODEL_PROVIDER_TERMINAL"}
        and result.claim is AgentClaim.NONE
        and all(
            turn.output is None
            and turn.action is None
            and turn.result is None
            and turn.event is None
            for turn in result.trace
        )
    )


def _development_plan_hash(episodes: tuple[DevelopmentPlannedEpisode, ...]) -> str:
    if (
        type(episodes) is not tuple
        or not episodes
        or any(type(episode) is not DevelopmentPlannedEpisode for episode in episodes)
    ):
        raise ValueError("episodes must be a nonempty exact development tuple")
    return sha256_ref(
        {
            "profile": _DEVELOPMENT_PROFILE,
            "episodes": [development_episode_projection(episode) for episode in episodes],
        }
    )


def _development_run_manifest(
    episodes: tuple[DevelopmentPlannedEpisode, ...], plan_hash: str
) -> dict[str, object]:
    return {
        "profile": _DEVELOPMENT_PROFILE,
        "classification": "PRACTICE_NOT_EVIDENCE",
        "plan_hash": plan_hash,
        "episodes": [development_episode_projection(episode) for episode in episodes],
    }


def _development_result_envelope(
    *,
    plan_hash: str,
    episode: DevelopmentPlannedEpisode,
    execution_plan: EpisodePlan,
    result: EpisodeResult,
) -> dict[str, object]:
    if result.episode_id != development_episode_id(episode):
        raise RunnerError("development result does not bind the planned episode")
    return {
        "profile": _DEVELOPMENT_PROFILE,
        "classification": "PRACTICE_NOT_EVIDENCE",
        "plan_hash": plan_hash,
        "episode": development_episode_projection(episode),
        "episode_id": result.episode_id,
        "execution_manifest_hash": result.manifest_hash,
        "expected_execution_manifest_hash": hash_episode_manifest(execution_plan.manifest),
        "result_content_hash": result.content_hash,
        "result": result.model_dump(mode="json"),
    }


def _development_telemetry(
    *,
    plan_hash: str,
    episode: DevelopmentPlannedEpisode,
    execution_plan: EpisodePlan,
    result: EpisodeResult,
) -> dict[str, object]:
    visible = visible_execution_input(execution_plan)
    response = json.loads(scripted_development_response(visible))
    if type(response) is not dict:
        raise RunnerError("scripted development response is invalid")
    return {
        "profile": _DEVELOPMENT_TELEMETRY_PROFILE,
        "classification": "PRACTICE_NOT_EVIDENCE",
        "plan_hash": plan_hash,
        "episode_id": development_episode_id(episode),
        "execution_manifest_hash": hash_episode_manifest(execution_plan.manifest),
        "visible_input": visible.projection(),
        "visible_input_hash": visible.content_hash,
        "scripted_response_hash": sha256_ref(response),
        "result_content_hash": result.content_hash,
        "overhead": result.overhead.model_dump(mode="json"),
    }


def _development_execution_map(
    episodes: tuple[DevelopmentPlannedEpisode, ...], execution_plans: tuple[EpisodePlan, ...]
) -> dict[str, EpisodePlan]:
    if type(execution_plans) is not tuple or len(execution_plans) != len(episodes):
        raise ValueError("execution_plans must match the exact development episode tuple")
    plans: dict[str, EpisodePlan] = {}
    for episode, execution_plan in zip(episodes, execution_plans, strict=True):
        episode_id = development_episode_id(episode)
        if (
            type(execution_plan) is not EpisodePlan
            or execution_plan.manifest.episode_id != episode_id
            or execution_plan.manifest.stage.value != "development"
            or execution_plan.manifest.condition is not episode.condition
            or execution_plan.manifest.domain != episode.domain
        ):
            raise RunnerError("materialized development plan does not bind its planned episode")
        plans[episode_id] = execution_plan
    if len(plans) != len(episodes):
        raise RunnerError("development matrix contains duplicate episode identities")
    return plans


def _load_development_completed(
    run_directory: Path,
    episodes: tuple[DevelopmentPlannedEpisode, ...],
    execution_plans: dict[str, EpisodePlan],
    plan_hash: str,
) -> dict[str, _PersistedEpisode]:
    results_directory = run_directory / "results"
    if not results_directory.exists():
        return {}
    expected = {development_episode_id(episode): episode for episode in episodes}
    completed: dict[str, _PersistedEpisode] = {}
    for path in results_directory.glob("*.json"):
        envelope = _read_json(path, label="development episode result")
        episode_id = envelope.get("episode_id")
        if type(episode_id) is not str:
            raise RunCustodyError("development episode result has an invalid identity")
        episode = expected.get(episode_id)
        if (
            episode is None
            or path.name != f"{episode_id}.json"
            or envelope.get("profile") != _DEVELOPMENT_PROFILE
            or envelope.get("classification") != "PRACTICE_NOT_EVIDENCE"
            or envelope.get("plan_hash") != plan_hash
            or envelope.get("episode") != development_episode_projection(episode)
        ):
            raise RunCustodyError("development episode result is outside the exact planned matrix")
        result = _json_object(envelope.get("result"), label="development result payload")
        result_content_hash = _result_content_hash(result)
        try:
            parsed_result = parse_episode_result(result)
        except (TypeError, ValueError) as error:
            raise RunCustodyError("development result payload is semantically invalid") from error
        execution_plan = execution_plans[parsed_result.episode_id]
        expected_manifest_hash = hash_episode_manifest(execution_plan.manifest)
        if (
            parsed_result.episode_id != episode_id
            or parsed_result.content_hash != result_content_hash
            or envelope.get("result_content_hash") != result_content_hash
            or envelope.get("execution_manifest_hash") != parsed_result.manifest_hash
            or envelope.get("expected_execution_manifest_hash") != expected_manifest_hash
            or parsed_result.manifest_hash != expected_manifest_hash
        ):
            raise RunCustodyError("development result does not bind its materialized plan")
        telemetry_path = run_directory / "telemetry" / f"{episode_id}.json"
        telemetry = _read_json(telemetry_path, label="development telemetry")
        expected_telemetry = _development_telemetry(
            plan_hash=plan_hash,
            episode=episode,
            execution_plan=execution_plan,
            result=parsed_result,
        )
        if telemetry != expected_telemetry:
            raise RunCustodyError("development telemetry does not bind its visible execution")
        completed[episode_id] = _PersistedEpisode(
            content_hash=result_content_hash,
            execution_manifest_hash=parsed_result.manifest_hash,
        )
    return completed


async def run_development_plan(
    episodes: tuple[DevelopmentPlannedEpisode, ...],
    *,
    execution_plans: tuple[EpisodePlan, ...],
    run_directory: Path,
) -> DevelopmentRunSummary:
    """Run or resume selected ADR-001 cells as explicitly non-confirmatory practice."""

    if not isinstance(run_directory, Path):
        raise ValueError("run_directory must be a Path")
    plan_hash = _development_plan_hash(episodes)
    plans = _development_execution_map(episodes, execution_plans)
    _safe_directory(run_directory)
    manifest_path = run_directory / "run-manifest.json"
    manifest = _development_run_manifest(episodes, plan_hash)
    if (
        not _write_exclusive_json(manifest_path, manifest)
        and _read_json(manifest_path, label="development run manifest") != manifest
    ):
        raise RunCustodyError("development run manifest does not match the exact planned matrix")
    completed = _load_development_completed(run_directory, episodes, plans, plan_hash)
    skipped = tuple(
        development_episode_id(episode)
        for episode in episodes
        if development_episode_id(episode) in completed
    )
    completed_now: list[str] = []
    for episode in episodes:
        episode_id = development_episode_id(episode)
        if episode_id in completed:
            continue
        execution_plan = plans[episode_id]
        visible = visible_execution_input(execution_plan)
        result = await run_episode(execution_plan, scripted_development_client(visible))
        if type(result) is not EpisodeResult:
            raise RunnerError("development executor must return an exact EpisodeResult")
        result_path = run_directory / "results" / f"{episode_id}.json"
        envelope = _development_result_envelope(
            plan_hash=plan_hash,
            episode=episode,
            execution_plan=execution_plan,
            result=result,
        )
        if not _write_exclusive_json(result_path, envelope):
            raise RunCustodyError("development result already exists during execution")
        telemetry_path = run_directory / "telemetry" / f"{episode_id}.json"
        telemetry = _development_telemetry(
            plan_hash=plan_hash,
            episode=episode,
            execution_plan=execution_plan,
            result=result,
        )
        if not _write_exclusive_json(telemetry_path, telemetry):
            raise RunCustodyError("development telemetry already exists during execution")
        completed_now.append(episode_id)
    return DevelopmentRunSummary(
        plan_hash=plan_hash,
        completed_episode_ids=tuple(completed_now),
        skipped_episode_ids=skipped,
    )


def _development_audit_root(path: Path) -> Path:
    if not isinstance(path, Path):
        raise ValueError("development artifacts root must be a Path")
    root = path if path.is_absolute() else Path.cwd() / path
    try:
        root.relative_to(Path.cwd())
        current = Path(root.anchor)
        for part in root.parts[1:]:
            current = current / part
            if current.is_symlink():
                raise RunCustodyError("development artifacts root contains a symlink")
        if not root.is_dir():
            raise RunCustodyError("development artifacts root is unavailable")
        root.resolve(strict=True).relative_to(Path.cwd().resolve(strict=True))
        return root
    except RunCustodyError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise RunCustodyError("development artifacts root is unsafe") from error


def _development_manifest_episodes(
    manifest: dict[str, object],
) -> tuple[DevelopmentPlannedEpisode, ...]:
    if (
        set(manifest) != {"profile", "classification", "plan_hash", "episodes"}
        or manifest.get("profile") != _DEVELOPMENT_PROFILE
        or manifest.get("classification") != "PRACTICE_NOT_EVIDENCE"
        or type(manifest.get("episodes")) is not list
    ):
        raise RunCustodyError("development run manifest is invalid")
    from shadowskillbench.corpus.development import generate_development_corpus
    from shadowskillbench.experiments.development_plan import build_development_plan

    recorded = cast(list[object], manifest["episodes"])
    full = build_development_plan(generate_development_corpus(4242)).episodes
    matches = tuple(
        candidates[: len(recorded)]
        for candidates in (
            tuple(item for item in full if item.stage == "stage_a"),
            tuple(item for item in full if item.stage == "stage_b"),
        )
        if recorded
        == [development_episode_projection(item) for item in candidates[: len(recorded)]]
    )
    if len(recorded) == 0 or len(matches) != 1:
        raise RunCustodyError("development run manifest is not an exact bounded stage prefix")
    episodes = matches[0]
    if manifest.get("plan_hash") != _development_plan_hash(episodes):
        raise RunCustodyError("development run manifest does not bind its planned matrix")
    return episodes


def audit_development_runs(artifacts_root: Path) -> tuple[DevelopmentRunSummary, ...]:
    """Audit bounded scripted development runs without promoting them to evidence."""

    root = _development_audit_root(artifacts_root)
    manifests = tuple(sorted(root.rglob("run-manifest.json")))
    if not manifests:
        raise RunCustodyError("development run manifest is unavailable")
    from shadowskillbench.corpus.development import generate_development_corpus
    from shadowskillbench.experiments.development_execution import materialize_development_episodes

    summaries: list[DevelopmentRunSummary] = []
    for manifest_path in manifests:
        run_directory = manifest_path.parent
        if run_directory.is_symlink() or any(
            parent.is_symlink() for parent in run_directory.parents
        ):
            raise RunCustodyError("development run path contains a symlink")
        manifest = _read_json(manifest_path, label="development run manifest")
        episodes = _development_manifest_episodes(manifest)
        plans = _development_execution_map(
            episodes,
            materialize_development_episodes(generate_development_corpus(4242), episodes),
        )
        completed = _load_development_completed(
            run_directory,
            episodes,
            plans,
            _development_plan_hash(episodes),
        )
        expected_ids = {development_episode_id(episode) for episode in episodes}
        result_paths = (
            ()
            if not (run_directory / "results").is_dir()
            else tuple((run_directory / "results").glob("*.json"))
        )
        telemetry_paths = (
            ()
            if not (run_directory / "telemetry").is_dir()
            else tuple((run_directory / "telemetry").glob("*.json"))
        )
        if (
            set(completed) != expected_ids
            or {path.stem for path in result_paths} != expected_ids
            or {path.stem for path in telemetry_paths} != expected_ids
        ):
            raise RunCustodyError("development run has missing or unplanned artifacts")
        summaries.append(
            DevelopmentRunSummary(
                plan_hash=_development_plan_hash(episodes),
                completed_episode_ids=tuple(sorted(completed)),
                skipped_episode_ids=(),
            )
        )
    return tuple(summaries)


async def run_confirmatory_plan(
    plan: ConfirmatoryEpisodePlan,
    *,
    run_directory: Path,
    materialize: EpisodeMaterializer,
    client_factory: ModelClientFactory,
    catalog: FrozenArtifactCatalog | None = None,
    exclusion_rule: FrozenExclusionRule | None = None,
    runtime_binding: dict[str, object] | None = None,
    package_manifest: dict[str, object] | None = None,
    max_concurrency: int = 1,
    max_technical_retries: int = 0,
    dispatch_episode_ids: frozenset[str] | None = None,
    execute: EpisodeExecutor = run_episode,
) -> RunSummary:
    """Run or resume an exact planned matrix using caller-injected execution inputs."""
    if type(plan) is not ConfirmatoryEpisodePlan:
        raise ValueError("plan must be an exact ConfirmatoryEpisodePlan")
    if not isinstance(run_directory, Path):
        raise ValueError("run_directory must be a Path")
    if not callable(materialize) or not callable(client_factory) or not callable(execute):
        raise ValueError("materialize, client_factory, and execute must be callable")
    if catalog is not None and type(catalog) is not FrozenArtifactCatalog:
        raise ValueError("catalog must be a FrozenArtifactCatalog")
    if exclusion_rule is not None and type(exclusion_rule) is not FrozenExclusionRule:
        raise ValueError("exclusion_rule must be a FrozenExclusionRule")
    if runtime_binding is not None and (
        type(runtime_binding) is not dict or any(type(key) is not str for key in runtime_binding)
    ):
        raise ValueError("runtime_binding must be a JSON object")
    if package_manifest is not None and (
        type(package_manifest) is not dict or any(type(key) is not str for key in package_manifest)
    ):
        raise ValueError("package_manifest must be a JSON object")
    if exclusion_rule is not None and catalog is None:
        raise ValueError("exclusion_rule requires a catalog")
    if catalog is not None and catalog.exclusion_rule_hash != (
        None if exclusion_rule is None else exclusion_rule.rule_hash
    ):
        raise ValueError("catalog exclusion_rule_hash does not bind exclusion_rule")
    if type(max_concurrency) is not int or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive exact integer")
    if type(max_technical_retries) is not int or max_technical_retries < 0:
        raise ValueError("max_technical_retries must be a nonnegative exact integer")
    expected_episode_ids = frozenset(episode.episode_id for episode in plan.episodes)
    if dispatch_episode_ids is not None and (
        type(dispatch_episode_ids) is not frozenset
        or not dispatch_episode_ids
        or any(type(episode_id) is not str for episode_id in dispatch_episode_ids)
        or not dispatch_episode_ids <= expected_episode_ids
    ):
        raise ValueError("dispatch_episode_ids must be a nonempty exact plan-id subset")
    retry_budget = (
        max_technical_retries if exclusion_rule is None else exclusion_rule.max_retries_per_episode
    )

    _safe_directory(run_directory)
    if catalog is not None:
        _write_audit_artifacts(run_directory, plan, catalog, exclusion_rule, package_manifest)
    if runtime_binding is not None:
        runtime_path = run_directory / "runtime-binding.json"
        if (
            not _write_exclusive_json(runtime_path, runtime_binding)
            and _read_json(runtime_path, label="runtime binding") != runtime_binding
        ):
            raise RunCustodyError("runtime binding does not match prior dispatch")
    manifest_path = run_directory / "run-manifest.json"
    if not _write_exclusive_json(manifest_path, _run_manifest(plan)):
        _validate_run_manifest(manifest_path, plan)
    ledger_path = run_directory / "run-ledger.jsonl"
    sequence, previous_hash = _read_ledger(ledger_path, plan.plan_hash)
    completed = _load_completed(run_directory, plan)
    for episode in plan.episodes:
        persisted = completed.get(episode.episode_id)
        if persisted is None:
            continue
        execution_plan = materialize(episode)
        if type(execution_plan) is not EpisodePlan:
            raise RunnerError("materializer must return an exact EpisodePlan")
        _materialized_plan_binds(episode, execution_plan)
        if hash_episode_manifest(execution_plan.manifest) != persisted.execution_manifest_hash:
            raise RunCustodyError("completed episode execution manifest no longer matches")
        _persist_audit_binding(run_directory, episode, execution_plan)
    ledger_lock = asyncio.Lock()

    async def record(event: str, payload: dict[str, object]) -> None:
        nonlocal sequence, previous_hash
        async with ledger_lock:
            sequence += 1
            previous_hash = _append_ledger(
                ledger_path,
                sequence=sequence,
                previous_hash=previous_hash,
                plan_hash=plan.plan_hash,
                event=event,
                payload=payload,
            )

    await record("RUN_RESUMED", {"completed_before_resume": len(completed)})
    dispatch = tuple(
        episode
        for episode in plan.episodes
        if dispatch_episode_ids is None or episode.episode_id in dispatch_episode_ids
    )
    skipped = tuple(episode.episode_id for episode in dispatch if episode.episode_id in completed)
    pending = tuple(episode for episode in dispatch if episode.episode_id not in completed)
    semaphore = asyncio.Semaphore(max_concurrency)
    retried: list[str] = []

    async def execute_one(episode: PlannedEpisode) -> str:
        async with semaphore:
            retries = 0
            while True:
                execution_plan = materialize(episode)
                if type(execution_plan) is not EpisodePlan:
                    raise RunnerError("materializer must return an exact EpisodePlan")
                _materialized_plan_binds(episode, execution_plan)
                _persist_audit_binding(run_directory, episode, execution_plan)
                client = client_factory(episode)
                try:
                    result = await execute(execution_plan, client)
                except ModelAdapterError as error:
                    if _retryable(error) and retries < retry_budget:
                        retries += 1
                        retried.append(episode.episode_id)
                        await record(
                            "EPISODE_RETRY",
                            {
                                "episode_id": episode.episode_id,
                                "manifest_hash": episode.manifest_hash,
                                "attempt": retries,
                                "error_code": error.code,
                            },
                        )
                        continue
                    raise
                if type(result) is not EpisodeResult:
                    raise RunnerError("executor must return an exact EpisodeResult")
                if _retryable_result(result) and retries < retry_budget:
                    retries += 1
                    retried.append(episode.episode_id)
                    await record(
                        "EPISODE_RETRY",
                        {
                            "episode_id": episode.episode_id,
                            "manifest_hash": episode.manifest_hash,
                            "attempt": retries,
                            "error_code": result.error_code,
                            "result_content_hash": result.content_hash,
                        },
                    )
                    continue
                envelope = _result_envelope(
                    plan_hash=plan.plan_hash,
                    episode=episode,
                    execution_plan=execution_plan,
                    result=result,
                )
                result_path = run_directory / "results" / f"{episode.episode_id}.json"
                if not _write_exclusive_json(result_path, envelope):
                    existing = _load_completed(run_directory, plan).get(episode.episode_id)
                    if existing is None or existing.content_hash != result.content_hash:
                        raise RunCustodyError(
                            "existing episode result differs from attempted result"
                        )
                await record(
                    "EPISODE_COMPLETED",
                    {
                        "episode_id": episode.episode_id,
                        "manifest_hash": episode.manifest_hash,
                        "result_content_hash": result.content_hash,
                    },
                )
                return episode.episode_id

    completed_now = await asyncio.gather(*(execute_one(episode) for episode in pending))
    return RunSummary(
        plan_hash=plan.plan_hash,
        completed_episode_ids=tuple(completed_now),
        skipped_episode_ids=skipped,
        retried_episode_ids=tuple(retried),
    )


__all__ = [
    "DevelopmentRunSummary",
    "EpisodeExecutor",
    "EpisodeMaterializer",
    "ModelClientFactory",
    "RunCustodyError",
    "RunnerError",
    "RunSummary",
    "audit_development_runs",
    "run_development_plan",
    "run_confirmatory_plan",
]
