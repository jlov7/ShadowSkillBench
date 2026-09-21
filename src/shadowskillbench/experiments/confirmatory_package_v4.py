"""Forward-only V4 execution package custody.

V4 deliberately does not parse, upgrade, or reuse a legacy confirmatory
package.  Its package is a compact binding over the frozen V4 corpus and the
complete V4 planned matrix; episode materialization remains an explicit local
operation at execution time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.confirmatory_v4 import (
    CORPUS_SEED_V4,
    DEVELOPMENT_SEEDS_V4,
    ConfirmatoryAuthorizationV4,
    ConfirmatoryCorpusV4,
    authorization_projection_v4,
    validate_confirmatory_authorization_v4,
    validate_confirmatory_corpus_v4,
)
from shadowskillbench.experiments.confirmatory_preparation_v4 import (
    V4_PACKAGE_NAMESPACE,
    V4_RUNTIME_NAMESPACE,
)
from shadowskillbench.experiments.planner import ConfirmatoryEpisodePlan
from shadowskillbench.experiments.planner_v4 import (
    STAGE_A_EPISODE_COUNT,
    STAGE_B_EPISODE_COUNT,
)
from shadowskillbench.models.protocol import TokenPricing
from shadowskillbench.models.runtime import ModelRunDescriptor, RunDescriptorError

V4_PACKAGE_PROFILE = "SSB-CONFIRMATORY-EXECUTION-PACKAGE4"
V4_PACKAGE_SCHEMA_VERSION = "4.0"
V4_RESULT_NAMESPACE = "artifacts/experiments/confirmatory-v4/results"
V4_RUN_DESCRIPTOR_PROFILE = "SSB-CONFIRMATORY-RUN-DESCRIPTOR4"


class ConfirmatoryPackageV4Hold(ValueError):
    """Raised when a V4 package is absent, malformed, or from another route."""


def _hold(message: str) -> ConfirmatoryPackageV4Hold:
    return ConfirmatoryPackageV4Hold(f"HOLD_CONFIRMATORY_V4_PACKAGE: {message}")


@dataclass(frozen=True, slots=True)
class ConfirmatoryExecutionPackageV4Inputs:
    authorization: ConfirmatoryAuthorizationV4
    corpus: ConfirmatoryCorpusV4
    plan: ConfirmatoryEpisodePlan
    run_descriptor: ModelRunDescriptor


@dataclass(frozen=True, slots=True)
class ConfirmatoryExecutionPackageV4:
    authorization: ConfirmatoryAuthorizationV4
    corpus_hash: str
    plan_hash: str
    run_descriptor: ModelRunDescriptor
    episode_manifest_hashes: dict[str, str]
    package_hash: str
    package_manifest: dict[str, object]
    package_namespace: str = V4_PACKAGE_NAMESPACE
    runtime_namespace: str = V4_RUNTIME_NAMESPACE
    result_namespace: str = V4_RESULT_NAMESPACE

    def __post_init__(self) -> None:
        try:
            validate_confirmatory_authorization_v4(self.authorization)
        except ValueError as error:
            raise _hold("package authorization was not issuer-bound") from error
        if (
            type(self.corpus_hash) is not str
            or type(self.plan_hash) is not str
            or type(self.package_hash) is not str
            or type(self.episode_manifest_hashes) is not dict
            or any(
                type(episode_id) is not str or type(manifest_hash) is not str
                for episode_id, manifest_hash in self.episode_manifest_hashes.items()
            )
        ):
            raise _hold("package fields are invalid")

    def require_episode(self, *, episode_id: str, manifest_hash: str) -> None:
        if (
            type(episode_id) is not str
            or type(manifest_hash) is not str
            or self.episode_manifest_hashes.get(episode_id) != manifest_hash
        ):
            raise _hold("episode is not in the exact V4 planned matrix")


def _validate_inputs(
    inputs: ConfirmatoryExecutionPackageV4Inputs,
) -> tuple[dict[str, str], dict[str, str]]:
    if type(inputs) is not ConfirmatoryExecutionPackageV4Inputs:
        raise _hold("inputs must be an exact V4 package input")
    try:
        validate_confirmatory_authorization_v4(inputs.authorization)
    except ValueError as error:
        raise _hold("V4 freeze authorization is required") from error
    if type(inputs.corpus) is not ConfirmatoryCorpusV4:
        raise _hold("V3 corpus is not admissible to V4 packaging")
    if type(inputs.plan) is not ConfirmatoryEpisodePlan:
        raise _hold("V3 or unplanned matrix is not admissible to V4 packaging")
    _validate_v4_run_descriptor(inputs.run_descriptor)
    validate_confirmatory_corpus_v4(inputs.corpus)
    if inputs.corpus.authorization != inputs.authorization:
        raise _hold("corpus does not bind the supplied V4 authorization")
    if inputs.corpus.corpus_seed != CORPUS_SEED_V4:
        raise _hold("corpus seed is not the frozen V4 seed")
    if len(inputs.plan.episodes) != STAGE_A_EPISODE_COUNT + STAGE_B_EPISODE_COUNT:
        raise _hold("plan does not have the frozen V4 episode counts")
    stage_a = sum(episode.stage.value == "confirmatory_a" for episode in inputs.plan.episodes)
    stage_b = sum(episode.stage.value == "confirmatory_b" for episode in inputs.plan.episodes)
    if (stage_a, stage_b) != (STAGE_A_EPISODE_COUNT, STAGE_B_EPISODE_COUNT):
        raise _hold("plan stage counts are not the frozen V4 counts")
    stage_a_values = {
        episode.episode_id: episode.manifest_hash
        for episode in inputs.plan.episodes
        if episode.stage.value == "confirmatory_a"
    }
    stage_b_values = {
        episode.episode_id: episode.manifest_hash
        for episode in inputs.plan.episodes
        if episode.stage.value == "confirmatory_b"
    }
    if (
        len(stage_a_values) != STAGE_A_EPISODE_COUNT
        or len(stage_b_values) != STAGE_B_EPISODE_COUNT
        or set(stage_a_values) & set(stage_b_values)
    ):
        raise _hold("V4 plan has duplicate episode identities")
    return stage_a_values, stage_b_values


def _validate_v4_run_descriptor(descriptor: object) -> ModelRunDescriptor:
    if type(descriptor) is not ModelRunDescriptor:
        raise _hold("exact pinned V4 run descriptor is required")
    runtime = descriptor.executor_runtime_profile
    if (
        type(runtime) is not dict
        or type(runtime.get("profile")) is not str
        or not cast(str, runtime["profile"]).endswith("4")
    ):
        raise _hold("run descriptor does not bind a V4 executor runtime profile")
    return descriptor


def _descriptor_payload(descriptor: ModelRunDescriptor) -> dict[str, object]:
    _validate_v4_run_descriptor(descriptor)
    value = {
        "profile": V4_RUN_DESCRIPTOR_PROFILE,
        "provider": descriptor.provider,
        "model": descriptor.model,
        "model_version": descriptor.model_version,
        "endpoint": descriptor.endpoint,
        "api_key_environment": descriptor.api_key_environment,
        "max_attempts": descriptor.max_attempts,
        "supports_system_role": descriptor.supports_system_role,
        "supports_developer_role": descriptor.supports_developer_role,
        "supports_seed": descriptor.supports_seed,
        "supports_structured_output": descriptor.supports_structured_output,
        "pricing": (
            None if descriptor.pricing is None else descriptor.pricing.model_dump(mode="json")
        ),
        "executor_runtime_profile": descriptor.executor_runtime_profile,
    }
    return {**value, "content_hash": sha256_ref(value)}


def _parse_descriptor(value: object) -> ModelRunDescriptor:
    if type(value) is not dict:
        raise _hold("V4 run descriptor is invalid")
    payload = cast(dict[str, object], value)
    keys = {
        "profile",
        "provider",
        "model",
        "model_version",
        "endpoint",
        "api_key_environment",
        "max_attempts",
        "supports_system_role",
        "supports_developer_role",
        "supports_seed",
        "supports_structured_output",
        "pricing",
        "executor_runtime_profile",
        "content_hash",
    }
    if set(payload) != keys or payload.get("profile") != V4_RUN_DESCRIPTOR_PROFILE:
        raise _hold("V4 run descriptor profile is invalid")
    body = {key: payload[key] for key in keys - {"content_hash"}}
    if payload.get("content_hash") != sha256_ref(body):
        raise _hold("V4 run descriptor hash is invalid")
    pricing_value = payload["pricing"]
    try:
        pricing = None if pricing_value is None else TokenPricing.model_validate(pricing_value)
        descriptor = ModelRunDescriptor(
            provider=cast(str, payload["provider"]),
            model=cast(str, payload["model"]),
            model_version=cast(str, payload["model_version"]),
            endpoint=cast(str, payload["endpoint"]),
            api_key_environment=cast(str, payload["api_key_environment"]),
            max_attempts=cast(int, payload["max_attempts"]),
            supports_system_role=cast(bool, payload["supports_system_role"]),
            supports_developer_role=cast(bool, payload["supports_developer_role"]),
            supports_seed=cast(bool, payload["supports_seed"]),
            supports_structured_output=cast(bool, payload["supports_structured_output"]),
            pricing=pricing,
            executor_runtime_profile=cast(
                dict[str, object] | None, payload["executor_runtime_profile"]
            ),
        )
        return _validate_v4_run_descriptor(descriptor)
    except (RunDescriptorError, TypeError, ValueError) as error:
        raise _hold("V4 run descriptor is invalid") from error


def _require_safe_new_directory(path: Path) -> Path:
    if not isinstance(path, Path) or path.exists() or path.is_symlink():
        raise _hold("package directory must be a new nonsymlink path")
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise _hold("package path contains a symbolic link")
    except OSError as error:
        raise _hold("package path is unavailable") from error
    return absolute


def _manifest(
    inputs: ConfirmatoryExecutionPackageV4Inputs,
    stage_a_episodes: dict[str, str],
    stage_b_episodes: dict[str, str],
    descriptor_payload: dict[str, object],
) -> dict[str, object]:
    return {
        "profile": V4_PACKAGE_PROFILE,
        "schema_version": V4_PACKAGE_SCHEMA_VERSION,
        "package_namespace": V4_PACKAGE_NAMESPACE,
        "runtime_namespace": V4_RUNTIME_NAMESPACE,
        "result_namespace": V4_RESULT_NAMESPACE,
        "authorization": authorization_projection_v4(inputs.authorization),
        "corpus_hash": inputs.corpus.content_hash,
        "corpus_projection": inputs.corpus.projection(),
        "corpus_seed": CORPUS_SEED_V4,
        "development_seeds": sorted(DEVELOPMENT_SEEDS_V4),
        "plan_hash": inputs.plan.plan_hash,
        "plan_projection": inputs.plan.plan_projection(),
        "run_descriptor": descriptor_payload,
        "run_descriptor_hash": descriptor_payload["content_hash"],
        "stage_a_episode_count": STAGE_A_EPISODE_COUNT,
        "stage_b_episode_count": STAGE_B_EPISODE_COUNT,
        "stage_a_episode_manifest_hashes": stage_a_episodes,
        "stage_b_episode_manifest_hashes": stage_b_episodes,
    }


def write_confirmatory_execution_package_v4(
    inputs: ConfirmatoryExecutionPackageV4Inputs, package_dir: Path
) -> ConfirmatoryExecutionPackageV4:
    """Write one canonical V4 package; this never contacts a model or service."""

    stage_a_episodes, stage_b_episodes = _validate_inputs(inputs)
    destination = _require_safe_new_directory(package_dir)
    descriptor_payload = _descriptor_payload(inputs.run_descriptor)
    manifest = _manifest(inputs, stage_a_episodes, stage_b_episodes, descriptor_payload)
    sealed = {**manifest, "package_hash": sha256_ref(manifest)}
    try:
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "package-manifest.v4.json").write_bytes(canonical_json_bytes(sealed))
    except OSError as error:
        raise _hold("package write failed") from error
    return ConfirmatoryExecutionPackageV4(
        inputs.authorization,
        inputs.corpus.content_hash,
        inputs.plan.plan_hash,
        inputs.run_descriptor,
        {**stage_a_episodes, **stage_b_episodes},
        cast(str, sealed["package_hash"]),
        sealed,
    )


def _read_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise _hold("package manifest is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _hold("package manifest is unreadable") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise _hold("package manifest is not canonical")
    return cast(dict[str, object], value)


def load_confirmatory_execution_package_v4(
    package_dir: Path, *, authorization: ConfirmatoryAuthorizationV4
) -> ConfirmatoryExecutionPackageV4:
    """Load only an exact V4 package, rejecting every legacy package shape."""

    if not isinstance(package_dir, Path):
        raise _hold("V4 package authorization is required")
    try:
        validate_confirmatory_authorization_v4(authorization)
    except ValueError as error:
        raise _hold("V4 package authorization is required") from error
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise _hold("package directory is unavailable")
    try:
        if {item.name for item in package_dir.iterdir()} != {"package-manifest.v4.json"}:
            raise _hold("package directory is not an exact V4 package")
    except OSError as error:
        raise _hold("package directory is unreadable") from error
    manifest = _read_manifest(package_dir / "package-manifest.v4.json")
    required = {
        "profile",
        "schema_version",
        "package_namespace",
        "runtime_namespace",
        "result_namespace",
        "authorization",
        "corpus_hash",
        "corpus_projection",
        "corpus_seed",
        "development_seeds",
        "plan_hash",
        "plan_projection",
        "stage_a_episode_count",
        "stage_b_episode_count",
        "run_descriptor",
        "run_descriptor_hash",
        "stage_a_episode_manifest_hashes",
        "stage_b_episode_manifest_hashes",
        "package_hash",
    }
    if set(manifest) != required or manifest.get("profile") != V4_PACKAGE_PROFILE:
        raise _hold("package profile is not V4")
    hash_view = {key: manifest[key] for key in required - {"package_hash"}}
    if manifest.get("package_hash") != sha256_ref(hash_view):
        raise _hold("package hash is invalid")
    expected = {
        "schema_version": V4_PACKAGE_SCHEMA_VERSION,
        "package_namespace": V4_PACKAGE_NAMESPACE,
        "runtime_namespace": V4_RUNTIME_NAMESPACE,
        "result_namespace": V4_RESULT_NAMESPACE,
        "authorization": authorization_projection_v4(authorization),
        "corpus_seed": CORPUS_SEED_V4,
        "development_seeds": sorted(DEVELOPMENT_SEEDS_V4),
        "stage_a_episode_count": STAGE_A_EPISODE_COUNT,
        "stage_b_episode_count": STAGE_B_EPISODE_COUNT,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise _hold("package does not bind the frozen V4 route")
    corpus_projection = manifest.get("corpus_projection")
    plan_projection = manifest.get("plan_projection")
    if (
        type(corpus_projection) is not dict
        or manifest.get("corpus_hash") != sha256_ref(corpus_projection)
        or type(plan_projection) is not dict
        or manifest.get("plan_hash") != sha256_ref(plan_projection)
        or plan_projection.get("profile") != "SSB-PLAN1"
        or type(plan_projection.get("episode_manifest_hashes")) is not list
    ):
        raise _hold("package immutable corpus or plan bytes do not bind")
    descriptor = _parse_descriptor(manifest.get("run_descriptor"))
    if manifest.get("run_descriptor_hash") != _descriptor_payload(descriptor)["content_hash"]:
        raise _hold("package run descriptor does not bind")
    stage_a_episodes = manifest.get("stage_a_episode_manifest_hashes")
    stage_b_episodes = manifest.get("stage_b_episode_manifest_hashes")
    if (
        type(stage_a_episodes) is not dict
        or type(stage_b_episodes) is not dict
        or len(stage_a_episodes) != STAGE_A_EPISODE_COUNT
        or len(stage_b_episodes) != STAGE_B_EPISODE_COUNT
        or set(stage_a_episodes) & set(stage_b_episodes)
        or any(
            type(key) is not str or type(value) is not str
            for episodes in (stage_a_episodes, stage_b_episodes)
            for key, value in episodes.items()
        )
    ):
        raise _hold("package episode inventory is invalid")
    if len(plan_projection["episode_manifest_hashes"]) != len(stage_a_episodes) + len(
        stage_b_episodes
    ) or set(plan_projection["episode_manifest_hashes"]) != set(stage_a_episodes.values()) | set(
        stage_b_episodes.values()
    ):
        raise _hold("package episode inventory does not derive the immutable plan")
    return ConfirmatoryExecutionPackageV4(
        authorization,
        cast(str, manifest["corpus_hash"]),
        cast(str, manifest["plan_hash"]),
        descriptor,
        {**cast(dict[str, str], stage_a_episodes), **cast(dict[str, str], stage_b_episodes)},
        cast(str, manifest["package_hash"]),
        manifest,
    )


__all__ = [
    "ConfirmatoryExecutionPackageV4",
    "ConfirmatoryExecutionPackageV4Inputs",
    "ConfirmatoryPackageV4Hold",
    "V4_PACKAGE_PROFILE",
    "V4_RESULT_NAMESPACE",
    "V4_RUN_DESCRIPTOR_PROFILE",
    "load_confirmatory_execution_package_v4",
    "write_confirmatory_execution_package_v4",
]
