"""Sealed, non-secret confirmatory execution packages."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from shadowskillbench.authority.models import (
    AccessAuthorityQuery,
    AuthorityDecision,
    AuthorityRecord,
    FinanceAuthorityQuery,
)
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.confirmatory import ConfirmatoryAuthorization
from shadowskillbench.domains.access.models import AccessTaskCase
from shadowskillbench.domains.finance.models import FinanceTaskCase
from shadowskillbench.engine import TaskCase, WorldState
from shadowskillbench.episodes.executor import AgentTurn
from shadowskillbench.episodes.models import BlockOrder, EpisodeManifest, EpisodePlan
from shadowskillbench.episodes.pilot_turn_wire import (
    CONFIRMATORY_TURN_PROMPT,
    CONFIRMATORY_TURN_PROMPT_PROFILE,
    LEGACY_CONFIRMATORY_TURN_PROMPT,
    LEGACY_CONFIRMATORY_TURN_PROMPT_PROFILE,
    LEGACY_V2_CONFIRMATORY_TURN_PROMPT,
    LEGACY_V2_CONFIRMATORY_TURN_PROMPT_PROFILE,
    ConfirmatoryAgentTurnWire,
    confirmatory_active_turn_schema_hash,
    confirmatory_finished_turn_schema_hash,
    confirmatory_initial_turn_schema_hash,
)
from shadowskillbench.experiments.audit import FrozenArtifactCatalog, FrozenExclusionRule
from shadowskillbench.experiments.io import (
    ExperimentArtifactHold,
    _load_catalog,
    _load_exclusion_rule,
    _load_plan,
)
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_CONTEXT_LENGTH,
    OLLAMA_GPT_OSS_EXECUTOR_MODEFILE_SHA256,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
    OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_EXECUTOR_TEMPLATE_SHA256,
    OLLAMA_GPT_OSS_LEGACY_EXECUTOR_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_PROFILE,
    OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE_HASH,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
    OLLAMA_GPT_OSS_SAMPLING_TOP_P,
)
from shadowskillbench.experiments.planner import ConfirmatoryEpisodePlan, PlannedEpisode
from shadowskillbench.experiments.runner import (
    _catalog_artifact,
    _exclusion_rule_artifact,
    _materialized_plan_binds,
    _plan_artifact,
)
from shadowskillbench.models.protocol import TokenPricing
from shadowskillbench.models.runtime import ModelRunDescriptor
from shadowskillbench.policy.documents import RenderedPolicy
from shadowskillbench.skills.compiler import parse_compiled_skill_artifact

_PROFILE = "SSB-CONFIRMATORY-EXECUTION-PACKAGE2"
_LEGACY_PROFILE = "SSB-CONFIRMATORY-EXECUTION-PACKAGE1"
_DESCRIPTOR_PROFILE = "SSB-MODEL-RUN-DESCRIPTOR2"
_LEGACY_DESCRIPTOR_PROFILE = "SSB-MODEL-RUN-DESCRIPTOR1"
_EXECUTION_PROFILE = "SSB-EXECUTION-PLAN1"
_LEGACY_OLLAMA_EXECUTOR_RUNTIME_PROFILE = "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR1"
_LEGACY_V2_OLLAMA_EXECUTOR_RUNTIME_PROFILE = "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR2"
_LEGACY_V3_OLLAMA_EXECUTOR_RUNTIME_PROFILE = "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR3"
_LEGACY_V4_OLLAMA_EXECUTOR_RUNTIME_PROFILE = "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR4"
_LEGACY_V5_OLLAMA_EXECUTOR_RUNTIME_PROFILE = "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR5"
_OLLAMA_EXECUTOR_RUNTIME_PROFILE = "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR6"
_LEGACY_CONFIRMATORY_PROMPT_PROFILE = "SSB-CONFIRMATORY-AGENT-TURN1"
_LEGACY_CONFIRMATORY_PROMPT_PROFILE_HASH = sha256_ref(
    {"profile": _LEGACY_CONFIRMATORY_PROMPT_PROFILE, "additional_messages": []}
)
_LEGACY_CONFIRMATORY_OUTPUT_SCHEMA_HASH = sha256_ref(AgentTurn.model_json_schema())
_LEGACY_V3_CONFIRMATORY_PROMPT_PROFILE = LEGACY_CONFIRMATORY_TURN_PROMPT_PROFILE
_LEGACY_V3_CONFIRMATORY_PROMPT_PROFILE_HASH = (
    "sha256:" + sha256(LEGACY_CONFIRMATORY_TURN_PROMPT.encode("utf-8")).hexdigest()
)
_LEGACY_V4_CONFIRMATORY_PROMPT_PROFILE = LEGACY_V2_CONFIRMATORY_TURN_PROMPT_PROFILE
_LEGACY_V4_CONFIRMATORY_PROMPT_PROFILE_HASH = (
    "sha256:" + sha256(LEGACY_V2_CONFIRMATORY_TURN_PROMPT.encode("utf-8")).hexdigest()
)
_CONFIRMATORY_PROMPT_PROFILE = CONFIRMATORY_TURN_PROMPT_PROFILE
_CONFIRMATORY_PROMPT_PROFILE_HASH = (
    "sha256:" + sha256(CONFIRMATORY_TURN_PROMPT.encode("utf-8")).hexdigest()
)
_LEGACY_V3_CONFIRMATORY_OUTPUT_SCHEMA_HASH = sha256_ref(
    ConfirmatoryAgentTurnWire.model_json_schema()
)
_CONFIRMATORY_INITIAL_OUTPUT_SCHEMA_HASH = confirmatory_initial_turn_schema_hash()
_CONFIRMATORY_ACTIVE_OUTPUT_SCHEMA_HASH = confirmatory_active_turn_schema_hash()
_CONFIRMATORY_FINISHED_OUTPUT_SCHEMA_HASH = confirmatory_finished_turn_schema_hash()
_CONFIRMATORY_OUTPUT_SCHEMA_HASH = sha256_ref(
    {
        "initial": _CONFIRMATORY_INITIAL_OUTPUT_SCHEMA_HASH,
        "active": _CONFIRMATORY_ACTIVE_OUTPUT_SCHEMA_HASH,
        "finished": _CONFIRMATORY_FINISHED_OUTPUT_SCHEMA_HASH,
    }
)
_CONFIRMATORY_MAX_TURNS = 12
_CONFIRMATORY_MAX_TOOL_CALLS = 12
_CONFIRMATORY_MAX_AGGREGATE_TOKENS = 8192


class ConfirmatoryPackageHold(ValueError):
    """Raised when a package is not an exact, safe, freeze-bound custody object."""


@dataclass(frozen=True, slots=True)
class ConfirmatoryExecutionPackageInputs:
    authorization: ConfirmatoryAuthorization
    plan: ConfirmatoryEpisodePlan
    catalog: FrozenArtifactCatalog
    exclusion_rule: FrozenExclusionRule
    execution_plans: tuple[EpisodePlan, ...]
    run_descriptor: ModelRunDescriptor


@dataclass(frozen=True, slots=True)
class ConfirmatoryExecutionPackage:
    plan: ConfirmatoryEpisodePlan
    catalog: FrozenArtifactCatalog
    exclusion_rule: FrozenExclusionRule
    execution_plans: Mapping[str, EpisodePlan]
    run_descriptor: ModelRunDescriptor
    freeze_manifest_hash: str
    anchor_receipt_hash: str
    package_hash: str
    package_manifest: Mapping[str, object]

    def materialize(self, episode: PlannedEpisode) -> EpisodePlan:
        try:
            return self.execution_plans[episode.episode_id]
        except KeyError as error:
            raise ConfirmatoryPackageHold("missing exact episode materialization") from error


def _hold(message: str) -> ConfirmatoryPackageHold:
    return ConfirmatoryPackageHold(f"HOLD_CONFIRMATORY_EXECUTION_PACKAGE: {message}")


def ollama_gpt_oss_confirmatory_executor_profile(
    *, role_profile_receipt_hash: str
) -> dict[str, object]:
    """Return the sealed V6 native executor profile for a package input."""

    value = {
        "profile": _OLLAMA_EXECUTOR_RUNTIME_PROFILE,
        "transport": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
        "request_profile": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
        "request_profile_hash": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
        "model": OLLAMA_GPT_OSS_EXECUTOR_MODEL,
        "model_version": OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
        "executor_profile": OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
        "modelfile_sha256": OLLAMA_GPT_OSS_EXECUTOR_MODEFILE_SHA256,
        "template_sha256": OLLAMA_GPT_OSS_EXECUTOR_TEMPLATE_SHA256,
        "think": False,
        "temperature": OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
        "top_p": OLLAMA_GPT_OSS_SAMPLING_TOP_P,
        "context_length": OLLAMA_GPT_OSS_CONTEXT_LENGTH,
        "timeout_seconds": 900,
        "output_schema_hash": _CONFIRMATORY_OUTPUT_SCHEMA_HASH,
        "initial_output_schema_hash": _CONFIRMATORY_INITIAL_OUTPUT_SCHEMA_HASH,
        "active_output_schema_hash": _CONFIRMATORY_ACTIVE_OUTPUT_SCHEMA_HASH,
        "finished_output_schema_hash": _CONFIRMATORY_FINISHED_OUTPUT_SCHEMA_HASH,
        "prompt_profile": _CONFIRMATORY_PROMPT_PROFILE,
        "prompt_profile_hash": _CONFIRMATORY_PROMPT_PROFILE_HASH,
        "max_turns": _CONFIRMATORY_MAX_TURNS,
        "max_tool_calls": _CONFIRMATORY_MAX_TOOL_CALLS,
        "max_aggregate_tokens": _CONFIRMATORY_MAX_AGGREGATE_TOKENS,
        "role_profile_receipt_hash": role_profile_receipt_hash,
    }
    _validate_ollama_executor_runtime_profile(value)
    return value


def _validate_ollama_executor_runtime_profile(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise _hold("executor runtime profile is invalid")
    base_required = {
        "profile",
        "transport",
        "request_profile",
        "request_profile_hash",
        "model",
        "model_version",
        "executor_profile",
        "modelfile_sha256",
        "template_sha256",
        "think",
        "temperature",
        "top_p",
        "context_length",
        "timeout_seconds",
        "output_schema_hash",
        "prompt_profile",
        "prompt_profile_hash",
        "role_profile_receipt_hash",
    }
    receipt_hash = value.get("role_profile_receipt_hash")
    profile = value.get("profile")
    legacy_v1 = profile == _LEGACY_OLLAMA_EXECUTOR_RUNTIME_PROFILE
    legacy_v2 = profile == _LEGACY_V2_OLLAMA_EXECUTOR_RUNTIME_PROFILE
    legacy_v3 = profile == _LEGACY_V3_OLLAMA_EXECUTOR_RUNTIME_PROFILE
    legacy_v4 = profile == _LEGACY_V4_OLLAMA_EXECUTOR_RUNTIME_PROFILE
    legacy_v5 = profile == _LEGACY_V5_OLLAMA_EXECUTOR_RUNTIME_PROFILE
    if profile not in {
        _LEGACY_OLLAMA_EXECUTOR_RUNTIME_PROFILE,
        _LEGACY_V2_OLLAMA_EXECUTOR_RUNTIME_PROFILE,
        _LEGACY_V3_OLLAMA_EXECUTOR_RUNTIME_PROFILE,
        _LEGACY_V4_OLLAMA_EXECUTOR_RUNTIME_PROFILE,
        _LEGACY_V5_OLLAMA_EXECUTOR_RUNTIME_PROFILE,
        _OLLAMA_EXECUTOR_RUNTIME_PROFILE,
    }:
        raise _hold("executor runtime profile is not pinned")
    expected_value = {
        "profile": profile,
        "transport": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
        "request_profile": (
            OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE
            if legacy_v1
            else OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE
        ),
        "request_profile_hash": (
            OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE_HASH
            if legacy_v1
            else OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH
        ),
        "model": OLLAMA_GPT_OSS_EXECUTOR_MODEL,
        "model_version": OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
        "executor_profile": (
            OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_PROFILE
            if legacy_v1
            else OLLAMA_GPT_OSS_EXECUTOR_PROFILE
        ),
        "modelfile_sha256": OLLAMA_GPT_OSS_EXECUTOR_MODEFILE_SHA256,
        "template_sha256": OLLAMA_GPT_OSS_EXECUTOR_TEMPLATE_SHA256,
        "think": False,
        "temperature": (
            OLLAMA_GPT_OSS_LEGACY_EXECUTOR_SAMPLING_TEMPERATURE
            if legacy_v1 or legacy_v2 or legacy_v3 or legacy_v4
            else OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE
        ),
        "top_p": OLLAMA_GPT_OSS_SAMPLING_TOP_P,
        "context_length": OLLAMA_GPT_OSS_CONTEXT_LENGTH,
        "timeout_seconds": 900,
        "output_schema_hash": (
            _LEGACY_CONFIRMATORY_OUTPUT_SCHEMA_HASH
            if legacy_v1 or legacy_v2
            else _LEGACY_V3_CONFIRMATORY_OUTPUT_SCHEMA_HASH
            if legacy_v3 or legacy_v4 or legacy_v5
            else _CONFIRMATORY_OUTPUT_SCHEMA_HASH
        ),
        "prompt_profile": (
            _LEGACY_CONFIRMATORY_PROMPT_PROFILE
            if legacy_v1 or legacy_v2
            else _LEGACY_V3_CONFIRMATORY_PROMPT_PROFILE
            if legacy_v3
            else _LEGACY_V4_CONFIRMATORY_PROMPT_PROFILE
            if legacy_v4 or legacy_v5
            else _CONFIRMATORY_PROMPT_PROFILE
        ),
        "prompt_profile_hash": (
            _LEGACY_CONFIRMATORY_PROMPT_PROFILE_HASH
            if legacy_v1 or legacy_v2
            else _LEGACY_V3_CONFIRMATORY_PROMPT_PROFILE_HASH
            if legacy_v3
            else _LEGACY_V4_CONFIRMATORY_PROMPT_PROFILE_HASH
            if legacy_v4 or legacy_v5
            else _CONFIRMATORY_PROMPT_PROFILE_HASH
        ),
        "role_profile_receipt_hash": receipt_hash,
    }
    required = set(base_required)
    if not legacy_v1 and not legacy_v2:
        expected_value.update(
            {
                "max_turns": _CONFIRMATORY_MAX_TURNS,
                "max_tool_calls": _CONFIRMATORY_MAX_TOOL_CALLS,
                "max_aggregate_tokens": _CONFIRMATORY_MAX_AGGREGATE_TOKENS,
            }
        )
        required.update({"max_turns", "max_tool_calls", "max_aggregate_tokens"})
    if profile == _OLLAMA_EXECUTOR_RUNTIME_PROFILE:
        expected_value.update(
            {
                "initial_output_schema_hash": _CONFIRMATORY_INITIAL_OUTPUT_SCHEMA_HASH,
                "active_output_schema_hash": _CONFIRMATORY_ACTIVE_OUTPUT_SCHEMA_HASH,
                "finished_output_schema_hash": _CONFIRMATORY_FINISHED_OUTPUT_SCHEMA_HASH,
            }
        )
        required.update(
            {
                "initial_output_schema_hash",
                "active_output_schema_hash",
                "finished_output_schema_hash",
            }
        )
    if (
        set(value) != required
        or type(receipt_hash) is not str
        or len(receipt_hash) != 71
        or not receipt_hash.startswith("sha256:")
        or value != expected_value
    ):
        raise _hold("executor runtime profile is not pinned")
    return cast(dict[str, object], value)


def confirmatory_executor_runtime_profile_projection(value: object) -> dict[str, object]:
    """Validate and return the exact sealed native executor projection."""

    return _validate_ollama_executor_runtime_profile(value)


def _validate_execution_budget_binding(
    runtime_profile: Mapping[str, object] | None, plan: EpisodePlan
) -> None:
    if runtime_profile is None or runtime_profile.get("profile") not in {
        _LEGACY_V3_OLLAMA_EXECUTOR_RUNTIME_PROFILE,
        _LEGACY_V4_OLLAMA_EXECUTOR_RUNTIME_PROFILE,
        _LEGACY_V5_OLLAMA_EXECUTOR_RUNTIME_PROFILE,
        _OLLAMA_EXECUTOR_RUNTIME_PROFILE,
    }:
        return
    budgets = plan.manifest.budgets
    if (
        budgets.max_turns != runtime_profile.get("max_turns")
        or budgets.max_tool_calls != runtime_profile.get("max_tool_calls")
        or budgets.max_tokens != runtime_profile.get("max_aggregate_tokens")
    ):
        raise _hold("execution plan does not bind executor runtime budgets")


def _write_new(path: Path, value: dict[str, object]) -> None:
    payload = canonical_json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise _hold(f"refusing to overwrite {path.name}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _descriptor_payload(descriptor: ModelRunDescriptor) -> dict[str, object]:
    value: dict[str, object] = {
        "profile": _DESCRIPTOR_PROFILE,
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
        "pricing": None
        if descriptor.pricing is None
        else descriptor.pricing.model_dump(mode="json"),
        "executor_runtime_profile": descriptor.executor_runtime_profile,
    }
    return {**value, "content_hash": sha256_ref(value)}


def _parse_descriptor(value: object) -> ModelRunDescriptor:
    if type(value) is not dict:
        raise _hold("run descriptor is invalid")
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
    legacy_keys = keys - {"executor_runtime_profile"}
    if payload.get("profile") == _LEGACY_DESCRIPTOR_PROFILE and set(payload) == legacy_keys:
        raw = {key: payload[key] for key in legacy_keys - {"content_hash"}}
        runtime_profile = None
    elif payload.get("profile") == _DESCRIPTOR_PROFILE and set(payload) == keys:
        raw = {key: payload[key] for key in keys - {"content_hash"}}
        runtime_profile = raw["executor_runtime_profile"]
    else:
        raise _hold("run descriptor shape is invalid")
    if payload["content_hash"] != sha256_ref(raw):
        raise _hold("run descriptor hash is invalid")
    try:
        pricing = None if raw["pricing"] is None else TokenPricing.model_validate(raw["pricing"])
        if runtime_profile is not None:
            runtime_profile = _validate_ollama_executor_runtime_profile(runtime_profile)
        return ModelRunDescriptor(
            provider=cast(str, raw["provider"]),
            model=cast(str, raw["model"]),
            model_version=cast(str, raw["model_version"]),
            endpoint=cast(str, raw["endpoint"]),
            api_key_environment=cast(str, raw["api_key_environment"]),
            max_attempts=cast(int, raw["max_attempts"]),
            supports_system_role=cast(bool, raw["supports_system_role"]),
            supports_developer_role=cast(bool, raw["supports_developer_role"]),
            supports_seed=cast(bool, raw["supports_seed"]),
            supports_structured_output=cast(bool, raw["supports_structured_output"]),
            pricing=pricing,
            executor_runtime_profile=runtime_profile,
        )
    except (TypeError, ValueError) as error:
        raise _hold("run descriptor is invalid") from error


def _execution_payload(episode: PlannedEpisode, plan: EpisodePlan) -> dict[str, object]:
    value = {
        "profile": _EXECUTION_PROFILE,
        "episode_id": episode.episode_id,
        "planned_manifest_hash": episode.manifest_hash,
        "execution_plan": plan.model_dump(mode="json"),
    }
    return {**value, "content_hash": sha256_ref(value)}


def _parse_execution_plan(value: object) -> EpisodePlan:
    if type(value) is not dict:
        raise _hold("execution plan is invalid")
    raw = cast(dict[str, object], value)
    try:
        manifest = EpisodeManifest.model_validate(raw["manifest"])
        domain_case_raw = cast(dict[str, object], raw["domain_case"])
        domain_case = (
            AccessTaskCase.model_validate(domain_case_raw)
            if manifest.domain == "access_provisioning"
            else FinanceTaskCase.model_validate(domain_case_raw)
        )
        query_raw = cast(dict[str, object], raw["authority_query"])
        query = (
            AccessAuthorityQuery.model_validate(query_raw)
            if manifest.domain == "access_provisioning"
            else FinanceAuthorityQuery.model_validate(query_raw)
        )
        skill_raw = raw["skill"]
        policy_raw = raw["policy"]
        return EpisodePlan(
            manifest=manifest,
            task=TaskCase.model_validate(raw["task"]),
            initial_state=WorldState.model_validate(raw["initial_state"]),
            domain_case=domain_case,
            authority_decision=AuthorityDecision.model_validate(raw["authority_decision"]),
            authority_records=tuple(
                AuthorityRecord.model_validate(item)
                for item in cast(list[object], raw["authority_records"])
            ),
            authority_query=query,
            skill=None if skill_raw is None else parse_compiled_skill_artifact(skill_raw),
            policy=None if policy_raw is None else RenderedPolicy.model_validate(policy_raw),
            order_assignment=cast(BlockOrder | None, raw["order_assignment"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _hold("execution plan is invalid") from error


def _validate_inputs(inputs: ConfirmatoryExecutionPackageInputs) -> dict[str, EpisodePlan]:
    if type(inputs) is not ConfirmatoryExecutionPackageInputs:
        raise _hold("inputs must be exact")
    if inputs.catalog.exclusion_rule_hash != inputs.exclusion_rule.rule_hash:
        raise _hold("catalog does not bind exclusion rule")
    runtime_profile: dict[str, object] | None = None
    if inputs.run_descriptor.executor_runtime_profile is not None:
        runtime_profile = _validate_ollama_executor_runtime_profile(
            inputs.run_descriptor.executor_runtime_profile
        )
        if (
            inputs.run_descriptor.model != runtime_profile["model"]
            or inputs.run_descriptor.model_version != runtime_profile["model_version"]
        ):
            raise _hold("run descriptor does not bind executor runtime profile")
    if type(inputs.execution_plans) is not tuple or len(inputs.execution_plans) != len(
        inputs.plan.episodes
    ):
        raise _hold("execution plans do not match complete plan")
    plans: dict[str, EpisodePlan] = {}
    for episode, plan in zip(inputs.plan.episodes, inputs.execution_plans, strict=True):
        if type(plan) is not EpisodePlan:
            raise _hold("execution plan must be exact")
        _validate_execution_budget_binding(runtime_profile, plan)
        try:
            _materialized_plan_binds(episode, plan)
        except ValueError as error:
            raise _hold("execution plan does not bind planned matrix") from error
        manifest_model = plan.manifest.executor_model
        if (manifest_model.provider, manifest_model.model, manifest_model.model_version_date) != (
            inputs.run_descriptor.provider,
            inputs.run_descriptor.model,
            inputs.run_descriptor.model_version,
        ):
            raise _hold("run descriptor does not bind execution model")
        plans[episode.episode_id] = plan
    if len(plans) != len(inputs.plan.episodes):
        raise _hold("planned matrix has duplicate identities")
    return plans


def _require_nonsymlink_ancestors(path: Path) -> Path:
    """Reject a custody path if any existing component is a symbolic link."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise _hold(f"unsafe custody path {path.name}")
    except ConfirmatoryPackageHold:
        raise
    except OSError as error:
        raise _hold(f"unavailable custody path {path.name}") from error
    return absolute


def _require_nonsymlink_path(path: Path, *, directory: bool) -> None:
    absolute = _require_nonsymlink_ancestors(path)
    if absolute.is_symlink() or (not absolute.is_dir() if directory else not absolute.is_file()):
        raise _hold(f"unsafe custody path {path.name}")


def write_confirmatory_execution_package(
    inputs: ConfirmatoryExecutionPackageInputs, package_dir: Path
) -> ConfirmatoryExecutionPackage:
    """Publish a new package only from caller-supplied, frozen in-memory inputs."""
    plans = _validate_inputs(inputs)
    if not isinstance(package_dir, Path) or package_dir.exists() or package_dir.is_symlink():
        raise _hold("package directory must be a new nonsymlink path")
    _require_nonsymlink_ancestors(package_dir.parent)
    try:
        package_dir.mkdir(parents=True, exist_ok=False)
        execution_dir = package_dir / "execution-plans"
        execution_dir.mkdir()
        plan_payload = _plan_artifact(inputs.plan)
        catalog_payload = _catalog_artifact(inputs.catalog)
        rule_payload = _exclusion_rule_artifact(inputs.exclusion_rule)
        descriptor_payload = _descriptor_payload(inputs.run_descriptor)
        _write_new(package_dir / "plan.json", plan_payload)
        _write_new(package_dir / "catalog.json", catalog_payload)
        _write_new(package_dir / "exclusion-rule.json", rule_payload)
        _write_new(package_dir / "run-descriptor.json", descriptor_payload)
        files: list[dict[str, str]] = []
        for episode in inputs.plan.episodes:
            payload = _execution_payload(episode, plans[episode.episode_id])
            relative = f"execution-plans/{episode.episode_id}.json"
            _write_new(package_dir / relative, payload)
            files.append({"path": relative, "content_hash": cast(str, payload["content_hash"])})
        manifest = {
            "profile": _PROFILE,
            "freeze_manifest_hash": inputs.authorization.freeze.manifest_hash,
            "anchor_receipt_hash": inputs.authorization.receipt.receipt_hash,
            "plan_hash": inputs.plan.plan_hash,
            "catalog_hash": sha256_ref(catalog_payload),
            "exclusion_rule_hash": inputs.exclusion_rule.rule_hash,
            "run_descriptor_hash": cast(str, descriptor_payload["content_hash"]),
            "executor_runtime_profile_hash": (
                None
                if inputs.run_descriptor.executor_runtime_profile is None
                else sha256_ref(inputs.run_descriptor.executor_runtime_profile)
            ),
            "execution_plans": files,
        }
        sealed_manifest = {**manifest, "package_hash": sha256_ref(manifest)}
        _write_new(package_dir / "package-manifest.json", sealed_manifest)
    except OSError as error:
        raise _hold("package write failed") from error
    return ConfirmatoryExecutionPackage(
        inputs.plan,
        inputs.catalog,
        inputs.exclusion_rule,
        plans,
        inputs.run_descriptor,
        inputs.authorization.freeze.manifest_hash,
        inputs.authorization.receipt.receipt_hash,
        sha256_ref(manifest),
        sealed_manifest,
    )


def _read_canonical(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise _hold(f"unsafe package entry {path.name}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise _hold(f"unreadable package entry {path.name}") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise _hold(f"noncanonical package entry {path.name}")
    return cast(dict[str, object], value)


def load_confirmatory_execution_package_inputs(
    source_dir: Path, *, authorization: ConfirmatoryAuthorization
) -> ConfirmatoryExecutionPackageInputs:
    """Load a complete, immutable producer source without regenerating any material.

    The source is deliberately package-manifest-free: its complete filesystem
    inventory is the plan plus one exact materialization for every planned episode.
    """

    if type(authorization) is not ConfirmatoryAuthorization:
        raise _hold("authenticated anchor is required")
    if not isinstance(source_dir, Path):
        raise _hold("source directory is unsafe")
    _require_nonsymlink_path(source_dir, directory=True)
    expected_entries = {
        "plan.json",
        "catalog.json",
        "exclusion-rule.json",
        "run-descriptor.json",
        "execution-plans",
    }
    try:
        if {item.name for item in source_dir.iterdir()} != expected_entries:
            raise _hold("source directory has unexpected entries")
    except ConfirmatoryPackageHold:
        raise
    except OSError as error:
        raise _hold("source directory is unreadable") from error
    for name in ("plan.json", "catalog.json", "exclusion-rule.json", "run-descriptor.json"):
        _require_nonsymlink_path(source_dir / name, directory=False)
    try:
        plan = _load_plan(source_dir / "plan.json")
        catalog = _load_catalog(source_dir / "catalog.json")
        rule = _load_exclusion_rule(source_dir / "exclusion-rule.json")
    except ExperimentArtifactHold as error:
        raise _hold("frozen source input is invalid") from error
    descriptor_payload = _read_canonical(source_dir / "run-descriptor.json")
    descriptor = _parse_descriptor(descriptor_payload)
    if catalog.exclusion_rule_hash != rule.rule_hash:
        raise _hold("source catalog does not bind exclusion rule")
    execution_dir = source_dir / "execution-plans"
    _require_nonsymlink_path(execution_dir, directory=True)
    expected_ids = {episode.episode_id: episode for episode in plan.episodes}
    plans: dict[str, EpisodePlan] = {}
    try:
        names = {item.name for item in execution_dir.iterdir()}
    except OSError as error:
        raise _hold("execution plan inventory is unreadable") from error
    if names != {f"{episode_id}.json" for episode_id in expected_ids}:
        raise _hold("execution plan inventory is incomplete or unsafe")
    for episode in plan.episodes:
        payload = _read_canonical(execution_dir / f"{episode.episode_id}.json")
        if set(payload) != {
            "profile",
            "episode_id",
            "planned_manifest_hash",
            "execution_plan",
            "content_hash",
        }:
            raise _hold("execution plan source shape is invalid")
        body = {key: value for key, value in payload.items() if key != "content_hash"}
        if (
            payload.get("profile") != _EXECUTION_PROFILE
            or payload.get("episode_id") != episode.episode_id
            or payload.get("planned_manifest_hash") != episode.manifest_hash
            or payload.get("content_hash") != sha256_ref(body)
        ):
            raise _hold("execution plan content hash is invalid")
        execution_plan = _parse_execution_plan(payload.get("execution_plan"))
        try:
            _materialized_plan_binds(episode, execution_plan)
        except ValueError as error:
            raise _hold("execution plan does not bind planned matrix") from error
        model = execution_plan.manifest.executor_model
        if (model.provider, model.model, model.model_version_date) != (
            descriptor.provider,
            descriptor.model,
            descriptor.model_version,
        ):
            raise _hold("execution plan does not bind run descriptor")
        _validate_execution_budget_binding(descriptor.executor_runtime_profile, execution_plan)
        plans[episode.episode_id] = execution_plan
    inputs = ConfirmatoryExecutionPackageInputs(
        authorization=authorization,
        plan=plan,
        catalog=catalog,
        exclusion_rule=rule,
        execution_plans=tuple(plans[episode.episode_id] for episode in plan.episodes),
        run_descriptor=descriptor,
    )
    _validate_inputs(inputs)
    return inputs


def load_confirmatory_execution_package(
    package_dir: Path, *, authorization: ConfirmatoryAuthorization | None = None
) -> ConfirmatoryExecutionPackage:
    """Load a package with no symlink traversal and exact content-hash custody checks."""
    if not isinstance(package_dir, Path):
        raise _hold("package directory is unsafe")
    _require_nonsymlink_path(package_dir, directory=True)
    manifest = _read_canonical(package_dir / "package-manifest.json")
    manifest_keys = {
        "profile",
        "freeze_manifest_hash",
        "anchor_receipt_hash",
        "plan_hash",
        "catalog_hash",
        "exclusion_rule_hash",
        "run_descriptor_hash",
        "executor_runtime_profile_hash",
        "execution_plans",
        "package_hash",
    }
    legacy_manifest_keys = manifest_keys - {"executor_runtime_profile_hash"}
    if manifest.get("profile") == _LEGACY_PROFILE and set(manifest) == legacy_manifest_keys:
        hash_view = {key: manifest[key] for key in legacy_manifest_keys - {"package_hash"}}
    elif manifest.get("profile") == _PROFILE and set(manifest) == manifest_keys:
        hash_view = {key: manifest[key] for key in manifest_keys - {"package_hash"}}
    else:
        raise _hold("package manifest is invalid")
    if manifest["package_hash"] != sha256_ref(hash_view):
        raise _hold("package manifest hash is invalid")
    if authorization is not None and (
        type(authorization) is not ConfirmatoryAuthorization
        or manifest["freeze_manifest_hash"] != authorization.freeze.manifest_hash
        or manifest["anchor_receipt_hash"] != authorization.receipt.receipt_hash
    ):
        raise _hold("package does not bind authenticated anchor")
    try:
        _read_canonical(package_dir / "plan.json")
        _read_canonical(package_dir / "catalog.json")
        _read_canonical(package_dir / "exclusion-rule.json")
        plan = _load_plan(package_dir / "plan.json")
        catalog = _load_catalog(package_dir / "catalog.json")
        rule = _load_exclusion_rule(package_dir / "exclusion-rule.json")
    except ExperimentArtifactHold as error:
        raise _hold("frozen audit input is invalid") from error
    descriptor_payload = _read_canonical(package_dir / "run-descriptor.json")
    descriptor = _parse_descriptor(descriptor_payload)
    if (
        manifest["plan_hash"] != plan.plan_hash
        or manifest["catalog_hash"] != sha256_ref(_catalog_artifact(catalog))
        or manifest["exclusion_rule_hash"] != rule.rule_hash
        or manifest["run_descriptor_hash"] != descriptor_payload["content_hash"]
        or manifest.get("executor_runtime_profile_hash")
        != (
            None
            if descriptor.executor_runtime_profile is None
            else sha256_ref(descriptor.executor_runtime_profile)
        )
        or catalog.exclusion_rule_hash != rule.rule_hash
    ):
        raise _hold("package frozen input hashes do not bind")
    entries = manifest["execution_plans"]
    if type(entries) is not list or len(entries) != len(plan.episodes):
        raise _hold("execution plan inventory is invalid")
    expected_paths = {
        "package-manifest.json",
        "plan.json",
        "catalog.json",
        "exclusion-rule.json",
        "run-descriptor.json",
        "execution-plans",
    }
    if {item.name for item in package_dir.iterdir()} != expected_paths:
        raise _hold("package directory has unexpected entries")
    execution_dir = package_dir / "execution-plans"
    if execution_dir.is_symlink() or not execution_dir.is_dir():
        raise _hold("execution plan directory is unsafe")
    plans: dict[str, EpisodePlan] = {}
    expected_ids = {episode.episode_id: episode for episode in plan.episodes}
    expected_names: set[str] = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"path", "content_hash"}:
            raise _hold("execution plan inventory entry is invalid")
        item = cast(dict[str, object], entry)
        path = item["path"]
        if type(path) is not str or not path.startswith("execution-plans/") or "/" in path[16:]:
            raise _hold("execution plan path is invalid")
        filename = path.removeprefix("execution-plans/")
        expected_names.add(filename)
        payload = _read_canonical(execution_dir / filename)
        if payload.get("content_hash") != item["content_hash"]:
            raise _hold("execution plan inventory hash is invalid")
        body = {key: value for key, value in payload.items() if key != "content_hash"}
        if payload.get("profile") != _EXECUTION_PROFILE or payload.get(
            "content_hash"
        ) != sha256_ref(body):
            raise _hold("execution plan content hash is invalid")
        episode_id = payload.get("episode_id")
        episode = expected_ids.get(episode_id) if type(episode_id) is str else None
        if (
            episode is None
            or filename != f"{episode.episode_id}.json"
            or payload.get("planned_manifest_hash") != episode.manifest_hash
        ):
            raise _hold("execution plan is outside exact planned matrix")
        execution_plan = _parse_execution_plan(payload.get("execution_plan"))
        try:
            _materialized_plan_binds(episode, execution_plan)
        except ValueError as error:
            raise _hold("execution plan does not bind planned matrix") from error
        model = execution_plan.manifest.executor_model
        if (model.provider, model.model, model.model_version_date) != (
            descriptor.provider,
            descriptor.model,
            descriptor.model_version,
        ):
            raise _hold("execution plan does not bind run descriptor")
        _validate_execution_budget_binding(descriptor.executor_runtime_profile, execution_plan)
        plans[episode.episode_id] = execution_plan
    if set(path.name for path in execution_dir.iterdir()) != expected_names or set(plans) != set(
        expected_ids
    ):
        raise _hold("execution plan files are incomplete or unsafe")
    return ConfirmatoryExecutionPackage(
        plan,
        catalog,
        rule,
        plans,
        descriptor,
        cast(str, manifest["freeze_manifest_hash"]),
        cast(str, manifest["anchor_receipt_hash"]),
        cast(str, manifest["package_hash"]),
        manifest,
    )


__all__ = [
    "ConfirmatoryExecutionPackage",
    "ConfirmatoryExecutionPackageInputs",
    "ConfirmatoryPackageHold",
    "confirmatory_executor_runtime_profile_projection",
    "load_confirmatory_execution_package_inputs",
    "load_confirmatory_execution_package",
    "ollama_gpt_oss_confirmatory_executor_profile",
    "write_confirmatory_execution_package",
]
