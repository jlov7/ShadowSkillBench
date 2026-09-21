"""Strict local loaders for confirmatory experiment audit artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.episodes.models import BlockOrder, EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.audit import (
    FrozenArtifactCatalog,
    FrozenExclusionRule,
    PolicyText,
    RuntimeBinding,
)
from shadowskillbench.experiments.planner import (
    AuthorityClass,
    ConditionBinding,
    ConfirmatoryEpisodePlan,
    Domain,
    HeldOutCaseBinding,
    PlannedEpisode,
    SkillBundleBinding,
    hash_confirmatory_episode_plan,
    hash_planned_episode_manifest,
)

_PLAN_PROFILE = "SSB-PLAN1"
_CATALOG_PROFILE = "SSB-AUDIT-CATALOG1"
_EXCLUSION_PROFILE = "SSB-EXCLUSION-RULE2"
_PLAN_KEYS = {"profile", "plan_hash", "episodes"}
_PLAN_ENTRY_KEYS = {"manifest", "manifest_hash"}
_MANIFEST_KEYS = {
    "profile",
    "episode_id",
    "stage",
    "condition",
    "domain",
    "case_id",
    "case_manifest_hash",
    "world_hash",
    "authority_graph_hash",
    "authority_class",
    "context_contract_hash",
    "policy_hash",
    "skill_bundle_id",
    "contamination_ratio",
    "source_manifest_hash",
    "compiler_manifest_hash",
    "compiled_skill_artifact_hash",
    "rendered_skill_hash",
    "repeat_index",
    "order_assignment",
}
_CATALOG_KEYS = {
    "profile",
    "case_manifest_hashes",
    "context_contract_hashes",
    "source_manifest_hashes",
    "compiler_manifest_hashes",
    "compiled_skill_artifact_hashes",
    "rendered_skill_hashes",
    "policies",
    "development_entities",
    "confirmatory_entities",
    "development_values",
    "confirmatory_values",
    "exclusion_rule_hash",
}


class ExperimentArtifactHold(RuntimeError):
    """A fail-closed local custody error intended for the CLI boundary."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ConfirmatoryAuditInputs:
    plan: ConfirmatoryEpisodePlan
    catalog: FrozenArtifactCatalog
    exclusion_rule: FrozenExclusionRule | None
    runtime_binding: RuntimeBinding
    package_manifest: dict[str, object]
    run_directory: Path


def _invalid(detail: str) -> ExperimentArtifactHold:
    return ExperimentArtifactHold("HOLD_INVALID_EXPERIMENT_ARTIFACTS", detail)


def _safe_relative(value: Path) -> tuple[str, ...]:
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or "\\" in raw
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ExperimentArtifactHold("HOLD_UNSAFE_EXPERIMENT_ARTIFACTS", raw or "root")
    return path.parts


def _safe_root(value: Path) -> Path:
    parts = _safe_relative(value)
    root = Path.cwd()
    try:
        if root.is_symlink() or not root.is_dir():
            raise ExperimentArtifactHold("HOLD_UNSAFE_EXPERIMENT_ARTIFACTS", "working directory")
        for part in parts:
            root = root / part
            if root.is_symlink():
                raise ExperimentArtifactHold("HOLD_UNSAFE_EXPERIMENT_ARTIFACTS", str(root))
        if not root.exists():
            raise ExperimentArtifactHold("HOLD_MISSING_EXPERIMENT_ARTIFACTS", str(root))
        if not root.is_dir():
            raise _invalid(f"not a directory: {root}")
        root.resolve(strict=True).relative_to(Path.cwd().resolve(strict=True))
    except ExperimentArtifactHold:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ExperimentArtifactHold("HOLD_UNSAFE_EXPERIMENT_ARTIFACTS", str(root)) from error
    return root


def _safe_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    candidate = root
    try:
        for part in PurePosixPath(relative).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ExperimentArtifactHold("HOLD_UNSAFE_EXPERIMENT_ARTIFACTS", relative)
        if not candidate.exists():
            raise ExperimentArtifactHold("HOLD_MISSING_EXPERIMENT_ARTIFACTS", relative)
        if (not candidate.is_dir()) if directory else (not candidate.is_file()):
            raise _invalid(relative)
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ExperimentArtifactHold:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ExperimentArtifactHold("HOLD_UNSAFE_EXPERIMENT_ARTIFACTS", relative) from error
    return candidate


def _reject_duplicate_keys(pairs: list[tuple[object, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in parsed:
            raise ValueError("duplicate or invalid JSON object key")
        parsed[key] = value
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"noncanonical JSON constant: {value}")


def _read_canonical_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise _invalid(label) from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise _invalid(label)
    return cast(dict[str, object], value)


def _exact_object(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise _invalid(label)
    return cast(dict[str, object], value)


def _sorted_string_list(
    value: object, *, label: str, allow_none: bool = False
) -> frozenset[str] | None:
    if value is None and allow_none:
        return None
    if type(value) is not list or any(type(item) is not str for item in value):
        raise _invalid(label)
    values = cast(list[str], value)
    if values != sorted(set(values)):
        raise _invalid(label)
    return frozenset(values)


def _episode_from_manifest(value: object) -> PlannedEpisode:
    manifest = _exact_object(value, _MANIFEST_KEYS, label="episode manifest")
    try:
        if manifest["profile"] != _PLAN_PROFILE:
            raise ValueError("profile")
        condition = ExperimentCondition(cast(str, manifest["condition"]))
        stage = EpisodeStage(cast(str, manifest["stage"]))
        case = HeldOutCaseBinding(
            domain=cast(Domain, manifest["domain"]),
            case_id=cast(str, manifest["case_id"]),
            case_manifest_hash=cast(str, manifest["case_manifest_hash"]),
            world_hash=cast(str, manifest["world_hash"]),
            authority_graph_hash=cast(str, manifest["authority_graph_hash"]),
            authority_class=cast(AuthorityClass | None, manifest["authority_class"]),
        )
        binding = ConditionBinding(
            domain=case.domain,
            condition=condition,
            context_contract_hash=cast(str, manifest["context_contract_hash"]),
            policy_hash=cast(str | None, manifest["policy_hash"]),
        )
        skill_values = (
            manifest["skill_bundle_id"],
            manifest["contamination_ratio"],
            manifest["source_manifest_hash"],
            manifest["compiler_manifest_hash"],
            manifest["compiled_skill_artifact_hash"],
            manifest["rendered_skill_hash"],
        )
        skill: SkillBundleBinding | None = None
        if any(item is not None for item in skill_values):
            if any(item is None for item in skill_values):
                raise ValueError("partial skill binding")
            ratio = Decimal(cast(str, manifest["contamination_ratio"]))
            skill = SkillBundleBinding(
                domain=case.domain,
                bundle_id=cast(str, manifest["skill_bundle_id"]),
                contamination_ratio=ratio,
                source_manifest_hash=cast(str, manifest["source_manifest_hash"]),
                compiler_manifest_hash=cast(str, manifest["compiler_manifest_hash"]),
                compiled_skill_artifact_hash=cast(str, manifest["compiled_skill_artifact_hash"]),
                rendered_skill_hash=cast(str, manifest["rendered_skill_hash"]),
            )
        order_raw = manifest["order_assignment"]
        order = None if order_raw is None else BlockOrder(cast(str, order_raw))
        episode = PlannedEpisode(
            stage=stage,
            condition=condition,
            case=case,
            condition_binding=binding,
            repeat_index=cast(int, manifest["repeat_index"]),
            skill=skill,
            order_assignment=order,
        )
    except (InvalidOperation, TypeError, ValueError) as error:
        raise _invalid("episode manifest") from error
    if manifest != episode.manifest_projection():
        raise _invalid("episode manifest")
    return episode


def _load_plan(path: Path) -> ConfirmatoryEpisodePlan:
    payload = _exact_object(
        _read_canonical_object(path, label="plan.json"), _PLAN_KEYS, label="plan.json"
    )
    if payload["profile"] != _PLAN_PROFILE or type(payload["plan_hash"]) is not str:
        raise _invalid("plan.json")
    entries = payload["episodes"]
    if type(entries) is not list or not entries:
        raise _invalid("plan.json")
    episodes: list[PlannedEpisode] = []
    for value in entries:
        entry = _exact_object(value, _PLAN_ENTRY_KEYS, label="plan episode")
        episode = _episode_from_manifest(entry["manifest"])
        if entry["manifest_hash"] != hash_planned_episode_manifest(episode):
            raise _invalid("plan episode hash")
        episodes.append(episode)
    try:
        plan = ConfirmatoryEpisodePlan(episodes=tuple(episodes))
    except ValueError as error:
        raise _invalid("plan episode matrix") from error
    if payload["plan_hash"] != hash_confirmatory_episode_plan(plan):
        raise _invalid("plan hash")
    return plan


def _load_catalog(path: Path) -> FrozenArtifactCatalog:
    payload = _exact_object(
        _read_canonical_object(path, label="catalog.json"), _CATALOG_KEYS, label="catalog.json"
    )
    if payload["profile"] != _CATALOG_PROFILE:
        raise _invalid("catalog.json")
    policies_value = payload["policies"]
    if type(policies_value) is not list:
        raise _invalid("catalog policies")
    policies: list[PolicyText] = []
    for value in policies_value:
        item = _exact_object(value, {"rendered_hash", "rendered_text"}, label="catalog policy")
        try:
            policies.append(
                PolicyText(
                    rendered_hash=cast(str, item["rendered_hash"]),
                    rendered_text=cast(str, item["rendered_text"]),
                )
            )
        except ValueError as error:
            raise _invalid("catalog policy") from error
    if [policy.rendered_hash for policy in policies] != sorted(
        policy.rendered_hash for policy in policies
    ):
        raise _invalid("catalog policies")
    try:
        return FrozenArtifactCatalog(
            case_manifest_hashes=cast(
                frozenset[str],
                _sorted_string_list(payload["case_manifest_hashes"], label="case hashes"),
            ),
            context_contract_hashes=cast(
                frozenset[str],
                _sorted_string_list(payload["context_contract_hashes"], label="context hashes"),
            ),
            source_manifest_hashes=cast(
                frozenset[str],
                _sorted_string_list(payload["source_manifest_hashes"], label="source hashes"),
            ),
            compiler_manifest_hashes=cast(
                frozenset[str],
                _sorted_string_list(payload["compiler_manifest_hashes"], label="compiler hashes"),
            ),
            compiled_skill_artifact_hashes=cast(
                frozenset[str],
                _sorted_string_list(
                    payload["compiled_skill_artifact_hashes"], label="compiled hashes"
                ),
            ),
            rendered_skill_hashes=cast(
                frozenset[str],
                _sorted_string_list(
                    payload["rendered_skill_hashes"], label="rendered skill hashes"
                ),
            ),
            policies=tuple(policies),
            development_entities=_sorted_string_list(
                payload["development_entities"], label="development entities", allow_none=True
            ),
            confirmatory_entities=_sorted_string_list(
                payload["confirmatory_entities"], label="confirmatory entities", allow_none=True
            ),
            development_values=_sorted_string_list(
                payload["development_values"], label="development values", allow_none=True
            ),
            confirmatory_values=_sorted_string_list(
                payload["confirmatory_values"], label="confirmatory values", allow_none=True
            ),
            exclusion_rule_hash=cast(str | None, payload["exclusion_rule_hash"]),
        )
    except ValueError as error:
        raise _invalid("catalog.json") from error


def _load_exclusion_rule(path: Path) -> FrozenExclusionRule:
    payload = _exact_object(
        _read_canonical_object(path, label="exclusion-rule.json"),
        {
            "profile",
            "allowed_error_codes",
            "max_retries_per_episode",
            "max_total_exclusions",
            "max_exclusions_per_primary_cell",
            "rule_hash",
        },
        label="exclusion-rule.json",
    )
    if (
        payload["profile"] != _EXCLUSION_PROFILE
        or type(payload["allowed_error_codes"]) is not list
        or type(payload["max_retries_per_episode"]) is not int
        or type(payload["max_total_exclusions"]) is not int
        or type(payload["max_exclusions_per_primary_cell"]) is not int
    ):
        raise _invalid("exclusion-rule.json")
    try:
        rule = FrozenExclusionRule(
            allowed_error_codes=tuple(cast(list[str], payload["allowed_error_codes"])),
            max_retries_per_episode=cast(int, payload["max_retries_per_episode"]),
            max_total_exclusions=cast(int, payload["max_total_exclusions"]),
            max_exclusions_per_primary_cell=cast(int, payload["max_exclusions_per_primary_cell"]),
        )
    except ValueError as error:
        raise _invalid("exclusion-rule.json") from error
    if payload["rule_hash"] != rule.rule_hash:
        raise _invalid("exclusion-rule hash")
    return rule


def _validate_run_paths(root: Path) -> None:
    _safe_path(root, "run-manifest.json")
    for directory in ("results", "execution-bindings"):
        path = _safe_path(root, directory, directory=True)
        try:
            for child in path.iterdir():
                if child.is_symlink() or not child.is_file():
                    raise ExperimentArtifactHold("HOLD_UNSAFE_EXPERIMENT_ARTIFACTS", str(child))
        except ExperimentArtifactHold:
            raise
        except OSError as error:
            raise ExperimentArtifactHold("HOLD_UNSAFE_EXPERIMENT_ARTIFACTS", directory) from error


def _validate_runtime_binding(root: Path, plan: ConfirmatoryEpisodePlan) -> RuntimeBinding:
    payload = _read_canonical_object(
        _safe_path(root, "runtime-binding.json"), label="runtime binding"
    )
    try:
        binding = RuntimeBinding(
            plan_hash=cast(str, payload["plan_hash"]),
            package_hash=cast(str, payload["package_hash"]),
            run_descriptor_hash=cast(str, payload["run_descriptor_hash"]),
            freeze_manifest_hash=cast(str, payload["freeze_manifest_hash"]),
            anchor_receipt_hash=cast(str, payload["anchor_receipt_hash"]),
            operator_endpoint_hash=cast(str, payload["operator_endpoint_hash"]),
            operator_api_key_environment=cast(str, payload["operator_api_key_environment"]),
            provider=cast(str, payload["provider"]),
            model=cast(str, payload["model"]),
            model_version=cast(str, payload["model_version"]),
            executor_runtime_profile_hash=(
                None
                if payload.get("executor_runtime_profile_hash") is None
                else cast(str, payload["executor_runtime_profile_hash"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _invalid("runtime binding") from error
    if payload != binding.projection() or binding.plan_hash != plan.plan_hash:
        raise _invalid("runtime binding")
    return binding


def _validate_package_manifest(
    root: Path,
    plan: ConfirmatoryEpisodePlan,
    catalog_payload: dict[str, object],
    exclusion_rule: FrozenExclusionRule | None,
    runtime: RuntimeBinding,
) -> dict[str, object]:
    payload = _read_canonical_object(
        _safe_path(root, "package-manifest.json"), label="package manifest"
    )
    required = {
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
    legacy_required = required - {"executor_runtime_profile_hash"}
    view = (
        {key: payload[key] for key in legacy_required - {"package_hash"}}
        if payload.get("profile") == "SSB-CONFIRMATORY-EXECUTION-PACKAGE1"
        and set(payload) == legacy_required
        else {key: payload[key] for key in required - {"package_hash"}}
        if payload.get("profile") == "SSB-CONFIRMATORY-EXECUTION-PACKAGE2"
        and set(payload) == required
        else {}
    )
    entries = payload.get("execution_plans")
    expected_paths = {f"execution-plans/{episode.episode_id}.json" for episode in plan.episodes}
    if (
        payload.get("profile")
        not in {"SSB-CONFIRMATORY-EXECUTION-PACKAGE1", "SSB-CONFIRMATORY-EXECUTION-PACKAGE2"}
        or payload.get("package_hash") != sha256_ref(view)
        or payload.get("package_hash") != runtime.package_hash
        or payload.get("plan_hash") != plan.plan_hash
        or payload.get("catalog_hash") != sha256_ref(catalog_payload)
        or exclusion_rule is None
        or payload.get("exclusion_rule_hash") != exclusion_rule.rule_hash
        or payload.get("run_descriptor_hash") != runtime.run_descriptor_hash
        or payload.get("freeze_manifest_hash") != runtime.freeze_manifest_hash
        or payload.get("anchor_receipt_hash") != runtime.anchor_receipt_hash
        or payload.get("executor_runtime_profile_hash") != runtime.executor_runtime_profile_hash
        or type(entries) is not list
        or len(entries) != len(plan.episodes)
        or any(
            type(entry) is not dict
            or set(entry) != {"path", "content_hash"}
            or type(entry.get("path")) is not str
            or type(entry.get("content_hash")) is not str
            for entry in entries
        )
        or {cast(dict[str, object], entry)["path"] for entry in entries} != expected_paths
    ):
        raise _invalid("package manifest")
    return payload


def load_confirmatory_audit_inputs(artifacts_root: Path) -> ConfirmatoryAuditInputs:
    """Load an exact local audit bundle without resolving symlinks or repairing data."""
    root = _safe_root(artifacts_root)
    plan = _load_plan(_safe_path(root, "plan.json"))
    catalog_payload = _read_canonical_object(_safe_path(root, "catalog.json"), label="catalog")
    catalog = _load_catalog(_safe_path(root, "catalog.json"))
    exclusion_path = root / "exclusion-rule.json"
    exclusion_rule = None
    if exclusion_path.exists() or exclusion_path.is_symlink():
        exclusion_rule = _load_exclusion_rule(_safe_path(root, "exclusion-rule.json"))
        if catalog.exclusion_rule_hash != exclusion_rule.rule_hash:
            raise _invalid("catalog exclusion rule hash")
    elif catalog.exclusion_rule_hash is not None:
        raise _invalid("missing exclusion-rule.json")
    _validate_run_paths(root)
    runtime_binding = _validate_runtime_binding(root, plan)
    package_manifest = _validate_package_manifest(
        root, plan, catalog_payload, exclusion_rule, runtime_binding
    )
    return ConfirmatoryAuditInputs(
        plan=plan,
        catalog=catalog,
        exclusion_rule=exclusion_rule,
        runtime_binding=runtime_binding,
        package_manifest=package_manifest,
        run_directory=root,
    )


__all__ = [
    "ConfirmatoryAuditInputs",
    "ExperimentArtifactHold",
    "load_confirmatory_audit_inputs",
]
