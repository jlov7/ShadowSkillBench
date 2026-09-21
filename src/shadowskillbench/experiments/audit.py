"""Fail-closed integrity audit for persisted confirmatory experiment runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.episodes.executor import (
    AgentClaim,
    EpisodeDisposition,
    EpisodeResult,
    parse_episode_result,
)
from shadowskillbench.episodes.models import EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.confirmatory_execution_v4 import (
    V4_RESULT_PROFILE,
    parse_confirmatory_run_bundle_result_v4,
)
from shadowskillbench.experiments.planner import ConfirmatoryEpisodePlan, PlannedEpisode

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUN_PROFILE = "SSB-RUN1"
_BINDINGS_PROFILE = "SSB-AUDIT-BINDINGS1"
_REPORT_PROFILE = "SSB-AUDIT2"
_EXCLUSION_RULE_PROFILE = "SSB-EXCLUSION-RULE2"
_ALLOWED_PROVIDER_EXCLUSION_CODES = (
    "MODEL_PROVIDER_TERMINAL",
    "MODEL_PROVIDER_TRANSIENT",
)
_MATCHED_POLICY_GROUPS = (
    (
        ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
        ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
        ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
    ),
    (
        ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
        ExperimentCondition.B2_AUTHORITY_RESOLVER,
        ExperimentCondition.B3_DETERMINISTIC_GATE,
    ),
)


@dataclass(frozen=True, slots=True)
class PolicyText:
    rendered_hash: str
    rendered_text: str

    def __post_init__(self) -> None:
        if _HASH.fullmatch(self.rendered_hash) is None or type(self.rendered_text) is not str:
            raise ValueError("policy text binding is invalid")
        if self.rendered_hash != _policy_hash(self.rendered_text):
            raise ValueError("policy text does not bind its rendered hash")


@dataclass(frozen=True, slots=True)
class FrozenArtifactCatalog:
    """Verified immutable referents supplied to an audit; absent referents are not trusted."""

    case_manifest_hashes: frozenset[str] = frozenset()
    context_contract_hashes: frozenset[str] = frozenset()
    source_manifest_hashes: frozenset[str] = frozenset()
    compiler_manifest_hashes: frozenset[str] = frozenset()
    compiled_skill_artifact_hashes: frozenset[str] = frozenset()
    rendered_skill_hashes: frozenset[str] = frozenset()
    policies: tuple[PolicyText, ...] = ()
    development_entities: frozenset[str] | None = None
    confirmatory_entities: frozenset[str] | None = None
    development_values: frozenset[str] | None = None
    confirmatory_values: frozenset[str] | None = None
    exclusion_rule_hash: str | None = None

    def __post_init__(self) -> None:
        for values in (
            self.case_manifest_hashes,
            self.context_contract_hashes,
            self.source_manifest_hashes,
            self.compiler_manifest_hashes,
            self.compiled_skill_artifact_hashes,
            self.rendered_skill_hashes,
        ):
            if type(values) is not frozenset or any(
                _HASH.fullmatch(item) is None for item in values
            ):
                raise ValueError("artifact catalog hashes are invalid")
        if type(self.policies) is not tuple or any(
            type(item) is not PolicyText for item in self.policies
        ):
            raise ValueError("artifact catalog policies are invalid")
        if len({item.rendered_hash for item in self.policies}) != len(self.policies):
            raise ValueError("artifact catalog policy hashes must be unique")
        for values in (
            self.development_entities,
            self.confirmatory_entities,
            self.development_values,
            self.confirmatory_values,
        ):
            if values is not None and (
                type(values) is not frozenset or any(type(item) is not str for item in values)
            ):
                raise ValueError("corpus identities must be exact string sets")
        if (
            self.exclusion_rule_hash is not None
            and _HASH.fullmatch(self.exclusion_rule_hash) is None
        ):
            raise ValueError("exclusion_rule_hash is invalid")

    def policy_for(self, rendered_hash: str) -> PolicyText | None:
        return next((item for item in self.policies if item.rendered_hash == rendered_hash), None)

    def projection(self) -> dict[str, object]:
        return {
            "profile": "SSB-AUDIT-CATALOG1",
            "case_manifest_hashes": sorted(self.case_manifest_hashes),
            "context_contract_hashes": sorted(self.context_contract_hashes),
            "source_manifest_hashes": sorted(self.source_manifest_hashes),
            "compiler_manifest_hashes": sorted(self.compiler_manifest_hashes),
            "compiled_skill_artifact_hashes": sorted(self.compiled_skill_artifact_hashes),
            "rendered_skill_hashes": sorted(self.rendered_skill_hashes),
            "policies": [
                {"rendered_hash": item.rendered_hash, "rendered_text": item.rendered_text}
                for item in sorted(self.policies, key=lambda item: item.rendered_hash)
            ],
            "development_entities": (
                None if self.development_entities is None else sorted(self.development_entities)
            ),
            "confirmatory_entities": (
                None if self.confirmatory_entities is None else sorted(self.confirmatory_entities)
            ),
            "development_values": (
                None if self.development_values is None else sorted(self.development_values)
            ),
            "confirmatory_values": (
                None if self.confirmatory_values is None else sorted(self.confirmatory_values)
            ),
            "exclusion_rule_hash": self.exclusion_rule_hash,
        }


@dataclass(frozen=True, slots=True)
class FrozenExclusionRule:
    """Freeze-bound limits for pre-behavior provider outages only."""

    allowed_error_codes: tuple[str, ...]
    max_retries_per_episode: int
    max_total_exclusions: int
    max_exclusions_per_primary_cell: int
    rule_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.allowed_error_codes) is not tuple
            or not self.allowed_error_codes
            or any(type(code) is not str or not code for code in self.allowed_error_codes)
            or tuple(sorted(set(self.allowed_error_codes))) != self.allowed_error_codes
            or any(
                code not in _ALLOWED_PROVIDER_EXCLUSION_CODES for code in self.allowed_error_codes
            )
        ):
            raise ValueError("technical exclusions must be sorted provider outage codes")
        if (
            type(self.max_retries_per_episode) is not int
            or not 1 <= self.max_retries_per_episode <= 5
        ):
            raise ValueError(
                "max_retries_per_episode must be an exact integer from one through five"
            )
        if type(self.max_total_exclusions) is not int or self.max_total_exclusions < 1:
            raise ValueError("max_total_exclusions must be a positive exact integer")
        if self.max_exclusions_per_primary_cell != 1:
            raise ValueError("max_exclusions_per_primary_cell must retain two of three repeats")
        object.__setattr__(
            self,
            "rule_hash",
            sha256_ref(
                {
                    "profile": _EXCLUSION_RULE_PROFILE,
                    "allowed_error_codes": list(self.allowed_error_codes),
                    "max_retries_per_episode": self.max_retries_per_episode,
                    "max_total_exclusions": self.max_total_exclusions,
                    "max_exclusions_per_primary_cell": self.max_exclusions_per_primary_cell,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ExecutionBinding:
    """Persisted materialization facts not present in an EpisodeResult envelope."""

    episode_id: str
    execution_manifest_hash: str
    condition: ExperimentCondition
    context_contract_hash: str
    policy_hash: str | None
    policy_text: str | None
    source_manifest_hash: str | None
    compiler_manifest_hash: str | None
    compiled_skill_artifact_hash: str | None
    rendered_skill_hash: str | None
    provider: str
    model: str
    model_version: str
    package_execution_plan_hash: str

    def __post_init__(self) -> None:
        if type(self.episode_id) is not str or not self.episode_id:
            raise ValueError("execution binding episode_id is invalid")
        for value in (
            self.execution_manifest_hash,
            self.context_contract_hash,
            self.package_execution_plan_hash,
        ):
            if _HASH.fullmatch(value) is None:
                raise ValueError("execution binding hash is invalid")
        for value in (
            self.policy_hash,
            self.source_manifest_hash,
            self.compiler_manifest_hash,
            self.compiled_skill_artifact_hash,
            self.rendered_skill_hash,
        ):
            if value is not None and _HASH.fullmatch(value) is None:
                raise ValueError("execution binding optional hash is invalid")
        if self.policy_text is not None and type(self.policy_text) is not str:
            raise ValueError("execution binding policy_text is invalid")
        if any(
            type(value) is not str or not value.strip() or "\x00" in value
            for value in (self.provider, self.model, self.model_version)
        ):
            raise ValueError("execution binding model identity is invalid")

    def projection(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "execution_manifest_hash": self.execution_manifest_hash,
            "condition": self.condition.value,
            "context_contract_hash": self.context_contract_hash,
            "policy_hash": self.policy_hash,
            "policy_text": self.policy_text,
            "source_manifest_hash": self.source_manifest_hash,
            "compiler_manifest_hash": self.compiler_manifest_hash,
            "compiled_skill_artifact_hash": self.compiled_skill_artifact_hash,
            "rendered_skill_hash": self.rendered_skill_hash,
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "package_execution_plan_hash": self.package_execution_plan_hash,
        }


@dataclass(frozen=True, slots=True)
class RuntimeBinding:
    plan_hash: str
    package_hash: str
    run_descriptor_hash: str
    freeze_manifest_hash: str
    anchor_receipt_hash: str
    operator_endpoint_hash: str
    operator_api_key_environment: str
    provider: str
    model: str
    model_version: str
    executor_runtime_profile_hash: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.plan_hash,
            self.package_hash,
            self.run_descriptor_hash,
            self.freeze_manifest_hash,
            self.anchor_receipt_hash,
            self.operator_endpoint_hash,
        ):
            if type(value) is not str or _HASH.fullmatch(value) is None:
                raise ValueError("runtime binding hash is invalid")
        if any(
            type(value) is not str or not value.strip() or "\x00" in value
            for value in (
                self.operator_api_key_environment,
                self.provider,
                self.model,
                self.model_version,
            )
        ):
            raise ValueError("runtime binding identity is invalid")
        if self.executor_runtime_profile_hash is not None and (
            type(self.executor_runtime_profile_hash) is not str
            or _HASH.fullmatch(self.executor_runtime_profile_hash) is None
        ):
            raise ValueError("runtime executor profile hash is invalid")

    def projection(self) -> dict[str, object]:
        value: dict[str, object] = {
            "profile": (
                "SSB-RUNTIME-BINDING2"
                if self.executor_runtime_profile_hash is not None
                else "SSB-RUNTIME-BINDING1"
            ),
            "plan_hash": self.plan_hash,
            "package_hash": self.package_hash,
            "run_descriptor_hash": self.run_descriptor_hash,
            "freeze_manifest_hash": self.freeze_manifest_hash,
            "anchor_receipt_hash": self.anchor_receipt_hash,
            "operator_endpoint_hash": self.operator_endpoint_hash,
            "operator_api_key_environment": self.operator_api_key_environment,
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
        }
        if self.executor_runtime_profile_hash is not None:
            value["executor_runtime_profile_hash"] = self.executor_runtime_profile_hash
        return value


@dataclass(frozen=True, slots=True, order=True)
class AuditFinding:
    code: str
    subject: str
    detail: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value for value in (self.code, self.subject, self.detail)
        ):
            raise ValueError("audit finding fields must be nonempty strings")

    def projection(self) -> dict[str, str]:
        return {"code": self.code, "subject": self.subject, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ArtifactIntegrityReport:
    status: Literal["PASS", "HOLD"]
    plan_hash: str
    runtime_binding_hash: str | None
    audited_episode_ids: tuple[str, ...]
    findings: tuple[AuditFinding, ...]
    model_failure_count: int = 0
    pre_dispatch_configuration_failure_count: int = 0
    report_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "HOLD"} or _HASH.fullmatch(self.plan_hash) is None:
            raise ValueError("audit report header is invalid")
        if (
            self.runtime_binding_hash is not None
            and _HASH.fullmatch(self.runtime_binding_hash) is None
        ):
            raise ValueError("audit runtime binding hash is invalid")
        if type(self.audited_episode_ids) is not tuple or any(
            type(item) is not str for item in self.audited_episode_ids
        ):
            raise ValueError("audit report episode ids are invalid")
        if type(self.findings) is not tuple or any(
            type(item) is not AuditFinding for item in self.findings
        ):
            raise ValueError("audit report findings are invalid")
        if (
            type(self.model_failure_count) is not int
            or self.model_failure_count < 0
            or type(self.pre_dispatch_configuration_failure_count) is not int
            or self.pre_dispatch_configuration_failure_count < 0
            or self.pre_dispatch_configuration_failure_count > self.model_failure_count
        ):
            raise ValueError("audit report failure counts are invalid")
        if self.findings != tuple(sorted(self.findings)):
            raise ValueError("audit report findings must be ordered")
        if (self.status == "PASS") != (not self.findings):
            raise ValueError("audit status must bind findings")
        object.__setattr__(self, "report_hash", sha256_ref(self.projection()))

    def projection(self) -> dict[str, object]:
        return {
            "profile": _REPORT_PROFILE,
            "status": self.status,
            "plan_hash": self.plan_hash,
            "runtime_binding_hash": self.runtime_binding_hash,
            "audited_episode_ids": list(self.audited_episode_ids),
            "findings": [item.projection() for item in self.findings],
            "model_failure_count": self.model_failure_count,
            "pre_dispatch_configuration_failure_count": (
                self.pre_dispatch_configuration_failure_count
            ),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.projection(), "report_hash": self.report_hash}


def _policy_hash(text: str) -> str:
    return sha256_ref({"profile": "SSB-POLICY1", "rendered_text": text})


def _run_manifest(plan: ConfirmatoryEpisodePlan) -> dict[str, object]:
    return {
        "profile": _RUN_PROFILE,
        "plan_hash": plan.plan_hash,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "manifest_hash": episode.manifest_hash,
                "manifest": episode.manifest_projection(),
            }
            for episode in plan.episodes
        ],
    }


def _read_json(path: Path, findings: list[AuditFinding], subject: str) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        findings.append(AuditFinding("UNREADABLE_ARTIFACT", subject, str(path)))
        return None
    if type(value) is not dict or any(type(key) is not str for key in value):
        findings.append(AuditFinding("INVALID_ARTIFACT", subject, str(path)))
        return None
    return cast(dict[str, object], value)


def _cell_from_manifest(manifest: object) -> tuple[object, ...] | None:
    if type(manifest) is not dict:
        return None
    fields = (
        "stage",
        "condition",
        "domain",
        "case_id",
        "skill_bundle_id",
        "repeat_index",
    )
    if any(field not in manifest for field in fields):
        return None
    cell = tuple(manifest[field] for field in fields)
    if (
        any(type(value) is not str for value in cell[:4])
        or cell[4] is not None
        and type(cell[4]) is not str
        or type(cell[5]) is not int
    ):
        return None
    return cell


def _load_execution_bindings(
    run_directory: Path, findings: list[AuditFinding]
) -> dict[str, ExecutionBinding]:
    directory = run_directory / "execution-bindings"
    if not directory.is_dir():
        findings.append(AuditFinding("UNREADABLE_ARTIFACT", "execution-bindings", str(directory)))
        return {}
    bindings: dict[str, ExecutionBinding] = {}
    for path in sorted(directory.glob("*.json")):
        data = _read_json(path, findings, path.name)
        if data is None:
            continue
        try:
            binding = ExecutionBinding(
                episode_id=cast(str, data["episode_id"]),
                execution_manifest_hash=cast(str, data["execution_manifest_hash"]),
                condition=ExperimentCondition(cast(str, data["condition"])),
                context_contract_hash=cast(str, data["context_contract_hash"]),
                policy_hash=cast(str | None, data["policy_hash"]),
                policy_text=cast(str | None, data["policy_text"]),
                source_manifest_hash=cast(str | None, data["source_manifest_hash"]),
                compiler_manifest_hash=cast(str | None, data["compiler_manifest_hash"]),
                compiled_skill_artifact_hash=cast(str | None, data["compiled_skill_artifact_hash"]),
                rendered_skill_hash=cast(str | None, data["rendered_skill_hash"]),
                provider=cast(str, data["provider"]),
                model=cast(str, data["model"]),
                model_version=cast(str, data["model_version"]),
                package_execution_plan_hash=cast(str, data["package_execution_plan_hash"]),
            )
            if data.get("profile") != _BINDINGS_PROFILE or set(data) != {
                "profile",
                *binding.projection(),
            }:
                raise ValueError("fields")
        except (KeyError, TypeError, ValueError):
            findings.append(AuditFinding("INVALID_EXECUTION_BINDING", path.name, "schema"))
            continue
        if path.name != f"{binding.episode_id}.json":
            findings.append(AuditFinding("INVALID_EXECUTION_BINDING", path.name, "filename"))
            continue
        if binding.episode_id in bindings:
            findings.append(
                AuditFinding("DUPLICATE_EXECUTION_BINDING", binding.episode_id, "episode")
            )
            continue
        bindings[binding.episode_id] = binding
    return bindings


def _load_runtime_binding(
    run_directory: Path, findings: list[AuditFinding]
) -> RuntimeBinding | None:
    data = _read_json(run_directory / "runtime-binding.json", findings, "runtime-binding")
    if data is None:
        return None
    try:
        binding = RuntimeBinding(
            plan_hash=cast(str, data["plan_hash"]),
            package_hash=cast(str, data["package_hash"]),
            run_descriptor_hash=cast(str, data["run_descriptor_hash"]),
            freeze_manifest_hash=cast(str, data["freeze_manifest_hash"]),
            anchor_receipt_hash=cast(str, data["anchor_receipt_hash"]),
            operator_endpoint_hash=cast(str, data["operator_endpoint_hash"]),
            operator_api_key_environment=cast(str, data["operator_api_key_environment"]),
            provider=cast(str, data["provider"]),
            model=cast(str, data["model"]),
            model_version=cast(str, data["model_version"]),
            executor_runtime_profile_hash=(
                None
                if data.get("executor_runtime_profile_hash") is None
                else cast(str, data["executor_runtime_profile_hash"])
            ),
        )
        if data != binding.projection():
            raise ValueError("fields")
    except (KeyError, TypeError, ValueError):
        findings.append(AuditFinding("INVALID_RUNTIME_BINDING", "runtime-binding", "schema"))
        return None
    return binding


def _load_package_inventory(
    run_directory: Path,
    plan: ConfirmatoryEpisodePlan,
    catalog: FrozenArtifactCatalog,
    exclusion_rule: FrozenExclusionRule | None,
    runtime: RuntimeBinding | None,
    findings: list[AuditFinding],
) -> dict[str, str]:
    data = _read_json(run_directory / "package-manifest.json", findings, "package-manifest")
    if data is None:
        return {}
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
    package_profile = data.get("profile")
    view = (
        {key: data[key] for key in legacy_required - {"package_hash"}}
        if package_profile == "SSB-CONFIRMATORY-EXECUTION-PACKAGE1" and set(data) == legacy_required
        else {key: data[key] for key in required - {"package_hash"}}
        if package_profile == "SSB-CONFIRMATORY-EXECUTION-PACKAGE2" and set(data) == required
        else {}
    )
    entries = data.get("execution_plans")
    expected_rule_hash = (
        catalog.exclusion_rule_hash if exclusion_rule is None else exclusion_rule.rule_hash
    )
    if (
        package_profile
        not in {"SSB-CONFIRMATORY-EXECUTION-PACKAGE1", "SSB-CONFIRMATORY-EXECUTION-PACKAGE2"}
        or data.get("package_hash") != sha256_ref(view)
        or data.get("plan_hash") != plan.plan_hash
        or data.get("catalog_hash") != sha256_ref(catalog.projection())
        or expected_rule_hash is None
        or data.get("exclusion_rule_hash") != expected_rule_hash
        or runtime is None
        or data.get("package_hash") != runtime.package_hash
        or data.get("run_descriptor_hash") != runtime.run_descriptor_hash
        or data.get("freeze_manifest_hash") != runtime.freeze_manifest_hash
        or data.get("anchor_receipt_hash") != runtime.anchor_receipt_hash
        or runtime.executor_runtime_profile_hash != data.get("executor_runtime_profile_hash")
        or type(entries) is not list
    ):
        findings.append(AuditFinding("INVALID_PACKAGE_MANIFEST", "package-manifest", "binding"))
        return {}
    inventory: dict[str, str] = {}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"path", "content_hash"}:
            findings.append(
                AuditFinding("INVALID_PACKAGE_MANIFEST", "package-manifest", "inventory")
            )
            return {}
        path = entry.get("path")
        content_hash = entry.get("content_hash")
        if (
            type(path) is not str
            or not path.startswith("execution-plans/")
            or not path.endswith(".json")
            or "/" in path.removeprefix("execution-plans/")
            or type(content_hash) is not str
            or _HASH.fullmatch(content_hash) is None
        ):
            findings.append(
                AuditFinding("INVALID_PACKAGE_MANIFEST", "package-manifest", "inventory")
            )
            return {}
        episode_id = path.removeprefix("execution-plans/").removesuffix(".json")
        if episode_id in inventory:
            findings.append(
                AuditFinding("INVALID_PACKAGE_MANIFEST", "package-manifest", "duplicate")
            )
            return {}
        inventory[episode_id] = content_hash
    if set(inventory) != {episode.episode_id for episode in plan.episodes}:
        findings.append(AuditFinding("INVALID_PACKAGE_MANIFEST", "package-manifest", "matrix"))
        return {}
    return inventory


def _technical_exclusion(result: EpisodeResult) -> bool:
    return (
        result.disposition is EpisodeDisposition.MODEL_FAILURE
        and result.error_code in _ALLOWED_PROVIDER_EXCLUSION_CODES
        and result.claim is AgentClaim.NONE
        and all(
            turn.output is None
            and turn.action is None
            and turn.result is None
            and turn.event is None
            for turn in result.trace
        )
    )


def _pre_dispatch_configuration_failure(result: EpisodeResult) -> str | None:
    """Return the custody failure that proves the model request never dispatched cleanly."""
    if result.disposition is not EpisodeDisposition.MODEL_FAILURE:
        return None
    if result.error_code == "CONFIGURATION_ERROR":
        return "configuration error"
    if any(turn.receipt.attempts == 0 for turn in result.trace):
        return "zero-attempt model call"
    if (
        result.error_code not in _ALLOWED_PROVIDER_EXCLUSION_CODES
        and not _technical_exclusion(result)
        and any(turn.receipt.raw_request_hash is None for turn in result.trace)
    ):
        return "missing request hash"
    return None


def _check_pre_dispatch_configuration_failures(
    results: dict[str, list[EpisodeResult]], findings: list[AuditFinding]
) -> int:
    failures = 0
    for episode_id, result_items in results.items():
        if len(result_items) != 1:
            continue
        detail = _pre_dispatch_configuration_failure(result_items[0])
        if detail is None:
            continue
        failures += 1
        findings.append(AuditFinding("PRE_DISPATCH_CONFIGURATION_FAILURE", episode_id, detail))
    return failures


def _primary_cell(episode: PlannedEpisode) -> tuple[str, str, str, str, str | None]:
    return (
        episode.stage.value,
        episode.condition.value,
        episode.case.domain,
        episode.case.case_id,
        None if episode.skill is None else episode.skill.bundle_id,
    )


def _retry_counts(
    run_directory: Path,
    plan: ConfirmatoryEpisodePlan,
    findings: list[AuditFinding],
) -> dict[str, int]:
    path = run_directory / "run-ledger.jsonl"
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        findings.append(AuditFinding("UNREADABLE_ARTIFACT", "run-ledger.jsonl", str(path)))
        return {}
    previous_hash: str | None = None
    counts: Counter[str] = Counter()
    expected = {episode.episode_id: episode for episode in plan.episodes}
    for sequence, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            findings.append(AuditFinding("INVALID_RUN_LEDGER", "run-ledger.jsonl", "json"))
            return {}
        if type(entry) is not dict:
            findings.append(AuditFinding("INVALID_RUN_LEDGER", "run-ledger.jsonl", "entry"))
            return {}
        entry_hash = entry.get("entry_hash")
        projection = {key: value for key, value in entry.items() if key != "entry_hash"}
        if (
            set(entry)
            != {
                "profile",
                "sequence",
                "previous_hash",
                "plan_hash",
                "event",
                "payload",
                "entry_hash",
            }
            or entry.get("profile") != _RUN_PROFILE
            or entry.get("sequence") != sequence
            or entry.get("previous_hash") != previous_hash
            or entry.get("plan_hash") != plan.plan_hash
            or entry.get("event") not in {"RUN_RESUMED", "EPISODE_RETRY", "EPISODE_COMPLETED"}
            or type(entry.get("payload")) is not dict
            or type(entry_hash) is not str
            or sha256_ref(projection) != entry_hash
        ):
            findings.append(AuditFinding("INVALID_RUN_LEDGER", "run-ledger.jsonl", "custody"))
            return {}
        previous_hash = entry_hash
        if entry["event"] != "EPISODE_RETRY":
            continue
        payload = cast(dict[str, object], entry["payload"])
        episode_id = payload.get("episode_id")
        episode = expected.get(episode_id) if type(episode_id) is str else None
        if (
            episode is None
            or payload.get("manifest_hash") != episode.manifest_hash
            or type(payload.get("attempt")) is not int
            or cast(int, payload["attempt"]) < 1
            or payload.get("error_code") not in _ALLOWED_PROVIDER_EXCLUSION_CODES
        ):
            findings.append(AuditFinding("INVALID_RUN_LEDGER", "run-ledger.jsonl", "retry"))
            return {}
        counts[episode.episode_id] += 1
    return dict(counts)


def _check_technical_exclusions(
    plan: ConfirmatoryEpisodePlan,
    results: dict[str, list[EpisodeResult]],
    catalog: FrozenArtifactCatalog,
    exclusion_rule: FrozenExclusionRule | None,
    retry_counts: dict[str, int],
    findings: list[AuditFinding],
) -> None:
    technical = {
        episode_id: result_items[0]
        for episode_id, result_items in results.items()
        if len(result_items) == 1 and _technical_exclusion(result_items[0])
    }
    rule_is_bound = (
        exclusion_rule is not None and catalog.exclusion_rule_hash == exclusion_rule.rule_hash
    )
    for episode_id, retry_count in retry_counts.items():
        if (
            not rule_is_bound
            or exclusion_rule is None
            or retry_count > exclusion_rule.max_retries_per_episode
        ):
            findings.append(AuditFinding("UNAPPROVED_TECHNICAL_RETRY", episode_id, "retry budget"))
    for episode_id, result in technical.items():
        if (
            not rule_is_bound
            or exclusion_rule is None
            or result.error_code not in exclusion_rule.allowed_error_codes
        ):
            findings.append(
                AuditFinding("UNAPPROVED_TECHNICAL_EXCLUSION", episode_id, "error code")
            )
            continue
        if retry_counts.get(episode_id, 0) != exclusion_rule.max_retries_per_episode:
            findings.append(
                AuditFinding("TECHNICAL_EXCLUSION_RETRY_NOT_EXHAUSTED", episode_id, "retry budget")
            )
    if not rule_is_bound or exclusion_rule is None:
        return
    if len(technical) > exclusion_rule.max_total_exclusions:
        findings.append(
            AuditFinding("TECHNICAL_EXCLUSION_TOTAL_CAP_EXCEEDED", "confirmatory", "total")
        )
    episodes_by_id = {episode.episode_id: episode for episode in plan.episodes}
    technical_by_cell = Counter(
        _primary_cell(episodes_by_id[episode_id]) for episode_id in technical
    )
    planned_by_cell: dict[tuple[str, str, str, str, str | None], tuple[PlannedEpisode, ...]] = {}
    for episode in plan.episodes:
        key = _primary_cell(episode)
        planned_by_cell[key] = (*planned_by_cell.get(key, ()), episode)
    for cell, count in technical_by_cell.items():
        if count > exclusion_rule.max_exclusions_per_primary_cell:
            findings.append(
                AuditFinding("TECHNICAL_EXCLUSION_CELL_CAP_EXCEEDED", repr(cell), "primary cell")
            )
    for cell, episodes in planned_by_cell.items():
        if len(episodes) != 3:
            continue
        retained = sum(episode.episode_id not in technical for episode in episodes)
        if retained < 2:
            findings.append(
                AuditFinding("TECHNICAL_EXCLUSION_ESTIMABILITY_FAILURE", repr(cell), "repeats")
            )


def _check_referents(
    plan: ConfirmatoryEpisodePlan, catalog: FrozenArtifactCatalog, findings: list[AuditFinding]
) -> None:
    requirements: tuple[tuple[str, frozenset[str], str], ...] = (
        ("case_manifest", catalog.case_manifest_hashes, "case_manifest_hash"),
        ("context_contract", catalog.context_contract_hashes, "context_contract_hash"),
        ("source_manifest", catalog.source_manifest_hashes, "source_manifest_hash"),
        ("compiler_manifest", catalog.compiler_manifest_hashes, "compiler_manifest_hash"),
        (
            "compiled_skill_artifact",
            catalog.compiled_skill_artifact_hashes,
            "compiled_skill_artifact_hash",
        ),
        ("rendered_skill", catalog.rendered_skill_hashes, "rendered_skill_hash"),
    )
    emitted: set[tuple[str, str]] = set()
    for episode in plan.episodes:
        source = episode.manifest_projection()
        for name, available, field_name in requirements:
            value = source[field_name]
            if (
                value is not None
                and value not in available
                and (name, cast(str, value)) not in emitted
            ):
                findings.append(AuditFinding("UNVERIFIED_MISSING_REFERENT", cast(str, value), name))
                emitted.add((name, cast(str, value)))
        policy_hash = cast(str | None, source["policy_hash"])
        if (
            policy_hash is not None
            and catalog.policy_for(policy_hash) is None
            and ("policy", policy_hash) not in emitted
        ):
            findings.append(AuditFinding("UNVERIFIED_MISSING_REFERENT", policy_hash, "policy"))
            emitted.add(("policy", policy_hash))


def _check_corpus_isolation(catalog: FrozenArtifactCatalog, findings: list[AuditFinding]) -> None:
    for kind, development, confirmatory in (
        ("entity", catalog.development_entities, catalog.confirmatory_entities),
        ("value", catalog.development_values, catalog.confirmatory_values),
    ):
        if not development or not confirmatory:
            findings.append(AuditFinding("UNVERIFIED_MISSING_REFERENT", kind, "corpus inventory"))
            continue
        for value in sorted(development & confirmatory):
            findings.append(AuditFinding("CORPUS_LEAKAGE", value, kind))


def _check_binding(
    episode: PlannedEpisode,
    binding: ExecutionBinding,
    result: EpisodeResult,
    catalog: FrozenArtifactCatalog,
    findings: list[AuditFinding],
) -> None:
    expected = episode.manifest_projection()
    if binding.execution_manifest_hash != result.manifest_hash:
        findings.append(AuditFinding("EXECUTION_MANIFEST_MISMATCH", episode.episode_id, "result"))
    if binding.condition is not episode.condition:
        findings.append(
            AuditFinding("EXECUTION_CONDITION_MISMATCH", episode.episode_id, "condition")
        )
    if binding.context_contract_hash != expected["context_contract_hash"]:
        findings.append(
            AuditFinding("CONTEXT_CONTRACT_MISMATCH", episode.episode_id, "context_contract_hash")
        )
    if binding.policy_hash != expected["policy_hash"]:
        findings.append(AuditFinding("POLICY_HASH_MISMATCH", episode.episode_id, "policy_hash"))
    if binding.policy_hash is None:
        if binding.policy_text is not None:
            findings.append(
                AuditFinding("POLICY_TEXT_MISMATCH", episode.episode_id, "unexpected policy text")
            )
    elif binding.policy_text is None or _policy_hash(binding.policy_text) != binding.policy_hash:
        findings.append(
            AuditFinding("POLICY_TEXT_MISMATCH", episode.episode_id, "text does not bind hash")
        )
    elif (catalog_policy := catalog.policy_for(binding.policy_hash)) is not None and (
        catalog_policy.rendered_text != binding.policy_text
    ):
        findings.append(AuditFinding("POLICY_TEXT_MISMATCH", episode.episode_id, "frozen policy"))
    for code, field_name in (
        ("CHANGED_SOURCE_MANIFEST", "source_manifest_hash"),
        ("CHANGED_COMPILER_MANIFEST", "compiler_manifest_hash"),
        ("CHANGED_COMPILED_SKILL", "compiled_skill_artifact_hash"),
        ("CHANGED_RENDERED_SKILL", "rendered_skill_hash"),
    ):
        if getattr(binding, field_name) != expected[field_name]:
            findings.append(AuditFinding(code, episode.episode_id, field_name))


def _check_matched_policies(
    episodes: tuple[PlannedEpisode, ...],
    bindings: dict[str, ExecutionBinding],
    findings: list[AuditFinding],
) -> None:
    for domain in ("access_provisioning", "financial_adjustments"):
        for group in _MATCHED_POLICY_GROUPS:
            matching = [
                episode
                for episode in episodes
                if episode.case.domain == domain and episode.condition in group
            ]
            if not matching:
                continue
            expected_hashes = {episode.condition_binding.policy_hash for episode in matching}
            actual = [
                bindings[episode.episode_id]
                for episode in matching
                if episode.episode_id in bindings
            ]
            actual_hashes = {binding.policy_hash for binding in actual}
            actual_texts = {binding.policy_text for binding in actual}
            subject = f"{domain}:{group[0].value}"
            if len(expected_hashes) != 1 or None in expected_hashes or len(actual_hashes) != 1:
                findings.append(
                    AuditFinding("MATCHED_POLICY_HASH_MISMATCH", subject, "policy hash")
                )
            if len(actual_texts) != 1 or None in actual_texts:
                findings.append(
                    AuditFinding("MATCHED_POLICY_TEXT_MISMATCH", subject, "policy text")
                )


def audit_confirmatory_run(
    plan: ConfirmatoryEpisodePlan,
    run_directory: Path,
    *,
    catalog: FrozenArtifactCatalog,
    exclusion_rule: FrozenExclusionRule | None = None,
    stage: EpisodeStage | None = None,
) -> ArtifactIntegrityReport:
    """Audit a complete confirmatory run or one frozen stage without repairing it."""
    if type(plan) is not ConfirmatoryEpisodePlan or not isinstance(run_directory, Path):
        raise ValueError("plan and run_directory must be exact values")
    if type(catalog) is not FrozenArtifactCatalog:
        raise ValueError("catalog must be an exact FrozenArtifactCatalog")
    if exclusion_rule is not None and type(exclusion_rule) is not FrozenExclusionRule:
        raise ValueError("exclusion_rule must be a FrozenExclusionRule")
    if stage is not None and type(stage) is not EpisodeStage:
        raise ValueError("stage must be an EpisodeStage")
    findings: list[AuditFinding] = []
    all_expected = {episode.episode_id: episode for episode in plan.episodes}
    scoped_episodes = tuple(
        episode for episode in plan.episodes if stage is None or episode.stage is stage
    )
    if not scoped_episodes:
        raise ValueError("selected confirmatory stage has no planned episodes")
    expected = {episode.episode_id: episode for episode in scoped_episodes}
    runtime_binding = _load_runtime_binding(run_directory, findings)
    if runtime_binding is not None and runtime_binding.plan_hash != plan.plan_hash:
        findings.append(AuditFinding("RUNTIME_PLAN_MISMATCH", "runtime-binding", "plan hash"))
    package_inventory = _load_package_inventory(
        run_directory, plan, catalog, exclusion_rule, runtime_binding, findings
    )
    manifest = _read_json(run_directory / "run-manifest.json", findings, "run-manifest")
    if manifest is not None:
        entries = manifest.get("episodes")
        if type(entries) is not list:
            findings.append(AuditFinding("INVALID_RUN_MANIFEST", "run-manifest", "episodes"))
        else:
            ids = [
                entry.get("episode_id")
                for entry in entries
                if type(entry) is dict and type(entry.get("episode_id")) is str
            ]
            for episode_id, count in Counter(ids).items():
                if type(episode_id) is str and count > 1:
                    findings.append(
                        AuditFinding("DUPLICATE_PLANNED_RESULT", episode_id, "episode id")
                    )
            cells = [
                _cell_from_manifest(entry.get("manifest"))
                for entry in entries
                if type(entry) is dict
            ]
            for cell, count in Counter(cell for cell in cells if cell is not None).items():
                if count > 1:
                    findings.append(AuditFinding("DUPLICATE_CELL", repr(cell), "run manifest"))
        if manifest != _run_manifest(plan):
            findings.append(AuditFinding("RUN_MANIFEST_MISMATCH", "run-manifest", "planned matrix"))
    parsed_results: dict[str, list[EpisodeResult]] = {}
    results_directory = run_directory / "results"
    if not results_directory.is_dir():
        findings.append(AuditFinding("UNREADABLE_ARTIFACT", "results", str(results_directory)))
    else:
        for path in sorted(results_directory.glob("*.json")):
            envelope = _read_json(path, findings, path.name)
            if envelope is None:
                continue
            episode_id = envelope.get("episode_id")
            try:
                parsed = (
                    parse_confirmatory_run_bundle_result_v4(envelope)
                    if envelope.get("profile") == V4_RESULT_PROFILE
                    else parse_episode_result(envelope["result"])
                )
            except (KeyError, TypeError, ValueError):
                findings.append(
                    AuditFinding("INVALID_RESULT_ENVELOPE", path.name, "semantic result")
                )
                continue
            episode = all_expected.get(episode_id) if type(episode_id) is str else None
            v4 = envelope.get("profile") == V4_RESULT_PROFILE
            if (
                episode is None
                or (
                    not v4
                    and set(envelope)
                    != {
                        "profile",
                        "plan_hash",
                        "episode_id",
                        "planned_manifest_hash",
                        "execution_manifest_hash",
                        "expected_execution_manifest_hash",
                        "result_content_hash",
                        "result",
                    }
                )
                or (not v4 and envelope.get("profile") != _RUN_PROFILE)
                or envelope.get("plan_hash") != plan.plan_hash
                or envelope.get("planned_manifest_hash") != episode.manifest_hash
                or parsed.episode_id != episode.episode_id
                or envelope.get("execution_manifest_hash") != parsed.manifest_hash
                or envelope.get("expected_execution_manifest_hash") != parsed.manifest_hash
                or envelope.get("result_content_hash") != parsed.content_hash
            ):
                findings.append(AuditFinding("INVALID_RESULT_ENVELOPE", path.name, "custody"))
                continue
            if episode.episode_id in expected:
                parsed_results.setdefault(episode.episode_id, []).append(parsed)
    for episode_id, count in Counter(
        {key: len(value) for key, value in parsed_results.items()}
    ).items():
        if count > 1:
            findings.append(
                AuditFinding("DUPLICATE_PLANNED_RESULT", episode_id, "persisted result")
            )
    for episode_id in expected:
        if len(parsed_results.get(episode_id, [])) != 1:
            findings.append(AuditFinding("MISSING_PLANNED_EPISODE", episode_id, "result"))
    retry_counts = {
        episode_id: count
        for episode_id, count in _retry_counts(run_directory, plan, findings).items()
        if episode_id in expected
    }
    bindings = _load_execution_bindings(run_directory, findings)
    for episode_id in sorted(set(bindings) - set(all_expected)):
        findings.append(
            AuditFinding("UNPLANNED_EXECUTION_BINDING", episode_id, "execution binding")
        )
    for episode_id, episode in expected.items():
        result_items = parsed_results.get(episode_id, [])
        binding = bindings.get(episode_id)
        if binding is None:
            findings.append(
                AuditFinding("UNVERIFIED_MISSING_REFERENT", episode_id, "execution binding")
            )
            continue
        if len(result_items) == 1:
            _check_binding(episode, binding, result_items[0], catalog, findings)
        if runtime_binding is not None and (
            binding.provider,
            binding.model,
            binding.model_version,
        ) != (
            runtime_binding.provider,
            runtime_binding.model,
            runtime_binding.model_version,
        ):
            findings.append(AuditFinding("RUNTIME_MODEL_MISMATCH", episode_id, "execution binding"))
        if package_inventory.get(episode_id) != binding.package_execution_plan_hash:
            findings.append(
                AuditFinding("PACKAGE_EXECUTION_PLAN_MISMATCH", episode_id, "package inventory")
            )
    _check_technical_exclusions(
        plan,
        parsed_results,
        catalog,
        exclusion_rule,
        retry_counts,
        findings,
    )
    pre_dispatch_configuration_failure_count = _check_pre_dispatch_configuration_failures(
        parsed_results, findings
    )
    _check_matched_policies(scoped_episodes, bindings, findings)
    _check_referents(plan, catalog, findings)
    _check_corpus_isolation(catalog, findings)
    ordered = tuple(sorted(set(findings)))
    return ArtifactIntegrityReport(
        status="PASS" if not ordered else "HOLD",
        plan_hash=plan.plan_hash,
        runtime_binding_hash=(
            None if runtime_binding is None else sha256_ref(runtime_binding.projection())
        ),
        audited_episode_ids=tuple(sorted(parsed_results)),
        findings=ordered,
        model_failure_count=sum(
            len(result_items)
            for result_items in parsed_results.values()
            if len(result_items) == 1
            and result_items[0].disposition is EpisodeDisposition.MODEL_FAILURE
        ),
        pre_dispatch_configuration_failure_count=pre_dispatch_configuration_failure_count,
    )


audit_confirmatory_integrity = audit_confirmatory_run


__all__ = [
    "ArtifactIntegrityReport",
    "AuditFinding",
    "ExecutionBinding",
    "FrozenArtifactCatalog",
    "FrozenExclusionRule",
    "PolicyText",
    "RuntimeBinding",
    "audit_confirmatory_integrity",
    "audit_confirmatory_run",
]
