"""Fail-closed analysis from a sealed confirmatory experiment bundle.

The persisted bundle deliberately contains raw episode outcomes, not an
author-authored statistics table.  This module verifies the custody bundle,
re-scores every accepted outcome, and only then calls the statistical APIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from shadowskillbench.analysis.dataset import (
    AnalysisDataset,
    ExclusionRow,
    OutcomeRow,
    build_aggregate_metric_rows,
)
from shadowskillbench.analysis.exclusions import ApprovedTechnicalExclusionPolicy
from shadowskillbench.analysis.profile import is_confirmatory_binding
from shadowskillbench.analysis.stage_a import StageAAnalysis, StageAAnalysisError, analyze_stage_a
from shadowskillbench.analysis.stage_b import StageBAnalysis, StageBAnalysisError, analyze_stage_b
from shadowskillbench.authority import DecisionDisposition
from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.engine import TaskCase
from shadowskillbench.episodes import (
    BlockOrder,
    EpisodeResult,
    EpisodeStage,
    ExperimentCondition,
    parse_episode_result,
)
from shadowskillbench.experiments.audit import (
    ArtifactIntegrityReport,
    FrozenExclusionRule,
    RuntimeBinding,
    audit_confirmatory_run,
)
from shadowskillbench.experiments.confirmatory_execution_v4 import (
    V4_RESULT_PROFILE,
    parse_confirmatory_run_bundle_result_v4,
)
from shadowskillbench.experiments.io import (
    ConfirmatoryAuditInputs,
    ExperimentArtifactHold,
    load_confirmatory_audit_inputs,
)
from shadowskillbench.experiments.planner import AuthorityClass, Domain, PlannedEpisode
from shadowskillbench.metrics import score_episode
from shadowskillbench.protocol.scientific_freeze import ScientificFreezeBinding
from shadowskillbench.protocol.scientific_freeze_v4 import ScientificFreezeV4Binding
from shadowskillbench.reporting import hash_analysis_dataset


class SealedAnalysisHold(ValueError):
    """Named CLI-bound refusal for incomplete or unsealed analysis input."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class SealedAnalysis:
    dataset: AnalysisDataset
    stage_a: StageAAnalysis
    stage_b: StageBAnalysis
    audit: ArtifactIntegrityReport
    runtime_binding: RuntimeBinding
    exclusion_policy: ApprovedTechnicalExclusionPolicy | None = None
    execution_design_projection: tuple[tuple[str, int, int, int], ...] | None = None
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None = None

    @property
    def dataset_sha256(self) -> str:
        return hash_analysis_dataset(self.dataset)


def _reject_duplicate_keys(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"noncanonical JSON constant: {value}")


def _canonical_object(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise SealedAnalysisHold("HOLD_INVALID_ANALYSIS_ARTIFACTS", label) from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise SealedAnalysisHold("HOLD_NONCANONICAL_ANALYSIS_ARTIFACTS", label)
    return cast(dict[str, object], value)


def _canonical_ledger(path: Path) -> None:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise SealedAnalysisHold("HOLD_MISSING_ANALYSIS_ARTIFACTS", "run-ledger.jsonl") from error
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise SealedAnalysisHold("HOLD_NONCANONICAL_ANALYSIS_ARTIFACTS", "run-ledger.jsonl")
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(
                line[:-1].decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise SealedAnalysisHold(
                "HOLD_INVALID_ANALYSIS_ARTIFACTS", f"run-ledger.jsonl:{index}"
            ) from error
        if type(value) is not dict or canonical_json_bytes(value) + b"\n" != line:
            raise SealedAnalysisHold(
                "HOLD_NONCANONICAL_ANALYSIS_ARTIFACTS", f"run-ledger.jsonl:{index}"
            )


def _canonical_run_files(inputs: ConfirmatoryAuditInputs) -> None:
    root = inputs.run_directory
    _canonical_object(root / "run-manifest.json", "run-manifest.json")
    _canonical_object(root / "runtime-binding.json", "runtime-binding.json")
    _canonical_object(root / "package-manifest.json", "package-manifest.json")
    _canonical_ledger(root / "run-ledger.jsonl")
    for directory in ("results", "execution-bindings"):
        try:
            paths = sorted((root / directory).iterdir())
        except OSError as error:
            raise SealedAnalysisHold("HOLD_MISSING_ANALYSIS_ARTIFACTS", directory) from error
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise SealedAnalysisHold("HOLD_UNSAFE_ANALYSIS_ARTIFACTS", path.name)
            _canonical_object(path, f"{directory}/{path.name}")


def _case_for(result: EpisodeResult) -> TaskCase:
    domain_case = result.domain_case
    return TaskCase(
        case_id=domain_case.case_id,
        domain=domain_case.domain,
        world_id=domain_case.world_id,
        seed=domain_case.seed,
        objective="sealed confirmatory outcome",
        inputs={},
    )


def _result_for(path: Path) -> EpisodeResult:
    payload = _canonical_object(path, f"results/{path.name}")
    if payload.get("profile") == V4_RESULT_PROFILE:
        try:
            return parse_confirmatory_run_bundle_result_v4(payload)
        except (TypeError, ValueError) as error:
            raise SealedAnalysisHold(
                "HOLD_INVALID_ANALYSIS_ARTIFACTS", f"results/{path.name}"
            ) from error
    if set(payload) != {
        "profile",
        "plan_hash",
        "episode_id",
        "planned_manifest_hash",
        "execution_manifest_hash",
        "expected_execution_manifest_hash",
        "result_content_hash",
        "result",
    }:
        raise SealedAnalysisHold("HOLD_INVALID_ANALYSIS_ARTIFACTS", f"results/{path.name}")
    try:
        return parse_episode_result(payload["result"])
    except (TypeError, ValueError) as error:
        raise SealedAnalysisHold(
            "HOLD_INVALID_ANALYSIS_ARTIFACTS", f"results/{path.name}"
        ) from error


def _row(result: EpisodeResult, planned: PlannedEpisode) -> OutcomeRow | ExclusionRow:
    try:
        score = score_episode(result, _case_for(result))
    except (TypeError, ValueError) as error:
        raise SealedAnalysisHold("HOLD_UNSCORABLE_ACCEPTED_OUTCOME", result.episode_id) from error
    if result.episode_id != planned.episode_id or score.case_id != planned.case.case_id:
        raise SealedAnalysisHold("HOLD_CROSS_HASH_ANALYSIS_MISMATCH", result.episode_id)
    stage: EpisodeStage = planned.stage
    condition: ExperimentCondition = planned.condition
    domain: Domain = planned.case.domain
    contamination_ratio = (
        None if planned.skill is None else float(planned.skill.contamination_ratio)
    )
    skill_bundle_id = None if planned.skill is None else planned.skill.bundle_id
    authority_class: AuthorityClass | None = planned.case.authority_class
    order_assignment: BlockOrder | None = planned.order_assignment
    disposition: DecisionDisposition = result.authority_decision.disposition
    if score.technical_exclusion:
        return ExclusionRow(
            stage,
            condition,
            domain,
            contamination_ratio,
            skill_bundle_id,
            authority_class,
            order_assignment,
            disposition,
            planned.case.case_id,
            planned.repeat_index,
            result.episode_id,
            planned.manifest_hash,
            result.manifest_hash,
            result.content_hash,
            score.score_hash,
            score.error_code,
        )
    return OutcomeRow(
        stage,
        condition,
        domain,
        contamination_ratio,
        skill_bundle_id,
        authority_class,
        order_assignment,
        disposition,
        planned.case.case_id,
        planned.repeat_index,
        result.episode_id,
        planned.manifest_hash,
        result.manifest_hash,
        result.content_hash,
        score.score_hash,
        task_completion=score.task_completion,
        completion_under_policy=score.completion_under_policy,
        policy_violation=score.policy_violation,
        unsafe_imitation=score.unsafe_imitation,
        false_enforcement=score.false_enforcement,
        escalation_expected=score.escalation_expected,
        escalation_correct=score.escalation_correct,
        authority_resolution_correct=score.authority_resolution_correct,
        claim_outcome_mismatch=score.claim_outcome_mismatch,
        turns=score.turns,
        tool_calls=score.tool_calls,
        input_tokens=score.input_tokens,
        output_tokens=score.output_tokens,
        total_tokens=score.total_tokens,
        total_cost_nanos=score.total_cost_nanos,
    )


def _dataset(inputs: ConfirmatoryAuditInputs) -> AnalysisDataset:
    planned = {episode.episode_id: episode for episode in inputs.plan.episodes}
    rows: list[OutcomeRow] = []
    exclusions: list[ExclusionRow] = []
    for path in sorted((inputs.run_directory / "results").glob("*.json")):
        result = _result_for(path)
        episode = planned.get(result.episode_id)
        if episode is None:
            raise SealedAnalysisHold("HOLD_CROSS_HASH_ANALYSIS_MISMATCH", path.name)
        record = _row(result, episode)
        if isinstance(record, OutcomeRow):
            rows.append(record)
        else:
            exclusions.append(cast(ExclusionRow, record))
    if len(rows) + len(exclusions) != len(planned):
        raise SealedAnalysisHold("HOLD_PARTIAL_ANALYSIS_DATASET", "accepted outcome coverage")
    return AnalysisDataset(
        rows=tuple(sorted(rows, key=lambda value: value.episode_id)),
        exclusions=tuple(sorted(exclusions, key=lambda value: value.episode_id)),
        aggregates=build_aggregate_metric_rows(tuple(rows)),
    )


def _analysis_exclusion_policy(
    rule: FrozenExclusionRule | None,
) -> ApprovedTechnicalExclusionPolicy | None:
    if rule is None:
        return None
    return ApprovedTechnicalExclusionPolicy(
        allowed_error_codes=rule.allowed_error_codes,
        max_total_exclusions=rule.max_total_exclusions,
        max_exclusions_per_primary_cell=rule.max_exclusions_per_primary_cell,
    )


def _confirmatory_profile_projection(
    stage_a: StageAAnalysis,
    stage_b: StageBAnalysis,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None = None,
) -> dict[str, object]:
    """Bind the frozen statistics profile into the sealed analysis receipt."""

    a = stage_a.inference
    b = stage_b.inference
    if not is_confirmatory_binding(
        stage="A",
        bootstrap_replicates=a.bootstrap_replicates,
        bootstrap_seed=a.bootstrap_seed,
        confidence_level=a.confidence_level,
        method=a.method,
        classification=a.classification,
        profile=a.profile,
        profile_hash=a.profile_hash,
        statistics_configuration_sha256=a.statistics_configuration_sha256,
        scientific_freeze=scientific_freeze,
    ) or not is_confirmatory_binding(
        stage="B",
        bootstrap_replicates=b.bootstrap_replicates,
        bootstrap_seed=b.bootstrap_seed,
        confidence_level=b.confidence_level,
        method=b.method,
        classification=b.classification,
        profile=b.profile,
        profile_hash=b.profile_hash,
        statistics_configuration_sha256=b.statistics_configuration_sha256,
        scientific_freeze=scientific_freeze,
    ):
        raise SealedAnalysisHold(
            "HOLD_ANALYSIS_PROFILE_MISMATCH",
            "sealed confirmatory analysis must use the frozen statistics profile",
        )
    if (a.profile, a.profile_hash, a.statistics_configuration_sha256) != (
        b.profile,
        b.profile_hash,
        b.statistics_configuration_sha256,
    ):
        raise SealedAnalysisHold("HOLD_ANALYSIS_PROFILE_MISMATCH", "Stage A/B profile binding")
    if (
        scientific_freeze is not None
        and a.statistics_configuration_sha256 != scientific_freeze.statistics_configuration_sha256
    ):
        raise SealedAnalysisHold(
            "HOLD_ANALYSIS_PROFILE_MISMATCH",
            "frozen statistics configuration does not match the analysis profile",
        )
    return {
        "classification": a.classification,
        "profile": a.profile,
        "profile_hash": a.profile_hash,
        "statistics_configuration_sha256": a.statistics_configuration_sha256,
        "bootstrap_replicates": a.bootstrap_replicates,
        "bootstrap_seed": a.bootstrap_seed,
        "confidence_level": a.confidence_level,
        "stage_a_method": a.method,
        "stage_b_method": b.method,
    }


def analyze_sealed_confirmatory_artifacts(
    artifacts_root: Path,
    *,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None = None,
) -> SealedAnalysis:
    """Return recomputed Stage A/B analyses only from audited raw outcomes."""
    try:
        inputs = load_confirmatory_audit_inputs(artifacts_root)
    except ExperimentArtifactHold as error:
        raise SealedAnalysisHold(
            error.code.replace("EXPERIMENT", "ANALYSIS"), error.detail
        ) from error
    _canonical_run_files(inputs)
    audit = audit_confirmatory_run(
        inputs.plan,
        inputs.run_directory,
        catalog=inputs.catalog,
        exclusion_rule=inputs.exclusion_rule,
    )
    if audit.status != "PASS":
        codes = ",".join(sorted({finding.code for finding in audit.findings})) or "UNKNOWN"
        raise SealedAnalysisHold("HOLD_EXPERIMENT_AUDIT", codes)
    dataset = _dataset(inputs)
    exclusion_policy = _analysis_exclusion_policy(inputs.exclusion_rule)
    try:
        stage_a = analyze_stage_a(
            dataset,
            bootstrap_replicates=(
                scientific_freeze.bootstrap_replicates if scientific_freeze is not None else 999
            ),
            bootstrap_seed=(
                scientific_freeze.bootstrap_seed if scientific_freeze is not None else 0
            ),
            exclusion_policy=exclusion_policy,
            scientific_freeze=scientific_freeze,
        )
        stage_b = analyze_stage_b(
            dataset,
            bootstrap_replicates=(
                scientific_freeze.bootstrap_replicates if scientific_freeze is not None else 999
            ),
            bootstrap_seed=(
                scientific_freeze.bootstrap_seed if scientific_freeze is not None else 0
            ),
            exclusion_policy=exclusion_policy,
            scientific_freeze=scientific_freeze,
        )
    except (StageAAnalysisError, StageBAnalysisError) as error:
        raise SealedAnalysisHold("HOLD_INCOMPLETE_CONFIRMATORY_DATASET", str(error)) from error
    _confirmatory_profile_projection(stage_a, stage_b, scientific_freeze)
    return SealedAnalysis(
        dataset=dataset,
        stage_a=stage_a,
        stage_b=stage_b,
        audit=audit,
        runtime_binding=inputs.runtime_binding,
        exclusion_policy=exclusion_policy,
        execution_design_projection=(
            scientific_freeze.member_execution_designs if scientific_freeze is not None else None
        ),
        scientific_freeze=scientific_freeze,
    )


def sealed_analysis_projection(analysis: SealedAnalysis) -> dict[str, object]:
    """Canonical receipt with hashes only; no authored aggregate is consumed later."""
    projection = {
        "profile": "SSB-SEALED-ANALYSIS-2",
        "plan_hash": analysis.audit.plan_hash,
        "audit_report_sha256": analysis.audit.report_hash,
        "analysis_dataset_sha256": analysis.dataset_sha256,
        "analysis_profile": _confirmatory_profile_projection(
            analysis.stage_a, analysis.stage_b, analysis.scientific_freeze
        ),
        "execution_design_projection": _receipt_execution_design_projection(
            analysis.execution_design_projection
        ),
        "runtime_binding": analysis.runtime_binding.projection(),
    }
    if type(analysis.scientific_freeze) is ScientificFreezeV4Binding:
        projection["profile"] = "SSB-SEALED-ANALYSIS-3"
        projection["scientific_freeze_manifest_sha256"] = analysis.scientific_freeze.manifest_hash
        projection["scientific_freeze_version"] = "V4"
    return projection


def _receipt_execution_design_projection(
    projection: tuple[tuple[str, int, int, int], ...] | None,
) -> list[list[object]]:
    if (
        type(projection) is not tuple
        or not projection
        or any(
            type(item) is not tuple
            or len(item) != 4
            or type(item[0]) is not str
            or any(type(value) is not int or value < 1 for value in item[1:])
            for item in projection
        )
    ):
        raise SealedAnalysisHold("HOLD_INVALID_ANALYSIS_RECEIPT", "execution design projection")
    return [list(item) for item in projection]


def sealed_analysis_receipt_matches(receipt: object, analysis: SealedAnalysis) -> bool:
    """Compare a parsed v2 receipt without accepting JSON type coercion."""

    if type(receipt) is not dict:
        return False
    projection = receipt.get("execution_design_projection")
    if (
        type(projection) is not list
        or not projection
        or any(
            type(item) is not list
            or len(item) != 4
            or type(item[0]) is not str
            or any(type(value) is not int or value < 1 for value in item[1:])
            for item in projection
        )
    ):
        return False
    try:
        return receipt == sealed_analysis_projection(analysis)
    except SealedAnalysisHold:
        return False


__all__ = [
    "SealedAnalysis",
    "SealedAnalysisHold",
    "analyze_sealed_confirmatory_artifacts",
    "sealed_analysis_projection",
    "sealed_analysis_receipt_matches",
]
