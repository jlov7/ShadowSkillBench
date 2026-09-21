"""Deterministic, self-contained, evidence-bound HTML reporting."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from html import escape
from math import isfinite
from pathlib import Path
from string import Template
from typing import Literal

from shadowskillbench.analysis.dataset import (
    AggregateMetricRow,
    AnalysisDataset,
    ExclusionRow,
    OutcomeRow,
)
from shadowskillbench.analysis.exclusions import ApprovedTechnicalExclusionPolicy
from shadowskillbench.analysis.profile import is_confirmatory_binding, scientific_freeze_version
from shadowskillbench.analysis.stage_a import (
    AverageMarginalEffect,
    RawCuPCount,
    ResponseCurvePoint,
    StageAAnalysis,
    StageAEstimand,
    StageAInference,
    StageAModel,
)
from shadowskillbench.analysis.stage_b import (
    DeltaEstimate,
    Figure2Point,
    MacroGovernanceEffect,
    RateEstimate,
    StageBAnalysis,
    StageBInference,
)
from shadowskillbench.authority import DecisionDisposition
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.episodes import EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.audit import ArtifactIntegrityReport, RuntimeBinding
from shadowskillbench.experiments.planner import (
    AUTHORITY_CLASSES,
    DOMAINS,
    RATIOS,
    STAGE_A_EPISODE_COUNT,
    STAGE_B_EPISODE_COUNT,
)
from shadowskillbench.experiments.planner_v4 import (
    STAGE_A_EPISODE_COUNT as V4_STAGE_A_EPISODE_COUNT,
)
from shadowskillbench.experiments.planner_v4 import (
    STAGE_B_EPISODE_COUNT as V4_STAGE_B_EPISODE_COUNT,
)
from shadowskillbench.protocol import ClaimsReport, PreregistrationReport
from shadowskillbench.protocol.scientific_freeze import ScientificFreezeBinding
from shadowskillbench.protocol.scientific_freeze_v4 import ScientificFreezeV4Binding
from shadowskillbench.reporting.selection import (
    SelectionCandidate,
    SelectionResult,
    SelectionStatus,
    select_stage_a,
    select_stage_b,
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{7,64}\Z")
_TEMPLATE_PATH = Path(__file__).with_name("templates") / "report.html.j2"


class ReportInputError(ValueError):
    """Raised when a renderer input cannot safely bind a report artifact."""


class ReportStatus(StrEnum):
    CONFIRMATORY = "confirmatory"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class ProtocolEvidence:
    """Post-freeze identifiers required to render a confirmatory report."""

    protocol_tag: str
    freeze_manifest_sha256: str
    prompt_hashes: tuple[str, ...]
    code_commit: str
    reproduce_command: str
    external_anchor_locator: str | None = None
    custody_locator: str | None = None
    custody_mode: Literal["EXTERNAL_SIGNED_ANCHOR", "LOCAL_HASH_CUSTODY"] | None = None
    anchor_receipt_sha256: str | None = None
    anchor_verified: bool = False
    experiment_plan_sha256: str | None = None
    experiment_runtime_binding_sha256: str | None = None
    experiment_audit_report_sha256: str | None = None
    experiment_audit_status: Literal["PASS"] | None = None

    def __post_init__(self) -> None:
        for field in ("protocol_tag", "reproduce_command"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip() or "\x00" in value:
                raise ReportInputError(f"{field} must be nonblank text")
        if self.external_anchor_locator is not None and (
            type(self.external_anchor_locator) is not str
            or not self.external_anchor_locator.strip()
            or "\x00" in self.external_anchor_locator
        ):
            raise ReportInputError("external_anchor_locator must be nonblank text when present")
        if self.custody_locator is not None and (
            type(self.custody_locator) is not str
            or not self.custody_locator.strip()
            or "\x00" in self.custody_locator
        ):
            raise ReportInputError("custody_locator must be nonblank text when present")
        if self.custody_mode is not None and self.custody_mode not in {
            "EXTERNAL_SIGNED_ANCHOR",
            "LOCAL_HASH_CUSTODY",
        }:
            raise ReportInputError("custody_mode is invalid")
        if self.custody_mode is not None and self.custody_locator is None:
            raise ReportInputError("custody_mode requires a custody_locator")
        if (
            type(self.freeze_manifest_sha256) is not str
            or _SHA256.fullmatch(self.freeze_manifest_sha256) is None
        ):
            raise ReportInputError("freeze_manifest_sha256 must be a sha256 reference")
        if type(self.prompt_hashes) is not tuple or not self.prompt_hashes:
            raise ReportInputError("prompt_hashes must be a non-empty tuple")
        if any(
            type(item) is not str or _SHA256.fullmatch(item) is None for item in self.prompt_hashes
        ):
            raise ReportInputError("prompt_hashes must contain sha256 references")
        if type(self.code_commit) is not str or _COMMIT.fullmatch(self.code_commit) is None:
            raise ReportInputError("code_commit must be a lowercase hexadecimal revision")
        for field in (
            "anchor_receipt_sha256",
            "experiment_plan_sha256",
            "experiment_runtime_binding_sha256",
            "experiment_audit_report_sha256",
        ):
            value = getattr(self, field)
            if value is not None and (type(value) is not str or _SHA256.fullmatch(value) is None):
                raise ReportInputError(f"{field} must be a sha256 reference when present")
        if type(self.anchor_verified) is not bool:
            raise ReportInputError("anchor_verified must be an exact boolean")
        if self.experiment_audit_status is not None and (
            type(self.experiment_audit_status) is not str or self.experiment_audit_status != "PASS"
        ):
            raise ReportInputError("experiment_audit_status must be PASS when present")


@dataclass(frozen=True, slots=True)
class RepresentativeTrace:
    """An explicitly selected trace or an explicit selection HOLD."""

    stage: str
    selection: SelectionResult
    trace_artifact_sha256: str | None = None
    summary: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"A", "B"}:
            raise ReportInputError("representative trace stage must be A or B")
        if type(self.selection) is not SelectionResult or self.selection.stage != self.stage:
            raise ReportInputError("representative trace selection must bind its stage")
        selected = self.selection.status is SelectionStatus.SELECTED
        if selected:
            if self.selection.selected is None:
                raise ReportInputError("selected trace is missing its candidate")
            if (
                type(self.trace_artifact_sha256) is not str
                or _SHA256.fullmatch(self.trace_artifact_sha256) is None
            ):
                raise ReportInputError("selected trace requires an artifact hash")
            if type(self.summary) is not str or not self.summary.strip() or "\x00" in self.summary:
                raise ReportInputError("selected trace requires a nonblank summary")
        elif self.selection.status is SelectionStatus.HOLD_NO_CANDIDATE:
            if self.trace_artifact_sha256 is not None or self.summary is not None:
                raise ReportInputError("a held trace selection cannot render a substitute trace")
        else:
            raise ReportInputError("representative trace has an unknown selection status")


def select_representative_traces(
    stage_a_candidates: tuple[SelectionCandidate, ...],
    stage_b_candidates: tuple[SelectionCandidate, ...],
    trace_bindings: Mapping[str, tuple[str, str]],
) -> tuple[RepresentativeTrace, RepresentativeTrace]:
    """Select real representative cases from explicit, custody-bound summaries.

    The report builder supplies case-level summaries and the corresponding public
    trace hash/summary.  No placeholder case IDs are invented; a selected case
    without a public binding fails closed.
    """
    selections = (select_stage_a(stage_a_candidates), select_stage_b(stage_b_candidates))
    traces: list[RepresentativeTrace] = []
    for stage, selection in zip(("A", "B"), selections):
        if selection.status is SelectionStatus.HOLD_NO_CANDIDATE:
            traces.append(RepresentativeTrace(stage, selection))
            continue
        if selection.selected is None:
            raise ReportInputError("selected representative case is missing")
        binding = trace_bindings.get(selection.selected.case_id)
        if binding is None or len(binding) != 2:
            raise ReportInputError("selected representative case has no public trace binding")
        traces.append(
            RepresentativeTrace(
                stage,
                selection,
                trace_artifact_sha256=binding[0],
                summary=binding[1],
            )
        )
    return (traces[0], traces[1])


def render_report_artifacts(
    evidence: ReportEvidence,
    *,
    public_episode_traces: tuple[object, ...] = (),
) -> tuple[bytes, bytes]:
    """Render HTML and workbench bytes from one immutable ``ReportEvidence``.

    The local import keeps the reporting modules acyclic while making it
    difficult for callers to accidentally build the two artifacts from separate
    evidence objects.
    """
    from shadowskillbench.reporting.workbench_export import render_workbench_json

    return (
        render_report(evidence).encode("utf-8"),
        render_workbench_json(evidence, public_episode_traces=public_episode_traces),  # type: ignore[arg-type]
    )


def write_report_artifacts(
    evidence: ReportEvidence,
    html_destination: Path,
    workbench_destination: Path,
    *,
    public_episode_traces: tuple[object, ...] = (),
) -> None:
    """Exclusively publish both artifacts generated from one evidence object."""
    if not isinstance(html_destination, Path) or not isinstance(workbench_destination, Path):
        raise ReportInputError("report artifact destinations must be Paths")
    if html_destination == workbench_destination:
        raise ReportInputError("report artifact destinations must be distinct")
    destinations = (html_destination, workbench_destination)
    if any(
        path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink()
        for path in destinations
    ):
        raise ReportInputError("report artifact destination exists or is unsafe")
    html_bytes, workbench_bytes = render_report_artifacts(
        evidence, public_episode_traces=public_episode_traces
    )
    temporaries: list[Path] = []
    linked: list[Path] = []
    try:
        for destination, payload in zip(destinations, (html_bytes, workbench_bytes)):
            temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
            if temporary.exists() or temporary.is_symlink():
                raise ReportInputError("report artifact temporary path exists")
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporaries.append(temporary)
        for temporary, destination in zip(temporaries, destinations):
            os.link(temporary, destination)
            linked.append(destination)
    except FileExistsError as error:
        raise ReportInputError("report artifact destination appeared during publish") from error
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
        if len(linked) != len(destinations):
            for destination in linked:
                destination.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ReportEvidence:
    """All evidence that can appear in a rendered report.

    The renderer does not load files, call models, or infer missing trace
    metadata.  A caller must bind every displayed value to an already-validated
    artifact before rendering.
    """

    dataset: AnalysisDataset
    stage_a: StageAAnalysis
    stage_b: StageBAnalysis
    preregistration: PreregistrationReport
    claims: ClaimsReport
    protocol: ProtocolEvidence
    analysis_dataset_sha256: str
    representative_traces: tuple[RepresentativeTrace, ...]
    limitations: tuple[str, ...]
    exclusion_policy: ApprovedTechnicalExclusionPolicy | None = None
    runtime_binding: RuntimeBinding | None = None
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None = None

    def __post_init__(self) -> None:
        if type(self.dataset) is not AnalysisDataset:
            raise ReportInputError("dataset must be an exact AnalysisDataset")
        if type(self.stage_a) is not StageAAnalysis or type(self.stage_b) is not StageBAnalysis:
            raise ReportInputError("stage analyses must be exact analysis records")
        if type(self.preregistration) is not PreregistrationReport:
            raise ReportInputError("preregistration must be an exact PreregistrationReport")
        if type(self.claims) is not ClaimsReport:
            raise ReportInputError("claims must be an exact ClaimsReport")
        if type(self.protocol) is not ProtocolEvidence:
            raise ReportInputError("protocol must be exact ProtocolEvidence")
        if self.runtime_binding is not None and type(self.runtime_binding) is not RuntimeBinding:
            raise ReportInputError("runtime_binding must be an exact RuntimeBinding when present")
        if (
            self.scientific_freeze is not None
            and scientific_freeze_version(self.scientific_freeze) is None
        ):
            raise ReportInputError(
                "scientific_freeze must be an exact V3 or V4 binding when present"
            )
        if (
            type(self.analysis_dataset_sha256) is not str
            or _SHA256.fullmatch(self.analysis_dataset_sha256) is None
        ):
            raise ReportInputError("analysis_dataset_sha256 must be a sha256 reference")
        if self.analysis_dataset_sha256 != hash_analysis_dataset(self.dataset):
            raise ReportInputError("analysis_dataset_sha256 does not bind the dataset")
        if type(self.representative_traces) is not tuple:
            raise ReportInputError("representative_traces must be an exact tuple")
        traces = tuple(self.representative_traces)
        if any(type(trace) is not RepresentativeTrace for trace in traces):
            raise ReportInputError("representative_traces must contain exact RepresentativeTrace")
        if tuple(trace.stage for trace in traces) != ("A", "B"):
            raise ReportInputError("representative traces must be ordered Stage A then Stage B")
        if type(self.limitations) is not tuple or not self.limitations:
            raise ReportInputError("limitations must be a non-empty tuple")
        if any(
            type(item) is not str or not item.strip() or "\x00" in item for item in self.limitations
        ):
            raise ReportInputError("limitations must contain nonblank text")


def _confirmatory_counts(dataset: AnalysisDataset) -> tuple[int, int, int]:
    stage_a = sum(row.stage is EpisodeStage.CONFIRMATORY_A for row in dataset.rows)
    stage_b = sum(row.stage is EpisodeStage.CONFIRMATORY_B for row in dataset.rows)
    excluded = sum(
        row.stage in {EpisodeStage.CONFIRMATORY_A, EpisodeStage.CONFIRMATORY_B}
        for row in dataset.exclusions
    )
    return stage_a, stage_b, excluded


def _expected_confirmatory_counts(
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None,
) -> tuple[int, int]:
    if scientific_freeze is None or type(scientific_freeze) is ScientificFreezeBinding:
        return (STAGE_A_EPISODE_COUNT, STAGE_B_EPISODE_COUNT)
    if type(scientific_freeze) is ScientificFreezeV4Binding:
        return (V4_STAGE_A_EPISODE_COUNT, V4_STAGE_B_EPISODE_COUNT)
    raise ReportInputError("scientific_freeze must be an exact V3 or V4 binding")


def _exclusions_are_approved(
    dataset: AnalysisDataset,
    policy: ApprovedTechnicalExclusionPolicy | None,
) -> bool:
    exclusions = tuple(
        item
        for item in dataset.exclusions
        if item.stage in {EpisodeStage.CONFIRMATORY_A, EpisodeStage.CONFIRMATORY_B}
    )
    if not exclusions:
        return True
    if policy is None or len(exclusions) > policy.max_total_exclusions:
        return False
    if any(
        item.error_code is None or item.error_code not in policy.allowed_error_codes
        for item in exclusions
    ):
        return False
    cell_counts: dict[tuple[object, ...], int] = {}
    for item in exclusions:
        key = (
            item.stage,
            item.condition,
            item.domain,
            item.contamination_ratio,
            item.skill_bundle_id,
            item.held_out_case_id,
        )
        cell_counts[key] = cell_counts.get(key, 0) + 1
    return all(count <= policy.max_exclusions_per_primary_cell for count in cell_counts.values())


def _json_projection(value: object) -> object:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, StrEnum):
        return value.value
    if type(value) is tuple:
        return [_json_projection(item) for item in value]
    if type(value) is dict:
        return {str(key): _json_projection(item) for key, item in value.items()}
    raise ReportInputError(f"dataset projection has unsupported value type {type(value).__name__}")


def _record_projection(record: OutcomeRow | ExclusionRow | AggregateMetricRow) -> dict[str, object]:
    return {name: _json_projection(value) for name, value in asdict(record).items()}


def _sorted_records(
    records: tuple[OutcomeRow, ...] | tuple[ExclusionRow, ...] | tuple[AggregateMetricRow, ...],
    expected_type: type[OutcomeRow] | type[ExclusionRow] | type[AggregateMetricRow],
) -> list[dict[str, object]]:
    if any(type(record) is not expected_type for record in records):
        raise ReportInputError(f"dataset contains non-exact {expected_type.__name__} records")
    values = [_record_projection(record) for record in records]
    return sorted(values, key=canonical_json_bytes)


def dataset_projection(dataset: AnalysisDataset) -> dict[str, object]:
    """Return a canonical, order-independent projection of rendered dataset evidence."""
    if type(dataset) is not AnalysisDataset:
        raise ReportInputError("dataset must be an exact AnalysisDataset")
    return {
        "profile": "SSB-REPORT-DATASET-1",
        "rows": _sorted_records(dataset.rows, OutcomeRow),
        "exclusions": _sorted_records(dataset.exclusions, ExclusionRow),
        "aggregates": _sorted_records(dataset.aggregates, AggregateMetricRow),
    }


def hash_analysis_dataset(dataset: AnalysisDataset) -> str:
    """Hash the exact table rendered by this report using the project canonicalizer."""
    return sha256_ref(dataset_projection(dataset))


_SCOPES = (*DOMAINS, "pooled")
_RATIO_VALUES = tuple(float(value) for value in RATIOS)
_STAGE_A_SKILL_CONDITIONS = (
    ExperimentCondition.A2_SKILL_ONLY,
    ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
    ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
    ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER,
)
_STAGE_A_CURVE_CONDITIONS = (ExperimentCondition.A1_POLICY_ONLY_SYSTEM, *_STAGE_A_SKILL_CONDITIONS)
_STAGE_B_CONDITIONS = (
    ExperimentCondition.B0_SKILL_ONLY,
    ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
    ExperimentCondition.B2_AUTHORITY_RESOLVER,
    ExperimentCondition.B3_DETERMINISTIC_GATE,
)
_VALID_EXCEPTION_OR_SUPERSEDED = ("APPROVED_SCOPED_EXCEPTION", "POLICY_SUPERSEDED")
_ESTIMAND_LAYOUT = (
    ("E1", "slope"),
    ("E2", "slope"),
    ("E3", "slope"),
    ("E3", "average_cup_contrast"),
    ("E4", "average_cup_contrast"),
    ("E5", "average_cup_contrast"),
)


def _rate_is_valid(rate: object) -> bool:
    return (
        type(rate) is RateEstimate
        and type(rate.numerator) is int
        and type(rate.denominator) is int
        and 0 <= rate.numerator <= rate.denominator
        and rate.denominator > 0
        and all(isfinite(value) and 0 <= value <= 1 for value in (rate.ci_low, rate.ci_high))
        and rate.ci_low <= rate.rate <= rate.ci_high
    )


def _stage_a_is_complete(
    analysis: StageAAnalysis,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None = None,
) -> bool:
    expected_raw = {
        (scope, condition, ratio)
        for scope in _SCOPES
        for condition in (ExperimentCondition.A0_BARE, ExperimentCondition.A1_POLICY_ONLY_SYSTEM)
        for ratio in (None,)
    } | {
        (scope, condition, ratio)
        for scope in _SCOPES
        for condition in _STAGE_A_SKILL_CONDITIONS
        for ratio in _RATIO_VALUES
    }
    if any(type(row) is not RawCuPCount for row in analysis.raw_counts):
        return False
    raw_keys = {(row.scope, row.condition, row.contamination_ratio) for row in analysis.raw_counts}
    if raw_keys != expected_raw or len(raw_keys) != len(analysis.raw_counts):
        return False
    if any(
        type(row.numerator) is not int
        or type(row.denominator) is not int
        or not 0 <= row.numerator <= row.denominator
        or row.denominator == 0
        for row in analysis.raw_counts
    ):
        return False

    expected_curves = {
        (scope, condition, ratio)
        for scope in _SCOPES
        for condition in _STAGE_A_CURVE_CONDITIONS
        for ratio in _RATIO_VALUES
    }
    if any(type(point) is not ResponseCurvePoint for point in analysis.response_curves):
        return False
    curve_keys = {
        (point.scope, point.condition, point.contamination_ratio)
        for point in analysis.response_curves
    }
    if curve_keys != expected_curves or len(curve_keys) != len(analysis.response_curves):
        return False
    if any(
        type(point.numerator) is not int
        or type(point.denominator) is not int
        or not 0 <= point.numerator <= point.denominator
        or point.denominator == 0
        or any(
            not isfinite(value) or not 0 <= value <= 1
            for value in (point.estimate, point.ci_low, point.ci_high)
        )
        or not point.ci_low <= point.estimate <= point.ci_high
        for point in analysis.response_curves
    ):
        return False

    expected_estimands = {
        (estimand, component, scope)
        for scope in _SCOPES
        for estimand, component in _ESTIMAND_LAYOUT
    }
    if any(type(estimand) is not StageAEstimand for estimand in analysis.estimands):
        return False
    estimand_keys = {
        (estimand.estimand, estimand.component, estimand.scope) for estimand in analysis.estimands
    }
    if estimand_keys != expected_estimands or len(estimand_keys) != len(analysis.estimands):
        return False
    if any(
        any(not isfinite(value) for value in (estimand.estimate, estimand.ci_low, estimand.ci_high))
        or not estimand.ci_low <= estimand.estimate <= estimand.ci_high
        for estimand in analysis.estimands
    ):
        return False

    expected_marginal_effects = {
        ("contamination_ratio", scope, condition, None, ratio)
        for scope in _SCOPES
        for condition in _STAGE_A_SKILL_CONDITIONS
        for ratio in _RATIO_VALUES
    } | {
        (
            "condition_contrast",
            scope,
            condition,
            ExperimentCondition.A2_SKILL_ONLY,
            ratio,
        )
        for scope in _SCOPES
        for condition in _STAGE_A_SKILL_CONDITIONS[1:]
        for ratio in _RATIO_VALUES
    }
    if any(
        type(effect) is not AverageMarginalEffect for effect in analysis.average_marginal_effects
    ):
        return False
    marginal_effect_keys = {
        (
            effect.effect,
            effect.scope,
            effect.condition,
            effect.reference_condition,
            effect.contamination_ratio,
        )
        for effect in analysis.average_marginal_effects
    }
    if (
        marginal_effect_keys != expected_marginal_effects
        or len(marginal_effect_keys) != len(analysis.average_marginal_effects)
        or any(
            any(not isfinite(value) for value in (effect.estimate, effect.ci_low, effect.ci_high))
            or not effect.ci_low <= effect.estimate <= effect.ci_high
            for effect in analysis.average_marginal_effects
        )
    ):
        return False

    if type(scientific_freeze) is ScientificFreezeV4Binding:
        expected_observations, expected_skill_clusters, expected_case_clusters = (16_800, 70, 40)
    elif scientific_freeze is None or type(scientific_freeze) is ScientificFreezeBinding:
        expected_observations, expected_skill_clusters, expected_case_clusters = (7_200, 30, 40)
    else:
        return False
    model = analysis.model
    if type(model) is not StageAModel or (
        model.formula != "CuP ~ contamination_ratio * condition + domain"
        or model.family != "Binomial"
        or model.link != "logit"
        or len(model.parameter_names) != 9
        or len(model.coefficients) != 9
        or len(model.covariance) != 9
        or any(len(row) != 9 for row in model.covariance)
        or any(not isfinite(value) for value in (*model.coefficients, *sum(model.covariance, ())))
        or model.observation_count != expected_observations
        or model.skill_bundle_cluster_count != expected_skill_clusters
        or model.held_out_case_cluster_count != expected_case_clusters
        or model.converged is not True
    ):
        return False
    inference = analysis.inference
    return type(inference) is StageAInference and is_confirmatory_binding(
        stage="A",
        bootstrap_replicates=inference.bootstrap_replicates,
        bootstrap_seed=inference.bootstrap_seed,
        confidence_level=inference.confidence_level,
        method=inference.method,
        classification=inference.classification,
        profile=inference.profile,
        profile_hash=inference.profile_hash,
        statistics_configuration_sha256=inference.statistics_configuration_sha256,
        scientific_freeze=scientific_freeze,
    )


def _stage_b_is_complete(
    analysis: StageBAnalysis,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None = None,
) -> bool:
    inference = analysis.inference
    if type(inference) is not StageBInference or not is_confirmatory_binding(
        stage="B",
        bootstrap_replicates=inference.bootstrap_replicates,
        bootstrap_seed=inference.bootstrap_seed,
        confidence_level=inference.confidence_level,
        method=inference.method,
        classification=inference.classification,
        profile=inference.profile,
        profile_hash=inference.profile_hash,
        statistics_configuration_sha256=inference.statistics_configuration_sha256,
        scientific_freeze=scientific_freeze,
    ):
        return False
    expected_e6 = {
        (domain, authority_class) for domain in _SCOPES for authority_class in AUTHORITY_CLASSES
    }
    if any(type(rate) is not RateEstimate for rate in analysis.e6_unsafe_imitation):
        return False
    e6_keys = {(rate.domain, rate.authority_class) for rate in analysis.e6_unsafe_imitation}
    if e6_keys != expected_e6 or len(e6_keys) != len(analysis.e6_unsafe_imitation):
        return False
    if any(
        not _rate_is_valid(rate)
        or rate.condition is not ExperimentCondition.B0_SKILL_ONLY
        or rate.metric != "unsafe_imitation"
        or rate.authority_class is None
        or rate.eligible_authority_classes != (rate.authority_class,)
        or rate.expected_authority_disposition is not None
        for rate in analysis.e6_unsafe_imitation
    ):
        return False

    expected_e7 = {
        (domain, authority_class)
        for domain in _SCOPES
        for authority_class in _VALID_EXCEPTION_OR_SUPERSEDED
    }
    if any(type(rate) is not RateEstimate for rate in analysis.e7_false_enforcement):
        return False
    e7_keys = {(rate.domain, rate.authority_class) for rate in analysis.e7_false_enforcement}
    if e7_keys != expected_e7 or len(e7_keys) != len(analysis.e7_false_enforcement):
        return False
    if any(
        not _rate_is_valid(rate)
        or rate.condition is not ExperimentCondition.B1_FLAT_POLICY_SYSTEM
        or rate.metric != "false_enforcement"
        or rate.authority_class is None
        or rate.eligible_authority_classes != (rate.authority_class,)
        or rate.expected_authority_disposition is not None
        for rate in analysis.e7_false_enforcement
    ):
        return False

    if any(
        type(effect) is not MacroGovernanceEffect for effect in analysis.e8_authority_aware_gain
    ):
        return False
    e8_domains = {effect.domain for effect in analysis.e8_authority_aware_gain}
    if e8_domains != set(_SCOPES) or len(e8_domains) != len(analysis.e8_authority_aware_gain):
        return False
    for effect in analysis.e8_authority_aware_gain:
        components = (*effect.baseline_components, *effect.comparison_components)
        if (
            len(effect.baseline_components) != len(AUTHORITY_CLASSES)
            or len(effect.comparison_components) != len(AUTHORITY_CLASSES)
            or any(not _rate_is_valid(rate) for rate in components)
            or {rate.authority_class for rate in effect.baseline_components}
            != set(AUTHORITY_CLASSES)
            or {rate.authority_class for rate in effect.comparison_components}
            != set(AUTHORITY_CLASSES)
            or any(
                rate.eligible_authority_classes != (rate.authority_class,)
                or rate.expected_authority_disposition is not None
                for rate in components
            )
            or any(rate.domain != effect.domain for rate in components)
            or any(
                rate.condition is not ExperimentCondition.B1_FLAT_POLICY_SYSTEM
                for rate in effect.baseline_components
            )
            or any(
                rate.condition is not ExperimentCondition.B2_AUTHORITY_RESOLVER
                for rate in effect.comparison_components
            )
            or any(rate.metric != "authority_resolution_correct" for rate in components)
            or any(not isfinite(value) for value in (effect.delta, effect.ci_low, effect.ci_high))
            or not effect.ci_low <= effect.delta <= effect.ci_high
        ):
            return False

    if not _delta_series_is_complete(
        analysis.e9_completion_under_policy_delta, "completion_under_policy"
    ) or not _delta_series_is_complete(analysis.e9_unsafe_imitation_delta, "unsafe_imitation"):
        return False

    expected_figure_2 = {
        (domain, condition) for domain in _SCOPES for condition in _STAGE_B_CONDITIONS
    }
    if any(type(point) is not Figure2Point for point in analysis.figure_2_points):
        return False
    figure_2_keys = {(point.domain, point.condition) for point in analysis.figure_2_points}
    if figure_2_keys != expected_figure_2 or len(figure_2_keys) != len(analysis.figure_2_points):
        return False
    return all(
        _rate_is_valid(point.unsafe_imitation)
        and _rate_is_valid(point.false_enforcement)
        and point.unsafe_imitation.domain == point.domain
        and point.false_enforcement.domain == point.domain
        and point.unsafe_imitation.condition is point.condition
        and point.false_enforcement.condition is point.condition
        and point.unsafe_imitation.metric == "unsafe_imitation"
        and point.false_enforcement.metric == "false_enforcement"
        and point.unsafe_imitation.authority_class is None
        and point.false_enforcement.authority_class is None
        and point.unsafe_imitation.eligible_authority_classes == AUTHORITY_CLASSES
        and point.false_enforcement.eligible_authority_classes == AUTHORITY_CLASSES
        and point.unsafe_imitation.expected_authority_disposition is None
        and point.false_enforcement.expected_authority_disposition is DecisionDisposition.PROCEED
        for point in analysis.figure_2_points
    )


def _delta_series_is_complete(
    effects: tuple[DeltaEstimate, ...],
    metric: Literal["completion_under_policy", "unsafe_imitation"],
) -> bool:
    if any(type(effect) is not DeltaEstimate for effect in effects):
        return False
    domains = {effect.baseline.domain for effect in effects}
    if domains != set(_SCOPES) or len(domains) != len(effects):
        return False
    return all(
        _rate_is_valid(effect.baseline)
        and _rate_is_valid(effect.comparison)
        and effect.comparison.domain == effect.baseline.domain
        and effect.baseline.condition is ExperimentCondition.B2_AUTHORITY_RESOLVER
        and effect.comparison.condition is ExperimentCondition.B3_DETERMINISTIC_GATE
        and effect.baseline.metric == metric
        and effect.comparison.metric == metric
        and effect.baseline.authority_class is None
        and effect.comparison.authority_class is None
        and effect.baseline.eligible_authority_classes == AUTHORITY_CLASSES
        and effect.comparison.eligible_authority_classes == AUTHORITY_CLASSES
        and effect.baseline.expected_authority_disposition is None
        and effect.comparison.expected_authority_disposition is None
        and all(isfinite(value) for value in (effect.delta, effect.ci_low, effect.ci_high))
        and effect.ci_low <= effect.delta <= effect.ci_high
        for effect in effects
    )


def _analysis_is_complete(
    stage_a: object,
    stage_b: object,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None = None,
) -> bool:
    if type(stage_a) is not StageAAnalysis or type(stage_b) is not StageBAnalysis:
        return False
    try:
        return _stage_a_is_complete(stage_a, scientific_freeze) and _stage_b_is_complete(
            stage_b, scientific_freeze
        )
    except (AttributeError, TypeError, ValueError):
        return False


def sealed_analysis_is_release_complete(analysis: object) -> bool:
    """Return whether an exact sealed analysis can support a release claim."""

    from shadowskillbench.analysis.sealed import SealedAnalysis

    if type(analysis) is not SealedAnalysis:
        return False
    dataset = analysis.dataset
    audit = analysis.audit
    runtime = analysis.runtime_binding
    if (
        type(dataset) is not AnalysisDataset
        or type(audit) is not ArtifactIntegrityReport
        or type(runtime) is not RuntimeBinding
        or type(analysis.stage_a) is not StageAAnalysis
        or type(analysis.stage_b) is not StageBAnalysis
        or audit.status != "PASS"
        or audit.findings
        or audit.plan_hash != runtime.plan_hash
        or audit.runtime_binding_hash != sha256_ref(runtime.projection())
        or not _analysis_is_complete(analysis.stage_a, analysis.stage_b, analysis.scientific_freeze)
        or not _exclusions_are_approved(dataset, analysis.exclusion_policy)
    ):
        return False
    stage_a_count, stage_b_count, _ = _confirmatory_counts(dataset)
    try:
        expected_stage_a, expected_stage_b = _expected_confirmatory_counts(
            analysis.scientific_freeze
        )
    except ReportInputError:
        return False
    if stage_a_count != expected_stage_a or stage_b_count != expected_stage_b:
        return False
    expected_ids = tuple(sorted(row.episode_id for row in (*dataset.rows, *dataset.exclusions)))
    return audit.audited_episode_ids == expected_ids


def report_status(evidence: ReportEvidence) -> ReportStatus:
    """Return the claim ceiling without converting incomplete evidence into a finding."""
    stage_a_count, stage_b_count, excluded = _confirmatory_counts(evidence.dataset)
    runtime = evidence.runtime_binding
    models = getattr(evidence.preregistration.core, "models", None)
    compiler = getattr(models, "skill_compiler", None)
    executor = getattr(models, "skill_executor", None)
    try:
        expected_stage_a, expected_stage_b = _expected_confirmatory_counts(
            evidence.scientific_freeze
        )
    except ReportInputError:
        return ReportStatus.HOLD
    complete = (
        evidence.preregistration.valid
        and evidence.preregistration.core is not None
        and evidence.claims.valid
        and stage_a_count
        + sum(item.stage is EpisodeStage.CONFIRMATORY_A for item in evidence.dataset.exclusions)
        == expected_stage_a
        and stage_b_count
        + sum(item.stage is EpisodeStage.CONFIRMATORY_B for item in evidence.dataset.exclusions)
        == expected_stage_b
        and _exclusions_are_approved(evidence.dataset, evidence.exclusion_policy)
        and _analysis_is_complete(evidence.stage_a, evidence.stage_b, evidence.scientific_freeze)
        and (
            evidence.protocol.custody_locator is not None
            or evidence.protocol.external_anchor_locator is not None
        )
        and evidence.protocol.anchor_receipt_sha256 is not None
        and evidence.protocol.anchor_verified is True
        and evidence.protocol.experiment_plan_sha256 is not None
        and evidence.protocol.experiment_runtime_binding_sha256 is not None
        and evidence.protocol.experiment_audit_report_sha256 is not None
        and evidence.protocol.experiment_audit_status == "PASS"
        and runtime is not None
        and runtime.plan_hash == evidence.protocol.experiment_plan_sha256
        and sha256_ref(runtime.projection()) == evidence.protocol.experiment_runtime_binding_sha256
        and runtime.freeze_manifest_hash == evidence.protocol.freeze_manifest_sha256
        and runtime.anchor_receipt_hash == evidence.protocol.anchor_receipt_sha256
        and (
            evidence.scientific_freeze is None
            or evidence.protocol.freeze_manifest_sha256 == evidence.scientific_freeze.manifest_hash
        )
        and executor is not None
        and runtime.provider
        in {
            executor.provider,
            getattr(compiler, "provider", None),
        }
        and (runtime.model, runtime.model_version) == (executor.model, executor.model_version_date)
    )
    return ReportStatus.CONFIRMATORY if complete else ReportStatus.HOLD


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _table(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...], caption: str) -> str:
    head = "".join(f'<th scope="col">{_text(header)}</th>' for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        f"<table><caption>{_text(caption)}</caption><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _condition_label(condition: object) -> str:
    return str(getattr(condition, "value", condition)).replace("_", " ")


def _figure_1(analysis: StageAAnalysis) -> str:
    points = tuple(point for point in analysis.response_curves if point.scope == "pooled")
    rows = tuple(
        (
            _condition_label(point.condition),
            f"{point.contamination_ratio:.2f}",
            _percent(point.estimate),
            _percent(point.ci_low),
            _percent(point.ci_high),
            f"{point.numerator}/{point.denominator}",
        )
        for point in points
    )
    return _table(
        ("Condition", "Contamination ratio", "CuP", "95% CI low", "95% CI high", "Raw count"),
        rows,
        "Figure 1. Pooled completion-under-policy response curves; accessible chart data.",
    )


def _figure_2(analysis: StageBAnalysis) -> str:
    rows = tuple(
        (
            point.domain,
            _condition_label(point.condition),
            " ".join(
                (
                    _percent(point.unsafe_imitation.rate),
                    f"[{_percent(point.unsafe_imitation.ci_low)}, "
                    f"{_percent(point.unsafe_imitation.ci_high)}]",
                )
            ),
            f"{point.unsafe_imitation.numerator}/{point.unsafe_imitation.denominator}",
            " ".join(
                (
                    _percent(point.false_enforcement.rate),
                    f"[{_percent(point.false_enforcement.ci_low)}, "
                    f"{_percent(point.false_enforcement.ci_high)}]",
                )
            ),
            f"{point.false_enforcement.numerator}/{point.false_enforcement.denominator}",
        )
        for point in analysis.figure_2_points
    )
    return _table(
        (
            "Domain",
            "Condition",
            "Unsafe imitation (95% CI)",
            "Unsafe numerator/denominator",
            "False enforcement (95% CI)",
            "False-enforcement numerator/denominator",
        ),
        rows,
        "Figure 2. Compliance and over-enforcement plane; accessible chart data.",
    )


def _stage_a_marginal_effects(analysis: StageAAnalysis) -> str:
    rows = tuple(
        (
            effect.effect,
            effect.scope,
            _condition_label(effect.condition),
            (
                "—"
                if effect.reference_condition is None
                else _condition_label(effect.reference_condition)
            ),
            f"{effect.contamination_ratio:.2f}",
            _percent(effect.estimate),
            f"[{_percent(effect.ci_low)}, {_percent(effect.ci_high)}]",
        )
        for effect in sorted(
            analysis.average_marginal_effects,
            key=lambda item: (
                item.effect,
                item.scope,
                item.condition.value,
                item.reference_condition.value if item.reference_condition is not None else "",
                item.contamination_ratio,
            ),
        )
    )
    return _table(
        (
            "Effect",
            "Scope",
            "Condition",
            "Reference condition",
            "Contamination ratio",
            "Estimate",
            "95% CI",
        ),
        rows,
        "Stage A average marginal effects on completion under policy.",
    )


def _stage_b_estimands(analysis: StageBAnalysis) -> str:
    rows: list[tuple[str, ...]] = []
    for rate in sorted(
        (*analysis.e6_unsafe_imitation, *analysis.e7_false_enforcement),
        key=lambda item: (item.metric, item.domain, item.condition.value),
    ):
        label = (
            "E6 unsafe imitation" if rate.metric == "unsafe_imitation" else "E7 false enforcement"
        )
        rows.append(
            (
                label,
                rate.domain,
                _condition_label(rate.condition),
                _percent(rate.rate),
                f"[{_percent(rate.ci_low)}, {_percent(rate.ci_high)}]",
                f"{rate.numerator}/{rate.denominator}",
            )
        )
    for effect in sorted(analysis.e8_authority_aware_gain, key=lambda item: item.domain):
        rows.append(
            (
                "E8 authority-aware gain",
                effect.domain,
                "B2 minus B1",
                _percent(effect.delta),
                f"[{_percent(effect.ci_low)}, {_percent(effect.ci_high)}]",
                (
                    f"{_percent(effect.baseline_macro_rate)} → "
                    f"{_percent(effect.comparison_macro_rate)}"
                ),
            )
        )
    for label, effects in (
        ("E9 CuP delta", analysis.e9_completion_under_policy_delta),
        ("E9 unsafe-imitation delta", analysis.e9_unsafe_imitation_delta),
    ):
        for effect in sorted(effects, key=lambda item: item.baseline.domain):
            rows.append(
                (
                    label,
                    effect.baseline.domain,
                    (
                        f"{_condition_label(effect.baseline.condition)} vs "
                        f"{_condition_label(effect.comparison.condition)}"
                    ),
                    _percent(effect.delta),
                    f"[{_percent(effect.ci_low)}, {_percent(effect.ci_high)}]",
                    f"{_percent(effect.baseline.rate)} → {_percent(effect.comparison.rate)}",
                )
            )
    return _table(
        ("Estimand", "Domain", "Comparison", "Estimate", "95% CI", "Bound rates"),
        tuple(rows),
        "Stage B primary estimands E6–E9 with deterministic bootstrap intervals.",
    )


def _domain_breakdown(dataset: AnalysisDataset) -> str:
    rows = []
    for domain in ("access_provisioning", "financial_adjustments"):
        values = tuple(
            row
            for row in dataset.rows
            if row.domain == domain
            and row.stage in {EpisodeStage.CONFIRMATORY_A, EpisodeStage.CONFIRMATORY_B}
        )
        rows.append(
            (
                domain,
                str(len(values)),
                _percent(sum(row.completion_under_policy for row in values) / len(values))
                if values
                else "HOLD — no included rows",
                _percent(sum(row.unsafe_imitation for row in values) / len(values))
                if values
                else "HOLD — no included rows",
            )
        )
    return _table(
        ("Domain", "Included episodes", "CuP across included episodes", "Unsafe imitation"),
        tuple(rows),
        "Domain breakdown from included, observed episodes.",
    )


def _cost(dataset: AnalysisDataset) -> str:
    rows = tuple(
        row
        for row in dataset.rows
        if row.stage in {EpisodeStage.CONFIRMATORY_A, EpisodeStage.CONFIRMATORY_B}
    )
    unknown = sum(row.total_cost_nanos is None for row in rows)
    tokens = sum(row.total_tokens for row in rows)
    known_cost = sum(row.total_cost_nanos or 0 for row in rows)
    cost = (
        "HOLD — one or more included episodes lack a bound cost receipt"
        if unknown
        else str(known_cost)
    )
    return _table(
        ("Included episodes", "Total tokens", "Total cost (nanos)", "Missing cost receipts"),
        ((str(len(rows)), str(tokens), cost, str(unknown)),),
        (
            "Cost accounting. A numeric total is withheld unless every included episode "
            "binds a cost receipt."
        ),
    )


def _trace_rows(traces: tuple[RepresentativeTrace, ...]) -> str:
    rows = []
    for trace in traces:
        selected = trace.selection.selected
        if trace.selection.status is SelectionStatus.HOLD_NO_CANDIDATE:
            rows.append((trace.stage, "HOLD — no eligible candidate", "—", "—"))
        else:
            assert selected is not None
            assert trace.trace_artifact_sha256 is not None
            assert trace.summary is not None
            rows.append(
                (
                    trace.stage,
                    selected.case_id,
                    trace.trace_artifact_sha256,
                    trace.summary,
                )
            )
    return _table(
        ("Stage", "Selected case", "Trace artifact", "Bound summary"),
        tuple(rows),
        "Representative traces selected by the preregistered selection algorithm.",
    )


def _claims(claims: ClaimsReport) -> str:
    if not claims.valid:
        return "<p>HOLD — claims ledger is invalid; no findings are rendered.</p>"
    rows = tuple(
        (claim.id, claim.status, claim.scope, claim.wording, claim.finding_disposition or "—")
        for claim in claims.claims
    )
    return _table(
        ("ID", "Status", "Scope", "Wording", "Finding disposition"),
        rows,
        "Claims ledger. Findings appear only from a validated ledger.",
    )


def _null_adverse(analysis: StageBAnalysis, claims: ClaimsReport) -> str:
    rows = [
        (
            point.domain,
            _condition_label(point.condition),
            _percent(point.unsafe_imitation.rate),
            _percent(point.false_enforcement.rate),
        )
        for point in analysis.figure_2_points
    ]
    rendered_findings = (
        ", ".join(claim.id for claim in claims.renderable_findings) if claims.valid else "HOLD"
    )
    return (
        "<p>Observed adverse measures are shown without directional interpretation. "
        f"Validated finding IDs: {_text(rendered_findings)}.</p>"
        + _table(
            ("Domain", "Condition", "Unsafe imitation", "False enforcement"),
            tuple(rows),
            "Null and adverse finding measures from Figure 2 data.",
        )
    )


def _protocol_integrity(evidence: ReportEvidence, status: ReportStatus) -> str:
    core = evidence.preregistration.core
    stage_a_count, stage_b_count, excluded = _confirmatory_counts(evidence.dataset)
    required_stage_a, required_stage_b = _expected_confirmatory_counts(evidence.scientific_freeze)
    model = "HOLD — preregistration core unavailable"
    corpus_seed = "HOLD — preregistration core unavailable"
    if core is not None:
        model = f"{core.models.skill_executor.provider}/{core.models.skill_executor.model}"
        corpus_seed = str(core.demonstrations.bundle_seeds_per_domain_ratio)
    return _table(
        ("Field", "Bound value"),
        (
            ("Report status", status.value.upper()),
            ("Protocol tag", evidence.protocol.protocol_tag),
            (
                "Custody locator",
                evidence.protocol.custody_locator
                or evidence.protocol.external_anchor_locator
                or "HOLD — absent",
            ),
            ("Custody mode", evidence.protocol.custody_mode or "HOLD — unspecified"),
            (
                "External anchor locator",
                evidence.protocol.external_anchor_locator
                or "HOLD — local custody is not externally verified",
            ),
            (
                "Anchor receipt",
                evidence.protocol.anchor_receipt_sha256 or "HOLD — absent",
            ),
            ("Custody verified", str(evidence.protocol.anchor_verified)),
            ("Freeze manifest", evidence.protocol.freeze_manifest_sha256),
            (
                "Experiment audit report",
                evidence.protocol.experiment_audit_report_sha256 or "HOLD — absent",
            ),
            (
                "Experiment plan",
                evidence.protocol.experiment_plan_sha256 or "HOLD — absent",
            ),
            (
                "Runtime binding",
                evidence.protocol.experiment_runtime_binding_sha256 or "HOLD — absent",
            ),
            (
                "Experiment audit status",
                evidence.protocol.experiment_audit_status or "HOLD — absent",
            ),
            ("Analysis dataset", evidence.analysis_dataset_sha256),
            (
                "Runtime package",
                evidence.runtime_binding.package_hash
                if evidence.runtime_binding is not None
                else "HOLD — absent",
            ),
            ("Code commit", evidence.protocol.code_commit),
            ("Executor model", model),
            ("Corpus bundle seeds per domain/ratio", corpus_seed),
            ("Prompt hashes", ", ".join(evidence.protocol.prompt_hashes)),
            ("Stage A included / required", f"{stage_a_count}/{required_stage_a}"),
            ("Stage B included / required", f"{stage_b_count}/{required_stage_b}"),
            ("Confirmatory technical exclusions", str(excluded)),
            (
                "Exclusion approval / caps",
                (
                    "approved; total cap "
                    f"{evidence.exclusion_policy.max_total_exclusions}; per-cell cap "
                    f"{evidence.exclusion_policy.max_exclusions_per_primary_cell}"
                    if evidence.exclusion_policy is not None
                    and _exclusions_are_approved(evidence.dataset, evidence.exclusion_policy)
                    else "HOLD — no approved bounded exclusion policy"
                ),
            ),
        ),
        "Protocol and evidence bindings used to determine the report claim ceiling.",
    )


def render_report(evidence: ReportEvidence) -> str:
    """Render one deterministic, self-contained HTML artifact.

    Invalid or incomplete confirmatory evidence is visibly held.  The renderer
    never promotes an analysis value to a confirmatory claim on its own.
    """
    if type(evidence) is not ReportEvidence:
        raise ReportInputError("evidence must be an exact ReportEvidence")
    status = report_status(evidence)
    prior_art = tuple(
        claim for claim in evidence.claims.claims if claim.status == "supported_prior_art"
    )
    prior_art_html = _claims(
        ClaimsReport(
            ledger_path=evidence.claims.ledger_path,
            version=evidence.claims.version,
            valid=evidence.claims.valid,
            claims=prior_art,
            violations=evidence.claims.violations,
            warnings=evidence.claims.warnings,
            renderable_findings=(),
        )
    )
    claim_ceiling = (
        (
            "Confirmatory evidence is complete; consult the validated claims ledger for "
            "bounded findings."
        )
        if status is ReportStatus.CONFIRMATORY
        else (
            "HOLD — this artifact is descriptive only because required confirmatory evidence "
            "is incomplete or invalid."
        )
    )
    limitations = "".join(f"<li>{_text(item)}</li>" for item in evidence.limitations)
    sections = {
        "status": _text(status.value.upper()),
        "claim_ceiling": _text(claim_ceiling),
        "prior_art": prior_art_html,
        "protocol_integrity": _protocol_integrity(evidence, status),
        "figure_1": _figure_1(evidence.stage_a),
        "figure_2": _figure_2(evidence.stage_b),
        "stage_a_marginal_effects": _stage_a_marginal_effects(evidence.stage_a),
        "stage_b_estimands": _stage_b_estimands(evidence.stage_b),
        "domain_breakdown": _domain_breakdown(evidence.dataset),
        "cost": _cost(evidence.dataset),
        "representative_traces": _trace_rows(evidence.representative_traces),
        "null_adverse": _null_adverse(evidence.stage_b, evidence.claims),
        "limitations": limitations,
        "claims_ledger": _claims(evidence.claims),
        "reproduce_command": _text(evidence.protocol.reproduce_command),
    }
    try:
        template = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise ReportInputError("report template is unavailable") from error
    return template.substitute(sections)


__all__ = [
    "ProtocolEvidence",
    "dataset_projection",
    "hash_analysis_dataset",
    "ReportEvidence",
    "ReportInputError",
    "ReportStatus",
    "RepresentativeTrace",
    "render_report_artifacts",
    "render_report",
    "report_status",
    "select_representative_traces",
    "write_report_artifacts",
]
