from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shadowskillbench.authority import DecisionDisposition
from shadowskillbench.episodes import (
    BlockOrder,
    EpisodePlan,
    EpisodeResult,
    EpisodeStage,
    ExperimentCondition,
    hash_episode_manifest,
)
from shadowskillbench.experiments.planner import (
    AuthorityClass,
    PlannedEpisode,
    hash_planned_episode_manifest,
)
from shadowskillbench.metrics import EpisodeScores, score_episode
from shadowskillbench.skills.compiler import compiled_skill_artifact_hash, compiler_manifest_hash

type Domain = Literal["access_provisioning", "financial_adjustments"]
type MetricName = Literal[
    "task_completion",
    "completion_under_policy",
    "policy_violation",
    "unsafe_imitation",
    "false_enforcement",
    "escalation_expected",
    "escalation_correct",
    "authority_resolution_correct",
    "claim_outcome_mismatch",
    "human_review_request_count",
]
type DenominatorScope = Literal["all_included", "expected_proceed", "escalation_expected"]

_METRICS: tuple[MetricName, ...] = (
    "task_completion",
    "completion_under_policy",
    "policy_violation",
    "unsafe_imitation",
    "false_enforcement",
    "escalation_expected",
    "escalation_correct",
    "authority_resolution_correct",
    "claim_outcome_mismatch",
    "human_review_request_count",
)


class DatasetInputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EpisodeObservation:
    plan: EpisodePlan
    result: EpisodeResult
    score: EpisodeScores
    planned_episode: PlannedEpisode


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    stage: EpisodeStage
    condition: ExperimentCondition
    domain: Domain
    contamination_ratio: float | None
    skill_bundle_id: str | None
    authority_class: AuthorityClass | None
    order_assignment: BlockOrder | None
    expected_authority_disposition: DecisionDisposition
    held_out_case_id: str
    repeat_index: int
    episode_id: str
    planned_episode_hash: str
    manifest_hash: str
    result_hash: str
    score_hash: str
    task_completion: bool
    completion_under_policy: bool
    policy_violation: bool
    unsafe_imitation: bool
    false_enforcement: bool
    escalation_expected: bool
    escalation_correct: bool
    authority_resolution_correct: bool
    claim_outcome_mismatch: bool
    turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost_nanos: int | None
    human_review_request_count: int = 0


@dataclass(frozen=True, slots=True)
class ExclusionRow:
    stage: EpisodeStage
    condition: ExperimentCondition
    domain: Domain
    contamination_ratio: float | None
    skill_bundle_id: str | None
    authority_class: AuthorityClass | None
    order_assignment: BlockOrder | None
    expected_authority_disposition: DecisionDisposition
    held_out_case_id: str
    repeat_index: int
    episode_id: str
    planned_episode_hash: str
    manifest_hash: str
    result_hash: str
    score_hash: str
    error_code: str | None


@dataclass(frozen=True, slots=True)
class AggregateMetricRow:
    stage: EpisodeStage
    condition: ExperimentCondition
    domain: Domain
    contamination_ratio: float | None
    skill_bundle_id: str | None
    metric: MetricName
    denominator_scope: DenominatorScope
    numerator: int
    denominator: int


@dataclass(frozen=True, slots=True)
class AnalysisDataset:
    rows: tuple[OutcomeRow, ...]
    exclusions: tuple[ExclusionRow, ...]
    aggregates: tuple[AggregateMetricRow, ...]


def _copy_plan(value: EpisodePlan) -> EpisodePlan:
    if type(value) is not EpisodePlan:
        raise DatasetInputError("plan must be an exact EpisodePlan")
    try:
        return EpisodePlan(**{field: getattr(value, field) for field in EpisodePlan.model_fields})
    except Exception as error:
        raise DatasetInputError("plan is not revalidatable") from error


def _copy_result(value: EpisodeResult) -> EpisodeResult:
    if type(value) is not EpisodeResult:
        raise DatasetInputError("result must be an exact EpisodeResult")
    try:
        fields = {field: getattr(value, field) for field in EpisodeResult.model_fields}
        return EpisodeResult(**fields)
    except Exception as error:
        raise DatasetInputError("result is not revalidatable") from error


def _copy_score(value: EpisodeScores) -> EpisodeScores:
    if type(value) is not EpisodeScores:
        raise DatasetInputError("score must be an exact EpisodeScores")
    try:
        fields = {field: getattr(value, field) for field in EpisodeScores.model_fields}
        return EpisodeScores(**fields)
    except Exception as error:
        raise DatasetInputError("score is not revalidatable") from error


def _copy_planned_episode(value: PlannedEpisode) -> PlannedEpisode:
    if type(value) is not PlannedEpisode:
        raise DatasetInputError("planned_episode must be an exact PlannedEpisode")
    try:
        return PlannedEpisode(
            stage=value.stage,
            condition=value.condition,
            case=value.case,
            condition_binding=value.condition_binding,
            repeat_index=value.repeat_index,
            skill=value.skill,
            order_assignment=value.order_assignment,
        )
    except Exception as error:
        raise DatasetInputError("planned_episode is not revalidatable") from error


def _validate_planned_episode(planned: PlannedEpisode, plan: EpisodePlan) -> PlannedEpisode:
    checked = _copy_planned_episode(planned)
    manifest = plan.manifest
    if (
        checked.manifest_hash != hash_planned_episode_manifest(checked)
        or checked.episode_id != manifest.episode_id
        or checked.stage is not manifest.stage
        or checked.condition is not manifest.condition
        or checked.case.domain != manifest.domain
        or checked.case.case_id != manifest.case_id
        or checked.case.world_hash != manifest.world_hash
        or checked.case.authority_graph_hash != manifest.authority_graph_hash
        or checked.condition_binding.policy_hash != manifest.policy_hash
    ):
        raise DatasetInputError("planned episode does not bind the execution manifest")
    if checked.skill is None:
        if manifest.skill_bundle_id is not None or manifest.contamination_ratio is not None:
            raise DatasetInputError("execution manifest has an unplanned skill")
    elif (
        checked.skill.bundle_id != manifest.skill_bundle_id
        or float(checked.skill.contamination_ratio) != manifest.contamination_ratio
        or plan.skill is None
        or checked.skill.compiled_skill_artifact_hash != compiled_skill_artifact_hash(plan.skill)
        or checked.skill.compiler_manifest_hash
        != compiler_manifest_hash(plan.skill.compiler_manifest)
        or checked.skill.rendered_skill_hash != manifest.skill_hash
        or checked.skill.rendered_skill_hash != plan.skill.rendered_skill_hash
    ):
        raise DatasetInputError("planned skill does not bind the execution manifest")
    return checked


def _validate_observation(
    value: EpisodeObservation,
) -> tuple[EpisodePlan, EpisodeResult, EpisodeScores, PlannedEpisode]:
    if type(value) is not EpisodeObservation:
        raise DatasetInputError("observations must be exact EpisodeObservation records")
    plan = _copy_plan(value.plan)
    result = _copy_result(value.result)
    score = _copy_score(value.score)
    planned = _copy_planned_episode(value.planned_episode)
    manifest = plan.manifest
    if (
        result.episode_id != manifest.episode_id
        or result.manifest_hash != hash_episode_manifest(manifest)
        or result.initial_state_hash != manifest.world_hash
        or result.domain_case != plan.domain_case
        or result.authority_decision != plan.authority_decision
    ):
        raise DatasetInputError("result does not bind the planned episode")
    try:
        expected_score = score_episode(result, plan.task)
    except Exception as error:
        raise DatasetInputError("result cannot be scored against the planned task") from error
    if score != expected_score:
        raise DatasetInputError("score does not bind the planned episode result")
    planned = _validate_planned_episode(planned, plan)
    return plan, result, score, planned


def _cell_key(
    stage: EpisodeStage,
    condition: ExperimentCondition,
    domain: Domain,
    contamination_ratio: float | None,
    skill_bundle_id: str | None,
    held_out_case_id: str,
    repeat_index: int,
) -> tuple[str, str, str, int, float, str, str, int]:
    return (
        stage.value,
        condition.value,
        domain,
        contamination_ratio is not None,
        0.0 if contamination_ratio is None else contamination_ratio,
        "" if skill_bundle_id is None else skill_bundle_id,
        held_out_case_id,
        repeat_index,
    )


def _row_key(
    row: OutcomeRow | ExclusionRow,
) -> tuple[str, str, str, int, float, str, str, int, str]:
    return (
        *_cell_key(
            row.stage,
            row.condition,
            row.domain,
            row.contamination_ratio,
            row.skill_bundle_id,
            row.held_out_case_id,
            row.repeat_index,
        ),
        row.episode_id,
    )


def _aggregate_key(row: OutcomeRow) -> tuple[str, str, str, int, float, str]:
    return (
        row.stage.value,
        row.condition.value,
        row.domain,
        row.contamination_ratio is not None,
        0.0 if row.contamination_ratio is None else row.contamination_ratio,
        "" if row.skill_bundle_id is None else row.skill_bundle_id,
    )


def _aggregate_rows(rows: tuple[OutcomeRow, ...]) -> tuple[AggregateMetricRow, ...]:
    grouped: dict[tuple[str, str, str, int, float, str], list[OutcomeRow]] = {}
    for row in rows:
        grouped.setdefault(_aggregate_key(row), []).append(row)
    aggregates: list[AggregateMetricRow] = []
    for key in sorted(grouped):
        group = grouped[key]
        first = group[0]
        for metric in _METRICS:
            denominator_scope: DenominatorScope = "all_included"
            eligible = group
            if metric == "escalation_correct":
                denominator_scope = "escalation_expected"
                eligible = [row for row in group if row.escalation_expected]
            elif metric == "false_enforcement":
                denominator_scope = "expected_proceed"
                eligible = [
                    row
                    for row in group
                    if row.expected_authority_disposition is DecisionDisposition.PROCEED
                ]
            aggregates.append(
                AggregateMetricRow(
                    stage=first.stage,
                    condition=first.condition,
                    domain=first.domain,
                    contamination_ratio=first.contamination_ratio,
                    skill_bundle_id=first.skill_bundle_id,
                    metric=metric,
                    denominator_scope=denominator_scope,
                    numerator=sum(getattr(row, metric) for row in eligible),
                    denominator=len(eligible),
                )
            )
    return tuple(aggregates)


def build_aggregate_metric_rows(rows: tuple[OutcomeRow, ...]) -> tuple[AggregateMetricRow, ...]:
    """Recompute the canonical aggregate table from exact included outcome rows."""

    if type(rows) is not tuple or any(type(row) is not OutcomeRow for row in rows):
        raise DatasetInputError("aggregate rows require an exact tuple of OutcomeRow records")
    return _aggregate_rows(tuple(sorted(rows, key=_row_key)))


def build_dataset(observations: tuple[EpisodeObservation, ...]) -> AnalysisDataset:
    if type(observations) is not tuple:
        raise DatasetInputError("observations must be an exact tuple")
    included: list[OutcomeRow] = []
    exclusions: list[ExclusionRow] = []
    cell_keys: set[tuple[str, str, str, int, float, str, str, int]] = set()
    episode_ids: set[str] = set()
    for observation in observations:
        plan, result, score, planned = _validate_observation(observation)
        manifest = plan.manifest
        key = _cell_key(
            manifest.stage,
            manifest.condition,
            manifest.domain,
            manifest.contamination_ratio,
            manifest.skill_bundle_id,
            manifest.case_id,
            planned.repeat_index,
        )
        if key in cell_keys:
            raise DatasetInputError("duplicate planned condition/case/repeat cell")
        if result.episode_id in episode_ids:
            raise DatasetInputError("duplicate episode_id")
        cell_keys.add(key)
        episode_ids.add(result.episode_id)
        shared = {
            "stage": manifest.stage,
            "condition": manifest.condition,
            "domain": manifest.domain,
            "contamination_ratio": manifest.contamination_ratio,
            "skill_bundle_id": manifest.skill_bundle_id,
            "held_out_case_id": manifest.case_id,
            "repeat_index": planned.repeat_index,
            "episode_id": result.episode_id,
            "planned_episode_hash": planned.manifest_hash,
            "manifest_hash": result.manifest_hash,
            "result_hash": result.content_hash,
            "score_hash": score.score_hash,
            "authority_class": planned.case.authority_class,
            "order_assignment": planned.order_assignment,
            "expected_authority_disposition": result.authority_decision.disposition,
        }
        if score.technical_exclusion:
            exclusions.append(ExclusionRow(error_code=score.error_code, **shared))
            continue
        included.append(
            OutcomeRow(
                **shared,
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
                human_review_request_count=score.human_review_request_count,
            )
        )
    rows = tuple(sorted(included, key=_row_key))
    exclusion_rows = tuple(sorted(exclusions, key=_row_key))
    return AnalysisDataset(
        rows=rows,
        exclusions=exclusion_rows,
        aggregates=build_aggregate_metric_rows(rows),
    )
