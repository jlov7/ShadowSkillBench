from __future__ import annotations

from dataclasses import fields, replace
from math import isfinite

import pytest

from shadowskillbench.analysis.dataset import AnalysisDataset, ExclusionRow, OutcomeRow
from shadowskillbench.analysis.exclusions import ApprovedTechnicalExclusionPolicy
from shadowskillbench.analysis.profile import NONCONFIRMATORY_ANALYSIS_PROFILE
from shadowskillbench.analysis.stage_a import StageAAnalysisError, analyze_stage_a
from shadowskillbench.authority import DecisionDisposition
from shadowskillbench.episodes import BlockOrder, EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.planner import DOMAINS, RATIOS, REPEATS
from shadowskillbench.reporting.report import _stage_a_is_complete


def _row(
    *,
    condition: ExperimentCondition,
    domain: str,
    case_index: int,
    repeat_index: int,
    complete: bool,
    ratio: float | None = None,
    bundle_id: str | None = None,
) -> OutcomeRow:
    identifier = "-".join(
        (
            condition.value,
            domain,
            str(ratio),
            str(bundle_id),
            str(case_index),
            str(repeat_index),
        )
    )
    return OutcomeRow(
        stage=EpisodeStage.CONFIRMATORY_A,
        condition=condition,
        domain=domain,  # type: ignore[arg-type]
        contamination_ratio=ratio,
        skill_bundle_id=bundle_id,
        authority_class=None,
        order_assignment=(
            (
                BlockOrder.POLICY_THEN_SKILL
                if (case_index * REPEATS + repeat_index - 1) % 2 == 0
                else BlockOrder.SKILL_THEN_POLICY
            )
            if condition is ExperimentCondition.A3_SKILL_POLICY_SAME_TIER
            else None
        ),
        expected_authority_disposition=DecisionDisposition.PROCEED,
        held_out_case_id=f"{domain}-case-{case_index:02d}",
        repeat_index=repeat_index,
        episode_id=identifier,
        planned_episode_hash=f"planned-{identifier}",
        manifest_hash=f"manifest-{identifier}",
        result_hash=f"result-{identifier}",
        score_hash=f"score-{identifier}",
        task_completion=complete,
        completion_under_policy=complete,
        policy_violation=not complete,
        unsafe_imitation=False,
        false_enforcement=False,
        escalation_expected=False,
        escalation_correct=False,
        authority_resolution_correct=True,
        claim_outcome_mismatch=False,
        turns=1,
        tool_calls=1,
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        total_cost_nanos=1,
    )


def _synthetic_dataset() -> AnalysisDataset:
    success_per_sixty = {
        ExperimentCondition.A2_SKILL_ONLY: (60, 48, 36, 24, 12),
        ExperimentCondition.A3_SKILL_POLICY_SAME_TIER: (60, 54, 48, 42, 36),
        ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER: (60, 57, 54, 51, 48),
        ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER: (54, 48, 42, 36, 30),
    }
    rows: list[OutcomeRow] = []
    for domain in DOMAINS:
        for condition in (ExperimentCondition.A0_BARE, ExperimentCondition.A1_POLICY_ONLY_SYSTEM):
            for case_index in range(20):
                for repeat_index in range(1, REPEATS + 1):
                    rows.append(
                        _row(
                            condition=condition,
                            domain=domain,
                            case_index=case_index,
                            repeat_index=repeat_index,
                            complete=True,
                        )
                    )
        for ratio_index, ratio in enumerate(RATIOS):
            for bundle_index in range(3):
                bundle_id = f"{domain}-r{ratio_index}-b{bundle_index}"
                for condition, successes in success_per_sixty.items():
                    for case_index in range(20):
                        for repeat_index in range(1, REPEATS + 1):
                            within_bundle = case_index * REPEATS + repeat_index - 1
                            rows.append(
                                _row(
                                    condition=condition,
                                    domain=domain,
                                    case_index=case_index,
                                    repeat_index=repeat_index,
                                    complete=within_bundle < successes[ratio_index],
                                    ratio=float(ratio),
                                    bundle_id=bundle_id,
                                )
                            )
    return AnalysisDataset(rows=tuple(rows), exclusions=(), aggregates=())


def _estimate_by_name(result: object) -> dict[tuple[str, str], float]:
    estimands = getattr(result, "estimands")
    return {
        (item.estimand, item.component): item.estimate
        for item in estimands
        if item.scope == "pooled"
    }


def test_synthetic_known_effects_and_intervals_are_deterministic() -> None:
    dataset = _synthetic_dataset()
    first = analyze_stage_a(dataset, bootstrap_replicates=200, bootstrap_seed=71)
    second = analyze_stage_a(dataset, bootstrap_replicates=200, bootstrap_seed=71)
    estimates = _estimate_by_name(first)

    assert first == second
    assert first.inference.classification == "NONCONFIRMATORY"
    assert first.inference.profile == NONCONFIRMATORY_ANALYSIS_PROFILE
    assert not _stage_a_is_complete(first)
    assert len(first.raw_counts) == 66
    assert len(first.response_curves) == 75
    assert first.average_marginal_effects
    assert estimates == pytest.approx(
        {
            ("E1", "slope"): -0.8,
            ("E2", "slope"): 0.4,
            ("E3", "slope"): 0.2,
            ("E3", "average_cup_contrast"): 0.1,
            ("E4", "average_cup_contrast"): -0.1,
            ("E5", "average_cup_contrast"): -0.1,
        }
    )
    for estimate in first.estimands:
        assert estimate.ci_low <= estimate.estimate <= estimate.ci_high
    skill_points = [
        point
        for point in first.response_curves
        if point.condition is ExperimentCondition.A2_SKILL_ONLY
    ]
    assert any(point.estimate != point.numerator / point.denominator for point in skill_points)
    assert all(0.0 <= point.ci_low <= point.ci_high <= 1.0 for point in first.response_curves)
    for effect in first.average_marginal_effects:
        assert isfinite(effect.estimate)
        assert isfinite(effect.ci_low)
        assert isfinite(effect.ci_high)
        assert effect.ci_low <= effect.ci_high
    assert {
        (effect.effect, effect.condition, effect.reference_condition)
        for effect in first.average_marginal_effects
    } >= {
        ("contamination_ratio", ExperimentCondition.A2_SKILL_ONLY, None),
        (
            "condition_contrast",
            ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
            ExperimentCondition.A2_SKILL_ONLY,
        ),
    }
    assert first.model == second.model
    assert first.model.formula == "CuP ~ contamination_ratio * condition + domain"
    assert first.model.family == "Binomial"
    assert first.model.link == "logit"
    assert first.model.converged
    assert first.model.observation_count == 7200
    assert first.model.skill_bundle_cluster_count == 30
    assert first.model.held_out_case_cluster_count == 40
    assert first.model.coefficients[first.model.parameter_names.index("contamination_ratio")] < 0
    assert len(first.model.parameter_names) == 9
    assert len(first.model.covariance) == len(first.model.parameter_names)
    assert all(len(row) == len(first.model.parameter_names) for row in first.model.covariance)
    assert "Binomial logistic GLM" in first.inference.method


def test_report_rejects_stage_a_estimates_outside_their_intervals() -> None:
    analysis = analyze_stage_a(_synthetic_dataset())

    assert _stage_a_is_complete(analysis)
    assert not _stage_a_is_complete(
        replace(analysis, inference=replace(analysis.inference, bootstrap_seed=1))
    )
    assert not _stage_a_is_complete(
        replace(analysis, inference=replace(analysis.inference, bootstrap_replicates=100))
    )
    assert not _stage_a_is_complete(
        replace(analysis, inference=replace(analysis.inference, method="drift"))
    )
    assert not _stage_a_is_complete(
        replace(analysis, inference=replace(analysis.inference, profile="SSB-DRIFT-1"))
    )
    assert not _stage_a_is_complete(
        replace(analysis, inference=replace(analysis.inference, profile_hash="sha256:" + "0" * 64))
    )
    curve = analysis.response_curves[0]
    assert not _stage_a_is_complete(
        replace(
            analysis,
            response_curves=(
                replace(curve, estimate=curve.ci_high + 0.01),
                *analysis.response_curves[1:],
            ),
        )
    )
    estimand = analysis.estimands[0]
    assert not _stage_a_is_complete(
        replace(
            analysis,
            estimands=(
                replace(estimand, estimate=estimand.ci_high + 0.01),
                *analysis.estimands[1:],
            ),
        )
    )
    effect = analysis.average_marginal_effects[0]
    assert not _stage_a_is_complete(
        replace(
            analysis,
            average_marginal_effects=(
                replace(effect, estimate=effect.ci_high + 0.01),
                *analysis.average_marginal_effects[1:],
            ),
        )
    )


def test_rejects_a3_block_order_balance_drift() -> None:
    source = _synthetic_dataset()
    rows = list(source.rows)
    index = next(
        index
        for index, row in enumerate(rows)
        if row.condition is ExperimentCondition.A3_SKILL_POLICY_SAME_TIER
        and row.order_assignment is BlockOrder.SKILL_THEN_POLICY
    )
    rows[index] = replace(rows[index], order_assignment=BlockOrder.POLICY_THEN_SKILL)
    drifted = AnalysisDataset(rows=tuple(rows), exclusions=(), aggregates=())

    with pytest.raises(StageAAnalysisError, match="30/30"):
        analyze_stage_a(drifted, bootstrap_replicates=100)


def _technical_exclusion(row: OutcomeRow) -> ExclusionRow:
    values = {
        field.name: getattr(row, field.name)
        for field in fields(ExclusionRow)
        if field.name != "error_code"
    }
    values["error_code"] = "MODEL_PROVIDER_TERMINAL"
    return ExclusionRow(**values)


def test_allows_only_capped_preapproved_technical_exclusions() -> None:
    source = _synthetic_dataset()
    first = next(
        row
        for row in source.rows
        if row.condition is ExperimentCondition.A2_SKILL_ONLY
        and row.domain == "access_provisioning"
        and row.repeat_index == 1
    )
    policy = ApprovedTechnicalExclusionPolicy(
        allowed_error_codes=("MODEL_PROVIDER_TERMINAL",),
        max_total_exclusions=1,
        max_exclusions_per_primary_cell=1,
    )
    excluded = AnalysisDataset(
        rows=tuple(row for row in source.rows if row is not first),
        exclusions=(_technical_exclusion(first),),
        aggregates=(),
    )

    analysis = analyze_stage_a(
        excluded,
        bootstrap_replicates=100,
        exclusion_policy=policy,
    )

    assert any(
        count.condition is first.condition
        and count.scope == first.domain
        and count.contamination_ratio == first.contamination_ratio
        and count.denominator == 179
        for count in analysis.raw_counts
    )
    with pytest.raises(StageAAnalysisError, match="preapproved"):
        analyze_stage_a(excluded, bootstrap_replicates=100)

    second = next(
        row
        for row in source.rows
        if row.condition is first.condition
        and row.domain == first.domain
        and row.contamination_ratio == first.contamination_ratio
        and row.skill_bundle_id == first.skill_bundle_id
        and row.held_out_case_id == first.held_out_case_id
        and row.repeat_index == 2
    )
    underpowered = AnalysisDataset(
        rows=tuple(row for row in source.rows if row not in {first, second}),
        exclusions=(_technical_exclusion(first), _technical_exclusion(second)),
        aggregates=(),
    )
    with pytest.raises(StageAAnalysisError, match="two included repeats"):
        analyze_stage_a(
            underpowered,
            bootstrap_replicates=100,
            exclusion_policy=ApprovedTechnicalExclusionPolicy(
                allowed_error_codes=("MODEL_PROVIDER_TERMINAL",),
                max_total_exclusions=2,
                max_exclusions_per_primary_cell=2,
            ),
        )
