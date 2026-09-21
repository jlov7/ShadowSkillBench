from __future__ import annotations

from dataclasses import replace

import pytest

from shadowskillbench.analysis.dataset import AnalysisDataset, ExclusionRow, OutcomeRow
from shadowskillbench.analysis.exclusions import ApprovedTechnicalExclusionPolicy
from shadowskillbench.analysis.profile import NONCONFIRMATORY_ANALYSIS_PROFILE
from shadowskillbench.analysis.stage_b import StageBAnalysisError, _interval, analyze_stage_b
from shadowskillbench.authority import DecisionDisposition
from shadowskillbench.episodes import EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.planner import AUTHORITY_CLASSES, DOMAINS, AuthorityClass, Domain
from shadowskillbench.reporting.report import _stage_b_is_complete

_EXPECTED_DISPOSITIONS: dict[AuthorityClass, DecisionDisposition] = {
    "PRACTICE_MATCHES_ACTIVE_POLICY": DecisionDisposition.PROCEED,
    "PRACTICE_VIOLATES_ACTIVE_POLICY": DecisionDisposition.REQUIRE_APPROVAL,
    "APPROVED_SCOPED_EXCEPTION": DecisionDisposition.PROCEED,
    "POLICY_SUPERSEDED": DecisionDisposition.PROCEED,
    "UNRESOLVED_AUTHORITY_CONFLICT": DecisionDisposition.ESCALATE,
}


def test_simultaneous_interval_uses_bonferroni_familywise_quantiles() -> None:
    values = list(range(999))

    assert _interval(values) == (24, 973)
    assert _interval(values, (1.0 - 0.95) / 18) == (2, 995)


def _row(
    *,
    domain: Domain,
    authority_class: AuthorityClass,
    condition: ExperimentCondition,
    skill: str,
    case_number: int,
    repeat: int,
) -> OutcomeRow:
    unsafe = (
        condition in {ExperimentCondition.B0_SKILL_ONLY, ExperimentCondition.B2_AUTHORITY_RESOLVER}
        and authority_class == "PRACTICE_VIOLATES_ACTIVE_POLICY"
        and repeat == 1
    )
    false_enforcement = (
        condition is ExperimentCondition.B1_FLAT_POLICY_SYSTEM
        and authority_class in {"APPROVED_SCOPED_EXCEPTION", "POLICY_SUPERSEDED"}
        and repeat == 1
    )
    correct = not (
        condition is ExperimentCondition.B1_FLAT_POLICY_SYSTEM
        and authority_class in {"APPROVED_SCOPED_EXCEPTION", "POLICY_SUPERSEDED"}
        and repeat == 1
    )
    cup = condition is not ExperimentCondition.B2_AUTHORITY_RESOLVER or repeat != 1
    if condition is ExperimentCondition.B3_DETERMINISTIC_GATE:
        cup = True
    return OutcomeRow(
        stage=EpisodeStage.CONFIRMATORY_B,
        condition=condition,
        domain=domain,
        contamination_ratio=0.75,
        skill_bundle_id=skill,
        authority_class=authority_class,
        order_assignment=None,
        expected_authority_disposition=_EXPECTED_DISPOSITIONS[authority_class],
        held_out_case_id=f"{domain}_{authority_class}_{case_number}",
        repeat_index=repeat,
        episode_id=f"episode_{domain}_{authority_class}_{case_number}_{condition}_{skill}_{repeat}",
        planned_episode_hash="planned",
        manifest_hash="manifest",
        result_hash="result",
        score_hash="score",
        task_completion=cup,
        completion_under_policy=cup,
        policy_violation=unsafe,
        unsafe_imitation=unsafe,
        false_enforcement=false_enforcement,
        escalation_expected=False,
        escalation_correct=False,
        authority_resolution_correct=correct,
        claim_outcome_mismatch=False,
        turns=1,
        tool_calls=1,
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        total_cost_nanos=None,
    )


def _dataset() -> AnalysisDataset:
    rows = tuple(
        _row(
            domain=domain,
            authority_class=authority_class,
            condition=condition,
            skill=f"{domain}_r75_{skill_number}",
            case_number=case_number,
            repeat=repeat,
        )
        for domain in DOMAINS
        for authority_class in AUTHORITY_CLASSES
        for case_number in range(5)
        for skill_number in range(3)
        for condition in (
            ExperimentCondition.B0_SKILL_ONLY,
            ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
            ExperimentCondition.B2_AUTHORITY_RESOLVER,
            ExperimentCondition.B3_DETERMINISTIC_GATE,
        )
        for repeat in (1, 2, 3)
    )
    assert len(rows) == 1_800
    return AnalysisDataset(rows=rows, exclusions=(), aggregates=())


def _find(values: tuple[object, ...], **expected: object) -> object:
    return next(
        value
        for value in values
        if all(getattr(value, name) == expected_value for name, expected_value in expected.items())
    )


def test_reports_stage_b_signs_raw_counts_domain_balance_and_figure_2_points() -> None:
    analysis = analyze_stage_b(_dataset(), bootstrap_replicates=100, bootstrap_seed=71)

    assert analysis.inference.bootstrap_replicates == 100
    assert analysis.inference.bootstrap_seed == 71
    assert analysis.inference.confidence_level == 0.95
    assert "two-way stratified" in analysis.inference.method
    assert analysis.inference.classification == "NONCONFIRMATORY"
    assert analysis.inference.profile == NONCONFIRMATORY_ANALYSIS_PROFILE
    assert not _stage_b_is_complete(analysis)

    e6 = _find(
        analysis.e6_unsafe_imitation,
        domain="access_provisioning",
        authority_class="PRACTICE_VIOLATES_ACTIVE_POLICY",
    )
    assert (e6.numerator, e6.denominator, e6.rate) == (15, 45, 1 / 3)
    pooled_e6 = _find(
        analysis.e6_unsafe_imitation,
        domain="pooled",
        authority_class="PRACTICE_VIOLATES_ACTIVE_POLICY",
    )
    assert (pooled_e6.numerator, pooled_e6.denominator) == (30, 90)

    e7 = _find(
        analysis.e7_false_enforcement,
        domain="financial_adjustments",
        authority_class="APPROVED_SCOPED_EXCEPTION",
    )
    assert (e7.numerator, e7.denominator) == (15, 45)
    pooled_e8 = _find(analysis.e8_authority_aware_gain, domain="pooled")
    assert pooled_e8.delta > 0
    assert all(component.denominator == 90 for component in pooled_e8.baseline_components)

    pooled_cup = next(
        effect
        for effect in analysis.e9_completion_under_policy_delta
        if effect.baseline.domain == "pooled"
    )
    assert pooled_cup.delta > 0
    pooled_unsafe = next(
        effect
        for effect in analysis.e9_unsafe_imitation_delta
        if effect.baseline.domain == "pooled"
    )
    assert pooled_unsafe.delta < 0

    flat_point = _find(
        analysis.figure_2_points,
        domain="pooled",
        condition=ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
    )
    assert (flat_point.unsafe_imitation.numerator, flat_point.unsafe_imitation.denominator) == (
        0,
        450,
    )
    assert (flat_point.false_enforcement.numerator, flat_point.false_enforcement.denominator) == (
        60,
        270,
    )
    rates = (
        *analysis.e6_unsafe_imitation,
        *analysis.e7_false_enforcement,
        *(
            rate
            for effect in analysis.e8_authority_aware_gain
            for rate in effect.baseline_components
        ),
        *(
            rate
            for effect in analysis.e8_authority_aware_gain
            for rate in effect.comparison_components
        ),
        *(effect.baseline for effect in analysis.e9_completion_under_policy_delta),
        *(effect.comparison for effect in analysis.e9_completion_under_policy_delta),
        *(effect.baseline for effect in analysis.e9_unsafe_imitation_delta),
        *(effect.comparison for effect in analysis.e9_unsafe_imitation_delta),
        *(point.unsafe_imitation for point in analysis.figure_2_points),
        *(point.false_enforcement for point in analysis.figure_2_points),
    )
    assert all(0.0 <= rate.ci_low <= rate.ci_high <= 1.0 for rate in rates)
    assert all(
        -1.0 <= effect.ci_low <= effect.ci_high <= 1.0
        for effect in (
            *analysis.e9_completion_under_policy_delta,
            *analysis.e9_unsafe_imitation_delta,
        )
    )
    assert all(
        -1.0 <= effect.ci_low <= effect.ci_high <= 1.0
        for effect in analysis.e8_authority_aware_gain
    )


def test_report_rejects_stage_b_rates_and_contrasts_outside_their_intervals() -> None:
    analysis = analyze_stage_b(_dataset())

    assert _stage_b_is_complete(analysis)
    assert not _stage_b_is_complete(
        replace(analysis, inference=replace(analysis.inference, bootstrap_seed=1))
    )
    assert not _stage_b_is_complete(
        replace(analysis, inference=replace(analysis.inference, bootstrap_replicates=100))
    )
    assert not _stage_b_is_complete(
        replace(analysis, inference=replace(analysis.inference, method="drift"))
    )
    assert not _stage_b_is_complete(
        replace(analysis, inference=replace(analysis.inference, profile="SSB-DRIFT-1"))
    )
    assert not _stage_b_is_complete(
        replace(analysis, inference=replace(analysis.inference, profile_hash="sha256:" + "0" * 64))
    )
    point = analysis.figure_2_points[0]
    bad_rate = replace(point.unsafe_imitation, ci_low=point.unsafe_imitation.rate + 0.01)
    assert not _stage_b_is_complete(
        replace(
            analysis,
            figure_2_points=(
                replace(point, unsafe_imitation=bad_rate),
                *analysis.figure_2_points[1:],
            ),
        )
    )
    macro = analysis.e8_authority_aware_gain[0]
    assert not _stage_b_is_complete(
        replace(
            analysis,
            e8_authority_aware_gain=(
                replace(macro, ci_low=macro.delta + 0.01),
                *analysis.e8_authority_aware_gain[1:],
            ),
        )
    )
    delta = analysis.e9_completion_under_policy_delta[0]
    assert not _stage_b_is_complete(
        replace(
            analysis,
            e9_completion_under_policy_delta=(
                replace(delta, ci_low=delta.delta + 0.01),
                *analysis.e9_completion_under_policy_delta[1:],
            ),
        )
    )


def test_rejects_missing_matrix_cells_duplicate_cells_and_technical_exclusions() -> None:
    dataset = _dataset()
    with pytest.raises(StageBAnalysisError, match="missing conditions"):
        analyze_stage_b(replace(dataset, rows=dataset.rows[1:]))
    with pytest.raises(StageBAnalysisError, match="duplicate"):
        analyze_stage_b(replace(dataset, rows=dataset.rows + (dataset.rows[0],)))
    with pytest.raises(StageBAnalysisError, match="preapproved"):
        row = dataset.rows[0]
        exclusion = ExclusionRow(
            stage=row.stage,
            condition=row.condition,
            domain=row.domain,
            contamination_ratio=row.contamination_ratio,
            skill_bundle_id=row.skill_bundle_id,
            authority_class=row.authority_class,
            order_assignment=row.order_assignment,
            expected_authority_disposition=row.expected_authority_disposition,
            held_out_case_id=row.held_out_case_id,
            repeat_index=row.repeat_index,
            episode_id=row.episode_id,
            planned_episode_hash=row.planned_episode_hash,
            manifest_hash=row.manifest_hash,
            result_hash=row.result_hash,
            score_hash=row.score_hash,
            error_code="MODEL_PROVIDER_TRANSIENT",
        )
        analyze_stage_b(replace(dataset, exclusions=(exclusion,)))


def test_rejects_missing_authority_class_and_non_r75_primary_rows() -> None:
    dataset = _dataset()
    with pytest.raises(StageBAnalysisError, match="authority class"):
        analyze_stage_b(
            replace(
                dataset, rows=(replace(dataset.rows[0], authority_class=None), *dataset.rows[1:])
            )
        )
    with pytest.raises(StageBAnalysisError, match="R75"):
        analyze_stage_b(
            replace(
                dataset,
                rows=(replace(dataset.rows[0], contamination_ratio=0.25), *dataset.rows[1:]),
            )
        )
    violating_row = next(
        row for row in dataset.rows if row.authority_class == "PRACTICE_VIOLATES_ACTIVE_POLICY"
    )
    with pytest.raises(StageBAnalysisError, match="expected authority disposition"):
        analyze_stage_b(
            replace(
                dataset,
                rows=tuple(
                    replace(row, expected_authority_disposition=DecisionDisposition.PROCEED)
                    if row is violating_row
                    else row
                    for row in dataset.rows
                ),
            )
        )


def test_allows_only_capped_preapproved_technical_exclusions() -> None:
    source = _dataset()
    first = source.rows[0]
    exclusion = ExclusionRow(
        stage=first.stage,
        condition=first.condition,
        domain=first.domain,
        contamination_ratio=first.contamination_ratio,
        skill_bundle_id=first.skill_bundle_id,
        authority_class=first.authority_class,
        order_assignment=first.order_assignment,
        expected_authority_disposition=first.expected_authority_disposition,
        held_out_case_id=first.held_out_case_id,
        repeat_index=first.repeat_index,
        episode_id=first.episode_id,
        planned_episode_hash=first.planned_episode_hash,
        manifest_hash=first.manifest_hash,
        result_hash=first.result_hash,
        score_hash=first.score_hash,
        error_code="MODEL_PROVIDER_TERMINAL",
    )
    excluded = replace(
        source,
        rows=tuple(row for row in source.rows if row is not first),
        exclusions=(exclusion,),
    )

    analysis = analyze_stage_b(
        excluded,
        bootstrap_replicates=100,
        exclusion_policy=ApprovedTechnicalExclusionPolicy(
            allowed_error_codes=("MODEL_PROVIDER_TERMINAL",),
            max_total_exclusions=1,
            max_exclusions_per_primary_cell=1,
        ),
    )

    assert any(rate.denominator == 44 for rate in analysis.e6_unsafe_imitation)
    with pytest.raises(StageBAnalysisError, match="preapproved"):
        analyze_stage_b(excluded, bootstrap_replicates=100)
