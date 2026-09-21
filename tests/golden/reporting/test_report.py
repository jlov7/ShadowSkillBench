from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest

from shadowskillbench.analysis.dataset import AnalysisDataset, ExclusionRow, OutcomeRow
from shadowskillbench.analysis.exclusions import ApprovedTechnicalExclusionPolicy
from shadowskillbench.analysis.profile import STAGE_A_METHOD, STAGE_B_METHOD, binding_for
from shadowskillbench.analysis.stage_a import (
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
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.episodes import EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.audit import RuntimeBinding
from shadowskillbench.experiments.planner import AUTHORITY_CLASSES
from shadowskillbench.protocol import ClaimsReport, PreregistrationCore, PreregistrationReport
from shadowskillbench.protocol.models import (
    DemonstrationDeclaration,
    ModelsDeclaration,
    SkillCompilerDeclaration,
    SkillExecutorDeclaration,
)
from shadowskillbench.reporting.report import (
    ProtocolEvidence,
    ReportEvidence,
    ReportInputError,
    ReportStatus,
    RepresentativeTrace,
    hash_analysis_dataset,
    render_report,
    report_status,
)
from shadowskillbench.reporting.selection import SelectionResult, SelectionStatus

_HASH = "sha256:" + "a" * 64


def _stage_a() -> StageAAnalysis:
    return StageAAnalysis(
        raw_counts=(
            RawCuPCount(
                scope="pooled",
                condition=ExperimentCondition.A0_BARE,
                contamination_ratio=None,
                numerator=1,
                denominator=2,
            ),
        ),
        response_curves=(
            ResponseCurvePoint(
                scope="pooled",
                condition=ExperimentCondition.A2_SKILL_ONLY,
                contamination_ratio=0.5,
                numerator=1,
                denominator=2,
                estimate=0.5,
                ci_low=0.25,
                ci_high=0.75,
            ),
        ),
        estimands=(
            StageAEstimand(
                estimand="E1",
                component="slope",
                scope="pooled",
                estimate=0.0,
                ci_low=-0.1,
                ci_high=0.1,
            ),
        ),
        average_marginal_effects=(),
        model=StageAModel(
            formula="test",
            family="binomial",
            link="logit",
            parameter_names=(),
            coefficients=(),
            covariance=(),
            observation_count=0,
            skill_bundle_cluster_count=0,
            held_out_case_cluster_count=0,
            converged=False,
        ),
        inference=StageAInference(
            bootstrap_replicates=100,
            bootstrap_seed=0,
            confidence_level=0.95,
            method="test",
        ),
    )


def _stage_b() -> StageBAnalysis:
    unsafe = RateEstimate(
        domain="pooled",
        condition=ExperimentCondition.B0_SKILL_ONLY,
        metric="unsafe_imitation",
        authority_class=None,
        eligible_authority_classes=AUTHORITY_CLASSES,
        expected_authority_disposition=None,
        numerator=1,
        denominator=2,
    )
    false_enforcement = RateEstimate(
        domain="pooled",
        condition=ExperimentCondition.B0_SKILL_ONLY,
        metric="false_enforcement",
        authority_class=None,
        eligible_authority_classes=AUTHORITY_CLASSES,
        expected_authority_disposition=None,
        numerator=0,
        denominator=2,
    )
    return StageBAnalysis(
        (),
        (),
        (),
        (),
        (),
        (
            Figure2Point(
                domain="pooled",
                condition=ExperimentCondition.B0_SKILL_ONLY,
                unsafe_imitation=unsafe,
                false_enforcement=false_enforcement,
            ),
        ),
        StageBInference(100, 0, 0.95, "test"),
    )


def _held_trace(stage: Literal["A", "B"]) -> RepresentativeTrace:
    return RepresentativeTrace(
        stage=stage,
        selection=SelectionResult(stage, SelectionStatus.HOLD_NO_CANDIDATE, None, ()),
    )


def _evidence() -> ReportEvidence:
    dataset = AnalysisDataset(rows=(), exclusions=(), aggregates=())
    return ReportEvidence(
        dataset=dataset,
        stage_a=_stage_a(),
        stage_b=_stage_b(),
        preregistration=PreregistrationReport(
            source_path=None,
            valid=False,
            core=None,
            violations=(),
        ),
        claims=ClaimsReport(
            ledger_path=Path("claims.yaml"),
            version="1.0",
            valid=True,
            claims=(),
            violations=(),
            renderable_findings=(),
        ),
        protocol=ProtocolEvidence(
            protocol_tag="v1.0",
            freeze_manifest_sha256=_HASH,
            prompt_hashes=(_HASH,),
            code_commit="abcdef0",
            reproduce_command="uv run shadowskillbench reproduce --protocol protocol.md",
        ),
        analysis_dataset_sha256=hash_analysis_dataset(dataset),
        representative_traces=(_held_trace("A"), _held_trace("B")),
        limitations=("Synthetic benchmark results do not establish workplace outcomes.",),
    )


def test_report_golden_hold_is_self_contained_and_includes_all_required_sections() -> None:
    evidence = _evidence()

    first = render_report(evidence)
    second = render_report(evidence)

    assert first == second
    assert report_status(evidence) is ReportStatus.HOLD
    assert "HOLD — this artifact is descriptive only" in first
    assert "<!doctype html>" in first
    assert "<script" not in first
    assert "<link" not in first
    for heading in (
        "Executive summary",
        "Prior art / claim",
        "Protocol integrity",
        "Figure 1",
        "Figure 2",
        "Domain breakdown",
        "Cost",
        "Representative traces",
        "Null / adverse findings",
        "Limitations",
        "Claims ledger",
        "Reproduce command",
    ):
        assert heading in first
    assert "<table>" in first
    assert "50.0%" in first
    assert "HOLD — no eligible candidate" in first


def test_report_rejects_missing_protocol_hash_and_trace_substitution() -> None:
    with pytest.raises(ReportInputError, match="freeze_manifest_sha256"):
        ProtocolEvidence(
            protocol_tag="v1.0",
            external_anchor_locator="anchor",
            freeze_manifest_sha256="missing",
            prompt_hashes=(_HASH,),
            code_commit="abcdef0",
            reproduce_command="uv run reproduce",
        )
    with pytest.raises(ReportInputError, match="substitute trace"):
        RepresentativeTrace(
            stage="A",
            selection=SelectionResult("A", SelectionStatus.HOLD_NO_CANDIDATE, None, ()),
            summary="invented replacement",
        )
    with pytest.raises(ReportInputError, match="does not bind"):
        replace(_evidence(), analysis_dataset_sha256="sha256:" + "b" * 64)


def _outcome(stage: EpisodeStage, index: int) -> OutcomeRow:
    return OutcomeRow(
        stage=stage,
        condition=ExperimentCondition.A0_BARE,
        domain="access_provisioning",
        contamination_ratio=None,
        skill_bundle_id=None,
        authority_class=None,
        order_assignment=None,
        expected_authority_disposition=DecisionDisposition.PROCEED,
        held_out_case_id=f"case-{index}",
        repeat_index=1,
        episode_id=f"episode-{stage.value}-{index}",
        planned_episode_hash="planned",
        manifest_hash="manifest",
        result_hash="result",
        score_hash="score",
        task_completion=True,
        completion_under_policy=True,
        policy_violation=False,
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


def _complete_looking_evidence() -> ReportEvidence:
    base = _evidence()
    stage_a_binding = binding_for(
        stage="A",
        bootstrap_replicates=999,
        bootstrap_seed=0,
        confidence_level=0.95,
        method=STAGE_A_METHOD,
    )
    stage_b_binding = binding_for(
        stage="B",
        bootstrap_replicates=999,
        bootstrap_seed=0,
        confidence_level=0.95,
        method=STAGE_B_METHOD,
    )
    dataset = AnalysisDataset(
        rows=(
            *(_outcome(EpisodeStage.CONFIRMATORY_A, index) for index in range(7_440)),
            *(_outcome(EpisodeStage.CONFIRMATORY_B, index) for index in range(1_800)),
        ),
        exclusions=(),
        aggregates=(),
    )
    stage_a = replace(
        base.stage_a,
        raw_counts=base.stage_a.raw_counts * 66,
        response_curves=base.stage_a.response_curves * 75,
        estimands=base.stage_a.estimands * 18,
        inference=replace(
            base.stage_a.inference,
            bootstrap_replicates=999,
            bootstrap_seed=0,
            method=STAGE_A_METHOD,
            classification=stage_a_binding.classification,
            profile=stage_a_binding.profile,
            profile_hash=stage_a_binding.profile_hash,
            statistics_configuration_sha256=stage_a_binding.statistics_configuration_sha256,
        ),
    )
    governance_baseline = RateEstimate(
        domain="pooled",
        condition=ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
        metric="authority_resolution_correct",
        authority_class="PRACTICE_MATCHES_ACTIVE_POLICY",
        eligible_authority_classes=("PRACTICE_MATCHES_ACTIVE_POLICY",),
        expected_authority_disposition=None,
        numerator=1,
        denominator=2,
    )
    governance_comparison = replace(
        governance_baseline,
        condition=ExperimentCondition.B2_AUTHORITY_RESOLVER,
    )
    governance_effect = MacroGovernanceEffect(
        domain="pooled",
        baseline_components=(governance_baseline,),
        comparison_components=(governance_comparison,),
    )
    cup_baseline = RateEstimate(
        domain="pooled",
        condition=ExperimentCondition.B2_AUTHORITY_RESOLVER,
        metric="completion_under_policy",
        authority_class=None,
        eligible_authority_classes=AUTHORITY_CLASSES,
        expected_authority_disposition=None,
        numerator=1,
        denominator=2,
    )
    cup_delta = DeltaEstimate(
        baseline=cup_baseline,
        comparison=replace(cup_baseline, condition=ExperimentCondition.B3_DETERMINISTIC_GATE),
    )
    unsafe_baseline = replace(
        cup_baseline,
        metric="unsafe_imitation",
    )
    unsafe_delta = DeltaEstimate(
        baseline=unsafe_baseline,
        comparison=replace(unsafe_baseline, condition=ExperimentCondition.B3_DETERMINISTIC_GATE),
    )
    stage_b = replace(
        base.stage_b,
        e6_unsafe_imitation=(base.stage_b.figure_2_points[0].unsafe_imitation,) * 15,
        e7_false_enforcement=(base.stage_b.figure_2_points[0].false_enforcement,) * 6,
        e8_authority_aware_gain=(governance_effect,) * 3,
        e9_completion_under_policy_delta=(cup_delta,) * 3,
        e9_unsafe_imitation_delta=(unsafe_delta,) * 3,
        figure_2_points=base.stage_b.figure_2_points * 12,
        inference=replace(
            base.stage_b.inference,
            bootstrap_replicates=999,
            bootstrap_seed=0,
            method=STAGE_B_METHOD,
            classification=stage_b_binding.classification,
            profile=stage_b_binding.profile,
            profile_hash=stage_b_binding.profile_hash,
            statistics_configuration_sha256=stage_b_binding.statistics_configuration_sha256,
        ),
    )
    return replace(
        base,
        dataset=dataset,
        stage_a=stage_a,
        stage_b=stage_b,
        preregistration=base.preregistration.model_copy(
            update={
                "valid": True,
                "core": PreregistrationCore.model_construct(
                    models=ModelsDeclaration.model_construct(
                        skill_executor=SkillExecutorDeclaration.model_construct(
                            provider="provider",
                            model="model",
                            model_version_date="2026-08-24",
                        )
                    ),
                    demonstrations=DemonstrationDeclaration.model_construct(
                        bundle_seeds_per_domain_ratio=3,
                        ratios=(Decimal("0"),),
                    ),
                ),
            }
        ),
        analysis_dataset_sha256=hash_analysis_dataset(dataset),
        runtime_binding=RuntimeBinding(
            plan_hash=_HASH,
            package_hash=_HASH,
            run_descriptor_hash=_HASH,
            freeze_manifest_hash=_HASH,
            anchor_receipt_hash=_HASH,
            operator_endpoint_hash=_HASH,
            operator_api_key_environment="SSB_TEST_KEY",
            provider="provider",
            model="model",
            model_version="2026-08-24",
        ),
    )


def test_complete_looking_evidence_stays_hold_without_verified_anchor_or_audit_pass() -> None:
    evidence = _complete_looking_evidence()

    assert report_status(evidence) is ReportStatus.HOLD

    unverified = replace(
        evidence,
        protocol=replace(
            evidence.protocol,
            external_anchor_locator="anchor",
            anchor_receipt_sha256=_HASH,
            experiment_audit_report_sha256=_HASH,
            experiment_audit_status="PASS",
        ),
    )
    assert report_status(unverified) is ReportStatus.HOLD

    anchored = replace(
        evidence,
        protocol=replace(
            evidence.protocol,
            external_anchor_locator="anchor",
            anchor_receipt_sha256=_HASH,
            anchor_verified=True,
        ),
    )
    assert report_status(anchored) is ReportStatus.HOLD

    audited = replace(
        anchored,
        protocol=replace(
            anchored.protocol,
            experiment_audit_report_sha256=_HASH,
            experiment_audit_status="PASS",
        ),
    )
    assert report_status(audited) is ReportStatus.HOLD


def test_report_status_admits_one_approved_bounded_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _complete_looking_evidence()
    monkeypatch.setattr("shadowskillbench.reporting.report._analysis_is_complete", lambda *_: True)
    excluded = ExclusionRow(
        stage=EpisodeStage.CONFIRMATORY_A,
        condition=ExperimentCondition.A0_BARE,
        domain="access_provisioning",
        contamination_ratio=None,
        skill_bundle_id=None,
        authority_class=None,
        order_assignment=None,
        expected_authority_disposition=DecisionDisposition.PROCEED,
        held_out_case_id="excluded-case",
        repeat_index=1,
        episode_id="excluded-episode",
        planned_episode_hash="planned",
        manifest_hash="manifest",
        result_hash="result",
        score_hash="score",
        error_code="MODEL_PROVIDER_TRANSIENT",
    )
    removed = next(
        index
        for index, row in enumerate(evidence.dataset.rows)
        if row.stage is EpisodeStage.CONFIRMATORY_A
    )
    rows = evidence.dataset.rows[:removed] + evidence.dataset.rows[removed + 1 :]
    dataset = AnalysisDataset(rows=rows, exclusions=(excluded,), aggregates=())
    assert evidence.runtime_binding is not None
    evidence = replace(
        evidence,
        dataset=dataset,
        analysis_dataset_sha256=hash_analysis_dataset(dataset),
        protocol=replace(
            evidence.protocol,
            external_anchor_locator="anchor",
            anchor_receipt_sha256=_HASH,
            anchor_verified=True,
            experiment_plan_sha256=_HASH,
            experiment_runtime_binding_sha256=sha256_ref(evidence.runtime_binding.projection()),
            experiment_audit_report_sha256=_HASH,
            experiment_audit_status="PASS",
        ),
        exclusion_policy=ApprovedTechnicalExclusionPolicy(
            allowed_error_codes=("MODEL_PROVIDER_TRANSIENT",),
            max_total_exclusions=1,
            max_exclusions_per_primary_cell=1,
        ),
    )
    assert report_status(evidence) is ReportStatus.CONFIRMATORY
    local_custody = replace(
        evidence,
        protocol=replace(
            evidence.protocol,
            external_anchor_locator=None,
            custody_locator="urn:git:" + "a" * 40,
            custody_mode="LOCAL_HASH_CUSTODY",
        ),
    )
    assert report_status(local_custody) is ReportStatus.CONFIRMATORY
    assert "local custody is not externally verified" in render_report(local_custody)
    connector_provider = replace(
        evidence,
        preregistration=evidence.preregistration.model_copy(
            update={
                "core": evidence.preregistration.core.model_copy(
                    update={
                        "models": ModelsDeclaration.model_construct(
                            skill_compiler=SkillCompilerDeclaration.model_construct(
                                provider="provider"
                            ),
                            skill_executor=SkillExecutorDeclaration.model_construct(
                                provider="executor-provider",
                                model="model",
                                model_version_date="2026-08-24",
                            ),
                        )
                    }
                )
            }
        ),
    )
    assert report_status(connector_provider) is ReportStatus.CONFIRMATORY
    assert evidence.runtime_binding is not None
    assert (
        report_status(
            replace(
                evidence,
                runtime_binding=replace(
                    evidence.runtime_binding,
                    package_hash="sha256:" + "b" * 64,
                    freeze_manifest_hash="sha256:" + "b" * 64,
                ),
            )
        )
        is ReportStatus.HOLD
    )
    assert report_status(replace(evidence, exclusion_policy=None)) is ReportStatus.HOLD
    assert (
        report_status(
            replace(
                evidence,
                exclusion_policy=ApprovedTechnicalExclusionPolicy(
                    allowed_error_codes=("MODEL_PROVIDER_TERMINAL",),
                    max_total_exclusions=1,
                    max_exclusions_per_primary_cell=1,
                ),
            )
        )
        is ReportStatus.HOLD
    )
    held_dataset = AnalysisDataset(
        rows=(_outcome(EpisodeStage.CONFIRMATORY_A, 0),),
        exclusions=(excluded,),
        aggregates=(),
    )
    held = replace(
        _evidence(),
        dataset=held_dataset,
        analysis_dataset_sha256=hash_analysis_dataset(held_dataset),
        exclusion_policy=evidence.exclusion_policy,
    )
    rendered = render_report(held)
    assert "Confirmatory technical exclusions" in rendered
    assert "approved; total cap 1; per-cell cap 1" in rendered
