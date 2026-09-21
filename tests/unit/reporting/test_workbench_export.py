from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from shadowskillbench.analysis.dataset import AnalysisDataset, OutcomeRow
from shadowskillbench.analysis.stage_a import (
    ResponseCurvePoint,
    StageAAnalysis,
    StageAInference,
    StageAModel,
)
from shadowskillbench.analysis.stage_b import (
    DeltaEstimate,
    MacroGovernanceEffect,
    RateEstimate,
    StageBAnalysis,
    StageBInference,
)
from shadowskillbench.authority import DecisionDisposition
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.episodes import EpisodeStage, ExperimentCondition
from shadowskillbench.protocol import ClaimsReport, PreregistrationReport
from shadowskillbench.reporting.report import (
    ProtocolEvidence,
    ReportEvidence,
    RepresentativeTrace,
    hash_analysis_dataset,
    render_report_artifacts,
    write_report_artifacts,
)
from shadowskillbench.reporting.selection import (
    SelectionCandidate,
    SelectionResult,
    SelectionStatus,
)
from shadowskillbench.reporting.workbench_export import (
    WORKBENCH_SCHEMA,
    PublicEpisodeTrace,
    PublicTraceEvent,
    WorkbenchExportError,
    build_workbench_export,
    render_workbench_json,
    validate_workbench_export,
    write_workbench_export,
)

_HASH = "sha256:" + "a" * 64


def _row(index: int, *, stage: EpisodeStage = EpisodeStage.CONFIRMATORY_A) -> OutcomeRow:
    return OutcomeRow(
        stage=stage,
        condition=ExperimentCondition.A2_SKILL_ONLY,
        domain="access_provisioning",
        contamination_ratio=0.5,
        skill_bundle_id="bundle-a",
        authority_class=None,
        order_assignment=None,
        expected_authority_disposition=DecisionDisposition.PROCEED,
        held_out_case_id=f"case-{index}",
        repeat_index=1,
        episode_id=f"episode-{index}",
        planned_episode_hash=_HASH,
        manifest_hash=_HASH,
        result_hash=_HASH,
        score_hash=_HASH,
        task_completion=True,
        completion_under_policy=index == 1,
        policy_violation=False,
        unsafe_imitation=False,
        false_enforcement=False,
        escalation_expected=False,
        escalation_correct=False,
        authority_resolution_correct=True,
        claim_outcome_mismatch=False,
        turns=1,
        tool_calls=1,
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
        total_cost_nanos=7,
    )


def _evidence() -> ReportEvidence:
    dataset = AnalysisDataset(rows=(_row(2), _row(1)), exclusions=(), aggregates=())
    stage_a = StageAAnalysis(
        raw_counts=(),
        response_curves=(
            ResponseCurvePoint(
                scope="pooled",
                condition=ExperimentCondition.A2_SKILL_ONLY,
                contamination_ratio=0.5,
                numerator=1,
                denominator=2,
                estimate=0.5,
                ci_low=0.1,
                ci_high=0.9,
            ),
        ),
        estimands=(),
        average_marginal_effects=(),
        model=StageAModel("test", "binomial", "logit", (), (), (), 0, 0, 0, False),
        inference=StageAInference(100, 0, 0.95, "test"),
    )
    return ReportEvidence(
        dataset=dataset,
        stage_a=stage_a,
        stage_b=StageBAnalysis((), (), (), (), (), (), StageBInference(100, 0, 0.95, "test")),
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
            reproduce_command="uv run shadowskillbench reproduce",
        ),
        analysis_dataset_sha256=hash_analysis_dataset(dataset),
        representative_traces=(
            RepresentativeTrace(
                "A", SelectionResult("A", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
            ),
            RepresentativeTrace(
                "B", SelectionResult("B", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
            ),
        ),
        limitations=("Synthetic evidence is not evidence of workplace behavior.",),
    )


def _public_trace(episode_id: str = "episode-1") -> PublicEpisodeTrace:
    return PublicEpisodeTrace(
        episode_id=episode_id,
        trace_artifact_sha256=_HASH,
        events=(
            PublicTraceEvent(0, "message", "Visible system-message summary.", role="system"),
            PublicTraceEvent(
                1,
                "tool_call",
                "Public tool invocation summary.",
                tool_name="grant_access",
                local_status="success",
            ),
            PublicTraceEvent(
                2,
                "state_change",
                "Public state transition summary.",
                before_state_hash=_HASH,
                after_state_hash=_HASH,
            ),
            PublicTraceEvent(
                3,
                "authority_lookup",
                "Public authority lookup summary.",
                authority_lookup_hash=_HASH,
            ),
            PublicTraceEvent(4, "reason_code", "Public reason-code summary.", reason_code="D-017"),
            PublicTraceEvent(
                5, "final_state", "Public final-state summary.", final_state_hash=_HASH
            ),
        ),
    )


def test_hold_export_is_deterministic_read_only_and_excludes_hidden_truth() -> None:
    evidence = _evidence()

    first = build_workbench_export(evidence)
    second = build_workbench_export(evidence)
    parsed = json.loads(render_workbench_json(evidence))

    assert first.dataset == second.dataset == parsed
    assert parsed["schema"] == WORKBENCH_SCHEMA
    assert parsed["read_only"] is True
    assert parsed["claim_ceiling"]["status"] == "hold"
    assert parsed["claim_ceiling"]["descriptive_only"] is True
    assert parsed["claims"]["renderable_finding_ids"] == []
    assert parsed["summaries"]["metrics"]["latency"]["status"] == "unavailable"
    assert parsed["summaries"]["metrics"]["human_review_burden"]["status"] == "available"
    assert parsed["summaries"]["metrics"]["human_review_burden"]["value"] == 0.0
    assert [row["episode_id"] for row in parsed["episode_index"]] == ["episode-1", "episode-2"]
    serialized = render_workbench_json(evidence).decode("utf-8")
    for forbidden in ("expected_authority_disposition", "authority_class", "prompt_text"):
        assert forbidden not in serialized
    validate_workbench_export(first.dataset)


def test_report_artifacts_are_rendered_from_the_same_evidence() -> None:
    report_bytes, workbench_bytes = render_report_artifacts(_evidence())

    assert report_bytes.startswith(b"<!doctype html>")
    exported = json.loads(workbench_bytes)
    assert exported["content_hash"].startswith("sha256:")
    assert [(row["episode_id"], row["stage"]) for row in exported["episode_index"]] == [
        ("episode-1", "confirmatory_a"),
        ("episode-2", "confirmatory_a"),
    ]


def test_report_artifacts_publish_exclusively_as_a_pair(tmp_path: Path) -> None:
    html = tmp_path / "report.html"
    workbench = tmp_path / "workbench.json"

    write_report_artifacts(_evidence(), html, workbench)

    assert html.read_bytes().startswith(b"<!doctype html>")
    exported = json.loads(workbench.read_bytes())
    assert exported["content_hash"].startswith("sha256:")
    with pytest.raises(ValueError, match="exists"):
        write_report_artifacts(_evidence(), html, workbench)


def test_stage_b_primary_estimands_are_exported_and_rendered_with_intervals() -> None:
    baseline = RateEstimate(
        domain="pooled",
        condition=ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
        metric="completion_under_policy",
        authority_class=None,
        eligible_authority_classes=(),
        expected_authority_disposition=None,
        numerator=3,
        denominator=4,
        ci_low=0.5,
        ci_high=1.0,
    )
    comparison = RateEstimate(
        domain="pooled",
        condition=ExperimentCondition.B2_AUTHORITY_RESOLVER,
        metric="completion_under_policy",
        authority_class=None,
        eligible_authority_classes=(),
        expected_authority_disposition=None,
        numerator=4,
        denominator=4,
        ci_low=0.75,
        ci_high=1.0,
    )
    unsafe = replace(baseline, metric="unsafe_imitation", numerator=1, ci_low=0.0)
    false = replace(baseline, metric="false_enforcement", numerator=1, ci_low=0.0)
    stage_b = StageBAnalysis(
        e6_unsafe_imitation=(unsafe,),
        e7_false_enforcement=(false,),
        e8_authority_aware_gain=(
            MacroGovernanceEffect("pooled", (baseline,), (comparison,), 0.1, 0.4),
        ),
        e9_completion_under_policy_delta=(DeltaEstimate(baseline, comparison, 0.0, 0.5),),
        e9_unsafe_imitation_delta=(
            DeltaEstimate(
                unsafe,
                replace(unsafe, condition=ExperimentCondition.B2_AUTHORITY_RESOLVER),
                -0.5,
                0.5,
            ),
        ),
        figure_2_points=(),
        inference=StageBInference(100, 0, 0.95, "test"),
    )
    evidence = replace(_evidence(), stage_b=stage_b)
    exported = build_workbench_export(evidence).dataset["analysis"]["stage_b"]
    assert exported["e6_unsafe_imitation"][0]["ci_high"] == 1.0
    assert exported["e7_false_enforcement"][0]["numerator"] == 1
    assert exported["e8_authority_aware_gain"][0]["estimate"] == 0.25
    assert exported["e9_completion_under_policy_delta"][0]["ci_low"] == 0.0
    assert exported["e9_unsafe_imitation_delta"][0]["ci_high"] == 0.5
    html, _ = render_report_artifacts(evidence)
    rendered = html.decode()
    assert "Stage B primary estimands" in rendered
    assert "E6 unsafe imitation" in rendered
    assert "E9 unsafe-imitation delta" in rendered


def test_export_declares_unavailable_operational_diagnostics_without_inventing_values() -> None:
    metrics = build_workbench_export(_evidence()).dataset["summaries"]["metrics"]

    assert metrics["ordinary_task_completion"]["value"] == 1.0
    assert metrics["completion_under_policy"]["value"] == 0.5
    assert metrics["unsafe_imitation"]["status"] == "unavailable"
    assert metrics["false_enforcement"]["status"] == "unavailable"
    assert metrics["instruction_hierarchy_delta"]["status"] == "unavailable"
    assert metrics["latency"]["status"] == "unavailable"
    assert metrics["human_review_burden"] == {
        "status": "available",
        "unit": "review_requests_per_episode",
        "value": 0.0,
        "numerator": 0,
        "denominator": 2,
        "reason": "unique successful pending-to-approval_requested transitions",
    }


def test_export_projects_average_marginal_effects_with_intervals() -> None:
    from shadowskillbench.analysis.stage_a import AverageMarginalEffect

    effect = AverageMarginalEffect(
        effect="condition_contrast",
        scope="pooled",
        condition=ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
        reference_condition=ExperimentCondition.A2_SKILL_ONLY,
        contamination_ratio=0.5,
        estimate=0.25,
        ci_low=0.1,
        ci_high=0.4,
    )
    evidence = _evidence()
    exported = build_workbench_export(
        replace(evidence, stage_a=replace(evidence.stage_a, average_marginal_effects=(effect,)))
    ).dataset
    assert exported["analysis"]["average_marginal_effects"] == [
        {
            "effect": "condition_contrast",
            "scope": "pooled",
            "condition": "A4_SKILL_POLICY_SYSTEM_TIER",
            "reference_condition": "A2_SKILL_ONLY",
            "contamination_ratio": 0.5,
            "estimate": 0.25,
            "ci_low": 0.1,
            "ci_high": 0.4,
        }
    ]
    validate_workbench_export(exported)
    html, _ = render_report_artifacts(
        replace(evidence, stage_a=replace(evidence.stage_a, average_marginal_effects=(effect,)))
    )
    assert "Stage A average marginal effects" in html.decode()
    assert "95% CI" in html.decode()


def test_export_binds_human_review_burden_numerator_denominator_and_rate() -> None:
    evidence = _evidence()
    rows = (
        replace(evidence.dataset.rows[0], human_review_request_count=1),
        evidence.dataset.rows[1],
    )
    dataset = AnalysisDataset(rows=rows, exclusions=(), aggregates=())
    exported = build_workbench_export(
        replace(evidence, dataset=dataset, analysis_dataset_sha256=hash_analysis_dataset(dataset))
    ).dataset

    assert exported["summaries"]["metrics"]["human_review_burden"] == {
        "status": "available",
        "unit": "review_requests_per_episode",
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
        "reason": "unique successful pending-to-approval_requested transitions",
    }


def test_export_rejects_inconsistent_figure_and_tampered_self_hash() -> None:
    evidence = _evidence()
    bad_curve = replace(evidence.stage_a.response_curves[0], numerator=2)

    with pytest.raises(WorkbenchExportError, match="Figure 1 point"):
        build_workbench_export(
            replace(evidence, stage_a=replace(evidence.stage_a, response_curves=(bad_curve,)))
        )

    dataset = json.loads(render_workbench_json(evidence))
    dataset["limitations"].append("tampered")
    with pytest.raises(WorkbenchExportError, match="content_hash"):
        validate_workbench_export(dataset)


def test_explicit_writer_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "workbench.json"

    export = write_workbench_export(_evidence(), destination)

    assert destination.read_bytes() == export.json_bytes()
    with pytest.raises(WorkbenchExportError, match="refusing to overwrite"):
        write_workbench_export(_evidence(), destination)


def test_public_episode_trace_exports_sanitized_synchronized_timeline() -> None:
    exported = build_workbench_export(_evidence(), public_episode_traces=(_public_trace(),)).dataset

    trace = exported["episode_traces"][0]
    assert trace["episode_id"] == "episode-1"
    assert [event["kind"] for event in trace["events"]] == [
        "message",
        "tool_call",
        "state_change",
        "authority_lookup",
        "reason_code",
        "final_state",
    ]
    assert "raw_provider_payload" not in render_workbench_json(
        _evidence(), public_episode_traces=(_public_trace(),)
    ).decode("utf-8")


def test_public_episode_trace_rejects_unbound_and_hidden_raw_fields() -> None:
    with pytest.raises(WorkbenchExportError, match="not bound"):
        build_workbench_export(
            _evidence(), public_episode_traces=(_public_trace("episode-unbound"),)
        )
    with pytest.raises(WorkbenchExportError, match="duplicate"):
        build_workbench_export(
            _evidence(), public_episode_traces=(_public_trace(), _public_trace())
        )

    dataset = json.loads(
        render_workbench_json(_evidence(), public_episode_traces=(_public_trace(),))
    )
    dataset["episode_traces"][0]["events"][0]["raw_provider_payload"] = "forbidden"
    unsigned = {key: value for key, value in dataset.items() if key != "content_hash"}
    dataset["content_hash"] = sha256_ref(unsigned)
    with pytest.raises(WorkbenchExportError, match="forbidden or missing"):
        validate_workbench_export(dataset)


def test_selected_representative_trace_must_bind_exported_public_trace() -> None:
    candidate = SelectionCandidate(
        stage="A",
        case_id="case-1",
        contamination_ratio=0.5,
        cascade_action_trace_length=1,
        a2_cup_by_repeat=(False,),
        a4_cup_by_repeat=(True,),
    )
    selected = RepresentativeTrace(
        "A",
        SelectionResult("A", SelectionStatus.SELECTED, candidate, ("case-1",)),
        trace_artifact_sha256=_HASH,
        summary="Selected public trace summary.",
    )
    evidence = replace(
        _evidence(), representative_traces=(selected, _evidence().representative_traces[1])
    )

    with pytest.raises(WorkbenchExportError, match="not bound"):
        build_workbench_export(evidence)
    assert (
        build_workbench_export(evidence, public_episode_traces=(_public_trace(),)).dataset[
            "summaries"
        ]["representative_traces"][0]["status"]
        == "selected"
    )
