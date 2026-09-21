from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace
from typing import Any, cast

import pytest

import shadowskillbench.analysis.sealed as sealed_module
from shadowskillbench import cli
from shadowskillbench.analysis import build_aggregate_metric_rows
from shadowskillbench.analysis.dataset import AnalysisDataset, OutcomeRow
from shadowskillbench.analysis.profile import STAGE_A_METHOD, STAGE_B_METHOD, binding_for
from shadowskillbench.analysis.sealed import (
    SealedAnalysis,
    SealedAnalysisHold,
    analyze_sealed_confirmatory_artifacts,
    sealed_analysis_projection,
    sealed_analysis_receipt_matches,
)
from shadowskillbench.analysis.stage_a import StageAInference
from shadowskillbench.analysis.stage_b import StageBInference
from shadowskillbench.authority import DecisionDisposition
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.episodes import EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.audit import (
    FrozenArtifactCatalog,
    audit_confirmatory_run,
)
from shadowskillbench.experiments.confirmatory_execution_v4 import (
    V4_RESULT_PROFILE,
    V4_RUNTIME_BINDING_PROFILE,
)
from shadowskillbench.experiments.io import load_confirmatory_audit_inputs
from shadowskillbench.experiments.planner import ConfirmatoryEpisodePlan
from shadowskillbench.protocol.scientific_freeze import MULTIPLICITY_MEMBERS
from tests.unit.experiments.test_audit import _prepared, _write_v15d_configuration_failure


def _plan_payload(plan: ConfirmatoryEpisodePlan) -> dict[str, object]:
    return {
        "profile": "SSB-PLAN1",
        "plan_hash": plan.plan_hash,
        "episodes": [
            {"manifest": episode.manifest_projection(), "manifest_hash": episode.manifest_hash}
            for episode in plan.episodes
        ],
    }


def _catalog_payload(catalog: FrozenArtifactCatalog) -> dict[str, object]:
    return {
        "profile": "SSB-AUDIT-CATALOG1",
        "case_manifest_hashes": sorted(catalog.case_manifest_hashes),
        "context_contract_hashes": sorted(catalog.context_contract_hashes),
        "source_manifest_hashes": sorted(catalog.source_manifest_hashes),
        "compiler_manifest_hashes": sorted(catalog.compiler_manifest_hashes),
        "compiled_skill_artifact_hashes": sorted(catalog.compiled_skill_artifact_hashes),
        "rendered_skill_hashes": sorted(catalog.rendered_skill_hashes),
        "policies": [
            {"rendered_hash": policy.rendered_hash, "rendered_text": policy.rendered_text}
            for policy in sorted(catalog.policies, key=lambda value: value.rendered_hash)
        ],
        "development_entities": sorted(catalog.development_entities or ()),
        "confirmatory_entities": sorted(catalog.confirmatory_entities or ()),
        "development_values": sorted(catalog.development_values or ()),
        "confirmatory_values": sorted(catalog.confirmatory_values or ()),
        "exclusion_rule_hash": catalog.exclusion_rule_hash,
    }


def _bundle(tmp_path: Path) -> Path:
    plan, _, run_directory, catalog = _prepared(tmp_path)
    root = tmp_path / "artifacts/experiments/confirmatory"
    copytree(run_directory, root)
    (root / "plan.json").write_bytes(canonical_json_bytes(_plan_payload(plan)))
    (root / "catalog.json").write_bytes(canonical_json_bytes(_catalog_payload(catalog)))
    return root


def test_audited_canonical_minimal_bundle_is_held_as_incomplete_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SealedAnalysisHold, match="HOLD_INCOMPLETE_CONFIRMATORY_DATASET"):
        analyze_sealed_confirmatory_artifacts(Path("artifacts/experiments/confirmatory"))

    assert root.is_dir()


def test_v4_run_bundle_result_flows_through_offline_audit_and_sealed_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)
    result_path = next((root / "results").glob("*.json"))
    legacy = json.loads(result_path.read_bytes())
    result = legacy["result"]
    runtime = {
        "profile": V4_RUNTIME_BINDING_PROFILE,
        "package_hash": "sha256:" + "a" * 64,
        "plan_hash": legacy["plan_hash"],
        "planned_manifest_hash": legacy["planned_manifest_hash"],
        "run_descriptor_hash": "sha256:" + "b" * 64,
        "result_hash": legacy["result_content_hash"],
        "result_ref": "artifacts/experiments/confirmatory-v4/results/fixture.v4.json",
    }
    value = {
        "profile": V4_RESULT_PROFILE,
        "schema_version": "4.0",
        "result_namespace": "artifacts/experiments/confirmatory-v4/results",
        "result_ref": runtime["result_ref"],
        "package_hash": runtime["package_hash"],
        "plan_hash": legacy["plan_hash"],
        "episode_id": legacy["episode_id"],
        "planned_manifest_hash": legacy["planned_manifest_hash"],
        "execution_manifest_hash": legacy["execution_manifest_hash"],
        "expected_execution_manifest_hash": legacy["expected_execution_manifest_hash"],
        "result_content_hash": legacy["result_content_hash"],
        "runtime_binding": runtime,
        "episode_result": result,
    }
    result_path.write_bytes(canonical_json_bytes({**value, "content_hash": sha256_ref(value)}))
    monkeypatch.chdir(tmp_path)
    inputs = load_confirmatory_audit_inputs(Path("artifacts/experiments/confirmatory"))
    audit = audit_confirmatory_run(inputs.plan, inputs.run_directory, catalog=inputs.catalog)
    assert audit.status == "PASS"
    with pytest.raises(SealedAnalysisHold, match="HOLD_INCOMPLETE_CONFIRMATORY_DATASET"):
        analyze_sealed_confirmatory_artifacts(Path("artifacts/experiments/confirmatory"))


def test_missing_and_symlinked_inputs_hold_before_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SealedAnalysisHold, match="HOLD_MISSING_ANALYSIS_ARTIFACTS"):
        analyze_sealed_confirmatory_artifacts(Path("artifacts/experiments/confirmatory"))

    root = _bundle(tmp_path)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes((root / "plan.json").read_bytes())
    (root / "plan.json").unlink()
    (root / "plan.json").symlink_to(replacement)
    with pytest.raises(SealedAnalysisHold, match="HOLD_UNSAFE_ANALYSIS_ARTIFACTS"):
        analyze_sealed_confirmatory_artifacts(Path("artifacts/experiments/confirmatory"))


@pytest.mark.parametrize("mutation", ("duplicate", "tamper", "noncanonical"))
def test_duplicate_tampered_and_noncanonical_results_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = _bundle(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = next((root / "results").glob("*.json"))
    if mutation == "duplicate":
        (root / "results" / "duplicate.json").write_bytes(result.read_bytes())
        expected = "HOLD_EXPERIMENT_AUDIT"
    elif mutation == "tamper":
        payload = json.loads(result.read_bytes())
        payload["result_content_hash"] = "sha256:" + "0" * 64
        result.write_bytes(canonical_json_bytes(payload))
        expected = "HOLD_EXPERIMENT_AUDIT"
    else:
        result.write_bytes(result.read_bytes() + b"\n")
        expected = "HOLD_NONCANONICAL_ANALYSIS_ARTIFACTS"

    with pytest.raises(SealedAnalysisHold, match=expected):
        analyze_sealed_confirmatory_artifacts(Path("artifacts/experiments/confirmatory"))


def test_pre_dispatch_configuration_failure_holds_before_outcome_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)
    _write_v15d_configuration_failure(next((root / "results").glob("*.json")))
    monkeypatch.chdir(tmp_path)

    def _outcome_analysis_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("outcome analysis must not run for a held audit")

    monkeypatch.setattr(sealed_module, "analyze_stage_a", _outcome_analysis_must_not_run)
    monkeypatch.setattr(sealed_module, "analyze_stage_b", _outcome_analysis_must_not_run)

    with pytest.raises(SealedAnalysisHold, match="HOLD_EXPERIMENT_AUDIT") as error:
        analyze_sealed_confirmatory_artifacts(Path("artifacts/experiments/confirmatory"))

    assert "PRE_DISPATCH_CONFIGURATION_FAILURE" in error.value.detail


def test_manual_sealed_rows_recompute_canonical_bound_aggregate_table() -> None:
    row = OutcomeRow(
        stage=EpisodeStage.CONFIRMATORY_A,
        condition=ExperimentCondition.A0_BARE,
        domain="access_provisioning",
        contamination_ratio=None,
        skill_bundle_id=None,
        authority_class=None,
        order_assignment=None,
        expected_authority_disposition=DecisionDisposition.PROCEED,
        held_out_case_id="case-1",
        repeat_index=1,
        episode_id="episode-1",
        planned_episode_hash="sha256:" + "a" * 64,
        manifest_hash="sha256:" + "b" * 64,
        result_hash="sha256:" + "c" * 64,
        score_hash="sha256:" + "d" * 64,
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
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
        total_cost_nanos=7,
    )

    aggregates = build_aggregate_metric_rows((row,))

    assert len(aggregates) == 10
    assert [aggregate.metric for aggregate in aggregates] == [
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
    completion = next(item for item in aggregates if item.metric == "completion_under_policy")
    assert completion.numerator == completion.denominator == 1
    escalation = next(item for item in aggregates if item.metric == "escalation_correct")
    assert escalation.denominator == 0


def test_sealed_receipt_rejects_nonfrozen_analysis_profile() -> None:
    a_binding = binding_for(
        stage="A",
        bootstrap_replicates=999,
        bootstrap_seed=0,
        confidence_level=0.95,
        method=STAGE_A_METHOD,
    )
    b_binding = binding_for(
        stage="B",
        bootstrap_replicates=999,
        bootstrap_seed=0,
        confidence_level=0.95,
        method=STAGE_B_METHOD,
    )
    stage_a = SimpleNamespace(
        inference=StageAInference(
            999,
            0,
            0.95,
            STAGE_A_METHOD,
            a_binding.classification,
            a_binding.profile,
            a_binding.profile_hash,
            a_binding.statistics_configuration_sha256,
        )
    )
    stage_b = SimpleNamespace(
        inference=StageBInference(
            999,
            0,
            0.95,
            STAGE_B_METHOD,
            b_binding.classification,
            b_binding.profile,
            b_binding.profile_hash,
            b_binding.statistics_configuration_sha256,
        )
    )

    projection = sealed_module._confirmatory_profile_projection(
        cast(Any, stage_a), cast(Any, stage_b)
    )

    assert projection["bootstrap_replicates"] == 999
    assert projection["bootstrap_seed"] == 0
    assert projection["profile"] == "SSB-CONFIRMATORY-STATISTICS1"
    for drifted_stage_a, drifted_stage_b in (
        (replace(stage_a.inference, bootstrap_seed=1), stage_b.inference),
        (replace(stage_a.inference, bootstrap_replicates=100), stage_b.inference),
        (replace(stage_a.inference, method="drift"), stage_b.inference),
        (replace(stage_a.inference, profile="SSB-DRIFT-1"), stage_b.inference),
        (stage_a.inference, replace(stage_b.inference, profile_hash="sha256:" + "0" * 64)),
    ):
        with pytest.raises(SealedAnalysisHold, match="HOLD_ANALYSIS_PROFILE_MISMATCH"):
            sealed_module._confirmatory_profile_projection(
                cast(Any, SimpleNamespace(inference=drifted_stage_a)),
                cast(Any, SimpleNamespace(inference=drifted_stage_b)),
            )


def _receipt_ready_analysis(monkeypatch: pytest.MonkeyPatch) -> SealedAnalysis:
    monkeypatch.setattr(
        sealed_module,
        "_confirmatory_profile_projection",
        lambda *_: {"profile": "fixture"},
    )
    return SealedAnalysis(
        dataset=AnalysisDataset((), (), ()),
        stage_a=cast(Any, SimpleNamespace()),
        stage_b=cast(Any, SimpleNamespace()),
        audit=cast(
            Any,
            SimpleNamespace(
                plan_hash="sha256:" + "a" * 64,
                report_hash="sha256:" + "b" * 64,
            ),
        ),
        runtime_binding=cast(Any, SimpleNamespace(projection=lambda: {"runtime": "fixture"})),
        execution_design_projection=tuple((member, 1, 1, 1) for member in MULTIPLICITY_MEMBERS),
    )


def test_profile2_receipt_is_json_safe_and_parsed_receipt_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = _receipt_ready_analysis(monkeypatch)

    receipt = sealed_analysis_projection(analysis)
    encoded = canonical_json_bytes(receipt)
    parsed = json.loads(encoded)
    path = tmp_path / "analysis.json"
    path.write_bytes(encoded)

    assert receipt["profile"] == "SSB-SEALED-ANALYSIS-2"
    assert type(receipt["execution_design_projection"]) is list
    assert all(type(item) is list for item in receipt["execution_design_projection"])
    assert parsed == receipt
    assert sealed_analysis_receipt_matches(cli._read_analysis_receipt(path), analysis)


def test_profile2_receipt_rejects_tampered_execution_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _receipt_ready_analysis(monkeypatch)
    receipt = json.loads(canonical_json_bytes(sealed_analysis_projection(analysis)))
    projection = receipt["execution_design_projection"]
    assert type(projection) is list
    projection[0][1] = 2

    assert not sealed_analysis_receipt_matches(receipt, analysis)
