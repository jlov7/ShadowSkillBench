from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from shadowskillbench.analysis.dataset import AnalysisDataset
from shadowskillbench.analysis.profile import STATISTICS_CONFIGURATION_SHA256
from shadowskillbench.analysis.sealed import SealedAnalysis
from shadowskillbench.analysis.stage_a import (
    StageAAnalysis,
    StageAEstimand,
    StageAInference,
    StageAModel,
)
from shadowskillbench.analysis.stage_b import (
    MacroGovernanceEffect,
    RateEstimate,
    StageBAnalysis,
    StageBInference,
)
from shadowskillbench.episodes import ExperimentCondition
from shadowskillbench.experiments.audit import ArtifactIntegrityReport, RuntimeBinding
from shadowskillbench.protocol import claims as claims_module
from shadowskillbench.protocol.claim_estimands import claim_estimand_mapping
from shadowskillbench.protocol.claims import (
    ReleaseClaimsStatus,
    validate_claims_ledger,
    validate_release_claims,
)
from shadowskillbench.protocol.scientific_freeze import (
    MULTIPLICITY_MEMBERS,
    SCIENTIFIC_FREEZE_INPUTS,
    ScientificFreezeBinding,
)
from shadowskillbench.reporting.report import ReportStatus, sealed_analysis_is_release_complete

ROOT = Path(__file__).parents[3]
CURRENT_LEDGER = ROOT / "protocol" / "CLAIMS_LEDGER.yaml"
HASH = "sha256:" + "a" * 64
LIMITATIONS = (
    "Synthetic domains limit external validity beyond the benchmark setting.",
    "The selected model provider and version bound these observations.",
    "Technical exclusions and audit coverage bound the included evidence.",
)


def prior_claim(claim_id: str = "PRIOR-001") -> dict[str, object]:
    return {
        "id": claim_id,
        "wording": "A bounded prior-art statement.",
        "status": "supported_prior_art",
        "evidence": ["Source A", "Source B"],
        "scope": "A narrow scope.",
    }


def hypothesis_claim(claim_id: str = "EXP-001") -> dict[str, object]:
    return {
        "id": claim_id,
        "wording": "A bounded hypothesis.",
        "status": "hypothesis",
        "evidence_command": f"shadowskillbench report claim {claim_id}",
        "artifact": "artifacts/results/claim.json",
        "scope": "A narrow scope.",
    }


def finding_claim(
    artifact_sha256: str,
    finding_disposition: str = "supports",
    claim_id: str = "EXP-001",
) -> dict[str, object]:
    claim = hypothesis_claim(claim_id)
    mapping = claim_estimand_mapping(claim_id)
    assert mapping is not None
    claim.update(
        {
            "status": "finding",
            "artifact_sha256": artifact_sha256,
            "finding_disposition": finding_disposition,
            "claim_mapping_profile": mapping.profile,
            "claim_mapping_id": mapping.mapping_id,
            "claim_mapping_sha256": mapping.sha256,
        }
    )
    return claim


def write_ledger(tmp_path: Path, data: object) -> tuple[Path, Path]:
    repository_root = tmp_path / "repository"
    repository_root.mkdir(parents=True)
    path = repository_root / "claims.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path, repository_root


def codes(report: Any) -> set[str]:
    return {violation.code for violation in report.violations}


def release_gate(report: object, **overrides: Any):
    arguments: dict[str, Any] = {
        "report_status": "confirmatory",
        "experiment_audit_status": "PASS",
        "experiment_audit_hash": HASH,
        "analysis_dataset_hash": HASH,
        "limitations": LIMITATIONS,
    }
    arguments.update(overrides)
    return validate_release_claims(report, **arguments)


def scientific_freeze() -> ScientificFreezeBinding:
    return ScientificFreezeBinding(
        manifest_hash=HASH,
        input_hashes=tuple(
            (
                item.path,
                STATISTICS_CONFIGURATION_SHA256 if item.path == "config/statistics.yaml" else HASH,
            )
            for item in SCIENTIFIC_FREEZE_INPUTS
        ),
        analysis_plan_sha256=HASH,
        power_precision_plan_sha256=HASH,
        power_precision_evidence_sha256=HASH,
        analysis_profile="SSB-CONFIRMATORY-STATISTICS3-BONFERRONI",
        multiplicity_family="E1-E9-primary-components-16",
        multiplicity_members=MULTIPLICITY_MEMBERS,
        multiplicity_scope="primary_confirmatory_estimands",
        multiplicity_method="bonferroni_simultaneous_bootstrap_intervals",
        interval_rule="simultaneous_95pct_ci_entirely_beyond_material_threshold",
        familywise_confidence_level=0.95,
        material_threshold=0.1,
        bootstrap_replicates=40_000,
        bootstrap_seed=104_731,
        member_execution_designs=tuple((member, 1, 1, 1) for member in MULTIPLICITY_MEMBERS),
    )


def sealed_analysis(
    *,
    e1: tuple[float, float, float] = (-0.2, -0.3, -0.11),
) -> SealedAnalysis:
    audit = ArtifactIntegrityReport("PASS", HASH, None, (), ())
    stage_a = StageAAnalysis(
        raw_counts=(),
        response_curves=(),
        estimands=(StageAEstimand("E1", "slope", "pooled", *e1),),
        average_marginal_effects=(),
        model=cast(StageAModel, None),
        inference=cast(StageAInference, None),
    )
    baseline = RateEstimate(
        domain="pooled",
        condition=ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
        metric="authority_resolution_correct",
        authority_class=None,
        eligible_authority_classes=(),
        expected_authority_disposition=None,
        numerator=6,
        denominator=10,
        ci_low=0.5,
        ci_high=0.7,
    )
    comparison = RateEstimate(
        domain="pooled",
        condition=ExperimentCondition.B2_AUTHORITY_RESOLVER,
        metric="authority_resolution_correct",
        authority_class=None,
        eligible_authority_classes=(),
        expected_authority_disposition=None,
        numerator=8,
        denominator=10,
        ci_low=0.7,
        ci_high=0.9,
    )
    stage_b = StageBAnalysis(
        e6_unsafe_imitation=(),
        e7_false_enforcement=(),
        e8_authority_aware_gain=(
            MacroGovernanceEffect("pooled", (baseline,), (comparison,), 0.11, 0.3),
        ),
        e9_completion_under_policy_delta=(),
        e9_unsafe_imitation_delta=(),
        figure_2_points=(),
        inference=cast(StageBInference, None),
    )
    return SealedAnalysis(
        dataset=AnalysisDataset((), (), ()),
        stage_a=stage_a,
        stage_b=stage_b,
        audit=audit,
        runtime_binding=cast(RuntimeBinding, None),
        execution_design_projection=scientific_freeze().member_execution_designs,
    )


def supported_release_gate(report: object, **overrides: object):
    sealed = sealed_analysis()
    return release_gate(
        report,
        sealed_analysis=sealed,
        experiment_audit_hash=sealed.audit.report_hash,
        analysis_dataset_hash=sealed.dataset_sha256,
        scientific_freeze=scientific_freeze(),
        **overrides,
    )


def test_current_immutable_ledger_has_expected_composition() -> None:
    report = validate_claims_ledger(CURRENT_LEDGER, ROOT)

    assert report.valid
    assert report.version == "1.0"
    assert report.ledger_path == CURRENT_LEDGER.resolve()
    assert report.counts == {"supported_prior_art": 3, "hypothesis": 4, "finding": 0}
    assert len(report.claims) == 7
    assert [claim.status for claim in report.claims].count("supported_prior_art") == 3
    assert [claim.status for claim in report.claims].count("hypothesis") == 4
    assert report.renderable_findings == ()
    assert report.warnings == ()


def test_current_hypothesis_only_ledger_holds_without_promoting_claims() -> None:
    report = validate_claims_ledger(CURRENT_LEDGER, ROOT)

    gate = release_gate(report, report_status=ReportStatus.CONFIRMATORY)

    assert gate.status is ReleaseClaimsStatus.HOLD_NO_EXPERIMENTAL_FINDINGS
    assert gate.finding_ids == ()
    assert not gate.passed


def test_release_gate_invalid_ledger_holds_before_evidence_checks(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    report = validate_claims_ledger(repository_root / "missing.yaml", repository_root)

    gate = release_gate(report, scientific_freeze=scientific_freeze())

    assert gate.status is ReleaseClaimsStatus.HOLD_INVALID_CLAIMS_LEDGER


def test_release_gate_requires_confirmatory_report_audit_and_hashes(tmp_path: Path) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    artifact_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {"version": "1.0", "claims": [finding_claim(artifact_hash)]}, sort_keys=False
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")

    assert (
        release_gate(report, report_status="hold").status
        is ReleaseClaimsStatus.HOLD_REPORT_NOT_CONFIRMATORY
    )
    assert (
        release_gate(report, experiment_audit_status="HOLD").status
        is ReleaseClaimsStatus.HOLD_EXPERIMENT_AUDIT_NOT_PASS
    )
    assert (
        release_gate(report, experiment_audit_hash=None).status
        is ReleaseClaimsStatus.HOLD_MISSING_EXPERIMENT_AUDIT_HASH
    )
    assert (
        release_gate(report, analysis_dataset_hash=None).status
        is ReleaseClaimsStatus.HOLD_MISSING_ANALYSIS_DATASET_HASH
    )


@pytest.mark.parametrize(
    "finding_disposition", ["supports", "does_not_support", "mixed", "inconclusive"]
)
def test_release_gate_rejects_skeletal_sealed_analysis(
    tmp_path: Path, finding_disposition: str
) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    artifact_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "claims": [finding_claim(artifact_hash, finding_disposition)],
            },
            sort_keys=False,
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")

    gate = supported_release_gate(report)

    assert gate.status is ReleaseClaimsStatus.HOLD_CROSS_HASH_SEALED_ANALYSIS_MISMATCH
    assert gate.finding_ids == ("EXP-001",)
    assert gate.passed is False


def test_release_gate_requires_a_scientific_freeze_binding(tmp_path: Path) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "claims": [
                    finding_claim("sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest())
                ],
            },
            sort_keys=False,
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")

    gate = release_gate(report, sealed_analysis=sealed_analysis())

    assert gate.status is ReleaseClaimsStatus.HOLD_MISSING_SCIENTIFIC_FREEZE


@pytest.mark.parametrize(
    ("disposition", "has_exclusion", "expected"),
    [
        (
            "supports",
            True,
            ReleaseClaimsStatus.HOLD_CLAIM_EXCLUSIONS_REQUIRE_NON_SUPPORT,
        ),
        ("supports", False, ReleaseClaimsStatus.PASS),
        ("does_not_support", True, ReleaseClaimsStatus.PASS),
    ],
)
def test_profile2_support_requires_zero_technical_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
    has_exclusion: bool,
    expected: ReleaseClaimsStatus,
) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "claims": [
                    finding_claim(
                        "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        disposition,
                    )
                ],
            },
            sort_keys=False,
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")
    analysis = sealed_analysis()
    if has_exclusion:
        analysis = replace(
            analysis,
            dataset=AnalysisDataset((), (cast(Any, object()),), ()),
        )
    monkeypatch.setattr(
        claims_module,
        "_sealed_claim_evidence",
        lambda *_: {
            ("E1", "slope", "pooled", "completion_under_policy", None): ((-0.2, -0.3, -0.11),)
        },
    )

    status = claims_module._claim_support_status(
        report.renderable_findings, analysis, HASH, HASH, scientific_freeze()
    )

    assert status is expected


@pytest.mark.parametrize("field", ["stage_a", "stage_b"])
def test_malformed_exact_sealed_analysis_is_held_without_attribute_error(field: str) -> None:
    sealed = sealed_analysis()

    assert not sealed_analysis_is_release_complete(replace(sealed, **{field: cast(Any, object())}))


def test_malformed_scientific_freeze_input_shape_holds_without_unpacking_error() -> None:
    malformed = replace(scientific_freeze(), input_hashes=(("valid", HASH), ("missing",)))

    assert (
        claims_module._claim_support_status((), object(), HASH, HASH, malformed)
        is ReleaseClaimsStatus.HOLD_MISSING_SCIENTIFIC_FREEZE
    )


@pytest.mark.parametrize(
    "designs",
    [
        (("E1", 1, 1),),
        tuple((member, 0, 1, 1) for member in MULTIPLICITY_MEMBERS),
        tuple(reversed(scientific_freeze().member_execution_designs)),
    ],
)
def test_malformed_or_reordered_execution_design_holds_without_type_error(
    designs: object,
) -> None:
    malformed = replace(scientific_freeze(), member_execution_designs=cast(Any, designs))

    assert (
        claims_module._claim_support_status((), object(), HASH, HASH, malformed)
        is ReleaseClaimsStatus.HOLD_MISSING_SCIENTIFIC_FREEZE
    )


def test_release_claim_support_holds_when_prebuilt_sealed_design_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "claims": [
                    finding_claim("sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest())
                ],
            },
            sort_keys=False,
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")
    freeze = scientific_freeze()
    wrong_design = list(freeze.member_execution_designs)
    wrong_design[0] = (wrong_design[0][0], 2, 1, 1)
    sealed = replace(sealed_analysis(), execution_design_projection=tuple(wrong_design))
    monkeypatch.setattr(
        claims_module,
        "_sealed_claim_evidence",
        lambda *_: {
            ("E1", "slope", "pooled", "completion_under_policy", None): ((-0.2, -0.3, -0.11),)
        },
    )

    status = claims_module._claim_support_status(
        report.renderable_findings, sealed, HASH, HASH, freeze
    )

    assert status is ReleaseClaimsStatus.HOLD_SEALED_ANALYSIS_EXECUTION_DESIGN_MISMATCH


def test_release_claim_support_accepts_matching_prebuilt_sealed_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claims_module, "_sealed_claim_evidence", lambda *_: {})

    status = claims_module._claim_support_status(
        (), sealed_analysis(), HASH, HASH, scientific_freeze()
    )

    assert status is ReleaseClaimsStatus.PASS


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manifest_hash", None),
        ("analysis_plan_sha256", None),
        ("multiplicity_members", ("wrong-member",)),
    ],
)
def test_malformed_scientific_freeze_scalars_hold_without_type_error(
    field: str, value: object
) -> None:
    malformed = replace(scientific_freeze(), **{field: cast(Any, value)})

    assert (
        claims_module._claim_support_status((), object(), HASH, HASH, malformed)
        is ReleaseClaimsStatus.HOLD_MISSING_SCIENTIFIC_FREEZE
    )


def test_nested_malformed_stage_a_holds_without_iteration_error() -> None:
    sealed = sealed_analysis()
    malformed_stage_a = replace(sealed.stage_a, raw_counts=cast(Any, None))

    assert not sealed_analysis_is_release_complete(replace(sealed, stage_a=malformed_stage_a))


@pytest.mark.parametrize(
    ("e1", "expected"),
    [
        ((0.2, 0.11, 0.3), ReleaseClaimsStatus.HOLD_CLAIM_DIRECTION_MISMATCH),
        ((-0.05, -0.2, -0.01), ReleaseClaimsStatus.HOLD_CLAIM_EFFECT_NOT_MATERIAL),
        ((-0.2, -0.3, -0.05), ReleaseClaimsStatus.HOLD_CLAIM_INTERVAL_INCONCLUSIVE),
    ],
)
def test_release_gate_holds_wrong_sign_submaterial_and_interval_spanning_estimands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    e1: tuple[float, float, float],
    expected: ReleaseClaimsStatus,
) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    artifact_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {"version": "1.0", "claims": [finding_claim(artifact_hash)]}, sort_keys=False
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")
    monkeypatch.setattr(
        claims_module,
        "_sealed_claim_evidence",
        lambda *_: {("E1", "slope", "pooled", "completion_under_policy", None): (e1,)},
    )

    status = claims_module._claim_support_status(
        report.renderable_findings,
        sealed_analysis(),
        HASH,
        HASH,
        scientific_freeze(),
    )

    assert status is expected


@pytest.mark.parametrize(
    "binding",
    [
        lambda: replace(scientific_freeze(), interval_rule=""),
        lambda: replace(
            scientific_freeze(), multiplicity_method="unadjusted_intervals_no_pvalue_threshold"
        ),
        lambda: replace(scientific_freeze(), power_precision_evidence_sha256="sha256:" + "b" * 64),
    ],
)
def test_release_claim_support_holds_without_consistent_simultaneous_interval_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: Callable[[], ScientificFreezeBinding],
) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "claims": [
                    finding_claim("sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest())
                ],
            },
            sort_keys=False,
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")
    monkeypatch.setattr(
        claims_module,
        "_sealed_claim_evidence",
        lambda *_: {
            ("E1", "slope", "pooled", "completion_under_policy", None): ((-0.2, -0.3, -0.11),)
        },
    )

    status = claims_module._claim_support_status(
        report.renderable_findings, object(), HASH, HASH, binding()
    )

    assert status is ReleaseClaimsStatus.HOLD_MISSING_SCIENTIFIC_FREEZE


def test_release_gate_requires_sealed_analysis_not_authored_numeric_evidence(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    artifact_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {"version": "1.0", "claims": [finding_claim(artifact_hash)]}, sort_keys=False
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")

    gate = release_gate(report, scientific_freeze=scientific_freeze())

    assert gate.status is ReleaseClaimsStatus.HOLD_MISSING_SEALED_ANALYSIS_EVIDENCE
    assert not gate.passed


def test_finding_mapping_must_bind_the_canonical_profile(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    artifact = repository_root / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    artifact_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    claim = finding_claim(artifact_hash)
    claim["claim_mapping_sha256"] = HASH
    path = repository_root / "claims.yaml"
    path.write_text(yaml.safe_dump({"version": "1.0", "claims": [claim]}, sort_keys=False))

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "INVALID_CLAIM_ESTIMAND_MAPPING" in codes(report)


@pytest.mark.parametrize(
    "limitations",
    [
        (),
        ("Synthetic domains limit external validity.",) * 3,
        (
            " ",
            "The selected model provider and version bound these observations.",
            "Technical exclusions and audit coverage bound the included evidence.",
        ),
        (
            "Synthetic domains limit external validity.",
            "Selected model scope is limited.",
            "Audit coverage is bounded.",
        ),
    ],
)
def test_release_gate_requires_distinct_semantic_limitations(
    tmp_path: Path, limitations: tuple[str, ...]
) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    artifact_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {"version": "1.0", "claims": [finding_claim(artifact_hash)]}, sort_keys=False
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")

    gate = supported_release_gate(report, limitations=limitations)

    assert gate.status is ReleaseClaimsStatus.HOLD_CROSS_HASH_SEALED_ANALYSIS_MISMATCH


def test_release_limitations_use_concepts_not_a_required_phrase(tmp_path: Path) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"release evidence")
    artifact_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {"version": "1.0", "claims": [finding_claim(artifact_hash)]}, sort_keys=False
        )
    )
    report = validate_claims_ledger(path, tmp_path / "repository")

    gate = supported_release_gate(
        report,
        limitations=(
            "Simulated domains constrain generalizability to real-world workplaces.",
            "Configured provider/model versions bound the reported behavior.",
            "Omitted cases remain within the predeclared exclusion audit boundaries.",
        ),
    )

    assert gate.status is ReleaseClaimsStatus.HOLD_CROSS_HASH_SEALED_ANALYSIS_MISMATCH


@pytest.mark.parametrize(
    "text",
    [
        "version: !custom 1.0\nclaims: []\n",
        "version: '1.0'\nclaims:\n  - id: EXP-001\n    id: EXP-002\n",
    ],
)
def test_malformed_or_duplicate_yaml_fails_closed(tmp_path: Path, text: str) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    path = repository_root / "claims.yaml"
    path.write_text(text)

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert report.claims == ()
    assert report.renderable_findings == ()
    assert report.violations


def test_duplicate_claim_ids_fail_closed_and_sort_deterministically(tmp_path: Path) -> None:
    path, repository_root = write_ledger(
        tmp_path,
        {"version": "1.0", "claims": [hypothesis_claim(), hypothesis_claim()]},
    )

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "DUPLICATE_CLAIM_ID" in codes(report)
    assert report.renderable_findings == ()
    assert tuple(report.violations) == tuple(
        sorted(
            report.violations,
            key=lambda item: (item.location, item.code, item.claim_id or "", item.detail),
        )
    )


@pytest.mark.parametrize(
    "data",
    [
        {"version": "1.0", "claims": [], "unexpected": True},
        {"version": 1.0, "claims": []},
        {"version": "2.0", "claims": []},
        {"version": "1.0", "claims": "not-a-list"},
        {"version": "1.0", "claims": [{**prior_claim(), "unknown": "field"}]},
        {"version": "1.0", "claims": [{**prior_claim(), "status": 7}]},
    ],
)
def test_unknown_root_claim_fields_and_types_fail_closed(tmp_path: Path, data: object) -> None:
    path, repository_root = write_ledger(tmp_path, data)

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert report.renderable_findings == ()
    assert report.violations


def test_missing_ledger_is_a_typed_invalid_report(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    report = validate_claims_ledger(repository_root / "missing.yaml", repository_root)

    assert not report.valid
    assert "LEDGER_IO" in codes(report)


@pytest.mark.parametrize("status", ["supported", "supported_experimental"])
def test_ambiguous_supported_statuses_are_rejected(tmp_path: Path, status: str) -> None:
    claim = hypothesis_claim()
    claim["status"] = status
    path, repository_root = write_ledger(tmp_path, {"version": "1.0", "claims": [claim]})

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "AMBIGUOUS_STATUS" in codes(report)


def test_lifecycle_omissions_and_prior_art_forbidden_fields_fail(tmp_path: Path) -> None:
    prior = prior_claim()
    prior["evidence"] = ["Source A", "Source A"]
    prior["artifact"] = "artifacts/results/claim.json"
    hypothesis = hypothesis_claim("EXP-002")
    hypothesis.pop("artifact")
    finding = hypothesis_claim("EXP-003")
    finding["status"] = "finding"
    path, repository_root = write_ledger(
        tmp_path,
        {"version": "1.0", "claims": [prior, hypothesis, finding]},
    )

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert {
        "INVALID_PRIOR_ART",
        "MISSING_ARTIFACT",
        "MISSING_ARTIFACT_SHA256",
        "MISSING_FINDING_DISPOSITION",
        "MISSING_CLAIM_ESTIMAND_MAPPING",
    } <= codes(report)


def test_absent_hypothesis_artifact_is_accepted_but_never_renderable(tmp_path: Path) -> None:
    path, repository_root = write_ledger(
        tmp_path,
        {"version": "1.0", "claims": [hypothesis_claim()]},
    )

    report = validate_claims_ledger(path, repository_root)

    assert report.valid
    assert report.renderable_findings == ()


@pytest.mark.parametrize(
    "finding_disposition", ["supports", "does_not_support", "mixed", "inconclusive"]
)
def test_valid_findings_render_for_all_explicit_dispositions(
    tmp_path: Path, finding_disposition: str
) -> None:
    artifact = tmp_path / "repository" / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'{ "raw": "artifact" }\n')
    artifact_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    path = tmp_path / "repository" / "claims.yaml"
    path.write_text(
        yaml.safe_dump(
            {"version": "1.0", "claims": [finding_claim(artifact_hash, finding_disposition)]},
            sort_keys=False,
        )
    )

    report = validate_claims_ledger(path, tmp_path / "repository")

    assert report.valid
    assert [claim.id for claim in report.renderable_findings] == ["EXP-001"]


@pytest.mark.parametrize("artifact_sha256", [None, "sha256:" + "A" * 64])
def test_findings_require_a_lowercase_hash(tmp_path: Path, artifact_sha256: str | None) -> None:
    claim = hypothesis_claim()
    claim["status"] = "finding"
    claim["finding_disposition"] = "supports"
    if artifact_sha256 is not None:
        claim["artifact_sha256"] = artifact_sha256
    path, repository_root = write_ledger(tmp_path, {"version": "1.0", "claims": [claim]})

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "MISSING_ARTIFACT_SHA256" in codes(report) or "INVALID_ARTIFACT_SHA256" in codes(report)


def test_finding_rejects_missing_and_mismatched_artifacts(tmp_path: Path) -> None:
    missing_hash = "sha256:" + "0" * 64
    missing_path, missing_root = write_ledger(
        tmp_path / "missing",
        {"version": "1.0", "claims": [finding_claim(missing_hash)]},
    )
    mismatch_root = tmp_path / "mismatch" / "repository"
    artifact = mismatch_root / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"actual")
    mismatch_path = mismatch_root / "claims.yaml"
    mismatch_path.write_text(
        yaml.safe_dump(
            {"version": "1.0", "claims": [finding_claim("sha256:" + "1" * 64)]},
            sort_keys=False,
        )
    )

    missing_report = validate_claims_ledger(missing_path, missing_root)
    mismatch_report = validate_claims_ledger(mismatch_path, mismatch_root)

    assert "MISSING_ARTIFACT" in codes(missing_report)
    assert "ARTIFACT_HASH_MISMATCH" in codes(mismatch_report)


def test_hypothesis_hash_requires_existing_matching_artifact(tmp_path: Path) -> None:
    claim = hypothesis_claim()
    claim["artifact_sha256"] = "sha256:" + "0" * 64
    path, repository_root = write_ledger(tmp_path, {"version": "1.0", "claims": [claim]})

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "MISSING_ARTIFACT" in codes(report)


@pytest.mark.parametrize(
    "artifact",
    [
        "../outside.json",
        "artifacts/../outside.json",
        "/tmp/outside.json",
        "artifacts\\results\\claim.json",
        "artifacts//results/claim.json",
        "./artifacts/results/claim.json",
        "artifacts/results/\x00claim.json",
        "artifacts/results/~claim.json",
        "C:/temp/claim.json",
        "artifacts/results/claim:one.json",
    ],
)
def test_unsafe_artifact_paths_are_rejected(tmp_path: Path, artifact: str) -> None:
    claim = hypothesis_claim()
    claim["artifact"] = artifact
    path, repository_root = write_ledger(tmp_path, {"version": "1.0", "claims": [claim]})

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "UNSAFE_ARTIFACT_PATH" in codes(report)


@pytest.mark.parametrize("finding_disposition", [None, "supports"])
def test_hypotheses_forbid_any_present_finding_disposition(
    tmp_path: Path, finding_disposition: object
) -> None:
    claim = hypothesis_claim()
    claim["finding_disposition"] = finding_disposition
    path, repository_root = write_ledger(tmp_path, {"version": "1.0", "claims": [claim]})

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "FORBIDDEN_FINDING_DISPOSITION" in codes(report)


@pytest.mark.parametrize("status", ["supported_prior_art", "hypothesis", "finding"])
@pytest.mark.parametrize("evidence", [None, "not-a-list", 1, ["Source A", "Source A"]])
def test_present_invalid_or_duplicate_evidence_is_never_silently_normalized(
    tmp_path: Path, status: str, evidence: object
) -> None:
    claim = prior_claim() if status == "supported_prior_art" else hypothesis_claim()
    claim["status"] = status
    claim["evidence"] = evidence
    path, repository_root = write_ledger(tmp_path, {"version": "1.0", "claims": [claim]})

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "INVALID_EVIDENCE" in codes(report)


def test_hypothesis_wrong_type_artifact_sha256_is_not_treated_as_absent(tmp_path: Path) -> None:
    claim = hypothesis_claim()
    claim["artifact_sha256"] = 1
    path, repository_root = write_ledger(tmp_path, {"version": "1.0", "claims": [claim]})

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    violation = next(
        violation for violation in report.violations if violation.code == "INVALID_ARTIFACT_SHA256"
    )
    assert violation.location == "claims[EXP-001].artifact_sha256"
    assert violation.detail


def test_finding_with_present_null_evidence_is_invalid_and_nonrenderable(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    artifact = repository_root / "artifacts" / "results" / "claim.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"finding artifact")
    artifact_sha256 = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    claim = finding_claim(artifact_sha256)
    claim["evidence"] = None
    path = repository_root / "claims.yaml"
    path.write_text(yaml.safe_dump({"version": "1.0", "claims": [claim]}, sort_keys=False))

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "INVALID_EVIDENCE" in codes(report)
    assert report.renderable_findings == ()


def test_existing_directory_cannot_satisfy_unhashed_hypothesis_artifact(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    artifact = repository_root / "artifacts" / "results" / "claim.json"
    artifact.mkdir(parents=True)
    path = repository_root / "claims.yaml"
    path.write_text(
        yaml.safe_dump({"version": "1.0", "claims": [hypothesis_claim()]}, sort_keys=False)
    )

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "INVALID_ARTIFACT" in codes(report)


def test_inside_and_dangling_symlink_components_are_rejected(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    real = repository_root / "artifacts" / "real"
    real.mkdir(parents=True)
    (real / "claim.json").write_bytes(b"artifact")
    (repository_root / "artifacts" / "inside").symlink_to(real, target_is_directory=True)
    (repository_root / "artifacts" / "dangling").symlink_to(
        repository_root / "missing", target_is_directory=True
    )
    claims = [hypothesis_claim("EXP-001"), hypothesis_claim("EXP-002")]
    claims[0]["artifact"] = "artifacts/inside/claim.json"
    claims[1]["artifact"] = "artifacts/dangling/claim.json"
    path = repository_root / "claims.yaml"
    path.write_text(yaml.safe_dump({"version": "1.0", "claims": claims}, sort_keys=False))

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "SYMLINK_ARTIFACT_PATH" in codes(report)


def test_command_injection_and_id_mismatch_never_execute_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"subprocess execution attempted: {args!r} {kwargs!r}")

    monkeypatch.setattr(subprocess, "run", forbidden)
    claim = hypothesis_claim()
    claim["evidence_command"] = "shadowskillbench report claim EXP-002; touch should-not-run"
    path, repository_root = write_ledger(tmp_path, {"version": "1.0", "claims": [claim]})

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert "INVALID_EVIDENCE_COMMAND" in codes(report)


def test_partial_reports_are_sorted_and_never_render_findings_when_invalid(tmp_path: Path) -> None:
    claim_a = hypothesis_claim("EXP-002")
    claim_a["evidence_command"] = "wrong"
    claim_b = prior_claim("PRIOR-002")
    claim_b["scope"] = ""
    path, repository_root = write_ledger(
        tmp_path,
        {"version": "1.0", "claims": [claim_a, claim_b]},
    )

    report = validate_claims_ledger(path, repository_root)

    assert not report.valid
    assert [claim.id for claim in report.claims] == ["EXP-002", "PRIOR-002"]
    assert report.renderable_findings == ()
    assert tuple(report.violations) == tuple(
        sorted(
            report.violations,
            key=lambda item: (item.location, item.code, item.claim_id or "", item.detail),
        )
    )
