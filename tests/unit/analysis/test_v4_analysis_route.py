from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

import shadowskillbench.analysis.sealed as sealed_module
import shadowskillbench.reporting.report as report_module
from shadowskillbench.analysis.dataset import AnalysisDataset
from shadowskillbench.analysis.profile import (
    ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
    ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
    ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
    STAGE_A_METHOD,
    STAGE_B_METHOD,
    is_confirmatory_binding,
    simultaneous_profile_hash,
)
from shadowskillbench.analysis.sealed import SealedAnalysis, sealed_analysis_projection
from shadowskillbench.analysis.stage_a import _freeze_layout as stage_a_layout
from shadowskillbench.analysis.stage_b import _freeze_layout as stage_b_layout
from shadowskillbench.experiments.planner_v4 import (
    STAGE_A_EPISODE_COUNT,
    STAGE_B_EPISODE_COUNT,
)
from shadowskillbench.protocol.scientific_freeze_v4 import (
    MULTIPLICITY_MEMBERS_V4,
    ScientificFreezeV4Binding,
)
from shadowskillbench.release.reproduce import (
    ReproductionHold,
    _confirmatory_protocol_evidence_v4,
    _selected_scientific_profile,
)
from shadowskillbench.reporting.report import _expected_confirmatory_counts

HASH = "sha256:" + "a" * 64


def _freeze() -> ScientificFreezeV4Binding:
    return ScientificFreezeV4Binding(
        manifest_hash=HASH,
        input_hashes=(),
        analysis_profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
        statistics_configuration_sha256=HASH,
        multiplicity_members=MULTIPLICITY_MEMBERS_V4,
        material_threshold=0.1,
        bootstrap_replicates=ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
        bootstrap_seed=ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
        member_execution_designs=tuple((member, 1, 1, 1) for member in MULTIPLICITY_MEMBERS_V4),
    )


def test_v4_layouts_are_selected_only_by_the_exact_v4_freeze_type() -> None:
    freeze = _freeze()

    assert stage_a_layout(freeze) == (STAGE_A_EPISODE_COUNT, 70, 7)
    assert stage_b_layout(freeze) == (STAGE_B_EPISODE_COUNT, 22, 7)
    assert _expected_confirmatory_counts(freeze) == (
        STAGE_A_EPISODE_COUNT,
        STAGE_B_EPISODE_COUNT,
    )


def test_v4_confirmatory_binding_requires_the_v4_freeze() -> None:
    freeze = _freeze()
    profile_hash = simultaneous_profile_hash(HASH, profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE)

    def assert_bound(stage: Literal["A", "B"], method: str) -> None:
        assert is_confirmatory_binding(
            stage=stage,
            bootstrap_replicates=ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
            bootstrap_seed=ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
            confidence_level=0.95,
            method=method,
            classification="CONFIRMATORY",
            profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
            profile_hash=profile_hash,
            statistics_configuration_sha256=HASH,
            scientific_freeze=freeze,
        )

    assert_bound("A", STAGE_A_METHOD)
    assert_bound("B", STAGE_B_METHOD)


def test_v4_sealed_receipt_carries_the_selected_freeze_manifest(
    monkeypatch: Any,
) -> None:
    freeze = _freeze()
    monkeypatch.setattr(
        sealed_module, "_confirmatory_profile_projection", lambda *_: {"profile": "v4"}
    )
    analysis = SealedAnalysis(
        dataset=AnalysisDataset((), (), ()),
        stage_a=cast(Any, SimpleNamespace()),
        stage_b=cast(Any, SimpleNamespace()),
        audit=cast(Any, SimpleNamespace(plan_hash=HASH, report_hash=HASH)),
        runtime_binding=cast(Any, SimpleNamespace(projection=lambda: {"runtime": "fixture"})),
        execution_design_projection=freeze.member_execution_designs,
        scientific_freeze=freeze,
    )

    receipt = sealed_analysis_projection(analysis)

    assert receipt["profile"] == "SSB-SEALED-ANALYSIS-3"
    assert receipt["scientific_freeze_version"] == "V4"
    assert receipt["scientific_freeze_manifest_sha256"] == freeze.manifest_hash


def test_reproduction_profile_selection_is_version_bound_to_the_manifest_name() -> None:
    assert _selected_scientific_profile(Path("protocol/freeze_manifest.v4.json"), None) == "V4"
    assert _selected_scientific_profile(Path("protocol/freeze_manifest.json"), None) == "V3"


def test_v4_report_renders_v4_required_matrix_counts(monkeypatch: Any) -> None:
    from tests.golden.reporting.test_report import _evidence

    evidence = _evidence()
    evidence = replace(evidence, scientific_freeze=_freeze())
    monkeypatch.setattr(report_module, "_confirmatory_counts", lambda _: (59_640, 18_480, 0))

    rendered = report_module.render_report(evidence)

    assert "59640/59640" in rendered
    assert "18480/18480" in rendered


def test_v4_reproduction_holds_until_full_version_bound_custody(tmp_path: Path) -> None:
    protocol_directory = tmp_path / "protocol"
    protocol_directory.mkdir()
    anchor = tmp_path / "anchor.bin"
    anchor.write_bytes(b"v4 fixture anchor")
    manifest = {
        "anchor_status": "PENDING_V4_CUSTODY_RECEIPT",
        "inputs": [{"path": "config/statistics.v4.yaml", "role": "fixture", "sha256": HASH}],
        "profile": "SSB-PROTOCOL-FREEZE-4",
        "schema_version": "4.0",
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    manifest_path = protocol_directory / "freeze_manifest.v4.json"
    manifest_path.write_bytes(manifest_bytes)
    (protocol_directory / "freeze_manifest.v4.sha256").write_bytes(
        f"{sha256(manifest_bytes).hexdigest()}  freeze_manifest.v4.json\n".encode()
    )

    with pytest.raises(ReproductionHold, match="HOLD_MISSING_V4_CONFIRMATORY_AUTHORIZATION"):
        _confirmatory_protocol_evidence_v4(
            tmp_path,
            manifest_path,
            {
                "path": "anchor.bin",
                "sha256": "sha256:" + sha256(anchor.read_bytes()).hexdigest(),
            },
        )
