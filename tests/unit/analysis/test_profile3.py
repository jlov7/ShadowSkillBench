from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from shadowskillbench.analysis.dataset import AnalysisDataset
from shadowskillbench.analysis.profile import (
    ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
    ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
    ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
    STAGE_A_METHOD,
    is_confirmatory_binding,
    simultaneous_profile_hash,
)
from shadowskillbench.analysis.stage_a import StageAAnalysisError, _interval, analyze_stage_a
from shadowskillbench.analysis.stage_b import StageBAnalysisError, analyze_stage_b
from shadowskillbench.protocol.scientific_freeze import (
    MULTIPLICITY_MEMBERS,
    SCIENTIFIC_FREEZE_INPUTS,
    ScientificFreezeBinding,
)
from tests.golden.analysis.test_stage_a_synthetic import _synthetic_dataset as stage_a_dataset
from tests.unit.analysis.test_stage_b import _dataset as stage_b_dataset

HASH = "sha256:" + "a" * 64


def _freeze() -> ScientificFreezeBinding:
    return ScientificFreezeBinding(
        manifest_hash=HASH,
        input_hashes=tuple((item.path, HASH) for item in SCIENTIFIC_FREEZE_INPUTS),
        analysis_plan_sha256=HASH,
        power_precision_plan_sha256=HASH,
        power_precision_evidence_sha256=HASH,
        analysis_profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
        multiplicity_family="E1-E9-primary-components-16",
        multiplicity_members=MULTIPLICITY_MEMBERS,
        multiplicity_scope="primary_confirmatory_estimands",
        multiplicity_method="bonferroni_simultaneous_bootstrap_intervals",
        interval_rule="simultaneous_95pct_ci_entirely_beyond_material_threshold",
        familywise_confidence_level=0.95,
        material_threshold=0.1,
        bootstrap_replicates=ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
        bootstrap_seed=ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
        member_execution_designs=tuple(
            (
                member,
                40
                if member.startswith(("E1:", "E2:", "E3:", "E4:", "E5:"))
                else 10
                if member.startswith(("E6:", "E7:"))
                else 50,
                30 if member.startswith(("E1:", "E2:", "E3:", "E4:", "E5:")) else 6,
                3,
            )
            for member in MULTIPLICITY_MEMBERS
        ),
    )


def test_profile3_bonferroni_quantiles_are_wider_and_tail_resolved() -> None:
    values = list(range(ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES))
    per_tail_alpha = (1.0 - 0.95) / (2 * len(MULTIPLICITY_MEMBERS))

    ordinary = _interval(values)
    simultaneous = _interval(values, per_tail_alpha)

    assert ordinary == (999, 38_999)
    assert simultaneous == (62, 39_936)
    assert simultaneous[0] < ordinary[0] < ordinary[1] < simultaneous[1]
    assert int(ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES * per_tail_alpha) >= 50


def test_profile3_rejects_legacy_draw_count_or_seed() -> None:
    freeze = _freeze()
    profile_hash = simultaneous_profile_hash(
        HASH,
        profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
    )
    arguments = {
        "stage": "A",
        "bootstrap_replicates": ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "method": STAGE_A_METHOD,
        "classification": "CONFIRMATORY",
        "profile": ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
        "profile_hash": profile_hash,
        "statistics_configuration_sha256": HASH,
        "scientific_freeze": freeze,
    }

    assert is_confirmatory_binding(**arguments)
    assert not is_confirmatory_binding(**{**arguments, "bootstrap_replicates": 999})
    assert not is_confirmatory_binding(**{**arguments, "bootstrap_seed": 0})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manifest_hash", None),
        ("analysis_plan_sha256", None),
        ("power_precision_plan_sha256", "not-a-hash"),
        ("power_precision_evidence_sha256", None),
        ("member_execution_designs", (("E1", 1, 1, 1),)),
        ("multiplicity_method", "unadjusted"),
    ],
)
def test_profile3_rejects_malformed_scientific_freeze(field: str, value: object) -> None:
    freeze = replace(_freeze(), **{field: cast(Any, value)})
    profile_hash = simultaneous_profile_hash(
        HASH,
        profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
    )

    assert not is_confirmatory_binding(
        stage="A",
        bootstrap_replicates=ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
        bootstrap_seed=ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
        confidence_level=0.95,
        method=STAGE_A_METHOD,
        classification="CONFIRMATORY",
        profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
        profile_hash=profile_hash,
        statistics_configuration_sha256=HASH,
        scientific_freeze=freeze,
    )


def test_profile3_rejects_malformed_execution_design_binding() -> None:
    freeze = _freeze()
    hashes = dict(freeze.input_hashes)
    hashes["src/shadowskillbench/protocol/power_precision_design.py"] = "not-a-hash"
    malformed = replace(freeze, input_hashes=tuple(hashes.items()))
    profile_hash = simultaneous_profile_hash(
        HASH,
        profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
    )

    assert not is_confirmatory_binding(
        stage="A",
        bootstrap_replicates=ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
        bootstrap_seed=ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
        confidence_level=0.95,
        method=STAGE_A_METHOD,
        classification="CONFIRMATORY",
        profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
        profile_hash=profile_hash,
        statistics_configuration_sha256=HASH,
        scientific_freeze=malformed,
    )


@pytest.mark.parametrize(
    ("analyze", "error"),
    [(analyze_stage_a, StageAAnalysisError), (analyze_stage_b, StageBAnalysisError)],
)
def test_profile3_stage_analysis_rejects_legacy_draw_count_before_estimation(
    analyze, error
) -> None:
    with pytest.raises(error, match="invalid simultaneous"):
        analyze(
            AnalysisDataset(rows=(), exclusions=(), aggregates=()),
            bootstrap_replicates=999,
            bootstrap_seed=ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
            scientific_freeze=_freeze(),
        )


@pytest.mark.parametrize(
    ("analyze", "dataset", "error"),
    [
        (analyze_stage_a, stage_a_dataset, StageAAnalysisError),
        (analyze_stage_b, stage_b_dataset, StageBAnalysisError),
    ],
)
def test_profile3_stage_analysis_rejects_frozen_execution_design_mismatch(
    analyze: object, dataset: object, error: type[ValueError]
) -> None:
    mismatched = replace(
        _freeze(),
        member_execution_designs=tuple((member, 1, 1, 1) for member in MULTIPLICITY_MEMBERS),
    )
    with pytest.raises(error, match="HOLD_FROZEN_EXECUTION_DESIGN_MISMATCH"):
        analyze(  # type: ignore[operator]
            dataset(),  # type: ignore[operator]
            bootstrap_replicates=ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
            bootstrap_seed=ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
            scientific_freeze=mismatched,
        )


def test_profile3_real_analysis_kernel_uses_the_frozen_40k_profile() -> None:
    """Kernel-only regression: it does not construct a report, release, or claim."""

    freeze = _freeze()
    stage_a = analyze_stage_a(
        stage_a_dataset(),
        bootstrap_replicates=ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
        bootstrap_seed=ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
        scientific_freeze=freeze,
    )
    stage_b = analyze_stage_b(
        stage_b_dataset(),
        bootstrap_replicates=ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
        bootstrap_seed=ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
        scientific_freeze=freeze,
    )

    assert stage_a.estimands and stage_b.e6_unsafe_imitation
    for analysis in (stage_a, stage_b):
        assert analysis.inference.classification == "CONFIRMATORY"
        assert analysis.inference.profile == ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE
        assert analysis.inference.bootstrap_replicates == ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES
        assert analysis.inference.bootstrap_seed == ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED
