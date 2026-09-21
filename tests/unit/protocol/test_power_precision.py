from __future__ import annotations

import pytest

from shadowskillbench.protocol.power_precision import (
    PowerPrecisionSimulationError,
    simulate_power_precision,
)
from shadowskillbench.protocol.power_precision_v4 import simulate_power_precision_v4
from shadowskillbench.protocol.scientific_freeze import MULTIPLICITY_MEMBERS


def test_clustered_power_simulation_is_deterministic_and_uses_both_iccs() -> None:
    scenario = {
        "baseline_rate": 0.5,
        "cluster_icc_held_out_case": 0.01,
        "cluster_icc_skill_bundle": 0.01,
        "contrast_structure": "two_arm_equivalent",
        "effect_size": 0.2,
        "expected_direction": "positive",
        "held_out_case_clusters": 40,
        "repetitions": 3,
        "skill_bundle_clusters": 30,
        "target_threshold": 0.1,
        "variance_multiplier": 1.0,
    }

    first = simulate_power_precision(
        scenario, replicates=10_000, seed=104_732, family_size=len(MULTIPLICITY_MEMBERS)
    )
    second = simulate_power_precision(
        scenario, replicates=10_000, seed=104_732, family_size=len(MULTIPLICITY_MEMBERS)
    )
    no_cluster = simulate_power_precision(
        {**scenario, "cluster_icc_held_out_case": 0.0, "cluster_icc_skill_bundle": 0.0},
        replicates=10_000,
        seed=104_732,
        family_size=len(MULTIPLICITY_MEMBERS),
    )
    lower_boundary = simulate_power_precision(
        {**scenario, "target_threshold": 0.05},
        replicates=10_000,
        seed=104_732,
        family_size=len(MULTIPLICITY_MEMBERS),
    )

    assert first == second
    assert first["expected_interval_width"] > no_cluster["expected_interval_width"]
    assert first["estimated_power"] < lower_boundary["estimated_power"]
    assert 0.0 <= first["estimated_power"] <= 1.0


def test_power_simulation_supports_negative_alternatives_and_contrast_variance() -> None:
    scenario = {
        "baseline_rate": 0.5,
        "cluster_icc_held_out_case": 0.01,
        "cluster_icc_skill_bundle": 0.01,
        "contrast_structure": "two_arm_equivalent",
        "effect_size": -0.2,
        "expected_direction": "negative",
        "held_out_case_clusters": 60,
        "repetitions": 3,
        "skill_bundle_clusters": 40,
        "target_threshold": 0.1,
        "variance_multiplier": 1.0,
    }

    negative = simulate_power_precision(
        scenario, replicates=10_000, seed=104_732, family_size=len(MULTIPLICITY_MEMBERS)
    )
    difference_in_differences = simulate_power_precision(
        {
            **scenario,
            "contrast_structure": "four_arm_difference_in_differences",
            "variance_multiplier": 2.0,
        },
        replicates=10_000,
        seed=104_732,
        family_size=len(MULTIPLICITY_MEMBERS),
    )

    assert negative["estimated_power"] > difference_in_differences["estimated_power"]
    assert (
        negative["expected_interval_width"] < difference_in_differences["expected_interval_width"]
    )


def test_power_simulation_requires_an_alternative_beyond_the_claim_boundary() -> None:
    scenario = {
        "baseline_rate": 0.5,
        "cluster_icc_held_out_case": 0.01,
        "cluster_icc_skill_bundle": 0.01,
        "contrast_structure": "two_arm_equivalent",
        "effect_size": 0.1,
        "expected_direction": "positive",
        "held_out_case_clusters": 40,
        "repetitions": 3,
        "skill_bundle_clusters": 30,
        "target_threshold": 0.1,
        "variance_multiplier": 1.0,
    }

    with pytest.raises(PowerPrecisionSimulationError, match="scenario values"):
        simulate_power_precision(
            scenario,
            replicates=10_000,
            seed=104_732,
            family_size=len(MULTIPLICITY_MEMBERS),
        )


def test_v4_arm_specific_simulation_does_not_assign_bundle_clustering_to_a1() -> None:
    scenario = {
        "alternative_arm_skill_bundle_clusters": 70,
        "baseline_arm_skill_bundle_clusters": 0,
        "baseline_rate": 0.5,
        "cluster_icc_held_out_case": 0.01,
        "cluster_icc_skill_bundle": 0.01,
        "contrast_structure": "two_arm_equivalent",
        "effect_size": -0.2,
        "expected_direction": "negative",
        "held_out_case_clusters": 140,
        "repetitions": 3,
        "skill_bundle_clusters": 70,
        "target_threshold": 0.1,
        "variance_multiplier": 1.0,
    }

    corrected = simulate_power_precision_v4(
        scenario, replicates=20_000, seed=104_732, family_size=len(MULTIPLICITY_MEMBERS)
    )
    incorrectly_clustered = simulate_power_precision_v4(
        {**scenario, "baseline_arm_skill_bundle_clusters": 70},
        replicates=20_000,
        seed=104_732,
        family_size=len(MULTIPLICITY_MEMBERS),
    )

    assert corrected["expected_interval_width"] > incorrectly_clustered["expected_interval_width"]
    assert corrected["estimated_power"] < incorrectly_clustered["estimated_power"]
    assert corrected == simulate_power_precision_v4(
        scenario, replicates=20_000, seed=104_732, family_size=len(MULTIPLICITY_MEMBERS)
    )
