"""Deterministic arm-specific power simulation for the forward-only V4 design."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import cast

import numpy as np

SIMULATOR_PROFILE_V4 = "SSB-POWER-PRECISION-SIMULATOR-2-ARM-SPECIFIC"


class PowerPrecisionV4SimulationError(ValueError):
    pass


def simulate_power_precision_v4(
    scenario: dict[str, object],
    *,
    replicates: int,
    seed: int,
    family_size: int,
) -> dict[str, float]:
    """Simulate V4 contrast power with covariance denominators per arm.

    A zero arm-specific bundle count denotes a no-skill arm: it has no
    skill-bundle ICC component and its repeated-observation residual is nested
    only in held-out cases.  V1--V3 keep using ``power_precision.py``.
    """

    fields = {
        "alternative_arm_skill_bundle_clusters",
        "baseline_arm_skill_bundle_clusters",
        "baseline_rate",
        "cluster_icc_held_out_case",
        "cluster_icc_skill_bundle",
        "contrast_structure",
        "effect_size",
        "expected_direction",
        "held_out_case_clusters",
        "repetitions",
        "skill_bundle_clusters",
        "target_threshold",
        "variance_multiplier",
    }
    if set(scenario) != fields:
        raise PowerPrecisionV4SimulationError("scenario shape is invalid")
    numeric = fields - {
        "alternative_arm_skill_bundle_clusters",
        "baseline_arm_skill_bundle_clusters",
        "contrast_structure",
        "expected_direction",
        "held_out_case_clusters",
        "repetitions",
        "skill_bundle_clusters",
    }
    if any(type(scenario[field]) is not float for field in numeric) or any(
        type(scenario[field]) is not int
        for field in {
            "alternative_arm_skill_bundle_clusters",
            "baseline_arm_skill_bundle_clusters",
            "held_out_case_clusters",
            "repetitions",
            "skill_bundle_clusters",
        }
    ):
        raise PowerPrecisionV4SimulationError("scenario types are invalid")
    baseline = cast(float, scenario["baseline_rate"])
    effect = cast(float, scenario["effect_size"])
    contrast_structure = cast(str, scenario["contrast_structure"])
    direction = cast(str, scenario["expected_direction"])
    target = cast(float, scenario["target_threshold"])
    case_icc = cast(float, scenario["cluster_icc_held_out_case"])
    bundle_icc = cast(float, scenario["cluster_icc_skill_bundle"])
    cases = cast(int, scenario["held_out_case_clusters"])
    baseline_bundles = cast(int, scenario["baseline_arm_skill_bundle_clusters"])
    alternative_bundles = cast(int, scenario["alternative_arm_skill_bundle_clusters"])
    bundles = cast(int, scenario["skill_bundle_clusters"])
    repetitions = cast(int, scenario["repetitions"])
    variance_multiplier = cast(float, scenario["variance_multiplier"])
    if (
        replicates < 10_000
        or seed < 0
        or family_size < 1
        or not 0.0 < baseline < 1.0
        or not 0.0 < baseline + effect < 1.0
        or direction not in {"positive", "negative"}
        or contrast_structure not in {"two_arm_equivalent", "four_arm_difference_in_differences"}
        or not 0.0 < target < abs(effect)
        or (direction == "positive" and effect <= 0.0)
        or (direction == "negative" and effect >= 0.0)
        or not 0.0 <= case_icc < 1.0
        or not 0.0 <= bundle_icc < 1.0
        or case_icc + bundle_icc >= 1.0
        or cases < 2
        or bundles < 1
        or baseline_bundles < 0
        or alternative_bundles < 0
        or bundles != max(baseline_bundles, alternative_bundles)
        or repetitions < 1
        or (contrast_structure == "two_arm_equivalent" and variance_multiplier != 1.0)
        or (
            contrast_structure == "four_arm_difference_in_differences"
            and variance_multiplier != 2.0
        )
    ):
        raise PowerPrecisionV4SimulationError("scenario values are invalid")

    def arm_variance(rate: float, arm_bundles: int) -> float:
        unit_variance = rate * (1.0 - rate)
        if arm_bundles == 0:
            return unit_variance * (case_icc / cases + (1.0 - case_icc) / (cases * repetitions))
        return unit_variance * (
            case_icc / cases
            + bundle_icc / arm_bundles
            + (1.0 - case_icc - bundle_icc) / (cases * arm_bundles * repetitions)
        )

    standard_error = sqrt(
        variance_multiplier
        * (
            arm_variance(baseline, baseline_bundles)
            + arm_variance(baseline + effect, alternative_bundles)
        )
    )
    critical = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * family_size))
    rng = np.random.default_rng(seed)
    estimates = effect + rng.normal(0.0, standard_error, size=replicates)
    if direction == "positive":
        successes = int(np.count_nonzero(estimates - critical * standard_error > target))
    else:
        successes = int(np.count_nonzero(estimates + critical * standard_error < -target))
    power = successes / replicates
    return {
        "estimated_power": float(power),
        "expected_interval_width": float(2.0 * critical * standard_error),
        "monte_carlo_standard_error": float(sqrt(power * (1.0 - power) / replicates)),
    }


__all__ = [
    "SIMULATOR_PROFILE_V4",
    "PowerPrecisionV4SimulationError",
    "simulate_power_precision_v4",
]
