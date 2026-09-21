"""Immutable analysis-profile bindings for confirmatory evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from shadowskillbench.core.hashing import sha256_ref

CONFIRMATORY_ANALYSIS_PROFILE = "SSB-CONFIRMATORY-STATISTICS1"
SIMULTANEOUS_ANALYSIS_PROFILE = "SSB-CONFIRMATORY-STATISTICS2-SIMULTANEOUS"
ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE = "SSB-CONFIRMATORY-STATISTICS3-BONFERRONI"
NONCONFIRMATORY_ANALYSIS_PROFILE = "SSB-NONCONFIRMATORY-STATISTICS1"
STATISTICS_CONFIGURATION_SHA256 = (
    "sha256:a1aa62d66d29569d3641d94676d71e8050b7ba57d0c914e077b77fa58fb3f120"
)
CONFIRMATORY_BOOTSTRAP_REPLICATES = 999
CONFIRMATORY_BOOTSTRAP_SEED = 0
CONFIRMATORY_CONFIDENCE_LEVEL = 0.95
ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES = 40_000
ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED = 104_731
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")

STAGE_A_METHOD = (
    "Binomial logistic GLM with logit link for skill-bearing A2-A5 rows; "
    "two-way cluster-robust covariance uses skill_bundle_id and held_out_case_id. "
    "Response-curve and contrast intervals use a deterministic two-way stratified "
    "pigeonhole bootstrap that resamples held-out cases and, within domain and "
    "contamination ratio, skill bundles independently."
)
STAGE_B_METHOD = (
    "Raw-rate and contrast estimates with a deterministic two-way stratified "
    "pigeonhole bootstrap that resamples held-out cases within authority class "
    "and R75 skill bundles within domain."
)


def confirmatory_profile_projection() -> dict[str, object]:
    """Return the versioned profile that is frozen by ``config/statistics.yaml``."""

    return {
        "profile": CONFIRMATORY_ANALYSIS_PROFILE,
        "statistics_configuration_sha256": STATISTICS_CONFIGURATION_SHA256,
        "bootstrap": {
            "method": "deterministic_two_way_stratified_pigeonhole",
            "replicates": CONFIRMATORY_BOOTSTRAP_REPLICATES,
            "seed": CONFIRMATORY_BOOTSTRAP_SEED,
            "confidence_level": CONFIRMATORY_CONFIDENCE_LEVEL,
        },
        "stage_methods": {"A": STAGE_A_METHOD, "B": STAGE_B_METHOD},
    }


CONFIRMATORY_ANALYSIS_PROFILE_HASH = sha256_ref(confirmatory_profile_projection())


@dataclass(frozen=True, slots=True)
class AnalysisProfileBinding:
    """Profile metadata carried by every analysis output."""

    classification: Literal["CONFIRMATORY", "NONCONFIRMATORY"]
    profile: str
    profile_hash: str
    statistics_configuration_sha256: str | None


def simultaneous_profile_hash(
    statistics_configuration_sha256: str,
    *,
    profile: str = SIMULTANEOUS_ANALYSIS_PROFILE,
) -> str:
    """Bind the successor inference label to its exact statistics configuration."""

    return sha256_ref(
        {
            "profile": profile,
            "statistics_configuration_sha256": statistics_configuration_sha256,
        }
    )


def frozen_member_execution_designs(scientific_freeze: object) -> dict[str, tuple[int, int, int]]:
    """Return the exact frozen primary-analysis design, or fail closed."""

    try:
        from shadowskillbench.protocol.scientific_freeze import (
            MULTIPLICITY_MEMBERS,
            ScientificFreezeBinding,
        )
        from shadowskillbench.protocol.scientific_freeze_v4 import (
            MULTIPLICITY_MEMBERS_V4,
            ScientificFreezeV4Binding,
        )

        if type(scientific_freeze) is ScientificFreezeV4Binding:
            members = MULTIPLICITY_MEMBERS_V4
        elif type(scientific_freeze) is ScientificFreezeBinding:
            members = MULTIPLICITY_MEMBERS
        else:
            raise ValueError("scientific freeze must be an exact V3 or V4 binding")

        designs = getattr(scientific_freeze, "member_execution_designs", None)
        if (
            type(designs) is not tuple
            or len(designs) != len(members)
            or any(
                type(item) is not tuple
                or len(item) != 4
                or type(item[0]) is not str
                or any(type(value) is not int or value < 1 for value in item[1:])
                for item in designs
            )
            or tuple(item[0] for item in designs) != members
        ):
            raise ValueError("execution design is invalid")
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("execution design is invalid") from error
    result: dict[str, tuple[int, int, int]] = {}
    for member, cases, bundles, repetitions in designs:
        result[member] = (cases, bundles, repetitions)
    return result


def scientific_freeze_version(scientific_freeze: object) -> Literal["V3", "V4"] | None:
    """Return the exact supported freeze generation; never infer it structurally."""

    from shadowskillbench.protocol.scientific_freeze import ScientificFreezeBinding
    from shadowskillbench.protocol.scientific_freeze_v4 import ScientificFreezeV4Binding

    if type(scientific_freeze) is ScientificFreezeBinding:
        return "V3"
    if type(scientific_freeze) is ScientificFreezeV4Binding:
        return "V4"
    return None


def binding_for(
    *,
    stage: Literal["A", "B"],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
    method: str,
) -> AnalysisProfileBinding:
    """Bind exact frozen settings, or label the output as non-confirmatory."""

    expected_method = STAGE_A_METHOD if stage == "A" else STAGE_B_METHOD
    if (
        bootstrap_replicates == CONFIRMATORY_BOOTSTRAP_REPLICATES
        and bootstrap_seed == CONFIRMATORY_BOOTSTRAP_SEED
        and confidence_level == CONFIRMATORY_CONFIDENCE_LEVEL
        and method == expected_method
    ):
        return AnalysisProfileBinding(
            classification="CONFIRMATORY",
            profile=CONFIRMATORY_ANALYSIS_PROFILE,
            profile_hash=CONFIRMATORY_ANALYSIS_PROFILE_HASH,
            statistics_configuration_sha256=STATISTICS_CONFIGURATION_SHA256,
        )
    return AnalysisProfileBinding(
        classification="NONCONFIRMATORY",
        profile=NONCONFIRMATORY_ANALYSIS_PROFILE,
        profile_hash=sha256_ref(
            {
                "profile": NONCONFIRMATORY_ANALYSIS_PROFILE,
                "stage": stage,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
                "confidence_level": confidence_level,
                "method": method,
            }
        ),
        statistics_configuration_sha256=None,
    )


def is_confirmatory_binding(
    *,
    stage: Literal["A", "B"],
    bootstrap_replicates: object,
    bootstrap_seed: object,
    confidence_level: object,
    method: object,
    classification: object,
    profile: object,
    profile_hash: object,
    statistics_configuration_sha256: object,
    scientific_freeze: object = None,
) -> bool:
    """Return whether an inference record exactly matches the frozen profile."""

    expected_method = STAGE_A_METHOD if stage == "A" else STAGE_B_METHOD
    common = (
        type(bootstrap_replicates) is int
        and bootstrap_replicates == CONFIRMATORY_BOOTSTRAP_REPLICATES
        and type(bootstrap_seed) is int
        and bootstrap_seed == CONFIRMATORY_BOOTSTRAP_SEED
        and type(confidence_level) is float
        and confidence_level == CONFIRMATORY_CONFIDENCE_LEVEL
        and type(method) is str
        and method == expected_method
        and classification == "CONFIRMATORY"
    )
    legacy = (
        profile == CONFIRMATORY_ANALYSIS_PROFILE
        and profile_hash == CONFIRMATORY_ANALYSIS_PROFILE_HASH
        and statistics_configuration_sha256 == STATISTICS_CONFIGURATION_SHA256
    )
    simultaneous = (
        profile == SIMULTANEOUS_ANALYSIS_PROFILE
        and type(statistics_configuration_sha256) is str
        and _SHA256.fullmatch(statistics_configuration_sha256) is not None
        and profile_hash == simultaneous_profile_hash(statistics_configuration_sha256)
    )
    robust_simultaneous = (
        profile == ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE
        and type(bootstrap_replicates) is int
        and bootstrap_replicates == ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES
        and type(bootstrap_seed) is int
        and bootstrap_seed == ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED
        and type(confidence_level) is float
        and confidence_level == CONFIRMATORY_CONFIDENCE_LEVEL
        and type(method) is str
        and method == expected_method
        and classification == "CONFIRMATORY"
        and type(statistics_configuration_sha256) is str
        and _SHA256.fullmatch(statistics_configuration_sha256) is not None
        and profile_hash
        == simultaneous_profile_hash(
            statistics_configuration_sha256,
            profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
        )
    )
    if scientific_freeze is not None:
        try:
            from shadowskillbench.protocol.scientific_freeze import (
                MULTIPLICITY_MEMBERS,
                SCIENTIFIC_FREEZE_INPUTS,
                ScientificFreezeBinding,
            )
            from shadowskillbench.protocol.scientific_freeze_v4 import (
                MULTIPLICITY_MEMBERS_V4,
                ScientificFreezeV4Binding,
            )

            if type(scientific_freeze) is ScientificFreezeV4Binding:
                if (
                    type(scientific_freeze.manifest_hash) is not str
                    or _SHA256.fullmatch(scientific_freeze.manifest_hash) is None
                    or scientific_freeze.analysis_profile != ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE
                    or type(scientific_freeze.statistics_configuration_sha256) is not str
                    or _SHA256.fullmatch(scientific_freeze.statistics_configuration_sha256) is None
                    or scientific_freeze.multiplicity_members != MULTIPLICITY_MEMBERS_V4
                    or scientific_freeze.material_threshold != 0.1
                    or scientific_freeze.bootstrap_replicates
                    != ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES
                    or scientific_freeze.bootstrap_seed != ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED
                    or tuple(frozen_member_execution_designs(scientific_freeze))
                    != MULTIPLICITY_MEMBERS_V4
                ):
                    return False
                return robust_simultaneous and (
                    scientific_freeze.analysis_profile == profile
                    and scientific_freeze.statistics_configuration_sha256
                    == statistics_configuration_sha256
                )

            if (
                type(scientific_freeze) is not ScientificFreezeBinding
                or type(scientific_freeze.manifest_hash) is not str
                or _SHA256.fullmatch(scientific_freeze.manifest_hash) is None
                or type(scientific_freeze.input_hashes) is not tuple
                or type(scientific_freeze.analysis_plan_sha256) is not str
                or _SHA256.fullmatch(scientific_freeze.analysis_plan_sha256) is None
                or type(scientific_freeze.power_precision_plan_sha256) is not str
                or _SHA256.fullmatch(scientific_freeze.power_precision_plan_sha256) is None
                or type(scientific_freeze.power_precision_evidence_sha256) is not str
                or _SHA256.fullmatch(scientific_freeze.power_precision_evidence_sha256) is None
                or any(
                    type(item) is not tuple
                    or len(item) != 2
                    or type(item[0]) is not str
                    or type(item[1]) is not str
                    or _SHA256.fullmatch(item[1]) is None
                    for item in scientific_freeze.input_hashes
                )
                or tuple(path for path, _ in scientific_freeze.input_hashes)
                != tuple(item.path for item in SCIENTIFIC_FREEZE_INPUTS)
                or scientific_freeze.analysis_plan_sha256
                != dict(scientific_freeze.input_hashes)["protocol/analysis_plan.v3.json"]
                or scientific_freeze.power_precision_plan_sha256
                != dict(scientific_freeze.input_hashes)["protocol/power_precision_plan.v3.json"]
                or scientific_freeze.power_precision_evidence_sha256
                != dict(scientific_freeze.input_hashes)["protocol/power_precision_evidence.v3.json"]
                or any(
                    _SHA256.fullmatch(dict(scientific_freeze.input_hashes)[path]) is None
                    for path in (
                        "config/corpora.yaml",
                        "src/shadowskillbench/experiments/planner.py",
                        "src/shadowskillbench/protocol/power_precision_design.py",
                    )
                )
                or scientific_freeze.multiplicity_family != "E1-E9-primary-components-16"
                or scientific_freeze.multiplicity_members != MULTIPLICITY_MEMBERS
                or scientific_freeze.multiplicity_scope != "primary_confirmatory_estimands"
                or scientific_freeze.multiplicity_method
                != "bonferroni_simultaneous_bootstrap_intervals"
                or scientific_freeze.interval_rule
                != "simultaneous_95pct_ci_entirely_beyond_material_threshold"
                or scientific_freeze.familywise_confidence_level != 0.95
                or scientific_freeze.material_threshold != 0.1
                or scientific_freeze.bootstrap_replicates
                != ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES
                or scientific_freeze.bootstrap_seed != ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED
                or tuple(frozen_member_execution_designs(scientific_freeze)) != MULTIPLICITY_MEMBERS
            ):
                return False
            freeze_profile = getattr(scientific_freeze, "analysis_profile", None)
            freeze_statistics_hash = getattr(
                scientific_freeze, "statistics_configuration_sha256", None
            )
        except (AttributeError, TypeError, ValueError):
            return False
        legacy = False
        simultaneous = simultaneous and (
            freeze_profile == profile and freeze_statistics_hash == statistics_configuration_sha256
        )
        robust_simultaneous = robust_simultaneous and (
            freeze_profile == profile and freeze_statistics_hash == statistics_configuration_sha256
        )
    return (common and (legacy or simultaneous)) or robust_simultaneous


__all__ = [
    "AnalysisProfileBinding",
    "CONFIRMATORY_ANALYSIS_PROFILE",
    "CONFIRMATORY_ANALYSIS_PROFILE_HASH",
    "CONFIRMATORY_BOOTSTRAP_REPLICATES",
    "CONFIRMATORY_BOOTSTRAP_SEED",
    "CONFIRMATORY_CONFIDENCE_LEVEL",
    "NONCONFIRMATORY_ANALYSIS_PROFILE",
    "ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE",
    "ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES",
    "ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED",
    "SIMULTANEOUS_ANALYSIS_PROFILE",
    "STAGE_A_METHOD",
    "STAGE_B_METHOD",
    "STATISTICS_CONFIGURATION_SHA256",
    "binding_for",
    "confirmatory_profile_projection",
    "frozen_member_execution_designs",
    "is_confirmatory_binding",
    "scientific_freeze_version",
    "simultaneous_profile_hash",
]
