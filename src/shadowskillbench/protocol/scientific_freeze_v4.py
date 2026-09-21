"""Exact-byte scientific-freeze verification for the forward-only V4 design."""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, ulp
from pathlib import Path, PurePosixPath

import yaml

from shadowskillbench.protocol.power_precision_design_v4 import (
    PowerPrecisionDesignV4Error,
    execution_design_projection_v4,
)
from shadowskillbench.protocol.power_precision_v4 import (
    SIMULATOR_PROFILE_V4,
    PowerPrecisionV4SimulationError,
    simulate_power_precision_v4,
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CORPORA_PATH = "config/corpora.v4.yaml"
_ANALYSIS_PATH = "protocol/analysis_plan.v4.json"
_POWER_PLAN_PATH = "protocol/power_precision_plan.v4.json"
_POWER_EVIDENCE_PATH = "protocol/power_precision_evidence.v4.json"
_PLANNER_PATH = "src/shadowskillbench/experiments/planner_v4.py"
_DESIGN_PATH = "src/shadowskillbench/protocol/power_precision_design_v4.py"
_SIMULATOR_PATH = "src/shadowskillbench/protocol/power_precision_v4.py"
_STATISTICS_PATH = "config/statistics.v4.yaml"
MULTIPLICITY_MEMBERS_V4 = (
    "E1:slope:pooled:completion_under_policy",
    "E2:slope:pooled:completion_under_policy",
    "E3:slope:pooled:completion_under_policy",
    "E3:average_cup_contrast:pooled:completion_under_policy",
    "E4:average_cup_contrast:pooled:completion_under_policy",
    "E5:average_cup_contrast:pooled:completion_under_policy",
    "E6:rate:pooled:unsafe_imitation:PRACTICE_MATCHES_ACTIVE_POLICY",
    "E6:rate:pooled:unsafe_imitation:PRACTICE_VIOLATES_ACTIVE_POLICY",
    "E6:rate:pooled:unsafe_imitation:APPROVED_SCOPED_EXCEPTION",
    "E6:rate:pooled:unsafe_imitation:POLICY_SUPERSEDED",
    "E6:rate:pooled:unsafe_imitation:UNRESOLVED_AUTHORITY_CONFLICT",
    "E7:rate:pooled:false_enforcement:APPROVED_SCOPED_EXCEPTION",
    "E7:rate:pooled:false_enforcement:POLICY_SUPERSEDED",
    "E8:macro_delta:pooled:authority_resolution_correct",
    "E9:delta:pooled:completion_under_policy",
    "E9:delta:pooled:unsafe_imitation",
)
POWER_MEMBER_SPECS_V4: dict[str, dict[str, str | float]] = {
    "E1:slope:pooled:completion_under_policy": {
        "contrast_structure": "two_arm_equivalent",
        "expected_direction": "negative",
        "variance_multiplier": 1.0,
    },
    "E2:slope:pooled:completion_under_policy": {
        "contrast_structure": "four_arm_difference_in_differences",
        "expected_direction": "positive",
        "variance_multiplier": 2.0,
    },
    "E3:slope:pooled:completion_under_policy": {
        "contrast_structure": "four_arm_difference_in_differences",
        "expected_direction": "positive",
        "variance_multiplier": 2.0,
    },
    "E3:average_cup_contrast:pooled:completion_under_policy": {
        "contrast_structure": "two_arm_equivalent",
        "expected_direction": "positive",
        "variance_multiplier": 1.0,
    },
    "E4:average_cup_contrast:pooled:completion_under_policy": {
        "contrast_structure": "two_arm_equivalent",
        "expected_direction": "negative",
        "variance_multiplier": 1.0,
    },
    "E5:average_cup_contrast:pooled:completion_under_policy": {
        "contrast_structure": "two_arm_equivalent",
        "expected_direction": "negative",
        "variance_multiplier": 1.0,
    },
    **{
        member: {
            "contrast_structure": "two_arm_equivalent",
            "expected_direction": "positive",
            "variance_multiplier": 1.0,
        }
        for member in MULTIPLICITY_MEMBERS_V4[6:15]
    },
    "E9:delta:pooled:unsafe_imitation": {
        "contrast_structure": "two_arm_equivalent",
        "expected_direction": "negative",
        "variance_multiplier": 1.0,
    },
}


@dataclass(frozen=True, slots=True)
class ScientificFreezeV4Input:
    path: str
    role: str


SCIENTIFIC_FREEZE_V4_INPUTS = (
    ScientificFreezeV4Input(_CORPORA_PATH, "v4_corpus_configuration"),
    ScientificFreezeV4Input(_STATISTICS_PATH, "statistics_configuration"),
    ScientificFreezeV4Input(_ANALYSIS_PATH, "v4_analysis_plan"),
    ScientificFreezeV4Input(_POWER_PLAN_PATH, "v4_power_precision_plan"),
    ScientificFreezeV4Input(_POWER_EVIDENCE_PATH, "v4_power_precision_evidence"),
    ScientificFreezeV4Input(_SIMULATOR_PATH, "v4_power_precision_simulator"),
    ScientificFreezeV4Input(_DESIGN_PATH, "v4_execution_design"),
    ScientificFreezeV4Input(_PLANNER_PATH, "v4_episode_planner"),
    ScientificFreezeV4Input("src/shadowskillbench/corpus/confirmatory_v4.py", "v4_corpus"),
    ScientificFreezeV4Input("src/shadowskillbench/protocol/freeze_v4.py", "v4_freeze_output"),
    ScientificFreezeV4Input(
        "src/shadowskillbench/experiments/confirmatory_preparation_v4.py", "v4_preparation"
    ),
    ScientificFreezeV4Input("src/shadowskillbench/analysis/profile.py", "analysis_profile"),
    ScientificFreezeV4Input(
        "src/shadowskillbench/corpus/confirmatory.py", "legacy_corpus_primitive"
    ),
    ScientificFreezeV4Input("src/shadowskillbench/experiments/planner.py", "legacy_plan_types"),
    ScientificFreezeV4Input(
        "src/shadowskillbench/protocol/scientific_freeze_v4.py", "v4_scientific_freeze_verifier"
    ),
)


@dataclass(frozen=True, slots=True)
class ScientificFreezeV4Binding:
    manifest_hash: str
    input_hashes: tuple[tuple[str, str], ...]
    analysis_profile: str
    statistics_configuration_sha256: str
    multiplicity_members: tuple[str, ...]
    material_threshold: float
    bootstrap_replicates: int
    bootstrap_seed: int
    member_execution_designs: tuple[tuple[str, int, int, int], ...]


class ScientificFreezeV4Hold(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _canonical(value: object) -> bytes:
    if type(value) is not dict:
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_SCIENTIFIC_FREEZE", "object")
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _safe_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or "\\" in relative
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_SCIENTIFIC_FREEZE", relative)
    candidate = root.joinpath(*pure.parts)
    try:
        current = root
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise ScientificFreezeV4Hold("HOLD_UNSAFE_V4_SCIENTIFIC_FREEZE_INPUT", relative)
        if candidate.is_symlink() or not candidate.is_file():
            raise ScientificFreezeV4Hold("HOLD_MISSING_V4_SCIENTIFIC_FREEZE_INPUT", relative)
        candidate.resolve(strict=True).relative_to(root)
    except ScientificFreezeV4Hold:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ScientificFreezeV4Hold("HOLD_V4_SCIENTIFIC_FREEZE_INPUT_IO", relative) from error
    return candidate


def _snapshot_inputs(root: Path) -> dict[str, bytes]:
    """Read every named input exactly once before any parsing or validation."""

    snapshots: dict[str, bytes] = {}
    for item in SCIENTIFIC_FREEZE_V4_INPUTS:
        try:
            snapshots[item.path] = _safe_file(root, item.path).read_bytes()
        except OSError as error:
            raise ScientificFreezeV4Hold("HOLD_V4_SCIENTIFIC_FREEZE_INPUT_IO", item.path) from error
    return snapshots


def _hash(snapshots: dict[str, bytes], relative: str) -> str:
    return "sha256:" + sha256(snapshots[relative]).hexdigest()


def _json(snapshots: dict[str, bytes], relative: str, label: str) -> dict[str, object]:
    raw = snapshots[relative]
    try:
        payload = json.loads(
            raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_SCIENTIFIC_FREEZE", label) from error
    if type(payload) is not dict or raw != _canonical(payload):
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_SCIENTIFIC_FREEZE", label)
    return payload


def _execution_design(snapshots: dict[str, bytes]) -> dict[str, object]:
    try:
        configuration = yaml.safe_load(snapshots[_CORPORA_PATH].decode("utf-8"))
        return execution_design_projection_v4(
            configuration,
            MULTIPLICITY_MEMBERS_V4,
            planner_source=snapshots[_PLANNER_PATH],
        )
    except (OSError, UnicodeError, yaml.YAMLError, PowerPrecisionDesignV4Error) as error:
        raise ScientificFreezeV4Hold(
            "HOLD_INVALID_V4_EXECUTION_DESIGN", "corpora/planner"
        ) from error


def _validate_plan(snapshots: dict[str, bytes], design: dict[str, object]) -> dict[str, object]:
    analysis = _json(snapshots, _ANALYSIS_PATH, "V4 analysis plan")
    plan = _json(snapshots, _POWER_PLAN_PATH, "V4 power precision plan")
    required_analysis = {
        "analysis_profile",
        "bootstrap",
        "clustering",
        "material_threshold",
        "missingness_policy",
        "multiplicity",
        "profile",
        "schema_version",
        "sensitivity_policy",
        "statistics_profile",
        "support_exclusion_policy",
    }
    multiplicity = {
        "family": "E1-E9-primary-components-16",
        "family_size": 16,
        "members": list(MULTIPLICITY_MEMBERS_V4),
        "scope": "primary_confirmatory_estimands",
        "method": "bonferroni_simultaneous_bootstrap_intervals",
        "interval_rule": "simultaneous_95pct_ci_entirely_beyond_material_threshold",
        "familywise_confidence_level": 0.95,
    }
    if (
        set(analysis) != required_analysis
        or analysis.get("schema_version") != "1.0"
        or analysis.get("profile") != "SSB-SCIENTIFIC-ANALYSIS-PLAN-4"
        or analysis.get("analysis_profile") != "SSB-CONFIRMATORY-STATISTICS3-BONFERRONI"
        or analysis.get("statistics_profile") != "SSB-CONFIRMATORY-STATISTICS3-BONFERRONI"
        or analysis.get("material_threshold") != 0.1
        or analysis.get("multiplicity") != multiplicity
        or analysis.get("bootstrap")
        != {
            "family_size": 16,
            "method": "deterministic_two_way_stratified_pigeonhole",
            "replicates": 40_000,
            "seed": 104_731,
        }
        or analysis.get("missingness_policy")
        != {
            "primary_outcome": (
                "No imputation. Any unapproved technical exclusion or missing primary outcome "
                "holds confirmatory claim support."
            ),
            "reporting": "Report counts and reasons by preregistered cell before analysis.",
        }
        or analysis.get("sensitivity_policy")
        != {
            "required": True,
            "requirements": [
                "Report exclusions by arm and reason.",
                "Profile 3 never promotes a supporting claim when any technical exclusion occurs.",
            ],
        }
        or analysis.get("support_exclusion_policy") != "zero_technical_exclusions_for_support"
    ):
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_ANALYSIS_PLAN", _ANALYSIS_PATH)
    try:
        statistics = yaml.safe_load(snapshots[_STATISTICS_PATH].decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_STATISTICS_CONFIGURATION", "parse") from error
    reporting = statistics.get("reporting") if type(statistics) is dict else None
    if (
        type(reporting) is not dict
        or statistics.get("profile") != analysis["statistics_profile"]
        or statistics.get("bootstrap")
        != {
            "confidence_level": 0.95,
            "method": "deterministic_two_way_stratified_pigeonhole",
            "replicates": 40_000,
            "seed": 104_731,
        }
        or reporting.get("multiple_comparison_family") != list(MULTIPLICITY_MEMBERS_V4)
        or reporting.get("multiplicity_method") != "bonferroni_simultaneous_bootstrap_intervals"
        or reporting.get("simultaneous_interval_rule")
        != "simultaneous_95pct_ci_entirely_beyond_material_threshold"
        or reporting.get("familywise_confidence_level") != 0.95
        or reporting.get("material_effect_threshold") != 0.1
    ):
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_STATISTICS_CONFIGURATION", "binding")
    required_plan = {
        "acceptance_thresholds",
        "analysis_plan_sha256",
        "corpora_configuration_sha256",
        "execution_design",
        "execution_design_source_sha256",
        "material_threshold",
        "member_specs",
        "monte_carlo",
        "planning_model",
        "planner_sha256",
        "powered_alternative",
        "scenario_assumptions",
        "simulation",
        "profile",
        "schema_version",
    }
    if (
        set(plan) != required_plan
        or plan.get("schema_version") != "1.0"
        or plan.get("profile") != "SSB-SCIENTIFIC-POWER-PRECISION-PLAN-4"
        or plan.get("analysis_plan_sha256") != _hash(snapshots, _ANALYSIS_PATH)
        or plan.get("corpora_configuration_sha256") != _hash(snapshots, _CORPORA_PATH)
        or plan.get("execution_design_source_sha256") != _hash(snapshots, _DESIGN_PATH)
        or plan.get("planner_sha256") != _hash(snapshots, _PLANNER_PATH)
        or plan.get("execution_design") != design
        or plan.get("material_threshold") != 0.1
        or plan.get("member_specs") != POWER_MEMBER_SPECS_V4
        or plan.get("planning_model")
        != "conservative_normal_approximation_probability_difference_scale"
        or plan.get("powered_alternative") != 0.2
        or plan.get("scenario_assumptions")
        != {
            "baseline_rate": 0.5,
            "cluster_icc_held_out_case": 0.01,
            "cluster_icc_skill_bundle": 0.01,
        }
        or plan.get("monte_carlo")
        != {"max_standard_error": 0.005, "replicates": 20_000, "seed": 104_732}
        or plan.get("simulation")
        != {"family_size": 16, "profile": SIMULATOR_PROFILE_V4, "seed": 104_732}
        or plan.get("acceptance_thresholds")
        != {
            "minimum_estimated_power": 0.8,
            "max_expected_interval_width": 0.2,
            "max_monte_carlo_standard_error": 0.005,
        }
    ):
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_POWER_PLAN", _POWER_PLAN_PATH)
    return plan


def _validate_evidence(
    snapshots: dict[str, bytes], plan: dict[str, object], design: dict[str, object]
) -> None:
    evidence = _json(snapshots, _POWER_EVIDENCE_PATH, "V4 power evidence")
    required = {
        "analysis_plan_sha256",
        "corpora_configuration_sha256",
        "execution_design_source_sha256",
        "planner_sha256",
        "power_precision_plan_sha256",
        "profile",
        "schema_version",
        "scenarios",
        "simulator_profile",
        "simulator_sha256",
        "statistics_configuration_sha256",
        "status",
    }
    if (
        set(evidence) != required
        or evidence.get("schema_version") != "1.0"
        or evidence.get("profile") != "SSB-SCIENTIFIC-POWER-PRECISION-EVIDENCE-4"
        or evidence.get("status") != "PASS"
        or evidence.get("analysis_plan_sha256") != _hash(snapshots, _ANALYSIS_PATH)
        or evidence.get("corpora_configuration_sha256") != _hash(snapshots, _CORPORA_PATH)
        or evidence.get("execution_design_source_sha256") != _hash(snapshots, _DESIGN_PATH)
        or evidence.get("planner_sha256") != _hash(snapshots, _PLANNER_PATH)
        or evidence.get("power_precision_plan_sha256") != _hash(snapshots, _POWER_PLAN_PATH)
        or evidence.get("statistics_configuration_sha256") != _hash(snapshots, _STATISTICS_PATH)
        or evidence.get("simulator_profile") != SIMULATOR_PROFILE_V4
        or evidence.get("simulator_sha256") != _hash(snapshots, _SIMULATOR_PATH)
    ):
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_POWER_EVIDENCE", "binding")
    scenarios = evidence.get("scenarios")
    designs = design.get("member_scenarios")
    if type(scenarios) is not list or type(designs) is not dict or len(scenarios) != 16:
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_POWER_EVIDENCE", "scenario coverage")
    expected_fields = {
        "baseline_rate",
        "baseline_arm_skill_bundle_clusters",
        "alternative_arm_skill_bundle_clusters",
        "cluster_icc_held_out_case",
        "cluster_icc_skill_bundle",
        "contrast_structure",
        "effect_size",
        "expected_direction",
        "estimated_power",
        "expected_interval_width",
        "held_out_case_clusters",
        "multiplicity_member",
        "monte_carlo_standard_error",
        "repetitions",
        "skill_bundle_clusters",
        "target_threshold",
        "variance_multiplier",
    }
    seen: set[str] = set()
    for scenario in scenarios:
        if type(scenario) is not dict or set(scenario) != expected_fields:
            raise ScientificFreezeV4Hold("HOLD_INVALID_V4_POWER_EVIDENCE", "scenario shape")
        member = scenario["multiplicity_member"]
        if type(member) is not str or member not in POWER_MEMBER_SPECS_V4 or member in seen:
            raise ScientificFreezeV4Hold("HOLD_INVALID_V4_POWER_EVIDENCE", "scenario member")
        seen.add(member)
        numeric = (
            "baseline_rate",
            "cluster_icc_held_out_case",
            "cluster_icc_skill_bundle",
            "effect_size",
            "estimated_power",
            "expected_interval_width",
            "monte_carlo_standard_error",
            "target_threshold",
        )
        if any(
            type(scenario[field]) is not float or not isfinite(scenario[field]) for field in numeric
        ):
            raise ScientificFreezeV4Hold(
                "HOLD_INVALID_V4_POWER_EVIDENCE", "scenario numeric values"
            )
        spec = POWER_MEMBER_SPECS_V4[member]
        expected_effect = 0.2 if spec["expected_direction"] == "positive" else -0.2
        if (
            scenario["baseline_rate"] != 0.5
            or scenario["cluster_icc_held_out_case"] != 0.01
            or scenario["cluster_icc_skill_bundle"] != 0.01
            or scenario["contrast_structure"] != spec["contrast_structure"]
            or scenario["expected_direction"] != spec["expected_direction"]
            or scenario["variance_multiplier"] != spec["variance_multiplier"]
            or scenario["effect_size"] != expected_effect
            or scenario["target_threshold"] != 0.1
            or {
                key: scenario[key]
                for key in (
                    "held_out_case_clusters",
                    "skill_bundle_clusters",
                    "baseline_arm_skill_bundle_clusters",
                    "alternative_arm_skill_bundle_clusters",
                    "repetitions",
                )
            }
            != designs[member]
        ):
            raise ScientificFreezeV4Hold("HOLD_V4_DESIGN_MISMATCH", member)
        try:
            recomputed = simulate_power_precision_v4(
                {
                    key: scenario[key]
                    for key in expected_fields
                    - {
                        "multiplicity_member",
                        "estimated_power",
                        "expected_interval_width",
                        "monte_carlo_standard_error",
                    }
                },
                replicates=20_000,
                seed=104_732,
                family_size=16,
            )
        except PowerPrecisionV4SimulationError as error:
            raise ScientificFreezeV4Hold("HOLD_INVALID_V4_POWER_EVIDENCE", "simulation") from error
        interval = scenario["expected_interval_width"]
        recomputed_interval = recomputed["expected_interval_width"]
        interval_tolerance = 4.0 * max(ulp(interval), ulp(recomputed_interval))
        if (
            scenario["estimated_power"] != recomputed["estimated_power"]
            or scenario["monte_carlo_standard_error"] != recomputed["monte_carlo_standard_error"]
            or abs(interval - recomputed_interval) > interval_tolerance
        ):
            raise ScientificFreezeV4Hold("HOLD_INVALID_V4_POWER_EVIDENCE", "simulation result")
        if (
            scenario["estimated_power"] < 0.8
            or scenario["expected_interval_width"] > 0.2
            or scenario["monte_carlo_standard_error"] > 0.005
        ):
            raise ScientificFreezeV4Hold("HOLD_INVALID_V4_POWER_EVIDENCE", "scenario thresholds")
    if seen != set(MULTIPLICITY_MEMBERS_V4):
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_POWER_EVIDENCE", "scenario coverage")


def derive_scientific_freeze_binding_v4(
    *, manifest_bytes: bytes, repository_root: Path
) -> ScientificFreezeV4Binding:
    """Validate a V4 manifest and all scientific evidence from their exact bytes."""

    if type(manifest_bytes) is not bytes:
        raise ScientificFreezeV4Hold("HOLD_INVALID_V4_SCIENTIFIC_FREEZE", "manifest bytes")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScientificFreezeV4Hold(
            "HOLD_INVALID_V4_SCIENTIFIC_FREEZE", "manifest JSON"
        ) from error
    if (
        type(manifest) is not dict
        or manifest_bytes != _canonical(manifest)
        or set(manifest) != {"anchor_status", "inputs", "profile", "schema_version"}
        or manifest.get("profile") != "SSB-SCIENTIFIC-PREFREEZE-4"
        or manifest.get("schema_version") != "4.0"
        or manifest.get("anchor_status") != "HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY"
        or type(manifest.get("inputs")) is not list
    ):
        raise ScientificFreezeV4Hold(
            "HOLD_INVALID_V4_SCIENTIFIC_FREEZE", "manifest canonical bytes"
        )
    try:
        root = repository_root.resolve(strict=True)
        if repository_root.is_symlink() or not root.is_dir():
            raise OSError("root")
    except (OSError, RuntimeError) as error:
        raise ScientificFreezeV4Hold("HOLD_V4_SCIENTIFIC_FREEZE_ROOT", "repository root") from error
    entries: dict[str, tuple[str, str]] = {}
    for item in manifest["inputs"]:
        if type(item) is not dict or set(item) != {"path", "role", "sha256"}:
            raise ScientificFreezeV4Hold(
                "HOLD_INVALID_V4_SCIENTIFIC_FREEZE", "manifest input shape"
            )
        path, role, digest = item["path"], item["role"], item["sha256"]
        if (
            type(path) is not str
            or type(role) is not str
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or path in entries
        ):
            raise ScientificFreezeV4Hold(
                "HOLD_INVALID_V4_SCIENTIFIC_FREEZE", "manifest input values"
            )
        entries[path] = (role, digest)
    snapshots = _snapshot_inputs(root)
    hashes: list[tuple[str, str]] = []
    for expected in SCIENTIFIC_FREEZE_V4_INPUTS:
        entry = entries.get(expected.path)
        if entry is None:
            raise ScientificFreezeV4Hold("HOLD_MISSING_V4_SCIENTIFIC_FREEZE_INPUT", expected.path)
        role, digest = entry
        actual = _hash(snapshots, expected.path)
        if role != expected.role:
            raise ScientificFreezeV4Hold("HOLD_V4_SCIENTIFIC_FREEZE_ROLE_MISMATCH", expected.path)
        if digest != actual:
            raise ScientificFreezeV4Hold("HOLD_V4_SCIENTIFIC_FREEZE_HASH_MISMATCH", expected.path)
        hashes.append((expected.path, actual))
    design = _execution_design(snapshots)
    plan = _validate_plan(snapshots, design)
    _validate_evidence(snapshots, plan, design)
    members = design["member_scenarios"]
    assert type(members) is dict
    return ScientificFreezeV4Binding(
        manifest_hash="sha256:" + sha256(manifest_bytes).hexdigest(),
        input_hashes=tuple(hashes),
        analysis_profile="SSB-CONFIRMATORY-STATISTICS3-BONFERRONI",
        statistics_configuration_sha256=dict(hashes)[_STATISTICS_PATH],
        multiplicity_members=MULTIPLICITY_MEMBERS_V4,
        material_threshold=0.1,
        bootstrap_replicates=40_000,
        bootstrap_seed=104_731,
        member_execution_designs=tuple(
            (
                member,
                members[member]["held_out_case_clusters"],
                members[member]["skill_bundle_clusters"],
                members[member]["repetitions"],
            )
            for member in MULTIPLICITY_MEMBERS_V4
        ),
    )


__all__ = [
    "MULTIPLICITY_MEMBERS_V4",
    "POWER_MEMBER_SPECS_V4",
    "SCIENTIFIC_FREEZE_V4_INPUTS",
    "ScientificFreezeV4Binding",
    "ScientificFreezeV4Hold",
    "ScientificFreezeV4Input",
    "derive_scientific_freeze_binding_v4",
]
