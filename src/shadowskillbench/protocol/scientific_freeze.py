"""Forward-only verification of scientific inputs at release boundaries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path, PurePosixPath

import yaml

from shadowskillbench.protocol.power_precision_design import (
    PowerPrecisionDesignError,
    execution_design_projection,
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ANALYSIS_PLAN_PATH = "protocol/analysis_plan.v3.json"
_POWER_PRECISION_PLAN_PATH = "protocol/power_precision_plan.v3.json"
_POWER_PRECISION_EVIDENCE_PATH = "protocol/power_precision_evidence.v3.json"
_POWER_PRECISION_SIMULATOR_PATH = "src/shadowskillbench/protocol/power_precision.py"
_POWER_PRECISION_DESIGN_PATH = "src/shadowskillbench/protocol/power_precision_design.py"
_CORPORA_CONFIGURATION_PATH = "config/corpora.yaml"
_PLANNER_PATH = "src/shadowskillbench/experiments/planner.py"
_CONFIRMATORY_CORPUS_PATH = "src/shadowskillbench/corpus/confirmatory.py"
_ANALYSIS_PROFILE = "SSB-CONFIRMATORY-STATISTICS3-BONFERRONI"
MULTIPLICITY_MEMBERS = (
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
_MULTIPLICITY = {
    "family": "E1-E9-primary-components-16",
    "family_size": len(MULTIPLICITY_MEMBERS),
    "members": list(MULTIPLICITY_MEMBERS),
    "scope": "primary_confirmatory_estimands",
    "method": "bonferroni_simultaneous_bootstrap_intervals",
    "interval_rule": "simultaneous_95pct_ci_entirely_beyond_material_threshold",
    "familywise_confidence_level": 0.95,
}
POWER_MEMBER_SPECS: dict[str, dict[str, str | float]] = {
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
        for member in MULTIPLICITY_MEMBERS[6:15]
    },
    "E9:delta:pooled:unsafe_imitation": {
        "contrast_structure": "two_arm_equivalent",
        "expected_direction": "negative",
        "variance_multiplier": 1.0,
    },
}
_POWER_PLANNING_MODEL = "conservative_normal_approximation_probability_difference_scale"


@dataclass(frozen=True, slots=True)
class ScientificFreezeInput:
    path: str
    role: str


SCIENTIFIC_FREEZE_INPUTS: tuple[ScientificFreezeInput, ...] = (
    ScientificFreezeInput(_CORPORA_CONFIGURATION_PATH, "corpus_configuration"),
    ScientificFreezeInput("config/statistics.yaml", "statistics_configuration"),
    ScientificFreezeInput(_ANALYSIS_PLAN_PATH, "successor_analysis_plan"),
    ScientificFreezeInput(_POWER_PRECISION_PLAN_PATH, "successor_power_precision_plan"),
    ScientificFreezeInput(_POWER_PRECISION_EVIDENCE_PATH, "successor_power_precision_evidence"),
    ScientificFreezeInput(_POWER_PRECISION_SIMULATOR_PATH, "power_precision_simulator"),
    ScientificFreezeInput(_POWER_PRECISION_DESIGN_PATH, "power_precision_execution_design"),
    ScientificFreezeInput(_PLANNER_PATH, "episode_planner"),
    ScientificFreezeInput(
        "src/shadowskillbench/protocol/claim_estimands.py", "claim_estimand_mapping"
    ),
    ScientificFreezeInput("src/shadowskillbench/protocol/claims.py", "claims_validator"),
    ScientificFreezeInput(
        "src/shadowskillbench/protocol/scientific_freeze.py", "scientific_freeze_verifier"
    ),
    ScientificFreezeInput("src/shadowskillbench/analysis/profile.py", "analysis_profile"),
    ScientificFreezeInput("src/shadowskillbench/analysis/dataset.py", "analysis_dataset"),
    ScientificFreezeInput("src/shadowskillbench/analysis/exclusions.py", "analysis_exclusions"),
    ScientificFreezeInput("src/shadowskillbench/analysis/sealed.py", "sealed_analysis"),
    ScientificFreezeInput("src/shadowskillbench/analysis/stage_a.py", "stage_a_estimators"),
    ScientificFreezeInput("src/shadowskillbench/analysis/stage_b.py", "stage_b_estimators"),
    ScientificFreezeInput("src/shadowskillbench/experiments/audit.py", "experiment_audit"),
    ScientificFreezeInput("src/shadowskillbench/experiments/io.py", "experiment_artifact_loader"),
    ScientificFreezeInput("src/shadowskillbench/corpus/confirmatory.py", "confirmatory_corpus"),
    ScientificFreezeInput("src/shadowskillbench/reporting/report.py", "report_renderer"),
)


@dataclass(frozen=True, slots=True)
class ScientificFreezeBinding:
    manifest_hash: str
    input_hashes: tuple[tuple[str, str], ...]
    analysis_plan_sha256: str
    power_precision_plan_sha256: str
    power_precision_evidence_sha256: str
    analysis_profile: str
    multiplicity_family: str
    multiplicity_members: tuple[str, ...]
    multiplicity_scope: str
    multiplicity_method: str
    interval_rule: str
    familywise_confidence_level: float
    material_threshold: float
    bootstrap_replicates: int
    bootstrap_seed: int
    member_execution_designs: tuple[tuple[str, int, int, int], ...]

    @property
    def statistics_configuration_sha256(self) -> str:
        if type(self.input_hashes) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in self.input_hashes
        ):
            return ""
        return dict(self.input_hashes).get("config/statistics.yaml", "")


class ScientificFreezeHold(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _canonical_manifest(value: object) -> bytes:
    if type(value) is not dict:
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", "manifest payload")
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _safe_input_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or "\\" in relative
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", relative)
    candidate = root.joinpath(*path.parts)
    try:
        current = root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise ScientificFreezeHold("HOLD_UNSAFE_SCIENTIFIC_FREEZE_INPUT", relative)
        if candidate.is_symlink() or not candidate.is_file():
            raise ScientificFreezeHold("HOLD_MISSING_SCIENTIFIC_FREEZE_INPUT", relative)
        candidate.resolve(strict=True).relative_to(root)
    except ScientificFreezeHold:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ScientificFreezeHold("HOLD_SCIENTIFIC_FREEZE_INPUT_IO", relative) from error
    return candidate


def _canonical_json_object(raw: bytes, label: str) -> dict[str, object]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(value)

    try:
        value = json.loads(raw, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", label) from error
    if type(value) is not dict or raw != _canonical_manifest(value):
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", label)
    return value


def _file_sha256(root: Path, relative: str) -> str:
    return f"sha256:{sha256(_safe_input_path(root, relative).read_bytes()).hexdigest()}"


def _execution_design(root: Path) -> dict[str, object]:
    try:
        configuration = yaml.safe_load(
            _safe_input_path(root, _CORPORA_CONFIGURATION_PATH).read_text(encoding="utf-8")
        )
        return execution_design_projection(
            configuration,
            MULTIPLICITY_MEMBERS,
            planner_source=_safe_input_path(root, _PLANNER_PATH).read_bytes(),
        )
    except (OSError, UnicodeError, yaml.YAMLError, PowerPrecisionDesignError) as error:
        raise ScientificFreezeHold(
            "HOLD_INVALID_POWER_PRECISION_EXECUTION_DESIGN", "corpora/planner"
        ) from error


def _scientific_plan_metadata(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    analysis_plan = _canonical_json_object(
        _safe_input_path(root, _ANALYSIS_PLAN_PATH).read_bytes(), "analysis plan"
    )
    power_plan = _canonical_json_object(
        _safe_input_path(root, _POWER_PRECISION_PLAN_PATH).read_bytes(), "power precision plan"
    )
    power_evidence = _canonical_json_object(
        _safe_input_path(root, _POWER_PRECISION_EVIDENCE_PATH).read_bytes(),
        "power precision evidence",
    )
    material_threshold = analysis_plan.get("material_threshold")
    if set(analysis_plan) != {
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
    } or (
        analysis_plan.get("schema_version") != "1.0"
        or analysis_plan.get("profile") != "SSB-SCIENTIFIC-ANALYSIS-PLAN-3"
        or analysis_plan.get("analysis_profile") != _ANALYSIS_PROFILE
        or analysis_plan.get("statistics_profile") != _ANALYSIS_PROFILE
        or analysis_plan.get("support_exclusion_policy") != "zero_technical_exclusions_for_support"
        or type(material_threshold) is not float
        or material_threshold != 0.1
    ):
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_ANALYSIS_PLAN", _ANALYSIS_PLAN_PATH)
    multiplicity = analysis_plan.get("multiplicity")
    if type(multiplicity) is not dict or multiplicity != _MULTIPLICITY:
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_ANALYSIS_PLAN", "multiplicity")
    bootstrap = analysis_plan.get("bootstrap")
    if bootstrap != {
        "family_size": len(MULTIPLICITY_MEMBERS),
        "method": "deterministic_two_way_stratified_pigeonhole",
        "replicates": 40_000,
        "seed": 104_731,
    }:
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_ANALYSIS_PLAN", "bootstrap")
    clustering = analysis_plan.get("clustering")
    missingness = analysis_plan.get("missingness_policy")
    sensitivity = analysis_plan.get("sensitivity_policy")
    if (
        type(clustering) is not dict
        or set(clustering) != {"repeat_observation_assumption", "units"}
        or clustering.get("units") != ["skill_bundle_id", "held_out_case_id"]
        or type(clustering.get("repeat_observation_assumption")) is not str
        or not clustering["repeat_observation_assumption"].strip()
        or type(missingness) is not dict
        or set(missingness) != {"primary_outcome", "reporting"}
        or not all(type(value) is str and value.strip() for value in missingness.values())
        or type(sensitivity) is not dict
        or set(sensitivity) != {"required", "requirements"}
        or sensitivity.get("required") is not True
        or type(sensitivity.get("requirements")) is not list
        or not sensitivity["requirements"]
        or not all(type(value) is str and value.strip() for value in sensitivity["requirements"])
    ):
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_ANALYSIS_PLAN", "policies")
    execution_design = _execution_design(root)
    if set(power_plan) != {
        "acceptance_thresholds",
        "analysis_plan_sha256",
        "clustered_repeat_assumptions",
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
    } or (
        power_plan.get("schema_version") != "1.0"
        or power_plan.get("profile") != "SSB-SCIENTIFIC-POWER-PRECISION-PLAN-3"
        or power_plan.get("analysis_plan_sha256") != _file_sha256(root, _ANALYSIS_PLAN_PATH)
        or power_plan.get("corpora_configuration_sha256")
        != _file_sha256(root, _CORPORA_CONFIGURATION_PATH)
        or power_plan.get("planner_sha256") != _file_sha256(root, _PLANNER_PATH)
        or power_plan.get("execution_design_source_sha256")
        != _file_sha256(root, _POWER_PRECISION_DESIGN_PATH)
        or power_plan.get("execution_design") != execution_design
        or power_plan.get("material_threshold") != analysis_plan["material_threshold"]
        or power_plan.get("member_specs") != POWER_MEMBER_SPECS
        or power_plan.get("planning_model") != _POWER_PLANNING_MODEL
        or power_plan.get("powered_alternative") != 0.2
        or power_plan.get("scenario_assumptions")
        != {
            "baseline_rate": 0.5,
            "cluster_icc_held_out_case": 0.01,
            "cluster_icc_skill_bundle": 0.01,
        }
        or power_plan.get("clustered_repeat_assumptions")
        != {
            "repeat_observation_assumption": (
                "Repeated outcomes within a skill bundle or held-out case are not independent; "
                "simulations must incorporate both cluster variance components."
            ),
            "units": ["skill_bundle_id", "held_out_case_id"],
        }
    ):
        raise ScientificFreezeHold(
            "HOLD_INVALID_SCIENTIFIC_POWER_PRECISION_PLAN", _POWER_PRECISION_PLAN_PATH
        )
    monte_carlo = power_plan.get("monte_carlo")
    simulation = power_plan.get("simulation")
    thresholds = power_plan.get("acceptance_thresholds")
    if (
        type(monte_carlo) is not dict
        or set(monte_carlo) != {"max_standard_error", "replicates", "seed"}
        or type(monte_carlo.get("replicates")) is not int
        or monte_carlo["replicates"] < 10_000
        or type(monte_carlo.get("seed")) is not int
        or type(monte_carlo.get("max_standard_error")) is not float
        or not 0.0 < monte_carlo["max_standard_error"] <= 0.005
        or simulation
        != {
            "family_size": len(MULTIPLICITY_MEMBERS),
            "profile": "SSB-POWER-PRECISION-SIMULATOR-1",
            "seed": 104_732,
        }
        or type(thresholds) is not dict
        or set(thresholds)
        != {
            "max_expected_interval_width",
            "max_monte_carlo_standard_error",
            "minimum_estimated_power",
        }
        or thresholds.get("minimum_estimated_power") != 0.8
        or thresholds.get("max_expected_interval_width") != 0.2
        or thresholds.get("max_monte_carlo_standard_error") != 0.005
    ):
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_POWER_PRECISION_PLAN", "requirements")
    _validate_statistics_profile(root, analysis_plan)
    _validate_power_precision_evidence(
        root, analysis_plan, power_plan, power_evidence, execution_design
    )
    return analysis_plan, power_plan, power_evidence


def _validate_statistics_profile(root: Path, analysis_plan: dict[str, object]) -> None:
    try:
        configuration = yaml.safe_load(
            _safe_input_path(root, "config/statistics.yaml").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ScientificFreezeHold(
            "HOLD_INVALID_SCIENTIFIC_STATISTICS_CONFIGURATION", "parse"
        ) from error
    reporting = configuration.get("reporting") if type(configuration) is dict else None
    if (
        type(reporting) is not dict
        or configuration.get("profile") != analysis_plan["statistics_profile"]
        or reporting.get("multiple_comparison_family") != list(MULTIPLICITY_MEMBERS)
        or reporting.get("multiplicity_method") != _MULTIPLICITY["method"]
        or reporting.get("simultaneous_interval_rule") != _MULTIPLICITY["interval_rule"]
        or reporting.get("familywise_confidence_level")
        != _MULTIPLICITY["familywise_confidence_level"]
        or reporting.get("material_effect_threshold") != analysis_plan["material_threshold"]
        or configuration.get("bootstrap")
        != {
            "confidence_level": _MULTIPLICITY["familywise_confidence_level"],
            "method": analysis_plan["bootstrap"]["method"],  # type: ignore[index]
            "replicates": analysis_plan["bootstrap"]["replicates"],  # type: ignore[index]
            "seed": analysis_plan["bootstrap"]["seed"],  # type: ignore[index]
        }
    ):
        raise ScientificFreezeHold(
            "HOLD_INVALID_SCIENTIFIC_STATISTICS_CONFIGURATION", "successor profile"
        )


def _validate_power_precision_evidence(
    root: Path,
    analysis_plan: dict[str, object],
    power_plan: dict[str, object],
    evidence: dict[str, object],
    execution_design: dict[str, object],
) -> None:
    required = {
        "analysis_plan_sha256",
        "corpora_configuration_sha256",
        "execution_design_source_sha256",
        "power_precision_plan_sha256",
        "planner_sha256",
        "profile",
        "schema_version",
        "statistics_configuration_sha256",
        "status",
        "simulator_profile",
        "simulator_sha256",
    }
    if evidence.get("status") == "PASS":
        required.add("scenarios")
    if (
        set(evidence) != required
        or evidence.get("schema_version") != "1.0"
        or evidence.get("profile") != "SSB-SCIENTIFIC-POWER-PRECISION-EVIDENCE-3"
        or evidence.get("analysis_plan_sha256") != _file_sha256(root, _ANALYSIS_PLAN_PATH)
        or evidence.get("corpora_configuration_sha256")
        != _file_sha256(root, _CORPORA_CONFIGURATION_PATH)
        or evidence.get("power_precision_plan_sha256")
        != _file_sha256(root, _POWER_PRECISION_PLAN_PATH)
        or evidence.get("planner_sha256") != _file_sha256(root, _PLANNER_PATH)
        or evidence.get("execution_design_source_sha256")
        != _file_sha256(root, _POWER_PRECISION_DESIGN_PATH)
        or evidence.get("statistics_configuration_sha256")
        != _file_sha256(root, "config/statistics.yaml")
        or evidence.get("simulator_profile") != "SSB-POWER-PRECISION-SIMULATOR-1"
        or evidence.get("simulator_sha256") != _file_sha256(root, _POWER_PRECISION_SIMULATOR_PATH)
    ):
        raise ScientificFreezeHold("HOLD_INVALID_POWER_PRECISION_EVIDENCE", "binding")
    if evidence.get("status") != "PASS":
        raise ScientificFreezeHold("HOLD_POWER_PRECISION_EVIDENCE_NOT_PASS", "status")
    scenarios = evidence.get("scenarios")
    thresholds = power_plan["acceptance_thresholds"]
    powered_alternative = power_plan["powered_alternative"]
    scenario_assumptions = power_plan["scenario_assumptions"]
    assert type(thresholds) is dict
    assert type(powered_alternative) is float
    assert type(scenario_assumptions) is dict
    designs = execution_design["member_scenarios"]
    assert type(designs) is dict
    if type(scenarios) is not list or len(scenarios) != len(MULTIPLICITY_MEMBERS):
        raise ScientificFreezeHold("HOLD_INVALID_POWER_PRECISION_EVIDENCE", "scenarios")
    expected_fields = {
        "baseline_rate",
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
    observed_members: set[str] = set()
    for scenario in scenarios:
        if type(scenario) is not dict or set(scenario) != expected_fields:
            raise ScientificFreezeHold("HOLD_INVALID_POWER_PRECISION_EVIDENCE", "scenario shape")
        member = scenario["multiplicity_member"]
        if (
            type(member) is not str
            or member not in MULTIPLICITY_MEMBERS
            or member in observed_members
        ):
            raise ScientificFreezeHold("HOLD_INVALID_POWER_PRECISION_EVIDENCE", "scenario member")
        observed_members.add(member)
        spec = POWER_MEMBER_SPECS[member]
        design = designs[member]
        assert type(design) is dict
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
        if (
            any(type(scenario[field]) is not float for field in numeric)
            or any(not isfinite(scenario[field]) for field in numeric)
            or any(
                type(scenario[field]) is not int
                for field in ("held_out_case_clusters", "repetitions", "skill_bundle_clusters")
            )
            or not 0.0 < scenario["baseline_rate"] < 1.0
            or not 0.0 <= scenario["cluster_icc_held_out_case"] < 1.0
            or not 0.0 <= scenario["cluster_icc_skill_bundle"] < 1.0
            or scenario["expected_direction"] != spec["expected_direction"]
            or scenario["contrast_structure"] != spec["contrast_structure"]
            or scenario["variance_multiplier"] != spec["variance_multiplier"]
            or (
                scenario["expected_direction"] == "positive"
                and scenario["effect_size"] != powered_alternative
            )
            or (
                scenario["expected_direction"] == "negative"
                and scenario["effect_size"] != -powered_alternative
            )
            or scenario["target_threshold"] != analysis_plan["material_threshold"]
            or scenario["baseline_rate"] != scenario_assumptions["baseline_rate"]
            or scenario["cluster_icc_held_out_case"]
            != scenario_assumptions["cluster_icc_held_out_case"]
            or scenario["cluster_icc_skill_bundle"]
            != scenario_assumptions["cluster_icc_skill_bundle"]
            or scenario["repetitions"] < 1
        ):
            raise ScientificFreezeHold("HOLD_INVALID_POWER_PRECISION_EVIDENCE", "scenario values")
        if {
            "held_out_case_clusters": scenario["held_out_case_clusters"],
            "skill_bundle_clusters": scenario["skill_bundle_clusters"],
            "repetitions": scenario["repetitions"],
        } != design:
            raise ScientificFreezeHold("HOLD_DESIGN_MISMATCH", member)
        from shadowskillbench.protocol.power_precision import (
            PowerPrecisionSimulationError,
            simulate_power_precision,
        )

        try:
            recomputed = simulate_power_precision(
                {
                    key: scenario[key]
                    for key in expected_fields
                    - {
                        "estimated_power",
                        "expected_interval_width",
                        "monte_carlo_standard_error",
                        "multiplicity_member",
                    }
                },
                replicates=power_plan["monte_carlo"]["replicates"],  # type: ignore[index]
                seed=power_plan["simulation"]["seed"],  # type: ignore[index]
                family_size=power_plan["simulation"]["family_size"],  # type: ignore[index]
            )
        except PowerPrecisionSimulationError as error:
            raise ScientificFreezeHold(
                "HOLD_INVALID_POWER_PRECISION_EVIDENCE", "simulation"
            ) from error
        if any(scenario[key] != value for key, value in recomputed.items()):
            raise ScientificFreezeHold("HOLD_INVALID_POWER_PRECISION_EVIDENCE", "simulation result")
        if (
            scenario["estimated_power"] < thresholds["minimum_estimated_power"]
            or scenario["expected_interval_width"] > thresholds["max_expected_interval_width"]
            or scenario["monte_carlo_standard_error"] > thresholds["max_monte_carlo_standard_error"]
        ):
            raise ScientificFreezeHold(
                "HOLD_INVALID_POWER_PRECISION_EVIDENCE", "scenario thresholds"
            )
    if observed_members != set(MULTIPLICITY_MEMBERS):
        raise ScientificFreezeHold("HOLD_INVALID_POWER_PRECISION_EVIDENCE", "scenario coverage")


def derive_scientific_freeze_binding(
    *, manifest_bytes: bytes, repository_root: Path
) -> ScientificFreezeBinding:
    """Verify the release-critical scientific inputs against validated freeze bytes."""

    if type(manifest_bytes) is not bytes:
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", "manifest bytes")
    try:
        payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", "manifest JSON") from error
    if manifest_bytes != _canonical_manifest(payload):
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", "manifest canonical bytes")
    inputs = payload.get("inputs")
    if type(inputs) is not list:
        raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", "manifest inputs")

    entries: dict[str, tuple[str, str]] = {}
    for item in inputs:
        if type(item) is not dict or set(item) != {"path", "role", "sha256"}:
            raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", "manifest input shape")
        path = item["path"]
        role = item["role"]
        digest = item["sha256"]
        if (
            type(path) is not str
            or type(role) is not str
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or path in entries
        ):
            raise ScientificFreezeHold("HOLD_INVALID_SCIENTIFIC_FREEZE", "manifest input values")
        entries[path] = (role, digest)

    try:
        root = repository_root.resolve(strict=True)
        if repository_root.is_symlink() or not root.is_dir():
            raise OSError("repository root")
    except (OSError, RuntimeError) as error:
        raise ScientificFreezeHold("HOLD_SCIENTIFIC_FREEZE_ROOT", "repository root") from error

    bound_hashes: list[tuple[str, str]] = []
    for expected in SCIENTIFIC_FREEZE_INPUTS:
        entry = entries.get(expected.path)
        if entry is None:
            raise ScientificFreezeHold("HOLD_MISSING_SCIENTIFIC_FREEZE_INPUT", expected.path)
        role, expected_hash = entry
        if role != expected.role:
            raise ScientificFreezeHold("HOLD_SCIENTIFIC_FREEZE_ROLE_MISMATCH", expected.path)
        actual_bytes = _safe_input_path(root, expected.path).read_bytes()
        actual_hash = f"sha256:{sha256(actual_bytes).hexdigest()}"
        if actual_hash != expected_hash:
            raise ScientificFreezeHold("HOLD_SCIENTIFIC_FREEZE_HASH_MISMATCH", expected.path)
        bound_hashes.append((expected.path, actual_hash))
    analysis_plan, power_plan, _ = _scientific_plan_metadata(root)
    multiplicity = analysis_plan["multiplicity"]
    assert type(multiplicity) is dict
    execution_design = power_plan["execution_design"]
    assert type(execution_design) is dict
    member_scenarios = execution_design["member_scenarios"]
    assert type(member_scenarios) is dict
    member_execution_designs = tuple(
        (
            member,
            member_scenarios[member]["held_out_case_clusters"],
            member_scenarios[member]["skill_bundle_clusters"],
            member_scenarios[member]["repetitions"],
        )
        for member in MULTIPLICITY_MEMBERS
    )
    return ScientificFreezeBinding(
        manifest_hash=f"sha256:{sha256(manifest_bytes).hexdigest()}",
        input_hashes=tuple(bound_hashes),
        analysis_plan_sha256=dict(bound_hashes)[_ANALYSIS_PLAN_PATH],
        power_precision_plan_sha256=dict(bound_hashes)[_POWER_PRECISION_PLAN_PATH],
        power_precision_evidence_sha256=dict(bound_hashes)[_POWER_PRECISION_EVIDENCE_PATH],
        analysis_profile=analysis_plan["analysis_profile"],  # type: ignore[arg-type]
        multiplicity_family=multiplicity["family"],  # type: ignore[arg-type]
        multiplicity_members=tuple(multiplicity["members"]),  # type: ignore[arg-type]
        multiplicity_scope=multiplicity["scope"],  # type: ignore[arg-type]
        multiplicity_method=multiplicity["method"],  # type: ignore[arg-type]
        interval_rule=multiplicity["interval_rule"],  # type: ignore[arg-type]
        familywise_confidence_level=multiplicity["familywise_confidence_level"],  # type: ignore[arg-type]
        material_threshold=analysis_plan["material_threshold"],  # type: ignore[arg-type]
        bootstrap_replicates=analysis_plan["bootstrap"]["replicates"],  # type: ignore[index,arg-type]
        bootstrap_seed=analysis_plan["bootstrap"]["seed"],  # type: ignore[index,arg-type]
        member_execution_designs=member_execution_designs,  # type: ignore[arg-type]
    )


__all__ = [
    "SCIENTIFIC_FREEZE_INPUTS",
    "POWER_MEMBER_SPECS",
    "ScientificFreezeBinding",
    "ScientificFreezeHold",
    "ScientificFreezeInput",
    "derive_scientific_freeze_binding",
]
