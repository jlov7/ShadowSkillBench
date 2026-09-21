"""Frozen execution-design projection for the forward-only V4 matrix."""

from __future__ import annotations

import ast
from collections.abc import Mapping

EXECUTION_DESIGN_PROFILE_V4 = "SSB-POWER-PRECISION-EXECUTION-DESIGN-4"


class PowerPrecisionDesignV4Error(ValueError):
    pass


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise PowerPrecisionDesignV4Error(f"{label} must be an object")
    return value


def _integer(mapping: Mapping[str, object], field: str) -> int:
    value = mapping.get(field)
    if type(value) is not int or value < 1:
        raise PowerPrecisionDesignV4Error(f"{field} must be a positive integer")
    return value


def _assignments(source: bytes) -> dict[str, ast.expr]:
    try:
        module = ast.parse(source.decode("utf-8"), filename="V4 episode planner")
    except (SyntaxError, UnicodeDecodeError) as error:
        raise PowerPrecisionDesignV4Error("V4 planner source is invalid") from error
    result: dict[str, ast.expr] = {}
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            result[statement.targets[0].id] = statement.value
    return result


def _source_integer(assignments: Mapping[str, ast.expr], name: str) -> int:
    try:
        value = ast.literal_eval(assignments[name])
    except (KeyError, ValueError) as error:
        raise PowerPrecisionDesignV4Error(f"{name} is not an integer source constant") from error
    if type(value) is not int or value < 1:
        raise PowerPrecisionDesignV4Error(f"{name} is not a positive integer source constant")
    return value


def _source_sequence_length(assignments: Mapping[str, ast.expr], name: str) -> int:
    expression = assignments.get(name)
    if not isinstance(expression, (ast.List, ast.Tuple)) or not expression.elts:
        raise PowerPrecisionDesignV4Error(f"{name} is not a nonempty source sequence")
    return len(expression.elts)


def execution_design_projection_v4(
    corpora_configuration: object,
    multiplicity_members: tuple[str, ...],
    *,
    planner_source: bytes,
) -> dict[str, object]:
    """Derive V4 member cluster counts from exact config and planner bytes."""

    configuration = _mapping(corpora_configuration, "V4 corpora configuration")
    if (
        configuration.get("profile") != "SSB-CONFIRMATORY-CORPUS4"
        or configuration.get("corpus_seed") != 104733
        or configuration.get("development_seeds") != [4242, 4243]
    ):
        raise PowerPrecisionDesignV4Error("corpora configuration is not V4")
    stage_a = _mapping(configuration.get("stage_a"), "stage_a")
    stage_b = _mapping(configuration.get("stage_b"), "stage_b")
    planner = _assignments(planner_source)
    domains = _source_sequence_length(planner, "DOMAINS")
    ratios = _source_sequence_length(planner, "RATIOS")
    authority_classes = _source_sequence_length(planner, "AUTHORITY_CLASSES")
    repeats = _source_integer(planner, "REPEATS")
    stage_a_conditions = _source_sequence_length(planner, "STAGE_A_CONDITIONS")
    stage_b_conditions = _source_sequence_length(planner, "STAGE_B_CONDITIONS")
    seeds_per_ratio = _source_integer(planner, "STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO")
    stage_a_cases = _source_integer(planner, "STAGE_A_CASES_PER_DOMAIN")
    r75_skills = _source_integer(planner, "STAGE_B_R75_SKILLS_PER_DOMAIN")
    stage_b_cases = _source_integer(planner, "STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN")
    stage_a_total = _source_integer(planner, "STAGE_A_EPISODE_COUNT")
    stage_b_total = _source_integer(planner, "STAGE_B_EPISODE_COUNT")
    if (
        _integer(stage_a, "bundle_seeds_per_domain_ratio") != seeds_per_ratio
        or _integer(stage_a, "held_out_cases_per_domain") != stage_a_cases
        or _integer(stage_a, "repeats") != repeats
        or _integer(stage_b, "r75_skills_per_domain") != r75_skills
        or _integer(stage_b, "held_out_cases_per_authority_class_domain") != stage_b_cases
        or _integer(stage_b, "authority_classes") != authority_classes
        or _integer(stage_b, "conditions") != stage_b_conditions
        or _integer(stage_b, "repeats") != repeats
    ):
        raise PowerPrecisionDesignV4Error("V4 configuration does not match frozen planner")
    expected_a = (
        domains
        * stage_a_cases
        * repeats
        * (ratios * seeds_per_ratio * (stage_a_conditions - 2) + 2)
    )
    expected_b = (
        domains * authority_classes * stage_b_cases * repeats * r75_skills * stage_b_conditions
    )
    if stage_a_total != expected_a or stage_b_total != expected_b:
        raise PowerPrecisionDesignV4Error("V4 planner count formula is invalid")
    if _integer(stage_a, "total") != expected_a or _integer(stage_b, "total") != expected_b:
        raise PowerPrecisionDesignV4Error("V4 configuration totals do not match frozen planner")
    stage_a_design = {
        "held_out_case_clusters": domains * stage_a_cases,
        "skill_bundle_clusters": domains * ratios * seeds_per_ratio,
        "repetitions": repeats,
    }
    stage_b_class_design = {
        "held_out_case_clusters": domains * stage_b_cases,
        "skill_bundle_clusters": domains * r75_skills,
        "repetitions": repeats,
    }
    stage_b_all_classes_design = {
        "held_out_case_clusters": domains * authority_classes * stage_b_cases,
        "skill_bundle_clusters": domains * r75_skills,
        "repetitions": repeats,
    }
    designs: dict[str, dict[str, int]] = {}
    for member in multiplicity_members:
        if member.startswith(("E1:", "E2:", "E3:", "E4:", "E5:")):
            design = stage_a_design
        elif member.startswith(("E6:", "E7:")):
            design = stage_b_class_design
        elif member.startswith(("E8:", "E9:")):
            design = stage_b_all_classes_design
        else:
            raise PowerPrecisionDesignV4Error("unknown multiplicity member")
        designs[member] = {
            **design,
            "baseline_arm_skill_bundle_clusters": design["skill_bundle_clusters"],
            "alternative_arm_skill_bundle_clusters": design["skill_bundle_clusters"],
        }
    e4 = "E4:average_cup_contrast:pooled:completion_under_policy"
    if e4 in designs:
        # E4 is A4 (skill + current system policy) against A1 (policy-only).
        # A1 has no skill bundles, so its covariance must not use the A4
        # 70-bundle divisor.
        designs[e4]["baseline_arm_skill_bundle_clusters"] = 0
    if len(designs) != len(multiplicity_members):
        raise PowerPrecisionDesignV4Error("multiplicity members must be unique")
    return {"profile": EXECUTION_DESIGN_PROFILE_V4, "member_scenarios": designs}


__all__ = [
    "EXECUTION_DESIGN_PROFILE_V4",
    "PowerPrecisionDesignV4Error",
    "execution_design_projection_v4",
]
