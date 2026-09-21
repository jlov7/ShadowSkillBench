"""Frozen execution-design projection for Profile 3 power scenarios."""

from __future__ import annotations

import ast
from collections.abc import Mapping

EXECUTION_DESIGN_PROFILE = "SSB-POWER-PRECISION-EXECUTION-DESIGN-1"


class PowerPrecisionDesignError(ValueError):
    pass


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise PowerPrecisionDesignError(f"{label} must be an object")
    return value


def _integer(mapping: Mapping[str, object], field: str) -> int:
    value = mapping.get(field)
    if type(value) is not int or value < 1:
        raise PowerPrecisionDesignError(f"{field} must be a positive integer")
    return value


def _assignments(source: bytes, label: str) -> dict[str, ast.expr]:
    try:
        module = ast.parse(source.decode("utf-8"), filename=label)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise PowerPrecisionDesignError(f"{label} is not valid Python source") from error
    assignments: dict[str, ast.expr] = {}
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            assignments[statement.targets[0].id] = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            assignments[statement.target.id] = statement.value
    return assignments


def _source_integer(assignments: Mapping[str, ast.expr], name: str) -> int:
    try:
        value = ast.literal_eval(assignments[name])
    except (KeyError, ValueError) as error:
        raise PowerPrecisionDesignError(f"{name} is not an integer source constant") from error
    if type(value) is not int or value < 1:
        raise PowerPrecisionDesignError(f"{name} is not a positive integer source constant")
    return value


def _source_sequence_length(assignments: Mapping[str, ast.expr], name: str) -> int:
    expression = assignments.get(name)
    if not isinstance(expression, (ast.List, ast.Tuple)) or not expression.elts:
        raise PowerPrecisionDesignError(f"{name} is not a nonempty source sequence")
    return len(expression.elts)


def member_execution_designs(
    corpora_configuration: object,
    multiplicity_members: tuple[str, ...],
    *,
    planner_source: bytes,
) -> dict[str, dict[str, int]]:
    """Derive the actual primary-analysis cluster counts from the frozen matrix."""

    configuration = _mapping(corpora_configuration, "corpora configuration")
    stage_a = _mapping(configuration.get("stage_a"), "stage_a")
    stage_b = _mapping(configuration.get("stage_b"), "stage_b")
    planner = _assignments(planner_source, "episode planner")
    domains = _source_sequence_length(planner, "DOMAINS")
    authority_classes = _source_sequence_length(planner, "AUTHORITY_CLASSES")
    ratios = _source_sequence_length(planner, "RATIOS")
    repeats = _source_integer(planner, "REPEATS")
    stage_a_conditions = _source_sequence_length(planner, "STAGE_A_CONDITIONS")
    stage_b_conditions = _source_sequence_length(planner, "STAGE_B_CONDITIONS")
    stage_a_episode_count = _source_integer(planner, "STAGE_A_EPISODE_COUNT")
    stage_b_episode_count = _source_integer(planner, "STAGE_B_EPISODE_COUNT")

    stage_a_bundles = _integer(stage_a, "skill_bundles")
    stage_a_cases_per_domain = _integer(stage_a, "held_out_cases_per_domain")
    stage_a_repetitions = _integer(stage_a, "repeats")
    stage_b_bundles_per_domain = _integer(stage_b, "r75_skills_per_domain")
    stage_b_cases_per_class_domain = _integer(stage_b, "held_out_cases_per_authority_class_domain")
    stage_b_repetitions = _integer(stage_b, "repeats")
    if stage_a_repetitions != repeats or stage_b_repetitions != repeats:
        raise PowerPrecisionDesignError("corpora configuration does not match the frozen planner")
    stage_a_bundles_per_domain, stage_a_remainder = divmod(stage_a_bundles, domains)
    if stage_a_remainder or stage_a_bundles_per_domain % ratios:
        raise PowerPrecisionDesignError("corpora configuration does not match the frozen planner")
    stage_b_cases_per_domain = stage_b_cases_per_class_domain * authority_classes
    expected_stage_a_episodes = (
        domains
        * stage_a_cases_per_domain
        * repeats
        * (stage_a_bundles_per_domain * (stage_a_conditions - 2) + 2)
    )
    expected_stage_b_episodes = (
        domains
        * stage_b_cases_per_domain
        * repeats
        * stage_b_bundles_per_domain
        * stage_b_conditions
    )
    if (
        stage_a_conditions < 3
        or stage_b_conditions < 1
        or expected_stage_a_episodes != stage_a_episode_count
        or expected_stage_b_episodes != stage_b_episode_count
    ):
        raise PowerPrecisionDesignError("corpora configuration does not match the frozen planner")

    stage_a_design = {
        "held_out_case_clusters": stage_a_cases_per_domain * domains,
        "skill_bundle_clusters": stage_a_bundles,
        "repetitions": stage_a_repetitions,
    }
    stage_b_class_design = {
        "held_out_case_clusters": stage_b_cases_per_class_domain * domains,
        "skill_bundle_clusters": stage_b_bundles_per_domain * domains,
        "repetitions": stage_b_repetitions,
    }
    stage_b_all_classes_design = {
        "held_out_case_clusters": (stage_b_cases_per_class_domain * authority_classes * domains),
        "skill_bundle_clusters": stage_b_bundles_per_domain * domains,
        "repetitions": stage_b_repetitions,
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
            raise PowerPrecisionDesignError("unknown multiplicity member")
        designs[member] = dict(design)
    if len(designs) != len(multiplicity_members):
        raise PowerPrecisionDesignError("multiplicity members must be unique")
    return designs


def execution_design_projection(
    corpora_configuration: object,
    multiplicity_members: tuple[str, ...],
    *,
    planner_source: bytes,
) -> dict[str, object]:
    return {
        "profile": EXECUTION_DESIGN_PROFILE,
        "member_scenarios": member_execution_designs(
            corpora_configuration,
            multiplicity_members,
            planner_source=planner_source,
        ),
    }


__all__ = [
    "EXECUTION_DESIGN_PROFILE",
    "PowerPrecisionDesignError",
    "execution_design_projection",
    "member_execution_designs",
]
