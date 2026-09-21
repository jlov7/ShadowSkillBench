# ruff: noqa: E501

"""Explicit V4 confirmatory episode matrix.

V4 is deliberately a closed successor design.  It does not mutate or reinterpret
the V1--V3 planner; callers must opt in to this module and its larger matrix.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

from shadowskillbench.episodes.models import BlockOrder, EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.planner import (
    ConditionBinding,
    ConfirmatoryEpisodePlan,
    HeldOutCaseBinding,
    PlannedEpisode,
    PlannerInputs,
    SkillBundleBinding,
)

DOMAINS = ("access_provisioning", "financial_adjustments")
RATIOS = (
    Decimal("0"),
    Decimal("0.25"),
    Decimal("0.5"),
    Decimal("0.75"),
    Decimal("1"),
)
AUTHORITY_CLASSES = (
    "PRACTICE_MATCHES_ACTIVE_POLICY",
    "PRACTICE_VIOLATES_ACTIVE_POLICY",
    "APPROVED_SCOPED_EXCEPTION",
    "POLICY_SUPERSEDED",
    "UNRESOLVED_AUTHORITY_CONFLICT",
)
STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO = 7
STAGE_A_CASES_PER_DOMAIN = 70
STAGE_B_R75_SKILLS_PER_DOMAIN = 7
STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN = 22
REPEATS = 3
STAGE_A_CONDITIONS = (
    ExperimentCondition.A0_BARE,
    ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
    ExperimentCondition.A2_SKILL_ONLY,
    ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
    ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
    ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER,
)
STAGE_B_CONDITIONS = (
    ExperimentCondition.B0_SKILL_ONLY,
    ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
    ExperimentCondition.B2_AUTHORITY_RESOLVER,
    ExperimentCondition.B3_DETERMINISTIC_GATE,
)
STAGE_A_EPISODE_COUNT = 59_640
STAGE_B_EPISODE_COUNT = 18_480


def _skills_by_domain(
    values: tuple[SkillBundleBinding, ...], domain: str
) -> tuple[SkillBundleBinding, ...]:
    return tuple(value for value in values if value.domain == domain)


def _cases_by_domain(
    values: tuple[HeldOutCaseBinding, ...], domain: str
) -> tuple[HeldOutCaseBinding, ...]:
    return tuple(value for value in values if value.domain == domain)


def _condition_map(
    bindings: tuple[ConditionBinding, ...],
) -> dict[tuple[str, ExperimentCondition], ConditionBinding]:
    expected = {
        (domain, condition)
        for domain in DOMAINS
        for condition in STAGE_A_CONDITIONS + STAGE_B_CONDITIONS
    }
    mapping = {(binding.domain, binding.condition): binding for binding in bindings}
    if len(mapping) != len(bindings) or set(mapping) != expected:
        raise ValueError(
            "condition_bindings must contain every V4 domain-condition pair exactly once"
        )
    return mapping


def _require_shared_policy(
    bindings: dict[tuple[str, ExperimentCondition], ConditionBinding],
    domain: str,
    conditions: tuple[ExperimentCondition, ...],
    label: str,
) -> None:
    hashes = {bindings[(domain, condition)].policy_hash for condition in conditions}
    if len(hashes) != 1 or None in hashes:
        raise ValueError(f"{label} policy hashes must match within a domain")


def _validate_stage_a(
    skills: tuple[SkillBundleBinding, ...], cases: tuple[HeldOutCaseBinding, ...]
) -> None:
    expected_skills = len(DOMAINS) * len(RATIOS) * STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO
    expected_cases = len(DOMAINS) * STAGE_A_CASES_PER_DOMAIN
    if len(skills) != expected_skills or len(cases) != expected_cases:
        raise ValueError("Stage A does not match the frozen V4 bundle/case counts")
    if any(case.authority_class is not None for case in cases):
        raise ValueError("Stage A cases cannot declare Stage B authority classes")
    for domain in DOMAINS:
        domain_skills = _skills_by_domain(skills, domain)
        domain_cases = _cases_by_domain(cases, domain)
        if len(domain_cases) != STAGE_A_CASES_PER_DOMAIN:
            raise ValueError("Stage A requires 70 held-out cases per domain")
        if Counter(skill.contamination_ratio for skill in domain_skills) != Counter(
            {ratio: STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO for ratio in RATIOS}
        ):
            raise ValueError("Stage A requires seven bundle seeds per ratio and domain")
        if len({case.case_id for case in domain_cases}) != len(domain_cases):
            raise ValueError("Stage A case IDs must be unique within a domain")
        if len({skill.bundle_id for skill in domain_skills}) != len(domain_skills):
            raise ValueError("Stage A bundle IDs must be unique within a domain")


def _validate_stage_b(cases: tuple[HeldOutCaseBinding, ...]) -> None:
    expected_per_domain = len(AUTHORITY_CLASSES) * STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN
    if len(cases) != len(DOMAINS) * expected_per_domain:
        raise ValueError("Stage B does not match the frozen V4 authority-class case count")
    for domain in DOMAINS:
        domain_cases = _cases_by_domain(cases, domain)
        if len(domain_cases) != expected_per_domain:
            raise ValueError("Stage B requires 110 held-out cases per domain")
        if Counter(case.authority_class for case in domain_cases) != Counter(
            {
                authority_class: STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN
                for authority_class in AUTHORITY_CLASSES
            }
        ):
            raise ValueError("Stage B requires 22 cases per authority class and domain")
        if len({case.case_id for case in domain_cases}) != len(domain_cases):
            raise ValueError("Stage B case IDs must be unique within a domain")


def _validate_inputs(
    inputs: PlannerInputs,
) -> dict[tuple[str, ExperimentCondition], ConditionBinding]:
    if type(inputs) is not PlannerInputs:
        raise ValueError("inputs must be an exact PlannerInputs")
    for name, expected_type in (
        ("stage_a_skills", SkillBundleBinding),
        ("stage_a_cases", HeldOutCaseBinding),
        ("stage_b_cases", HeldOutCaseBinding),
        ("condition_bindings", ConditionBinding),
    ):
        if any(type(value) is not expected_type for value in getattr(inputs, name)):
            raise ValueError(f"{name} must contain exact {expected_type.__name__} values")
    _validate_stage_a(inputs.stage_a_skills, inputs.stage_a_cases)
    _validate_stage_b(inputs.stage_b_cases)
    bindings = _condition_map(inputs.condition_bindings)
    for domain in DOMAINS:
        _require_shared_policy(
            bindings,
            domain,
            (
                ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
                ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
                ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
            ),
            "Stage A current-policy",
        )
        _require_shared_policy(
            bindings,
            domain,
            (
                ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
                ExperimentCondition.B2_AUTHORITY_RESOLVER,
                ExperimentCondition.B3_DETERMINISTIC_GATE,
            ),
            "Stage B policy",
        )
    return bindings


def _episode(
    *,
    stage: EpisodeStage,
    condition: ExperimentCondition,
    case: HeldOutCaseBinding,
    bindings: dict[tuple[str, ExperimentCondition], ConditionBinding],
    repeat_index: int,
    skill: SkillBundleBinding | None = None,
    order_assignment: BlockOrder | None = None,
) -> PlannedEpisode:
    return PlannedEpisode(
        stage=stage,
        condition=condition,
        case=case,
        condition_binding=bindings[(case.domain, condition)],
        repeat_index=repeat_index,
        skill=skill,
        order_assignment=order_assignment,
    )


def plan_confirmatory_episodes_v4(inputs: PlannerInputs) -> ConfirmatoryEpisodePlan:
    """Generate exactly 59,640 Stage A and 18,480 Stage B V4 episodes."""

    bindings = _validate_inputs(inputs)
    episodes: list[PlannedEpisode] = []
    for domain in DOMAINS:
        cases = tuple(
            sorted(_cases_by_domain(inputs.stage_a_cases, domain), key=lambda item: item.case_id)
        )
        for condition in STAGE_A_CONDITIONS[:2]:
            for case in cases:
                for repeat_index in range(1, REPEATS + 1):
                    episodes.append(
                        _episode(
                            stage=EpisodeStage.CONFIRMATORY_A,
                            condition=condition,
                            case=case,
                            bindings=bindings,
                            repeat_index=repeat_index,
                        )
                    )
        skills = tuple(
            sorted(
                _skills_by_domain(inputs.stage_a_skills, domain),
                key=lambda item: (item.contamination_ratio, item.bundle_id),
            )
        )
        for skill in skills:
            for condition in STAGE_A_CONDITIONS[2:]:
                for case_index, case in enumerate(cases):
                    for repeat_index in range(1, REPEATS + 1):
                        order = None
                        if condition is ExperimentCondition.A3_SKILL_POLICY_SAME_TIER:
                            order = (
                                BlockOrder.POLICY_THEN_SKILL
                                if (case_index * REPEATS + repeat_index - 1) % 2 == 0
                                else BlockOrder.SKILL_THEN_POLICY
                            )
                        episodes.append(
                            _episode(
                                stage=EpisodeStage.CONFIRMATORY_A,
                                condition=condition,
                                case=case,
                                bindings=bindings,
                                repeat_index=repeat_index,
                                skill=skill,
                                order_assignment=order,
                            )
                        )
    for domain in DOMAINS:
        cases = tuple(
            sorted(_cases_by_domain(inputs.stage_b_cases, domain), key=lambda item: item.case_id)
        )
        skills = tuple(
            skill
            for skill in sorted(
                _skills_by_domain(inputs.stage_a_skills, domain), key=lambda item: item.bundle_id
            )
            if skill.contamination_ratio == Decimal("0.75")
        )
        if len(skills) != STAGE_B_R75_SKILLS_PER_DOMAIN:
            raise ValueError("Stage B requires the seven V4 R75 skills per domain")
        for skill in skills:
            for case in cases:
                for condition in STAGE_B_CONDITIONS:
                    for repeat_index in range(1, REPEATS + 1):
                        episodes.append(
                            _episode(
                                stage=EpisodeStage.CONFIRMATORY_B,
                                condition=condition,
                                case=case,
                                bindings=bindings,
                                repeat_index=repeat_index,
                                skill=skill,
                            )
                        )
    counts = (
        sum(item.stage is EpisodeStage.CONFIRMATORY_A for item in episodes),
        sum(item.stage is EpisodeStage.CONFIRMATORY_B for item in episodes),
    )
    if counts != (STAGE_A_EPISODE_COUNT, STAGE_B_EPISODE_COUNT):
        raise ValueError("episode matrix does not match preregistered V4 counts")
    return ConfirmatoryEpisodePlan(episodes=tuple(episodes))


__all__ = [
    "AUTHORITY_CLASSES",
    "DOMAINS",
    "RATIOS",
    "REPEATS",
    "STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO",
    "STAGE_A_CASES_PER_DOMAIN",
    "STAGE_A_CONDITIONS",
    "STAGE_A_EPISODE_COUNT",
    "STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN",
    "STAGE_B_CONDITIONS",
    "STAGE_B_EPISODE_COUNT",
    "STAGE_B_R75_SKILLS_PER_DOMAIN",
    "plan_confirmatory_episodes_v4",
]
