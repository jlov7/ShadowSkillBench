from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, cast

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.episodes.models import BlockOrder, EpisodeStage, ExperimentCondition

type Domain = Literal["access_provisioning", "financial_adjustments"]
type AuthorityClass = Literal[
    "PRACTICE_MATCHES_ACTIVE_POLICY",
    "PRACTICE_VIOLATES_ACTIVE_POLICY",
    "APPROVED_SCOPED_EXCEPTION",
    "POLICY_SUPERSEDED",
    "UNRESOLVED_AUTHORITY_CONFLICT",
]

DOMAINS: tuple[Domain, ...] = ("access_provisioning", "financial_adjustments")
RATIOS = (
    Decimal("0"),
    Decimal("0.25"),
    Decimal("0.5"),
    Decimal("0.75"),
    Decimal("1"),
)
AUTHORITY_CLASSES: tuple[AuthorityClass, ...] = (
    "PRACTICE_MATCHES_ACTIVE_POLICY",
    "PRACTICE_VIOLATES_ACTIVE_POLICY",
    "APPROVED_SCOPED_EXCEPTION",
    "POLICY_SUPERSEDED",
    "UNRESOLVED_AUTHORITY_CONFLICT",
)
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
REPEATS = 3
STAGE_A_EPISODE_COUNT = 7_440
STAGE_B_EPISODE_COUNT = 1_800

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _require_hash(value: str, field_name: str) -> None:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase sha256 reference")


def _require_identifier(value: str, field_name: str) -> None:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a path-safe identifier")


def _require_domain(value: str) -> None:
    if value not in DOMAINS:
        raise ValueError("domain is invalid")


@dataclass(frozen=True, slots=True)
class SkillBundleBinding:
    """Hashes required to trace a planned skill back to its induced artifact."""

    domain: Domain
    bundle_id: str
    contamination_ratio: Decimal
    source_manifest_hash: str
    compiler_manifest_hash: str
    compiled_skill_artifact_hash: str
    rendered_skill_hash: str

    def __post_init__(self) -> None:
        _require_domain(self.domain)
        _require_identifier(self.bundle_id, "bundle_id")
        if type(self.contamination_ratio) is not Decimal or self.contamination_ratio not in RATIOS:
            raise ValueError("contamination_ratio is not preregistered")
        for field_name in (
            "source_manifest_hash",
            "compiler_manifest_hash",
            "compiled_skill_artifact_hash",
            "rendered_skill_hash",
        ):
            _require_hash(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class HeldOutCaseBinding:
    """Caller-supplied custody for one held-out execution case."""

    domain: Domain
    case_id: str
    case_manifest_hash: str
    world_hash: str
    authority_graph_hash: str
    authority_class: AuthorityClass | None = None

    def __post_init__(self) -> None:
        _require_domain(self.domain)
        _require_identifier(self.case_id, "case_id")
        for field_name in ("case_manifest_hash", "world_hash", "authority_graph_hash"):
            _require_hash(getattr(self, field_name), field_name)
        if self.authority_class is not None and self.authority_class not in AUTHORITY_CLASSES:
            raise ValueError("authority_class is invalid")


@dataclass(frozen=True, slots=True)
class ConditionBinding:
    """Frozen domain-condition template, not a case-specific assembled prompt."""

    domain: Domain
    condition: ExperimentCondition
    context_contract_hash: str
    policy_hash: str | None

    def __post_init__(self) -> None:
        _require_domain(self.domain)
        if type(self.condition) is not ExperimentCondition:
            raise ValueError("condition must be an exact ExperimentCondition")
        _require_hash(self.context_contract_hash, "context_contract_hash")
        if self.condition.requires_policy:
            if self.policy_hash is None:
                raise ValueError("policy condition requires policy_hash")
        elif self.policy_hash is not None:
            raise ValueError("non-policy condition cannot bind policy_hash")
        if self.policy_hash is not None:
            _require_hash(self.policy_hash, "policy_hash")


@dataclass(frozen=True, slots=True)
class PlannerInputs:
    """All confirmatory inputs; no corpus or artifact is fabricated by the planner."""

    stage_a_skills: tuple[SkillBundleBinding, ...]
    stage_a_cases: tuple[HeldOutCaseBinding, ...]
    stage_b_cases: tuple[HeldOutCaseBinding, ...]
    condition_bindings: tuple[ConditionBinding, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "stage_a_skills",
            "stage_a_cases",
            "stage_b_cases",
            "condition_bindings",
        ):
            if type(getattr(self, field_name)) is not tuple:
                raise ValueError(f"{field_name} must be an exact tuple")


@dataclass(frozen=True, slots=True)
class PlannedEpisode:
    """One immutable execution cell, bound to all available upstream custody hashes."""

    stage: EpisodeStage
    condition: ExperimentCondition
    case: HeldOutCaseBinding
    condition_binding: ConditionBinding
    repeat_index: int
    skill: SkillBundleBinding | None = None
    order_assignment: BlockOrder | None = None
    episode_id: str = field(init=False)
    manifest_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.stage not in {EpisodeStage.CONFIRMATORY_A, EpisodeStage.CONFIRMATORY_B}:
            raise ValueError("planned episode stage is invalid")
        if self.condition_binding.condition is not self.condition:
            raise ValueError("condition binding does not match episode condition")
        if self.condition_binding.domain != self.case.domain:
            raise ValueError("condition binding does not match episode domain")
        if type(self.repeat_index) is not int or not 1 <= self.repeat_index <= REPEATS:
            raise ValueError("repeat_index is invalid")
        if self.condition.requires_skill != (self.skill is not None):
            raise ValueError("skill binding does not match condition")
        if self.skill is not None and self.skill.domain != self.case.domain:
            raise ValueError("skill and case domains must match")
        if self.condition.requires_block_order != (self.order_assignment is not None):
            raise ValueError("block order does not match condition")
        if self.order_assignment is not None and type(self.order_assignment) is not BlockOrder:
            raise ValueError("order_assignment is invalid")
        identity = self.cell_key
        episode_id = "episode_" + sha256_ref(list(identity)).removeprefix("sha256:")[:32]
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "manifest_hash", sha256_ref(self.manifest_projection()))

    @property
    def cell_key(self) -> tuple[str, str, str, str, str | None, int]:
        return (
            self.stage.value,
            self.condition.value,
            self.case.domain,
            self.case.case_id,
            None if self.skill is None else self.skill.bundle_id,
            self.repeat_index,
        )

    def manifest_projection(self) -> dict[str, object]:
        skill = self.skill
        return {
            "profile": "SSB-PLAN1",
            "episode_id": self.episode_id,
            "stage": self.stage.value,
            "condition": self.condition.value,
            "domain": self.case.domain,
            "case_id": self.case.case_id,
            "case_manifest_hash": self.case.case_manifest_hash,
            "world_hash": self.case.world_hash,
            "authority_graph_hash": self.case.authority_graph_hash,
            "authority_class": self.case.authority_class,
            "context_contract_hash": self.condition_binding.context_contract_hash,
            "policy_hash": self.condition_binding.policy_hash,
            "skill_bundle_id": None if skill is None else skill.bundle_id,
            "contamination_ratio": None if skill is None else str(skill.contamination_ratio),
            "source_manifest_hash": None if skill is None else skill.source_manifest_hash,
            "compiler_manifest_hash": None if skill is None else skill.compiler_manifest_hash,
            "compiled_skill_artifact_hash": None
            if skill is None
            else skill.compiled_skill_artifact_hash,
            "rendered_skill_hash": None if skill is None else skill.rendered_skill_hash,
            "repeat_index": self.repeat_index,
            "order_assignment": None
            if self.order_assignment is None
            else self.order_assignment.value,
        }


@dataclass(frozen=True, slots=True)
class ConfirmatoryEpisodePlan:
    episodes: tuple[PlannedEpisode, ...]
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.episodes) is not tuple or not self.episodes:
            raise ValueError("episodes must be a nonempty exact tuple")
        keys = tuple(episode.cell_key for episode in self.episodes)
        if len(set(keys)) != len(keys):
            raise ValueError("episode matrix contains duplicate cells")
        hashes = tuple(episode.manifest_hash for episode in self.episodes)
        if len(set(hashes)) != len(hashes):
            raise ValueError("episode matrix contains duplicate manifests")
        object.__setattr__(self, "plan_hash", sha256_ref(self.plan_projection()))

    def plan_projection(self) -> dict[str, object]:
        return {
            "profile": "SSB-PLAN1",
            "episode_manifest_hashes": [episode.manifest_hash for episode in self.episodes],
        }


def episode_manifest_projection(episode: PlannedEpisode) -> dict[str, object]:
    if type(episode) is not PlannedEpisode:
        raise ValueError("episode must be an exact PlannedEpisode")
    return episode.manifest_projection()


def hash_planned_episode_manifest(episode: PlannedEpisode) -> str:
    return sha256_ref(episode_manifest_projection(episode))


def episode_plan_projection(plan: ConfirmatoryEpisodePlan) -> dict[str, object]:
    if type(plan) is not ConfirmatoryEpisodePlan:
        raise ValueError("plan must be an exact ConfirmatoryEpisodePlan")
    return plan.plan_projection()


def hash_confirmatory_episode_plan(plan: ConfirmatoryEpisodePlan) -> str:
    return sha256_ref(episode_plan_projection(plan))


def _by_domain[T](values: tuple[T, ...], domain: Domain) -> tuple[T, ...]:
    return tuple(value for value in values if getattr(value, "domain") == domain)


def _condition_map(
    bindings: tuple[ConditionBinding, ...],
) -> dict[tuple[Domain, ExperimentCondition], ConditionBinding]:
    expected = {
        (domain, condition)
        for domain in DOMAINS
        for condition in STAGE_A_CONDITIONS + STAGE_B_CONDITIONS
    }
    mapping = {(cast(Domain, binding.domain), binding.condition): binding for binding in bindings}
    if len(mapping) != len(bindings) or set(mapping) != expected:
        raise ValueError("condition_bindings must contain every domain-condition pair exactly once")
    return cast(dict[tuple[Domain, ExperimentCondition], ConditionBinding], mapping)


def _require_shared_policy(
    bindings: dict[tuple[Domain, ExperimentCondition], ConditionBinding],
    domain: Domain,
    conditions: tuple[ExperimentCondition, ...],
    field_name: str,
) -> None:
    hashes = {bindings[(domain, condition)].policy_hash for condition in conditions}
    if len(hashes) != 1 or None in hashes:
        raise ValueError(f"{field_name} policy hashes must match within a domain")


def _validate_stage_a(
    skills: tuple[SkillBundleBinding, ...], cases: tuple[HeldOutCaseBinding, ...]
) -> None:
    if len(skills) != 30 or len(cases) != 40:
        raise ValueError("Stage A requires 30 skills and 40 held-out cases")
    if any(case.authority_class is not None for case in cases):
        raise ValueError("Stage A cases cannot declare Stage B authority classes")
    for domain in DOMAINS:
        domain_skills = _by_domain(skills, domain)
        domain_cases = _by_domain(cases, domain)
        if len(domain_cases) != 20:
            raise ValueError("Stage A requires 20 held-out cases per domain")
        ratios = Counter(skill.contamination_ratio for skill in domain_skills)
        if ratios != Counter({ratio: 3 for ratio in RATIOS}):
            raise ValueError("Stage A requires three skill bundles per ratio and domain")
        if len({case.case_id for case in domain_cases}) != len(domain_cases):
            raise ValueError("Stage A case IDs must be unique within a domain")
        if len({skill.bundle_id for skill in domain_skills}) != len(domain_skills):
            raise ValueError("Stage A bundle IDs must be unique within a domain")


def _validate_stage_b(cases: tuple[HeldOutCaseBinding, ...]) -> None:
    if len(cases) != 50:
        raise ValueError("Stage B requires 50 held-out cases")
    for domain in DOMAINS:
        domain_cases = _by_domain(cases, domain)
        if len(domain_cases) != 25:
            raise ValueError("Stage B requires 25 held-out cases per domain")
        classes = Counter(case.authority_class for case in domain_cases)
        if classes != Counter({authority_class: 5 for authority_class in AUTHORITY_CLASSES}):
            raise ValueError("Stage B requires five cases per authority class and domain")
        if len({case.case_id for case in domain_cases}) != len(domain_cases):
            raise ValueError("Stage B case IDs must be unique within a domain")


def _validate_inputs(
    inputs: PlannerInputs,
) -> dict[tuple[Domain, ExperimentCondition], ConditionBinding]:
    for field_name, expected_type in (
        ("stage_a_skills", SkillBundleBinding),
        ("stage_a_cases", HeldOutCaseBinding),
        ("stage_b_cases", HeldOutCaseBinding),
        ("condition_bindings", ConditionBinding),
    ):
        if any(type(value) is not expected_type for value in getattr(inputs, field_name)):
            raise ValueError(f"{field_name} must contain exact {expected_type.__name__} values")
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


def _planned_episode(
    *,
    stage: EpisodeStage,
    condition: ExperimentCondition,
    case: HeldOutCaseBinding,
    bindings: dict[tuple[Domain, ExperimentCondition], ConditionBinding],
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


def plan_confirmatory_episodes(inputs: PlannerInputs) -> ConfirmatoryEpisodePlan:
    """Generate the preregistered Stage A/B matrix from explicit confirmatory custody."""

    if type(inputs) is not PlannerInputs:
        raise ValueError("inputs must be an exact PlannerInputs")
    bindings = _validate_inputs(inputs)
    episodes: list[PlannedEpisode] = []

    for domain in DOMAINS:
        cases = tuple(
            sorted(_by_domain(inputs.stage_a_cases, domain), key=lambda case: case.case_id)
        )
        for condition in (
            ExperimentCondition.A0_BARE,
            ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
        ):
            for case in cases:
                for repeat_index in range(1, REPEATS + 1):
                    episodes.append(
                        _planned_episode(
                            stage=EpisodeStage.CONFIRMATORY_A,
                            condition=condition,
                            case=case,
                            bindings=bindings,
                            repeat_index=repeat_index,
                        )
                    )
        skills = tuple(
            sorted(
                _by_domain(inputs.stage_a_skills, domain),
                key=lambda skill: (skill.contamination_ratio, skill.bundle_id),
            )
        )
        for skill in skills:
            for condition in STAGE_A_CONDITIONS[2:]:
                for case_index, case in enumerate(cases):
                    for repeat_index in range(1, REPEATS + 1):
                        order_assignment = None
                        if condition is ExperimentCondition.A3_SKILL_POLICY_SAME_TIER:
                            order_assignment = (
                                BlockOrder.POLICY_THEN_SKILL
                                if (case_index * REPEATS + repeat_index - 1) % 2 == 0
                                else BlockOrder.SKILL_THEN_POLICY
                            )
                        episodes.append(
                            _planned_episode(
                                stage=EpisodeStage.CONFIRMATORY_A,
                                condition=condition,
                                case=case,
                                bindings=bindings,
                                repeat_index=repeat_index,
                                skill=skill,
                                order_assignment=order_assignment,
                            )
                        )

    for domain in DOMAINS:
        cases = tuple(
            sorted(_by_domain(inputs.stage_b_cases, domain), key=lambda case: case.case_id)
        )
        skills = tuple(
            skill
            for skill in sorted(
                _by_domain(inputs.stage_a_skills, domain), key=lambda skill: skill.bundle_id
            )
            if skill.contamination_ratio == Decimal("0.75")
        )
        for skill in skills:
            for case in cases:
                for condition in STAGE_B_CONDITIONS:
                    for repeat_index in range(1, REPEATS + 1):
                        episodes.append(
                            _planned_episode(
                                stage=EpisodeStage.CONFIRMATORY_B,
                                condition=condition,
                                case=case,
                                bindings=bindings,
                                repeat_index=repeat_index,
                                skill=skill,
                            )
                        )

    stage_a_count = sum(episode.stage is EpisodeStage.CONFIRMATORY_A for episode in episodes)
    stage_b_count = sum(episode.stage is EpisodeStage.CONFIRMATORY_B for episode in episodes)
    if (stage_a_count, stage_b_count) != (STAGE_A_EPISODE_COUNT, STAGE_B_EPISODE_COUNT):
        raise ValueError("episode matrix does not match preregistered confirmatory counts")
    return ConfirmatoryEpisodePlan(episodes=tuple(episodes))


__all__ = [
    "AUTHORITY_CLASSES",
    "DOMAINS",
    "RATIOS",
    "REPEATS",
    "STAGE_A_CONDITIONS",
    "STAGE_A_EPISODE_COUNT",
    "STAGE_B_CONDITIONS",
    "STAGE_B_EPISODE_COUNT",
    "ConditionBinding",
    "ConfirmatoryEpisodePlan",
    "HeldOutCaseBinding",
    "PlannerInputs",
    "PlannedEpisode",
    "SkillBundleBinding",
    "episode_manifest_projection",
    "episode_plan_projection",
    "hash_confirmatory_episode_plan",
    "hash_planned_episode_manifest",
    "plan_confirmatory_episodes",
]
