"""Explicit V4 staging route with a version-bound package/output namespace."""

from __future__ import annotations

from dataclasses import dataclass

from shadowskillbench.corpus.confirmatory import SplitInventory
from shadowskillbench.corpus.confirmatory_v4 import (
    CORPUS_SEED_V4,
    ConfirmatoryAuthorizationV4,
    ConfirmatoryCorpusV4,
    generate_confirmatory_corpus_v4,
)
from shadowskillbench.experiments.planner import (
    ConditionBinding,
    ConfirmatoryEpisodePlan,
    HeldOutCaseBinding,
    PlannerInputs,
    SkillBundleBinding,
)
from shadowskillbench.experiments.planner_v4 import (
    STAGE_A_EPISODE_COUNT,
    STAGE_B_EPISODE_COUNT,
    plan_confirmatory_episodes_v4,
)

V4_PACKAGE_NAMESPACE = "artifacts/experiments/confirmatory-v4-package"
V4_RUNTIME_NAMESPACE = "artifacts/experiments/confirmatory-v4"


@dataclass(frozen=True, slots=True)
class ConfirmatoryPreparationV4:
    corpus: ConfirmatoryCorpusV4
    plan: ConfirmatoryEpisodePlan
    package_namespace: str = V4_PACKAGE_NAMESPACE
    runtime_namespace: str = V4_RUNTIME_NAMESPACE

    @property
    def plan_episode_count(self) -> int:
        return len(self.plan.episodes)

    @property
    def plan_hash(self) -> str:
        return self.plan.plan_hash


def stage_confirmatory_execution_v4(
    *,
    authorization: ConfirmatoryAuthorizationV4,
    development_inventory: SplitInventory,
    stage_a_skills: tuple[SkillBundleBinding, ...],
    stage_a_cases: tuple[HeldOutCaseBinding, ...],
    stage_b_cases: tuple[HeldOutCaseBinding, ...],
    condition_bindings: tuple[ConditionBinding, ...],
) -> ConfirmatoryPreparationV4:
    """Stage the fixed V4 matrix without provider calls or artifact writes."""

    corpus = generate_confirmatory_corpus_v4(
        authorization=authorization,
        corpus_seed=CORPUS_SEED_V4,
        development_inventory=development_inventory,
    )
    plan = plan_confirmatory_episodes_v4(
        PlannerInputs(stage_a_skills, stage_a_cases, stage_b_cases, condition_bindings)
    )
    if len(plan.episodes) != STAGE_A_EPISODE_COUNT + STAGE_B_EPISODE_COUNT:
        raise ValueError("V4 staged plan does not have frozen counts")
    return ConfirmatoryPreparationV4(corpus, plan)


__all__ = [
    "ConfirmatoryPreparationV4",
    "V4_PACKAGE_NAMESPACE",
    "V4_RUNTIME_NAMESPACE",
    "stage_confirmatory_execution_v4",
]
