from __future__ import annotations

from collections import Counter
from dataclasses import replace
from decimal import Decimal

import pytest

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.episodes.models import BlockOrder, EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.planner import (
    AUTHORITY_CLASSES,
    DOMAINS,
    RATIOS,
    STAGE_A_EPISODE_COUNT,
    STAGE_B_EPISODE_COUNT,
    ConditionBinding,
    HeldOutCaseBinding,
    PlannerInputs,
    SkillBundleBinding,
    hash_confirmatory_episode_plan,
    hash_planned_episode_manifest,
    plan_confirmatory_episodes,
)


def _hash(*parts: object) -> str:
    return sha256_ref(list(parts))


def _skill(domain: str, ratio: Decimal, seed: int, stage: str) -> SkillBundleBinding:
    identity = (stage, domain, str(ratio), seed)
    return SkillBundleBinding(
        domain=domain,  # type: ignore[arg-type]
        bundle_id=f"{stage}_{domain}_{str(ratio).replace('.', '_')}_{seed}",
        contamination_ratio=ratio,
        source_manifest_hash=_hash("source", *identity),
        compiler_manifest_hash=_hash("compiler", *identity),
        compiled_skill_artifact_hash=_hash("artifact", *identity),
        rendered_skill_hash=_hash("rendered", *identity),
    )


def _case(
    domain: str, stage: str, number: int, authority_class: str | None = None
) -> HeldOutCaseBinding:
    identity = (stage, domain, number)
    return HeldOutCaseBinding(
        domain=domain,  # type: ignore[arg-type]
        case_id=f"{stage}_{domain}_{number:02d}",
        case_manifest_hash=_hash("case", *identity),
        world_hash=_hash("world", *identity),
        authority_graph_hash=_hash("authority", *identity),
        authority_class=authority_class,  # type: ignore[arg-type]
    )


def _inputs() -> PlannerInputs:
    stage_a_skills = tuple(
        _skill(domain, ratio, seed, "a")
        for domain in DOMAINS
        for ratio in RATIOS
        for seed in range(3)
    )
    stage_a_cases = tuple(_case(domain, "a", number) for domain in DOMAINS for number in range(20))
    stage_b_cases = tuple(
        _case(domain, "b", class_index * 5 + number, authority_class)
        for domain in DOMAINS
        for class_index, authority_class in enumerate(AUTHORITY_CLASSES)
        for number in range(5)
    )
    condition_bindings = tuple(
        ConditionBinding(
            domain=domain,
            condition=condition,
            context_contract_hash=_hash("context", domain, condition.value),
            policy_hash=(
                _hash("a-current-policy", domain)
                if condition
                in {
                    ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
                    ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
                    ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
                }
                else _hash("b-policy", domain)
                if condition
                in {
                    ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
                    ExperimentCondition.B2_AUTHORITY_RESOLVER,
                    ExperimentCondition.B3_DETERMINISTIC_GATE,
                }
                else _hash("a5-handbook", domain)
                if condition is ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER
                else None
            ),
        )
        for domain in DOMAINS
        for condition in ExperimentCondition
    )
    return PlannerInputs(
        stage_a_skills=stage_a_skills,
        stage_a_cases=stage_a_cases,
        stage_b_cases=stage_b_cases,
        condition_bindings=condition_bindings,
    )


def test_plan_has_preregistered_counts_and_control_replication() -> None:
    plan = plan_confirmatory_episodes(_inputs())
    counts = Counter((episode.stage, episode.condition) for episode in plan.episodes)

    assert len(plan.episodes) == STAGE_A_EPISODE_COUNT + STAGE_B_EPISODE_COUNT
    assert (
        sum(count for (stage, _), count in counts.items() if stage is EpisodeStage.CONFIRMATORY_A)
        == 7_440
    )
    assert (
        sum(count for (stage, _), count in counts.items() if stage is EpisodeStage.CONFIRMATORY_B)
        == 1_800
    )
    assert counts[(EpisodeStage.CONFIRMATORY_A, ExperimentCondition.A0_BARE)] == 120
    assert counts[(EpisodeStage.CONFIRMATORY_A, ExperimentCondition.A1_POLICY_ONLY_SYSTEM)] == 120
    for condition in (
        ExperimentCondition.A2_SKILL_ONLY,
        ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
        ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
        ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER,
    ):
        assert counts[(EpisodeStage.CONFIRMATORY_A, condition)] == 1_800
    for condition in (
        ExperimentCondition.B0_SKILL_ONLY,
        ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
        ExperimentCondition.B2_AUTHORITY_RESOLVER,
        ExperimentCondition.B3_DETERMINISTIC_GATE,
    ):
        assert counts[(EpisodeStage.CONFIRMATORY_B, condition)] == 450
    stage_a_r75_ids = {
        skill.bundle_id
        for skill in _inputs().stage_a_skills
        if skill.contamination_ratio == Decimal("0.75")
    }
    stage_b_skill_ids = {
        episode.skill.bundle_id
        for episode in plan.episodes
        if episode.stage is EpisodeStage.CONFIRMATORY_B and episode.skill is not None
    }
    assert stage_b_skill_ids == stage_a_r75_ids


def test_plan_has_no_duplicate_cells_or_manifests() -> None:
    plan = plan_confirmatory_episodes(_inputs())

    assert len({episode.cell_key for episode in plan.episodes}) == len(plan.episodes)
    assert len({episode.manifest_hash for episode in plan.episodes}) == len(plan.episodes)


def test_plan_is_deterministic_and_a3_is_counterbalanced_per_skill() -> None:
    first = plan_confirmatory_episodes(_inputs())
    second = plan_confirmatory_episodes(_inputs())
    a3 = [
        episode
        for episode in first.episodes
        if episode.condition is ExperimentCondition.A3_SKILL_POLICY_SAME_TIER
    ]

    assert first == second
    assert first.plan_hash == hash_confirmatory_episode_plan(first)
    assert first.plan_hash == second.plan_hash
    assert Counter(episode.order_assignment for episode in a3) == Counter(
        {BlockOrder.POLICY_THEN_SKILL: 900, BlockOrder.SKILL_THEN_POLICY: 900}
    )
    for skill_id in {episode.skill.bundle_id for episode in a3 if episode.skill is not None}:
        assignments = Counter(
            episode.order_assignment
            for episode in a3
            if episode.skill is not None and episode.skill.bundle_id == skill_id
        )
        assert assignments == Counter(
            {BlockOrder.POLICY_THEN_SKILL: 30, BlockOrder.SKILL_THEN_POLICY: 30}
        )


def test_conditions_admit_only_the_required_context_and_skill_bindings() -> None:
    plan = plan_confirmatory_episodes(_inputs())

    for episode in plan.episodes:
        assert (episode.skill is not None) is episode.condition.requires_skill
        assert (
            episode.condition_binding.policy_hash is not None
        ) is episode.condition.requires_policy
        assert (episode.order_assignment is not None) is episode.condition.requires_block_order


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        (ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER, "Stage A current-policy"),
        (ExperimentCondition.B2_AUTHORITY_RESOLVER, "Stage B policy"),
    ],
)
def test_planner_rejects_policy_drift_inside_matched_conditions(
    condition: ExperimentCondition, message: str
) -> None:
    valid = _inputs()
    bindings = tuple(
        replace(binding, policy_hash=_hash("drift", binding.domain, condition.value))
        if binding.domain == "access_provisioning" and binding.condition is condition
        else binding
        for binding in valid.condition_bindings
    )

    with pytest.raises(ValueError, match=message):
        plan_confirmatory_episodes(
            PlannerInputs(
                stage_a_skills=valid.stage_a_skills,
                stage_a_cases=valid.stage_a_cases,
                stage_b_cases=valid.stage_b_cases,
                condition_bindings=bindings,
            )
        )


def test_every_episode_and_plan_hash_binds_its_supplied_custody() -> None:
    plan = plan_confirmatory_episodes(_inputs())
    episode = next(
        item
        for item in plan.episodes
        if item.condition is ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER
    )
    assert episode.skill is not None
    projection = episode.manifest_projection()

    assert projection["case_manifest_hash"] == episode.case.case_manifest_hash
    assert projection["world_hash"] == episode.case.world_hash
    assert projection["authority_graph_hash"] == episode.case.authority_graph_hash
    assert projection["source_manifest_hash"] == episode.skill.source_manifest_hash
    assert projection["compiler_manifest_hash"] == episode.skill.compiler_manifest_hash
    assert projection["compiled_skill_artifact_hash"] == episode.skill.compiled_skill_artifact_hash
    assert projection["rendered_skill_hash"] == episode.skill.rendered_skill_hash
    assert projection["policy_hash"] == episode.condition_binding.policy_hash
    assert projection["context_contract_hash"] == episode.condition_binding.context_contract_hash
    assert projection["repeat_index"] == episode.repeat_index
    assert 1 <= episode.repeat_index <= 3
    assert episode.repeat_index == episode.cell_key[-1]
    assert episode.manifest_hash == hash_planned_episode_manifest(episode)

    with pytest.raises(ValueError, match="Stage B requires"):
        bad = _inputs()
        plan_confirmatory_episodes(
            PlannerInputs(
                stage_a_skills=bad.stage_a_skills,
                stage_a_cases=bad.stage_a_cases,
                stage_b_cases=bad.stage_b_cases[:-1],
                condition_bindings=bad.condition_bindings,
            )
        )
