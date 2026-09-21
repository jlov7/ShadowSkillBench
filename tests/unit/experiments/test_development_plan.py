from __future__ import annotations

from collections import Counter

import pytest

from shadowskillbench.corpus.development import generate_development_corpus
from shadowskillbench.episodes import ExperimentCondition
from shadowskillbench.experiments.development_plan import (
    DevelopmentEpisodePlan,
    DevelopmentPlannedEpisode,
    build_development_plan,
)


def test_adr001_development_plan_has_exact_stage_counts() -> None:
    plan = build_development_plan(generate_development_corpus(4242))

    assert len(plan.episodes) == 600
    assert Counter(episode.stage for episode in plan.episodes) == {"stage_a": 440, "stage_b": 160}
    assert Counter(episode.domain for episode in plan.episodes) == {
        "access_provisioning": 300,
        "financial_adjustments": 300,
    }


def test_development_plan_admits_only_the_five_committed_corpus_seeds() -> None:
    episode = DevelopmentPlannedEpisode(
        "stage_a",
        ExperimentCondition.A0_BARE,
        "development_test",
        "access_provisioning",
        None,
    )
    episodes = (episode,) * 600

    assert DevelopmentEpisodePlan(seed=4242, episodes=episodes).seed == 4242
    assert DevelopmentEpisodePlan(seed=4243, episodes=episodes).seed == 4243
    assert DevelopmentEpisodePlan(seed=4244, episodes=episodes).seed == 4244
    assert DevelopmentEpisodePlan(seed=4245, episodes=episodes).seed == 4245
    assert DevelopmentEpisodePlan(seed=4246, episodes=episodes).seed == 4246
    with pytest.raises(ValueError, match="ADR-001 development matrix"):
        DevelopmentEpisodePlan(seed=4247, episodes=episodes)
