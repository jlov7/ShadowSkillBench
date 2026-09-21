from __future__ import annotations

import pytest

from shadowskillbench.analysis.dataset import AnalysisDataset, ExclusionRow
from shadowskillbench.analysis.stage_a import StageAAnalysisError, _interval, analyze_stage_a
from shadowskillbench.authority import DecisionDisposition
from shadowskillbench.episodes import EpisodeStage, ExperimentCondition


def test_simultaneous_interval_uses_bonferroni_familywise_quantiles() -> None:
    values = list(range(999))

    assert _interval(values) == (24, 973)
    assert _interval(values, (1.0 - 0.95) / 18) == (2, 995)


def test_rejects_an_incomplete_primary_matrix() -> None:
    with pytest.raises(StageAAnalysisError, match="incomplete"):
        analyze_stage_a(
            AnalysisDataset(rows=(), exclusions=(), aggregates=()), bootstrap_replicates=100
        )


def test_rejects_a_stage_a_technical_exclusion_before_estimation() -> None:
    exclusion = ExclusionRow(
        stage=EpisodeStage.CONFIRMATORY_A,
        condition=ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
        domain="access_provisioning",
        contamination_ratio=None,
        skill_bundle_id=None,
        authority_class=None,
        order_assignment=None,
        expected_authority_disposition=DecisionDisposition.PROCEED,
        held_out_case_id="case-00",
        repeat_index=1,
        episode_id="excluded",
        planned_episode_hash="planned",
        manifest_hash="manifest",
        result_hash="result",
        score_hash="score",
        error_code="MODEL_PROVIDER_TRANSIENT",
    )
    dataset = AnalysisDataset(rows=(), exclusions=(exclusion,), aggregates=())

    with pytest.raises(StageAAnalysisError, match="preapproved"):
        analyze_stage_a(dataset, bootstrap_replicates=100)


@pytest.mark.parametrize("replicates, seed", [(99, 0), (100, -1)])
def test_rejects_invalid_bootstrap_parameters(replicates: int, seed: int) -> None:
    dataset = AnalysisDataset(rows=(), exclusions=(), aggregates=())

    with pytest.raises(StageAAnalysisError):
        analyze_stage_a(dataset, bootstrap_replicates=replicates, bootstrap_seed=seed)
