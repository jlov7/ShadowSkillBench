"""Deterministic ADR-001 development episode matrix."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from shadowskillbench.corpus.development import DevelopmentCorpus
from shadowskillbench.episodes.models import ExperimentCondition
from shadowskillbench.traces.bundles import generate_bundle

_RATIOS = (Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1"))
_STAGE_A = (
    ExperimentCondition.A0_BARE,
    ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
    ExperimentCondition.A2_SKILL_ONLY,
    ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
    ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
    ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER,
)
_STAGE_B = (
    ExperimentCondition.B0_SKILL_ONLY,
    ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
    ExperimentCondition.B2_AUTHORITY_RESOLVER,
    ExperimentCondition.B3_DETERMINISTIC_GATE,
)


@dataclass(frozen=True, slots=True)
class DevelopmentPlannedEpisode:
    stage: str
    condition: ExperimentCondition
    case_id: str
    domain: str
    bundle_id: str | None


@dataclass(frozen=True, slots=True)
class DevelopmentEpisodePlan:
    seed: int
    episodes: tuple[DevelopmentPlannedEpisode, ...]

    def __post_init__(self) -> None:
        if self.seed not in {4242, 4243, 4244, 4245, 4246} or len(self.episodes) != 600:
            raise ValueError("ADR-001 development matrix is invalid")


def build_development_plan(corpus: DevelopmentCorpus) -> DevelopmentEpisodePlan:
    if type(corpus) is not DevelopmentCorpus:
        raise ValueError("corpus must be an exact DevelopmentCorpus")
    episodes: list[DevelopmentPlannedEpisode] = []
    for domain in ("access_provisioning", "financial_adjustments"):
        cases = tuple(case for case in corpus.cases if case.domain == domain)
        active_cases = cases[:10]
        bundles = {
            ratio: generate_bundle(domain, ratio, 12, corpus.seed).bundle.source_manifest.bundle_id
            for ratio in _RATIOS
        }
        for case in active_cases:
            for condition in _STAGE_A[:2]:
                episodes.append(
                    DevelopmentPlannedEpisode("stage_a", condition, case.case_id, domain, None)
                )
        for ratio in _RATIOS:
            for case in active_cases:
                for condition in _STAGE_A[2:]:
                    episodes.append(
                        DevelopmentPlannedEpisode(
                            "stage_a", condition, case.case_id, domain, bundles[ratio]
                        )
                    )
        for case in cases:
            for condition in _STAGE_B:
                episodes.append(
                    DevelopmentPlannedEpisode(
                        "stage_b", condition, case.case_id, domain, bundles[Decimal("0.75")]
                    )
                )
    return DevelopmentEpisodePlan(seed=corpus.seed, episodes=tuple(episodes))


__all__ = ["DevelopmentEpisodePlan", "DevelopmentPlannedEpisode", "build_development_plan"]
