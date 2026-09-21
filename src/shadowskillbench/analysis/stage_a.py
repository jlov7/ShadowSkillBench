from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import floor
from random import Random
from typing import Any, Literal, cast

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from shadowskillbench.analysis.dataset import AnalysisDataset, ExclusionRow, OutcomeRow
from shadowskillbench.analysis.exclusions import ApprovedTechnicalExclusionPolicy
from shadowskillbench.analysis.profile import (
    ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
    ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES,
    ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED,
    STAGE_A_METHOD,
    AnalysisProfileBinding,
    binding_for,
    frozen_member_execution_designs,
    simultaneous_profile_hash,
)
from shadowskillbench.episodes import BlockOrder, EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.planner import DOMAINS, RATIOS, REPEATS, STAGE_A_EPISODE_COUNT
from shadowskillbench.experiments.planner_v4 import (
    STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO as V4_STAGE_A_BUNDLES_PER_RATIO,
)
from shadowskillbench.experiments.planner_v4 import (
    STAGE_A_CASES_PER_DOMAIN as V4_STAGE_A_CASES_PER_DOMAIN,
)
from shadowskillbench.experiments.planner_v4 import (
    STAGE_A_EPISODE_COUNT as V4_STAGE_A_EPISODE_COUNT,
)
from shadowskillbench.protocol.scientific_freeze import (
    MULTIPLICITY_MEMBERS,
    ScientificFreezeBinding,
)
from shadowskillbench.protocol.scientific_freeze_v4 import (
    MULTIPLICITY_MEMBERS_V4,
    ScientificFreezeV4Binding,
)

type Domain = Literal["access_provisioning", "financial_adjustments"]
type Scope = Domain | Literal["pooled"]
type EstimandName = Literal["E1", "E2", "E3", "E4", "E5"]
type EstimandComponent = Literal["slope", "average_cup_contrast"]
type MarginalEffect = Literal["contamination_ratio", "condition_contrast"]

_SKILL_CONDITIONS = (
    ExperimentCondition.A2_SKILL_ONLY,
    ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
    ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
    ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER,
)
_CURVE_CONDITIONS = (ExperimentCondition.A1_POLICY_ONLY_SYSTEM, *_SKILL_CONDITIONS)
_ALL_CONDITIONS = (ExperimentCondition.A0_BARE, *_CURVE_CONDITIONS)
_RATIO_VALUES = tuple(float(ratio) for ratio in RATIOS)
_SCOPE_VALUES: tuple[Scope, ...] = (*DOMAINS, "pooled")
_ESTIMAND_LAYOUT: tuple[tuple[EstimandName, EstimandComponent], ...] = (
    ("E1", "slope"),
    ("E2", "slope"),
    ("E3", "slope"),
    ("E3", "average_cup_contrast"),
    ("E4", "average_cup_contrast"),
    ("E5", "average_cup_contrast"),
)
_MODEL_CONDITIONS = _SKILL_CONDITIONS[1:]
_PARAMETER_NAMES = (
    "intercept",
    "contamination_ratio",
    "condition_A3_SKILL_POLICY_SAME_TIER",
    "condition_A4_SKILL_POLICY_SYSTEM_TIER",
    "condition_A5_SKILL_BURIED_POLICY_SAME_TIER",
    "contamination_ratio:condition_A3_SKILL_POLICY_SAME_TIER",
    "contamination_ratio:condition_A4_SKILL_POLICY_SYSTEM_TIER",
    "contamination_ratio:condition_A5_SKILL_BURIED_POLICY_SAME_TIER",
    "domain_financial_adjustments",
)


class StageAAnalysisError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RawCuPCount:
    """Observed completion-under-policy numerator and denominator for one analysis cell."""

    scope: Scope
    condition: ExperimentCondition
    contamination_ratio: float | None
    numerator: int
    denominator: int


@dataclass(frozen=True, slots=True)
class ResponseCurvePoint:
    scope: Scope
    condition: ExperimentCondition
    contamination_ratio: float
    numerator: int
    denominator: int
    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class StageAEstimand:
    estimand: EstimandName
    component: EstimandComponent
    scope: Scope
    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class AverageMarginalEffect:
    """Model-based average marginal effect on Completion Under Policy."""

    effect: MarginalEffect
    scope: Scope
    condition: ExperimentCondition
    reference_condition: ExperimentCondition | None
    contamination_ratio: float
    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class StageAInference:
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    method: str
    classification: Literal["CONFIRMATORY", "NONCONFIRMATORY"] = "NONCONFIRMATORY"
    profile: str = "SSB-NONCONFIRMATORY-STATISTICS1"
    profile_hash: str = ""
    statistics_configuration_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class StageAModel:
    formula: str
    family: str
    link: str
    parameter_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    observation_count: int
    skill_bundle_cluster_count: int
    held_out_case_cluster_count: int
    converged: bool


@dataclass(frozen=True, slots=True)
class StageAAnalysis:
    raw_counts: tuple[RawCuPCount, ...]
    response_curves: tuple[ResponseCurvePoint, ...]
    estimands: tuple[StageAEstimand, ...]
    average_marginal_effects: tuple[AverageMarginalEffect, ...]
    model: StageAModel
    inference: StageAInference


type GroupKey = tuple[ExperimentCondition, Domain, float | None]
type CurveKey = tuple[Scope, ExperimentCondition, float]
type EstimandKey = tuple[str, EstimandComponent, Scope]
type MarginalEffectKey = tuple[
    MarginalEffect,
    Scope,
    ExperimentCondition,
    ExperimentCondition | None,
    float,
]


def _ratio(value: float | None) -> float:
    if type(value) is not float:
        raise StageAAnalysisError("skill condition contamination_ratio must be an exact float")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as error:
        raise StageAAnalysisError("skill condition contamination_ratio is invalid") from error
    if not decimal.is_finite() or float(decimal) not in _RATIO_VALUES:
        raise StageAAnalysisError("skill condition contamination_ratio is not preregistered")
    return float(decimal)


def _stage_a_rows(
    dataset: AnalysisDataset,
    exclusion_policy: ApprovedTechnicalExclusionPolicy | None,
    expected_episode_count: int,
) -> tuple[tuple[OutcomeRow, ...], tuple[ExclusionRow, ...]]:
    if type(dataset) is not AnalysisDataset:
        raise StageAAnalysisError("dataset must be an exact AnalysisDataset")
    if (
        exclusion_policy is not None
        and type(exclusion_policy) is not ApprovedTechnicalExclusionPolicy
    ):
        raise StageAAnalysisError("exclusion_policy must be an exact approved policy")
    exclusions = tuple(
        row for row in dataset.exclusions if row.stage is EpisodeStage.CONFIRMATORY_A
    )
    if any(type(row) is not ExclusionRow for row in exclusions):
        raise StageAAnalysisError("Stage A exclusions must be exact ExclusionRow records")
    if exclusions and exclusion_policy is None:
        raise StageAAnalysisError(
            "Stage A exclusions require a preapproved technical exclusion policy"
        )
    if exclusion_policy is not None:
        if len(exclusions) > exclusion_policy.max_total_exclusions:
            raise StageAAnalysisError("Stage A exclusions exceed the preapproved total cap")
        if any(row.error_code not in exclusion_policy.allowed_error_codes for row in exclusions):
            raise StageAAnalysisError("Stage A exclusion is not preapproved")
    rows = tuple(row for row in dataset.rows if row.stage is EpisodeStage.CONFIRMATORY_A)
    if len(rows) + len(exclusions) != expected_episode_count:
        raise StageAAnalysisError("Stage A primary matrix is incomplete")
    if any(type(row) is not OutcomeRow for row in rows):
        raise StageAAnalysisError("Stage A rows must be exact OutcomeRow records")
    return rows, exclusions


def _validate_primary_matrix(
    rows: tuple[OutcomeRow, ...],
    exclusions: tuple[ExclusionRow, ...],
    exclusion_policy: ApprovedTechnicalExclusionPolicy | None,
    *,
    cases_per_domain: int,
    bundles_per_ratio: int,
) -> None:
    seen_episode_ids: set[str] = set()
    observed_keys: set[tuple[object, ...]] = set()
    cases_by_domain: dict[Domain, set[str]] = {domain: set() for domain in DOMAINS}
    bundles_by_domain_ratio: dict[tuple[Domain, float], set[str]] = defaultdict(set)
    a3_order_counts: Counter[tuple[Domain, float, str, BlockOrder]] = Counter()
    included_repeats: Counter[tuple[object, ...]] = Counter()

    for row in (*rows, *exclusions):
        if row.condition not in _ALL_CONDITIONS or row.domain not in DOMAINS:
            raise StageAAnalysisError("Stage A has an invalid condition or domain")
        if row.episode_id in seen_episode_ids:
            raise StageAAnalysisError("Stage A has duplicate episode_id values")
        seen_episode_ids.add(row.episode_id)
        if row.repeat_index not in range(1, REPEATS + 1):
            raise StageAAnalysisError("Stage A has an invalid repeat_index")
        if row.condition in _SKILL_CONDITIONS:
            ratio = _ratio(row.contamination_ratio)
            if type(row.skill_bundle_id) is not str or not row.skill_bundle_id:
                raise StageAAnalysisError("skill condition requires skill_bundle_id")
            if row.condition is ExperimentCondition.A3_SKILL_POLICY_SAME_TIER:
                if row.order_assignment is None or type(row.order_assignment) is not BlockOrder:
                    raise StageAAnalysisError("A3 requires a block-order assignment")
                a3_order_counts[(row.domain, ratio, row.skill_bundle_id, row.order_assignment)] += 1
            elif row.order_assignment is not None:
                raise StageAAnalysisError("only A3 may have a block-order assignment")
            bundles_by_domain_ratio[(row.domain, ratio)].add(row.skill_bundle_id)
        else:
            if row.contamination_ratio is not None or row.skill_bundle_id is not None:
                raise StageAAnalysisError("non-skill Stage A conditions cannot bind a skill")
            if row.order_assignment is not None:
                raise StageAAnalysisError("non-skill Stage A conditions cannot have block order")
            ratio = None
        cases_by_domain[row.domain].add(row.held_out_case_id)
        key = (
            row.condition,
            row.domain,
            ratio,
            row.skill_bundle_id,
            row.held_out_case_id,
            row.repeat_index,
        )
        if key in observed_keys:
            raise StageAAnalysisError("Stage A has duplicate primary cells")
        observed_keys.add(key)
        if type(row) is OutcomeRow:
            included_repeats[
                (row.condition, row.domain, ratio, row.skill_bundle_id, row.held_out_case_id)
            ] += 1

    expected_keys: set[tuple[object, ...]] = set()
    for domain in DOMAINS:
        cases = cases_by_domain[domain]
        if len(cases) != cases_per_domain:
            raise StageAAnalysisError(
                f"Stage A requires {cases_per_domain} held-out cases per domain"
            )
        for ratio in _RATIO_VALUES:
            bundles = bundles_by_domain_ratio[(domain, ratio)]
            if len(bundles) != bundles_per_ratio:
                raise StageAAnalysisError(
                    f"Stage A requires {bundles_per_ratio} bundles per ratio and domain"
                )
            for bundle_id in bundles:
                expected_order_count = cases_per_domain * REPEATS // 2
                if (
                    a3_order_counts[(domain, ratio, bundle_id, BlockOrder.POLICY_THEN_SKILL)]
                    != expected_order_count
                    or a3_order_counts[(domain, ratio, bundle_id, BlockOrder.SKILL_THEN_POLICY)]
                    != expected_order_count
                ):
                    raise StageAAnalysisError(
                        f"A3 requires a {expected_order_count}/{expected_order_count} "
                        "block-order balance per domain, ratio, and bundle"
                    )
            for condition in _SKILL_CONDITIONS:
                for bundle_id in bundles:
                    for case_id in cases:
                        for repeat_index in range(1, REPEATS + 1):
                            expected_keys.add(
                                (condition, domain, ratio, bundle_id, case_id, repeat_index)
                            )
        for condition in (ExperimentCondition.A0_BARE, ExperimentCondition.A1_POLICY_ONLY_SYSTEM):
            for case_id in cases:
                for repeat_index in range(1, REPEATS + 1):
                    expected_keys.add((condition, domain, None, None, case_id, repeat_index))
    if observed_keys != expected_keys:
        raise StageAAnalysisError("Stage A primary matrix has missing or extra cells")
    if exclusion_policy is not None:
        exclusion_counts = Counter(
            (
                row.condition,
                row.domain,
                None
                if row.condition
                in {ExperimentCondition.A0_BARE, ExperimentCondition.A1_POLICY_ONLY_SYSTEM}
                else _ratio(row.contamination_ratio),
                row.skill_bundle_id,
                row.held_out_case_id,
            )
            for row in exclusions
        )
        if any(
            count > exclusion_policy.max_exclusions_per_primary_cell
            for count in exclusion_counts.values()
        ):
            raise StageAAnalysisError("Stage A exclusions exceed the preapproved per-cell cap")
    primary_cells = {
        (condition, domain, ratio, bundle_id, case_id)
        for condition, domain, ratio, bundle_id, case_id, _ in expected_keys
    }
    if any(included_repeats[cell] < 2 for cell in primary_cells):
        raise StageAAnalysisError("Stage A exclusions must retain at least two included repeats")


def _validate_frozen_execution_design(
    rows: tuple[OutcomeRow, ...],
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding,
) -> None:
    try:
        designs = frozen_member_execution_designs(scientific_freeze)
        members = (
            MULTIPLICITY_MEMBERS_V4
            if type(scientific_freeze) is ScientificFreezeV4Binding
            else MULTIPLICITY_MEMBERS
        )
        expected = {
            designs[member]
            for member in members
            if member.startswith(("E1:", "E2:", "E3:", "E4:", "E5:"))
        }
    except (KeyError, ValueError) as error:
        raise StageAAnalysisError("invalid frozen execution design") from error
    if len(expected) != 1:
        raise StageAAnalysisError("invalid frozen execution design")
    observed = (
        len({row.held_out_case_id for row in rows}),
        len({row.skill_bundle_id for row in rows if row.skill_bundle_id is not None}),
        len({row.repeat_index for row in rows}),
    )
    if observed != expected.pop():
        raise StageAAnalysisError("HOLD_FROZEN_EXECUTION_DESIGN_MISMATCH")


def _freeze_layout(
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None,
) -> tuple[int, int, int]:
    """Return only the design selected by an exact scientific-freeze type."""

    if scientific_freeze is None or type(scientific_freeze) is ScientificFreezeBinding:
        return (STAGE_A_EPISODE_COUNT, 20, 3)
    if type(scientific_freeze) is ScientificFreezeV4Binding:
        return (
            V4_STAGE_A_EPISODE_COUNT,
            V4_STAGE_A_CASES_PER_DOMAIN,
            V4_STAGE_A_BUNDLES_PER_RATIO,
        )
    raise StageAAnalysisError("scientific_freeze must be an exact V3 or V4 binding")


def _validate_scientific_freeze(
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> None:
    if scientific_freeze is None:
        return
    if type(scientific_freeze) is ScientificFreezeBinding:
        valid = (
            scientific_freeze.analysis_profile == ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE
            and scientific_freeze.multiplicity_family == "E1-E9-primary-components-16"
            and scientific_freeze.multiplicity_members == MULTIPLICITY_MEMBERS
            and scientific_freeze.multiplicity_method
            == "bonferroni_simultaneous_bootstrap_intervals"
            and scientific_freeze.familywise_confidence_level == 0.95
            and scientific_freeze.bootstrap_replicates == ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES
            and scientific_freeze.bootstrap_seed == ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED
        )
    elif type(scientific_freeze) is ScientificFreezeV4Binding:
        valid = (
            scientific_freeze.analysis_profile == ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE
            and scientific_freeze.multiplicity_members == MULTIPLICITY_MEMBERS_V4
            and scientific_freeze.material_threshold == 0.1
            and scientific_freeze.bootstrap_replicates == ROBUST_SIMULTANEOUS_BOOTSTRAP_REPLICATES
            and scientific_freeze.bootstrap_seed == ROBUST_SIMULTANEOUS_BOOTSTRAP_SEED
        )
    else:
        valid = False
    if not valid or (
        bootstrap_replicates != scientific_freeze.bootstrap_replicates
        or bootstrap_seed != scientific_freeze.bootstrap_seed
    ):
        raise StageAAnalysisError("invalid simultaneous scientific freeze binding")


def _weighted_counts(
    rows: tuple[OutcomeRow, ...],
    case_weights: dict[tuple[Domain, str], int] | None = None,
    bundle_weights: dict[tuple[Domain, float, str], int] | None = None,
) -> dict[GroupKey, tuple[int, int]]:
    totals: dict[GroupKey, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        case_weight = (
            1 if case_weights is None else case_weights.get((row.domain, row.held_out_case_id), 0)
        )
        bundle_weight = 1
        ratio: float | None = None
        if row.condition in _SKILL_CONDITIONS:
            ratio = _ratio(row.contamination_ratio)
            assert row.skill_bundle_id is not None
            if bundle_weights is not None:
                bundle_weight = bundle_weights.get((row.domain, ratio, row.skill_bundle_id), 0)
        weight = case_weight * bundle_weight
        total = totals[(row.condition, row.domain, ratio)]
        total[0] += weight if row.completion_under_policy else 0
        total[1] += weight
    return {key: (value[0], value[1]) for key, value in totals.items()}


def _count_for_scope(
    counts: dict[GroupKey, tuple[int, int]],
    scope: Scope,
    condition: ExperimentCondition,
    ratio: float | None,
) -> tuple[int, int]:
    domains: tuple[Domain, ...] = DOMAINS if scope == "pooled" else (scope,)
    numerator = 0
    denominator = 0
    for domain in domains:
        count = counts[(condition, domain, ratio)]
        numerator += count[0]
        denominator += count[1]
    if denominator == 0:
        raise StageAAnalysisError("bootstrap produced an empty primary analysis cell")
    return numerator, denominator


def _rate(
    counts: dict[GroupKey, tuple[int, int]],
    scope: Scope,
    condition: ExperimentCondition,
    ratio: float | None,
) -> float:
    numerator, denominator = _count_for_scope(counts, scope, condition, ratio)
    return numerator / denominator


def _empirical_curve_rates(
    counts: dict[GroupKey, tuple[int, int]], scope: Scope
) -> dict[CurveKey, float]:
    values: dict[CurveKey, float] = {}
    for ratio in _RATIO_VALUES:
        values[(scope, ExperimentCondition.A1_POLICY_ONLY_SYSTEM, ratio)] = _rate(
            counts, scope, ExperimentCondition.A1_POLICY_ONLY_SYSTEM, None
        )
    return values


def _slope(values: tuple[float, ...]) -> float:
    mean_ratio = 0.5
    denominator = sum((ratio - mean_ratio) ** 2 for ratio in _RATIO_VALUES)
    numerator = sum((ratio - mean_ratio) * value for ratio, value in zip(_RATIO_VALUES, values))
    return numerator / denominator


def _estimands_from_curves(curves: dict[CurveKey, float], scope: Scope) -> dict[EstimandKey, float]:
    def by_ratio(condition: ExperimentCondition) -> tuple[float, ...]:
        return tuple(curves[(scope, condition, ratio)] for ratio in _RATIO_VALUES)

    a1 = by_ratio(ExperimentCondition.A1_POLICY_ONLY_SYSTEM)
    a2 = by_ratio(ExperimentCondition.A2_SKILL_ONLY)
    a3 = by_ratio(ExperimentCondition.A3_SKILL_POLICY_SAME_TIER)
    a4 = by_ratio(ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER)
    a5 = by_ratio(ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER)
    return {
        ("E1", "slope", scope): _slope(a2),
        ("E2", "slope", scope): _slope(a3) - _slope(a2),
        ("E3", "slope", scope): _slope(a4) - _slope(a3),
        ("E3", "average_cup_contrast", scope): sum(a4) / len(a4) - sum(a3) / len(a3),
        ("E4", "average_cup_contrast", scope): sum(a4) / len(a4) - sum(a1) / len(a1),
        ("E5", "average_cup_contrast", scope): sum(a5) / len(a5) - sum(a3) / len(a3),
    }


def _design_row(condition: ExperimentCondition, ratio: float, domain: Domain) -> list[float]:
    condition_indicators = [float(condition is item) for item in _MODEL_CONDITIONS]
    return [
        1.0,
        ratio,
        *condition_indicators,
        *(ratio * indicator for indicator in condition_indicators),
        float(domain == "financial_adjustments"),
    ]


@dataclass(frozen=True, slots=True)
class _LogitFit:
    coefficients: np.ndarray[Any, Any]
    covariance: np.ndarray[Any, Any] | None
    observation_count: int
    skill_bundle_cluster_count: int
    held_out_case_cluster_count: int
    converged: bool


def _fit_logit(
    rows: tuple[OutcomeRow, ...],
    case_weights: dict[tuple[Domain, str], int] | None = None,
    bundle_weights: dict[tuple[Domain, float, str], int] | None = None,
    *,
    with_cluster_covariance: bool,
) -> _LogitFit:
    responses: list[float] = []
    designs: list[list[float]] = []
    weights: list[int] = []
    skill_clusters: list[str] = []
    case_clusters: list[str] = []
    for row in rows:
        if row.condition not in _SKILL_CONDITIONS:
            continue
        ratio = _ratio(row.contamination_ratio)
        assert row.skill_bundle_id is not None
        case_weight = (
            1 if case_weights is None else case_weights.get((row.domain, row.held_out_case_id), 0)
        )
        bundle_weight = (
            1
            if bundle_weights is None
            else bundle_weights.get((row.domain, ratio, row.skill_bundle_id), 0)
        )
        weight = case_weight * bundle_weight
        if weight == 0:
            continue
        responses.append(float(row.completion_under_policy))
        designs.append(_design_row(row.condition, ratio, row.domain))
        weights.append(weight)
        skill_clusters.append(f"{row.domain}:{row.skill_bundle_id}")
        case_clusters.append(f"{row.domain}:{row.held_out_case_id}")
    if not responses:
        raise StageAAnalysisError("bootstrap produced no skill-bearing rows")
    endog = np.asarray(responses, dtype=float)
    exog = np.asarray(designs, dtype=float)
    frequency_weights = np.asarray(weights, dtype=float)
    try:
        result = sm.GLM(
            endog,
            exog,
            family=sm.families.Binomial(link=sm.families.links.Logit()),
            freq_weights=frequency_weights,
        ).fit()
    except Exception as error:
        raise StageAAnalysisError("Stage A binomial logistic GLM did not fit") from error
    if not bool(result.converged):
        raise StageAAnalysisError("Stage A binomial logistic GLM did not converge")
    coefficients = np.asarray(result.params, dtype=float)
    if coefficients.shape != (len(_PARAMETER_NAMES),) or not np.isfinite(coefficients).all():
        raise StageAAnalysisError("Stage A binomial logistic GLM has invalid coefficients")
    covariance: np.ndarray[Any, Any] | None = None
    if with_cluster_covariance:
        try:
            covariance = np.asarray(
                cov_cluster_2groups(
                    result,
                    np.asarray(skill_clusters),
                    np.asarray(case_clusters),
                    use_correction=True,
                )[0],
                dtype=float,
            )
        except Exception as error:
            raise StageAAnalysisError(
                "two-way clustered covariance could not be estimated"
            ) from error
        expected_shape = (len(_PARAMETER_NAMES), len(_PARAMETER_NAMES))
        if covariance.shape != expected_shape or not np.isfinite(covariance).all():
            raise StageAAnalysisError("two-way clustered covariance is invalid")
    return _LogitFit(
        coefficients=coefficients,
        covariance=covariance,
        observation_count=len(responses),
        skill_bundle_cluster_count=len(set(skill_clusters)),
        held_out_case_cluster_count=len(set(case_clusters)),
        converged=True,
    )


def _predict(fit: _LogitFit, condition: ExperimentCondition, ratio: float, domain: Domain) -> float:
    value = np.asarray(_design_row(condition, ratio, domain), dtype=float) @ fit.coefficients
    return float(1.0 / (1.0 + np.exp(-value)))


def _model_curve_rates(
    fit: _LogitFit, counts: dict[GroupKey, tuple[int, int]], scope: Scope
) -> dict[CurveKey, float]:
    values = _empirical_curve_rates(counts, scope)
    domains: tuple[Domain, ...] = DOMAINS if scope == "pooled" else (scope,)
    for condition in _SKILL_CONDITIONS:
        for ratio in _RATIO_VALUES:
            values[(scope, condition, ratio)] = sum(
                _predict(fit, condition, ratio, domain) for domain in domains
            ) / len(domains)
    return values


def _contamination_ratio_effect(
    fit: _LogitFit, condition: ExperimentCondition, ratio: float, domain: Domain
) -> float:
    if condition not in _SKILL_CONDITIONS:
        raise StageAAnalysisError("marginal effects require a skill-bearing condition")
    slope = float(fit.coefficients[1])
    if condition in _MODEL_CONDITIONS:
        interaction_index = 5 + _MODEL_CONDITIONS.index(condition)
        slope += float(fit.coefficients[interaction_index])
    prediction = _predict(fit, condition, ratio, domain)
    return prediction * (1.0 - prediction) * slope


def _average_marginal_effects(fit: _LogitFit, scope: Scope) -> dict[MarginalEffectKey, float]:
    """Return model-derived continuous and discrete AMEs for every planned ratio."""

    domains: tuple[Domain, ...] = DOMAINS if scope == "pooled" else (scope,)
    values: dict[MarginalEffectKey, float] = {}
    for condition in _SKILL_CONDITIONS:
        for ratio in _RATIO_VALUES:
            values[("contamination_ratio", scope, condition, None, ratio)] = sum(
                _contamination_ratio_effect(fit, condition, ratio, domain) for domain in domains
            ) / len(domains)
    for condition in _MODEL_CONDITIONS:
        for ratio in _RATIO_VALUES:
            values[
                (
                    "condition_contrast",
                    scope,
                    condition,
                    ExperimentCondition.A2_SKILL_ONLY,
                    ratio,
                )
            ] = sum(
                _predict(fit, condition, ratio, domain)
                - _predict(fit, ExperimentCondition.A2_SKILL_ONLY, ratio, domain)
                for domain in domains
            ) / len(domains)
    return values


def _bootstrap_weights(
    rows: tuple[OutcomeRow, ...], rng: Random
) -> tuple[dict[tuple[Domain, str], int], dict[tuple[Domain, float, str], int]]:
    cases: dict[Domain, tuple[str, ...]] = {}
    bundles: dict[tuple[Domain, float], tuple[str, ...]] = {}
    for domain in DOMAINS:
        cases[domain] = tuple(
            sorted({row.held_out_case_id for row in rows if row.domain == domain})
        )
        for ratio in _RATIO_VALUES:
            bundles[(domain, ratio)] = tuple(
                sorted(
                    cast(
                        set[str],
                        {
                            row.skill_bundle_id
                            for row in rows
                            if row.domain == domain
                            and row.condition in _SKILL_CONDITIONS
                            and _ratio(row.contamination_ratio) == ratio
                        },
                    )
                )
            )
    case_weights: dict[tuple[Domain, str], int] = {}
    bundle_weights: dict[tuple[Domain, float, str], int] = {}
    for domain in DOMAINS:
        case_draws = Counter(rng.choice(cases[domain]) for _ in cases[domain])
        case_weights.update({(domain, case_id): count for case_id, count in case_draws.items()})
        for ratio in _RATIO_VALUES:
            bundle_draws = Counter(
                rng.choice(bundles[(domain, ratio)]) for _ in bundles[(domain, ratio)]
            )
            bundle_weights.update(
                {(domain, ratio, bundle_id): count for bundle_id, count in bundle_draws.items()}
            )
    return case_weights, bundle_weights


def _interval(values: list[float], tail_probability: float = 0.025) -> tuple[float, float]:
    ordered = sorted(values)
    return (
        ordered[floor(tail_probability * (len(ordered) - 1))],
        ordered[floor((1.0 - tail_probability) * (len(ordered) - 1))],
    )


def analyze_stage_a(
    dataset: AnalysisDataset,
    *,
    bootstrap_replicates: int = 999,
    bootstrap_seed: int = 0,
    exclusion_policy: ApprovedTechnicalExclusionPolicy | None = None,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None = None,
) -> StageAAnalysis:
    """Estimate preregistered Stage A CuP curves and contrasts from a complete frozen matrix."""

    if type(bootstrap_replicates) is not int or bootstrap_replicates < 100:
        raise StageAAnalysisError("bootstrap_replicates must be an integer of at least 100")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise StageAAnalysisError("bootstrap_seed must be a nonnegative integer")
    _validate_scientific_freeze(scientific_freeze, bootstrap_replicates, bootstrap_seed)
    expected_episode_count, cases_per_domain, bundles_per_ratio = _freeze_layout(scientific_freeze)
    if scientific_freeze is None:
        tail_probability = 0.025
    else:
        confidence_level = (
            scientific_freeze.familywise_confidence_level
            if type(scientific_freeze) is ScientificFreezeBinding
            else 0.95
        )
        tail_probability = (1.0 - confidence_level) / (
            2 * len(scientific_freeze.multiplicity_members)
        )
    rows, exclusions = _stage_a_rows(dataset, exclusion_policy, expected_episode_count)
    _validate_primary_matrix(
        rows,
        exclusions,
        exclusion_policy,
        cases_per_domain=cases_per_domain,
        bundles_per_ratio=bundles_per_ratio,
    )
    if scientific_freeze is not None:
        _validate_frozen_execution_design(rows, scientific_freeze)
    observed = _weighted_counts(rows)
    fitted_model = _fit_logit(rows, with_cluster_covariance=True)
    if fitted_model.covariance is None:
        raise StageAAnalysisError("two-way clustered covariance is missing")
    observed_curves: dict[CurveKey, float] = {}
    observed_estimands: dict[EstimandKey, float] = {}
    observed_marginal_effects: dict[MarginalEffectKey, float] = {}
    for scope in _SCOPE_VALUES:
        curves = _model_curve_rates(fitted_model, observed, scope)
        observed_curves.update(curves)
        observed_estimands.update(_estimands_from_curves(curves, scope))
        observed_marginal_effects.update(_average_marginal_effects(fitted_model, scope))

    curve_bootstrap: dict[CurveKey, list[float]] = defaultdict(list)
    estimand_bootstrap: dict[EstimandKey, list[float]] = defaultdict(list)
    marginal_effect_bootstrap: dict[MarginalEffectKey, list[float]] = defaultdict(list)
    rng = Random(bootstrap_seed)
    for _ in range(bootstrap_replicates):
        case_weights, bundle_weights = _bootstrap_weights(rows, rng)
        replicate = _weighted_counts(rows, case_weights, bundle_weights)
        replicate_model = _fit_logit(
            rows,
            case_weights,
            bundle_weights,
            with_cluster_covariance=False,
        )
        for scope in _SCOPE_VALUES:
            curves = _model_curve_rates(replicate_model, replicate, scope)
            for curve_key, value in curves.items():
                curve_bootstrap[curve_key].append(value)
            for estimand_key, value in _estimands_from_curves(curves, scope).items():
                estimand_bootstrap[estimand_key].append(value)
            for marginal_effect_key, value in _average_marginal_effects(
                replicate_model, scope
            ).items():
                marginal_effect_bootstrap[marginal_effect_key].append(value)

    raw_counts: list[RawCuPCount] = []
    for scope in _SCOPE_VALUES:
        for condition in _ALL_CONDITIONS:
            ratios = (
                (None,)
                if condition
                in {
                    ExperimentCondition.A0_BARE,
                    ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
                }
                else _RATIO_VALUES
            )
            for ratio in ratios:
                numerator, denominator = _count_for_scope(observed, scope, condition, ratio)
                raw_counts.append(RawCuPCount(scope, condition, ratio, numerator, denominator))

    response_curves: list[ResponseCurvePoint] = []
    for scope in _SCOPE_VALUES:
        for condition in _CURVE_CONDITIONS:
            for ratio in _RATIO_VALUES:
                source_ratio = (
                    None if condition is ExperimentCondition.A1_POLICY_ONLY_SYSTEM else ratio
                )
                numerator, denominator = _count_for_scope(observed, scope, condition, source_ratio)
                ci_low, ci_high = _interval(
                    curve_bootstrap[(scope, condition, ratio)], tail_probability
                )
                response_curves.append(
                    ResponseCurvePoint(
                        scope,
                        condition,
                        ratio,
                        numerator,
                        denominator,
                        observed_curves[(scope, condition, ratio)],
                        ci_low,
                        ci_high,
                    )
                )

    estimands: list[StageAEstimand] = []
    for scope in _SCOPE_VALUES:
        for name, component in _ESTIMAND_LAYOUT:
            estimand_key = (name, component, scope)
            ci_low, ci_high = _interval(estimand_bootstrap[estimand_key], tail_probability)
            estimands.append(
                StageAEstimand(
                    name,
                    component,
                    scope,
                    observed_estimands[estimand_key],
                    ci_low,
                    ci_high,
                )
            )
    average_marginal_effects: list[AverageMarginalEffect] = []
    for scope in _SCOPE_VALUES:
        for condition in _SKILL_CONDITIONS:
            for ratio in _RATIO_VALUES:
                marginal_effect_key: MarginalEffectKey = (
                    "contamination_ratio",
                    scope,
                    condition,
                    None,
                    ratio,
                )
                ci_low, ci_high = _interval(
                    marginal_effect_bootstrap[marginal_effect_key], tail_probability
                )
                average_marginal_effects.append(
                    AverageMarginalEffect(
                        effect="contamination_ratio",
                        scope=scope,
                        condition=condition,
                        reference_condition=None,
                        contamination_ratio=ratio,
                        estimate=observed_marginal_effects[marginal_effect_key],
                        ci_low=ci_low,
                        ci_high=ci_high,
                    )
                )
        for condition in _MODEL_CONDITIONS:
            for ratio in _RATIO_VALUES:
                marginal_effect_key = (
                    "condition_contrast",
                    scope,
                    condition,
                    ExperimentCondition.A2_SKILL_ONLY,
                    ratio,
                )
                ci_low, ci_high = _interval(
                    marginal_effect_bootstrap[marginal_effect_key], tail_probability
                )
                average_marginal_effects.append(
                    AverageMarginalEffect(
                        effect="condition_contrast",
                        scope=scope,
                        condition=condition,
                        reference_condition=ExperimentCondition.A2_SKILL_ONLY,
                        contamination_ratio=ratio,
                        estimate=observed_marginal_effects[marginal_effect_key],
                        ci_low=ci_low,
                        ci_high=ci_high,
                    )
                )
    method = STAGE_A_METHOD
    binding = binding_for(
        stage="A",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=0.95,
        method=method,
    )
    if scientific_freeze is not None:
        binding = AnalysisProfileBinding(
            classification="CONFIRMATORY",
            profile=scientific_freeze.analysis_profile,
            profile_hash=simultaneous_profile_hash(
                scientific_freeze.statistics_configuration_sha256,
                profile=ROBUST_SIMULTANEOUS_ANALYSIS_PROFILE,
            ),
            statistics_configuration_sha256=(scientific_freeze.statistics_configuration_sha256),
        )
    return StageAAnalysis(
        raw_counts=tuple(raw_counts),
        response_curves=tuple(response_curves),
        estimands=tuple(estimands),
        average_marginal_effects=tuple(average_marginal_effects),
        model=StageAModel(
            formula="CuP ~ contamination_ratio * condition + domain",
            family="Binomial",
            link="logit",
            parameter_names=_PARAMETER_NAMES,
            coefficients=tuple(float(value) for value in fitted_model.coefficients),
            covariance=tuple(
                tuple(float(value) for value in row) for row in fitted_model.covariance
            ),
            observation_count=fitted_model.observation_count,
            skill_bundle_cluster_count=fitted_model.skill_bundle_cluster_count,
            held_out_case_cluster_count=fitted_model.held_out_case_cluster_count,
            converged=fitted_model.converged,
        ),
        inference=StageAInference(
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            confidence_level=0.95,
            method=method,
            classification=binding.classification,
            profile=binding.profile,
            profile_hash=binding.profile_hash,
            statistics_configuration_sha256=binding.statistics_configuration_sha256,
        ),
    )


__all__ = [
    "AverageMarginalEffect",
    "RawCuPCount",
    "ResponseCurvePoint",
    "StageAAnalysis",
    "StageAAnalysisError",
    "StageAEstimand",
    "StageAInference",
    "StageAModel",
    "analyze_stage_a",
]
