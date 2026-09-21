"""Deterministic, preregistered representative-case selection.

The selector consumes explicit case summaries rather than deriving metadata from
``OutcomeRow`` or ``EpisodeResult``.  In particular, trace length and authority
graph depth must be supplied by the caller from the corresponding authoritative
artifacts; this module never treats an available counter as a proxy for either.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Literal

type SelectionStage = Literal["A", "B"]


class SelectionInputError(ValueError):
    """Raised when a candidate is not an explicit, well-bound summary."""


class SelectionStatus(StrEnum):
    SELECTED = "selected"
    HOLD_NO_CANDIDATE = "hold_no_candidate"


def _repeats(
    value: object, field: str, *, strings: bool = False
) -> tuple[bool, ...] | tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise SelectionInputError(f"{field} must be a non-empty tuple")
    values = value
    if strings:
        if any(type(item) is not str or not item for item in values):
            raise SelectionInputError(f"{field} must contain nonblank exact strings")
        return values  # type: ignore[return-value]
    if any(type(item) is not bool for item in values):
        raise SelectionInputError(f"{field} must contain exact booleans")
    return values  # type: ignore[return-value]


def _nonnegative_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise SelectionInputError(f"{field} must be an exact nonnegative integer")
    return value


def _ratio(value: object) -> float:
    if type(value) is not float or not isfinite(value) or not 0 <= value <= 1:
        raise SelectionInputError("contamination_ratio must be a finite float in [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    """One case/condition summary eligible for deterministic selection.

    Stage A binds CuP repeat outcomes for A2 and A4.  Stage B binds the
    condition outcome signatures for B1 and B2; differing signatures establish
    the preregistered ``B1/B2 differ`` predicate.  The two scalar metadata
    fields are intentionally explicit bindings supplied by the caller from
    trace/authority artifacts.  They are never inferred from episode counters.
    """

    stage: SelectionStage
    case_id: str
    contamination_ratio: float | None = None
    cascade_action_trace_length: int | None = None
    authority_graph_depth: int | None = None
    a2_cup_by_repeat: tuple[bool, ...] | None = None
    a4_cup_by_repeat: tuple[bool, ...] | None = None
    b1_outcome_by_repeat: tuple[str, ...] | None = None
    b2_outcome_by_repeat: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.stage not in ("A", "B"):
            raise SelectionInputError("stage must be 'A' or 'B'")
        if type(self.case_id) is not str or not self.case_id:
            raise SelectionInputError("case_id must be a nonblank exact string")
        if self.stage == "A":
            if self.contamination_ratio is None:
                raise SelectionInputError("Stage A requires contamination_ratio")
            _ratio(self.contamination_ratio)
            if self.cascade_action_trace_length is None:
                raise SelectionInputError("Stage A requires cascade_action_trace_length")
            _nonnegative_integer(self.cascade_action_trace_length, "cascade_action_trace_length")
            if self.authority_graph_depth is not None:
                raise SelectionInputError("Stage A cannot bind authority_graph_depth")
            a2 = _repeats(self.a2_cup_by_repeat, "a2_cup_by_repeat")
            a4 = _repeats(self.a4_cup_by_repeat, "a4_cup_by_repeat")
            if len(a2) != len(a4):
                raise SelectionInputError("A2 and A4 repeat counts must match")
            if self.b1_outcome_by_repeat is not None or self.b2_outcome_by_repeat is not None:
                raise SelectionInputError("Stage A cannot bind Stage B outcomes")
        else:
            if self.contamination_ratio is not None:
                raise SelectionInputError("Stage B cannot bind contamination_ratio")
            if self.cascade_action_trace_length is not None:
                raise SelectionInputError("Stage B cannot bind cascade_action_trace_length")
            if self.authority_graph_depth is None:
                raise SelectionInputError("Stage B requires authority_graph_depth")
            _nonnegative_integer(self.authority_graph_depth, "authority_graph_depth")
            b1 = _repeats(self.b1_outcome_by_repeat, "b1_outcome_by_repeat", strings=True)
            b2 = _repeats(self.b2_outcome_by_repeat, "b2_outcome_by_repeat", strings=True)
            if len(b1) != len(b2):
                raise SelectionInputError("B1 and B2 repeat counts must match")
            if self.a2_cup_by_repeat is not None or self.a4_cup_by_repeat is not None:
                raise SelectionInputError("Stage B cannot bind Stage A outcomes")

    @property
    def a2_fails_cup(self) -> bool:
        """Whether A2 fails CuP in a strict majority of repeats."""
        if self.stage != "A" or self.a2_cup_by_repeat is None:
            raise SelectionInputError("a2_fails_cup is only available for Stage A")
        return sum(self.a2_cup_by_repeat) < len(self.a2_cup_by_repeat) / 2

    @property
    def a4_passes_majority(self) -> bool:
        """Whether A4 passes CuP in a strict majority of repeats."""
        if self.stage != "A" or self.a4_cup_by_repeat is None:
            raise SelectionInputError("a4_passes_majority is only available for Stage A")
        return sum(self.a4_cup_by_repeat) > len(self.a4_cup_by_repeat) / 2

    @property
    def stage_b_conditions_differ(self) -> bool:
        """Whether the bound B1 and B2 outcome signatures differ."""
        if (
            self.stage != "B"
            or self.b1_outcome_by_repeat is None
            or self.b2_outcome_by_repeat is None
        ):
            raise SelectionInputError("stage_b_conditions_differ is only available for Stage B")
        return self.b1_outcome_by_repeat != self.b2_outcome_by_repeat


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Selection output, including an explicit no-candidate HOLD path."""

    stage: SelectionStage
    status: SelectionStatus
    selected: SelectionCandidate | None
    eligible_case_ids: tuple[str, ...]
    median_contamination_ratio: float | None = None
    median_cascade_action_trace_length: float | None = None
    median_authority_graph_depth: float | None = None

    @property
    def case_id(self) -> str | None:
        return None if self.selected is None else self.selected.case_id


def _median(values: tuple[float, ...]) -> float:
    if not values:
        raise SelectionInputError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _nearest(
    candidates: tuple[SelectionCandidate, ...], values: tuple[float, ...], target: float
) -> tuple[SelectionCandidate, ...]:
    distances = tuple(abs(value - target) for value in values)
    minimum = min(distances)
    return tuple(
        candidate for candidate, distance in zip(candidates, distances) if distance == minimum
    )


def select_stage_a(candidates: tuple[SelectionCandidate, ...]) -> SelectionResult:
    """Select Stage A's illustrative case under the preregistered algorithm."""
    _validate_input(candidates, "A")
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.a2_fails_cup and candidate.a4_passes_majority
    )
    if not eligible:
        return SelectionResult("A", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
    ratio_median = _median(
        tuple(
            candidate.contamination_ratio
            for candidate in eligible
            if candidate.contamination_ratio is not None
        )
    )
    ratio_candidates = _nearest(
        eligible,
        tuple(
            candidate.contamination_ratio
            for candidate in eligible
            if candidate.contamination_ratio is not None
        ),
        ratio_median,
    )
    trace_values = tuple(
        float(candidate.cascade_action_trace_length)
        for candidate in ratio_candidates
        if candidate.cascade_action_trace_length is not None
    )
    trace_median = _median(trace_values)
    trace_candidates = _nearest(ratio_candidates, trace_values, trace_median)
    selected = min(trace_candidates, key=lambda candidate: candidate.case_id)
    return SelectionResult(
        "A",
        SelectionStatus.SELECTED,
        selected,
        tuple(sorted(candidate.case_id for candidate in eligible)),
        ratio_median,
        trace_median,
    )


def select_stage_b(candidates: tuple[SelectionCandidate, ...]) -> SelectionResult:
    """Select Stage B's illustrative case under the preregistered algorithm."""
    _validate_input(candidates, "B")
    eligible = tuple(candidate for candidate in candidates if candidate.stage_b_conditions_differ)
    if not eligible:
        return SelectionResult("B", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
    depth_values = tuple(
        float(candidate.authority_graph_depth)
        for candidate in eligible
        if candidate.authority_graph_depth is not None
    )
    depth_median = _median(depth_values)
    depth_candidates = _nearest(eligible, depth_values, depth_median)
    selected = min(depth_candidates, key=lambda candidate: candidate.case_id)
    return SelectionResult(
        "B",
        SelectionStatus.SELECTED,
        selected,
        tuple(sorted(candidate.case_id for candidate in eligible)),
        median_authority_graph_depth=depth_median,
    )


def _validate_input(candidates: tuple[SelectionCandidate, ...], stage: SelectionStage) -> None:
    if type(candidates) is not tuple:
        raise SelectionInputError("candidates must be an exact tuple")
    if any(type(candidate) is not SelectionCandidate for candidate in candidates):
        raise SelectionInputError("candidates must contain exact SelectionCandidate records")
    if any(candidate.stage != stage for candidate in candidates):
        raise SelectionInputError(f"all candidates must be Stage {stage}")
    case_ids = [candidate.case_id for candidate in candidates]
    if len(case_ids) != len(set(case_ids)):
        raise SelectionInputError("candidate case_id values must be unique")


__all__ = [
    "SelectionCandidate",
    "SelectionInputError",
    "SelectionResult",
    "SelectionStatus",
    "select_stage_a",
    "select_stage_b",
]
