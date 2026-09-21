from __future__ import annotations

from itertools import permutations

import pytest

from shadowskillbench.reporting.selection import (
    SelectionCandidate,
    SelectionInputError,
    SelectionStatus,
    select_stage_a,
    select_stage_b,
)


def stage_a(
    case_id: str,
    ratio: float,
    trace_length: int,
    a2: tuple[bool, ...],
    a4: tuple[bool, ...],
) -> SelectionCandidate:
    return SelectionCandidate(
        stage="A",
        case_id=case_id,
        contamination_ratio=ratio,
        cascade_action_trace_length=trace_length,
        a2_cup_by_repeat=a2,
        a4_cup_by_repeat=a4,
    )


def stage_b(
    case_id: str, depth: int, b1: tuple[str, ...], b2: tuple[str, ...]
) -> SelectionCandidate:
    return SelectionCandidate(
        stage="B",
        case_id=case_id,
        authority_graph_depth=depth,
        b1_outcome_by_repeat=b1,
        b2_outcome_by_repeat=b2,
    )


def test_stage_a_applies_eligibility_then_median_narrowing_and_lexical_tie_break() -> None:
    candidates = (
        stage_a("case-z", 0.0, 8, (False, False, False), (True, True, False)),
        stage_a("case-b", 0.5, 6, (False, False, False), (True, True, False)),
        stage_a("case-a", 0.5, 6, (False, False, False), (True, True, False)),
        stage_a("case-out", 0.75, 1, (True, False, True), (True, True, True)),
    )

    result = select_stage_a(candidates)

    assert result.status is SelectionStatus.SELECTED
    assert result.case_id == "case-a"
    assert result.eligible_case_ids == ("case-a", "case-b", "case-z")
    assert result.median_contamination_ratio == 0.5
    assert result.median_cascade_action_trace_length == 6.0


def test_stage_a_majority_is_strict_and_no_candidate_is_hold() -> None:
    candidate = stage_a("case-a", 0.5, 3, (False, True, False), (True, False, False))
    result = select_stage_a((candidate,))

    assert result.status is SelectionStatus.HOLD_NO_CANDIDATE
    assert result.selected is None
    assert result.case_id is None
    assert result.eligible_case_ids == ()


def test_stage_a_uses_strict_majorities_and_even_medians_use_arithmetic_midpoint() -> None:
    candidates = (
        stage_a("case-b", 0.25, 4, (False, True, False, False), (True, True, False, True)),
        stage_a("case-a", 0.75, 8, (False, False, True, False), (True, False, True, True)),
    )

    result = select_stage_a(candidates)

    assert result.status is SelectionStatus.SELECTED
    assert result.case_id == "case-a"
    assert result.median_contamination_ratio == 0.5
    assert result.median_cascade_action_trace_length == 6.0


def test_stage_b_selects_median_depth_and_lexical_tie_break() -> None:
    candidates = (
        stage_b("case-z", 3, ("block", "proceed"), ("proceed", "proceed")),
        stage_b("case-b", 2, ("block", "block"), ("proceed", "block")),
        stage_b("case-a", 2, ("block", "block"), ("proceed", "block")),
        stage_b("case-same", 2, ("block",), ("block",)),
    )

    result = select_stage_b(candidates)

    assert result.status is SelectionStatus.SELECTED
    assert result.case_id == "case-a"
    assert result.eligible_case_ids == ("case-a", "case-b", "case-z")
    assert result.median_authority_graph_depth == 2.0


def test_selection_is_invariant_to_input_permutation() -> None:
    candidates = (
        stage_a("case-c", 0.25, 4, (False, False, False), (True, True, False)),
        stage_a("case-a", 0.5, 7, (False, False, False), (True, True, False)),
        stage_a("case-b", 0.75, 9, (False, False, False), (True, True, False)),
    )

    results = {select_stage_a(tuple(order)).case_id for order in permutations(candidates)}

    assert results == {"case-a"}


def test_candidate_rejects_missing_trace_or_authority_metadata() -> None:
    with pytest.raises(SelectionInputError):
        SelectionCandidate(
            stage="A",
            case_id="case-a",
            contamination_ratio=0.5,
            a2_cup_by_repeat=(False, False, False),
            a4_cup_by_repeat=(True, True, True),
        )
    with pytest.raises(SelectionInputError):
        SelectionCandidate(
            stage="B",
            case_id="case-b",
            b1_outcome_by_repeat=("block",),
            b2_outcome_by_repeat=("proceed",),
        )


def test_selector_requires_tuple_and_rejects_duplicate_case_ids() -> None:
    candidate = stage_b("case-a", 1, ("block",), ("proceed",))

    with pytest.raises(SelectionInputError):
        select_stage_b([candidate])  # type: ignore[arg-type]
    with pytest.raises(SelectionInputError):
        select_stage_b((candidate, candidate))
