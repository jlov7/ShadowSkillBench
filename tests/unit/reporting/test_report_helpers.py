from shadowskillbench.reporting import select_representative_traces
from shadowskillbench.reporting.selection import SelectionCandidate

_HASH = "sha256:" + "a" * 64


def test_representative_helper_uses_preregistered_deterministic_selectors() -> None:
    stage_a = (
        SelectionCandidate(
            stage="A",
            case_id="case-a",
            contamination_ratio=0.5,
            cascade_action_trace_length=2,
            a2_cup_by_repeat=(False, False),
            a4_cup_by_repeat=(True, True),
        ),
    )
    stage_b = (
        SelectionCandidate(
            stage="B",
            case_id="case-b",
            authority_graph_depth=2,
            b1_outcome_by_repeat=("proceed",),
            b2_outcome_by_repeat=("block",),
        ),
    )

    traces = select_representative_traces(
        stage_a,
        stage_b,
        {"case-a": (_HASH, "Stage A public trace"), "case-b": (_HASH, "Stage B public trace")},
    )

    assert tuple(trace.selection.selected.case_id for trace in traces) == ("case-a", "case-b")
    assert tuple(trace.summary for trace in traces) == (
        "Stage A public trace",
        "Stage B public trace",
    )


def test_representative_helper_holds_only_when_selector_has_no_candidate() -> None:
    traces = select_representative_traces(
        (
            SelectionCandidate(
                stage="A",
                case_id="case-a",
                contamination_ratio=0.5,
                cascade_action_trace_length=2,
                a2_cup_by_repeat=(True,),
                a4_cup_by_repeat=(True,),
            ),
        ),
        (
            SelectionCandidate(
                stage="B",
                case_id="case-b",
                authority_graph_depth=2,
                b1_outcome_by_repeat=("proceed",),
                b2_outcome_by_repeat=("proceed",),
            ),
        ),
        {},
    )

    assert all(trace.selection.status.value == "hold_no_candidate" for trace in traces)
