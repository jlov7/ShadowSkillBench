from __future__ import annotations

from shadowskillbench.episodes.context import a5_content_matches
from shadowskillbench.policy.documents import (
    PolicyClause,
    render_matched_handbooks,
    render_policy_card,
)


def _clause(clause_id: str) -> PolicyClause:
    return PolicyClause(
        clause_id=clause_id,
        domain="access_provisioning",
        heading=f"Heading {clause_id}",
        body=f"Body {clause_id}.",
    )


def test_a5_matched_content_helper_accepts_only_equivalent_handbooks() -> None:
    salient, buried = render_matched_handbooks(
        _clause("target"), [_clause("one"), _clause("two")], seed=4
    )

    assert a5_content_matches(salient, buried)
    assert a5_content_matches(buried, salient)
    assert not a5_content_matches(salient, salient)
    assert not a5_content_matches(salient, render_policy_card(_clause("other")))
