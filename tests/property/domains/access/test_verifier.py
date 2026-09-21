from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.domains.access.verifier import verify_access_case
from tests.unit.domains.access.test_verifier import _decision, _finish_and_grant, _fixture


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_verifier_is_deterministic_for_valid_proceed_cases(seed: int) -> None:
    fixture = _fixture(seed=seed)
    decision = _decision(fixture)
    state = _finish_and_grant(fixture)
    assert verify_access_case(
        state, fixture.case, authority_decision=decision
    ) == verify_access_case(state, fixture.case, authority_decision=decision)
