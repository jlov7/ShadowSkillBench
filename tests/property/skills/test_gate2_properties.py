from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from shadowskillbench.skills import gate2
from tests.unit.skills import test_gate2 as gate2_tests


@pytest.fixture(scope="session")
def gate2_report() -> gate2.Gate2Report:
    return gate2_tests._built_report()


@given(
    domain=st.sampled_from(gate2.DOMAINS),
    ratio=st.sampled_from(gate2.RATIOS),
)
@pytest.mark.property
def test_every_matrix_coordinate_has_its_required_classification(
    gate2_report: gate2.Gate2Report, domain: str, ratio: str
) -> None:
    cell = next(
        item
        for item in gate2_report.cells
        if item.domain == domain and item.contamination_ratio == ratio
    )
    assert cell.classification_status == ("CLEAN" if ratio == "0" else "CONTAMINATED")


@pytest.mark.property
def test_rendered_report_is_stable_json_that_reloads_as_the_same_report(
    gate2_report: gate2.Gate2Report, tmp_path: Path
) -> None:
    rendered = gate2.render_gate2_report(gate2_report)
    assert rendered == gate2.render_gate2_report(gate2_report)
    output = tmp_path / "gate2.json"
    output.write_text(rendered, encoding="utf-8")
    assert '"classification": "PRACTICE_NOT_EVIDENCE"' in rendered
    assert gate2.load_gate2_report(output) == gate2_report
