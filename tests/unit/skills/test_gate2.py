from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

import pytest

from shadowskillbench.skills import gate2

ROOT = Path(__file__).resolve().parents[3]
PROMPT = (ROOT / "prompts" / "skill_compiler.md").read_bytes()


@lru_cache(maxsize=1)
def _built_report() -> gate2.Gate2Report:
    return asyncio.run(gate2.build_gate2_development_report(prompt_bytes=PROMPT))


@pytest.fixture(scope="session")
def gate2_report() -> gate2.Gate2Report:
    """The fixed matrix is expensive enough to build once per focused test run."""
    return _built_report()


def test_fixed_matrix_has_two_domains_and_five_ratios(gate2_report: gate2.Gate2Report) -> None:
    assert gate2_report.schema_version == "1.1"
    assert gate2_report.classification == "PRACTICE_NOT_EVIDENCE"
    assert [(cell.domain, cell.contamination_ratio) for cell in gate2_report.cells] == [
        (domain, ratio) for domain in gate2.DOMAINS for ratio in gate2.RATIOS
    ]
    assert all((cell.count, cell.seed) == (gate2.COUNT, gate2.SEED) for cell in gate2_report.cells)


def test_compiled_cells_have_the_expected_classifications(
    gate2_report: gate2.Gate2Report,
) -> None:
    for cell in gate2_report.cells:
        expected = "CLEAN" if cell.contamination_ratio == "0" else "CONTAMINATED"
        assert cell.classification_status == expected
        assert bool(cell.finding_codes) is (expected == "CONTAMINATED")


def test_compiler_surface_is_clear_for_every_compiled_cell(
    gate2_report: gate2.Gate2Report,
) -> None:
    assert all(
        cell.leakage_status == "CLEAR" and cell.leakage_findings == ()
        for cell in gate2_report.cells
    )


def test_compilation_is_deterministic_for_a_fixed_matrix_coordinate() -> None:
    first = asyncio.run(gate2._compile_cell("access_provisioning", "0.5", PROMPT))
    second = asyncio.run(gate2._compile_cell("access_provisioning", "0.5", PROMPT))
    assert second == first


def test_builder_rejects_invalid_inputs() -> None:
    with pytest.raises(gate2.Gate2Error):
        asyncio.run(gate2.build_gate2_development_report(prompt_bytes=b""))
    with pytest.raises(gate2.Gate2Error):
        asyncio.run(gate2.build_gate2_development_report(prompt_bytes=PROMPT, seed=1))
