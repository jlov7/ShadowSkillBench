from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from shadowskillbench.skills import gate2
from tests.unit.skills import test_gate2 as gate2_tests


@pytest.fixture(scope="session")
def gate2_report() -> gate2.Gate2Report:
    return gate2_tests._built_report()


def _payload(report: gate2.Gate2Report) -> dict[str, object]:
    return json.loads(json.dumps(deepcopy(gate2.gate2_report_projection(report))))


def _write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "gate2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_tampered_classification_cannot_reload_as_pass(
    gate2_report: gate2.Gate2Report, tmp_path: Path
) -> None:
    payload = _payload(gate2_report)
    cells = payload["cells"]
    assert isinstance(cells, list)
    cells[1]["classification_status"] = "CLEAN"
    with pytest.raises(gate2.Gate2Error):
        gate2.load_gate2_report(_write_payload(tmp_path, payload))


def test_missing_matrix_cell_cannot_reload_as_pass(
    gate2_report: gate2.Gate2Report, tmp_path: Path
) -> None:
    payload = _payload(gate2_report)
    cells = payload["cells"]
    assert isinstance(cells, list)
    cells.pop()
    with pytest.raises(gate2.Gate2Error):
        gate2.load_gate2_report(_write_payload(tmp_path, payload))


def test_reordered_or_duplicate_matrix_cell_cannot_reload_as_pass(
    gate2_report: gate2.Gate2Report, tmp_path: Path
) -> None:
    payload = _payload(gate2_report)
    cells = payload["cells"]
    assert isinstance(cells, list)
    cells[1] = dict(cells[0])
    with pytest.raises(gate2.Gate2Error):
        gate2.load_gate2_report(_write_payload(tmp_path, payload))


def test_malformed_report_data_cannot_reload_as_pass(
    gate2_report: gate2.Gate2Report, tmp_path: Path
) -> None:
    payload = _payload(gate2_report)
    payload["status"] = "FAIL"
    with pytest.raises(gate2.Gate2Error):
        gate2.load_gate2_report(_write_payload(tmp_path, payload))


def test_confirmatory_classification_cannot_reload_as_practice_report(
    gate2_report: gate2.Gate2Report, tmp_path: Path
) -> None:
    payload = _payload(gate2_report)
    payload["classification"] = "CONFIRMATORY"
    with pytest.raises(gate2.Gate2Error):
        gate2.load_gate2_report(_write_payload(tmp_path, payload))
