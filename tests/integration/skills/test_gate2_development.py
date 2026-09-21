from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from shadowskillbench.models import ModelAdapterError
from shadowskillbench.skills import gate2
from tests.unit.skills import test_gate2 as gate2_tests
from tests.unit.skills.test_gate2 import ROOT


@pytest.fixture(scope="session")
def gate2_report() -> gate2.Gate2Report:
    return gate2_tests._built_report()


RUNNER_PATH = ROOT / "scripts" / "run_gate2_development.py"
CANONICAL_REPORT_PATH = ROOT / "artifacts" / "development" / "gate2-development-report.json"
_SPEC = importlib.util.spec_from_file_location("run_gate2_development", RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_runner_default_output_path_is_canonical() -> None:
    assert runner._parser().parse_args([]).output == CANONICAL_REPORT_PATH


def test_runner_writes_a_reloadable_pass_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gate2_report: gate2.Gate2Report
) -> None:
    async def build_shared_report(*, prompt_bytes: bytes) -> gate2.Gate2Report:
        assert prompt_bytes
        return gate2_report

    output = tmp_path / "gate2.json"
    monkeypatch.setattr(runner, "build_gate2_development_report", build_shared_report)
    assert runner.main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["classification"] == "PRACTICE_NOT_EVIDENCE"
    loaded = gate2.load_gate2_report(output)
    assert loaded.status == "PASS"
    assert loaded == gate2_report


def test_malformed_compiler_output_fails_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate2, "_scripted_response", lambda *_args: b"{}")
    with pytest.raises(ModelAdapterError, match="MODEL_OUTPUT_INVALID"):
        asyncio.run(
            gate2.build_gate2_development_report(
                prompt_bytes=(ROOT / "prompts" / "skill_compiler.md").read_bytes()
            )
        )


def test_detected_compiler_surface_leakage_fails_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leak = gate2.LeakageFinding(
        "compiler_input", "hidden_class_manifest", "hidden_label", "hidden_class_manifest"
    )
    monkeypatch.setattr(gate2, "scan_compiler_surface", lambda *_args: (leak,))
    with pytest.raises(gate2.Gate2Error, match="forbidden compiler data"):
        asyncio.run(
            gate2.build_gate2_development_report(
                prompt_bytes=(ROOT / "prompts" / "skill_compiler.md").read_bytes()
            )
        )


def test_compatibility_builder_delegates_to_the_report_builder(
    monkeypatch: pytest.MonkeyPatch, gate2_report: gate2.Gate2Report
) -> None:
    async def build_shared_report(*, prompt_bytes: bytes, seed: int) -> gate2.Gate2Report:
        assert prompt_bytes
        assert seed == gate2.SEED
        return gate2_report

    monkeypatch.setattr(gate2, "build_gate2_development_report", build_shared_report)
    report = asyncio.run(
        gate2.build_gate2_development_evidence(
            prompt_bytes=(ROOT / "prompts" / "skill_compiler.md").read_bytes()
        )
    )
    assert report == gate2_report
