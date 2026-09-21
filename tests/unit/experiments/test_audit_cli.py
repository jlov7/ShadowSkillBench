from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree

import pytest
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.experiments.audit import FrozenArtifactCatalog
from shadowskillbench.experiments.planner import ConfirmatoryEpisodePlan
from tests.unit.experiments.test_audit import _prepared


def _plan_payload(plan: ConfirmatoryEpisodePlan) -> dict[str, object]:
    return {
        "profile": "SSB-PLAN1",
        "plan_hash": plan.plan_hash,
        "episodes": [
            {"manifest": episode.manifest_projection(), "manifest_hash": episode.manifest_hash}
            for episode in plan.episodes
        ],
    }


def _catalog_payload(catalog: FrozenArtifactCatalog) -> dict[str, object]:
    return {
        "profile": "SSB-AUDIT-CATALOG1",
        "case_manifest_hashes": sorted(catalog.case_manifest_hashes),
        "context_contract_hashes": sorted(catalog.context_contract_hashes),
        "source_manifest_hashes": sorted(catalog.source_manifest_hashes),
        "compiler_manifest_hashes": sorted(catalog.compiler_manifest_hashes),
        "compiled_skill_artifact_hashes": sorted(catalog.compiled_skill_artifact_hashes),
        "rendered_skill_hashes": sorted(catalog.rendered_skill_hashes),
        "policies": [
            {"rendered_hash": policy.rendered_hash, "rendered_text": policy.rendered_text}
            for policy in sorted(catalog.policies, key=lambda value: value.rendered_hash)
        ],
        "development_entities": sorted(catalog.development_entities or ()),
        "confirmatory_entities": sorted(catalog.confirmatory_entities or ()),
        "development_values": sorted(catalog.development_values or ()),
        "confirmatory_values": sorted(catalog.confirmatory_values or ()),
        "exclusion_rule_hash": catalog.exclusion_rule_hash,
    }


def _bundle(tmp_path: Path) -> Path:
    plan, _, run_directory, catalog = _prepared(tmp_path)
    root = tmp_path / "artifacts/experiments/confirmatory"
    copytree(run_directory, root)
    (root / "plan.json").write_bytes(canonical_json_bytes(_plan_payload(plan)))
    (root / "catalog.json").write_bytes(canonical_json_bytes(_catalog_payload(catalog)))
    return root


def _invoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *args: str):
    monkeypatch.chdir(tmp_path)
    return CliRunner().invoke(cli.app, ["experiments", "audit", *args])


def test_experiments_audit_runs_the_real_auditor_for_a_minimal_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)

    result = _invoke(monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    assert "PASS_EXPERIMENT_AUDIT: report_sha256=sha256:" in result.output
    assert not (root / "audit-report.json").exists()


def test_experiments_audit_holds_for_missing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _invoke(monkeypatch, tmp_path)

    assert result.exit_code == 1
    assert "HOLD_MISSING_EXPERIMENT_ARTIFACTS" in result.output


@pytest.mark.parametrize("field", ("plan_hash", "manifest_hash"))
def test_experiments_audit_rejects_tampered_plan_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    root = _bundle(tmp_path)
    plan = json.loads((root / "plan.json").read_bytes())
    if field == "plan_hash":
        plan[field] = "sha256:" + "0" * 64
    else:
        plan["episodes"][0][field] = "sha256:" + "0" * 64
    (root / "plan.json").write_bytes(canonical_json_bytes(plan))

    result = _invoke(monkeypatch, tmp_path)

    assert result.exit_code == 1
    assert "HOLD_INVALID_EXPERIMENT_ARTIFACTS" in result.output


def test_experiments_audit_reports_duplicate_persisted_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)
    original = next((root / "results").glob("*.json"))
    (root / "results" / "duplicate.json").write_bytes(original.read_bytes())

    result = _invoke(monkeypatch, tmp_path)

    assert result.exit_code == 1
    assert "HOLD_EXPERIMENT_AUDIT" in result.output
    assert "DUPLICATE_PLANNED_RESULT" in result.output


def test_experiments_audit_rejects_symlinked_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes((root / "plan.json").read_bytes())
    (root / "plan.json").unlink()
    (root / "plan.json").symlink_to(replacement)

    result = _invoke(monkeypatch, tmp_path)

    assert result.exit_code == 1
    assert "HOLD_UNSAFE_EXPERIMENT_ARTIFACTS" in result.output


@pytest.mark.parametrize(
    ("args", "code"),
    (
        (("--split", "development"), "HOLD_DEVELOPMENT_EXPERIMENT_AUDIT"),
        (("--stage-a",), "HOLD_MISSING_EXPERIMENT_ARTIFACTS"),
        (("--stage-b",), "HOLD_MISSING_EXPERIMENT_ARTIFACTS"),
        (("--stage-a", "--stage-b"), "HOLD_CONTRADICTORY_EXPERIMENT_AUDIT_SCOPE"),
    ),
)
def test_experiments_audit_scopes_fail_closed_without_required_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...], code: str
) -> None:
    result = _invoke(monkeypatch, tmp_path, *args)

    assert result.exit_code == 1
    assert code in result.output
