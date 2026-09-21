from __future__ import annotations

import json
import socket
import subprocess
import urllib.request
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from shadowskillbench.analysis import (
    SealedAnalysis,
    SealedAnalysisHold,
    build_aggregate_metric_rows,
    sealed_analysis_projection,
)
from shadowskillbench.analysis.dataset import AnalysisDataset
from shadowskillbench.analysis.stage_a import StageAAnalysis, StageAAnalysisError, analyze_stage_a
from shadowskillbench.analysis.stage_b import StageBAnalysis, analyze_stage_b
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.experiments.audit import ArtifactIntegrityReport, RuntimeBinding
from shadowskillbench.models.openai_compatible import OpenAICompatibleClient
from shadowskillbench.protocol import validate_claims_ledger, validate_preregistration
from shadowskillbench.protocol.claim_estimands import claim_estimand_mapping
from shadowskillbench.protocol.power_precision import simulate_power_precision
from shadowskillbench.protocol.power_precision_design import execution_design_projection
from shadowskillbench.protocol.scientific_freeze import (
    MULTIPLICITY_MEMBERS,
    POWER_MEMBER_SPECS,
    SCIENTIFIC_FREEZE_INPUTS,
    ScientificFreezeBinding,
    derive_scientific_freeze_binding,
)
from shadowskillbench.release import reproduce
from shadowskillbench.release.reproduce import ReproductionHold, reproduce_sealed_artifacts
from shadowskillbench.reporting import (
    ReportEvidence,
    RepresentativeTrace,
    hash_analysis_dataset,
    render_report,
    render_workbench_json,
    report_status,
)
from shadowskillbench.reporting.selection import SelectionResult, SelectionStatus
from tests.custody import write_local_custody_receipt
from tests.golden.analysis.test_stage_a_synthetic import _synthetic_dataset as _stage_a_dataset
from tests.unit.analysis.test_stage_b import _dataset as _stage_b_dataset
from tests.unit.protocol.test_preregistration import completed_text

_HASH = "sha256:" + "a" * 64
ROOT = Path(__file__).parents[2]
_PROFILE3_STAGE_CACHE: tuple[str, StageAAnalysis, StageBAnalysis] | None = None


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _hash(data: bytes) -> str:
    return f"sha256:{sha256(data).hexdigest()}"


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(_canonical(value))


def _write_successor_scientific_profile(root: Path) -> None:
    """Build a hypothetical high-count successor for software composition only."""

    corpora_path = root / "config/corpora.yaml"
    corpora = yaml.safe_load(corpora_path.read_text(encoding="utf-8"))
    assert type(corpora) is dict
    stage_a_corpora = corpora["stage_a"]
    stage_b_corpora = corpora["stage_b"]
    assert type(stage_a_corpora) is dict and type(stage_b_corpora) is dict
    stage_a_corpora.update({"skill_bundles": 200, "held_out_cases_per_domain": 30})
    stage_b_corpora.update(
        {"r75_skills_per_domain": 20, "held_out_cases_per_authority_class_domain": 30}
    )
    corpora_path.write_text(yaml.safe_dump(corpora, sort_keys=False), encoding="utf-8")
    configuration_path = root / "config/statistics.yaml"
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    assert type(configuration) is dict
    reporting = configuration["reporting"]
    assert type(reporting) is dict
    configuration["profile"] = "SSB-CONFIRMATORY-STATISTICS3-BONFERRONI"
    reporting.update(
        {
            "familywise_confidence_level": 0.95,
            "multiple_comparison_family": list(MULTIPLICITY_MEMBERS),
            "multiplicity_method": "bonferroni_simultaneous_bootstrap_intervals",
            "simultaneous_interval_rule": (
                "simultaneous_95pct_ci_entirely_beyond_material_threshold"
            ),
        }
    )
    configuration["bootstrap"] = {
        "confidence_level": 0.95,
        "method": "deterministic_two_way_stratified_pigeonhole",
        "replicates": 40_000,
        "seed": 104_731,
    }
    configuration_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    planner_path = root / "src/shadowskillbench/experiments/planner.py"
    planner_path.write_bytes(
        planner_path.read_bytes()
        .replace(b"STAGE_A_EPISODE_COUNT = 7_440", b"STAGE_A_EPISODE_COUNT = 72_360")
        .replace(b"STAGE_B_EPISODE_COUNT = 1_800", b"STAGE_B_EPISODE_COUNT = 72_000")
        .replace(
            b"len(skills) != 30 or len(cases) != 40",
            b"len(skills) != 200 or len(cases) != 60",
        )
        .replace(b"len(domain_cases) != 20", b"len(domain_cases) != 30")
        .replace(
            b"Counter({ratio: 3 for ratio in RATIOS})", b"Counter({ratio: 20 for ratio in RATIOS})"
        )
        .replace(b"len(cases) != 50", b"len(cases) != 300")
        .replace(b"len(domain_cases) != 25", b"len(domain_cases) != 150")
        .replace(
            b"{authority_class: 5 for authority_class in AUTHORITY_CLASSES}",
            b"{authority_class: 30 for authority_class in AUTHORITY_CLASSES}",
        )
        .replace(b"len(skills[domain]) != 3", b"len(skills[domain]) != 20")
    )
    plan_path = root / "protocol/power_precision_plan.v3.json"
    plan = json.loads(plan_path.read_bytes())
    design = execution_design_projection(
        corpora,
        MULTIPLICITY_MEMBERS,
        planner_source=planner_path.read_bytes(),
    )
    plan.update(
        {
            "corpora_configuration_sha256": _hash(corpora_path.read_bytes()),
            "execution_design": design,
            "planner_sha256": _hash(planner_path.read_bytes()),
        }
    )
    plan_path.write_bytes(_canonical(plan))
    evidence: dict[str, object] = {
        "analysis_plan_sha256": _hash((root / "protocol/analysis_plan.v3.json").read_bytes()),
        "corpora_configuration_sha256": _hash(corpora_path.read_bytes()),
        "execution_design_source_sha256": _hash(
            (root / "src/shadowskillbench/protocol/power_precision_design.py").read_bytes()
        ),
        "power_precision_plan_sha256": _hash(
            (root / "protocol/power_precision_plan.v3.json").read_bytes()
        ),
        "planner_sha256": _hash(planner_path.read_bytes()),
        "profile": "SSB-SCIENTIFIC-POWER-PRECISION-EVIDENCE-3",
        "schema_version": "1.0",
        "statistics_configuration_sha256": _hash(configuration_path.read_bytes()),
        "status": "PASS",
        "simulator_profile": "SSB-POWER-PRECISION-SIMULATOR-1",
        "simulator_sha256": _hash(
            (root / "src/shadowskillbench/protocol/power_precision.py").read_bytes()
        ),
    }
    designs = design["member_scenarios"]
    assert type(designs) is dict
    scenarios: list[dict[str, object]] = []
    for member in MULTIPLICITY_MEMBERS:
        spec = POWER_MEMBER_SPECS[member]
        scenario_design = designs[member]
        assert type(scenario_design) is dict
        scenario = {
            "baseline_rate": 0.5,
            "cluster_icc_held_out_case": 0.01,
            "cluster_icc_skill_bundle": 0.01,
            "contrast_structure": spec["contrast_structure"],
            "effect_size": 0.2 if spec["expected_direction"] == "positive" else -0.2,
            "expected_direction": spec["expected_direction"],
            **scenario_design,
            "target_threshold": 0.1,
            "variance_multiplier": spec["variance_multiplier"],
        }
        scenarios.append(
            {
                "multiplicity_member": member,
                **scenario,
                **simulate_power_precision(
                    scenario,
                    replicates=20_000,
                    seed=104_732,
                    family_size=len(MULTIPLICITY_MEMBERS),
                ),
            }
        )
    evidence["scenarios"] = scenarios
    (root / "protocol/power_precision_evidence.v3.json").write_bytes(_canonical(evidence))


def _write_practice_release(root: Path) -> tuple[Path, Path]:
    (root / "protocol").mkdir(parents=True)
    (root / "artifacts" / "release").mkdir(parents=True)
    (root / "artifacts" / "reports").mkdir(parents=True)
    preregistration_path = root / "protocol" / "preregistration.md"
    preregistration_bytes = b"practice preregistration custody input\n"
    preregistration_path.write_bytes(preregistration_bytes)
    preregistration_hash = _hash(preregistration_bytes)
    protocol = {
        "anchor_status": "PENDING_HUMAN_ANCHOR",
        "inputs": [
            {
                "path": "protocol/preregistration.md",
                "role": "preregistration_core",
                "sha256": preregistration_hash,
            }
        ],
        "preregistration_core": {
            "path": "protocol/preregistration.md",
            "sha256": preregistration_hash,
        },
        "referents": [],
        "schema_version": "1.0",
    }
    protocol_path = root / "protocol" / "freeze_manifest.json"
    protocol_bytes = _canonical(protocol)
    protocol_path.write_bytes(protocol_bytes)
    protocol_hash = _hash(protocol_bytes)
    source = {
        "profile": "SSB-PRACTICE-REBUILD-1",
        "schema_version": "1.0",
        "classification": "PRACTICE_NOT_EVIDENCE",
        "protocol": {
            "protocol_tag": "practice-v1",
            "freeze_manifest_sha256": protocol_hash,
            "prompt_hashes": [_HASH],
            "code_commit": "abcdef0",
            "reproduce_command": (
                "uv run shadowskillbench reproduce --protocol protocol/freeze_manifest.json"
            ),
        },
        "limitations": ["PRACTICE_NOT_EVIDENCE synthetic release fixture."],
    }
    source_path = root / "artifacts" / "release" / "practice_rebuild.json"
    source_bytes = _canonical(source)
    source_path.write_bytes(source_bytes)
    evidence = reproduce._practice_evidence(source, protocol_hash)
    report = render_report(evidence).encode()
    workbench = render_workbench_json(evidence)
    sealed = {
        "profile": "SSB-SEALED-ARTIFACTS-1",
        "schema_version": "1.0",
        "classification": "PRACTICE_NOT_EVIDENCE",
        "protocol_manifest_sha256": protocol_hash,
        "sources": [
            {
                "path": "artifacts/release/practice_rebuild.json",
                "sha256": _hash(source_bytes),
                "role": "practice_rebuild",
            }
        ],
        "outputs": [
            {
                "kind": "report_html",
                "path": "artifacts/reports/report.html",
                "sha256": _hash(report),
            },
            {
                "kind": "workbench_json",
                "path": "artifacts/reports/workbench.json",
                "sha256": _hash(workbench),
            },
        ],
    }
    manifest_path = root / "artifacts" / "release" / "sealed_artifacts_manifest.json"
    manifest_path.write_bytes(_canonical(sealed))
    return protocol_path, manifest_path


def _write_confirmatory_release(
    root: Path, *, preregistration_bytes: bytes = b"confirmatory fixture preregistration\n"
) -> tuple[Path, Path]:
    """Write only sealed source evidence; analysis itself remains a test seam."""
    (root / "protocol").mkdir(parents=True)
    (root / "artifacts" / "release").mkdir(parents=True)
    (root / "artifacts" / "reports").mkdir(parents=True)
    preregistration_path = root / "protocol" / "preregistration.md"
    preregistration_path.write_bytes(preregistration_bytes)
    confirmatory_bytes = (ROOT / "src/shadowskillbench/corpus/confirmatory.py").read_bytes()
    confirmatory_path = root / "src/shadowskillbench/corpus/confirmatory.py"
    confirmatory_path.parent.mkdir(parents=True)
    confirmatory_path.write_bytes(confirmatory_bytes)
    compiler_bytes = b"fixture compiler prompt\n"
    compiler_path = root / "prompts/skill_compiler.md"
    compiler_path.parent.mkdir(parents=True)
    compiler_path.write_bytes(compiler_bytes)
    executor_bytes = b"fixture executor prompt\n"
    executor_path = root / "prompts/executor_system.md"
    executor_path.write_bytes(executor_bytes)
    explicit_paths = {
        "protocol/preregistration.md",
        "src/shadowskillbench/corpus/confirmatory.py",
        "prompts/skill_compiler.md",
        "prompts/executor_system.md",
    }
    for item in SCIENTIFIC_FREEZE_INPUTS:
        if item.path in explicit_paths:
            continue
        destination = root / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / item.path).read_bytes())
    _write_successor_scientific_profile(root)
    confirmatory_bytes = confirmatory_path.read_bytes()
    scientific_inputs: list[dict[str, str]] = []
    for item in SCIENTIFIC_FREEZE_INPUTS:
        if item.path in explicit_paths:
            continue
        destination = root / item.path
        scientific_inputs.append(
            {
                "path": item.path,
                "role": item.role,
                "sha256": _hash(destination.read_bytes()),
            }
        )
    inputs = [
        {
            "path": "protocol/preregistration.md",
            "role": "preregistration_core",
            "sha256": _hash(preregistration_bytes),
        },
        {
            "path": "src/shadowskillbench/corpus/confirmatory.py",
            "role": "confirmatory_corpus",
            "sha256": _hash(confirmatory_bytes),
        },
        {
            "path": "prompts/skill_compiler.md",
            "role": "compiler_prompt",
            "sha256": _hash(compiler_bytes),
        },
        {
            "path": "prompts/executor_system.md",
            "role": "executor_system_prompt",
            "sha256": _hash(executor_bytes),
        },
        *scientific_inputs,
    ]
    protocol = {
        "anchor_status": "PENDING_HUMAN_ANCHOR",
        "inputs": inputs,
        "preregistration_core": {
            "path": "protocol/preregistration.md",
            "sha256": _hash(preregistration_bytes),
        },
        "schema_version": "1.0",
    }
    protocol_path = root / "protocol" / "freeze_manifest.json"
    protocol_bytes = _canonical(protocol)
    protocol_path.write_bytes(protocol_bytes)
    protocol_hash = _hash(protocol_bytes)
    (root / "protocol" / "freeze_manifest.sha256").write_bytes(
        f"{protocol_hash.removeprefix('sha256:')}  freeze_manifest.json\n".encode()
    )
    receipt_payload = {"profile": "SSB-SEALED-ANALYSIS-1", "fixture": "complete"}
    receipt_path = root / "artifacts" / "release" / "analysis_receipt.json"
    receipt_bytes = _canonical(receipt_payload)
    receipt_path.write_bytes(receipt_bytes)
    source = {
        "profile": "SSB-CONFIRMATORY-REPRODUCTION1",
        "schema_version": "1.0",
        "classification": "CONFIRMATORY",
        "artifacts_root": "artifacts/experiments/confirmatory",
        "analysis_receipt": {
            "path": "artifacts/release/analysis_receipt.json",
            "sha256": _hash(receipt_bytes),
        },
        "limitations": ["Synthetic sealed confirmatory reproduction fixture."],
    }
    source_path = root / "artifacts" / "release" / "confirmatory_rebuild.json"
    source_bytes = _canonical(source)
    source_path.write_bytes(source_bytes)
    anchor_bytes = write_local_custody_receipt(root, protocol_bytes)
    practice_evidence = reproduce._practice_evidence(
        {
            "profile": "SSB-PRACTICE-REBUILD-1",
            "schema_version": "1.0",
            "classification": "PRACTICE_NOT_EVIDENCE",
            "protocol": {
                "protocol_tag": "fixture-v1",
                "freeze_manifest_sha256": protocol_hash,
                "prompt_hashes": [_HASH],
                "code_commit": "abcdef0",
                "reproduce_command": (
                    "uv run shadowskillbench reproduce --protocol protocol/freeze_manifest.json"
                ),
            },
            "limitations": ["Synthetic renderer fixture; not confirmatory evidence."],
        },
        protocol_hash,
    )
    sealed = {
        "profile": "SSB-SEALED-ARTIFACTS-2",
        "schema_version": "1.0",
        "classification": "CONFIRMATORY",
        "protocol_manifest_sha256": protocol_hash,
        "anchor_receipt": {"path": "protocol/anchor_receipt.json", "sha256": _hash(anchor_bytes)},
        "sources": [
            {
                "path": "artifacts/release/confirmatory_rebuild.json",
                "sha256": _hash(source_bytes),
                "role": "confirmatory_rebuild",
            }
        ],
        "outputs": [
            {
                "kind": "report_html",
                "path": "artifacts/reports/report.html",
                "sha256": _hash(render_report(practice_evidence).encode()),
            },
            {
                "kind": "workbench_json",
                "path": "artifacts/reports/workbench.json",
                "sha256": _hash(render_workbench_json(practice_evidence)),
            },
        ],
    }
    manifest_path = root / "artifacts" / "release" / "sealed_artifacts_manifest.json"
    manifest_path.write_bytes(_canonical(sealed))
    return protocol_path, manifest_path


def test_reproduction_protocol_evidence_accepts_verified_local_confirmatory_custody(
    tmp_path: Path,
) -> None:
    protocol_path, _ = _write_confirmatory_release(tmp_path)
    protocol = json.loads(protocol_path.read_bytes())
    compiler_input = next(item for item in protocol["inputs"] if item["role"] == "compiler_prompt")
    compiler_input["path"] = "prompts/confirmatory_skill_compiler.md"
    compiler_input["role"] = "confirmatory_compiler_prompt"
    (tmp_path / "prompts/confirmatory_skill_compiler.md").write_bytes(
        (tmp_path / "prompts/skill_compiler.md").read_bytes()
    )
    protocol_bytes = _canonical(protocol)
    protocol_path.write_bytes(protocol_bytes)
    protocol_hash = _hash(protocol_bytes)
    (tmp_path / "protocol" / "freeze_manifest.sha256").write_bytes(
        f"{protocol_hash.removeprefix('sha256:')}  freeze_manifest.json\n".encode()
    )
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "ShadowSkillBench test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "freeze")
    freeze_commit = _git(tmp_path, "rev-parse", "HEAD")
    protocol_tag = "shadowskillbench-protocol-v1.0.0-local"
    _git(tmp_path, "tag", "-f", protocol_tag, freeze_commit)
    receipt = {
        "schema_version": "2.0",
        "custody_mode": "LOCAL_HASH_CUSTODY",
        "freeze_commit": freeze_commit,
        "protocol_tag": protocol_tag,
        "freeze_manifest_hash": protocol_hash,
        "custody_locator": "urn:git:" + freeze_commit,
        "created_at": "2026-08-24T12:00:00Z",
        "verification_result": "VERIFIED_LOCAL",
    }
    receipt_path = tmp_path / "protocol" / "anchor_receipt.json"
    receipt_bytes = _canonical(receipt)
    receipt_path.write_bytes(receipt_bytes)
    _git(tmp_path, "add", "protocol/anchor_receipt.json")
    _git(tmp_path, "commit", "-m", "record local custody receipt")

    evidence = reproduce._confirmatory_protocol_evidence(
        tmp_path,
        protocol_path,
        {"path": "protocol/anchor_receipt.json", "sha256": _hash(receipt_bytes)},
    )

    assert evidence.external_anchor_locator is None
    assert evidence.custody_locator == "urn:git:" + freeze_commit
    assert evidence.custody_mode == "LOCAL_HASH_CUSTODY"
    assert evidence.anchor_verified is True


def _complete_sealed_analysis(root: Path, protocol_hash: str, anchor_hash: str) -> SealedAnalysis:
    stage_a_data = _stage_a_dataset()
    stage_b_data = _stage_b_dataset()
    rows = tuple(
        replace(
            row,
            planned_episode_hash=sha256_ref(["planned", row.episode_id]),
            manifest_hash=sha256_ref(["manifest", row.episode_id]),
            result_hash=sha256_ref(["result", row.episode_id]),
            score_hash=sha256_ref(["score", row.episode_id]),
        )
        for row in stage_a_data.rows + stage_b_data.rows
    )
    dataset = type(stage_a_data)(
        rows=rows,
        exclusions=(),
        aggregates=build_aggregate_metric_rows(rows),
    )
    runtime = RuntimeBinding(
        plan_hash="sha256:" + "1" * 64,
        package_hash="sha256:" + "2" * 64,
        run_descriptor_hash="sha256:" + "3" * 64,
        freeze_manifest_hash=protocol_hash,
        anchor_receipt_hash=anchor_hash,
        operator_endpoint_hash="sha256:" + "4" * 64,
        operator_api_key_environment="SSB_TEST_KEY",
        provider="fixture",
        model="fixture",
        model_version="fixture",
    )
    audit = ArtifactIntegrityReport(
        status="PASS",
        plan_hash=runtime.plan_hash,
        runtime_binding_hash=sha256_ref(runtime.projection()),
        audited_episode_ids=tuple(sorted(row.episode_id for row in rows)),
        findings=(),
    )
    scientific_freeze = derive_scientific_freeze_binding(
        manifest_bytes=(root / "protocol/freeze_manifest.json").read_bytes(),
        repository_root=root,
    )
    stage_a, stage_b = _profile3_stages(dataset, scientific_freeze)
    return SealedAnalysis(
        dataset=dataset,
        stage_a=stage_a,
        stage_b=stage_b,
        audit=audit,
        runtime_binding=runtime,
        execution_design_projection=scientific_freeze.member_execution_designs,
    )


def _profile3_stages(
    dataset: AnalysisDataset, scientific_freeze: ScientificFreezeBinding
) -> tuple[StageAAnalysis, StageBAnalysis]:
    """Run the real frozen analysis once and reuse its immutable test outputs."""

    global _PROFILE3_STAGE_CACHE
    key = sha256_ref(
        {
            "dataset": hash_analysis_dataset(dataset),
            "statistics": scientific_freeze.statistics_configuration_sha256,
            "profile": scientific_freeze.analysis_profile,
            "bootstrap_replicates": scientific_freeze.bootstrap_replicates,
            "bootstrap_seed": scientific_freeze.bootstrap_seed,
        }
    )
    if _PROFILE3_STAGE_CACHE is None:
        stage_a = analyze_stage_a(
            dataset,
            bootstrap_replicates=scientific_freeze.bootstrap_replicates,
            bootstrap_seed=scientific_freeze.bootstrap_seed,
            scientific_freeze=scientific_freeze,
        )
        stage_b = analyze_stage_b(
            dataset,
            bootstrap_replicates=scientific_freeze.bootstrap_replicates,
            bootstrap_seed=scientific_freeze.bootstrap_seed,
            scientific_freeze=scientific_freeze,
        )
        assert stage_a.inference.bootstrap_replicates == 40_000
        assert stage_b.inference.bootstrap_replicates == 40_000
        assert stage_a.inference.profile == scientific_freeze.analysis_profile
        assert stage_b.inference.profile == scientific_freeze.analysis_profile
        _PROFILE3_STAGE_CACHE = (key, stage_a, stage_b)
    cached_key, stage_a, stage_b = _PROFILE3_STAGE_CACHE
    if cached_key != key:
        raise AssertionError("Profile 3 stage cache key mismatch")
    return stage_a, stage_b


def _write_composed_confirmatory_release(
    root: Path, *, include_null_finding: bool = False
) -> tuple[Path, SealedAnalysis]:
    protocol_path, manifest_path = _write_confirmatory_release(
        root, preregistration_bytes=completed_text().encode()
    )
    protocol_bytes = protocol_path.read_bytes()
    protocol_hash = _hash(protocol_bytes)
    anchor_path = root / "protocol" / "anchor_receipt.json"
    anchor = _manifest(anchor_path)
    sealed = _complete_sealed_analysis(root, protocol_hash, sha256_ref(anchor))
    finding_path = root / "artifacts" / "results" / "claim.json"
    finding_path.parent.mkdir(parents=True)
    finding_path.write_bytes(b"synthetic sealed finding\n")
    claim_mapping = claim_estimand_mapping("EXP-001")
    assert claim_mapping is not None
    claims: list[dict[str, object]] = [
        {
            "id": "EXP-001",
            "wording": "Synthetic fixture finding.",
            "status": "finding",
            "evidence_command": "shadowskillbench report claim EXP-001",
            "artifact": "artifacts/results/claim.json",
            "scope": "Synthetic fixture scope.",
            "artifact_sha256": _hash(finding_path.read_bytes()),
            "finding_disposition": "supports",
            "claim_mapping_profile": claim_mapping.profile,
            "claim_mapping_id": claim_mapping.mapping_id,
            "claim_mapping_sha256": claim_mapping.sha256,
        }
    ]
    if include_null_finding:
        null_mapping = claim_estimand_mapping("EXP-002")
        assert null_mapping is not None
        claims.append(
            {
                "id": "EXP-002",
                "wording": "Synthetic null fixture finding.",
                "status": "finding",
                "evidence_command": "shadowskillbench report claim EXP-002",
                "artifact": "artifacts/results/claim.json",
                "scope": "Synthetic fixture scope.",
                "artifact_sha256": _hash(finding_path.read_bytes()),
                "finding_disposition": "does_not_support",
                "claim_mapping_profile": null_mapping.profile,
                "claim_mapping_id": null_mapping.mapping_id,
                "claim_mapping_sha256": null_mapping.sha256,
            }
        )
    (root / "protocol" / "CLAIMS_LEDGER.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "claims": claims,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    receipt_path = root / "artifacts" / "release" / "analysis_receipt.json"
    receipt_bytes = _canonical(sealed_analysis_projection(sealed))
    receipt_path.write_bytes(receipt_bytes)
    limitations = [
        "Synthetic domains limit external validity beyond the benchmark setting.",
        "The selected model provider and version bound these observations.",
        "Technical exclusions and audit coverage bound the included evidence.",
    ]
    source = {
        "profile": "SSB-CONFIRMATORY-REPRODUCTION1",
        "schema_version": "1.0",
        "classification": "CONFIRMATORY",
        "artifacts_root": "artifacts/experiments/confirmatory",
        "analysis_receipt": {
            "path": "artifacts/release/analysis_receipt.json",
            "sha256": _hash(receipt_bytes),
        },
        "limitations": limitations,
    }
    source_path = root / "artifacts" / "release" / "confirmatory_rebuild.json"
    source_bytes = _canonical(source)
    source_path.write_bytes(source_bytes)
    protocol = reproduce._confirmatory_protocol_evidence(
        root,
        protocol_path,
        {"path": "protocol/anchor_receipt.json", "sha256": _hash(anchor_path.read_bytes())},
    )
    protocol = replace(
        protocol,
        experiment_plan_sha256=sealed.audit.plan_hash,
        experiment_runtime_binding_sha256=sha256_ref(sealed.runtime_binding.projection()),
        experiment_audit_report_sha256=sealed.audit.report_hash,
        experiment_audit_status="PASS",
    )
    evidence = ReportEvidence(
        dataset=sealed.dataset,
        stage_a=sealed.stage_a,
        stage_b=sealed.stage_b,
        preregistration=validate_preregistration(root / "protocol" / "preregistration.md"),
        claims=validate_claims_ledger(root / "protocol" / "CLAIMS_LEDGER.yaml", root),
        protocol=protocol,
        analysis_dataset_sha256=sealed.dataset_sha256,
        representative_traces=(
            RepresentativeTrace(
                "A", SelectionResult("A", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
            ),
            RepresentativeTrace(
                "B", SelectionResult("B", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
            ),
        ),
        limitations=tuple(limitations),
        runtime_binding=sealed.runtime_binding,
    )
    assert report_status(evidence).value == "confirmatory"
    manifest = _manifest(manifest_path)
    manifest["sources"] = [
        {
            "path": "artifacts/release/confirmatory_rebuild.json",
            "sha256": _hash(source_bytes),
            "role": "confirmatory_rebuild",
        }
    ]
    manifest["outputs"] = [
        {
            "kind": "report_html",
            "path": "artifacts/reports/report.html",
            "sha256": _hash(render_report(evidence).encode()),
        },
        {
            "kind": "workbench_json",
            "path": "artifacts/reports/workbench.json",
            "sha256": _hash(render_workbench_json(evidence)),
        },
    ]
    _write_manifest(manifest_path, manifest)
    return protocol_path, sealed


def test_practice_release_rebuilds_report_and_workbench_without_authored_outputs(
    tmp_path: Path,
) -> None:
    protocol, _ = _write_practice_release(tmp_path)

    result = reproduce_sealed_artifacts(tmp_path, protocol)

    assert result.classification == "PRACTICE_NOT_EVIDENCE"
    assert result.report_sha256 == _hash((tmp_path / "artifacts/reports/report.html").read_bytes())
    assert result.workbench_sha256 == _hash(
        (tmp_path / "artifacts/reports/workbench.json").read_bytes()
    )
    assert "HOLD" in (tmp_path / "artifacts/reports/report.html").read_text(encoding="utf-8")


def test_reproduction_refuses_tampered_or_authored_outputs(tmp_path: Path) -> None:
    protocol, _ = _write_practice_release(tmp_path)
    source = tmp_path / "artifacts/release/practice_rebuild.json"
    source.write_bytes(source.read_bytes().replace(b"practice-v1", b"practice-x1"))

    with pytest.raises(ReproductionHold, match="HOLD_HASH_MISMATCH"):
        reproduce_sealed_artifacts(tmp_path, protocol)


def test_reproduction_rejects_duplicate_manifest_key(tmp_path: Path) -> None:
    protocol, manifest = _write_practice_release(tmp_path)
    manifest.write_bytes(
        manifest.read_bytes().replace(b'"profile":', b'"profile":"x","profile":', 1)
    )

    with pytest.raises(ReproductionHold, match="HOLD_DUPLICATE_JSON_KEY"):
        reproduce_sealed_artifacts(tmp_path, protocol)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("sources", "path", "../practice_rebuild.json"),
        ("sources", "path", "/tmp/practice_rebuild.json"),
        ("outputs", "path", "../report.html"),
        ("outputs", "path", "/tmp/report.html"),
    ],
)
def test_reproduction_rejects_unsafe_source_and_output_paths(
    tmp_path: Path, section: str, field: str, value: str
) -> None:
    protocol, manifest_path = _write_practice_release(tmp_path)
    manifest = _manifest(manifest_path)
    records = manifest[section]
    assert type(records) is list
    records[0][field] = value
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ReproductionHold, match="HOLD_UNSAFE_PATH"):
        reproduce_sealed_artifacts(tmp_path, protocol)


def test_reproduction_rejects_source_and_output_parent_symlinks(tmp_path: Path) -> None:
    protocol, _ = _write_practice_release(tmp_path)
    source = tmp_path / "artifacts/release/practice_rebuild.json"
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(replacement)
    with pytest.raises(ReproductionHold, match="HOLD_SYMLINK_PATH"):
        reproduce_sealed_artifacts(tmp_path, protocol)

    protocol, _ = _write_practice_release(tmp_path / "output")
    reports = tmp_path / "output/artifacts/reports"
    reports.rmdir()
    reports.symlink_to(tmp_path)
    with pytest.raises(ReproductionHold, match="HOLD_SYMLINK_PATH"):
        reproduce_sealed_artifacts(tmp_path / "output", protocol)


def test_reproduction_rejects_missing_source_duplicate_outputs_and_hash_tamper(
    tmp_path: Path,
) -> None:
    protocol, manifest_path = _write_practice_release(tmp_path)
    (tmp_path / "artifacts/release/practice_rebuild.json").unlink()
    with pytest.raises(ReproductionHold, match="HOLD_MISSING_SEALED_INPUT"):
        reproduce_sealed_artifacts(tmp_path, protocol)

    protocol, manifest_path = _write_practice_release(tmp_path / "duplicate")
    manifest = _manifest(manifest_path)
    outputs = manifest["outputs"]
    assert type(outputs) is list
    outputs[1]["kind"] = "report_html"
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ReproductionHold, match="sealed output kind"):
        reproduce_sealed_artifacts(tmp_path / "duplicate", protocol)

    protocol, manifest_path = _write_practice_release(tmp_path / "duplicate-path")
    manifest = _manifest(manifest_path)
    outputs = manifest["outputs"]
    assert type(outputs) is list
    outputs[1]["path"] = outputs[0]["path"]
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ReproductionHold, match="HOLD_DUPLICATE_REFERENCE"):
        reproduce_sealed_artifacts(tmp_path / "duplicate-path", protocol)

    protocol, manifest_path = _write_practice_release(tmp_path / "hash")
    manifest = _manifest(manifest_path)
    outputs = manifest["outputs"]
    assert type(outputs) is list
    outputs[0]["sha256"] = "sha256:" + "b" * 64
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ReproductionHold, match="HOLD_REBUILD_HASH_MISMATCH"):
        reproduce_sealed_artifacts(tmp_path / "hash", protocol)


def test_confirmatory_reproduction_requires_human_anchor_inputs(
    tmp_path: Path,
) -> None:
    protocol, manifest_path = _write_practice_release(tmp_path)
    manifest = _manifest(manifest_path)
    manifest["classification"] = "CONFIRMATORY"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ReproductionHold, match="HOLD_MISSING_ANCHOR_RECEIPT"):
        reproduce_sealed_artifacts(tmp_path, protocol)


def test_practice_reproduction_is_deterministic_without_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_protocol, _ = _write_practice_release(left)
    right_protocol, _ = _write_practice_release(right)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("subprocess"))
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: pytest.fail("network"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network"))

    async def forbid_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("provider")

    monkeypatch.setattr(OpenAICompatibleClient, "structured", forbid_provider)
    left_result = reproduce_sealed_artifacts(left, left_protocol)
    right_result = reproduce_sealed_artifacts(right, right_protocol)
    assert left_result.report_sha256 == right_result.report_sha256
    assert left_result.workbench_sha256 == right_result.workbench_sha256
    assert (left / "artifacts/reports/report.html").read_bytes() == (
        right / "artifacts/reports/report.html"
    ).read_bytes()


def test_confirmatory_reproduction_holds_when_rows_do_not_match_frozen_design(
    tmp_path: Path,
) -> None:
    with pytest.raises(StageAAnalysisError, match="HOLD_FROZEN_EXECUTION_DESIGN_MISMATCH"):
        _write_composed_confirmatory_release(tmp_path)


def test_confirmatory_reproduction_holds_for_null_fixture_before_claims(
    tmp_path: Path,
) -> None:
    with pytest.raises(StageAAnalysisError, match="HOLD_FROZEN_EXECUTION_DESIGN_MISMATCH"):
        _write_composed_confirmatory_release(tmp_path, include_null_finding=True)


def test_confirmatory_reproduction_holds_before_post_analysis_claims(
    tmp_path: Path,
) -> None:
    with pytest.raises(StageAAnalysisError, match="HOLD_FROZEN_EXECUTION_DESIGN_MISMATCH"):
        _write_composed_confirmatory_release(tmp_path)


def test_confirmatory_reproduction_holds_for_missing_anchor_and_forged_source(
    tmp_path: Path,
) -> None:
    protocol, _ = _write_confirmatory_release(tmp_path / "anchor")
    (tmp_path / "anchor/protocol/anchor_receipt.json").unlink()
    with pytest.raises(ReproductionHold, match="HOLD_MISSING_SEALED_INPUT"):
        reproduce_sealed_artifacts(tmp_path / "anchor", protocol)

    protocol, _ = _write_confirmatory_release(tmp_path / "forged")
    source = tmp_path / "forged/artifacts/release/confirmatory_rebuild.json"
    source.write_bytes(source.read_bytes().replace(b"CONFIRMATORY", b"FORGED____", 1))
    with pytest.raises(ReproductionHold, match="HOLD_HASH_MISMATCH"):
        reproduce_sealed_artifacts(tmp_path / "forged", protocol)


def test_confirmatory_reproduction_holds_for_audit_or_runtime_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, _ = _write_confirmatory_release(tmp_path / "audit")

    def audit_hold(*_args: object, **_kwargs: object) -> object:
        raise SealedAnalysisHold("HOLD_EXPERIMENT_AUDIT", "fixture audit hold")

    monkeypatch.setattr(reproduce, "analyze_sealed_confirmatory_artifacts", audit_hold)
    with pytest.raises(ReproductionHold, match="HOLD_EXPERIMENT_AUDIT"):
        reproduce_sealed_artifacts(tmp_path / "audit", protocol)

    protocol, _ = _write_confirmatory_release(tmp_path / "runtime")
    bad_runtime = SimpleNamespace(
        freeze_manifest_hash="sha256:" + "d" * 64,
        anchor_receipt_hash="sha256:" + "e" * 64,
        plan_hash="sha256:" + "f" * 64,
    )
    sealed = SimpleNamespace(
        runtime_binding=bad_runtime,
        audit=SimpleNamespace(plan_hash="sha256:" + "f" * 64),
    )
    monkeypatch.setattr(
        reproduce, "analyze_sealed_confirmatory_artifacts", lambda *_args, **_kwargs: sealed
    )
    monkeypatch.setattr(reproduce, "sealed_analysis_receipt_matches", lambda *_: True)
    with pytest.raises(ReproductionHold, match="HOLD_RUNTIME_PROTOCOL_BINDING_MISMATCH"):
        reproduce_sealed_artifacts(tmp_path / "runtime", protocol)
