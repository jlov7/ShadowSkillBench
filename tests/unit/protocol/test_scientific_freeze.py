from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from shadowskillbench.protocol.power_precision import simulate_power_precision
from shadowskillbench.protocol.power_precision_design import execution_design_projection
from shadowskillbench.protocol.scientific_freeze import (
    MULTIPLICITY_MEMBERS,
    POWER_MEMBER_SPECS,
    SCIENTIFIC_FREEZE_INPUTS,
    ScientificFreezeHold,
    derive_scientific_freeze_binding,
)

ROOT = Path(__file__).parents[3]


def _manifest(root: Path) -> bytes:
    payload = {
        "inputs": [
            {
                "path": item.path,
                "role": item.role,
                "sha256": "sha256:" + sha256(root.joinpath(item.path).read_bytes()).hexdigest(),
            }
            for item in SCIENTIFIC_FREEZE_INPUTS
        ]
    }
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )


def _successor_root(tmp_path: Path) -> Path:
    root = tmp_path / "successor"
    for item in SCIENTIFIC_FREEZE_INPUTS:
        destination = root / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / item.path).read_bytes())
    return root


def _write_power_precision_evidence(root: Path, *, status: str = "PASS") -> None:
    configuration_path = root / "config/statistics.yaml"
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    assert type(configuration) is dict
    reporting = configuration["reporting"]
    assert type(reporting) is dict
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
    configuration["profile"] = "SSB-CONFIRMATORY-STATISTICS3-BONFERRONI"
    configuration["bootstrap"] = {
        "confidence_level": 0.95,
        "method": "deterministic_two_way_stratified_pigeonhole",
        "replicates": 40_000,
        "seed": 104_731,
    }
    configuration_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    evidence = {
        "analysis_plan_sha256": "sha256:"
        + sha256((root / "protocol/analysis_plan.v3.json").read_bytes()).hexdigest(),
        "power_precision_plan_sha256": "sha256:"
        + sha256((root / "protocol/power_precision_plan.v3.json").read_bytes()).hexdigest(),
        "corpora_configuration_sha256": "sha256:"
        + sha256((root / "config/corpora.yaml").read_bytes()).hexdigest(),
        "execution_design_source_sha256": "sha256:"
        + sha256(
            (root / "src/shadowskillbench/protocol/power_precision_design.py").read_bytes()
        ).hexdigest(),
        "planner_sha256": "sha256:"
        + sha256((root / "src/shadowskillbench/experiments/planner.py").read_bytes()).hexdigest(),
        "profile": "SSB-SCIENTIFIC-POWER-PRECISION-EVIDENCE-3",
        "schema_version": "1.0",
        "statistics_configuration_sha256": "sha256:"
        + sha256(configuration_path.read_bytes()).hexdigest(),
        "status": status,
        "simulator_profile": "SSB-POWER-PRECISION-SIMULATOR-1",
        "simulator_sha256": "sha256:"
        + sha256(
            (root / "src/shadowskillbench/protocol/power_precision.py").read_bytes()
        ).hexdigest(),
    }
    if status == "PASS":
        corpora = yaml.safe_load((root / "config/corpora.yaml").read_text(encoding="utf-8"))
        designs = execution_design_projection(
            corpora,
            MULTIPLICITY_MEMBERS,
            planner_source=(root / "src/shadowskillbench/experiments/planner.py").read_bytes(),
        )["member_scenarios"]
        assert type(designs) is dict
        scenarios: list[dict[str, object]] = []
        for member in MULTIPLICITY_MEMBERS:
            spec = POWER_MEMBER_SPECS[member]
            design = designs[member]
            assert type(design) is dict
            scenario = {
                "baseline_rate": 0.5,
                "cluster_icc_held_out_case": 0.01,
                "cluster_icc_skill_bundle": 0.01,
                "contrast_structure": spec["contrast_structure"],
                "effect_size": 0.2 if spec["expected_direction"] == "positive" else -0.2,
                "expected_direction": spec["expected_direction"],
                **design,
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
    (root / "protocol/power_precision_evidence.v3.json").write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )


def _exact_design_successor_root(tmp_path: Path) -> Path:
    root = _successor_root(tmp_path)
    _write_power_precision_evidence(root)
    return root


def test_current_v1_freeze_cannot_bind_forward_only_scientific_inputs() -> None:
    with pytest.raises(ScientificFreezeHold, match="HOLD_MISSING_SCIENTIFIC_FREEZE_INPUT"):
        derive_scientific_freeze_binding(
            manifest_bytes=(ROOT / "protocol" / "freeze_manifest.json").read_bytes(),
            repository_root=ROOT,
        )


def test_exact_execution_design_holds_until_real_power_evidence_passes(
    tmp_path: Path,
) -> None:
    root = _exact_design_successor_root(tmp_path)
    manifest = _manifest(root)

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_POWER_PRECISION_EVIDENCE"):
        derive_scientific_freeze_binding(manifest_bytes=manifest, repository_root=root)


@pytest.mark.parametrize(
    "path",
    [
        "config/corpora.yaml",
        "config/statistics.yaml",
        "src/shadowskillbench/experiments/planner.py",
        "src/shadowskillbench/protocol/power_precision_design.py",
        "src/shadowskillbench/protocol/scientific_freeze.py",
    ],
)
def test_scientific_freeze_holds_config_or_verifier_source_drift(tmp_path: Path, path: str) -> None:
    root = _exact_design_successor_root(tmp_path)
    manifest = _manifest(root)
    target = root / path
    target.write_bytes(target.read_bytes() + b"# drift\n")

    with pytest.raises(ScientificFreezeHold, match="HOLD_SCIENTIFIC_FREEZE_HASH_MISMATCH"):
        derive_scientific_freeze_binding(manifest_bytes=manifest, repository_root=root)


def test_former_6040_power_fixture_holds_against_the_frozen_execution_design(
    tmp_path: Path,
) -> None:
    root = _exact_design_successor_root(tmp_path)
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    scenarios = evidence["scenarios"]
    assert type(scenarios) is list and type(scenarios[0]) is dict
    scenarios[0]["held_out_case_clusters"] = 60
    scenarios[0]["skill_bundle_clusters"] = 40
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_DESIGN_MISMATCH"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_stage_b_member_cannot_borrow_the_pooled_stage_a_design(tmp_path: Path) -> None:
    root = _exact_design_successor_root(tmp_path)
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    scenarios = evidence["scenarios"]
    assert type(scenarios) is list
    e7 = next(
        scenario
        for scenario in scenarios
        if type(scenario) is dict
        and scenario["multiplicity_member"]
        == "E7:rate:pooled:false_enforcement:APPROVED_SCOPED_EXCEPTION"
    )
    e7["held_out_case_clusters"] = 40
    e7["skill_bundle_clusters"] = 30
    scenarios[:] = [e7, *(scenario for scenario in scenarios if scenario is not e7)]
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_DESIGN_MISMATCH"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


@pytest.mark.parametrize(
    ("path", "replacement", "code"),
    [
        (
            "protocol/analysis_plan.v3.json",
            b'{"schema_version":"1.0"}\n',
            "HOLD_INVALID_SCIENTIFIC_ANALYSIS_PLAN",
        ),
        (
            "protocol/power_precision_plan.v3.json",
            b'{"schema_version":"1.0"}\n',
            "HOLD_INVALID_SCIENTIFIC_POWER_PRECISION_PLAN",
        ),
    ],
)
def test_successor_scientific_freeze_requires_valid_analysis_and_power_plans(
    tmp_path: Path, path: str, replacement: bytes, code: str
) -> None:
    root = _exact_design_successor_root(tmp_path)
    target = root / path
    target.write_bytes(replacement)

    with pytest.raises(ScientificFreezeHold, match=code):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_successor_scientific_freeze_binds_zero_exclusion_support_policy(tmp_path: Path) -> None:
    root = _exact_design_successor_root(tmp_path)
    path = root / "protocol/analysis_plan.v3.json"
    plan = json.loads(path.read_bytes())
    plan["support_exclusion_policy"] = "sensitivity_evidence_required"
    path.write_bytes(
        json.dumps(plan, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_SCIENTIFIC_ANALYSIS_PLAN"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_template_not_run_evidence_holds_even_with_a_future_statistics_profile(
    tmp_path: Path,
) -> None:
    root = _successor_root(tmp_path)
    _write_power_precision_evidence(root, status="NOT_RUN")

    with pytest.raises(ScientificFreezeHold, match="HOLD_POWER_PRECISION_EVIDENCE_NOT_PASS"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_successor_plan_rejects_a_statistics1_configuration_even_when_evidence_rebinds(
    tmp_path: Path,
) -> None:
    root = _exact_design_successor_root(tmp_path)
    configuration_path = root / "config/statistics.yaml"
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    assert type(configuration) is dict
    configuration["profile"] = "SSB-CONFIRMATORY-STATISTICS1"
    configuration_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    evidence["statistics_configuration_sha256"] = (
        "sha256:" + sha256(configuration_path.read_bytes()).hexdigest()
    )
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(
        ScientificFreezeHold, match="HOLD_INVALID_SCIENTIFIC_STATISTICS_CONFIGURATION"
    ):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda root: (root / "protocol/power_precision_evidence.v3.json").write_bytes(
                b'{"status":"PASS"}\n'
            ),
            "HOLD_INVALID_POWER_PRECISION_EVIDENCE",
        ),
        (
            lambda root: (root / "protocol/power_precision_evidence.v3.json").write_bytes(
                json.dumps(
                    {
                        **json.loads(
                            (root / "protocol/power_precision_evidence.v3.json").read_bytes()
                        ),
                        "analysis_plan_sha256": "sha256:" + "0" * 64,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            ),
            "HOLD_INVALID_POWER_PRECISION_EVIDENCE",
        ),
    ],
)
def test_power_precision_evidence_must_be_canonical_complete_and_hash_bound(
    tmp_path: Path, mutate: Callable[[Path], object], code: str
) -> None:
    root = _exact_design_successor_root(tmp_path)
    mutate(root)

    with pytest.raises(ScientificFreezeHold, match=code):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("effect_size", 0.200_001), ("target_threshold", 0.100_001)],
)
def test_power_precision_evidence_requires_exact_powered_alternative_and_boundary(
    tmp_path: Path, field: str, value: float
) -> None:
    root = _exact_design_successor_root(tmp_path)
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    scenarios = evidence["scenarios"]
    assert type(scenarios) is list and type(scenarios[0]) is dict
    scenarios[0][field] = value
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_POWER_PRECISION_EVIDENCE"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_rate", 0.49),
        ("cluster_icc_held_out_case", 0.02),
        ("cluster_icc_skill_bundle", 0.02),
    ],
)
def test_power_precision_evidence_requires_frozen_scenario_assumptions(
    tmp_path: Path, field: str, value: float
) -> None:
    root = _exact_design_successor_root(tmp_path)
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    scenarios = evidence["scenarios"]
    assert type(scenarios) is list and type(scenarios[0]) is dict
    scenarios[0][field] = value
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_POWER_PRECISION_EVIDENCE"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_power_precision_evidence_recomputes_reported_scenario_values(tmp_path: Path) -> None:
    root = _exact_design_successor_root(tmp_path)
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    scenarios = evidence["scenarios"]
    assert type(scenarios) is list and type(scenarios[0]) is dict
    scenarios[0]["estimated_power"] = 0.8
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_POWER_PRECISION_EVIDENCE"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


@pytest.mark.parametrize("mutate", ["missing", "duplicate"])
def test_power_precision_evidence_requires_exact_multiplicity_member_coverage(
    tmp_path: Path, mutate: str
) -> None:
    root = _exact_design_successor_root(tmp_path)
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    scenarios = evidence["scenarios"]
    assert type(scenarios) is list
    if mutate == "missing":
        scenarios.pop()
    else:
        assert type(scenarios[0]) is dict and type(scenarios[1]) is dict
        scenarios[1]["multiplicity_member"] = scenarios[0]["multiplicity_member"]
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_POWER_PRECISION_EVIDENCE"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("expected_direction", "negative"), ("variance_multiplier", 1.0)],
)
def test_power_precision_evidence_rejects_member_direction_or_variance_tampering(
    tmp_path: Path, field: str, value: object
) -> None:
    root = _exact_design_successor_root(tmp_path)
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    scenarios = evidence["scenarios"]
    assert type(scenarios) is list
    e2 = next(
        scenario
        for scenario in scenarios
        if type(scenario) is dict
        and scenario["multiplicity_member"] == "E2:slope:pooled:completion_under_policy"
    )
    e2[field] = value
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_POWER_PRECISION_EVIDENCE"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_successor_statistics_configuration_requires_exact_member_labels(tmp_path: Path) -> None:
    root = _exact_design_successor_root(tmp_path)
    configuration_path = root / "config/statistics.yaml"
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    assert type(configuration) is dict
    reporting = configuration["reporting"]
    assert type(reporting) is dict
    reporting["multiple_comparison_family"] = [f"E{index}" for index in range(1, 10)]
    configuration_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    evidence["statistics_configuration_sha256"] = (
        "sha256:" + sha256(configuration_path.read_bytes()).hexdigest()
    )
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(
        ScientificFreezeHold, match="HOLD_INVALID_SCIENTIFIC_STATISTICS_CONFIGURATION"
    ):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_power_precision_evidence_rejects_nonfinite_json_numbers(tmp_path: Path) -> None:
    root = _exact_design_successor_root(tmp_path)
    evidence_path = root / "protocol/power_precision_evidence.v3.json"
    evidence = json.loads(evidence_path.read_bytes())
    scenarios = evidence["scenarios"]
    assert type(scenarios) is list and type(scenarios[0]) is dict
    scenarios[0]["estimated_power"] = float("nan")
    evidence_path.write_bytes(
        json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_SCIENTIFIC_FREEZE"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_successor_analysis_plan_rejects_multiplicity_member_drift(tmp_path: Path) -> None:
    root = _exact_design_successor_root(tmp_path)
    plan_path = root / "protocol/analysis_plan.v3.json"
    plan = json.loads(plan_path.read_bytes())
    plan["multiplicity"]["members"][0] = "E1:slope:pooled:task_completion"
    plan_path.write_bytes(
        json.dumps(plan, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_SCIENTIFIC_ANALYSIS_PLAN"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_power_precision_evidence_binds_exact_simulator_source(tmp_path: Path) -> None:
    root = _exact_design_successor_root(tmp_path)
    simulator = root / "src/shadowskillbench/protocol/power_precision.py"
    simulator.write_bytes(simulator.read_bytes() + b"# drift\n")

    with pytest.raises(ScientificFreezeHold, match="HOLD_INVALID_POWER_PRECISION_EVIDENCE"):
        derive_scientific_freeze_binding(manifest_bytes=_manifest(root), repository_root=root)


def test_successor_scientific_freeze_requires_the_power_precision_evidence_input(
    tmp_path: Path,
) -> None:
    root = _exact_design_successor_root(tmp_path)
    manifest = json.loads(_manifest(root))
    manifest["inputs"] = [
        item
        for item in manifest["inputs"]
        if item["path"] != "protocol/power_precision_evidence.v3.json"
    ]

    with pytest.raises(ScientificFreezeHold, match="HOLD_MISSING_SCIENTIFIC_FREEZE_INPUT"):
        derive_scientific_freeze_binding(
            manifest_bytes=json.dumps(
                manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode()
            + b"\n",
            repository_root=root,
        )
