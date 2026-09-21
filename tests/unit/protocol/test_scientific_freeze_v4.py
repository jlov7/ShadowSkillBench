# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.episodes.models import EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.planner import (
    ConditionBinding,
    HeldOutCaseBinding,
    PlannerInputs,
    SkillBundleBinding,
)
from shadowskillbench.experiments.planner_v4 import (
    AUTHORITY_CLASSES,
    DOMAINS,
    RATIOS,
    STAGE_A_EPISODE_COUNT,
    STAGE_B_EPISODE_COUNT,
    plan_confirmatory_episodes_v4,
)
from shadowskillbench.protocol.power_precision_design_v4 import execution_design_projection_v4
from shadowskillbench.protocol.scientific_freeze_v4 import (
    MULTIPLICITY_MEMBERS_V4,
    SCIENTIFIC_FREEZE_V4_INPUTS,
    ScientificFreezeV4Hold,
    derive_scientific_freeze_binding_v4,
)

ROOT = Path(__file__).parents[3]


def _hash(*parts: object) -> str:
    return sha256_ref(list(parts))


def _inputs() -> PlannerInputs:
    skills = tuple(
        SkillBundleBinding(
            domain=domain,  # type: ignore[arg-type]
            bundle_id=f"v4_{domain}_{str(ratio).replace('.', '_')}_{ordinal}",
            contamination_ratio=ratio,
            source_manifest_hash=_hash("source", domain, str(ratio), ordinal),
            compiler_manifest_hash=_hash("compiler", domain, str(ratio), ordinal),
            compiled_skill_artifact_hash=_hash("artifact", domain, str(ratio), ordinal),
            rendered_skill_hash=_hash("rendered", domain, str(ratio), ordinal),
        )
        for domain in DOMAINS
        for ratio in RATIOS
        for ordinal in range(7)
    )
    stage_a = tuple(
        HeldOutCaseBinding(
            domain=domain,  # type: ignore[arg-type]
            case_id=f"a_{domain}_{index}",
            case_manifest_hash=_hash("a-case", domain, index),
            world_hash=_hash("a-world", domain, index),
            authority_graph_hash=_hash("a-authority", domain, index),
        )
        for domain in DOMAINS
        for index in range(70)
    )
    stage_b = tuple(
        HeldOutCaseBinding(
            domain=domain,  # type: ignore[arg-type]
            case_id=f"b_{domain}_{authority}_{index}",
            case_manifest_hash=_hash("b-case", domain, authority, index),
            world_hash=_hash("b-world", domain, authority, index),
            authority_graph_hash=_hash("b-authority", domain, authority, index),
            authority_class=authority,  # type: ignore[arg-type]
        )
        for domain in DOMAINS
        for authority in AUTHORITY_CLASSES
        for index in range(22)
    )
    bindings = tuple(
        ConditionBinding(
            domain=domain,  # type: ignore[arg-type]
            condition=condition,
            context_contract_hash=_hash("context", domain, condition.value),
            policy_hash=(
                _hash("a-policy", domain)
                if condition
                in {
                    ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
                    ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
                    ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
                }
                else _hash("b-policy", domain)
                if condition
                in {
                    ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
                    ExperimentCondition.B2_AUTHORITY_RESOLVER,
                    ExperimentCondition.B3_DETERMINISTIC_GATE,
                }
                else _hash("a5-handbook", domain)
                if condition is ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER
                else None
            ),
        )
        for domain in DOMAINS
        for condition in ExperimentCondition
    )
    return PlannerInputs(skills, stage_a, stage_b, bindings)


def _copy_v4_surface(tmp_path: Path) -> Path:
    root = tmp_path / "v4"
    for item in SCIENTIFIC_FREEZE_V4_INPUTS:
        target = root / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item.path, target)
    return root


def _manifest(root: Path) -> bytes:
    payload = {
        "anchor_status": "HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY",
        "inputs": [
            {
                "path": item.path,
                "role": item.role,
                "sha256": "sha256:" + sha256((root / item.path).read_bytes()).hexdigest(),
            }
            for item in SCIENTIFIC_FREEZE_V4_INPUTS
        ],
        "profile": "SSB-SCIENTIFIC-PREFREEZE-4",
        "schema_version": "4.0",
    }
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )


def _rewrite(path: Path, payload: object) -> None:
    path.write_bytes(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )


def test_v4_planner_has_exact_closed_matrix_counts() -> None:
    plan = plan_confirmatory_episodes_v4(_inputs())
    counts = Counter(item.stage for item in plan.episodes)

    assert counts == {
        EpisodeStage.CONFIRMATORY_A: STAGE_A_EPISODE_COUNT,
        EpisodeStage.CONFIRMATORY_B: STAGE_B_EPISODE_COUNT,
    }
    assert len(plan.episodes) == 78_120


def test_v4_e4_design_marks_the_a1_control_as_unbundled() -> None:
    configuration = yaml.safe_load((ROOT / "config/corpora.v4.yaml").read_text(encoding="utf-8"))
    design = execution_design_projection_v4(
        configuration,
        MULTIPLICITY_MEMBERS_V4,
        planner_source=(ROOT / "src/shadowskillbench/experiments/planner_v4.py").read_bytes(),
    )
    members = design["member_scenarios"]
    assert type(members) is dict
    e4 = members["E4:average_cup_contrast:pooled:completion_under_policy"]
    assert e4 == {
        "alternative_arm_skill_bundle_clusters": 70,
        "baseline_arm_skill_bundle_clusters": 0,
        "held_out_case_clusters": 140,
        "repetitions": 3,
        "skill_bundle_clusters": 70,
    }


def test_v4_freeze_validates_exact_bytes_and_all_thresholds_pass(tmp_path: Path) -> None:
    root = _copy_v4_surface(tmp_path)
    binding = derive_scientific_freeze_binding_v4(
        manifest_bytes=_manifest(root), repository_root=root
    )

    assert len(binding.member_execution_designs) == len(MULTIPLICITY_MEMBERS_V4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "SSB-SCIENTIFIC-FREEZE-4"),
        ("schema_version", "4.1"),
        ("anchor_status", "PASS_FULL_V4_CUSTODY"),
    ],
)
def test_v4_freeze_rejects_any_non_prefreeze_top_level_boundary(
    tmp_path: Path, field: str, value: str
) -> None:
    root = _copy_v4_surface(tmp_path)
    manifest = json.loads(_manifest(root))
    manifest[field] = value

    with pytest.raises(ScientificFreezeV4Hold, match="HOLD_INVALID_V4_SCIENTIFIC_FREEZE"):
        derive_scientific_freeze_binding_v4(
            manifest_bytes=json.dumps(
                manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode()
            + b"\n",
            repository_root=root,
        )


def test_v4_freeze_uses_one_snapshot_per_input_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_v4_surface(tmp_path)
    target = root / "protocol/power_precision_plan.v4.json"
    manifest_bytes = _manifest(root)
    original_read_bytes = Path.read_bytes
    changed = False

    def read_then_mutate(path: Path) -> bytes:
        nonlocal changed
        value = original_read_bytes(path)
        if path == target and not changed:
            changed = True
            target.write_bytes(value.replace(b'"schema_version":"1.0"', b'"schema_version":"0.0"'))
        return value

    monkeypatch.setattr(Path, "read_bytes", read_then_mutate)
    binding = derive_scientific_freeze_binding_v4(
        manifest_bytes=manifest_bytes, repository_root=root
    )

    assert changed
    assert binding.manifest_hash.startswith("sha256:")
    evidence = json.loads((root / "protocol/power_precision_evidence.v4.json").read_bytes())
    scenarios = evidence["scenarios"]
    assert min(item["estimated_power"] for item in scenarios) == 0.81185
    assert all(item["estimated_power"] >= 0.8 for item in scenarios)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("stale", "HOLD_INVALID_V4_POWER_EVIDENCE"),
        ("mixed", "HOLD_V4_DESIGN_MISMATCH"),
        ("threshold", "HOLD_INVALID_V4_POWER_EVIDENCE"),
        ("interval", "HOLD_INVALID_V4_POWER_EVIDENCE"),
    ],
)
def test_v4_evidence_rejects_stale_mixed_and_threshold_cases(
    tmp_path: Path, mutation: str, code: str
) -> None:
    root = _copy_v4_surface(tmp_path)
    evidence_path = root / "protocol/power_precision_evidence.v4.json"
    evidence = json.loads(evidence_path.read_bytes())
    scenarios = evidence["scenarios"]
    if mutation == "stale":
        evidence["power_precision_plan_sha256"] = "sha256:" + "0" * 64
    elif mutation == "mixed":
        scenarios[6]["held_out_case_clusters"] = 40
    elif mutation == "interval":
        scenarios[6]["expected_interval_width"] += 1e-12
    else:
        scenarios[6]["estimated_power"] = 0.799
    _rewrite(evidence_path, evidence)

    with pytest.raises(ScientificFreezeV4Hold, match=code):
        derive_scientific_freeze_binding_v4(manifest_bytes=_manifest(root), repository_root=root)


def test_v4_freeze_rejects_wrong_planner_count_formula(tmp_path: Path) -> None:
    root = _copy_v4_surface(tmp_path)
    planner = root / "src/shadowskillbench/experiments/planner_v4.py"
    planner.write_bytes(
        planner.read_bytes().replace(
            b"STAGE_B_EPISODE_COUNT = 18_480", b"STAGE_B_EPISODE_COUNT = 18_479"
        )
    )

    with pytest.raises(ScientificFreezeV4Hold, match="HOLD_INVALID_V4_EXECUTION_DESIGN"):
        derive_scientific_freeze_binding_v4(manifest_bytes=_manifest(root), repository_root=root)
