from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.episodes import EpisodePlan, EpisodeStage, ExperimentCondition
from shadowskillbench.episodes.executor import episode_result_projection
from shadowskillbench.experiments.audit import (
    ExecutionBinding,
    FrozenArtifactCatalog,
    FrozenExclusionRule,
    PolicyText,
    audit_confirmatory_run,
)
from shadowskillbench.experiments.planner import (
    ConditionBinding,
    ConfirmatoryEpisodePlan,
    HeldOutCaseBinding,
    PlannedEpisode,
    SkillBundleBinding,
)
from shadowskillbench.experiments.runner import run_confirmatory_plan
from shadowskillbench.models import ProviderCapabilities, ScriptedModelClient
from shadowskillbench.skills.compiler import compiled_skill_artifact_hash
from tests.integration.episodes.test_scripted_executor import HASH, _client, _plan


def _episode(
    condition: ExperimentCondition = ExperimentCondition.A0_BARE,
) -> tuple[ConfirmatoryEpisodePlan, dict[str, EpisodePlan]]:
    base, _ = _plan(condition=condition)
    skill = None
    if base.skill is not None:
        skill = SkillBundleBinding(
            domain="access_provisioning",
            bundle_id="skill_access_provisioning",
            contamination_ratio=Decimal("0.75"),
            source_manifest_hash=HASH,
            compiler_manifest_hash=base.skill.compiler_manifest_hash,
            compiled_skill_artifact_hash=compiled_skill_artifact_hash(base.skill),
            rendered_skill_hash=base.skill.rendered_skill_hash,
        )
    planned = PlannedEpisode(
        stage=(
            EpisodeStage.CONFIRMATORY_B
            if condition.value.startswith("B")
            else EpisodeStage.CONFIRMATORY_A
        ),
        condition=condition,
        case=HeldOutCaseBinding(
            domain="access_provisioning",
            case_id=base.task.case_id,
            case_manifest_hash=HASH,
            world_hash=base.manifest.world_hash,
            authority_graph_hash=base.authority_decision.authority_set_hash,
        ),
        condition_binding=ConditionBinding(
            domain="access_provisioning",
            condition=condition,
            context_contract_hash=HASH,
            policy_hash=None if base.policy is None else base.policy.rendered_hash,
        ),
        repeat_index=1,
        skill=skill,
    )
    materialized = EpisodePlan(
        manifest=base.manifest.model_copy(
            update={
                "episode_id": planned.episode_id,
                "stage": planned.stage,
                "contamination_ratio": None if skill is None else float(skill.contamination_ratio),
            }
        ),
        task=base.task,
        initial_state=base.initial_state,
        domain_case=base.domain_case,
        authority_decision=base.authority_decision,
        authority_records=base.authority_records,
        authority_query=base.authority_query,
        skill=base.skill,
        policy=base.policy,
        order_assignment=base.order_assignment,
    )
    return ConfirmatoryEpisodePlan(episodes=(planned,)), {planned.episode_id: materialized}


def _catalog(plan: ConfirmatoryEpisodePlan, plans: dict[str, EpisodePlan]) -> FrozenArtifactCatalog:
    episode = plan.episodes[0]
    materialized = plans[episode.episode_id]
    policy = materialized.policy
    skill = episode.skill
    return FrozenArtifactCatalog(
        case_manifest_hashes=frozenset({episode.case.case_manifest_hash}),
        context_contract_hashes=frozenset({episode.condition_binding.context_contract_hash}),
        source_manifest_hashes=frozenset()
        if skill is None
        else frozenset({skill.source_manifest_hash}),
        compiler_manifest_hashes=frozenset()
        if skill is None
        else frozenset({skill.compiler_manifest_hash}),
        compiled_skill_artifact_hashes=(
            frozenset() if skill is None else frozenset({skill.compiled_skill_artifact_hash})
        ),
        rendered_skill_hashes=frozenset()
        if skill is None
        else frozenset({skill.rendered_skill_hash}),
        policies=()
        if policy is None
        else (PolicyText(rendered_hash=policy.rendered_hash, rendered_text=policy.rendered_text),),
        development_entities=frozenset({"development-entity"}),
        confirmatory_entities=frozenset({"confirmatory-entity"}),
        development_values=frozenset({"development-value"}),
        confirmatory_values=frozenset({"confirmatory-value"}),
    )


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
            {"rendered_hash": item.rendered_hash, "rendered_text": item.rendered_text}
            for item in sorted(catalog.policies, key=lambda item: item.rendered_hash)
        ],
        "development_entities": sorted(catalog.development_entities or ()),
        "confirmatory_entities": sorted(catalog.confirmatory_entities or ()),
        "development_values": sorted(catalog.development_values or ()),
        "confirmatory_values": sorted(catalog.confirmatory_values or ()),
        "exclusion_rule_hash": catalog.exclusion_rule_hash,
    }


def _package_manifest(
    plan: ConfirmatoryEpisodePlan,
    plans: dict[str, EpisodePlan],
    catalog: FrozenArtifactCatalog,
    rule: FrozenExclusionRule,
) -> dict[str, object]:
    entries = []
    for episode in plan.episodes:
        value = {
            "profile": "SSB-EXECUTION-PLAN1",
            "episode_id": episode.episode_id,
            "planned_manifest_hash": episode.manifest_hash,
            "execution_plan": plans[episode.episode_id].model_dump(mode="json"),
        }
        entries.append(
            {
                "path": f"execution-plans/{episode.episode_id}.json",
                "content_hash": sha256_ref(value),
            }
        )
    view: dict[str, object] = {
        "profile": "SSB-CONFIRMATORY-EXECUTION-PACKAGE1",
        "freeze_manifest_hash": HASH,
        "anchor_receipt_hash": HASH,
        "plan_hash": plan.plan_hash,
        "catalog_hash": sha256_ref(_catalog_payload(catalog)),
        "exclusion_rule_hash": rule.rule_hash,
        "run_descriptor_hash": HASH,
        "execution_plans": entries,
    }
    return {**view, "package_hash": sha256_ref(view)}


def _write_bindings(
    run_directory: Path,
    plan: ConfirmatoryEpisodePlan,
    plans: dict[str, EpisodePlan],
) -> None:
    directory = run_directory / "execution-bindings"
    directory.mkdir(exist_ok=True)
    for episode in plan.episodes:
        execution = plans[episode.episode_id]
        skill = episode.skill
        binding = ExecutionBinding(
            episode_id=episode.episode_id,
            execution_manifest_hash=sha256_ref(execution.manifest.model_dump(mode="json")),
            condition=episode.condition,
            context_contract_hash=episode.condition_binding.context_contract_hash,
            policy_hash=None if execution.policy is None else execution.policy.rendered_hash,
            policy_text=None if execution.policy is None else execution.policy.rendered_text,
            source_manifest_hash=None if skill is None else skill.source_manifest_hash,
            compiler_manifest_hash=None if skill is None else skill.compiler_manifest_hash,
            compiled_skill_artifact_hash=(
                None if skill is None else skill.compiled_skill_artifact_hash
            ),
            rendered_skill_hash=None if skill is None else skill.rendered_skill_hash,
            provider=execution.manifest.executor_model.provider,
            model=execution.manifest.executor_model.model,
            model_version=execution.manifest.executor_model.model_version_date,
            package_execution_plan_hash=sha256_ref(
                {
                    "profile": "SSB-EXECUTION-PLAN1",
                    "episode_id": episode.episode_id,
                    "planned_manifest_hash": episode.manifest_hash,
                    "execution_plan": execution.model_dump(mode="json"),
                }
            ),
        )
        (directory / f"{episode.episode_id}.json").write_bytes(
            canonical_json_bytes({"profile": "SSB-AUDIT-BINDINGS1", **binding.projection()})
        )


def _prepared(
    tmp_path: Path, condition: ExperimentCondition = ExperimentCondition.A0_BARE
) -> tuple[ConfirmatoryEpisodePlan, dict[str, EpisodePlan], Path, FrozenArtifactCatalog]:
    plan, plans = _episode(condition)
    run_directory = tmp_path / "run"
    model = next(iter(plans.values())).manifest.executor_model
    rule = FrozenExclusionRule(
        allowed_error_codes=("MODEL_PROVIDER_TRANSIENT",),
        max_retries_per_episode=1,
        max_total_exclusions=1,
        max_exclusions_per_primary_cell=1,
    )
    catalog = replace(_catalog(plan, plans), exclusion_rule_hash=rule.rule_hash)
    package_manifest = _package_manifest(plan, plans, catalog, rule)
    _run_in(
        tmp_path,
        run_confirmatory_plan(
            plan,
            run_directory=run_directory,
            materialize=lambda episode: plans[episode.episode_id],
            client_factory=lambda episode: _client(({"kind": "finish", "summary": "Done."},)),
            runtime_binding={
                "profile": "SSB-RUNTIME-BINDING1",
                "plan_hash": plan.plan_hash,
                "package_hash": package_manifest["package_hash"],
                "run_descriptor_hash": HASH,
                "freeze_manifest_hash": HASH,
                "anchor_receipt_hash": HASH,
                "operator_endpoint_hash": HASH,
                "operator_api_key_environment": "SSB_TEST_KEY",
                "provider": model.provider,
                "model": model.model,
                "model_version": model.model_version_date,
            },
            catalog=catalog,
            exclusion_rule=rule,
            package_manifest=package_manifest,
        ),
    )
    _write_bindings(run_directory, plan, plans)
    return plan, plans, run_directory, catalog


def _run_in(path: Path, coroutine: Any) -> Any:
    previous = Path.cwd()
    try:
        os.chdir(path)
        return asyncio.run(coroutine)
    finally:
        os.chdir(previous)


def _bindings(run_directory: Path) -> tuple[Path, dict[str, Any]]:
    path = next((run_directory / "execution-bindings").glob("*.json"))
    return path, json.loads(path.read_bytes())


def _replace_bindings(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _write_v15d_configuration_failure(path: Path) -> None:
    envelope = json.loads(path.read_bytes())
    result = envelope["result"]
    result["claim"] = "none"
    result["disposition"] = "model_failure"
    result["error_code"] = "CONFIGURATION_ERROR"
    for turn in result["trace"]:
        turn["output"] = None
        turn["receipt"].update(
            {
                "raw_request_hash": None,
                "raw_response_hash": None,
                "structured_output_schema_hash": None,
                "usage": None,
                "cost": None,
                "attempts": 0,
                "error_code": "CONFIGURATION_ERROR",
                "finish_reason": None,
                "before_meaningful_behavior": False,
                "output_cap_exhausted": None,
            }
        )
    result["overhead"].update(
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": None,
        }
    )
    result["content_hash"] = sha256_ref(episode_result_projection(result))
    digest = result["content_hash"].removeprefix("sha256:")
    result["artifact_ref"] = f"artifacts/episode_result/{digest[:2]}/{digest}.json"
    envelope["result_content_hash"] = result["content_hash"]
    path.write_bytes(canonical_json_bytes(envelope))


def test_clean_synthetic_run_passes_and_report_is_self_hashed(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path)

    report = audit_confirmatory_run(plan, run_directory, catalog=catalog)

    assert report.status == "PASS"
    assert report.findings == ()
    assert report.report_hash == sha256_ref(report.projection())


def test_v15d_shaped_pre_dispatch_configuration_failure_holds(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path)
    _write_v15d_configuration_failure(next((run_directory / "results").glob("*.json")))

    report = audit_confirmatory_run(plan, run_directory, catalog=catalog)

    assert report.status == "HOLD"
    assert {(finding.code, finding.detail) for finding in report.findings} >= {
        ("PRE_DISPATCH_CONFIGURATION_FAILURE", "configuration error")
    }
    assert report.model_failure_count == 1
    assert report.pre_dispatch_configuration_failure_count == 1


def test_stage_scoped_audit_preserves_full_plan_custody(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path)

    report = audit_confirmatory_run(
        plan,
        run_directory,
        catalog=catalog,
        stage=EpisodeStage.CONFIRMATORY_A,
    )

    assert report.status == "PASS"
    assert report.audited_episode_ids == (plan.episodes[0].episode_id,)


def test_audit_rejects_runtime_model_drift(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path)
    binding_path, binding = _bindings(run_directory)
    binding["provider"] = "different-provider"
    _replace_bindings(binding_path, binding)

    report = audit_confirmatory_run(plan, run_directory, catalog=catalog)

    assert report.status == "HOLD"
    assert "RUNTIME_MODEL_MISMATCH" in {finding.code for finding in report.findings}


def test_audit_rejects_forged_package_manifest(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path)
    manifest_path = run_directory / "package-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["package_hash"] = HASH
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    report = audit_confirmatory_run(plan, run_directory, catalog=catalog)

    assert report.status == "HOLD"
    assert "INVALID_PACKAGE_MANIFEST" in {finding.code for finding in report.findings}


def test_missing_and_duplicate_result_cells_hold(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path)
    result = next((run_directory / "results").glob("*.json"))
    (run_directory / "results" / "duplicate.json").write_bytes(result.read_bytes())

    duplicate_report = audit_confirmatory_run(plan, run_directory, catalog=catalog)
    result.unlink()
    (run_directory / "results" / "duplicate.json").unlink()
    missing_report = audit_confirmatory_run(plan, run_directory, catalog=catalog)

    assert "DUPLICATE_PLANNED_RESULT" in {finding.code for finding in duplicate_report.findings}
    assert "MISSING_PLANNED_EPISODE" in {finding.code for finding in missing_report.findings}


def test_duplicate_planned_cell_and_invalid_semantic_result_hold(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path)
    manifest_path = run_directory / "run-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["episodes"].append(manifest["episodes"][0])
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    result_path = next((run_directory / "results").glob("*.json"))
    envelope = json.loads(result_path.read_bytes())
    envelope["result"]["overhead"]["model_calls"] += 1
    result_path.write_bytes(canonical_json_bytes(envelope))

    report = audit_confirmatory_run(plan, run_directory, catalog=catalog)

    assert {finding.code for finding in report.findings} >= {
        "DUPLICATE_CELL",
        "INVALID_RESULT_ENVELOPE",
        "MISSING_PLANNED_EPISODE",
    }


def test_condition_context_policy_and_changed_skill_mismatches_hold(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path, ExperimentCondition.B1_FLAT_POLICY_SYSTEM)
    binding_path, binding = _bindings(run_directory)
    binding["condition"] = ExperimentCondition.B2_AUTHORITY_RESOLVER.value
    binding["context_contract_hash"] = sha256_ref("different-context")
    binding["policy_hash"] = sha256_ref("different-policy")
    binding["policy_text"] = "different policy"
    binding["source_manifest_hash"] = sha256_ref("different-source")
    binding["compiler_manifest_hash"] = sha256_ref("different-compiler")
    binding["compiled_skill_artifact_hash"] = sha256_ref("different-skill")
    binding["rendered_skill_hash"] = sha256_ref("different-rendered-skill")
    _replace_bindings(binding_path, binding)

    report = audit_confirmatory_run(plan, run_directory, catalog=catalog)

    assert {finding.code for finding in report.findings} >= {
        "EXECUTION_CONDITION_MISMATCH",
        "CONTEXT_CONTRACT_MISMATCH",
        "POLICY_HASH_MISMATCH",
        "POLICY_TEXT_MISMATCH",
        "CHANGED_SOURCE_MANIFEST",
        "CHANGED_COMPILER_MANIFEST",
        "CHANGED_COMPILED_SKILL",
        "CHANGED_RENDERED_SKILL",
    }


def test_missing_referents_and_corpus_entity_value_leakage_hold(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path)
    leaking = replace(
        catalog,
        case_manifest_hashes=frozenset(),
        development_entities=frozenset({"employee_1"}),
        confirmatory_entities=frozenset({"employee_1"}),
        development_values=frozenset({"500"}),
        confirmatory_values=frozenset({"500"}),
    )

    report = audit_confirmatory_run(plan, run_directory, catalog=leaking)

    assert {finding.code for finding in report.findings} >= {
        "UNVERIFIED_MISSING_REFERENT",
        "CORPUS_LEAKAGE",
    }


def test_unplanned_execution_binding_holds(tmp_path: Path) -> None:
    plan, _, run_directory, catalog = _prepared(tmp_path)
    binding_path, binding = _bindings(run_directory)
    binding["episode_id"] = "episode_unplanned"
    (binding_path.parent / "episode_unplanned.json").write_bytes(canonical_json_bytes(binding))

    report = audit_confirmatory_run(plan, run_directory, catalog=catalog)

    assert "UNPLANNED_EXECUTION_BINDING" in {finding.code for finding in report.findings}


def test_unapproved_technical_exclusion_holds_then_accepts_frozen_allowlist(tmp_path: Path) -> None:
    plan, plans = _episode()
    run_directory = tmp_path / "run"
    catalog = _catalog(plan, plans)
    rule = FrozenExclusionRule(
        allowed_error_codes=("MODEL_PROVIDER_TRANSIENT",),
        max_retries_per_episode=1,
        max_total_exclusions=1,
        max_exclusions_per_primary_cell=1,
    )
    frozen_catalog = replace(catalog, exclusion_rule_hash=rule.rule_hash)
    package_manifest = _package_manifest(plan, plans, frozen_catalog, rule)

    def client_factory(episode: PlannedEpisode) -> ScriptedModelClient:
        return ScriptedModelClient(
            capabilities=ProviderCapabilities(
                provider="fixture",
                model="scripted-executor",
                model_version="2026-08-24",
                supports_system_role=True,
                supports_developer_role=True,
                supports_seed=True,
                supports_structured_output=True,
            ),
            script=("transient",),
            max_attempts=1,
        )

    _run_in(
        tmp_path,
        run_confirmatory_plan(
            plan,
            run_directory=run_directory,
            materialize=lambda episode: plans[episode.episode_id],
            client_factory=client_factory,
            max_technical_retries=1,
            runtime_binding={
                "profile": "SSB-RUNTIME-BINDING1",
                "plan_hash": plan.plan_hash,
                "package_hash": package_manifest["package_hash"],
                "run_descriptor_hash": HASH,
                "freeze_manifest_hash": HASH,
                "anchor_receipt_hash": HASH,
                "operator_endpoint_hash": HASH,
                "operator_api_key_environment": "SSB_TEST_KEY",
                "provider": next(iter(plans.values())).manifest.executor_model.provider,
                "model": next(iter(plans.values())).manifest.executor_model.model,
                "model_version": next(
                    iter(plans.values())
                ).manifest.executor_model.model_version_date,
            },
            catalog=frozen_catalog,
            exclusion_rule=rule,
            package_manifest=package_manifest,
        ),
    )
    _write_bindings(run_directory, plan, plans)

    held = audit_confirmatory_run(plan, run_directory, catalog=catalog)
    approved = audit_confirmatory_run(
        plan,
        run_directory,
        catalog=frozen_catalog,
        exclusion_rule=rule,
    )

    assert "UNAPPROVED_TECHNICAL_EXCLUSION" in {finding.code for finding in held.findings}
    assert "UNAPPROVED_TECHNICAL_EXCLUSION" not in {finding.code for finding in approved.findings}
    assert approved.status == "PASS"
    assert approved.model_failure_count == 1
    assert approved.pre_dispatch_configuration_failure_count == 0


def test_frozen_exclusion_rule_hash_binds_retry_and_exclusion_caps() -> None:
    base = FrozenExclusionRule(
        allowed_error_codes=("MODEL_PROVIDER_TERMINAL",),
        max_retries_per_episode=1,
        max_total_exclusions=1,
        max_exclusions_per_primary_cell=1,
    )
    changed_budget = FrozenExclusionRule(
        allowed_error_codes=("MODEL_PROVIDER_TERMINAL",),
        max_retries_per_episode=2,
        max_total_exclusions=1,
        max_exclusions_per_primary_cell=1,
    )

    assert base.rule_hash != changed_budget.rule_hash
    with pytest.raises(ValueError, match="two of three"):
        FrozenExclusionRule(
            allowed_error_codes=("MODEL_PROVIDER_TERMINAL",),
            max_retries_per_episode=1,
            max_total_exclusions=1,
            max_exclusions_per_primary_cell=2,
        )
