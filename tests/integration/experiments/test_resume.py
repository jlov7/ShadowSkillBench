from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.episodes import (
    EpisodeBudgets,
    EpisodePlan,
    EpisodeStage,
    ExperimentCondition,
    parse_episode_result,
    run_episode,
)
from shadowskillbench.episodes.executor import episode_result_projection
from shadowskillbench.experiments.audit import FrozenArtifactCatalog, FrozenExclusionRule
from shadowskillbench.experiments.io import ExperimentArtifactHold, load_confirmatory_audit_inputs
from shadowskillbench.experiments.planner import (
    ConditionBinding,
    ConfirmatoryEpisodePlan,
    HeldOutCaseBinding,
    PlannedEpisode,
    SkillBundleBinding,
)
from shadowskillbench.experiments.runner import RunCustodyError, RunnerError, run_confirmatory_plan
from shadowskillbench.models import ModelAdapterError, ProviderCapabilities, ScriptedModelClient
from shadowskillbench.skills.compiler import compiled_skill_artifact_hash
from tests.integration.episodes.test_scripted_executor import HASH, _client, _plan


def _matrix(count: int) -> tuple[ConfirmatoryEpisodePlan, dict[str, EpisodePlan]]:
    base, _ = _plan()
    case = HeldOutCaseBinding(
        domain="access_provisioning",
        case_id=base.task.case_id,
        case_manifest_hash=HASH,
        world_hash=base.manifest.world_hash,
        authority_graph_hash=base.authority_decision.authority_set_hash,
    )
    binding = ConditionBinding(
        domain="access_provisioning",
        condition=base.manifest.condition,
        context_contract_hash=HASH,
        policy_hash=None,
    )
    episodes = tuple(
        PlannedEpisode(
            stage=EpisodeStage.CONFIRMATORY_A,
            condition=base.manifest.condition,
            case=case,
            condition_binding=binding,
            repeat_index=index + 1,
        )
        for index in range(count)
    )
    materialized: dict[str, EpisodePlan] = {}
    for episode in episodes:
        manifest = base.manifest.model_copy(
            update={"episode_id": episode.episode_id, "stage": episode.stage}
        )
        materialized[episode.episode_id] = EpisodePlan(
            manifest=manifest,
            task=base.task,
            initial_state=base.initial_state,
            domain_case=base.domain_case,
            authority_decision=base.authority_decision,
            authority_records=base.authority_records,
            authority_query=base.authority_query,
        )
    return ConfirmatoryEpisodePlan(episodes=episodes), materialized


def _materializer(materialized: dict[str, EpisodePlan]) -> Any:
    return lambda episode: materialized[episode.episode_id]


def _client_factory(_: object) -> Any:
    return _client(({"kind": "finish", "summary": "Done."},))


def _package_manifest(
    plan: ConfirmatoryEpisodePlan,
    materialized: dict[str, EpisodePlan],
    catalog: FrozenArtifactCatalog,
    rule: FrozenExclusionRule,
) -> dict[str, object]:
    catalog_payload = {
        "profile": "SSB-AUDIT-CATALOG1",
        "case_manifest_hashes": sorted(catalog.case_manifest_hashes),
        "context_contract_hashes": sorted(catalog.context_contract_hashes),
        "source_manifest_hashes": sorted(catalog.source_manifest_hashes),
        "compiler_manifest_hashes": sorted(catalog.compiler_manifest_hashes),
        "compiled_skill_artifact_hashes": sorted(catalog.compiled_skill_artifact_hashes),
        "rendered_skill_hashes": sorted(catalog.rendered_skill_hashes),
        "policies": [],
        "development_entities": sorted(catalog.development_entities or ()),
        "confirmatory_entities": sorted(catalog.confirmatory_entities or ()),
        "development_values": sorted(catalog.development_values or ()),
        "confirmatory_values": sorted(catalog.confirmatory_values or ()),
        "exclusion_rule_hash": catalog.exclusion_rule_hash,
    }
    entries = []
    for episode in plan.episodes:
        value = {
            "profile": "SSB-EXECUTION-PLAN1",
            "episode_id": episode.episode_id,
            "planned_manifest_hash": episode.manifest_hash,
            "execution_plan": materialized[episode.episode_id].model_dump(mode="json"),
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
        "catalog_hash": sha256_ref(catalog_payload),
        "exclusion_rule_hash": rule.rule_hash,
        "run_descriptor_hash": HASH,
        "execution_plans": entries,
    }
    return {**view, "package_hash": sha256_ref(view)}


def _run(
    plan: ConfirmatoryEpisodePlan,
    materialized: dict[str, EpisodePlan],
    run_directory: Path,
    **kwargs: Any,
) -> Any:
    model = next(iter(materialized.values())).manifest.executor_model
    package_manifest = kwargs.get("package_manifest")
    catalog = kwargs.get("catalog")
    rule = kwargs.get("exclusion_rule")
    if (
        package_manifest is None
        and type(catalog) is FrozenArtifactCatalog
        and type(rule) is FrozenExclusionRule
    ):
        package_manifest = _package_manifest(plan, materialized, catalog, rule)
        kwargs["package_manifest"] = package_manifest
    kwargs.setdefault(
        "runtime_binding",
        {
            "profile": "SSB-RUNTIME-BINDING1",
            "plan_hash": plan.plan_hash,
            "package_hash": (
                HASH if type(package_manifest) is not dict else package_manifest["package_hash"]
            ),
            "run_descriptor_hash": HASH,
            "freeze_manifest_hash": HASH,
            "anchor_receipt_hash": HASH,
            "operator_endpoint_hash": HASH,
            "operator_api_key_environment": "SSB_TEST_KEY",
            "provider": model.provider,
            "model": model.model,
            "model_version": model.model_version_date,
        },
    )
    return asyncio.run(
        run_confirmatory_plan(
            plan,
            run_directory=run_directory,
            materialize=_materializer(materialized),
            client_factory=_client_factory,
            **kwargs,
        )
    )


def test_resume_skips_completed_exact_manifest_and_preserves_atomic_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(2)
    run_directory = tmp_path / "run"
    calls: list[str] = []

    async def execute(plan: EpisodePlan, client: Any) -> Any:
        calls.append(plan.manifest.episode_id)
        return await run_episode(plan, client)

    first = _run(plan, materialized, run_directory, execute=execute)
    result_paths = sorted((run_directory / "results").glob("*.json"))
    result_bytes = {path.name: path.read_bytes() for path in result_paths}
    second = _run(plan, materialized, run_directory, execute=execute)

    assert set(first.completed_episode_ids) == {episode.episode_id for episode in plan.episodes}
    assert second.completed_episode_ids == ()
    assert set(second.skipped_episode_ids) == {episode.episode_id for episode in plan.episodes}
    assert len(calls) == 2
    assert {path.name: path.read_bytes() for path in result_paths} == result_bytes
    assert all(
        json.loads(path.read_bytes())["plan_hash"] == plan.plan_hash for path in result_paths
    )


def test_runner_publishes_reloadable_audit_inputs_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(1)
    episode = plan.episodes[0]
    catalog = FrozenArtifactCatalog(
        case_manifest_hashes=frozenset({episode.case.case_manifest_hash}),
        context_contract_hashes=frozenset({episode.condition_binding.context_contract_hash}),
        development_entities=frozenset({"development_entity"}),
        confirmatory_entities=frozenset({"confirmatory_entity"}),
        development_values=frozenset({"development_value"}),
        confirmatory_values=frozenset({"confirmatory_value"}),
    )
    rule = FrozenExclusionRule(
        allowed_error_codes=("MODEL_PROVIDER_TRANSIENT",),
        max_retries_per_episode=1,
        max_total_exclusions=1,
        max_exclusions_per_primary_cell=1,
    )
    catalog = replace(catalog, exclusion_rule_hash=rule.rule_hash)

    _run(plan, materialized, Path("run"), catalog=catalog, exclusion_rule=rule)
    loaded = load_confirmatory_audit_inputs(Path("run"))

    assert loaded.plan == plan
    assert loaded.catalog == catalog
    assert loaded.exclusion_rule == rule
    assert loaded.runtime_binding.plan_hash == plan.plan_hash
    assert (tmp_path / "run/plan.json").is_file()
    assert (tmp_path / "run/catalog.json").is_file()

    (tmp_path / "run/runtime-binding.json").unlink()
    with pytest.raises(ExperimentArtifactHold, match="runtime-binding.json"):
        load_confirmatory_audit_inputs(Path("run"))


def test_crash_after_completed_episode_resumes_only_missing_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(2)
    run_directory = tmp_path / "run"
    calls = 0

    async def crash_after_first(execution_plan: EpisodePlan, client: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated crash")
        return await run_episode(execution_plan, client)

    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(plan, materialized, run_directory, execute=crash_after_first, max_concurrency=1)

    resumed = _run(plan, materialized, run_directory)

    assert resumed.completed_episode_ids == (plan.episodes[1].episode_id,)
    assert resumed.skipped_episode_ids == (plan.episodes[0].episode_id,)


def test_resume_refuses_a_different_planned_manifest_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(1)
    run_directory = tmp_path / "run"
    _run(plan, materialized, run_directory)
    changed = ConfirmatoryEpisodePlan(
        episodes=(
            PlannedEpisode(
                stage=plan.episodes[0].stage,
                condition=plan.episodes[0].condition,
                case=plan.episodes[0].case,
                condition_binding=ConditionBinding(
                    domain="access_provisioning",
                    condition=plan.episodes[0].condition,
                    context_contract_hash="sha256:" + "b" * 64,
                    policy_hash=None,
                ),
                repeat_index=plan.episodes[0].repeat_index,
            ),
        )
    )
    dispatched = False

    def materialize(_: object) -> EpisodePlan:
        nonlocal dispatched
        dispatched = True
        return next(iter(materialized.values()))

    with pytest.raises(RunCustodyError, match="exact planned matrix"):
        asyncio.run(
            run_confirmatory_plan(
                changed,
                run_directory=run_directory,
                materialize=materialize,
                client_factory=_client_factory,
            )
        )
    assert not dispatched


def test_resume_refuses_changed_execution_manifest_for_completed_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(1)
    run_directory = tmp_path / "run"
    _run(plan, materialized, run_directory)
    original = materialized[plan.episodes[0].episode_id]
    changed_manifest = original.manifest.model_copy(
        update={
            "budgets": EpisodeBudgets(
                max_turns=original.manifest.budgets.max_turns,
                max_tool_calls=original.manifest.budgets.max_tool_calls,
                max_tokens=original.manifest.budgets.max_tokens + 1,
            )
        }
    )
    materialized[plan.episodes[0].episode_id] = EpisodePlan(
        manifest=changed_manifest,
        task=original.task,
        initial_state=original.initial_state,
        domain_case=original.domain_case,
        authority_decision=original.authority_decision,
        authority_records=original.authority_records,
        authority_query=original.authority_query,
    )

    with pytest.raises(RunCustodyError, match="execution manifest"):
        _run(plan, materialized, run_directory)


def test_resume_revalidates_persisted_result_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(1)
    run_directory = tmp_path / "run"
    _run(plan, materialized, run_directory)
    path = run_directory / "results" / f"{plan.episodes[0].episode_id}.json"
    envelope = json.loads(path.read_bytes())
    result = envelope["result"]
    assert parse_episode_result(result).content_hash == result["content_hash"]
    result["overhead"]["model_calls"] += 1
    projection = episode_result_projection(result)
    content_hash = sha256_ref(projection)
    result["content_hash"] = content_hash
    digest = content_hash.removeprefix("sha256:")
    result["artifact_ref"] = f"artifacts/episode_result/{digest[:2]}/{digest}.json"
    envelope["result_content_hash"] = content_hash
    path.write_bytes(canonical_json_bytes(envelope))

    with pytest.raises(RunCustodyError, match="semantically invalid"):
        _run(plan, materialized, run_directory)


def test_resume_rejects_rehashed_result_with_an_unknown_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(1)
    run_directory = tmp_path / "run"
    _run(plan, materialized, run_directory)
    path = run_directory / "results" / f"{plan.episodes[0].episode_id}.json"
    envelope = json.loads(path.read_bytes())
    result = envelope["result"]
    result["unexpected"] = "tampered"
    projection = episode_result_projection(result)
    content_hash = sha256_ref(projection)
    result["content_hash"] = content_hash
    digest = content_hash.removeprefix("sha256:")
    result["artifact_ref"] = f"artifacts/episode_result/{digest[:2]}/{digest}.json"
    envelope["result_content_hash"] = content_hash
    path.write_bytes(canonical_json_bytes(envelope))

    with pytest.raises(RunCustodyError, match="semantically invalid"):
        _run(plan, materialized, run_directory)


def test_retries_only_transient_provider_failure_before_meaningful_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(1)
    attempts = 0

    async def retry_once(execution_plan: EpisodePlan, client: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelAdapterError(
                code="MODEL_PROVIDER_TRANSIENT",
                attempts=1,
                before_meaningful_behavior=True,
                raw_request_hash=HASH,
                raw_response_hash=None,
            )
        return await run_episode(execution_plan, client)

    summary = _run(
        plan,
        materialized,
        tmp_path / "run",
        execute=retry_once,
        max_technical_retries=1,
    )

    assert attempts == 2
    assert summary.retried_episode_ids == (plan.episodes[0].episode_id,)
    ledger = (tmp_path / "run" / "run-ledger.jsonl").read_text().splitlines()
    assert any(json.loads(line)["event"] == "EPISODE_RETRY" for line in ledger)


def test_default_executor_retries_a_transient_provider_result_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(1)
    factories = 0

    def client_factory(_: object) -> Any:
        nonlocal factories
        factories += 1
        if factories == 1:
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
        return _client(({"kind": "finish", "summary": "Done."},))

    summary = asyncio.run(
        run_confirmatory_plan(
            plan,
            run_directory=tmp_path / "run",
            materialize=_materializer(materialized),
            client_factory=client_factory,
            max_technical_retries=1,
        )
    )

    assert factories == 2
    assert summary.retried_episode_ids == (plan.episodes[0].episode_id,)


@pytest.mark.parametrize("field", ["compiler_manifest_hash", "compiled_skill_artifact_hash"])
def test_materializer_rejects_skill_custody_hash_mismatch_before_client_factory(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base, _ = _plan(condition=ExperimentCondition.B1_FLAT_POLICY_SYSTEM)
    assert base.skill is not None
    assert base.policy is not None
    case = HeldOutCaseBinding(
        domain="access_provisioning",
        case_id=base.task.case_id,
        case_manifest_hash=HASH,
        world_hash=base.manifest.world_hash,
        authority_graph_hash=base.authority_decision.authority_set_hash,
    )
    values = {
        "domain": "access_provisioning",
        "bundle_id": "skill_access_provisioning",
        "contamination_ratio": Decimal("0"),
        "source_manifest_hash": HASH,
        "compiler_manifest_hash": base.skill.compiler_manifest_hash,
        "compiled_skill_artifact_hash": compiled_skill_artifact_hash(base.skill),
        "rendered_skill_hash": base.skill.rendered_skill_hash,
    }
    values[field] = "sha256:" + "b" * 64
    episode = PlannedEpisode(
        stage=EpisodeStage.CONFIRMATORY_B,
        condition=base.manifest.condition,
        case=case,
        condition_binding=ConditionBinding(
            domain="access_provisioning",
            condition=base.manifest.condition,
            context_contract_hash=HASH,
            policy_hash=base.policy.rendered_hash,
        ),
        repeat_index=1,
        skill=SkillBundleBinding(**values),
    )
    materialized = EpisodePlan(
        manifest=base.manifest.model_copy(
            update={"episode_id": episode.episode_id, "stage": episode.stage}
        ),
        task=base.task,
        initial_state=base.initial_state,
        domain_case=base.domain_case,
        authority_decision=base.authority_decision,
        authority_records=base.authority_records,
        authority_query=base.authority_query,
        skill=base.skill,
        policy=base.policy,
    )

    with pytest.raises(RunnerError, match="does not bind"):
        asyncio.run(
            run_confirmatory_plan(
                ConfirmatoryEpisodePlan(episodes=(episode,)),
                run_directory=tmp_path / field,
                materialize=lambda _: materialized,
                client_factory=lambda _: pytest.fail("client factory must not be called"),
            )
        )


def test_concurrency_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(3)
    active = maximum = 0

    class TrackingClient:
        def __init__(self) -> None:
            self._inner = _client(({"kind": "finish", "summary": "Done."},))

        @property
        def capabilities(self) -> Any:
            return self._inner.capabilities

        async def structured(self, request: Any, schema: Any) -> Any:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.01)
                return await self._inner.structured(request, schema)
            finally:
                active -= 1

    summary = asyncio.run(
        run_confirmatory_plan(
            plan,
            run_directory=tmp_path / "run",
            materialize=_materializer(materialized),
            client_factory=lambda _: TrackingClient(),
            max_concurrency=2,
        )
    )

    assert len(summary.completed_episode_ids) == 3
    assert maximum == 2


def test_frozen_exclusion_rule_controls_terminal_retry_budget_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan, materialized = _matrix(1)
    rule = FrozenExclusionRule(
        allowed_error_codes=("MODEL_PROVIDER_TERMINAL",),
        max_retries_per_episode=1,
        max_total_exclusions=1,
        max_exclusions_per_primary_cell=1,
    )
    catalog = FrozenArtifactCatalog(exclusion_rule_hash=rule.rule_hash)
    attempts = 0

    async def terminal_once(execution_plan: EpisodePlan, client: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelAdapterError(
                code="MODEL_PROVIDER_TERMINAL",
                attempts=1,
                before_meaningful_behavior=True,
                raw_request_hash=HASH,
                raw_response_hash=None,
            )
        return await run_episode(execution_plan, client)

    summary = _run(
        plan,
        materialized,
        tmp_path / "run",
        catalog=catalog,
        exclusion_rule=rule,
        max_technical_retries=0,
        execute=terminal_once,
    )

    artifact = json.loads((tmp_path / "run" / "exclusion-rule.json").read_bytes())
    assert attempts == 2
    assert summary.retried_episode_ids == (plan.episodes[0].episode_id,)
    assert artifact == {
        "profile": "SSB-EXCLUSION-RULE2",
        "allowed_error_codes": ["MODEL_PROVIDER_TERMINAL"],
        "max_retries_per_episode": 1,
        "max_total_exclusions": 1,
        "max_exclusions_per_primary_cell": 1,
        "rule_hash": rule.rule_hash,
    }
