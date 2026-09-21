from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from shadowskillbench.analysis import DatasetInputError, EpisodeObservation, build_dataset
from shadowskillbench.analysis.dataset import _validate_planned_episode
from shadowskillbench.authority import (
    AuthoritySourceType,
    NormativeStatus,
    create_access_authority_query,
    create_authority_record,
    create_issuer_registry,
)
from shadowskillbench.authority.resolver import resolve_authority
from shadowskillbench.domains.access.fixtures import build_access_fixture
from shadowskillbench.engine import TaskCase, hash_state
from shadowskillbench.episodes import (
    EpisodeBudgets,
    EpisodeManifest,
    EpisodePlan,
    EpisodeStage,
    EpisodeStatus,
    ExecutorModel,
    ExperimentCondition,
    PromptHashes,
    run_episode,
)
from shadowskillbench.experiments.planner import (
    ConditionBinding,
    HeldOutCaseBinding,
    PlannedEpisode,
    SkillBundleBinding,
)
from shadowskillbench.metrics import score_episode
from shadowskillbench.models import ProviderCapabilities, ScriptedModelClient, TransportResponse
from shadowskillbench.skills.compiler import compiled_skill_artifact_hash
from tests.unit.episodes.test_context import _skill

HASH = "sha256:" + "a" * 64


def _decision(fixture: Any) -> tuple[tuple[Any, ...], Any, Any]:
    request = fixture.initial_world.data["access_requests"][0]
    record = create_authority_record(
        authority_id="dataset_rule",
        schema_version="1.0",
        source_type=AuthoritySourceType.POLICY,
        title="dataset_rule",
        issuer_id="issuer_policy",
        issuer_role="policy_owner",
        authority_rank=10,
        action_type="grant_access",
        scope={
            "domain": "access_provisioning",
            "scope_kind": "rule",
            "subject_id": request["employee_id"],
            "resource_id": request["application_id"],
            "organization_id": None,
            "geography_id": None,
            "role_id": request["requested_role"],
            "rule_disposition": "PROCEED",
            "allowed_action": "grant_access",
            "requires_security_approval": False,
            "role_derived_without_approval": True,
        },
        effective_at="2026-08-01T00:00:00Z",
        expires_at=None,
        supersedes=[],
        exception_to=[],
        provenance_locator="authority/dataset_rule",
        normative_status=NormativeStatus.ACTIVE_AUTHORITY,
    )
    registry = create_issuer_registry(
        authorizations=[
            {
                "issuer_id": "issuer_policy",
                "issuer_role": "policy_owner",
                "authority_rank": 10,
                "allowed_domains": ["access_provisioning"],
                "allowed_source_types": ["POLICY"],
                "can_supersede": False,
                "can_issue_scoped_evidence": False,
            }
        ]
    )
    query = create_access_authority_query(
        subject_id=request["employee_id"],
        resource_id=request["application_id"],
        action_type="grant_access",
        at_time="2026-08-01T12:00:00Z",
        organization_id=None,
        geography_id=None,
        reference_authority_ids=[],
        issuer_registry=registry,
        role_id=request["requested_role"],
        role_derived_access=True,
    )
    return (record,), query, resolve_authority((record,), query)


def _body(value: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "model": "dataset-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(value)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
        separators=(",", ":"),
    ).encode()


def _observation(
    tmp_path: Path, *, repeat_index: int, outcome: str = "complete"
) -> EpisodeObservation:
    fixture = build_access_fixture("role_derived", seed=811)
    task = TaskCase(
        case_id=fixture.case.case_id,
        domain="access_provisioning",
        world_id=fixture.case.world_id,
        seed=fixture.case.seed,
        objective="Review access.",
        inputs={"request_id": fixture.case.target_request_id},
    )
    records, query, decision = _decision(fixture)
    planned = PlannedEpisode(
        stage=EpisodeStage.CONFIRMATORY_A,
        condition=ExperimentCondition.A0_BARE,
        case=HeldOutCaseBinding(
            domain="access_provisioning",
            case_id=task.case_id,
            case_manifest_hash=HASH,
            world_hash=hash_state(fixture.initial_world),
            authority_graph_hash=decision.authority_set_hash,
        ),
        condition_binding=ConditionBinding(
            domain="access_provisioning",
            condition=ExperimentCondition.A0_BARE,
            context_contract_hash=HASH,
            policy_hash=None,
        ),
        repeat_index=repeat_index,
    )
    plan = EpisodePlan(
        manifest=EpisodeManifest(
            episode_id=planned.episode_id,
            schema_version="1.0",
            stage=EpisodeStage.CONFIRMATORY_A.value,  # pyright: ignore[reportArgumentType]
            condition=ExperimentCondition.A0_BARE.value,  # pyright: ignore[reportArgumentType]
            domain="access_provisioning",
            case_id=task.case_id,
            world_hash=hash_state(fixture.initial_world),
            executor_model=ExecutorModel(
                provider="fixture", model="dataset-test", model_version_date="2026-08-24"
            ),
            prompt_hashes=PromptHashes(system=HASH, developer=HASH, user=HASH),
            policy_hash=None,
            skill_hash=None,
            authority_graph_hash=decision.authority_set_hash,
            seed=task.seed,
            budgets=EpisodeBudgets(max_turns=4, max_tool_calls=4, max_tokens=100),
            status=EpisodeStatus.PLANNED.value,  # pyright: ignore[reportArgumentType]
            artifact_refs=(),
        ),
        task=task,
        initial_state=fixture.initial_world,
        domain_case=fixture.case,
        authority_records=records,
        authority_query=query,
        authority_decision=decision,
    )
    if outcome == "complete":
        output = (
            {
                "kind": "tool_call",
                "action": {
                    "action_id": "grant",
                    "tool_name": "grant_access",
                    "arguments": {"request_id": fixture.case.target_request_id},
                },
            },
            {
                "kind": "tool_call",
                "action": {
                    "action_id": "finish_tool",
                    "tool_name": "finish_task",
                    "arguments": {"summary": "done"},
                },
            },
            {"kind": "finish", "summary": "done"},
        )
        responses = tuple(TransportResponse(status_code=200, body=_body(item)) for item in output)
    elif outcome == "refuse":
        refusal = {"kind": "refuse", "summary": "no"}
        responses = (TransportResponse(status_code=200, body=_body(refusal)),)
    else:
        responses = (TransportResponse(status_code=503, body=b"unavailable"),)
    client = ScriptedModelClient(
        capabilities=ProviderCapabilities(
            provider="fixture",
            model="dataset-test",
            model_version="2026-08-24",
            supports_system_role=True,
            supports_developer_role=True,
            supports_seed=True,
            supports_structured_output=True,
        ),
        script=responses,
        max_attempts=1,
    )
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = asyncio.run(run_episode(plan, client))
    finally:
        os.chdir(cwd)
    return EpisodeObservation(
        plan=plan,
        result=result,
        score=score_episode(result, task),
        planned_episode=planned,
    )


def _aggregates(dataset: object) -> dict[str, object]:
    return {item.metric: item for item in getattr(dataset, "aggregates")}


def test_builds_sorted_rows_and_exact_raw_metric_totals(tmp_path: Path) -> None:
    complete = _observation(tmp_path, repeat_index=1)
    refused = _observation(tmp_path, repeat_index=2, outcome="refuse")

    dataset = build_dataset((refused, complete))

    assert [row.repeat_index for row in dataset.rows] == [1, 2]
    assert dataset.exclusions == ()
    aggregates = _aggregates(dataset)
    completion = aggregates["completion_under_policy"]
    false_enforcement = aggregates["false_enforcement"]
    escalation = aggregates["escalation_correct"]
    assert (completion.numerator, completion.denominator) == (1, 2)
    assert (false_enforcement.numerator, false_enforcement.denominator) == (1, 2)
    assert false_enforcement.denominator_scope == "expected_proceed"
    assert (escalation.numerator, escalation.denominator) == (0, 0)
    assert escalation.denominator_scope == "escalation_expected"


def test_keeps_only_technical_failures_in_the_exclusion_table(tmp_path: Path) -> None:
    excluded = _observation(tmp_path, repeat_index=3, outcome="provider_failure")

    dataset = build_dataset((excluded,))

    assert dataset.rows == ()
    assert len(dataset.exclusions) == 1
    assert dataset.exclusions[0].error_code == "MODEL_PROVIDER_TRANSIENT"
    assert dataset.aggregates == ()


def test_rejects_duplicate_cells_and_mismatched_custody(tmp_path: Path) -> None:
    first = _observation(tmp_path, repeat_index=1)
    second = _observation(tmp_path, repeat_index=2, outcome="refuse")

    with pytest.raises(DatasetInputError, match="duplicate"):
        build_dataset((first, first))
    with pytest.raises(DatasetInputError, match="score does not bind"):
        build_dataset(
            (
                EpisodeObservation(
                    plan=first.plan,
                    result=first.result,
                    score=second.score,
                    planned_episode=first.planned_episode,
                ),
            )
        )
    with pytest.raises(DatasetInputError, match="planned episode does not bind"):
        build_dataset(
            (
                EpisodeObservation(
                    plan=first.plan,
                    result=first.result,
                    score=first.score,
                    planned_episode=second.planned_episode,
                ),
            )
        )


def test_rejects_skill_artifact_hash_mismatch_despite_matching_rendered_text(
    tmp_path: Path,
) -> None:
    base = _observation(tmp_path, repeat_index=1)
    artifact = _skill()
    bundle_id = "bundle_dataset"
    manifest = base.plan.manifest.model_copy(
        update={
            "condition": ExperimentCondition.A2_SKILL_ONLY,
            "skill_bundle_id": bundle_id,
            "contamination_ratio": 0.5,
            "skill_hash": artifact.rendered_skill_hash,
        }
    )
    planned = PlannedEpisode(
        stage=EpisodeStage.CONFIRMATORY_A,
        condition=ExperimentCondition.A2_SKILL_ONLY,
        case=base.planned_episode.case,
        condition_binding=ConditionBinding(
            domain="access_provisioning",
            condition=ExperimentCondition.A2_SKILL_ONLY,
            context_contract_hash=HASH,
            policy_hash=None,
        ),
        repeat_index=1,
        skill=SkillBundleBinding(
            domain="access_provisioning",
            bundle_id=bundle_id,
            contamination_ratio=Decimal("0.5"),
            source_manifest_hash=HASH,
            compiler_manifest_hash=artifact.compiler_manifest_hash,
            compiled_skill_artifact_hash="sha256:" + "b" * 64,
            rendered_skill_hash=artifact.rendered_skill_hash,
        ),
    )
    manifest = manifest.model_copy(update={"episode_id": planned.episode_id})
    plan = base.plan.model_copy(update={"manifest": manifest, "skill": artifact})

    assert planned.skill is not None
    assert planned.skill.compiled_skill_artifact_hash != compiled_skill_artifact_hash(artifact)
    with pytest.raises(DatasetInputError, match="planned skill"):
        _validate_planned_episode(planned, plan)


def test_dataset_order_is_deterministic(tmp_path: Path) -> None:
    first = _observation(tmp_path, repeat_index=1)
    second = _observation(tmp_path, repeat_index=2, outcome="refuse")

    assert build_dataset((first, second)) == build_dataset((second, first))
