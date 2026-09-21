from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import shadowskillbench.corpus.confirmatory_v4 as confirmatory_v4
from shadowskillbench import cli
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.confirmatory_v4 import (
    ConfirmatoryAuthorizationV4,
    authorization_projection_v4,
)
from shadowskillbench.episodes import EpisodePlan, EpisodeStage
from shadowskillbench.experiments import confirmatory_preparation_v4
from shadowskillbench.experiments.confirmatory_execution_v4 import (
    V4_RESULT_PROFILE,
    V4_RUNTIME_BINDING_PROFILE,
    parse_confirmatory_run_bundle_result_v4,
    run_confirmatory_episode_v4,
)
from shadowskillbench.experiments.confirmatory_package_v4 import (
    V4_PACKAGE_PROFILE,
    V4_RESULT_NAMESPACE,
    V4_RUN_DESCRIPTOR_PROFILE,
    ConfirmatoryExecutionPackageV4,
    ConfirmatoryPackageV4Hold,
    load_confirmatory_execution_package_v4,
)
from shadowskillbench.experiments.confirmatory_preparation_v4 import (
    V4_PACKAGE_NAMESPACE,
    V4_RUNTIME_NAMESPACE,
)
from shadowskillbench.experiments.planner import (
    ConditionBinding,
    HeldOutCaseBinding,
    PlannedEpisode,
)
from shadowskillbench.experiments.planner_v4 import (
    STAGE_A_EPISODE_COUNT,
    STAGE_B_EPISODE_COUNT,
)
from shadowskillbench.models.runtime import ModelRunDescriptor
from tests.integration.episodes.test_scripted_executor import HASH, _client, _plan


def _authorization() -> ConfirmatoryAuthorizationV4:
    return ConfirmatoryAuthorizationV4._issue(
        freeze_manifest_hash="sha256:" + "a" * 64,
        anchor_receipt_hash="sha256:" + "b" * 64,
        issuer=confirmatory_v4._AUTHORIZATION_ISSUER,
    )


def _descriptor() -> ModelRunDescriptor:
    return ModelRunDescriptor(
        provider="fixture",
        model="scripted-executor",
        model_version="2026-08-24",
        endpoint="http://localhost:9000/v1/chat/completions",
        api_key_environment="SSB_TEST_KEY",
        max_attempts=1,
        executor_runtime_profile={"profile": "SSB-TEST-EXECUTOR4"},
    )


def _descriptor_manifest(descriptor: ModelRunDescriptor) -> dict[str, object]:
    body = {
        "profile": V4_RUN_DESCRIPTOR_PROFILE,
        "provider": descriptor.provider,
        "model": descriptor.model,
        "model_version": descriptor.model_version,
        "endpoint": descriptor.endpoint,
        "api_key_environment": descriptor.api_key_environment,
        "max_attempts": descriptor.max_attempts,
        "supports_system_role": descriptor.supports_system_role,
        "supports_developer_role": descriptor.supports_developer_role,
        "supports_seed": descriptor.supports_seed,
        "supports_structured_output": descriptor.supports_structured_output,
        "pricing": None,
        "executor_runtime_profile": descriptor.executor_runtime_profile,
    }
    return {**body, "content_hash": sha256_ref(body)}


def _manifest(authorization: ConfirmatoryAuthorizationV4) -> dict[str, object]:
    episodes = {
        f"episode_{index:05d}": f"sha256:{index:064x}"
        for index in range(STAGE_A_EPISODE_COUNT + STAGE_B_EPISODE_COUNT)
    }
    descriptor = _descriptor_manifest(_descriptor())
    corpus_projection = {
        "schema_version": "4.0",
        "freeze_manifest_hash": authorization.freeze_manifest_hash,
        "anchor_receipt_hash": authorization.anchor_receipt_hash,
        "corpus_seed": 104733,
        "bundle_hashes": [],
        "stage_a_hidden_content_hashes": [],
        "stage_b_hidden_content_hashes": [],
    }
    plan_projection = {
        "profile": "SSB-PLAN1",
        "episode_manifest_hashes": list(episodes.values()),
    }
    value: dict[str, object] = {
        "profile": V4_PACKAGE_PROFILE,
        "schema_version": "4.0",
        "package_namespace": V4_PACKAGE_NAMESPACE,
        "runtime_namespace": V4_RUNTIME_NAMESPACE,
        "result_namespace": V4_RESULT_NAMESPACE,
        "authorization": authorization_projection_v4(authorization),
        "corpus_hash": sha256_ref(corpus_projection),
        "corpus_projection": corpus_projection,
        "corpus_seed": 104733,
        "development_seeds": [4242, 4243],
        "plan_hash": sha256_ref(plan_projection),
        "plan_projection": plan_projection,
        "run_descriptor": descriptor,
        "run_descriptor_hash": descriptor["content_hash"],
        "stage_a_episode_count": STAGE_A_EPISODE_COUNT,
        "stage_b_episode_count": STAGE_B_EPISODE_COUNT,
        "stage_a_episode_manifest_hashes": {
            key: value for key, value in list(episodes.items())[:STAGE_A_EPISODE_COUNT]
        },
        "stage_b_episode_manifest_hashes": {
            key: value for key, value in list(episodes.items())[STAGE_A_EPISODE_COUNT:]
        },
    }
    return {**value, "package_hash": sha256_ref(value)}


def test_v4_loader_rejects_legacy_manifest_and_requires_exact_v4_counts(tmp_path: Path) -> None:
    authorization = _authorization()
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    legacy = {"profile": "SSB-CONFIRMATORY-EXECUTION-PACKAGE2"}
    (package_dir / "package-manifest.v4.json").write_bytes(canonical_json_bytes(legacy))

    with pytest.raises(ConfirmatoryPackageV4Hold, match="not V4"):
        load_confirmatory_execution_package_v4(package_dir, authorization=authorization)

    manifest = _manifest(authorization)
    (package_dir / "package-manifest.v4.json").write_bytes(canonical_json_bytes(manifest))
    package = load_confirmatory_execution_package_v4(package_dir, authorization=authorization)

    assert len(package.episode_manifest_hashes) == STAGE_A_EPISODE_COUNT + STAGE_B_EPISODE_COUNT
    assert package.result_namespace == V4_RESULT_NAMESPACE


def test_v4_production_preparation_has_no_fabricated_local_binding_route() -> None:
    assert not hasattr(confirmatory_preparation_v4, "stage_confirmatory_execution_v4_local")


def test_v4_router_uses_descriptor_matched_client_and_writes_v4_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    planned = PlannedEpisode(
        stage=EpisodeStage.CONFIRMATORY_A,
        condition=base.manifest.condition,
        case=case,
        condition_binding=binding,
        repeat_index=1,
    )
    execution_plan = EpisodePlan(
        manifest=base.manifest.model_copy(
            update={"episode_id": planned.episode_id, "stage": EpisodeStage.CONFIRMATORY_A}
        ),
        task=base.task,
        initial_state=base.initial_state,
        domain_case=base.domain_case,
        authority_decision=base.authority_decision,
        authority_records=base.authority_records,
        authority_query=base.authority_query,
    )
    authorization = _authorization()
    package = ConfirmatoryExecutionPackageV4(
        authorization,
        "sha256:" + "c" * 64,
        "sha256:" + "d" * 64,
        _descriptor(),
        {planned.episode_id: planned.manifest_hash},
        "sha256:" + "e" * 64,
        {
            "profile": V4_PACKAGE_PROFILE,
            "runtime_namespace": V4_RUNTIME_NAMESPACE,
            "result_namespace": V4_RESULT_NAMESPACE,
            "package_hash": "sha256:" + "e" * 64,
            "run_descriptor_hash": "sha256:" + "f" * 64,
        },
    )
    monkeypatch.chdir(tmp_path)

    routed = asyncio.run(
        run_confirmatory_episode_v4(
            package=package,
            planned_episode=planned,
            execution_plan=execution_plan,
            client=_client(({"kind": "finish", "summary": "finished locally"},)),
            result_dir=tmp_path / "v4-results",
        )
    )

    assert routed.result_ref.startswith(V4_RESULT_NAMESPACE + "/")
    assert routed.runtime_binding["profile"] == V4_RUNTIME_BINDING_PROFILE
    payload = next((tmp_path / "v4-results").iterdir()).read_bytes()
    assert V4_RESULT_PROFILE.encode() in payload
    assert parse_confirmatory_run_bundle_result_v4(json.loads(payload)) == routed.result

    with pytest.raises(ConfirmatoryPackageV4Hold, match="capabilities"):
        asyncio.run(
            run_confirmatory_episode_v4(
                package=package,
                planned_episode=planned,
                execution_plan=execution_plan,
                client=_client(({"kind": "finish", "summary": "wrong client"},), model="other"),
                result_dir=tmp_path / "v4-results",
            )
        )


@pytest.mark.parametrize("command", ("build-confirmatory-package-v4", "run-confirmatory-v4"))
def test_v4_cli_never_reports_a_fake_package_or_execution_pass(command: str) -> None:
    result = CliRunner().invoke(cli.app, ["episodes", command])

    assert result.exit_code == 1
    assert "HOLD_CONFIRMATORY_V4_" in result.output
    assert "PASS_CONFIRMATORY_V4" not in result.output
