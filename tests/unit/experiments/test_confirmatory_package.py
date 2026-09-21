from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.confirmatory import authorize_confirmatory_generation
from shadowskillbench.episodes import AgentTurn, EpisodePlan, EpisodeStage
from shadowskillbench.episodes.pilot_turn_wire import (
    CONFIRMATORY_TURN_PROMPT,
    CONFIRMATORY_TURN_PROMPT_PROFILE,
    LEGACY_V2_CONFIRMATORY_TURN_PROMPT,
    ConfirmatoryActiveTurnWire,
    ConfirmatoryAgentTurnWire,
    ConfirmatoryFinishedTurnWire,
    ConfirmatoryInitialToolTurnWire,
)
from shadowskillbench.experiments.audit import FrozenArtifactCatalog, FrozenExclusionRule
from shadowskillbench.experiments.confirmatory_package import (
    ConfirmatoryExecutionPackageInputs,
    ConfirmatoryPackageHold,
    confirmatory_executor_runtime_profile_projection,
    load_confirmatory_execution_package,
    ollama_gpt_oss_confirmatory_executor_profile,
    write_confirmatory_execution_package,
)
from shadowskillbench.experiments.planner import (
    ConditionBinding,
    ConfirmatoryEpisodePlan,
    HeldOutCaseBinding,
    PlannedEpisode,
)
from shadowskillbench.models.runtime import ModelRunDescriptor
from tests.custody import write_local_custody_receipt
from tests.integration.episodes.test_scripted_executor import HASH, _client, _plan

_LEGACY_COMPACT_WIRE1_PROMPT_HASH = (
    "sha256:23ccd11341af1efb266471444232e8a3820b20c7e502055a5889e9babf210406"
)


def _authorization(root: Path):
    protocol = root / "protocol"
    protocol.mkdir()
    source = Path(cli.__file__).parent / "corpus" / "confirmatory.py"
    frozen_source = root / "src" / "shadowskillbench" / "corpus" / "confirmatory.py"
    frozen_source.parent.mkdir(parents=True)
    frozen_source.write_bytes(source.read_bytes())
    source_hash = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = (
        canonical_json_bytes(
            {
                "anchor_status": "PENDING_HUMAN_ANCHOR",
                "inputs": [
                    {
                        "path": "src/shadowskillbench/corpus/confirmatory.py",
                        "role": "confirmatory_generator",
                        "sha256": source_hash,
                    }
                ],
                "preregistration_core": {
                    "path": "protocol/preregistration.md",
                    "sha256": "sha256:" + "a" * 64,
                },
                "schema_version": "1.0",
            }
        )
        + b"\n"
    )
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    (protocol / "freeze_manifest.json").write_bytes(manifest)
    (protocol / "freeze_manifest.sha256").write_text(f"{manifest_hash}  freeze_manifest.json\n")
    receipt = write_local_custody_receipt(root, manifest)
    return authorize_confirmatory_generation(
        manifest,
        f"{manifest_hash}  freeze_manifest.json\n".encode(),
        receipt,
        repository_root=root,
    )


def _package_inputs(root: Path) -> ConfirmatoryExecutionPackageInputs:
    authorization = _authorization(root)
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
            stage=stage,
            condition=base.manifest.condition,
            case=case,
            condition_binding=binding,
            repeat_index=1,
        )
        for stage in (EpisodeStage.CONFIRMATORY_A, EpisodeStage.CONFIRMATORY_B)
    )
    execution_plans: list[EpisodePlan] = []
    for episode in episodes:
        execution_plans.append(
            EpisodePlan(
                manifest=base.manifest.model_copy(
                    update={"episode_id": episode.episode_id, "stage": episode.stage}
                ),
                task=base.task,
                initial_state=base.initial_state,
                domain_case=base.domain_case,
                authority_decision=base.authority_decision,
                authority_records=base.authority_records,
                authority_query=base.authority_query,
            )
        )
    rule = FrozenExclusionRule(
        allowed_error_codes=("MODEL_PROVIDER_TERMINAL", "MODEL_PROVIDER_TRANSIENT"),
        max_retries_per_episode=1,
        max_total_exclusions=2,
        max_exclusions_per_primary_cell=1,
    )
    catalog = FrozenArtifactCatalog(
        case_manifest_hashes=frozenset({case.case_manifest_hash}),
        context_contract_hashes=frozenset({binding.context_contract_hash}),
        exclusion_rule_hash=rule.rule_hash,
    )
    return ConfirmatoryExecutionPackageInputs(
        authorization=authorization,
        plan=ConfirmatoryEpisodePlan(episodes=episodes),
        catalog=catalog,
        exclusion_rule=rule,
        execution_plans=tuple(execution_plans),
        run_descriptor=ModelRunDescriptor(
            provider="fixture",
            model="scripted-executor",
            model_version="2026-08-24",
            endpoint="http://localhost:9000/v1/chat/completions",
            api_key_environment="SSB_TEST_KEY",
            max_attempts=1,
        ),
    )


def test_native_ollama_executor_profile_is_exact_and_secret_free() -> None:
    profile = ollama_gpt_oss_confirmatory_executor_profile(
        role_profile_receipt_hash="sha256:" + "a" * 64
    )

    assert profile["transport"] == "ollama_native_api_generate_raw"
    assert profile["profile"] == "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR6"
    assert profile["think"] is False
    assert profile["temperature"] == 1.0
    assert profile["top_p"] == 1.0
    assert profile["context_length"] == 131_072
    assert profile["timeout_seconds"] == 900
    assert profile["prompt_profile"] == "SSB-CONFIRMATORY-TURN-WIRE3"
    assert profile["prompt_profile"] == CONFIRMATORY_TURN_PROMPT_PROFILE
    assert profile["prompt_profile_hash"] == (
        "sha256:" + hashlib.sha256(CONFIRMATORY_TURN_PROMPT.encode("utf-8")).hexdigest()
    )
    assert profile["initial_output_schema_hash"] == sha256_ref(
        ConfirmatoryInitialToolTurnWire.model_json_schema()
    )
    assert profile["active_output_schema_hash"] == sha256_ref(
        ConfirmatoryActiveTurnWire.model_json_schema()
    )
    assert profile["finished_output_schema_hash"] == sha256_ref(
        ConfirmatoryFinishedTurnWire.model_json_schema()
    )
    assert profile["output_schema_hash"] == sha256_ref(
        {
            "initial": profile["initial_output_schema_hash"],
            "active": profile["active_output_schema_hash"],
            "finished": profile["finished_output_schema_hash"],
        }
    )
    assert profile["max_turns"] == profile["max_tool_calls"] == 12
    assert profile["max_aggregate_tokens"] == 8192
    assert profile["role_profile_receipt_hash"] == "sha256:" + "a" * 64
    assert confirmatory_executor_runtime_profile_projection(profile) == profile
    changed = {**profile, "temperature": 0.0}
    with pytest.raises(ConfirmatoryPackageHold, match="runtime profile"):
        confirmatory_executor_runtime_profile_projection(changed)

    legacy = {
        **profile,
        "profile": "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR1",
        "request_profile": "SSB-OLLAMA-NATIVE-GENERATE-EXECUTOR-REQUEST1",
        "executor_profile": "SSB-OLLAMA-GPT-OSS-20B-HARMONY-FINAL2",
    }
    from shadowskillbench.experiments.ollama_gpt_oss_profile import (
        OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE_HASH,
    )

    legacy["request_profile_hash"] = (
        OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE_HASH
    )
    legacy["temperature"] = 0.0
    legacy["output_schema_hash"] = sha256_ref(AgentTurn.model_json_schema())
    legacy["prompt_profile"] = "SSB-CONFIRMATORY-AGENT-TURN1"
    legacy["prompt_profile_hash"] = sha256_ref(
        {"profile": "SSB-CONFIRMATORY-AGENT-TURN1", "additional_messages": []}
    )
    for field in (
        "max_turns",
        "max_tool_calls",
        "max_aggregate_tokens",
        "initial_output_schema_hash",
        "active_output_schema_hash",
        "finished_output_schema_hash",
    ):
        del legacy[field]
    assert confirmatory_executor_runtime_profile_projection(legacy) == legacy

    legacy_v3 = {
        **profile,
        "profile": "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR3",
        "temperature": 0.0,
        "output_schema_hash": sha256_ref(ConfirmatoryAgentTurnWire.model_json_schema()),
        "prompt_profile": "SSB-CONFIRMATORY-TURN-WIRE1",
        "prompt_profile_hash": _LEGACY_COMPACT_WIRE1_PROMPT_HASH,
    }
    for field in (
        "initial_output_schema_hash",
        "active_output_schema_hash",
        "finished_output_schema_hash",
    ):
        del legacy_v3[field]
    assert legacy_v3["output_schema_hash"] == sha256_ref(
        ConfirmatoryAgentTurnWire.model_json_schema()
    )
    assert legacy_v3["max_aggregate_tokens"] == 8192
    assert confirmatory_executor_runtime_profile_projection(legacy_v3) == legacy_v3

    legacy_v4 = {
        **profile,
        "profile": "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR4",
        "temperature": 0.0,
        "output_schema_hash": sha256_ref(ConfirmatoryAgentTurnWire.model_json_schema()),
        "prompt_profile": "SSB-CONFIRMATORY-TURN-WIRE2",
        "prompt_profile_hash": "sha256:"
        + hashlib.sha256(LEGACY_V2_CONFIRMATORY_TURN_PROMPT.encode("utf-8")).hexdigest(),
    }
    for field in (
        "initial_output_schema_hash",
        "active_output_schema_hash",
        "finished_output_schema_hash",
    ):
        del legacy_v4[field]
    assert legacy_v4["prompt_profile"] == "SSB-CONFIRMATORY-TURN-WIRE2"
    assert legacy_v4["output_schema_hash"] == sha256_ref(
        ConfirmatoryAgentTurnWire.model_json_schema()
    )
    assert legacy_v4["max_aggregate_tokens"] == 8192
    assert confirmatory_executor_runtime_profile_projection(legacy_v4) == legacy_v4

    legacy_v5 = {
        **legacy_v4,
        "profile": "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR5",
        "temperature": 1.0,
    }
    assert legacy_v5["prompt_profile"] == "SSB-CONFIRMATORY-TURN-WIRE2"
    assert legacy_v5["output_schema_hash"] == sha256_ref(
        ConfirmatoryAgentTurnWire.model_json_schema()
    )
    assert legacy_v5["max_aggregate_tokens"] == 8192
    assert confirmatory_executor_runtime_profile_projection(legacy_v5) == legacy_v5

    legacy_v2 = {
        **profile,
        "profile": "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR2",
        "temperature": 0.0,
        "output_schema_hash": sha256_ref(AgentTurn.model_json_schema()),
        "prompt_profile": "SSB-CONFIRMATORY-AGENT-TURN1",
        "prompt_profile_hash": sha256_ref(
            {"profile": "SSB-CONFIRMATORY-AGENT-TURN1", "additional_messages": []}
        ),
    }
    for field in (
        "max_turns",
        "max_tool_calls",
        "max_aggregate_tokens",
        "initial_output_schema_hash",
        "active_output_schema_hash",
        "finished_output_schema_hash",
    ):
        del legacy_v2[field]
    assert confirmatory_executor_runtime_profile_projection(legacy_v2) == legacy_v2


def test_descriptor_payload_binds_native_executor_profile_hash() -> None:
    from shadowskillbench.experiments import confirmatory_package

    profile = ollama_gpt_oss_confirmatory_executor_profile(
        role_profile_receipt_hash="sha256:" + "b" * 64
    )
    descriptor = ModelRunDescriptor(
        provider="ollama-gpt-oss-20b-harmony-roles",
        model=profile["model"],  # type: ignore[arg-type]
        model_version=profile["model_version"],  # type: ignore[arg-type]
        endpoint="http://127.0.0.1:11435/api/generate",
        api_key_environment="SSB_TEST_KEY",
        max_attempts=1,
        executor_runtime_profile=profile,
    )

    payload = confirmatory_package._descriptor_payload(descriptor)

    assert payload["profile"] == "SSB-MODEL-RUN-DESCRIPTOR2"
    assert payload["executor_runtime_profile"] == profile
    assert payload["content_hash"] == sha256_ref(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )
    assert confirmatory_package._parse_descriptor(payload) == descriptor


def test_current_runtime_budget_binding_rejects_4096_while_legacy_profiles_admit_it(
    tmp_path: Path,
) -> None:
    current = ollama_gpt_oss_confirmatory_executor_profile(
        role_profile_receipt_hash="sha256:" + "c" * 64
    )
    base = _package_inputs(tmp_path)

    def _inputs_for(profile: dict[str, object]) -> ConfirmatoryExecutionPackageInputs:
        plans = tuple(
            plan.model_copy(
                update={
                    "manifest": plan.manifest.model_copy(
                        update={
                            "executor_model": plan.manifest.executor_model.model_copy(
                                update={
                                    "provider": "ollama-gpt-oss-20b-harmony-roles",
                                    "model": profile["model"],
                                    "model_version_date": profile["model_version"],
                                }
                            ),
                            "budgets": plan.manifest.budgets.model_copy(
                                update={"max_tokens": 4096}
                            ),
                        }
                    )
                }
            )
            for plan in base.execution_plans
        )
        return replace(
            base,
            execution_plans=plans,
            run_descriptor=ModelRunDescriptor(
                provider="ollama-gpt-oss-20b-harmony-roles",
                model=profile["model"],  # type: ignore[arg-type]
                model_version=profile["model_version"],  # type: ignore[arg-type]
                endpoint="http://127.0.0.1:11435/api/generate",
                api_key_environment="SSB_TEST_KEY",
                max_attempts=1,
                executor_runtime_profile=profile,
            ),
        )

    with pytest.raises(ConfirmatoryPackageHold, match="executor runtime budgets"):
        write_confirmatory_execution_package(_inputs_for(current), tmp_path / "current")

    legacy_v2 = {
        **current,
        "profile": "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR2",
        "temperature": 0.0,
        "output_schema_hash": sha256_ref(AgentTurn.model_json_schema()),
        "prompt_profile": "SSB-CONFIRMATORY-AGENT-TURN1",
        "prompt_profile_hash": sha256_ref(
            {"profile": "SSB-CONFIRMATORY-AGENT-TURN1", "additional_messages": []}
        ),
    }
    for field in (
        "max_turns",
        "max_tool_calls",
        "max_aggregate_tokens",
        "initial_output_schema_hash",
        "active_output_schema_hash",
        "finished_output_schema_hash",
    ):
        del legacy_v2[field]
    legacy_v1 = {
        **legacy_v2,
        "profile": "SSB-CONFIRMATORY-OLLAMA-GPT-OSS-NATIVE-EXECUTOR1",
        "request_profile": "SSB-OLLAMA-NATIVE-GENERATE-EXECUTOR-REQUEST1",
        "executor_profile": "SSB-OLLAMA-GPT-OSS-20B-HARMONY-FINAL2",
    }
    from shadowskillbench.experiments.ollama_gpt_oss_profile import (
        OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE_HASH,
    )

    legacy_v1["request_profile_hash"] = (
        OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE_HASH
    )
    for name, profile in (("legacy-v1", legacy_v1), ("legacy-v2", legacy_v2)):
        package = write_confirmatory_execution_package(_inputs_for(profile), tmp_path / name)
        assert package.run_descriptor.executor_runtime_profile == profile


def _package_source(inputs: ConfirmatoryExecutionPackageInputs, source_dir: Path) -> Path:
    staged = source_dir.parent / "staged-package"
    write_confirmatory_execution_package(inputs, staged)
    source_dir.mkdir()
    for name in ("plan.json", "catalog.json", "exclusion-rule.json", "run-descriptor.json"):
        (source_dir / name).write_bytes((staged / name).read_bytes())
    (source_dir / "execution-plans").mkdir()
    for path in (staged / "execution-plans").iterdir():
        (source_dir / "execution-plans" / path.name).write_bytes(path.read_bytes())
    return source_dir


def test_package_loader_and_stage_filtered_cli_share_the_unchanged_full_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs = _package_inputs(tmp_path)
    package_dir = tmp_path / "package"
    package = write_confirmatory_execution_package(inputs, package_dir)
    assert (
        load_confirmatory_execution_package(package_dir, authorization=inputs.authorization).plan
        == package.plan
    )

    monkeypatch.setattr(
        cli,
        "client_from_run_descriptor",
        lambda descriptor: _client(({"kind": "finish", "summary": "Done."},)),
    )
    output = tmp_path / "run"
    runner = CliRunner()
    args = [
        "--package-dir",
        str(package_dir),
        "--output-dir",
        str(output),
        "--manifest",
        str(tmp_path / "protocol/freeze_manifest.json"),
        "--manifest-sha256",
        str(tmp_path / "protocol/freeze_manifest.sha256"),
        "--anchor-receipt",
        str(tmp_path / "protocol/anchor_receipt.json"),
        "--endpoint",
        "http://localhost:9000/v1/chat/completions",
        "--api-key-environment",
        "SSB_TEST_KEY",
    ]
    stage_a = runner.invoke(cli.app, ["episodes", "run-confirmatory", "--stage-a", *args])
    runtime_binding = (output / "runtime-binding.json").read_bytes()
    stage_b = runner.invoke(cli.app, ["episodes", "run-confirmatory", "--stage-b", *args])

    assert stage_a.exit_code == 0, stage_a.output
    assert stage_b.exit_code == 0, stage_b.output
    assert len(list((output / "results").glob("*.json"))) == 2
    assert (output / "runtime-binding.json").read_bytes() == runtime_binding
    assert (
        json.loads((output / "run-manifest.json").read_bytes())["plan_hash"]
        == package.plan.plan_hash
    )


def test_cli_builds_sealed_package_from_explicit_source_then_runs_both_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs = _package_inputs(tmp_path)
    source_dir = _package_source(inputs, tmp_path / "source")
    package_dir = tmp_path / "package"
    runner = CliRunner()
    authorization_args = [
        "--manifest",
        str(tmp_path / "protocol/freeze_manifest.json"),
        "--manifest-sha256",
        str(tmp_path / "protocol/freeze_manifest.sha256"),
        "--anchor-receipt",
        str(tmp_path / "protocol/anchor_receipt.json"),
    ]

    built = runner.invoke(
        cli.app,
        [
            "episodes",
            "build-confirmatory-package",
            "--source-dir",
            str(source_dir),
            "--package-dir",
            str(package_dir),
            *authorization_args,
        ],
    )

    assert built.exit_code == 0, built.output
    package = load_confirmatory_execution_package(package_dir, authorization=inputs.authorization)
    assert package.plan == inputs.plan
    monkeypatch.setattr(
        cli,
        "client_from_run_descriptor",
        lambda descriptor: _client(({"kind": "finish", "summary": "Done."},)),
    )
    run_args = [
        "--package-dir",
        str(package_dir),
        "--output-dir",
        str(tmp_path / "run"),
        "--endpoint",
        "http://localhost:9000/v1/chat/completions",
        "--api-key-environment",
        "SSB_TEST_KEY",
        *authorization_args,
    ]
    stage_a = runner.invoke(cli.app, ["episodes", "run-confirmatory", "--stage-a", *run_args])
    stage_b = runner.invoke(cli.app, ["episodes", "run-confirmatory", "--stage-b", *run_args])

    assert stage_a.exit_code == 0, stage_a.output
    assert stage_b.exit_code == 0, stage_b.output
    assert len(list((tmp_path / "run/results").glob("*.json"))) == 2


def test_cli_build_holds_for_tampered_source_before_creating_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _package_inputs(tmp_path)
    source_dir = _package_source(inputs, tmp_path / "source")
    plan_path = source_dir / "plan.json"
    plan_path.write_bytes(plan_path.read_bytes() + b" ")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "episodes",
            "build-confirmatory-package",
            "--source-dir",
            str(source_dir),
            "--package-dir",
            str(tmp_path / "package"),
            "--manifest",
            str(tmp_path / "protocol/freeze_manifest.json"),
            "--manifest-sha256",
            str(tmp_path / "protocol/freeze_manifest.sha256"),
            "--anchor-receipt",
            str(tmp_path / "protocol/anchor_receipt.json"),
        ],
    )

    assert result.exit_code == 1
    assert "HOLD_CONFIRMATORY_PACKAGE_BUILD" in result.output
    assert not (tmp_path / "package").exists()


def test_cli_build_holds_for_symlinked_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _package_inputs(tmp_path)
    source_dir = _package_source(inputs, tmp_path / "source")
    linked_source = tmp_path / "linked-source"
    linked_source.symlink_to(source_dir, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "episodes",
            "build-confirmatory-package",
            "--source-dir",
            str(linked_source),
            "--package-dir",
            str(tmp_path / "package"),
            "--manifest",
            str(tmp_path / "protocol/freeze_manifest.json"),
            "--manifest-sha256",
            str(tmp_path / "protocol/freeze_manifest.sha256"),
            "--anchor-receipt",
            str(tmp_path / "protocol/anchor_receipt.json"),
        ],
    )

    assert result.exit_code == 1
    assert "HOLD_CONFIRMATORY_PACKAGE_BUILD" in result.output
    assert not (tmp_path / "package").exists()


def test_package_loader_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    inputs = _package_inputs(tmp_path)
    custody = tmp_path / "custody"
    custody.mkdir()
    package_dir = custody / "package"
    write_confirmatory_execution_package(inputs, package_dir)
    alias = tmp_path / "alias"
    alias.symlink_to(custody, target_is_directory=True)

    with pytest.raises(ConfirmatoryPackageHold, match="unsafe custody path"):
        load_confirmatory_execution_package(alias / "package", authorization=inputs.authorization)


def test_cli_build_refuses_to_overwrite_existing_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _package_inputs(tmp_path)
    source_dir = _package_source(inputs, tmp_path / "source")
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    marker = package_dir / "marker"
    marker.write_text("preserve")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "episodes",
            "build-confirmatory-package",
            "--source-dir",
            str(source_dir),
            "--package-dir",
            str(package_dir),
            "--manifest",
            str(tmp_path / "protocol/freeze_manifest.json"),
            "--manifest-sha256",
            str(tmp_path / "protocol/freeze_manifest.sha256"),
            "--anchor-receipt",
            str(tmp_path / "protocol/anchor_receipt.json"),
        ],
    )

    assert result.exit_code == 1
    assert "HOLD_CONFIRMATORY_PACKAGE_BUILD" in result.output
    assert marker.read_text() == "preserve"


def test_unauthorized_cli_never_constructs_a_runtime_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _package_inputs(tmp_path)
    inputs = replace(
        inputs,
        run_descriptor=replace(
            inputs.run_descriptor,
            endpoint="https://attacker.example.invalid/exfiltrate",
            api_key_environment="AWS_SESSION_TOKEN",
        ),
    )
    package_dir = tmp_path / "package"
    write_confirmatory_execution_package(inputs, package_dir)
    monkeypatch.chdir(tmp_path)
    constructed = False

    def forbidden(_: ModelRunDescriptor):
        nonlocal constructed
        constructed = True
        raise AssertionError("client construction must follow anchor admission")

    monkeypatch.setattr(cli, "client_from_run_descriptor", forbidden)
    result = CliRunner().invoke(
        cli.app,
        [
            "episodes",
            "run-confirmatory",
            "--stage-a",
            "--package-dir",
            str(package_dir),
            "--output-dir",
            str(tmp_path / "run"),
            "--manifest",
            str(tmp_path / "protocol/freeze_manifest.json"),
            "--manifest-sha256",
            str(tmp_path / "protocol/freeze_manifest.sha256"),
            "--anchor-receipt",
            str(tmp_path / "protocol/anchor_receipt.json"),
        ],
    )

    assert result.exit_code == 1
    assert "operator must explicitly supply --endpoint and --api-key-environment" in result.output
    assert not constructed


def test_cli_rejects_a_symlinked_output_root_before_client_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _package_inputs(tmp_path)
    package_dir = tmp_path / "package"
    write_confirmatory_execution_package(inputs, package_dir)
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "linked-output"
    output.symlink_to(target, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    constructed = False

    def forbidden(_: ModelRunDescriptor):
        nonlocal constructed
        constructed = True
        raise AssertionError("unsafe output must fail before credential access")

    monkeypatch.setattr(cli, "client_from_run_descriptor", forbidden)
    result = CliRunner().invoke(
        cli.app,
        [
            "episodes",
            "run-confirmatory",
            "--stage-a",
            "--package-dir",
            str(package_dir),
            "--output-dir",
            str(output),
            "--endpoint",
            "http://localhost:9000/v1/chat/completions",
            "--api-key-environment",
            "SSB_TEST_KEY",
            "--manifest",
            str(tmp_path / "protocol/freeze_manifest.json"),
            "--manifest-sha256",
            str(tmp_path / "protocol/freeze_manifest.sha256"),
            "--anchor-receipt",
            str(tmp_path / "protocol/anchor_receipt.json"),
        ],
    )

    assert result.exit_code == 1
    assert "output path contains a symlink" in result.output
    assert not constructed
