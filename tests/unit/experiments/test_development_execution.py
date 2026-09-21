from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from functools import cache
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.corpus.development import generate_development_corpus
from shadowskillbench.episodes import parse_episode_result
from shadowskillbench.experiments import development_execution
from shadowskillbench.experiments.development_execution import (
    development_model_seed,
    materialize_development_episode,
    materialize_development_episodes,
    scripted_development_response,
    visible_execution_input,
)
from shadowskillbench.experiments.development_plan import (
    DevelopmentPlannedEpisode,
    build_development_plan,
)
from shadowskillbench.experiments.runner import (
    RunCustodyError,
    audit_development_runs,
    run_development_plan,
)
from shadowskillbench.models import ProviderCapabilities, TokenUsage
from shadowskillbench.skills import SkillIR, hash_rendered_skill, hash_skill_ir, render_skill
from shadowskillbench.skills.compiler import (
    CompiledSkillArtifact,
    CompilerManifest,
    compiler_manifest_hash,
)
from shadowskillbench.skills.projection import compiler_view, hash_compiler_input
from shadowskillbench.traces.bundles import generate_bundle
from tests.integration.episodes.test_scripted_executor import _plan


@cache
def _materialization_skill() -> CompiledSkillArtifact:
    hash_ref = sha256_ref(["development-materialization-test"])
    capabilities = ProviderCapabilities(
        provider="scripted",
        model="development-materialization-test",
        model_version="D-001",
        supports_system_role=True,
        supports_developer_role=False,
        supports_seed=True,
        supports_structured_output=True,
    )
    manifest = CompilerManifest(
        manifest_profile="SSB-COMPILER-MANIFEST1",
        compiler_input_hash=hash_ref,
        prompt_profile="SSB-SKILL-COMPILER1",
        system_prompt_raw_hash=hash_ref,
        request_profile="SSB-SKILL-COMPILER-REQUEST1",
        instruction_provenance_profile="SSB-INSTRUCTION-PROVENANCE1",
        structured_output_schema_hash=sha256_ref(SkillIR.model_json_schema()),
        declared_capabilities=capabilities,
        temperature=0.0,
        seed=4242,
        max_tokens=1,
    )
    manifest_hash = compiler_manifest_hash(manifest)
    skill = SkillIR.model_validate(
        {
            "skill_id": "skill_materialization_test",
            "schema_version": "1.0",
            "domain": "access_provisioning",
            "objective": "Test materialization.",
            "applicability": [],
            "required_inputs": [],
            "preconditions": [],
            "ordered_steps": [
                {
                    "step_id": "step_001",
                    "action_intent": "Read the visible task.",
                    "tool_name": "get_access_request",
                    "argument_bindings": {"request_id": "visible"},
                    "preconditions": [],
                    "optional": False,
                    "evidence_refs": ["event_001"],
                }
            ],
            "decision_hints": [],
            "verification_steps": [],
            "stop_conditions": [],
            "escalation_hints": [],
            "source_trace_ids": ["trace_00000000000000000000000000000000"],
            "instruction_provenance": {
                "profile": "SSB-INSTRUCTION-PROVENANCE1",
                "compiler_input_hash": hash_ref,
                "instruction_evidence": {
                    "/objective": ["event_001"],
                    "/ordered_steps/0": ["event_001"],
                },
            },
            "compiler_manifest_ref": manifest_hash,
        }
    )
    rendered = render_skill(skill)
    return CompiledSkillArtifact(
        artifact_profile="SSB-COMPILED-SKILL1",
        compiler_manifest=manifest,
        compiler_manifest_hash=manifest_hash,
        request_envelope_hash=hash_ref,
        skill_ir=skill,
        skill_ir_hash=hash_skill_ir(skill),
        rendered_skill=rendered,
        rendered_skill_hash=hash_rendered_skill(skill),
        raw_request_hash=hash_ref,
        raw_response_hash=hash_ref,
        structured_output_schema_hash=manifest.structured_output_schema_hash,
        usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        cost=None,
        attempts=1,
    )


def test_visible_script_is_deterministic_and_contains_no_hidden_truth() -> None:
    plan, fixture = _plan()
    visible = visible_execution_input(plan)
    first = scripted_development_response(visible)
    second = scripted_development_response(visible)
    hidden = plan.authority_records[0].authority_id

    assert first == second
    assert hidden.encode() not in first
    assert fixture.case.case_id.encode() not in first


def test_development_materialization_retains_hidden_truth_outside_visible_input() -> None:
    corpus = generate_development_corpus(4242)
    planned = build_development_plan(corpus).episodes[:2]
    execution_plans = materialize_development_episodes(corpus, planned)

    for episode, execution_plan in zip(planned, execution_plans, strict=True):
        case = next(case for case in corpus.cases if case.case_id == episode.case_id)
        visible = json.dumps(visible_execution_input(execution_plan).projection(), sort_keys=True)

        assert execution_plan.authority_records == case.hidden_truth.authority_records
        assert execution_plan.authority_query == case.hidden_truth.authority_query
        assert (
            execution_plan.authority_decision.decision_hash
            == case.hidden_truth.authority_decision_hash
        )
        assert case.hidden_truth.authority_records[0].authority_id not in visible
        assert case.hidden_truth.expected_decision.value not in visible


def test_development_materializes_a3_with_a_strict_block_order() -> None:
    corpus = generate_development_corpus(4242)
    planned = next(
        episode
        for episode in build_development_plan(corpus).episodes
        if episode.condition.value == "A3_SKILL_POLICY_SAME_TIER"
    )

    execution_plan = materialize_development_episode(corpus, planned)

    assert execution_plan.order_assignment is not None
    assert execution_plan.order_assignment.value in {
        "policy_then_skill",
        "skill_then_policy",
    }


def test_development_executor_budget_defaults_to_4096_and_allows_8192_override() -> None:
    corpus = generate_development_corpus(4242)
    planned = build_development_plan(corpus).episodes[0]

    default = materialize_development_episode(corpus, planned)
    expanded = materialize_development_episode(corpus, planned, max_tokens=8192)

    assert default.manifest.budgets.max_tokens == 4096
    assert expanded.manifest.budgets.max_tokens == 8192


@pytest.mark.parametrize("max_tokens", (0, -1, True, 4096.0))
def test_development_executor_budget_rejects_invalid_override(max_tokens: object) -> None:
    corpus = generate_development_corpus(4242)
    planned = build_development_plan(corpus).episodes[0]

    with pytest.raises(ValueError, match="max_tokens"):
        materialize_development_episode(corpus, planned, max_tokens=max_tokens)  # type: ignore[arg-type]


def test_development_materialization_defaults_to_4242_and_admits_explicit_4243() -> None:
    legacy_corpus = generate_development_corpus(4242)
    legacy_episode = build_development_plan(legacy_corpus).episodes[0]
    fresh_corpus = generate_development_corpus(4243)
    fresh_episode = build_development_plan(fresh_corpus).episodes[0]

    assert materialize_development_episode(legacy_corpus, legacy_episode).manifest.case_id
    with pytest.raises(ValueError, match="corpus_seed"):
        materialize_development_episode(fresh_corpus, fresh_episode)
    assert materialize_development_episode(
        fresh_corpus, fresh_episode, corpus_seed=4243
    ).manifest.case_id


def test_development_skill_materialization_binds_bundle_and_compiler_input_to_corpus_seed() -> None:
    def skill_episode(seed: int) -> DevelopmentPlannedEpisode:
        corpus = generate_development_corpus(seed)
        return next(
            episode
            for episode in build_development_plan(corpus).episodes
            if episode.condition.requires_skill
        )

    def expected_bundle(episode: DevelopmentPlannedEpisode, seed: int):
        ratios = (
            Decimal("0"),
            Decimal("0.25"),
            Decimal("0.5"),
            Decimal("0.75"),
            Decimal("1"),
        )
        matches = tuple(
            generated.bundle
            for ratio in ratios
            if (
                generated := generate_bundle(episode.domain, ratio, 12, seed)
            ).bundle.source_manifest.bundle_id
            == episode.bundle_id
        )
        assert len(matches) == 1
        return matches[0]

    fresh_corpus = generate_development_corpus(4243)
    fresh_episode = skill_episode(4243)
    fresh = materialize_development_episode(fresh_corpus, fresh_episode, corpus_seed=4243)
    fresh_bundle = expected_bundle(fresh_episode, 4243)

    legacy_corpus = generate_development_corpus(4242)
    legacy_episode = skill_episode(4242)
    legacy_default = materialize_development_episode(legacy_corpus, legacy_episode)
    legacy_explicit = materialize_development_episode(
        legacy_corpus, legacy_episode, corpus_seed=4242
    )
    legacy_bundle = expected_bundle(legacy_episode, 4242)

    assert fresh.manifest.skill_bundle_id == fresh_bundle.source_manifest.bundle_id
    assert fresh.skill is not None
    assert fresh.skill.compiler_manifest.compiler_input_hash == hash_compiler_input(
        compiler_view(fresh_bundle)
    )
    assert legacy_default.model_dump(mode="json") == legacy_explicit.model_dump(mode="json")
    assert legacy_default.manifest.skill_bundle_id == legacy_bundle.source_manifest.bundle_id
    assert legacy_default.skill is not None
    assert legacy_default.skill.compiler_manifest.compiler_input_hash == hash_compiler_input(
        compiler_view(legacy_bundle)
    )


@pytest.mark.parametrize("corpus_seed", (True, 4243.0, "4243", None))
def test_development_materialization_rejects_non_integer_corpus_seed(corpus_seed: object) -> None:
    corpus = generate_development_corpus(4242)
    planned = build_development_plan(corpus).episodes[0]

    with pytest.raises(ValueError, match="corpus_seed"):
        materialize_development_episode(corpus, planned, corpus_seed=corpus_seed)  # type: ignore[arg-type]


def test_development_materialization_rejects_mismatched_corpus_seed() -> None:
    corpus = generate_development_corpus(4242)
    planned = build_development_plan(corpus).episodes[0]

    with pytest.raises(ValueError, match="does not match"):
        materialize_development_episode(corpus, planned, corpus_seed=4243)


def test_development_model_seed_maps_uint64_to_signed_int64_exactly() -> None:
    assert development_model_seed(0) == 0
    assert development_model_seed(2**63 - 1) == 2**63 - 1
    assert development_model_seed(2**63) == 0
    assert development_model_seed(2**64 - 1) == 2**63 - 1
    for invalid in (-1, 2**64, True):
        with pytest.raises(ValueError, match="uint64"):
            development_model_seed(invalid)


def test_development_materializes_an_overflowing_case_with_a_signed_request_seed() -> None:
    corpus = generate_development_corpus(4242)
    case = next(
        case
        for case in corpus.cases
        if case.domain == "financial_adjustments" and case.seed > 2**63 - 1
    )
    episode = next(
        episode
        for episode in build_development_plan(corpus).episodes
        if episode.case_id == case.case_id and episode.condition.value == "A0_BARE"
    )

    execution_plan = materialize_development_episode(corpus, episode)

    assert execution_plan.manifest.seed == development_model_seed(case.seed)
    assert execution_plan.manifest.seed >= 0
    assert execution_plan.task.seed == case.seed


def test_development_materializes_every_case_with_a_signed_manifest_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = generate_development_corpus(4242)
    planned = build_development_plan(corpus).episodes
    baseline_skill = _materialization_skill()
    monkeypatch.setattr(
        development_execution,
        "_compiled_skill",
        lambda domain, ratio, corpus_seed: baseline_skill,
    )
    selected_by_case: dict[str, DevelopmentPlannedEpisode] = {}
    for episode in planned:
        selected_by_case.setdefault(episode.case_id, episode)
    selected = tuple(selected_by_case[case.case_id] for case in corpus.cases)

    execution_plans = materialize_development_episodes(corpus, selected)

    assert len(execution_plans) == len(selected) == len(corpus.cases) == 40
    assert {episode.case_id for episode in selected} == {case.case_id for case in corpus.cases}
    case_model_seeds = {development_model_seed(case.seed) for case in corpus.cases}
    assert len(case_model_seeds) == len(corpus.cases)
    for episode, execution_plan in zip(selected, execution_plans, strict=True):
        case = next(case for case in corpus.cases if case.case_id == episode.case_id)
        assert execution_plan.manifest.seed == development_model_seed(case.seed)
        assert execution_plan.manifest.seed >= 0
        assert execution_plan.task.seed == execution_plan.initial_state.seed == case.seed


def test_development_runner_persists_atomic_results_telemetry_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    corpus = generate_development_corpus(4242)
    planned = build_development_plan(corpus).episodes[:2]
    execution_plans = materialize_development_episodes(corpus, planned)
    run_directory = tmp_path / "development-run"

    first = asyncio.run(
        run_development_plan(
            planned,
            execution_plans=execution_plans,
            run_directory=run_directory,
        )
    )
    result_paths = sorted((run_directory / "results").glob("*.json"))
    telemetry_paths = sorted((run_directory / "telemetry").glob("*.json"))
    second = asyncio.run(
        run_development_plan(
            planned,
            execution_plans=execution_plans,
            run_directory=run_directory,
        )
    )

    assert len(first.completed_episode_ids) == 2
    assert len(result_paths) == len(telemetry_paths) == 2
    assert all(
        parse_episode_result(json.loads(path.read_bytes())["result"]) for path in result_paths
    )
    assert all(
        json.loads(path.read_bytes())["profile"] == "SSB-DEVELOPMENT-TELEMETRY1"
        for path in telemetry_paths
    )
    assert second.completed_episode_ids == ()
    assert second.skipped_episode_ids == first.completed_episode_ids


def test_cli_runs_a_limited_development_stage_and_resumes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_directory = tmp_path / "cli-run"
    arguments = [
        "episodes",
        "run",
        "--stage-a",
        "--split",
        "development",
        "--limit",
        "2",
        "--output-dir",
        str(run_directory),
    ]

    first = CliRunner().invoke(cli.app, arguments)
    second = CliRunner().invoke(cli.app, arguments)

    assert first.exit_code == 0, first.output
    assert "PRACTICE_NOT_EVIDENCE" in first.output
    assert "completed=2" in first.output
    assert second.exit_code == 0, second.output
    assert "skipped=2" in second.output


def test_development_audit_requires_complete_custodied_practice_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_directory = tmp_path / "artifacts/experiments/development/stage_a/limit-2"
    run = CliRunner().invoke(
        cli.app,
        [
            "episodes",
            "run",
            "--stage-a",
            "--split",
            "development",
            "--limit",
            "2",
            "--output-dir",
            str(run_directory),
        ],
    )
    assert run.exit_code == 0, run.output

    audited = CliRunner().invoke(cli.app, ["experiments", "audit", "--split", "development"])
    assert audited.exit_code == 0, audited.output
    assert "PRACTICE_NOT_EVIDENCE" in audited.output
    assert "completed=2" in audited.output

    next((run_directory / "telemetry").glob("*.json")).unlink()
    with pytest.raises(RunCustodyError, match="development telemetry is unreadable"):
        audit_development_runs(Path("artifacts/experiments/development"))
