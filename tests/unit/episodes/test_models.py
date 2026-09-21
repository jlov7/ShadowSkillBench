from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from shadowskillbench.engine import TaskCase
from shadowskillbench.episodes.models import (
    BlockOrder,
    EpisodeBudgets,
    EpisodePlan,
    EpisodeStage,
    EpisodeStatus,
    ExecutorModel,
    ExperimentCondition,
    PromptHashes,
    episode_manifest_projection,
    hash_episode_manifest,
    parse_episode_manifest,
)

HASH = "sha256:" + "a" * 64
SCHEMA_REQUIRED_FIELDS = (
    "episode_id",
    "schema_version",
    "stage",
    "condition",
    "domain",
    "case_id",
    "world_hash",
    "executor_model",
    "prompt_hashes",
    "policy_hash",
    "skill_hash",
    "seed",
    "budgets",
    "status",
)


def raw_manifest(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "episode_id": "episode_a3_skill_access_r75_case_01_r02",
        "schema_version": "1.0",
        "stage": "confirmatory_a",
        "condition": "A3_SKILL_POLICY_SAME_TIER",
        "domain": "access_provisioning",
        "case_id": "case_access_01",
        "skill_bundle_id": "bundle_access_r75_s02",
        "contamination_ratio": 0.75,
        "world_hash": HASH,
        "executor_model": {
            "provider": "fixture",
            "model": "scripted-executor",
            "model_version_date": "2026-08-24",
        },
        "prompt_hashes": {"system": HASH, "developer": HASH, "user": HASH},
        "policy_hash": HASH,
        "skill_hash": HASH,
        "authority_graph_hash": None,
        "seed": 2,
        "budgets": {"max_turns": 12, "max_tool_calls": 12, "max_tokens": 4096},
        "status": "planned",
        "artifact_refs": [],
    }
    value.update(overrides)
    return value


def test_manifest_is_strict_frozen_and_validates_against_the_authoritative_schema() -> None:
    raw = raw_manifest()
    parsed = parse_episode_manifest(raw)
    schema = json.loads(
        (Path(__file__).parents[3] / "schemas" / "episode_manifest.schema.json").read_text()
    )

    assert parsed.condition is ExperimentCondition.A3_SKILL_POLICY_SAME_TIER
    assert parsed.stage is EpisodeStage.CONFIRMATORY_A
    assert parsed.status is EpisodeStatus.PLANNED
    assert parsed.budgets == EpisodeBudgets(max_turns=12, max_tool_calls=12, max_tokens=4096)
    assert parsed.executor_model == ExecutorModel(
        provider="fixture", model="scripted-executor", model_version_date="2026-08-24"
    )
    assert parsed.prompt_hashes == PromptHashes(system=HASH, developer=HASH, user=HASH)
    assert type(parsed.artifact_refs) is tuple
    Draft202012Validator(schema).validate(episode_manifest_projection(parsed))
    with pytest.raises(ValidationError):
        parsed.status = EpisodeStatus.RUNNING  # type: ignore[misc]


@pytest.mark.parametrize("condition", [member.value for member in ExperimentCondition])
def test_schema_permits_each_condition_with_nullable_context_fields(condition: str) -> None:
    parsed = parse_episode_manifest(
        raw_manifest(
            condition=condition,
            skill_bundle_id=None,
            contamination_ratio=None,
            skill_hash=None,
            policy_hash=None,
            authority_graph_hash=None,
        )
    )
    assert parsed.condition.value == condition


@pytest.mark.parametrize("field", SCHEMA_REQUIRED_FIELDS)
def test_schema_required_fields_are_required_by_the_model(field: str) -> None:
    raw = raw_manifest()
    del raw[field]
    with pytest.raises(ValueError):
        parse_episode_manifest(raw)


def test_schema_valid_condition_mixtures_are_left_for_context_and_execution_validation() -> None:
    for condition, stage in (
        ("A0_BARE", "confirmatory_b"),
        ("B3_DETERMINISTIC_GATE", "confirmatory_a"),
    ):
        assert parse_episode_manifest(raw_manifest(condition=condition, stage=stage))


def test_condition_metadata_and_block_orders_are_explicit() -> None:
    assert ExperimentCondition.A3_SKILL_POLICY_SAME_TIER.requires_skill
    assert ExperimentCondition.A3_SKILL_POLICY_SAME_TIER.requires_policy
    assert ExperimentCondition.B2_AUTHORITY_RESOLVER.requires_policy
    assert ExperimentCondition.B3_DETERMINISTIC_GATE.requires_policy
    assert ExperimentCondition.B2_AUTHORITY_RESOLVER.uses_authority_resolver
    assert ExperimentCondition.B3_DETERMINISTIC_GATE.uses_deterministic_gate
    assert not ExperimentCondition.A0_BARE.requires_skill
    assert not ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER.requires_block_order
    assert BlockOrder.POLICY_THEN_SKILL != BlockOrder.SKILL_THEN_POLICY


def test_projection_is_detached_and_hashes_all_manifest_content() -> None:
    parsed = parse_episode_manifest(raw_manifest())
    projection = episode_manifest_projection(parsed)

    assert hash_episode_manifest(parsed) == hash_episode_manifest(projection)
    projection["episode_id"] = "episode_changed"
    assert episode_manifest_projection(parsed)["episode_id"] != "episode_changed"
    assert hash_episode_manifest(parsed) != hash_episode_manifest(projection)


def test_plan_binds_held_out_task() -> None:
    manifest = parse_episode_manifest(
        raw_manifest(
            condition="A0_BARE",
            skill_bundle_id=None,
            contamination_ratio=None,
            skill_hash=None,
            policy_hash=None,
        )
    )
    task = TaskCase(
        case_id=manifest.case_id,
        domain=manifest.domain,
        world_id="world_access_01",
        seed=manifest.seed,
        objective="Review the visible request.",
        inputs={},
    )
    with pytest.raises(ValueError, match="Field required"):
        EpisodePlan(manifest=manifest, task=task)


@pytest.mark.parametrize(
    "overrides",
    [
        {"episode_id": "bad/id"},
        {"schema_version": "1"},
        {"stage": "other"},
        {"condition": "A9"},
        {"world_hash": "sha256:" + "A" * 64},
        {"policy_hash": "not-a-hash"},
        {"seed": True},
        {"budgets": {"max_turns": 0, "max_tool_calls": 1, "max_tokens": 1}},
        {"prompt_hashes": {"system": HASH, "developer": HASH}},
        {"artifact_refs": ["ok", "ok"]},
        {
            "executor_model": {
                "provider": "fixture",
                "model": "scripted-executor",
                "model_version_date": "2026-08-24",
                "unexpected": "extra",
            }
        },
        {"prompt_hashes": {"system": HASH, "developer": HASH, "user": HASH, "extra": HASH}},
        {"unexpected": "extra"},
    ],
)
def test_manifest_rejects_malformed_or_noncanonical_data(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_episode_manifest(raw_manifest(**overrides))
