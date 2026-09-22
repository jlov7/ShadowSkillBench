from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.text import Text
from typer import rich_utils
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.episodes import ExperimentCondition
from shadowskillbench.episodes.pilot_turn_wire import (
    NAMED_ACTION_TURN_PROMPT_PROFILE,
    NamedActionInitialTurnWire,
    named_action_active_turn_schema_hash,
    named_action_finished_turn_schema_hash,
    named_action_initial_turn_schema_hash,
)
from shadowskillbench.experiments import capability_pilot
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    canonical_role_profile_receipt,
    role_conformance_probe_projection,
)


def _synthetic_role_receipt() -> dict[str, object]:
    """Build a deterministic fixture without loading a private run receipt."""

    probe = role_conformance_probe_projection()
    return json.loads(
        canonical_role_profile_receipt(
            "sha256:" + "1" * 64,
            baseline={
                "raw_request_hash": "sha256:" + "a" * 64,
                "raw_response_hash": "sha256:" + "b" * 64,
                "prompt_tokens": 100,
                "response_marker": probe["baseline_expected_marker"],
            },
            developer={
                "raw_request_hash": "sha256:" + "c" * 64,
                "raw_response_hash": "sha256:" + "d" * 64,
                "prompt_tokens": 101,
                "response_marker": probe["developer_expected_marker"],
            },
            executor_model_digest=OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
        )
    )


def test_synthetic_role_receipt_is_canonical_and_deterministic() -> None:
    first = _synthetic_role_receipt()
    second = _synthetic_role_receipt()

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert sha256_ref(first) == sha256_ref(second)
    assert first["record_kind"] == "OLLAMA_ROLE_CONFORMANCE_RECEIPT1"


def test_runtime_and_plan_bind_the_synthetic_receipt() -> None:
    receipt = _synthetic_role_receipt()
    cell = capability_pilot.CapabilityPilotCell(
        stage="stage_a",
        condition=ExperimentCondition.A0_BARE,
        domain="access_provisioning",
        case_id="access_case_001",
        bundle_id=None,
    )

    runtime = capability_pilot._runtime(
        "http://127.0.0.1:11435/api/generate",
        "SSB_TEST_KEY",
        receipt,
        phase="screen",
    )
    plan = capability_pilot._plan((cell,), phase="screen")

    assert runtime["role_profile_receipt_hash"] == sha256_ref(receipt)
    assert runtime["turn_prompt_profile"] == NAMED_ACTION_TURN_PROMPT_PROFILE
    assert runtime["initial_turn_schema_hash"] == named_action_initial_turn_schema_hash()
    assert runtime["active_turn_schema_hash"] == named_action_active_turn_schema_hash()
    assert runtime["finished_turn_schema_hash"] == named_action_finished_turn_schema_hash()
    assert plan["turn_schema_hash"] == runtime["turn_schema_hash"]
    assert plan["max_aggregate_tokens"] == runtime["max_aggregate_tokens"] == 8192


def test_missing_commitment_fails_closed_without_private_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "not-present.json"
    monkeypatch.setattr(capability_pilot, "GROUNDING_COMMITMENT_PATH", missing)

    with pytest.raises(capability_pilot.CapabilityPilotHold, match="unreadable"):
        capability_pilot._grounding_commitment()


def test_configuration_and_provider_errors_are_classified_fail_closed() -> None:
    for error_code in {
        "CONFIGURATION_ERROR",
        "SCHEMA_ERROR",
        "MODEL_PROVIDER_TRANSIENT",
        "MODEL_PROVIDER_TERMINAL",
        "ENVIRONMENT_INVARIANT_FAILURE",
    }:
        assert capability_pilot._is_configuration_or_provider_error(error_code)

    assert not capability_pilot._is_configuration_or_provider_error("MODEL_OUTPUT_INVALID")
    assert not capability_pilot._is_configuration_or_provider_error("MODEL_UNKNOWN_FAILURE")
    assert not capability_pilot._is_configuration_or_provider_error(None)


def test_native_builder_rejects_the_chat_route(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(_synthetic_role_receipt()))

    with pytest.raises(capability_pilot.CapabilityPilotHold, match="/api/generate"):
        capability_pilot.build_native_capability_client(
            "http://127.0.0.1:11435/v1/chat/completions",
            "SSB_TEST_KEY",
            receipt_path,
        )


def test_wire_schema_hash_matches_the_public_model() -> None:
    assert named_action_initial_turn_schema_hash() == sha256_ref(
        NamedActionInitialTurnWire.model_json_schema()
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "pilot",
            "capability-run",
            "--endpoint",
            "http://127.0.0.1:11435/api/generate",
            "--api-key-environment",
            "SSB_TEST_KEY",
            "--role-profile-receipt",
            "receipt.json",
            "--output-dir",
            "capability",
        ],
        ["pilot", "capability-audit", "--output-dir", "capability"],
    ],
    ids=("capability-run", "capability-audit"),
)
def test_capability_cli_requires_an_explicit_phase_with_color(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", True)
    result = CliRunner().invoke(cli.app, arguments, color=True)
    normalized_output = Text.from_ansi(result.output).plain

    assert result.exit_code == 2
    assert "\x1b[" in result.output
    assert "Missing option '--phase'." in normalized_output


def test_capability_cli_rejects_an_unknown_bundle_before_audit() -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "pilot",
            "capability-audit",
            "--phase",
            "screen",
            "--output-dir",
            "capability",
            "--bundle",
            "unknown",
        ],
    )

    assert result.exit_code != 0
    assert "HOLD_CAPABILITY_BUNDLE" in result.output
