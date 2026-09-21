from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.corpus.development import generate_development_corpus
from shadowskillbench.episodes.native_tool_turn_wire import (
    NATIVE_TOOL_TURN_PROMPT_HASH,
    NATIVE_TOOL_TURN_PROMPT_PROFILE,
)
from shadowskillbench.experiments import capability_pilot
from shadowskillbench.experiments.ollama_gemma4_native_tools_profile import (
    Gemma4NativeToolsProfileError,
    ollama_gemma4_native_tools_projection,
    validate_gemma4_native_tools_identity,
    validate_gemma4_native_tools_role_profile_receipt,
)
from shadowskillbench.models.runtime import (
    ModelRunDescriptor,
    RunDescriptorError,
    ollama_native_tool_client_from_run_descriptor,
)


def _call(marker: str, request: str, response: str, input_tokens: int) -> dict[str, object]:
    return {
        "arguments_hash": "sha256:" + "a" * 64,
        "attempts": 1,
        "finish_reason": "tool_calls",
        "provider_done": True,
        "provider_finish_reason": "stop",
        "raw_request_hash": "sha256:" + request * 64,
        "raw_response_hash": "sha256:" + response * 64,
        "role_marker": marker,
        "tool_declaration_hash": "sha256:" + "b" * 64,
        "tool_name": "finish_task",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": 2,
            "total_tokens": input_tokens + 2,
        },
    }


def _role_receipt() -> dict[str, object]:
    baseline = _call("baseline", "1", "2", 83)
    developer = _call("developer", "3", "4", 101)
    tool_call = _call("tool_call", "3", "5", 101)
    body = {
        "record_kind": "OLLAMA_NATIVE_TOOLS_MATRIX_ROLE_TOOL_CONFORMANCE_RECEIPT1",
        **ollama_gemma4_native_tools_projection(),
        "baseline": baseline,
        "developer": developer,
        "developer_prompt_token_delta": 18,
        "tool_call": tool_call,
    }
    return {**body, "receipt_hash": sha256_ref(body)}


def _identity_receipt() -> dict[str, object]:
    body = {
        "record_kind": "OLLAMA_NATIVE_TOOLS_MATRIX_IDENTITY_RECEIPT1",
        **ollama_gemma4_native_tools_projection(),
        "capabilities": ["completion", "vision", "audio", "tools", "thinking"],
        "model_context": 262144,
        "show_response_hash": "sha256:" + "c" * 64,
    }
    return {**body, "receipt_hash": sha256_ref(body)}


def test_gemma_native_tools_receipts_bind_response2_and_role_delta() -> None:
    receipt = _role_receipt()
    assert validate_gemma4_native_tools_role_profile_receipt(receipt) == receipt
    receipt["developer_prompt_token_delta"] = 17
    try:
        validate_gemma4_native_tools_role_profile_receipt(receipt)
    except Gemma4NativeToolsProfileError:
        pass
    else:
        raise AssertionError("a stale prompt delta was admitted")


def test_gemma_native_tools_identity_and_fresh_selection_are_bound() -> None:
    identity = _identity_receipt()
    assert validate_gemma4_native_tools_identity(identity) == identity
    corpus = generate_development_corpus(4248)
    screen = capability_pilot.select_capability_cells(
        corpus, phase="screen", bundle="gemma4-native-tools-context32768"
    )
    validation = capability_pilot.select_capability_cells(
        corpus, phase="validation", bundle="gemma4-native-tools-context32768"
    )
    commitment = capability_pilot._grounding_commitment("gemma4-native-tools-context32768")
    successor = capability_pilot._grounding_commitment("gemma4-native-tools-context32768-v2")
    successor_v3 = capability_pilot._grounding_commitment("gemma4-native-tools-context32768-v3")
    successor_v4 = capability_pilot._grounding_commitment("gemma4-native-tools-context32768-v4")
    assert len(screen) == 8
    assert len(validation) == 40
    assert commitment["selection_hash"] == sha256_ref(commitment["selection"])
    assert successor["base_commitment_hash"] == sha256_ref(commitment)
    assert successor["selection"] == commitment["selection"]
    assert successor_v3["base_commitment_hash"] == sha256_ref(successor)
    assert successor_v3["selection"] == successor["selection"]
    assert successor_v4["base_commitment_hash"] == sha256_ref(successor_v3)
    assert successor_v4["selection"] == successor_v3["selection"]
    assert (
        successor["native_turn_projection"]
        == capability_pilot.GEMMA4_NATIVE_TOOLS_NATIVE_TURN_PROJECTION
    )
    assert (
        successor_v3["native_turn_projection"]
        == capability_pilot.GEMMA4_NATIVE_TOOLS_V3_NATIVE_TURN_PROJECTION
    )
    one_factor_delta = successor_v4["one_factor_delta"]
    assert type(one_factor_delta) is dict
    assert one_factor_delta["field"] == ("transport.native_turn_prompt_and_request_profile")
    assert {cell.case_id for cell in screen}.isdisjoint(
        {"development_0863fa8260723020", "development_d6b479ccd34e7720"}
    )


def test_gemma_native_tools_v2_plan_never_uses_structured_turn_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = generate_development_corpus(4248)
    cells = capability_pilot.select_capability_cells(
        corpus, phase="screen", bundle="gemma4-native-tools-context32768-v2"
    )

    def unexpected_structured_projection(_bundle: object) -> dict[str, str]:
        raise AssertionError("native tools must not construct a structured-turn projection")

    monkeypatch.setattr(capability_pilot, "_turn_projection", unexpected_structured_projection)
    plan = capability_pilot._plan(
        cells, phase="screen", bundle="gemma4-native-tools-context32768-v2"
    )

    assert plan["native_turn_projection"] == {
        "endpoint_path": "/api/chat",
        "profile": "SSB-OLLAMA-NATIVE-TOOLS-TURN1",
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
        "response_contract": "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
        "response_format": "omitted",
        "stream": False,
        "tool_declaration_source": "ToolRegistry.visible_specs",
    }
    assert "turn_schema_hash" not in plan
    runtime = capability_pilot._runtime(
        "http://127.0.0.1:11435/api/chat",
        "SSB_GEMMA4_TEST_KEY",
        _role_receipt(),
        phase="screen",
        model_identity_receipt=_identity_receipt(),
        bundle="gemma4-native-tools-context32768-v2",
    )
    assert runtime["request_profile"] == "SSB-OLLAMA-NATIVE-TOOLS1"
    assert "turn_schema_hash" not in runtime


def test_gemma_native_tools_v3_binds_profile2_history_and_successor_root() -> None:
    corpus = generate_development_corpus(4248)
    cells = capability_pilot.select_capability_cells(
        corpus, phase="screen", bundle="gemma4-native-tools-context32768-v3"
    )
    plan = capability_pilot._plan(
        cells, phase="screen", bundle="gemma4-native-tools-context32768-v3"
    )
    runtime = capability_pilot._runtime(
        "http://127.0.0.1:11435/api/chat",
        "SSB_GEMMA4_TEST_KEY",
        _role_receipt(),
        phase="screen",
        model_identity_receipt=_identity_receipt(),
        bundle="gemma4-native-tools-context32768-v3",
    )

    assert plan["action_interface_profile"] == "SSB-OLLAMA-NATIVE-TOOLS2"
    assert (
        plan["native_turn_projection"]
        == capability_pilot.GEMMA4_NATIVE_TOOLS_V3_NATIVE_TURN_PROJECTION
    )
    assert runtime["request_profile"] == "SSB-OLLAMA-NATIVE-TOOLS2"
    assert runtime["history_projection"] == "SSB-OLLAMA-NATIVE-TOOLS-HISTORY1"


def test_gemma_native_tools_v4_binds_profile3_prompt_and_history() -> None:
    corpus = generate_development_corpus(4248)
    cells = capability_pilot.select_capability_cells(
        corpus, phase="screen", bundle="gemma4-native-tools-context32768-v4"
    )
    plan = capability_pilot._plan(
        cells, phase="screen", bundle="gemma4-native-tools-context32768-v4"
    )
    runtime = capability_pilot._runtime(
        "http://127.0.0.1:11435/api/chat",
        "SSB_GEMMA4_TEST_KEY",
        _role_receipt(),
        phase="screen",
        model_identity_receipt=_identity_receipt(),
        bundle="gemma4-native-tools-context32768-v4",
    )

    assert plan["action_interface_profile"] == "SSB-OLLAMA-NATIVE-TOOLS3"
    assert plan["native_turn_prompt_profile"] == NATIVE_TOOL_TURN_PROMPT_PROFILE
    assert plan["native_turn_prompt_hash"] == NATIVE_TOOL_TURN_PROMPT_HASH
    assert (
        plan["native_turn_projection"]
        == capability_pilot.GEMMA4_NATIVE_TOOLS_V4_NATIVE_TURN_PROJECTION
    )
    assert runtime["request_profile"] == "SSB-OLLAMA-NATIVE-TOOLS3"
    assert runtime["history_projection"] == "SSB-OLLAMA-NATIVE-TOOLS-HISTORY1"
    assert runtime["native_turn_prompt_profile"] == NATIVE_TOOL_TURN_PROMPT_PROFILE
    assert runtime["native_turn_prompt_hash"] == NATIVE_TOOL_TURN_PROMPT_HASH


def test_gemma_native_tools_v3_runtime_factory_requires_history_profile_binding() -> None:
    descriptor = ModelRunDescriptor(
        provider="ollama",
        model="gemma4:12b-it-q4_K_M",
        model_version="sha256:4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c",
        endpoint="http://127.0.0.1:11435/api/chat",
        api_key_environment="SSB_GEMMA4_TEST_KEY",
        max_attempts=1,
        supports_structured_output=False,
        executor_runtime_profile={
            "request_profile": "SSB-OLLAMA-NATIVE-TOOLS2",
            "response_contract": "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
            "endpoint_path": "/api/chat",
            "context_length": 32768,
            "top_p_wire": "omitted",
            "reasoning_effort_wire": "omitted",
            "history_projection": "SSB-OLLAMA-NATIVE-TOOLS-HISTORY1",
        },
    )
    client = ollama_native_tool_client_from_run_descriptor(
        descriptor, environment={"SSB_GEMMA4_TEST_KEY": "test-key"}, trust_env=False
    )
    assert client.native_capabilities.request_profile == "SSB-OLLAMA-NATIVE-TOOLS2"
    assert client.response_contract == "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2"
    profile = descriptor.executor_runtime_profile
    assert profile is not None

    with pytest.raises(RunDescriptorError):
        ollama_native_tool_client_from_run_descriptor(
            replace(
                descriptor,
                executor_runtime_profile={
                    **profile,
                    "history_projection": "mismatch",
                },
            ),
            environment={"SSB_GEMMA4_TEST_KEY": "test-key"},
            trust_env=False,
        )


def test_gemma_native_tools_v4_runtime_factory_requires_prompt_binding() -> None:
    descriptor = ModelRunDescriptor(
        provider="ollama",
        model="gemma4:12b-it-q4_K_M",
        model_version="sha256:4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c",
        endpoint="http://127.0.0.1:11435/api/chat",
        api_key_environment="SSB_GEMMA4_TEST_KEY",
        max_attempts=1,
        supports_structured_output=False,
        executor_runtime_profile={
            "request_profile": "SSB-OLLAMA-NATIVE-TOOLS3",
            "response_contract": "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
            "endpoint_path": "/api/chat",
            "context_length": 32768,
            "top_p_wire": "omitted",
            "reasoning_effort_wire": "omitted",
            "history_projection": "SSB-OLLAMA-NATIVE-TOOLS-HISTORY1",
            "native_turn_prompt_profile": NATIVE_TOOL_TURN_PROMPT_PROFILE,
            "native_turn_prompt_hash": NATIVE_TOOL_TURN_PROMPT_HASH,
        },
    )
    client = ollama_native_tool_client_from_run_descriptor(
        descriptor, environment={"SSB_GEMMA4_TEST_KEY": "test-key"}, trust_env=False
    )
    assert client.native_capabilities.request_profile == "SSB-OLLAMA-NATIVE-TOOLS3"
    profile = descriptor.executor_runtime_profile
    assert profile is not None
    with pytest.raises(RunDescriptorError):
        ollama_native_tool_client_from_run_descriptor(
            replace(
                descriptor,
                executor_runtime_profile={
                    **profile,
                    "native_turn_prompt_hash": "sha256:" + "0" * 64,
                },
            ),
            environment={"SSB_GEMMA4_TEST_KEY": "test-key"},
            trust_env=False,
        )


def test_gemma_native_tools_cli_routes_receipts_and_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    role: dict[str, object] = {"kind": "role"}
    identity: dict[str, object] = {"kind": "identity"}

    def load_role(_path: Path) -> dict[str, object]:
        seen.append("role")
        return role

    def load_identity(_path: Path) -> dict[str, object]:
        seen.append("identity")
        return identity

    def build_client(_endpoint: str, _environment: str, _path: Path) -> object:
        seen.append("builder")
        return object()

    async def run_pilot(**kwargs: Any) -> dict[str, object]:
        assert kwargs["bundle"] == "gemma4-native-tools-context32768"
        assert kwargs["role_receipt"] is role
        assert kwargs["model_identity_receipt"] is identity
        seen.append("run")
        return {"phase": "screen", "completed": 0, "skipped": 0, "total": 8}

    monkeypatch.setattr(cli, "load_gemma4_native_tools_role_profile_receipt", load_role)
    monkeypatch.setattr(cli, "load_gemma4_native_tools_identity", load_identity)
    monkeypatch.setattr(
        cli, "build_gemma4_native_tools_context32768_capability_client", build_client
    )
    monkeypatch.setattr(cli, "run_capability_pilot", run_pilot)

    result = CliRunner().invoke(
        cli.app,
        [
            "pilot",
            "capability-run",
            "--endpoint",
            "http://127.0.0.1:11435/api/chat",
            "--api-key-environment",
            "SSB_GEMMA4_TEST_KEY",
            "--role-profile-receipt",
            "role.json",
            "--model-identity-receipt",
            "identity.json",
            "--phase",
            "screen",
            "--output-dir",
            "artifacts/development/v4-gemma4-native-tools-context32768-seed-4248-screen",
            "--bundle",
            "gemma4-native-tools-context32768",
        ],
    )

    assert result.exit_code == 0
    assert seen == ["role", "identity", "builder", "run"]


def test_gemma_native_tools_v3_cli_routes_successor_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    role: dict[str, object] = {"kind": "role"}
    identity: dict[str, object] = {"kind": "identity"}

    monkeypatch.setattr(cli, "load_gemma4_native_tools_role_profile_receipt", lambda _path: role)
    monkeypatch.setattr(cli, "load_gemma4_native_tools_identity", lambda _path: identity)
    monkeypatch.setattr(
        cli,
        "build_gemma4_native_tools_context32768_v3_capability_client",
        lambda *_args: seen.append("builder") or object(),
    )

    async def run_pilot(**kwargs: Any) -> dict[str, object]:
        assert kwargs["bundle"] == "gemma4-native-tools-context32768-v3"
        assert kwargs["role_receipt"] is role
        assert kwargs["model_identity_receipt"] is identity
        seen.append("run")
        return {"phase": "screen", "completed": 0, "skipped": 0, "total": 8}

    monkeypatch.setattr(cli, "run_capability_pilot", run_pilot)
    result = CliRunner().invoke(
        cli.app,
        [
            "pilot",
            "capability-run",
            "--endpoint",
            "http://127.0.0.1:11435/api/chat",
            "--api-key-environment",
            "SSB_GEMMA4_TEST_KEY",
            "--role-profile-receipt",
            "role.json",
            "--model-identity-receipt",
            "identity.json",
            "--phase",
            "screen",
            "--output-dir",
            "artifacts/development/v4-gemma4-native-tools-context32768-seed-4248-v3-screen",
            "--bundle",
            "gemma4-native-tools-context32768-v3",
        ],
    )

    assert result.exit_code == 0
    assert seen == ["builder", "run"]


def test_gemma_native_tools_v4_cli_routes_successor_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role: dict[str, object] = {"kind": "role"}
    identity: dict[str, object] = {"kind": "identity"}
    seen: list[str] = []
    monkeypatch.setattr(cli, "load_gemma4_native_tools_role_profile_receipt", lambda _path: role)
    monkeypatch.setattr(cli, "load_gemma4_native_tools_identity", lambda _path: identity)
    monkeypatch.setattr(
        cli,
        "build_gemma4_native_tools_context32768_v4_capability_client",
        lambda *_args: seen.append("builder") or object(),
    )

    async def run_pilot(**kwargs: Any) -> dict[str, object]:
        assert kwargs["bundle"] == "gemma4-native-tools-context32768-v4"
        seen.append("run")
        return {"phase": "screen", "completed": 0, "skipped": 0, "total": 8}

    monkeypatch.setattr(cli, "run_capability_pilot", run_pilot)
    result = CliRunner().invoke(
        cli.app,
        [
            "pilot",
            "capability-run",
            "--endpoint",
            "http://127.0.0.1:11435/api/chat",
            "--api-key-environment",
            "SSB_GEMMA4_TEST_KEY",
            "--role-profile-receipt",
            "role.json",
            "--model-identity-receipt",
            "identity.json",
            "--phase",
            "screen",
            "--output-dir",
            "artifacts/development/v4-gemma4-native-tools-context32768-seed-4248-v4-screen",
            "--bundle",
            "gemma4-native-tools-context32768-v4",
        ],
    )
    assert result.exit_code == 0
    assert seen == ["builder", "run"]
