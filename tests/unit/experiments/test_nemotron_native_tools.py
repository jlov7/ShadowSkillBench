from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.development import generate_development_corpus
from shadowskillbench.engine import JsonObject
from shadowskillbench.episodes.executor import (
    AgentClaim,
    EpisodeDisposition,
    ExecutorConfigurationError,
)
from shadowskillbench.episodes.models import ExperimentCondition
from shadowskillbench.episodes.native_tool_executor import run_native_tool_episode
from shadowskillbench.episodes.native_tool_turn_wire import (
    NATIVE_TOOL_TURN_PROMPT,
    NATIVE_TOOL_TURN_PROMPT_PROFILE,
)
from shadowskillbench.episodes.tools import ToolRegistry
from shadowskillbench.experiments import capability_pilot
from shadowskillbench.experiments.development_execution import materialize_development_episode
from shadowskillbench.experiments.ollama_nemotron_context32k_profile import (
    canonical_nemotron_context32k_role_profile_receipt,
    nemotron_context32k_live_show_identity_projection,
    ollama_nemotron_context32k_profile_projection,
)
from shadowskillbench.experiments.ollama_nemotron_native_tools_profile import (
    NemotronNativeToolsRoleProfileError,
    canonical_nemotron_native_tools_role_profile_failure_receipt,
    canonical_nemotron_native_tools_role_profile_receipt,
    validate_nemotron_native_tools_role_profile_receipt,
)
from shadowskillbench.models import (
    ModelAdapterError,
    ModelRequest,
    ProviderCapabilities,
    TokenUsage,
)
from shadowskillbench.models.native_tools import (
    NativeToolDefinition,
    NativeToolHistoryTurn,
    NativeToolProviderCapabilities,
    NativeToolResponse,
    native_tool_declaration_hash,
)
from shadowskillbench.models.runtime import (
    ModelRunDescriptor,
    RunDescriptorError,
    ollama_native_tool_client_from_run_descriptor,
)


def _native_receipt() -> dict[str, object]:
    def call(request: str, response: str, marker: str, input_tokens: int) -> dict[str, object]:
        return {
            "raw_request_hash": "sha256:" + request * 64,
            "raw_response_hash": "sha256:" + response * 64,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 2,
                "total_tokens": input_tokens + 2,
            },
            "attempts": 1,
            "finish_reason": "tool_calls",
            "role_marker": marker,
        }

    tool_call = {
        **call("5", "6", "SSB_NATIVE_TOOL_CALL_MARKER", 12),
        "tool_name": "finish_task",
        "arguments_hash": "sha256:" + "7" * 64,
        "tool_declaration_hash": "sha256:" + "8" * 64,
    }
    return json.loads(
        canonical_nemotron_native_tools_role_profile_receipt(
            baseline=call("1", "2", "SSB_NATIVE_SYSTEM_ROLE_MARKER", 10),
            developer=call("3", "4", "SSB_NATIVE_DEVELOPER_ROLE_MARKER", 11),
            tool_call=tool_call,
        )
    )


def _identity_receipt() -> dict[str, object]:
    body = {
        "record_kind": "OLLAMA_NEMOTRON_CONTEXT32K_LIVE_SHOW_IDENTITY_RECEIPT2",
        **nemotron_context32k_live_show_identity_projection(),
        "show_response_hash": "sha256:" + "9" * 64,
    }
    return {**body, "receipt_hash": sha256_ref(body)}


def _old_context32k_receipt() -> dict[str, object]:
    schema_hash = ollama_nemotron_context32k_profile_projection()["structured_output_schema_hash"]
    return json.loads(
        canonical_nemotron_context32k_role_profile_receipt(
            baseline={
                "attempts": 1,
                "raw_request_hash": "sha256:" + "a" * 64,
                "raw_response_hash": "sha256:" + "b" * 64,
                "response_marker": "SSB_SYSTEM_ROLE_MARKER_7fa38c",
                "structured_output_schema_hash": schema_hash,
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            },
            developer={
                "attempts": 1,
                "raw_request_hash": "sha256:" + "c" * 64,
                "raw_response_hash": "sha256:" + "d" * 64,
                "response_marker": "SSB_DEVELOPER_ROLE_MARKER_81c2ad",
                "structured_output_schema_hash": schema_hash,
                "usage": {"input_tokens": 11, "output_tokens": 2, "total_tokens": 13},
            },
        )
    )


class _NativeScript:
    def __init__(
        self,
        plan,
        responses: list[tuple[str, dict[str, object]] | ModelAdapterError],
        *,
        request_profile: str = "SSB-OLLAMA-NATIVE-TOOLS1",
    ) -> None:
        expected = plan.manifest.executor_model
        self.capabilities = ProviderCapabilities(
            provider=expected.provider,
            model=expected.model,
            model_version=expected.model_version_date,
            supports_system_role=True,
            supports_developer_role=True,
            supports_seed=True,
            supports_structured_output=False,
        )
        self.native_capabilities = NativeToolProviderCapabilities(
            base=self.capabilities,
            endpoint_path="/api/chat",
            supports_native_tool_calls=True,
            request_profile=request_profile,
        )
        self._max_attempts = 1
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []
        self.histories: list[tuple[NativeToolHistoryTurn, ...]] = []

    async def call_tools(
        self,
        request: ModelRequest,
        declarations: tuple[NativeToolDefinition, ...],
        *,
        history: tuple[NativeToolHistoryTurn, ...] = (),
    ) -> NativeToolResponse:
        self.requests.append(request)
        self.histories.append(history)
        response = self._responses.pop(0)
        if type(response) is ModelAdapterError:
            raise response
        name, arguments = cast(tuple[str, dict[str, object]], response)
        index = len(self.requests)
        return NativeToolResponse(
            tool_name=name,
            arguments=cast(JsonObject, arguments),
            raw_request_hash="sha256:" + f"{index:x}" * 64,
            raw_response_hash="sha256:" + f"{index + 1:x}" * 64,
            tool_declaration_hash=native_tool_declaration_hash(declarations),
            usage=TokenUsage(input_tokens=5, output_tokens=3, total_tokens=8),
            cost=None,
            attempts=1,
            finish_reason="tool_calls",
        )


def _native_plan():
    corpus = generate_development_corpus(4246)
    cell = capability_pilot.select_capability_cells(
        corpus, phase="screen", bundle="nemotron-native-tools-context32k"
    )[0]
    return materialize_development_episode(
        corpus,
        capability_pilot._planned(cell),
        executor_model_override=capability_pilot._executor_model(
            "nemotron-native-tools-context32k"
        ),
        max_tokens=8192,
        corpus_seed=4246,
    )


def test_native_executor_validates_before_execution_and_finish_task_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _native_plan()
    client = _NativeScript(plan, [("finish_task", {"summary": "done"})])
    observed: list[str] = []
    original_validate = ToolRegistry.validate_call
    original_execute = ToolRegistry.execute

    def validate(self, call, *, condition):
        observed.append("validate")
        return original_validate(self, call, condition=condition)

    def execute(self, state, call, *, cursor, condition):
        assert observed == ["validate"]
        observed.append("execute")
        return original_execute(self, state, call, cursor=cursor, condition=condition)

    monkeypatch.setattr(ToolRegistry, "validate_call", validate)
    monkeypatch.setattr(ToolRegistry, "execute", execute)
    result = asyncio.run(run_native_tool_episode(plan, client, temperature=0.15, persist=False))

    assert observed[:2] == ["validate", "execute"]
    assert result.claim is AgentClaim.FINISH
    assert result.disposition is EpisodeDisposition.COMPLETED
    assert result.trace[0].action is not None
    assert result.trace[0].action.action_id == "native_action_0"
    assert len(client.requests) == 1


def test_native_executor_rejects_registry_calls_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _native_plan()
    client = _NativeScript(plan, [("unregistered_tool", {})])

    def unexpected_execute(*_args, **_kwargs):
        raise AssertionError("invalid registry call reached execution")

    monkeypatch.setattr(ToolRegistry, "execute", unexpected_execute)
    result = asyncio.run(run_native_tool_episode(plan, client, temperature=0.15, persist=False))

    assert result.disposition is EpisodeDisposition.INVALID_ACTION
    assert result.error_code == "AGENT_INVALID_ACTION"
    assert result.trace[0].receipt.raw_request_hash is not None
    assert result.trace[0].action is None


def test_native_executor_preserves_sanitized_invalid_output_custody() -> None:
    plan = _native_plan()
    client = _NativeScript(
        plan,
        [
            ModelAdapterError(
                code="MODEL_OUTPUT_INVALID",
                attempts=1,
                before_meaningful_behavior=False,
                raw_request_hash="sha256:" + "a" * 64,
                raw_response_hash="sha256:" + "b" * 64,
                reported_usage=TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3),
            )
        ],
    )
    result = asyncio.run(run_native_tool_episode(plan, client, temperature=0.15, persist=False))

    assert result.disposition is EpisodeDisposition.INVALID_ACTION
    assert result.error_code == "MODEL_OUTPUT_INVALID"
    assert result.trace[0].receipt.raw_request_hash == "sha256:" + "a" * 64
    assert result.trace[0].receipt.raw_response_hash == "sha256:" + "b" * 64
    assert result.trace[0].receipt.usage == TokenUsage(
        input_tokens=1, output_tokens=2, total_tokens=3
    )
    assert result.trace[0].output is None


def test_native_executor_profile2_supplies_only_validated_native_history() -> None:
    corpus = generate_development_corpus(4246)
    cell = next(
        cell
        for cell in capability_pilot.select_capability_cells(
            corpus, phase="screen", bundle="nemotron-native-tools-context32k"
        )
        if cell.condition is ExperimentCondition.B3_DETERMINISTIC_GATE
    )
    plan = materialize_development_episode(
        corpus,
        capability_pilot._planned(cell),
        executor_model_override=capability_pilot._executor_model(
            "nemotron-native-tools-context32k"
        ),
        max_tokens=8192,
        corpus_seed=4246,
    )
    client = _NativeScript(
        plan,
        [("resolve_authority", {}), ("finish_task", {"summary": "done"})],
        request_profile="SSB-OLLAMA-NATIVE-TOOLS2",
    )

    result = asyncio.run(
        run_native_tool_episode(
            plan,
            client,
            temperature=0.15,
            persist=False,
            history_profile="SSB-OLLAMA-NATIVE-TOOLS2",
        )
    )

    assert result.disposition is EpisodeDisposition.COMPLETED
    assert client.histories[0] == ()
    assert len(client.histories[1]) == 1
    history = client.histories[1][0]
    assert history.tool_name == "resolve_authority"
    assert history.arguments == {}
    assert history.result["local_status"] == "success"
    assert all(message.role != "tool" for message in client.requests[1].messages)


def test_native_executor_profile3_adds_prompt_and_requires_profile_pairing() -> None:
    corpus = generate_development_corpus(4246)
    cell = next(
        cell
        for cell in capability_pilot.select_capability_cells(
            corpus, phase="screen", bundle="nemotron-native-tools-context32k"
        )
        if cell.condition is ExperimentCondition.B3_DETERMINISTIC_GATE
    )
    plan = materialize_development_episode(
        corpus,
        capability_pilot._planned(cell),
        executor_model_override=capability_pilot._executor_model(
            "nemotron-native-tools-context32k"
        ),
        max_tokens=8192,
        corpus_seed=4246,
    )
    client = _NativeScript(
        plan,
        [("resolve_authority", {}), ("finish_task", {"summary": "done"})],
        request_profile="SSB-OLLAMA-NATIVE-TOOLS3",
    )

    result = asyncio.run(
        run_native_tool_episode(
            plan,
            client,
            temperature=0.15,
            persist=False,
            history_profile="SSB-OLLAMA-NATIVE-TOOLS3",
            turn_prompt_profile=NATIVE_TOOL_TURN_PROMPT_PROFILE,
        )
    )

    assert result.disposition is EpisodeDisposition.COMPLETED
    assert client.requests[0].messages[-2].role == "developer"
    assert client.requests[0].messages[-2].content == NATIVE_TOOL_TURN_PROMPT
    assert client.requests[0].messages[-1].role == "user"
    assert client.histories[0] == ()
    assert client.histories[1][0].tool_name == "resolve_authority"
    with pytest.raises(ExecutorConfigurationError, match="turn prompt profile"):
        asyncio.run(
            run_native_tool_episode(
                plan,
                client,
                temperature=0.15,
                persist=False,
                history_profile="SSB-OLLAMA-NATIVE-TOOLS3",
            )
        )


def test_native_profile_plan_runtime_audit_and_cli_reject_cross_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus = generate_development_corpus(4246)
    cells = capability_pilot.select_capability_cells(
        corpus, phase="screen", bundle="nemotron-native-tools-context32k"
    )
    receipt = _native_receipt()
    identity = _identity_receipt()
    plan = capability_pilot._plan(cells, phase="screen", bundle="nemotron-native-tools-context32k")
    runtime = capability_pilot._runtime(
        "http://127.0.0.1:11435/api/chat",
        "SSB_NEMOTRON_TEST_KEY",
        receipt,
        phase="screen",
        model_identity_receipt=identity,
        bundle="nemotron-native-tools-context32k",
    )
    assert plan["action_interface_profile"] == "SSB-OLLAMA-NATIVE-TOOLS1"
    assert "turn_schema_hash" not in plan
    assert runtime["request_profile"] == "SSB-OLLAMA-NATIVE-TOOLS1"
    assert runtime["response_format"] == "omitted"

    (tmp_path / "results").mkdir()
    (tmp_path / "capability-plan.json").write_bytes(
        canonical_json_bytes({**plan, "plan_hash": sha256_ref(plan)})
    )
    (tmp_path / "runtime.json").write_bytes(
        canonical_json_bytes({**runtime, "runtime_hash": sha256_ref(runtime)})
    )
    (tmp_path / "role-profile-receipt.json").write_bytes(
        canonical_json_bytes(_old_context32k_receipt())
    )
    (tmp_path / "model-identity-receipt.json").write_bytes(canonical_json_bytes(identity))
    with pytest.raises(capability_pilot.CapabilityPilotHold, match="role receipt"):
        capability_pilot.audit_capability_pilot(
            tmp_path, phase="screen", bundle="nemotron-native-tools-context32k"
        )

    seen: list[str] = []

    def load_receipt(_path: Path) -> dict[str, object]:
        seen.append("receipt")
        return receipt

    def load_identity(_path: Path) -> dict[str, object]:
        seen.append("identity")
        return identity

    def build_client(_endpoint: str, _environment: str, _path: Path) -> object:
        seen.append("builder")
        return object()

    async def run_pilot(**kwargs: object) -> dict[str, object]:
        assert kwargs["bundle"] == "nemotron-native-tools-context32k"
        seen.append("run")
        return {"phase": "screen", "completed": 0, "skipped": 0, "total": 8}

    monkeypatch.setattr(cli, "load_nemotron_native_tools_role_profile_receipt", load_receipt)
    monkeypatch.setattr(cli, "load_nemotron_context32k_live_show_identity", load_identity)
    monkeypatch.setattr(
        cli, "build_nemotron_native_tools_context32k_capability_client", build_client
    )
    monkeypatch.setattr(cli, "run_capability_pilot", run_pilot)
    command = CliRunner().invoke(
        cli.app,
        [
            "pilot",
            "capability-run",
            "--endpoint",
            "http://127.0.0.1:11435/api/chat",
            "--api-key-environment",
            "SSB_NEMOTRON_TEST_KEY",
            "--role-profile-receipt",
            "receipt.json",
            "--model-identity-receipt",
            "identity.json",
            "--phase",
            "screen",
            "--output-dir",
            "capability",
            "--bundle",
            "nemotron-native-tools-context32k",
        ],
    )
    assert command.exit_code == 0
    assert seen == ["receipt", "identity", "builder", "run"]


def test_native_failure_receipts_are_hash_only_and_never_admit_the_old_profile() -> None:
    failure = json.loads(
        canonical_nemotron_native_tools_role_profile_failure_receipt(
            failure_code="MODEL_OUTPUT_INVALID",
            completed_calls=1,
            failed_call={
                "raw_request_hash": "sha256:" + "a" * 64,
                "raw_response_hash": "sha256:" + "b" * 64,
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                "attempts": 1,
                "finish_reason": None,
                "output_cap_exhausted": False,
            },
        )
    )
    assert set(failure["failed_call"]) == {
        "raw_request_hash",
        "raw_response_hash",
        "usage",
        "attempts",
        "finish_reason",
        "output_cap_exhausted",
    }
    with pytest.raises(NemotronNativeToolsRoleProfileError):
        validate_nemotron_native_tools_role_profile_receipt(failure)


def test_native_runtime_descriptor_rejects_old_structured_route() -> None:
    descriptor = ModelRunDescriptor(
        provider="ollama-nemotron35-lightning-30b-mlx",
        model="nemotron-3.5-lightning:30b-mlx",
        model_version="sha256:8b1474be6e54dc19eb7aa08bebfb9bda147c4b9ef9796a726131ad29ad15645a",
        endpoint="http://127.0.0.1:11435/api/chat",
        api_key_environment="SSB_NEMOTRON_TEST_KEY",
        max_attempts=1,
        supports_structured_output=False,
        executor_runtime_profile={
            "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
            "endpoint_path": "/api/chat",
            "context_length": 32768,
            "top_p_wire": "omitted",
            "reasoning_effort_wire": "omitted",
        },
    )
    client = ollama_native_tool_client_from_run_descriptor(
        descriptor, environment={"SSB_NEMOTRON_TEST_KEY": "test"}, trust_env=False
    )
    assert client.native_capabilities.endpoint_path == "/api/chat"
    with pytest.raises(RunDescriptorError):
        ollama_native_tool_client_from_run_descriptor(
            replace(
                descriptor,
                endpoint="http://127.0.0.1:11435/v1/chat/completions",
            ),
            environment={"SSB_NEMOTRON_TEST_KEY": "test"},
            trust_env=False,
        )
