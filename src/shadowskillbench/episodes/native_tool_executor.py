"""Executor for the separately frozen native provider tool-call interface."""

from __future__ import annotations

import math

from shadowskillbench.authority import GateInputError, GateOutcome, gate_action
from shadowskillbench.authority.view import build_authority_evidence_view
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.engine import (
    ActionCall,
    EnvironmentInvariantFailure,
    EventCursor,
    JsonObject,
    StateEvent,
    WorldState,
    execute_action,
    hash_state,
    make_snapshot,
    make_terminal_record,
)
from shadowskillbench.episodes.context import assemble_context
from shadowskillbench.episodes.executor import (
    AgentClaim,
    AgentTurn,
    EpisodeDisposition,
    EpisodeResult,
    ExecutionOverhead,
    ExecutionTurn,
    ExecutorConfigurationError,
    ModelCallReceipt,
    _aggregate_cost,
    _artifact_ref,
    _GateBlockAdapter,
    _json_message,
    _json_ready,
    _persist,
    _ready_observation,
    _token_count,
    _verify,
    episode_result_projection,
)
from shadowskillbench.episodes.models import EpisodePlan, EpisodeStatus, ExperimentCondition
from shadowskillbench.episodes.native_tool_turn_wire import (
    NATIVE_TOOL_TURN_PROMPT,
    NATIVE_TOOL_TURN_PROMPT_PROFILE,
)
from shadowskillbench.episodes.native_tools import native_tool_definitions
from shadowskillbench.episodes.tools import ToolRegistry, ToolRegistryError
from shadowskillbench.models.native_tools import (
    NativeToolClient,
    NativeToolHistoryTurn,
    NativeToolProviderCapabilities,
)
from shadowskillbench.models.protocol import (
    Message,
    ModelAdapterError,
    ModelRequest,
    ProviderCapabilities,
    TokenCost,
)


def _identity_check(plan: EpisodePlan, client: NativeToolClient, *, request_profile: str) -> None:
    capabilities = client.capabilities
    native = client.native_capabilities
    if (
        type(capabilities) is not ProviderCapabilities
        or type(native) is not NativeToolProviderCapabilities
    ):
        raise ExecutorConfigurationError("native client capabilities are invalid")
    expected = plan.manifest.executor_model
    if (
        capabilities.provider != expected.provider
        or capabilities.model != expected.model
        or capabilities.model_version != expected.model_version_date
        or not capabilities.supports_system_role
        or not capabilities.supports_developer_role
        or not capabilities.supports_seed
        or native.base != capabilities
        or native.endpoint_path != "/api/chat"
        or native.request_profile != request_profile
        or not native.supports_native_tool_calls
    ):
        raise ExecutorConfigurationError("native client capabilities do not match manifest")


async def run_native_tool_episode(
    plan: EpisodePlan,
    client: NativeToolClient,
    *,
    temperature: float,
    persist: bool = True,
    history_profile: str = "SSB-OLLAMA-NATIVE-TOOLS1",
    turn_prompt_profile: str | None = None,
) -> EpisodeResult:
    """Execute only response-native tool calls; text terminal claims are unavailable."""

    if type(plan) is not EpisodePlan:
        raise ExecutorConfigurationError("plan must be an exact EpisodePlan")
    if type(temperature) is not float or not math.isfinite(temperature) or temperature < 0.0:
        raise ExecutorConfigurationError("native tool temperature is invalid")
    if type(persist) is not bool:
        raise ExecutorConfigurationError("episode persistence flag is invalid")
    if history_profile not in {
        "SSB-OLLAMA-NATIVE-TOOLS1",
        "SSB-OLLAMA-NATIVE-TOOLS2",
        "SSB-OLLAMA-NATIVE-TOOLS3",
    }:
        raise ExecutorConfigurationError("native tool history profile is invalid")
    if turn_prompt_profile not in {None, NATIVE_TOOL_TURN_PROMPT_PROFILE} or (
        history_profile == "SSB-OLLAMA-NATIVE-TOOLS3"
    ) != (turn_prompt_profile == NATIVE_TOOL_TURN_PROMPT_PROFILE):
        raise ExecutorConfigurationError("native tool turn prompt profile is invalid")
    _identity_check(plan, client, request_profile=history_profile)
    if plan.manifest.status is not EpisodeStatus.PLANNED:
        raise ExecutorConfigurationError("episode manifest must be planned")

    authority_view = build_authority_evidence_view(
        plan.authority_records, plan.authority_query, plan.authority_decision
    )
    registry = ToolRegistry.for_domain(plan.manifest.domain, authority_view=authority_view)
    declarations = native_tool_definitions(registry.visible_specs(plan.manifest.condition))
    context = assemble_context(
        plan.manifest.condition,
        plan.task,
        plan.skill,
        plan.policy,
        plan.order_assignment,
    )
    state = WorldState.model_validate(plan.initial_state.model_dump(mode="json"))
    initial_hash = hash_state(state)
    make_snapshot(
        state,
        episode_id=plan.manifest.episode_id,
        snapshot_id=f"snapshot_{plan.manifest.episode_id}",
        observation={"status": "ready"},
    )
    cursor = EventCursor(episode_id=plan.manifest.episode_id, next_index=0)
    history: list[Message] = []
    native_history: list[NativeToolHistoryTurn] = []
    turn_prompt = (
        ()
        if turn_prompt_profile is None
        else (Message(role="developer", content=NATIVE_TOOL_TURN_PROMPT),)
    )
    turns: list[ExecutionTurn] = []
    events: list[StateEvent] = []
    observations: list[JsonObject] = []
    claim = AgentClaim.NONE
    disposition = EpisodeDisposition.BUDGET_EXHAUSTED
    error_code: str | None = None
    input_tokens = output_tokens = total_tokens = model_calls = tool_calls = 0
    costs: list[TokenCost] = []

    for turn_index in range(plan.manifest.budgets.max_turns):
        remaining_tokens = plan.manifest.budgets.max_tokens - total_tokens
        if remaining_tokens <= 0:
            error_code = "AGENT_BUDGET_EXHAUSTED"
            break
        observation = _ready_observation(
            registry, plan, turns=turn_index, calls=tool_calls, tokens=total_tokens
        )
        if observations:
            observation["last_tool_observation"] = observations[-1]
        request = ModelRequest(
            messages=tuple(context.messages)
            + turn_prompt
            + tuple(history)
            + (_json_message("OBSERVATION", observation),),
            temperature=temperature,
            seed=plan.manifest.seed,
            max_tokens=max(1, min(remaining_tokens, 1_000_000)),
        )
        try:
            response = (
                await client.call_tools(request, declarations, history=tuple(native_history))
                if history_profile != "SSB-OLLAMA-NATIVE-TOOLS1"
                else await client.call_tools(request, declarations)
            )
        except ModelAdapterError as error:
            model_calls += 1
            if error.reported_usage is not None:
                input_tokens += error.reported_usage.input_tokens
                output_tokens += error.reported_usage.output_tokens
                total_tokens += error.reported_usage.total_tokens
            turns.append(
                ExecutionTurn(
                    turn_index=turn_index,
                    output=None,
                    receipt=ModelCallReceipt(
                        turn_index=turn_index,
                        raw_request_hash=error.raw_request_hash,
                        raw_response_hash=error.raw_response_hash,
                        attempts=error.attempts,
                        error_code=error.code,
                        usage=error.reported_usage,
                        finish_reason=error.finish_reason,
                        before_meaningful_behavior=error.before_meaningful_behavior,
                        output_cap_exhausted=(
                            None
                            if error.reported_usage is None
                            else error.reported_usage.output_tokens >= request.max_tokens
                        ),
                    ),
                )
            )
            error_code = error.code
            disposition = (
                EpisodeDisposition.INVALID_ACTION
                if error.code == "MODEL_OUTPUT_INVALID"
                else EpisodeDisposition.MODEL_FAILURE
            )
            break

        model_calls += 1
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        total_tokens += response.usage.total_tokens
        if response.cost is not None:
            costs.append(response.cost)
        receipt = ModelCallReceipt(
            turn_index=turn_index,
            raw_request_hash=response.raw_request_hash,
            raw_response_hash=response.raw_response_hash,
            action_interface_hash=response.tool_declaration_hash,
            usage=response.usage,
            cost=response.cost,
            attempts=response.attempts,
            finish_reason=response.finish_reason,
            output_cap_exhausted=response.usage.output_tokens >= request.max_tokens,
        )
        try:
            action = ActionCall(
                action_id=f"native_action_{turn_index}",
                tool_name=response.tool_name,
                arguments=response.arguments,
            )
            output = AgentTurn(kind="tool_call", action=action)
        except ValueError:
            error_code = "MODEL_OUTPUT_INVALID"
            disposition = EpisodeDisposition.INVALID_ACTION
            turns.append(ExecutionTurn(turn_index=turn_index, output=None, receipt=receipt))
            break
        if total_tokens > plan.manifest.budgets.max_tokens:
            error_code = "AGENT_BUDGET_EXHAUSTED"
            disposition = EpisodeDisposition.BUDGET_EXHAUSTED
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break
        if tool_calls >= plan.manifest.budgets.max_tool_calls:
            error_code = "AGENT_BUDGET_EXHAUSTED"
            disposition = EpisodeDisposition.BUDGET_EXHAUSTED
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break
        try:
            checked = registry.validate_call(action, condition=plan.manifest.condition)
        except ToolRegistryError:
            error_code = "AGENT_INVALID_ACTION"
            disposition = EpisodeDisposition.INVALID_ACTION
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break

        gate_decision = None
        try:
            if (
                plan.manifest.condition is ExperimentCondition.B3_DETERMINISTIC_GATE
                and checked.tool_name != "resolve_authority"
            ):
                gate_decision = gate_action(
                    state,
                    checked,
                    plan.task,
                    plan.authority_records,
                    query=plan.authority_query,
                )
            step = (
                registry.execute(state, checked, cursor=cursor, condition=plan.manifest.condition)
                if gate_decision is None
                or gate_decision.outcome not in {GateOutcome.BLOCK, GateOutcome.ESCALATE}
                else execute_action(state, checked, _GateBlockAdapter(gate_decision), cursor=cursor)
            )
        except (GateInputError, ToolRegistryError):
            error_code = "AGENT_INVALID_ACTION"
            disposition = EpisodeDisposition.INVALID_ACTION
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break
        except EnvironmentInvariantFailure:
            error_code = "ENVIRONMENT_INVARIANT_FAILURE"
            disposition = EpisodeDisposition.ENVIRONMENT_FAILURE
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break

        tool_calls += 1
        state, cursor = step.state, step.cursor
        events.append(step.event)
        observations.append(step.result.observation)
        if history_profile != "SSB-OLLAMA-NATIVE-TOOLS1":
            native_history.append(
                NativeToolHistoryTurn(
                    tool_name=checked.tool_name,
                    arguments=checked.arguments,
                    result=step.result.model_dump(mode="json"),
                )
            )
        else:
            history.append(
                Message(
                    role="assistant",
                    content=canonical_json_bytes(output.model_dump(mode="json")).decode("utf-8"),
                )
            )
            history.append(_json_message("TOOL_RESULT", step.result.model_dump(mode="json")))
        turns.append(
            ExecutionTurn(
                turn_index=turn_index,
                output=output,
                receipt=receipt,
                action=checked,
                result=step.result,
                event=step.event,
                gate_decision=gate_decision,
            )
        )
        if checked.tool_name == "finish_task" and step.result.local_status == "success":
            claim = AgentClaim.FINISH
            disposition = EpisodeDisposition.COMPLETED
            break
    else:
        error_code = "AGENT_BUDGET_EXHAUSTED"

    verifier = _verify(plan, state)
    terminal = make_terminal_record(
        episode_id=plan.manifest.episode_id,
        initial_state_hash=initial_hash,
        terminal_state_hash=hash_state(state),
        event_root_hash=events[-1].event_hash if events else None,
        final_observation=observations[-1] if observations else {"status": "ready"},
        verdict=verifier,
    )
    overhead = ExecutionOverhead(
        model_calls=model_calls,
        turns=len(turns),
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        context_tokens=context.token_count,
        tool_schema_tokens=_token_count(
            canonical_json_bytes([item.projection() for item in declarations]).decode("utf-8")
        ),
        history_tokens=(
            sum(_token_count(message.content) for message in history)
            if history_profile == "SSB-OLLAMA-NATIVE-TOOLS1"
            else sum(
                _token_count(turn.tool_name)
                + _token_count(canonical_json_bytes(turn.arguments).decode("utf-8"))
                + _token_count(canonical_json_bytes(turn.result).decode("utf-8"))
                for turn in native_history
            )
        ),
        cost=_aggregate_cost(costs),
    )
    values: dict[str, object] = {
        "episode_id": plan.manifest.episode_id,
        "manifest_hash": sha256_ref(plan.manifest.model_dump(mode="json")),
        "initial_state_hash": initial_hash,
        "terminal_state": state,
        "terminal_state_hash": hash_state(state),
        "domain_case": plan.domain_case,
        "authority_decision": plan.authority_decision,
        "trace": tuple(turns),
        "terminal_record": terminal,
        "objective_verdict": verifier,
        "claim": claim,
        "disposition": disposition,
        "overhead": overhead,
        "error_code": error_code,
    }
    content_hash = sha256_ref(episode_result_projection(_json_ready(values)))
    result = EpisodeResult(
        **values, content_hash=content_hash, artifact_ref=_artifact_ref(content_hash)
    )
    if persist:
        _persist(result)
    return result


__all__ = ["run_native_tool_episode"]
