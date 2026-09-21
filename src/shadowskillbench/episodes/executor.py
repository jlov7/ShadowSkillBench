from __future__ import annotations

import json
import math
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from shadowskillbench.authority import GateDecision, GateInputError, GateOutcome, gate_action
from shadowskillbench.authority.models import AuthorityDecision
from shadowskillbench.authority.view import build_authority_evidence_view
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.domains.access.models import AccessTaskCase, parse_access_world
from shadowskillbench.domains.access.verifier import verify_access_case
from shadowskillbench.domains.finance.models import FinanceTaskCase, parse_finance_world
from shadowskillbench.domains.finance.verifier import verify_finance_case
from shadowskillbench.engine import (
    ActionCall,
    ActionResult,
    EnvironmentInvariantFailure,
    EventCursor,
    JsonObject,
    OutcomeVerdict,
    StateEvent,
    TerminalRecord,
    TransitionProposal,
    WorldState,
    execute_action,
    hash_action,
    hash_state,
    make_snapshot,
    make_terminal_record,
)
from shadowskillbench.episodes.context import ContextBundle, assemble_context
from shadowskillbench.episodes.models import EpisodePlan, EpisodeStatus, ExperimentCondition
from shadowskillbench.episodes.pilot_turn_wire import (
    CAPABILITY_TURN_PROMPT,
    CONFIRMATORY_TURN_PROMPT,
    NAMED_ACTION_TURN_PROMPT,
    PILOT_TURN_PROMPT,
    CapabilityActiveTurnWire,
    CapabilityAgentTurnWire,
    CapabilityFinishedTurnWire,
    CapabilityInitialTurnWire,
    ConfirmatoryActiveTurnWire,
    ConfirmatoryAgentTurnWire,
    ConfirmatoryFinishedTurnWire,
    ConfirmatoryInitialToolTurnWire,
    NamedActionActiveTurnWire,
    NamedActionAgentTurnWire,
    NamedActionFinishedTurnWire,
    NamedActionInitialTurnWire,
    PilotAgentTurnWire,
    named_action_turn_to_agent_turn_data,
    pilot_turn_to_agent_turn_data,
    pilot_turn_tool_layout,
)
from shadowskillbench.episodes.tools import ToolRegistry, ToolRegistryError
from shadowskillbench.models.protocol import (
    Message,
    ModelAdapterError,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    TokenCost,
    TokenUsage,
)


class AgentClaim(StrEnum):
    NONE = "none"
    FINISH = "finish"
    ESCALATE = "escalate"
    REFUSE = "refuse"


class EpisodeDisposition(StrEnum):
    COMPLETED = "completed"
    ESCALATED = "escalated"
    INVALID_ACTION = "invalid_action"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODEL_FAILURE = "model_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    REFUSED = "refused"


class _ExecutorModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _bounded_text(value: object, *, field: str, maximum: int = 512) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be bounded nonblank text")
    if any(ord(char) < 32 and char not in "\t" for char in value):
        raise ValueError(f"{field} contains control characters")
    return value


class AgentTurn(_ExecutorModel):
    kind: Literal["tool_call", "finish", "escalate", "refuse"]
    action: ActionCall | None = None
    summary: str | None = None

    @field_validator("action", mode="before")
    @classmethod
    def _action(cls, value: object) -> ActionCall | None:
        if value is None:
            return None
        if type(value) is dict:
            return ActionCall.model_validate(value, strict=True)
        if type(value) is not ActionCall:
            raise ValueError("action must be an exact ActionCall")
        return ActionCall.model_validate(value.model_dump(mode="json"))

    @field_validator("summary", mode="before")
    @classmethod
    def _summary(cls, value: object) -> str | None:
        return None if value is None else _bounded_text(value, field="summary")

    @model_validator(mode="after")
    def _combination(self) -> AgentTurn:
        if self.kind == "tool_call" and (self.action is None or self.summary is not None):
            raise ValueError("tool_call requires action and forbids summary")
        if self.kind in {"finish", "escalate", "refuse"} and self.action is not None:
            raise ValueError("terminal claim forbids action")
        if self.kind in {"finish", "refuse"} and self.summary is None:
            raise ValueError(f"{self.kind} requires summary")
        return self


class ModelCallReceipt(_ExecutorModel):
    turn_index: int
    raw_request_hash: str | None = None
    raw_response_hash: str | None = None
    structured_output_schema_hash: str | None = None
    action_interface_hash: str | None = None
    usage: TokenUsage | None = None
    cost: TokenCost | None = None
    attempts: int
    error_code: str | None = None
    finish_reason: str | None = None
    before_meaningful_behavior: bool | None = None
    output_cap_exhausted: bool | None = None


class ExecutionTurn(_ExecutorModel):
    turn_index: int
    output: AgentTurn | None
    receipt: ModelCallReceipt
    action: ActionCall | None = None
    result: ActionResult | None = None
    event: StateEvent | None = None
    gate_decision: GateDecision | None = None


class ExecutionOverhead(_ExecutorModel):
    model_calls: int
    turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_tokens: int
    tool_schema_tokens: int
    history_tokens: int
    cost: TokenCost | None = None


def _artifact_ref(content_hash: str) -> str:
    digest = content_hash.removeprefix("sha256:")
    return f"artifacts/episode_result/{digest[:2]}/{digest}.json"


def episode_result_projection(values: dict[str, object]) -> dict[str, object]:
    def compatible(value: object) -> object:
        if type(value) is list:
            return [compatible(item) for item in value]
        if type(value) is dict:
            return {
                key: compatible(item)
                for key, item in value.items()
                # Historical structured receipts predate the native-interface
                # binding.  Omit only its absent value so their custody hash
                # remains reproducible; native receipts always carry a hash.
                if not (key == "action_interface_hash" and item is None)
            }
        return value

    return compatible(
        {key: value for key, value in values.items() if key not in {"content_hash", "artifact_ref"}}
    )  # type: ignore[return-value]


def _json_ready(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_ready(value.model_dump(mode="json"))
    if type(value) is tuple:
        return [_json_ready(item) for item in value]
    if type(value) is list:
        return [_json_ready(item) for item in value]
    if type(value) is dict:
        return {key: _json_ready(item) for key, item in value.items()}
    return value


_ASCII_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_IGNORED_TOKEN_CHARACTERS = frozenset("\t\n\v\f\r ")


def _token_count(text: str) -> int:
    count = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character in _ASCII_WORD:
            count += 1
            index += 1
            while index < len(text) and text[index] in _ASCII_WORD:
                index += 1
        elif character in _IGNORED_TOKEN_CHARACTERS:
            index += 1
        else:
            count += 1
            index += 1
    return count


class EpisodeResult(_ExecutorModel):
    episode_id: str
    manifest_hash: str
    initial_state_hash: str
    terminal_state: WorldState
    terminal_state_hash: str
    domain_case: AccessTaskCase | FinanceTaskCase
    authority_decision: AuthorityDecision
    trace: tuple[ExecutionTurn, ...]
    terminal_record: TerminalRecord
    objective_verdict: OutcomeVerdict
    claim: AgentClaim
    disposition: EpisodeDisposition
    overhead: ExecutionOverhead
    error_code: str | None = None
    content_hash: str
    artifact_ref: str

    @field_validator("domain_case", mode="before")
    @classmethod
    def _domain_case(cls, value: object) -> AccessTaskCase | FinanceTaskCase:
        if type(value) is AccessTaskCase:
            return AccessTaskCase.model_validate(value.model_dump(mode="json"))
        if type(value) is FinanceTaskCase:
            return FinanceTaskCase.model_validate(value.model_dump(mode="json"))
        raise ValueError("domain_case must be an exact typed domain case")

    @field_validator("authority_decision", mode="before")
    @classmethod
    def _authority_decision(cls, value: object) -> AuthorityDecision:
        if type(value) is not AuthorityDecision:
            raise ValueError("authority_decision must be an exact AuthorityDecision")
        return AuthorityDecision.model_validate(value.model_dump(mode="json"))

    @model_validator(mode="after")
    def _bind(self) -> EpisodeResult:
        values = self.model_dump(mode="json")
        if any(turn.turn_index != index for index, turn in enumerate(self.trace)):
            raise ValueError("trace turn indices must be contiguous")
        previous_event: StateEvent | None = None
        expected_event_index = 0
        action_turns = 0
        input_tokens = output_tokens = total_tokens = 0
        for turn in self.trace:
            present = (turn.action is not None, turn.result is not None, turn.event is not None)
            if any(present) and not all(present):
                raise ValueError("trace action, result, and event must be recorded together")
            if all(present):
                if turn.output is None or turn.output.action != turn.action:
                    raise ValueError("trace action does not bind model output")
                if turn.event is None or turn.result is None or turn.action is None:
                    raise ValueError("trace action record is incomplete")
                if turn.event.result != turn.result:
                    raise ValueError("trace event does not bind action result")
                if turn.event.action_id != turn.action.action_id:
                    raise ValueError("trace event does not bind action")
                if turn.gate_decision is not None:
                    if turn.gate_decision.action_hash != hash_action(turn.action):
                        raise ValueError("gate decision does not bind action")
                    if turn.gate_decision.state_hash != turn.event.before_hash:
                        raise ValueError("gate decision does not bind pre-action state")
                if turn.event.index != expected_event_index:
                    raise ValueError("trace event indices must be contiguous")
                if previous_event is None:
                    if (
                        turn.event.parent_event_id is not None
                        or turn.event.parent_event_hash is not None
                    ):
                        raise ValueError("first trace event cannot have a parent")
                elif (
                    turn.event.parent_event_id != previous_event.event_id
                    or turn.event.parent_event_hash != previous_event.event_hash
                ):
                    raise ValueError("trace event parent does not bind prior event")
                previous_event = turn.event
                expected_event_index += 1
                action_turns += 1
            usage = turn.receipt.usage
            if usage is not None:
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
                total_tokens += usage.total_tokens
        if self.terminal_record.initial_state_hash != self.initial_state_hash:
            raise ValueError("terminal record does not bind initial state")
        expected_root = None if previous_event is None else previous_event.event_hash
        if self.terminal_record.event_root_hash != expected_root:
            raise ValueError("terminal record does not bind event trace")
        if (
            self.overhead.model_calls != len(self.trace)
            or self.overhead.turns != len(self.trace)
            or self.overhead.tool_calls != action_turns
            or self.overhead.input_tokens != input_tokens
            or self.overhead.output_tokens != output_tokens
            or self.overhead.total_tokens != total_tokens
        ):
            raise ValueError("execution overhead does not bind trace")
        expected_disposition = {
            AgentClaim.FINISH: EpisodeDisposition.COMPLETED,
            AgentClaim.ESCALATE: EpisodeDisposition.ESCALATED,
            AgentClaim.REFUSE: EpisodeDisposition.REFUSED,
        }.get(self.claim)
        if expected_disposition is not None and self.disposition is not expected_disposition:
            raise ValueError("claim does not bind disposition")
        if self.claim is AgentClaim.NONE and self.disposition not in {
            EpisodeDisposition.INVALID_ACTION,
            EpisodeDisposition.BUDGET_EXHAUSTED,
            EpisodeDisposition.MODEL_FAILURE,
            EpisodeDisposition.ENVIRONMENT_FAILURE,
        }:
            raise ValueError("none claim requires a nonterminal failure disposition")
        if self.terminal_state_hash != hash_state(self.terminal_state):
            raise ValueError("terminal_state_hash does not bind terminal_state")
        if self.terminal_record.episode_id != self.episode_id:
            raise ValueError("terminal record does not bind episode")
        if self.terminal_record.terminal_state_hash != self.terminal_state_hash:
            raise ValueError("terminal record does not bind terminal state")
        if self.terminal_state.domain == "access_provisioning":
            expected_case_type = AccessTaskCase
        elif self.terminal_state.domain == "financial_adjustments":
            expected_case_type = FinanceTaskCase
        else:
            raise ValueError("terminal state has an unsupported domain")
        if type(self.domain_case) is not expected_case_type:
            raise ValueError("domain_case does not bind terminal domain")
        if self.domain_case.case_id != self.terminal_state.data.get("case_id"):
            raise ValueError("domain_case does not bind terminal case")
        if self.domain_case.world_id != self.terminal_state.world_id:
            raise ValueError("domain_case does not bind terminal world")
        if self.authority_decision.domain != self.terminal_state.domain:
            raise ValueError("authority_decision does not bind terminal domain")
        decision_hash = self.objective_verdict.details.get("authority_decision_hash")
        if decision_hash != self.authority_decision.decision_hash:
            raise ValueError("objective verdict does not bind authority decision")
        if self.objective_verdict != self.terminal_record.verdict:
            raise ValueError("objective_verdict does not bind terminal record")
        if self.content_hash != sha256_ref(episode_result_projection(values)):
            raise ValueError("content_hash does not bind result")
        if self.artifact_ref != _artifact_ref(self.content_hash):
            raise ValueError("artifact_ref does not bind content_hash")
        return self


def parse_episode_result(value: object) -> EpisodeResult:
    """Rebuild a persisted episode result through its full semantic validators."""
    if type(value) is not dict:
        raise ValueError("episode result must be an exact mapping")
    data = cast(dict[str, object], value)
    if set(data) != set(EpisodeResult.model_fields):
        raise ValueError("episode result fields are invalid")

    def mapping(
        item: object, label: str, expected_fields: set[str] | None = None
    ) -> dict[str, object]:
        if type(item) is not dict:
            raise ValueError(f"{label} must be an exact mapping")
        parsed = cast(dict[str, object], item)
        if expected_fields is not None and set(parsed) != expected_fields:
            raise ValueError(f"{label} fields are invalid")
        return parsed

    def token_usage(item: object) -> TokenUsage | None:
        return None if item is None else TokenUsage.model_validate(item)

    def token_cost(item: object) -> TokenCost | None:
        return None if item is None else TokenCost.model_validate(item)

    terminal_state = WorldState.model_validate(data.get("terminal_state"))
    domain_case_data = data.get("domain_case")
    domain_case = (
        AccessTaskCase.model_validate(domain_case_data)
        if terminal_state.domain == "access_provisioning"
        else FinanceTaskCase.model_validate(domain_case_data)
    )
    parsed_turns: list[ExecutionTurn] = []
    trace = data.get("trace")
    if type(trace) is not list:
        raise ValueError("trace must be an exact list")
    for index, raw_turn in enumerate(trace):
        turn = mapping(raw_turn, "trace entry", set(ExecutionTurn.model_fields))
        expected_receipt_fields = frozenset(ModelCallReceipt.model_fields)
        receipt_data = mapping(turn.get("receipt"), "model receipt")
        if frozenset(receipt_data) not in {
            expected_receipt_fields,
            expected_receipt_fields - {"action_interface_hash"},
        }:
            raise ValueError("model receipt fields are invalid")
        receipt = ModelCallReceipt(
            turn_index=cast(int, receipt_data.get("turn_index")),
            raw_request_hash=cast(str | None, receipt_data.get("raw_request_hash")),
            raw_response_hash=cast(str | None, receipt_data.get("raw_response_hash")),
            structured_output_schema_hash=cast(
                str | None, receipt_data.get("structured_output_schema_hash")
            ),
            action_interface_hash=cast(str | None, receipt_data.get("action_interface_hash")),
            usage=token_usage(receipt_data.get("usage")),
            cost=token_cost(receipt_data.get("cost")),
            attempts=cast(int, receipt_data.get("attempts")),
            error_code=cast(str | None, receipt_data.get("error_code")),
            finish_reason=cast(str | None, receipt_data.get("finish_reason")),
            before_meaningful_behavior=cast(
                bool | None, receipt_data.get("before_meaningful_behavior")
            ),
            output_cap_exhausted=cast(bool | None, receipt_data.get("output_cap_exhausted")),
        )
        output_data = turn.get("output")
        action_data = turn.get("action")
        result_data = turn.get("result")
        event_data = turn.get("event")
        gate_data = turn.get("gate_decision")
        event = None
        if event_data is not None:
            event_mapping = mapping(event_data, "state event")
            delta = event_mapping.get("delta")
            if type(delta) is not list:
                raise ValueError("state event delta must be an exact list")
            event = StateEvent.model_validate({**event_mapping, "delta": tuple(delta)})
        parsed_turns.append(
            ExecutionTurn(
                turn_index=cast(int, turn.get("turn_index")),
                output=None if output_data is None else AgentTurn.model_validate(output_data),
                receipt=receipt,
                action=None if action_data is None else ActionCall.model_validate(action_data),
                result=None if result_data is None else ActionResult.model_validate(result_data),
                event=event,
                gate_decision=None if gate_data is None else GateDecision.model_validate(gate_data),
            )
        )
    overhead_data = mapping(
        data.get("overhead"), "execution overhead", set(ExecutionOverhead.model_fields)
    )
    overhead = ExecutionOverhead(
        model_calls=cast(int, overhead_data.get("model_calls")),
        turns=cast(int, overhead_data.get("turns")),
        tool_calls=cast(int, overhead_data.get("tool_calls")),
        input_tokens=cast(int, overhead_data.get("input_tokens")),
        output_tokens=cast(int, overhead_data.get("output_tokens")),
        total_tokens=cast(int, overhead_data.get("total_tokens")),
        context_tokens=cast(int, overhead_data.get("context_tokens")),
        tool_schema_tokens=cast(int, overhead_data.get("tool_schema_tokens")),
        history_tokens=cast(int, overhead_data.get("history_tokens")),
        cost=token_cost(overhead_data.get("cost")),
    )
    return EpisodeResult(
        episode_id=cast(str, data.get("episode_id")),
        manifest_hash=cast(str, data.get("manifest_hash")),
        initial_state_hash=cast(str, data.get("initial_state_hash")),
        terminal_state=terminal_state,
        terminal_state_hash=cast(str, data.get("terminal_state_hash")),
        domain_case=domain_case,
        authority_decision=AuthorityDecision.model_validate(data.get("authority_decision")),
        trace=tuple(parsed_turns),
        terminal_record=TerminalRecord.model_validate(data.get("terminal_record")),
        objective_verdict=OutcomeVerdict.model_validate(data.get("objective_verdict")),
        claim=AgentClaim(data.get("claim")),
        disposition=EpisodeDisposition(data.get("disposition")),
        overhead=overhead,
        error_code=cast(str | None, data.get("error_code")),
        content_hash=cast(str, data.get("content_hash")),
        artifact_ref=cast(str, data.get("artifact_ref")),
    )


class ExecutorConfigurationError(ValueError):
    pass


class _GateBlockAdapter:
    def __init__(self, decision: GateDecision) -> None:
        self._decision = decision

    def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
        return TransitionProposal(
            local_status=("failure" if self._decision.outcome is GateOutcome.BLOCK else "clarify"),
            next_state=None,
            observation={
                "gate_hash": self._decision.gate_hash,
                "outcome": self._decision.outcome.value,
            },
            error_code=(
                "AUTHORITY_GATE_BLOCKED"
                if self._decision.outcome is GateOutcome.BLOCK
                else "AUTHORITY_GATE_ESCALATED"
            ),
        )


def _identity_check(plan: EpisodePlan, client: ModelClient) -> None:
    capabilities = client.capabilities
    if type(capabilities) is not ProviderCapabilities:
        raise ExecutorConfigurationError("client capabilities are invalid")
    expected = plan.manifest.executor_model
    if (
        capabilities.provider != expected.provider
        or capabilities.model != expected.model
        or capabilities.model_version != expected.model_version_date
        or not capabilities.supports_structured_output
        or not capabilities.supports_seed
    ):
        raise ExecutorConfigurationError("client capabilities do not match manifest")


def _json_message(label: str, value: object) -> Message:
    return Message(role="user", content=f"{label}\n{canonical_json_bytes(value).decode('utf-8')}")


def _ready_observation(
    registry: ToolRegistry, plan: EpisodePlan, *, turns: int, calls: int, tokens: int
) -> dict[str, object]:
    return {
        "status": "ready",
        "available_tools": registry.visible_projection(plan.manifest.condition),
        "remaining": {
            "turns": max(plan.manifest.budgets.max_turns - turns, 0),
            "tool_calls": max(plan.manifest.budgets.max_tool_calls - calls, 0),
            "tokens": max(plan.manifest.budgets.max_tokens - tokens, 0),
        },
    }


def _verify(plan: EpisodePlan, state: WorldState) -> OutcomeVerdict:
    if plan.manifest.domain == "access_provisioning":
        if type(plan.domain_case) is not AccessTaskCase:
            raise ExecutorConfigurationError("access episode has the wrong typed case")
        return verify_access_case(
            parse_access_world(state),
            cast(AccessTaskCase, plan.domain_case),
            authority_decision=plan.authority_decision,
        )
    if type(plan.domain_case) is not FinanceTaskCase:
        raise ExecutorConfigurationError("finance episode has the wrong typed case")
    return verify_finance_case(
        parse_finance_world(state),
        cast(FinanceTaskCase, plan.domain_case),
        authority_decision=plan.authority_decision,
    )


def _persist(result: EpisodeResult) -> None:
    path = Path.cwd() / result.artifact_ref
    path.parent.mkdir(parents=True, exist_ok=True)
    projection = episode_result_projection(result.model_dump(mode="json"))
    if sha256_ref(projection) != result.content_hash:
        raise RuntimeError("episode artifact projection does not bind content hash")
    payload = canonical_json_bytes(
        {
            "content_hash": result.content_hash,
            "artifact_ref": result.artifact_ref,
            "payload": projection,
        }
    )
    if path.exists():
        try:
            existing = json.loads(path.read_bytes())
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError("existing episode artifact is not valid JSON") from error
        if type(existing) is not dict or set(existing) != {
            "content_hash",
            "artifact_ref",
            "payload",
        }:
            raise RuntimeError("existing episode artifact envelope is invalid")
        if (
            existing["content_hash"] != result.content_hash
            or existing["artifact_ref"] != result.artifact_ref
            or type(existing["payload"]) is not dict
            or sha256_ref(existing["payload"]) != existing["content_hash"]
        ):
            raise RuntimeError("existing episode artifact payload does not bind content hash")
        if path.read_bytes() != payload:
            raise RuntimeError("existing episode artifact differs from content hash")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=".episode-result-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = json.loads(path.read_bytes())
            except (OSError, ValueError, TypeError) as error:
                raise RuntimeError("existing episode artifact is not valid JSON") from error
            if (
                type(existing) is not dict
                or set(existing) != {"content_hash", "artifact_ref", "payload"}
                or existing.get("content_hash") != result.content_hash
                or existing.get("artifact_ref") != result.artifact_ref
                or type(existing.get("payload")) is not dict
                or sha256_ref(existing["payload"]) != existing["content_hash"]
            ):
                raise RuntimeError("existing episode artifact payload does not bind content hash")
            if path.read_bytes() != payload:
                raise RuntimeError("existing episode artifact differs from content hash")
        finally:
            os.unlink(temporary)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _aggregate_cost(costs: list[TokenCost]) -> TokenCost | None:
    if not costs:
        return None
    first = costs[0]
    if any(
        (item.input_nanos_per_token, item.output_nanos_per_token)
        != (first.input_nanos_per_token, first.output_nanos_per_token)
        for item in costs[1:]
    ):
        return None
    input_nanos = sum(item.input_nanos for item in costs)
    output_nanos = sum(item.output_nanos for item in costs)
    return TokenCost(
        currency="USD",
        input_nanos_per_token=first.input_nanos_per_token,
        output_nanos_per_token=first.output_nanos_per_token,
        input_nanos=input_nanos,
        output_nanos=output_nanos,
        total_nanos=input_nanos + output_nanos,
    )


async def run_episode(
    plan: EpisodePlan,
    client: ModelClient,
    *,
    output_schema: type[BaseModel] = AgentTurn,
    temperature: float = 0.0,
    persist: bool = True,
) -> EpisodeResult:
    if type(plan) is not EpisodePlan:
        raise ExecutorConfigurationError("plan must be an exact EpisodePlan")
    if output_schema not in {
        AgentTurn,
        PilotAgentTurnWire,
        ConfirmatoryAgentTurnWire,
        CapabilityAgentTurnWire,
        NamedActionAgentTurnWire,
    }:
        raise ExecutorConfigurationError("episode output schema is invalid")
    if (
        type(temperature) is not float
        or not math.isfinite(temperature)
        or temperature < 0.0
        or temperature == 0.0
        and math.copysign(1.0, temperature) < 0.0
    ):
        raise ExecutorConfigurationError("episode temperature is invalid")
    if type(persist) is not bool:
        raise ExecutorConfigurationError("episode persistence flag is invalid")
    _identity_check(plan, client)
    if plan.manifest.status is not EpisodeStatus.PLANNED:
        raise ExecutorConfigurationError("episode manifest must be planned")
    authority_view = build_authority_evidence_view(
        plan.authority_records, plan.authority_query, plan.authority_decision
    )
    registry = ToolRegistry.for_domain(
        plan.manifest.domain,
        authority_view=authority_view,
    )
    turn_layout = (
        None
        if output_schema in {AgentTurn, NamedActionAgentTurnWire}
        else pilot_turn_tool_layout(registry, plan.manifest.condition)
    )
    turn_prompt = (
        ()
        if output_schema is AgentTurn
        else (
            Message(
                role="developer",
                content=(
                    PILOT_TURN_PROMPT
                    if output_schema is PilotAgentTurnWire
                    else CAPABILITY_TURN_PROMPT
                    if output_schema is CapabilityAgentTurnWire
                    else NAMED_ACTION_TURN_PROMPT
                    if output_schema is NamedActionAgentTurnWire
                    else CONFIRMATORY_TURN_PROMPT
                ),
            ),
        )
    )
    context: ContextBundle = assemble_context(
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
            disposition = EpisodeDisposition.BUDGET_EXHAUSTED
            error_code = "AGENT_BUDGET_EXHAUSTED"
            break
        observation = _ready_observation(
            registry,
            plan,
            turns=turn_index,
            calls=tool_calls,
            tokens=total_tokens,
        )
        if observations:
            observation["last_tool_observation"] = observations[-1]
        request_messages = (
            tuple(context.messages)
            + turn_prompt
            + tuple(history)
            + (_json_message("OBSERVATION", observation),)
        )
        request = ModelRequest(
            messages=request_messages,
            temperature=temperature,
            seed=plan.manifest.seed,
            max_tokens=max(1, min(remaining_tokens, 1_000_000)),
        )
        request_output_schema: type[BaseModel]
        if output_schema is ConfirmatoryAgentTurnWire:
            request_output_schema = (
                ConfirmatoryInitialToolTurnWire
                if turn_index == 0
                else ConfirmatoryFinishedTurnWire
                if state.data.get("finished") is True
                else ConfirmatoryActiveTurnWire
            )
        elif output_schema is CapabilityAgentTurnWire:
            request_output_schema = (
                CapabilityInitialTurnWire
                if turn_index == 0
                else CapabilityFinishedTurnWire
                if state.data.get("finished") is True
                else CapabilityActiveTurnWire
            )
        elif output_schema is NamedActionAgentTurnWire:
            request_output_schema = (
                NamedActionInitialTurnWire
                if turn_index == 0
                else NamedActionFinishedTurnWire
                if state.data.get("finished") is True
                else NamedActionActiveTurnWire
            )
        else:
            request_output_schema = output_schema
        try:
            response = await client.structured(request, request_output_schema)
        except ModelAdapterError as error:
            model_calls += 1
            if error.reported_usage is not None:
                input_tokens += error.reported_usage.input_tokens
                output_tokens += error.reported_usage.output_tokens
                total_tokens += error.reported_usage.total_tokens
            receipt = ModelCallReceipt(
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
            )
            turns.append(ExecutionTurn(turn_index=turn_index, output=None, receipt=receipt))
            error_code = error.code
            disposition = (
                EpisodeDisposition.INVALID_ACTION
                if error.code == "MODEL_OUTPUT_INVALID"
                else EpisodeDisposition.MODEL_FAILURE
            )
            break
        expected_response_type = ModelResponse[request_output_schema]
        if type(response) is not expected_response_type:
            raise ExecutorConfigurationError("client returned the wrong response schema")
        model_calls += 1
        usage = response.usage
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        total_tokens += usage.total_tokens
        if response.cost is not None:
            costs.append(response.cost)
        try:
            output_data = response.output.model_dump(mode="json")
            if output_schema in {
                PilotAgentTurnWire,
                ConfirmatoryAgentTurnWire,
                CapabilityAgentTurnWire,
            }:
                if turn_layout is None:
                    raise ValueError("pilot turn layout is unavailable")
                output_data = pilot_turn_to_agent_turn_data(
                    cast(
                        PilotAgentTurnWire
                        | ConfirmatoryAgentTurnWire
                        | ConfirmatoryInitialToolTurnWire
                        | ConfirmatoryActiveTurnWire
                        | ConfirmatoryFinishedTurnWire
                        | CapabilityAgentTurnWire
                        | CapabilityInitialTurnWire
                        | CapabilityActiveTurnWire
                        | CapabilityFinishedTurnWire,
                        response.output,
                    ),
                    turn_layout,
                    turn_index,
                )
            elif output_schema is NamedActionAgentTurnWire:
                output_data = named_action_turn_to_agent_turn_data(
                    cast(
                        NamedActionAgentTurnWire
                        | NamedActionInitialTurnWire
                        | NamedActionActiveTurnWire
                        | NamedActionFinishedTurnWire,
                        response.output,
                    ),
                    registry,
                    plan.manifest.condition,
                    turn_index,
                )
            checked_output = AgentTurn.model_validate(output_data, strict=True)
        except ValueError:
            receipt = ModelCallReceipt(
                turn_index=turn_index,
                raw_request_hash=response.raw_request_hash,
                raw_response_hash=response.raw_response_hash,
                structured_output_schema_hash=response.structured_output_schema_hash,
                usage=response.usage,
                cost=response.cost,
                attempts=response.attempts,
                error_code="MODEL_OUTPUT_INVALID",
                finish_reason="stop",
                before_meaningful_behavior=False,
                output_cap_exhausted=response.usage.output_tokens >= request.max_tokens,
            )
            turns.append(ExecutionTurn(turn_index=turn_index, output=None, receipt=receipt))
            error_code = "MODEL_OUTPUT_INVALID"
            disposition = EpisodeDisposition.INVALID_ACTION
            break
        receipt = ModelCallReceipt(
            turn_index=turn_index,
            raw_request_hash=response.raw_request_hash,
            raw_response_hash=response.raw_response_hash,
            structured_output_schema_hash=response.structured_output_schema_hash,
            usage=usage,
            cost=response.cost,
            attempts=response.attempts,
        )
        output = checked_output
        history_output = (
            cast(
                PilotAgentTurnWire
                | ConfirmatoryAgentTurnWire
                | ConfirmatoryInitialToolTurnWire
                | ConfirmatoryActiveTurnWire
                | ConfirmatoryFinishedTurnWire
                | CapabilityAgentTurnWire
                | CapabilityInitialTurnWire
                | CapabilityActiveTurnWire
                | CapabilityFinishedTurnWire
                | NamedActionAgentTurnWire
                | NamedActionInitialTurnWire
                | NamedActionActiveTurnWire
                | NamedActionFinishedTurnWire,
                response.output,
            )
            if output_schema
            in {
                PilotAgentTurnWire,
                ConfirmatoryAgentTurnWire,
                CapabilityAgentTurnWire,
                NamedActionAgentTurnWire,
            }
            else output
        )
        history.append(
            Message(
                role="assistant",
                content=canonical_json_bytes(history_output.model_dump(mode="json")).decode(
                    "utf-8"
                ),
            )
        )
        if total_tokens > plan.manifest.budgets.max_tokens:
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            disposition = EpisodeDisposition.BUDGET_EXHAUSTED
            error_code = "AGENT_BUDGET_EXHAUSTED"
            break
        if output.kind == "finish":
            claim = AgentClaim.FINISH
            disposition = EpisodeDisposition.COMPLETED
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break
        if output.kind == "escalate":
            claim = AgentClaim.ESCALATE
            disposition = EpisodeDisposition.ESCALATED
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break
        if output.kind == "refuse":
            claim = AgentClaim.REFUSE
            disposition = EpisodeDisposition.REFUSED
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break
        if output.action is None:
            error_code = "AGENT_INVALID_ACTION"
            disposition = EpisodeDisposition.INVALID_ACTION
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break
        if tool_calls >= plan.manifest.budgets.max_tool_calls:
            error_code = "AGENT_BUDGET_EXHAUSTED"
            disposition = EpisodeDisposition.BUDGET_EXHAUSTED
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break
        gate_decision: GateDecision | None = None
        try:
            if (
                plan.manifest.condition is ExperimentCondition.B3_DETERMINISTIC_GATE
                and output.action.tool_name != "resolve_authority"
            ):
                gate_decision = gate_action(
                    state,
                    output.action,
                    plan.task,
                    plan.authority_records,
                    query=plan.authority_query,
                )
            if gate_decision is not None and gate_decision.outcome in {
                GateOutcome.BLOCK,
                GateOutcome.ESCALATE,
            }:
                step = execute_action(
                    state,
                    output.action,
                    _GateBlockAdapter(gate_decision),
                    cursor=cursor,
                )
            else:
                step = registry.execute(
                    state,
                    output.action,
                    cursor=cursor,
                    condition=plan.manifest.condition,
                )
        except GateInputError:
            error_code = "AGENT_INVALID_ACTION"
            disposition = EpisodeDisposition.INVALID_ACTION
            turns.append(ExecutionTurn(turn_index=turn_index, output=output, receipt=receipt))
            break
        except ToolRegistryError:
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
        state = step.state
        cursor = step.cursor
        events.append(step.event)
        observations.append(step.result.observation)
        history.append(_json_message("TOOL_RESULT", step.result.model_dump(mode="json")))
        turns.append(
            ExecutionTurn(
                turn_index=turn_index,
                output=output,
                receipt=receipt,
                action=output.action,
                result=step.result,
                event=step.event,
                gate_decision=gate_decision,
            )
        )
    else:
        disposition = EpisodeDisposition.BUDGET_EXHAUSTED
        error_code = "AGENT_BUDGET_EXHAUSTED"

    verifier = _verify(plan, state)
    final_observation: JsonObject = observations[-1] if observations else {"status": "ready"}
    terminal = make_terminal_record(
        episode_id=plan.manifest.episode_id,
        initial_state_hash=initial_hash,
        terminal_state_hash=hash_state(state),
        event_root_hash=events[-1].event_hash if events else None,
        final_observation=final_observation,
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
            canonical_json_bytes(registry.visible_projection(plan.manifest.condition)).decode(
                "utf-8"
            )
        ),
        history_tokens=sum(_token_count(message.content) for message in history),
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
    content_hash = sha256_ref(
        episode_result_projection(cast(dict[str, object], _json_ready(values)))
    )
    result = EpisodeResult(
        **values,
        content_hash=content_hash,
        artifact_ref=_artifact_ref(content_hash),
    )
    if persist:
        _persist(result)
    return result


__all__ = [
    "AgentClaim",
    "AgentTurn",
    "EpisodeDisposition",
    "EpisodeResult",
    "ExecutionOverhead",
    "ExecutionTurn",
    "ExecutorConfigurationError",
    "ModelCallReceipt",
    "episode_result_projection",
    "parse_episode_result",
    "run_episode",
]
