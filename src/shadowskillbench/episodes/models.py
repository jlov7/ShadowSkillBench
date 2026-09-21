from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from shadowskillbench.authority.models import (
    AccessAuthorityQuery,
    AuthorityDecision,
    AuthorityRecord,
    FinanceAuthorityQuery,
)
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.domains.access.models import AccessTaskCase
from shadowskillbench.domains.finance.models import FinanceTaskCase
from shadowskillbench.engine import TaskCase, WorldState, hash_state
from shadowskillbench.policy.documents import RenderedPolicy
from shadowskillbench.skills.compiler import CompiledSkillArtifact

type Sha256Ref = str

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOMAINS = frozenset({"access_provisioning", "financial_adjustments"})
_MAX_I64 = 2**63 - 1


class ExperimentCondition(StrEnum):
    A0_BARE = "A0_BARE"
    A1_POLICY_ONLY_SYSTEM = "A1_POLICY_ONLY_SYSTEM"
    A2_SKILL_ONLY = "A2_SKILL_ONLY"
    A3_SKILL_POLICY_SAME_TIER = "A3_SKILL_POLICY_SAME_TIER"
    A4_SKILL_POLICY_SYSTEM_TIER = "A4_SKILL_POLICY_SYSTEM_TIER"
    A5_SKILL_BURIED_POLICY_SAME_TIER = "A5_SKILL_BURIED_POLICY_SAME_TIER"
    B0_SKILL_ONLY = "B0_SKILL_ONLY"
    B1_FLAT_POLICY_SYSTEM = "B1_FLAT_POLICY_SYSTEM"
    B2_AUTHORITY_RESOLVER = "B2_AUTHORITY_RESOLVER"
    B3_DETERMINISTIC_GATE = "B3_DETERMINISTIC_GATE"

    @property
    def requires_skill(self) -> bool:
        return self not in {
            ExperimentCondition.A0_BARE,
            ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
        }

    @property
    def requires_policy(self) -> bool:
        return self in {
            ExperimentCondition.A1_POLICY_ONLY_SYSTEM,
            ExperimentCondition.A3_SKILL_POLICY_SAME_TIER,
            ExperimentCondition.A4_SKILL_POLICY_SYSTEM_TIER,
            ExperimentCondition.A5_SKILL_BURIED_POLICY_SAME_TIER,
            ExperimentCondition.B1_FLAT_POLICY_SYSTEM,
            ExperimentCondition.B2_AUTHORITY_RESOLVER,
            ExperimentCondition.B3_DETERMINISTIC_GATE,
        }

    @property
    def uses_authority_resolver(self) -> bool:
        return self is ExperimentCondition.B2_AUTHORITY_RESOLVER

    @property
    def uses_deterministic_gate(self) -> bool:
        return self is ExperimentCondition.B3_DETERMINISTIC_GATE

    @property
    def requires_authority_graph(self) -> bool:
        return self.uses_authority_resolver or self.uses_deterministic_gate

    @property
    def requires_block_order(self) -> bool:
        return self is ExperimentCondition.A3_SKILL_POLICY_SAME_TIER


class EpisodeStage(StrEnum):
    DEVELOPMENT = "development"
    CONFIRMATORY_A = "confirmatory_a"
    CONFIRMATORY_B = "confirmatory_b"
    EXPLORATORY_C = "exploratory_c"


class EpisodeStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    TECHNICAL_FAILURE = "technical_failure"


class BlockOrder(StrEnum):
    POLICY_THEN_SKILL = "policy_then_skill"
    SKILL_THEN_POLICY = "skill_then_policy"


class _EpisodeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _enum_value[E: StrEnum](value: object, *, enum_type: type[E], field: str) -> E:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{field} is invalid") from error


def _identifier(value: object, *, field: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a path-safe identifier")
    return cast(str, value)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be nonblank text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field} must be UTF-8 encodable") from error
    return cast(str, value)


def _hash_ref(value: object, *, field: str) -> Sha256Ref:
    if type(value) is not str or _SHA256_REF.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase sha256 reference")
    return cast(Sha256Ref, value)


def _optional_hash_ref(value: object, *, field: str) -> Sha256Ref | None:
    if value is None:
        return None
    return _hash_ref(value, field=field)


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_I64:
        raise ValueError(f"{field} must be a positive bounded integer")
    return cast(int, value)


class ExecutorModel(_EpisodeModel):
    provider: str
    model: str
    model_version_date: str

    @field_validator("provider", "model", "model_version_date", mode="before")
    @classmethod
    def _text_fields(cls, value: object, info: object) -> str:
        return _text(value, field=str(getattr(info, "field_name")))


class PromptHashes(_EpisodeModel):
    system: Sha256Ref
    developer: Sha256Ref
    user: Sha256Ref

    @field_validator("system", "developer", "user", mode="before")
    @classmethod
    def _hash_fields(cls, value: object, info: object) -> Sha256Ref:
        return _hash_ref(value, field=str(getattr(info, "field_name")))


class EpisodeBudgets(_EpisodeModel):
    max_turns: int
    max_tool_calls: int
    max_tokens: int

    @field_validator("max_turns", "max_tool_calls", "max_tokens", mode="before")
    @classmethod
    def _limits(cls, value: object, info: object) -> int:
        return _positive_int(value, field=str(getattr(info, "field_name")))


class EpisodeManifest(_EpisodeModel):
    episode_id: str
    schema_version: Literal["1.0"]
    stage: EpisodeStage
    condition: ExperimentCondition
    domain: Literal["access_provisioning", "financial_adjustments"]
    case_id: str
    skill_bundle_id: str | None = None
    contamination_ratio: float | None = None
    world_hash: Sha256Ref
    executor_model: ExecutorModel
    prompt_hashes: PromptHashes
    policy_hash: Sha256Ref | None
    skill_hash: Sha256Ref | None
    authority_graph_hash: Sha256Ref | None = None
    seed: int
    budgets: EpisodeBudgets
    status: EpisodeStatus
    artifact_refs: tuple[str, ...] = ()

    @field_validator("episode_id", "case_id", mode="before")
    @classmethod
    def _identifiers(cls, value: object, info: object) -> str:
        return _identifier(value, field=str(getattr(info, "field_name")))

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_version(cls, value: object) -> Literal["1.0"]:
        if type(value) is not str or value != "1.0":
            raise ValueError("schema_version must be 1.0")
        return "1.0"

    @field_validator("stage", mode="before")
    @classmethod
    def _stage(cls, value: object) -> EpisodeStage:
        return _enum_value(value, enum_type=EpisodeStage, field="stage")

    @field_validator("condition", mode="before")
    @classmethod
    def _condition(cls, value: object) -> ExperimentCondition:
        return _enum_value(value, enum_type=ExperimentCondition, field="condition")

    @field_validator("domain", mode="before")
    @classmethod
    def _domain(cls, value: object) -> Literal["access_provisioning", "financial_adjustments"]:
        if type(value) is not str or value not in _DOMAINS:
            raise ValueError("domain is invalid")
        return cast(Literal["access_provisioning", "financial_adjustments"], value)

    @field_validator("skill_bundle_id", mode="before")
    @classmethod
    def _bundle_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return _identifier(value, field="skill_bundle_id")

    @field_validator("contamination_ratio", mode="before")
    @classmethod
    def _ratio(cls, value: object) -> float | None:
        if value is None:
            return None
        if type(value) is not int and type(value) is not float:
            raise ValueError("contamination_ratio must be between zero and one")
        ratio = cast(int | float, value)
        if not 0 <= ratio <= 1:
            raise ValueError("contamination_ratio must be between zero and one")
        return float(ratio)

    @field_validator("world_hash", mode="before")
    @classmethod
    def _world_hash(cls, value: object) -> Sha256Ref:
        return _hash_ref(value, field="world_hash")

    @field_validator("policy_hash", "skill_hash", "authority_graph_hash", mode="before")
    @classmethod
    def _optional_hashes(cls, value: object, info: object) -> Sha256Ref | None:
        return _optional_hash_ref(value, field=str(getattr(info, "field_name")))

    @field_validator("seed", mode="before")
    @classmethod
    def _seed(cls, value: object) -> int:
        if type(value) is not int or not -(2**63) <= value <= _MAX_I64:
            raise ValueError("seed must be a bounded exact integer")
        return cast(int, value)

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, value: object) -> EpisodeStatus:
        return _enum_value(value, enum_type=EpisodeStatus, field="status")

    @field_validator("artifact_refs", mode="before")
    @classmethod
    def _artifacts(cls, value: object) -> tuple[str, ...]:
        if type(value) not in {list, tuple}:
            raise ValueError("artifact_refs must be an array")
        items = cast(list[object] | tuple[object, ...], value)
        refs = tuple(_text(item, field="artifact_refs") for item in items)
        if len(set(refs)) != len(refs):
            raise ValueError("artifact_refs must be unique")
        return refs


class EpisodePlan(_EpisodeModel):
    """Frozen inputs for one executor invocation."""

    manifest: EpisodeManifest
    task: TaskCase
    initial_state: WorldState
    domain_case: AccessTaskCase | FinanceTaskCase
    authority_decision: AuthorityDecision
    authority_records: tuple[AuthorityRecord, ...]
    authority_query: AccessAuthorityQuery | FinanceAuthorityQuery
    skill: CompiledSkillArtifact | None = None
    policy: RenderedPolicy | None = None
    order_assignment: BlockOrder | None = None

    @field_validator("initial_state", mode="before")
    @classmethod
    def _initial_state(cls, value: object) -> WorldState:
        if type(value) is not WorldState:
            raise ValueError("initial_state must be an exact WorldState")
        return WorldState.model_validate(
            {
                field: getattr(value, field)
                for field in ("schema_version", "world_id", "domain", "seed", "data")
            }
        )

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

    @field_validator("authority_records", mode="before")
    @classmethod
    def _authority_records(cls, value: object) -> tuple[AuthorityRecord, ...]:
        if type(value) is not tuple or not value:
            raise ValueError("authority_records must be a nonempty exact tuple")
        records: list[AuthorityRecord] = []
        for item in value:
            if type(item) is not AuthorityRecord:
                raise ValueError("authority_records must contain exact AuthorityRecord models")
            records.append(AuthorityRecord.model_validate(item.model_dump(mode="json")))
        return tuple(records)

    @field_validator("authority_query", mode="before")
    @classmethod
    def _authority_query(cls, value: object) -> AccessAuthorityQuery | FinanceAuthorityQuery:
        if type(value) is AccessAuthorityQuery:
            return AccessAuthorityQuery.model_validate(value.model_dump(mode="json"))
        if type(value) is FinanceAuthorityQuery:
            return FinanceAuthorityQuery.model_validate(value.model_dump(mode="json"))
        raise ValueError("authority_query must be an exact typed authority query")

    @field_validator("skill", mode="before")
    @classmethod
    def _skill(cls, value: object) -> CompiledSkillArtifact | None:
        if value is None:
            return None
        if type(value) is not CompiledSkillArtifact:
            raise ValueError("skill must be an exact CompiledSkillArtifact")
        return value

    @field_validator("policy", mode="before")
    @classmethod
    def _policy(cls, value: object) -> RenderedPolicy | None:
        if value is None:
            return None
        if type(value) is not RenderedPolicy:
            raise ValueError("policy must be an exact RenderedPolicy")
        return RenderedPolicy.model_validate(value.model_dump(mode="json"))

    @field_validator("order_assignment", mode="before")
    @classmethod
    def _order_assignment(cls, value: object) -> BlockOrder | None:
        if value is None:
            return None
        return _enum_value(value, enum_type=BlockOrder, field="order_assignment")

    @model_validator(mode="after")
    def _bind_inputs(self) -> EpisodePlan:
        if (
            self.task.case_id != self.manifest.case_id
            or self.task.domain != self.manifest.domain
            or self.task.world_id != self.initial_state.world_id
            or self.task.seed != self.initial_state.seed
            or self.initial_state.domain != self.manifest.domain
            or hash_state(self.initial_state) != self.manifest.world_hash
        ):
            raise ValueError("task does not bind the manifest case and domain")
        if (
            self.domain_case.case_id != self.task.case_id
            or self.domain_case.world_id != self.task.world_id
            or self.domain_case.domain != self.task.domain
            or self.domain_case.seed != self.task.seed
            or self.authority_decision.domain != self.manifest.domain
            or self.authority_query.domain != self.manifest.domain
        ):
            raise ValueError("execution inputs do not bind the manifest task")
        if self.manifest.authority_graph_hash != self.authority_decision.authority_set_hash:
            raise ValueError("manifest authority_graph_hash does not bind authority decision")
        if type(self.domain_case) is AccessTaskCase:
            from shadowskillbench.authority.models import AccessAuthorityQuery
            from shadowskillbench.domains.access.models import parse_access_world

            if type(self.authority_query) is not AccessAuthorityQuery:
                raise ValueError("access case requires an access authority query")
            world = parse_access_world(self.initial_state)
            request = next(
                (
                    item
                    for item in world.access_requests
                    if item.request_id == self.domain_case.target_request_id
                ),
                None,
            )
            application = (
                None
                if request is None
                else next(
                    (
                        item
                        for item in world.applications
                        if item.application_id == request.application_id
                    ),
                    None,
                )
            )
            if (
                request is None
                or application is None
                or (
                    self.authority_query.action_type != "grant_access"
                    or self.authority_query.subject_id != request.employee_id
                    or self.authority_query.resource_id != request.application_id
                    or self.authority_query.role_id != request.requested_role
                    or self.authority_query.role_derived_access
                    != application.supports_role_derived_access
                )
            ):
                raise ValueError("authority query does not bind access case facts")
        else:
            from shadowskillbench.authority.models import FinanceAuthorityQuery
            from shadowskillbench.domains.finance.models import parse_finance_world

            if type(self.authority_query) is not FinanceAuthorityQuery:
                raise ValueError("finance case requires a finance authority query")
            finance_case = cast(FinanceTaskCase, self.domain_case)
            world = parse_finance_world(self.initial_state)
            fact = next(
                (
                    item
                    for item in world.control_facts
                    if item.control_fact_id == finance_case.target_control_fact_id
                ),
                None,
            )
            if fact is None or (
                self.authority_query.action_type != "post_adjustment"
                or self.authority_query.subject_id != fact.adjustment_id
                or self.authority_query.resource_id != fact.portco_id
                or self.authority_query.category_id != fact.original_economic_category
                or self.authority_query.period_id != fact.period_id
                or self.authority_query.amount_minor != fact.signed_amount_minor
                or self.authority_query.currency != fact.currency
                or self.authority_query.unit != fact.unit
                or self.authority_query.currency_exponent != fact.currency_exponent
            ):
                raise ValueError("authority query does not bind finance case facts")
        from shadowskillbench.authority.resolver import resolve_authority

        try:
            resolved = resolve_authority(self.authority_records, self.authority_query)
        except (TypeError, ValueError) as error:
            raise ValueError("authority records and query are not resolvable") from error
        if resolved != self.authority_decision:
            raise ValueError("authority decision does not bind authority records and query")
        if self.manifest.skill_hash != (
            None if self.skill is None else self.skill.rendered_skill_hash
        ):
            raise ValueError("manifest skill_hash does not bind skill")
        if self.manifest.policy_hash != (
            None if self.policy is None else self.policy.rendered_hash
        ):
            raise ValueError("manifest policy_hash does not bind policy")
        if self.manifest.condition.requires_block_order:
            if self.order_assignment is None:
                raise ValueError("condition requires order_assignment")
        elif self.order_assignment is not None:
            raise ValueError("order_assignment is not admitted for this condition")
        from shadowskillbench.episodes.context import assemble_context

        assemble_context(
            self.manifest.condition,
            self.task,
            self.skill,
            self.policy,
            self.order_assignment,
        )
        return self


def parse_episode_manifest(value: object) -> EpisodeManifest:
    return EpisodeManifest.model_validate(value)


def episode_manifest_projection(value: object) -> dict[str, object]:
    manifest = parse_episode_manifest(value)
    return cast(dict[str, object], manifest.model_dump(mode="json"))


def hash_episode_manifest(value: object) -> Sha256Ref:
    return cast(Sha256Ref, sha256_ref(episode_manifest_projection(value)))
