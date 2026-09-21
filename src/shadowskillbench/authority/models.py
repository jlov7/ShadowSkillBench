from __future__ import annotations

import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.config import ExtraValues

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.engine.models import PathSafeId, Sha256Ref

ACCESS_DOMAIN = "access_provisioning"
FINANCE_DOMAIN = "financial_adjustments"
_DOMAIN_VALUES = frozenset({ACCESS_DOMAIN, FINANCE_DOMAIN})
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


class AuthorityAdmissionError(ValueError):
    """Signals malformed authority input before deterministic resolution."""


class AuthoritySourceType(StrEnum):
    BEHAVIOR_TRACE = "BEHAVIOR_TRACE"
    PROCEDURE_GUIDE = "PROCEDURE_GUIDE"
    POLICY = "POLICY"
    SIGNED_DIRECTIVE = "SIGNED_DIRECTIVE"
    CHANGE_RECORD = "CHANGE_RECORD"
    APPROVAL = "APPROVAL"
    WAIVER = "WAIVER"
    SYSTEM_CONFIGURATION = "SYSTEM_CONFIGURATION"
    VERIFIER_RESULT = "VERIFIER_RESULT"


class NormativeStatus(StrEnum):
    DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"
    ACTIVE_AUTHORITY = "ACTIVE_AUTHORITY"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"
    SCOPED_EXCEPTION = "SCOPED_EXCEPTION"
    CONFLICTING = "CONFLICTING"
    UNRESOLVED = "UNRESOLVED"


class ScopeKind(StrEnum):
    RULE = "rule"
    EVIDENCE = "evidence"
    DESCRIPTIVE = "descriptive"


class RuleDisposition(StrEnum):
    PROCEED = "PROCEED"
    BLOCK = "BLOCK"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class Decision(StrEnum):
    PROCEED = "PROCEED"
    BLOCK = "BLOCK"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    PROCEED_UNDER_EXCEPTION = "PROCEED_UNDER_EXCEPTION"
    FOLLOW_SUPERSEDING_AUTHORITY = "FOLLOW_SUPERSEDING_AUTHORITY"
    ESCALATE = "ESCALATE"


class DecisionDisposition(StrEnum):
    PROCEED = "PROCEED"
    BLOCK = "BLOCK"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    ESCALATE = "ESCALATE"


class DiscardReasonCode(StrEnum):
    DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"
    NOT_YET_EFFECTIVE = "NOT_YET_EFFECTIVE"
    EXPIRED_AT_QUERY = "EXPIRED_AT_QUERY"
    ACTION_MISMATCH = "ACTION_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    SUPERSEDED = "SUPERSEDED"
    LOWER_AUTHORITY_RANK = "LOWER_AUTHORITY_RANK"
    EVIDENCE_TARGET_NOT_EFFECTIVE = "EVIDENCE_TARGET_NOT_EFFECTIVE"
    EVIDENCE_NOT_NEEDED = "EVIDENCE_NOT_NEEDED"


class DecisionReasonCode(StrEnum):
    NO_APPLICABLE_AUTHORITY = "NO_APPLICABLE_AUTHORITY"
    UNRESOLVED_TOP_RANK_CONFLICT = "UNRESOLVED_TOP_RANK_CONFLICT"
    EFFECTIVE_RULE = "EFFECTIVE_RULE"
    EFFECTIVE_SUPERSEDING_RULE = "EFFECTIVE_SUPERSEDING_RULE"
    VALID_APPROVAL = "VALID_APPROVAL"
    VALID_WAIVER = "VALID_WAIVER"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="never"
    )
    _validated: bool = PrivateAttr(default=False)

    @classmethod
    def _reject_weakened_validation_options(
        cls,
        *,
        strict: bool | None,
        extra: ExtraValues | None,
        from_attributes: bool | None,
    ) -> None:
        if strict is False:
            raise ValueError(f"{cls.__name__} requires strict validation")
        if extra not in {None, "forbid"}:
            raise ValueError(f"{cls.__name__} requires extra='forbid'")
        if from_attributes is True:
            raise ValueError(f"{cls.__name__} rejects attribute-based validation")

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        cls._reject_weakened_validation_options(
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
        )
        if type(obj) is not dict:
            raise ValueError(f"{cls.__name__} input must be an exact built-in object")
        return super().model_validate(
            obj,
            strict=True,
            extra="forbid",
            from_attributes=False,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        cls._reject_weakened_validation_options(
            strict=strict,
            extra=extra,
            from_attributes=None,
        )
        try:
            payload = json.loads(json_data)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{cls.__name__} input must be valid JSON") from error
        return cls.model_validate(
            payload,
            strict=True,
            extra="forbid",
            from_attributes=False,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def model_validate_strings(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        raise ValueError(f"{cls.__name__} rejects coercive string validation")

    @model_validator(mode="after")
    def _mark_validated(self) -> Self:
        object.__setattr__(self, "_validated", True)
        return self


def _path_safe_id(value: object, *, field: str) -> PathSafeId:
    if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact path-safe identifier")
    return cast(PathSafeId, value)


def _optional_path_safe_id(value: object, *, field: str) -> PathSafeId | None:
    if value is None:
        return None
    return _path_safe_id(value, field=field)


def _sha256_reference(value: object, *, field: str) -> Sha256Ref:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase sha256 reference")
    return cast(Sha256Ref, value)


def _utc_seconds(value: object, *, field: str) -> str:
    if type(value) is not str or _UTC_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must use canonical UTC seconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"{field} must be a valid UTC timestamp") from error
    return value


def _optional_utc_seconds(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _utc_seconds(value, field=field)


def _exact_int(value: object, *, field: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" no less than {minimum}"
        raise ValueError(f"{field} must be an exact integer{suffix}")
    return cast(int, value)


def _exact_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be an exact boolean")
    return cast(bool, value)


def _nonblank_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be nonblank text without NUL")
    return cast(str, value)


def _currency(value: object, *, field: str) -> str:
    if type(value) is not str or _CURRENCY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be an uppercase ISO 4217 currency")
    return value


def _currency_exponent(value: object) -> Literal[2]:
    if type(value) is not int or value != 2:
        raise ValueError("currency_exponent must be the exact integer 2")
    return cast(Literal[2], value)


def _exact_array(value: object, *, field: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{field} must be an exact JSON array")
    return cast(list[object], value)


def _sorted_unique_ids(value: object, *, field: str) -> tuple[PathSafeId, ...]:
    items = tuple(_path_safe_id(item, field=field) for item in _exact_array(value, field=field))
    if items != tuple(sorted(items)) or len(set(items)) != len(items):
        raise ValueError(f"{field} must be Unicode-sorted and unique")
    return items


def _enum_value[E: StrEnum](value: object, *, enum_type: type[E], field: str) -> E:
    if type(value) is str:
        return enum_type(value)
    if type(value) is enum_type:
        return cast(E, value)
    raise ValueError(f"{field} must be an explicit {enum_type.__name__}")


def _owned_model_input[T: _StrictFrozenModel](
    value: object, *, expected_type: type[T], field: str
) -> dict[str, object]:
    if type(value) is expected_type:
        model = cast(T, value)
        if not model._validated:
            raise ValueError(f"{field} must be a validated {expected_type.__name__}")
        return cast(dict[str, object], model.model_dump(mode="json"))
    if type(value) is dict:
        return cast(dict[str, object], value)
    raise ValueError(f"{field} must be an exact JSON object or {expected_type.__name__}")


class _AccessScopeBase(_StrictFrozenModel):
    domain: Literal["access_provisioning"]
    scope_kind: ScopeKind
    subject_id: PathSafeId | None
    resource_id: PathSafeId | None
    organization_id: PathSafeId | None
    geography_id: PathSafeId | None
    role_id: PathSafeId | None

    @field_validator(
        "subject_id", "resource_id", "organization_id", "geography_id", "role_id", mode="before"
    )
    @classmethod
    def _validate_optional_ids(cls, value: object, info: ValidationInfo) -> PathSafeId | None:
        return _optional_path_safe_id(value, field=cast(str, info.field_name))


class AccessDescriptiveScope(_AccessScopeBase):
    scope_kind: Literal[ScopeKind.DESCRIPTIVE]  # pyright: ignore[reportIncompatibleVariableOverride]


class AccessEvidenceScope(_AccessScopeBase):
    scope_kind: Literal[ScopeKind.EVIDENCE]  # pyright: ignore[reportIncompatibleVariableOverride]

    @model_validator(mode="after")
    def _require_evidence_subject_and_resource(self) -> Self:
        if self.subject_id is None or self.resource_id is None:
            raise ValueError("access evidence scope requires exact subject_id and resource_id")
        return self


class AccessRuleScope(_AccessScopeBase):
    scope_kind: Literal[ScopeKind.RULE]  # pyright: ignore[reportIncompatibleVariableOverride]
    rule_disposition: RuleDisposition
    allowed_action: PathSafeId
    requires_security_approval: bool
    role_derived_without_approval: bool

    @field_validator("rule_disposition", mode="before")
    @classmethod
    def _validate_rule_disposition(cls, value: object) -> RuleDisposition:
        return _enum_value(value, enum_type=RuleDisposition, field="rule_disposition")

    @field_validator("allowed_action", mode="before")
    @classmethod
    def _validate_allowed_action(cls, value: object) -> PathSafeId:
        return _path_safe_id(value, field="allowed_action")

    @field_validator("requires_security_approval", "role_derived_without_approval", mode="before")
    @classmethod
    def _validate_rule_booleans(cls, value: object, info: ValidationInfo) -> bool:
        return _exact_bool(value, field=cast(str, info.field_name))


class _FinanceScopeBase(_StrictFrozenModel):
    domain: Literal["financial_adjustments"]
    scope_kind: ScopeKind
    subject_id: PathSafeId | None
    resource_id: PathSafeId | None
    organization_id: PathSafeId | None
    geography_id: PathSafeId | None
    category_id: PathSafeId | None
    period_id: PathSafeId | None
    minimum_amount_minor: int | None
    minimum_inclusive: bool
    maximum_amount_minor: int | None
    maximum_inclusive: bool
    currency: str
    unit: Literal["minor_units"]
    currency_exponent: Literal[2]

    @field_validator(
        "subject_id",
        "resource_id",
        "organization_id",
        "geography_id",
        "category_id",
        "period_id",
        mode="before",
    )
    @classmethod
    def _validate_optional_ids(cls, value: object, info: ValidationInfo) -> PathSafeId | None:
        return _optional_path_safe_id(value, field=cast(str, info.field_name))

    @field_validator("minimum_amount_minor", "maximum_amount_minor", mode="before")
    @classmethod
    def _validate_optional_amounts(cls, value: object, info: ValidationInfo) -> int | None:
        if value is None:
            return None
        return _exact_int(value, field=cast(str, info.field_name))

    @field_validator("minimum_inclusive", "maximum_inclusive", mode="before")
    @classmethod
    def _validate_inclusivity(cls, value: object, info: ValidationInfo) -> bool:
        return _exact_bool(value, field=cast(str, info.field_name))

    @field_validator("currency", mode="before")
    @classmethod
    def _validate_currency(cls, value: object) -> str:
        return _currency(value, field="currency")

    @field_validator("currency_exponent", mode="before")
    @classmethod
    def _validate_currency_exponent(cls, value: object) -> Literal[2]:
        return _currency_exponent(value)

    @model_validator(mode="after")
    def _validate_amount_interval(self) -> Self:
        if self.minimum_amount_minor is None and self.minimum_inclusive:
            raise ValueError("unbounded minimum amount requires minimum_inclusive=false")
        if self.maximum_amount_minor is None and self.maximum_inclusive:
            raise ValueError("unbounded maximum amount requires maximum_inclusive=false")
        if (
            self.minimum_amount_minor is not None
            and self.maximum_amount_minor is not None
            and self.minimum_amount_minor > self.maximum_amount_minor
        ):
            raise ValueError("minimum amount must not exceed maximum amount")
        if (
            self.minimum_amount_minor is not None
            and self.minimum_amount_minor == self.maximum_amount_minor
            and not (self.minimum_inclusive and self.maximum_inclusive)
        ):
            raise ValueError("equal amount bounds require both endpoints inclusive")
        return self


class FinanceDescriptiveScope(_FinanceScopeBase):
    scope_kind: Literal[ScopeKind.DESCRIPTIVE]  # pyright: ignore[reportIncompatibleVariableOverride]


class FinanceEvidenceScope(_FinanceScopeBase):
    scope_kind: Literal[ScopeKind.EVIDENCE]  # pyright: ignore[reportIncompatibleVariableOverride]

    @model_validator(mode="after")
    def _validate_evidence_scope(self) -> Self:
        if self.subject_id is None or self.resource_id is None:
            raise ValueError("finance evidence scope requires exact subject_id and resource_id")
        if self.category_id is None or self.period_id is None:
            raise ValueError("finance evidence scope requires exact category_id and period_id")
        if (
            self.minimum_amount_minor != 0
            or not self.minimum_inclusive
            or self.maximum_amount_minor is None
            or not self.maximum_inclusive
            or self.maximum_amount_minor < 0
        ):
            raise ValueError(
                "finance evidence scope requires an inclusive [0, nonnegative cap] interval"
            )
        return self


class FinanceRuleScope(_FinanceScopeBase):
    scope_kind: Literal[ScopeKind.RULE]  # pyright: ignore[reportIncompatibleVariableOverride]
    rule_disposition: RuleDisposition
    threshold_minor: int
    restricted_categories: tuple[PathSafeId, ...]

    @field_validator("rule_disposition", mode="before")
    @classmethod
    def _validate_rule_disposition(cls, value: object) -> RuleDisposition:
        return _enum_value(value, enum_type=RuleDisposition, field="rule_disposition")

    @field_validator("threshold_minor", mode="before")
    @classmethod
    def _validate_threshold(cls, value: object) -> int:
        return _exact_int(value, field="threshold_minor", minimum=0)

    @field_validator("restricted_categories", mode="before")
    @classmethod
    def _validate_restricted_categories(cls, value: object) -> tuple[PathSafeId, ...]:
        return _sorted_unique_ids(value, field="restricted_categories")


AccessScope = Annotated[
    AccessRuleScope | AccessEvidenceScope | AccessDescriptiveScope,
    Field(discriminator="scope_kind"),
]
FinanceScope = Annotated[
    FinanceRuleScope | FinanceEvidenceScope | FinanceDescriptiveScope,
    Field(discriminator="scope_kind"),
]
AuthorityScope = Annotated[AccessScope | FinanceScope, Field(discriminator="domain")]


_SCOPE_TYPES = (
    AccessRuleScope,
    AccessEvidenceScope,
    AccessDescriptiveScope,
    FinanceRuleScope,
    FinanceEvidenceScope,
    FinanceDescriptiveScope,
)


def _owned_scope_input(value: object) -> dict[str, object]:
    if type(value) in _SCOPE_TYPES:
        model = cast(_StrictFrozenModel, value)
        if not model._validated:
            raise ValueError("scope must be a validated authority scope")
        return cast(dict[str, object], model.model_dump(mode="json"))
    if type(value) is dict:
        return cast(dict[str, object], value)
    raise ValueError("scope must be an exact JSON object or validated authority scope")


class IssuerAuthorization(_StrictFrozenModel):
    issuer_id: PathSafeId
    issuer_role: PathSafeId
    authority_rank: int
    allowed_domains: tuple[Literal["access_provisioning", "financial_adjustments"], ...]
    allowed_source_types: tuple[AuthoritySourceType, ...]
    can_supersede: bool
    can_issue_scoped_evidence: bool

    @field_validator("issuer_id", "issuer_role", mode="before")
    @classmethod
    def _validate_ids(cls, value: object, info: ValidationInfo) -> PathSafeId:
        return _path_safe_id(value, field=cast(str, info.field_name))

    @field_validator("authority_rank", mode="before")
    @classmethod
    def _validate_rank(cls, value: object) -> int:
        return _exact_int(value, field="authority_rank", minimum=0)

    @field_validator("allowed_domains", mode="before")
    @classmethod
    def _validate_domains(
        cls, value: object
    ) -> tuple[Literal["access_provisioning", "financial_adjustments"], ...]:
        values = tuple(_exact_array(value, field="allowed_domains"))
        if any(type(item) is not str or item not in _DOMAIN_VALUES for item in values):
            raise ValueError("allowed_domains must contain exact supported domains")
        strings = cast(tuple[str, ...], values)
        if strings != tuple(sorted(strings)) or len(set(strings)) != len(strings):
            raise ValueError("allowed_domains must be Unicode-sorted and unique")
        return cast(tuple[Literal["access_provisioning", "financial_adjustments"], ...], strings)

    @field_validator("allowed_source_types", mode="before")
    @classmethod
    def _validate_source_types(cls, value: object) -> tuple[AuthoritySourceType, ...]:
        values = tuple(
            _enum_value(item, enum_type=AuthoritySourceType, field="allowed_source_types")
            for item in _exact_array(value, field="allowed_source_types")
        )
        names = tuple(item.value for item in values)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("allowed_source_types must be Unicode-sorted and unique")
        return values

    @field_validator("can_supersede", "can_issue_scoped_evidence", mode="before")
    @classmethod
    def _validate_booleans(cls, value: object, info: ValidationInfo) -> bool:
        return _exact_bool(value, field=cast(str, info.field_name))


def _owned_authorization_input(value: object) -> dict[str, object]:
    return _owned_model_input(value, expected_type=IssuerAuthorization, field="authorization")


class _IssuerRegistryPayload(_StrictFrozenModel):
    authorizations: tuple[IssuerAuthorization, ...]

    @field_validator("authorizations", mode="before")
    @classmethod
    def _validate_authorizations(cls, value: object) -> tuple[IssuerAuthorization, ...]:
        authorizations = tuple(
            IssuerAuthorization.model_validate(_owned_authorization_input(item))
            for item in _exact_array(value, field="authorizations")
        )
        keys = tuple((item.issuer_id, item.issuer_role) for item in authorizations)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError(
                "authorizations must be Unicode-sorted and unique by issuer_id and issuer_role"
            )
        return authorizations


def _registry_hash_payload(registry: _IssuerRegistryPayload) -> Sha256Ref:
    projection = registry.model_dump(mode="json")
    projection.pop("registry_hash", None)
    return cast(Sha256Ref, sha256_ref(projection))


class IssuerRegistry(_IssuerRegistryPayload):
    registry_hash: Sha256Ref

    @field_validator("registry_hash", mode="before")
    @classmethod
    def _validate_registry_hash(cls, value: object) -> Sha256Ref:
        return _sha256_reference(value, field="registry_hash")

    @model_validator(mode="after")
    def _bind_registry_hash(self) -> Self:
        if self.registry_hash != _registry_hash_payload(self):
            raise ValueError("registry_hash does not bind authorizations")
        return self


def create_issuer_registry(*, authorizations: object) -> IssuerRegistry:
    provisional = _IssuerRegistryPayload.model_validate({"authorizations": authorizations})
    return IssuerRegistry.model_validate(
        {
            "authorizations": [item.model_dump(mode="json") for item in provisional.authorizations],
            "registry_hash": _registry_hash_payload(provisional),
        }
    )


def _owned_registry_input(value: object) -> dict[str, object]:
    return _owned_model_input(value, expected_type=IssuerRegistry, field="issuer_registry")


class _AuthorityQueryPayloadBase(_StrictFrozenModel):
    domain: Literal["access_provisioning", "financial_adjustments"]
    subject_id: PathSafeId
    resource_id: PathSafeId
    action_type: PathSafeId
    at_time: str
    organization_id: PathSafeId | None
    geography_id: PathSafeId | None
    reference_authority_ids: tuple[PathSafeId, ...]
    issuer_registry: IssuerRegistry

    @field_validator(
        "subject_id", "resource_id", "action_type", "organization_id", "geography_id", mode="before"
    )
    @classmethod
    def _validate_ids(cls, value: object, info: ValidationInfo) -> PathSafeId | None:
        if info.field_name in {"organization_id", "geography_id"}:
            return _optional_path_safe_id(value, field=cast(str, info.field_name))
        return _path_safe_id(value, field=cast(str, info.field_name))

    @field_validator("at_time", mode="before")
    @classmethod
    def _validate_time(cls, value: object) -> str:
        return _utc_seconds(value, field="at_time")

    @field_validator("reference_authority_ids", mode="before")
    @classmethod
    def _validate_reference_ids(cls, value: object) -> tuple[PathSafeId, ...]:
        return _sorted_unique_ids(value, field="reference_authority_ids")

    @field_validator("issuer_registry", mode="before")
    @classmethod
    def _validate_registry(cls, value: object) -> dict[str, object]:
        return _owned_registry_input(value)


class _AccessAuthorityQueryPayload(_AuthorityQueryPayloadBase):
    domain: Literal["access_provisioning"]  # pyright: ignore[reportIncompatibleVariableOverride]
    role_id: PathSafeId
    role_derived_access: bool

    @field_validator("role_id", mode="before")
    @classmethod
    def _validate_role(cls, value: object) -> PathSafeId:
        return _path_safe_id(value, field="role_id")

    @field_validator("role_derived_access", mode="before")
    @classmethod
    def _validate_role_derived(cls, value: object) -> bool:
        return _exact_bool(value, field="role_derived_access")


class _FinanceAuthorityQueryPayload(_AuthorityQueryPayloadBase):
    domain: Literal["financial_adjustments"]  # pyright: ignore[reportIncompatibleVariableOverride]
    category_id: PathSafeId
    period_id: PathSafeId
    amount_minor: int
    currency: str
    unit: Literal["minor_units"]
    currency_exponent: Literal[2]

    @field_validator("category_id", "period_id", mode="before")
    @classmethod
    def _validate_ids(cls, value: object, info: ValidationInfo) -> PathSafeId:
        return _path_safe_id(value, field=cast(str, info.field_name))

    @field_validator("amount_minor", mode="before")
    @classmethod
    def _validate_amount(cls, value: object) -> int:
        return _exact_int(value, field="amount_minor")

    @field_validator("currency", mode="before")
    @classmethod
    def _validate_currency(cls, value: object) -> str:
        return _currency(value, field="currency")

    @field_validator("currency_exponent", mode="before")
    @classmethod
    def _validate_currency_exponent(cls, value: object) -> Literal[2]:
        return _currency_exponent(value)


def _authority_query_hash_trusted(query: _AuthorityQueryPayloadBase) -> Sha256Ref:
    projection = query.model_dump(mode="json")
    projection.pop("authority_query_hash", None)
    projection["issuer_registry"] = query.issuer_registry.registry_hash
    return cast(Sha256Ref, sha256_ref(projection))


class AccessAuthorityQuery(_AccessAuthorityQueryPayload):
    authority_query_hash: Sha256Ref

    @field_validator("authority_query_hash", mode="before")
    @classmethod
    def _validate_query_hash(cls, value: object) -> Sha256Ref:
        return _sha256_reference(value, field="authority_query_hash")

    @model_validator(mode="after")
    def _bind_query_hash(self) -> Self:
        if self.authority_query_hash != _authority_query_hash_trusted(self):
            raise ValueError("authority_query_hash does not bind query")
        return self


class FinanceAuthorityQuery(_FinanceAuthorityQueryPayload):
    authority_query_hash: Sha256Ref

    @field_validator("authority_query_hash", mode="before")
    @classmethod
    def _validate_query_hash(cls, value: object) -> Sha256Ref:
        return _sha256_reference(value, field="authority_query_hash")

    @model_validator(mode="after")
    def _bind_query_hash(self) -> Self:
        if self.authority_query_hash != _authority_query_hash_trusted(self):
            raise ValueError("authority_query_hash does not bind query")
        return self


AuthorityQuery = Annotated[
    AccessAuthorityQuery | FinanceAuthorityQuery,
    Field(discriminator="domain"),
]


def _owned_query_input(value: object) -> dict[str, object]:
    if type(value) is AccessAuthorityQuery:
        model = cast(AccessAuthorityQuery, value)
    elif type(value) is FinanceAuthorityQuery:
        model = cast(FinanceAuthorityQuery, value)
    else:
        raise ValueError("authority_query must be an exact validated authority query")
    if not model._validated:
        raise ValueError("authority_query must be validated")
    return cast(dict[str, object], model.model_dump(mode="json"))


def authority_query_hash(query: object) -> Sha256Ref:
    if type(query) is AccessAuthorityQuery:
        model = cast(AccessAuthorityQuery, query)
        if not model._validated:
            raise ValueError("authority_query_hash requires a validated AccessAuthorityQuery")
        return _authority_query_hash_trusted(
            AccessAuthorityQuery.model_validate(model.model_dump(mode="json"))
        )
    if type(query) is FinanceAuthorityQuery:
        model = cast(FinanceAuthorityQuery, query)
        if not model._validated:
            raise ValueError("authority_query_hash requires a validated FinanceAuthorityQuery")
        return _authority_query_hash_trusted(
            FinanceAuthorityQuery.model_validate(model.model_dump(mode="json"))
        )
    raise ValueError("authority_query_hash requires an exact authority query")


def _create_access_query(raw: dict[str, object]) -> AccessAuthorityQuery:
    provisional = _AccessAuthorityQueryPayload.model_validate(raw)
    return AccessAuthorityQuery.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "authority_query_hash": _authority_query_hash_trusted(provisional),
        }
    )


def _create_finance_query(raw: dict[str, object]) -> FinanceAuthorityQuery:
    provisional = _FinanceAuthorityQueryPayload.model_validate(raw)
    return FinanceAuthorityQuery.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "authority_query_hash": _authority_query_hash_trusted(provisional),
        }
    )


def create_access_authority_query(
    *,
    subject_id: object,
    resource_id: object,
    action_type: object,
    at_time: object,
    organization_id: object,
    geography_id: object,
    reference_authority_ids: object,
    issuer_registry: object,
    role_id: object,
    role_derived_access: object,
) -> AccessAuthorityQuery:
    return _create_access_query(
        {
            "domain": ACCESS_DOMAIN,
            "subject_id": subject_id,
            "resource_id": resource_id,
            "action_type": action_type,
            "at_time": at_time,
            "organization_id": organization_id,
            "geography_id": geography_id,
            "reference_authority_ids": reference_authority_ids,
            "issuer_registry": issuer_registry,
            "role_id": role_id,
            "role_derived_access": role_derived_access,
        },
    )


def create_finance_authority_query(
    *,
    subject_id: object,
    resource_id: object,
    action_type: object,
    at_time: object,
    organization_id: object,
    geography_id: object,
    reference_authority_ids: object,
    issuer_registry: object,
    category_id: object,
    period_id: object,
    amount_minor: object,
    currency: object,
    unit: object,
    currency_exponent: object,
) -> FinanceAuthorityQuery:
    return _create_finance_query(
        {
            "domain": FINANCE_DOMAIN,
            "subject_id": subject_id,
            "resource_id": resource_id,
            "action_type": action_type,
            "at_time": at_time,
            "organization_id": organization_id,
            "geography_id": geography_id,
            "reference_authority_ids": reference_authority_ids,
            "issuer_registry": issuer_registry,
            "category_id": category_id,
            "period_id": period_id,
            "amount_minor": amount_minor,
            "currency": currency,
            "unit": unit,
            "currency_exponent": currency_exponent,
        },
    )


class _AuthorityRecordPayload(_StrictFrozenModel):
    authority_id: PathSafeId
    schema_version: Literal["1.0"]
    source_type: AuthoritySourceType
    title: str
    issuer_id: PathSafeId
    issuer_role: PathSafeId
    authority_rank: int
    action_type: PathSafeId
    scope: AuthorityScope
    effective_at: str
    expires_at: str | None
    supersedes: tuple[PathSafeId, ...]
    exception_to: tuple[PathSafeId, ...]
    provenance_locator: str
    normative_status: NormativeStatus

    @field_validator("authority_id", "issuer_id", "issuer_role", "action_type", mode="before")
    @classmethod
    def _validate_ids(cls, value: object, info: ValidationInfo) -> PathSafeId:
        return _path_safe_id(value, field=cast(str, info.field_name))

    @field_validator("source_type", mode="before")
    @classmethod
    def _validate_source_type(cls, value: object) -> AuthoritySourceType:
        return _enum_value(value, enum_type=AuthoritySourceType, field="source_type")

    @field_validator("normative_status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> NormativeStatus:
        return _enum_value(value, enum_type=NormativeStatus, field="normative_status")

    @field_validator("title", "provenance_locator", mode="before")
    @classmethod
    def _validate_text(cls, value: object, info: ValidationInfo) -> str:
        return _nonblank_text(value, field=cast(str, info.field_name))

    @field_validator("authority_rank", mode="before")
    @classmethod
    def _validate_rank(cls, value: object) -> int:
        return _exact_int(value, field="authority_rank", minimum=0)

    @field_validator("scope", mode="before")
    @classmethod
    def _validate_scope(cls, value: object) -> dict[str, object]:
        return _owned_scope_input(value)

    @field_validator("effective_at", mode="before")
    @classmethod
    def _validate_effective_at(cls, value: object) -> str:
        return _utc_seconds(value, field="effective_at")

    @field_validator("expires_at", mode="before")
    @classmethod
    def _validate_expires_at(cls, value: object) -> str | None:
        return _optional_utc_seconds(value, field="expires_at")

    @field_validator("supersedes", "exception_to", mode="before")
    @classmethod
    def _validate_id_arrays(cls, value: object, info: ValidationInfo) -> tuple[PathSafeId, ...]:
        return _sorted_unique_ids(value, field=cast(str, info.field_name))

    @model_validator(mode="after")
    def _validate_record_contract(self) -> Self:
        if self.expires_at is not None and self.effective_at >= self.expires_at:
            raise AuthorityAdmissionError("expires_at must be later than effective_at")
        if self.normative_status is NormativeStatus.EXPIRED and self.expires_at is None:
            raise AuthorityAdmissionError("expired authority must have an explicit expires_at")

        descriptive_sources = {
            AuthoritySourceType.BEHAVIOR_TRACE,
            AuthoritySourceType.PROCEDURE_GUIDE,
            AuthoritySourceType.VERIFIER_RESULT,
        }
        rule_sources = {
            AuthoritySourceType.POLICY,
            AuthoritySourceType.SIGNED_DIRECTIVE,
            AuthoritySourceType.CHANGE_RECORD,
            AuthoritySourceType.SYSTEM_CONFIGURATION,
        }
        if self.source_type in descriptive_sources:
            if (
                self.normative_status is not NormativeStatus.DESCRIPTIVE_ONLY
                or self.scope.scope_kind is not ScopeKind.DESCRIPTIVE
                or self.supersedes
                or self.exception_to
            ):
                raise AuthorityAdmissionError(
                    "descriptive sources must be descriptive-only without edges"
                )
        elif self.source_type in rule_sources:
            if (
                self.normative_status
                not in {
                    NormativeStatus.ACTIVE_AUTHORITY,
                    NormativeStatus.SUPERSEDED,
                    NormativeStatus.EXPIRED,
                    NormativeStatus.CONFLICTING,
                    NormativeStatus.UNRESOLVED,
                }
                or self.scope.scope_kind is not ScopeKind.RULE
                or self.exception_to
            ):
                raise AuthorityAdmissionError(
                    "rule sources require a rule scope and admitted rule status"
                )
        elif self.source_type is AuthoritySourceType.APPROVAL:
            if (
                self.normative_status is not NormativeStatus.ACTIVE_AUTHORITY
                or self.scope.scope_kind is not ScopeKind.EVIDENCE
                or not self.exception_to
                or self.supersedes
            ):
                raise AuthorityAdmissionError(
                    "approval must be active scoped evidence without supersession"
                )
        elif self.source_type is AuthoritySourceType.WAIVER:
            if (
                self.normative_status is not NormativeStatus.SCOPED_EXCEPTION
                or self.scope.scope_kind is not ScopeKind.EVIDENCE
                or not self.exception_to
                or self.supersedes
            ):
                raise AuthorityAdmissionError("waiver must be scoped evidence without supersession")

        if (
            isinstance(self.scope, AccessRuleScope)
            and self.scope.allowed_action != self.action_type
        ):
            raise AuthorityAdmissionError(
                "access rule allowed_action must equal record action_type"
            )

        return self


def _authority_record_hash_trusted(record: _AuthorityRecordPayload) -> Sha256Ref:
    projection = record.model_dump(mode="json")
    projection.pop("content_hash", None)
    return cast(Sha256Ref, sha256_ref(projection))


class AuthorityRecord(_AuthorityRecordPayload):
    content_hash: Sha256Ref

    @field_validator("content_hash", mode="before")
    @classmethod
    def _validate_content_hash(cls, value: object) -> Sha256Ref:
        return _sha256_reference(value, field="content_hash")

    @model_validator(mode="after")
    def _bind_content_hash(self) -> Self:
        if self.content_hash != _authority_record_hash_trusted(self):
            raise ValueError("content_hash does not bind authority record")
        return self


def authority_record_hash(record: object) -> Sha256Ref:
    if type(record) is not AuthorityRecord:
        raise ValueError("authority_record_hash requires an exact AuthorityRecord")
    model = cast(AuthorityRecord, record)
    if not model._validated:
        raise ValueError("authority_record_hash requires a validated AuthorityRecord")
    return _authority_record_hash_trusted(
        AuthorityRecord.model_validate(model.model_dump(mode="json"))
    )


def create_authority_record(
    *,
    authority_id: object,
    schema_version: object,
    source_type: object,
    title: object,
    issuer_id: object,
    issuer_role: object,
    authority_rank: object,
    action_type: object,
    scope: object,
    effective_at: object,
    expires_at: object,
    supersedes: object,
    exception_to: object,
    provenance_locator: object,
    normative_status: object,
) -> AuthorityRecord:
    raw = {
        "authority_id": authority_id,
        "schema_version": schema_version,
        "source_type": source_type,
        "title": title,
        "issuer_id": issuer_id,
        "issuer_role": issuer_role,
        "authority_rank": authority_rank,
        "action_type": action_type,
        "scope": scope,
        "effective_at": effective_at,
        "expires_at": expires_at,
        "supersedes": supersedes,
        "exception_to": exception_to,
        "provenance_locator": provenance_locator,
        "normative_status": normative_status,
    }
    provisional = _AuthorityRecordPayload.model_validate(raw)
    return AuthorityRecord.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "content_hash": _authority_record_hash_trusted(provisional),
        }
    )


class AccessEffectiveParameters(_StrictFrozenModel):
    domain: Literal["access_provisioning"]
    allowed_action: PathSafeId
    requires_security_approval: bool
    role_derived_without_approval: bool

    @field_validator("allowed_action", mode="before")
    @classmethod
    def _validate_action(cls, value: object) -> PathSafeId:
        return _path_safe_id(value, field="allowed_action")

    @field_validator("requires_security_approval", "role_derived_without_approval", mode="before")
    @classmethod
    def _validate_booleans(cls, value: object, info: ValidationInfo) -> bool:
        return _exact_bool(value, field=cast(str, info.field_name))


class FinanceEffectiveParameters(_StrictFrozenModel):
    domain: Literal["financial_adjustments"]
    threshold_minor: int
    currency: str
    currency_exponent: Literal[2]
    restricted_categories: tuple[PathSafeId, ...]

    @field_validator("threshold_minor", mode="before")
    @classmethod
    def _validate_threshold(cls, value: object) -> int:
        return _exact_int(value, field="threshold_minor", minimum=0)

    @field_validator("currency", mode="before")
    @classmethod
    def _validate_currency(cls, value: object) -> str:
        return _currency(value, field="currency")

    @field_validator("currency_exponent", mode="before")
    @classmethod
    def _validate_currency_exponent(cls, value: object) -> Literal[2]:
        return _currency_exponent(value)

    @field_validator("restricted_categories", mode="before")
    @classmethod
    def _validate_categories(cls, value: object) -> tuple[PathSafeId, ...]:
        return _sorted_unique_ids(value, field="restricted_categories")


EffectiveParameters = Annotated[
    AccessEffectiveParameters | FinanceEffectiveParameters,
    Field(discriminator="domain"),
]


def _owned_effective_parameters_input(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if type(value) is AccessEffectiveParameters:
        model = cast(AccessEffectiveParameters, value)
    elif type(value) is FinanceEffectiveParameters:
        model = cast(FinanceEffectiveParameters, value)
    elif type(value) is dict:
        return cast(dict[str, object], value)
    else:
        raise ValueError(
            "effective_parameters must be an exact JSON object or validated parameters"
        )
    if not model._validated:
        raise ValueError("effective_parameters must be validated")
    return cast(dict[str, object], model.model_dump(mode="json"))


class DiscardedAuthority(_StrictFrozenModel):
    authority_id: PathSafeId
    reason_code: DiscardReasonCode

    @field_validator("authority_id", mode="before")
    @classmethod
    def _validate_id(cls, value: object) -> PathSafeId:
        return _path_safe_id(value, field="authority_id")

    @field_validator("reason_code", mode="before")
    @classmethod
    def _validate_reason(cls, value: object) -> DiscardReasonCode:
        return _enum_value(value, enum_type=DiscardReasonCode, field="reason_code")


def _owned_discarded_input(value: object) -> dict[str, object]:
    return _owned_model_input(value, expected_type=DiscardedAuthority, field="discarded_authority")


class _AuthorityDecisionPayload(_StrictFrozenModel):
    decision: Decision
    disposition: DecisionDisposition
    authority_query_hash: Sha256Ref
    at_time: str
    domain: Literal["access_provisioning", "financial_adjustments"]
    authority_set_hash: Sha256Ref
    effective_parameters: EffectiveParameters | None
    applicable_authority_ids: tuple[PathSafeId, ...]
    discarded_authorities: tuple[DiscardedAuthority, ...]
    supersession_path: tuple[PathSafeId, ...]
    exception_or_approval_ids: tuple[PathSafeId, ...]
    unresolved_conflict_ids: tuple[PathSafeId, ...]
    effective_rule_ids: tuple[PathSafeId, ...]
    reason_code: DecisionReasonCode

    @field_validator("decision", mode="before")
    @classmethod
    def _validate_decision(cls, value: object) -> Decision:
        return _enum_value(value, enum_type=Decision, field="decision")

    @field_validator("disposition", mode="before")
    @classmethod
    def _validate_disposition(cls, value: object) -> DecisionDisposition:
        return _enum_value(value, enum_type=DecisionDisposition, field="disposition")

    @field_validator("authority_query_hash", "authority_set_hash", mode="before")
    @classmethod
    def _validate_hashes(cls, value: object, info: ValidationInfo) -> Sha256Ref:
        return _sha256_reference(value, field=cast(str, info.field_name))

    @field_validator("at_time", mode="before")
    @classmethod
    def _validate_time(cls, value: object) -> str:
        return _utc_seconds(value, field="at_time")

    @field_validator("effective_parameters", mode="before")
    @classmethod
    def _validate_parameters(cls, value: object) -> dict[str, object] | None:
        return _owned_effective_parameters_input(value)

    @field_validator(
        "applicable_authority_ids",
        "exception_or_approval_ids",
        "unresolved_conflict_ids",
        "effective_rule_ids",
        mode="before",
    )
    @classmethod
    def _validate_sorted_id_arrays(
        cls, value: object, info: ValidationInfo
    ) -> tuple[PathSafeId, ...]:
        return _sorted_unique_ids(value, field=cast(str, info.field_name))

    @field_validator("supersession_path", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> tuple[PathSafeId, ...]:
        path = tuple(
            _path_safe_id(item, field="supersession_path")
            for item in _exact_array(value, field="supersession_path")
        )
        if len(set(path)) != len(path):
            raise ValueError("supersession_path must not repeat authority IDs")
        return path

    @field_validator("discarded_authorities", mode="before")
    @classmethod
    def _validate_discarded(cls, value: object) -> tuple[DiscardedAuthority, ...]:
        discarded = tuple(
            DiscardedAuthority.model_validate(_owned_discarded_input(item))
            for item in _exact_array(value, field="discarded_authorities")
        )
        ids = tuple(item.authority_id for item in discarded)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise ValueError(
                "discarded_authorities must be Unicode-sorted and unique by authority_id"
            )
        return discarded

    @field_validator("reason_code", mode="before")
    @classmethod
    def _validate_reason(cls, value: object) -> DecisionReasonCode:
        return _enum_value(value, enum_type=DecisionReasonCode, field="reason_code")

    @model_validator(mode="after")
    def _validate_decision_contract(self) -> Self:
        is_escalation = self.decision is Decision.ESCALATE
        has_escalation_disposition = self.disposition is DecisionDisposition.ESCALATE
        has_no_parameters = self.effective_parameters is None
        if not (is_escalation == has_escalation_disposition == has_no_parameters):
            raise AuthorityAdmissionError(
                "decision, disposition, and effective_parameters must agree on escalation"
            )

        if (
            self.effective_parameters is not None
            and self.effective_parameters.domain != self.domain
        ):
            raise AuthorityAdmissionError("effective_parameters domain must equal decision domain")

        expected_dispositions = {
            Decision.PROCEED: DecisionDisposition.PROCEED,
            Decision.BLOCK: DecisionDisposition.BLOCK,
            Decision.REQUIRE_APPROVAL: DecisionDisposition.REQUIRE_APPROVAL,
            Decision.PROCEED_UNDER_EXCEPTION: DecisionDisposition.PROCEED,
        }
        if (
            self.decision in expected_dispositions
            and self.disposition is not expected_dispositions[self.decision]
        ):
            raise AuthorityAdmissionError(
                "decision and disposition do not have the required mapping"
            )
        if self.decision is Decision.FOLLOW_SUPERSEDING_AUTHORITY and self.disposition not in {
            DecisionDisposition.PROCEED,
            DecisionDisposition.BLOCK,
            DecisionDisposition.REQUIRE_APPROVAL,
        }:
            raise AuthorityAdmissionError(
                "follow-superseding decision must have a governing disposition"
            )

        if self.reason_code in {
            DecisionReasonCode.NO_APPLICABLE_AUTHORITY,
            DecisionReasonCode.UNRESOLVED_TOP_RANK_CONFLICT,
        }:
            if self.decision is not Decision.ESCALATE:
                raise AuthorityAdmissionError("escalation reason code requires ESCALATE")
        elif self.reason_code is DecisionReasonCode.EFFECTIVE_RULE:
            if self.decision not in {Decision.PROCEED, Decision.BLOCK, Decision.REQUIRE_APPROVAL}:
                raise AuthorityAdmissionError(
                    "effective-rule reason requires an ordinary governing decision"
                )
        elif self.reason_code is DecisionReasonCode.EFFECTIVE_SUPERSEDING_RULE:
            if self.decision is not Decision.FOLLOW_SUPERSEDING_AUTHORITY:
                raise AuthorityAdmissionError(
                    "superseding-rule reason requires follow-superseding decision"
                )
        elif self.reason_code is DecisionReasonCode.VALID_APPROVAL:
            if self.decision is not Decision.PROCEED or not self.exception_or_approval_ids:
                raise AuthorityAdmissionError("valid approval requires PROCEED and evidence IDs")
        elif self.reason_code is DecisionReasonCode.VALID_WAIVER:
            if (
                self.decision is not Decision.PROCEED_UNDER_EXCEPTION
                or not self.exception_or_approval_ids
            ):
                raise AuthorityAdmissionError(
                    "valid waiver requires exception decision and evidence IDs"
                )

        if (
            self.decision is Decision.PROCEED_UNDER_EXCEPTION
            and self.reason_code is not DecisionReasonCode.VALID_WAIVER
        ):
            raise AuthorityAdmissionError("exception decision requires valid-waiver reason")
        if (
            self.decision is Decision.FOLLOW_SUPERSEDING_AUTHORITY
            and self.reason_code is not DecisionReasonCode.EFFECTIVE_SUPERSEDING_RULE
        ):
            raise AuthorityAdmissionError(
                "follow-superseding decision requires superseding-rule reason"
            )
        if (
            self.reason_code
            not in {DecisionReasonCode.VALID_APPROVAL, DecisionReasonCode.VALID_WAIVER}
            and self.exception_or_approval_ids
        ):
            raise AuthorityAdmissionError("evidence IDs require a valid approval or waiver")
        if not is_escalation and not self.effective_rule_ids:
            raise AuthorityAdmissionError("non-escalation decisions require effective rule IDs")

        return self


def _authority_decision_hash_trusted(decision: _AuthorityDecisionPayload) -> Sha256Ref:
    projection = decision.model_dump(mode="json")
    projection.pop("decision_hash", None)
    return cast(Sha256Ref, sha256_ref(projection))


class AuthorityDecision(_AuthorityDecisionPayload):
    decision_hash: Sha256Ref

    @field_validator("decision_hash", mode="before")
    @classmethod
    def _validate_decision_hash(cls, value: object) -> Sha256Ref:
        return _sha256_reference(value, field="decision_hash")

    @model_validator(mode="after")
    def _bind_decision_hash(self) -> Self:
        if self.decision_hash != _authority_decision_hash_trusted(self):
            raise ValueError("decision_hash does not bind authority decision")
        return self


def authority_decision_hash(decision: object) -> Sha256Ref:
    if type(decision) is not AuthorityDecision:
        raise ValueError("authority_decision_hash requires an exact AuthorityDecision")
    model = cast(AuthorityDecision, decision)
    if not model._validated:
        raise ValueError("authority_decision_hash requires a validated AuthorityDecision")
    return _authority_decision_hash_trusted(
        AuthorityDecision.model_validate(model.model_dump(mode="json"))
    )


def authority_set_hash(records: object) -> Sha256Ref:
    if type(records) not in {list, tuple}:
        raise ValueError("authority_records must be an exact built-in sequence")
    validated: list[AuthorityRecord] = []
    for item in cast(list[object] | tuple[object, ...], records):
        if type(item) is not AuthorityRecord:
            raise ValueError("authority_records must contain exact AuthorityRecord models")
        model = cast(AuthorityRecord, item)
        if not model._validated:
            raise ValueError("authority_records must be validated")
        validated.append(AuthorityRecord.model_validate(model.model_dump(mode="json")))
    ids = tuple(item.authority_id for item in validated)
    if len(set(ids)) != len(ids):
        raise AuthorityAdmissionError("authority_records must have unique authority IDs")
    ordered = sorted(validated, key=lambda item: item.authority_id)
    return cast(Sha256Ref, sha256_ref([item.model_dump(mode="json") for item in ordered]))


def create_authority_decision(
    *,
    authority_query: object,
    authority_set_hash: object,
    decision: object,
    disposition: object,
    effective_parameters: object,
    applicable_authority_ids: object,
    discarded_authorities: object,
    supersession_path: object,
    exception_or_approval_ids: object,
    unresolved_conflict_ids: object,
    effective_rule_ids: object,
    reason_code: object,
) -> AuthorityDecision:
    query_raw = _owned_query_input(authority_query)
    if query_raw["domain"] == ACCESS_DOMAIN:
        query = AccessAuthorityQuery.model_validate(query_raw)
    else:
        query = FinanceAuthorityQuery.model_validate(query_raw)
    raw = {
        "decision": decision,
        "disposition": disposition,
        "authority_query_hash": query.authority_query_hash,
        "at_time": query.at_time,
        "domain": query.domain,
        "authority_set_hash": authority_set_hash,
        "effective_parameters": effective_parameters,
        "applicable_authority_ids": applicable_authority_ids,
        "discarded_authorities": discarded_authorities,
        "supersession_path": supersession_path,
        "exception_or_approval_ids": exception_or_approval_ids,
        "unresolved_conflict_ids": unresolved_conflict_ids,
        "effective_rule_ids": effective_rule_ids,
        "reason_code": reason_code,
    }
    provisional = _AuthorityDecisionPayload.model_validate(raw)
    return AuthorityDecision.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "decision_hash": _authority_decision_hash_trusted(provisional),
        }
    )


__all__ = [
    "ACCESS_DOMAIN",
    "FINANCE_DOMAIN",
    "AccessAuthorityQuery",
    "AccessDescriptiveScope",
    "AccessEffectiveParameters",
    "AccessEvidenceScope",
    "AccessRuleScope",
    "AccessScope",
    "AuthorityAdmissionError",
    "AuthorityDecision",
    "AuthorityQuery",
    "AuthorityRecord",
    "AuthorityScope",
    "AuthoritySourceType",
    "Decision",
    "DecisionDisposition",
    "DecisionReasonCode",
    "DiscardReasonCode",
    "DiscardedAuthority",
    "EffectiveParameters",
    "FinanceAuthorityQuery",
    "FinanceDescriptiveScope",
    "FinanceEffectiveParameters",
    "FinanceEvidenceScope",
    "FinanceRuleScope",
    "FinanceScope",
    "IssuerAuthorization",
    "IssuerRegistry",
    "NormativeStatus",
    "RuleDisposition",
    "ScopeKind",
    "authority_decision_hash",
    "authority_query_hash",
    "authority_record_hash",
    "authority_set_hash",
    "create_access_authority_query",
    "create_authority_decision",
    "create_authority_record",
    "create_finance_authority_query",
    "create_issuer_registry",
]
