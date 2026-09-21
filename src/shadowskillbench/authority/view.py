from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from shadowskillbench.authority.models import (
    AccessAuthorityQuery,
    AuthorityDecision,
    AuthorityRecord,
    FinanceAuthorityQuery,
)
from shadowskillbench.authority.resolver import resolve_authority
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.engine.models import JsonObject


class AuthorityEvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    authority_id: str
    source_type: str
    issuer_id: str
    issuer_role: str
    authority_rank: int
    effective_at: str
    expires_at: str | None
    scope_dimensions: dict[str, str | int | bool | None]
    supersedes: tuple[str, ...]
    exception_to: tuple[str, ...]
    normative_status: str


class AuthorityEvidenceView(BaseModel):
    """Neutral authority metadata visible to the model, never a directive."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    records: tuple[AuthorityEvidenceRecord, ...]
    applicable_record_ids: tuple[str, ...]
    supersession_path: tuple[str, ...]
    evidence_reference_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    view_hash: str

    @field_validator("view_hash", mode="before")
    @classmethod
    def _hash(cls, value: object) -> str:
        if type(value) is not str or not value.startswith("sha256:") or len(value) != 71:
            raise ValueError("view_hash must be a sha256 reference")
        return value

    @model_validator(mode="after")
    def _bind(self) -> AuthorityEvidenceView:
        projection = self.model_dump(mode="json")
        projection.pop("view_hash")
        if self.view_hash != sha256_ref(projection):
            raise ValueError("view_hash does not bind authority evidence view")
        return self

    def projection(self) -> JsonObject:
        return cast(JsonObject, self.model_dump(mode="json"))


_SCOPE_DIMENSIONS = (
    "domain",
    "scope_kind",
    "subject_id",
    "resource_id",
    "organization_id",
    "geography_id",
    "role_id",
    "category_id",
    "period_id",
    "minimum_amount_minor",
    "minimum_inclusive",
    "maximum_amount_minor",
    "maximum_inclusive",
    "currency",
    "unit",
    "currency_exponent",
)


def _record_view(record: AuthorityRecord) -> AuthorityEvidenceRecord:
    scope = record.scope
    dimensions: dict[str, str | int | bool | None] = {}
    for name in _SCOPE_DIMENSIONS:
        value = getattr(scope, name, None)
        if value is not None and hasattr(value, "value"):
            value = value.value
        if type(value) in {str, int, bool} or value is None:
            dimensions[name] = cast(str | int | bool | None, value)
    return AuthorityEvidenceRecord(
        authority_id=record.authority_id,
        source_type=record.source_type.value,
        issuer_id=record.issuer_id,
        issuer_role=record.issuer_role,
        authority_rank=record.authority_rank,
        effective_at=record.effective_at,
        expires_at=record.expires_at,
        scope_dimensions=dimensions,
        supersedes=record.supersedes,
        exception_to=record.exception_to,
        normative_status=record.normative_status.value,
    )


def build_authority_evidence_view(
    records: tuple[AuthorityRecord, ...],
    query: AccessAuthorityQuery | FinanceAuthorityQuery,
    decision: AuthorityDecision,
) -> AuthorityEvidenceView:
    if type(records) is not tuple or not records:
        raise ValueError("records must be a nonempty exact tuple")
    if type(query) not in {AccessAuthorityQuery, FinanceAuthorityQuery}:
        raise ValueError("query must be an exact typed authority query")
    if type(decision) is not AuthorityDecision:
        raise ValueError("decision must be an exact AuthorityDecision")
    resolved = resolve_authority(records, query)
    if resolved != decision:
        raise ValueError("decision does not bind authority evidence")
    projection = {
        "records": [
            _record_view(record).model_dump(mode="json")
            for record in sorted(records, key=lambda item: item.authority_id)
        ],
        "applicable_record_ids": list(decision.applicable_authority_ids),
        "supersession_path": list(decision.supersession_path),
        "evidence_reference_ids": list(decision.exception_or_approval_ids),
        "conflict_ids": list(decision.unresolved_conflict_ids),
    }
    return AuthorityEvidenceView(
        records=tuple(
            _record_view(record) for record in sorted(records, key=lambda item: item.authority_id)
        ),
        applicable_record_ids=tuple(decision.applicable_authority_ids),
        supersession_path=tuple(decision.supersession_path),
        evidence_reference_ids=tuple(decision.exception_or_approval_ids),
        conflict_ids=tuple(decision.unresolved_conflict_ids),
        view_hash=sha256_ref(projection),
    )


__all__ = ["AuthorityEvidenceRecord", "AuthorityEvidenceView", "build_authority_evidence_view"]
