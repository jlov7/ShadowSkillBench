from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic.config import ExtraValues

from shadowskillbench.engine.models import JsonObject, WorldState

type PathSafeId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
type UsdCurrency = Literal["USD"]
type MinorUnit = Literal["minor_units"]

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class _FinanceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="never"
    )

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
        if type(obj) is not dict:
            raise ValueError(f"{cls.__name__} input must be an exact built-in object")
        return super().model_validate(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )


def _path_safe(value: object, *, field: str) -> PathSafeId:
    if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact path-safe identifier")
    return cast(PathSafeId, value)


def _signed_minor(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an exact signed integer minor-unit amount")
    return cast(int, value)


def _date_only(value: object, *, field: str) -> str:
    if type(value) is not str or _DATE_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ISO date-only value")
    try:
        from datetime import date

        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a valid ISO date-only value") from error
    return value


def _utc_seconds(value: object, *, field: str) -> str:
    if type(value) is not str or _UTC_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be canonical UTC seconds ending in Z")
    try:
        from datetime import datetime

        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"{field} must be a valid UTC timestamp") from error
    return value


def _exact_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise ValueError(f"{field} must be an exact JSON array")
    return tuple(cast(list[object], value))


def _sorted_unique_ids(value: object, *, field: str) -> tuple[PathSafeId, ...]:
    items = tuple(_path_safe(item, field=field) for item in _exact_sequence(value, field=field))
    if items != tuple(sorted(items)) or len(set(items)) != len(items):
        raise ValueError(f"{field} must be Unicode-sorted and unique")
    return items


def _sorted_entities[T: _FinanceModel](
    value: object, *, field: str, id_name: str, expected_type: type[T]
) -> tuple[T, ...]:
    items = _exact_sequence(value, field=field)
    validated_items: list[T] = []
    for item in items:
        if type(item) is expected_type:
            try:
                item = {name: getattr(item, name) for name in expected_type.model_fields}
            except AttributeError as error:
                raise ValueError(f"{expected_type.__name__} is incomplete") from error
        elif type(item) is not dict:
            raise ValueError(
                f"{field} must contain exact JSON objects or exact {expected_type.__name__} models"
            )
        validated_items.append(expected_type.model_validate(item))
    validated = tuple(validated_items)
    ids = tuple(cast(str, getattr(item, id_name)) for item in validated)
    if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
        raise ValueError(f"{field} must be Unicode ID-sorted with unique IDs")
    return validated


def _id_set(values: tuple[_FinanceModel, ...], attribute: str) -> set[str]:
    return {cast(str, getattr(value, attribute)) for value in values}


class ReportingPeriod(_FinanceModel):
    period_id: PathSafeId
    start_date: str
    end_date: str
    status: Literal["open", "closed"]

    @field_validator("period_id", mode="before")
    @classmethod
    def validate_period_id(cls, value: object) -> PathSafeId:
        return _path_safe(value, field="period_id")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def validate_dates(cls, value: object, info: object) -> str:
        return _date_only(value, field=cast(str, getattr(info, "field_name")))

    @model_validator(mode="after")
    def validate_range(self) -> ReportingPeriod:
        if self.start_date > self.end_date:
            raise ValueError("reporting period start_date must not exceed end_date")
        return self


class LedgerBalance(_FinanceModel):
    balance_id: PathSafeId
    signed_amount_minor: int
    currency: UsdCurrency
    unit: MinorUnit
    currency_exponent: Literal[2]

    @field_validator("balance_id", mode="before")
    @classmethod
    def validate_balance_id(cls, value: object) -> PathSafeId:
        return _path_safe(value, field="balance_id")

    @field_validator("signed_amount_minor", mode="before")
    @classmethod
    def validate_amount(cls, value: object) -> int:
        return _signed_minor(value, field="signed_amount_minor")


class LedgerSnapshot(_FinanceModel):
    snapshot_id: PathSafeId
    period_id: PathSafeId
    source_version: PathSafeId
    currency: UsdCurrency
    unit: MinorUnit
    currency_exponent: Literal[2]
    balances: tuple[LedgerBalance, ...]

    @field_validator("snapshot_id", "period_id", "source_version", mode="before")
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _path_safe(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("balances", mode="before")
    @classmethod
    def validate_balances(cls, value: object) -> tuple[LedgerBalance, ...]:
        return _sorted_entities(
            value, field="balances", id_name="balance_id", expected_type=LedgerBalance
        )

    @model_validator(mode="after")
    def validate_money(self) -> LedgerSnapshot:
        if any(
            balance.currency != self.currency
            or balance.unit != self.unit
            or balance.currency_exponent != self.currency_exponent
            for balance in self.balances
        ):
            raise ValueError("ledger balances must use the snapshot money representation")
        return self


class Adjustment(_FinanceModel):
    adjustment_id: PathSafeId
    reported_category: PathSafeId
    signed_amount_minor: int
    currency: UsdCurrency
    unit: MinorUnit
    currency_exponent: Literal[2]
    period_id: PathSafeId
    portco_id: PathSafeId
    rationale: str
    approval_refs: tuple[PathSafeId, ...]
    exception_ref: PathSafeId | None
    status: Literal["draft", "attached", "published"]

    @field_validator("adjustment_id", "reported_category", "period_id", "portco_id", mode="before")
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _path_safe(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("signed_amount_minor", mode="before")
    @classmethod
    def validate_amount(cls, value: object) -> int:
        return _signed_minor(value, field="signed_amount_minor")

    @field_validator("rationale", mode="before")
    @classmethod
    def validate_rationale(cls, value: object) -> str:
        if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
            raise ValueError("rationale must be nonblank single-line text")
        return value

    @field_validator("approval_refs", mode="before")
    @classmethod
    def validate_approval_refs(cls, value: object) -> tuple[PathSafeId, ...]:
        return _sorted_unique_ids(value, field="approval_refs")

    @field_validator("exception_ref", mode="before")
    @classmethod
    def validate_exception_ref(cls, value: object) -> PathSafeId | None:
        if value is None:
            return None
        return _path_safe(value, field="exception_ref")


class MonthlyMetric(_FinanceModel):
    metric_id: PathSafeId
    signed_amount_minor: int
    currency: UsdCurrency
    unit: MinorUnit
    currency_exponent: Literal[2]

    @field_validator("metric_id", mode="before")
    @classmethod
    def validate_metric_id(cls, value: object) -> PathSafeId:
        return _path_safe(value, field="metric_id")

    @field_validator("signed_amount_minor", mode="before")
    @classmethod
    def validate_amount(cls, value: object) -> int:
        return _signed_minor(value, field="signed_amount_minor")


class MonthlyReport(_FinanceModel):
    report_id: PathSafeId
    period_id: PathSafeId
    source_snapshot_id: PathSafeId
    adjustment_ids: tuple[PathSafeId, ...]
    metrics: tuple[MonthlyMetric, ...]
    status: Literal["draft", "published"]

    @field_validator("report_id", "period_id", "source_snapshot_id", mode="before")
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _path_safe(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("adjustment_ids", mode="before")
    @classmethod
    def validate_adjustment_ids(cls, value: object) -> tuple[PathSafeId, ...]:
        return _sorted_unique_ids(value, field="adjustment_ids")

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, value: object) -> tuple[MonthlyMetric, ...]:
        return _sorted_entities(
            value, field="metrics", id_name="metric_id", expected_type=MonthlyMetric
        )


class ApprovalEvidence(_FinanceModel):
    approval_id: PathSafeId
    evidence_kind: Literal["cfo_approval", "scoped_waiver"]
    approver_role: Literal["CFO"]
    adjustment_id: PathSafeId
    category_id: PathSafeId
    portco_id: PathSafeId
    period_id: PathSafeId
    source_snapshot_id: PathSafeId
    maximum_amount_minor: int
    currency: UsdCurrency
    unit: MinorUnit
    currency_exponent: Literal[2]
    effective_at: str
    expires_at: str | None
    source_version: PathSafeId
    rule_reference_id: PathSafeId

    @field_validator(
        "approval_id",
        "adjustment_id",
        "category_id",
        "portco_id",
        "period_id",
        "source_snapshot_id",
        "source_version",
        "rule_reference_id",
        mode="before",
    )
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _path_safe(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("maximum_amount_minor", mode="before")
    @classmethod
    def validate_cap(cls, value: object) -> int:
        return _signed_minor(value, field="maximum_amount_minor")

    @field_validator("effective_at", mode="before")
    @classmethod
    def validate_effective_at(cls, value: object) -> str:
        return _utc_seconds(value, field="effective_at")

    @field_validator("expires_at", mode="before")
    @classmethod
    def validate_expires_at(cls, value: object) -> str | None:
        if value is None:
            return None
        return _utc_seconds(value, field="expires_at")

    @model_validator(mode="after")
    def validate_window(self) -> ApprovalEvidence:
        if self.maximum_amount_minor < 0:
            raise ValueError("maximum_amount_minor must be nonnegative")
        if self.expires_at is not None and self.effective_at >= self.expires_at:
            raise ValueError("approval effective_at must precede expires_at")
        return self


class FinanceControlFact(_FinanceModel):
    control_fact_id: PathSafeId
    adjustment_id: PathSafeId
    original_economic_category: PathSafeId
    portco_id: PathSafeId
    signed_amount_minor: int
    currency: UsdCurrency
    unit: MinorUnit
    currency_exponent: Literal[2]
    period_id: PathSafeId
    source_snapshot_id: PathSafeId
    source_version: PathSafeId

    @field_validator(
        "control_fact_id",
        "adjustment_id",
        "original_economic_category",
        "portco_id",
        "period_id",
        "source_snapshot_id",
        "source_version",
        mode="before",
    )
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _path_safe(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("signed_amount_minor", mode="before")
    @classmethod
    def validate_amount(cls, value: object) -> int:
        return _signed_minor(value, field="signed_amount_minor")


class FinanceTaskCase(_FinanceModel):
    case_id: PathSafeId
    world_id: PathSafeId
    domain: Literal["financial_adjustments"]
    seed: int
    target_report_id: PathSafeId
    target_adjustment_id: PathSafeId
    target_control_fact_id: PathSafeId
    source_snapshot_id: PathSafeId
    period_id: PathSafeId
    source_version: PathSafeId

    @field_validator(
        "case_id",
        "world_id",
        "target_report_id",
        "target_adjustment_id",
        "target_control_fact_id",
        "source_snapshot_id",
        "period_id",
        "source_version",
        mode="before",
    )
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _path_safe(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("seed must be an exact integer")
        return cast(int, value)


def _exact_finance_case(value: object) -> FinanceTaskCase:
    if type(value) is not FinanceTaskCase:
        raise ValueError("case must be an exact FinanceTaskCase instance")
    fields = (
        "case_id",
        "world_id",
        "domain",
        "seed",
        "target_report_id",
        "target_adjustment_id",
        "target_control_fact_id",
        "source_snapshot_id",
        "period_id",
        "source_version",
    )
    try:
        raw = {field: getattr(value, field) for field in fields}
    except AttributeError as error:
        raise ValueError("FinanceTaskCase is incomplete") from error
    return FinanceTaskCase.model_validate(raw)


class FinanceWorldState(_FinanceModel):
    domain_schema_version: Literal["1.0"]
    case_id: PathSafeId
    reporting_periods: tuple[ReportingPeriod, ...]
    ledger_snapshots: tuple[LedgerSnapshot, ...]
    adjustments: tuple[Adjustment, ...]
    monthly_reports: tuple[MonthlyReport, ...]
    approvals: tuple[ApprovalEvidence, ...]
    control_facts: tuple[FinanceControlFact, ...]
    finished: bool
    finish_summary: str | None

    @field_validator("case_id", mode="before")
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _path_safe(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("reporting_periods", mode="before")
    @classmethod
    def validate_periods(cls, value: object) -> tuple[ReportingPeriod, ...]:
        return _sorted_entities(
            value, field="reporting_periods", id_name="period_id", expected_type=ReportingPeriod
        )

    @field_validator("ledger_snapshots", mode="before")
    @classmethod
    def validate_snapshots(cls, value: object) -> tuple[LedgerSnapshot, ...]:
        return _sorted_entities(
            value, field="ledger_snapshots", id_name="snapshot_id", expected_type=LedgerSnapshot
        )

    @field_validator("adjustments", mode="before")
    @classmethod
    def validate_adjustments(cls, value: object) -> tuple[Adjustment, ...]:
        return _sorted_entities(
            value, field="adjustments", id_name="adjustment_id", expected_type=Adjustment
        )

    @field_validator("monthly_reports", mode="before")
    @classmethod
    def validate_reports(cls, value: object) -> tuple[MonthlyReport, ...]:
        return _sorted_entities(
            value, field="monthly_reports", id_name="report_id", expected_type=MonthlyReport
        )

    @field_validator("approvals", mode="before")
    @classmethod
    def validate_approvals(cls, value: object) -> tuple[ApprovalEvidence, ...]:
        return _sorted_entities(
            value, field="approvals", id_name="approval_id", expected_type=ApprovalEvidence
        )

    @field_validator("control_facts", mode="before")
    @classmethod
    def validate_control_facts(cls, value: object) -> tuple[FinanceControlFact, ...]:
        return _sorted_entities(
            value,
            field="control_facts",
            id_name="control_fact_id",
            expected_type=FinanceControlFact,
        )

    @model_validator(mode="after")
    def validate_integrity(self) -> FinanceWorldState:
        period_ids = _id_set(self.reporting_periods, "period_id")
        snapshots = {snapshot.snapshot_id: snapshot for snapshot in self.ledger_snapshots}
        adjustment_ids = _id_set(self.adjustments, "adjustment_id")
        adjustments = {adjustment.adjustment_id: adjustment for adjustment in self.adjustments}
        approval_ids = _id_set(self.approvals, "approval_id")
        approvals = {approval.approval_id: approval for approval in self.approvals}
        facts_by_adjustment = {fact.adjustment_id: fact for fact in self.control_facts}
        if len(facts_by_adjustment) != len(self.control_facts) or not adjustment_ids.issubset(
            facts_by_adjustment
        ):
            raise ValueError("every adjustment must have exactly one immutable control fact")
        if any(snapshot.period_id not in period_ids for snapshot in self.ledger_snapshots):
            raise ValueError("ledger snapshot references an unknown reporting period")
        for adjustment in self.adjustments:
            refs = set(adjustment.approval_refs)
            if not refs.issubset(approval_ids):
                raise ValueError("adjustment references an unknown approval")
            if any(approvals[reference].evidence_kind != "cfo_approval" for reference in refs):
                raise ValueError("adjustment approval_refs must target CFO approvals")
            if (
                adjustment.exception_ref is not None
                and adjustment.exception_ref not in approval_ids
            ):
                raise ValueError("adjustment references an unknown exception")
            if (
                adjustment.exception_ref is not None
                and approvals[adjustment.exception_ref].evidence_kind != "scoped_waiver"
            ):
                raise ValueError("adjustment exception_ref must target a scoped waiver")
        for fact in self.control_facts:
            if fact.period_id not in period_ids:
                raise ValueError("control fact references an unknown reporting period")
            source_snapshot = snapshots.get(fact.source_snapshot_id)
            if source_snapshot is None:
                raise ValueError("control fact references an unknown source snapshot")
            if (
                source_snapshot.period_id != fact.period_id
                or source_snapshot.source_version != fact.source_version
            ):
                raise ValueError("control fact source snapshot must bind period and source version")
            if (
                source_snapshot.currency != fact.currency
                or source_snapshot.unit != fact.unit
                or source_snapshot.currency_exponent != fact.currency_exponent
            ):
                raise ValueError("control fact must match source snapshot money representation")
            adjustment = adjustments.get(fact.adjustment_id)
            if adjustment is not None and (
                adjustment.signed_amount_minor != fact.signed_amount_minor
                or adjustment.currency != fact.currency
                or adjustment.unit != fact.unit
                or adjustment.currency_exponent != fact.currency_exponent
                or adjustment.period_id != fact.period_id
                or adjustment.portco_id != fact.portco_id
            ):
                raise ValueError("adjustment operational facts must match immutable control facts")
        for approval in self.approvals:
            fact = facts_by_adjustment.get(approval.adjustment_id)
            if fact is None or approval.period_id not in period_ids:
                raise ValueError("approval references an unknown adjustment or reporting period")
            if (
                approval.currency != fact.currency
                or approval.unit != fact.unit
                or approval.currency_exponent != fact.currency_exponent
                or approval.source_snapshot_id != fact.source_snapshot_id
                or approval.source_version != fact.source_version
            ):
                raise ValueError(
                    "approval source and money representation must match its control fact"
                )
        for report in self.monthly_reports:
            snapshot = snapshots.get(report.source_snapshot_id)
            if snapshot is None or report.period_id not in period_ids:
                raise ValueError("report references an unknown snapshot or reporting period")
            if snapshot.period_id != report.period_id:
                raise ValueError("report period must match its source snapshot")
            if not set(report.adjustment_ids).issubset(adjustment_ids):
                raise ValueError("report references an unknown adjustment")
            for adjustment_id in report.adjustment_ids:
                fact = facts_by_adjustment[adjustment_id]
                if (
                    fact.period_id != report.period_id
                    or fact.source_version != snapshot.source_version
                ):
                    raise ValueError("report control facts must match period and source version")
                if (
                    fact.currency != snapshot.currency
                    or fact.unit != snapshot.unit
                    or fact.currency_exponent != snapshot.currency_exponent
                ):
                    raise ValueError(
                        "report control facts must match snapshot money representation"
                    )
            if any(
                metric.currency != snapshot.currency
                or metric.unit != snapshot.unit
                or metric.currency_exponent != snapshot.currency_exponent
                for metric in report.metrics
            ):
                raise ValueError("report metrics must match snapshot money representation")
        return self


def _exact_world_state(value: object) -> WorldState:
    if not isinstance(value, WorldState) or type(value) is not WorldState:
        raise ValueError("state must be an exact WorldState instance")
    try:
        raw = {
            "schema_version": value.schema_version,
            "world_id": value.world_id,
            "domain": value.domain,
            "seed": value.seed,
            "data": value.data,
        }
    except AttributeError as error:
        raise ValueError("WorldState is incomplete") from error
    return WorldState.model_validate(raw)


def parse_finance_world(state: object) -> FinanceWorldState:
    """Detach and validate an exact engine state as the finance projection."""
    exact_state = _exact_world_state(state)
    if exact_state.domain != "financial_adjustments":
        raise ValueError("WorldState domain must be financial_adjustments")
    projection_input: dict[str, object] = {}
    projection_input.update(exact_state.data)
    return FinanceWorldState.model_validate(projection_input)


def _entity_data(
    value: _FinanceModel, fields: tuple[str, ...], expected: type[_FinanceModel]
) -> dict[str, object]:
    if type(value) is not expected:
        raise ValueError(f"{expected.__name__} must be an exact constructed model")
    try:
        return {field: getattr(value, field) for field in fields}
    except AttributeError as error:
        raise ValueError(f"{expected.__name__} is incomplete") from error


def _balance_data(value: LedgerBalance) -> dict[str, object]:
    return _entity_data(
        value,
        ("balance_id", "signed_amount_minor", "currency", "unit", "currency_exponent"),
        LedgerBalance,
    )


def _snapshot_data(value: LedgerSnapshot) -> dict[str, object]:
    data = _entity_data(
        value,
        ("snapshot_id", "period_id", "source_version", "currency", "unit", "currency_exponent"),
        LedgerSnapshot,
    )
    data["balances"] = [_balance_data(balance) for balance in value.balances]
    return data


def _adjustment_data(value: Adjustment) -> dict[str, object]:
    data = _entity_data(
        value,
        (
            "adjustment_id",
            "reported_category",
            "signed_amount_minor",
            "currency",
            "unit",
            "currency_exponent",
            "period_id",
            "portco_id",
            "rationale",
            "exception_ref",
            "status",
        ),
        Adjustment,
    )
    data["approval_refs"] = list(value.approval_refs)
    return data


def _metric_data(value: MonthlyMetric) -> dict[str, object]:
    return _entity_data(
        value,
        ("metric_id", "signed_amount_minor", "currency", "unit", "currency_exponent"),
        MonthlyMetric,
    )


def _report_data(value: MonthlyReport) -> dict[str, object]:
    data = _entity_data(
        value, ("report_id", "period_id", "source_snapshot_id", "status"), MonthlyReport
    )
    data["adjustment_ids"] = list(value.adjustment_ids)
    data["metrics"] = [_metric_data(metric) for metric in value.metrics]
    return data


def _approval_data(value: ApprovalEvidence) -> dict[str, object]:
    return _entity_data(
        value,
        (
            "approval_id",
            "evidence_kind",
            "approver_role",
            "adjustment_id",
            "category_id",
            "portco_id",
            "period_id",
            "source_snapshot_id",
            "maximum_amount_minor",
            "currency",
            "unit",
            "currency_exponent",
            "effective_at",
            "expires_at",
            "source_version",
            "rule_reference_id",
        ),
        ApprovalEvidence,
    )


def _fact_data(value: FinanceControlFact) -> dict[str, object]:
    return _entity_data(
        value,
        (
            "control_fact_id",
            "adjustment_id",
            "original_economic_category",
            "portco_id",
            "signed_amount_minor",
            "currency",
            "unit",
            "currency_exponent",
            "period_id",
            "source_snapshot_id",
            "source_version",
        ),
        FinanceControlFact,
    )


def _period_data(value: ReportingPeriod) -> dict[str, object]:
    return _entity_data(value, ("period_id", "start_date", "end_date", "status"), ReportingPeriod)


def _projection_input(value: object) -> dict[str, object]:
    if not isinstance(value, FinanceWorldState) or type(value) is not FinanceWorldState:
        raise ValueError("projection must be an exact FinanceWorldState instance")
    try:
        return {
            "domain_schema_version": value.domain_schema_version,
            "case_id": value.case_id,
            "reporting_periods": [_period_data(period) for period in value.reporting_periods],
            "ledger_snapshots": [_snapshot_data(snapshot) for snapshot in value.ledger_snapshots],
            "adjustments": [_adjustment_data(adjustment) for adjustment in value.adjustments],
            "monthly_reports": [_report_data(report) for report in value.monthly_reports],
            "approvals": [_approval_data(approval) for approval in value.approvals],
            "control_facts": [_fact_data(fact) for fact in value.control_facts],
            "finished": value.finished,
            "finish_summary": value.finish_summary,
        }
    except AttributeError as error:
        raise ValueError("FinanceWorldState is incomplete") from error


def render_finance_world(projection: object, *, world_id: PathSafeId, seed: int) -> WorldState:
    """Render a detached, exact engine WorldState from a finance projection."""
    validated = FinanceWorldState.model_validate(_projection_input(projection))
    checked_world_id = _path_safe(world_id, field="world_id")
    if type(seed) is not int:
        raise ValueError("seed must be an exact integer")
    data: JsonObject = cast(
        JsonObject,
        {
            "domain_schema_version": validated.domain_schema_version,
            "case_id": validated.case_id,
            "reporting_periods": [_period_data(period) for period in validated.reporting_periods],
            "ledger_snapshots": [
                _snapshot_data(snapshot) for snapshot in validated.ledger_snapshots
            ],
            "adjustments": [_adjustment_data(adjustment) for adjustment in validated.adjustments],
            "monthly_reports": [_report_data(report) for report in validated.monthly_reports],
            "approvals": [_approval_data(approval) for approval in validated.approvals],
            "control_facts": [_fact_data(fact) for fact in validated.control_facts],
            "finished": validated.finished,
            "finish_summary": validated.finish_summary,
        },
    )
    return WorldState(
        schema_version="1.0",
        world_id=checked_world_id,
        domain="financial_adjustments",
        seed=seed,
        data=data,
    )


class FinanceFixture(_FinanceModel):
    fixture_id: PathSafeId
    seed: int
    case: FinanceTaskCase
    initial_state: WorldState
    source_snapshot_id: PathSafeId
    control_fact_ids: tuple[PathSafeId, ...]
    authority_reference_ids: tuple[PathSafeId, ...]

    @field_validator("fixture_id", "source_snapshot_id", mode="before")
    @classmethod
    def validate_ids(cls, value: object, info: object) -> PathSafeId:
        return _path_safe(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("case", mode="before")
    @classmethod
    def validate_case(cls, value: object) -> FinanceTaskCase:
        return _exact_finance_case(value)

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("seed must be an exact integer")
        return cast(int, value)

    @field_validator("control_fact_ids", "authority_reference_ids", mode="before")
    @classmethod
    def validate_id_arrays(cls, value: object, info: object) -> tuple[PathSafeId, ...]:
        return _sorted_unique_ids(value, field=cast(str, getattr(info, "field_name")))

    @field_validator("initial_state", mode="before")
    @classmethod
    def validate_initial_state(cls, value: object) -> WorldState:
        return _exact_world_state(value)

    @model_validator(mode="after")
    def bind_case_and_state(self) -> FinanceFixture:
        projection = parse_finance_world(self.initial_state)
        if self.seed != self.initial_state.seed or self.seed != self.case.seed:
            raise ValueError("fixture, case, and state seeds must match")
        if (
            self.case.world_id != self.initial_state.world_id
            or self.case.case_id != projection.case_id
        ):
            raise ValueError("case must bind the projected world and case identities")
        reports = {report.report_id: report for report in projection.monthly_reports}
        report = reports.get(self.case.target_report_id)
        if report is None or report.report_id not in reports:
            raise ValueError("case target report must exist")
        facts_by_id = {fact.control_fact_id: fact for fact in projection.control_facts}
        target_fact = facts_by_id.get(self.case.target_control_fact_id)
        if target_fact is None or target_fact.adjustment_id != self.case.target_adjustment_id:
            raise ValueError("case target control fact must bind its target adjustment")
        adjustment_ids = {adjustment.adjustment_id for adjustment in projection.adjustments}
        if (
            self.case.target_adjustment_id in adjustment_ids
            and self.case.target_adjustment_id not in report.adjustment_ids
        ):
            raise ValueError("materialized case target adjustment must belong to target report")
        snapshot = next(
            (
                item
                for item in projection.ledger_snapshots
                if item.snapshot_id == self.case.source_snapshot_id
            ),
            None,
        )
        if snapshot is None or snapshot.snapshot_id != self.source_snapshot_id:
            raise ValueError("fixture and case must bind the same source snapshot")
        if (
            report.source_snapshot_id != snapshot.snapshot_id
            or report.period_id != self.case.period_id
        ):
            raise ValueError("case report source and period must match")
        if snapshot.source_version != self.case.source_version:
            raise ValueError("case source version must match snapshot")
        control_ids = tuple(fact.control_fact_id for fact in projection.control_facts)
        if self.control_fact_ids != control_ids:
            raise ValueError("fixture control fact IDs must exactly bind projected facts")
        opaque_rule_refs = {approval.rule_reference_id for approval in projection.approvals}
        if not opaque_rule_refs.issubset(self.authority_reference_ids):
            raise ValueError("fixture must declare every opaque rule evidence reference")
        return self


Approval = ApprovalEvidence
