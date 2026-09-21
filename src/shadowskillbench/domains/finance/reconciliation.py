from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from shadowskillbench.domains.finance.models import (
    FinanceWorldState,
    parse_finance_world,
    render_finance_world,
)


class ReconciliationError(ValueError):
    pass


_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class ReconciliationResult(BaseModel):
    """A local numerical comparison; it contains no authority or policy conclusion."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    report_id: str
    snapshot_id: str
    base_total_minor: int
    adjustment_total_minor: int
    reported_total_minor: int
    delta_minor: int
    reconciled: bool
    reason: Literal["RECONCILED", "METRIC_MISMATCH"]


def reconcile_report(world: FinanceWorldState, report_id: str) -> ReconciliationResult:
    """Reconcile exactly one report using its bound snapshot and materialized adjustments."""
    if type(world) is not FinanceWorldState:
        raise ReconciliationError("WORLD_INVALID")
    try:
        world = parse_finance_world(render_finance_world(world, world_id="reconcile_world", seed=0))
    except Exception as error:
        raise ReconciliationError("WORLD_INVALID") from error
    if type(report_id) is not str or _PATH_SAFE_ID.fullmatch(report_id) is None:
        raise ReconciliationError("ARGUMENTS_INVALID")
    reports = {report.report_id: report for report in world.monthly_reports}
    report = reports.get(report_id)
    if report is None:
        raise ReconciliationError("ENTITY_NOT_FOUND")
    snapshots = {snapshot.snapshot_id: snapshot for snapshot in world.ledger_snapshots}
    snapshot = snapshots.get(report.source_snapshot_id)
    if snapshot is None or snapshot.period_id != report.period_id:
        raise ReconciliationError("SOURCE_MISMATCH")
    if len(report.metrics) != 1:
        raise ReconciliationError("METRIC_SHAPE_INVALID")
    metric = report.metrics[0]
    if (
        metric.currency != snapshot.currency
        or metric.unit != snapshot.unit
        or metric.currency_exponent != snapshot.currency_exponent
    ):
        raise ReconciliationError("CURRENCY_UNIT_MISMATCH")
    adjustments = {adjustment.adjustment_id: adjustment for adjustment in world.adjustments}
    facts = {fact.adjustment_id: fact for fact in world.control_facts}
    report_adjustments = []
    for adjustment_id in report.adjustment_ids:
        adjustment = adjustments.get(adjustment_id)
        fact = facts.get(adjustment_id)
        if adjustment is None or fact is None:
            raise ReconciliationError("ENTITY_NOT_FOUND")
        if (
            fact.period_id != report.period_id
            or fact.source_snapshot_id != snapshot.snapshot_id
            or fact.source_version != snapshot.source_version
        ):
            raise ReconciliationError("SOURCE_MISMATCH")
        report_adjustments.append(adjustment)
    base_total = sum(balance.signed_amount_minor for balance in snapshot.balances)
    adjustment_total = sum(item.signed_amount_minor for item in report_adjustments)
    reported_total = metric.signed_amount_minor
    delta = reported_total - (base_total + adjustment_total)
    reconciled = delta == 0
    return ReconciliationResult(
        report_id=report.report_id,
        snapshot_id=snapshot.snapshot_id,
        base_total_minor=base_total,
        adjustment_total_minor=adjustment_total,
        reported_total_minor=reported_total,
        delta_minor=delta,
        reconciled=reconciled,
        reason="RECONCILED" if reconciled else "METRIC_MISMATCH",
    )
