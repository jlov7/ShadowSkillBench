from __future__ import annotations

import re
from typing import Any, cast

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.domains.finance.models import (
    Adjustment,
    FinanceWorldState,
    MonthlyMetric,
    MonthlyReport,
    parse_finance_world,
    render_finance_world,
)
from shadowskillbench.domains.finance.reconciliation import ReconciliationError, reconcile_report
from shadowskillbench.engine.models import ActionCall, JsonObject, TransitionProposal, WorldState

_READ_ARGUMENTS: dict[str, tuple[str, str]] = {
    "get_period": ("period_id", "reporting_periods"),
    "get_ledger_snapshot": ("snapshot_id", "ledger_snapshots"),
    "get_adjustment": ("adjustment_id", "adjustments"),
    "get_approval": ("approval_id", "approvals"),
    "reconcile_report": ("report_id", "monthly_reports"),
}
_EFFECTFUL_TOOLS = {
    "create_adjustment",
    "attach_approval",
    "classify_adjustment",
    "publish_report",
    "finish_task",
}
_IMMUTABLE_ROOTS = ("reporting_periods", "ledger_snapshots", "approvals", "control_facts")
_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_ERROR_CODES = {
    "ACTION_INVALID",
    "ARGUMENTS_INVALID",
    "CONTROL_FACT_MISMATCH",
    "CREATE_CONFLICT",
    "CURRENCY_UNIT_MISMATCH",
    "ENTITY_NOT_FOUND",
    "ENTITY_STATE_INVALID",
    "EXCEPTION_CONFLICT",
    "FINISH_CONFLICT",
    "METRIC_SHAPE_INVALID",
    "RECONCILIATION_FAILED",
    "SOURCE_MISMATCH",
    "STATE_INVALID",
    "TASK_FINISHED",
    "TOOL_UNKNOWN",
}


class LocalActionError(ValueError):
    pass


def _failure(code: str) -> TransitionProposal:
    if code not in _ERROR_CODES:
        code = "STATE_INVALID"
    return TransitionProposal(
        local_status="failure",
        observation={"kind": "finance_error", "code": code},
        error_code=code,
    )


def _exact_call(value: object) -> ActionCall:
    if type(value) is not ActionCall:
        raise LocalActionError("ACTION_INVALID")
    try:
        return ActionCall.model_validate(
            {
                "action_id": value.action_id,
                "tool_name": value.tool_name,
                "arguments": value.arguments,
            }
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise LocalActionError("ACTION_INVALID") from error


def _validated_finance_state(value: object) -> tuple[WorldState, FinanceWorldState]:
    if type(value) is not WorldState:
        raise LocalActionError("STATE_INVALID")
    try:
        state = WorldState.model_validate(
            {
                "schema_version": value.schema_version,
                "world_id": value.world_id,
                "domain": value.domain,
                "seed": value.seed,
                "data": value.data,
            }
        )
        return state, parse_finance_world(state)
    except (AttributeError, TypeError, ValueError) as error:
        raise LocalActionError("STATE_INVALID") from error


def _arguments(call: ActionCall, required: tuple[str, ...]) -> dict[str, object]:
    if type(call.arguments) is not dict or set(call.arguments) != set(required):
        raise LocalActionError("ARGUMENTS_INVALID")
    return cast(dict[str, object], call.arguments)


def _identifier(arguments: dict[str, object], name: str) -> str:
    value = arguments[name]
    if type(value) is not str or _PATH_SAFE_ID.fullmatch(value) is None:
        raise LocalActionError("ARGUMENTS_INVALID")
    return value


def _summary(arguments: dict[str, object]) -> str:
    value = arguments["summary"]
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > 240
        or "\n" in value
        or "\r" in value
    ):
        raise LocalActionError("ARGUMENTS_INVALID")
    return value


def _rationale(arguments: dict[str, object]) -> str:
    value = arguments["rationale"]
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise LocalActionError("ARGUMENTS_INVALID")
    return value


def _detached_state(source: WorldState, world: FinanceWorldState) -> WorldState:
    return render_finance_world(world, world_id=source.world_id, seed=source.seed)


def _plain(value: object) -> dict[str, object]:
    if not hasattr(value, "model_dump"):
        raise RuntimeError("finance projection entity is invalid")
    payload = getattr(value, "model_dump")(mode="json")
    if type(payload) is not dict:
        raise RuntimeError("finance projection entity did not serialize to an object")
    return cast(dict[str, object], payload)


def _next_world(
    world: FinanceWorldState,
    *,
    adjustments: tuple[Adjustment, ...] | None = None,
    reports: tuple[MonthlyReport, ...] | None = None,
    finished: bool | None = None,
    finish_summary: str | None | object = ...,
) -> FinanceWorldState:
    return FinanceWorldState.model_validate(
        {
            "domain_schema_version": world.domain_schema_version,
            "case_id": world.case_id,
            "reporting_periods": [_plain(item) for item in world.reporting_periods],
            "ledger_snapshots": [_plain(item) for item in world.ledger_snapshots],
            "adjustments": [
                _plain(item) for item in (world.adjustments if adjustments is None else adjustments)
            ],
            "monthly_reports": [
                _plain(item) for item in (world.monthly_reports if reports is None else reports)
            ],
            "approvals": [_plain(item) for item in world.approvals],
            "control_facts": [_plain(item) for item in world.control_facts],
            "finished": world.finished if finished is None else finished,
            "finish_summary": world.finish_summary if finish_summary is ... else finish_summary,
        }
    )


def _success(
    source: WorldState, before: FinanceWorldState, after: FinanceWorldState, observation: JsonObject
) -> TransitionProposal:
    next_state = _detached_state(source, after)
    for root in _IMMUTABLE_ROOTS:
        if canonical_json_bytes(source.data[root]) != canonical_json_bytes(next_state.data[root]):
            raise RuntimeError("immutable finance facts drifted during action")
    return TransitionProposal(
        local_status="success", next_state=next_state, observation=observation
    )


def _entity_observation(tool: str, entity: object) -> JsonObject:
    if not hasattr(entity, "model_dump"):
        raise RuntimeError("finance entity is invalid")
    payload = getattr(entity, "model_dump")(mode="json")
    if type(payload) is not dict:
        raise RuntimeError("finance entity did not serialize to an object")
    return {"tool": tool, "entity": cast(Any, payload)}


def _replace_report_metric(world: FinanceWorldState, report: MonthlyReport) -> MonthlyReport:
    result = reconcile_report(world, report.report_id)
    if len(report.metrics) != 1:
        raise LocalActionError("METRIC_SHAPE_INVALID")
    metric = report.metrics[0]
    return MonthlyReport.model_validate(
        {
            "report_id": report.report_id,
            "period_id": report.period_id,
            "source_snapshot_id": report.source_snapshot_id,
            "adjustment_ids": list(report.adjustment_ids),
            "metrics": [
                MonthlyMetric(
                    metric_id=metric.metric_id,
                    signed_amount_minor=result.base_total_minor + result.adjustment_total_minor,
                    currency=metric.currency,
                    unit=metric.unit,
                    currency_exponent=metric.currency_exponent,
                )
            ],
            "status": report.status,
        }
    )


class FinanceAdapter:
    """Deterministic local financial operations, deliberately without policy resolution."""

    def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal:
        try:
            source, world = _validated_finance_state(state)
            action = _exact_call(call)
        except LocalActionError as error:
            return _failure(str(error))
        if action.tool_name not in {*_READ_ARGUMENTS, *_EFFECTFUL_TOOLS}:
            return _failure("TOOL_UNKNOWN")
        if (
            world.finished
            and action.tool_name in _EFFECTFUL_TOOLS
            and action.tool_name != "finish_task"
        ):
            return _failure("TASK_FINISHED")
        try:
            if action.tool_name in _READ_ARGUMENTS:
                return self._read(source, world, action)
            if action.tool_name == "create_adjustment":
                return self._create(source, world, action)
            if action.tool_name == "attach_approval":
                return self._attach(source, world, action)
            if action.tool_name == "classify_adjustment":
                return self._classify(source, world, action)
            if action.tool_name == "publish_report":
                return self._publish(source, world, action)
            return self._finish(source, world, action)
        except (LocalActionError, ReconciliationError) as error:
            return _failure(str(error))

    def _read(
        self, source: WorldState, world: FinanceWorldState, action: ActionCall
    ) -> TransitionProposal:
        argument, attribute = _READ_ARGUMENTS[action.tool_name]
        arguments = _arguments(action, (argument,))
        entity_id = _identifier(arguments, argument)
        if action.tool_name == "reconcile_report":
            result = reconcile_report(world, entity_id)
            observation: JsonObject = {
                "tool": action.tool_name,
                "report_id": result.report_id,
                "base_total_minor": result.base_total_minor,
                "adjustment_total_minor": result.adjustment_total_minor,
                "reported_total_minor": result.reported_total_minor,
                "delta_minor": result.delta_minor,
                "reconciled": result.reconciled,
                "reason": result.reason,
            }
            return _success(source, world, world, observation)
        identifier_name = argument
        entities = getattr(world, attribute)
        entity = next(
            (item for item in entities if getattr(item, identifier_name) == entity_id), None
        )
        if entity is None:
            raise LocalActionError("ENTITY_NOT_FOUND")
        return _success(source, world, world, _entity_observation(action.tool_name, entity))

    def _create(
        self, source: WorldState, world: FinanceWorldState, action: ActionCall
    ) -> TransitionProposal:
        names = (
            "report_id",
            "control_fact_id",
            "reported_category",
            "signed_amount_minor",
            "currency",
            "unit",
            "currency_exponent",
            "period_id",
            "portco_id",
            "rationale",
        )
        arguments = _arguments(action, names)
        report_id = _identifier(arguments, "report_id")
        fact_id = _identifier(arguments, "control_fact_id")
        report = next((item for item in world.monthly_reports if item.report_id == report_id), None)
        fact = next((item for item in world.control_facts if item.control_fact_id == fact_id), None)
        if report is None or fact is None:
            raise LocalActionError("ENTITY_NOT_FOUND")
        for name in ("reported_category", "period_id", "portco_id"):
            _identifier(arguments, name)
        _rationale(arguments)
        money_values = (
            arguments["signed_amount_minor"],
            arguments["currency"],
            arguments["unit"],
            arguments["currency_exponent"],
        )
        if (
            type(money_values[0]) is not int
            or type(money_values[1]) is not str
            or type(money_values[2]) is not str
            or type(money_values[3]) is not int
        ):
            raise LocalActionError("CURRENCY_UNIT_MISMATCH")
        if (money_values[1], money_values[2], money_values[3]) != (
            fact.currency,
            fact.unit,
            fact.currency_exponent,
        ):
            raise LocalActionError("CURRENCY_UNIT_MISMATCH")
        existing = next(
            (item for item in world.adjustments if item.adjustment_id == fact.adjustment_id), None
        )
        if existing is not None:
            if (
                existing.reported_category != arguments["reported_category"]
                or existing.rationale != arguments["rationale"]
                or existing.signed_amount_minor != money_values[0]
                or existing.period_id != arguments["period_id"]
                or existing.portco_id != arguments["portco_id"]
                or fact.adjustment_id not in report.adjustment_ids
            ):
                raise LocalActionError("CREATE_CONFLICT")
            return _success(
                source,
                world,
                world,
                {"tool": action.tool_name, "adjustment_id": existing.adjustment_id},
            )
        if report.status == "published":
            raise LocalActionError("ENTITY_STATE_INVALID")
        if (
            money_values[0] != fact.signed_amount_minor
            or arguments["period_id"] != fact.period_id
            or arguments["portco_id"] != fact.portco_id
        ):
            raise LocalActionError("CONTROL_FACT_MISMATCH")
        snapshot = next(
            (
                item
                for item in world.ledger_snapshots
                if item.snapshot_id == report.source_snapshot_id
            ),
            None,
        )
        if (
            snapshot is None
            or report.period_id != fact.period_id
            or report.source_snapshot_id != fact.source_snapshot_id
            or snapshot.source_version != fact.source_version
        ):
            raise LocalActionError("SOURCE_MISMATCH")
        if len(report.metrics) != 1:
            raise LocalActionError("METRIC_SHAPE_INVALID")
        adjustment = Adjustment.model_validate(
            {
                "adjustment_id": fact.adjustment_id,
                "reported_category": cast(str, arguments["reported_category"]),
                "signed_amount_minor": fact.signed_amount_minor,
                "currency": fact.currency,
                "unit": fact.unit,
                "currency_exponent": fact.currency_exponent,
                "period_id": fact.period_id,
                "portco_id": fact.portco_id,
                "rationale": cast(str, arguments["rationale"]),
                "approval_refs": [],
                "exception_ref": None,
                "status": "draft",
            }
        )
        draft = _next_world(
            world,
            adjustments=tuple(
                sorted((*world.adjustments, adjustment), key=lambda item: item.adjustment_id)
            ),
        )
        target = next(item for item in draft.monthly_reports if item.report_id == report_id)
        with_adjustment = MonthlyReport.model_validate(
            {
                "report_id": target.report_id,
                "period_id": target.period_id,
                "source_snapshot_id": target.source_snapshot_id,
                "adjustment_ids": sorted((*target.adjustment_ids, adjustment.adjustment_id)),
                "metrics": list(target.metrics),
                "status": target.status,
            }
        )
        reports = tuple(
            with_adjustment if item.report_id == report_id else item
            for item in draft.monthly_reports
        )
        after = _next_world(draft, reports=reports)
        recomputed = _replace_report_metric(after, with_adjustment)
        after = _next_world(
            after,
            reports=tuple(
                recomputed if item.report_id == report_id else item
                for item in after.monthly_reports
            ),
        )
        return _success(
            source,
            world,
            after,
            {"tool": action.tool_name, "adjustment_id": adjustment.adjustment_id},
        )

    def _attach(
        self, source: WorldState, world: FinanceWorldState, action: ActionCall
    ) -> TransitionProposal:
        arguments = _arguments(action, ("adjustment_id", "approval_id"))
        adjustment_id = _identifier(arguments, "adjustment_id")
        approval_id = _identifier(arguments, "approval_id")
        adjustment = next(
            (item for item in world.adjustments if item.adjustment_id == adjustment_id), None
        )
        approval = next((item for item in world.approvals if item.approval_id == approval_id), None)
        if adjustment is None or approval is None:
            raise LocalActionError("ENTITY_NOT_FOUND")
        if adjustment.status == "published":
            raise LocalActionError("ENTITY_STATE_INVALID")
        approval_refs = adjustment.approval_refs
        exception_ref = adjustment.exception_ref
        if approval.evidence_kind == "cfo_approval":
            approval_refs = tuple(sorted(set((*approval_refs, approval_id))))
        elif exception_ref is None or exception_ref == approval_id:
            exception_ref = approval_id
        else:
            raise LocalActionError("EXCEPTION_CONFLICT")
        replacement = Adjustment.model_validate(
            {
                "adjustment_id": adjustment.adjustment_id,
                "reported_category": adjustment.reported_category,
                "signed_amount_minor": adjustment.signed_amount_minor,
                "currency": adjustment.currency,
                "unit": adjustment.unit,
                "currency_exponent": adjustment.currency_exponent,
                "period_id": adjustment.period_id,
                "portco_id": adjustment.portco_id,
                "rationale": adjustment.rationale,
                "approval_refs": list(approval_refs),
                "exception_ref": exception_ref,
                "status": "attached",
            }
        )
        after = _next_world(
            world,
            adjustments=tuple(
                replacement if item.adjustment_id == adjustment_id else item
                for item in world.adjustments
            ),
        )
        return _success(
            source,
            world,
            after,
            {"tool": action.tool_name, "adjustment_id": adjustment_id, "approval_id": approval_id},
        )

    def _classify(
        self, source: WorldState, world: FinanceWorldState, action: ActionCall
    ) -> TransitionProposal:
        arguments = _arguments(action, ("adjustment_id", "reported_category"))
        adjustment_id = _identifier(arguments, "adjustment_id")
        category = _identifier(arguments, "reported_category")
        adjustment = next(
            (item for item in world.adjustments if item.adjustment_id == adjustment_id), None
        )
        if adjustment is None:
            raise LocalActionError("ENTITY_NOT_FOUND")
        if adjustment.status == "published":
            raise LocalActionError("ENTITY_STATE_INVALID")
        replacement = Adjustment.model_validate(
            {
                "adjustment_id": adjustment.adjustment_id,
                "reported_category": category,
                "signed_amount_minor": adjustment.signed_amount_minor,
                "currency": adjustment.currency,
                "unit": adjustment.unit,
                "currency_exponent": adjustment.currency_exponent,
                "period_id": adjustment.period_id,
                "portco_id": adjustment.portco_id,
                "rationale": adjustment.rationale,
                "approval_refs": list(adjustment.approval_refs),
                "exception_ref": adjustment.exception_ref,
                "status": adjustment.status,
            }
        )
        after = _next_world(
            world,
            adjustments=tuple(
                replacement if item.adjustment_id == adjustment_id else item
                for item in world.adjustments
            ),
        )
        return _success(
            source, world, after, {"tool": action.tool_name, "adjustment_id": adjustment_id}
        )

    def _publish(
        self, source: WorldState, world: FinanceWorldState, action: ActionCall
    ) -> TransitionProposal:
        arguments = _arguments(action, ("report_id",))
        report_id = _identifier(arguments, "report_id")
        report = next((item for item in world.monthly_reports if item.report_id == report_id), None)
        if report is None:
            raise LocalActionError("ENTITY_NOT_FOUND")
        if not reconcile_report(world, report_id).reconciled:
            raise LocalActionError("RECONCILIATION_FAILED")
        published_report = MonthlyReport.model_validate(
            {
                "report_id": report.report_id,
                "period_id": report.period_id,
                "source_snapshot_id": report.source_snapshot_id,
                "adjustment_ids": list(report.adjustment_ids),
                "metrics": list(report.metrics),
                "status": "published",
            }
        )
        included = set(report.adjustment_ids)
        adjustments = tuple(
            Adjustment.model_validate(
                {
                    "adjustment_id": item.adjustment_id,
                    "reported_category": item.reported_category,
                    "signed_amount_minor": item.signed_amount_minor,
                    "currency": item.currency,
                    "unit": item.unit,
                    "currency_exponent": item.currency_exponent,
                    "period_id": item.period_id,
                    "portco_id": item.portco_id,
                    "rationale": item.rationale,
                    "approval_refs": list(item.approval_refs),
                    "exception_ref": item.exception_ref,
                    "status": "published",
                }
            )
            if item.adjustment_id in included
            else item
            for item in world.adjustments
        )
        after = _next_world(
            world,
            adjustments=adjustments,
            reports=tuple(
                published_report if item.report_id == report_id else item
                for item in world.monthly_reports
            ),
        )
        return _success(source, world, after, {"tool": action.tool_name, "report_id": report_id})

    def _finish(
        self, source: WorldState, world: FinanceWorldState, action: ActionCall
    ) -> TransitionProposal:
        arguments = _arguments(action, ("summary",))
        summary = _summary(arguments)
        if world.finished:
            if world.finish_summary != summary:
                raise LocalActionError("FINISH_CONFLICT")
            return _success(source, world, world, {"tool": action.tool_name, "summary": summary})
        after = _next_world(world, finished=True, finish_summary=summary)
        return _success(source, world, after, {"tool": action.tool_name, "summary": summary})
