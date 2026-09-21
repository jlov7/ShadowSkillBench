"""Static, read-only workbench dataset export.

The workbench is a consumer of already-validated reporting evidence.  This
module deliberately exports only public report projections: it does not expose
prompts, raw provider payloads, hidden authority truth, development records,
or any operation that can mutate a benchmark result.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from shadowskillbench.analysis.dataset import OutcomeRow
from shadowskillbench.analysis.stage_a import RawCuPCount, ResponseCurvePoint
from shadowskillbench.analysis.stage_b import DeltaEstimate, Figure2Point, RateEstimate
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.episodes import EpisodeStage
from shadowskillbench.reporting.report import (
    ReportEvidence,
    ReportStatus,
    RepresentativeTrace,
    _exclusions_are_approved,
    report_status,
)
from shadowskillbench.reporting.selection import SelectionStatus

WORKBENCH_SCHEMA = "SSB-WORKBENCH-1"
WORKBENCH_SCHEMA_VERSION = "1.0"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_PUBLIC_STAGES = frozenset({EpisodeStage.CONFIRMATORY_A, EpisodeStage.CONFIRMATORY_B})

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type WorkbenchDataset = dict[str, JsonValue]


class WorkbenchExportError(ValueError):
    """Raised when a static export cannot be safely bound to report evidence."""


type PublicEventKind = Literal[
    "message",
    "tool_call",
    "state_change",
    "authority_lookup",
    "final_state",
    "reason_code",
]
type PublicMessageRole = Literal["system", "developer", "user", "tool"]


def _public_text(value: object, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise WorkbenchExportError(f"{field} must be nonblank sanitized text")
    return value


def _identifier(value: object, field: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise WorkbenchExportError(f"{field} must be a safe identifier")
    return value


@dataclass(frozen=True, slots=True)
class PublicTraceEvent:
    """One sanitized, model-visible timeline entry for a read-only episode compare.

    It intentionally has no prompt text, tool arguments, provider payload, oracle
    verdict, or extensible metadata field.  Each event type only permits its
    explicitly public, deterministic fields.
    """

    sequence: int
    kind: PublicEventKind
    summary: str
    role: PublicMessageRole | None = None
    tool_name: str | None = None
    local_status: Literal["success", "failure", "clarify"] | None = None
    before_state_hash: str | None = None
    after_state_hash: str | None = None
    authority_lookup_hash: str | None = None
    final_state_hash: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise WorkbenchExportError("event sequence must be a nonnegative exact integer")
        if self.kind not in {
            "message",
            "tool_call",
            "state_change",
            "authority_lookup",
            "final_state",
            "reason_code",
        }:
            raise WorkbenchExportError("event kind is invalid")
        _public_text(self.summary, "event summary")
        if self.role is not None and self.role not in {"system", "developer", "user", "tool"}:
            raise WorkbenchExportError("event role is invalid")
        fields = (
            self.role,
            self.tool_name,
            self.local_status,
            self.before_state_hash,
            self.after_state_hash,
            self.authority_lookup_hash,
            self.final_state_hash,
            self.reason_code,
        )
        if self.kind == "message":
            if self.role is None or any(value is not None for value in fields[1:]):
                raise WorkbenchExportError("message event has forbidden non-message fields")
        elif self.kind == "tool_call":
            if (
                self.role is not None
                or self.tool_name is None
                or self.local_status not in {"success", "failure", "clarify"}
                or any(value is not None for value in fields[3:])
            ):
                raise WorkbenchExportError("tool_call event must contain only public tool status")
            _identifier(self.tool_name, "tool_name")
        elif self.kind == "state_change":
            if (
                self.role is not None
                or self.tool_name is not None
                or self.local_status is not None
                or self.before_state_hash is None
                or self.after_state_hash is None
                or any(value is not None for value in fields[5:])
            ):
                raise WorkbenchExportError("state_change event must contain only state hashes")
            _sha256(self.before_state_hash, "before_state_hash")
            _sha256(self.after_state_hash, "after_state_hash")
        elif self.kind == "authority_lookup":
            if any(value is not None for value in fields[:5]) or any(
                value is not None for value in fields[6:]
            ):
                raise WorkbenchExportError("authority_lookup event has forbidden fields")
            _sha256(self.authority_lookup_hash, "authority_lookup_hash")
        elif self.kind == "final_state":
            if (
                any(value is not None for value in fields[:6])
                or self.final_state_hash is None
                or self.reason_code is not None
            ):
                raise WorkbenchExportError("final_state event must contain only final_state_hash")
            _sha256(self.final_state_hash, "final_state_hash")
        else:
            if any(value is not None for value in fields[:7]) or self.reason_code is None:
                raise WorkbenchExportError("reason_code event must contain only reason_code")
            _identifier(self.reason_code, "reason_code")


@dataclass(frozen=True, slots=True)
class PublicEpisodeTrace:
    """A custody-bound, sanitized public trace for one included episode."""

    episode_id: str
    trace_artifact_sha256: str
    events: tuple[PublicTraceEvent, ...]

    def __post_init__(self) -> None:
        _identifier(self.episode_id, "episode_id")
        _sha256(self.trace_artifact_sha256, "trace_artifact_sha256")
        if type(self.events) is not tuple or not self.events:
            raise WorkbenchExportError("trace events must be a non-empty exact tuple")
        if any(type(event) is not PublicTraceEvent for event in self.events):
            raise WorkbenchExportError("trace events must be exact PublicTraceEvent records")
        sequences = tuple(event.sequence for event in self.events)
        if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
            raise WorkbenchExportError("trace event sequences must be strictly ordered and unique")
        if sum(event.kind == "final_state" for event in self.events) != 1:
            raise WorkbenchExportError("trace must contain exactly one public final_state event")


@dataclass(frozen=True, slots=True)
class WorkbenchExport:
    """Canonical static dataset and its content-addressed identifier."""

    dataset: WorkbenchDataset

    @property
    def content_hash(self) -> str:
        value = self.dataset.get("content_hash")
        if type(value) is not str:
            raise WorkbenchExportError("workbench dataset is missing content_hash")
        return value

    def json_bytes(self) -> bytes:
        """Return canonical JSON suitable for a static frontend import."""
        return canonical_json_bytes(self.dataset)


def _enum_value(value: object) -> str:
    raw = getattr(value, "value", value)
    if type(raw) is not str:
        raise WorkbenchExportError("workbench projection contains a non-string enum value")
    return raw


def _sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise WorkbenchExportError(f"{field} must be a sha256 reference")
    return value


def _unique(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise WorkbenchExportError(f"{field} contains duplicate references")
    return values


def _public_rows(evidence: ReportEvidence) -> tuple[OutcomeRow, ...]:
    if any(type(row) is not OutcomeRow for row in evidence.dataset.rows):
        raise WorkbenchExportError("report dataset contains a non-exact outcome row")
    rows = tuple(row for row in evidence.dataset.rows if row.stage in _PUBLIC_STAGES)
    episode_ids = tuple(row.episode_id for row in rows)
    _unique(episode_ids, "confirmatory episode_id")
    return tuple(sorted(rows, key=lambda row: row.episode_id))


def _rows_for_scope(
    rows: tuple[OutcomeRow, ...],
    *,
    stage: EpisodeStage,
    scope: str,
    condition: str,
    contamination_ratio: float | None,
) -> tuple[OutcomeRow, ...]:
    return tuple(
        row
        for row in rows
        if row.stage is stage
        and (scope == "pooled" or row.domain == scope)
        and _enum_value(row.condition) == condition
        and row.contamination_ratio == contamination_ratio
    )


def _validate_response_curve(point: ResponseCurvePoint, rows: tuple[OutcomeRow, ...]) -> None:
    ratio = None if point.condition.value == "A1_POLICY_ONLY_SYSTEM" else point.contamination_ratio
    selected = _rows_for_scope(
        rows,
        stage=EpisodeStage.CONFIRMATORY_A,
        scope=point.scope,
        condition=point.condition.value,
        contamination_ratio=ratio,
    )
    numerator = sum(row.completion_under_policy for row in selected)
    if point.denominator != len(selected) or point.numerator != numerator:
        raise WorkbenchExportError("Figure 1 point does not match the bound analysis dataset")


def _validate_raw_count(count: RawCuPCount, rows: tuple[OutcomeRow, ...]) -> None:
    selected = _rows_for_scope(
        rows,
        stage=EpisodeStage.CONFIRMATORY_A,
        scope=count.scope,
        condition=count.condition.value,
        contamination_ratio=count.contamination_ratio,
    )
    numerator = sum(row.completion_under_policy for row in selected)
    if count.denominator != len(selected) or count.numerator != numerator:
        raise WorkbenchExportError("raw count does not match the bound analysis dataset")


def _validate_rate(rate: RateEstimate, rows: tuple[OutcomeRow, ...]) -> None:
    selected = tuple(
        row
        for row in rows
        if row.stage is EpisodeStage.CONFIRMATORY_B
        and (rate.domain == "pooled" or row.domain == rate.domain)
        and _enum_value(row.condition) == rate.condition.value
        and row.authority_class in rate.eligible_authority_classes
        and (
            rate.expected_authority_disposition is None
            or row.expected_authority_disposition is rate.expected_authority_disposition
        )
    )
    numerator = sum(getattr(row, rate.metric) for row in selected)
    if rate.denominator != len(selected) or rate.numerator != numerator:
        raise WorkbenchExportError("Figure 2 rate does not match the bound analysis dataset")


def _validate_figure_2(point: Figure2Point, rows: tuple[OutcomeRow, ...]) -> None:
    _validate_rate(point.unsafe_imitation, rows)
    _validate_rate(point.false_enforcement, rows)


def _figure_1(evidence: ReportEvidence, rows: tuple[OutcomeRow, ...]) -> dict[str, JsonValue]:
    points = tuple(point for point in evidence.stage_a.response_curves if point.scope == "pooled")
    for point in points:
        _validate_response_curve(point, rows)
    table_rows: list[JsonValue] = [
        {
            "condition": point.condition.value,
            "contamination_ratio": point.contamination_ratio,
            "estimate": point.estimate,
            "ci_low": point.ci_low,
            "ci_high": point.ci_high,
            "numerator": point.numerator,
            "denominator": point.denominator,
        }
        for point in sorted(
            points, key=lambda item: (item.condition.value, item.contamination_ratio)
        )
    ]
    return {
        "id": "figure_1",
        "title": "Completion-under-policy response curves",
        "summary": "Pooled Stage A response curves with confidence intervals and raw counts.",
        "columns": [
            "condition",
            "contamination_ratio",
            "estimate",
            "ci_low",
            "ci_high",
            "numerator",
            "denominator",
        ],
        "rows": table_rows,
    }


def _figure_2(evidence: ReportEvidence, rows: tuple[OutcomeRow, ...]) -> dict[str, JsonValue]:
    points = evidence.stage_b.figure_2_points
    for point in points:
        _validate_figure_2(point, rows)
    table_rows: list[JsonValue] = [
        {
            "domain": point.domain,
            "condition": point.condition.value,
            "unsafe_imitation": {
                "rate": point.unsafe_imitation.rate,
                "numerator": point.unsafe_imitation.numerator,
                "denominator": point.unsafe_imitation.denominator,
            },
            "false_enforcement": {
                "rate": point.false_enforcement.rate,
                "numerator": point.false_enforcement.numerator,
                "denominator": point.false_enforcement.denominator,
            },
        }
        for point in sorted(points, key=lambda item: (item.domain, item.condition.value))
    ]
    return {
        "id": "figure_2",
        "title": "Compliance and over-enforcement plane",
        "summary": "Stage B unsafe-imitation and false-enforcement rates with raw counts.",
        "columns": ["domain", "condition", "unsafe_imitation", "false_enforcement"],
        "rows": table_rows,
    }


def _stage_a_analysis(
    evidence: ReportEvidence, rows: tuple[OutcomeRow, ...]
) -> dict[str, JsonValue]:
    raw_counts = evidence.stage_a.raw_counts
    for count in raw_counts:
        _validate_raw_count(count, rows)
    counts: list[JsonValue] = [
        {
            "scope": count.scope,
            "condition": count.condition.value,
            "contamination_ratio": count.contamination_ratio,
            "numerator": count.numerator,
            "denominator": count.denominator,
        }
        for count in sorted(
            raw_counts,
            key=lambda item: (
                item.scope,
                item.condition.value,
                item.contamination_ratio is not None,
                item.contamination_ratio or 0.0,
            ),
        )
    ]
    estimands: list[JsonValue] = [
        {
            "estimand": estimand.estimand,
            "component": estimand.component,
            "scope": estimand.scope,
            "estimate": estimand.estimate,
            "ci_low": estimand.ci_low,
            "ci_high": estimand.ci_high,
        }
        for estimand in sorted(
            evidence.stage_a.estimands,
            key=lambda item: (item.estimand, item.component, item.scope),
        )
    ]
    marginal_effects: list[JsonValue] = [
        {
            "effect": effect.effect,
            "scope": effect.scope,
            "condition": effect.condition.value,
            "reference_condition": (
                None if effect.reference_condition is None else effect.reference_condition.value
            ),
            "contamination_ratio": effect.contamination_ratio,
            "estimate": effect.estimate,
            "ci_low": effect.ci_low,
            "ci_high": effect.ci_high,
        }
        for effect in sorted(
            evidence.stage_a.average_marginal_effects,
            key=lambda item: (
                item.effect,
                item.scope,
                item.condition.value,
                item.reference_condition.value if item.reference_condition is not None else "",
                item.contamination_ratio,
            ),
        )
    ]
    return {
        "raw_counts": counts,
        "estimands": estimands,
        "average_marginal_effects": marginal_effects,
    }


def _rate_projection(rate: RateEstimate) -> dict[str, JsonValue]:
    return {
        "domain": rate.domain,
        "condition": rate.condition.value,
        "metric": rate.metric,
        "authority_class": rate.authority_class,
        "eligible_authority_classes": list(rate.eligible_authority_classes),
        "expected_authority_disposition": (
            None
            if rate.expected_authority_disposition is None
            else rate.expected_authority_disposition.value
        ),
        "numerator": rate.numerator,
        "denominator": rate.denominator,
        "estimate": rate.rate,
        "ci_low": rate.ci_low,
        "ci_high": rate.ci_high,
    }


def _stage_b_analysis(evidence: ReportEvidence) -> dict[str, JsonValue]:
    stage_b = evidence.stage_b

    def delta_projection(effect: object) -> dict[str, JsonValue]:
        assert isinstance(effect, DeltaEstimate)
        return {
            "domain": effect.baseline.domain,
            "metric": effect.baseline.metric,
            "baseline": _rate_projection(effect.baseline),
            "comparison": _rate_projection(effect.comparison),
            "estimate": effect.delta,
            "ci_low": effect.ci_low,
            "ci_high": effect.ci_high,
        }

    return {
        "e6_unsafe_imitation": [
            _rate_projection(rate)
            for rate in sorted(
                stage_b.e6_unsafe_imitation,
                key=lambda item: (item.domain, item.condition.value),
            )
        ],
        "e7_false_enforcement": [
            _rate_projection(rate)
            for rate in sorted(
                stage_b.e7_false_enforcement,
                key=lambda item: (item.domain, item.condition.value),
            )
        ],
        "e8_authority_aware_gain": [
            {
                "domain": effect.domain,
                "baseline": [_rate_projection(rate) for rate in effect.baseline_components],
                "comparison": [_rate_projection(rate) for rate in effect.comparison_components],
                "baseline_estimate": effect.baseline_macro_rate,
                "comparison_estimate": effect.comparison_macro_rate,
                "estimate": effect.delta,
                "ci_low": effect.ci_low,
                "ci_high": effect.ci_high,
            }
            for effect in sorted(stage_b.e8_authority_aware_gain, key=lambda item: item.domain)
        ],
        "e9_completion_under_policy_delta": [
            delta_projection(effect)
            for effect in sorted(
                stage_b.e9_completion_under_policy_delta,
                key=lambda item: item.baseline.domain,
            )
        ],
        "e9_unsafe_imitation_delta": [
            delta_projection(effect)
            for effect in sorted(
                stage_b.e9_unsafe_imitation_delta,
                key=lambda item: item.baseline.domain,
            )
        ],
    }


def _domain_summary(rows: tuple[OutcomeRow, ...]) -> list[JsonValue]:
    summaries: list[JsonValue] = []
    for domain in ("access_provisioning", "financial_adjustments"):
        selected = tuple(row for row in rows if row.domain == domain)
        count = len(selected)
        summaries.append(
            {
                "domain": domain,
                "included_episodes": count,
                "completion_under_policy_rate": (
                    None
                    if count == 0
                    else sum(row.completion_under_policy for row in selected) / count
                ),
                "unsafe_imitation_rate": (
                    None if count == 0 else sum(row.unsafe_imitation for row in selected) / count
                ),
            }
        )
    return summaries


def _cost_summary(rows: tuple[OutcomeRow, ...]) -> dict[str, JsonValue]:
    missing = sum(row.total_cost_nanos is None for row in rows)
    return {
        "included_episodes": len(rows),
        "total_tokens": sum(row.total_tokens for row in rows),
        "missing_cost_receipts": missing,
        "total_cost_nanos": None if missing else sum(row.total_cost_nanos or 0 for row in rows),
    }


def _operational_metrics(rows: tuple[OutcomeRow, ...]) -> dict[str, JsonValue]:
    """Declare diagnostics that the current evidence contract cannot measure.

    ``OutcomeRow`` binds result and token/cost counts.  Human-review burden is
    the explicitly derived approval-workflow transition count; wall-clock
    latency remains unavailable because no timing field exists.
    """
    review_numerator = sum(row.human_review_request_count for row in rows)
    review_denominator = len(rows)
    denominator = len(rows)

    def rate(
        numerator: int,
        *,
        denominator_override: int | None = None,
        unit: str = "rate",
    ) -> dict[str, JsonValue]:
        bound_denominator = denominator if denominator_override is None else denominator_override
        return {
            "status": "available" if bound_denominator else "unavailable",
            "unit": unit,
            "value": None if not bound_denominator else numerator / bound_denominator,
            "numerator": numerator,
            "denominator": bound_denominator,
            "reason": "no included episodes are available" if not bound_denominator else None,
        }

    stage_b_rows = tuple(row for row in rows if row.stage is EpisodeStage.CONFIRMATORY_B)
    proceed_rows = tuple(
        row for row in stage_b_rows if row.expected_authority_disposition.value == "PROCEED"
    )
    a3 = tuple(
        row
        for row in rows
        if row.stage is EpisodeStage.CONFIRMATORY_A
        and row.condition.value == "A3_SKILL_POLICY_SAME_TIER"
    )
    a4 = tuple(
        row
        for row in rows
        if row.stage is EpisodeStage.CONFIRMATORY_A
        and row.condition.value == "A4_SKILL_POLICY_SYSTEM_TIER"
    )
    a3_mean = None if not a3 else sum(row.completion_under_policy for row in a3) / len(a3)
    a4_mean = None if not a4 else sum(row.completion_under_policy for row in a4) / len(a4)
    hierarchy_delta = None if a3_mean is None or a4_mean is None else a4_mean - a3_mean
    hierarchy_metric: dict[str, JsonValue] = {
        "status": "available" if hierarchy_delta is not None else "unavailable",
        "unit": "completion_under_policy_delta",
        "value": hierarchy_delta,
        "numerator": None,
        "denominator": None,
        "reason": "A3 and A4 rows are required" if hierarchy_delta is None else None,
    }
    return {
        "ordinary_task_completion": rate(sum(row.task_completion for row in rows)),
        "completion_under_policy": rate(sum(row.completion_under_policy for row in rows)),
        "unsafe_imitation": rate(
            sum(row.unsafe_imitation for row in stage_b_rows),
            denominator_override=len(stage_b_rows),
        ),
        "false_enforcement": rate(
            sum(row.false_enforcement for row in proceed_rows),
            denominator_override=len(proceed_rows),
        ),
        "instruction_hierarchy_delta": hierarchy_metric,
        "latency": {
            "status": "unavailable",
            "unit": "milliseconds",
            "value": None,
            "reason": "episode artifacts do not record wall-clock latency",
        },
        "human_review_burden": {
            "status": "available" if review_denominator else "unavailable",
            "unit": "review_requests_per_episode",
            "value": (None if not review_denominator else review_numerator / review_denominator),
            "numerator": review_numerator,
            "denominator": review_denominator,
            "reason": "unique successful pending-to-approval_requested transitions",
        },
    }


def _trace_summary(
    traces: tuple[RepresentativeTrace, ...],
    rows: tuple[OutcomeRow, ...],
    public_traces: tuple[PublicEpisodeTrace, ...],
) -> list[JsonValue]:
    values: list[JsonValue] = []
    for trace in traces:
        selection = trace.selection
        if selection.status is SelectionStatus.HOLD_NO_CANDIDATE:
            values.append({"stage": trace.stage, "status": "hold_no_candidate"})
            continue
        if selection.status is not SelectionStatus.SELECTED or selection.selected is None:
            raise WorkbenchExportError("representative trace has an unsupported selection status")
        case_id = selection.selected.case_id
        episode_stage = (
            EpisodeStage.CONFIRMATORY_A if trace.stage == "A" else EpisodeStage.CONFIRMATORY_B
        )
        if not any(row.stage is episode_stage and row.held_out_case_id == case_id for row in rows):
            raise WorkbenchExportError("representative trace case is not bound to a public episode")
        assert trace.trace_artifact_sha256 is not None
        if not any(
            public_trace.trace_artifact_sha256 == trace.trace_artifact_sha256
            and any(
                row.episode_id == public_trace.episode_id
                and row.stage is episode_stage
                and row.held_out_case_id == case_id
                for row in rows
            )
            for public_trace in public_traces
        ):
            raise WorkbenchExportError(
                "selected representative trace artifact is not bound to an exported public trace"
            )
        values.append(
            {
                "stage": trace.stage,
                "status": "selected",
                "case_id": case_id,
                "trace_artifact_sha256": _sha256(
                    trace.trace_artifact_sha256, "trace_artifact_sha256"
                ),
                "summary": cast(str, trace.summary),
            }
        )
    return values


def _public_event_projection(event: PublicTraceEvent) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {
        "sequence": event.sequence,
        "kind": event.kind,
        "summary": event.summary,
    }
    if event.role is not None:
        value["role"] = event.role
    if event.tool_name is not None:
        value["tool_name"] = event.tool_name
    if event.local_status is not None:
        value["local_status"] = event.local_status
    if event.before_state_hash is not None:
        value["before_state_hash"] = event.before_state_hash
    if event.after_state_hash is not None:
        value["after_state_hash"] = event.after_state_hash
    if event.authority_lookup_hash is not None:
        value["authority_lookup_hash"] = event.authority_lookup_hash
    if event.final_state_hash is not None:
        value["final_state_hash"] = event.final_state_hash
    if event.reason_code is not None:
        value["reason_code"] = event.reason_code
    return value


def _validate_public_event_projection(value: object) -> PublicTraceEvent:
    if type(value) is not dict:
        raise WorkbenchExportError("public trace event must be a dictionary")
    kind = value.get("kind")
    allowed_fields: dict[str, set[str]] = {
        "message": {"sequence", "kind", "summary", "role"},
        "tool_call": {"sequence", "kind", "summary", "tool_name", "local_status"},
        "state_change": {
            "sequence",
            "kind",
            "summary",
            "before_state_hash",
            "after_state_hash",
        },
        "authority_lookup": {"sequence", "kind", "summary", "authority_lookup_hash"},
        "final_state": {"sequence", "kind", "summary", "final_state_hash"},
        "reason_code": {"sequence", "kind", "summary", "reason_code"},
    }
    if type(kind) is not str or kind not in allowed_fields or set(value) != allowed_fields[kind]:
        raise WorkbenchExportError("public trace event has forbidden or missing fields")
    return PublicTraceEvent(
        sequence=cast(int, value["sequence"]),
        kind=cast(PublicEventKind, kind),
        summary=cast(str, value["summary"]),
        role=cast(PublicMessageRole | None, value.get("role")),
        tool_name=cast(str | None, value.get("tool_name")),
        local_status=cast(
            Literal["success", "failure", "clarify"] | None, value.get("local_status")
        ),
        before_state_hash=cast(str | None, value.get("before_state_hash")),
        after_state_hash=cast(str | None, value.get("after_state_hash")),
        authority_lookup_hash=cast(str | None, value.get("authority_lookup_hash")),
        final_state_hash=cast(str | None, value.get("final_state_hash")),
        reason_code=cast(str | None, value.get("reason_code")),
    )


def _public_trace_projection(
    traces: tuple[PublicEpisodeTrace, ...], rows: tuple[OutcomeRow, ...]
) -> list[JsonValue]:
    if type(traces) is not tuple:
        raise WorkbenchExportError("public_episode_traces must be an exact tuple")
    if any(type(trace) is not PublicEpisodeTrace for trace in traces):
        raise WorkbenchExportError(
            "public_episode_traces must contain exact PublicEpisodeTrace records"
        )
    episode_ids = {row.episode_id for row in rows}
    trace_ids = tuple(trace.episode_id for trace in traces)
    _unique(trace_ids, "public episode trace IDs")
    if any(trace_id not in episode_ids for trace_id in trace_ids):
        raise WorkbenchExportError("public episode trace is not bound to an included episode")
    return [
        {
            "episode_id": trace.episode_id,
            "trace_artifact_sha256": trace.trace_artifact_sha256,
            "events": [_public_event_projection(event) for event in trace.events],
        }
        for trace in sorted(traces, key=lambda item: item.episode_id)
    ]


def _episode_index(rows: tuple[OutcomeRow, ...]) -> list[JsonValue]:
    """Export identifiers and custody refs only; never raw trace/model data or oracle truth."""
    return [
        {
            "episode_id": row.episode_id,
            "stage": row.stage.value,
            "condition": row.condition.value,
            "domain": row.domain,
            "held_out_case_id": row.held_out_case_id,
            "repeat_index": row.repeat_index,
            "planned_episode_hash": _sha256(row.planned_episode_hash, "planned_episode_hash"),
            "manifest_hash": _sha256(row.manifest_hash, "manifest_hash"),
            "result_hash": _sha256(row.result_hash, "result_hash"),
            "score_hash": _sha256(row.score_hash, "score_hash"),
        }
        for row in rows
    ]


def _claim_ceiling(evidence: ReportEvidence, status: ReportStatus) -> dict[str, JsonValue]:
    excluded = sum(
        item.stage in {EpisodeStage.CONFIRMATORY_A, EpisodeStage.CONFIRMATORY_B}
        for item in evidence.dataset.exclusions
    )
    policy = evidence.exclusion_policy
    exclusion_summary: dict[str, JsonValue] = {
        "count": excluded,
        "approved": _exclusions_are_approved(evidence.dataset, policy),
        "max_total": None if policy is None else policy.max_total_exclusions,
        "max_per_primary_cell": (
            None if policy is None else policy.max_exclusions_per_primary_cell
        ),
    }
    if status is ReportStatus.CONFIRMATORY:
        return {
            "status": "confirmatory",
            "descriptive_only": False,
            "technical_exclusions": exclusion_summary,
            "message": (
                "Confirmatory evidence is complete; findings remain bounded by the claims ledger."
            ),
        }
    return {
        "status": "hold",
        "descriptive_only": True,
        "technical_exclusions": exclusion_summary,
        "message": (
            "HOLD: this dataset is descriptive only and contains no confirmatory finding label."
        ),
    }


def _protocol_projection(evidence: ReportEvidence) -> dict[str, JsonValue]:
    protocol = evidence.protocol
    prompt_hashes = _unique(protocol.prompt_hashes, "prompt_hashes")
    return {
        "protocol_tag": protocol.protocol_tag,
        "code_commit": protocol.code_commit,
        "freeze_manifest_sha256": _sha256(
            protocol.freeze_manifest_sha256, "freeze_manifest_sha256"
        ),
        "analysis_dataset_sha256": _sha256(
            evidence.analysis_dataset_sha256, "analysis_dataset_sha256"
        ),
        "prompt_hashes": list(prompt_hashes),
        "external_anchor_locator": protocol.external_anchor_locator,
        "custody_locator": protocol.custody_locator,
        "custody_mode": protocol.custody_mode,
        "anchor_receipt_sha256": (
            None
            if protocol.anchor_receipt_sha256 is None
            else _sha256(protocol.anchor_receipt_sha256, "anchor_receipt_sha256")
        ),
        "anchor_verified": protocol.anchor_verified,
        "experiment_plan_sha256": (
            None
            if protocol.experiment_plan_sha256 is None
            else _sha256(protocol.experiment_plan_sha256, "experiment_plan_sha256")
        ),
        "experiment_runtime_binding_sha256": (
            None
            if protocol.experiment_runtime_binding_sha256 is None
            else _sha256(
                protocol.experiment_runtime_binding_sha256,
                "experiment_runtime_binding_sha256",
            )
        ),
        "experiment_audit_report_sha256": (
            None
            if protocol.experiment_audit_report_sha256 is None
            else _sha256(protocol.experiment_audit_report_sha256, "experiment_audit_report_sha256")
        ),
        "experiment_audit_status": protocol.experiment_audit_status,
        "runtime_binding": (
            None
            if evidence.runtime_binding is None
            else {
                "plan_hash": evidence.runtime_binding.plan_hash,
                "package_hash": evidence.runtime_binding.package_hash,
                "run_descriptor_hash": evidence.runtime_binding.run_descriptor_hash,
                "freeze_manifest_hash": evidence.runtime_binding.freeze_manifest_hash,
                "anchor_receipt_hash": evidence.runtime_binding.anchor_receipt_hash,
                "operator_endpoint_hash": evidence.runtime_binding.operator_endpoint_hash,
                "operator_api_key_environment": (
                    evidence.runtime_binding.operator_api_key_environment
                ),
                "provider": evidence.runtime_binding.provider,
                "model": evidence.runtime_binding.model,
                "model_version": evidence.runtime_binding.model_version,
            }
        ),
        "reproduce_command": protocol.reproduce_command,
    }


def _preregistration_projection(evidence: ReportEvidence) -> dict[str, JsonValue]:
    """Return public configuration/corpus metadata, or an explicit HOLD projection."""
    core = evidence.preregistration.core
    if not evidence.preregistration.valid or core is None:
        return {"status": "hold_missing_valid_preregistration"}
    try:
        compiler = core.models.skill_compiler
        executor = core.models.skill_executor
        corpus = core.corpus_counts
        return {
            "status": "bound",
            "model_configurations": {
                "skill_compiler": {
                    "provider": compiler.provider,
                    "model": compiler.model,
                    "model_version_date": compiler.model_version_date,
                    "system_prompt_hash": compiler.system_prompt_hash,
                    "structured_output_schema_hash": compiler.structured_output_schema_hash,
                    "temperature": str(compiler.temperature),
                    "seed": compiler.seed,
                    "max_tokens": compiler.max_tokens,
                },
                "skill_executor": {
                    "provider": executor.provider,
                    "model": executor.model,
                    "model_version_date": executor.model_version_date,
                    "runtime_prompt_hashes": executor.runtime_prompt_hashes,
                    "temperature": str(executor.temperature),
                    "seed_policy": executor.seed_policy,
                    "max_turns": executor.max_turns,
                    "max_tool_calls": executor.max_tool_calls,
                    "max_tokens": executor.max_tokens,
                },
            },
            "corpus_summary": {
                "stage_a": {
                    "skill_bundles": corpus.stage_a.skill_bundles,
                    "held_out_cases_per_domain": corpus.stage_a.held_out_cases_per_domain,
                    "repeats": corpus.stage_a.repeats,
                    "total": corpus.stage_a.total,
                },
                "stage_b": {
                    "r75_skills_per_domain": corpus.stage_b.r75_skills_per_domain,
                    "held_out_cases_per_authority_class_domain": (
                        corpus.stage_b.held_out_cases_per_authority_class_domain
                    ),
                    "authority_classes": corpus.stage_b.authority_classes,
                    "conditions": corpus.stage_b.conditions,
                    "repeats": corpus.stage_b.repeats,
                    "total": corpus.stage_b.total,
                },
            },
        }
    except AttributeError as error:
        raise WorkbenchExportError("validated preregistration core is incomplete") from error


def build_workbench_export(
    evidence: ReportEvidence,
    *,
    public_episode_traces: tuple[PublicEpisodeTrace, ...] = (),
) -> WorkbenchExport:
    """Build a deterministic static workbench contract from report evidence.

    A HOLD is exportable as explicitly descriptive data.  Any contradiction
    between displayed figures, selected traces, custody references, and the
    bound analysis table fails closed.
    """
    if type(evidence) is not ReportEvidence:
        raise WorkbenchExportError("evidence must be an exact ReportEvidence")
    rows = _public_rows(evidence)
    status = report_status(evidence)
    trace_projection = _public_trace_projection(public_episode_traces, rows)
    dataset: WorkbenchDataset = {
        "schema": WORKBENCH_SCHEMA,
        "schema_version": WORKBENCH_SCHEMA_VERSION,
        "read_only": True,
        "claim_ceiling": _claim_ceiling(evidence, status),
        "protocol": _protocol_projection(evidence),
        "preregistration": _preregistration_projection(evidence),
        "analysis": {
            **_stage_a_analysis(evidence, rows),
            "stage_b": _stage_b_analysis(evidence),
        },
        "figures": [_figure_1(evidence, rows), _figure_2(evidence, rows)],
        "summaries": {
            "domains": _domain_summary(rows),
            "cost": _cost_summary(rows),
            "metrics": _operational_metrics(rows),
            "representative_traces": _trace_summary(
                evidence.representative_traces, rows, public_episode_traces
            ),
        },
        "episode_index": _episode_index(rows),
        "episode_traces": trace_projection,
        "claims": {
            "ledger_valid": evidence.claims.valid,
            "renderable_finding_ids": (
                [claim.id for claim in evidence.claims.renderable_findings]
                if status is ReportStatus.CONFIRMATORY and evidence.claims.valid
                else []
            ),
        },
        "limitations": list(evidence.limitations),
    }
    dataset["content_hash"] = sha256_ref(dataset)
    validate_workbench_export(dataset)
    return WorkbenchExport(dataset)


def validate_workbench_export(dataset: WorkbenchDataset) -> None:
    """Validate exact static-contract fields, uniqueness, and its self-hash."""
    if type(dataset) is not dict:
        raise WorkbenchExportError("workbench dataset must be an exact dictionary")
    required = {
        "schema",
        "schema_version",
        "read_only",
        "claim_ceiling",
        "protocol",
        "preregistration",
        "analysis",
        "figures",
        "summaries",
        "episode_index",
        "episode_traces",
        "claims",
        "limitations",
        "content_hash",
    }
    if set(dataset) != required:
        raise WorkbenchExportError("workbench dataset fields do not match the static contract")
    if (
        dataset["schema"] != WORKBENCH_SCHEMA
        or dataset["schema_version"] != WORKBENCH_SCHEMA_VERSION
    ):
        raise WorkbenchExportError("workbench dataset has an unsupported schema or version")
    if dataset["read_only"] is not True:
        raise WorkbenchExportError("workbench dataset must declare read_only true")
    content_hash = _sha256(dataset["content_hash"], "content_hash")
    unsigned = {key: value for key, value in dataset.items() if key != "content_hash"}
    if content_hash != sha256_ref(unsigned):
        raise WorkbenchExportError("workbench dataset content_hash does not bind its contents")
    claim_ceiling = dataset["claim_ceiling"]
    if type(claim_ceiling) is not dict:
        raise WorkbenchExportError("claim_ceiling must be a dictionary")
    if claim_ceiling.get("status") == "hold" and claim_ceiling.get("descriptive_only") is not True:
        raise WorkbenchExportError("HOLD dataset must be descriptive only")
    if claim_ceiling.get("status") not in {"hold", "confirmatory"}:
        raise WorkbenchExportError("claim_ceiling has an invalid status")
    analysis = dataset["analysis"]
    if type(analysis) is not dict or set(analysis) != {
        "raw_counts",
        "estimands",
        "average_marginal_effects",
        "stage_b",
    }:
        raise WorkbenchExportError("analysis fields do not match the static contract")
    for name in ("raw_counts", "estimands", "average_marginal_effects"):
        if type(analysis[name]) is not list:
            raise WorkbenchExportError(f"analysis.{name} must be a list")
    stage_b = analysis["stage_b"]
    if type(stage_b) is not dict or set(stage_b) != {
        "e6_unsafe_imitation",
        "e7_false_enforcement",
        "e8_authority_aware_gain",
        "e9_completion_under_policy_delta",
        "e9_unsafe_imitation_delta",
    }:
        raise WorkbenchExportError("analysis.stage_b fields do not match the static contract")
    for name in stage_b:
        if type(stage_b[name]) is not list:
            raise WorkbenchExportError(f"analysis.stage_b.{name} must be a list")
    rate_fields = {
        "domain",
        "condition",
        "metric",
        "authority_class",
        "eligible_authority_classes",
        "expected_authority_disposition",
        "numerator",
        "denominator",
        "estimate",
        "ci_low",
        "ci_high",
    }
    for name in ("e6_unsafe_imitation", "e7_false_enforcement"):
        rates = cast(list[JsonValue], stage_b[name])
        for rate in rates:
            if type(rate) is not dict or set(rate) != rate_fields:
                raise WorkbenchExportError(f"analysis.stage_b.{name} has invalid rate fields")
    for name in ("e9_completion_under_policy_delta", "e9_unsafe_imitation_delta"):
        deltas = cast(list[JsonValue], stage_b[name])
        for delta in deltas:
            if type(delta) is not dict or set(delta) != {
                "domain",
                "metric",
                "baseline",
                "comparison",
                "estimate",
                "ci_low",
                "ci_high",
            }:
                raise WorkbenchExportError(f"analysis.stage_b.{name} has invalid delta fields")
            delta_record = cast(dict[str, JsonValue], delta)
            for side in ("baseline", "comparison"):
                side_record = delta_record[side]
                if (
                    type(side_record) is not dict
                    or set(cast(dict[str, JsonValue], side_record)) != rate_fields
                ):
                    raise WorkbenchExportError(f"analysis.stage_b.{name} has invalid bound rate")
    macro_effects = cast(list[JsonValue], stage_b["e8_authority_aware_gain"])
    for effect in macro_effects:
        if type(effect) is not dict or set(effect) != {
            "domain",
            "baseline",
            "comparison",
            "baseline_estimate",
            "comparison_estimate",
            "estimate",
            "ci_low",
            "ci_high",
        }:
            raise WorkbenchExportError(
                "analysis.stage_b.e8_authority_aware_gain has invalid fields"
            )
        if type(effect["baseline"]) is not list or type(effect["comparison"]) is not list:
            raise WorkbenchExportError("analysis.stage_b.e8_authority_aware_gain has invalid rates")
        bound_rates = (*effect["baseline"], *effect["comparison"])
        if any(
            type(rate) is not dict or set(cast(dict[str, JsonValue], rate)) != rate_fields
            for rate in bound_rates
        ):
            raise WorkbenchExportError(
                "analysis.stage_b.e8_authority_aware_gain has invalid bound rate"
            )
    marginal_effects = cast(list[JsonValue], analysis["average_marginal_effects"])
    for effect in marginal_effects:
        if type(effect) is not dict or set(effect) != {
            "effect",
            "scope",
            "condition",
            "reference_condition",
            "contamination_ratio",
            "estimate",
            "ci_low",
            "ci_high",
        }:
            raise WorkbenchExportError(
                "analysis.average_marginal_effects fields do not match the static contract"
            )
        if (
            type(effect["effect"]) is not str
            or type(effect["scope"]) is not str
            or type(effect["condition"]) is not str
            or (
                effect["reference_condition"] is not None
                and type(effect["reference_condition"]) is not str
            )
            or type(effect["contamination_ratio"]) is not float
            or any(
                type(effect[field]) not in {int, float}
                for field in ("estimate", "ci_low", "ci_high")
            )
            or cast(float, effect["ci_low"]) > cast(float, effect["ci_high"])
        ):
            raise WorkbenchExportError("analysis.average_marginal_effects contains invalid values")
    episodes = dataset["episode_index"]
    if type(episodes) is not list:
        raise WorkbenchExportError("episode_index must be a list of dictionaries")
    ids: list[object] = []
    for item in episodes:
        if type(item) is not dict:
            raise WorkbenchExportError("episode_index must be a list of dictionaries")
        ids.append(item.get("episode_id"))
    if any(type(item) is not str for item in ids) or len(ids) != len(set(ids)):
        raise WorkbenchExportError("episode_index contains duplicate or invalid episode IDs")
    traces = dataset["episode_traces"]
    if type(traces) is not list:
        raise WorkbenchExportError("episode_traces must be a list")
    trace_ids: list[object] = []
    for trace in traces:
        if type(trace) is not dict or set(trace) != {
            "episode_id",
            "trace_artifact_sha256",
            "events",
        }:
            raise WorkbenchExportError(
                "episode trace fields do not match the public trace contract"
            )
        trace_id = trace["episode_id"]
        if type(trace_id) is not str or trace_id not in ids:
            raise WorkbenchExportError("episode trace is not bound to episode_index")
        _sha256(trace["trace_artifact_sha256"], "trace_artifact_sha256")
        if type(trace["events"]) is not list:
            raise WorkbenchExportError("episode trace events must be a list")
        events = tuple(_validate_public_event_projection(event) for event in trace["events"])
        PublicEpisodeTrace(
            episode_id=trace_id,
            trace_artifact_sha256=cast(str, trace["trace_artifact_sha256"]),
            events=events,
        )
        trace_ids.append(trace_id)
    if len(trace_ids) != len(set(trace_ids)):
        raise WorkbenchExportError("episode_traces contains duplicate episode IDs")
    claims = dataset["claims"]
    if type(claims) is not dict or type(claims.get("renderable_finding_ids")) is not list:
        raise WorkbenchExportError("claims must contain renderable_finding_ids")
    if claim_ceiling["status"] == "hold" and claims["renderable_finding_ids"]:
        raise WorkbenchExportError("HOLD dataset cannot label findings confirmatory")


def render_workbench_json(
    evidence: ReportEvidence,
    *,
    public_episode_traces: tuple[PublicEpisodeTrace, ...] = (),
) -> bytes:
    """Render canonical bytes for ``artifacts/reports/workbench.json``."""
    return build_workbench_export(
        evidence, public_episode_traces=public_episode_traces
    ).json_bytes()


def write_workbench_export(
    evidence: ReportEvidence,
    destination: Path,
    *,
    public_episode_traces: tuple[PublicEpisodeTrace, ...] = (),
) -> WorkbenchExport:
    """Atomically create a new static export without overwriting an existing file."""
    if not isinstance(destination, Path):
        raise WorkbenchExportError("destination must be a Path")
    export = build_workbench_export(evidence, public_episode_traces=public_episode_traces)
    if destination.exists() or destination.is_symlink():
        raise WorkbenchExportError("refusing to overwrite an existing workbench export")
    parent = destination.parent
    if not parent.is_dir():
        raise WorkbenchExportError("destination parent directory does not exist")
    temporary = parent / f".{destination.name}.{export.content_hash.removeprefix('sha256:')}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise WorkbenchExportError("temporary export path already exists") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(export.json_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise WorkbenchExportError(
                "refusing to overwrite an existing workbench export"
            ) from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return export


__all__ = [
    "WORKBENCH_SCHEMA",
    "WORKBENCH_SCHEMA_VERSION",
    "WorkbenchDataset",
    "WorkbenchExport",
    "WorkbenchExportError",
    "PublicEpisodeTrace",
    "PublicTraceEvent",
    "build_workbench_export",
    "render_workbench_json",
    "validate_workbench_export",
    "write_workbench_export",
]
