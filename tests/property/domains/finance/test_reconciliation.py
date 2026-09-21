from __future__ import annotations

from typing import cast

import pytest

from shadowskillbench.domains.finance.fixtures import FinanceFixtureVariant, build_finance_fixture
from shadowskillbench.domains.finance.models import FinanceWorldState, parse_finance_world
from shadowskillbench.domains.finance.reconciliation import ReconciliationError, reconcile_report
from shadowskillbench.engine.models import JsonObject


@pytest.mark.property
@pytest.mark.parametrize("seed", [0, 1, 12012, 99991])
def test_reconciliation_is_deterministic_for_fixed_fixture(seed: int) -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=seed)
    world = parse_finance_world(fixture.initial_state)
    first = reconcile_report(world, fixture.case.target_report_id)
    second = reconcile_report(world, fixture.case.target_report_id)
    assert first == second
    assert first.reconciled is True
    assert first.base_total_minor == 100_000_000
    assert first.adjustment_total_minor == 1_000_000
    assert first.reported_total_minor == 101_000_000


@pytest.mark.mutation
def test_reconciliation_rejects_unsupported_metric_shape() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=7)
    raw = fixture.initial_state.data
    reports = cast(list[JsonObject], raw["monthly_reports"])
    metrics = cast(list[JsonObject], reports[0]["metrics"])
    extra = metrics[0].copy()
    extra["metric_id"] = "metric_z"
    metrics.append(extra)
    with pytest.raises(ValueError, match="METRIC_SHAPE_INVALID"):
        reconcile_report(parse_finance_world(fixture.initial_state), fixture.case.target_report_id)


@pytest.mark.mutation
def test_reconciliation_revalidates_constructed_and_model_copy_worlds() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=7)
    world = parse_finance_world(fixture.initial_state)
    malformed_copy = world.model_copy(update={"monthly_reports": ("not-a-report",)})
    malformed_constructed = FinanceWorldState.model_construct(
        domain_schema_version="1.0",
        case_id=world.case_id,
        reporting_periods=world.reporting_periods,
        ledger_snapshots=world.ledger_snapshots,
        adjustments=world.adjustments,
        monthly_reports=("not-a-report",),
        approvals=world.approvals,
        control_facts=world.control_facts,
        finished=False,
        finish_summary=None,
    )

    class FinanceWorldSubclass(FinanceWorldState):
        pass

    malformed_subclass = FinanceWorldSubclass.model_construct()
    for candidate in (malformed_copy, malformed_constructed, malformed_subclass):
        with pytest.raises(ReconciliationError, match="WORLD_INVALID"):
            reconcile_report(cast(FinanceWorldState, candidate), fixture.case.target_report_id)


@pytest.mark.mutation
def test_reconciliation_rejects_non_path_safe_report_ids() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=7)
    with pytest.raises(ReconciliationError, match="ARGUMENTS_INVALID"):
        reconcile_report(parse_finance_world(fixture.initial_state), "report invalid")
