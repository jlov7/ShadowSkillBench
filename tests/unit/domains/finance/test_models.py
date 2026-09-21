from __future__ import annotations

import re
import subprocess
import sys
from typing import ClassVar

import pytest
from pydantic import ValidationError

from shadowskillbench.domains.finance.fixtures import (
    FinanceFixtureVariant,
    build_finance_fixture,
    finance_fixtures,
)
from shadowskillbench.domains.finance.models import (
    Adjustment,
    ApprovalEvidence,
    FinanceControlFact,
    FinanceTaskCase,
    FinanceWorldState,
    LedgerBalance,
    ReportingPeriod,
    parse_finance_world,
    render_finance_world,
)
from shadowskillbench.engine.models import WorldState


def test_strict_money_dates_and_entity_models() -> None:
    period = ReportingPeriod(
        period_id="period-a",
        start_date="2026-01-01",
        end_date="2026-01-31",
        status="closed",
    )
    assert period.period_id == "period-a"
    with pytest.raises(ValidationError):
        LedgerBalance(
            balance_id="balance-a",
            signed_amount_minor=True,
            currency="USD",
            unit="minor_units",
            currency_exponent=2,
        )
    with pytest.raises(ValidationError):
        LedgerBalance(
            balance_id="balance-a",
            signed_amount_minor=1.0,  # type: ignore[arg-type]
            currency="USD",
            unit="minor_units",
            currency_exponent=2,
        )
    with pytest.raises(ValidationError):
        ReportingPeriod(
            period_id="period-a",
            start_date="2026-02-01",
            end_date="2026-01-31",
            status="closed",
        )
    with pytest.raises(ValidationError):
        ApprovalEvidence(
            approval_id="approval-a",
            evidence_kind="cfo_approval",
            approver_role="CFO",
            adjustment_id="adjustment-a",
            category_id="category-a",
            portco_id="portco-a",
            period_id="period-a",
            source_snapshot_id="snapshot-a",
            maximum_amount_minor=1,
            currency="USD",
            unit="minor_units",
            currency_exponent=2,
            effective_at="2026-01-01T00:00:00+00:00",
            expires_at=None,
            source_version="source-a",
            rule_reference_id="rule-a",
        )


def test_models_forbid_extra_fields_and_are_frozen() -> None:
    period = ReportingPeriod(
        period_id="period-a",
        start_date="2026-01-01",
        end_date="2026-01-31",
        status="closed",
    )
    with pytest.raises(ValidationError):
        ReportingPeriod.model_validate({**period.__dict__, "unexpected": "x"})
    with pytest.raises(ValidationError):
        period.status = "open"  # type: ignore[misc]


def test_fixture_projection_is_exact_sorted_referential_and_nonaliased() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.VALID_CFO_CAP, seed=17)
    parsed = parse_finance_world(fixture.initial_state)
    assert isinstance(parsed, FinanceWorldState)
    assert tuple(item.period_id for item in parsed.reporting_periods) == tuple(
        sorted(item.period_id for item in parsed.reporting_periods)
    )
    rebuilt = render_finance_world(parsed, world_id=fixture.case.world_id, seed=fixture.case.seed)
    assert rebuilt == fixture.initial_state
    assert rebuilt is not fixture.initial_state
    assert rebuilt.data is not fixture.initial_state.data
    fixture.initial_state.data["finished"] = True  # type: ignore[index]
    assert parsed.finished is False


def test_case_and_fixture_bind_exact_world_identity_and_no_variant_leakage() -> None:
    first = build_finance_fixture(FinanceFixtureVariant.RESTRICTED_NO_APPROVAL, seed=19)
    second = build_finance_fixture(FinanceFixtureVariant.RESTRICTED_NO_APPROVAL, seed=19)
    changed = build_finance_fixture(FinanceFixtureVariant.RESTRICTED_NO_APPROVAL, seed=20)
    different = build_finance_fixture(FinanceFixtureVariant.VALID_CFO_CAP, seed=19)
    assert first == second
    assert first.initial_state == second.initial_state
    assert first.fixture_id != changed.fixture_id
    assert first.fixture_id != different.fixture_id
    visible = repr(first.case) + repr(first.initial_state.data)
    assert "restricted_no_approval" not in visible
    assert "FinanceFixtureVariant" not in visible


def test_visible_ids_do_not_encode_fixture_variant_position_or_seed() -> None:
    variant = FinanceFixtureVariant.RESTRICTED_NO_APPROVAL
    seed = 19
    fixture = build_finance_fixture(variant, seed=seed)
    projection = parse_finance_world(fixture.initial_state)
    visible_ids = [
        fixture.fixture_id,
        fixture.case.case_id,
        fixture.case.world_id,
        fixture.case.target_report_id,
        fixture.case.target_adjustment_id,
        fixture.case.target_control_fact_id,
        fixture.case.source_snapshot_id,
        fixture.case.period_id,
        fixture.case.source_version,
        fixture.source_snapshot_id,
        *fixture.control_fact_ids,
        *fixture.authority_reference_ids,
        *(period.period_id for period in projection.reporting_periods),
        *(snapshot.snapshot_id for snapshot in projection.ledger_snapshots),
        *(snapshot.source_version for snapshot in projection.ledger_snapshots),
        *(adjustment.adjustment_id for adjustment in projection.adjustments),
        *(adjustment.period_id for adjustment in projection.adjustments),
        *(adjustment.portco_id for adjustment in projection.adjustments),
        *(report.report_id for report in projection.monthly_reports),
        *(fact.control_fact_id for fact in projection.control_facts),
        *(fact.source_snapshot_id for fact in projection.control_facts),
        *(fact.source_version for fact in projection.control_facts),
    ]
    for identifier in visible_ids:
        assert variant.value not in identifier
        assert re.fullmatch(r"[a-z]+_[0-9a-f]{16}", identifier)
        assert f"_{seed}_" not in identifier
        assert not identifier.endswith(f"_{seed}")


def test_projection_refuses_wrong_roots_order_duplicates_references_and_engine_identity() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=23)
    state = fixture.initial_state
    bad_data = dict(state.data)
    bad_data["unexpected"] = "x"
    with pytest.raises(ValidationError):
        parse_finance_world(
            WorldState(
                schema_version="1.0",
                world_id=state.world_id,
                domain="financial_adjustments",
                seed=state.seed,
                data=bad_data,
            )
        )
    bad_data = dict(state.data)
    bad_data["reporting_periods"] = list(reversed(bad_data["reporting_periods"]))
    with pytest.raises((ValidationError, ValueError)):
        parse_finance_world(
            WorldState(
                schema_version="1.0",
                world_id=state.world_id,
                domain="financial_adjustments",
                seed=state.seed,
                data=bad_data,
            )
        )
    with pytest.raises((ValidationError, ValueError)):
        parse_finance_world(
            WorldState(
                schema_version="1.0",
                world_id=state.world_id,
                domain="wrong-domain",
                seed=state.seed,
                data=state.data,
            )
        )


def test_projection_refuses_duplicate_and_dangling_entities_and_money_mismatches() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=27)
    state = fixture.initial_state
    duplicate_data = dict(state.data)
    duplicate_data["adjustments"] = [
        state.data["adjustments"][0],
        state.data["adjustments"][0],
    ]
    with pytest.raises((ValidationError, ValueError)):
        parse_finance_world(
            WorldState(
                schema_version="1.0",
                world_id=state.world_id,
                domain="financial_adjustments",
                seed=state.seed,
                data=duplicate_data,
            )
        )
    dangling_data = dict(state.data)
    dangling_report = dict(state.data["monthly_reports"][0])
    dangling_report["adjustment_ids"] = ["missing-adjustment"]
    dangling_data["monthly_reports"] = [dangling_report]
    with pytest.raises((ValidationError, ValueError)):
        parse_finance_world(
            WorldState(
                schema_version="1.0",
                world_id=state.world_id,
                domain="financial_adjustments",
                seed=state.seed,
                data=dangling_data,
            )
        )
    mismatch_data = dict(state.data)
    mismatch_snapshot = dict(state.data["ledger_snapshots"][0])
    mismatch_balance = dict(mismatch_snapshot["balances"][0])
    mismatch_balance["currency"] = "EUR"
    mismatch_snapshot["balances"] = [mismatch_balance]
    mismatch_data["ledger_snapshots"] = [mismatch_snapshot]
    with pytest.raises((ValidationError, ValueError)):
        parse_finance_world(
            WorldState(
                schema_version="1.0",
                world_id=state.world_id,
                domain="financial_adjustments",
                seed=state.seed,
                data=mismatch_data,
            )
        )


def test_orphan_adjustment_is_rejected_but_latent_control_fact_is_admitted() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=26)
    orphan_data = dict(fixture.initial_state.data)
    orphan = dict(orphan_data["adjustments"][0])
    orphan["adjustment_id"] = "orphan-adjustment"
    orphan_data["adjustments"] = [orphan_data["adjustments"][0], orphan]
    with pytest.raises((ValidationError, ValueError)):
        parse_finance_world(
            WorldState(
                schema_version="1.0",
                world_id=fixture.initial_state.world_id,
                domain="financial_adjustments",
                seed=fixture.initial_state.seed,
                data=orphan_data,
            )
        )

    pending = build_finance_fixture(FinanceFixtureVariant.CREATE_PENDING, seed=26)
    projection = parse_finance_world(pending.initial_state)
    assert projection.adjustments == ()
    assert projection.monthly_reports[0].adjustment_ids == ()
    assert pending.case.target_control_fact_id == projection.control_facts[0].control_fact_id
    assert pending.case.target_adjustment_id == projection.control_facts[0].adjustment_id


def test_projection_refuses_swapped_approval_and_waiver_attachment_kinds() -> None:
    approval_fixture = build_finance_fixture(FinanceFixtureVariant.VALID_CFO_CAP, seed=28)
    approval_data = dict(approval_fixture.initial_state.data)
    approval = dict(approval_data["approvals"][0])
    approval["evidence_kind"] = "scoped_waiver"
    approval_data["approvals"] = [approval]
    with pytest.raises((ValidationError, ValueError)):
        parse_finance_world(
            WorldState(
                schema_version="1.0",
                world_id=approval_fixture.initial_state.world_id,
                domain="financial_adjustments",
                seed=approval_fixture.initial_state.seed,
                data=approval_data,
            )
        )
    waiver_fixture = build_finance_fixture(FinanceFixtureVariant.WRONG_ENTITY_WAIVER, seed=28)
    waiver_data = dict(waiver_fixture.initial_state.data)
    waiver = dict(waiver_data["approvals"][0])
    waiver["evidence_kind"] = "cfo_approval"
    waiver_data["approvals"] = [waiver]
    with pytest.raises((ValidationError, ValueError)):
        parse_finance_world(
            WorldState(
                schema_version="1.0",
                world_id=waiver_fixture.initial_state.world_id,
                domain="financial_adjustments",
                seed=waiver_fixture.initial_state.seed,
                data=waiver_data,
            )
        )


def test_direct_model_validation_rejects_constructed_models_and_hostile_dicts() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=35)
    with pytest.raises((ValidationError, ValueError)):
        FinanceTaskCase.model_validate(fixture.case)
    with pytest.raises((ValidationError, ValueError)):
        FinanceTaskCase.model_validate(
            FinanceTaskCase.model_construct(
                case_id=fixture.case.case_id,
                world_id=fixture.case.world_id,
                domain="financial_adjustments",
                seed=fixture.case.seed,
                target_report_id=fixture.case.target_report_id,
                target_adjustment_id=fixture.case.target_adjustment_id,
                target_control_fact_id=fixture.case.target_control_fact_id,
                source_snapshot_id=fixture.case.source_snapshot_id,
                period_id=fixture.case.period_id,
            )
        )

    class HostileDict(dict[str, object]):
        calls = 0

        def items(self) -> object:
            type(self).calls += 1
            raise AssertionError("hostile mapping hook must not run")

    with pytest.raises((ValidationError, ValueError)):
        FinanceWorldState.model_validate(HostileDict())
    assert HostileDict.calls == 0


def test_fixture_builder_rejects_nonexact_variant_inputs_without_hooks() -> None:
    with pytest.raises(ValueError):
        build_finance_fixture("small_permitted", seed=47)  # type: ignore[arg-type]

    class HostileVariant(str):
        calls: ClassVar[int] = 0

        def __hash__(self) -> int:
            type(self).calls += 1
            raise AssertionError("variant hook must not run")

    with pytest.raises(ValueError):
        build_finance_fixture(HostileVariant("small_permitted"), seed=47)  # type: ignore[arg-type]
    assert HostileVariant.calls == 0


def test_control_facts_remain_separate_from_editable_presentation() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.RESTRICTED_NO_APPROVAL, seed=29)
    parsed = parse_finance_world(fixture.initial_state)
    adjustment = parsed.adjustments[0]
    control = parsed.control_facts[0]
    assert isinstance(adjustment, Adjustment)
    assert isinstance(control, FinanceControlFact)
    assert adjustment.reported_category != control.original_economic_category
    assert control.signed_amount_minor == 7_500_000
    with pytest.raises(ValidationError):
        FinanceControlFact.model_validate({**control.__dict__, "extra": True})


def test_threshold_values_are_exact_minor_unit_facts_not_verdicts() -> None:
    fixtures = finance_fixtures(seed=31)
    values = {
        fact.signed_amount_minor
        for fixture in fixtures
        for fact in parse_finance_world(fixture.initial_state).control_facts
    }
    assert {5_000_000, 7_500_000, 10_000_000}.issubset(values)


def test_hostile_construct_subclass_and_custom_container_are_rejected() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=37)

    class EvilWorldState(WorldState):
        calls: ClassVar[int] = 0

        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            type(self).calls += 1
            raise AssertionError("hostile hook must not run")

    with pytest.raises((ValidationError, ValueError)):
        parse_finance_world(
            EvilWorldState(
                schema_version="1.0",
                world_id=fixture.initial_state.world_id,
                domain="financial_adjustments",
                seed=fixture.initial_state.seed,
                data=fixture.initial_state.data,
            )
        )
    assert EvilWorldState.calls == 0
    constructed = WorldState.model_construct(
        schema_version="1.0",
        world_id=fixture.initial_state.world_id,
        domain="financial_adjustments",
        seed=fixture.initial_state.seed,
        data={"not": "a-world"},
    )
    with pytest.raises(ValidationError):
        parse_finance_world(constructed)

    class CustomList(list[object]):
        pass

    bad = WorldState.model_construct(
        schema_version="1.0",
        world_id=fixture.initial_state.world_id,
        domain="financial_adjustments",
        seed=fixture.initial_state.seed,
        data=CustomList(),
    )
    with pytest.raises(ValidationError):
        parse_finance_world(bad)


def test_fixture_revalidates_constructed_case_at_its_public_boundary() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=38)
    malformed_case = FinanceTaskCase.model_construct(
        case_id=fixture.case.case_id,
        world_id=fixture.case.world_id,
        domain="wrong-domain",
        seed=fixture.case.seed,
        target_report_id=fixture.case.target_report_id,
        target_adjustment_id=fixture.case.target_adjustment_id,
        target_control_fact_id=fixture.case.target_control_fact_id,
        source_snapshot_id=fixture.case.source_snapshot_id,
        period_id=fixture.case.period_id,
        source_version=fixture.case.source_version,
    )
    with pytest.raises((ValidationError, ValueError)):
        type(fixture)(
            fixture_id=fixture.fixture_id,
            seed=fixture.seed,
            case=malformed_case,
            initial_state=fixture.initial_state,
            source_snapshot_id=fixture.source_snapshot_id,
            control_fact_ids=list(fixture.control_fact_ids),
            authority_reference_ids=list(fixture.authority_reference_ids),
        )


def test_exact_root_and_boundary_stay_authority_free() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.SMALL_PERMITTED, seed=39)
    assert set(fixture.initial_state.data) == {
        "domain_schema_version",
        "case_id",
        "reporting_periods",
        "ledger_snapshots",
        "adjustments",
        "monthly_reports",
        "approvals",
        "control_facts",
        "finished",
        "finish_summary",
    }
    assert "at_time" not in FinanceTaskCase.model_fields
    assert "at_time" not in FinanceWorldState.model_fields
    assert set(FinanceTaskCase.model_fields) == {
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
    }
    assert set(FinanceControlFact.model_fields) == {
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
    }
    assert set(FinanceWorldState.model_fields) == {
        "domain_schema_version",
        "case_id",
        "reporting_periods",
        "ledger_snapshots",
        "adjustments",
        "monthly_reports",
        "approvals",
        "control_facts",
        "finished",
        "finish_summary",
    }
    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import shadowskillbench.domains.finance.models; "
                "raise SystemExit(any(name.startswith('shadowskillbench.authority') "
                "for name in sys.modules))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_check.returncode == 0, import_check.stderr


@pytest.mark.property
def test_projection_roundtrip_is_deterministic_for_every_fixture_variant() -> None:
    for fixture in finance_fixtures(seed=41):
        assert (
            render_finance_world(
                parse_finance_world(fixture.initial_state),
                world_id=fixture.case.world_id,
                seed=fixture.case.seed,
            )
            == fixture.initial_state
        )


@pytest.mark.mutation
def test_mutation_marker_control_category_cannot_be_presentation_category() -> None:
    fixture = build_finance_fixture(FinanceFixtureVariant.RESTRICTED_NO_APPROVAL, seed=43)
    projection = parse_finance_world(fixture.initial_state)
    assert (
        projection.control_facts[0].original_economic_category
        != projection.adjustments[0].reported_category
    )
