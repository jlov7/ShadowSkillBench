from __future__ import annotations

from typing import cast

import pytest

import shadowskillbench.domains.finance.adapter as finance_adapter_module
from shadowskillbench.domains.finance.adapter import FinanceAdapter
from shadowskillbench.domains.finance.fixtures import (
    FinanceFixtureVariant,
    build_finance_fixture,
)
from shadowskillbench.domains.finance.models import parse_finance_world
from shadowskillbench.domains.finance.reconciliation import reconcile_report
from shadowskillbench.engine.models import (
    ActionCall,
    EventCursor,
    JsonObject,
    OutcomeVerdict,
    WorldState,
    make_snapshot,
)
from shadowskillbench.engine.replay import replay
from shadowskillbench.engine.runtime import EnvironmentInvariantFailure, execute_action


def fixture(variant: FinanceFixtureVariant = FinanceFixtureVariant.CREATE_PENDING):
    return build_finance_fixture(variant, seed=12012)


def call(tool: str, arguments: dict[str, object], *, action_id: str = "action_1") -> ActionCall:
    return ActionCall(action_id=action_id, tool_name=tool, arguments=cast(JsonObject, arguments))


def assert_failure(proposal: object, code: str) -> None:
    assert getattr(proposal, "local_status") == "failure"
    assert getattr(proposal, "next_state") is None
    assert getattr(proposal, "error_code") == code


@pytest.mark.parametrize(
    ("tool", "entity_field", "fixture_variant"),
    [
        ("get_period", "period_id", FinanceFixtureVariant.CREATE_PENDING),
        ("get_ledger_snapshot", "source_snapshot_id", FinanceFixtureVariant.CREATE_PENDING),
        ("get_adjustment", "target_adjustment_id", FinanceFixtureVariant.SMALL_PERMITTED),
        ("get_approval", "approval_id", FinanceFixtureVariant.VALID_CFO_CAP),
        ("reconcile_report", "target_report_id", FinanceFixtureVariant.SMALL_PERMITTED),
    ],
)
def test_read_tools_are_detached_and_exact(
    tool: str, entity_field: str, fixture_variant: FinanceFixtureVariant
) -> None:
    source = fixture(fixture_variant)
    world = parse_finance_world(source.initial_state)
    if entity_field == "period_id":
        value = world.monthly_reports[0].period_id
    elif entity_field == "approval_id":
        value = world.approvals[0].approval_id
    else:
        value = getattr(source.case, entity_field)
    argument = {
        "get_period": "period_id",
        "get_ledger_snapshot": "snapshot_id",
        "get_adjustment": "adjustment_id",
        "get_approval": "approval_id",
        "reconcile_report": "report_id",
    }[tool]
    adapter = FinanceAdapter()
    proposal = adapter.apply(source.initial_state, call(tool, {argument: value}))
    assert proposal.local_status == "success"
    assert proposal.next_state is not source.initial_state
    assert proposal.next_state == source.initial_state
    proposal.observation["tool"] = "tampered"
    assert (
        adapter.apply(source.initial_state, call(tool, {argument: value})).observation["tool"]
        == tool
    )
    assert_failure(
        adapter.apply(source.initial_state, call(tool, {argument: value, "x": 1})),
        "ARGUMENTS_INVALID",
    )


def test_create_pending_materializes_only_latent_fact_and_recomputes_metric() -> None:
    source = fixture()
    world = parse_finance_world(source.initial_state)
    report = world.monthly_reports[0]
    fact = world.control_facts[0]
    args = {
        "report_id": report.report_id,
        "control_fact_id": fact.control_fact_id,
        "reported_category": "reported-normalization",
        "signed_amount_minor": fact.signed_amount_minor,
        "currency": fact.currency,
        "unit": fact.unit,
        "currency_exponent": fact.currency_exponent,
        "period_id": fact.period_id,
        "portco_id": fact.portco_id,
        "rationale": "deterministic materialization",
    }
    adapter = FinanceAdapter()
    proposal = adapter.apply(source.initial_state, call("create_adjustment", args))
    assert proposal.local_status == "success"
    assert proposal.next_state is not None
    next_world = parse_finance_world(proposal.next_state)
    assert next_world.control_facts == world.control_facts
    assert next_world.monthly_reports[0].adjustment_ids == (fact.adjustment_id,)
    assert next_world.adjustments[0].signed_amount_minor == fact.signed_amount_minor
    assert next_world.monthly_reports[0].metrics[0].signed_amount_minor == 107_500_000
    assert (
        adapter.apply(proposal.next_state, call("create_adjustment", args)).local_status
        == "success"
    )
    wrong = {**args, "signed_amount_minor": fact.signed_amount_minor + 1}
    assert_failure(
        adapter.apply(proposal.next_state, call("create_adjustment", wrong)), "CREATE_CONFLICT"
    )


def test_create_rejects_new_adjustment_into_published_report_atomically() -> None:
    source = fixture()
    world = parse_finance_world(source.initial_state)
    report = world.monthly_reports[0]
    fact = world.control_facts[0]
    adapter = FinanceAdapter()
    published = adapter.apply(
        source.initial_state,
        call("publish_report", {"report_id": report.report_id}),
    )
    assert published.local_status == "success"
    assert published.next_state is not None
    args = {
        "report_id": report.report_id,
        "control_fact_id": fact.control_fact_id,
        "reported_category": "reported-normalization",
        "signed_amount_minor": fact.signed_amount_minor,
        "currency": fact.currency,
        "unit": fact.unit,
        "currency_exponent": fact.currency_exponent,
        "period_id": fact.period_id,
        "portco_id": fact.portco_id,
        "rationale": "published report must remain unchanged",
    }
    proposal = adapter.apply(published.next_state, call("create_adjustment", args))
    assert_failure(proposal, "ENTITY_STATE_INVALID")
    after = parse_finance_world(published.next_state)
    assert after.monthly_reports[0].status == "published"
    assert after.monthly_reports[0].adjustment_ids == ()
    assert after.adjustments == ()

    materialized = adapter.apply(source.initial_state, call("create_adjustment", args))
    assert materialized.local_status == "success"
    assert materialized.next_state is not None
    published_adjusted = adapter.apply(
        materialized.next_state,
        call("publish_report", {"report_id": report.report_id}),
    )
    assert published_adjusted.local_status == "success"
    assert published_adjusted.next_state is not None
    repeated = adapter.apply(published_adjusted.next_state, call("create_adjustment", args))
    assert repeated.local_status == "success"
    assert repeated.next_state == published_adjusted.next_state
    repeated_world = parse_finance_world(repeated.next_state)
    assert repeated_world.monthly_reports[0].status == "published"
    assert repeated_world.adjustments[0].status == "published"


@pytest.mark.parametrize("bad", [True, 7.5, "7500000"])
def test_create_rejects_nonexact_money_and_money_representation_mismatch(bad: object) -> None:
    source = fixture()
    world = parse_finance_world(source.initial_state)
    fact = world.control_facts[0]
    args: dict[str, object] = {
        "report_id": world.monthly_reports[0].report_id,
        "control_fact_id": fact.control_fact_id,
        "reported_category": "reported-normalization",
        "signed_amount_minor": bad,
        "currency": fact.currency,
        "unit": fact.unit,
        "currency_exponent": fact.currency_exponent,
        "period_id": fact.period_id,
        "portco_id": fact.portco_id,
        "rationale": "exact money required",
    }
    assert_failure(
        FinanceAdapter().apply(source.initial_state, call("create_adjustment", args)),
        "CURRENCY_UNIT_MISMATCH",
    )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("currency", "EUR"),
        ("unit", "whole_units"),
        ("currency_exponent", 0),
    ],
)
def test_materialized_create_keeps_money_mismatch_taxonomy(field: str, wrong_value: object) -> None:
    source = fixture()
    world = parse_finance_world(source.initial_state)
    report = world.monthly_reports[0]
    fact = world.control_facts[0]
    args: dict[str, object] = {
        "report_id": report.report_id,
        "control_fact_id": fact.control_fact_id,
        "reported_category": "reported-normalization",
        "signed_amount_minor": fact.signed_amount_minor,
        "currency": fact.currency,
        "unit": fact.unit,
        "currency_exponent": fact.currency_exponent,
        "period_id": fact.period_id,
        "portco_id": fact.portco_id,
        "rationale": "materialized money taxonomy",
    }
    materialized = FinanceAdapter().apply(source.initial_state, call("create_adjustment", args))
    assert materialized.next_state is not None
    mismatch = {**args, field: wrong_value}
    assert_failure(
        FinanceAdapter().apply(materialized.next_state, call("create_adjustment", mismatch)),
        "CURRENCY_UNIT_MISMATCH",
    )


def test_attach_and_classify_are_local_evidence_and_presentation_only() -> None:
    source = fixture(FinanceFixtureVariant.PRIOR_PERIOD_APPROVAL)
    world = parse_finance_world(source.initial_state)
    adjustment = world.adjustments[0]
    approval = world.approvals[0]
    before_reconciliation = reconcile_report(world, source.case.target_report_id)
    adapter = FinanceAdapter()
    attached = adapter.apply(
        source.initial_state,
        call(
            "attach_approval",
            {"adjustment_id": adjustment.adjustment_id, "approval_id": approval.approval_id},
        ),
    )
    assert attached.local_status == "success"
    assert attached.next_state is not None
    classified = adapter.apply(
        attached.next_state,
        call(
            "classify_adjustment",
            {"adjustment_id": adjustment.adjustment_id, "reported_category": "ordinary"},
        ),
    )
    assert classified.next_state is not None
    after = parse_finance_world(classified.next_state)
    assert after.control_facts == world.control_facts
    assert after.adjustments[0].signed_amount_minor == adjustment.signed_amount_minor
    assert after.adjustments[0].reported_category == "ordinary"
    assert reconcile_report(after, source.case.target_report_id) == before_reconciliation


def test_numerically_correct_but_unauthorized_publish_locally_succeeds() -> None:
    source = fixture(FinanceFixtureVariant.RESTRICTED_NO_APPROVAL)
    report_id = source.case.target_report_id
    proposal = FinanceAdapter().apply(
        source.initial_state, call("publish_report", {"report_id": report_id})
    )
    assert proposal.local_status == "success"
    assert proposal.next_state is not None
    assert parse_finance_world(proposal.next_state).monthly_reports[0].status == "published"


def test_finished_gating_and_summary_idempotence() -> None:
    source = fixture()
    adapter = FinanceAdapter()
    finished = adapter.apply(source.initial_state, call("finish_task", {"summary": "done"}))
    assert finished.local_status == "success"
    assert finished.next_state is not None
    assert (
        adapter.apply(finished.next_state, call("finish_task", {"summary": "done"})).local_status
        == "success"
    )
    assert_failure(
        adapter.apply(finished.next_state, call("finish_task", {"summary": "other"})),
        "FINISH_CONFLICT",
    )
    assert_failure(
        adapter.apply(
            finished.next_state, call("publish_report", {"report_id": source.case.target_report_id})
        ),
        "TASK_FINISHED",
    )


def test_unknown_tool_wrong_ids_and_exact_arguments_fail_without_state() -> None:
    source = fixture()
    adapter = FinanceAdapter()
    assert_failure(adapter.apply(source.initial_state, call("unknown", {})), "TOOL_UNKNOWN")
    assert_failure(
        adapter.apply(source.initial_state, call("get_period", {"period_id": "missing"})),
        "ENTITY_NOT_FOUND",
    )
    assert_failure(
        adapter.apply(source.initial_state, call("get_period", {"period_id": 1})),
        "ARGUMENTS_INVALID",
    )


def test_constructed_and_custom_container_inputs_are_revalidated_without_aliasing() -> None:
    source = fixture()

    class CustomArguments(dict[str, object]):
        pass

    malformed_call = ActionCall.model_construct(
        action_id="action_1",
        tool_name="get_period",
        arguments=CustomArguments(period_id="bad"),
    )
    proposal = FinanceAdapter().apply(source.initial_state, malformed_call)
    assert_failure(proposal, "ACTION_INVALID")
    assert source.initial_state.data["finished"] is False


def test_publish_rejects_local_numeric_mismatch_without_mutating_input() -> None:
    source = fixture(FinanceFixtureVariant.SMALL_PERMITTED)
    reports = cast(list[JsonObject], source.initial_state.data["monthly_reports"])
    metrics = cast(list[JsonObject], reports[0]["metrics"])
    metrics[0]["signed_amount_minor"] = 0
    proposal = FinanceAdapter().apply(
        source.initial_state,
        call("publish_report", {"report_id": source.case.target_report_id}),
    )
    assert_failure(proposal, "RECONCILIATION_FAILED")
    assert metrics[0]["signed_amount_minor"] == 0


def test_wrong_report_source_cannot_reconcile_or_publish() -> None:
    source = fixture(FinanceFixtureVariant.SMALL_PERMITTED)
    reports = cast(list[JsonObject], source.initial_state.data["monthly_reports"])
    reports[0]["source_snapshot_id"] = "snapshot_missing"
    proposal = FinanceAdapter().apply(
        source.initial_state,
        call("publish_report", {"report_id": source.case.target_report_id}),
    )
    assert_failure(proposal, "STATE_INVALID")


def test_wrong_entity_waiver_can_attach_as_local_evidence_only() -> None:
    source = fixture(FinanceFixtureVariant.WRONG_ENTITY_WAIVER)
    world = parse_finance_world(source.initial_state)
    proposal = FinanceAdapter().apply(
        source.initial_state,
        call(
            "attach_approval",
            {
                "adjustment_id": world.adjustments[0].adjustment_id,
                "approval_id": world.approvals[0].approval_id,
            },
        ),
    )
    assert proposal.local_status == "success"
    assert (
        parse_finance_world(proposal.next_state).adjustments[0].exception_ref
        == world.approvals[0].approval_id
    )


@pytest.mark.mutation
def test_immutable_fact_drift_is_an_engine_adapter_invariant_not_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = fixture()

    def drifted_state(_source: WorldState, _world: object) -> WorldState:
        return WorldState(
            schema_version=source.initial_state.schema_version,
            world_id=source.initial_state.world_id,
            domain=source.initial_state.domain,
            seed=source.initial_state.seed,
            data={**source.initial_state.data, "control_facts": []},
        )

    monkeypatch.setattr(finance_adapter_module, "_detached_state", drifted_state)
    with pytest.raises(EnvironmentInvariantFailure, match="ADAPTER_EXCEPTION"):
        execute_action(
            source.initial_state,
            call(
                "get_period",
                {
                    "period_id": parse_finance_world(source.initial_state)
                    .monthly_reports[0]
                    .period_id
                },
            ),
            FinanceAdapter(),
            cursor=EventCursor(episode_id="episode_drift", next_index=0),
        )


def test_execute_and_replay_are_deterministic_integration_only() -> None:
    source = fixture(FinanceFixtureVariant.SMALL_PERMITTED)
    adapter = FinanceAdapter()
    action = call("publish_report", {"report_id": source.case.target_report_id})
    step = execute_action(
        source.initial_state,
        action,
        adapter,
        cursor=EventCursor(episode_id="episode_1", next_index=0),
    )
    assert step.result.local_status == "success"
    snapshot = make_snapshot(
        snapshot_id="snapshot_1", episode_id="episode_1", state=source.initial_state, observation={}
    )

    def verifier(state: WorldState) -> OutcomeVerdict:
        assert state.domain == "financial_adjustments"
        return OutcomeVerdict(status="PASS", reason_code="LOCAL_ONLY", details={})

    first = replay(snapshot, (action,), adapter, verifier=verifier)
    second = replay(snapshot, (action,), adapter, verifier=verifier)
    assert first == second
