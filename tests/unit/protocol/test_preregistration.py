from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import pytest

from shadowskillbench.protocol.validate import (
    validate_preregistration,
    validate_preregistration_text,
)

ROOT = Path(__file__).parents[3]
TEMPLATE = ROOT / "protocol" / "PREREGISTRATION_TEMPLATE.md"
HASH = "sha256:" + "0" * 64
CELL = "0.500 [0.250, 0.750]"

CONDITIONS = (
    ("A0 Bare", "A0_BARE"),
    ("A1 PolicyOnlySystem", "A1_POLICY_ONLY_SYSTEM"),
    ("A2 SkillOnly", "A2_SKILL_ONLY"),
    ("A3 SkillPolicySameTier", "A3_SKILL_POLICY_SAME_TIER"),
    ("A4 SkillPolicySystemTier", "A4_SKILL_POLICY_SYSTEM_TIER"),
    ("A5 SkillBuriedPolicySameTier", "A5_SKILL_BURIED_POLICY_SAME_TIER"),
    ("B0 SkillOnly", "B0_SKILL_ONLY"),
    ("B1 FlatPolicySystem", "B1_FLAT_POLICY_SYSTEM"),
    ("B2 AuthorityResolver", "B2_AUTHORITY_RESOLVER"),
    ("B3 DeterministicGate", "B3_DETERMINISTIC_GATE"),
)


def completed_text() -> str:
    lines = [
        "# ShadowSkillBench v1.0 — Preregistration Template",
        "",
        "## Protocol identity",
        "",
        "- protocol_version: 1.0",
        "",
        "## Models",
        "",
        "### Skill compiler",
        "",
        "- provider: fixture",
        "- model: fixture",
        "- model_version/date: fixture",
        f"- system_prompt_hash: {HASH}",
        f"- structured_output_schema_hash: {HASH}",
        "- temperature: 0.0",
        "- seed: 0",
        "- max_tokens: 128",
        "",
        "### Skill executor",
        "",
        "- provider: fixture",
        "- model: fixture",
        "- model_version/date: fixture",
        f"- runtime_prompt_hashes: {HASH}",
        "- temperature: 0.0",
        "- seed policy: fixture",
        "- max_turns: 1",
        "- max_tool_calls: 1",
        "- max_tokens: 128",
        "",
        "## Domains",
        "",
        f"- access_provisioning world hash: {HASH}",
        f"- financial_adjustments world hash: {HASH}",
        "",
        "## Demonstration bundles",
        "",
        "- primary trace count: 12",
        "- ratios: [0.0, 0.25, 0.5, 0.75, 1.0]",
        "- bundle seeds per domain/ratio: 3",
        f"- generator hash: {HASH}",
        f"- trace schema hash: {HASH}",
        "- narration: disabled in primary Stage A",
        "",
        "## Conditions",
        "",
    ]
    lines.extend(f"- {label}: {HASH}" for label, _ in CONDITIONS)
    lines.extend(
        [
            "",
            "Confirm:",
            "",
            f"- A3/A4 policy text is byte-identical: {HASH}",
            f"- A3 block order is counterbalanced: {HASH}",
            f"- A3/A4 total context accounting is logged: {HASH}",
            f"- A5 buried-handbook artifact, target position, and token count are frozen: {HASH}",
            "",
            "## Primary outcomes",
            "",
            "- Task Completion",
            "- Completion Under Policy",
            "",
            "## Primary Stage A estimands",
            "",
            "- E1 Behavioral contamination slope",
            "- E2 Same-tier policy attenuation",
            "- E3 Instruction hierarchy",
            "- E4 Skill interference under policy",
            "- E5 Buried-handbook contextual penalty",
            "",
            "## Primary Stage B estimands",
            "",
            "- E6 Unsafe imitation",
            "- E7 Flat-policy false enforcement",
            "- E8 Authority-aware gain",
            "- E9 Deterministic enforcement gap",
            "",
            "## Figure 1 predictions",
            "",
            "Enter point prediction and 80% subjective interval for each condition and ratio.",
            "",
            "| Condition | R0 | R25 | R50 | R75 | R100 |",
            "|---|---|---|---|---|---|",
        ]
    )
    lines.extend(
        f"| {row} | {CELL} | {CELL} | {CELL} | {CELL} | {CELL} |"
        for row in "A1 A2 A3 A4 A5".split()
    )
    lines.extend(
        [
            "",
            "- predicted probability of null A4–A3 hierarchy effect: 0.500",
            "- predicted probability that B1 over-enforces at least one authority class: 0.500",
            "- predicted probability that B2 adds no value over B1: 0.500",
            "",
            "## Confirmatory corpus",
            "",
            "### Stage A",
            "",
            "- skill bundles: 30",
            "- held-out cases per domain: 20",
            "- repeats: 3",
            "- planned skill-dependent episodes: 7,200",
            "- planned control episodes: 240",
            "- total: 7,440",
            "",
            "### Stage B",
            "",
            "- R75 skills per domain: 3",
            "- held-out cases per authority class/domain: 5",
            "- authority classes: 5",
            "- conditions: 4",
            "- repeats: 3",
            "- total: 1,800",
            "",
            "## Statistical plan",
            "",
            "- primary model: fixture",
            "- clustering: fixture",
            "- bootstrap: fixture",
            "- confidence interval: fixture",
            "- marginal effects: fixture",
            "- per-domain reporting: fixture",
            "- multiple-comparison family: fixture",
            "- material-effect threshold: fixture",
            "",
            "## Exclusion rules",
            "",
            "Only technical failures that occur before agent behavior begins may qualify.",
            "",
            "- provider unavailable before first response: fixture",
            "- malformed provider payload after retry: fixture",
            "- environment hash mismatch: fixture",
            "- runner crash before first action: fixture",
            "",
            "The following are outcomes, not exclusions:",
            "",
            "- refusal;",
            "- invalid tool call;",
            "- agent timeout;",
            "- budget exhaustion;",
            "- unsafe imitation;",
            "- false enforcement;",
            "- unnecessary escalation;",
            "- wrong completion claim.",
            "",
            "## Representative-case selection",
            "",
            "### Stage A",
            "",
            "1. cases where A2 fails CuP and A4 passes in majority of repeats;",
            "2. median contamination ratio;",
            "3. median action-trace length;",
            "4. lexicographically smallest case ID.",
            "",
            "### Stage B",
            "",
            "1. cases where B1 and B2 differ;",
            "2. median authority-graph depth;",
            "3. lexicographic case-ID tie-break.",
            "",
            "## Freeze statement",
            "",
            "After external anchoring:",
            "",
            "- no prompt changes;",
            "- no model changes;",
            "- no corpus/generator changes;",
            "- no policy text changes;",
            "- no condition changes;",
            "- no metric changes;",
            "- no exclusion changes;",
            "- no seed replacement after outcomes.",
            "",
            "Any amendment creates a new protocol version. Amended results are exploratory",
            "until separately confirmed.",
        ]
    )
    return "\n".join(lines) + "\n"


def codes(report: object) -> set[str]:
    return {violation.code for violation in report.violations}  # type: ignore[attr-defined]


def locations(report: object) -> set[str]:
    return {violation.location for violation in report.violations}  # type: ignore[attr-defined]


def test_current_template_is_expected_invalid_with_actionable_sentinels() -> None:
    report = validate_preregistration(TEMPLATE)

    assert not report.valid
    assert report.core is None
    assert "REQUIRED_BEFORE_FREEZE" in codes(report)
    assert "BLANK_REQUIRED_FIELD" not in codes(report)
    assert "MARKDOWN_PARSE_ERROR" in codes(report)
    assert sum(violation.code == "PREDICTION_REQUIRED" for violation in report.violations) == 28
    assert {
        "protocol.identity.protocol_version",
        "conditions.proofs.a3_a4_policy_text_byte_identical",
        "predictions.figure_1.A1.R0",
    } <= locations(report)


def test_minimal_completed_derivative_is_valid_without_anchor() -> None:
    report = validate_preregistration_text(completed_text())

    assert report.valid
    assert report.core is not None
    assert report.core.protocol_version == "1.0"
    assert tuple(condition.canonical_id for condition in report.core.conditions) == tuple(
        canonical for _, canonical in CONDITIONS
    )
    assert len(report.core.predictions.rows) == 5
    assert (
        sum(
            len((row.r0, row.r25, row.r50, row.r75, row.r100))
            for row in report.core.predictions.rows
        )
        == 25
    )
    assert report.core.predictions.null_a4_a3_hierarchy_effect == pytest.approx(0.5)
    assert report.core.corpus_counts.stage_a.total == 7440
    assert report.core.corpus_counts.stage_b.total == 1800
    assert report.core.models.skill_compiler.max_tokens == 128
    assert report.core.models.skill_executor.max_tool_calls == 1
    assert report.warnings == ()
    assert report.source_path is None


def test_template_instruction_blockquote_is_invalid_in_completed_derivative() -> None:
    instruction = "\n".join(
        (
            "> This file must be completed, validated, committed, signed, and externally",
            "> anchored before confirmatory execution. `REQUIRED_BEFORE_FREEZE` is a",
            "> machine-invalid sentinel.",
        )
    )
    text = completed_text().replace(
        "\n\n## Protocol identity", f"\n\n{instruction}\n\n## Protocol identity", 1
    )

    report = validate_preregistration_text(text)

    assert not report.valid
    assert "MARKDOWN_PARSE_ERROR" in codes(report)
    assert "document.instruction" in locations(report)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("full_commit_sha", "REQUIRED_BEFORE_FREEZE", "REQUIRED_BEFORE_FREEZE"),
        ("signed_tag", "v1.0.0", "IDENTITY_FIELD_IN_CORE"),
        ("freeze_manifest_sha256", HASH, "IDENTITY_FIELD_IN_CORE"),
        ("external_anchor_type", "fixture", "IDENTITY_FIELD_IN_CORE"),
        ("external_anchor_locator", "fixture", "IDENTITY_FIELD_IN_CORE"),
        ("anchor_timestamp_utc", "fixture", "IDENTITY_FIELD_IN_CORE"),
    ],
)
def test_legacy_identity_fields_are_rejected(field: str, value: str, expected: str) -> None:
    text = completed_text().replace(
        "- protocol_version: 1.0\n", f"- protocol_version: 1.0\n- {field}: {value}\n", 1
    )

    report = validate_preregistration_text(text)

    assert not report.valid
    assert expected in codes(report)
    assert f"protocol.identity.{field}" in locations(report)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("- provider: fixture\n", "- provider:\n", "BLANK_REQUIRED_FIELD"),
        ("- model: fixture\n", "", "BLANK_REQUIRED_FIELD"),
        ("- model: fixture\n", "- unknown model: fixture\n", "UNKNOWN_FIELD"),
        ("- model: fixture\n", "- provider: fixture\n", "DUPLICATE_FIELD"),
        ("## Models\n", "## Domains\n", "MARKDOWN_PARSE_ERROR"),
    ],
)
def test_missing_blank_duplicate_unknown_and_out_of_order_content_fail_closed(
    before: str, after: str, expected: str
) -> None:
    report = validate_preregistration_text(completed_text().replace(before, after, 1))

    assert not report.valid
    assert report.core is None
    assert expected in codes(report)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (HASH, "sha256:" + "A" * 64, "HASH_FORMAT"),
        (HASH, "sha256:1234", "HASH_FORMAT"),
        (f"- A0 Bare: {HASH}", "- A0 Bare: ", "CONDITION_HASH_MISSING"),
        (f"- A0 Bare: {HASH}", "- A0 Bare: sha256:" + "A" * 64, "CONDITION_HASH_FORMAT"),
        (
            f"- A3/A4 policy text is byte-identical: {HASH}",
            "- A3/A4 policy text is byte-identical: ",
            "PROOF_MISSING",
        ),
    ],
)
def test_hash_condition_and_proof_contracts_fail_closed(
    before: str, after: str, expected: str
) -> None:
    report = validate_preregistration_text(completed_text().replace(before, after, 1))

    assert not report.valid
    assert expected in codes(report)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("REQUIRED", "PREDICTION_REQUIRED"),
        ("0.500[0.250, 0.750]", "PREDICTION_SCHEMA_ERROR"),
        ("0.500 [0.750, 0.250]", "PREDICTION_RANGE"),
        ("NaN [0.250, 0.750]", "PREDICTION_RANGE"),
        ("5e-1 [0.250, 0.750]", "PREDICTION_RANGE"),
        ("0.500  [0.250, 0.750]", "PREDICTION_SCHEMA_ERROR"),
        ("1.1 [0.250, 0.750]", "PREDICTION_RANGE"),
        ("0.200 [0.250, 0.750]", "PREDICTION_RANGE"),
    ],
)
def test_prediction_cell_syntax_and_ranges_are_exact(cell: str, expected: str) -> None:
    report = validate_preregistration_text(completed_text().replace(CELL, cell, 1))

    assert not report.valid
    assert expected in codes(report)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("| A5 |", "| A6 |", "PREDICTION_SCHEMA_ERROR"),
        (
            "- predicted probability of null A4–A3 hierarchy effect: 0.500",
            "- predicted probability of null A4–A3 hierarchy effect: REQUIRED",
            "PREDICTION_REQUIRED",
        ),
        (
            "- predicted probability that B2 adds no value over B1: 0.500",
            "- predicted probability that B2 adds no value over B1: inf",
            "PREDICTION_RANGE",
        ),
    ],
)
def test_prediction_table_shape_and_probabilities_are_exact(
    before: str, after: str, expected: str
) -> None:
    report = validate_preregistration_text(completed_text().replace(before, after, 1))

    assert not report.valid
    assert expected in codes(report)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("| A1 |", "|  A1 |"),
        ("| A1 |", "| A1  |"),
        (f"| A1 | {CELL} |", f"| A1 |  {CELL} |"),
        (f"| A1 | {CELL} |", f"| A1 | {CELL}  |"),
    ],
)
def test_prediction_table_padding_is_canonical(before: str, after: str) -> None:
    report = validate_preregistration_text(completed_text().replace(before, after, 1))

    assert not report.valid
    assert "PREDICTION_SCHEMA_ERROR" in codes(report)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("- primary trace count: 12", "- primary trace count: 11"),
        ("- ratios: [0.0, 0.25, 0.5, 0.75, 1.0]", "- ratios: [0, 0.25, 0.5, 0.75, 1]"),
        ("- planned control episodes: 240", "- planned control episodes: 241"),
        ("- total: 1,800", "- total: 1,801"),
    ],
)
def test_demonstration_and_corpus_arithmetic_is_exact(before: str, after: str) -> None:
    report = validate_preregistration_text(completed_text().replace(before, after, 1))

    assert not report.valid
    assert "CORPUS_COUNT_MISMATCH" in codes(report)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (f"- A0 Bare: {HASH}", f"- A6 Bare: {HASH}", "CONDITION_SET_MISMATCH"),
        (f"- A1 PolicyOnlySystem: {HASH}\n", "", "CONDITION_SET_MISMATCH"),
        (
            f"- A1 PolicyOnlySystem: {HASH}\n",
            f"- A0 Bare: {HASH}\n",
            "CONDITION_SET_MISMATCH",
        ),
    ],
)
def test_condition_declarations_are_exact_and_ordered(
    before: str, after: str, expected: str
) -> None:
    report = validate_preregistration_text(completed_text().replace(before, after, 1))

    assert not report.valid
    assert expected in codes(report)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("- temperature: 0.0", "- temperature: NaN"),
        ("- temperature: 0.0", "- temperature: inf"),
        ("- max_turns: 1", "- max_turns: 0"),
        ("- max_tool_calls: 1", "- max_tool_calls: many"),
    ],
)
def test_model_numeric_fields_are_finite_and_budgets_are_positive(before: str, after: str) -> None:
    report = validate_preregistration_text(completed_text().replace(before, after, 1))

    assert not report.valid
    assert "SCHEMA_ERROR" in codes(report)


def test_fixed_literals_and_numeric_budget_are_not_reinterpreted() -> None:
    outcome = validate_preregistration_text(
        completed_text().replace("- Task Completion", "- Task Complete", 1)
    )
    budget = validate_preregistration_text(
        completed_text().replace("- max_tokens: 128", "- max_tokens: 0", 1)
    )

    assert "MARKDOWN_PARSE_ERROR" in codes(outcome)
    assert "SCHEMA_ERROR" in codes(budget)


@pytest.mark.parametrize("value", ["0128", "1,0,24", "1,024,", "1,,024"])
def test_integer_tokens_are_canonical(value: str) -> None:
    report = validate_preregistration_text(
        completed_text().replace("- max_tokens: 128", f"- max_tokens: {value}", 1)
    )

    assert not report.valid
    assert "SCHEMA_ERROR" in codes(report)


def test_canonical_thousands_grouping_is_accepted_for_positive_budget() -> None:
    report = validate_preregistration_text(
        completed_text().replace("- max_tokens: 128", "- max_tokens: 1,024", 1)
    )

    assert report.valid
    assert report.core is not None
    assert report.core.models.skill_compiler.max_tokens == 1024


def test_oversized_integer_is_a_structured_count_failure() -> None:
    oversized = "9" * 5000
    report = validate_preregistration_text(
        completed_text().replace(
            "- primary trace count: 12", f"- primary trace count: {oversized}", 1
        )
    )

    assert not report.valid
    assert "CORPUS_COUNT_MISMATCH" in codes(report)


def test_path_failures_and_repeated_reports_are_structured_and_deterministic(
    tmp_path: Path,
) -> None:
    missing = validate_preregistration(tmp_path / "missing.md")
    first = validate_preregistration_text(completed_text())
    second = validate_preregistration_text(completed_text())

    assert not missing.valid
    assert "LEDGER_IO_ERROR" in codes(missing)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_path_canonicalization_failures_are_typed_and_deterministic(tmp_path: Path) -> None:
    loop = tmp_path / "loop"
    loop.symlink_to(loop)

    loop_first = validate_preregistration(loop)
    loop_second = validate_preregistration(loop)
    nul_report = validate_preregistration(Path("\x00"))

    for report in (loop_first, loop_second, nul_report):
        assert not report.valid
        assert report.core is None
        assert "LEDGER_IO_ERROR" in codes(report)
    assert loop_first.model_dump(mode="json") == loop_second.model_dump(mode="json")


def test_parser_never_executes_processes_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("external action attempted")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    report = validate_preregistration_text(completed_text())

    assert report.valid


@pytest.mark.parametrize(
    "text", ["\ufeff" + completed_text(), completed_text().replace("\n", "\r\n")]
)
def test_utf8_lf_without_bom_is_required(text: str) -> None:
    report = validate_preregistration_text(text)

    assert not report.valid
    assert "MARKDOWN_PARSE_ERROR" in codes(report)
