from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import ValidationError

from shadowskillbench.protocol.models import (
    ConditionDefinition,
    ConditionProofs,
    CorpusCounts,
    DemonstrationDeclaration,
    DomainWorlds,
    FigurePredictions,
    ModelsDeclaration,
    PredictionCell,
    PredictionRow,
    PreregistrationCore,
    PreregistrationReport,
    ProtocolViolation,
    SkillCompilerDeclaration,
    SkillExecutorDeclaration,
    StageACorpusCounts,
    StageBCorpusCounts,
    StatisticalPlan,
    TechnicalExclusions,
)

_SENTINEL = "REQUIRED_BEFORE_FREEZE"
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NUMBER = re.compile(r"(?:0|1|0\.[0-9]+|1\.0+)\Z")
_CELL = re.compile(r"([^ ]+) \[([^,]+), ([^\]]+)\]\Z")
_PLAIN_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_GROUPED_INTEGER = re.compile(r"[1-9][0-9]{0,2}(?:,[0-9]{3})+\Z")
_TITLE = "# ShadowSkillBench v1.0 — Preregistration Template"
_TEMPLATE_INSTRUCTION = (
    "> This file must be completed, validated, committed, signed, and externally",
    "> anchored before confirmatory execution. `REQUIRED_BEFORE_FREEZE` is a",
    "> machine-invalid sentinel.",
)
_LEGACY_IDENTITY_FIELDS = frozenset(
    {
        "full_commit_sha",
        "signed_tag",
        "freeze_manifest_sha256",
        "external_anchor_type",
        "external_anchor_locator",
        "anchor_timestamp_utc",
    }
)
_CONDITIONS = (
    ("A0 Bare", "A0", "A0_BARE"),
    ("A1 PolicyOnlySystem", "A1", "A1_POLICY_ONLY_SYSTEM"),
    ("A2 SkillOnly", "A2", "A2_SKILL_ONLY"),
    ("A3 SkillPolicySameTier", "A3", "A3_SKILL_POLICY_SAME_TIER"),
    ("A4 SkillPolicySystemTier", "A4", "A4_SKILL_POLICY_SYSTEM_TIER"),
    ("A5 SkillBuriedPolicySameTier", "A5", "A5_SKILL_BURIED_POLICY_SAME_TIER"),
    ("B0 SkillOnly", "B0", "B0_SKILL_ONLY"),
    ("B1 FlatPolicySystem", "B1", "B1_FLAT_POLICY_SYSTEM"),
    ("B2 AuthorityResolver", "B2", "B2_AUTHORITY_RESOLVER"),
    ("B3 DeterministicGate", "B3", "B3_DETERMINISTIC_GATE"),
)
_PROOF_FIELDS = (
    (
        "A3/A4 policy text is byte-identical",
        "a3_a4_policy_text_byte_identical",
    ),
    (
        "A3 block order is counterbalanced",
        "a3_block_order_counterbalanced",
    ),
    (
        "A3/A4 total context accounting is logged",
        "a3_a4_context_accounting_logged",
    ),
    (
        "A5 buried-handbook artifact, target position, and token count are frozen",
        "a5_buried_handbook_artifact_frozen",
    ),
)
_LEGACY_A5_PROOF_LABEL = "A5 handbooks have identical content/tokens and only section order differs"
_LEGACY_E5_ESTIMAND = "- E5 Policy salience"
_COMPILER_FIELDS = (
    ("provider", "provider"),
    ("model", "model"),
    ("model_version/date", "model_version_date"),
    ("system_prompt_hash", "system_prompt_hash"),
    ("structured_output_schema_hash", "structured_output_schema_hash"),
    ("temperature", "temperature"),
    ("seed", "seed"),
    ("max_tokens", "max_tokens"),
)
_EXECUTOR_FIELDS = (
    ("provider", "provider"),
    ("model", "model"),
    ("model_version/date", "model_version_date"),
    ("runtime_prompt_hashes", "runtime_prompt_hashes"),
    ("temperature", "temperature"),
    ("seed policy", "seed_policy"),
    ("max_turns", "max_turns"),
    ("max_tool_calls", "max_tool_calls"),
    ("max_tokens", "max_tokens"),
)
_STATISTICAL_FIELDS = (
    ("primary model", "primary_model"),
    ("clustering", "clustering"),
    ("bootstrap", "bootstrap"),
    ("confidence interval", "confidence_interval"),
    ("marginal effects", "marginal_effects"),
    ("per-domain reporting", "per_domain_reporting"),
    ("multiple-comparison family", "multiple_comparison_family"),
    ("material-effect threshold", "material_effect_threshold"),
)
_EXCLUSION_FIELDS = (
    ("provider unavailable before first response", "provider_unavailable_before_first_response"),
    ("malformed provider payload after retry", "malformed_provider_payload_after_retry"),
    ("environment hash mismatch", "environment_hash_mismatch"),
    ("runner crash before first action", "runner_crash_before_first_action"),
)
_KNOWN_LABELS = frozenset(
    {
        "protocol_version",
        *(_LEGACY_IDENTITY_FIELDS),
        *(label for label, _ in _COMPILER_FIELDS),
        *(label for label, _ in _EXECUTOR_FIELDS),
        "access_provisioning world hash",
        "financial_adjustments world hash",
        "primary trace count",
        "ratios",
        "bundle seeds per domain/ratio",
        "generator hash",
        "trace schema hash",
        "narration",
        *(label for label, _, _ in _CONDITIONS),
        *(label for label, _ in _PROOF_FIELDS),
        _LEGACY_A5_PROOF_LABEL,
        "predicted probability of null A4–A3 hierarchy effect",
        "predicted probability that B1 over-enforces at least one authority class",
        "predicted probability that B2 adds no value over B1",
        "skill bundles",
        "held-out cases per domain",
        "repeats",
        "planned skill-dependent episodes",
        "planned control episodes",
        "total",
        "R75 skills per domain",
        "held-out cases per authority class/domain",
        "authority classes",
        "conditions",
        *(label for label, _ in _STATISTICAL_FIELDS),
        *(label for label, _ in _EXCLUSION_FIELDS),
    }
)


@dataclass
class _Parser:
    lines: tuple[str, ...]
    index: int = 0
    violations: list[ProtocolViolation] = field(default_factory=list)
    seen_by_section: dict[str, set[str]] = field(default_factory=dict)

    def violation(self, code: str, location: str, detail: str) -> None:
        self.violations.append(ProtocolViolation(code=code, location=location, detail=detail))

    def current(self) -> str | None:
        if self.index >= len(self.lines):
            return None
        return self.lines[self.index]

    def expect_literal(
        self,
        literal: str,
        location: str,
        *,
        superseded_aliases: tuple[str, ...] = (),
    ) -> None:
        if self.current() == literal:
            self.index += 1
            return
        if self.current() in superseded_aliases:
            self.index += 1
            self.violation(
                "SUPERSEDED_PREREGISTRATION_CONTRACT",
                location,
                "legacy E5 wording is superseded by the buried-handbook contextual penalty",
            )
            return
        self.violation("MARKDOWN_PARSE_ERROR", location, "expected fixed literal")
        if self.current() is not None:
            self.index += 1

    def optional_template_instruction(self) -> None:
        if self.current() != _TEMPLATE_INSTRUCTION[0]:
            return
        if (
            self.lines[self.index : self.index + len(_TEMPLATE_INSTRUCTION)]
            == _TEMPLATE_INSTRUCTION
        ):
            self.index += len(_TEMPLATE_INSTRUCTION)
            self.violation(
                "MARKDOWN_PARSE_ERROR",
                "document.instruction",
                "template instruction blockquote is forbidden in a completed derivative",
            )
            return
        self.violation(
            "MARKDOWN_PARSE_ERROR", "document.instruction", "invalid instruction blockquote"
        )
        while (line := self.current()) is not None and line.startswith(">"):
            self.index += 1

    def scalar(
        self,
        label: str,
        location: str,
        section: str,
        *,
        blank_code: str = "BLANK_REQUIRED_FIELD",
        mismatch_code: str = "MARKDOWN_PARSE_ERROR",
        template_continuation: bool = False,
        superseded_aliases: tuple[str, ...] = (),
    ) -> str:
        line = self.current()
        labels = (label, *superseded_aliases)
        matched_label = next(
            (
                candidate
                for candidate in labels
                if line is not None and line.startswith(f"- {candidate}:")
            ),
            None,
        )
        if matched_label is not None:
            assert line is not None
            prefix = f"- {matched_label}:"
            tail = line[len(prefix) :]
            self.index += 1
            self.seen_by_section.setdefault(section, set()).add(label)
            if matched_label != label:
                self.violation(
                    "SUPERSEDED_PREREGISTRATION_CONTRACT",
                    location,
                    "legacy E5 wording is superseded by the buried-handbook contextual penalty",
                )
            if template_continuation and tail == "" and self.current() == f"  {_SENTINEL}":
                self.index += 1
                value = _SENTINEL
            elif tail == "":
                value = ""
            elif tail.startswith(" ") and not tail.startswith("  "):
                value = tail[1:]
            else:
                self.violation(
                    "MARKDOWN_PARSE_ERROR", location, "scalar must use colon-space syntax"
                )
                value = ""
            if value == "":
                self.violation(blank_code, location, "required value is blank")
            elif value == _SENTINEL:
                self.violation(
                    "REQUIRED_BEFORE_FREEZE", location, "required value is the freeze sentinel"
                )
            return value

        self.violation(blank_code, location, "required field is missing")
        actual_label = _label_from_line(line)
        if actual_label is None:
            return ""
        if mismatch_code == "CONDITION_SET_MISMATCH":
            self.violation(
                mismatch_code, location, "condition declaration is out of order or unknown"
            )
        if actual_label in self.seen_by_section.setdefault(section, set()):
            self.violation("DUPLICATE_FIELD", location, "field appears more than once")
        elif actual_label not in _KNOWN_LABELS:
            self.violation("UNKNOWN_FIELD", f"{section}.unknown", "field is not allowed")
        else:
            self.violation(mismatch_code, location, "field is out of order or unexpected")
        self.index += 1
        return ""


def _label_from_line(line: str | None) -> str | None:
    if line is None or not line.startswith("- "):
        return None
    body = line[2:]
    if ":" not in body:
        return None
    return body.split(":", 1)[0]


def _read_fields(
    parser: _Parser, fields: tuple[tuple[str, str], ...], prefix: str, section: str
) -> dict[str, str]:
    return {field: parser.scalar(label, f"{prefix}.{field}", section) for label, field in fields}


def _is_complete(value: str) -> bool:
    return bool(value) and value != _SENTINEL


def _hash(parser: _Parser, value: str, location: str, code: str = "HASH_FORMAT") -> None:
    if _is_complete(value) and not _HASH.fullmatch(value):
        parser.violation(code, location, "value must be a lowercase sha256 reference")


def _integer(
    parser: _Parser,
    value: str,
    location: str,
    *,
    positive: bool = False,
    mismatch_code: str = "SCHEMA_ERROR",
) -> int:
    if not _is_complete(value):
        return 0
    if _PLAIN_INTEGER.fullmatch(value):
        normalized = value
    elif _GROUPED_INTEGER.fullmatch(value):
        normalized = value.replace(",", "")
    else:
        parser.violation(mismatch_code, location, "value must be an integer")
        return 0
    try:
        number = int(normalized)
    except ValueError:
        parser.violation(mismatch_code, location, "value must be an integer")
        return 0
    if positive and number <= 0:
        parser.violation(mismatch_code, location, "value must be positive")
    return number


def _finite_decimal(parser: _Parser, value: str, location: str) -> Decimal:
    if not _is_complete(value):
        return Decimal("0")
    try:
        decimal = Decimal(value)
    except InvalidOperation:
        parser.violation("SCHEMA_ERROR", location, "value must be a finite decimal")
        return Decimal("0")
    if not decimal.is_finite():
        parser.violation("SCHEMA_ERROR", location, "value must be a finite decimal")
        return Decimal("0")
    return decimal


def _prediction_number(
    parser: _Parser, value: str, location: str, *, required: bool = False
) -> Decimal:
    if value == "":
        return Decimal("0")
    if value == "REQUIRED":
        parser.violation("PREDICTION_REQUIRED", location, "prediction is required")
        return Decimal("0")
    if not _NUMBER.fullmatch(value):
        parser.violation(
            "PREDICTION_RANGE", location, "prediction must be a finite decimal in range"
        )
        return Decimal("0")
    decimal = Decimal(value)
    if decimal < 0 or decimal > 1:
        parser.violation("PREDICTION_RANGE", location, "prediction must be in [0, 1]")
    return decimal


def _prediction_cell(parser: _Parser, value: str, location: str) -> PredictionCell:
    if value == "REQUIRED":
        parser.violation("PREDICTION_REQUIRED", location, "prediction is required")
        return PredictionCell(
            point=Decimal("0"), interval80_low=Decimal("0"), interval80_high=Decimal("0")
        )
    match = _CELL.fullmatch(value)
    if match is None:
        parser.violation("PREDICTION_SCHEMA_ERROR", location, "prediction cell syntax is invalid")
        return PredictionCell(
            point=Decimal("0"), interval80_low=Decimal("0"), interval80_high=Decimal("0")
        )
    point_raw, low_raw, high_raw = match.groups()
    point = _prediction_number(parser, point_raw, location)
    low = _prediction_number(parser, low_raw, location)
    high = _prediction_number(parser, high_raw, location)
    if all(_NUMBER.fullmatch(number) for number in (point_raw, low_raw, high_raw)) and not (
        low <= point <= high
    ):
        parser.violation("PREDICTION_RANGE", location, "interval must contain point")
    return PredictionCell(point=point, interval80_low=low, interval80_high=high)


def _prediction_table(parser: _Parser) -> tuple[PredictionRow, ...]:
    parser.expect_literal(
        "| Condition | R0 | R25 | R50 | R75 | R100 |", "predictions.figure_1.header"
    )
    parser.expect_literal("|---|---|---|---|---|---|", "predictions.figure_1.separator")
    rows: list[PredictionRow] = []
    for row_id in ("A1", "A2", "A3", "A4", "A5"):
        line = parser.current()
        if line is None:
            parser.violation(
                "PREDICTION_SCHEMA_ERROR", f"predictions.figure_1.{row_id}", "row is missing"
            )
            cells = (_prediction_cell(parser, "", f"predictions.figure_1.{row_id}.R0"),) * 5
        else:
            parser.index += 1
            parts = line.split("|")
            if len(parts) != 8 or parts[0] != "" or parts[-1] != "":
                parser.violation(
                    "PREDICTION_SCHEMA_ERROR",
                    f"predictions.figure_1.{row_id}",
                    "row has the wrong shape",
                )
                raw_cells = ("",) * 5
            else:
                if parts[1] != f" {row_id} ":
                    parser.violation(
                        "PREDICTION_SCHEMA_ERROR",
                        f"predictions.figure_1.{row_id}",
                        "row identifier is out of order",
                    )
                raw_cell_values: list[str] = []
                for padded_cell, ratio in zip(
                    parts[2:-1], ("R0", "R25", "R50", "R75", "R100"), strict=True
                ):
                    location = f"predictions.figure_1.{row_id}.{ratio}"
                    if (
                        len(padded_cell) < 2
                        or not padded_cell.startswith(" ")
                        or not padded_cell.endswith(" ")
                        or padded_cell.startswith("  ")
                        or padded_cell.endswith("  ")
                    ):
                        parser.violation(
                            "PREDICTION_SCHEMA_ERROR",
                            location,
                            "cell must use exactly one Markdown padding space",
                        )
                        raw_cell_values.append("")
                    else:
                        raw_cell_values.append(padded_cell[1:-1])
                raw_cells = tuple(raw_cell_values)
            cells = tuple(
                _prediction_cell(parser, raw, f"predictions.figure_1.{row_id}.{ratio}")
                for raw, ratio in zip(raw_cells, ("R0", "R25", "R50", "R75", "R100"), strict=True)
            )
        rows.append(
            PredictionRow(
                condition=row_id,
                r0=cells[0],
                r25=cells[1],
                r50=cells[2],
                r75=cells[3],
                r100=cells[4],
            )
        )
    return tuple(rows)


def _legacy_identity(parser: _Parser) -> None:
    while (label := _label_from_line(parser.current())) in _LEGACY_IDENTITY_FIELDS:
        assert label is not None
        line = parser.current()
        assert line is not None
        parser.index += 1
        prefix = f"- {label}:"
        tail = line[len(prefix) :] if line.startswith(prefix) else ""
        value = tail[1:] if tail.startswith(" ") else ""
        location = f"protocol.identity.{label}"
        if value == _SENTINEL:
            parser.violation(
                "REQUIRED_BEFORE_FREEZE", location, "legacy identity field is the freeze sentinel"
            )
        else:
            parser.violation(
                "IDENTITY_FIELD_IN_CORE", location, "post-freeze identity field is forbidden"
            )


def _fixed_literals(parser: _Parser, literals: tuple[str, ...], location: str) -> tuple[str, ...]:
    for literal in literals:
        parser.expect_literal(literal, location)
    return literals


def _report(
    source_path: Path | None, core: PreregistrationCore | None, violations: list[ProtocolViolation]
) -> PreregistrationReport:
    ordered = tuple(sorted(violations, key=lambda item: (item.location, item.code, item.detail)))
    return PreregistrationReport(
        source_path=source_path,
        valid=not ordered,
        core=core if not ordered else None,
        violations=ordered,
        warnings=(),
    )


def _validate_text(text: str, source_path: Path | None) -> PreregistrationReport:
    if text.startswith("\ufeff") or "\r" in text:
        return _report(
            source_path,
            None,
            [
                ProtocolViolation(
                    code="MARKDOWN_PARSE_ERROR",
                    location="document.encoding",
                    detail="input must be UTF-8 without BOM and use LF line endings",
                )
            ],
        )
    parser = _Parser(tuple(line for line in text.split("\n") if line != ""))
    parser.expect_literal(_TITLE, "document.title")
    parser.optional_template_instruction()
    parser.expect_literal("## Protocol identity", "protocol.identity")
    protocol_version = parser.scalar(
        "protocol_version", "protocol.identity.protocol_version", "identity"
    )
    if _is_complete(protocol_version) and protocol_version != "1.0":
        parser.violation(
            "SCHEMA_ERROR", "protocol.identity.protocol_version", "version must be 1.0"
        )
    _legacy_identity(parser)

    parser.expect_literal("## Models", "models")
    parser.expect_literal("### Skill compiler", "models.skill_compiler")
    compiler = _read_fields(parser, _COMPILER_FIELDS, "models.skill_compiler", "compiler")
    parser.expect_literal("### Skill executor", "models.skill_executor")
    executor = _read_fields(parser, _EXECUTOR_FIELDS, "models.skill_executor", "executor")
    for hash_field in ("system_prompt_hash", "structured_output_schema_hash"):
        _hash(parser, compiler[hash_field], f"models.skill_compiler.{hash_field}")
    _hash(parser, executor["runtime_prompt_hashes"], "models.skill_executor.runtime_prompt_hashes")
    compiler_temperature = _finite_decimal(
        parser, compiler["temperature"], "models.skill_compiler.temperature"
    )
    compiler_seed = _integer(parser, compiler["seed"], "models.skill_compiler.seed")
    compiler_max_tokens = _integer(
        parser, compiler["max_tokens"], "models.skill_compiler.max_tokens", positive=True
    )
    executor_temperature = _finite_decimal(
        parser, executor["temperature"], "models.skill_executor.temperature"
    )
    executor_max_turns = _integer(
        parser, executor["max_turns"], "models.skill_executor.max_turns", positive=True
    )
    executor_max_tool_calls = _integer(
        parser, executor["max_tool_calls"], "models.skill_executor.max_tool_calls", positive=True
    )
    executor_max_tokens = _integer(
        parser, executor["max_tokens"], "models.skill_executor.max_tokens", positive=True
    )

    parser.expect_literal("## Domains", "domains")
    access_hash = parser.scalar(
        "access_provisioning world hash", "domains.access_provisioning.world_hash", "domains"
    )
    financial_hash = parser.scalar(
        "financial_adjustments world hash", "domains.financial_adjustments.world_hash", "domains"
    )
    _hash(parser, access_hash, "domains.access_provisioning.world_hash")
    _hash(parser, financial_hash, "domains.financial_adjustments.world_hash")

    parser.expect_literal("## Demonstration bundles", "demonstrations")
    primary_trace_count = parser.scalar(
        "primary trace count", "demonstrations.primary_trace_count", "demonstrations"
    )
    ratios = parser.scalar("ratios", "demonstrations.ratios", "demonstrations")
    bundle_seeds = parser.scalar(
        "bundle seeds per domain/ratio",
        "demonstrations.bundle_seeds_per_domain_ratio",
        "demonstrations",
    )
    generator_hash = parser.scalar(
        "generator hash", "demonstrations.generator_hash", "demonstrations"
    )
    trace_schema_hash = parser.scalar(
        "trace schema hash", "demonstrations.trace_schema_hash", "demonstrations"
    )
    narration = parser.scalar("narration", "demonstrations.narration", "demonstrations")
    trace_count = _integer(
        parser,
        primary_trace_count,
        "demonstrations.primary_trace_count",
        mismatch_code="CORPUS_COUNT_MISMATCH",
    )
    seed_count = _integer(
        parser,
        bundle_seeds,
        "demonstrations.bundle_seeds_per_domain_ratio",
        mismatch_code="CORPUS_COUNT_MISMATCH",
    )
    if trace_count != 12 or seed_count != 3 or ratios != "[0.0, 0.25, 0.5, 0.75, 1.0]":
        parser.violation(
            "CORPUS_COUNT_MISMATCH", "demonstrations", "trace count, ratios, or seed count differs"
        )
    if _is_complete(narration) and narration != "disabled in primary Stage A":
        parser.violation("SCHEMA_ERROR", "demonstrations.narration", "narration must be disabled")
    _hash(parser, generator_hash, "demonstrations.generator_hash")
    _hash(parser, trace_schema_hash, "demonstrations.trace_schema_hash")

    parser.expect_literal("## Conditions", "conditions")
    condition_values: list[tuple[str, str, str]] = []
    for label, template_id, canonical_id in _CONDITIONS:
        value = parser.scalar(
            label,
            f"conditions.{template_id}",
            "conditions",
            blank_code="CONDITION_HASH_MISSING",
            mismatch_code="CONDITION_SET_MISMATCH",
        )
        _hash(parser, value, f"conditions.{template_id}", "CONDITION_HASH_FORMAT")
        condition_values.append((template_id, canonical_id, value))
    parser.expect_literal("Confirm:", "conditions.proofs")
    proofs: dict[str, str] = {}
    for label, proof_field in _PROOF_FIELDS:
        location = f"conditions.proofs.{proof_field}"
        value = parser.scalar(
            label,
            location,
            "proofs",
            blank_code="PROOF_MISSING",
            mismatch_code="PROOF_MISSING",
            template_continuation=True,
            superseded_aliases=(_LEGACY_A5_PROOF_LABEL,)
            if proof_field == "a5_buried_handbook_artifact_frozen"
            else (),
        )
        _hash(parser, value, location)
        proofs[proof_field] = value

    parser.expect_literal("## Primary outcomes", "outcomes")
    outcomes = _fixed_literals(
        parser, ("- Task Completion", "- Completion Under Policy"), "outcomes"
    )
    parser.expect_literal("## Primary Stage A estimands", "estimands.stage_a")
    stage_a_prefix = _fixed_literals(
        parser,
        (
            "- E1 Behavioral contamination slope",
            "- E2 Same-tier policy attenuation",
            "- E3 Instruction hierarchy",
            "- E4 Skill interference under policy",
        ),
        "estimands.stage_a",
    )
    parser.expect_literal(
        "- E5 Buried-handbook contextual penalty",
        "estimands.stage_a",
        superseded_aliases=(_LEGACY_E5_ESTIMAND,),
    )
    stage_a_estimands = (*stage_a_prefix, "- E5 Buried-handbook contextual penalty")
    parser.expect_literal("## Primary Stage B estimands", "estimands.stage_b")
    stage_b_estimands = _fixed_literals(
        parser,
        (
            "- E6 Unsafe imitation",
            "- E7 Flat-policy false enforcement",
            "- E8 Authority-aware gain",
            "- E9 Deterministic enforcement gap",
        ),
        "estimands.stage_b",
    )

    parser.expect_literal("## Figure 1 predictions", "predictions")
    parser.expect_literal(
        "Enter point prediction and 80% subjective interval for each condition and ratio.",
        "predictions.instructions",
    )
    prediction_rows = _prediction_table(parser)
    null_probability = parser.scalar(
        "predicted probability of null A4–A3 hierarchy effect",
        "predictions.null_a4_a3_hierarchy_effect",
        "predictions",
        blank_code="PREDICTION_REQUIRED",
    )
    b1_probability = parser.scalar(
        "predicted probability that B1 over-enforces at least one authority class",
        "predictions.b1_over_enforces_authority_class",
        "predictions",
        blank_code="PREDICTION_REQUIRED",
    )
    b2_probability = parser.scalar(
        "predicted probability that B2 adds no value over B1",
        "predictions.b2_adds_no_value_over_b1",
        "predictions",
        blank_code="PREDICTION_REQUIRED",
    )
    probabilities = tuple(
        _prediction_number(parser, value, location, required=True)
        for value, location in (
            (null_probability, "predictions.null_a4_a3_hierarchy_effect"),
            (b1_probability, "predictions.b1_over_enforces_authority_class"),
            (b2_probability, "predictions.b2_adds_no_value_over_b1"),
        )
    )

    parser.expect_literal("## Confirmatory corpus", "corpus")
    parser.expect_literal("### Stage A", "corpus.stage_a")
    stage_a_raw = _read_fields(
        parser,
        (
            ("skill bundles", "skill_bundles"),
            ("held-out cases per domain", "held_out_cases_per_domain"),
            ("repeats", "repeats"),
            ("planned skill-dependent episodes", "planned_skill_dependent_episodes"),
            ("planned control episodes", "planned_control_episodes"),
            ("total", "total"),
        ),
        "corpus.stage_a",
        "stage_a",
    )
    parser.expect_literal("### Stage B", "corpus.stage_b")
    stage_b_raw = _read_fields(
        parser,
        (
            ("R75 skills per domain", "r75_skills_per_domain"),
            (
                "held-out cases per authority class/domain",
                "held_out_cases_per_authority_class_domain",
            ),
            ("authority classes", "authority_classes"),
            ("conditions", "conditions"),
            ("repeats", "repeats"),
            ("total", "total"),
        ),
        "corpus.stage_b",
        "stage_b",
    )
    stage_a_numbers = {
        field: _integer(
            parser, value, f"corpus.stage_a.{field}", mismatch_code="CORPUS_COUNT_MISMATCH"
        )
        for field, value in stage_a_raw.items()
    }
    stage_b_numbers = {
        field: _integer(
            parser, value, f"corpus.stage_b.{field}", mismatch_code="CORPUS_COUNT_MISMATCH"
        )
        for field, value in stage_b_raw.items()
    }
    if stage_a_numbers != {
        "skill_bundles": 30,
        "held_out_cases_per_domain": 20,
        "repeats": 3,
        "planned_skill_dependent_episodes": 7200,
        "planned_control_episodes": 240,
        "total": 7440,
    }:
        parser.violation("CORPUS_COUNT_MISMATCH", "corpus.stage_a", "Stage A counts differ")
    if stage_b_numbers != {
        "r75_skills_per_domain": 3,
        "held_out_cases_per_authority_class_domain": 5,
        "authority_classes": 5,
        "conditions": 4,
        "repeats": 3,
        "total": 1800,
    }:
        parser.violation("CORPUS_COUNT_MISMATCH", "corpus.stage_b", "Stage B counts differ")

    parser.expect_literal("## Statistical plan", "statistical_plan")
    statistical = _read_fields(parser, _STATISTICAL_FIELDS, "statistical_plan", "statistical_plan")
    parser.expect_literal("## Exclusion rules", "exclusions")
    parser.expect_literal(
        "Only technical failures that occur before agent behavior begins may qualify.",
        "exclusions.introduction",
    )
    exclusions = _read_fields(parser, _EXCLUSION_FIELDS, "exclusions", "exclusions")
    parser.expect_literal("The following are outcomes, not exclusions:", "exclusions.outcomes")
    _fixed_literals(
        parser,
        (
            "- refusal;",
            "- invalid tool call;",
            "- agent timeout;",
            "- budget exhaustion;",
            "- unsafe imitation;",
            "- false enforcement;",
            "- unnecessary escalation;",
            "- wrong completion claim.",
        ),
        "exclusions.outcomes",
    )

    parser.expect_literal("## Representative-case selection", "representative_selection")
    parser.expect_literal("### Stage A", "representative_selection.stage_a")
    stage_a_selection = _fixed_literals(
        parser,
        (
            "1. cases where A2 fails CuP and A4 passes in majority of repeats;",
            "2. median contamination ratio;",
            "3. median action-trace length;",
            "4. lexicographically smallest case ID.",
        ),
        "representative_selection.stage_a",
    )
    parser.expect_literal("### Stage B", "representative_selection.stage_b")
    stage_b_selection = _fixed_literals(
        parser,
        (
            "1. cases where B1 and B2 differ;",
            "2. median authority-graph depth;",
            "3. lexicographic case-ID tie-break.",
        ),
        "representative_selection.stage_b",
    )

    parser.expect_literal("## Freeze statement", "freeze_statement")
    if parser.current() == "After external anchoring:":
        parser.index += 1
    else:
        parser.expect_literal("After custody receipt creation:", "freeze_statement")
    freeze_statement = _fixed_literals(
        parser,
        (
            "- no prompt changes;",
            "- no model changes;",
            "- no corpus/generator changes;",
            "- no policy text changes;",
            "- no condition changes;",
            "- no metric changes;",
            "- no exclusion changes;",
            "- no seed replacement after outcomes.",
            "Any amendment creates a new protocol version. Amended results are exploratory",
            "until separately confirmed.",
        ),
        "freeze_statement",
    )
    if parser.current() is not None:
        parser.violation(
            "MARKDOWN_PARSE_ERROR", "document.trailing", "trailing content is not allowed"
        )

    if parser.violations:
        return _report(source_path, None, parser.violations)
    try:
        core = PreregistrationCore(
            protocol_version="1.0",
            models=ModelsDeclaration(
                skill_compiler=SkillCompilerDeclaration(
                    provider=compiler["provider"],
                    model=compiler["model"],
                    model_version_date=compiler["model_version_date"],
                    system_prompt_hash=compiler["system_prompt_hash"],
                    structured_output_schema_hash=compiler["structured_output_schema_hash"],
                    temperature=compiler_temperature,
                    seed=compiler_seed,
                    max_tokens=compiler_max_tokens,
                ),
                skill_executor=SkillExecutorDeclaration(
                    provider=executor["provider"],
                    model=executor["model"],
                    model_version_date=executor["model_version_date"],
                    runtime_prompt_hashes=executor["runtime_prompt_hashes"],
                    temperature=executor_temperature,
                    seed_policy=executor["seed_policy"],
                    max_turns=executor_max_turns,
                    max_tool_calls=executor_max_tool_calls,
                    max_tokens=executor_max_tokens,
                ),
            ),
            domains=DomainWorlds(
                access_provisioning_world_hash=access_hash,
                financial_adjustments_world_hash=financial_hash,
            ),
            demonstrations=DemonstrationDeclaration(
                primary_trace_count=trace_count,
                ratios=(
                    Decimal("0.0"),
                    Decimal("0.25"),
                    Decimal("0.5"),
                    Decimal("0.75"),
                    Decimal("1.0"),
                ),
                bundle_seeds_per_domain_ratio=seed_count,
                generator_hash=generator_hash,
                trace_schema_hash=trace_schema_hash,
                narration=narration,
            ),
            conditions=tuple(
                ConditionDefinition(
                    template_id=template_id,
                    canonical_id=canonical_id,
                    definition_hash=value,
                )
                for template_id, canonical_id, value in condition_values
            ),
            condition_proofs=ConditionProofs(**proofs),
            primary_outcomes=(outcomes[0], outcomes[1]),
            stage_a_estimands=(
                stage_a_estimands[0],
                stage_a_estimands[1],
                stage_a_estimands[2],
                stage_a_estimands[3],
                stage_a_estimands[4],
            ),
            stage_b_estimands=(
                stage_b_estimands[0],
                stage_b_estimands[1],
                stage_b_estimands[2],
                stage_b_estimands[3],
            ),
            predictions=FigurePredictions(
                rows=prediction_rows,
                null_a4_a3_hierarchy_effect=probabilities[0],
                b1_over_enforces_authority_class=probabilities[1],
                b2_adds_no_value_over_b1=probabilities[2],
            ),
            corpus_counts=CorpusCounts(
                stage_a=StageACorpusCounts(**stage_a_numbers),
                stage_b=StageBCorpusCounts(**stage_b_numbers),
            ),
            statistical_plan=StatisticalPlan(**statistical),
            technical_exclusions=TechnicalExclusions(**exclusions),
            stage_a_representative_selection=(
                stage_a_selection[0],
                stage_a_selection[1],
                stage_a_selection[2],
                stage_a_selection[3],
            ),
            stage_b_representative_selection=(
                stage_b_selection[0],
                stage_b_selection[1],
                stage_b_selection[2],
            ),
            freeze_statement=(
                freeze_statement[0],
                freeze_statement[1],
                freeze_statement[2],
                freeze_statement[3],
                freeze_statement[4],
                freeze_statement[5],
                freeze_statement[6],
                freeze_statement[7],
                freeze_statement[8],
                freeze_statement[9],
            ),
        )
    except ValidationError:
        parser.violation("SCHEMA_ERROR", "core", "completed core violates the typed schema")
        return _report(source_path, None, parser.violations)
    return _report(source_path, core, parser.violations)


def validate_preregistration_text(text: str) -> PreregistrationReport:
    if type(text) is not str:
        return _report(
            None,
            None,
            [
                ProtocolViolation(
                    code="MARKDOWN_PARSE_ERROR",
                    location="document",
                    detail="input must be text",
                )
            ],
        )
    return _validate_text(text, None)


def validate_preregistration(path: Path) -> PreregistrationReport:
    try:
        source_path: Path | None = path.resolve(strict=False)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        source_path = None
    try:
        text = path.read_text(encoding="utf-8")
    except (AttributeError, OSError, RuntimeError, TypeError, UnicodeError, ValueError):
        return _report(
            source_path,
            None,
            [
                ProtocolViolation(
                    code="LEDGER_IO_ERROR",
                    location="document",
                    detail="preregistration could not be read",
                )
            ],
        )
    return _validate_text(text, source_path)
