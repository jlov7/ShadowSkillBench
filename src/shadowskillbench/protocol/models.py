from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProtocolViolation(_ProtocolModel):
    code: str
    location: str
    detail: str


class SkillCompilerDeclaration(_ProtocolModel):
    provider: str
    model: str
    model_version_date: str
    system_prompt_hash: str
    structured_output_schema_hash: str
    temperature: Decimal
    seed: int
    max_tokens: int


class SkillExecutorDeclaration(_ProtocolModel):
    provider: str
    model: str
    model_version_date: str
    runtime_prompt_hashes: str
    temperature: Decimal
    seed_policy: str
    max_turns: int
    max_tool_calls: int
    max_tokens: int


class ModelsDeclaration(_ProtocolModel):
    skill_compiler: SkillCompilerDeclaration
    skill_executor: SkillExecutorDeclaration


class DomainWorlds(_ProtocolModel):
    access_provisioning_world_hash: str
    financial_adjustments_world_hash: str


class DemonstrationDeclaration(_ProtocolModel):
    primary_trace_count: int
    ratios: tuple[Decimal, ...]
    bundle_seeds_per_domain_ratio: int
    generator_hash: str
    trace_schema_hash: str
    narration: str


class ConditionDefinition(_ProtocolModel):
    template_id: str
    canonical_id: str
    definition_hash: str


class ConditionProofs(_ProtocolModel):
    a3_a4_policy_text_byte_identical: str
    a3_block_order_counterbalanced: str
    a3_a4_context_accounting_logged: str
    a5_buried_handbook_artifact_frozen: str


class PredictionCell(_ProtocolModel):
    point: Decimal
    interval80_low: Decimal
    interval80_high: Decimal


class PredictionRow(_ProtocolModel):
    condition: Literal["A1", "A2", "A3", "A4", "A5"]
    r0: PredictionCell
    r25: PredictionCell
    r50: PredictionCell
    r75: PredictionCell
    r100: PredictionCell


class FigurePredictions(_ProtocolModel):
    rows: tuple[PredictionRow, ...]
    null_a4_a3_hierarchy_effect: Decimal
    b1_over_enforces_authority_class: Decimal
    b2_adds_no_value_over_b1: Decimal


class StageACorpusCounts(_ProtocolModel):
    skill_bundles: int
    held_out_cases_per_domain: int
    repeats: int
    planned_skill_dependent_episodes: int
    planned_control_episodes: int
    total: int


class StageBCorpusCounts(_ProtocolModel):
    r75_skills_per_domain: int
    held_out_cases_per_authority_class_domain: int
    authority_classes: int
    conditions: int
    repeats: int
    total: int


class CorpusCounts(_ProtocolModel):
    stage_a: StageACorpusCounts
    stage_b: StageBCorpusCounts


class StatisticalPlan(_ProtocolModel):
    primary_model: str
    clustering: str
    bootstrap: str
    confidence_interval: str
    marginal_effects: str
    per_domain_reporting: str
    multiple_comparison_family: str
    material_effect_threshold: str


class TechnicalExclusions(_ProtocolModel):
    provider_unavailable_before_first_response: str
    malformed_provider_payload_after_retry: str
    environment_hash_mismatch: str
    runner_crash_before_first_action: str


class PreregistrationCore(_ProtocolModel):
    protocol_version: Literal["1.0"]
    models: ModelsDeclaration
    domains: DomainWorlds
    demonstrations: DemonstrationDeclaration
    conditions: tuple[ConditionDefinition, ...]
    condition_proofs: ConditionProofs
    primary_outcomes: tuple[str, str]
    stage_a_estimands: tuple[str, str, str, str, str]
    stage_b_estimands: tuple[str, str, str, str]
    predictions: FigurePredictions
    corpus_counts: CorpusCounts
    statistical_plan: StatisticalPlan
    technical_exclusions: TechnicalExclusions
    stage_a_representative_selection: tuple[str, str, str, str]
    stage_b_representative_selection: tuple[str, str, str]
    freeze_statement: tuple[str, str, str, str, str, str, str, str, str, str]


class PreregistrationReport(_ProtocolModel):
    source_path: Path | None
    valid: bool
    core: PreregistrationCore | None
    violations: tuple[ProtocolViolation, ...]
    warnings: tuple[ProtocolViolation, ...] = ()
