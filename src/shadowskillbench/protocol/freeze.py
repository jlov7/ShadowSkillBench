"""Deterministic, offline construction of the protocol-freeze artifacts.

This module deliberately stops before commit and post-commit custody receipt
creation. Those operations are performed after these artifacts are reviewed.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from shadowskillbench.protocol.claims import validate_claims_ledger
from shadowskillbench.protocol.runtime_prompt_manifest import runtime_prompt_manifest_matches
from shadowskillbench.protocol.schema_artifacts import runtime_skill_ir_schema_matches
from shadowskillbench.protocol.validate import validate_preregistration

_SHA256_REFERENCE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SCHEMA_VERSION = "1.0"
_PREREGISTRATION = "protocol/preregistration.md"
_OUTPUT_PATHS = frozenset(
    {
        "protocol/freeze_manifest.json",
        "protocol/freeze_manifest.sha256",
        "protocol/FREEZE_SUMMARY.md",
        "protocol/anchor_receipt.json",
    }
)


@dataclass(frozen=True, slots=True)
class FreezeInput:
    """One named, repository-relative byte input to the scientific freeze."""

    path: str
    role: str


@dataclass(frozen=True, slots=True)
class PreregistrationHashBinding:
    """One preregistration hash field and its sole admitted freeze input."""

    location: str
    path: str
    role: str


@dataclass(frozen=True, slots=True)
class FreezeViolation:
    code: str
    path: str
    detail: str


class ProtocolFreezeError(ValueError):
    """Raised when a freeze cannot be constructed without weakening custody."""

    def __init__(self, violations: Iterable[FreezeViolation]) -> None:
        self.violations = tuple(
            sorted(violations, key=lambda item: (item.path, item.code, item.detail))
        )
        message = (
            "; ".join(f"{item.code}:{item.path}:{item.detail}" for item in self.violations)
            or "protocol freeze failed"
        )
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class FreezeManifest:
    """The deterministic manifest bytes and their detached SHA-256 reference."""

    payload: Mapping[str, object]
    bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class FreezeArtifacts:
    manifest_path: Path
    sha256_path: Path
    summary_path: Path
    manifest: FreezeManifest


# This is intentionally an explicit, closed list.  A glob would make the protocol
# surface depend on incidental local files, while an omitted path would silently
# leave a preregistered mechanism mutable.
REQUIRED_INPUTS: tuple[FreezeInput, ...] = (
    FreezeInput("protocol/preregistration.md", "preregistration_core"),
    FreezeInput("protocol/analysis_plan.v3.json", "successor_analysis_plan"),
    FreezeInput("protocol/power_precision_plan.v3.json", "successor_power_precision_plan"),
    FreezeInput("protocol/power_precision_evidence.v3.json", "successor_power_precision_evidence"),
    FreezeInput("src/shadowskillbench/protocol/power_precision.py", "power_precision_simulator"),
    FreezeInput(
        "src/shadowskillbench/protocol/power_precision_design.py",
        "power_precision_execution_design",
    ),
    FreezeInput("protocol/CLAIMS_LEDGER.yaml", "claims_ledger"),
    FreezeInput("protocol/confound_register.md", "confound_register"),
    FreezeInput("protocol/requirement_ledger.md", "requirement_ledger"),
    FreezeInput("docs/03_BENCHMARK_PROTOCOL.md", "benchmark_protocol"),
    FreezeInput("docs/04_DOMAIN_ACCESS_PROVISIONING.md", "access_domain_specification"),
    FreezeInput("docs/05_DOMAIN_FINANCIAL_ADJUSTMENTS.md", "finance_domain_specification"),
    FreezeInput("docs/06_AUTHORITY_AND_EVIDENCE_MODEL.md", "authority_specification"),
    FreezeInput("docs/07_SKILL_INDUCTION_AND_EXECUTION_SPEC.md", "skill_specification"),
    FreezeInput("docs/08_EXPERIMENT_AND_STATISTICS_SPEC.md", "statistics_specification"),
    FreezeInput("docs/10_SECURITY_PRIVACY_GOVERNANCE.md", "governance_specification"),
    FreezeInput("docs/15_TECHNICAL_ARCHITECTURE.md", "architecture_specification"),
    FreezeInput("docs/16_TEST_STRATEGY.md", "test_strategy"),
    FreezeInput("docs/17_PROTOCOL_FREEZE_RUNBOOK.md", "anchor_runbook"),
    FreezeInput("docs/PROTOCOL_FREEZE_RUNBOOK.md", "freeze_builder_runbook"),
    FreezeInput("docs/adr/ADR-000-architecture-audit.md", "architecture_decisions"),
    FreezeInput("docs/adr/ADR-002-frozen-runner-outage-retry.md", "runner_outage_retry_decision"),
    FreezeInput("prompts/skill_compiler.md", "base_compiler_prompt_source"),
    FreezeInput("prompts/confirmatory_skill_compiler.md", "confirmatory_compiler_prompt"),
    FreezeInput("prompts/executor_system.md", "executor_system_prompt"),
    FreezeInput("config/models.yaml", "model_configuration"),
    FreezeInput("config/compiler.yaml", "compiler_configuration"),
    FreezeInput("config/executor.yaml", "executor_configuration"),
    FreezeInput("config/corpora.yaml", "corpus_configuration"),
    FreezeInput("config/statistics.yaml", "statistics_configuration"),
    FreezeInput("protocol/commitments/exclusion_rule.json", "confirmatory_exclusion_rule"),
    FreezeInput("schemas/action_trace.schema.json", "action_trace_schema"),
    FreezeInput("schemas/authority_record.schema.json", "authority_schema"),
    FreezeInput("schemas/episode_manifest.schema.json", "episode_schema"),
    FreezeInput("schemas/skill_ir.schema.json", "skill_schema"),
    FreezeInput("protocol/commitments/compiler_skill_ir_schema.json", "runtime_skill_ir_schema"),
    FreezeInput(
        "protocol/commitments/confirmatory_compiler_wire.schema.json",
        "confirmatory_compiler_wire_schema",
    ),
    FreezeInput(
        "protocol/commitments/role_conformance_receipt.json",
        "confirmatory_role_conformance_receipt",
    ),
    FreezeInput("protocol/commitments/run_descriptor.json", "confirmatory_run_descriptor"),
    FreezeInput("protocol/commitments/runtime_prompt_manifest.json", "runtime_prompt_manifest"),
    FreezeInput("protocol/commitments/access_worlds.json", "access_worlds_aggregate"),
    FreezeInput("protocol/commitments/financial_worlds.json", "financial_worlds_aggregate"),
    FreezeInput("protocol/commitments/conditions/A0_BARE.json", "condition_a0_bare"),
    FreezeInput(
        "protocol/commitments/conditions/A1_POLICY_ONLY_SYSTEM.json",
        "condition_a1_policy_only_system",
    ),
    FreezeInput("protocol/commitments/conditions/A2_SKILL_ONLY.json", "condition_a2_skill_only"),
    FreezeInput(
        "protocol/commitments/conditions/A3_SKILL_POLICY_SAME_TIER.json",
        "condition_a3_skill_policy_same_tier",
    ),
    FreezeInput(
        "protocol/commitments/conditions/A4_SKILL_POLICY_SYSTEM_TIER.json",
        "condition_a4_skill_policy_system_tier",
    ),
    FreezeInput(
        "protocol/commitments/conditions/A5_SKILL_BURIED_POLICY_SAME_TIER.json",
        "condition_a5_skill_buried_policy_same_tier",
    ),
    FreezeInput("protocol/commitments/conditions/B0_SKILL_ONLY.json", "condition_b0_skill_only"),
    FreezeInput(
        "protocol/commitments/conditions/B1_FLAT_POLICY_SYSTEM.json",
        "condition_b1_flat_policy_system",
    ),
    FreezeInput(
        "protocol/commitments/conditions/B2_AUTHORITY_RESOLVER.json",
        "condition_b2_authority_resolver",
    ),
    FreezeInput(
        "protocol/commitments/conditions/B3_DETERMINISTIC_GATE.json",
        "condition_b3_deterministic_gate",
    ),
    FreezeInput(
        "protocol/commitments/proofs/a3_a4_policy_text_equal.json", "proof_a3_a4_policy_text_equal"
    ),
    FreezeInput("protocol/commitments/proofs/a3_counterbalance.json", "proof_a3_counterbalance"),
    FreezeInput(
        "protocol/commitments/proofs/a3_a4_context_accounting.json",
        "proof_a3_a4_context_accounting",
    ),
    FreezeInput("protocol/commitments/proofs/a5_buried_handbook.json", "proof_a5_buried_handbook"),
    FreezeInput("protocol/confirmatory_design_commitment.json", "confirmatory_design_commitment"),
    FreezeInput(
        "src/shadowskillbench/experiments/confirmatory_preparation.py", "design_commitment_builder"
    ),
    FreezeInput(
        "src/shadowskillbench/experiments/confirmatory_preparation_cli.py",
        "confirmatory_compiler_cli",
    ),
    FreezeInput(
        "src/shadowskillbench/experiments/confirmatory_compilation.py",
        "confirmatory_compiler_producer",
    ),
    FreezeInput("src/shadowskillbench/core/artifacts.py", "artifact_custody"),
    FreezeInput("src/shadowskillbench/core/hashing.py", "canonical_hashing"),
    FreezeInput("src/shadowskillbench/engine/models.py", "engine_models"),
    FreezeInput("src/shadowskillbench/engine/runtime.py", "engine_runtime"),
    FreezeInput("src/shadowskillbench/engine/replay.py", "engine_replay"),
    FreezeInput("src/shadowskillbench/domains/semantics.py", "domain_semantics"),
    FreezeInput("src/shadowskillbench/domains/access/fixtures.py", "access_worlds"),
    FreezeInput("src/shadowskillbench/domains/access/models.py", "access_models"),
    FreezeInput("src/shadowskillbench/domains/access/adapter.py", "access_adapter"),
    FreezeInput("src/shadowskillbench/domains/access/semantics.py", "access_semantics"),
    FreezeInput("src/shadowskillbench/domains/access/verifier.py", "access_verifier"),
    FreezeInput("src/shadowskillbench/domains/finance/fixtures.py", "finance_worlds"),
    FreezeInput("src/shadowskillbench/domains/finance/models.py", "finance_models"),
    FreezeInput("src/shadowskillbench/domains/finance/adapter.py", "finance_adapter"),
    FreezeInput("src/shadowskillbench/domains/finance/reconciliation.py", "finance_reconciliation"),
    FreezeInput("src/shadowskillbench/domains/finance/semantics.py", "finance_semantics"),
    FreezeInput("src/shadowskillbench/domains/finance/verifier.py", "finance_verifier"),
    FreezeInput("src/shadowskillbench/authority/models.py", "authority_models"),
    FreezeInput("src/shadowskillbench/authority/resolver.py", "authority_resolver"),
    FreezeInput("src/shadowskillbench/authority/view.py", "authority_evidence_view"),
    FreezeInput("src/shadowskillbench/authority/gate.py", "deterministic_gate"),
    FreezeInput("src/shadowskillbench/policy/documents.py", "policy_documents"),
    FreezeInput("src/shadowskillbench/traces/models.py", "trace_models"),
    FreezeInput("src/shadowskillbench/traces/workers.py", "source_workers"),
    FreezeInput("src/shadowskillbench/traces/bundles.py", "source_bundle_generator"),
    FreezeInput("src/shadowskillbench/traces/narration.py", "trace_narration"),
    FreezeInput("src/shadowskillbench/skills/models.py", "skill_models"),
    FreezeInput("src/shadowskillbench/skills/projection.py", "compiler_projection"),
    FreezeInput("src/shadowskillbench/skills/compiler.py", "skill_compiler"),
    FreezeInput("src/shadowskillbench/skills/render.py", "skill_renderer"),
    FreezeInput("src/shadowskillbench/skills/contamination.py", "contamination_classifier"),
    FreezeInput("src/shadowskillbench/skills/gate2.py", "skill_gate"),
    FreezeInput("src/shadowskillbench/models/protocol.py", "model_protocol"),
    FreezeInput("src/shadowskillbench/models/runtime.py", "model_runtime"),
    FreezeInput("src/shadowskillbench/models/scripted.py", "scripted_model"),
    FreezeInput("src/shadowskillbench/models/openai_compatible.py", "model_adapter"),
    FreezeInput("src/shadowskillbench/models/ollama_native.py", "native_model_adapter"),
    FreezeInput("src/shadowskillbench/episodes/models.py", "episode_models"),
    FreezeInput("src/shadowskillbench/episodes/context.py", "context_assembler"),
    FreezeInput("src/shadowskillbench/episodes/executor.py", "episode_executor"),
    FreezeInput("src/shadowskillbench/episodes/tools.py", "episode_tools"),
    FreezeInput("src/shadowskillbench/corpus/development.py", "development_corpus"),
    FreezeInput("src/shadowskillbench/corpus/confirmatory.py", "confirmatory_corpus"),
    FreezeInput("src/shadowskillbench/experiments/planner.py", "episode_planner"),
    FreezeInput("src/shadowskillbench/experiments/confirmatory_package.py", "confirmatory_package"),
    FreezeInput(
        "src/shadowskillbench/experiments/ollama_gpt_oss_profile.py",
        "ollama_gpt_oss_runtime_profile",
    ),
    FreezeInput("src/shadowskillbench/experiments/runner.py", "episode_runner"),
    FreezeInput("src/shadowskillbench/experiments/audit.py", "experiment_audit"),
    FreezeInput("src/shadowskillbench/experiments/io.py", "experiment_artifact_loader"),
    FreezeInput("src/shadowskillbench/metrics/outcomes.py", "metrics"),
    FreezeInput("src/shadowskillbench/analysis/dataset.py", "analysis_dataset"),
    FreezeInput("src/shadowskillbench/analysis/exclusions.py", "analysis_exclusions"),
    FreezeInput("src/shadowskillbench/analysis/profile.py", "analysis_profile"),
    FreezeInput("src/shadowskillbench/analysis/sealed.py", "sealed_analysis"),
    FreezeInput("src/shadowskillbench/analysis/stage_a.py", "stage_a_estimators"),
    FreezeInput("src/shadowskillbench/analysis/stage_b.py", "stage_b_estimators"),
    FreezeInput("src/shadowskillbench/reporting/selection.py", "representative_selector"),
    FreezeInput("src/shadowskillbench/reporting/report.py", "report_renderer"),
    FreezeInput("src/shadowskillbench/reporting/templates/report.html.j2", "report_template"),
    FreezeInput("src/shadowskillbench/reporting/workbench_export.py", "workbench_exporter"),
    FreezeInput("src/shadowskillbench/release/reproduce.py", "release_reproducer"),
    FreezeInput("src/shadowskillbench/cli.py", "command_line_boundary"),
    FreezeInput("scripts/reproduce.sh", "reproduction_entrypoint"),
    FreezeInput("src/shadowskillbench/protocol/claim_estimands.py", "claim_estimand_mapping"),
    FreezeInput("src/shadowskillbench/protocol/claims.py", "claims_validator"),
    FreezeInput("src/shadowskillbench/protocol/scientific_freeze.py", "scientific_freeze_verifier"),
    FreezeInput("src/shadowskillbench/protocol/models.py", "preregistration_models"),
    FreezeInput("src/shadowskillbench/protocol/validate.py", "preregistration_validator"),
    FreezeInput("src/shadowskillbench/protocol/freeze.py", "freeze_builder"),
    FreezeInput(
        "src/shadowskillbench/protocol/runtime_prompt_manifest.py",
        "runtime_prompt_manifest_builder",
    ),
    FreezeInput("src/shadowskillbench/protocol/schema_artifacts.py", "schema_artifact_builder"),
    FreezeInput("src/shadowskillbench/skills/confirmatory_wire.py", "confirmatory_compiler_wire"),
)


PREREGISTRATION_HASH_BINDINGS: tuple[PreregistrationHashBinding, ...] = (
    PreregistrationHashBinding(
        "models.skill_compiler.system_prompt_hash",
        "prompts/confirmatory_skill_compiler.md",
        "confirmatory_compiler_prompt",
    ),
    PreregistrationHashBinding(
        "models.skill_compiler.structured_output_schema_hash",
        "protocol/commitments/confirmatory_compiler_wire.schema.json",
        "confirmatory_compiler_wire_schema",
    ),
    PreregistrationHashBinding(
        "models.skill_executor.runtime_prompt_hashes",
        "protocol/commitments/runtime_prompt_manifest.json",
        "runtime_prompt_manifest",
    ),
    PreregistrationHashBinding(
        "domains.access_provisioning.world_hash",
        "protocol/commitments/access_worlds.json",
        "access_worlds_aggregate",
    ),
    PreregistrationHashBinding(
        "domains.financial_adjustments.world_hash",
        "protocol/commitments/financial_worlds.json",
        "financial_worlds_aggregate",
    ),
    PreregistrationHashBinding(
        "demonstrations.generator_hash",
        "src/shadowskillbench/traces/bundles.py",
        "source_bundle_generator",
    ),
    PreregistrationHashBinding(
        "demonstrations.trace_schema_hash",
        "schemas/action_trace.schema.json",
        "action_trace_schema",
    ),
    PreregistrationHashBinding(
        "conditions.A0", "protocol/commitments/conditions/A0_BARE.json", "condition_a0_bare"
    ),
    PreregistrationHashBinding(
        "conditions.A1",
        "protocol/commitments/conditions/A1_POLICY_ONLY_SYSTEM.json",
        "condition_a1_policy_only_system",
    ),
    PreregistrationHashBinding(
        "conditions.A2",
        "protocol/commitments/conditions/A2_SKILL_ONLY.json",
        "condition_a2_skill_only",
    ),
    PreregistrationHashBinding(
        "conditions.A3",
        "protocol/commitments/conditions/A3_SKILL_POLICY_SAME_TIER.json",
        "condition_a3_skill_policy_same_tier",
    ),
    PreregistrationHashBinding(
        "conditions.A4",
        "protocol/commitments/conditions/A4_SKILL_POLICY_SYSTEM_TIER.json",
        "condition_a4_skill_policy_system_tier",
    ),
    PreregistrationHashBinding(
        "conditions.A5",
        "protocol/commitments/conditions/A5_SKILL_BURIED_POLICY_SAME_TIER.json",
        "condition_a5_skill_buried_policy_same_tier",
    ),
    PreregistrationHashBinding(
        "conditions.B0",
        "protocol/commitments/conditions/B0_SKILL_ONLY.json",
        "condition_b0_skill_only",
    ),
    PreregistrationHashBinding(
        "conditions.B1",
        "protocol/commitments/conditions/B1_FLAT_POLICY_SYSTEM.json",
        "condition_b1_flat_policy_system",
    ),
    PreregistrationHashBinding(
        "conditions.B2",
        "protocol/commitments/conditions/B2_AUTHORITY_RESOLVER.json",
        "condition_b2_authority_resolver",
    ),
    PreregistrationHashBinding(
        "conditions.B3",
        "protocol/commitments/conditions/B3_DETERMINISTIC_GATE.json",
        "condition_b3_deterministic_gate",
    ),
    PreregistrationHashBinding(
        "conditions.proofs.a3_a4_policy_text_byte_identical",
        "protocol/commitments/proofs/a3_a4_policy_text_equal.json",
        "proof_a3_a4_policy_text_equal",
    ),
    PreregistrationHashBinding(
        "conditions.proofs.a3_block_order_counterbalanced",
        "protocol/commitments/proofs/a3_counterbalance.json",
        "proof_a3_counterbalance",
    ),
    PreregistrationHashBinding(
        "conditions.proofs.a3_a4_context_accounting_logged",
        "protocol/commitments/proofs/a3_a4_context_accounting.json",
        "proof_a3_a4_context_accounting",
    ),
    PreregistrationHashBinding(
        "conditions.proofs.a5_buried_handbook_artifact_frozen",
        "protocol/commitments/proofs/a5_buried_handbook.json",
        "proof_a5_buried_handbook",
    ),
)


def _safe_relative_path(value: str) -> str | None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        return None
    return value


def _is_post_freeze_artifact(path: str) -> bool:
    return path in _OUTPUT_PATHS or path.startswith("protocol/anchor_receipt.")


def _read_inputs(
    repository_root: Path, required_inputs: tuple[FreezeInput, ...]
) -> tuple[list[dict[str, str]], dict[str, list[str]], list[FreezeViolation]]:
    records: list[dict[str, str]] = []
    hashes: dict[str, list[str]] = {}
    violations: list[FreezeViolation] = []
    paths: set[str] = set()
    roles: set[str] = set()
    try:
        root = repository_root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return [], {}, [FreezeViolation("ROOT_IO", ".", "repository root is not accessible")]

    for freeze_input in required_inputs:
        relative = _safe_relative_path(freeze_input.path)
        if relative is None:
            violations.append(
                FreezeViolation("UNSAFE_INPUT_PATH", freeze_input.path, "path must be canonical")
            )
            continue
        if _is_post_freeze_artifact(relative):
            violations.append(
                FreezeViolation(
                    "POST_FREEZE_ARTIFACT",
                    relative,
                    "manifest outputs and anchor receipts are excluded",
                )
            )
            continue
        if relative in paths:
            violations.append(
                FreezeViolation("DUPLICATE_INPUT_PATH", relative, "path appears twice")
            )
            continue
        if not freeze_input.role or freeze_input.role in roles:
            violations.append(
                FreezeViolation("DUPLICATE_INPUT_ROLE", relative, "role must be present and unique")
            )
            continue
        paths.add(relative)
        roles.add(freeze_input.role)
        candidate = root.joinpath(*PurePosixPath(relative).parts)
        try:
            if candidate.is_symlink() or not candidate.is_file():
                violations.append(
                    FreezeViolation(
                        "MISSING_REFERENT", relative, "regular non-symlink file is required"
                    )
                )
                continue
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            data = candidate.read_bytes()
        except (OSError, RuntimeError, ValueError):
            violations.append(
                FreezeViolation("REFERENT_IO", relative, "referent cannot be read safely")
            )
            continue
        digest = f"sha256:{sha256(data).hexdigest()}"
        records.append({"path": relative, "role": freeze_input.role, "sha256": digest})
        hashes.setdefault(digest, []).append(relative)
    return records, hashes, violations


def _hash_references(value: object) -> set[str]:
    if isinstance(value, str):
        return {value} if _SHA256_REFERENCE.fullmatch(value) else set()
    if isinstance(value, dict):
        return set().union(*(_hash_references(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_hash_references(item) for item in value)) if value else set()
    return set()


def _preregistration_references(core: object) -> dict[str, str]:
    """Return every hash field in the completed preregistration by exact location."""

    model_dump = getattr(core, "model_dump", None)
    if not callable(model_dump):
        raise ValueError("preregistration core is invalid")
    value = model_dump(mode="json")
    if type(value) is not dict:
        raise ValueError("preregistration core is invalid")
    models = value.get("models")
    domains = value.get("domains")
    demonstrations = value.get("demonstrations")
    conditions = value.get("conditions")
    proofs = value.get("condition_proofs")
    if (
        type(models) is not dict
        or type(domains) is not dict
        or type(demonstrations) is not dict
        or type(conditions) is not list
        or type(proofs) is not dict
    ):
        raise ValueError("preregistration core is invalid")
    compiler = models.get("skill_compiler")
    executor = models.get("skill_executor")
    if type(compiler) is not dict or type(executor) is not dict:
        raise ValueError("preregistration core is invalid")
    references: dict[str, str] = {
        "models.skill_compiler.system_prompt_hash": compiler["system_prompt_hash"],
        "models.skill_compiler.structured_output_schema_hash": compiler[
            "structured_output_schema_hash"
        ],
        "models.skill_executor.runtime_prompt_hashes": executor["runtime_prompt_hashes"],
        "domains.access_provisioning.world_hash": domains["access_provisioning_world_hash"],
        "domains.financial_adjustments.world_hash": domains["financial_adjustments_world_hash"],
        "demonstrations.generator_hash": demonstrations["generator_hash"],
        "demonstrations.trace_schema_hash": demonstrations["trace_schema_hash"],
        "conditions.proofs.a3_a4_policy_text_byte_identical": proofs[
            "a3_a4_policy_text_byte_identical"
        ],
        "conditions.proofs.a3_block_order_counterbalanced": proofs[
            "a3_block_order_counterbalanced"
        ],
        "conditions.proofs.a3_a4_context_accounting_logged": proofs[
            "a3_a4_context_accounting_logged"
        ],
        "conditions.proofs.a5_buried_handbook_artifact_frozen": proofs[
            "a5_buried_handbook_artifact_frozen"
        ],
    }
    for condition in conditions:
        if type(condition) is not dict:
            raise ValueError("preregistration core is invalid")
        template_id = condition.get("template_id")
        definition_hash = condition.get("definition_hash")
        if type(template_id) is not str or type(definition_hash) is not str:
            raise ValueError("preregistration core is invalid")
        references[f"conditions.{template_id}"] = definition_hash
    if any(type(reference) is not str for reference in references.values()):
        raise ValueError("preregistration core is invalid")
    return references


def _validate_preregistration_bindings(
    core: object, records: list[dict[str, str]], hashes: dict[str, list[str]]
) -> tuple[list[dict[str, object]], list[FreezeViolation]]:
    """Bind each declared hash to its sole predeclared file and role."""

    records_by_path = {record["path"]: record for record in records}
    bindings = {binding.location: binding for binding in PREREGISTRATION_HASH_BINDINGS}
    violations: list[FreezeViolation] = []
    references: list[dict[str, object]] = []
    try:
        declared = _preregistration_references(core)
    except (KeyError, ValueError):
        return [], [
            FreezeViolation(
                "INVALID_PREREGISTRATION_BINDINGS",
                _PREREGISTRATION,
                "completed core does not expose the expected hash fields",
            )
        ]
    if set(declared) != set(bindings):
        return [], [
            FreezeViolation(
                "INVALID_PREREGISTRATION_BINDINGS",
                _PREREGISTRATION,
                "completed core hash fields do not match the freeze binding schema",
            )
        ]
    for location, reference in sorted(declared.items()):
        binding = bindings[location]
        record = records_by_path.get(binding.path)
        if record is None:
            violations.append(
                FreezeViolation(
                    "PREREGISTRATION_BINDING_INPUT_MISSING",
                    location,
                    f"required frozen input is {binding.path}",
                )
            )
            continue
        if record["role"] != binding.role:
            violations.append(
                FreezeViolation(
                    "PREREGISTRATION_BINDING_ROLE_MISMATCH",
                    location,
                    f"required role is {binding.role}",
                )
            )
            continue
        if record["sha256"] != reference:
            violations.append(
                FreezeViolation(
                    "PREREGISTRATION_BINDING_HASH_MISMATCH",
                    location,
                    f"required frozen path is {binding.path}",
                )
            )
            continue
        referent_paths = sorted(hashes.get(reference, ()))
        if referent_paths != [binding.path]:
            violations.append(
                FreezeViolation(
                    "AMBIGUOUS_PREREGISTRATION_REFERENT",
                    location,
                    "hash must resolve to exactly one frozen input path",
                )
            )
            continue
        references.append(
            {
                "location": location,
                "reference": reference,
                "path": binding.path,
                "role": binding.role,
            }
        )
    return references, violations


def _canonical_json(value: Mapping[str, object]) -> bytes:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return encoded.encode() + b"\n"


def build_freeze_manifest(
    repository_root: Path, *, required_inputs: tuple[FreezeInput, ...] = REQUIRED_INPUTS
) -> FreezeManifest:
    """Validate all custody gates and return, but do not write, a manifest."""

    records, hashes, violations = _read_inputs(repository_root, required_inputs)
    paths = {record["path"] for record in records}
    if _PREREGISTRATION not in paths:
        violations.append(
            FreezeViolation(
                "MISSING_PREREGISTRATION_CORE", _PREREGISTRATION, "must be a frozen input"
            )
        )

    preregistration = validate_preregistration(repository_root / _PREREGISTRATION)
    if not preregistration.valid or preregistration.core is None:
        violations.append(
            FreezeViolation(
                "INVALID_PREREGISTRATION_CORE", _PREREGISTRATION, "completed core is required"
            )
        )
    claims = validate_claims_ledger(
        repository_root / "protocol/CLAIMS_LEDGER.yaml", repository_root
    )
    if not claims.valid:
        violations.append(
            FreezeViolation(
                "INVALID_CLAIMS_LEDGER", "protocol/CLAIMS_LEDGER.yaml", "ledger is invalid"
            )
        )

    references: list[dict[str, object]] = []
    if preregistration.core is not None:
        bound_references, binding_violations = _validate_preregistration_bindings(
            preregistration.core, records, hashes
        )
        references.extend(bound_references)
        violations.extend(binding_violations)
    runtime_schema = next(
        (record for record in records if record["role"] == "runtime_skill_ir_schema"), None
    )
    if runtime_schema is not None and not runtime_skill_ir_schema_matches(
        repository_root / runtime_schema["path"]
    ):
        violations.append(
            FreezeViolation(
                "STALE_RUNTIME_SKILL_IR_SCHEMA",
                runtime_schema["path"],
                "artifact must equal the canonical SkillIR runtime schema bytes",
            )
        )
    runtime_prompt_manifest = next(
        (record for record in records if record["role"] == "runtime_prompt_manifest"), None
    )
    if runtime_prompt_manifest is not None and not runtime_prompt_manifest_matches(
        repository_root / runtime_prompt_manifest["path"]
    ):
        violations.append(
            FreezeViolation(
                "STALE_RUNTIME_PROMPT_MANIFEST",
                runtime_prompt_manifest["path"],
                "artifact must equal the canonical runtime prompt manifest bytes",
            )
        )
    if violations:
        raise ProtocolFreezeError(violations)

    ordered_inputs = sorted(records, key=lambda item: item["path"])
    preregistration_hash = next(
        record["sha256"] for record in ordered_inputs if record["path"] == _PREREGISTRATION
    )
    payload: dict[str, object] = {
        "anchor_status": "PENDING_CUSTODY_RECEIPT",
        "inputs": ordered_inputs,
        "preregistration_core": {
            "path": _PREREGISTRATION,
            "sha256": preregistration_hash,
        },
        "referents": references,
        "schema_version": _SCHEMA_VERSION,
    }
    manifest_bytes = _canonical_json(payload)
    return FreezeManifest(
        payload=payload,
        bytes=manifest_bytes,
        sha256=f"sha256:{sha256(manifest_bytes).hexdigest()}",
    )


def render_freeze_summary(manifest: FreezeManifest) -> bytes:
    """Render a deterministic, human-readable statement of the freeze boundary."""

    inputs = manifest.payload["inputs"]
    if not isinstance(inputs, list):
        raise ValueError("freeze manifest inputs must be a list")
    count = len(inputs)
    return (
        "# ShadowSkillBench protocol freeze summary\n\n"
        f"- manifest_sha256: {manifest.sha256}\n"
        f"- frozen_input_count: {count}\n"
        "- status: HOLD_PENDING_CUSTODY_RECEIPT\n"
        "- anchor_receipt: excluded from manifest inputs; created after the freeze commit\n\n"
        "This artifact does not create a commit or authorize confirmatory execution.\n"
    ).encode()


def _publish_exclusively(temporary: Path, destination: Path) -> None:
    """Publish a staged output only when no other process created its destination."""

    os.link(temporary, destination)
    temporary.unlink()


def write_freeze_artifacts(
    repository_root: Path, *, required_inputs: tuple[FreezeInput, ...] = REQUIRED_INPUTS
) -> FreezeArtifacts:
    """Create manifest, detached SHA file, and summary without overwriting a prior freeze."""

    manifest = build_freeze_manifest(repository_root, required_inputs=required_inputs)
    output_dir = repository_root / "protocol"
    manifest_path = output_dir / "freeze_manifest.json"
    sha256_path = output_dir / "freeze_manifest.sha256"
    summary_path = output_dir / "FREEZE_SUMMARY.md"
    try:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise OSError("protocol output directory is invalid")
        existing = [path for path in (manifest_path, sha256_path, summary_path) if path.exists()]
    except OSError as error:
        raise ProtocolFreezeError(
            [FreezeViolation("FREEZE_OUTPUT_IO", "protocol", "output directory is not writable")]
        ) from error
    if existing:
        raise ProtocolFreezeError(
            [
                FreezeViolation(
                    "FREEZE_OUTPUT_EXISTS",
                    str(path.relative_to(repository_root)),
                    "refusing overwrite",
                )
                for path in existing
            ]
        )
    outputs = (
        (manifest_path, manifest.bytes),
        (
            sha256_path,
            f"{manifest.sha256.removeprefix('sha256:')}  freeze_manifest.json\n".encode(),
        ),
        (summary_path, render_freeze_summary(manifest)),
    )
    staged: list[tuple[Path, Path]] = []
    written: list[Path] = []
    try:
        for destination, content in outputs:
            with NamedTemporaryFile(
                dir=output_dir, prefix=f".{destination.name}.", delete=False
            ) as temporary:
                staged.append((Path(temporary.name), destination))
                temporary.write(content)
        for temporary, destination in staged:
            _publish_exclusively(temporary, destination)
            written.append(destination)
    except OSError as error:
        for temporary, _ in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        for destination in written:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise ProtocolFreezeError(
            [FreezeViolation("FREEZE_OUTPUT_IO", "protocol", "could not write all freeze outputs")]
        ) from error
    return FreezeArtifacts(manifest_path, sha256_path, summary_path, manifest)
