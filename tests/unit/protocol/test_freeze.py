from __future__ import annotations

import subprocess
import urllib.request
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shadowskillbench import cli
from shadowskillbench.protocol import freeze
from shadowskillbench.protocol.freeze import (
    PREREGISTRATION_HASH_BINDINGS,
    REQUIRED_INPUTS,
    FreezeInput,
    ProtocolFreezeError,
    build_freeze_manifest,
    write_freeze_artifacts,
)


def completed_preregistration(references: dict[str, str]) -> str:
    def reference(location: str) -> str:
        return references[location]

    conditions = (
        "A0 Bare",
        "A1 PolicyOnlySystem",
        "A2 SkillOnly",
        "A3 SkillPolicySameTier",
        "A4 SkillPolicySystemTier",
        "A5 SkillBuriedPolicySameTier",
        "B0 SkillOnly",
        "B1 FlatPolicySystem",
        "B2 AuthorityResolver",
        "B3 DeterministicGate",
    )
    fields = [
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
        f"- system_prompt_hash: {reference('models.skill_compiler.system_prompt_hash')}",
        (
            "- structured_output_schema_hash: "
            f"{reference('models.skill_compiler.structured_output_schema_hash')}"
        ),
        "- temperature: 0.0",
        "- seed: 0",
        "- max_tokens: 1",
        "",
        "### Skill executor",
        "",
        "- provider: fixture",
        "- model: fixture",
        "- model_version/date: fixture",
        f"- runtime_prompt_hashes: {reference('models.skill_executor.runtime_prompt_hashes')}",
        "- temperature: 0.0",
        "- seed policy: fixture",
        "- max_turns: 1",
        "- max_tool_calls: 1",
        "- max_tokens: 1",
        "",
        "## Domains",
        "",
        f"- access_provisioning world hash: {reference('domains.access_provisioning.world_hash')}",
        (
            "- financial_adjustments world hash: "
            f"{reference('domains.financial_adjustments.world_hash')}"
        ),
        "",
        "## Demonstration bundles",
        "",
        "- primary trace count: 12",
        "- ratios: [0.0, 0.25, 0.5, 0.75, 1.0]",
        "- bundle seeds per domain/ratio: 3",
        f"- generator hash: {reference('demonstrations.generator_hash')}",
        f"- trace schema hash: {reference('demonstrations.trace_schema_hash')}",
        "- narration: disabled in primary Stage A",
        "",
        "## Conditions",
        "",
    ]
    fields.extend(
        f"- {condition}: {reference(f'conditions.{condition.split()[0]}')}"
        for condition in conditions
    )
    fields.extend(
        [
            "",
            "Confirm:",
            "",
            (
                "- A3/A4 policy text is byte-identical: "
                f"{reference('conditions.proofs.a3_a4_policy_text_byte_identical')}"
            ),
            (
                "- A3 block order is counterbalanced: "
                f"{reference('conditions.proofs.a3_block_order_counterbalanced')}"
            ),
            (
                "- A3/A4 total context accounting is logged: "
                f"{reference('conditions.proofs.a3_a4_context_accounting_logged')}"
            ),
            (
                "- A5 buried-handbook artifact, target position, and token count are frozen: "
                f"{reference('conditions.proofs.a5_buried_handbook_artifact_frozen')}"
            ),
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
    prediction_cell = "0.5 [0.25, 0.75]"
    fields.extend(
        f"| {condition} | {prediction_cell} | {prediction_cell} | {prediction_cell} | "
        f"{prediction_cell} | {prediction_cell} |"
        for condition in ("A1", "A2", "A3", "A4", "A5")
    )
    fields.extend(
        [
            "",
            "- predicted probability of null A4–A3 hierarchy effect: 0.5",
            "- predicted probability that B1 over-enforces at least one authority class: 0.5",
            "- predicted probability that B2 adds no value over B1: 0.5",
            "",
            "## Confirmatory corpus",
            "",
            "### Stage A",
            "",
            "- skill bundles: 30",
            "- held-out cases per domain: 20",
            "- repeats: 3",
            "- planned skill-dependent episodes: 7200",
            "- planned control episodes: 240",
            "- total: 7440",
            "",
            "### Stage B",
            "",
            "- R75 skills per domain: 3",
            "- held-out cases per authority class/domain: 5",
            "- authority classes: 5",
            "- conditions: 4",
            "- repeats: 3",
            "- total: 1800",
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
            "After custody receipt creation:",
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
    return "\n".join(fields) + "\n"


def fixture_inputs(tmp_path: Path, *, matching_referent: bool = True) -> tuple[FreezeInput, ...]:
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    references: dict[str, str] = {}
    inputs: list[FreezeInput] = [
        FreezeInput("protocol/preregistration.md", "preregistration_core"),
        FreezeInput("protocol/CLAIMS_LEDGER.yaml", "claims_ledger"),
    ]
    for binding in PREREGISTRATION_HASH_BINDINGS:
        referent = binding.location.encode("utf-8")
        if binding.role == "runtime_skill_ir_schema":
            from shadowskillbench.protocol.schema_artifacts import compiler_skill_ir_schema_bytes

            referent = compiler_skill_ir_schema_bytes()
        if binding.role == "runtime_prompt_manifest":
            from shadowskillbench.protocol.runtime_prompt_manifest import (
                runtime_prompt_manifest_bytes,
            )

            referent = runtime_prompt_manifest_bytes()
        path = tmp_path / binding.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(referent)
        references[binding.location] = f"sha256:{sha256(referent).hexdigest()}"
        inputs.append(FreezeInput(binding.path, binding.role))
    if not matching_referent:
        references["models.skill_compiler.system_prompt_hash"] = "sha256:" + "0" * 64
    (protocol / "preregistration.md").write_text(
        completed_preregistration(references), encoding="utf-8"
    )
    (protocol / "CLAIMS_LEDGER.yaml").write_text(
        "\n".join(
            (
                'version: "1.0"',
                "claims:",
                "  - id: PRIOR-001",
                "    wording: prior art",
                "    status: supported_prior_art",
                "    evidence: [source]",
                "    scope: narrow",
                "",
            )
        ),
        encoding="utf-8",
    )
    return tuple(inputs)


def test_manifest_is_deterministic_and_binds_preregistration_referents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = fixture_inputs(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("process call"))
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network call")
    )

    first = build_freeze_manifest(tmp_path, required_inputs=inputs)
    second = build_freeze_manifest(tmp_path, required_inputs=inputs)

    assert first.bytes == second.bytes
    assert first.sha256 == second.sha256
    assert first.payload["anchor_status"] == "PENDING_CUSTODY_RECEIPT"
    referents = first.payload["referents"]
    assert type(referents) is list
    assert len(referents) == len(PREREGISTRATION_HASH_BINDINGS)
    assert all(type(item) is dict and "location" in item and "role" in item for item in referents)


def test_outputs_are_detached_and_refuse_overwrite(tmp_path: Path) -> None:
    inputs = fixture_inputs(tmp_path)

    artifacts = write_freeze_artifacts(tmp_path, required_inputs=inputs)

    assert artifacts.manifest_path.read_bytes() == artifacts.manifest.bytes
    assert artifacts.sha256_path.read_text(encoding="utf-8") == (
        f"{artifacts.manifest.sha256.removeprefix('sha256:')}  freeze_manifest.json\n"
    )
    assert "HOLD_PENDING_CUSTODY_RECEIPT" in artifacts.summary_path.read_text(encoding="utf-8")
    with pytest.raises(ProtocolFreezeError) as error:
        write_freeze_artifacts(tmp_path, required_inputs=inputs)
    assert {item.code for item in error.value.violations} == {"FREEZE_OUTPUT_EXISTS"}


def test_cli_validates_and_freezes_only_the_current_fixture_protocol_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = fixture_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "write_freeze_artifacts",
        lambda root: write_freeze_artifacts(root, required_inputs=inputs),
    )
    runner = CliRunner()

    validation = runner.invoke(cli.app, ["protocol", "validate"])
    frozen = runner.invoke(cli.app, ["protocol", "freeze"])
    repeated = runner.invoke(cli.app, ["protocol", "freeze"])

    assert validation.exit_code == 0
    assert validation.stdout == "PASS_PROTOCOL_VALIDATION\n"
    assert frozen.exit_code == 0
    assert frozen.stdout.startswith("PASS_PROTOCOL_FREEZE: manifest_sha256=sha256:")
    assert "HOLD_PENDING_CUSTODY_RECEIPT" in frozen.stdout
    assert (tmp_path / "protocol" / "freeze_manifest.json").is_file()
    assert (tmp_path / "protocol" / "freeze_manifest.sha256").is_file()
    assert (tmp_path / "protocol" / "FREEZE_SUMMARY.md").is_file()
    assert repeated.exit_code == 1
    assert "HOLD_PROTOCOL_FREEZE" in repeated.stderr
    assert "FREEZE_OUTPUT_EXISTS" in repeated.stderr


def test_output_failure_removes_partial_freeze_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = fixture_inputs(tmp_path)
    original_publish = freeze._publish_exclusively

    def inject_intervening_output(source: Path, destination: Path) -> None:
        if destination.name == "freeze_manifest.sha256":
            destination.write_bytes(b"intervening artifact")
        original_publish(source, destination)

    monkeypatch.setattr(freeze, "_publish_exclusively", inject_intervening_output)

    with pytest.raises(ProtocolFreezeError) as error:
        write_freeze_artifacts(tmp_path, required_inputs=inputs)

    assert {item.code for item in error.value.violations} == {"FREEZE_OUTPUT_IO"}
    assert not (tmp_path / "protocol" / "freeze_manifest.json").exists()
    assert (
        tmp_path / "protocol" / "freeze_manifest.sha256"
    ).read_bytes() == b"intervening artifact"
    assert not (tmp_path / "protocol" / "FREEZE_SUMMARY.md").exists()


InputFactory = Callable[[Path], tuple[FreezeInput, ...]]


@pytest.mark.parametrize(
    ("inputs", "code"),
    [
        (
            lambda tmp_path: fixture_inputs(tmp_path, matching_referent=False),
            "PREREGISTRATION_BINDING_HASH_MISMATCH",
        ),
        (
            lambda tmp_path: (
                fixture_inputs(tmp_path)
                + (FreezeInput("protocol/anchor_receipt.json", "anchor_receipt"),)
            ),
            "POST_FREEZE_ARTIFACT",
        ),
    ],
)
def test_freeze_fails_closed_for_unbound_or_post_freeze_inputs(
    tmp_path: Path, inputs: InputFactory, code: str
) -> None:
    required_inputs = inputs(tmp_path)

    with pytest.raises(ProtocolFreezeError) as error:
        build_freeze_manifest(tmp_path, required_inputs=required_inputs)

    assert code in {item.code for item in error.value.violations}


def test_freeze_rejects_a_hash_bound_to_the_wrong_role(tmp_path: Path) -> None:
    inputs = list(fixture_inputs(tmp_path))
    binding = PREREGISTRATION_HASH_BINDINGS[0]
    index = next(index for index, item in enumerate(inputs) if item.path == binding.path)
    inputs[index] = FreezeInput(binding.path, "wrong_compiler_prompt_role")

    with pytest.raises(ProtocolFreezeError) as error:
        build_freeze_manifest(tmp_path, required_inputs=tuple(inputs))

    assert {item.code for item in error.value.violations} == {
        "PREREGISTRATION_BINDING_ROLE_MISMATCH"
    }


def test_freeze_rejects_an_ambiguous_preregistration_hash(tmp_path: Path) -> None:
    inputs = list(fixture_inputs(tmp_path))
    binding = PREREGISTRATION_HASH_BINDINGS[0]
    duplicate_path = "fixtures/duplicate-compiler-prompt.bin"
    duplicate = tmp_path / duplicate_path
    duplicate.parent.mkdir()
    duplicate.write_bytes((tmp_path / binding.path).read_bytes())
    inputs.append(FreezeInput(duplicate_path, "duplicate_compiler_prompt_bytes"))

    with pytest.raises(ProtocolFreezeError) as error:
        build_freeze_manifest(tmp_path, required_inputs=tuple(inputs))

    assert {item.code for item in error.value.violations} == {"AMBIGUOUS_PREREGISTRATION_REFERENT"}


def test_closed_production_surface_covers_load_bearing_artifacts() -> None:
    paths = {item.path for item in REQUIRED_INPUTS}
    roles = [item.role for item in REQUIRED_INPUTS]

    assert {
        "protocol/analysis_plan.v3.json",
        "protocol/power_precision_plan.v3.json",
        "protocol/power_precision_evidence.v3.json",
        "src/shadowskillbench/protocol/power_precision.py",
        "src/shadowskillbench/protocol/power_precision_design.py",
        "src/shadowskillbench/corpus/development.py",
        "src/shadowskillbench/experiments/io.py",
        "src/shadowskillbench/experiments/planner.py",
        "src/shadowskillbench/experiments/confirmatory_package.py",
        "src/shadowskillbench/experiments/runner.py",
        "src/shadowskillbench/experiments/audit.py",
        "src/shadowskillbench/metrics/outcomes.py",
        "src/shadowskillbench/analysis/dataset.py",
        "src/shadowskillbench/analysis/exclusions.py",
        "src/shadowskillbench/analysis/profile.py",
        "src/shadowskillbench/analysis/sealed.py",
        "src/shadowskillbench/analysis/stage_a.py",
        "src/shadowskillbench/analysis/stage_b.py",
        "src/shadowskillbench/reporting/selection.py",
        "src/shadowskillbench/reporting/report.py",
        "src/shadowskillbench/reporting/templates/report.html.j2",
        "src/shadowskillbench/reporting/workbench_export.py",
        "src/shadowskillbench/release/reproduce.py",
        "src/shadowskillbench/cli.py",
        "scripts/reproduce.sh",
        "src/shadowskillbench/protocol/claims.py",
        "src/shadowskillbench/protocol/claim_estimands.py",
        "src/shadowskillbench/protocol/scientific_freeze.py",
        "src/shadowskillbench/traces/bundles.py",
        "src/shadowskillbench/skills/compiler.py",
        "src/shadowskillbench/models/runtime.py",
        "docs/adr/ADR-002-frozen-runner-outage-retry.md",
    } <= paths
    assert len(paths) == len(REQUIRED_INPUTS)
    assert len(set(roles)) == len(roles)
    assert not any(item.path.startswith("protocol/anchor_receipt.") for item in REQUIRED_INPUTS)
