from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Never, cast
from urllib.parse import urlparse, urlunparse

import typer

from shadowskillbench.analysis import (
    SealedAnalysisHold,
    analyze_sealed_confirmatory_artifacts,
    sealed_analysis_projection,
    sealed_analysis_receipt_matches,
)
from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.confirmatory import (
    ConfirmatoryHoldError,
    authorize_confirmatory_generation,
    generate_confirmatory_from_files,
    inventory_from_development,
    load_model_visible_corpus,
    write_confirmatory_artifacts,
)
from shadowskillbench.corpus.development import (
    generate_development_corpus,
    validate_development_corpus,
)
from shadowskillbench.episodes.executor import run_episode
from shadowskillbench.episodes.models import EpisodePlan, EpisodeStage
from shadowskillbench.episodes.pilot_turn_wire import (
    CONFIRMATORY_TURN_PROMPT_PROFILE,
    LEGACY_CONFIRMATORY_TURN_PROMPT_PROFILE,
    LEGACY_V2_CONFIRMATORY_TURN_PROMPT_PROFILE,
    ConfirmatoryAgentTurnWire,
)
from shadowskillbench.experiments.audit import audit_confirmatory_run
from shadowskillbench.experiments.capability_pilot import (
    CapabilityPilotHold,
    build_gemma4_capability_client,
    build_gemma4_greedy_capability_client,
    build_gemma4_native_tools_context32768_capability_client,
    build_gemma4_native_tools_context32768_v2_capability_client,
    build_gemma4_native_tools_context32768_v3_capability_client,
    build_gemma4_native_tools_context32768_v4_capability_client,
    build_glm47_native_tools_context32k_capability_client,
    build_glm47_native_tools_response_v2_capability_client,
    build_mistral_capability_client,
    build_native_capability_client,
    build_nemotron_capability_client,
    build_nemotron_context32k_capability_client,
    build_nemotron_native_tools_context32k_capability_client,
    is_gemma4_native_tools_capability_bundle,
    is_gemma4_native_tools_v2_capability_bundle,
    is_gemma4_native_tools_v3_capability_bundle,
    is_gemma4_native_tools_v4_capability_bundle,
    is_glm47_native_tools_capability_bundle,
    is_glm47_native_tools_response_v2_capability_bundle,
    is_nemotron_capability_bundle,
    is_nemotron_context32k_capability_bundle,
    is_nemotron_native_tools_capability_bundle,
    run_capability_pilot,
    write_capability_audit,
)
from shadowskillbench.experiments.confirmatory_package import (
    ConfirmatoryPackageHold,
    confirmatory_executor_runtime_profile_projection,
    load_confirmatory_execution_package,
    load_confirmatory_execution_package_inputs,
    write_confirmatory_execution_package,
)
from shadowskillbench.experiments.confirmatory_preparation import (
    CORPUS_SEED,
    DEVELOPMENT_CORPUS_SEED,
)
from shadowskillbench.experiments.confirmatory_preparation_cli import (
    register_confirmatory_preparation_commands,
)
from shadowskillbench.experiments.development_execution import materialize_development_episodes
from shadowskillbench.experiments.development_plan import build_development_plan
from shadowskillbench.experiments.gemma4_recovery import (
    Gemma4RecoveryHold,
    run_recovery,
    write_recovery_audit,
    write_recovery_commitment,
)
from shadowskillbench.experiments.io import (
    ExperimentArtifactHold,
    load_confirmatory_audit_inputs,
)
from shadowskillbench.experiments.live_pilot import (
    OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
    PILOT_LABEL,
    LivePilotDescriptor,
    LivePilotError,
    audit_live_pilot,
    run_live_pilot,
)
from shadowskillbench.experiments.ollama_gemma4_native_tools_profile import (
    Gemma4NativeToolsProfileError,
    load_gemma4_native_tools_identity,
    load_gemma4_native_tools_role_profile_receipt,
)
from shadowskillbench.experiments.ollama_gemma4_profile_v3 import (
    Gemma4RoleProfileError,
    load_gemma4_role_profile_receipt,
)
from shadowskillbench.experiments.ollama_gemma4_profile_v4 import (
    Gemma4RoleProfileError as Gemma4GreedyRoleProfileError,
)
from shadowskillbench.experiments.ollama_gemma4_profile_v4 import (
    load_gemma4_role_profile_receipt as load_gemma4_greedy_role_profile_receipt,
)
from shadowskillbench.experiments.ollama_glm47_native_tools_profile import (
    Glm47NativeToolsRoleProfileError,
    load_glm47_native_tools_live_show_identity,
    load_glm47_native_tools_role_profile_receipt,
)
from shadowskillbench.experiments.ollama_glm47_native_tools_response_v2_profile import (
    Glm47NativeToolsResponseV2Error,
    load_glm47_native_tools_response_v2_identity,
    load_glm47_native_tools_response_v2_receipt,
)
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
    OLLAMA_GPT_OSS_CONTEXT_LENGTH,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL,
    OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
    OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
    OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_MODEL,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
    OLLAMA_GPT_OSS_PROFILE,
    OLLAMA_GPT_OSS_PROVIDER,
    OLLAMA_GPT_OSS_REASONING_EFFORT,
    OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_SAMPLING_TOP_P,
    OLLAMA_GPT_OSS_SERVER_VERSION,
    OllamaRoleProfileError,
    load_ollama_role_profile_receipt,
    ollama_gpt_oss_executor_modelfile_hash,
    ollama_gpt_oss_executor_template_hash,
)
from shadowskillbench.experiments.ollama_mistral_profile import (
    MistralRoleProfileError,
    load_mistral_role_profile_receipt,
)
from shadowskillbench.experiments.ollama_nemotron_context32k_profile import (
    NemotronContext32kRoleProfileError,
    load_nemotron_context32k_live_show_identity,
    load_nemotron_context32k_role_profile_receipt,
)
from shadowskillbench.experiments.ollama_nemotron_native_tools_profile import (
    NemotronNativeToolsRoleProfileError,
    load_nemotron_native_tools_role_profile_receipt,
)
from shadowskillbench.experiments.ollama_nemotron_profile import (
    NemotronRoleProfileError,
    load_nemotron_live_show_identity,
    load_nemotron_role_profile_receipt,
)
from shadowskillbench.experiments.planner import PlannedEpisode
from shadowskillbench.experiments.runner import (
    RunCustodyError,
    RunnerError,
    audit_development_runs,
    run_confirmatory_plan,
    run_development_plan,
)
from shadowskillbench.models import ModelAdapterError, ModelClient
from shadowskillbench.models.runtime import (
    ModelRunDescriptor,
    RunDescriptorError,
    client_from_run_descriptor,
    ollama_native_client_from_run_descriptor,
)
from shadowskillbench.protocol import (
    validate_claims_ledger,
    validate_preregistration,
    validate_release_claims,
)
from shadowskillbench.protocol.freeze import ProtocolFreezeError, write_freeze_artifacts
from shadowskillbench.protocol.freeze_v4 import write_freeze_v4_artifacts
from shadowskillbench.protocol.scientific_freeze import (
    ScientificFreezeBinding,
    ScientificFreezeHold,
    derive_scientific_freeze_binding,
)
from shadowskillbench.release import ReproductionHold, reproduce_sealed_artifacts
from shadowskillbench.reporting import (
    ProtocolEvidence,
    ReportEvidence,
    RepresentativeTrace,
    SelectionResult,
    SelectionStatus,
    report_status,
    write_report_artifacts,
)
from shadowskillbench.skills.compiler import CompilerContractError
from shadowskillbench.skills.gate2 import (
    Gate2Error,
    build_gate2_development_report,
    render_gate2_report,
    verify_gate2_report,
)
from shadowskillbench.traces.bundles import generate_bundle

app = typer.Typer(help="ShadowSkillBench command-line interface.", no_args_is_help=True)
world_app = typer.Typer(help="Validate synthetic worlds.", no_args_is_help=True)
traces_app = typer.Typer(help="Generate action traces.", no_args_is_help=True)
skills_app = typer.Typer(help="Compile skills.", no_args_is_help=True)
episodes_app = typer.Typer(help="Run benchmark episodes.", no_args_is_help=True)
authority_app = typer.Typer(help="Validate authority records.", no_args_is_help=True)
protocol_app = typer.Typer(help="Validate and freeze protocol material.", no_args_is_help=True)
corpus_app = typer.Typer(help="Manage benchmark corpora.", no_args_is_help=True)
report_app = typer.Typer(help="Build benchmark reports.", no_args_is_help=True)
experiments_app = typer.Typer(help="Audit experiments.", no_args_is_help=True)
pilot_app = typer.Typer(
    help="Run isolated exploratory live development pilots.", no_args_is_help=True
)

app.add_typer(world_app, name="world")
app.add_typer(traces_app, name="traces")
app.add_typer(skills_app, name="skills")
app.add_typer(episodes_app, name="episodes")
app.add_typer(authority_app, name="authority")
app.add_typer(protocol_app, name="protocol")
app.add_typer(corpus_app, name="corpus")
app.add_typer(report_app, name="report")
app.add_typer(experiments_app, name="experiments")
app.add_typer(pilot_app, name="pilot")
register_confirmatory_preparation_commands(protocol_app, episodes_app)


def _format_violation_codes(violations: Iterable[object]) -> str:
    """Return stable violation identifiers without promoting parser detail to CLI API."""

    codes = sorted({getattr(violation, "code", "UNKNOWN") for violation in violations})
    return ",".join(codes) or "UNKNOWN"


def _protocol_hold(code: str, path: str, violations: Iterable[object]) -> Never:
    typer.echo(f"{code}: {path} [{_format_violation_codes(violations)}]", err=True)
    raise typer.Exit(code=1)


def _validate_protocol_material(repository_root: Path) -> None:
    preregistration_path = repository_root / "protocol/preregistration.md"
    preregistration = validate_preregistration(preregistration_path)
    if not preregistration.valid or preregistration.core is None:
        _protocol_hold(
            "HOLD_PREREGISTRATION_INVALID",
            "protocol/preregistration.md",
            preregistration.violations,
        )
    claims_path = repository_root / "protocol/CLAIMS_LEDGER.yaml"
    claims = validate_claims_ledger(claims_path, repository_root)
    if not claims.valid:
        _protocol_hold(
            "HOLD_CLAIMS_LEDGER_INVALID",
            "protocol/CLAIMS_LEDGER.yaml",
            claims.violations,
        )


def _cli_hold(error: SealedAnalysisHold) -> Never:
    typer.echo(str(error), err=True)
    raise typer.Exit(code=1)


def _safe_artifact_root(path: Path) -> Path:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise RunCustodyError("output path contains a symlink")
        absolute.mkdir(parents=True, exist_ok=True)
        if absolute.is_symlink() or not absolute.is_dir():
            raise RunCustodyError("output directory is unsafe")
    except RunCustodyError:
        raise
    except OSError as error:
        raise RunCustodyError("output directory is unavailable") from error
    return absolute


def _ollama_native_executor_endpoint(compiler_endpoint: str) -> str:
    parsed = urlparse(compiler_endpoint) if type(compiler_endpoint) is str else None
    if (
        parsed is None
        or parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.path != "/v1/chat/completions"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise LivePilotError("Ollama compiler endpoint is not the pinned chat-completions route")
    return urlunparse((parsed.scheme, parsed.netloc, "/api/generate", "", "", ""))


def _write_exclusive(path: Path, data: bytes) -> None:
    """Write an output once without following a user-controlled link."""

    try:
        if path.exists() or path.is_symlink():
            raise SealedAnalysisHold("HOLD_OUTPUT_EXISTS", str(path))
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise SealedAnalysisHold("HOLD_OUTPUT_DIRECTORY", str(path.parent))
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SealedAnalysisHold("HOLD_OUTPUT_EXISTS", str(path)) from error
        finally:
            temporary.unlink(missing_ok=True)
    except SealedAnalysisHold:
        raise
    except OSError as error:
        raise SealedAnalysisHold("HOLD_OUTPUT_IO", str(path)) from error


def _read_analysis_receipt(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()

        def no_duplicates(pairs: list[tuple[object, object]]) -> dict[str, object]:
            parsed: dict[str, object] = {}
            for key, value in pairs:
                if type(key) is not str or key in parsed:
                    raise ValueError("duplicate JSON key")
                parsed[key] = value
            return parsed

        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise SealedAnalysisHold("HOLD_INVALID_ANALYSIS_RECEIPT", str(path)) from error
    if path.is_symlink() or type(value) is not dict or canonical_json_bytes(value) != raw:
        raise SealedAnalysisHold("HOLD_NONCANONICAL_ANALYSIS_RECEIPT", str(path))
    return value


def _authorized_protocol_evidence(root: Path) -> ProtocolEvidence:
    """Derive report protocol fields from the canonical post-freeze authorization."""

    manifest_path = root / "protocol" / "freeze_manifest.json"
    detached_path = root / "protocol" / "freeze_manifest.sha256"
    receipt_path = root / "protocol" / "anchor_receipt.json"
    try:
        if (
            root.is_symlink()
            or not root.is_dir()
            or any(
                path.is_symlink() or not path.is_file()
                for path in (manifest_path, detached_path, receipt_path)
            )
        ):
            raise SealedAnalysisHold("HOLD_MISSING_CONFIRMATORY_AUTHORIZATION", "protocol custody")
        receipt_bytes = receipt_path.read_bytes()

        def no_duplicates(pairs: list[tuple[object, object]]) -> dict[str, object]:
            parsed: dict[str, object] = {}
            for key, value in pairs:
                if type(key) is not str or key in parsed:
                    raise ValueError("duplicate JSON key")
                parsed[key] = value
            return parsed

        receipt = json.loads(receipt_bytes.decode("utf-8"), object_pairs_hook=no_duplicates)
        if type(receipt) is not dict or canonical_json_bytes(receipt) != receipt_bytes:
            raise SealedAnalysisHold("HOLD_NONCANONICAL_CONFIRMATORY_AUTHORIZATION", "receipt")
        authorization = authorize_confirmatory_generation(
            manifest_path.read_bytes(),
            detached_path.read_bytes(),
            receipt_bytes,
            repository_root=root,
        )
        manifest = json.loads(authorization.freeze.manifest_bytes)
    except SealedAnalysisHold:
        raise
    except (ConfirmatoryHoldError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedAnalysisHold(
            "HOLD_INVALID_CONFIRMATORY_AUTHORIZATION", "protocol custody"
        ) from error
    if type(manifest) is not dict or type(manifest.get("inputs")) is not list:
        raise SealedAnalysisHold("HOLD_INVALID_CONFIRMATORY_AUTHORIZATION", "freeze inputs")
    prompt_roles = {"compiler_prompt", "executor_system_prompt"}
    role_aliases = {
        "compiler_prompt": "compiler_prompt",
        "confirmatory_compiler_prompt": "compiler_prompt",
        "executor_system_prompt": "executor_system_prompt",
    }
    prompt_hashes: dict[str, str] = {}
    for entry in manifest["inputs"]:
        if type(entry) is not dict:
            raise SealedAnalysisHold("HOLD_INVALID_CONFIRMATORY_AUTHORIZATION", "freeze input")
        role = entry.get("role")
        digest = entry.get("sha256")
        normalized_role = role_aliases.get(role) if type(role) is str else None
        if normalized_role is not None:
            if type(digest) is not str or normalized_role in prompt_hashes:
                raise SealedAnalysisHold("HOLD_INVALID_CONFIRMATORY_AUTHORIZATION", "prompt inputs")
            prompt_hashes[normalized_role] = digest
    if set(prompt_hashes) != prompt_roles:
        raise SealedAnalysisHold("HOLD_MISSING_CONFIRMATORY_PROMPT_BINDING", "freeze inputs")
    return ProtocolEvidence(
        protocol_tag=authorization.receipt.signed_tag,
        freeze_manifest_sha256=authorization.freeze.manifest_hash,
        prompt_hashes=tuple(prompt_hashes[role] for role in sorted(prompt_roles)),
        code_commit=authorization.receipt.freeze_commit,
        reproduce_command=(
            "uv run shadowskillbench reproduce --protocol protocol/freeze_manifest.json"
        ),
        external_anchor_locator=(
            authorization.receipt.immutable_locator
            if authorization.receipt.externally_verified
            else None
        ),
        custody_locator=authorization.receipt.immutable_locator,
        custody_mode=authorization.receipt.custody_mode,
        anchor_receipt_sha256=authorization.receipt.receipt_hash,
        anchor_verified=authorization.receipt.verification_result in {"VERIFIED", "VERIFIED_LOCAL"},
    )


def _scientific_freeze_binding(root: Path) -> ScientificFreezeBinding:
    """Bind release analysis to exact authorized scientific implementation bytes."""

    manifest_path = root / "protocol" / "freeze_manifest.json"
    detached_path = root / "protocol" / "freeze_manifest.sha256"
    receipt_path = root / "protocol" / "anchor_receipt.json"
    try:
        authorization = authorize_confirmatory_generation(
            manifest_path.read_bytes(),
            detached_path.read_bytes(),
            receipt_path.read_bytes(),
            repository_root=root,
        )
        binding = derive_scientific_freeze_binding(
            manifest_bytes=authorization.freeze.manifest_bytes,
            repository_root=root,
        )
    except ScientificFreezeHold as error:
        raise SealedAnalysisHold(error.code, str(error)) from error
    except (ConfirmatoryHoldError, OSError, ValueError) as error:
        raise SealedAnalysisHold(
            "HOLD_INVALID_CONFIRMATORY_AUTHORIZATION", "protocol custody"
        ) from error
    if binding.manifest_hash != authorization.freeze.manifest_hash:
        raise SealedAnalysisHold("HOLD_SCIENTIFIC_FREEZE_MANIFEST_MISMATCH", "protocol custody")
    return binding


@world_app.command("validate")
def world_validate(all_worlds: bool = typer.Option(False, "--all")) -> None:
    """Validate generated worlds."""
    if not all_worlds:
        typer.echo("HOLD_WORLD_VALIDATION_SCOPE: pass --all", err=True)
        raise typer.Exit(code=1)
    corpus = generate_development_corpus(104729)
    validate_development_corpus(corpus)
    typer.echo(f"PASS_WORLD_VALIDATION: development_cases={len(corpus.cases)}")


@traces_app.command("generate")
def traces_generate(split: str = typer.Option(..., "--split")) -> None:
    """Generate traces for a corpus split."""
    if split != "development":
        typer.echo("HOLD_TRACE_GENERATION_SCOPE: only development is available", err=True)
        raise typer.Exit(code=1)
    bundles = tuple(
        generate_bundle(
            cast(Literal["access_provisioning", "financial_adjustments"], domain),
            ratio,
            12,
            104729 + index,
        )
        for index, (domain, ratio) in enumerate(
            (domain, ratio)
            for domain in ("access_provisioning", "financial_adjustments")
            for ratio in (
                Decimal("0"),
                Decimal("0.25"),
                Decimal("0.5"),
                Decimal("0.75"),
                Decimal("1"),
            )
        )
    )
    typer.echo(
        "PRACTICE_NOT_EVIDENCE: "
        f"generated_development_bundles={len(bundles)} persistence=not_configured"
    )


@skills_app.command("compile")
def skills_compile(
    split: str = typer.Option(..., "--split"),
    output: Path = typer.Option(
        Path("artifacts/development/gate2-development-report.json"), "--output"
    ),
    prompt: Path = typer.Option(Path("prompts/skill_compiler.md"), "--prompt"),
) -> None:
    """Compile skills for a corpus split."""
    if split != "development":
        typer.echo("HOLD_SKILL_COMPILATION_SCOPE: only development is available", err=True)
        raise typer.Exit(code=1)
    try:
        report = asyncio.run(build_gate2_development_report(prompt_bytes=prompt.read_bytes()))
        verify_gate2_report(report)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_exclusive(output, render_gate2_report(report).encode("utf-8"))
    except (Gate2Error, OSError, SealedAnalysisHold) as error:
        typer.echo(f"HOLD_DEVELOPMENT_SKILL_COMPILATION: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"PASS_DEVELOPMENT_SKILL_COMPILATION: output={output}")


@episodes_app.command("run")
def episodes_run(
    stage_a: bool = typer.Option(False, "--stage-a"),
    stage_b: bool = typer.Option(False, "--stage-b"),
    split: str = typer.Option(..., "--split"),
    limit: int | None = typer.Option(None, "--limit"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Run development episodes."""
    if split != "development" or stage_a == stage_b:
        typer.echo("HOLD_DEVELOPMENT_EPISODE_SCOPE: select one development stage", err=True)
        raise typer.Exit(code=1)
    if limit is not None and (type(limit) is not int or limit < 1):
        typer.echo("HOLD_DEVELOPMENT_EPISODE_LIMIT: limit must be positive", err=True)
        raise typer.Exit(code=1)
    if limit is None:
        typer.echo(
            "HOLD_DEVELOPMENT_EPISODE_COMPOSITION: pass an explicit bounded --limit",
            err=True,
        )
        raise typer.Exit(code=1)
    corpus = generate_development_corpus(4242)
    plan = build_development_plan(corpus)
    selected = tuple(
        episode
        for episode in plan.episodes
        if episode.stage == ("stage_a" if stage_a else "stage_b")
    )
    selected = selected[:limit]
    stage_name = "stage_a" if stage_a else "stage_b"
    run_directory = (
        output_dir
        if output_dir is not None
        else Path("artifacts/experiments/development") / stage_name / f"limit-{limit}"
    )
    try:
        execution_plans = materialize_development_episodes(corpus, selected)
        run_directory = _safe_artifact_root(run_directory)
        previous_cwd = Path.cwd()
        try:
            os.chdir(run_directory)
            summary = asyncio.run(
                run_development_plan(
                    selected,
                    execution_plans=execution_plans,
                    run_directory=run_directory,
                )
            )
        finally:
            os.chdir(previous_cwd)
    except (RunCustodyError, RunnerError, ValueError, OSError) as error:
        typer.echo(f"HOLD_DEVELOPMENT_EPISODE_EXECUTION: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "PRACTICE_NOT_EVIDENCE: "
        f"development_plan_cells={len(selected)} total_matrix_cells={len(plan.episodes)} "
        f"completed={len(summary.completed_episode_ids)} "
        f"skipped={len(summary.skipped_episode_ids)} "
        f"run_directory={run_directory}"
    )


@pilot_app.command("live-run")
def pilot_live_run(
    endpoint: str = typer.Option(..., "--endpoint"),
    api_key_environment: str = typer.Option(..., "--api-key-environment"),
    role_profile_receipt: Path = typer.Option(..., "--role-profile-receipt"),
    limit: int = typer.Option(..., "--limit"),
    output_dir: Path = typer.Option(Path("artifacts/exploratory-live-pilot"), "--output-dir"),
    protocol_path: Path = typer.Option(
        Path("protocol/EXPLORATORY_LIVE_PILOT.md"), "--pilot-protocol"
    ),
    prompt: Path = typer.Option(Path("prompts/skill_compiler.md"), "--prompt"),
) -> None:
    """Run only the pinned gpt-oss role-faithful exploratory loopback pilot."""

    try:
        receipt = load_ollama_role_profile_receipt(role_profile_receipt)
        executor_endpoint = _ollama_native_executor_endpoint(endpoint)
        descriptor = LivePilotDescriptor(
            endpoint=endpoint,
            api_key_environment=api_key_environment,
            provider=OLLAMA_GPT_OSS_PROVIDER,
            model=OLLAMA_GPT_OSS_MODEL,
            model_version=cast(str, receipt["profile_model_digest"]),
            ollama_server_version=OLLAMA_GPT_OSS_SERVER_VERSION,
            base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
            reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
            executor_reasoning_effort=OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
            sampling_temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            executor_sampling_temperature=OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
            executor_endpoint=executor_endpoint,
            executor_transport=OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
            executor_request_profile=OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
            executor_request_profile_hash=OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
            executor_think=False,
            executor_model_name=OLLAMA_GPT_OSS_EXECUTOR_MODEL,
            executor_model_version=cast(str, receipt["executor_model_digest"]),
            executor_profile=OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
            executor_modelfile_sha256=ollama_gpt_oss_executor_modelfile_hash(),
            executor_template_sha256=ollama_gpt_oss_executor_template_hash(),
            compiler_max_tokens=OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
            declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
            provider_profile=OLLAMA_GPT_OSS_PROFILE,
            role_profile_receipt_hash=sha256_ref(receipt),
        )
        compiler_client = client_from_run_descriptor(
            ModelRunDescriptor(
                provider=descriptor.provider,
                model=descriptor.model,
                model_version=descriptor.model_version,
                endpoint=endpoint,
                api_key_environment=api_key_environment,
                max_attempts=1,
            ),
            trust_env=False,
            reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
            timeout_seconds=900.0,
        )
        executor_client = ollama_native_client_from_run_descriptor(
            ModelRunDescriptor(
                provider=descriptor.provider,
                model=cast(str, descriptor.executor_model_name),
                model_version=cast(str, descriptor.executor_model_version),
                endpoint=executor_endpoint,
                api_key_environment=api_key_environment,
                max_attempts=1,
            ),
            trust_env=False,
            timeout_seconds=900.0,
            top_p=OLLAMA_GPT_OSS_SAMPLING_TOP_P,
            context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
        )
        summary = asyncio.run(
            run_live_pilot(
                output_dir=output_dir,
                protocol_path=protocol_path,
                prompt_path=prompt,
                descriptor=descriptor,
                compiler_client=compiler_client,
                executor_client=executor_client,
                limit=limit,
                role_profile_receipt=receipt,
            )
        )
    except ModelAdapterError as error:
        detail = f"HOLD_EXPLORATORY_LIVE_PILOT: {error.code}"
        if error.finish_reason is not None:
            detail += f" finish_reason={error.finish_reason}"
        if error.reported_usage is not None:
            detail += (
                f" input_tokens={error.reported_usage.input_tokens}"
                f" output_tokens={error.reported_usage.output_tokens}"
            )
        typer.echo(detail, err=True)
        raise typer.Exit(code=1) from None
    except CompilerContractError as error:
        detail = f"HOLD_EXPLORATORY_LIVE_PILOT: {error.code}"
        if error.stage is not None:
            detail += f" stage={error.stage} validation_code={error.validation_code}"
        if error.finish_reason is not None:
            detail += f" finish_reason={error.finish_reason}"
        if error.reported_usage is not None:
            detail += (
                f" input_tokens={error.reported_usage.input_tokens}"
                f" output_tokens={error.reported_usage.output_tokens}"
            )
        typer.echo(detail, err=True)
        raise typer.Exit(code=1) from None
    except (
        LivePilotError,
        OllamaRoleProfileError,
        RunDescriptorError,
        OSError,
        ValueError,
    ) as error:
        typer.echo(f"HOLD_EXPLORATORY_LIVE_PILOT: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"{PILOT_LABEL}: completed={summary['completed']} skipped={summary['skipped']} "
        f"limit={summary['limit']} total_cells={summary['total_cells']} "
        f"output_dir={output_dir}"
    )


@pilot_app.command("capability-run")
def pilot_capability_run(
    endpoint: str = typer.Option(..., "--endpoint"),
    api_key_environment: str = typer.Option(..., "--api-key-environment"),
    role_profile_receipt: Path = typer.Option(..., "--role-profile-receipt"),
    model_identity_receipt: Path | None = typer.Option(None, "--model-identity-receipt"),
    phase: str = typer.Option(..., "--phase"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    screen_receipt: Path | None = typer.Option(None, "--screen-receipt"),
    bundle: str = typer.Option("gpt-oss", "--bundle"),
) -> None:
    """Run the bounded development-only executor capability gate once, without retries."""

    identity_receipt: dict[str, object] | None = None
    try:
        if is_glm47_native_tools_response_v2_capability_bundle(bundle):
            if model_identity_receipt is None:
                raise CapabilityPilotHold("HOLD_CAPABILITY_IDENTITY_RECEIPT: required")
            receipt = load_glm47_native_tools_response_v2_receipt(role_profile_receipt)
            identity_receipt = load_glm47_native_tools_response_v2_identity(model_identity_receipt)
            client = build_glm47_native_tools_response_v2_capability_client(
                endpoint, api_key_environment, role_profile_receipt
            )
        elif is_glm47_native_tools_capability_bundle(bundle):
            if model_identity_receipt is None:
                raise CapabilityPilotHold("HOLD_CAPABILITY_IDENTITY_RECEIPT: required")
            receipt = load_glm47_native_tools_role_profile_receipt(role_profile_receipt)
            identity_receipt = load_glm47_native_tools_live_show_identity(model_identity_receipt)
            client = build_glm47_native_tools_context32k_capability_client(
                endpoint, api_key_environment, role_profile_receipt
            )
        elif is_gemma4_native_tools_capability_bundle(bundle):
            if model_identity_receipt is None:
                raise CapabilityPilotHold("HOLD_CAPABILITY_IDENTITY_RECEIPT: required")
            receipt = load_gemma4_native_tools_role_profile_receipt(role_profile_receipt)
            identity_receipt = load_gemma4_native_tools_identity(model_identity_receipt)
            client = (
                build_gemma4_native_tools_context32768_v4_capability_client(
                    endpoint, api_key_environment, role_profile_receipt
                )
                if is_gemma4_native_tools_v4_capability_bundle(bundle)
                else build_gemma4_native_tools_context32768_v3_capability_client(
                    endpoint, api_key_environment, role_profile_receipt
                )
                if is_gemma4_native_tools_v3_capability_bundle(bundle)
                else build_gemma4_native_tools_context32768_v2_capability_client(
                    endpoint, api_key_environment, role_profile_receipt
                )
                if is_gemma4_native_tools_v2_capability_bundle(bundle)
                else build_gemma4_native_tools_context32768_capability_client(
                    endpoint, api_key_environment, role_profile_receipt
                )
            )
        elif is_nemotron_native_tools_capability_bundle(bundle):
            if model_identity_receipt is None:
                raise CapabilityPilotHold("HOLD_CAPABILITY_IDENTITY_RECEIPT: required")
            receipt = load_nemotron_native_tools_role_profile_receipt(role_profile_receipt)
            identity_receipt = load_nemotron_context32k_live_show_identity(model_identity_receipt)
            client = build_nemotron_native_tools_context32k_capability_client(
                endpoint, api_key_environment, role_profile_receipt
            )
        elif is_nemotron_context32k_capability_bundle(bundle):
            if model_identity_receipt is None:
                raise CapabilityPilotHold("HOLD_CAPABILITY_IDENTITY_RECEIPT: required")
            receipt = load_nemotron_context32k_role_profile_receipt(role_profile_receipt)
            identity_receipt = load_nemotron_context32k_live_show_identity(model_identity_receipt)
            client = build_nemotron_context32k_capability_client(
                endpoint, api_key_environment, role_profile_receipt
            )
        elif is_nemotron_capability_bundle(bundle):
            if model_identity_receipt is None:
                raise CapabilityPilotHold("HOLD_CAPABILITY_IDENTITY_RECEIPT: required")
            receipt = load_nemotron_role_profile_receipt(role_profile_receipt)
            identity_receipt = load_nemotron_live_show_identity(model_identity_receipt)
            client = build_nemotron_capability_client(
                endpoint, api_key_environment, role_profile_receipt
            )
        elif bundle == "mistral":
            receipt = load_mistral_role_profile_receipt(role_profile_receipt)
            client = build_mistral_capability_client(
                endpoint, api_key_environment, role_profile_receipt
            )
        elif bundle == "gemma4-indexed-greedy":
            receipt = load_gemma4_greedy_role_profile_receipt(role_profile_receipt)
            client = build_gemma4_greedy_capability_client(
                endpoint, api_key_environment, role_profile_receipt
            )
        elif bundle in {"gemma4", "gemma4-indexed"}:
            receipt = load_gemma4_role_profile_receipt(role_profile_receipt)
            client = build_gemma4_capability_client(
                endpoint, api_key_environment, role_profile_receipt
            )
        elif bundle == "gpt-oss":
            receipt = load_ollama_role_profile_receipt(role_profile_receipt)
            client = build_native_capability_client(
                endpoint, api_key_environment, role_profile_receipt
            )
        else:
            raise CapabilityPilotHold("HOLD_CAPABILITY_BUNDLE: invalid")
        summary = asyncio.run(
            run_capability_pilot(
                output_dir=output_dir,
                endpoint=endpoint,
                api_key_environment=api_key_environment,
                role_receipt=receipt,
                client=client,
                phase=cast(Literal["screen", "validation"], phase),
                bundle=cast(
                    Literal[
                        "gpt-oss",
                        "gemma4",
                        "gemma4-indexed",
                        "gemma4-indexed-greedy",
                        "mistral",
                        "nemotron",
                        "nemotron-indexed",
                        "nemotron-indexed-context32k",
                        "nemotron-native-tools-context32k",
                        "glm47-native-tools-context32k",
                        "glm47-native-tools-context32768-roles2",
                        "gemma4-native-tools-context32768",
                        "gemma4-native-tools-context32768-v2",
                        "gemma4-native-tools-context32768-v3",
                        "gemma4-native-tools-context32768-v4",
                    ],
                    bundle,
                ),
                screen_receipt_path=screen_receipt,
                model_identity_receipt=(
                    identity_receipt
                    if is_nemotron_capability_bundle(bundle)
                    or is_glm47_native_tools_capability_bundle(bundle)
                    or is_glm47_native_tools_response_v2_capability_bundle(bundle)
                    or is_gemma4_native_tools_capability_bundle(bundle)
                    else None
                ),
            )
        )
    except (
        CapabilityPilotHold,
        Gemma4RoleProfileError,
        Gemma4GreedyRoleProfileError,
        MistralRoleProfileError,
        NemotronContext32kRoleProfileError,
        NemotronNativeToolsRoleProfileError,
        Glm47NativeToolsRoleProfileError,
        Glm47NativeToolsResponseV2Error,
        Gemma4NativeToolsProfileError,
        NemotronRoleProfileError,
        OllamaRoleProfileError,
        RunDescriptorError,
        OSError,
        ValueError,
    ) as error:
        typer.echo(f"HOLD_DEVELOPMENT_CAPABILITY_PILOT: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "PASS_DEVELOPMENT_CAPABILITY_PILOT: "
        f"phase={summary['phase']} completed={summary['completed']} skipped={summary['skipped']} "
        f"total={summary['total']} "
        f"output_dir={output_dir}"
    )


@pilot_app.command("capability-audit")
def pilot_capability_audit(
    phase: str = typer.Option(..., "--phase"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    output: Path | None = typer.Option(None, "--output"),
    bundle: str = typer.Option("gpt-oss", "--bundle"),
) -> None:
    """Offline audit of the development-only capability gate; never retries or repairs."""

    try:
        receipt = write_capability_audit(
            output_dir,
            phase=cast(Literal["screen", "validation"], phase),
            receipt_path=output,
            bundle=cast(
                Literal[
                    "gpt-oss",
                    "gemma4",
                    "gemma4-indexed",
                    "gemma4-indexed-greedy",
                    "mistral",
                    "nemotron",
                    "nemotron-indexed",
                    "nemotron-indexed-context32k",
                    "nemotron-native-tools-context32k",
                    "glm47-native-tools-context32k",
                    "glm47-native-tools-context32768-roles2",
                    "gemma4-native-tools-context32768",
                    "gemma4-native-tools-context32768-v2",
                    "gemma4-native-tools-context32768-v3",
                    "gemma4-native-tools-context32768-v4",
                ],
                bundle,
            ),
        )
    except (CapabilityPilotHold, OSError, ValueError) as error:
        typer.echo(f"HOLD_DEVELOPMENT_CAPABILITY_AUDIT: {error}", err=True)
        raise typer.Exit(code=1) from error
    if receipt["status"] != "PASS":
        findings = ",".join(cast(list[str], receipt["findings"]))
        typer.echo(
            f"HOLD_DEVELOPMENT_CAPABILITY_AUDIT: receipt_sha256={receipt['receipt_hash']} "
            f"findings={findings}",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(
        "PASS_DEVELOPMENT_CAPABILITY_AUDIT: "
        f"receipt_sha256={receipt['receipt_hash']} output_dir={output_dir}"
    )


@pilot_app.command("recovery-freeze")
def pilot_recovery_freeze(
    candidate: str = typer.Option(..., "--candidate"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Freeze one predeclared bounded Gemma recovery candidate before any model call."""

    try:
        commitment = write_recovery_commitment(
            output, cast(Literal["primary-26b", "fallback-12b"], candidate)
        )
    except (Gemma4RecoveryHold, OSError, ValueError) as error:
        typer.echo(f"HOLD_GEMMA4_RECOVERY_FREEZE: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "PASS_GEMMA4_RECOVERY_FREEZE: "
        f"candidate={commitment['candidate']} commitment_sha256={commitment['commitment_hash']}"
    )


@pilot_app.command("recovery-run")
def pilot_recovery_run(
    candidate: str = typer.Option(..., "--candidate"),
    phase: str = typer.Option(..., "--phase"),
    endpoint: str = typer.Option(..., "--endpoint"),
    api_key_environment: str = typer.Option(..., "--api-key-environment"),
    commitment: Path = typer.Option(..., "--commitment"),
    role_profile_receipt: Path = typer.Option(..., "--role-profile-receipt"),
    model_identity_receipt: Path = typer.Option(..., "--model-identity-receipt"),
    identity_show: Path = typer.Option(..., "--identity-show"),
    identity_tags: Path = typer.Option(..., "--identity-tags"),
    identity_version: Path = typer.Option(..., "--identity-version"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    screen_audit: Path | None = typer.Option(None, "--screen-audit"),
    primary_terminal_audit: Path | None = typer.Option(None, "--primary-terminal-audit"),
    primary_role_failure: Path | None = typer.Option(None, "--primary-role-failure"),
) -> None:
    """Run one frozen recovery phase once; validation requires an audited screen pass."""

    if candidate not in {"primary-26b", "fallback-12b"} or phase not in {"screen", "validation"}:
        typer.echo("HOLD_GEMMA4_RECOVERY_RUN: invalid candidate or phase", err=True)
        raise typer.Exit(code=1)
    try:
        summary = asyncio.run(
            run_recovery(
                output_dir=output_dir,
                commitment_path=commitment,
                role_receipt_path=role_profile_receipt,
                identity_receipt_path=model_identity_receipt,
                endpoint=endpoint,
                api_key_environment=api_key_environment,
                phase=cast(Literal["screen", "validation"], phase),
                requested_candidate=cast(Literal["primary-26b", "fallback-12b"], candidate),
                identity_show_path=identity_show,
                identity_tags_path=identity_tags,
                identity_version_path=identity_version,
                screen_audit_path=screen_audit,
                primary_terminal_audit_path=primary_terminal_audit,
                primary_role_failure_path=primary_role_failure,
            )
        )
    except (Gemma4RecoveryHold, OSError, ValueError) as error:
        typer.echo(f"HOLD_GEMMA4_RECOVERY_RUN: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "RECOVERY_RUN_RECORDED: "
        f"candidate={summary['candidate']} phase={summary['phase']} "
        f"completed={summary['completed']} skipped={summary['skipped']} total={summary['total']} "
        f"phase_wall_seconds={summary['phase_wall_seconds']}"
    )


@pilot_app.command("recovery-audit")
def pilot_recovery_audit(
    phase: str = typer.Option(..., "--phase"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    commitment: Path = typer.Option(..., "--commitment"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Audit one bounded recovery phase without retries or repair."""

    if phase not in {"screen", "validation"}:
        typer.echo("HOLD_GEMMA4_RECOVERY_AUDIT: invalid phase", err=True)
        raise typer.Exit(code=1)
    try:
        receipt = write_recovery_audit(
            output_dir,
            commitment_path=commitment,
            phase=cast(Literal["screen", "validation"], phase),
            output=output,
        )
    except (Gemma4RecoveryHold, OSError, ValueError) as error:
        typer.echo(f"HOLD_GEMMA4_RECOVERY_AUDIT: {error}", err=True)
        raise typer.Exit(code=1) from error
    message = (
        "PASS_GEMMA4_RECOVERY_AUDIT"
        if cast(str, receipt["status"]).startswith("PASS_")
        else "HOLD_GEMMA4_RECOVERY_AUDIT"
    )
    typer.echo(f"{message}: status={receipt['status']} receipt_sha256={receipt['receipt_hash']}")
    if message.startswith("HOLD_"):
        raise typer.Exit(code=1)


@pilot_app.command("audit")
def pilot_audit(
    output_dir: Path = typer.Option(Path("artifacts/exploratory-live-pilot"), "--output-dir"),
) -> None:
    """Report raw exploratory task-completion and CuP counts only."""

    try:
        summary = audit_live_pilot(output_dir)
    except (LivePilotError, OSError, ValueError) as error:
        typer.echo(f"HOLD_EXPLORATORY_LIVE_PILOT_AUDIT: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@episodes_app.command("build-confirmatory-package")
def episodes_build_confirmatory_package(
    source_dir: Path = typer.Option(..., "--source-dir"),
    package_dir: Path = typer.Option(
        Path("artifacts/experiments/confirmatory-package"), "--package-dir"
    ),
    manifest_path: Path = typer.Option(Path("protocol/freeze_manifest.json"), "--manifest"),
    detached_sha256_path: Path = typer.Option(
        Path("protocol/freeze_manifest.sha256"), "--manifest-sha256"
    ),
    anchor_receipt_path: Path = typer.Option(
        Path("protocol/anchor_receipt.json"), "--anchor-receipt"
    ),
) -> None:
    """Seal explicit frozen execution inputs after custody admission."""

    try:
        authorization = authorize_confirmatory_generation(
            manifest_path.read_bytes(),
            detached_sha256_path.read_bytes(),
            anchor_receipt_path.read_bytes(),
            repository_root=Path.cwd(),
        )
    except (ConfirmatoryHoldError, OSError, ValueError) as error:
        typer.echo(f"HOLD_CONFIRMATORY_PACKAGE_AUTHORIZATION: {error}", err=True)
        raise typer.Exit(code=1) from error
    try:
        inputs = load_confirmatory_execution_package_inputs(source_dir, authorization=authorization)
        package = write_confirmatory_execution_package(inputs, package_dir)
    except (ConfirmatoryPackageHold, OSError, ValueError) as error:
        typer.echo(f"HOLD_CONFIRMATORY_PACKAGE_BUILD: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "PASS_CONFIRMATORY_PACKAGE_BUILD: "
        f"package_hash={package.package_hash} plan_hash={package.plan.plan_hash} "
        f"package_dir={package_dir}"
    )


@episodes_app.command("run-confirmatory")
def episodes_run_confirmatory(
    stage_a: bool = typer.Option(False, "--stage-a"),
    stage_b: bool = typer.Option(False, "--stage-b"),
    package_dir: Path = typer.Option(
        Path("artifacts/experiments/confirmatory-package"), "--package-dir"
    ),
    output_dir: Path = typer.Option(Path("artifacts/experiments/confirmatory"), "--output-dir"),
    endpoint: str | None = typer.Option(None, "--endpoint"),
    api_key_environment: str | None = typer.Option(None, "--api-key-environment"),
    manifest_path: Path = typer.Option(Path("protocol/freeze_manifest.json"), "--manifest"),
    detached_sha256_path: Path = typer.Option(
        Path("protocol/freeze_manifest.sha256"), "--manifest-sha256"
    ),
    anchor_receipt_path: Path = typer.Option(
        Path("protocol/anchor_receipt.json"), "--anchor-receipt"
    ),
) -> None:
    """Run confirmatory episodes after custody admission."""
    if stage_a == stage_b:
        typer.echo("HOLD_CONFIRMATORY_EPISODE_SCOPE: select exactly one stage", err=True)
        raise typer.Exit(code=1)
    try:
        authorization = authorize_confirmatory_generation(
            manifest_path.read_bytes(),
            detached_sha256_path.read_bytes(),
            anchor_receipt_path.read_bytes(),
            repository_root=Path.cwd(),
        )
    except (ConfirmatoryHoldError, OSError, ValueError) as error:
        typer.echo(f"HOLD_CONFIRMATORY_EXECUTION_AUTHORIZATION: {error}", err=True)
        raise typer.Exit(code=1) from error
    try:
        package = load_confirmatory_execution_package(package_dir, authorization=authorization)
        selected_stage = EpisodeStage.CONFIRMATORY_A if stage_a else EpisodeStage.CONFIRMATORY_B
        selected_ids = frozenset(
            episode.episode_id
            for episode in package.plan.episodes
            if episode.stage is selected_stage
        )
        if not selected_ids:
            raise ConfirmatoryPackageHold("selected stage has no planned episodes")
        if endpoint is None or api_key_environment is None:
            raise ConfirmatoryPackageHold(
                "operator must explicitly supply --endpoint and --api-key-environment"
            )
        output_dir = _safe_artifact_root(output_dir)
        packaged = package.run_descriptor
        if endpoint != packaged.endpoint:
            raise ConfirmatoryPackageHold("operator endpoint does not match the frozen descriptor")
        operator_descriptor = ModelRunDescriptor(
            provider=packaged.provider,
            model=packaged.model,
            model_version=packaged.model_version,
            endpoint=endpoint,
            api_key_environment=api_key_environment,
            max_attempts=packaged.max_attempts,
            supports_system_role=packaged.supports_system_role,
            supports_developer_role=packaged.supports_developer_role,
            supports_seed=packaged.supports_seed,
            supports_structured_output=packaged.supports_structured_output,
            pricing=packaged.pricing,
            executor_runtime_profile=packaged.executor_runtime_profile,
        )
        executor_profile = (
            None
            if packaged.executor_runtime_profile is None
            else confirmatory_executor_runtime_profile_projection(packaged.executor_runtime_profile)
        )
        runtime_binding: dict[str, object] = {
            "profile": "SSB-RUNTIME-BINDING2"
            if executor_profile is not None
            else "SSB-RUNTIME-BINDING1",
            "plan_hash": package.plan.plan_hash,
            "package_hash": package.package_hash,
            "run_descriptor_hash": package.package_manifest["run_descriptor_hash"],
            "freeze_manifest_hash": package.freeze_manifest_hash,
            "anchor_receipt_hash": package.anchor_receipt_hash,
            "operator_endpoint_hash": sha256_ref(endpoint),
            "operator_api_key_environment": api_key_environment,
            "provider": operator_descriptor.provider,
            "model": operator_descriptor.model,
            "model_version": operator_descriptor.model_version,
        }
        if executor_profile is None:
            client = client_from_run_descriptor(operator_descriptor)
            executor_temperature = 0.0
        else:
            parsed_endpoint = urlparse(endpoint)
            if parsed_endpoint.path != "/api/generate":
                raise ConfirmatoryPackageHold("native executor endpoint must be /api/generate")
            runtime_binding["executor_runtime_profile_hash"] = sha256_ref(executor_profile)
            client = ollama_native_client_from_run_descriptor(
                operator_descriptor,
                timeout_seconds=float(cast(int, executor_profile["timeout_seconds"])),
                top_p=cast(float, executor_profile["top_p"]),
                context_length=cast(int, executor_profile["context_length"]),
            )

            executor_temperature = cast(float, executor_profile["temperature"])

        async def execute_function(plan: EpisodePlan, executor_client: ModelClient):
            if executor_profile is not None and executor_profile["prompt_profile"] in {
                LEGACY_CONFIRMATORY_TURN_PROMPT_PROFILE,
                LEGACY_V2_CONFIRMATORY_TURN_PROMPT_PROFILE,
                CONFIRMATORY_TURN_PROMPT_PROFILE,
            }:
                return await run_episode(
                    plan,
                    executor_client,
                    output_schema=ConfirmatoryAgentTurnWire,
                    temperature=executor_temperature,
                )
            return await run_episode(plan, executor_client, temperature=executor_temperature)

        def client_factory(episode: PlannedEpisode):
            del episode
            return client

        previous_cwd = Path.cwd()
        try:
            os.chdir(output_dir)
            summary = asyncio.run(
                run_confirmatory_plan(
                    package.plan,
                    run_directory=output_dir,
                    materialize=package.materialize,
                    client_factory=client_factory,
                    catalog=package.catalog,
                    exclusion_rule=package.exclusion_rule,
                    runtime_binding=runtime_binding,
                    package_manifest=dict(package.package_manifest),
                    dispatch_episode_ids=selected_ids,
                    execute=execute_function,
                )
            )
        finally:
            os.chdir(previous_cwd)
    except (
        ConfirmatoryPackageHold,
        RunDescriptorError,
        RunCustodyError,
        RunnerError,
        ValueError,
        OSError,
    ) as error:
        typer.echo(f"HOLD_CONFIRMATORY_EPISODE_EXECUTION: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "PASS_CONFIRMATORY_EPISODE_EXECUTION: "
        f"stage={selected_stage.value} completed={len(summary.completed_episode_ids)} "
        f"skipped={len(summary.skipped_episode_ids)} run_directory={output_dir} "
        "connection=operator_supplied"
    )


@episodes_app.command("build-confirmatory-package-v4")
def episodes_build_confirmatory_package_v4(
    package_dir: Path = typer.Option(
        Path("artifacts/experiments/confirmatory-v4-package"), "--package-dir"
    ),
    manifest_path: Path = typer.Option(Path("protocol/freeze_manifest.v4.json"), "--manifest"),
    detached_sha256_path: Path = typer.Option(
        Path("protocol/freeze_manifest.v4.sha256"), "--manifest-sha256"
    ),
    anchor_receipt_path: Path = typer.Option(
        Path("protocol/anchor_receipt.v4.json"), "--anchor-receipt"
    ),
) -> None:
    """Reserve package construction for the caller-supplied, verified V4 bindings API."""

    del package_dir, manifest_path, detached_sha256_path, anchor_receipt_path
    typer.echo(
        "HOLD_CONFIRMATORY_V4_PACKAGE_NOT_IMPLEMENTED: "
        "CLI cannot fabricate compiled skill, held-out case, condition, or runtime bindings",
        err=True,
    )
    raise typer.Exit(code=1)


@episodes_app.command("run-confirmatory-v4")
def episodes_run_confirmatory_v4(
    package_dir: Path = typer.Option(
        Path("artifacts/experiments/confirmatory-v4-package"), "--package-dir"
    ),
    manifest_path: Path = typer.Option(Path("protocol/freeze_manifest.v4.json"), "--manifest"),
    detached_sha256_path: Path = typer.Option(
        Path("protocol/freeze_manifest.v4.sha256"), "--manifest-sha256"
    ),
    anchor_receipt_path: Path = typer.Option(
        Path("protocol/anchor_receipt.v4.json"), "--anchor-receipt"
    ),
) -> None:
    """Do not report V4 execution until the CLI can load and dispatch verified cells."""

    del package_dir, manifest_path, detached_sha256_path, anchor_receipt_path
    typer.echo(
        "HOLD_CONFIRMATORY_V4_EXECUTION_NOT_IMPLEMENTED: "
        "use the descriptor-pinned library route with caller-supplied materialized cells",
        err=True,
    )
    raise typer.Exit(code=1)


@authority_app.command("validate")
def authority_validate() -> None:
    """Validate authority records."""
    corpus = generate_development_corpus(104729)
    validate_development_corpus(corpus)
    typer.echo(f"PASS_AUTHORITY_VALIDATION: development_cases={len(corpus.cases)}")


@protocol_app.command("validate")
def protocol_validate() -> None:
    """Validate protocol material."""
    _validate_protocol_material(Path.cwd())
    typer.echo("PASS_PROTOCOL_VALIDATION")


@protocol_app.command("freeze")
def protocol_freeze() -> None:
    """Prepare a protocol freeze for a post-commit custody receipt."""
    repository_root = Path.cwd()
    _validate_protocol_material(repository_root)
    try:
        artifacts = write_freeze_artifacts(repository_root)
    except ProtocolFreezeError as error:
        _protocol_hold("HOLD_PROTOCOL_FREEZE", "protocol", error.violations)
    typer.echo(
        "PASS_PROTOCOL_FREEZE: "
        f"manifest_sha256={artifacts.manifest.sha256} status=HOLD_PENDING_CUSTODY_RECEIPT"
    )


@protocol_app.command("freeze-v4")
def protocol_freeze_v4() -> None:
    """Write the V4 scientific prefreeze; this never authorizes execution."""

    try:
        artifacts = write_freeze_v4_artifacts(Path.cwd())
    except (OSError, ValueError) as error:
        typer.echo(f"HOLD_PROTOCOL_FREEZE_V4: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "PASS_PROTOCOL_SCIENTIFIC_PREFREEZE_V4: "
        f"manifest_sha256={artifacts.manifest_hash} "
        "status=HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY"
    )


@corpus_app.command("generate-confirmatory")
def corpus_generate_confirmatory(
    corpus_seed: int = typer.Option(CORPUS_SEED, "--corpus-seed"),
    development_seed: int = typer.Option(4242, "--development-seed"),
    manifest_path: Path = typer.Option(Path("protocol/freeze_manifest.json"), "--manifest"),
    detached_sha256_path: Path = typer.Option(
        Path("protocol/freeze_manifest.sha256"), "--manifest-sha256"
    ),
    anchor_receipt_path: Path = typer.Option(
        Path("protocol/anchor_receipt.json"), "--anchor-receipt"
    ),
    output_dir: Path = typer.Option(Path("artifacts/corpus/confirmatory"), "--output-dir"),
) -> None:
    """Generate the confirmatory corpus after custody validation."""
    try:
        if corpus_seed != CORPUS_SEED:
            raise ConfirmatoryHoldError(f"corpus seed must be frozen value {CORPUS_SEED}")
        if development_seed != DEVELOPMENT_CORPUS_SEED:
            raise ConfirmatoryHoldError(
                f"development seed must be frozen value {DEVELOPMENT_CORPUS_SEED}"
            )
        development = generate_development_corpus(development_seed)
        corpus = generate_confirmatory_from_files(
            manifest_path=manifest_path,
            detached_sha256_path=detached_sha256_path,
            anchor_receipt_path=anchor_receipt_path,
            repository_root=Path.cwd(),
            corpus_seed=corpus_seed,
            development_inventory=inventory_from_development(development),
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        artifacts = write_confirmatory_artifacts(corpus, output_dir)
    except (ConfirmatoryHoldError, OSError) as error:
        typer.echo(f"HOLD_CONFIRMATORY_CORPUS_GENERATION: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "PASS_CONFIRMATORY_CORPUS_GENERATION: "
        f"content_sha256={corpus.content_hash} public_corpus={artifacts.public_corpus}"
    )


@corpus_app.command("generate-confirmatory-v4")
def corpus_generate_confirmatory_v4(
    manifest_path: Path = typer.Option(Path("protocol/freeze_manifest.v4.json"), "--manifest"),
    detached_sha256_path: Path = typer.Option(
        Path("protocol/freeze_manifest.v4.sha256"), "--manifest-sha256"
    ),
    anchor_receipt_path: Path = typer.Option(
        Path("protocol/anchor_receipt.v4.json"), "--anchor-receipt"
    ),
    output: Path = typer.Option(
        Path("artifacts/corpus/confirmatory-v4/corpus-projection.v4.json"), "--output"
    ),
) -> None:
    """Fail closed until fresh Nemotron validation and full V4 custody exist."""

    del manifest_path, detached_sha256_path, anchor_receipt_path, output
    typer.echo(
        "HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY: "
        "V4 confirmatory corpus generation is not authorized",
        err=True,
    )
    raise typer.Exit(code=1)


@corpus_app.command("audit")
def corpus_audit(
    split: str = typer.Option(..., "--split"),
    public_corpus: Path = typer.Option(
        Path("artifacts/corpus/confirmatory/confirmatory_public_corpus.json"), "--public-corpus"
    ),
) -> None:
    """Audit corpus custody and composition."""
    if split != "confirmatory":
        typer.echo("HOLD_CORPUS_AUDIT_SCOPE: only confirmatory is available", err=True)
        raise typer.Exit(code=1)
    try:
        payload = load_model_visible_corpus(public_corpus)
    except ConfirmatoryHoldError as error:
        typer.echo(f"HOLD_CONFIRMATORY_CORPUS_AUDIT: {error}", err=True)
        raise typer.Exit(code=1) from error
    stage_a_cases = payload["stage_a_cases"]
    stage_b_cases = payload["stage_b_cases"]
    if type(stage_a_cases) is not list or type(stage_b_cases) is not list:
        typer.echo("HOLD_CONFIRMATORY_CORPUS_AUDIT: invalid public corpus case lists", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        "PASS_MODEL_VISIBLE_CORPUS_BOUNDARY: "
        f"stage_a_cases={len(stage_a_cases)} stage_b_cases={len(stage_b_cases)}"
    )


@app.command("analyze")
def analyze(
    artifacts_root: Path = typer.Option(
        Path("artifacts/experiments/confirmatory"), "--artifacts-root"
    ),
    output: Path = typer.Option(Path("artifacts/analysis/confirmatory_analysis.json"), "--output"),
) -> None:
    """Recompute Stage A/B analysis from audited canonical episode outcomes."""

    try:
        binding = _scientific_freeze_binding(Path.cwd())
        sealed = analyze_sealed_confirmatory_artifacts(artifacts_root, scientific_freeze=binding)
        _write_exclusive(output, canonical_json_bytes(sealed_analysis_projection(sealed)))
    except SealedAnalysisHold as error:
        _cli_hold(error)
    typer.echo(
        "PASS_SEALED_ANALYSIS: "
        f"dataset_sha256={sealed.dataset_sha256} audit_sha256={sealed.audit.report_hash}"
    )


@report_app.command("build")
def report_build(
    artifacts_root: Path = typer.Option(
        Path("artifacts/experiments/confirmatory"), "--artifacts-root"
    ),
    analysis_receipt: Path = typer.Option(
        Path("artifacts/analysis/confirmatory_analysis.json"), "--analysis-receipt"
    ),
    output: Path = typer.Option(Path("artifacts/reports/report.html"), "--output"),
    workbench_output: Path = typer.Option(
        Path("artifacts/reports/workbench.json"), "--workbench-output"
    ),
    claims_ledger: Path = typer.Option(Path("protocol/CLAIMS_LEDGER.yaml"), "--claims-ledger"),
    limitation: list[str] = typer.Option([], "--limitation"),
) -> None:
    """Build a report only when every displayed result rebinds to sealed evidence."""

    try:
        root = Path.cwd()
        binding = _scientific_freeze_binding(root)
        sealed = analyze_sealed_confirmatory_artifacts(artifacts_root, scientific_freeze=binding)
        receipt = _read_analysis_receipt(analysis_receipt)
        if not sealed_analysis_receipt_matches(receipt, sealed):
            raise SealedAnalysisHold("HOLD_CROSS_HASH_ANALYSIS_MISMATCH", "analysis receipt")
        preregistration = validate_preregistration(root / "protocol/preregistration.md")
        if not preregistration.valid or preregistration.core is None:
            raise SealedAnalysisHold("HOLD_PREREGISTRATION_INVALID", "protocol/preregistration.md")
        claims = validate_claims_ledger(claims_ledger, root)
        if not claims.valid:
            raise SealedAnalysisHold("HOLD_CLAIMS_LEDGER_INVALID", str(claims_ledger))
        if not limitation:
            raise SealedAnalysisHold("HOLD_MISSING_REPORT_LIMITATIONS", "--limitation")
        protocol_evidence = _authorized_protocol_evidence(root)
        protocol = ProtocolEvidence(
            protocol_tag=protocol_evidence.protocol_tag,
            freeze_manifest_sha256=protocol_evidence.freeze_manifest_sha256,
            prompt_hashes=protocol_evidence.prompt_hashes,
            code_commit=protocol_evidence.code_commit,
            reproduce_command=protocol_evidence.reproduce_command,
            external_anchor_locator=protocol_evidence.external_anchor_locator,
            custody_locator=protocol_evidence.custody_locator,
            custody_mode=protocol_evidence.custody_mode,
            anchor_receipt_sha256=protocol_evidence.anchor_receipt_sha256,
            anchor_verified=protocol_evidence.anchor_verified,
            experiment_plan_sha256=sealed.audit.plan_hash,
            experiment_runtime_binding_sha256=sealed.audit.runtime_binding_hash,
            experiment_audit_report_sha256=sealed.audit.report_hash,
            experiment_audit_status="PASS",
        )
        if (
            sealed.runtime_binding.freeze_manifest_hash != protocol.freeze_manifest_sha256
            or sealed.runtime_binding.anchor_receipt_hash != protocol.anchor_receipt_sha256
            or sealed.runtime_binding.plan_hash != sealed.audit.plan_hash
        ):
            raise SealedAnalysisHold("HOLD_RUNTIME_PROTOCOL_BINDING_MISMATCH", "runtime binding")
        held_a = RepresentativeTrace(
            "A", SelectionResult("A", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
        )
        held_b = RepresentativeTrace(
            "B", SelectionResult("B", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
        )
        evidence = ReportEvidence(
            dataset=sealed.dataset,
            stage_a=sealed.stage_a,
            stage_b=sealed.stage_b,
            preregistration=preregistration,
            claims=claims,
            protocol=protocol,
            analysis_dataset_sha256=sealed.dataset_sha256,
            representative_traces=(held_a, held_b),
            limitations=tuple(limitation),
            exclusion_policy=sealed.exclusion_policy,
            runtime_binding=sealed.runtime_binding,
        )
        if report_status(evidence).value != "confirmatory":
            raise SealedAnalysisHold(
                "HOLD_REPORT_NOT_CONFIRMATORY", "sealed evidence is incomplete"
            )
        claim_gate = validate_release_claims(
            claims,
            report_status=report_status(evidence).value,
            experiment_audit_status="PASS",
            experiment_audit_hash=sealed.audit.report_hash,
            analysis_dataset_hash=sealed.dataset_sha256,
            limitations=tuple(limitation),
            sealed_analysis=sealed,
            scientific_freeze=binding,
        )
        if not claim_gate.passed:
            raise SealedAnalysisHold(claim_gate.status.value, "claims release gate")
        write_report_artifacts(evidence, output, workbench_output)
    except (SealedAnalysisHold, ValueError) as error:
        if type(error) is SealedAnalysisHold:
            _cli_hold(error)
        _cli_hold(SealedAnalysisHold("HOLD_INVALID_REPORT_EVIDENCE", str(error)))
    typer.echo(
        "PASS_REPORT_BUILD: "
        f"dataset_sha256={sealed.dataset_sha256} audit_sha256={sealed.audit.report_hash}"
    )


@app.command("reproduce")
def reproduce(
    protocol: Path = typer.Option(..., "--protocol"),
    artifacts_manifest: Path | None = typer.Option(None, "--artifacts-manifest"),
) -> None:
    """Offline reproduce sealed benchmark artifacts."""
    try:
        result = reproduce_sealed_artifacts(Path.cwd(), protocol, artifacts_manifest)
    except ReproductionHold as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    if result.classification == "CONFIRMATORY":
        typer.echo(
            "PASS_CONFIRMATORY_REPRODUCTION: offline sealed rebuild passed "
            "claim_ceiling=confirmatory "
            f"report={result.report_sha256} workbench={result.workbench_sha256}"
        )
    else:
        typer.echo(
            "PRACTICE_NOT_EVIDENCE: offline sealed rebuild passed "
            "claim_ceiling=practice_not_evidence "
            f"report={result.report_sha256} workbench={result.workbench_sha256}"
        )


@experiments_app.command("audit")
def experiments_audit(
    split: str | None = typer.Option(None, "--split"),
    stage_a: bool = typer.Option(False, "--stage-a"),
    stage_b: bool = typer.Option(False, "--stage-b"),
    artifacts_root: Path = typer.Option(
        Path("artifacts/experiments/confirmatory"), "--artifacts-root"
    ),
) -> None:
    """Audit experiment custody and resumability."""
    if stage_a and stage_b:
        typer.echo("HOLD_CONTRADICTORY_EXPERIMENT_AUDIT_SCOPE: stage-a,stage-b", err=True)
        raise typer.Exit(code=1)
    if split == "development":
        if stage_a or stage_b:
            typer.echo(
                "HOLD_DEVELOPMENT_EXPERIMENT_AUDIT_SCOPE: development audit has no stage filter",
                err=True,
            )
            raise typer.Exit(code=1)
        root = (
            artifacts_root
            if artifacts_root != Path("artifacts/experiments/confirmatory")
            else Path("artifacts/experiments/development")
        )
        try:
            summaries = audit_development_runs(root)
        except (RunCustodyError, ValueError, OSError) as error:
            typer.echo(f"HOLD_DEVELOPMENT_EXPERIMENT_AUDIT: {error}", err=True)
            raise typer.Exit(code=1) from error
        completed = sum(len(summary.completed_episode_ids) for summary in summaries)
        typer.echo(
            "PRACTICE_NOT_EVIDENCE: "
            f"development_runs={len(summaries)} completed={completed} audit=PASS"
        )
        return
    if split not in {None, "confirmatory"}:
        typer.echo(
            "HOLD_UNSUPPORTED_EXPERIMENT_AUDIT_SCOPE: complete confirmatory run required", err=True
        )
        raise typer.Exit(code=1)
    try:
        inputs = load_confirmatory_audit_inputs(artifacts_root)
        report = audit_confirmatory_run(
            inputs.plan,
            inputs.run_directory,
            catalog=inputs.catalog,
            exclusion_rule=inputs.exclusion_rule,
            stage=(
                EpisodeStage.CONFIRMATORY_A
                if stage_a
                else EpisodeStage.CONFIRMATORY_B
                if stage_b
                else None
            ),
        )
    except ExperimentArtifactHold as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    finding_codes = ",".join(sorted({finding.code for finding in report.findings}))
    if report.status == "PASS":
        typer.echo(f"PASS_EXPERIMENT_AUDIT: report_sha256={report.report_hash}")
        return
    typer.echo(
        f"HOLD_EXPERIMENT_AUDIT: report_sha256={report.report_hash} findings={finding_codes}",
        err=True,
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
