"""Typer registration for deterministic confirmatory commitment and staging commands."""

from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from pathlib import Path
from typing import cast

import typer

from shadowskillbench.corpus.confirmatory import (
    ConfirmatoryHoldError,
    authorize_confirmatory_generation,
    inventory_from_development,
)
from shadowskillbench.corpus.development import generate_development_corpus
from shadowskillbench.experiments.audit import FrozenExclusionRule
from shadowskillbench.experiments.confirmatory_compilation import (
    ConfirmatoryCompilationHold,
    compile_confirmatory_skills,
)
from shadowskillbench.experiments.confirmatory_package import (
    _descriptor_payload,
    _parse_descriptor,
    _write_new,
    ollama_gpt_oss_confirmatory_executor_profile,
)
from shadowskillbench.experiments.confirmatory_preparation import (
    DEVELOPMENT_CORPUS_SEED,
    ConfirmatoryPreparationHold,
    stage_confirmatory_execution_source,
    write_design_commitment,
)
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_EXECUTOR_MODEL,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    OLLAMA_GPT_OSS_MODEL,
    OLLAMA_GPT_OSS_PROVIDER,
    OLLAMA_GPT_OSS_REASONING_EFFORT,
    OllamaRoleProfileError,
    load_ollama_role_profile_receipt,
)
from shadowskillbench.models.runtime import (
    ModelRunDescriptor,
    RunDescriptorError,
    ollama_native_client_from_run_descriptor,
)
from shadowskillbench.skills.compiler import CompiledSkillArtifact, parse_compiled_skill_artifact


def _canonical_object(path: Path) -> dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryPreparationHold(f"unavailable input: {path}") from error
    if type(value) is not dict:
        raise ConfirmatoryPreparationHold(f"invalid object input: {path}")
    return cast(dict[str, object], value)


def _rule(path: Path) -> FrozenExclusionRule:
    payload = _canonical_object(path)
    try:
        rule = FrozenExclusionRule(
            allowed_error_codes=tuple(cast(list[str], payload["allowed_error_codes"])),
            max_retries_per_episode=cast(int, payload["max_retries_per_episode"]),
            max_total_exclusions=cast(int, payload["max_total_exclusions"]),
            max_exclusions_per_primary_cell=cast(int, payload["max_exclusions_per_primary_cell"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfirmatoryPreparationHold("invalid exclusion rule") from error
    if payload.get("rule_hash") != rule.rule_hash:
        raise ConfirmatoryPreparationHold("exclusion rule hash is invalid")
    return rule


def _compiled_skills(path: Path) -> dict[str, CompiledSkillArtifact]:
    try:
        if path.is_symlink() or not path.is_dir():
            raise OSError("not a regular directory")
        entries = tuple(sorted(path.iterdir(), key=lambda item: item.name))
    except OSError as error:
        raise ConfirmatoryPreparationHold("compiled skills directory is unavailable") from error
    skills: dict[str, CompiledSkillArtifact] = {}
    for entry in entries:
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
            raise ConfirmatoryPreparationHold("compiled skills directory is unsafe")
        bundle_id = entry.stem
        try:
            skills[bundle_id] = parse_compiled_skill_artifact(_canonical_object(entry))
        except ValueError as error:
            raise ConfirmatoryPreparationHold("compiled skill is invalid") from error
    return skills


def _descriptor(path: Path) -> ModelRunDescriptor:
    return _parse_descriptor(_canonical_object(path))


def register_confirmatory_preparation_commands(
    protocol_app: typer.Typer, episodes_app: typer.Typer
) -> None:
    @protocol_app.command("write-confirmatory-design-commitment")
    def write_confirmatory_design_commitment(
        protocol_dir: Path = typer.Option(Path("protocol"), "--protocol-dir"),
        exclusion_rule_path: Path = typer.Option(
            Path("protocol/commitments/exclusion_rule.json"),
            "--exclusion-rule",
            help="Frozen exclusion-rule.json committed before anchor.",
        ),
    ) -> None:
        """Write the deterministic pre-freeze design commitment; never contacts a provider."""

        try:
            path = write_design_commitment(protocol_dir, exclusion_rule=_rule(exclusion_rule_path))
        except (ConfirmatoryPreparationHold, OSError, ValueError) as error:
            typer.echo(f"HOLD_CONFIRMATORY_DESIGN_COMMITMENT: {error}", err=True)
            raise typer.Exit(code=1) from error
        typer.echo(f"PASS_CONFIRMATORY_DESIGN_COMMITMENT: path={path}")

    @protocol_app.command("write-confirmatory-run-descriptor")
    def write_confirmatory_run_descriptor(
        endpoint: str = typer.Option("http://127.0.0.1:11435/api/generate", "--endpoint"),
        api_key_environment: str = typer.Option("SSB_OLLAMA_LOCAL_TOKEN", "--api-key-environment"),
        role_profile_receipt: Path = typer.Option(
            Path("protocol/commitments/role_conformance_receipt.json"),
            "--role-profile-receipt",
        ),
        output_path: Path = typer.Option(
            Path("protocol/commitments/run_descriptor.json"), "--output"
        ),
    ) -> None:
        """Write the frozen native-executor descriptor after role preflight."""

        try:
            if endpoint != "http://127.0.0.1:11435/api/generate":
                raise ConfirmatoryPreparationHold("executor endpoint is not the pinned route")
            receipt = load_ollama_role_profile_receipt(role_profile_receipt)
            role_receipt_hash = "sha256:" + sha256(role_profile_receipt.read_bytes()).hexdigest()
            profile = ollama_gpt_oss_confirmatory_executor_profile(
                role_profile_receipt_hash=role_receipt_hash
            )
            descriptor = ModelRunDescriptor(
                provider=OLLAMA_GPT_OSS_PROVIDER,
                model=OLLAMA_GPT_OSS_EXECUTOR_MODEL,
                model_version=OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
                endpoint=endpoint,
                api_key_environment=api_key_environment,
                max_attempts=1,
                executor_runtime_profile=profile,
            )
            if receipt.get("profile_model_digest") is None:
                raise ConfirmatoryPreparationHold("role receipt omits compiler model digest")
            _write_new(output_path, _descriptor_payload(descriptor))
        except (
            ConfirmatoryPreparationHold,
            OllamaRoleProfileError,
            RunDescriptorError,
            OSError,
            ValueError,
        ) as error:
            typer.echo(f"HOLD_CONFIRMATORY_RUN_DESCRIPTOR: {error}", err=True)
            raise typer.Exit(code=1) from None
        typer.echo(f"PASS_CONFIRMATORY_RUN_DESCRIPTOR: path={output_path}")

    @episodes_app.command("stage-confirmatory-source")
    def stage_confirmatory_source(
        source_dir: Path = typer.Option(..., "--source-dir"),
        commitment_path: Path = typer.Option(
            Path("protocol/confirmatory_design_commitment.json"), "--commitment"
        ),
        compiled_skills_dir: Path = typer.Option(..., "--compiled-skills-dir"),
        run_descriptor_path: Path = typer.Option(
            Path("protocol/commitments/run_descriptor.json"), "--run-descriptor"
        ),
        exclusion_rule_path: Path = typer.Option(
            Path("protocol/commitments/exclusion_rule.json"), "--exclusion-rule"
        ),
        manifest_path: Path = typer.Option(Path("protocol/freeze_manifest.json"), "--manifest"),
        detached_sha256_path: Path = typer.Option(
            Path("protocol/freeze_manifest.sha256"), "--manifest-sha256"
        ),
        anchor_receipt_path: Path = typer.Option(
            Path("protocol/anchor_receipt.json"), "--anchor-receipt"
        ),
    ) -> None:
        """Regenerate a committed design and stage all 9,240 package-source inputs."""

        try:
            authorization = authorize_confirmatory_generation(
                manifest_path.read_bytes(),
                detached_sha256_path.read_bytes(),
                anchor_receipt_path.read_bytes(),
                repository_root=Path.cwd(),
            )
            staged = stage_confirmatory_execution_source(
                commitment_path=commitment_path,
                authorization=authorization,
                development_inventory=inventory_from_development(
                    generate_development_corpus(DEVELOPMENT_CORPUS_SEED)
                ),
                compiled_skills=_compiled_skills(compiled_skills_dir),
                run_descriptor=_descriptor(run_descriptor_path),
                exclusion_rule=_rule(exclusion_rule_path),
                source_dir=source_dir,
            )
        except (
            ConfirmatoryHoldError,
            ConfirmatoryPreparationHold,
            OSError,
            ValueError,
        ) as error:
            typer.echo(f"HOLD_CONFIRMATORY_SOURCE_STAGING: {error}", err=True)
            raise typer.Exit(code=1) from error
        typer.echo(f"PASS_CONFIRMATORY_SOURCE_STAGING: source_dir={staged} episodes=9240")

    @episodes_app.command("compile-confirmatory-skills")
    def compile_confirmatory_skills_command(
        endpoint: str = typer.Option(..., "--endpoint"),
        api_key_environment: str = typer.Option(..., "--api-key-environment"),
        role_profile_receipt: Path = typer.Option(..., "--role-profile-receipt"),
        output_dir: Path = typer.Option(..., "--output-dir"),
        commitment_path: Path = typer.Option(
            Path("protocol/confirmatory_design_commitment.json"), "--commitment"
        ),
        exclusion_rule_path: Path = typer.Option(..., "--exclusion-rule"),
        manifest_path: Path = typer.Option(Path("protocol/freeze_manifest.json"), "--manifest"),
        detached_sha256_path: Path = typer.Option(
            Path("protocol/freeze_manifest.sha256"), "--manifest-sha256"
        ),
        anchor_receipt_path: Path = typer.Option(
            Path("protocol/anchor_receipt.json"), "--anchor-receipt"
        ),
        repository_root: Path = typer.Option(Path("."), "--repository-root"),
    ) -> None:
        """Produce the 30 post-anchor skills sequentially under the frozen local profile."""

        try:
            if endpoint != "http://127.0.0.1:11435/api/generate":
                raise ConfirmatoryCompilationHold(
                    "compiler endpoint must be the pinned native generate route"
                )
            authorization = authorize_confirmatory_generation(
                manifest_path.read_bytes(),
                detached_sha256_path.read_bytes(),
                anchor_receipt_path.read_bytes(),
                repository_root=repository_root,
            )
            receipt = load_ollama_role_profile_receipt(role_profile_receipt)
            client = ollama_native_client_from_run_descriptor(
                ModelRunDescriptor(
                    provider=OLLAMA_GPT_OSS_PROVIDER,
                    model=OLLAMA_GPT_OSS_MODEL,
                    model_version=cast(str, receipt["profile_model_digest"]),
                    endpoint=endpoint,
                    api_key_environment=api_key_environment,
                    max_attempts=1,
                ),
                trust_env=False,
                timeout_seconds=900.0,
                reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
                include_top_p=False,
            )
            compiled = asyncio.run(
                compile_confirmatory_skills(
                    repository_root=repository_root,
                    output_dir=output_dir,
                    commitment_path=commitment_path,
                    exclusion_rule=_rule(exclusion_rule_path),
                    authorization=authorization,
                    role_profile_receipt_path=role_profile_receipt,
                    client=client,
                )
            )
        except (
            ConfirmatoryHoldError,
            ConfirmatoryCompilationHold,
            ConfirmatoryPreparationHold,
            OllamaRoleProfileError,
            RunDescriptorError,
            OSError,
            ValueError,
        ) as error:
            typer.echo(f"HOLD_CONFIRMATORY_COMPILATION: {error}", err=True)
            raise typer.Exit(code=1) from None
        typer.echo(f"PASS_CONFIRMATORY_COMPILATION: compiled_skills_dir={compiled} bundles=30")


__all__ = ["register_confirmatory_preparation_commands"]
