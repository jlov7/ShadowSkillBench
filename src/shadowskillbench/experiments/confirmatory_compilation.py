"""Post-anchor, sequential production of the 30 frozen confirmatory skills."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

import yaml

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.confirmatory import ConfirmatoryAuthorization
from shadowskillbench.experiments.audit import FrozenExclusionRule
from shadowskillbench.experiments.confirmatory_preparation import (
    ConfirmatoryPreparationHold,
    verify_design_commitment,
)
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_MODEL,
    OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH,
    OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT,
    OLLAMA_GPT_OSS_PROFILE,
    OLLAMA_GPT_OSS_PROVIDER,
    OLLAMA_GPT_OSS_REASONING_EFFORT,
    OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_SERVER_VERSION,
    OllamaRoleProfileError,
    validate_ollama_role_profile_receipt,
)
from shadowskillbench.models.protocol import ModelAdapterError, ModelClient
from shadowskillbench.skills.compiler import (
    CONFIRMATORY_WIRE_PROMPT_PROFILE,
    CompiledSkillArtifact,
    CompilerConfig,
    CompilerContractError,
    compile_skill,
    compiled_skill_artifact_hash,
    compiler_manifest_hash,
)
from shadowskillbench.skills.confirmatory_wire import ConfirmatorySkillIRWire
from shadowskillbench.skills.projection import compiler_view, hash_compiler_input
from shadowskillbench.traces.bundles import generate_bundle, hash_source_bundle_manifest

_PROFILE = "SSB-CONFIRMATORY-COMPILER6"
_EXPECTED_CONFIG = {
    "profile": _PROFILE,
    "provider_profile": OLLAMA_GPT_OSS_PROFILE,
    "model_configuration": "config/models.yaml",
    "prompt_path": "prompts/confirmatory_skill_compiler.md",
    "prompt_profile": CONFIRMATORY_WIRE_PROMPT_PROFILE,
    "output_schema_path": "protocol/commitments/confirmatory_compiler_wire.schema.json",
    "output_schema_model": "ConfirmatorySkillIRWire",
    "reasoning_effort": OLLAMA_GPT_OSS_REASONING_EFFORT,
    "temperature": OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
    "seed": 4243,
    "max_tokens": 16384,
    "max_attempts": 1,
    "transport": OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT,
    "request_profile": OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE,
    "request_profile_hash": OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH,
    "top_p": "omitted",
}
_FROZEN_INPUTS = {
    "config/compiler.yaml": "compiler_configuration",
    "config/models.yaml": "model_configuration",
    "prompts/confirmatory_skill_compiler.md": "confirmatory_compiler_prompt",
    "protocol/commitments/confirmatory_compiler_wire.schema.json": (
        "confirmatory_compiler_wire_schema"
    ),
    "protocol/commitments/role_conformance_receipt.json": ("confirmatory_role_conformance_receipt"),
    "src/shadowskillbench/experiments/confirmatory_compilation.py": (
        "confirmatory_compiler_producer"
    ),
    "src/shadowskillbench/experiments/confirmatory_preparation_cli.py": (
        "confirmatory_compiler_cli"
    ),
    "src/shadowskillbench/experiments/ollama_gpt_oss_profile.py": (
        "ollama_gpt_oss_runtime_profile"
    ),
    "src/shadowskillbench/models/runtime.py": "model_runtime",
    "src/shadowskillbench/models/ollama_native.py": "native_model_adapter",
    "src/shadowskillbench/skills/confirmatory_wire.py": "confirmatory_compiler_wire",
    "src/shadowskillbench/skills/compiler.py": "skill_compiler",
}


class ConfirmatoryCompilationHold(ValueError):
    """Raised when post-anchor compiler production has incomplete custody."""


@dataclass(frozen=True, slots=True)
class ConfirmatoryCompilerConfig:
    repository_root: Path
    prompt_bytes: bytes
    config_hash: str
    prompt_hash: str
    schema_hash: str

    @property
    def compiler_config(self) -> CompilerConfig:
        return CompilerConfig(
            prompt_profile=CONFIRMATORY_WIRE_PROMPT_PROFILE,
            prompt_bytes=self.prompt_bytes,
            temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            seed=4243,
            max_tokens=16384,
        )


def _canonical_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryCompilationHold(f"{label} is unavailable") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ConfirmatoryCompilationHold(f"{label} is not canonical")
    return cast(dict[str, object], value)


def _regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        return path.read_bytes()
    except OSError as error:
        raise ConfirmatoryCompilationHold(f"{label} is unavailable") from error


def _yaml_mapping(path: Path) -> dict[str, object]:
    raw = _regular_bytes(path, label="compiler configuration")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ConfirmatoryCompilationHold("compiler configuration is invalid") from error
    if type(value) is not dict or set(value) != set(_EXPECTED_CONFIG) or value != _EXPECTED_CONFIG:
        raise ConfirmatoryCompilationHold("compiler configuration is not the pinned profile")
    return cast(dict[str, object], value)


def load_confirmatory_compiler_config(repository_root: Path) -> ConfirmatoryCompilerConfig:
    if (
        not isinstance(repository_root, Path)
        or repository_root.is_symlink()
        or not repository_root.is_dir()
    ):
        raise ConfirmatoryCompilationHold("repository root must be a regular directory")
    config_path = repository_root / "config/compiler.yaml"
    _yaml_mapping(config_path)
    prompt_path = repository_root / "prompts/confirmatory_skill_compiler.md"
    schema_path = repository_root / "protocol/commitments/confirmatory_compiler_wire.schema.json"
    prompt_bytes = _regular_bytes(prompt_path, label="compiler prompt")
    expected_schema = canonical_json_bytes(ConfirmatorySkillIRWire.model_json_schema())
    actual_schema = _regular_bytes(schema_path, label="compiler wire schema")
    if actual_schema != expected_schema:
        raise ConfirmatoryCompilationHold("compiler wire schema is not canonical")
    try:
        CompilerConfig(
            prompt_profile=CONFIRMATORY_WIRE_PROMPT_PROFILE,
            prompt_bytes=prompt_bytes,
            temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            seed=4243,
            max_tokens=16384,
        )
    except ValueError as error:
        raise ConfirmatoryCompilationHold("compiler prompt is invalid") from error
    return ConfirmatoryCompilerConfig(
        repository_root=repository_root,
        prompt_bytes=prompt_bytes,
        config_hash="sha256:"
        + sha256(_regular_bytes(config_path, label="compiler configuration")).hexdigest(),
        prompt_hash="sha256:" + sha256(prompt_bytes).hexdigest(),
        schema_hash=sha256_ref(ConfirmatorySkillIRWire.model_json_schema()),
    )


def _authorization_inputs(authorization: ConfirmatoryAuthorization) -> dict[str, dict[str, str]]:
    if type(authorization) is not ConfirmatoryAuthorization:
        raise ConfirmatoryCompilationHold("post-anchor authorization is required")
    try:
        payload = json.loads(authorization.freeze.manifest_bytes)
        inputs = payload["inputs"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ConfirmatoryCompilationHold("freeze manifest inputs are invalid") from error
    if type(inputs) is not list:
        raise ConfirmatoryCompilationHold("freeze manifest inputs are invalid")
    records: dict[str, dict[str, str]] = {}
    for input_record in inputs:
        if type(input_record) is not dict or set(input_record) != {"path", "role", "sha256"}:
            raise ConfirmatoryCompilationHold("freeze manifest inputs are invalid")
        path = input_record.get("path")
        role = input_record.get("role")
        digest = input_record.get("sha256")
        if (
            type(path) is not str
            or type(role) is not str
            or type(digest) is not str
            or path in records
        ):
            raise ConfirmatoryCompilationHold("freeze manifest inputs are invalid")
        records[path] = {"role": role, "sha256": digest}
    return records


def _verify_frozen_inputs(authorization: ConfirmatoryAuthorization, repository_root: Path) -> None:
    inputs = _authorization_inputs(authorization)
    for relative_path, expected_role in _FROZEN_INPUTS.items():
        entry = inputs.get(relative_path)
        if entry is None or entry["role"] != expected_role:
            raise ConfirmatoryCompilationHold(f"freeze manifest omits {relative_path}")
        current = _regular_bytes(repository_root / relative_path, label=relative_path)
        digest = "sha256:" + sha256(current).hexdigest()
        if entry["sha256"] != digest:
            raise ConfirmatoryCompilationHold(f"frozen input changed: {relative_path}")


def _validated_receipt(value: object) -> dict[str, object]:
    try:
        receipt = validate_ollama_role_profile_receipt(value)
    except OllamaRoleProfileError as error:
        raise ConfirmatoryCompilationHold("role conformance receipt is invalid") from error
    if receipt.get("role_profile") != OLLAMA_GPT_OSS_PROFILE:
        raise ConfirmatoryCompilationHold("role conformance receipt has the wrong profile")
    return receipt


def _frozen_receipt(repository_root: Path, path: Path) -> tuple[dict[str, object], str]:
    expected = repository_root / "protocol/commitments/role_conformance_receipt.json"
    if (
        not isinstance(path, Path)
        or path.is_symlink()
        or path.resolve(strict=False) != expected.resolve(strict=False)
    ):
        raise ConfirmatoryCompilationHold("role conformance receipt must be the frozen receipt")
    raw = _regular_bytes(expected, label="role conformance receipt")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryCompilationHold("role conformance receipt is invalid") from error
    if canonical_json_bytes(value) != raw:
        raise ConfirmatoryCompilationHold("role conformance receipt is not canonical")
    return _validated_receipt(value), "sha256:" + sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise ConfirmatoryCompilationHold(f"refusing to overwrite {path}")
    try:
        path.write_bytes(canonical_json_bytes(value))
    except OSError as error:
        raise ConfirmatoryCompilationHold(f"cannot write {path}") from error


def _failure_receipt(
    *,
    bundle_id: str,
    compiler_input_hash: str,
    config: ConfirmatoryCompilerConfig,
    receipt_hash: str,
    error: ModelAdapterError | CompilerContractError,
) -> dict[str, object]:
    if type(error) is ModelAdapterError:
        usage = error.reported_usage
        return {
            "record_kind": "CONFIRMATORY_COMPILER_FAILURE_RECEIPT1",
            "bundle_id": bundle_id,
            "compiler_input_hash": compiler_input_hash,
            "compiler_profile": _PROFILE,
            "compiler_config_hash": config.config_hash,
            "compiler_prompt_hash": config.prompt_hash,
            "structured_output_schema_hash": config.schema_hash,
            "role_profile_receipt_hash": receipt_hash,
            "compiler_transport": OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT,
            "compiler_request_profile": OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE,
            "compiler_request_profile_hash": OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH,
            "compiler_top_p": "omitted",
            "error_code": error.code,
            "attempts": error.attempts,
            "before_meaningful_behavior": error.before_meaningful_behavior,
            "raw_request_hash": error.raw_request_hash,
            "raw_response_hash": error.raw_response_hash,
            "finish_reason": error.finish_reason,
            "reported_usage": None
            if usage is None
            else {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
            "validation_paths": [],
            "raw_provider_trace_retained": False,
        }
    contract_error = cast(CompilerContractError, error)
    usage = contract_error.reported_usage
    return {
        "record_kind": "CONFIRMATORY_COMPILER_FAILURE_RECEIPT1",
        "bundle_id": bundle_id,
        "compiler_input_hash": compiler_input_hash,
        "compiler_profile": _PROFILE,
        "compiler_config_hash": config.config_hash,
        "compiler_prompt_hash": config.prompt_hash,
        "structured_output_schema_hash": config.schema_hash,
        "role_profile_receipt_hash": receipt_hash,
        "compiler_transport": OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT,
        "compiler_request_profile": OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE,
        "compiler_request_profile_hash": OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH,
        "compiler_top_p": "omitted",
        "error_code": contract_error.code,
        "attempts": contract_error.attempts,
        "before_meaningful_behavior": False,
        "raw_request_hash": contract_error.raw_request_hash,
        "raw_response_hash": contract_error.raw_response_hash,
        "finish_reason": contract_error.finish_reason,
        "reported_usage": None
        if usage is None
        else {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        "validation_paths": list(contract_error.validation_paths),
        "raw_provider_trace_retained": False,
    }


def _artifact_matches_request(
    artifact: CompiledSkillArtifact,
    *,
    compiler_input_hash: str,
    config: ConfirmatoryCompilerConfig,
) -> bool:
    manifest = artifact.compiler_manifest
    return (
        manifest.compiler_input_hash == compiler_input_hash
        and manifest.prompt_profile == CONFIRMATORY_WIRE_PROMPT_PROFILE
        and manifest.system_prompt_raw_hash == config.prompt_hash
        and manifest.structured_output_schema_hash == config.schema_hash
        and manifest.temperature == OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE
        and manifest.seed == 4243
        and manifest.max_tokens == 16384
        and artifact.compiler_manifest_hash == compiler_manifest_hash(manifest)
        and artifact.structured_output_schema_hash == config.schema_hash
    )


async def compile_confirmatory_skills(
    *,
    repository_root: Path,
    output_dir: Path,
    commitment_path: Path,
    exclusion_rule: FrozenExclusionRule,
    authorization: ConfirmatoryAuthorization,
    role_profile_receipt_path: Path,
    client: ModelClient,
) -> Path:
    """Compile each committed bundle exactly once after validated custody admission."""

    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or output_dir.parent.is_symlink()
        or not output_dir.parent.is_dir()
    ):
        raise ConfirmatoryCompilationHold("output directory must be fresh under a regular parent")
    if not callable(getattr(client, "structured", None)):
        raise ConfirmatoryCompilationHold("compiler client is invalid")
    config = load_confirmatory_compiler_config(repository_root)
    _verify_frozen_inputs(authorization, repository_root)
    receipt, receipt_hash = _frozen_receipt(repository_root, role_profile_receipt_path)
    if receipt.get("profile_model_digest") is None:
        raise ConfirmatoryCompilationHold("role conformance receipt omits model digest")
    if (
        client.capabilities.provider != OLLAMA_GPT_OSS_PROVIDER
        or client.capabilities.model != OLLAMA_GPT_OSS_MODEL
    ):
        raise ConfirmatoryCompilationHold(
            "compiler client does not match the pinned provider profile"
        )
    if client.capabilities.model_version != receipt["profile_model_digest"]:
        raise ConfirmatoryCompilationHold("compiler client digest does not match the role receipt")
    if type(exclusion_rule) is not FrozenExclusionRule:
        raise ConfirmatoryCompilationHold("exclusion rule is invalid")
    try:
        commitment = verify_design_commitment(commitment_path, exclusion_rule=exclusion_rule)
    except (ConfirmatoryPreparationHold, ValueError) as error:
        raise ConfirmatoryCompilationHold("confirmatory design commitment is invalid") from error
    bundles = commitment.payload.get("bundles")
    if type(bundles) is not list or len(bundles) != 30:
        raise ConfirmatoryCompilationHold("confirmatory design does not contain 30 bundles")
    output_dir.mkdir()
    skills_dir = output_dir / "compiled-skills"
    skills_dir.mkdir()
    produced: list[dict[str, object]] = []
    seen: set[str] = set()
    for bundle_record in bundles:
        if type(bundle_record) is not dict:
            raise ConfirmatoryCompilationHold("bundle commitment is invalid")
        try:
            bundle_id = cast(str, bundle_record["bundle_id"])
            domain = cast(str, bundle_record["domain"])
            ratio = cast(str, bundle_record["contamination_ratio"])
            seed = cast(int, bundle_record["seed"])
            source_hash = cast(str, bundle_record["source_manifest_hash"])
        except (KeyError, TypeError) as error:
            raise ConfirmatoryCompilationHold("bundle commitment is invalid") from error
        if (
            type(bundle_id) is not str
            or bundle_id in seen
            or domain not in {"access_provisioning", "financial_adjustments"}
            or type(ratio) is not str
            or type(seed) is not int
            or type(source_hash) is not str
        ):
            raise ConfirmatoryCompilationHold("bundle commitment is invalid")
        seen.add(bundle_id)
        generated = generate_bundle(
            cast(Literal["access_provisioning", "financial_adjustments"], domain),
            Decimal(ratio),
            12,
            seed,
        )
        if (
            generated.bundle.source_manifest.bundle_id != bundle_id
            or hash_source_bundle_manifest(generated.bundle.source_manifest) != source_hash
        ):
            raise ConfirmatoryCompilationHold("bundle no longer matches the committed source")
        compiler_input_hash = hash_compiler_input(compiler_view(generated.bundle))
        try:
            artifact = await compile_skill(
                generated.bundle,
                client,
                config.compiler_config,
                output_schema=ConfirmatorySkillIRWire,
            )
        except (ModelAdapterError, CompilerContractError) as error:
            failure = _failure_receipt(
                bundle_id=bundle_id,
                compiler_input_hash=compiler_input_hash,
                config=config,
                receipt_hash=receipt_hash,
                error=error,
            )
            digest = sha256_ref(failure).removeprefix("sha256:")[:16]
            _write_new(skills_dir / f"{bundle_id}.failure-{digest}.json", failure)
            raise ConfirmatoryCompilationHold(
                f"compiler hold for {bundle_id}; failure receipt was written"
            ) from None
        if not _artifact_matches_request(
            artifact,
            compiler_input_hash=compiler_input_hash,
            config=config,
        ):
            error = CompilerContractError("COMPILER_OUTPUT_INVALID")
            failure = _failure_receipt(
                bundle_id=bundle_id,
                compiler_input_hash=compiler_input_hash,
                config=config,
                receipt_hash=receipt_hash,
                error=error,
            )
            digest = sha256_ref(failure).removeprefix("sha256:")[:16]
            _write_new(skills_dir / f"{bundle_id}.failure-{digest}.json", failure)
            raise ConfirmatoryCompilationHold(
                f"compiler hold for {bundle_id}; failure receipt was written"
            )
        _write_new(skills_dir / f"{bundle_id}.json", artifact.model_dump(mode="json"))
        produced.append(
            {
                "bundle_id": bundle_id,
                "artifact_hash": compiled_skill_artifact_hash(artifact),
                "compiler_manifest_hash": artifact.compiler_manifest_hash,
            }
        )
    if len(seen) != 30 or len(produced) != 30:
        raise ConfirmatoryCompilationHold("compiler coverage is incomplete")
    _write_new(
        output_dir / "compiler-run-manifest.json",
        {
            "record_kind": "CONFIRMATORY_COMPILER_RUN_MANIFEST1",
            "compiler_profile": _PROFILE,
            "provider_profile": OLLAMA_GPT_OSS_PROFILE,
            "provider": OLLAMA_GPT_OSS_PROVIDER,
            "model": OLLAMA_GPT_OSS_MODEL,
            "model_digest": receipt["profile_model_digest"],
            "ollama_server_version": OLLAMA_GPT_OSS_SERVER_VERSION,
            "reasoning_effort": OLLAMA_GPT_OSS_REASONING_EFFORT,
            "temperature": OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            "seed": 4243,
            "max_tokens": 16384,
            "max_attempts": 1,
            "compiler_transport": OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT,
            "compiler_request_profile": OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE,
            "compiler_request_profile_hash": OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH,
            "compiler_top_p": "omitted",
            "compiler_config_hash": config.config_hash,
            "compiler_prompt_hash": config.prompt_hash,
            "structured_output_schema_hash": config.schema_hash,
            "role_profile_receipt_hash": receipt_hash,
            "freeze_manifest_hash": authorization.freeze.manifest_hash,
            "anchor_receipt_hash": authorization.receipt.receipt_hash,
            "artifacts": produced,
        },
    )
    return skills_dir


__all__ = [
    "ConfirmatoryCompilationHold",
    "ConfirmatoryCompilerConfig",
    "compile_confirmatory_skills",
    "load_confirmatory_compiler_config",
]
