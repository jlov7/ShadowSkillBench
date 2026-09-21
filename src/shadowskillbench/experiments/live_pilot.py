"""Isolated custody and execution for the exploratory live development pilot.

This module deliberately has no confirmatory-package, freeze, anchor, or report
dependencies.  Its artifacts are labeled exploratory and cannot be consumed by
the confirmatory audit or reporting paths.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.development import DevelopmentCorpus, generate_development_corpus
from shadowskillbench.episodes.executor import (
    EpisodeDisposition,
    EpisodeResult,
    episode_result_projection,
    parse_episode_result,
    run_episode,
)
from shadowskillbench.episodes.models import (
    EpisodePlan,
    ExecutorModel,
    ExperimentCondition,
    hash_episode_manifest,
)
from shadowskillbench.episodes.pilot_turn_wire import (
    PILOT_TURN_PROMPT,
    PILOT_TURN_PROMPT_PROFILE,
    PilotAgentTurnWire,
    pilot_turn_layout_hash,
    pilot_turn_schema_hash,
    pilot_turn_tool_layout,
)
from shadowskillbench.episodes.tools import ToolRegistry
from shadowskillbench.experiments.development_execution import materialize_development_episode
from shadowskillbench.experiments.development_plan import DevelopmentPlannedEpisode
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
    OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
    OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_TRANSPORT,
    OLLAMA_GPT_OSS_LEGACY_V9_PROFILE,
    OLLAMA_GPT_OSS_LEGACY_V10_PROFILE,
    OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
    OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
    OLLAMA_GPT_OSS_PROFILE,
    OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_SERVER_VERSION,
    OllamaRoleProfileError,
    ollama_gpt_oss_executor_modelfile_hash,
    ollama_gpt_oss_executor_template_hash,
    ollama_gpt_oss_profile_projection,
    validate_ollama_gpt_oss_profile,
    validate_ollama_role_profile_receipt,
)
from shadowskillbench.metrics.outcomes import score_episode
from shadowskillbench.models import (
    ModelAdapterError,
    ModelClient,
    ProviderCapabilities,
    TokenCost,
    TokenUsage,
)
from shadowskillbench.skills.compiler import (
    PILOT_WIRE_PROMPT_PROFILE,
    CompiledSkillArtifact,
    CompilerConfig,
    CompilerContractError,
    CompilerManifest,
    compile_skill,
    compiled_skill_artifact_hash,
    compiled_skill_artifact_projection,
)
from shadowskillbench.skills.models import parse_skill_ir, skill_ir_projection
from shadowskillbench.skills.pilot_wire import (
    PilotSkillIRWire,
    pilot_wire_from_skill_ir,
    pilot_wire_schema_hash,
    pilot_wire_to_skill_ir,
)
from shadowskillbench.skills.projection import compiler_view, hash_compiler_input
from shadowskillbench.traces.bundles import generate_bundle

PILOT_LABEL = "EXPLORATORY_LIVE_PILOT_NOT_CONFIRMATORY"
PILOT_PROFILE = "SSB-EXPLORATORY-LIVE-PILOT1"
PILOT_SEED = 4242
PILOT_CONCURRENCY = 1
PILOT_MAX_ATTEMPTS = 1
PILOT_LIMIT = 60
PILOT_REQUEST_TIMEOUT_SECONDS = 900
PILOT_COMPILER_MAX_TOKENS = 8192
OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS = 16_384
OLLAMA_GPT_OSS_OBSERVED_ACCESS_R100_COMPILER_INPUT_TOKENS = 74_526
OLLAMA_GPT_OSS_OBSERVED_ACCESS_R100_CONTEXT_BOUND = (
    OLLAMA_GPT_OSS_OBSERVED_ACCESS_R100_COMPILER_INPUT_TOKENS + OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS
)
PILOT_DECLARED_SERVER_CONTEXT_LENGTH = 262_144
PILOT_SCRIPTED_PROVIDER_PROFILE = "SSB-PILOT-SCRIPTED1"
PILOT_WIRE_SCHEMA_HASH = pilot_wire_schema_hash()
PILOT_COMPILER_PROMPT_PROFILE = PILOT_WIRE_PROMPT_PROFILE
PILOT_TURN_SCHEMA_HASH = pilot_turn_schema_hash()
PILOT_TURN_PROMPT_HASH = "sha256:" + sha256(PILOT_TURN_PROMPT.encode("utf-8")).hexdigest()
_PILOT_WIRE_PROMPT = b"""

# SSB-PILOT-WIRE-COMPILER1

For this exploratory live pilot only, supersede the base prompt's two dynamic
object output shapes and mechanical fields with the compact PilotSkillIRWire
required by the response schema. Emit only `objective`,
`objective_evidence_indices`, and `ordered_steps`. The host supplies the
skill ID, schema version, domain, source trace IDs, provenance profile/input
hash, compiler-manifest reference, and empty descriptive arrays. For each
ordered step, emit `argument_bindings` as an array
of `{"key": <identifier>, "value": <scalar-or-flat-array>}` records. For
objective and each step, emit a nonempty `*_evidence_indices` array of unique
zero-based indexes into the flattened compiler-input event order (trace order,
then event order). Do not repeat a binding key within a step. All descriptive
arrays remain empty; all other SkillIR semantics and exact binding requirements
remain unchanged. Use printable ASCII only for free text; do not emit a double
quote or backslash within a free-text value.
Return only the exact PilotSkillIRWire JSON object.
"""
_DOMAINS: tuple[Literal["access_provisioning", "financial_adjustments"], ...] = (
    "access_provisioning",
    "financial_adjustments",
)
_RATIOS: tuple[tuple[Literal["R0", "R100"], Decimal], ...] = (
    ("R0", Decimal("0")),
    ("R100", Decimal("1")),
)


class LivePilotError(RuntimeError):
    """The exploratory pilot is not safe to start or resume."""


@dataclass(frozen=True, slots=True)
class LivePilotDescriptor:
    endpoint: str
    api_key_environment: str
    provider: str
    model: str
    model_version: str
    ollama_server_version: str | None = None
    base_model_digest: str | None = None
    reasoning_effort: Literal["none", "low"] = "none"
    executor_reasoning_effort: Literal["none", "low"] = "none"
    sampling_temperature: float = 0.0
    executor_sampling_temperature: float = 0.0
    executor_endpoint: str | None = None
    executor_transport: str | None = None
    executor_request_profile: str | None = None
    executor_request_profile_hash: str | None = None
    executor_think: bool | None = None
    executor_model_name: str | None = None
    executor_model_version: str | None = None
    executor_profile: str | None = None
    executor_modelfile_sha256: str | None = None
    executor_template_sha256: str | None = None
    request_timeout_seconds: Literal[900] = PILOT_REQUEST_TIMEOUT_SECONDS
    compiler_max_tokens: Literal[8192, 16_384] = PILOT_COMPILER_MAX_TOKENS
    compiler_wire_schema_hash: str = PILOT_WIRE_SCHEMA_HASH
    compiler_prompt_profile: Literal["SSB-PILOT-WIRE-COMPILER1"] = PILOT_COMPILER_PROMPT_PROFILE
    executor_turn_prompt_profile: Literal["SSB-PILOT-TURN-WIRE2"] = PILOT_TURN_PROMPT_PROFILE
    executor_turn_prompt_hash: str = PILOT_TURN_PROMPT_HASH
    executor_turn_schema_hash: str = PILOT_TURN_SCHEMA_HASH
    declared_server_context_length: int = PILOT_DECLARED_SERVER_CONTEXT_LENGTH
    provider_profile: str = PILOT_SCRIPTED_PROVIDER_PROFILE
    role_profile_receipt_hash: str | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint) if type(self.endpoint) is str else None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or not self.endpoint
            or any(
                type(item) is not str or not item.strip()
                for item in (
                    self.endpoint,
                    self.api_key_environment,
                    self.provider,
                    self.model,
                    self.model_version,
                )
            )
            or self.reasoning_effort not in {"none", "low"}
            or self.executor_reasoning_effort not in {"none", "low"}
            or type(self.sampling_temperature) is not float
            or type(self.executor_sampling_temperature) is not float
            or type(self.request_timeout_seconds) is not int
            or self.request_timeout_seconds != PILOT_REQUEST_TIMEOUT_SECONDS
            or type(self.compiler_max_tokens) is not int
            or type(self.compiler_wire_schema_hash) is not str
            or self.compiler_wire_schema_hash != PILOT_WIRE_SCHEMA_HASH
            or self.compiler_prompt_profile != PILOT_COMPILER_PROMPT_PROFILE
            or self.executor_turn_prompt_profile != PILOT_TURN_PROMPT_PROFILE
            or self.executor_turn_prompt_hash != PILOT_TURN_PROMPT_HASH
            or self.executor_turn_schema_hash != PILOT_TURN_SCHEMA_HASH
            or type(self.declared_server_context_length) is not int
        ):
            raise LivePilotError("pilot requires an explicit loopback endpoint and descriptors")
        if (
            not self.api_key_environment.replace("_", "").isalnum()
            or not self.api_key_environment[:1].isupper()
        ):
            raise LivePilotError("api_key_environment is invalid")
        if self.provider_profile == PILOT_SCRIPTED_PROVIDER_PROFILE:
            valid_scripted = (
                self.provider == "scripted-pilot"
                and self.reasoning_effort == "none"
                and self.executor_reasoning_effort == "none"
                and self.sampling_temperature == 0.0
                and self.executor_sampling_temperature == 0.0
                and self.declared_server_context_length == PILOT_DECLARED_SERVER_CONTEXT_LENGTH
                and self.compiler_max_tokens == PILOT_COMPILER_MAX_TOKENS
                and self.role_profile_receipt_hash is None
                and self.base_model_digest is None
                and self.ollama_server_version is None
                and self.executor_endpoint is None
                and self.executor_transport is None
                and self.executor_request_profile is None
                and self.executor_request_profile_hash is None
                and self.executor_think is None
                and self.executor_model_name is None
                and self.executor_model_version is None
                and self.executor_profile is None
                and self.executor_modelfile_sha256 is None
                and self.executor_template_sha256 is None
            )
            if not valid_scripted:
                raise LivePilotError("pilot requires an explicit loopback endpoint and descriptors")
            return
        if self.provider_profile != OLLAMA_GPT_OSS_PROFILE:
            raise LivePilotError("pilot provider profile is not admitted")
        try:
            validate_ollama_gpt_oss_profile(
                provider=self.provider,
                model=self.model,
                model_version=self.model_version,
                server_version=cast(str, self.ollama_server_version),
                reasoning_effort=self.reasoning_effort,
                executor_reasoning_effort=self.executor_reasoning_effort,
                sampling_temperature=self.sampling_temperature,
                executor_sampling_temperature=self.executor_sampling_temperature,
                context_length=self.declared_server_context_length,
            )
        except OllamaRoleProfileError as error:
            raise LivePilotError("pilot provider profile is not admitted") from error
        if self.base_model_digest != OLLAMA_GPT_OSS_BASE_MODEL_DIGEST:
            raise LivePilotError("Ollama base model digest is not pinned")
        if self.ollama_server_version != OLLAMA_GPT_OSS_SERVER_VERSION:
            raise LivePilotError("Ollama server version is not pinned")
        if self.compiler_max_tokens != OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS:
            raise LivePilotError("Ollama compiler cap is not pinned")
        if OLLAMA_GPT_OSS_OBSERVED_ACCESS_R100_CONTEXT_BOUND > self.declared_server_context_length:
            raise LivePilotError("Ollama compiler cap exceeds observed context bound")
        if (
            type(self.role_profile_receipt_hash) is not str
            or not self.role_profile_receipt_hash.startswith("sha256:")
            or len(self.role_profile_receipt_hash) != 71
        ):
            raise LivePilotError("Ollama role profile receipt is required")
        native_fields = (
            self.executor_endpoint,
            self.executor_transport,
            self.executor_request_profile,
            self.executor_request_profile_hash,
            self.executor_think,
            self.executor_model_name,
            self.executor_model_version,
            self.executor_profile,
            self.executor_modelfile_sha256,
            self.executor_template_sha256,
        )
        executor = urlparse(self.executor_endpoint) if type(self.executor_endpoint) is str else None
        if (
            executor is None
            or executor.scheme not in {"http", "https"}
            or executor.hostname not in {"127.0.0.1", "::1", "localhost"}
            or executor.path != "/api/generate"
            or executor.params
            or executor.query
            or executor.fragment
            or self.executor_transport != OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT
            or self.executor_request_profile != OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE
            or self.executor_request_profile_hash
            != OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH
            or self.executor_think is not False
            or self.executor_model_name != OLLAMA_GPT_OSS_EXECUTOR_MODEL
            or self.executor_model_version != OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST
            or self.executor_profile != OLLAMA_GPT_OSS_EXECUTOR_PROFILE
            or self.executor_modelfile_sha256 != ollama_gpt_oss_executor_modelfile_hash()
            or self.executor_template_sha256 != ollama_gpt_oss_executor_template_hash()
            or any(value is None for value in native_fields)
        ):
            raise LivePilotError("Ollama native executor transport is not pinned")
        compiler = urlparse(self.endpoint)
        if (
            executor.scheme != compiler.scheme
            or executor.netloc != compiler.netloc
            or compiler.path != "/v1/chat/completions"
            or compiler.params
            or compiler.query
            or compiler.fragment
        ):
            raise LivePilotError("Ollama compiler transport is not pinned")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.provider,
            model=self.model,
            model_version=self.model_version,
            supports_system_role=True,
            supports_developer_role=True,
            supports_seed=True,
            supports_structured_output=True,
        )

    @property
    def executor_capabilities(self) -> ProviderCapabilities:
        if self.executor_model_name is None or self.executor_model_version is None:
            if self.provider_profile == PILOT_SCRIPTED_PROVIDER_PROFILE:
                return self.capabilities
            raise LivePilotError("Ollama native executor model is not pinned")
        return ProviderCapabilities(
            provider=self.provider,
            model=self.executor_model_name,
            model_version=self.executor_model_version,
            supports_system_role=True,
            supports_developer_role=True,
            supports_seed=True,
            supports_structured_output=True,
        )

    @property
    def executor_episode_model(self) -> ExecutorModel:
        if self.executor_model_name is None or self.executor_model_version is None:
            if self.provider_profile == PILOT_SCRIPTED_PROVIDER_PROFILE:
                return ExecutorModel(
                    provider=self.provider, model=self.model, model_version_date=self.model_version
                )
            raise LivePilotError("Ollama native executor model is not pinned")
        return ExecutorModel(
            provider=self.provider,
            model=self.executor_model_name,
            model_version_date=self.executor_model_version,
        )

    def projection(self) -> dict[str, object]:
        return {
            "profile": PILOT_PROFILE,
            "classification": PILOT_LABEL,
            "endpoint_hash": sha256_ref(self.endpoint),
            "api_key_environment": self.api_key_environment,
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "ollama_server_version": self.ollama_server_version,
            "base_model_digest": self.base_model_digest,
            "reasoning_effort": self.reasoning_effort,
            "executor_reasoning_effort": self.executor_reasoning_effort,
            "sampling_temperature": self.sampling_temperature,
            "executor_sampling_temperature": self.executor_sampling_temperature,
            "executor_endpoint_hash": (
                sha256_ref(self.executor_endpoint) if self.executor_endpoint is not None else None
            ),
            "executor_transport": self.executor_transport,
            "executor_request_profile": self.executor_request_profile,
            "executor_request_profile_hash": self.executor_request_profile_hash,
            "executor_think": self.executor_think,
            "executor_model": self.executor_model_name,
            "executor_model_version": self.executor_model_version,
            "executor_profile": self.executor_profile,
            "executor_modelfile_sha256": self.executor_modelfile_sha256,
            "executor_template_sha256": self.executor_template_sha256,
            "request_timeout_seconds": self.request_timeout_seconds,
            "compiler_max_tokens": self.compiler_max_tokens,
            "compiler_wire_schema_hash": self.compiler_wire_schema_hash,
            "compiler_prompt_profile": self.compiler_prompt_profile,
            "executor_turn_prompt_profile": self.executor_turn_prompt_profile,
            "executor_turn_prompt_hash": self.executor_turn_prompt_hash,
            "executor_turn_schema_hash": self.executor_turn_schema_hash,
            "declared_server_context_length": self.declared_server_context_length,
            "provider_profile": self.provider_profile,
            "role_profile_receipt_hash": self.role_profile_receipt_hash,
            "ollama_role_profile": (
                ollama_gpt_oss_profile_projection()
                if self.provider_profile == OLLAMA_GPT_OSS_PROFILE
                else None
            ),
            "concurrency": PILOT_CONCURRENCY,
            "max_attempts": PILOT_MAX_ATTEMPTS,
        }


@dataclass(frozen=True, slots=True)
class LivePilotCell:
    domain: Literal["access_provisioning", "financial_adjustments"]
    case_id: str
    condition: ExperimentCondition
    ratio_label: Literal["R0", "R100"] | None
    contamination_ratio: Decimal | None
    bundle_id: str | None

    def projection(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "case_id": self.case_id,
            "condition": self.condition.value,
            "ratio": self.ratio_label,
            "bundle_id": self.bundle_id,
        }


def build_live_pilot_plan(corpus: DevelopmentCorpus) -> tuple[LivePilotCell, ...]:
    """Build the immutable 2 x 10 x (A0 + A2 x 2) exploratory matrix."""

    if type(corpus) is not DevelopmentCorpus or corpus.seed != PILOT_SEED:
        raise LivePilotError("pilot requires the seed-4242 development corpus")
    cases_by_domain = {
        domain: tuple(case for case in corpus.cases if case.domain == domain)[:10]
        for domain in _DOMAINS
    }
    if any(len(cases) != 10 for cases in cases_by_domain.values()):
        raise LivePilotError("pilot requires ten active development cases per domain")
    bundles_by_domain = {
        domain: {
            label: generate_bundle(domain, ratio, 12, PILOT_SEED).bundle.source_manifest.bundle_id
            for label, ratio in _RATIOS
        }
        for domain in _DOMAINS
    }
    cells: list[LivePilotCell] = []
    for case_index in range(10):
        for domain in _DOMAINS:
            case = cases_by_domain[domain][case_index]
            cells.append(
                LivePilotCell(domain, case.case_id, ExperimentCondition.A0_BARE, None, None, None)
            )
            for label, ratio in _RATIOS:
                cells.append(
                    LivePilotCell(
                        domain,
                        case.case_id,
                        ExperimentCondition.A2_SKILL_ONLY,
                        label,
                        ratio,
                        bundles_by_domain[domain][label],
                    )
                )
    plan = tuple(cells)
    if len(plan) != PILOT_LIMIT:
        raise LivePilotError("pilot design must contain exactly 60 cells")
    return plan


def _protocol_hash(protocol_path: Path) -> str:
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise LivePilotError("pilot protocol is unavailable")
    try:
        return "sha256:" + sha256(protocol_path.read_bytes()).hexdigest()
    except OSError as error:
        raise LivePilotError("pilot protocol is unreadable") from error


def _plan_projection(
    cells: tuple[LivePilotCell, ...], protocol_hash: str, prompt_hash: str
) -> dict[str, object]:
    return {
        "profile": PILOT_PROFILE,
        "classification": PILOT_LABEL,
        "seed": PILOT_SEED,
        "protocol_hash": protocol_hash,
        "compiler_prompt_hash": prompt_hash,
        "compiler_prompt_profile": PILOT_COMPILER_PROMPT_PROFILE,
        "executor_turn_prompt_profile": PILOT_TURN_PROMPT_PROFILE,
        "executor_turn_prompt_hash": PILOT_TURN_PROMPT_HASH,
        "executor_turn_schema_hash": PILOT_TURN_SCHEMA_HASH,
        "executor_turn_tool_layout_hashes": _pilot_turn_layout_hashes(cells),
        "cells": [cell.projection() for cell in cells],
    }


def _pilot_turn_layout_hash(
    domain: Literal["access_provisioning", "financial_adjustments"],
    condition: ExperimentCondition,
) -> str:
    return pilot_turn_layout_hash(
        pilot_turn_tool_layout(ToolRegistry.for_domain(domain), condition)
    )


def _pilot_turn_layout_hashes(cells: tuple[LivePilotCell, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for cell in cells:
        key = f"{cell.domain}:{cell.condition.value}"
        layout_hash = _pilot_turn_layout_hash(cell.domain, cell.condition)
        if key in hashes and hashes[key] != layout_hash:
            raise LivePilotError("pilot turn tool layouts are inconsistent")
        hashes[key] = layout_hash
    return {key: hashes[key] for key in sorted(hashes)}


def _safe_directory(path: Path) -> Path:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise LivePilotError("pilot artifact path contains a symlink")
        absolute.mkdir(parents=True, exist_ok=True)
        if absolute.is_symlink() or not absolute.is_dir():
            raise LivePilotError("pilot artifact directory is unsafe")
    except OSError as error:
        raise LivePilotError("pilot artifact directory is unavailable") from error
    return absolute


def _safe_existing_directory(path: Path) -> Path:
    absolute = path if path.is_absolute() else Path.cwd() / path
    if not absolute.exists() or absolute.is_symlink() or not absolute.is_dir():
        raise LivePilotError("pilot artifact directory is unavailable")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise LivePilotError("pilot artifact path contains a symlink")
    return absolute


def _read_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise LivePilotError(f"{label} is a symlink")
    try:
        raw = path.read_bytes()

        def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if type(key) is not str or key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = item
            return result

        value = json.loads(raw, object_pairs_hook=no_duplicates)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise LivePilotError(f"{label} is unreadable") from error
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or canonical_json_bytes(value) != raw
    ):
        raise LivePilotError(f"{label} is invalid")
    return cast(dict[str, object], value)


def _write_once(path: Path, value: dict[str, object]) -> None:
    payload = canonical_json_bytes(value)
    _safe_directory(path.parent)
    if path.exists():
        if _read_json(path, path.name) != value:
            raise LivePilotError(f"{path.name} differs from immutable pilot custody")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if _read_json(path, path.name) != value:
                raise LivePilotError(f"{path.name} differs from immutable pilot custody")
    finally:
        temporary_path.unlink(missing_ok=True)


def _compiler_key(domain: str, ratio_label: str) -> str:
    return f"{domain}-{ratio_label.lower()}"


def _pilot_prompt_bytes(base_prompt_bytes: bytes) -> bytes:
    if type(base_prompt_bytes) is not bytes or not base_prompt_bytes.endswith(b"\n"):
        raise LivePilotError("compiler prompt is invalid")
    return base_prompt_bytes + _PILOT_WIRE_PROMPT


def _compiler_input_hash(
    domain: Literal["access_provisioning", "financial_adjustments"], ratio: Decimal
) -> str:
    return hash_compiler_input(compiler_view(generate_bundle(domain, ratio, 12, PILOT_SEED).bundle))


def _pilot_wire_round_trip(
    artifact: CompiledSkillArtifact,
    domain: Literal["access_provisioning", "financial_adjustments"],
    ratio: Decimal,
) -> bool:
    try:
        compiler_input = compiler_view(generate_bundle(domain, ratio, 12, PILOT_SEED).bundle)
        hydrated = pilot_wire_to_skill_ir(
            pilot_wire_from_skill_ir(
                artifact.skill_ir, compiler_input, artifact.compiler_manifest_hash
            ),
            compiler_input,
            artifact.compiler_manifest_hash,
        )
        return skill_ir_projection(hydrated) == skill_ir_projection(artifact.skill_ir)
    except ValueError:
        return False


def _load_artifact(value: dict[str, object]) -> CompiledSkillArtifact:
    try:
        manifest_raw = cast(dict[str, object], value["compiler_manifest"])
        capabilities = ProviderCapabilities.model_validate(
            cast(Any, manifest_raw["declared_capabilities"])
        )
        manifest = CompilerManifest(
            **cast(Any, {**manifest_raw, "declared_capabilities": capabilities})
        )
        usage = TokenUsage.model_validate(cast(Any, value["usage"]))
        raw_cost = value["cost"]
        cost = None if raw_cost is None else TokenCost.model_validate(cast(Any, raw_cost))
        return CompiledSkillArtifact(
            **{
                **value,
                "compiler_manifest": manifest,
                "skill_ir": parse_skill_ir(value["skill_ir"]),
                "usage": usage,
                "cost": cost,
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LivePilotError("compiled skill artifact is invalid") from error


COMPILER_FAILURE_RECORD_KIND = "PILOT_COMPILER_FAILURE_RECEIPT1"
COMPILER_CONTRACT_FAILURE_RECORD_KIND = "PILOT_COMPILER_CONTRACT_FAILURE_RECEIPT1"


def _compiler_failure_receipt(
    *,
    domain: str,
    ratio_label: str,
    ratio: Decimal,
    compiler_input_hash: str,
    descriptor: LivePilotDescriptor,
    error: ModelAdapterError,
    plan_hash: str | None,
    runtime_hash: str | None,
) -> dict[str, object]:
    """Secret-free custody of a failed compiler call: hashes, usage, and finish reason only."""

    usage = error.reported_usage
    return {
        "record_kind": COMPILER_FAILURE_RECORD_KIND,
        "profile": PILOT_PROFILE,
        "classification": PILOT_LABEL,
        "plan_hash": plan_hash,
        "runtime_hash": runtime_hash,
        "domain": domain,
        "ratio": ratio_label,
        "contamination_ratio": str(ratio),
        "compiler_input_hash": compiler_input_hash,
        "compiler_prompt_profile": descriptor.compiler_prompt_profile,
        "compiler_wire_schema_hash": descriptor.compiler_wire_schema_hash,
        "compiler_max_tokens": descriptor.compiler_max_tokens,
        "provider": descriptor.provider,
        "model": descriptor.model,
        "model_version": descriptor.model_version,
        "error_code": error.code,
        "attempts": error.attempts,
        "before_meaningful_behavior": error.before_meaningful_behavior,
        "raw_request_hash": error.raw_request_hash,
        "raw_response_hash": error.raw_response_hash,
        "finish_reason": error.finish_reason,
        "reported_usage": None if usage is None else usage.model_dump(mode="json"),
        "output_cap_exhausted": (
            usage is not None and usage.output_tokens >= descriptor.compiler_max_tokens
        ),
        "raw_provider_trace_retained": False,
    }


def _write_compiler_failure_receipt(run_directory: Path, receipt: dict[str, object]) -> Path:
    key = _compiler_key(cast(str, receipt["domain"]), cast(str, receipt["ratio"]))
    path = run_directory / "compiled-skills" / f"{key}.failure-{sha256_ref(receipt)[7:19]}.json"
    _write_once(path, receipt)
    return path


def _compiler_contract_failure_receipt(
    *,
    domain: str,
    ratio_label: str,
    ratio: Decimal,
    compiler_input_hash: str,
    descriptor: LivePilotDescriptor,
    error: CompilerContractError,
    plan_hash: str | None,
    runtime_hash: str | None,
) -> dict[str, object]:
    usage = error.reported_usage
    return {
        "record_kind": COMPILER_CONTRACT_FAILURE_RECORD_KIND,
        "profile": PILOT_PROFILE,
        "classification": PILOT_LABEL,
        "plan_hash": plan_hash,
        "runtime_hash": runtime_hash,
        "domain": domain,
        "ratio": ratio_label,
        "contamination_ratio": str(ratio),
        "compiler_input_hash": compiler_input_hash,
        "compiler_prompt_profile": descriptor.compiler_prompt_profile,
        "compiler_wire_schema_hash": descriptor.compiler_wire_schema_hash,
        "compiler_max_tokens": descriptor.compiler_max_tokens,
        "sampling_temperature": descriptor.sampling_temperature,
        "provider": descriptor.provider,
        "model": descriptor.model,
        "model_version": descriptor.model_version,
        "error_code": error.code,
        "stage": error.stage,
        "validation_code": error.validation_code,
        "validation_paths": list(error.validation_paths),
        "attempts": error.attempts,
        "raw_request_hash": error.raw_request_hash,
        "raw_response_hash": error.raw_response_hash,
        "finish_reason": error.finish_reason,
        "finish_reason_source": error.finish_reason_source,
        "reported_usage": None if usage is None else usage.model_dump(mode="json"),
        "output_cap_exhausted": (
            usage is not None and usage.output_tokens >= descriptor.compiler_max_tokens
        ),
        "raw_provider_trace_retained": False,
    }


def _write_compiler_contract_failure_receipt(
    run_directory: Path, receipt: dict[str, object]
) -> Path:
    key = _compiler_key(cast(str, receipt["domain"]), cast(str, receipt["ratio"]))
    digest = sha256_ref(receipt)[7:19]
    path = run_directory / "compiled-skills" / f"{key}.contract-failure-{digest}.json"
    _write_once(path, receipt)
    return path


async def _compiled_skills(
    run_directory: Path,
    descriptor: LivePilotDescriptor,
    prompt_bytes: bytes,
    client: ModelClient,
    *,
    plan_hash: str | None = None,
    runtime_hash: str | None = None,
) -> dict[tuple[str, str], CompiledSkillArtifact]:
    if (
        client.capabilities != descriptor.capabilities
        or getattr(client, "reasoning_effort", None) != descriptor.reasoning_effort
    ):
        raise LivePilotError("injected live client does not match the explicit runtime descriptor")
    prompt_bytes = _pilot_prompt_bytes(prompt_bytes)
    config = CompilerConfig(
        prompt_profile=PILOT_COMPILER_PROMPT_PROFILE,
        prompt_bytes=prompt_bytes,
        temperature=descriptor.sampling_temperature,
        seed=PILOT_SEED,
        max_tokens=descriptor.compiler_max_tokens,
    )
    skills: dict[tuple[str, str], CompiledSkillArtifact] = {}
    for domain in _DOMAINS:
        for label, ratio in _RATIOS:
            path = run_directory / "compiled-skills" / f"{_compiler_key(domain, label)}.json"
            expected_input = _compiler_input_hash(domain, ratio)
            if path.exists():
                stored = _read_json(path, "compiled skill artifact")
                stored_manifest = stored.get("compiler_manifest")
                if (
                    type(stored_manifest) is dict
                    and stored_manifest.get("temperature") != descriptor.sampling_temperature
                ):
                    raise LivePilotError("compiled skill temperature does not match pilot runtime")
                artifact = _load_artifact(stored)
            else:
                try:
                    artifact = await compile_skill(
                        generate_bundle(domain, ratio, 12, PILOT_SEED).bundle,
                        client,
                        config,
                        output_schema=PilotSkillIRWire,
                    )
                except ModelAdapterError as error:
                    _write_compiler_failure_receipt(
                        run_directory,
                        _compiler_failure_receipt(
                            domain=domain,
                            ratio_label=label,
                            ratio=ratio,
                            compiler_input_hash=expected_input,
                            descriptor=descriptor,
                            error=error,
                            plan_hash=plan_hash,
                            runtime_hash=runtime_hash,
                        ),
                    )
                    raise
                except CompilerContractError as error:
                    _write_compiler_contract_failure_receipt(
                        run_directory,
                        _compiler_contract_failure_receipt(
                            domain=domain,
                            ratio_label=label,
                            ratio=ratio,
                            compiler_input_hash=expected_input,
                            descriptor=descriptor,
                            error=error,
                            plan_hash=plan_hash,
                            runtime_hash=runtime_hash,
                        ),
                    )
                    raise
                _write_once(
                    path, cast(dict[str, object], compiled_skill_artifact_projection(artifact))
                )
            if (
                artifact.compiler_manifest.compiler_input_hash != expected_input
                or artifact.compiler_manifest.declared_capabilities != descriptor.capabilities
                or artifact.compiler_manifest.system_prompt_raw_hash
                != "sha256:" + sha256(prompt_bytes).hexdigest()
                or artifact.compiler_manifest.max_tokens != descriptor.compiler_max_tokens
                or artifact.compiler_manifest.prompt_profile != descriptor.compiler_prompt_profile
                or artifact.compiler_manifest.structured_output_schema_hash
                != descriptor.compiler_wire_schema_hash
                or artifact.attempts != PILOT_MAX_ATTEMPTS
                or not _pilot_wire_round_trip(artifact, domain, ratio)
            ):
                raise LivePilotError("compiled skill does not bind the pilot design and runtime")
            if artifact.compiler_manifest.temperature != descriptor.sampling_temperature:
                raise LivePilotError("compiled skill temperature does not match pilot runtime")
            skills[(domain, label)] = artifact
    if len(skills) != 4:
        raise LivePilotError("pilot must compile exactly four live skills")
    return skills


def _episode_for(cell: LivePilotCell) -> DevelopmentPlannedEpisode:
    return DevelopmentPlannedEpisode(
        "stage_a", cell.condition, cell.case_id, cell.domain, cell.bundle_id
    )


def _materialize(
    corpus: DevelopmentCorpus,
    cell: LivePilotCell,
    executor_model: ExecutorModel,
    skills: Mapping[tuple[str, str], CompiledSkillArtifact],
) -> EpisodePlan:
    artifact = None if cell.ratio_label is None else skills[(cell.domain, cell.ratio_label)]
    return materialize_development_episode(
        corpus,
        _episode_for(cell),
        compiled_skill_override=artifact,
        executor_model_override=executor_model,
    )


def _result_envelope(
    plan_hash: str,
    runtime_hash: str,
    cell: LivePilotCell,
    result: EpisodeResult,
    compiled_hash: str | None,
    executor_sampling_temperature: float,
    executor_reasoning_effort: Literal["none", "low"],
    executor_model_name: str | None,
    executor_model_version: str | None,
) -> dict[str, object]:
    return {
        "profile": PILOT_PROFILE,
        "classification": PILOT_LABEL,
        "plan_hash": plan_hash,
        "runtime_hash": runtime_hash,
        "cell": cell.projection(),
        "compiled_skill_hash": compiled_hash,
        "executor_sampling_temperature": executor_sampling_temperature,
        "executor_reasoning_effort": executor_reasoning_effort,
        "executor_model": executor_model_name,
        "executor_model_version": executor_model_version,
        "executor_turn_prompt_profile": PILOT_TURN_PROMPT_PROFILE,
        "executor_turn_prompt_hash": PILOT_TURN_PROMPT_HASH,
        "executor_turn_schema_hash": PILOT_TURN_SCHEMA_HASH,
        "executor_turn_tool_layout_hash": _pilot_turn_layout_hash(cell.domain, cell.condition),
        "execution_manifest_hash": result.manifest_hash,
        "result": result.model_dump(mode="json"),
    }


def _validate_role_profile_receipt(
    descriptor: LivePilotDescriptor, receipt: dict[str, object] | None
) -> None:
    if descriptor.provider_profile == PILOT_SCRIPTED_PROVIDER_PROFILE:
        if receipt is not None:
            raise LivePilotError("scripted pilot cannot bind an Ollama role profile receipt")
        return
    if type(receipt) is not dict:
        raise LivePilotError("Ollama role profile receipt is required")
    try:
        validated = validate_ollama_role_profile_receipt(receipt)
    except OllamaRoleProfileError as error:
        raise LivePilotError("Ollama role profile receipt is invalid") from error
    if sha256_ref(validated) != descriptor.role_profile_receipt_hash:
        raise LivePilotError("Ollama role profile receipt does not bind the descriptor")
    if validated["role_profile"] != descriptor.provider_profile:
        raise LivePilotError("Ollama role profile receipt does not bind the provider profile")
    if validated["profile_model_digest"] != descriptor.model_version:
        raise LivePilotError("Ollama role profile receipt does not bind the model version")
    if (
        validated.get("executor_model_digest") != descriptor.executor_model_version
        or validated.get("executor_model") != descriptor.executor_model_name
    ):
        raise LivePilotError("Ollama role profile receipt does not bind the executor model")


def _runtime_profile_is_valid(runtime: dict[str, object]) -> bool:
    profile = runtime.get("provider_profile")
    if profile == PILOT_SCRIPTED_PROVIDER_PROFILE:
        return (
            runtime.get("reasoning_effort") == "none"
            and runtime.get("executor_reasoning_effort") == "none"
            and type(runtime.get("sampling_temperature")) is float
            and runtime.get("sampling_temperature") == 0.0
            and type(runtime.get("executor_sampling_temperature")) is float
            and runtime.get("executor_sampling_temperature") == 0.0
            and runtime.get("declared_server_context_length")
            == PILOT_DECLARED_SERVER_CONTEXT_LENGTH
            and runtime.get("role_profile_receipt_hash") is None
            and runtime.get("base_model_digest") is None
            and runtime.get("ollama_server_version") is None
            and runtime.get("ollama_role_profile") is None
        )
    if profile not in {
        OLLAMA_GPT_OSS_LEGACY_V9_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V10_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
        OLLAMA_GPT_OSS_PROFILE,
    }:
        return False
    try:
        validate_ollama_gpt_oss_profile(
            provider=cast(str, runtime["provider"]),
            model=cast(str, runtime["model"]),
            model_version=cast(str, runtime["model_version"]),
            server_version=cast(str, runtime["ollama_server_version"]),
            reasoning_effort=cast(str, runtime["reasoning_effort"]),
            executor_reasoning_effort=cast(str, runtime["executor_reasoning_effort"]),
            sampling_temperature=cast(float, runtime["sampling_temperature"]),
            executor_sampling_temperature=cast(
                float,
                runtime["executor_sampling_temperature"]
                if profile == OLLAMA_GPT_OSS_PROFILE
                else OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
            ),
            context_length=cast(int, runtime["declared_server_context_length"]),
        )
    except (KeyError, OllamaRoleProfileError):
        return False
    native_fields = (
        runtime.get("executor_endpoint_hash"),
        runtime.get("executor_transport"),
        runtime.get("executor_request_profile"),
        runtime.get("executor_request_profile_hash"),
        runtime.get("executor_think"),
        runtime.get("executor_model"),
        runtime.get("executor_model_version"),
        runtime.get("executor_profile"),
        runtime.get("executor_modelfile_sha256"),
        runtime.get("executor_template_sha256"),
    )
    executor_model_fields = native_fields[5:]
    native_transport_valid = False
    if profile == OLLAMA_GPT_OSS_LEGACY_V9_PROFILE:
        native_transport_valid = all(value is None for value in native_fields)
    elif profile == OLLAMA_GPT_OSS_LEGACY_V10_PROFILE:
        native_transport_valid = (
            type(runtime.get("executor_endpoint_hash")) is str
            and runtime.get("executor_transport") == OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_TRANSPORT
            and runtime.get("executor_request_profile")
            == OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE
            and runtime.get("executor_request_profile_hash")
            == OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH
            and runtime.get("executor_think") is False
            and all(value is None for value in executor_model_fields)
        )
    elif profile == OLLAMA_GPT_OSS_LEGACY_V11_PROFILE:
        native_transport_valid = (
            type(runtime.get("executor_endpoint_hash")) is str
            and runtime.get("executor_transport") == OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_TRANSPORT
            and runtime.get("executor_request_profile")
            == OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE
            and runtime.get("executor_request_profile_hash")
            == OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH
            and runtime.get("executor_think") is False
            and runtime.get("executor_model") == OLLAMA_GPT_OSS_EXECUTOR_MODEL
            and runtime.get("executor_model_version") == OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST
            and runtime.get("executor_profile") == OLLAMA_GPT_OSS_EXECUTOR_PROFILE
            and runtime.get("executor_modelfile_sha256") == ollama_gpt_oss_executor_modelfile_hash()
            and runtime.get("executor_template_sha256") == ollama_gpt_oss_executor_template_hash()
        )
    else:
        native_transport_valid = (
            type(runtime.get("executor_endpoint_hash")) is str
            and runtime.get("executor_transport") == OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT
            and runtime.get("executor_request_profile")
            == OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE
            and runtime.get("executor_request_profile_hash")
            == OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH
            and runtime.get("executor_think") is False
            and runtime.get("executor_model") == OLLAMA_GPT_OSS_EXECUTOR_MODEL
            and runtime.get("executor_model_version") == OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST
            and runtime.get("executor_profile") == OLLAMA_GPT_OSS_EXECUTOR_PROFILE
            and runtime.get("executor_modelfile_sha256") == ollama_gpt_oss_executor_modelfile_hash()
            and runtime.get("executor_template_sha256") == ollama_gpt_oss_executor_template_hash()
        )
    return (
        type(runtime.get("role_profile_receipt_hash")) is str
        and runtime.get("ollama_server_version") == OLLAMA_GPT_OSS_SERVER_VERSION
        and runtime.get("sampling_temperature") == OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE
        and (
            runtime.get("executor_sampling_temperature")
            == OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE
            if profile == OLLAMA_GPT_OSS_PROFILE
            else runtime.get("executor_sampling_temperature") is None
        )
        and runtime.get("base_model_digest") == OLLAMA_GPT_OSS_BASE_MODEL_DIGEST
        and runtime.get("ollama_role_profile")
        == ollama_gpt_oss_profile_projection(profile=cast(str, profile))
        and native_transport_valid
    )


def _admit_completed_result(result: EpisodeResult, execution_plan: EpisodePlan) -> None:
    if (
        result.disposition
        in {
            EpisodeDisposition.MODEL_FAILURE,
            EpisodeDisposition.ENVIRONMENT_FAILURE,
        }
        or result.error_code == "MODEL_OUTPUT_INVALID"
    ):
        detail = f"pilot stopped on infrastructure failure: {result.error_code}"
        if result.trace:
            receipt = result.trace[-1].receipt
            if receipt.finish_reason is not None:
                detail += f" finish_reason={receipt.finish_reason}"
            if receipt.usage is not None:
                detail += (
                    f" input_tokens={receipt.usage.input_tokens}"
                    f" output_tokens={receipt.usage.output_tokens}"
                )
            if receipt.output_cap_exhausted is not None:
                detail += f" output_cap_exhausted={str(receipt.output_cap_exhausted).lower()}"
        raise LivePilotError(detail)
    if result.manifest_hash != hash_episode_manifest(execution_plan.manifest):
        raise LivePilotError("live result does not bind the materialized episode")
    if any(turn.receipt.attempts != 1 for turn in result.trace):
        raise LivePilotError("pilot result records a retry")
    if any(
        turn.output is None or turn.receipt.structured_output_schema_hash != PILOT_TURN_SCHEMA_HASH
        for turn in result.trace
    ):
        raise LivePilotError("pilot result does not bind the turn wire schema")


def _validate_engine_artifact(root: Path, result: EpisodeResult) -> None:
    artifact_ref = result.artifact_ref
    digest = result.content_hash.removeprefix("sha256:")
    if artifact_ref != f"artifacts/episode_result/{digest[:2]}/{digest}.json":
        raise LivePilotError("episode artifact reference is invalid")
    relative = Path(artifact_ref)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise LivePilotError("episode artifact reference escapes the pilot root")
    path = root
    for part in relative.parts:
        path = path / part
        if path.exists() and path.is_symlink():
            raise LivePilotError("episode artifact path contains a symlink")
    artifact = _read_json(path, "episode artifact")
    payload = result.model_dump(mode="json")
    projection = episode_result_projection(payload)
    expected = {
        "content_hash": result.content_hash,
        "artifact_ref": artifact_ref,
        "payload": projection,
    }
    if (
        artifact != expected
        or type(artifact.get("payload")) is not dict
        or sha256_ref(projection) != result.content_hash
        or artifact.get("content_hash") != sha256_ref(artifact.get("payload"))
    ):
        raise LivePilotError("episode artifact does not bind the persisted result")


def _completed_result(
    path: Path,
    plan_hash: str,
    runtime_hash: str,
    cell: LivePilotCell,
    compiled_hash: str | None,
    executor_sampling_temperature: float,
    executor_reasoning_effort: Literal["none", "low"],
    executor_model_name: str | None,
    executor_model_version: str | None,
    execution_plan: EpisodePlan,
    run_directory: Path,
) -> bool:
    envelope = _read_json(path, "pilot result")
    if (
        envelope.get("profile") != PILOT_PROFILE
        or envelope.get("classification") != PILOT_LABEL
        or envelope.get("plan_hash") != plan_hash
        or envelope.get("runtime_hash") != runtime_hash
        or envelope.get("cell") != cell.projection()
        or envelope.get("compiled_skill_hash") != compiled_hash
        or envelope.get("executor_sampling_temperature") != executor_sampling_temperature
        or envelope.get("executor_reasoning_effort") != executor_reasoning_effort
        or envelope.get("executor_model") != executor_model_name
        or envelope.get("executor_model_version") != executor_model_version
        or envelope.get("executor_turn_prompt_profile") != PILOT_TURN_PROMPT_PROFILE
        or envelope.get("executor_turn_prompt_hash") != PILOT_TURN_PROMPT_HASH
        or envelope.get("executor_turn_schema_hash") != PILOT_TURN_SCHEMA_HASH
        or envelope.get("executor_turn_tool_layout_hash")
        != _pilot_turn_layout_hash(cell.domain, cell.condition)
    ):
        raise LivePilotError("stored result does not bind the immutable pilot plan")
    result = parse_episode_result(envelope.get("result"))
    if any(
        turn.output is None or turn.receipt.structured_output_schema_hash != PILOT_TURN_SCHEMA_HASH
        for turn in result.trace
    ):
        raise LivePilotError("stored result does not bind the pilot turn wire")
    _validate_engine_artifact(run_directory, result)
    expected_manifest_hash = hash_episode_manifest(execution_plan.manifest)
    return (
        path.name == f"{execution_plan.manifest.episode_id}.json"
        and result.episode_id == execution_plan.manifest.episode_id
        and result.manifest_hash == expected_manifest_hash
        and envelope.get("execution_manifest_hash") == expected_manifest_hash
    )


async def _run_pilot_episode(
    plan: EpisodePlan, client: ModelClient, descriptor: LivePilotDescriptor
) -> EpisodeResult:
    if (
        client.capabilities != descriptor.executor_capabilities
        or getattr(client, "reasoning_effort", None) != descriptor.executor_reasoning_effort
        or descriptor.executor_transport is not None
        and getattr(client, "transport_profile", None) != descriptor.executor_transport
    ):
        raise LivePilotError(
            "injected live executor does not match the explicit runtime descriptor"
        )
    return await run_episode(
        plan,
        client,
        output_schema=PilotAgentTurnWire,
        temperature=descriptor.executor_sampling_temperature,
    )


async def run_live_pilot(
    *,
    output_dir: Path,
    protocol_path: Path,
    prompt_path: Path,
    descriptor: LivePilotDescriptor,
    limit: int,
    client: ModelClient | None = None,
    role_profile_receipt: dict[str, object] | None = None,
    execute: Callable[[EpisodePlan, ModelClient], Awaitable[EpisodeResult]] | None = None,
    compiler_client: ModelClient | None = None,
    executor_client: ModelClient | None = None,
) -> dict[str, object]:
    """Compile four skills once, then execute a prefix of the immutable 60-cell plan."""

    if type(limit) is not int or not 1 <= limit <= PILOT_LIMIT:
        raise LivePilotError("--limit must be an explicit prefix in 1..60")
    if client is not None and (compiler_client is not None or executor_client is not None):
        raise LivePilotError(
            "pilot accepts either a shared client or explicit compiler and executor clients"
        )
    if client is None and (compiler_client is None or executor_client is None):
        raise LivePilotError("pilot requires explicit compiler and executor clients")
    resolved_compiler_client = compiler_client if compiler_client is not None else client
    resolved_executor_client = executor_client if executor_client is not None else client
    if resolved_compiler_client is None or resolved_executor_client is None:
        raise LivePilotError("pilot requires explicit compiler and executor clients")
    if (
        resolved_executor_client.capabilities != descriptor.executor_capabilities
        or getattr(resolved_executor_client, "reasoning_effort", None)
        != descriptor.executor_reasoning_effort
        or descriptor.executor_transport is not None
        and getattr(resolved_executor_client, "transport_profile", None)
        != descriptor.executor_transport
    ):
        raise LivePilotError(
            "injected live executor does not match the explicit runtime descriptor"
        )
    if not isinstance(protocol_path, Path) or not isinstance(prompt_path, Path):
        raise LivePilotError("pilot inputs must be paths")
    try:
        prompt_bytes = prompt_path.read_bytes()
    except OSError as error:
        raise LivePilotError("compiler prompt is unreadable") from error
    corpus = generate_development_corpus(PILOT_SEED)
    cells = build_live_pilot_plan(corpus)
    run_directory = _safe_directory(output_dir)
    _validate_role_profile_receipt(descriptor, role_profile_receipt)
    pilot_prompt_bytes = _pilot_prompt_bytes(prompt_bytes)
    plan = _plan_projection(
        cells, _protocol_hash(protocol_path), "sha256:" + sha256(pilot_prompt_bytes).hexdigest()
    )
    plan_hash = sha256_ref(plan)
    _write_once(run_directory / "pilot-plan.json", {**plan, "plan_hash": plan_hash})
    runtime = {
        **descriptor.projection(),
        "executor_turn_tool_layout_hashes": _pilot_turn_layout_hashes(cells),
    }
    runtime_hash = sha256_ref(runtime)
    _write_once(run_directory / "runtime.json", {**runtime, "runtime_hash": runtime_hash})
    if role_profile_receipt is not None:
        _write_once(run_directory / "ollama-role-profile-receipt.json", role_profile_receipt)
    skills = await _compiled_skills(
        run_directory,
        descriptor,
        prompt_bytes,
        resolved_compiler_client,
        plan_hash=plan_hash,
        runtime_hash=runtime_hash,
    )
    completed = 0
    skipped = 0
    for cell in cells[:limit]:
        artifact = None if cell.ratio_label is None else skills[(cell.domain, cell.ratio_label)]
        compiled_hash = None if artifact is None else compiled_skill_artifact_hash(artifact)
        execution_plan = _materialize(corpus, cell, descriptor.executor_episode_model, skills)
        result_path = run_directory / "results" / f"{execution_plan.manifest.episode_id}.json"
        if result_path.exists():
            if not _completed_result(
                result_path,
                plan_hash,
                runtime_hash,
                cell,
                compiled_hash,
                descriptor.executor_sampling_temperature,
                descriptor.executor_reasoning_effort,
                descriptor.executor_model_name,
                descriptor.executor_model_version,
                execution_plan,
                run_directory,
            ):
                raise LivePilotError("stored pilot result is invalid")
            skipped += 1
            continue
        previous_cwd = Path.cwd()
        try:
            os.chdir(run_directory)
            result = (
                await _run_pilot_episode(execution_plan, resolved_executor_client, descriptor)
                if execute is None
                else await execute(execution_plan, resolved_executor_client)
            )
        finally:
            os.chdir(previous_cwd)
        _admit_completed_result(result, execution_plan)
        _write_once(
            result_path,
            _result_envelope(
                plan_hash,
                runtime_hash,
                cell,
                result,
                compiled_hash,
                descriptor.executor_sampling_temperature,
                descriptor.executor_reasoning_effort,
                descriptor.executor_model_name,
                descriptor.executor_model_version,
            ),
        )
        completed += 1
    return {
        "classification": PILOT_LABEL,
        "plan_hash": plan_hash,
        "completed": completed,
        "skipped": skipped,
        "limit": limit,
        "total_cells": PILOT_LIMIT,
        "concurrency": PILOT_CONCURRENCY,
        "retries": 0,
    }


def audit_live_pilot(output_dir: Path) -> dict[str, object]:
    """Return raw observed counts only; never a confirmatory interpretation."""

    root = _safe_existing_directory(output_dir)
    plan = _read_json(root / "pilot-plan.json", "pilot plan")
    runtime = _read_json(root / "runtime.json", "pilot runtime")
    if plan.get("profile") != PILOT_PROFILE or runtime.get("profile") != PILOT_PROFILE:
        raise LivePilotError("artifact root is not an exploratory live pilot")
    runtime_hash = runtime.get("runtime_hash")
    runtime_body = {key: value for key, value in runtime.items() if key != "runtime_hash"}
    if type(runtime_hash) is not str or sha256_ref(runtime_body) != runtime_hash:
        raise LivePilotError("pilot runtime hash is invalid")
    expected_compiler_max_tokens = (
        OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS
        if runtime.get("provider_profile")
        in {
            OLLAMA_GPT_OSS_LEGACY_V9_PROFILE,
            OLLAMA_GPT_OSS_LEGACY_V10_PROFILE,
            OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
            OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
            OLLAMA_GPT_OSS_PROFILE,
        }
        else PILOT_COMPILER_MAX_TOKENS
    )
    if (
        runtime.get("classification") != PILOT_LABEL
        or runtime.get("concurrency") != PILOT_CONCURRENCY
        or runtime.get("max_attempts") != PILOT_MAX_ATTEMPTS
        or runtime.get("request_timeout_seconds") != PILOT_REQUEST_TIMEOUT_SECONDS
        or runtime.get("compiler_max_tokens") != expected_compiler_max_tokens
        or runtime.get("compiler_wire_schema_hash") != PILOT_WIRE_SCHEMA_HASH
        or runtime.get("compiler_prompt_profile") != PILOT_COMPILER_PROMPT_PROFILE
        or runtime.get("executor_turn_prompt_profile") != PILOT_TURN_PROMPT_PROFILE
        or runtime.get("executor_turn_prompt_hash") != PILOT_TURN_PROMPT_HASH
        or runtime.get("executor_turn_schema_hash") != PILOT_TURN_SCHEMA_HASH
        or not _runtime_profile_is_valid(runtime)
        or not all(
            type(runtime.get(field)) is str and runtime[field]
            for field in (
                "endpoint_hash",
                "api_key_environment",
                "provider",
                "model",
                "model_version",
            )
        )
    ):
        raise LivePilotError("pilot runtime descriptor is invalid")
    if runtime.get("provider_profile") in {
        OLLAMA_GPT_OSS_LEGACY_V9_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V10_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
        OLLAMA_GPT_OSS_PROFILE,
    }:
        receipt = _read_json(
            root / "ollama-role-profile-receipt.json", "Ollama role profile receipt"
        )
        try:
            validate_ollama_role_profile_receipt(receipt)
        except OllamaRoleProfileError as error:
            raise LivePilotError("Ollama role profile receipt is invalid") from error
        if sha256_ref(receipt) != runtime.get("role_profile_receipt_hash"):
            raise LivePilotError("Ollama role profile receipt does not bind runtime")
        if receipt["role_profile"] != runtime.get("provider_profile"):
            raise LivePilotError("Ollama role profile receipt does not bind provider profile")
        if receipt["profile_model_digest"] != runtime.get("model_version"):
            raise LivePilotError("Ollama role profile receipt does not bind model version")
        if runtime.get("provider_profile") in {
            OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
            OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
            OLLAMA_GPT_OSS_PROFILE,
        } and (
            receipt.get("executor_model_digest") != runtime.get("executor_model_version")
            or receipt.get("executor_model") != runtime.get("executor_model")
        ):
            raise LivePilotError("Ollama role profile receipt does not bind executor model")
    try:
        capabilities = ProviderCapabilities(
            provider=cast(str, runtime["provider"]),
            model=cast(str, runtime["model"]),
            model_version=cast(str, runtime["model_version"]),
            supports_system_role=True,
            supports_developer_role=True,
            supports_seed=True,
            supports_structured_output=True,
        )
        executor_model = ExecutorModel(
            provider=cast(str, runtime["provider"]),
            model=cast(str, runtime.get("executor_model") or runtime["model"]),
            model_version_date=cast(
                str, runtime.get("executor_model_version") or runtime["model_version"]
            ),
        )
    except ValueError as error:
        raise LivePilotError("pilot runtime descriptor is invalid") from error
    plan_hash = plan.get("plan_hash")
    plan_body = {key: value for key, value in plan.items() if key != "plan_hash"}
    if type(plan_hash) is not str or sha256_ref(plan_body) != plan_hash:
        raise LivePilotError("pilot plan hash is invalid")
    cells_value = plan.get("cells")
    corpus = generate_development_corpus(PILOT_SEED)
    expected_plan = build_live_pilot_plan(corpus)
    expected_cells = [cell.projection() for cell in expected_plan]
    expected_turn_layout_hashes = _pilot_turn_layout_hashes(expected_plan)
    if (
        plan.get("classification") != PILOT_LABEL
        or plan.get("seed") != PILOT_SEED
        or plan.get("compiler_prompt_profile") != PILOT_COMPILER_PROMPT_PROFILE
        or plan.get("executor_turn_prompt_profile") != PILOT_TURN_PROMPT_PROFILE
        or plan.get("executor_turn_prompt_hash") != PILOT_TURN_PROMPT_HASH
        or plan.get("executor_turn_schema_hash") != PILOT_TURN_SCHEMA_HASH
        or plan.get("executor_turn_tool_layout_hashes") != expected_turn_layout_hashes
        or runtime.get("executor_turn_tool_layout_hashes") != expected_turn_layout_hashes
        or type(cells_value) is not list
        or cells_value != expected_cells
    ):
        raise LivePilotError("pilot plan does not have 60 cells")
    prompt_hash = plan.get("compiler_prompt_hash")
    if type(prompt_hash) is not str:
        raise LivePilotError("pilot compiler prompt hash is invalid")
    compiled_directory = root / "compiled-skills"
    results_directory = root / "results"
    if (
        compiled_directory.is_symlink()
        or not compiled_directory.is_dir()
        or results_directory.exists()
        and (results_directory.is_symlink() or not results_directory.is_dir())
    ):
        raise LivePilotError("pilot artifact subdirectory is unsafe")
    skills: dict[tuple[str, str], CompiledSkillArtifact] = {}
    for domain in _DOMAINS:
        for ratio_label, ratio in _RATIOS:
            artifact = _load_artifact(
                _read_json(
                    compiled_directory / f"{_compiler_key(domain, ratio_label)}.json",
                    "compiled skill artifact",
                )
            )
            if (
                artifact.compiler_manifest.declared_capabilities != capabilities
                or artifact.compiler_manifest.compiler_input_hash
                != _compiler_input_hash(domain, ratio)
                or artifact.compiler_manifest.system_prompt_raw_hash != prompt_hash
                or artifact.compiler_manifest.max_tokens != expected_compiler_max_tokens
                or artifact.compiler_manifest.temperature != runtime.get("sampling_temperature")
                or artifact.compiler_manifest.prompt_profile != PILOT_COMPILER_PROMPT_PROFILE
                or artifact.compiler_manifest.structured_output_schema_hash
                != PILOT_WIRE_SCHEMA_HASH
                or artifact.attempts != PILOT_MAX_ATTEMPTS
                or not _pilot_wire_round_trip(artifact, domain, ratio)
            ):
                raise LivePilotError("compiled skill does not bind the audited pilot runtime")
            skills[(domain, ratio_label)] = artifact
    rows: dict[tuple[str, str, str], dict[str, int]] = {}
    for result_path in (
        sorted(results_directory.glob("*.json")) if results_directory.exists() else ()
    ):
        envelope = _read_json(result_path, "pilot result")
        if (
            envelope.get("profile") != PILOT_PROFILE
            or envelope.get("plan_hash") != plan_hash
            or envelope.get("runtime_hash") != runtime_hash
            or envelope.get("classification") != PILOT_LABEL
            or envelope.get("executor_sampling_temperature")
            != (
                runtime.get("executor_sampling_temperature")
                if runtime.get("provider_profile") == OLLAMA_GPT_OSS_PROFILE
                else runtime.get("sampling_temperature")
            )
            or envelope.get("executor_reasoning_effort") != runtime.get("executor_reasoning_effort")
            or envelope.get("executor_model") != runtime.get("executor_model")
            or envelope.get("executor_model_version") != runtime.get("executor_model_version")
            or envelope.get("executor_turn_prompt_profile") != PILOT_TURN_PROMPT_PROFILE
            or envelope.get("executor_turn_prompt_hash") != PILOT_TURN_PROMPT_HASH
            or envelope.get("executor_turn_schema_hash") != PILOT_TURN_SCHEMA_HASH
        ):
            raise LivePilotError("pilot result is not bound to this plan")
        raw_cell = envelope.get("cell")
        matching_cells = tuple(cell for cell in expected_plan if cell.projection() == raw_cell)
        if type(raw_cell) is not dict or len(matching_cells) != 1:
            raise LivePilotError("pilot result cell is invalid")
        cell = matching_cells[0]
        if envelope.get("executor_turn_tool_layout_hash") != _pilot_turn_layout_hash(
            cell.domain, cell.condition
        ):
            raise LivePilotError("pilot result turn layout is invalid")
        ratio = cell.ratio_label
        compiled_hash = envelope.get("compiled_skill_hash")
        if ratio is None:
            if compiled_hash is not None:
                raise LivePilotError("reference result unexpectedly binds a skill")
        elif type(compiled_hash) is str:
            if compiled_skill_artifact_hash(skills[(cell.domain, ratio)]) != compiled_hash:
                raise LivePilotError("pilot result compiled-skill binding is invalid")
        else:
            raise LivePilotError("pilot result compiled-skill binding is invalid")
        result = parse_episode_result(envelope.get("result"))
        if any(
            turn.output is None
            or turn.receipt.structured_output_schema_hash != PILOT_TURN_SCHEMA_HASH
            for turn in result.trace
        ):
            raise LivePilotError("pilot result does not bind the turn wire")
        _validate_engine_artifact(root, result)
        execution_plan = _materialize(corpus, cell, executor_model, skills)
        expected_manifest_hash = hash_episode_manifest(execution_plan.manifest)
        if (
            result_path.name != f"{execution_plan.manifest.episode_id}.json"
            or result.episode_id != execution_plan.manifest.episode_id
            or result.manifest_hash != expected_manifest_hash
            or envelope.get("execution_manifest_hash") != expected_manifest_hash
        ):
            raise LivePilotError("pilot result manifest binding is invalid")
        if result.disposition in {
            EpisodeDisposition.MODEL_FAILURE,
            EpisodeDisposition.ENVIRONMENT_FAILURE,
        } or any(turn.receipt.attempts != 1 for turn in result.trace):
            raise LivePilotError("pilot result records an infrastructure failure or retry")
        matched = tuple(case for case in corpus.cases if case.case_id == cell.case_id)
        if len(matched) != 1:
            raise LivePilotError("pilot result case is unavailable")
        score = score_episode(result, matched[0].task_case)
        key = (
            cell.domain,
            cell.condition.value,
            cell.ratio_label or "REFERENCE",
        )
        row = rows.setdefault(
            key, {"observed": 0, "task_completion": 0, "completion_under_policy": 0}
        )
        row["observed"] += 1
        row["task_completion"] += int(score.task_completion)
        row["completion_under_policy"] += int(score.completion_under_policy)
    return {
        "classification": PILOT_LABEL,
        "claim_ceiling": "raw exploratory counts only; not confirmatory evidence",
        "plan_hash": plan_hash,
        "groups": [
            {"domain": key[0], "condition": key[1], "ratio": key[2], **rows[key]}
            for key in sorted(rows)
        ],
    }


__all__ = [
    "LivePilotDescriptor",
    "LivePilotError",
    "PILOT_LABEL",
    "audit_live_pilot",
    "build_live_pilot_plan",
    "run_live_pilot",
]
