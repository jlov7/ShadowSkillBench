from __future__ import annotations

import hashlib
import math
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.models.protocol import (
    Message,
    ModelAdapterError,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    TokenCost,
    TokenUsage,
)
from shadowskillbench.skills.confirmatory_wire import (
    ConfirmatorySkillIRWire,
    confirmatory_wire_to_skill_ir,
)
from shadowskillbench.skills.models import (
    JsonObject,
    SkillIR,
    hash_skill_ir,
    parse_skill_ir,
    skill_ir_projection,
)
from shadowskillbench.skills.pilot_wire import PilotSkillIRWire, pilot_wire_to_skill_ir
from shadowskillbench.skills.projection import (
    CompilerInput,
    compiler_input_projection,
    compiler_view,
    hash_compiler_input,
)
from shadowskillbench.skills.render import hash_rendered_skill, render_skill
from shadowskillbench.traces.bundles import DemonstrationBundle

_CONFIG_FIELDS = ("prompt_profile", "prompt_bytes", "temperature", "seed", "max_tokens")
_MANIFEST_FIELDS = (
    "manifest_profile",
    "compiler_input_hash",
    "prompt_profile",
    "system_prompt_raw_hash",
    "request_profile",
    "instruction_provenance_profile",
    "structured_output_schema_hash",
    "declared_capabilities",
    "temperature",
    "seed",
    "max_tokens",
)
_ARTIFACT_FIELDS = (
    "artifact_profile",
    "compiler_manifest",
    "compiler_manifest_hash",
    "request_envelope_hash",
    "skill_ir",
    "skill_ir_hash",
    "rendered_skill",
    "rendered_skill_hash",
    "raw_request_hash",
    "raw_response_hash",
    "structured_output_schema_hash",
    "usage",
    "cost",
    "attempts",
)
_CAPABILITY_FIELDS = (
    "provider",
    "model",
    "model_version",
    "supports_system_role",
    "supports_developer_role",
    "supports_seed",
    "supports_structured_output",
)
_MANIFEST_PROFILE = "SSB-COMPILER-MANIFEST1"
_PROMPT_PROFILE = "SSB-SKILL-COMPILER1"
PILOT_WIRE_PROMPT_PROFILE = "SSB-PILOT-WIRE-COMPILER1"
CONFIRMATORY_WIRE_PROMPT_PROFILE = "SSB-CONFIRMATORY-COMPACT-WIRE-COMPILER1"
_REQUEST_PROFILE = "SSB-SKILL-COMPILER-REQUEST1"
_PROVENANCE_PROFILE = "SSB-INSTRUCTION-PROVENANCE1"
_ARTIFACT_PROFILE = "SSB-COMPILED-SKILL1"
_HASH_PREFIX = "sha256:"
_MAX_I64 = 2**63 - 1
_ADMITTED_OUTPUT_SCHEMAS: tuple[type[BaseModel], ...] = (
    SkillIR,
    PilotSkillIRWire,
    ConfirmatorySkillIRWire,
)

type CompilerPromptProfile = Literal[
    "SSB-SKILL-COMPILER1",
    "SSB-PILOT-WIRE-COMPILER1",
    "SSB-CONFIRMATORY-COMPACT-WIRE-COMPILER1",
]


def _hash_ref(value: object) -> str:
    if type(value) is not str or not value.startswith(_HASH_PREFIX) or len(value) != 71:
        raise ValueError("hash reference is invalid")
    if any(character not in "0123456789abcdef" for character in value[len(_HASH_PREFIX) :]):
        raise ValueError("hash reference is invalid")
    return cast(str, value)


def _raw_bytes_hash(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


type CompilerFailureStage = Literal[
    "response_bindings",
    "pilot_wire_conversion",
    "output_bindings",
    "rendering",
    "artifact_construction",
]
type CompilerValidationCode = Literal[
    "RESPONSE_BINDING_FAILED",
    "PILOT_WIRE_EVENT_INDEX_OUT_OF_RANGE",
    "PILOT_WIRE_DUPLICATE_ARGUMENT_KEY",
    "CANONICAL_SKILL_VALIDATION_FAILED",
    "OUTPUT_BINDING_FAILED",
    "RENDERING_FAILED",
    "ARTIFACT_CONSTRUCTION_FAILED",
    "UNCLASSIFIED",
]


class CompilerContractError(RuntimeError):
    __slots__ = (
        "code",
        "stage",
        "validation_code",
        "validation_paths",
        "raw_request_hash",
        "raw_response_hash",
        "finish_reason",
        "finish_reason_source",
        "reported_usage",
        "attempts",
        "_sealed",
    )

    def __init__(
        self,
        code: str,
        *,
        stage: CompilerFailureStage | None = None,
        validation_code: CompilerValidationCode | None = None,
        validation_paths: tuple[str, ...] = (),
        raw_request_hash: str | None = None,
        raw_response_hash: str | None = None,
        finish_reason: Literal["stop"] | None = None,
        finish_reason_source: Literal["adapter_success_contract"] | None = None,
        reported_usage: TokenUsage | None = None,
        attempts: int | None = None,
    ) -> None:
        if (
            type(self) is not CompilerContractError
            or type(code) is not str
            or code
            not in {
                "COMPILER_CONFIGURATION_ERROR",
                "COMPILER_REQUEST_INVALID",
                "COMPILER_OUTPUT_INVALID",
            }
        ):
            raise ValueError("invalid compiler contract error")
        diagnostics = (
            stage,
            validation_code,
            raw_request_hash,
            raw_response_hash,
            finish_reason,
            finish_reason_source,
            reported_usage,
            attempts,
        )
        if any(value is not None for value in diagnostics):
            if (
                code != "COMPILER_OUTPUT_INVALID"
                or stage is None
                or validation_code is None
                or type(raw_request_hash) is not str
                or type(raw_response_hash) is not str
                or finish_reason != "stop"
                or finish_reason_source != "adapter_success_contract"
                or type(reported_usage) is not TokenUsage
                or type(attempts) is not int
                or attempts < 1
                or attempts > 5
            ):
                raise ValueError("invalid compiler failure diagnostics")
            _hash_ref(raw_request_hash)
            _hash_ref(raw_response_hash)
        if type(validation_paths) is not tuple or any(
            type(path) is not str or not path.startswith("/") or len(path) > 256
            for path in validation_paths
        ):
            raise ValueError("invalid compiler validation paths")
        if validation_paths and stage is None:
            raise ValueError("compiler validation paths require diagnostics")
        super().__init__(code)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "validation_code", validation_code)
        object.__setattr__(self, "validation_paths", validation_paths)
        object.__setattr__(self, "raw_request_hash", raw_request_hash)
        object.__setattr__(self, "raw_response_hash", raw_response_hash)
        object.__setattr__(self, "finish_reason", finish_reason)
        object.__setattr__(self, "finish_reason_source", finish_reason_source)
        object.__setattr__(self, "reported_usage", reported_usage)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"__traceback__", "__cause__", "__context__", "__suppress_context__"}:
            BaseException.__setattr__(self, name, value)
            return
        if getattr(self, "_sealed", False):
            raise AttributeError("CompilerContractError is immutable")
        object.__setattr__(self, name, value)

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return f"CompilerContractError({self.code!r})"


def _validation_paths(error: Exception) -> tuple[str, ...]:
    if not isinstance(error, ValidationError):
        return ()
    paths: list[str] = []
    for record in error.errors(include_url=False, include_context=False, include_input=False):
        components = [
            str(component)
            for component in record["loc"]
            if type(component) is int
            or (type(component) is str and component.replace("_", "").isalnum())
        ]
        path = "/" + "/".join(components)
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _compiler_output_error(
    response: ModelResponse[BaseModel],
    stage: CompilerFailureStage,
    error: Exception,
) -> CompilerContractError:
    if stage == "response_bindings":
        validation_code: CompilerValidationCode = "RESPONSE_BINDING_FAILED"
    elif stage == "pilot_wire_conversion":
        if type(error) is ValueError and str(error) == "pilot wire event indices are invalid":
            validation_code = "PILOT_WIRE_EVENT_INDEX_OUT_OF_RANGE"
        elif (
            type(error) is ValueError
            and str(error) == "pilot wire argument binding keys must be unique"
        ):
            validation_code = "PILOT_WIRE_DUPLICATE_ARGUMENT_KEY"
        elif isinstance(error, ValidationError):
            validation_code = "CANONICAL_SKILL_VALIDATION_FAILED"
        else:
            validation_code = "UNCLASSIFIED"
    elif stage == "output_bindings":
        validation_code = "OUTPUT_BINDING_FAILED"
    elif stage == "rendering":
        validation_code = "RENDERING_FAILED"
    else:
        validation_code = "ARTIFACT_CONSTRUCTION_FAILED"
    return CompilerContractError(
        "COMPILER_OUTPUT_INVALID",
        stage=stage,
        validation_code=validation_code,
        validation_paths=_validation_paths(error),
        raw_request_hash=response.raw_request_hash,
        raw_response_hash=response.raw_response_hash,
        finish_reason="stop",
        finish_reason_source="adapter_success_contract",
        reported_usage=response.usage,
        attempts=response.attempts,
    )


class _CompilerValue(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class CompilerConfig(_CompilerValue):
    prompt_profile: CompilerPromptProfile
    prompt_bytes: bytes
    temperature: float
    seed: int | None
    max_tokens: int

    @field_validator("prompt_profile", mode="before")
    @classmethod
    def _prompt_profile(cls, value: object) -> CompilerPromptProfile:
        if type(value) is not str or value not in {
            _PROMPT_PROFILE,
            PILOT_WIRE_PROMPT_PROFILE,
            CONFIRMATORY_WIRE_PROMPT_PROFILE,
        }:
            raise ValueError("prompt_profile is invalid")
        return cast(CompilerPromptProfile, value)

    @field_validator("prompt_bytes", mode="before")
    @classmethod
    def _prompt_bytes(cls, value: object) -> bytes:
        if type(value) is not bytes or not value or len(value) > 65_536:
            raise ValueError("prompt_bytes is invalid")
        if (
            b"\x00" in value
            or b"\r" in value
            or not value.endswith(b"\n")
            or value.endswith(b"\n\n")
        ):
            raise ValueError("prompt_bytes must be LF-only with one terminal LF")
        try:
            value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("prompt_bytes must be UTF-8") from error
        return bytes(value)

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature(cls, value: object) -> float:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 2.0
            or value == 0.0
            and math.copysign(1.0, value) < 0
        ):
            raise ValueError("temperature is invalid")
        return cast(float, value)

    @field_validator("seed", mode="before")
    @classmethod
    def _seed(cls, value: object) -> int | None:
        if value is None:
            return None
        if type(value) is not int or not -(2**63) <= value <= _MAX_I64:
            raise ValueError("seed is invalid")
        return cast(int, value)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _max_tokens(cls, value: object) -> int:
        if type(value) is not int or not 1 <= value <= 1_000_000:
            raise ValueError("max_tokens is invalid")
        return cast(int, value)


class CompilerManifest(_CompilerValue):
    manifest_profile: Literal["SSB-COMPILER-MANIFEST1"]
    compiler_input_hash: str
    prompt_profile: CompilerPromptProfile
    system_prompt_raw_hash: str
    request_profile: Literal["SSB-SKILL-COMPILER-REQUEST1"]
    instruction_provenance_profile: Literal["SSB-INSTRUCTION-PROVENANCE1"]
    structured_output_schema_hash: str
    declared_capabilities: ProviderCapabilities
    temperature: float
    seed: int | None
    max_tokens: int

    @field_validator(
        "compiler_input_hash",
        "system_prompt_raw_hash",
        "structured_output_schema_hash",
        mode="before",
    )
    @classmethod
    def _hash(cls, value: object) -> str:
        return _hash_ref(value)

    @field_validator("manifest_profile", mode="before")
    @classmethod
    def _manifest_profile(cls, value: object) -> Literal["SSB-COMPILER-MANIFEST1"]:
        if type(value) is not str or value != _MANIFEST_PROFILE:
            raise ValueError("manifest_profile is invalid")
        return _MANIFEST_PROFILE

    @field_validator("prompt_profile", mode="before")
    @classmethod
    def _prompt_profile(cls, value: object) -> CompilerPromptProfile:
        return CompilerConfig._prompt_profile(value)

    @field_validator("request_profile", mode="before")
    @classmethod
    def _request_profile(cls, value: object) -> Literal["SSB-SKILL-COMPILER-REQUEST1"]:
        if type(value) is not str or value != _REQUEST_PROFILE:
            raise ValueError("request_profile is invalid")
        return _REQUEST_PROFILE

    @field_validator("instruction_provenance_profile", mode="before")
    @classmethod
    def _provenance_profile(cls, value: object) -> Literal["SSB-INSTRUCTION-PROVENANCE1"]:
        if type(value) is not str or value != _PROVENANCE_PROFILE:
            raise ValueError("instruction_provenance_profile is invalid")
        return _PROVENANCE_PROFILE

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature(cls, value: object) -> float:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 2.0
            or value == 0.0
            and math.copysign(1.0, value) < 0
        ):
            raise ValueError("temperature is invalid")
        return cast(float, value)

    @field_validator("seed", mode="before")
    @classmethod
    def _seed(cls, value: object) -> int | None:
        if value is None:
            return None
        if type(value) is not int or not -(2**63) <= value <= _MAX_I64:
            raise ValueError("seed is invalid")
        return cast(int, value)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _max_tokens(cls, value: object) -> int:
        if type(value) is not int or not 1 <= value <= 1_000_000:
            raise ValueError("max_tokens is invalid")
        return cast(int, value)

    @field_validator("declared_capabilities", mode="before")
    @classmethod
    def _capabilities(cls, value: object) -> ProviderCapabilities:
        return _validated_capabilities(value)


class CompiledSkillArtifact(_CompilerValue):
    artifact_profile: Literal["SSB-COMPILED-SKILL1"]
    compiler_manifest: CompilerManifest
    compiler_manifest_hash: str
    request_envelope_hash: str
    skill_ir: SkillIR
    skill_ir_hash: str
    rendered_skill: str
    rendered_skill_hash: str
    raw_request_hash: str
    raw_response_hash: str
    structured_output_schema_hash: str
    usage: TokenUsage
    cost: TokenCost | None
    attempts: int

    @field_validator(
        "compiler_manifest_hash",
        "request_envelope_hash",
        "skill_ir_hash",
        "rendered_skill_hash",
        "raw_request_hash",
        "raw_response_hash",
        "structured_output_schema_hash",
        mode="before",
    )
    @classmethod
    def _hash(cls, value: object) -> str:
        return _hash_ref(value)

    @field_validator("artifact_profile", mode="before")
    @classmethod
    def _artifact_profile(cls, value: object) -> Literal["SSB-COMPILED-SKILL1"]:
        if type(value) is not str or value != _ARTIFACT_PROFILE:
            raise ValueError("artifact_profile is invalid")
        return _ARTIFACT_PROFILE

    @field_validator("compiler_manifest", mode="before")
    @classmethod
    def _manifest(cls, value: object) -> CompilerManifest:
        return _validated_manifest(value)

    @field_validator("skill_ir", mode="before")
    @classmethod
    def _skill_ir(cls, value: object) -> SkillIR:
        return parse_skill_ir(value)

    @field_validator("rendered_skill", mode="before")
    @classmethod
    def _rendered_skill(cls, value: object) -> str:
        if (
            type(value) is not str
            or "\x00" in value
            or "\r" in value
            or not value.endswith("\n")
            or value.endswith("\n\n")
        ):
            raise ValueError("rendered_skill is invalid")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("rendered_skill must be UTF-8") from error
        return value

    @field_validator("usage", mode="before")
    @classmethod
    def _usage(cls, value: object) -> TokenUsage:
        return _validated_value(value, TokenUsage)

    @field_validator("cost", mode="before")
    @classmethod
    def _cost(cls, value: object) -> TokenCost | None:
        if value is None:
            return None
        return _validated_value(value, TokenCost)

    @field_validator("attempts", mode="before")
    @classmethod
    def _attempts(cls, value: object) -> int:
        if type(value) is not int or not 1 <= value <= 5:
            raise ValueError("attempts is invalid")
        return cast(int, value)

    @model_validator(mode="after")
    def _bindings(self) -> CompiledSkillArtifact:
        if self.compiler_manifest_hash != compiler_manifest_hash(self.compiler_manifest):
            raise ValueError("compiler_manifest_hash is invalid")
        if (
            self.structured_output_schema_hash
            != self.compiler_manifest.structured_output_schema_hash
        ):
            raise ValueError("structured_output_schema_hash is invalid")
        if self.skill_ir.compiler_manifest_ref != self.compiler_manifest_hash:
            raise ValueError("SkillIR compiler manifest binding is invalid")
        provenance = self.skill_ir.instruction_provenance
        if (
            type(provenance) is not dict
            or set(provenance) != {"profile", "compiler_input_hash", "instruction_evidence"}
            or type(provenance["instruction_evidence"]) is not dict
            or provenance["profile"] != self.compiler_manifest.instruction_provenance_profile
            or provenance["compiler_input_hash"] != self.compiler_manifest.compiler_input_hash
        ):
            raise ValueError("SkillIR instruction provenance bindings are invalid")
        if self.skill_ir_hash != hash_skill_ir(self.skill_ir):
            raise ValueError("skill_ir_hash is invalid")
        if self.rendered_skill != render_skill(self.skill_ir):
            raise ValueError("rendered_skill is invalid")
        if self.rendered_skill_hash != hash_rendered_skill(self.skill_ir):
            raise ValueError("rendered_skill_hash is invalid")
        if self.cost is not None and (
            self.cost.input_nanos != self.usage.input_tokens * self.cost.input_nanos_per_token
            or self.cost.output_nanos != self.usage.output_tokens * self.cost.output_nanos_per_token
            or self.cost.total_nanos != self.cost.input_nanos + self.cost.output_nanos
        ):
            raise ValueError("cost does not bind usage and rates")
        return self


def _exact_model_data(
    value: object, expected: type[BaseModel], fields: tuple[str, ...]
) -> dict[str, object]:
    if type(value) is not expected:
        raise ValueError("value has an invalid exact model class")
    try:
        data = object.__getattribute__(value, "__dict__")
        field_set = object.__getattribute__(value, "__pydantic_fields_set__")
        extra = object.__getattribute__(value, "__pydantic_extra__")
        private = object.__getattribute__(value, "__pydantic_private__")
    except AttributeError as error:
        raise ValueError("value is incomplete or forged") from error
    if (
        type(data) is not dict
        or type(field_set) is not set
        or extra is not None
        or private is not None
        or len(data) != len(fields)
        or len(field_set) != len(fields)
        or any(type(name) is not str or name not in fields for name in data)
        or any(type(name) is not str or name not in fields for name in field_set)
    ):
        raise ValueError("value is incomplete or forged")
    return {name: data[name] for name in fields}


def _validated_value(value: object, expected: type[BaseModel]) -> Any:
    fields = tuple(expected.model_fields)
    return expected.model_validate(_exact_model_data(value, expected, fields))


def _validated_capabilities(value: object) -> ProviderCapabilities:
    return cast(ProviderCapabilities, _validated_value(value, ProviderCapabilities))


def _validated_config(value: object) -> CompilerConfig:
    return cast(
        CompilerConfig,
        CompilerConfig.model_validate(_exact_model_data(value, CompilerConfig, _CONFIG_FIELDS)),
    )


def _validated_manifest(value: object) -> CompilerManifest:
    return cast(
        CompilerManifest,
        CompilerManifest.model_validate(
            _exact_model_data(value, CompilerManifest, _MANIFEST_FIELDS)
        ),
    )


def _validated_artifact(value: object) -> CompiledSkillArtifact:
    return cast(
        CompiledSkillArtifact,
        CompiledSkillArtifact.model_validate(
            _exact_model_data(value, CompiledSkillArtifact, _ARTIFACT_FIELDS)
        ),
    )


def parse_compiled_skill_artifact(value: object) -> CompiledSkillArtifact:
    """Hydrate one canonical JSON artifact through its exact nested model boundary."""

    if type(value) is not dict or set(value) != set(_ARTIFACT_FIELDS):
        raise ValueError("compiled skill artifact is invalid")
    raw = cast(dict[str, object], value)
    try:
        manifest_raw = cast(dict[str, object], raw["compiler_manifest"])
        if type(manifest_raw) is not dict or set(manifest_raw) != set(_MANIFEST_FIELDS):
            raise ValueError("compiler manifest is invalid")
        capabilities = ProviderCapabilities.model_validate(
            cast(Any, manifest_raw["declared_capabilities"])
        )
        manifest = CompilerManifest(
            **cast(Any, {**manifest_raw, "declared_capabilities": capabilities})
        )
        usage = TokenUsage.model_validate(cast(Any, raw["usage"]))
        cost_raw = raw["cost"]
        cost = None if cost_raw is None else TokenCost.model_validate(cast(Any, cost_raw))
        return CompiledSkillArtifact(
            **cast(
                Any,
                {
                    **raw,
                    "compiler_manifest": manifest,
                    "skill_ir": parse_skill_ir(raw["skill_ir"]),
                    "usage": usage,
                    "cost": cost,
                },
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("compiled skill artifact is invalid") from error


def _capabilities_projection(value: ProviderCapabilities) -> JsonObject:
    capabilities = _validated_capabilities(value)
    data = _exact_model_data(capabilities, ProviderCapabilities, _CAPABILITY_FIELDS)
    return cast(JsonObject, {name: data[name] for name in _CAPABILITY_FIELDS})


def compiler_manifest_projection(value: object) -> JsonObject:
    manifest = _validated_manifest(value)
    data = _exact_model_data(manifest, CompilerManifest, _MANIFEST_FIELDS)
    return cast(
        JsonObject,
        {
            "manifest_profile": data["manifest_profile"],
            "compiler_input_hash": data["compiler_input_hash"],
            "prompt_profile": data["prompt_profile"],
            "system_prompt_raw_hash": data["system_prompt_raw_hash"],
            "request_profile": data["request_profile"],
            "instruction_provenance_profile": data["instruction_provenance_profile"],
            "structured_output_schema_hash": data["structured_output_schema_hash"],
            "declared_capabilities": _capabilities_projection(
                cast(ProviderCapabilities, data["declared_capabilities"])
            ),
            "temperature": data["temperature"],
            "seed": data["seed"],
            "max_tokens": data["max_tokens"],
        },
    )


def compiler_manifest_hash(value: object) -> str:
    return sha256_ref(compiler_manifest_projection(value))


def _usage_projection(value: TokenUsage) -> JsonObject:
    usage = cast(TokenUsage, _validated_value(value, TokenUsage))
    data = _exact_model_data(usage, TokenUsage, tuple(TokenUsage.model_fields))
    return cast(JsonObject, data)


def _cost_projection(value: TokenCost | None) -> JsonObject | None:
    if value is None:
        return None
    cost = cast(TokenCost, _validated_value(value, TokenCost))
    data = _exact_model_data(cost, TokenCost, tuple(TokenCost.model_fields))
    return cast(JsonObject, data)


def compiled_skill_artifact_projection(value: object) -> JsonObject:
    artifact = _validated_artifact(value)
    data = _exact_model_data(artifact, CompiledSkillArtifact, _ARTIFACT_FIELDS)
    return cast(
        JsonObject,
        {
            "artifact_profile": data["artifact_profile"],
            "compiler_manifest": compiler_manifest_projection(data["compiler_manifest"]),
            "compiler_manifest_hash": data["compiler_manifest_hash"],
            "request_envelope_hash": data["request_envelope_hash"],
            "skill_ir": skill_ir_projection(data["skill_ir"]),
            "skill_ir_hash": data["skill_ir_hash"],
            "rendered_skill": data["rendered_skill"],
            "rendered_skill_hash": data["rendered_skill_hash"],
            "raw_request_hash": data["raw_request_hash"],
            "raw_response_hash": data["raw_response_hash"],
            "structured_output_schema_hash": data["structured_output_schema_hash"],
            "usage": _usage_projection(cast(TokenUsage, data["usage"])),
            "cost": _cost_projection(cast(TokenCost | None, data["cost"])),
            "attempts": data["attempts"],
        },
    )


def compiled_skill_artifact_hash(value: object) -> str:
    return sha256_ref(compiled_skill_artifact_projection(value))


def _output_schema_hash(output_schema: type[BaseModel]) -> str:
    if not any(output_schema is admitted for admitted in _ADMITTED_OUTPUT_SCHEMAS):
        raise ValueError("output schema is invalid")
    schema = output_schema.model_json_schema()
    if type(schema) is not dict:
        raise ValueError("output schema is invalid")
    canonical_json_bytes(schema)
    return sha256_ref(schema)


def _expected_prompt_profile(output_schema: type[BaseModel]) -> CompilerPromptProfile:
    if output_schema is SkillIR:
        return _PROMPT_PROFILE
    if output_schema is PilotSkillIRWire:
        return PILOT_WIRE_PROMPT_PROFILE
    if output_schema is ConfirmatorySkillIRWire:
        return CONFIRMATORY_WIRE_PROMPT_PROFILE
    raise ValueError("output schema is invalid")


def _request_envelope(
    compiler_input_data: JsonObject,
    compiler_input_hash: str,
    compiler_manifest_ref: str,
) -> JsonObject:
    return {
        "request_profile": _REQUEST_PROFILE,
        "compiler_input": compiler_input_data,
        "required_output_bindings": {
            "compiler_manifest_ref": compiler_manifest_ref,
            "compiler_input_hash": compiler_input_hash,
            "instruction_provenance_profile": _PROVENANCE_PROFILE,
        },
    }


def _validate_declared_capabilities(value: object, config: CompilerConfig) -> ProviderCapabilities:
    capabilities = _validated_capabilities(value)
    if (
        not capabilities.supports_system_role
        or not capabilities.supports_structured_output
        or config.seed is not None
        and not capabilities.supports_seed
    ):
        raise ValueError("compiler capabilities are insufficient")
    return capabilities


def _validated_response(value: object, output_schema: type[BaseModel]) -> ModelResponse[BaseModel]:
    response_type = ModelResponse[output_schema]
    if type(value) is not response_type:
        raise ValueError("response has an invalid output schema")
    return cast(ModelResponse[BaseModel], response_type.model_validate(value))


def _expected_provenance_keys(skill: SkillIR) -> set[str]:
    keys = {"/objective"}
    for field in (
        "applicability",
        "required_inputs",
        "preconditions",
        "decision_hints",
        "verification_steps",
        "stop_conditions",
        "escalation_hints",
    ):
        for index, _ in enumerate(cast(tuple[str, ...], getattr(skill, field))):
            keys.add(f"/{field}/{index}")
    for index, _ in enumerate(skill.ordered_steps):
        keys.add(f"/ordered_steps/{index}")
    return keys


def _validate_output_bindings(
    skill: SkillIR,
    compiler_input: CompilerInput,
    compiler_input_hash: str,
    compiler_manifest_ref: str,
) -> SkillIR:
    parsed = parse_skill_ir(skill)
    if (
        parsed.domain != compiler_input.domain
        or parsed.skill_id != f"skill_{compiler_input.domain}"
    ):
        raise ValueError("SkillIR domain binding is invalid")
    if parsed.source_trace_ids != tuple(trace.trace_id for trace in compiler_input.traces):
        raise ValueError("SkillIR source trace binding is invalid")
    if parsed.compiler_manifest_ref != compiler_manifest_ref:
        raise ValueError("SkillIR compiler manifest binding is invalid")
    provenance = parsed.instruction_provenance
    if type(provenance) is not dict or set(provenance) != {
        "profile",
        "compiler_input_hash",
        "instruction_evidence",
    }:
        raise ValueError("instruction provenance is invalid")
    if (
        provenance["profile"] != _PROVENANCE_PROFILE
        or provenance["compiler_input_hash"] != compiler_input_hash
    ):
        raise ValueError("instruction provenance bindings are invalid")
    evidence = provenance["instruction_evidence"]
    if type(evidence) is not dict or set(evidence) != _expected_provenance_keys(parsed):
        raise ValueError("instruction evidence keys are invalid")
    event_ids = {event.event_id for trace in compiler_input.traces for event in trace.events}
    for index, step in enumerate(parsed.ordered_steps):
        references = evidence[f"/ordered_steps/{index}"]
        if (
            type(references) is not list
            or not references
            or len(references) != len(set(references))
        ):
            raise ValueError("instruction evidence references are invalid")
        if tuple(references) != step.evidence_refs or any(
            type(reference) is not str or reference not in event_ids for reference in references
        ):
            raise ValueError("step evidence references are invalid")
    for pointer, references in evidence.items():
        if pointer.startswith("/ordered_steps/"):
            continue
        if (
            type(references) is not list
            or not references
            or len(references) != len(set(references))
        ):
            raise ValueError("instruction evidence references are invalid")
        if any(
            type(reference) is not str or reference not in event_ids for reference in references
        ):
            raise ValueError("instruction evidence references are invalid")
    return parsed


async def compile_skill(
    bundle: DemonstrationBundle,
    client: ModelClient,
    config: CompilerConfig,
    *,
    output_schema: type[BaseModel] = SkillIR,
) -> CompiledSkillArtifact:
    configuration: tuple[CompilerConfig, ProviderCapabilities] | None = None
    try:
        admitted_config = _validated_config(config)
        declared_capabilities = _validate_declared_capabilities(
            client.capabilities, admitted_config
        )
        configuration = (admitted_config, declared_capabilities)
    except Exception:
        pass
    if configuration is None:
        raise CompilerContractError("COMPILER_CONFIGURATION_ERROR")
    admitted_config, declared_capabilities = configuration

    prepared_request: (
        tuple[CompilerInput, str, str, CompilerManifest, str, JsonObject, ModelRequest] | None
    ) = None
    try:
        compiler_input = compiler_view(bundle)
        compiler_input_data = compiler_input_projection(compiler_input)
        compiler_input_hash = hash_compiler_input(compiler_input)
        schema_hash = _output_schema_hash(output_schema)
        if admitted_config.prompt_profile != _expected_prompt_profile(output_schema):
            raise ValueError("compiler prompt profile does not match output schema")
        manifest = CompilerManifest(
            manifest_profile=_MANIFEST_PROFILE,
            compiler_input_hash=compiler_input_hash,
            prompt_profile=admitted_config.prompt_profile,
            system_prompt_raw_hash=_raw_bytes_hash(admitted_config.prompt_bytes),
            request_profile=_REQUEST_PROFILE,
            instruction_provenance_profile=_PROVENANCE_PROFILE,
            structured_output_schema_hash=schema_hash,
            declared_capabilities=declared_capabilities,
            temperature=admitted_config.temperature,
            seed=admitted_config.seed,
            max_tokens=admitted_config.max_tokens,
        )
        manifest_hash = compiler_manifest_hash(manifest)
        envelope = _request_envelope(compiler_input_data, compiler_input_hash, manifest_hash)
        user_content = canonical_json_bytes(envelope).decode("utf-8")
        system_content = admitted_config.prompt_bytes.decode("utf-8")
        request = ModelRequest(
            messages=(
                Message(role="system", content=system_content),
                Message(role="user", content=user_content),
            ),
            temperature=admitted_config.temperature,
            seed=admitted_config.seed,
            max_tokens=admitted_config.max_tokens,
        )
        prepared_request = (
            compiler_input,
            compiler_input_hash,
            schema_hash,
            manifest,
            manifest_hash,
            envelope,
            request,
        )
    except Exception:
        pass
    if prepared_request is None:
        raise CompilerContractError("COMPILER_REQUEST_INVALID")
    (
        compiler_input,
        compiler_input_hash,
        schema_hash,
        manifest,
        manifest_hash,
        envelope,
        request,
    ) = prepared_request

    adapter_response: tuple[ModelResponse[BaseModel]] | None = None
    try:
        adapter_response = (
            cast(ModelResponse[BaseModel], await client.structured(request, output_schema)),
        )
    except ModelAdapterError:
        raise
    except Exception:
        pass
    if adapter_response is None:
        raise CompilerContractError("COMPILER_OUTPUT_INVALID")
    (response_value,) = adapter_response
    response = _response_or_error(response_value, output_schema)

    failure: CompilerContractError | None = None
    try:
        if (
            response.capabilities != declared_capabilities
            or response.structured_output_schema_hash != schema_hash
        ):
            raise ValueError("response bindings are invalid")
    except Exception as error:
        failure = _compiler_output_error(response, "response_bindings", error)
    if failure is not None:
        raise failure

    output: SkillIR | None = None
    try:
        output = _canonical_output(output_schema, response.output, compiler_input, manifest_hash)
    except Exception as error:
        failure = _compiler_output_error(response, "pilot_wire_conversion", error)
    if failure is not None:
        raise failure
    if output is None:
        raise CompilerContractError("COMPILER_OUTPUT_INVALID")

    skill: SkillIR | None = None
    try:
        skill = _validate_output_bindings(
            output,
            compiler_input,
            compiler_input_hash,
            manifest_hash,
        )
    except Exception as error:
        failure = _compiler_output_error(response, "output_bindings", error)
    if failure is not None:
        raise failure
    if skill is None:
        raise CompilerContractError("COMPILER_OUTPUT_INVALID")

    rendered: str | None = None
    try:
        rendered = render_skill(skill)
    except Exception as error:
        failure = _compiler_output_error(response, "rendering", error)
    if failure is not None:
        raise failure
    if rendered is None:
        raise CompilerContractError("COMPILER_OUTPUT_INVALID")

    artifact: CompiledSkillArtifact | None = None
    try:
        artifact = CompiledSkillArtifact(
            artifact_profile=_ARTIFACT_PROFILE,
            compiler_manifest=manifest,
            compiler_manifest_hash=manifest_hash,
            request_envelope_hash=sha256_ref(envelope),
            skill_ir=skill,
            skill_ir_hash=hash_skill_ir(skill),
            rendered_skill=rendered,
            rendered_skill_hash=hash_rendered_skill(skill),
            raw_request_hash=response.raw_request_hash,
            raw_response_hash=response.raw_response_hash,
            structured_output_schema_hash=response.structured_output_schema_hash,
            usage=response.usage,
            cost=response.cost,
            attempts=response.attempts,
        )
    except Exception as error:
        failure = _compiler_output_error(response, "artifact_construction", error)
    if failure is not None:
        raise failure
    if artifact is None:
        raise CompilerContractError("COMPILER_OUTPUT_INVALID")
    return artifact


def _response_or_error(value: object, output_schema: type[BaseModel]) -> ModelResponse[BaseModel]:
    response: ModelResponse[BaseModel] | None = None
    try:
        response = _validated_response(value, output_schema)
    except Exception:
        pass
    if response is None:
        raise CompilerContractError("COMPILER_OUTPUT_INVALID")
    return response


def _canonical_output(
    output_schema: type[BaseModel],
    value: BaseModel,
    compiler_input: CompilerInput,
    compiler_manifest_ref: str,
) -> SkillIR:
    if output_schema is SkillIR:
        return cast(SkillIR, value)
    if output_schema is PilotSkillIRWire:
        return pilot_wire_to_skill_ir(
            cast(PilotSkillIRWire, value), compiler_input, compiler_manifest_ref
        )
    if output_schema is ConfirmatorySkillIRWire:
        return confirmatory_wire_to_skill_ir(
            cast(ConfirmatorySkillIRWire, value), compiler_input, compiler_manifest_ref
        )
    raise ValueError("output schema is invalid")


__all__ = [
    "CompiledSkillArtifact",
    "CompilerConfig",
    "CompilerContractError",
    "CompilerManifest",
    "CONFIRMATORY_WIRE_PROMPT_PROFILE",
    "PILOT_WIRE_PROMPT_PROFILE",
    "compile_skill",
    "compiled_skill_artifact_hash",
    "compiled_skill_artifact_projection",
    "compiler_manifest_hash",
    "compiler_manifest_projection",
    "parse_compiled_skill_artifact",
]
