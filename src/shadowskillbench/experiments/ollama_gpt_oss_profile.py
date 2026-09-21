"""Pinned, role-faithful local Ollama profile for the exploratory pilot only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.models.protocol import Message

OLLAMA_GPT_OSS_LEGACY_V9_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES4"
OLLAMA_GPT_OSS_LEGACY_V10_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES5"
OLLAMA_GPT_OSS_LEGACY_V11_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES6"
OLLAMA_GPT_OSS_LEGACY_V12_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES7"
OLLAMA_GPT_OSS_LEGACY_V14_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES8"
OLLAMA_GPT_OSS_LEGACY_V15_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES9"
OLLAMA_GPT_OSS_LEGACY_V16_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES10"
OLLAMA_GPT_OSS_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES11"
OLLAMA_GPT_OSS_PROVIDER = "ollama-gpt-oss-20b-harmony-roles"
OLLAMA_GPT_OSS_SERVER_VERSION = "0.33.2"
OLLAMA_GPT_OSS_MODEL = "ssb-gpt-oss-20b-harmony-roles:v1"
OLLAMA_GPT_OSS_EXECUTOR_MODEL = "ssb-gpt-oss-20b-harmony-final:v2"
OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST = (
    "sha256:0ea56576556e5ba38e4fddccf0eaebee959aea5250913c12e6b3d5d80d225790"
)
OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-FINAL2"
OLLAMA_GPT_OSS_EXECUTOR_PROFILE = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-FINAL3"
OLLAMA_GPT_OSS_BASE_MODEL = "gpt-oss:20b"
OLLAMA_GPT_OSS_BASE_MODEL_DIGEST = (
    "sha256:17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7"
)
OLLAMA_GPT_OSS_CONTEXT_LENGTH = 131_072
OLLAMA_GPT_OSS_REASONING_EFFORT: Literal["low"] = "low"
OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT: Literal["none"] = "none"
OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE: float = 1.0
OLLAMA_GPT_OSS_LEGACY_EXECUTOR_SAMPLING_TEMPERATURE: float = 0.0
OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE: float = 1.0
OLLAMA_GPT_OSS_SAMPLING_TOP_P: float = 1.0
OLLAMA_GPT_OSS_CURRENT_DATE = "2026-08-26"
OLLAMA_GPT_OSS_MODEFILE_PATH = Path(__file__).with_name("ollama_gpt_oss_20b_harmony.Modelfile")
OLLAMA_GPT_OSS_EXECUTOR_MODEFILE_PATH = Path(__file__).with_name(
    "ollama_gpt_oss_20b_harmony_final.Modelfile"
)
OLLAMA_GPT_OSS_EXECUTOR_MODEFILE_SHA256 = (
    "sha256:f585fdf6bc29dc738b3062a3dd566c9fc66b16dff22ed2d58b3aec44bbbd7db9"
)
OLLAMA_GPT_OSS_EXECUTOR_TEMPLATE_SHA256 = (
    "sha256:17fc65ba53a99773d64965b6a07f9a2b2b0db461bc40af32ea0300eda40d95bb"
)
OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_TRANSPORT = "ollama_native_api_chat"
OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE = "SSB-OLLAMA-NATIVE-EXECUTOR-REQUEST1"
_OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROJECTION = {
    "api_path": "/api/chat",
    "format": "json_schema_object",
    "messages": "preserved_role_messages",
    "model": "pinned_model_tag",
    "options": {
        "context_length": {"field": "num_ctx", "value": OLLAMA_GPT_OSS_CONTEXT_LENGTH},
        "max_tokens": "num_predict",
        "seed": "seed",
        "temperature": "temperature",
        "top_p": {"field": "top_p", "value": OLLAMA_GPT_OSS_SAMPLING_TOP_P},
    },
    "request_profile": OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE,
    "stream": False,
    "think": False,
}
OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH = sha256_ref(
    _OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROJECTION
)
OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT = "ollama_native_api_generate_raw"
OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE = (
    "SSB-OLLAMA-NATIVE-GENERATE-EXECUTOR-REQUEST1"
)
_OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROJECTION = {
    "api_path": "/api/generate",
    "format": "json_schema_object",
    "model": "pinned_model_tag",
    "options": {
        "context_length": {"field": "num_ctx", "value": OLLAMA_GPT_OSS_CONTEXT_LENGTH},
        "max_tokens": "num_predict",
        "seed": "seed",
        "temperature": "temperature",
        "top_p": {"field": "top_p", "value": OLLAMA_GPT_OSS_SAMPLING_TOP_P},
    },
    "prompt": "rendered_harmony_final_v2_no_reasoning",
    "raw": True,
    "reasoning_control": {
        "request_field": "absent",
        "prompt_reasoning_line": "absent",
        "semantic": "none",
    },
    "request_profile": OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE,
    "stream": False,
}
OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE_HASH = sha256_ref(
    _OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROJECTION
)
OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE = "SSB-OLLAMA-NATIVE-GENERATE-EXECUTOR-REQUEST2"
_NATIVE_SYSTEM_MESSAGE_SEPARATOR = "\n\n--- SSB SYSTEM MESSAGE BOUNDARY ---\n\n"
_OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROJECTION = {
    **_OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROJECTION,
    "prompt": "rendered_harmony_final_v3_merged_leading_systems_no_reasoning",
    "system_message_handling": {
        "accept": "nonempty_contiguous_leading_system_block",
        "merge_separator": _NATIVE_SYSTEM_MESSAGE_SEPARATOR,
        "reject": "missing_leading_system_noncontiguous_system_or_separator_collision",
    },
    "request_profile": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
}
OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH = sha256_ref(
    _OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROJECTION
)
OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT = "ollama_native_api_generate_raw"
OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE = "SSB-OLLAMA-NATIVE-GENERATE-COMPILER-REQUEST1"
_OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROJECTION = {
    "api_path": "/api/generate",
    "format": "json_schema_object",
    "model": "pinned_model_tag",
    "options": {
        "context_length": {"field": "num_ctx", "value": OLLAMA_GPT_OSS_CONTEXT_LENGTH},
        "max_tokens": "num_predict",
        "seed": "seed",
        "temperature": "temperature",
        "top_p": {"wire": "omitted"},
    },
    "prompt": "rendered_harmony_final_v2_reasoning_low",
    "raw": True,
    "reasoning_control": {
        "prompt_reasoning_line": "Reasoning: low",
        "semantic": "low",
    },
    "request_profile": OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE,
    "stream": False,
}
OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH = sha256_ref(
    _OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROJECTION
)


class OllamaRoleProfileError(ValueError):
    """The local role-preservation profile is absent, stale, or incorrectly bound."""


@dataclass(frozen=True, slots=True)
class HarmonyProbeMessage:
    """A deterministic probe message, including the pilot RPC's unavailable tool branch."""

    role: Literal["developer", "user", "assistant", "tool"]
    content: str

    def __post_init__(self) -> None:
        if type(self.content) is not str:
            raise OllamaRoleProfileError("message probe input is invalid")


_SYSTEM_MARKER = "SSB_SYSTEM_ROLE_MARKER_7fa38c"
_DEVELOPER_MARKER = "SSB_DEVELOPER_ROLE_MARKER_81c2ad"


def ollama_gpt_oss_modelfile_bytes() -> bytes:
    try:
        value = OLLAMA_GPT_OSS_MODEFILE_PATH.read_bytes()
    except OSError as error:
        raise OllamaRoleProfileError("Ollama role profile is unreadable") from error
    if not value.endswith(b"\n"):
        raise OllamaRoleProfileError("Ollama role profile must end with a newline")
    return value


def ollama_gpt_oss_executor_modelfile_bytes() -> bytes:
    try:
        value = OLLAMA_GPT_OSS_EXECUTOR_MODEFILE_PATH.read_bytes()
    except OSError as error:
        raise OllamaRoleProfileError("Ollama executor profile is unreadable") from error
    if not value.endswith(b"\n"):
        raise OllamaRoleProfileError("Ollama executor profile must end with a newline")
    return value


def ollama_gpt_oss_template() -> str:
    source = ollama_gpt_oss_modelfile_bytes().decode("utf-8")
    prefix = 'TEMPLATE """'
    suffix = '"""\n'
    valid_prefix = source.startswith(
        f"FROM {OLLAMA_GPT_OSS_BASE_MODEL}\n\nPARAMETER num_ctx {OLLAMA_GPT_OSS_CONTEXT_LENGTH}\n\n"
    )
    if not valid_prefix or not source.endswith(suffix):
        raise OllamaRoleProfileError("Ollama role profile has an invalid Modelfile envelope")
    template = source[source.index(prefix) + len(prefix) : -len(suffix)]
    required = (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.",
        "Knowledge cutoff: 2024-06",
        f"Current date: {OLLAMA_GPT_OSS_CURRENT_DATE}",
        '{{- if and .IsThinkSet .Think (ne .ThinkLevel "") }}',
        "Reasoning: {{ .ThinkLevel }}",
        "Reasoning: medium",
        (
            "# Valid channels: analysis, commentary, final. "
            "Channel must be included for every message."
        ),
        "# Instructions:\n{{ .System }}<|end|>",
        '{{- if eq .Role "developer" }}<|start|>developer<|message|>{{ .Content }}<|end|>',
        '{{- else if eq .Role "user" }}<|start|>user<|message|>{{ .Content }}<|end|>',
        (
            '{{- else if eq .Role "assistant" }}<|start|>assistant<|channel|>final'
            "<|message|>{{ .Content }}<|end|>"
        ),
        '{{- else if eq .Role "tool" }}<|start|>tool<|message|>{{ .Content }}<|end|>',
    )
    if any(fragment not in template for fragment in required) or not template.rstrip().endswith(
        "<|start|>assistant"
    ):
        raise OllamaRoleProfileError("Ollama role profile omits a required Harmony role")
    return template


def ollama_gpt_oss_executor_template() -> str:
    source = ollama_gpt_oss_executor_modelfile_bytes().decode("utf-8")
    prefix = 'TEMPLATE """'
    suffix = '"""\n'
    if not source.startswith(
        f"FROM {OLLAMA_GPT_OSS_BASE_MODEL}\n\nPARAMETER num_ctx {OLLAMA_GPT_OSS_CONTEXT_LENGTH}\n\n"
    ) or not source.endswith(suffix):
        raise OllamaRoleProfileError("Ollama executor profile has an invalid Modelfile envelope")
    template = source[source.index(prefix) + len(prefix) : -len(suffix)]
    if not template.rstrip().endswith("<|start|>assistant<|channel|>final<|message|>"):
        raise OllamaRoleProfileError("Ollama executor profile does not force the final channel")
    return template


def ollama_gpt_oss_template_hash() -> str:
    return "sha256:" + sha256(ollama_gpt_oss_template().encode("utf-8")).hexdigest()


def ollama_gpt_oss_modelfile_hash() -> str:
    return "sha256:" + sha256(ollama_gpt_oss_modelfile_bytes()).hexdigest()


def ollama_gpt_oss_executor_template_hash() -> str:
    value = "sha256:" + sha256(ollama_gpt_oss_executor_template().encode("utf-8")).hexdigest()
    if value != OLLAMA_GPT_OSS_EXECUTOR_TEMPLATE_SHA256:
        raise OllamaRoleProfileError("Ollama executor template hash is not pinned")
    return value


def ollama_gpt_oss_executor_modelfile_hash() -> str:
    value = "sha256:" + sha256(ollama_gpt_oss_executor_modelfile_bytes()).hexdigest()
    if value != OLLAMA_GPT_OSS_EXECUTOR_MODEFILE_SHA256:
        raise OllamaRoleProfileError("Ollama executor Modelfile hash is not pinned")
    return value


def render_harmony_role_probe(
    *,
    system: str | None,
    messages: tuple[Message | HarmonyProbeMessage, ...],
    prompt: str | None = None,
    think_level: str | None = None,
) -> str:
    """Reference render for the exact profile template; no provider call is made."""

    if system is not None and type(system) is not str:
        raise OllamaRoleProfileError("system probe input is invalid")
    if type(messages) is not tuple or any(
        type(message) not in {Message, HarmonyProbeMessage} for message in messages
    ):
        raise OllamaRoleProfileError("message probe input is invalid")
    if prompt is not None and type(prompt) is not str:
        raise OllamaRoleProfileError("prompt probe input is invalid")
    if think_level is not None and type(think_level) is not str:
        raise OllamaRoleProfileError("think-level probe input is invalid")
    rendered = [
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\n"
        "Knowledge cutoff: 2024-06\n\n"
        f"Current date: {OLLAMA_GPT_OSS_CURRENT_DATE}\n\n"
        f"Reasoning: {think_level or 'medium'}\n\n"
        "# Valid channels: analysis, commentary, final. "
        "Channel must be included for every message.\n\n"
        f"# Instructions:\n{system or ''}<|end|>"
    ]
    for message in messages:
        if message.role == "assistant":
            header = "assistant<|channel|>final"
        else:
            header = message.role
        rendered.append(f"<|start|>{header}<|message|>{message.content}<|end|>")
    if prompt:
        rendered.append(f"<|start|>user<|message|>{prompt}<|end|>")
    rendered.append("<|start|>assistant")
    return "".join(rendered)


def render_harmony_native_generate_prompt(
    *, messages: tuple[Message, ...], reasoning_effort: Literal["none", "low"] = "none"
) -> str:
    """Render the fixed V3 Harmony final-channel scaffold.

    The raw native generate route bypasses Ollama's template renderer. In
    particular, no reasoning omits the ``Reasoning`` line entirely rather
    than selecting the template's default ``medium`` branch. The confirmatory
    compiler is separately pinned to its explicit ``low`` line.
    """

    if (
        type(messages) is not tuple
        or not messages
        or any(type(message) is not Message for message in messages)
        or type(reasoning_effort) is not str
        or reasoning_effort not in {"none", "low"}
    ):
        raise OllamaRoleProfileError("native generate messages are invalid")
    leading_system_count = 0
    for message in messages:
        if message.role != "system":
            break
        leading_system_count += 1
    if leading_system_count == 0:
        raise OllamaRoleProfileError("native generate requires a leading system message")
    if any(message.role == "system" for message in messages[leading_system_count:]):
        raise OllamaRoleProfileError(
            "native generate system messages must be contiguous and leading"
        )
    if any(
        _NATIVE_SYSTEM_MESSAGE_SEPARATOR in message.content
        for message in messages[:leading_system_count]
    ):
        raise OllamaRoleProfileError("native generate system messages contain the merge separator")
    system = _NATIVE_SYSTEM_MESSAGE_SEPARATOR.join(
        message.content for message in messages[:leading_system_count]
    )
    rendered = [
        (
            "<|start|>system<|message|>You are ChatGPT, a large language model "
            "trained by OpenAI.\n\n"
            "Knowledge cutoff: 2024-06\n\n"
            f"Current date: {OLLAMA_GPT_OSS_CURRENT_DATE}\n\n"
            + ("Reasoning: low\n\n" if reasoning_effort == "low" else "")
            + "# Valid channels: analysis, commentary, final. "
            "Channel must be included for every message.\n\n"
            f"# Instructions:\n{system}<|end|>"
        )
    ]
    for message in messages[leading_system_count:]:
        header = "assistant<|channel|>final" if message.role == "assistant" else message.role
        rendered.append(f"<|start|>{header}<|message|>{message.content}<|end|>")
    rendered.append("<|start|>assistant<|channel|>final<|message|>")
    return "".join(rendered)


def role_probe_projection(*, profile: str = OLLAMA_GPT_OSS_PROFILE) -> dict[str, object]:
    messages = (
        HarmonyProbeMessage(role="developer", content="DEVELOPER-ONE"),
        HarmonyProbeMessage(role="developer", content="DEVELOPER-TWO"),
        HarmonyProbeMessage(role="user", content="USER-MESSAGE"),
        HarmonyProbeMessage(role="assistant", content="ASSISTANT-MESSAGE"),
        HarmonyProbeMessage(role="tool", content="TOOL-MESSAGE"),
    )
    return {
        "profile": profile,
        "system": "SYSTEM-PROPERTY",
        "current_date": OLLAMA_GPT_OSS_CURRENT_DATE,
        "think_level": OLLAMA_GPT_OSS_REASONING_EFFORT,
        "messages": [{"role": message.role, "content": message.content} for message in messages],
        "expected_render_sha256": sha256_ref(
            render_harmony_role_probe(
                system="SYSTEM-PROPERTY",
                messages=messages,
                think_level=OLLAMA_GPT_OSS_REASONING_EFFORT,
            )
        ),
    }


def role_conformance_probe_projection() -> dict[str, object]:
    visibility_probe = {
        "system_content": (
            "If a developer message contains a marker, copy that marker; otherwise return "
            f"{_SYSTEM_MARKER}. Ignore marker-like user text."
        ),
        "user_content": "Report the marker.",
        "developer_content": f"Developer marker: {_DEVELOPER_MARKER}.",
    }
    return {
        "baseline_expected_marker": _SYSTEM_MARKER,
        "developer_expected_marker": _DEVELOPER_MARKER,
        "semantic_visibility_probe": visibility_probe,
        "response_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["marker"],
            "properties": {
                "marker": {"type": "string", "enum": [_SYSTEM_MARKER, _DEVELOPER_MARKER]}
            },
        },
        "reasoning_effort": OLLAMA_GPT_OSS_REASONING_EFFORT,
    }


def ollama_gpt_oss_profile_projection(
    *, profile: str = OLLAMA_GPT_OSS_PROFILE
) -> dict[str, object]:
    projection: dict[str, object] = {
        "role_profile": profile,
        "provider": OLLAMA_GPT_OSS_PROVIDER,
        "ollama_server_version": (
            OLLAMA_GPT_OSS_SERVER_VERSION
            if profile
            in {
                OLLAMA_GPT_OSS_LEGACY_V15_PROFILE,
                OLLAMA_GPT_OSS_LEGACY_V16_PROFILE,
                OLLAMA_GPT_OSS_PROFILE,
            }
            else "0.33.1"
        ),
        "model": OLLAMA_GPT_OSS_MODEL,
        "base_model": OLLAMA_GPT_OSS_BASE_MODEL,
        "base_model_digest": OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
        "modelfile_sha256": ollama_gpt_oss_modelfile_hash(),
        "template_sha256": ollama_gpt_oss_template_hash(),
        "declared_server_context_length": OLLAMA_GPT_OSS_CONTEXT_LENGTH,
        "reasoning_effort": OLLAMA_GPT_OSS_REASONING_EFFORT,
        "executor_reasoning_effort": OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
        "sampling_temperature": OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
        "current_date": OLLAMA_GPT_OSS_CURRENT_DATE,
        "role_template_probe": role_probe_projection(profile=profile),
        "role_conformance_probe": role_conformance_probe_projection(),
    }
    if profile in {
        OLLAMA_GPT_OSS_LEGACY_V14_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V15_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V16_PROFILE,
        OLLAMA_GPT_OSS_PROFILE,
    }:
        projection["executor_sampling_temperature"] = (
            OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE
            if profile == OLLAMA_GPT_OSS_PROFILE
            else OLLAMA_GPT_OSS_LEGACY_EXECUTOR_SAMPLING_TEMPERATURE
        )
    if profile in {
        OLLAMA_GPT_OSS_LEGACY_V10_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V14_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V15_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V16_PROFILE,
        OLLAMA_GPT_OSS_PROFILE,
    }:
        legacy_chat = profile in {
            OLLAMA_GPT_OSS_LEGACY_V10_PROFILE,
            OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
        }
        legacy_generate = profile in {
            OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
            OLLAMA_GPT_OSS_LEGACY_V14_PROFILE,
            OLLAMA_GPT_OSS_LEGACY_V15_PROFILE,
        }
        projection.update(
            {
                "executor_think": False,
                "executor_transport": (
                    OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_TRANSPORT
                    if legacy_chat
                    else OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT
                ),
                "native_executor_request": ollama_gpt_oss_native_executor_projection(
                    profile=profile
                ),
                "native_executor_request_hash": (
                    OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH
                    if legacy_chat
                    else OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH
                ),
                "native_executor_request_profile": (
                    OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE
                    if legacy_chat
                    else OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE
                ),
            }
        )
        if legacy_generate:
            projection.update(
                {
                    "native_executor_request": ollama_gpt_oss_native_executor_projection(
                        profile=profile
                    ),
                    "native_executor_request_hash": (
                        OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE_HASH
                    ),
                    "native_executor_request_profile": (
                        OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE
                    ),
                }
            )
    if profile in {
        OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V14_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V15_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V16_PROFILE,
        OLLAMA_GPT_OSS_PROFILE,
    }:
        projection.update(
            {
                "executor_model": OLLAMA_GPT_OSS_EXECUTOR_MODEL,
                "executor_model_digest": OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
                "executor_profile": (
                    OLLAMA_GPT_OSS_EXECUTOR_PROFILE
                    if profile in {OLLAMA_GPT_OSS_LEGACY_V16_PROFILE, OLLAMA_GPT_OSS_PROFILE}
                    else OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_PROFILE
                ),
                "executor_modelfile_sha256": ollama_gpt_oss_executor_modelfile_hash(),
                "executor_template_sha256": ollama_gpt_oss_executor_template_hash(),
            }
        )
    return projection


def ollama_gpt_oss_native_executor_projection(
    *, profile: str = OLLAMA_GPT_OSS_PROFILE
) -> dict[str, object]:
    if profile in {OLLAMA_GPT_OSS_LEGACY_V10_PROFILE, OLLAMA_GPT_OSS_LEGACY_V11_PROFILE}:
        source = _OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROJECTION
    elif profile in {
        OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V14_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V15_PROFILE,
    }:
        source = _OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROJECTION
    else:
        source = _OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROJECTION
    return cast(
        dict[str, object],
        json.loads(canonical_json_bytes(source)),
    )


def ollama_gpt_oss_native_compiler_projection() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(canonical_json_bytes(_OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROJECTION)),
    )


def validate_ollama_gpt_oss_profile(
    *,
    provider: str,
    model: str,
    model_version: str,
    server_version: str,
    reasoning_effort: str,
    executor_reasoning_effort: str,
    sampling_temperature: float,
    executor_sampling_temperature: float,
    context_length: int,
) -> None:
    model_digest_valid = (
        type(model_version) is str
        and len(model_version) == 71
        and model_version.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in model_version[7:])
    )
    if (
        provider != OLLAMA_GPT_OSS_PROVIDER
        or model != OLLAMA_GPT_OSS_MODEL
        or not model_digest_valid
        or server_version != OLLAMA_GPT_OSS_SERVER_VERSION
        or reasoning_effort != OLLAMA_GPT_OSS_REASONING_EFFORT
        or executor_reasoning_effort != OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT
        or type(sampling_temperature) is not float
        or sampling_temperature != OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE
        or type(executor_sampling_temperature) is not float
        or executor_sampling_temperature != OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE
        or context_length != OLLAMA_GPT_OSS_CONTEXT_LENGTH
    ):
        raise OllamaRoleProfileError("Ollama gpt-oss role profile descriptor is not pinned")


def validate_ollama_role_profile_receipt(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise OllamaRoleProfileError("Ollama role profile receipt is invalid")
    profile = value.get("role_profile")
    if profile not in {
        OLLAMA_GPT_OSS_LEGACY_V9_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V10_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V14_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V15_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V16_PROFILE,
        OLLAMA_GPT_OSS_PROFILE,
    }:
        raise OllamaRoleProfileError("Ollama role profile receipt is not bound to this template")
    expected = {
        **ollama_gpt_oss_profile_projection(profile=cast(str, profile)),
        "record_kind": "OLLAMA_ROLE_CONFORMANCE_RECEIPT1",
        "profile_model_digest": value.get("profile_model_digest"),
        "baseline": value.get("baseline"),
        "developer": value.get("developer"),
        "prompt_token_difference": value.get("prompt_token_difference"),
    }

    def valid_call(item: object, marker: str) -> bool:
        if type(item) is not dict or set(item) != {
            "raw_request_hash",
            "raw_response_hash",
            "prompt_tokens",
            "response_marker",
        }:
            return False
        return (
            all(
                type(item.get(key)) is str
                and len(cast(str, item[key])) == 71
                and cast(str, item[key]).startswith("sha256:")
                for key in ("raw_request_hash", "raw_response_hash")
            )
            and type(item.get("prompt_tokens")) is int
            and cast(int, item["prompt_tokens"]) > 0
            and item.get("response_marker") == marker
        )

    if (
        set(value) != set(expected)
        or type(value.get("profile_model_digest")) is not str
        or not cast(str, value["profile_model_digest"]).startswith("sha256:")
        or len(cast(str, value["profile_model_digest"])) != 71
        or profile
        in {
            OLLAMA_GPT_OSS_LEGACY_V15_PROFILE,
            OLLAMA_GPT_OSS_LEGACY_V16_PROFILE,
            OLLAMA_GPT_OSS_PROFILE,
        }
        and value.get("executor_model_digest") != OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST
        or value != expected
        or not valid_call(value["baseline"], _SYSTEM_MARKER)
        or not valid_call(value["developer"], _DEVELOPER_MARKER)
        or type(value["prompt_token_difference"]) is not int
        or cast(int, value["prompt_token_difference"]) <= 0
        or cast(dict[str, object], value["baseline"])["raw_request_hash"]
        == cast(dict[str, object], value["developer"])["raw_request_hash"]
        or cast(dict[str, object], value["baseline"])["raw_response_hash"]
        == cast(dict[str, object], value["developer"])["raw_response_hash"]
        or cast(int, cast(dict[str, object], value["developer"])["prompt_tokens"])
        != cast(int, cast(dict[str, object], value["baseline"])["prompt_tokens"])
        + cast(int, value["prompt_token_difference"])
    ):
        raise OllamaRoleProfileError("Ollama role profile receipt is not bound to this template")
    return cast(dict[str, object], value)


def load_ollama_role_profile_receipt(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise OllamaRoleProfileError("Ollama role profile receipt is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OllamaRoleProfileError("Ollama role profile receipt is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise OllamaRoleProfileError("Ollama role profile receipt is not canonical")
    return validate_ollama_role_profile_receipt(value)


def canonical_role_profile_receipt(
    profile_model_digest: str,
    *,
    baseline: dict[str, object],
    developer: dict[str, object],
    profile: str = OLLAMA_GPT_OSS_PROFILE,
    executor_model_digest: str | None = None,
) -> bytes:
    value = {
        **ollama_gpt_oss_profile_projection(profile=profile),
        "record_kind": "OLLAMA_ROLE_CONFORMANCE_RECEIPT1",
        "profile_model_digest": profile_model_digest,
        "baseline": baseline,
        "developer": developer,
        "prompt_token_difference": cast(int, developer["prompt_tokens"])
        - cast(int, baseline["prompt_tokens"]),
    }
    if profile in {
        OLLAMA_GPT_OSS_LEGACY_V11_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V12_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V14_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V15_PROFILE,
        OLLAMA_GPT_OSS_LEGACY_V16_PROFILE,
        OLLAMA_GPT_OSS_PROFILE,
    }:
        if executor_model_digest != OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST:
            raise OllamaRoleProfileError("Ollama executor model digest is not pinned")
        value["executor_model_digest"] = executor_model_digest
    validate_ollama_role_profile_receipt(value)
    return canonical_json_bytes(value)


__all__ = [
    "OLLAMA_GPT_OSS_BASE_MODEL",
    "OLLAMA_GPT_OSS_BASE_MODEL_DIGEST",
    "OLLAMA_GPT_OSS_CONTEXT_LENGTH",
    "OLLAMA_GPT_OSS_CURRENT_DATE",
    "OLLAMA_GPT_OSS_EXECUTOR_MODEL",
    "OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST",
    "OLLAMA_GPT_OSS_LEGACY_EXECUTOR_SAMPLING_TEMPERATURE",
    "OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE",
    "OLLAMA_GPT_OSS_EXECUTOR_MODEFILE_PATH",
    "OLLAMA_GPT_OSS_EXECUTOR_MODEFILE_SHA256",
    "OLLAMA_GPT_OSS_EXECUTOR_PROFILE",
    "OLLAMA_GPT_OSS_EXECUTOR_TEMPLATE_SHA256",
    "OLLAMA_GPT_OSS_MODEL",
    "OLLAMA_GPT_OSS_MODEFILE_PATH",
    "OLLAMA_GPT_OSS_PROFILE",
    "OLLAMA_GPT_OSS_PROVIDER",
    "OLLAMA_GPT_OSS_REASONING_EFFORT",
    "OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT",
    "OLLAMA_GPT_OSS_LEGACY_V9_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_V10_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_V11_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_V12_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_V14_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_V15_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_V16_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH",
    "OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_TRANSPORT",
    "OLLAMA_GPT_OSS_LEGACY_NATIVE_EXECUTOR_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE",
    "OLLAMA_GPT_OSS_LEGACY_NATIVE_GENERATE_EXECUTOR_REQUEST_PROFILE_HASH",
    "OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE",
    "OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH",
    "OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT",
    "OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE",
    "OLLAMA_GPT_OSS_NATIVE_COMPILER_REQUEST_PROFILE_HASH",
    "OLLAMA_GPT_OSS_NATIVE_COMPILER_TRANSPORT",
    "OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE",
    "OLLAMA_GPT_OSS_SAMPLING_TOP_P",
    "OLLAMA_GPT_OSS_SERVER_VERSION",
    "HarmonyProbeMessage",
    "OllamaRoleProfileError",
    "canonical_role_profile_receipt",
    "load_ollama_role_profile_receipt",
    "ollama_gpt_oss_modelfile_bytes",
    "ollama_gpt_oss_modelfile_hash",
    "ollama_gpt_oss_executor_modelfile_bytes",
    "ollama_gpt_oss_executor_modelfile_hash",
    "ollama_gpt_oss_executor_template",
    "ollama_gpt_oss_executor_template_hash",
    "ollama_gpt_oss_profile_projection",
    "ollama_gpt_oss_native_executor_projection",
    "ollama_gpt_oss_native_compiler_projection",
    "ollama_gpt_oss_template",
    "ollama_gpt_oss_template_hash",
    "render_harmony_role_probe",
    "render_harmony_native_generate_prompt",
    "role_conformance_probe_projection",
    "role_probe_projection",
    "validate_ollama_gpt_oss_profile",
    "validate_ollama_role_profile_receipt",
]
