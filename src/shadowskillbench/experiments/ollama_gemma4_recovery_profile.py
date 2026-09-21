"""Prospective, hash-bound Gemma 4 native-tool recovery candidates.

This module deliberately does not share the historical Gemma V4 profile.  The
two candidates have their own observed runtime facts and require new identity
and role/tool receipts before any capability work can start.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.models.native_tools import NativeToolDefinition, native_tool_declaration_hash

RecoveryCandidate = Literal["primary-26b", "fallback-12b"]
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class Gemma4RecoveryProfileError(ValueError):
    """A prospective recovery receipt is missing or does not bind its candidate."""


@dataclass(frozen=True, slots=True)
class RecoveryCandidateSpec:
    candidate: RecoveryCandidate
    model: str
    model_digest: str
    from_blob_sha256: str
    modelfile_sha256: str
    capabilities: tuple[str, ...]
    corpus_seed: int
    output_root: str


_SPECS: dict[RecoveryCandidate, RecoveryCandidateSpec] = {
    "primary-26b": RecoveryCandidateSpec(
        candidate="primary-26b",
        model="gemma4:26b-a4b-it-q4_K_M",
        model_digest="sha256:5571076f3d70050487b26b341705799e0ab29b808164f90d20d4cf84f699d251",
        from_blob_sha256="sha256:7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df",
        modelfile_sha256="sha256:4bce192bceb459504cd1a5619d4f57bc86ed01f47a9d87f613cfea800b4bbc2f",
        capabilities=("completion", "vision", "tools", "thinking"),
        corpus_seed=4249,
        output_root="v4-gemma4-recovery-primary-26b-seed-4249",
    ),
    "fallback-12b": RecoveryCandidateSpec(
        candidate="fallback-12b",
        model="gemma4:12b-it-q4_K_M",
        model_digest="sha256:4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c",
        from_blob_sha256="sha256:1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606",
        modelfile_sha256="sha256:3bf96f2f667028647bc9ea9db02f945e0ee6b156e68332ba345297384c0f2100",
        capabilities=("completion", "vision", "audio", "tools", "thinking"),
        corpus_seed=4250,
        output_root="v4-gemma4-recovery-fallback-12b-seed-4250",
    ),
}

OLLAMA_GEMMA4_RECOVERY_SERVER_VERSION = "0.34.0"
OLLAMA_GEMMA4_RECOVERY_CONTEXT_LENGTH = 32_768
OLLAMA_GEMMA4_RECOVERY_MODEL_CONTEXT = 262_144
OLLAMA_GEMMA4_RECOVERY_TEMPERATURE = 1.0
OLLAMA_GEMMA4_RECOVERY_SEED = 4242
OLLAMA_GEMMA4_RECOVERY_MAX_TOKENS = 16_384
OLLAMA_GEMMA4_RECOVERY_MAX_TURNS = 12
OLLAMA_GEMMA4_RECOVERY_MAX_TOOL_CALLS = 12
OLLAMA_GEMMA4_RECOVERY_ROLE_KIND = "OLLAMA_GEMMA4_RECOVERY_ROLE_TOOL_CONFORMANCE_RECEIPT1"
OLLAMA_GEMMA4_RECOVERY_ROLE_FAILURE_KIND = (
    "OLLAMA_GEMMA4_RECOVERY_ROLE_TOOL_CONFORMANCE_FAILURE_RECEIPT1"
)
OLLAMA_GEMMA4_RECOVERY_IDENTITY_KIND = "OLLAMA_GEMMA4_RECOVERY_IDENTITY_RECEIPT1"


def recovery_candidate_spec(candidate: RecoveryCandidate) -> RecoveryCandidateSpec:
    try:
        return _SPECS[candidate]
    except KeyError as error:
        raise Gemma4RecoveryProfileError("recovery candidate is invalid") from error


def recovery_projection(candidate: RecoveryCandidate) -> dict[str, object]:
    spec = recovery_candidate_spec(candidate)
    return {
        "candidate": spec.candidate,
        "provider": "ollama",
        "model": spec.model,
        "model_digest": spec.model_digest,
        "from_blob_sha256": spec.from_blob_sha256,
        "modelfile_sha256": spec.modelfile_sha256,
        "ollama_server_version": OLLAMA_GEMMA4_RECOVERY_SERVER_VERSION,
        "context_length": OLLAMA_GEMMA4_RECOVERY_CONTEXT_LENGTH,
        "model_context": OLLAMA_GEMMA4_RECOVERY_MODEL_CONTEXT,
        "temperature": OLLAMA_GEMMA4_RECOVERY_TEMPERATURE,
        "role_conformance_seed": OLLAMA_GEMMA4_RECOVERY_SEED,
        "max_output_tokens": OLLAMA_GEMMA4_RECOVERY_MAX_TOKENS,
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS3",
        "response_contract": "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
        "history_projection": "SSB-OLLAMA-NATIVE-TOOLS-HISTORY1",
        "native_turn_prompt_profile": "SSB-OLLAMA-NATIVE-TOOLS-TURN-PROMPT1",
    }


def _usage(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"input_tokens", "output_tokens", "total_tokens"}
        and all(type(value[key]) is int and value[key] >= 0 for key in value)
        and value["total_tokens"] == value["input_tokens"] + value["output_tokens"]
    )


def _call(value: object, marker: str) -> bool:
    fields = {
        "arguments_hash",
        "attempts",
        "finish_reason",
        "provider_done",
        "provider_finish_reason",
        "raw_request_hash",
        "raw_response_hash",
        "role_marker",
        "tool_declaration_hash",
        "tool_name",
        "usage",
    }
    return (
        type(value) is dict
        and set(value) == fields
        and all(
            type(value[key]) is str and _HASH.fullmatch(cast(str, value[key]))
            for key in (
                "arguments_hash",
                "raw_request_hash",
                "raw_response_hash",
                "tool_declaration_hash",
            )
        )
        and value["attempts"] == 1
        and value["finish_reason"] == "tool_calls"
        and value["provider_done"] is True
        and value["provider_finish_reason"] == "stop"
        and value["role_marker"] == marker
        and value["tool_name"] == "finish_task"
        and _usage(value["usage"])
    )


def validate_recovery_identity(value: object, candidate: RecoveryCandidate) -> dict[str, object]:
    spec = recovery_candidate_spec(candidate)
    if type(value) is not dict:
        raise Gemma4RecoveryProfileError("recovery identity is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_GEMMA4_RECOVERY_IDENTITY_KIND,
        **recovery_projection(candidate),
        "capabilities": list(spec.capabilities),
        "show_response_hash": value.get("show_response_hash"),
        "tags_response_hash": value.get("tags_response_hash"),
        "version_response_hash": value.get("version_response_hash"),
    }
    if (
        body != expected
        or any(
            type(value.get(field)) is not str or _HASH.fullmatch(cast(str, value[field])) is None
            for field in ("show_response_hash", "tags_response_hash", "version_response_hash")
        )
        or value.get("receipt_hash") != sha256_ref(body)
    ):
        raise Gemma4RecoveryProfileError("recovery identity is not bound")
    return cast(dict[str, object], value)


def validate_recovery_role_receipt(
    value: object, candidate: RecoveryCandidate
) -> dict[str, object]:
    if type(value) is not dict:
        raise Gemma4RecoveryProfileError("recovery role receipt is invalid")
    baseline = value.get("baseline")
    developer = value.get("developer")
    tool_call = value.get("tool_call")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_GEMMA4_RECOVERY_ROLE_KIND,
        **recovery_projection(candidate),
        "baseline": baseline,
        "developer": developer,
        "tool_call": tool_call,
        "developer_prompt_token_delta": value.get("developer_prompt_token_delta"),
    }
    expected_arguments_hash = sha256_ref({"summary": "native tool conformance"})
    expected_tool_hash = native_tool_declaration_hash(
        (
            NativeToolDefinition(
                name="finish_task",
                description="Finish the task with a concise summary.",
                parameters={
                    "type": "object",
                    "properties": {"summary": {}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            ),
        )
    )
    if (
        body != expected
        or value.get("receipt_hash") != sha256_ref(body)
        or not _call(baseline, "baseline")
        or not _call(developer, "developer")
        or not _call(tool_call, "tool_call")
        or cast(dict[str, object], baseline)["arguments_hash"] != expected_arguments_hash
        or cast(dict[str, object], baseline)["tool_declaration_hash"] != expected_tool_hash
        or cast(dict[str, object], baseline)["arguments_hash"]
        != cast(dict[str, object], developer)["arguments_hash"]
        or cast(dict[str, object], baseline)["arguments_hash"]
        != cast(dict[str, object], tool_call)["arguments_hash"]
        or cast(dict[str, object], baseline)["tool_declaration_hash"]
        != cast(dict[str, object], developer)["tool_declaration_hash"]
        or cast(dict[str, object], baseline)["tool_declaration_hash"]
        != cast(dict[str, object], tool_call)["tool_declaration_hash"]
        or type(value.get("developer_prompt_token_delta")) is not int
        or value["developer_prompt_token_delta"] == 0
        or value["developer_prompt_token_delta"] < 0
        or cast(dict[str, object], baseline)["raw_request_hash"]
        == cast(dict[str, object], developer)["raw_request_hash"]
        or value["developer_prompt_token_delta"]
        != cast(dict[str, int], cast(dict[str, object], developer)["usage"])["input_tokens"]
        - cast(dict[str, int], cast(dict[str, object], baseline)["usage"])["input_tokens"]
    ):
        raise Gemma4RecoveryProfileError("recovery role receipt is not bound")
    return cast(dict[str, object], value)


def canonical_recovery_identity(
    *,
    candidate: RecoveryCandidate,
    show_response: bytes,
    tags_response: bytes,
    version_response: bytes,
) -> bytes:
    spec = recovery_candidate_spec(candidate)
    try:
        show = json.loads(show_response)
        tags = json.loads(tags_response)
        version = json.loads(version_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma4RecoveryProfileError("recovery show response is not JSON") from error
    if type(show) is not dict or type(tags) is not dict or type(version) is not dict:
        raise Gemma4RecoveryProfileError("recovery show response is invalid")
    modelfile = show.get("modelfile")
    models = tags.get("models")
    matching_tags = (
        [item for item in models if type(item) is dict and item.get("name") == spec.model]
        if type(models) is list
        else []
    )
    if (
        show.get("capabilities") != list(spec.capabilities)
        or type(modelfile) is not str
        or "RENDERER gemma4" not in modelfile
        or "PARSER gemma4" not in modelfile
        or spec.from_blob_sha256.removeprefix("sha256:") not in modelfile
        or "sha256:" + sha256(modelfile.encode("utf-8")).hexdigest() != spec.modelfile_sha256
        or type(show.get("model_info")) is not dict
        or cast(dict[str, object], show["model_info"]).get("gemma4.context_length")
        != OLLAMA_GEMMA4_RECOVERY_MODEL_CONTEXT
        or len(matching_tags) != 1
        or matching_tags[0].get("digest") != spec.model_digest.removeprefix("sha256:")
        or version.get("version") != OLLAMA_GEMMA4_RECOVERY_SERVER_VERSION
    ):
        raise Gemma4RecoveryProfileError("recovery show identity is not pinned")
    body = {
        "record_kind": OLLAMA_GEMMA4_RECOVERY_IDENTITY_KIND,
        **recovery_projection(candidate),
        "capabilities": list(spec.capabilities),
        "show_response_hash": "sha256:" + sha256(show_response).hexdigest(),
        "tags_response_hash": "sha256:" + sha256(tags_response).hexdigest(),
        "version_response_hash": "sha256:" + sha256(version_response).hexdigest(),
    }
    value = {**body, "receipt_hash": sha256_ref(body)}
    validate_recovery_identity(value, candidate)
    return canonical_json_bytes(value)


def canonical_recovery_role_receipt(
    *,
    candidate: RecoveryCandidate,
    baseline: dict[str, object],
    developer: dict[str, object],
    tool_call: dict[str, object],
) -> bytes:
    baseline_usage = baseline.get("usage")
    developer_usage = developer.get("usage")
    if type(baseline_usage) is not dict or type(developer_usage) is not dict:
        raise Gemma4RecoveryProfileError("recovery role receipt usage is invalid")
    body = {
        "record_kind": OLLAMA_GEMMA4_RECOVERY_ROLE_KIND,
        **recovery_projection(candidate),
        "baseline": baseline,
        "developer": developer,
        "tool_call": tool_call,
        "developer_prompt_token_delta": cast(int, developer_usage["input_tokens"])
        - cast(int, baseline_usage["input_tokens"]),
    }
    value = {**body, "receipt_hash": sha256_ref(body)}
    validate_recovery_role_receipt(value, candidate)
    return canonical_json_bytes(value)


def canonical_recovery_role_failure(
    *, candidate: RecoveryCandidate, failure_code: str, completed_calls: int
) -> bytes:
    if (
        failure_code
        not in {
            "CONFIGURATION_ERROR",
            "MODEL_OUTPUT_INVALID",
            "MODEL_PROVIDER_TERMINAL",
            "MODEL_PROVIDER_TRANSIENT",
        }
        or type(completed_calls) is not int
        or not 0 <= completed_calls <= 3
    ):
        raise Gemma4RecoveryProfileError("recovery role failure is invalid")
    body = {
        "record_kind": OLLAMA_GEMMA4_RECOVERY_ROLE_FAILURE_KIND,
        **recovery_projection(candidate),
        "failure_code": failure_code,
        "completed_calls": completed_calls,
    }
    return canonical_json_bytes({**body, "receipt_hash": sha256_ref(body)})


def validate_recovery_role_failure(
    value: object, candidate: RecoveryCandidate
) -> dict[str, object]:
    """Validate a recorded model-origin conformance failure without relabelling it.

    Only ``MODEL_OUTPUT_INVALID`` is an eligible reason to continue to the
    precommitted fallback. Provider and configuration failures are retained as
    technical holds by the caller.
    """

    if type(value) is not dict:
        raise Gemma4RecoveryProfileError("recovery role failure is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    expected = {
        "record_kind": OLLAMA_GEMMA4_RECOVERY_ROLE_FAILURE_KIND,
        **recovery_projection(candidate),
        "failure_code": value.get("failure_code"),
        "completed_calls": value.get("completed_calls"),
    }
    if (
        body != expected
        or value.get("receipt_hash") != sha256_ref(body)
        or value.get("failure_code")
        not in {
            "CONFIGURATION_ERROR",
            "MODEL_OUTPUT_INVALID",
            "MODEL_PROVIDER_TERMINAL",
            "MODEL_PROVIDER_TRANSIENT",
        }
        or type(value.get("completed_calls")) is not int
        or not 0 <= cast(int, value["completed_calls"]) <= 3
    ):
        raise Gemma4RecoveryProfileError("recovery role failure is not bound")
    return cast(dict[str, object], value)


def _load(path: Path, candidate: RecoveryCandidate, validator, label: str) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise Gemma4RecoveryProfileError(f"recovery {label} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma4RecoveryProfileError(f"recovery {label} is unreadable") from error
    if canonical_json_bytes(value) != raw:
        raise Gemma4RecoveryProfileError(f"recovery {label} is not canonical")
    return validator(value, candidate)


def load_recovery_identity(path: Path, candidate: RecoveryCandidate) -> dict[str, object]:
    return _load(path, candidate, validate_recovery_identity, "identity")


def load_recovery_role_receipt(path: Path, candidate: RecoveryCandidate) -> dict[str, object]:
    return _load(path, candidate, validate_recovery_role_receipt, "role receipt")


def load_recovery_role_failure(path: Path, candidate: RecoveryCandidate) -> dict[str, object]:
    return _load(path, candidate, validate_recovery_role_failure, "role failure")


__all__ = [
    "Gemma4RecoveryProfileError",
    "OLLAMA_GEMMA4_RECOVERY_CONTEXT_LENGTH",
    "OLLAMA_GEMMA4_RECOVERY_IDENTITY_KIND",
    "OLLAMA_GEMMA4_RECOVERY_MAX_TOKENS",
    "OLLAMA_GEMMA4_RECOVERY_MAX_TOOL_CALLS",
    "OLLAMA_GEMMA4_RECOVERY_MAX_TURNS",
    "OLLAMA_GEMMA4_RECOVERY_ROLE_KIND",
    "OLLAMA_GEMMA4_RECOVERY_ROLE_FAILURE_KIND",
    "OLLAMA_GEMMA4_RECOVERY_SEED",
    "OLLAMA_GEMMA4_RECOVERY_SERVER_VERSION",
    "OLLAMA_GEMMA4_RECOVERY_TEMPERATURE",
    "RecoveryCandidate",
    "RecoveryCandidateSpec",
    "canonical_recovery_identity",
    "canonical_recovery_role_failure",
    "canonical_recovery_role_receipt",
    "load_recovery_identity",
    "load_recovery_role_failure",
    "load_recovery_role_receipt",
    "recovery_candidate_spec",
    "recovery_projection",
    "validate_recovery_identity",
    "validate_recovery_role_failure",
    "validate_recovery_role_receipt",
]
