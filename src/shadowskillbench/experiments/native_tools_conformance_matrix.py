"""Frozen conformance-only matrix for pending native-tool candidates."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Literal, cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref

Candidate = Literal["gpt-oss-20b", "gemma4-12b-it-q4-k-m", "llama3.1-8b"]

CANDIDATES: dict[Candidate, dict[str, object]] = {
    "gpt-oss-20b": {
        "model": "gpt-oss:20b",
        "digest": "sha256:17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7",
        "capabilities": ["completion", "tools", "thinking"],
        "model_context": 131072,
        "temperature": 1.0,
        "blob": "sha256:e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb",
        "modelfile": "sha256:b648fa2f39b3ae0838878c4a62752e033f48c3b1158f42d6cf84a738120e9c57",
    },
    "gemma4-12b-it-q4-k-m": {
        "model": "gemma4:12b-it-q4_K_M",
        "digest": "sha256:4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c",
        "capabilities": ["completion", "vision", "audio", "tools", "thinking"],
        "model_context": 262144,
        "temperature": 1.0,
        "blob": "sha256:1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606",
        "modelfile": "sha256:c6749ea4267cda459b08c1a04c95cf733f148fba470f8aa07070bbfba1599fbf",
    },
    "llama3.1-8b": {
        "model": "llama3.1:8b",
        "digest": "sha256:46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e",
        "capabilities": ["completion", "tools"],
        "model_context": 131072,
        "temperature": 0.15,
        "blob": "sha256:667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29",
        "modelfile": "sha256:7590c492c0089d0a0ee662a01f8ffc61b5c9ccec4451ae656e7cb7365813303b",
    },
}


class MatrixConformanceError(ValueError):
    pass


def projection(candidate: Candidate) -> dict[str, object]:
    item = CANDIDATES[candidate]
    return {
        "candidate": candidate,
        "role_profile": f"SSB-OLLAMA-{candidate.upper()}-NATIVE-TOOLS-CONTEXT32768-ROLES1",
        "provider": "ollama",
        "model": item["model"],
        "model_digest": item["digest"],
        "ollama_server_version": "0.33.2",
        "config_blob_sha256": item["blob"],
        "modelfile_sha256": item["modelfile"],
        "context_length": 32768,
        "request_profile": "SSB-OLLAMA-NATIVE-TOOLS1",
        "response_contract": "SSB-OLLAMA-NATIVE-TOOLS-RESPONSE2",
        "temperature": item["temperature"],
        "seed": 4242,
        "max_output_tokens": 8192,
        "top_p_wire": "omitted",
        "reasoning_effort_wire": "omitted",
    }


def canonical_identity(
    candidate: Candidate, *, show_response: bytes, ollama_version_text: str
) -> bytes:
    try:
        show = json.loads(show_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatrixConformanceError("identity JSON") from error
    item = CANDIDATES[candidate]
    modelfile = show.get("modelfile") if type(show) is dict else None
    if (
        type(modelfile) is not str
        or show.get("capabilities") != item["capabilities"]
        or "sha256:" + sha256(modelfile.encode()).hexdigest() != item["modelfile"]
        or str(item["blob"]).replace(":", "-", 1) not in modelfile
        or "0.33.2" not in ollama_version_text
    ):
        raise MatrixConformanceError("identity pins")
    body = {
        "record_kind": "OLLAMA_NATIVE_TOOLS_MATRIX_IDENTITY_RECEIPT1",
        **projection(candidate),
        "model_context": item["model_context"],
        "capabilities": item["capabilities"],
        "show_response_hash": "sha256:" + sha256(show_response).hexdigest(),
    }
    return canonical_json_bytes({**body, "receipt_hash": sha256_ref(body)})


def canonical_failure(
    candidate: Candidate,
    *,
    failure_type: str,
    stage: str,
    completed_calls: int,
    calls: list[dict[str, object]],
) -> bytes:
    if (
        failure_type
        not in {"MODEL_OUTPUT_INVALID", "ROLE_CONFORMANCE_INVALID", "CONFIGURATION_ERROR"}
        or stage not in {"baseline", "developer", "tool_call", "receipt"}
        or not 0 <= completed_calls <= 3
    ):
        raise MatrixConformanceError("failure shape")
    body = {
        "record_kind": "OLLAMA_NATIVE_TOOLS_MATRIX_FAILURE_RECEIPT1",
        **projection(candidate),
        "failure_type": failure_type,
        "stage": stage,
        "completed_calls": completed_calls,
        "calls": calls,
    }
    return canonical_json_bytes({**body, "receipt_hash": sha256_ref(body)})


def canonical_receipt(candidate: Candidate, *, calls: list[dict[str, object]]) -> bytes:
    keys = {
        "raw_request_hash",
        "raw_response_hash",
        "usage",
        "attempts",
        "finish_reason",
        "provider_finish_reason",
        "provider_done",
        "role_marker",
        "tool_name",
        "arguments_hash",
        "tool_declaration_hash",
    }
    if len(calls) != 3 or any(
        set(call) != keys
        or call["tool_name"] != "finish_task"
        or call["finish_reason"] != "tool_calls"
        or call["provider_done"] is not True
        or call["attempts"] != 1
        for call in calls
    ):
        raise MatrixConformanceError("receipt calls")
    baseline, developer, tool_call = calls
    if (
        type(baseline["usage"]) is not dict
        or type(developer["usage"]) is not dict
        or developer["usage"].get("input_tokens") == baseline["usage"].get("input_tokens")
    ):
        raise MatrixConformanceError("developer delta")
    body = {
        "record_kind": "OLLAMA_NATIVE_TOOLS_MATRIX_ROLE_TOOL_CONFORMANCE_RECEIPT1",
        **projection(candidate),
        "baseline": baseline,
        "developer": developer,
        "tool_call": tool_call,
        "developer_prompt_token_delta": developer["usage"]["input_tokens"]
        - baseline["usage"]["input_tokens"],
    }
    return canonical_json_bytes({**body, "receipt_hash": sha256_ref(body)})


def validate_receipt(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or value.get("record_kind") != "OLLAMA_NATIVE_TOOLS_MATRIX_ROLE_TOOL_CONFORMANCE_RECEIPT1"
    ):
        raise MatrixConformanceError("receipt kind")
    candidate = load_candidate(cast(str, value.get("candidate")))
    calls = [value.get("baseline"), value.get("developer"), value.get("tool_call")]
    try:
        expected = json.loads(
            canonical_receipt(candidate, calls=cast(list[dict[str, object]], calls))
        )
    except (KeyError, TypeError, MatrixConformanceError) as error:
        raise MatrixConformanceError("receipt fields") from error
    if expected != value:
        raise MatrixConformanceError("receipt projection")
    return cast(dict[str, object], value)


def load_candidate(value: str) -> Candidate:
    if value not in CANDIDATES:
        raise MatrixConformanceError("candidate")
    return cast(Candidate, value)
