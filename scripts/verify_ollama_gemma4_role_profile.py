"""Record one write-once, role-faithful Gemma4 V3 structured-output receipt."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import cast
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shadowskillbench.core.hashing import canonical_json_bytes  # noqa: E402
from shadowskillbench.experiments.ollama_gemma4_profile_v3 import (  # noqa: E402
    OLLAMA_GEMMA4_MODEFILE_SHA256,
    OLLAMA_GEMMA4_MODEL,
    OLLAMA_GEMMA4_MODEL_DIGEST,
    OLLAMA_GEMMA4_PROVIDER,
    OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS,
    OLLAMA_GEMMA4_SEED,
    OLLAMA_GEMMA4_SERVER_VERSION,
    OLLAMA_GEMMA4_TEMPERATURE,
    Gemma4RoleMarker,
    canonical_gemma4_role_profile_receipt,
    ollama_gemma4_profile_projection,
)
from shadowskillbench.models.protocol import (  # noqa: E402
    Message,
    ModelAdapterError,
    ModelRequest,
)
from shadowskillbench.models.runtime import (  # noqa: E402
    ModelRunDescriptor,
    client_from_run_descriptor,
)

_ENDPOINT_PATH = "/v1/chat/completions"
_MAX_TOKENS = OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS
_FAILURE_RECORD_KIND = "OLLAMA_GEMMA4_ROLE_STRUCTURED_OUTPUT_FAILURE_RECEIPT3"
_PASS_PREFIX = "PASS_OLLAMA_GEMMA4_ROLE_PROFILE_V3"
_SUBJECT = "Gemma4"


class ProbeFailure(RuntimeError):
    def __init__(self, evidence: dict[str, object]) -> None:
        super().__init__(cast(str, evidence["code"]))
        self.evidence = evidence


def _command(ollama: str, *args: str, host: str | None = None) -> str:
    environment = None if host is None else {**os.environ, "OLLAMA_HOST": host.rstrip("/")}
    try:
        result = subprocess.run(
            [ollama, *args], check=True, capture_output=True, text=True, timeout=30, env=environment
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Ollama inspection command failed") from error
    return result.stdout


def _canonical_digest(value: object) -> str:
    if type(value) is not str:
        raise RuntimeError("Ollama model digest is invalid")
    bare = value.removeprefix("sha256:").lower()
    if len(bare) != 64 or any(character not in "0123456789abcdef" for character in bare):
        raise RuntimeError("Ollama model digest is invalid")
    return f"sha256:{bare}"


def _model_digest(host: str, model: str) -> str:
    try:
        with urlopen(f"{host}/api/tags", timeout=30) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (OSError, TimeoutError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama tag inspection failed") from error
    models = payload.get("models") if type(payload) is dict else None
    if type(models) is not list:
        raise RuntimeError("Ollama tag inspection returned an invalid payload")
    for item in models:
        if type(item) is dict and item.get("name") == model and type(item.get("digest")) is str:
            return _canonical_digest(item["digest"])
    raise RuntimeError(f"Ollama tag inspection did not report {model}")


def _server_version(host: str) -> str:
    try:
        with urlopen(f"{host}/api/version", timeout=30) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (OSError, TimeoutError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama version inspection failed") from error
    version = payload.get("version") if type(payload) is dict else None
    if type(version) is not str or not version:
        raise RuntimeError("Ollama version inspection returned an invalid payload")
    return version


def _probe_messages(call: str) -> tuple[Message, ...]:
    projection = ollama_gemma4_profile_projection()
    probe = cast(dict[str, object], projection["role_structured_output_probe"])
    item = probe.get(call)
    if type(item) is not dict or type(item.get("messages")) is not list:
        raise RuntimeError(f"{_SUBJECT} role probe is invalid")
    return tuple(Message.model_validate(message) for message in item["messages"])


def _usage(value: object) -> dict[str, int] | None:
    if value is None or not all(
        type(getattr(value, field, None)) is int
        for field in ("input_tokens", "output_tokens", "total_tokens")
    ):
        return None
    return {
        "input_tokens": getattr(value, "input_tokens"),
        "output_tokens": getattr(value, "output_tokens"),
        "total_tokens": getattr(value, "total_tokens"),
    }


def _failure_evidence(error: ModelAdapterError) -> dict[str, object]:
    return {
        "code": error.code,
        "attempts": error.attempts,
        "raw_request_hash": error.raw_request_hash,
        "raw_response_hash": error.raw_response_hash,
        "finish_reason": error.finish_reason,
        "usage": _usage(error.reported_usage),
    }


def _failure_call_summary(call: dict[str, object]) -> dict[str, object]:
    return {
        "attempts": call["attempts"],
        "raw_request_hash": call["raw_request_hash"],
        "raw_response_hash": call["raw_response_hash"],
        "finish_reason": "stop",
        "usage": call["usage"],
    }


def _completion(host: str, api_key_environment: str, api_key: str, call: str) -> dict[str, object]:
    descriptor = ModelRunDescriptor(
        provider=OLLAMA_GEMMA4_PROVIDER,
        model=OLLAMA_GEMMA4_MODEL,
        model_version=OLLAMA_GEMMA4_MODEL_DIGEST,
        endpoint=f"{host.rstrip('/')}{_ENDPOINT_PATH}",
        api_key_environment=api_key_environment,
        max_attempts=1,
    )
    client = client_from_run_descriptor(
        descriptor,
        environment={api_key_environment: api_key},
        trust_env=False,
        timeout_seconds=900,
    )
    request = ModelRequest(
        messages=_probe_messages(call),
        temperature=OLLAMA_GEMMA4_TEMPERATURE,
        seed=OLLAMA_GEMMA4_SEED,
        max_tokens=_MAX_TOKENS,
    )
    try:
        response = asyncio.run(client.structured(request, Gemma4RoleMarker))
    except ModelAdapterError as error:
        raise ProbeFailure(_failure_evidence(error)) from None
    return {
        "attempts": response.attempts,
        "raw_request_hash": response.raw_request_hash,
        "raw_response_hash": response.raw_response_hash,
        "response_marker": response.output.marker,
        "structured_output_schema_hash": response.structured_output_schema_hash,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def _failure_path(output: Path, explicit: Path | None) -> Path:
    return explicit if explicit is not None else output.with_name(f"{output.stem}.failed.json")


def _valid_digest_or_none(value: object) -> bool:
    return value is None or (
        type(value) is str
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _valid_usage_or_none(value: object) -> bool:
    if value is None:
        return True
    if type(value) is not dict or set(value) != {"input_tokens", "output_tokens", "total_tokens"}:
        return False
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    total_tokens = value.get("total_tokens")
    return (
        type(input_tokens) is int
        and type(output_tokens) is int
        and type(total_tokens) is int
        and input_tokens >= 0
        and output_tokens >= 0
        and total_tokens == input_tokens + output_tokens
    )


def _valid_failure_summary(value: object, *, maximum_attempts: int) -> bool:
    if type(value) is not dict or set(value) != {
        "attempts",
        "raw_request_hash",
        "raw_response_hash",
        "finish_reason",
        "usage",
    }:
        return False
    attempts = value.get("attempts")
    finish_reason = value.get("finish_reason")
    return (
        type(attempts) is int
        and 0 <= attempts <= maximum_attempts
        and _valid_digest_or_none(value.get("raw_request_hash"))
        and _valid_digest_or_none(value.get("raw_response_hash"))
        and (finish_reason is None or type(finish_reason) is str and len(finish_reason) <= 32)
        and _valid_usage_or_none(value.get("usage"))
    )


def _valid_failure_evidence(value: object) -> bool:
    if type(value) is not dict or type(value.get("code")) is not str:
        return False
    direct_failure = {
        "code",
        "attempts",
        "raw_request_hash",
        "raw_response_hash",
        "finish_reason",
        "usage",
    }
    if set(value) == direct_failure:
        summary = {key: value[key] for key in direct_failure - {"code"}}
        return _valid_failure_summary(summary, maximum_attempts=1)
    postcall_failure = {"code", "attempts", "reasons", "baseline", "developer"}
    return (
        set(value) == postcall_failure
        and value.get("attempts") == 2
        and type(value.get("reasons")) is list
        and bool(value["reasons"])
        and all(type(reason) is str for reason in cast(list[object], value["reasons"]))
        and _valid_failure_summary(value.get("baseline"), maximum_attempts=1)
        and _valid_failure_summary(value.get("developer"), maximum_attempts=1)
    )


def _write_failure(path: Path, evidence: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"{_SUBJECT} role-profile failure receipt already exists")
    if not _valid_failure_evidence(evidence):
        raise RuntimeError(f"{_SUBJECT} role-profile failure evidence is invalid")
    receipt = {
        **ollama_gemma4_profile_projection(),
        "record_kind": _FAILURE_RECORD_KIND,
        "failure": evidence,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt))


def _preflight(host: str, ollama_bin: str) -> None:
    parsed = urlparse(host)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(f"{_SUBJECT} role-profile host must be exact loopback HTTP")
    if _server_version(host) != OLLAMA_GEMMA4_SERVER_VERSION:
        raise RuntimeError(f"Ollama server version does not match the frozen {_SUBJECT} profile")
    if _model_digest(host, OLLAMA_GEMMA4_MODEL) != OLLAMA_GEMMA4_MODEL_DIGEST:
        raise RuntimeError(f"{_SUBJECT} model digest does not match the frozen profile")
    modelfile = _command(ollama_bin, "show", OLLAMA_GEMMA4_MODEL, "--modelfile", host=host)
    if ("sha256:" + sha256(modelfile.encode("utf-8")).hexdigest()) != OLLAMA_GEMMA4_MODEFILE_SHA256:
        raise RuntimeError(f"{_SUBJECT} Modelfile does not match the frozen profile")


def _post_call_failure(
    baseline: dict[str, object], developer: dict[str, object]
) -> dict[str, object] | None:
    projection = ollama_gemma4_profile_projection()
    probe = cast(dict[str, object], projection["role_structured_output_probe"])
    expected_baseline = cast(dict[str, object], probe["baseline"])["expected_marker"]
    expected_developer = cast(dict[str, object], probe["developer"])["expected_marker"]
    baseline_usage = cast(dict[str, int], baseline["usage"])
    developer_usage = cast(dict[str, int], developer["usage"])
    reasons: list[str] = []
    if baseline["response_marker"] != expected_baseline:
        reasons.append("BASELINE_MARKER_MISMATCH")
    if developer["response_marker"] != expected_developer:
        reasons.append("DEVELOPER_MARKER_MISMATCH")
    if developer_usage["input_tokens"] == baseline_usage["input_tokens"]:
        reasons.append("PROMPT_TOKEN_DIFFERENCE_ZERO")
    if baseline["raw_request_hash"] == developer["raw_request_hash"]:
        reasons.append("RAW_REQUEST_HASH_EQUAL")
    if baseline["raw_response_hash"] == developer["raw_response_hash"]:
        reasons.append("RAW_RESPONSE_HASH_EQUAL")
    if not reasons:
        return None
    return {
        "code": "ROLE_PROBE_POSTCALL_CONFORMANCE_FAILED",
        "attempts": 2,
        "reasons": reasons,
        "baseline": _failure_call_summary(baseline),
        "developer": _failure_call_summary(developer),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ollama-bin", default="ollama")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--api-key-environment", default="OLLAMA_API_KEY")
    parser.add_argument("--failure-output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"{_SUBJECT} role-profile receipt already exists")
    _preflight(args.host, args.ollama_bin)
    api_key = os.environ.get(args.api_key_environment)
    if type(api_key) is not str or not api_key:
        raise RuntimeError("configured Ollama API key is unavailable")
    try:
        baseline = _completion(args.host, args.api_key_environment, api_key, "baseline")
        developer = _completion(args.host, args.api_key_environment, api_key, "developer")
    except ProbeFailure as error:
        _write_failure(_failure_path(args.output, args.failure_output), error.evidence)
        raise RuntimeError(
            f"{_SUBJECT} role-conformance probe failed; failure receipt was written"
        ) from None
    evidence = _post_call_failure(baseline, developer)
    if evidence is not None:
        _write_failure(_failure_path(args.output, args.failure_output), evidence)
        raise RuntimeError(f"{_SUBJECT} role-conformance probe failed after both calls")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        canonical_gemma4_role_profile_receipt(baseline=baseline, developer=developer)
    )
    baseline_marker = baseline["response_marker"]
    developer_marker = developer["response_marker"]
    prompt_delta = (
        cast(dict[str, int], developer["usage"])["input_tokens"]
        - cast(dict[str, int], baseline["usage"])["input_tokens"]
    )
    print(
        f"{_PASS_PREFIX}: "
        f"baseline_marker={baseline_marker} developer_marker={developer_marker} "
        f"prompt_delta={prompt_delta} temperature={OLLAMA_GEMMA4_TEMPERATURE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
