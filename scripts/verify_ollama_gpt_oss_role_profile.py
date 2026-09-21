"""Record a matched local role-conformance receipt for the pilot's Ollama profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import cast
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shadowskillbench.core.hashing import canonical_json_bytes  # noqa: E402
from shadowskillbench.experiments.ollama_gpt_oss_profile import (  # noqa: E402
    OLLAMA_GPT_OSS_BASE_MODEL,
    OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
    OLLAMA_GPT_OSS_CONTEXT_LENGTH,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    OLLAMA_GPT_OSS_MODEL,
    OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_SERVER_VERSION,
    canonical_role_profile_receipt,
    ollama_gpt_oss_executor_template,
    ollama_gpt_oss_profile_projection,
    ollama_gpt_oss_template,
    role_conformance_probe_projection,
)


class ProbeFailure(RuntimeError):
    def __init__(self, code: str, evidence: dict[str, object]) -> None:
        super().__init__(code)
        self.code = code
        self.evidence = evidence


def _command(ollama: str, *args: str) -> str:
    try:
        result = subprocess.run(
            [ollama, *args], check=True, capture_output=True, text=True, timeout=30
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


def _completion(
    host: str, api_key: str, messages: list[dict[str, str]], call: str
) -> dict[str, object]:
    probe = role_conformance_probe_projection()
    body = json.dumps(
        {
            "model": OLLAMA_GPT_OSS_MODEL,
            "messages": messages,
            "temperature": OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            "seed": 4242,
            "max_tokens": 256,
            "stream": False,
            "reasoning_effort": probe["reasoning_effort"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ollama_role_conformance",
                    "strict": True,
                    "schema": probe["response_schema"],
                },
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        f"{host.rstrip('/')}/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=900) as response:  # noqa: S310
            raw_response = response.read()
    except (OSError, TimeoutError, URLError) as error:
        raise RuntimeError("Ollama role-conformance call failed") from error
    evidence: dict[str, object] = {
        "call": call,
        "raw_request_hash": "sha256:" + sha256(body).hexdigest(),
        "raw_response_hash": "sha256:" + sha256(raw_response).hexdigest(),
        "usage": None,
    }
    try:
        payload = json.loads(raw_response)
        if type(payload) is not dict:
            raise TypeError("payload")
        usage = payload.get("usage")
        if type(usage) is not dict:
            raise TypeError("usage")
        parsed_usage = {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if type(usage.get(key)) is int
        }
        evidence["usage"] = parsed_usage or None
        choices = payload["choices"]
        message = choices[0]["message"]
        content = json.loads(message["content"])
        marker = content["marker"]
        prompt_tokens = usage["prompt_tokens"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ProbeFailure("ROLE_PROBE_RESPONSE_INVALID", evidence) from error
    if (
        payload.get("model") != OLLAMA_GPT_OSS_MODEL
        or type(marker) is not str
        or type(prompt_tokens) is not int
        or prompt_tokens <= 0
    ):
        raise ProbeFailure("ROLE_PROBE_CONFORMANCE_INVALID", evidence)
    return {
        "raw_request_hash": evidence["raw_request_hash"],
        "raw_response_hash": evidence["raw_response_hash"],
        "prompt_tokens": prompt_tokens,
        "response_marker": marker,
        "usage": evidence["usage"],
    }


def _failure_path(output: Path, explicit: Path | None) -> Path:
    return explicit if explicit is not None else output.with_name(f"{output.stem}.failed.json")


def _write_failure(
    path: Path, profile_digest: str, executor_digest: str, evidence: dict[str, object]
) -> None:
    code = evidence.get("code")
    if type(code) is not str:
        raise RuntimeError("role-probe failure evidence is invalid")
    receipt = {
        **ollama_gpt_oss_profile_projection(),
        "record_kind": "OLLAMA_ROLE_CONFORMANCE_FAILURE_RECEIPT1",
        "profile_model_digest": profile_digest,
        "executor_model_digest": executor_digest,
        "failure": {
            "code": code,
            **{key: value for key, value in evidence.items() if key != "code"},
        },
    }
    if path.exists():
        raise RuntimeError("role-probe failure receipt already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt))


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _receipt_call(call: dict[str, object]) -> dict[str, object]:
    return {
        key: call[key]
        for key in ("raw_request_hash", "raw_response_hash", "prompt_tokens", "response_marker")
    }


def _failure_call_summary(call: dict[str, object]) -> dict[str, object]:
    return {
        key: call[key]
        for key in (
            "raw_request_hash",
            "raw_response_hash",
            "prompt_tokens",
            "response_marker",
            "usage",
        )
    }


def _post_call_failure_evidence(
    profile_digest: str, baseline: dict[str, object], developer: dict[str, object]
) -> dict[str, object] | None:
    probe = role_conformance_probe_projection()
    reasons: list[str] = []
    if not _valid_digest(profile_digest):
        reasons.append("PROFILE_MODEL_DIGEST_INVALID")
    if baseline["response_marker"] != probe["baseline_expected_marker"]:
        reasons.append("BASELINE_MARKER_MISMATCH")
    if developer["response_marker"] != probe["developer_expected_marker"]:
        reasons.append("DEVELOPER_MARKER_MISMATCH")
    prompt_difference = cast(int, developer["prompt_tokens"]) - cast(int, baseline["prompt_tokens"])
    if prompt_difference <= 0:
        reasons.append("PROMPT_TOKEN_DIFFERENCE_NONPOSITIVE")
    if baseline["raw_request_hash"] == developer["raw_request_hash"]:
        reasons.append("RAW_REQUEST_HASH_EQUAL")
    if baseline["raw_response_hash"] == developer["raw_response_hash"]:
        reasons.append("RAW_RESPONSE_HASH_EQUAL")
    if not reasons:
        return None
    return {
        "code": "ROLE_PROBE_POSTCALL_CONFORMANCE_FAILED",
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
    if _server_version(args.host) != OLLAMA_GPT_OSS_SERVER_VERSION:
        raise RuntimeError("Ollama server version does not match the pinned pilot profile")
    base_digest = _model_digest(args.host, OLLAMA_GPT_OSS_BASE_MODEL).lower()
    profile_digest = _model_digest(args.host, OLLAMA_GPT_OSS_MODEL).lower()
    executor_digest = _model_digest(args.host, OLLAMA_GPT_OSS_EXECUTOR_MODEL).lower()
    if base_digest != OLLAMA_GPT_OSS_BASE_MODEL_DIGEST:
        raise RuntimeError("gpt-oss:20b digest does not match the pinned pilot profile")
    if executor_digest != OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST:
        raise RuntimeError("executor digest does not match the pinned pilot profile")
    modelfile = _command(args.ollama_bin, "show", OLLAMA_GPT_OSS_MODEL, "--modelfile")
    if (
        ollama_gpt_oss_template() not in modelfile
        or f"PARAMETER num_ctx {OLLAMA_GPT_OSS_CONTEXT_LENGTH}" not in modelfile
    ):
        raise RuntimeError("installed Ollama model does not preserve the pinned Harmony template")
    executor_modelfile = _command(
        args.ollama_bin, "show", OLLAMA_GPT_OSS_EXECUTOR_MODEL, "--modelfile"
    )
    if (
        ollama_gpt_oss_executor_template() not in executor_modelfile
        or f"PARAMETER num_ctx {OLLAMA_GPT_OSS_CONTEXT_LENGTH}" not in executor_modelfile
    ):
        raise RuntimeError(
            "installed Ollama executor model does not preserve the pinned final template"
        )
    if args.output.exists():
        raise RuntimeError("role profile receipt already exists")
    api_key = os.environ.get(args.api_key_environment)
    if type(api_key) is not str or not api_key:
        raise RuntimeError("configured Ollama API key is unavailable")
    probe = role_conformance_probe_projection()
    visibility_probe = cast(dict[str, str], probe["semantic_visibility_probe"])
    try:
        baseline = _completion(
            args.host,
            api_key,
            [
                {
                    "role": "system",
                    "content": visibility_probe["system_content"],
                },
                {
                    "role": "user",
                    "content": visibility_probe["user_content"],
                },
            ],
            "baseline",
        )
        developer = _completion(
            args.host,
            api_key,
            [
                {
                    "role": "system",
                    "content": visibility_probe["system_content"],
                },
                {
                    "role": "developer",
                    "content": visibility_probe["developer_content"],
                },
                {
                    "role": "user",
                    "content": visibility_probe["user_content"],
                },
            ],
            "developer",
        )
    except ProbeFailure as error:
        _write_failure(
            _failure_path(args.output, args.failure_output),
            profile_digest,
            executor_digest,
            {"code": error.code, **error.evidence},
        )
        raise RuntimeError(
            "Ollama role-conformance probe failed; failure receipt was written"
        ) from error
    failure_evidence = _post_call_failure_evidence(profile_digest, baseline, developer)
    if failure_evidence is not None:
        _write_failure(
            _failure_path(args.output, args.failure_output),
            profile_digest,
            executor_digest,
            failure_evidence,
        )
        reasons = ", ".join(cast(list[str], failure_evidence["reasons"]))
        raise RuntimeError(
            "Ollama role-conformance probe failed after both calls; "
            f"failure receipt was written ({reasons})"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        canonical_role_profile_receipt(
            profile_digest,
            baseline=_receipt_call(baseline),
            developer=_receipt_call(developer),
            executor_model_digest=executor_digest,
        )
    )
    print(f"PASS_OLLAMA_GPT_OSS_ROLE_PROFILE: receipt={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
