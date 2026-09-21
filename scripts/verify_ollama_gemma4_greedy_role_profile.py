"""Record one write-once, temperature-zero Gemma4 V4 conformance receipt."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shadowskillbench.core.hashing import canonical_json_bytes  # noqa: E402
from shadowskillbench.experiments.ollama_gemma4_profile_v4 import (  # noqa: E402
    OLLAMA_GEMMA4_MODEL,
    OLLAMA_GEMMA4_MODEL_DIGEST,
    OLLAMA_GEMMA4_PROVIDER,
    OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS,
    OLLAMA_GEMMA4_SEED,
    OLLAMA_GEMMA4_TEMPERATURE,
    Gemma4RoleMarker,
    canonical_gemma4_role_profile_receipt,
    ollama_gemma4_profile_projection,
)
from shadowskillbench.models.protocol import ModelAdapterError, ModelRequest  # noqa: E402
from shadowskillbench.models.runtime import ModelRunDescriptor, client_from_run_descriptor  # noqa: E402
import verify_ollama_gemma4_role_profile as _v3_checks  # noqa: E402

_FAILURE_RECORD_KIND = "OLLAMA_GEMMA4_ROLE_STRUCTURED_OUTPUT_FAILURE_RECEIPT4"
_PASS_PREFIX = "PASS_OLLAMA_GEMMA4_ROLE_PROFILE_V4"
_SUBJECT = "Gemma4"


class ProbeFailure(RuntimeError):
    def __init__(self, evidence: dict[str, object]) -> None:
        super().__init__(cast(str, evidence["code"]))
        self.evidence = evidence


def _completion(host: str, api_key_environment: str, api_key: str, call: str) -> dict[str, object]:
    descriptor = ModelRunDescriptor(
        provider=OLLAMA_GEMMA4_PROVIDER,
        model=OLLAMA_GEMMA4_MODEL,
        model_version=OLLAMA_GEMMA4_MODEL_DIGEST,
        endpoint=f"{host.rstrip('/')}/v1/chat/completions",
        api_key_environment=api_key_environment,
        max_attempts=1,
    )
    client = client_from_run_descriptor(
        descriptor,
        environment={api_key_environment: api_key},
        trust_env=False,
        timeout_seconds=900,
    )
    try:
        response = asyncio.run(
            client.structured(
                ModelRequest(
                    messages=_v3_checks._probe_messages(call),
                    temperature=OLLAMA_GEMMA4_TEMPERATURE,
                    seed=OLLAMA_GEMMA4_SEED,
                    max_tokens=OLLAMA_GEMMA4_ROLE_PROBE_MAX_TOKENS,
                ),
                Gemma4RoleMarker,
            )
        )
    except ModelAdapterError as error:
        raise ProbeFailure(_v3_checks._failure_evidence(error)) from None
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


def _write_failure(path: Path, evidence: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"{_SUBJECT} role-profile failure receipt already exists")
    if not _v3_checks._valid_failure_evidence(evidence):
        raise RuntimeError(f"{_SUBJECT} role-profile failure evidence is invalid")
    receipt = {
        **ollama_gemma4_profile_projection(),
        "record_kind": _FAILURE_RECORD_KIND,
        "failure": evidence,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt))


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
    _v3_checks._preflight(args.host, args.ollama_bin)
    api_key = os.environ.get(args.api_key_environment)
    if type(api_key) is not str or not api_key:
        raise RuntimeError("configured Ollama API key is unavailable")
    try:
        baseline = _completion(args.host, args.api_key_environment, api_key, "baseline")
        developer = _completion(args.host, args.api_key_environment, api_key, "developer")
    except ProbeFailure as error:
        _write_failure(_v3_checks._failure_path(args.output, args.failure_output), error.evidence)
        raise RuntimeError(
            f"{_SUBJECT} role-conformance probe failed; failure receipt was written"
        ) from None
    evidence = _v3_checks._post_call_failure(baseline, developer)
    if evidence is not None:
        _write_failure(_v3_checks._failure_path(args.output, args.failure_output), evidence)
        raise RuntimeError(f"{_SUBJECT} role-conformance probe failed after both calls")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        canonical_gemma4_role_profile_receipt(baseline=baseline, developer=developer)
    )
    prompt_delta = (
        cast(dict[str, int], developer["usage"])["input_tokens"]
        - cast(dict[str, int], baseline["usage"])["input_tokens"]
    )
    print(
        f"{_PASS_PREFIX}: baseline_marker={baseline['response_marker']} "
        f"developer_marker={developer['response_marker']} prompt_delta={prompt_delta} "
        f"temperature={OLLAMA_GEMMA4_TEMPERATURE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
