"""Record one write-once Mistral Small 3.2 role/structured-output receipt."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import verify_ollama_gemma4_role_profile as verifier  # noqa: E402
from shadowskillbench.experiments.ollama_mistral_profile import (  # noqa: E402
    OLLAMA_MISTRAL_MODEFILE_SHA256,
    OLLAMA_MISTRAL_MODEL,
    OLLAMA_MISTRAL_MODEL_DIGEST,
    OLLAMA_MISTRAL_PROVIDER,
    OLLAMA_MISTRAL_ROLE_PROBE_MAX_TOKENS,
    OLLAMA_MISTRAL_SEED,
    OLLAMA_MISTRAL_SERVER_VERSION,
    OLLAMA_MISTRAL_TEMPERATURE,
    MistralRoleMarker,
    canonical_mistral_role_profile_receipt,
    ollama_mistral_profile_projection,
)

verifier.OLLAMA_GEMMA4_MODEFILE_SHA256 = OLLAMA_MISTRAL_MODEFILE_SHA256
verifier.OLLAMA_GEMMA4_MODEL = OLLAMA_MISTRAL_MODEL
verifier.OLLAMA_GEMMA4_MODEL_DIGEST = OLLAMA_MISTRAL_MODEL_DIGEST
verifier.OLLAMA_GEMMA4_PROVIDER = OLLAMA_MISTRAL_PROVIDER
verifier.OLLAMA_GEMMA4_SEED = OLLAMA_MISTRAL_SEED
verifier.OLLAMA_GEMMA4_SERVER_VERSION = OLLAMA_MISTRAL_SERVER_VERSION
verifier.OLLAMA_GEMMA4_TEMPERATURE = OLLAMA_MISTRAL_TEMPERATURE
verifier.Gemma4RoleMarker = MistralRoleMarker
verifier.canonical_gemma4_role_profile_receipt = canonical_mistral_role_profile_receipt
verifier.ollama_gemma4_profile_projection = ollama_mistral_profile_projection
verifier._MAX_TOKENS = OLLAMA_MISTRAL_ROLE_PROBE_MAX_TOKENS
verifier._FAILURE_RECORD_KIND = "OLLAMA_MISTRAL_ROLE_STRUCTURED_OUTPUT_FAILURE_RECEIPT1"
verifier._PASS_PREFIX = "PASS_OLLAMA_MISTRAL_ROLE_PROFILE_V1"
verifier._SUBJECT = "Mistral Small 3.2"
verifier.__doc__ = __doc__


if __name__ == "__main__":
    raise SystemExit(verifier.main())
