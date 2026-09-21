"""Record one write-once V2 context-32k Nemotron role receipt."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import verify_ollama_gemma4_role_profile as verifier  # noqa: E402
from shadowskillbench.experiments.ollama_nemotron_context32k_profile import (  # noqa: E402
    OLLAMA_NEMOTRON_CONTEXT32K_MODEFILE_SHA256,
    OLLAMA_NEMOTRON_CONTEXT32K_MODEL,
    OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST,
    OLLAMA_NEMOTRON_CONTEXT32K_PROVIDER,
    OLLAMA_NEMOTRON_CONTEXT32K_ROLE_PROBE_MAX_TOKENS,
    OLLAMA_NEMOTRON_CONTEXT32K_SEED,
    OLLAMA_NEMOTRON_CONTEXT32K_SERVER_VERSION,
    OLLAMA_NEMOTRON_CONTEXT32K_TEMPERATURE,
    NemotronContext32kRoleMarker,
    canonical_nemotron_context32k_role_profile_receipt,
    ollama_nemotron_context32k_profile_projection,
)

verifier.OLLAMA_GEMMA4_MODEFILE_SHA256 = OLLAMA_NEMOTRON_CONTEXT32K_MODEFILE_SHA256
verifier.OLLAMA_GEMMA4_MODEL = OLLAMA_NEMOTRON_CONTEXT32K_MODEL
verifier.OLLAMA_GEMMA4_MODEL_DIGEST = OLLAMA_NEMOTRON_CONTEXT32K_MODEL_DIGEST
verifier.OLLAMA_GEMMA4_PROVIDER = OLLAMA_NEMOTRON_CONTEXT32K_PROVIDER
verifier.OLLAMA_GEMMA4_SEED = OLLAMA_NEMOTRON_CONTEXT32K_SEED
verifier.OLLAMA_GEMMA4_SERVER_VERSION = OLLAMA_NEMOTRON_CONTEXT32K_SERVER_VERSION
verifier.OLLAMA_GEMMA4_TEMPERATURE = OLLAMA_NEMOTRON_CONTEXT32K_TEMPERATURE
verifier.Gemma4RoleMarker = NemotronContext32kRoleMarker
verifier.canonical_gemma4_role_profile_receipt = canonical_nemotron_context32k_role_profile_receipt
verifier.ollama_gemma4_profile_projection = ollama_nemotron_context32k_profile_projection
verifier._MAX_TOKENS = OLLAMA_NEMOTRON_CONTEXT32K_ROLE_PROBE_MAX_TOKENS
verifier._FAILURE_RECORD_KIND = "OLLAMA_NEMOTRON_CONTEXT32K_ROLE_STRUCTURED_OUTPUT_FAILURE_RECEIPT2"
verifier._PASS_PREFIX = "PASS_OLLAMA_NEMOTRON_CONTEXT32K_ROLE_PROFILE_V2"
verifier._SUBJECT = "Nemotron 3.5 Lightning 30B MLX context 32768"
verifier.__doc__ = __doc__


if __name__ == "__main__":
    raise SystemExit(verifier.main())
