from __future__ import annotations

import json
from pathlib import Path

from shadowskillbench.core.hashing import sha256_ref

ROOT = Path(__file__).resolve().parents[3] / "protocol"
INDEXED_PATH = ROOT / "development_gemma4_indexed_executor_bundle_v1.json"
MISTRAL_PATH = ROOT / "development_mistral_executor_bundle_v1.json"
GEMMA4_V1_PATH = ROOT / "development_gemma4_executor_bundle_v1.json"


def test_gemma4_indexed_bundle_binds_its_identity_and_selection_sources() -> None:
    indexed = json.loads(INDEXED_PATH.read_bytes())
    mistral = json.loads(MISTRAL_PATH.read_bytes())
    gemma4_v1 = json.loads(GEMMA4_V1_PATH.read_bytes())

    assert sha256_ref(indexed) == (
        "sha256:32c78d145edffb8bf18c3e9b92168678204a46b18b9fd424b299f59e4def6a39"
    )
    assert indexed["profile"] == "SSB-DEVELOPMENT-GEMMA4-INDEXED-EXECUTOR-BUNDLE1"
    assert (
        indexed["selection_hash"]
        == "sha256:c5147f887cf8e97fbcea8f73de7d2dbfecb558c4d4fb47b9d9553c9cfe1f0e95"
    )
    assert indexed["selection"] == {
        "screen_source_commitment_hash": sha256_ref(mistral),
        "screen_status": "UNEXECUTED_BECAUSE_MISTRAL_ROLE_GATE_HELD",
        "validation_source_commitment_hash": sha256_ref(gemma4_v1),
        "validation_status": "UNEXECUTED_RESERVED",
    }
