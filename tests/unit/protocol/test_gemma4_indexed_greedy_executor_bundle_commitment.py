from __future__ import annotations

import json
from pathlib import Path

from shadowskillbench.core.hashing import sha256_ref

ROOT = Path(__file__).resolve().parents[3] / "protocol"
GREEDY_PATH = ROOT / "development_gemma4_indexed_greedy_executor_bundle_v1.json"
INDEXED_PATH = ROOT / "development_gemma4_indexed_executor_bundle_v1.json"
GEMMA4_V1_PATH = ROOT / "development_gemma4_executor_bundle_v1.json"


def test_greedy_bundle_binds_wire4_and_the_exact_temperature_zero_delta() -> None:
    greedy = json.loads(GREEDY_PATH.read_bytes())
    indexed = json.loads(INDEXED_PATH.read_bytes())
    gemma4_v1 = json.loads(GEMMA4_V1_PATH.read_bytes())

    assert sha256_ref(greedy) == (
        "sha256:f357a5ffb45e14df84e95c216ed974e614896b6cf746c0651f431f05b9376bd6"
    )
    assert greedy["base_commitment_hash"] == sha256_ref(indexed)
    assert greedy["profile"] == "SSB-DEVELOPMENT-GEMMA4-INDEXED-GREEDY-EXECUTOR-BUNDLE1"
    assert greedy["projection"] == indexed["projection"]
    assert greedy["model_runtime"] == gemma4_v1["model_runtime"]
    assert greedy["transport"] == indexed["transport"]
    assert greedy["rules"] == {**indexed["rules"], "temperature": 0.0}
    assert greedy["delta"]["field"] == "rules.temperature"
    assert greedy["delta"]["from"] == 1.0
    assert greedy["delta"]["to"] == 0.0


def test_greedy_selection_keeps_validation_reserved_and_screen_disjoint() -> None:
    greedy = json.loads(GREEDY_PATH.read_bytes())
    gemma4_v1 = json.loads(GEMMA4_V1_PATH.read_bytes())

    screen = greedy["selection"]["screen"]
    validation = gemma4_v1["selection"]["validation"]
    screen_cases = {
        item["case_id"]
        for stage in screen["stages"]
        for entries in stage["cases"].values()
        for item in entries
    }
    validation_cases = {
        item["case_id"]
        for stage in validation["stages"]
        for entries in stage["cases"].values()
        for item in entries
    }

    assert greedy["selection"]["validation"]["source_commitment_hash"] == sha256_ref(gemma4_v1)
    assert not screen_cases & validation_cases
    assert all(
        item["expected_disposition"] == "PROCEED"
        for stage in screen["stages"]
        for entries in stage["cases"].values()
        for item in entries
    )
