from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shadowskillbench.corpus.confirmatory import (
    ConfirmatoryHoldError,
    SplitInventory,
    _current_generator_hash,
    authorize_confirmatory_generation,
    generate_confirmatory_corpus,
)
from tests.custody import write_local_custody_receipt

DEVELOPMENT_SENTINEL = SplitInventory(
    frozenset({-1}),
    frozenset({"development_sentinel"}),
    frozenset({"sha256:" + "0" * 64}),
)


_PREREGISTRATION = b"property-test preregistration\n"


def _authorization(repository_root: Path):
    preregistration_hash = "sha256:" + sha256(_PREREGISTRATION).hexdigest()
    generator_bytes = (
        Path(__file__)
        .parents[3]
        .joinpath("src", "shadowskillbench", "corpus", "confirmatory.py")
        .read_bytes()
    )
    protocol = repository_root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    (protocol / "preregistration.md").write_bytes(_PREREGISTRATION)
    generator_path = repository_root / "src" / "shadowskillbench" / "corpus"
    generator_path.mkdir(parents=True, exist_ok=True)
    (generator_path / "confirmatory.py").write_bytes(generator_bytes)
    manifest_payload = {
        "anchor_status": "PENDING_HUMAN_ANCHOR",
        "inputs": [
            {
                "path": "protocol/preregistration.md",
                "role": "preregistration_core",
                "sha256": preregistration_hash,
            },
            {
                "path": "src/shadowskillbench/corpus/confirmatory.py",
                "role": "confirmatory_corpus",
                "sha256": _current_generator_hash(),
            },
        ],
        "preregistration_core": {
            "path": "protocol/preregistration.md",
            "sha256": preregistration_hash,
        },
        "schema_version": "1.0",
    }
    manifest = json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    (protocol / "freeze_manifest.json").write_bytes(manifest)
    detached = f"{sha256(manifest).hexdigest()}  freeze_manifest.json\n".encode()
    (protocol / "freeze_manifest.sha256").write_bytes(detached)
    receipt = write_local_custody_receipt(repository_root, manifest)
    return authorize_confirmatory_generation(
        manifest,
        detached,
        receipt,
        repository_root=repository_root,
    )


@settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_any_reintroduced_generated_seed_fails_split_isolation(tmp_path: Path, seed: int) -> None:
    authorization = _authorization(tmp_path / f"repository-{seed}")
    corpus = generate_confirmatory_corpus(
        authorization=authorization, corpus_seed=seed, development_inventory=DEVELOPMENT_SENTINEL
    )
    development = SplitInventory(
        frozenset({-1, next(iter(corpus.inventory.seeds))}),
        DEVELOPMENT_SENTINEL.entity_ids,
        DEVELOPMENT_SENTINEL.value_hashes,
    )
    with pytest.raises(ConfirmatoryHoldError, match="split overlap"):
        generate_confirmatory_corpus(
            authorization=authorization, corpus_seed=seed, development_inventory=development
        )


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.binary(min_size=1, max_size=32))
def test_any_detached_manifest_tamper_holds(tmp_path: Path, tamper: bytes) -> None:
    repository_root = tmp_path / f"repository-{sha256(tamper).hexdigest()}"
    authorization = _authorization(repository_root)
    manifest = authorization.freeze.manifest_bytes
    with pytest.raises(ConfirmatoryHoldError):
        authorize_confirmatory_generation(
            manifest + tamper,
            f"{sha256(manifest).hexdigest()}  freeze_manifest.json\n".encode(),
            b"{}",
            repository_root=repository_root,
        )
