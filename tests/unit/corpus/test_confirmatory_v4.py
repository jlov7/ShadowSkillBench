from __future__ import annotations

import json
from hashlib import sha256

import pytest
from typer.testing import CliRunner

import shadowskillbench.corpus.confirmatory_v4 as confirmatory_v4
from shadowskillbench import cli
from shadowskillbench.corpus.confirmatory import ConfirmatoryHoldError, SplitInventory
from shadowskillbench.corpus.confirmatory_v4 import (
    CORPUS_SEED_V4,
    ConfirmatoryAuthorizationV4,
    authorize_confirmatory_generation_v4,
    generate_confirmatory_corpus_v4,
)


def _issued_authorization() -> ConfirmatoryAuthorizationV4:
    return ConfirmatoryAuthorizationV4._issue(
        freeze_manifest_hash="sha256:" + "a" * 64,
        anchor_receipt_hash="sha256:" + "b" * 64,
        issuer=confirmatory_v4._AUTHORIZATION_ISSUER,
    )


def _manifest() -> tuple[bytes, bytes]:
    payload = {
        "anchor_status": "HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY",
        "inputs": [],
        "profile": "SSB-SCIENTIFIC-PREFREEZE-4",
        "schema_version": "4.0",
    }
    raw = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    return raw, f"{sha256(raw).hexdigest()}  freeze_manifest.v4.json\n".encode()


def test_v4_authorization_rejects_v3_manifest_bytes() -> None:
    raw, detached = _manifest()
    v3 = raw.replace(b"SSB-SCIENTIFIC-PREFREEZE-4", b"SSB-SCIENTIFIC-PREFREEZE-3")
    v3_detached = f"{sha256(v3).hexdigest()}  freeze_manifest.v4.json\n".encode()

    with pytest.raises(ConfirmatoryHoldError, match="not V4"):
        authorize_confirmatory_generation_v4(v3, v3_detached, b"v4-receipt")
    with pytest.raises(ConfirmatoryHoldError, match="HOLD_PENDING_NEMOTRON"):
        authorize_confirmatory_generation_v4(raw, detached, b"arbitrary-receipt")


def test_v4_generation_rejects_an_arbitrary_seed() -> None:
    authorization = _issued_authorization()
    inventory = SplitInventory(
        seeds=frozenset({1}),
        entity_ids=frozenset({"development_0000000000000000"}),
        value_hashes=frozenset({"sha256:" + "0" * 64}),
    )

    with pytest.raises(ConfirmatoryHoldError, match="corpus seed"):
        generate_confirmatory_corpus_v4(
            authorization=authorization,
            corpus_seed=104730,
            development_inventory=inventory,
        )


def test_v4_generation_requires_inventory_covering_fresh_seed_4243() -> None:
    inventory = SplitInventory(
        seeds=frozenset({4242}),
        entity_ids=frozenset({"development_0000000000000000"}),
        value_hashes=frozenset({"sha256:" + "0" * 64}),
    )
    with pytest.raises(ConfirmatoryHoldError, match="4242 and 4243"):
        generate_confirmatory_corpus_v4(
            authorization=_issued_authorization(),
            corpus_seed=CORPUS_SEED_V4,
            development_inventory=inventory,
        )


def test_v4_cli_corpus_generation_fails_closed_without_creating_output(tmp_path) -> None:
    output = tmp_path / "v4-corpus.json"
    result = CliRunner().invoke(
        cli.app, ["corpus", "generate-confirmatory-v4", "--output", str(output)]
    )

    assert result.exit_code == 1
    assert "HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY" in result.output
    assert not output.exists()
