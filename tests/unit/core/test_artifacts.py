from __future__ import annotations

import pytest
from pydantic import ValidationError

from shadowskillbench.core.artifacts import ArtifactEnvelope, ArtifactHashMismatch
from shadowskillbench.core.hashing import sha256_ref

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def payload_path(artifact_type: str, payload: object) -> str:
    digest = sha256_ref(payload).removeprefix("sha256:")
    return f"artifacts/{artifact_type}/{digest[:2]}/{digest}.json"


def test_factory_binds_payload_reference_hash_and_canonical_path() -> None:
    payload = {"case": "alpha", "values": [1, 2]}

    envelope = ArtifactEnvelope.from_payload(
        artifact_type="episode-plan",
        schema_version="1.0",
        created_by_commit=COMMIT,
        payload=payload,
    )

    expected_ref = sha256_ref(payload)
    digest = expected_ref.removeprefix("sha256:")
    assert envelope.content_hash == expected_ref
    assert envelope.payload_ref == f"artifacts/episode-plan/{digest[:2]}/{digest}.json"
    assert envelope.canonical_path == f"artifacts/episode-plan/{digest[:2]}/{digest}.json"
    assert envelope.verify_payload(payload) is None
    assert envelope.verify_payload_hash(payload) is None


def test_payload_tampering_fails_hash_verification() -> None:
    envelope = ArtifactEnvelope.create(
        artifact_type="trace",
        schema_version="2.1",
        created_by_commit=COMMIT,
        payload={"steps": ["first"]},
    )

    with pytest.raises(ArtifactHashMismatch):
        envelope.verify_payload({"steps": ["tampered"]})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_type", "../protocol"),
        ("artifact_type", "trace/escape"),
        ("artifact_type", "Trace"),
        ("schema_version", "1.0.0"),
        ("schema_version", "v1.0"),
        ("created_by_commit", "a" * 39),
        ("created_by_commit", "A" * 40),
        ("content_hash", "sha256:" + "A" * 64),
        ("payload_ref", "../../protocol/freeze_manifest.json"),
    ],
)
def test_envelope_rejects_invalid_identity_or_path_fields(field: str, value: str) -> None:
    fields = {
        "artifact_type": "trace",
        "schema_version": "1.0",
        "created_by_commit": COMMIT,
        "content_hash": sha256_ref({"payload": "one"}),
        "payload_ref": payload_path("trace", {"payload": "one"}),
    }
    fields[field] = value

    with pytest.raises(ValidationError):
        ArtifactEnvelope(**fields)


def test_envelope_rejects_payload_ref_that_does_not_bind_its_type_and_hash() -> None:
    payload = {"payload": "one"}

    with pytest.raises(ValidationError):
        ArtifactEnvelope(
            artifact_type="trace",
            schema_version="1.0",
            created_by_commit=COMMIT,
            content_hash=sha256_ref(payload),
            payload_ref=payload_path("other-type", payload),
        )
