from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from shadowskillbench.core.hashing import sha256_ref

_ARTIFACT_TYPE = re.compile(r"[a-z][a-z0-9_-]*\Z")
_SCHEMA_VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_REF = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PAYLOAD_REF = re.compile(r"artifacts/[a-z][a-z0-9_-]*/[0-9a-f]{2}/[0-9a-f]{64}\.json\Z")


class ArtifactHashMismatch(ValueError):
    pass


def _payload_ref_for(artifact_type: str, content_hash: str) -> str:
    digest = content_hash.removeprefix("sha256:")
    return f"artifacts/{artifact_type}/{digest[:2]}/{digest}.json"


class ArtifactEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_type: str
    schema_version: str
    content_hash: str
    created_by_commit: str
    payload_ref: str

    @field_validator("artifact_type")
    @classmethod
    def validate_artifact_type(cls, value: str) -> str:
        if not _ARTIFACT_TYPE.fullmatch(value):
            raise ValueError("artifact_type must be a lowercase path-safe identifier")
        return value

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if not _SCHEMA_VERSION.fullmatch(value):
            raise ValueError("schema_version must use the major.minor grammar")
        return value

    @field_validator("created_by_commit")
    @classmethod
    def validate_created_by_commit(cls, value: str) -> str:
        if not _COMMIT.fullmatch(value):
            raise ValueError("created_by_commit must be 40 lowercase hexadecimal characters")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_sha256_ref(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("content_hash must be a lowercase sha256 reference")
        return value

    @field_validator("payload_ref")
    @classmethod
    def validate_payload_ref(cls, value: str) -> str:
        if not _PAYLOAD_REF.fullmatch(value):
            raise ValueError("payload_ref must be a canonical relative artifact path")
        return value

    @model_validator(mode="after")
    def bind_content_to_payload(self) -> Self:
        if self.payload_ref != _payload_ref_for(self.artifact_type, self.content_hash):
            raise ValueError("payload_ref must bind the artifact type and content hash")
        return self

    @classmethod
    def from_payload(
        cls,
        *,
        artifact_type: str,
        schema_version: str,
        created_by_commit: str,
        payload: object,
    ) -> Self:
        content_hash = sha256_ref(payload)
        return cls(
            artifact_type=artifact_type,
            schema_version=schema_version,
            content_hash=content_hash,
            created_by_commit=created_by_commit,
            payload_ref=_payload_ref_for(artifact_type, content_hash),
        )

    @classmethod
    def create(
        cls,
        *,
        artifact_type: str,
        schema_version: str,
        created_by_commit: str,
        payload: object,
    ) -> Self:
        return cls.from_payload(
            artifact_type=artifact_type,
            schema_version=schema_version,
            created_by_commit=created_by_commit,
            payload=payload,
        )

    @property
    def canonical_path(self) -> str:
        return self.payload_ref

    def verify_payload(self, payload: object) -> None:
        actual = sha256_ref(payload)
        expected_ref = _payload_ref_for(self.artifact_type, actual)
        if actual != self.content_hash or expected_ref != self.payload_ref:
            raise ArtifactHashMismatch("payload hash does not match envelope")

    def verify_payload_hash(self, payload: object) -> None:
        self.verify_payload(payload)
