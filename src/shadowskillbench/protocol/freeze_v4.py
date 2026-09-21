"""Closed V4 protocol-freeze output; it never discovers a latest manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from shadowskillbench.protocol.scientific_freeze_v4 import SCIENTIFIC_FREEZE_V4_INPUTS

V4_MANIFEST_PATH = "protocol/freeze_manifest.v4.json"
V4_DETACHED_PATH = "protocol/freeze_manifest.v4.sha256"


@dataclass(frozen=True, slots=True)
class FreezeV4Artifacts:
    manifest_path: Path
    detached_path: Path
    manifest_hash: str


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def build_freeze_v4_manifest(repository_root: Path) -> bytes:
    root = repository_root.resolve(strict=True)
    inputs: list[dict[str, str]] = []
    for item in SCIENTIFIC_FREEZE_V4_INPUTS:
        source = root / item.path
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"V4 freeze input is unavailable: {item.path}")
        inputs.append(
            {
                "path": item.path,
                "role": item.role,
                "sha256": "sha256:" + sha256(source.read_bytes()).hexdigest(),
            }
        )
    return _canonical(
        {
            "anchor_status": "HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY",
            "inputs": inputs,
            "profile": "SSB-SCIENTIFIC-PREFREEZE-4",
            "schema_version": "4.0",
        }
    )


def write_freeze_v4_artifacts(repository_root: Path) -> FreezeV4Artifacts:
    root = repository_root.resolve(strict=True)
    manifest = build_freeze_v4_manifest(root)
    manifest_path = root / V4_MANIFEST_PATH
    detached_path = root / V4_DETACHED_PATH
    if manifest_path.exists() or detached_path.exists():
        raise FileExistsError("V4 freeze output already exists")
    manifest_path.write_bytes(manifest)
    digest = sha256(manifest).hexdigest()
    detached_path.write_bytes(f"{digest}  freeze_manifest.v4.json\n".encode())
    return FreezeV4Artifacts(manifest_path, detached_path, f"sha256:{digest}")


__all__ = [
    "FreezeV4Artifacts",
    "V4_DETACHED_PATH",
    "V4_MANIFEST_PATH",
    "build_freeze_v4_manifest",
    "write_freeze_v4_artifacts",
]
