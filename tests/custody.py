from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-08-30T12:00:00Z",
            "GIT_COMMITTER_DATE": "2026-08-30T12:00:00Z",
        },
    ).stdout.strip()


def write_local_custody_receipt(root: Path, manifest: bytes) -> bytes:
    """Create a real local Git-custody receipt for an already frozen test tree."""

    receipt_path = root / "protocol" / "anchor_receipt.json"
    if (root / ".git").is_dir() and receipt_path.is_file():
        receipt = receipt_path.read_bytes()
        payload = json.loads(receipt)
        expected_hash = "sha256:" + sha256(manifest).hexdigest()
        if payload.get("freeze_manifest_hash") != expected_hash:
            raise AssertionError("existing test custody receipt binds a different manifest")
        return receipt

    _git(root, "init")
    _git(root, "config", "user.name", "ShadowSkillBench test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "freeze")
    freeze_commit = _git(root, "rev-parse", "HEAD")
    protocol_tag = "shadowskillbench-protocol-v1.0.0-local"
    _git(root, "tag", protocol_tag, freeze_commit)
    receipt = json.dumps(
        {
            "schema_version": "2.0",
            "custody_mode": "LOCAL_HASH_CUSTODY",
            "freeze_commit": freeze_commit,
            "protocol_tag": protocol_tag,
            "freeze_manifest_hash": "sha256:" + sha256(manifest).hexdigest(),
            "custody_locator": "urn:git:" + freeze_commit,
            "created_at": "2026-08-30T12:00:00Z",
            "verification_result": "VERIFIED_LOCAL",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    receipt_path.write_bytes(receipt)
    _git(root, "add", "protocol/anchor_receipt.json")
    _git(root, "commit", "-m", "record local custody receipt")
    return receipt
