"""Verify the immutable public pack files recorded in PACK_MANIFEST.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACK_MANIFEST.json"


def main() -> int:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = document.get("files")
    if not isinstance(entries, list) or len(entries) != 30:
        raise ValueError("PACK_MANIFEST.json must contain exactly 30 files")
    failures: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append("manifest entry is not an object")
            continue
        relative = entry.get("path")
        expected_bytes = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            failures.append(f"invalid manifest path: {relative!r}")
            continue
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        payload = path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_bytes or actual_hash != expected_hash:
            failures.append(f"mismatch: {relative} bytes={len(payload)} sha256={actual_hash}")
    if failures:
        print("PACK_MANIFEST_FAILED")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print(f"PACK_MANIFEST_OK files={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
