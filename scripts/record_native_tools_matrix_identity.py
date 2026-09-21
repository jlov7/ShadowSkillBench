"""Write one hash-only identity receipt for a matrix conformance candidate."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shadowskillbench.experiments.native_tools_conformance_matrix import (
    canonical_identity,
    load_candidate,
)  # noqa: E402


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("output")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise ValueError("write-once output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--show-json", type=Path, required=True)
    parser.add_argument("--ollama-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = canonical_identity(
            load_candidate(args.candidate),
            show_response=args.show_json.read_bytes(),
            ollama_version_text=args.ollama_version_file.read_text(),
        )
        _write_once(args.output, payload)
    except (OSError, ValueError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
