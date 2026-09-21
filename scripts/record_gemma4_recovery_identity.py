"""Create one prospective Gemma recovery identity receipt from loopback captures."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shadowskillbench.experiments.ollama_gemma4_recovery_profile import (  # noqa: E402
    Gemma4RecoveryProfileError,
    canonical_recovery_identity,
)


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise Gemma4RecoveryProfileError("identity capture is unsafe")
    return path.read_bytes()


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise Gemma4RecoveryProfileError("identity output is unsafe")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise Gemma4RecoveryProfileError("identity output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("primary-26b", "fallback-12b"), required=True)
    parser.add_argument("--show-json", type=Path, required=True)
    parser.add_argument("--tags-json", type=Path, required=True)
    parser.add_argument("--version-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        _write_once(
            arguments.output,
            canonical_recovery_identity(
                candidate=arguments.candidate,
                show_response=_read(arguments.show_json),
                tags_response=_read(arguments.tags_json),
                version_response=_read(arguments.version_json),
            ),
        )
    except (Gemma4RecoveryProfileError, OSError):
        print("HOLD_GEMMA4_RECOVERY_IDENTITY: invalid capture")
        return 1
    print(
        f"PASS_GEMMA4_RECOVERY_IDENTITY: candidate={arguments.candidate} output={arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
