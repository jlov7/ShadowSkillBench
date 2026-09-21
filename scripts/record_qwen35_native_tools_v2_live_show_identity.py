"""Write the hash-only Qwen 4.7 Flash identity receipt from an operator capture."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shadowskillbench.experiments.ollama_qwen35_native_tools_v2_profile import (  # noqa: E402
    Qwen35NativeToolsV2ProfileError,
    canonical_qwen35_native_tools_v2_identity,
)


def _regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise Qwen35NativeToolsV2ProfileError(f"{label} is unsafe")
    try:
        return path.read_bytes()
    except OSError as error:
        raise Qwen35NativeToolsV2ProfileError(f"{label} is unreadable") from error


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise Qwen35NativeToolsV2ProfileError("identity receipt output is unsafe")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise Qwen35NativeToolsV2ProfileError("identity receipt output already exists")
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
    parser.add_argument("--show-json", type=Path, required=True)
    parser.add_argument("--ollama-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        _write_once(
            arguments.output,
            canonical_qwen35_native_tools_v2_identity(
                show_response=_regular_bytes(arguments.show_json, "show capture"),
                ollama_version_text=_regular_bytes(
                    arguments.ollama_version_file, "Ollama version capture"
                ).decode("utf-8"),
            ),
        )
    except (Qwen35NativeToolsV2ProfileError, UnicodeDecodeError):
        print("HOLD_QWEN35_NATIVE_TOOLS_V2_IDENTITY: invalid capture")
        return 1
    print(f"PASS_QWEN35_NATIVE_TOOLS_V2_IDENTITY: output={arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
