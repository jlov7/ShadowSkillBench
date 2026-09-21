"""Write a hashed-only V2 32k Nemotron live-show identity receipt from captures."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from shadowskillbench.experiments.ollama_nemotron_context32k_profile import (
    NemotronContext32kRoleProfileError,
    canonical_nemotron_context32k_live_show_identity_receipt,
)


def _regular_input(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise NemotronContext32kRoleProfileError(
            f"HOLD_NEMOTRON_CONTEXT32K_IDENTITY_PRECHECK: {label}"
        )
    try:
        return path.read_bytes()
    except OSError as error:
        raise NemotronContext32kRoleProfileError(
            f"HOLD_NEMOTRON_CONTEXT32K_IDENTITY_PRECHECK: {label}"
        ) from error


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise NemotronContext32kRoleProfileError(
            "HOLD_NEMOTRON_CONTEXT32K_IDENTITY_PRECHECK: output path"
        )
    if path.exists():
        if _regular_input(path, "output") == payload:
            return
        raise NemotronContext32kRoleProfileError(
            "HOLD_NEMOTRON_CONTEXT32K_IDENTITY_PRECHECK: output exists"
        )
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except OSError as error:
        raise NemotronContext32kRoleProfileError(
            "HOLD_NEMOTRON_CONTEXT32K_IDENTITY_PRECHECK: write"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-json", type=Path, required=True)
    parser.add_argument("--ollama-version", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        payload = canonical_nemotron_context32k_live_show_identity_receipt(
            show_response=_regular_input(arguments.show_json, "show capture"),
            ollama_version_text=_regular_input(arguments.ollama_version, "version capture").decode(
                "utf-8", errors="strict"
            ),
        )
        _write_once(arguments.output, payload)
    except (NemotronContext32kRoleProfileError, UnicodeDecodeError) as error:
        print(error)
        return 1
    print(f"PASS_NEMOTRON_CONTEXT32K_IDENTITY_PRECHECK: output={arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
