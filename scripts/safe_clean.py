from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISPOSABLE_PATHS = frozenset(
    {
        ".coverage",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".pyright",
        ".ruff_cache",
        "build",
        "dist",
        "htmlcov",
    }
)


class CleanRefusal(Exception):
    """A requested clean target violates the explicit disposable-path contract."""


def refuse(message: str) -> None:
    raise CleanRefusal(f"refusing clean: {message}")


def validated_target(value: str, root: Path = ROOT) -> Path:
    root = root.resolve()
    requested = Path(value)
    if requested.is_absolute() or ".." in requested.parts:
        refuse(f"out-of-root path {value!r}")

    normalized = requested.as_posix()
    if normalized not in DISPOSABLE_PATHS:
        refuse(f"protected or unlisted path {value!r}")

    target = root / requested
    if target.is_symlink():
        refuse(f"symlink path {value!r}")
    if root not in target.resolve().parents and target.resolve() != root:
        refuse(f"out-of-root path {value!r}")
    return target


def clean(values: list[str], root: Path = ROOT) -> None:
    targets = values or sorted(DISPOSABLE_PATHS)
    for value in targets:
        target = validated_target(value, root=root)
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


def main() -> None:
    try:
        clean(sys.argv[1:])
    except CleanRefusal as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
