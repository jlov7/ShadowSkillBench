"""Generate or check the frozen canonical confirmatory compiler wire schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.skills.confirmatory_wire import ConfirmatorySkillIRWire

TARGET = Path("protocol/commitments/confirmatory_compiler_wire.schema.json")


def _bytes() -> bytes:
    return canonical_json_bytes(ConfirmatorySkillIRWire.model_json_schema())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(_bytes())
        return 0
    if not TARGET.is_file() or TARGET.is_symlink() or TARGET.read_bytes() != _bytes():
        raise SystemExit(f"confirmatory compiler wire schema artifact is stale: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
