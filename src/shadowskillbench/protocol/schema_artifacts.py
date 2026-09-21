"""Canonical generated schema artifacts that are admitted to the protocol freeze."""

from __future__ import annotations

from pathlib import Path

from shadowskillbench.core.hashing import canonical_json_bytes
from shadowskillbench.skills.models import SkillIR

RUNTIME_SKILL_IR_SCHEMA_PATH = Path("protocol/commitments/compiler_skill_ir_schema.json")


def compiler_skill_ir_schema_bytes() -> bytes:
    """Return the canonical bytes for the runtime compiler output schema."""

    return canonical_json_bytes(SkillIR.model_json_schema())


def runtime_skill_ir_schema_matches(path: Path) -> bool:
    """Return whether ``path`` is the exact generated runtime schema artifact."""

    try:
        return (
            path.is_file()
            and not path.is_symlink()
            and path.read_bytes() == compiler_skill_ir_schema_bytes()
        )
    except OSError:
        return False


__all__ = [
    "RUNTIME_SKILL_IR_SCHEMA_PATH",
    "compiler_skill_ir_schema_bytes",
    "runtime_skill_ir_schema_matches",
]
