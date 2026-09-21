"""Generate or check the frozen canonical runtime SkillIR schema artifact."""

from __future__ import annotations

import argparse

from shadowskillbench.protocol.schema_artifacts import (
    RUNTIME_SKILL_IR_SCHEMA_PATH,
    compiler_skill_ir_schema_bytes,
    runtime_skill_ir_schema_matches,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    target = RUNTIME_SKILL_IR_SCHEMA_PATH
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(compiler_skill_ir_schema_bytes())
        return 0
    if not runtime_skill_ir_schema_matches(target):
        raise SystemExit(f"runtime SkillIR schema artifact is stale: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
