from __future__ import annotations

from shadowskillbench.protocol.runtime_prompt_manifest import (
    RUNTIME_PROMPT_MANIFEST_PATH,
    runtime_prompt_manifest_bytes,
    runtime_prompt_manifest_matches,
)
from shadowskillbench.protocol.schema_artifacts import (
    RUNTIME_SKILL_IR_SCHEMA_PATH,
    compiler_skill_ir_schema_bytes,
    runtime_skill_ir_schema_matches,
)


def test_frozen_runtime_skill_ir_schema_is_the_current_canonical_bytes() -> None:
    assert runtime_skill_ir_schema_matches(RUNTIME_SKILL_IR_SCHEMA_PATH)
    assert RUNTIME_SKILL_IR_SCHEMA_PATH.read_bytes() == compiler_skill_ir_schema_bytes()


def test_frozen_runtime_prompt_manifest_is_the_current_canonical_bytes() -> None:
    assert runtime_prompt_manifest_matches(RUNTIME_PROMPT_MANIFEST_PATH)
    assert RUNTIME_PROMPT_MANIFEST_PATH.read_bytes() == runtime_prompt_manifest_bytes()
