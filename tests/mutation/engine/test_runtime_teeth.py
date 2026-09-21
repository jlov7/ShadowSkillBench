from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ENGINE_SOURCE = Path(__file__).parents[3] / "src" / "shadowskillbench" / "engine"
RUNTIME_SOURCE = ENGINE_SOURCE / "runtime.py"
MODELS_SOURCE = ENGINE_SOURCE / "models.py"


def _run_mutant(
    tmp_path: Path, mutated_source: str, invariant: str
) -> subprocess.CompletedProcess[str]:
    mutant = tmp_path / "runtime_mutant.py"
    invariant_script = tmp_path / "runtime_invariant.py"
    mutant.write_text(mutated_source)
    invariant_script.write_text(invariant)
    return subprocess.run(
        [sys.executable, str(invariant_script), str(mutant)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.mutation
def test_mutation_kills_stale_before_precondition_bypass(tmp_path: Path) -> None:
    source = RUNTIME_SOURCE.read_text()
    needle = "if not _json_equal(parent_object[token], cast(JsonValue, patch.before)):"
    assert needle in source
    mutant = source.replace(needle, "if False:", 1)
    invariant = """\\
import importlib.util
import sys

from shadowskillbench.engine.models import StatePatch

spec = importlib.util.spec_from_file_location("mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
patch = StatePatch(
    op="replace", path="/value", before_present=True, before=True,
    after_present=True, after=2,
)
try:
    module.apply_state_data_patches({"value": 1}, (patch,))
except module.PatchApplicationError:
    pass
else:
    raise AssertionError("stale before value was accepted")
"""

    result = _run_mutant(tmp_path, mutant, invariant)

    assert result.returncode == 1


@pytest.mark.mutation
def test_mutation_kills_type_sensitive_json_equality_bypass(tmp_path: Path) -> None:
    source = RUNTIME_SOURCE.read_text()
    needle = "return canonical_json_bytes(left) == canonical_json_bytes(right)"
    assert needle in source
    mutant = source.replace(needle, "return left == right", 1)
    invariant = """\\
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
delta = module.diff_state_data({"value": True}, {"value": 1})
assert len(delta) == 1
assert delta[0].op == "replace"
"""

    result = _run_mutant(tmp_path, mutant, invariant)

    assert result.returncode == 1


@pytest.mark.mutation
def test_mutation_kills_pointer_escaping_bypass(tmp_path: Path) -> None:
    source = RUNTIME_SOURCE.read_text()
    needle = 'return token.replace("~", "~0").replace("/", "~1")'
    assert needle in source
    mutant = source.replace(needle, "return token", 1)
    invariant = """\\
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
delta = module.diff_state_data({"~/": 1}, {"~/": 2})
assert delta[0].path == "/~0~1"
"""

    result = _run_mutant(tmp_path, mutant, invariant)

    assert result.returncode == 1


@pytest.mark.mutation
def test_mutation_kills_callback_input_clone_bypass(tmp_path: Path) -> None:
    source = RUNTIME_SOURCE.read_text()
    needle = "before_state = _validated_world_state(state)"
    assert needle in source
    mutant = source.replace(needle, "before_state = state", 1).replace(
        "callback_state = _validated_world_state(before_state)", "callback_state = before_state", 1
    )
    invariant = """\\
import importlib.util
import sys

from shadowskillbench.engine.models import ActionCall, EventCursor, TransitionProposal, WorldState

spec = importlib.util.spec_from_file_location("mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
state = WorldState(
    schema_version="1.0", world_id="world", domain="domain", seed=1,
    data={"items": [1]},
)
call = ActionCall(action_id="action", tool_name="tool", arguments={})
cursor = EventCursor(episode_id="episode", next_index=0)
class Adapter:
    def apply(self, state, call):
        state.data["items"].append(2)
        return TransitionProposal(local_status="success", next_state=state, observation={})
try:
    module.execute_action(state, call, Adapter(), cursor=cursor)
except module.EnvironmentInvariantFailure:
    pass
assert state.data == {"items": [1]}
"""

    result = _run_mutant(tmp_path, mutant, invariant)

    assert result.returncode == 1


@pytest.mark.mutation
def test_mutation_kills_cursor_parent_or_action_binding_bypass(tmp_path: Path) -> None:
    source = MODELS_SOURCE.read_text()
    parent_needle = "parent_event_hash = cursor.parent_event_hash"
    action_needle = "action_hash = _hash_action_trusted(action)"
    assert parent_needle in source
    assert action_needle in source
    mutant = source.replace(parent_needle, "parent_event_hash = None", 1).replace(
        action_needle, 'action_hash = "sha256:" + "0" * 64', 1
    )
    invariant = """\\
import importlib.util
import sys

from shadowskillbench.engine.models import (
    ActionCall, ActionResult, EventCursor, WorldState, hash_action,
)

spec = importlib.util.spec_from_file_location("mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
state = WorldState(schema_version="1.0", world_id="world", domain="domain", seed=1, data={})
action = ActionCall(action_id="action", tool_name="tool", arguments={})
result = ActionResult(local_status="success", observation={})
cursor = EventCursor(
    episode_id="episode", next_index=1, parent_event_id="event_episode_000000",
    parent_event_hash="sha256:" + "1" * 64,
)
event = module.make_state_event_from_cursor(
    cursor=cursor, action=action, result=result, before_state=state,
    after_state=state, delta=(),
)
assert event.parent_event_hash == cursor.parent_event_hash
assert event.action_hash == hash_action(action)
"""

    result = _run_mutant(tmp_path, mutant, invariant)

    assert result.returncode == 1
