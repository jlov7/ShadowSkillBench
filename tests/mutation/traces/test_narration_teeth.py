from __future__ import annotations

# ruff: noqa: E501
import subprocess
import sys
from pathlib import Path

import pytest

NARRATION = Path(__file__).parents[3] / "src" / "shadowskillbench" / "traces" / "narration.py"
_TIMEOUT_SECONDS = 15


def _run_mutant(
    tmp_path: Path, needle: str, replacement: str, body: str
) -> subprocess.CompletedProcess[str]:
    source = NARRATION.read_text(encoding="utf-8")
    assert needle in source
    mutant = tmp_path / "narration_mutant.py"
    check = tmp_path / "check.py"
    mutant.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    check.write_text(_LOAD + _wrapped(body), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(check), str(mutant)],
        capture_output=True,
        text=True,
        check=False,
        timeout=_TIMEOUT_SECONDS,
    )


def _assert_killed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 1, result.stderr
    assert "AssertionError" in result.stderr, result.stderr


_LOAD = """\\
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("shadowskillbench.traces.narration_mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""

_SOURCE_TRACE = """\\
from shadowskillbench.traces.models import action_trace_projection, hash_action_trace, make_action_trace, make_trace_event
HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64
def source_trace():
    return make_action_trace(
        trace_id="trace_narration_mutant", schema_version="1.0", domain="access_provisioning",
        task_template_id="access_request", worker_policy_id="access_compliant_v1",
        worker_role="access_operator", world_hash=HASH, narration_mode="none",
        events=(
            make_trace_event(trace_id="trace_narration_mutant", index=0, kind="observation", payload={"request": "alice"}),
            make_trace_event(trace_id="trace_narration_mutant", index=1, kind="action", payload={"tool": "inspect"}),
            make_trace_event(trace_id="trace_narration_mutant", index=2, kind="tool_result", payload={"status": "ok"}),
            make_trace_event(trace_id="trace_narration_mutant", index=3, kind="state_delta", payload={"after": "reviewed"}),
        ), terminal_state_hash=OTHER_HASH, local_task_outcome="completed",
    )
"""


def _wrapped(body: str) -> str:
    indented = "\n".join(f"    {line}" if line else line for line in body.splitlines())
    return (
        "try:\n"
        f"{indented}\n"
        "except AssertionError:\n"
        "    raise\n"
        "except Exception as error:\n"
        "    raise AssertionError('mutant behavior rejected') from error\n"
    )


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("needle", "replacement", "body"),
    [
        (
            'mode: object = "none",',
            'mode: object = "neutral",',
            _SOURCE_TRACE
            + "source = source_trace()\n"
            + "assert action_trace_projection(module.narrate_trace(source)) == action_trace_projection(source)\n"
            + "assert hash_action_trace(module.narrate_trace(source)) == hash_action_trace(source)\n",
        ),
        (
            "index=0,",
            "index=1,",
            _SOURCE_TRACE
            + "result = module.narrate_trace(source_trace(), mode='neutral', text='cue')\n"
            + "assert result.events[0].kind == 'observation'\n"
            + "assert result.events[1].kind == 'narration'\n",
        ),
        (
            """make_trace_event(
            trace_id=source.trace_id,
            index=0,
            kind=source.events[0].kind,
            payload=source.events[0].payload,
        )""",
            "source.events[0]",
            _SOURCE_TRACE
            + "source = source_trace()\n"
            + "factory = module.make_trace_event\n"
            + "calls = []\n"
            + "def recording_factory(**kwargs):\n"
            + "    calls.append(kwargs)\n"
            + "    return factory(**kwargs)\n"
            + "module.make_trace_event = recording_factory\n"
            + "module.narrate_trace(source, mode='neutral', text='factory path')\n"
            + "assert [call['index'] for call in calls] == [0, 1, 2, 3, 4]\n",
        ),
        (
            "index=index,",
            "index=event.index,",
            _SOURCE_TRACE
            + "result = module.narrate_trace(source_trace(), mode='neutral', text='cue')\n"
            + "assert [event.index for event in result.events] == list(range(5))\n"
            + "assert [event.event_id for event in result.events] == [f'event_trace_narration_mutant_{index:06d}' for index in range(5)]\n",
        ),
        (
            'payload={"text": owned_text},',
            'payload={"text": "altered"},',
            _SOURCE_TRACE
            + "result = module.narrate_trace(source_trace(), mode='neutral', text='exact caller stimulus')\n"
            + "assert result.events[1].payload == {'text': 'exact caller stimulus'}\n",
        ),
        (
            "narration_mode=owned_mode,",
            'narration_mode="neutral",',
            _SOURCE_TRACE
            + "result = module.narrate_trace(source_trace(), mode='rationalized', text='cue')\n"
            + "assert result.narration_mode == 'rationalized'\n",
        ),
        (
            'return mode == "none" and all(kind != "narration" for kind in kinds)',
            "return True",
            _SOURCE_TRACE
            + "source = module.narrate_trace(source_trace(), mode='neutral', text='existing')\n"
            + "try:\n"
            + "    module.narrate_trace(source, mode='rationalized', text='stacked')\n"
            + "except ValueError:\n"
            + "    pass\n"
            + "else:\n"
            + "    raise AssertionError('existing narration was stacked')\n",
        ),
        (
            "or private is not None",
            "or False",
            _SOURCE_TRACE
            + "source = source_trace()\n"
            + "object.__setattr__(source, '__pydantic_private__', {'forged': True})\n"
            + "try:\n"
            + "    module.narrate_trace(source)\n"
            + "except ValueError:\n"
            + "    pass\n"
            + "else:\n"
            + "    raise AssertionError('private state forgery was accepted')\n",
        ),
    ],
)
def test_mutation_kills_narration_contract_bypasses(
    tmp_path: Path, needle: str, replacement: str, body: str
) -> None:
    _assert_killed(_run_mutant(tmp_path, needle, replacement, body))


@pytest.mark.mutation
def test_mutation_kills_forbidden_ambient_import(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "from typing import cast",
        "from typing import cast\nimport pathlib",
        "from pathlib import Path\nassert 'import pathlib' not in Path(sys.argv[1]).read_text()\n",
    )
    _assert_killed(result)
