from __future__ import annotations

# ruff: noqa: E501
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[3] / "src" / "shadowskillbench" / "skills" / "projection.py"


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("needle", "replacement", "assertion"),
    [
        ("if text_key in _FORBIDDEN_KEYS:", "if False:", "assert rejected({'policy': 'x'})"),
        (
            "if text in _FORBIDDEN_VALUES:",
            "if False:",
            "assert rejected_retained_scalar('compliant')",
        ),
        (
            '"worker_role": trace.worker_role,',
            '"worker_policy_id": trace.worker_policy_id,',
            "assert compiler_view_is_valid_and_ordered()",
        ),
        (
            "for trace in validated.traces",
            "for trace in reversed(validated.traces)",
            "assert compiler_view_is_valid_and_ordered()",
        ),
        (
            "for event in trace.events",
            "for event in reversed(trace.events)",
            "assert compiler_view_is_valid_and_ordered()",
        ),
        (
            'if len({cast(str, trace["trace_id"]) for trace in raw}) != 12:',
            "if False:",
            "assert rejects_duplicate_trace_ids()",
        ),
        (
            '"terminal_state_hash": trace["terminal_state_hash"],',
            '"terminal_state_hash": "sha256:" + "0" * 64,',
            "assert module.hash_compiler_input(raw()) != module.hash_compiler_input(changed_terminal())",
        ),
        (
            "return sha256_ref(compiler_input_projection(value))",
            "return sha256_ref({})",
            "assert module.hash_compiler_input(raw()) != module.hash_compiler_input(changed())",
        ),
        (
            "or any(type(name) is not str for name in field_set)",
            "or False",
            "assert rejects_forged_fields()",
        ),
        ("import math", "import math, os", "assert imports_are_exact()"),
        (
            "from shadowskillbench.traces.bundles import DemonstrationBundle, SourceBundleManifest",
            "from shadowskillbench.traces.bundles import DemonstrationBundle, SourceBundleManifest, generate_bundle",
            "assert imports_are_exact()",
        ),
        (
            "validated = _source_preflight(bundle)",
            "bundle.model_dump()\n    validated = _source_preflight(bundle)",
            "assert source_has_no_full_serialization_call()",
        ),
    ],
)
def test_projection_mutants_are_killed(
    tmp_path: Path, needle: str, replacement: str, assertion: str
) -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert needle in source
    mutant = tmp_path / "projection_mutant.py"
    check = tmp_path / "check.py"
    mutant.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    check.write_text(_CHECK.replace("ASSERTION", assertion), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(check), str(mutant)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1, result.stderr


_CHECK = """\
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from shadowskillbench.traces.bundles import generate_bundle
spec = importlib.util.spec_from_file_location("shadowskillbench.skills.projection_mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
def raw():
    traces = []
    for number in range(12):
        trace_id = f"trace_unit_{number}"
        event = lambda i, k, p: {"event_id": f"event_{trace_id}_{i:06d}", "index": i, "kind": k, "payload": p}
        traces.append({"trace_id": trace_id, "domain": "access_provisioning", "task_template_id": "task", "worker_role": "worker", "world_hash": "sha256:" + "a" * 64, "events": [event(0,"observation",{"visible": 0}), event(1,"action",{"visible": 1}), event(2,"tool_result",{"visible": 2}), event(3,"state_delta",{"visible": 3})], "terminal_state_hash": "sha256:" + "b" * 64, "local_task_outcome": "completed"})
    return {"projection_profile": "SSB-COMPILER-VIEW1", "domain": "access_provisioning", "traces": traces}
def rejected(payload):
    value = raw(); value["traces"][0]["events"][0]["payload"] = payload
    try: module.CompilerInput.model_validate(value)
    except ValueError: return True
    return False
def rejected_retained_scalar(value):
    candidate = raw(); candidate["traces"][0]["worker_role"] = value
    try: module.CompilerInput.model_validate(candidate)
    except ValueError: return True
    return False
def compiler_view_is_valid_and_ordered():
    generated = generate_bundle("access_provisioning", Decimal("0.5"), 12, 902)
    projected = module.compiler_view(generated.bundle)
    assert tuple(trace.trace_id for trace in projected.traces) == tuple(trace.trace_id for trace in generated.bundle.traces)
    assert tuple(event.index for event in projected.traces[0].events) == tuple(event.index for event in generated.bundle.traces[0].events)
    return True
def rejects_forged_fields():
    class S(str): pass
    value = module.CompilerInput.model_validate(raw())
    fields = object.__getattribute__(value, "__pydantic_fields_set__")
    fields.clear(); fields.update(S(name) for name in ("projection_profile", "domain", "traces"))
    try: module.compiler_input_projection(value)
    except ValueError: return True
    return False
def rejects_duplicate_trace_ids():
    import copy
    value = raw(); value["traces"][-1] = copy.deepcopy(value["traces"][0])
    try: module.CompilerInput.model_validate(value)
    except ValueError: return True
    return False
def imports_are_exact():
    import ast
    tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
    direct = {name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names}
    imported = {name.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for name in node.names}
    return direct == {"math", "re"} and "generate_bundle" not in imported and "action_trace_projection" not in imported
def source_has_no_full_serialization_call():
    import ast
    tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
    return not {call.func.attr for call in ast.walk(tree) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in {"model_dump", "model_dump_json", "dict", "json"}}
def changed():
    value = raw(); value["traces"][0]["worker_role"] = "other"; return value
def changed_terminal():
    value = raw(); value["traces"][0]["terminal_state_hash"] = "sha256:" + "c" * 64; return value
ASSERTION
"""
