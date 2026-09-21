from __future__ import annotations

import subprocess
import sys
import tracemalloc
from pathlib import Path
from typing import Any, cast

import pytest

from shadowskillbench.domains.access.fixtures import build_access_fixture
from shadowskillbench.domains.access.models import parse_access_world
from shadowskillbench.domains.access.verifier import AccessVerifierInputError, verify_access_case

SOURCE = (
    Path(__file__).parents[3] / "src" / "shadowskillbench" / "domains" / "access" / "verifier.py"
)
UNIT_HELPERS = Path(__file__).parents[2] / "unit" / "domains" / "access" / "test_verifier.py"


def _run_mutant(
    tmp_path: Path, needle: str, replacement: str, assertion: list[str]
) -> subprocess.CompletedProcess[str]:
    source = SOURCE.read_text(encoding="utf-8")
    assert needle in source
    mutant = tmp_path / "verifier_mutant.py"
    script = tmp_path / "check.py"
    mutant.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    script.write_text(
        "\n".join(
            [
                "import importlib.util, runpy, sys, types",
                "helpers=types.SimpleNamespace(**runpy.run_path(sys.argv[2]))",
                "spec=importlib.util.spec_from_file_location('mutant', sys.argv[1])",
                "m=importlib.util.module_from_spec(spec)",
                "assert spec.loader is not None",
                "spec.loader.exec_module(m)",
                *assertion,
            ]
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(script), str(mutant), str(UNIT_HELPERS)],
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_killed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 1
    assert "AssertionError" in result.stderr


@pytest.mark.mutation
def test_mutation_kills_finish_gate_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "state.finished\n        and",
        "True\n        and",
        [
            "f=helpers._fixture()",
            "state=helpers.parse_access_world(f.initial_world)",
            "state=state.model_copy(update={'finish_summary': 'completed'})",
            "v=m.verify_access_case(state, f.case, authority_decision=helpers._decision(f))",
            "assert v.reason_code == 'TASK_NOT_FINISHED'",
        ],
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_original_baseline_validation_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "_validate_original_baseline(baseline)",
        "None",
        [
            "f=helpers._fixture()",
            "state=helpers._finish_and_grant(f)",
            "baseline=f.case.required_preserved_entitlement_ids[0]",
            "without=tuple(x for x in state.entitlements if x.entitlement_id != baseline)",
            "bad=state.model_copy(update={'entitlements': without,",
            "    'baseline_entitlement_ids': (baseline, baseline)})",
            "try:",
            "    m.verify_access_case(bad, f.case, authority_decision=helpers._decision(f))",
            "except m.AccessVerifierInputError:",
            "    pass",
            "else:",
            "    raise AssertionError('malformed original baseline was accepted')",
        ],
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_target_grant_requirement_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if not target_grant:",
        "if False:",
        [
            "f=helpers._fixture()",
            "state=helpers._finish_refusal(f)",
            "v=m.verify_access_case(state, f.case, authority_decision=helpers._decision(f))",
            "assert v.reason_code == 'ACCESS_STATE_MISMATCH'",
        ],
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_required_evidence_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if evidence_reason is not None and not _valid_evidence(",
        "if False and not _valid_evidence(",
        [
            "f=helpers._fixture('expired_approval')",
            "state=helpers._valid_approval_state(f)",
            "request=state.access_requests[0].model_copy(update={'approval_refs': ()})",
            "bad=state.model_copy(update={'access_requests': (request,)})",
            "v=m.verify_access_case(bad, f.case, authority_decision=helpers._approval_decision(f))",
            "assert v.reason_code == 'ACCESS_EVIDENCE_INVALID'",
        ],
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_preallocation_visit_budget_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "scheduled_visits = _reserve_children(scheduled_visits, len(sequence), depth)",
        "_ = sequence",
        [
            "import tracemalloc",
            "f=helpers._fixture()",
            "state=helpers._finish_and_grant(f)",
            "payload=tuple('entitlement_nested' for _ in range(100_001))",
            "bad=state.model_copy(update={'entitlements': payload})",
            "tracemalloc.start()",
            "try:",
            "    m.verify_access_case(bad, f.case, authority_decision=helpers._decision(f))",
            "except m.AccessVerifierInputError:",
            "    pass",
            "_, peak=tracemalloc.get_traced_memory()",
            "tracemalloc.stop()",
            "assert peak < 1_000_000",
        ],
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_oversized_hostile_model_key_tree_is_rejected_before_key_traversal() -> None:
    class HostileKey:
        calls = 0

        def __hash__(self) -> int:
            type(self).calls += 1
            return 0

        def __eq__(self, other: object) -> bool:
            type(self).calls += 1
            return self is other

    fixture = build_access_fixture("role_derived", seed=109)
    state = parse_access_world(fixture.initial_world)
    values = object.__getattribute__(state, "__dict__")
    hostile_key = HostileKey()
    values[hostile_key] = None
    HostileKey.calls = 0
    values.update({f"hostile_extra_{index}": None for index in range(100_001)})

    tracemalloc.start()
    try:
        with pytest.raises(AccessVerifierInputError):
            verify_access_case(
                state,
                fixture.case,
                authority_decision=cast(Any, object()),
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000
    assert HostileKey.calls == 0
