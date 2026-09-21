from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HASHING_SOURCE = Path(__file__).parents[3] / "src" / "shadowskillbench" / "core" / "hashing.py"


def run_mutant_invariant(
    tmp_path: Path, mutated_source: str, invariant: str
) -> subprocess.CompletedProcess[str]:
    mutant = tmp_path / "hashing_mutant.py"
    invariant_script = tmp_path / "hashing_invariant.py"
    mutant.write_text(mutated_source)
    invariant_script.write_text(invariant)
    return subprocess.run(
        [sys.executable, str(invariant_script), str(mutant)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.mutation
def test_mutation_marker_kills_unsorted_mapping_hash_mutant(tmp_path: Path) -> None:
    source = HASHING_SOURCE.read_text()
    assert "sort_keys=True" in source
    mutant = source.replace("sort_keys=True", "sort_keys=False", 1)
    invariant = """\
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
assert module.sha256_ref({"a": 1, "b": 2}) == module.sha256_ref({"b": 2, "a": 1})
"""

    result = run_mutant_invariant(tmp_path, mutant, invariant)

    assert result.returncode == 1
    assert "AssertionError" in result.stderr


@pytest.mark.mutation
def test_mutation_marker_kills_negative_zero_normalization_mutant(tmp_path: Path) -> None:
    source = HASHING_SOURCE.read_text()
    assert "if value == 0.0:" in source
    mutant = source.replace("if value == 0.0:", "if False:", 1)
    invariant = """\
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
assert module.canonical_json_bytes(-0.0) == module.canonical_json_bytes(0.0)
"""

    result = run_mutant_invariant(tmp_path, mutant, invariant)

    assert result.returncode == 1
    assert "AssertionError" in result.stderr
