from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def load_safe_clean() -> object:
    script = Path(__file__).parents[1] / "scripts" / "safe_clean.py"
    spec = importlib.util.spec_from_file_location("safe_clean", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "protocol").mkdir()
    (tmp_path / "data" / "confirmatory").mkdir(parents=True)
    (tmp_path / ".venv").mkdir()
    return tmp_path


def test_safe_clean_removes_only_explicit_disposable_path(project_root: Path) -> None:
    safe_clean = load_safe_clean()
    disposable = project_root / ".pytest_cache"
    disposable.mkdir()
    marker = disposable / "bootstrap-contract.txt"
    marker.write_text("disposable")

    safe_clean.clean([".pytest_cache"], root=project_root)

    assert not disposable.exists()


def test_safe_clean_removes_normal_build_directory(project_root: Path) -> None:
    safe_clean = load_safe_clean()
    build = project_root / "build"
    build.mkdir()
    (build / "bootstrap-output.txt").write_text("disposable")

    safe_clean.clean(["build"], root=project_root)

    assert not build.exists()


def test_safe_clean_refuses_protected_and_out_of_root_paths(
    project_root: Path,
) -> None:
    safe_clean = load_safe_clean()
    for target in ("protocol", "data/confirmatory", ".venv", "../outside"):
        with pytest.raises(safe_clean.CleanRefusal) as error:
            safe_clean.clean([target], root=project_root)
        assert "refusing" in str(error.value).lower()


def test_safe_clean_refuses_a_symlink(project_root: Path, tmp_path: Path) -> None:
    safe_clean = load_safe_clean()
    link = project_root / "build"
    link.symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(safe_clean.CleanRefusal) as error:
        safe_clean.clean(["build"], root=project_root)

    assert "symlink" in str(error.value).lower()
    assert link.is_symlink()


@pytest.mark.mutation
def test_mutation_marker_kills_protocol_allowlist_mutant(
    project_root: Path, tmp_path: Path
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "safe_clean.py"
    mutant = tmp_path / "safe_clean_mutant.py"
    mutant.write_text(
        script.read_text().replace(
            '        "htmlcov",',
            '        "htmlcov",\n        "protocol",',
        )
    )
    invariant = tmp_path / "assert_clean_refusal.py"
    invariant.write_text(
        "\n".join(
            (
                "import importlib.util",
                "import sys",
                "from pathlib import Path",
                "spec = importlib.util.spec_from_file_location('mutant', sys.argv[1])",
                "module = importlib.util.module_from_spec(spec)",
                "assert spec.loader is not None",
                "spec.loader.exec_module(module)",
                "try:",
                "    module.clean(['protocol'], root=Path(sys.argv[2]))",
                "except module.CleanRefusal:",
                "    raise SystemExit(0)",
                "raise AssertionError('protocol allowlist mutant escaped CleanRefusal')",
            )
        )
    )

    result = subprocess.run(
        [sys.executable, str(invariant), str(mutant), str(project_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "escaped cleanrefusal" in result.stderr.lower()
