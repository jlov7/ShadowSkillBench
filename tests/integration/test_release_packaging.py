from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[2]
SDIST_ROOT_FILES = {
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "SECURITY.md",
    "config/executor.yaml",
    "prompts/executor_system.md",
    "pyproject.toml",
}
REQUIRED_SDIST_FILES = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
    "src/shadowskillbench/__init__.py",
    "src/shadowskillbench/cli.py",
    "src/shadowskillbench/examples/offline_access.py",
    "src/shadowskillbench/reporting/templates/report.html.j2",
    "src/shadowskillbench/experiments/ollama_gpt_oss_20b_harmony.Modelfile",
    "protocol/development_executor_grounding_screen_v1.json",
}
REQUIRED_WHEEL_FILES = {
    "shadowskillbench/__init__.py",
    "shadowskillbench/cli.py",
    "shadowskillbench/examples/offline_access.py",
    "shadowskillbench/reporting/templates/report.html.j2",
    "shadowskillbench/experiments/ollama_gpt_oss_20b_harmony.Modelfile",
    "shadowskillbench/_protocol/development_executor_grounding_screen_v1.json",
    "shadowskillbench/_runtime_sources/config/executor.yaml",
    "shadowskillbench/_runtime_sources/prompts/executor_system.md",
}
PRIVATE_PACKAGE_PARTS = {
    ".git",
    "artifacts",
    "evidence",
    "private-review",
    "tests",
}


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    dist_dir = tmp_path_factory.mktemp("release-dist")
    subprocess.run(
        ["uv", "build", "--no-build-isolation", "--out-dir", str(dist_dir)],
        check=True,
        cwd=ROOT,
    )
    return (
        next(dist_dir.glob("shadowskillbench-*.tar.gz")),
        next(dist_dir.glob("shadowskillbench-*.whl")),
    )


def _sdist_paths(sdist: Path) -> set[str]:
    with tarfile.open(sdist) as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    root = PurePosixPath(members[0]).parts[0]
    return {str(PurePosixPath(member).relative_to(root)) for member in members}


def _assert_no_cache_files(paths: set[str]) -> None:
    assert not any("__pycache__" in PurePosixPath(path).parts for path in paths)
    assert not any(path.endswith((".pyc", ".pyo", ".pyd")) for path in paths)


def _assert_no_private_material(paths: set[str]) -> None:
    for path in paths:
        member = PurePosixPath(path)
        assert not member.is_absolute()
        assert ".." not in member.parts
        assert not PRIVATE_PACKAGE_PARTS.intersection(member.parts)
        assert not any(part == ".env" or part.startswith(".env.") for part in member.parts)


def test_sdist_is_an_explicit_public_source_allowlist(
    built_distributions: tuple[Path, Path],
) -> None:
    sdist, _ = built_distributions
    paths = _sdist_paths(sdist)

    assert REQUIRED_SDIST_FILES <= paths
    assert all(
        path in SDIST_ROOT_FILES or path.startswith(("src/shadowskillbench/", "protocol/"))
        for path in paths
    )
    assert not any(path.startswith(("artifacts/", "tests/")) for path in paths)
    _assert_no_cache_files(paths)
    _assert_no_private_material(paths)


def test_wheel_contains_only_runtime_package_files_and_a_working_cli(
    built_distributions: tuple[Path, Path], tmp_path: Path
) -> None:
    _, wheel = built_distributions
    with zipfile.ZipFile(wheel) as archive:
        paths = set(archive.namelist())

    assert REQUIRED_WHEEL_FILES <= paths
    dist_info_paths = {path for path in paths if ".dist-info/" in path}
    assert all(path.startswith("shadowskillbench-") for path in dist_info_paths)
    assert {path.rsplit("/", 1)[-1] for path in dist_info_paths} >= {
        "METADATA",
        "WHEEL",
        "entry_points.txt",
        "RECORD",
    }
    assert all(path.startswith("shadowskillbench/") or path in dist_info_paths for path in paths)
    _assert_no_cache_files(paths)
    _assert_no_private_material(paths)

    install_root = tmp_path / "installed-wheel"
    subprocess.run(
        ["uv", "pip", "install", "--target", str(install_root), "--no-deps", str(wheel)],
        check=True,
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.path.insert(0, sys.argv.pop(1)); "
                "from shadowskillbench.cli import app; "
                "app()"
            ),
            str(install_root),
            "--help",
        ],
        check=True,
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert "ShadowSkillBench command-line interface." in result.stdout

    example = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys; "
                "sys.path.insert(0, sys.argv.pop(1)); "
                "runpy.run_module('shadowskillbench.examples.offline_access', "
                "run_name='__main__')"
            ),
            str(install_root),
        ],
        check=True,
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert "operational access granted: True" in example.stdout
    assert "verifier: FAIL UNAUTHORIZED_ACCESS_GRANTED" in example.stdout
    assert "policy rule: REQUIRE_APPROVAL; gate: BLOCK" in example.stdout
    assert "verifier: PASS ACCESS_APPROVAL_REQUESTED" in example.stdout
