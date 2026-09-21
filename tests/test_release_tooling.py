from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_public_surface_covers_the_complete_repository_candidate() -> None:
    path = ROOT / "scripts" / "public_surface_scan.py"
    spec = importlib.util.spec_from_file_location("public_surface_scan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    is_distribution_path = module.is_distribution_path

    assert is_distribution_path("README.md")
    assert is_distribution_path("src/shadowskillbench/examples/offline_access.py")
    assert is_distribution_path("docs/README.md")
    assert is_distribution_path("tests/test_release_tooling.py")
    assert is_distribution_path("REVIEW_START_HERE.md")
    assert not is_distribution_path("workbench/node_modules/pkg/index.js")


def test_generated_sbom_uses_the_package_version(tmp_path: Path) -> None:
    output = tmp_path / "shadowskillbench.cdx.json"

    result = subprocess.run(
        [sys.executable, "scripts/generate_sbom.py", "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert document["metadata"]["component"]["version"] == project["version"]

    document["metadata"]["component"]["version"] = "0.0.0-stale"
    output.write_text(json.dumps(document), encoding="utf-8")
    stale = subprocess.run(
        [sys.executable, "scripts/generate_sbom.py", "--check", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert stale.returncode != 0
    assert "does not match the current project and lockfiles" in stale.stderr
