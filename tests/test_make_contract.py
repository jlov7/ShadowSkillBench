from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_makefile_defines_stable_credential_free_targets() -> None:
    contents = (ROOT / "Makefile").read_text()

    for target in (
        "setup",
        "lint",
        "typecheck",
        "test",
        "property-test",
        "mutation-test",
        "verify",
        "clean",
    ):
        assert f"{target}:" in contents

    assert "verify: lint typecheck test property-test mutation-test" in contents
    assert "sync --all-groups --frozen --no-editable" in contents
    assert "uv run --no-sync" in contents
    assert "ruff format --check ." in contents
    assert "$(UV_RUN) pytest -m property -q" in contents
    assert "$(UV_RUN) pytest -m mutation -q" in contents
    assert "-k refusal_property" not in contents
    assert "pytest tests/test_cli_contract.py" not in contents
    assert "pytest tests/test_safe_clean.py" not in contents
    assert "OPENAI_API_KEY" not in contents
    assert "ANTHROPIC_API_KEY" not in contents


def test_verify_workflow_uses_exact_immutable_action_pins() -> None:
    contents = (ROOT / ".github" / "workflows" / "verify.yml").read_text()

    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1" in contents
    assert "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d # v10.0.1" in contents
    assert "uv sync --all-groups --frozen --no-editable" in contents


def test_pyright_checks_all_scripts() -> None:
    contents = (ROOT / "pyproject.toml").read_text()

    assert 'include = ["src", "scripts"]' in contents
    assert (
        'extend-exclude = ["artifacts", "docs", "evidence", '
        '"implementation/IMPLEMENTATION_PLAN.md"]' in contents
    )
    assert "property:" in contents
    assert "mutation:" in contents
