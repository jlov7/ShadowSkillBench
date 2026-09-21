from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BUNDLES = Path(__file__).parents[3] / "src" / "shadowskillbench" / "traces" / "bundles.py"
_TIMEOUT_SECONDS = 15


def _run_mutant(
    tmp_path: Path, needle: str, replacement: str, body: str
) -> subprocess.CompletedProcess[str]:
    source = BUNDLES.read_text()
    assert needle in source
    mutant = tmp_path / "bundles_mutant.py"
    check = tmp_path / "check.py"
    mutant.write_text(source.replace(needle, replacement, 1))
    check.write_text(_LOAD + _wrapped(body))
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
spec = importlib.util.spec_from_file_location("shadowskillbench.traces.bundles_mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
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


def _generation_invariant(assertion: str, domain: str = "access_provisioning") -> str:
    return (
        "from decimal import Decimal\n"
        f"value = module.generate_bundle('{domain}', Decimal('0.5'), 12, 0)\n"
        f"{assertion}\n"
    )


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("needle", "replacement", "body"),
    [
        (
            'Decimal("0.5"): ("0.5", 6),',
            'Decimal("0.5"): ("0.5", 5),',
            _generation_invariant(
                "assert sum(x.semantic_class == 'prohibited_workaround' for x in value.hidden_class_manifest.bindings) == 6"  # noqa: E501
            ),
        ),
        (
            '"contamination_ratio": _ratio_text(ratio),',
            '"contamination_ratio": "0.5",',
            "from decimal import Decimal\n"
            "left = module.generate_bundle('access_provisioning', Decimal('0'), 12, 0)\n"
            "right = module.generate_bundle('access_provisioning', Decimal('0.5'), 12, 0)\n"
            "assert left.bundle.source_manifest.bundle_id != right.bundle.source_manifest.bundle_id\n",  # noqa: E501
        ),
        (
            "slot * 0x9E3779B97F4A7C15",
            "slot * 1",
            _generation_invariant(
                "assert value.hidden_class_manifest.bindings[0].fixture_seed == -3604917875252592624"  # noqa: E501
            ),
        ),
        (
            "return unsigned if unsigned < 2**63 else unsigned - 2**64",
            "return unsigned",
            _generation_invariant(
                "assert value.hidden_class_manifest.bindings[0].fixture_seed < 0"
            ),
        ),
        (
            '"final_trace_order", self.domain, self.seed',
            '"class_slots", self.domain, self.seed',
            _generation_invariant(
                "assert value.hidden_class_manifest.class_slot_order != value.hidden_class_manifest.final_trace_order"  # noqa: E501
            ),
        ),
        (
            "class_order[:workaround_count]",
            "class_order[workaround_count:]",
            _generation_invariant(
                "assert sum(x.semantic_class == 'prohibited_workaround' for x in value.hidden_class_manifest.bindings) == 6"  # noqa: E501
            ),
        ),
        (
            "ordered_trace_hashes=tuple(hash_action_trace(trace) for trace in traces),",
            "ordered_trace_hashes=tuple(reversed(hash_action_trace(trace) for trace in traces)),",
            _generation_invariant("assert len(value.bundle.traces) == 12"),
        ),
        (
            "hidden_class_manifest_hash=hash_hidden_class_manifest(hidden),",
            "hidden_class_manifest_hash='sha256:' + '0' * 64,",
            _generation_invariant(
                "assert value.bundle.source_manifest.hidden_class_manifest_hash == module.hash_hidden_class_manifest(value.hidden_class_manifest)"  # noqa: E501
            ),
        ),
    ],
)
def test_mutation_kills_generation_contract_bypasses(
    tmp_path: Path, needle: str, replacement: str, body: str
) -> None:
    _assert_killed(_run_mutant(tmp_path, needle, replacement, body))


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("needle", "replacement", "domain"),
    [
        (
            'build_access_fixture("high_risk_no_approval", seed=fixture_seed)',
            'build_access_fixture("role_derived", seed=fixture_seed)',
            "access_provisioning",
        ),
        (
            "FinanceFixtureVariant.RESTRICTED_NO_APPROVAL, seed=fixture_seed",
            "FinanceFixtureVariant.SMALL_PERMITTED, seed=fixture_seed",
            "financial_adjustments",
        ),
        (
            'evaluation_time="2026-08-01T12:00:00Z"',
            'evaluation_time="2026-08-01T11:00:00Z"',
            "access_provisioning",
        ),
        (
            'evaluation_time="2026-01-15T00:00:00Z"',
            'evaluation_time="2026-01-15T01:00:00Z"',
            "financial_adjustments",
        ),
    ],
)
def test_mutation_kills_selected_fixture_and_time_bypasses(
    tmp_path: Path, needle: str, replacement: str, domain: str
) -> None:
    body = (
        "from decimal import Decimal\n"
        "from shadowskillbench.traces.bundles import generate_bundle\n"
        "from shadowskillbench.traces.models import action_trace_projection\n"
        f"expected = generate_bundle('{domain}', Decimal('0.5'), 12, 0)\n"
        f"actual = module.generate_bundle('{domain}', Decimal('0.5'), 12, 0)\n"
        "assert [action_trace_projection(trace) for trace in actual.bundle.traces] == [action_trace_projection(trace) for trace in expected.bundle.traces]\n"  # noqa: E501
    )
    _assert_killed(_run_mutant(tmp_path, needle, replacement, body))


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("needle", "replacement", "body"),
    [
        (
            'if info.mode != "python":',
            "if False:",
            "import json\n"
            "from pydantic import TypeAdapter\n"
            "payload = {'source_slot': 0, 'fixture_seed': 0, 'fixture_variant': 'high_risk_no_approval', 'trace_id': 'trace_' + '0' * 32, 'trace_hash': 'sha256:' + '0' * 64, 'semantic_class': 'compliant', 'worker_policy_id': 'access_compliant_v1'}\n"  # noqa: E501
            "try:\n"
            "    TypeAdapter(module.HiddenTraceBinding).validate_json(json.dumps(payload))\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('JSON schema ingress was accepted')\n",
        ),
        (
            "or _TRACE.fullmatch(self.trace_id) is None",
            "or False",
            "payload = {'source_slot': 0, 'fixture_seed': 0, 'fixture_variant': 'high_risk_no_approval', 'trace_id': 'trace_bad', 'trace_hash': 'sha256:' + '0' * 64, 'semantic_class': 'compliant', 'worker_policy_id': 'access_compliant_v1'}\n"  # noqa: E501
            "try:\n"
            "    module.HiddenTraceBinding.model_validate(payload)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('invalid binding was accepted')\n",
        ),
    ],
)
def test_mutation_kills_public_ingress_bypasses(
    tmp_path: Path, needle: str, replacement: str, body: str
) -> None:
    _assert_killed(_run_mutant(tmp_path, needle, replacement, body))


@pytest.mark.mutation
def test_mutation_kills_source_only_import_boundary(tmp_path: Path) -> None:
    body = (
        "from pathlib import Path\nassert 'import pathlib' not in Path(sys.argv[1]).read_text()\n"
    )
    _assert_killed(_run_mutant(tmp_path, "import re", "import re\nimport pathlib", body))
