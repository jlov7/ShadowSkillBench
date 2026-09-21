from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RESOLVER_SOURCE = (
    Path(__file__).parents[3] / "src" / "shadowskillbench" / "authority" / "resolver.py"
)
UNIT_HELPERS = Path(__file__).parents[2] / "unit" / "authority" / "test_resolver.py"


def _run_mutant(
    tmp_path: Path, needle: str, replacement: str, assertion: str
) -> subprocess.CompletedProcess[str]:
    source = RESOLVER_SOURCE.read_text(encoding="utf-8")
    assert needle in source
    mutant = tmp_path / "resolver_mutant.py"
    script = tmp_path / "resolver_invariant.py"
    mutant.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    script.write_text(
        "\n".join(
            [
                "import importlib.util, runpy, sys",
                "helpers = runpy.run_path(sys.argv[2])",
                "spec = importlib.util.spec_from_file_location('mutant', sys.argv[1])",
                "module = importlib.util.module_from_spec(spec)",
                "assert spec.loader is not None",
                "sys.modules[spec.name] = module",
                "spec.loader.exec_module(module)",
                "resolve = module.resolve_authority",
                assertion,
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


def _lines(*lines: str) -> str:
    return "\n".join(lines)


def _expect_admission(records_expression: str) -> str:
    return _lines(
        f"records = {records_expression}",
        "try:",
        "    resolve(records, helpers['_access_query']())",
        "except helpers['AuthorityAdmissionError']:",
        "    pass",
        "else:",
        "    raise AssertionError('malformed corpus was admitted')",
    )


@pytest.mark.mutation
def test_mutation_kills_filter_before_admission_validation(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "_validate_issuer_authorization(ordered, query)",
        "_ = query",
        _expect_admission("[helpers['_record']('irrelevant_bad_rank', authority_rank=9)]"),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_branched_supersession_admission(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if target.authority_id in incoming:",
        "if False:",
        _expect_admission(
            "[helpers['_record']('target', status='SUPERSEDED', "
            "scope=helpers['_access_scope'](role_id=None)), "
            "helpers['_record']('source_a', issuer_id='issuer_senior', authority_rank=20, "
            "effective_at='2026-01-02T00:00:00Z', supersedes=['target']), "
            "helpers['_record']('source_b', issuer_id='issuer_senior', authority_rank=20, "
            "effective_at='2026-01-03T00:00:00Z', supersedes=['target'])]"
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_lower_rank_superseder_admission(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if source.authority_rank < target.authority_rank:",
        "if False:",
        _expect_admission(
            "[helpers['_record']('target', issuer_id='issuer_senior', authority_rank=20, "
            "status='SUPERSEDED'), helpers['_record']('lower', "
            "effective_at='2026-01-02T00:00:00Z', supersedes=['target'])]"
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_open_ended_superseder_of_finite_target(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if not _time_contains(target, source):",
        "if False:",
        _expect_admission(
            "[helpers['_record']('target', status='SUPERSEDED', "
            "expires_at='2026-12-31T00:00:00Z'), "
            "helpers['_record']('source', issuer_id='issuer_senior', authority_rank=20, "
            "effective_at='2026-01-02T00:00:00Z', supersedes=['target'])]"
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_broader_evidence_scope(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if not _scope_contains(target, evidence):",
        "if False:",
        _expect_admission(
            "[helpers['_record']('target', scope=helpers['_access_scope']("
            "organization_id='organization_a', disposition='REQUIRE_APPROVAL')), "
            "helpers['_record']('approval', source_type='APPROVAL', status='ACTIVE_AUTHORITY', "
            "scope=helpers['_access_evidence_scope'](organization_id=None), "
            "exception_to=['target'])]"
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_explicit_top_status_conflict_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if status_conflict_ids or not is_equivalent:",
        "if False:",
        _lines(
            "record = helpers['_record']('conflict', status='CONFLICTING')",
            "decision = resolve([record], helpers['_access_query']())",
            "assert decision.decision is helpers['Decision'].ESCALATE",
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_waiver_precedence_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if waivers:",
        "if False:",
        _lines(
            "required = helpers['_record']('required', "
            "scope=helpers['_access_scope'](disposition='REQUIRE_APPROVAL'))",
            "approval = helpers['_record']('approval', source_type='APPROVAL', "
            "status='ACTIVE_AUTHORITY', scope=helpers['_access_evidence_scope'](), "
            "exception_to=['required'])",
            "waiver = helpers['_record']('waiver', source_type='WAIVER', "
            "status='SCOPED_EXCEPTION', scope=helpers['_access_evidence_scope'](), "
            "exception_to=['required'])",
            "decision = resolve([required, approval, waiver], helpers['_access_query']())",
            "assert decision.decision is helpers['Decision'].PROCEED_UNDER_EXCEPTION",
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_reference_aware_coalesced_path_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "referenced or paths",
        "paths",
        _lines(
            "base_a = helpers['_record']('base_a', status='SUPERSEDED', "
            "scope=helpers['_access_scope'](role_id=None))",
            "base_b = helpers['_record']('base_b', status='SUPERSEDED', "
            "scope=helpers['_access_scope'](role_id=None))",
            "effective_a = helpers['_record']('effective_a', issuer_id='issuer_senior', "
            "authority_rank=20, effective_at='2026-01-02T00:00:00Z', supersedes=['base_a'])",
            "effective_b = helpers['_record']('effective_b', issuer_id='issuer_senior', "
            "authority_rank=20, effective_at='2026-01-02T00:00:00Z', supersedes=['base_b'])",
            "decision = resolve([base_a, effective_a, base_b, effective_b], "
            "helpers['_access_query'](references=['base_b']))",
            "assert decision.supersession_path == ('base_b', 'effective_b')",
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_discarded_record_hash_exclusion(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "authority_set_hash(list(admitted_records))",
        "authority_set_hash([record for record in admitted_records "
        "if record.source_type not in _DESCRIPTIVE_SOURCES])",
        _lines(
            "rule = helpers['_record']('rule')",
            "trace = helpers['_record']('trace', source_type='BEHAVIOR_TRACE', "
            "status='DESCRIPTIVE_ONLY', scope={'domain': 'access_provisioning', "
            "'scope_kind': 'descriptive', 'subject_id': None, 'resource_id': None, "
            "'organization_id': None, 'geography_id': None, 'role_id': None})",
            "with_trace = resolve([rule, trace], helpers['_access_query']())",
            "without_trace = resolve([rule], helpers['_access_query']())",
            "assert with_trace.authority_set_hash != without_trace.authority_set_hash",
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_singleton_path_erasure(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "reporting_path = full_path",
        "reporting_path = ()",
        _lines(
            "rule = helpers['_record']('rule')",
            "decision = resolve([rule], helpers['_access_query']())",
            "assert decision.supersession_path == ('rule',)",
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_hostile_nested_container_preflight_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if not _has_exact_nested_structure(record):",
        "if False:",
        _lines(
            "class HostileList(list):",
            "    def __iter__(self):",
            "        raise AssertionError('hostile hook invoked')",
            "record = helpers['_record']('rule').model_copy(update={'supersedes': HostileList()})",
            "try:",
            "    resolve([record], helpers['_access_query']())",
            "except helpers['AuthorityAdmissionError']:",
            "    pass",
            "else:",
            "    raise AssertionError('hostile record was admitted')",
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_hostile_validation_marker_truthiness(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        'if "_validated" not in values or values["_validated"] is not True:',
        'if not values.get("_validated", False):',
        _lines(
            "class HostileValidatedMarker:",
            "    calls = 0",
            "    def __bool__(self):",
            "        type(self).calls += 1",
            "        raise AssertionError('hostile marker truthiness invoked')",
            "record = helpers['_record']('rule').model_copy("
            "update={'_validated': HostileValidatedMarker()})",
            "try:",
            "    resolve([record], helpers['_access_query']())",
            "except helpers['AuthorityAdmissionError']:",
            "    pass",
            "else:",
            "    raise AssertionError('hostile marker was admitted')",
            "assert HostileValidatedMarker.calls == 0",
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_extra_model_key_preflight_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "if len(values) != len(fields) + 1:",
        "if False:",
        _lines(
            "class HostileExtra:",
            "    calls = 0",
            "    def __bool__(self):",
            "        type(self).calls += 1",
            "        raise AssertionError('hostile extra hook invoked')",
            "record = helpers['_record']('rule').model_copy(update={'evil': HostileExtra()})",
            "try:",
            "    resolve([record], helpers['_access_query']())",
            "except helpers['AuthorityAdmissionError']:",
            "    pass",
            "else:",
            "    raise AssertionError('extra model key was admitted')",
            "assert HostileExtra.calls == 0",
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_active_path_cycle_guard_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        (
            "if node_id in active_path:\n                return False\n"
            "            active_path.add(node_id)"
        ),
        (
            "if node_id in active_path:\n                return True\n"
            "            active_path.add(node_id)"
        ),
        _lines(
            "cycle = []",
            "cycle.append(cycle)",
            "record = helpers['_record']('rule').model_copy(update={'supersedes': cycle})",
            "assert module._has_exact_nested_structure(record) is False",
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_threshold_equality_regression(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "amount > scope.threshold_minor",
        "amount >= scope.threshold_minor",
        "\n".join(
            [
                "rule = helpers['_record']('finance', action_type='post_adjustment',",
                "    scope=helpers['_finance_scope'](threshold_minor=500))",
                "decision = resolve([rule], helpers['_finance_query'](-500))",
                "assert decision.decision is helpers['Decision'].PROCEED",
            ]
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_follow_without_explicit_reference(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "and follows_reference",
        "and True",
        "\n".join(
            [
                "base = helpers['_record']('base', status='SUPERSEDED',",
                "    scope=helpers['_access_scope'](role_id=None))",
                "new = helpers['_record']('new', issuer_id='issuer_senior',",
                "    authority_rank=20, supersedes=['base'])",
                "decision = resolve([base, new], helpers['_access_query']())",
                "assert decision.decision is helpers['Decision'].PROCEED",
            ]
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_approval_unblocks_block(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "and first_disposition is RuleDisposition.REQUIRE_APPROVAL",
        "and first_disposition in {RuleDisposition.BLOCK, RuleDisposition.REQUIRE_APPROVAL}",
        "\n".join(
            [
                "blocked = helpers['_record']('blocked',",
                "    scope=helpers['_access_scope'](disposition='BLOCK'))",
                "approval = helpers['_record']('approval', source_type='APPROVAL',",
                "    status='ACTIVE_AUTHORITY', scope=helpers['_access_evidence_scope'](),",
                "    exception_to=['blocked'])",
                "decision = resolve([blocked, approval], helpers['_access_query']())",
                "assert decision.decision is helpers['Decision'].BLOCK",
            ]
        ),
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_mutation_kills_cyclic_graph_bypass(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        "_validate_acyclic_supersession(ordered)",
        "_ = ordered",
        "\n".join(
            [
                "a = helpers['_record']('a', status='SUPERSEDED', supersedes=['b'])",
                "b = helpers['_record']('b', status='SUPERSEDED', supersedes=['a'])",
                "try:",
                "    resolve([a, b], helpers['_access_query']())",
                "except helpers['AuthorityAdmissionError']:",
                "    pass",
                "else:",
                "    raise AssertionError('cyclic corpus admitted')",
            ]
        ),
    )
    _assert_killed(result)
