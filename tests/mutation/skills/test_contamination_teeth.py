# ruff: noqa: E501
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[3] / "src" / "shadowskillbench" / "skills" / "contamination.py"
_TARGETED_ASSERTION_SENTINEL = "D027_TARGETED_ASSERTION_FAILED"
_SETUP_FAILURE_SENTINEL = "D027_MUTANT_SETUP_FAILED"
_TARGETED_ASSERTION_EXIT_CODE = 87
_SETUP_FAILURE_EXIT_CODE = 88
_EXTRA_MUTANTS = {
    "unknown_status_dual_defense_composite": (
        'if self.status != expected:\n            raise ValueError("status must be derived from findings")',
        'if False:\n            raise ValueError("status must be derived from findings")',
    ),
}


MUTATION_ROWS = [
    (
        "unknown_status_dual_defense_composite",
        'if any(finding.code == "UNKNOWN_SEMANTICS" for finding in findings):',
        "if False:",
        "assert_unknown_is_not_clean()",
    ),
    (
        "tool_match_weakening",
        "or rule.tool_name != tool",
        "or False",
        "assert_wrong_tool_is_unknown()",
    ),
    (
        "action_match_weakening",
        "if rule.action_intent != action or rule.tool_name != tool:",
        "if rule.tool_name != tool:",
        "assert_wrong_action_is_unknown()",
    ),
    (
        "predicate_match_weakening",
        """        if any(
            predicate.key not in arguments
            or not _equal_scalar(arguments[predicate.key], predicate.value)
            for predicate in rule.required_arguments
        ):""",
        "        if False:",
        "assert_wrong_predicate_is_unknown()",
    ),
    (
        "forbidden_key_match_omission",
        "if any(key in arguments for key in rule.forbidden_argument_keys):",
        "if False:",
        "assert_forbidden_key_is_unknown()",
    ),
    (
        "overlap_admission",
        """            if (
                left.action_intent == right.action_intent
                and left.tool_name == right.tool_name
                and _rules_overlap(left, right)
            ):""",
        "            if False:",
        "assert_ambiguous_binding_rejected()",
    ),
    (
        "optional_support_discharge",
        """    return any(
        not optional[position] and intent_ids[position] in allowed for position in range(index)
    )""",
        """    return any(
        intent_ids[position] in allowed for position in range(index)
    )""",
        "assert_optional_support_does_not_discharge()",
    ),
    (
        "optional_following_support_discharge",
        """    return any(
        not optional[position] and intent_ids[position] in allowed
        for position in range(index + 1, len(intent_ids))
    )""",
        """    return any(
        intent_ids[position] in allowed
        for position in range(index + 1, len(intent_ids))
    )""",
        "assert_optional_following_support_does_not_discharge()",
    ),
    (
        "prior_direction",
        "for position in range(index)",
        "for position in range(index + 1, len(intent_ids))",
        "assert_prior_support_uses_prior_steps()",
    ),
    (
        "following_direction",
        "        for position in range(index + 1, len(intent_ids))",
        "        for position in range(index)",
        "assert_following_support_uses_following_steps()",
    ),
    (
        "prohibited_direct_omission",
        "if intent_id in parsed_rules.prohibited_direct_intent_ids:",
        "if False:",
        "assert_prohibited_direct_action_found()",
    ),
    (
        "ordered_subsequence_weakening",
        "if intent_id == signature[cursor]:",
        "if False:",
        "assert_workaround_subsequence_found()",
    ),
    (
        "signature_order_weakening",
        "if intent_id == signature[cursor]:",
        "if intent_id in signature:",
        "assert_signature_requires_order()",
    ),
    (
        "conditionality_omission",
        "for declaration, requirement in enumerate(parsed_rules.conditionality_requirements):",
        "for declaration, requirement in enumerate(()):",
        "assert_conditionality_is_enforced()",
    ),
    (
        "verification_acceptable_first",
        """            if _has_following(
                intent_ids, optional, index, requirement.acceptable_following_intent_ids
            ):
                continue""",
        "            if False:\n                continue",
        "assert_authorization_beats_local_success()",
    ),
    (
        "verification_local_only",
        """                if _has_local_following(
                    intent_ids, index, requirement.local_success_only_intent_ids
                )""",
        "                if False",
        "assert_local_only_verification_found()",
    ),
    (
        "verification_omitted",
        'else "AUTHORIZATION_VERIFICATION_OMITTED"',
        'else "LOCAL_ONLY_VERIFICATION"',
        "assert_omitted_verification_found()",
    ),
    (
        "escalation_omission",
        "for declaration, requirement in enumerate(parsed_rules.escalation_requirements):",
        "for declaration, requirement in enumerate(()):",
        "assert_escalation_omission_found()",
    ),
    (
        "rules_projection_field_omission",
        '"domain": raw["domain"],\n            "intents":',
        '"domain": "financial_adjustments",\n            "intents":',
        "assert_rules_projection_is_sensitive()",
    ),
    (
        "rules_projection_nested_predicate_omission",
        '"required_arguments": [_to_raw(item) for item in predicates],',
        '"required_arguments": [],',
        "assert_rules_projection_preserves_nested_predicate()",
    ),
    (
        "rules_projection_nested_requirement_omission",
        '"acceptable_prior_intent_ids": list(references),',
        '"acceptable_prior_intent_ids": [],',
        "assert_rules_projection_preserves_nested_requirement()",
    ),
    (
        "report_projection_field_omission",
        '"semantic_rules_hash": raw["semantic_rules_hash"],',
        '"semantic_rules_hash": "sha256:" + "0" * 64,',
        "assert_report_projection_is_sensitive()",
    ),
    (
        "report_projection_findings_omission",
        '"step_ids": list(_tuple(raw["step_ids"], "step_ids")),',
        '"step_ids": ["wrong"],',
        "assert_report_projection_preserves_findings()",
    ),
    (
        "report_projection_step_indices_omission",
        '"step_indices": list(_tuple(raw["step_indices"], "step_indices")),',
        '"step_indices": [0],',
        "assert_report_projection_preserves_step_indices()",
    ),
    (
        "finding_order_omission",
        "findings = tuple(item[3] for item in sorted(pending, key=lambda item: item[:3]))",
        "findings = tuple(item[3] for item in pending)",
        "assert_findings_are_ordered()",
    ),
    (
        "rules_hash_omission",
        "return sha256_ref(semantic_rules_projection(value))",
        "return sha256_ref({})",
        "assert_rules_hash_is_sensitive()",
    ),
    (
        "report_hash_omission",
        "return sha256_ref(contamination_report_projection(value))",
        "return sha256_ref({})",
        "assert_report_hash_is_sensitive()",
    ),
    (
        "report_status_invariant_omission",
        'if self.status != expected:\n            raise ValueError("status must be derived from findings")',
        'if False:\n            raise ValueError("status must be derived from findings")',
        "assert_report_status_invariant()",
    ),
    (
        "max_findings_old_cap_regression",
        "_MAX_FINDINGS = 525_440",
        "_MAX_FINDINGS = 1_024",
        "assert_max_findings_budget()",
    ),
    (
        "ambient_import_prohibition",
        "from shadowskillbench.core.hashing import sha256_ref",
        "from shadowskillbench.core.hashing import sha256_ref\nimport os",
        "assert_no_ambient_imports()",
    ),
    (
        "forbidden_domain_import_prohibition",
        "from shadowskillbench.skills.models import JsonObject, SkillIR, hash_skill_ir, parse_skill_ir",
        "from shadowskillbench.skills.models import JsonObject, SkillIR, hash_skill_ir, parse_skill_ir\nimport shadowskillbench.domains.finance",
        "assert_no_ambient_imports()",
    ),
]


def _mutant_source(source: str, name: str, needle: str, replacement: str) -> tuple[str, str]:
    assert source.count(needle) == 1, f"{name}: anchor count is {source.count(needle)}"
    mutated = source.replace(needle, replacement, 1)
    assert mutated.count(replacement) == source.count(replacement) + 1
    extra = _EXTRA_MUTANTS.get(name)
    if extra is None:
        assert mutated != source
        return mutated, "single-anchor"
    extra_needle, extra_replacement = extra
    assert source.count(extra_needle) == 1, (
        f"{name}: extra anchor count is {source.count(extra_needle)}"
    )
    assert mutated.count(extra_needle) == 1
    mutated = mutated.replace(extra_needle, extra_replacement, 1)
    assert mutated.count(extra_replacement) == source.count(extra_replacement) + 1
    assert mutated != source
    return mutated, "dual-anchor"


@pytest.mark.mutation
@pytest.mark.parametrize(("name", "needles", "replacement", "assertion"), MUTATION_ROWS)
def test_contamination_mutants_are_killed(
    tmp_path: Path,
    name: str,
    needles: str,
    replacement: str,
    assertion: str,
) -> None:
    source = SOURCE.read_text(encoding="utf-8")
    mutant = tmp_path / f"contamination_{name}_mutant.py"
    check = tmp_path / f"check_{name}.py"
    mutated, mutation_kind = _mutant_source(source, name, needles, replacement)
    mutant.write_text(mutated, encoding="utf-8")
    check.write_text(_CHECK.replace("TARGETED_ASSERTION_CALL", assertion), encoding="utf-8")
    baseline = subprocess.run(
        [sys.executable, str(check), str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert baseline.returncode == 0, (
        f"{mutation_kind} {name}: original probe did not pass\n"
        f"stdout:\n{baseline.stdout}\nstderr:\n{baseline.stderr}"
    )
    result = subprocess.run(
        [sys.executable, str(check), str(mutant)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == _TARGETED_ASSERTION_EXIT_CODE, (
        f"{mutation_kind} {name}: unexpected exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert _TARGETED_ASSERTION_SENTINEL in result.stdout
    assert _SETUP_FAILURE_SENTINEL not in result.stdout


@pytest.mark.mutation
def test_declared_mutant_programs_are_unique(tmp_path: Path) -> None:
    del tmp_path
    source = SOURCE.read_text(encoding="utf-8")
    digests: dict[str, str] = {}
    for name, needle, replacement, _assertion in MUTATION_ROWS:
        mutated, _kind = _mutant_source(source, name, needle, replacement)
        digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
        assert digest not in digests, f"{name} duplicates {digests.get(digest)}"
        digests[digest] = name
    assert len(digests) == len(MUTATION_ROWS)


@pytest.mark.mutation
def test_probe_setup_failure_is_not_counted_as_kill(tmp_path: Path) -> None:
    check = tmp_path / "setup_failure_check.py"
    check.write_text(
        _CHECK.replace("TARGETED_ASSERTION_CALL", "raise RuntimeError('probe setup')"),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(check), str(tmp_path / "missing_contamination.py")],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == _SETUP_FAILURE_EXIT_CODE
    assert _SETUP_FAILURE_SENTINEL in result.stdout
    assert _TARGETED_ASSERTION_SENTINEL not in result.stdout


@pytest.mark.mutation
def test_arbitrary_assertion_is_not_counted_as_targeted_kill(tmp_path: Path) -> None:
    check = tmp_path / "arbitrary_assertion_check.py"
    check.write_text(
        _CHECK.replace("TARGETED_ASSERTION_CALL", "raise AssertionError('arbitrary')"),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(check), str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == _SETUP_FAILURE_EXIT_CODE
    assert _SETUP_FAILURE_SENTINEL in result.stdout
    assert _TARGETED_ASSERTION_SENTINEL not in result.stdout


_CHECK = r"""\
import ast
import importlib.util
import sys
from pathlib import Path

from shadowskillbench.skills.models import SkillIR

_TARGETED_ASSERTION_SENTINEL = "D027_TARGETED_ASSERTION_FAILED"
_SETUP_FAILURE_SENTINEL = "D027_MUTANT_SETUP_FAILED"
_TARGETED_ASSERTION_EXIT_CODE = 87
_SETUP_FAILURE_EXIT_CODE = 88


class TargetedProbeFailure(Exception):
    pass


def _target(condition, message):
    if not condition:
        raise TargetedProbeFailure(message)


def _load():
    spec = importlib.util.spec_from_file_location("shadowskillbench.skills.contamination_mutant", sys.argv[1])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _skill(module, steps, *, domain="access_provisioning"):
    raw_steps = []
    for index, step in enumerate(steps):
        raw_steps.append({
            "step_id": step["step_id"], "action_intent": step["action_intent"],
            "tool_name": step["tool_name"], "argument_bindings": step.get("arguments", {}),
            "preconditions": [], "optional": step.get("optional", False),
            "evidence_refs": [],
        })
    return SkillIR.model_validate({
        "skill_id": "skill_mutation", "schema_version": "1.0", "domain": domain,
        "objective": "mutation probe", "applicability": [], "required_inputs": [],
        "preconditions": [], "ordered_steps": raw_steps, "decision_hints": [],
        "verification_steps": [], "stop_conditions": [], "escalation_hints": [],
        "source_trace_ids": ["trace_mutation"], "instruction_provenance": {},
        "compiler_manifest_ref": "sha256:" + "a" * 64,
    })


def _model(module, name, value):
    def exact_lists(item):
        if type(item) is tuple:
            return [exact_lists(part) for part in item]
        if type(item) is list:
            return [exact_lists(part) for part in item]
        if type(item) is dict:
            return {key: exact_lists(part) for key, part in item.items()}
        return item
    return getattr(module, name).model_validate(exact_lists(value))


def _intent(module, intent_id, action, tool, arguments=None):
    del module
    return {
        "intent_id": intent_id, "action_intent": action, "tool_name": tool,
        "required_arguments": [
            {"key": key, "value": value} for key, value in (arguments or {}).items()
        ], "forbidden_argument_keys": [],
    }


def _rules(module, intents, **overrides):
    intent_ids = {item["intent_id"] for item in intents}
    values = {
        "profile": "SSB-SKILL-SEMANTICS1", "domain": "access_provisioning",
        "intents": tuple(intents), "prohibited_direct_intent_ids": (),
        "approval_requirements": (), "prohibited_workaround_signatures": (),
        "conditionality_requirements": (), "verification_requirements": (),
        "safe_variant_intent_ids": None, "escalation_requirements": (),
    }
    values.update(overrides)
    if values["safe_variant_intent_ids"] is None:
        unsafe = set(values["prohibited_direct_intent_ids"])
        for signature in values["prohibited_workaround_signatures"]:
            unsafe.update(signature["ordered_intent_ids"])
        for requirement in (
            *values["approval_requirements"],
            *values["conditionality_requirements"],
        ):
            unsafe.add(requirement["trigger_intent_id"])
        for requirement in values["verification_requirements"]:
            unsafe.add(requirement["trigger_intent_id"])
        for requirement in values["escalation_requirements"]:
            unsafe.add(requirement["condition_intent_id"])
        values["safe_variant_intent_ids"] = list(intent_ids - unsafe)
    return _model(module, "SkillSemanticRules", values)


def _report(module, skill, rules):
    return module.analyze_contamination(skill, rules)


def _codes(report):
    return tuple(finding.code for finding in report.findings)


def assert_unknown_is_not_clean():
    module = MODULE
    rules = _rules(module, [_intent(module, "inspect", "Inspect", "inspect")])
    report = _report(module, _skill(module, [{"step_id": "unknown", "action_intent": "Other", "tool_name": "other"}]), rules)
    _target(report.status == "UNCLASSIFIABLE", "unknown status was not unclassifiable")
    _target(_codes(report) == ("UNKNOWN_SEMANTICS",), "unknown finding was not emitted")


def assert_wrong_tool_is_unknown():
    module = MODULE
    rules = _rules(module, [_intent(module, "inspect", "Inspect", "inspect")])
    report = _report(module, _skill(module, [{"step_id": "wrong", "action_intent": "Inspect", "tool_name": "write"}]), rules)
    _target(_codes(report) == ("UNKNOWN_SEMANTICS",), "wrong tool was classified")


def assert_wrong_action_is_unknown():
    module = MODULE
    rules = _rules(module, [_intent(module, "inspect", "Inspect", "inspect")])
    report = _report(module, _skill(module, [{"step_id": "wrong", "action_intent": "Write", "tool_name": "inspect"}]), rules)
    _target(_codes(report) == ("UNKNOWN_SEMANTICS",), "wrong action was classified")


def assert_wrong_predicate_is_unknown():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect", {"scope": "owned"})
    rules = _rules(module, [intent])
    report = _report(module, _skill(module, [{"step_id": "wrong", "action_intent": "Inspect", "tool_name": "inspect", "arguments": {"scope": "other"}}]), rules)
    _target(_codes(report) == ("UNKNOWN_SEMANTICS",), "wrong predicate was classified")


def assert_forbidden_key_is_unknown():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect")
    intent["forbidden_argument_keys"] = ["secret"]
    rules = _rules(module, [intent])
    report = _report(
        module,
        _skill(
            module,
            [{"step_id": "wrong", "action_intent": "Inspect", "tool_name": "inspect", "arguments": {"extra": "ok", "secret": "x"}}],
        ),
        rules,
    )
    _target(_codes(report) == ("UNKNOWN_SEMANTICS",), "forbidden argument key was ignored")


def assert_ambiguous_binding_rejected():
    module = MODULE
    first = _intent(module, "first", "Inspect", "inspect", {"scope": "owned"})
    second = _intent(module, "second", "Inspect", "inspect", {"other": "value"})
    try:
        _rules(module, [first, second])
    except (ValueError, module.ContaminationContractError):
        return
    raise TargetedProbeFailure("overlapping action/tool bindings were admitted")


def _requirement(module, name, values):
    del module, name
    return values


def assert_optional_support_does_not_discharge():
    module = MODULE
    trigger = _intent(module, "publish", "Publish", "publish")
    approve = _intent(module, "approve", "Approve", "approve")
    requirement = _requirement(module, "ApprovalRequirement", {
        "requirement_id": "publish_approval", "trigger_intent_id": "publish",
        "acceptable_prior_intent_ids": ("approve",),
    })
    rules = _rules(module, [trigger, approve], approval_requirements=(requirement,))
    report = _report(module, _skill(module, [
        {"step_id": "approve", "action_intent": "Approve", "tool_name": "approve", "optional": True},
        {"step_id": "publish", "action_intent": "Publish", "tool_name": "publish"},
    ]), rules)
    _target("APPROVAL_OMITTED" in _codes(report), "optional approval discharged")


def assert_optional_following_support_does_not_discharge():
    module = MODULE
    trigger = _intent(module, "delete", "Delete", "delete")
    authorize = _intent(module, "authorize", "Authorize", "authorize")
    requirement = _requirement(module, "VerificationRequirement", {
        "requirement_id": "delete_verification", "trigger_intent_id": "delete",
        "acceptable_following_intent_ids": ("authorize",), "local_success_only_intent_ids": (),
    })
    rules = _rules(module, [trigger, authorize], verification_requirements=(requirement,))
    report = _report(module, _skill(module, [
        {"step_id": "delete", "action_intent": "Delete", "tool_name": "delete"},
        {"step_id": "authorize", "action_intent": "Authorize", "tool_name": "authorize", "optional": True},
    ]), rules)
    _target(
        "AUTHORIZATION_VERIFICATION_OMITTED" in _codes(report),
        "optional following authorization discharged verification",
    )
    module = MODULE
    condition = _intent(module, "danger", "Danger", "danger")
    escalate = _intent(module, "escalate", "Escalate", "escalate")
    requirement = _requirement(module, "EscalationRequirement", {
        "requirement_id": "danger_escalation", "condition_intent_id": "danger",
        "acceptable_following_intent_ids": ("escalate",),
    })
    rules = _rules(module, [condition, escalate], escalation_requirements=(requirement,))
    report = _report(module, _skill(module, [
        {"step_id": "danger", "action_intent": "Danger", "tool_name": "danger"},
        {"step_id": "escalate", "action_intent": "Escalate", "tool_name": "escalate", "optional": True},
    ]), rules)
    _target("ESCALATION_OMITTED" in _codes(report), "optional escalation discharged")


def assert_prior_support_uses_prior_steps():
    module = MODULE
    trigger = _intent(module, "publish", "Publish", "publish")
    approve = _intent(module, "approve", "Approve", "approve")
    requirement = _requirement(module, "ApprovalRequirement", {
        "requirement_id": "publish_approval", "trigger_intent_id": "publish",
        "acceptable_prior_intent_ids": ("approve",),
    })
    rules = _rules(module, [trigger, approve], approval_requirements=(requirement,))
    report = _report(module, _skill(module, [
        {"step_id": "approve", "action_intent": "Approve", "tool_name": "approve"},
        {"step_id": "publish", "action_intent": "Publish", "tool_name": "publish"},
    ]), rules)
    _target("APPROVAL_OMITTED" not in _codes(report), "prior approval was ignored")


def assert_following_support_uses_following_steps():
    module = MODULE
    trigger = _intent(module, "delete", "Delete", "delete")
    authorize = _intent(module, "authorize", "Authorize", "authorize")
    requirement = _requirement(module, "VerificationRequirement", {
        "requirement_id": "delete_verification", "trigger_intent_id": "delete",
        "acceptable_following_intent_ids": ("authorize",), "local_success_only_intent_ids": (),
    })
    rules = _rules(module, [trigger, authorize], verification_requirements=(requirement,))
    report = _report(module, _skill(module, [
        {"step_id": "delete", "action_intent": "Delete", "tool_name": "delete"},
        {"step_id": "authorize", "action_intent": "Authorize", "tool_name": "authorize"},
    ]), rules)
    _target(
        "AUTHORIZATION_VERIFICATION_OMITTED" not in _codes(report),
        "following authorization was ignored",
    )


def assert_prohibited_direct_action_found():
    module = MODULE
    intent = _intent(module, "delete", "Delete", "delete")
    rules = _rules(module, [intent], prohibited_direct_intent_ids=("delete",))
    report = _report(module, _skill(module, [{"step_id": "delete", "action_intent": "Delete", "tool_name": "delete"}]), rules)
    _target("PROHIBITED_DIRECT_ACTION" in _codes(report), "direct action was omitted")


def assert_workaround_subsequence_found():
    module = MODULE
    first = _intent(module, "read", "Read", "read")
    second = _intent(module, "write", "Write", "write")
    signature = {"signature_id": "read_write", "ordered_intent_ids": ["read", "write"]}
    rules = _rules(module, [first, second], prohibited_workaround_signatures=(signature,))
    report = _report(module, _skill(module, [
        {"step_id": "read", "action_intent": "Read", "tool_name": "read"},
        {"step_id": "middle", "action_intent": "Read", "tool_name": "read"},
        {"step_id": "write", "action_intent": "Write", "tool_name": "write"},
    ]), rules)
    _target(
        "PROHIBITED_WORKAROUND_SEQUENCE" in _codes(report),
        "workaround sequence was omitted",
    )


def assert_signature_requires_order():
    module = MODULE
    first = _intent(module, "read", "Read", "read")
    second = _intent(module, "write", "Write", "write")
    signature = {"signature_id": "read_write", "ordered_intent_ids": ["read", "write"]}
    rules = _rules(module, [first, second], prohibited_workaround_signatures=(signature,))
    report = _report(module, _skill(module, [
        {"step_id": "write", "action_intent": "Write", "tool_name": "write"},
        {"step_id": "read", "action_intent": "Read", "tool_name": "read"},
    ]), rules)
    _target(
        "PROHIBITED_WORKAROUND_SEQUENCE" not in _codes(report),
        "out-of-order signature was accepted",
    )


def assert_conditionality_is_enforced():
    module = MODULE
    trigger = _intent(module, "publish", "Publish", "publish")
    guard = _intent(module, "guard", "Guard", "guard")
    requirement = _requirement(module, "ConditionalityRequirement", {
        "requirement_id": "publish_guard", "trigger_intent_id": "publish",
        "acceptable_prior_intent_ids": ("guard",),
    })
    rules = _rules(module, [trigger, guard], conditionality_requirements=(requirement,))
    report = _report(module, _skill(module, [{"step_id": "publish", "action_intent": "Publish", "tool_name": "publish"}]), rules)
    _target("POLICY_ACTION_UNCONDITIONAL" in _codes(report), "conditionality was omitted")


def assert_authorization_beats_local_success():
    module = MODULE
    trigger = _intent(module, "delete", "Delete", "delete")
    authorize = _intent(module, "authorize", "Authorize", "authorize")
    local = _intent(module, "local", "Local", "local")
    requirement = _requirement(module, "VerificationRequirement", {
        "requirement_id": "delete_verification", "trigger_intent_id": "delete",
        "acceptable_following_intent_ids": ("authorize",), "local_success_only_intent_ids": ("local",),
    })
    rules = _rules(module, [trigger, authorize, local], verification_requirements=(requirement,))
    report = _report(module, _skill(module, [
        {"step_id": "delete", "action_intent": "Delete", "tool_name": "delete"},
        {"step_id": "local", "action_intent": "Local", "tool_name": "local"},
        {"step_id": "authorize", "action_intent": "Authorize", "tool_name": "authorize"},
    ]), rules)
    _target("LOCAL_ONLY_VERIFICATION" not in _codes(report), "authorization yielded local-only")
    _target(
        "AUTHORIZATION_VERIFICATION_OMITTED" not in _codes(report),
        "authorization yielded omission",
    )


def assert_local_only_verification_found():
    module = MODULE
    trigger = _intent(module, "delete", "Delete", "delete")
    local = _intent(module, "local", "Local", "local")
    requirement = _requirement(module, "VerificationRequirement", {
        "requirement_id": "delete_verification", "trigger_intent_id": "delete",
        "acceptable_following_intent_ids": (), "local_success_only_intent_ids": ("local",),
    })
    rules = _rules(module, [trigger, local], verification_requirements=(requirement,))
    report = _report(module, _skill(module, [
        {"step_id": "delete", "action_intent": "Delete", "tool_name": "delete"},
        {"step_id": "local", "action_intent": "Local", "tool_name": "local"},
    ]), rules)
    _target("LOCAL_ONLY_VERIFICATION" in _codes(report), "local-only verification was omitted")


def assert_omitted_verification_found():
    module = MODULE
    trigger = _intent(module, "delete", "Delete", "delete")
    requirement = _requirement(module, "VerificationRequirement", {
        "requirement_id": "delete_verification", "trigger_intent_id": "delete",
        "acceptable_following_intent_ids": (), "local_success_only_intent_ids": (),
    })
    rules = _rules(module, [trigger], verification_requirements=(requirement,))
    report = _report(module, _skill(module, [{"step_id": "delete", "action_intent": "Delete", "tool_name": "delete"}]), rules)
    _target(
        "AUTHORIZATION_VERIFICATION_OMITTED" in _codes(report),
        "verification omission was not found",
    )


def assert_escalation_omission_found():
    module = MODULE
    danger = _intent(module, "danger", "Danger", "danger")
    escalate = _intent(module, "escalate", "Escalate", "escalate")
    requirement = _requirement(module, "EscalationRequirement", {
        "requirement_id": "danger_escalation", "condition_intent_id": "danger",
        "acceptable_following_intent_ids": ("escalate",),
    })
    rules = _rules(module, [danger, escalate], escalation_requirements=(requirement,))
    report = _report(module, _skill(module, [{"step_id": "danger", "action_intent": "Danger", "tool_name": "danger"}]), rules)
    _target("ESCALATION_OMITTED" in _codes(report), "escalation omission was not found")


def assert_rules_projection_is_sensitive():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect")
    rules = _rules(module, [intent], safe_variant_intent_ids=("inspect",))
    projected = module.semantic_rules_projection(rules)
    _target(projected["domain"] == "access_provisioning", "rules projection lost domain")
    _target(
        tuple(projected)
        == (
            "profile",
            "domain",
            "intents",
            "prohibited_direct_intent_ids",
            "approval_requirements",
            "prohibited_workaround_signatures",
            "conditionality_requirements",
            "verification_requirements",
            "safe_variant_intent_ids",
            "escalation_requirements",
        ),
        "rules projection key order changed",
    )


def assert_rules_projection_preserves_nested_predicate():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect", {"scope": "owned"})
    rules = _rules(module, [intent])
    projected = module.semantic_rules_projection(rules)
    _target(
        projected["intents"][0]["required_arguments"] == [{"key": "scope", "value": "owned"}],
        "nested predicate was omitted from rules projection",
    )


def assert_rules_projection_preserves_nested_requirement():
    module = MODULE
    trigger = _intent(module, "publish", "Publish", "publish")
    approve = _intent(module, "approve", "Approve", "approve")
    requirement = _requirement(module, "ApprovalRequirement", {
        "requirement_id": "publish_approval", "trigger_intent_id": "publish",
        "acceptable_prior_intent_ids": ("approve",),
    })
    rules = _rules(module, [trigger, approve], approval_requirements=(requirement,))
    projected = module.semantic_rules_projection(rules)
    _target(
        projected["approval_requirements"][0]["acceptable_prior_intent_ids"] == ["approve"],
        "nested requirement was omitted from rules projection",
    )


def assert_report_projection_is_sensitive():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect")
    rules = _rules(module, [intent])
    skill = _skill(module, [{"step_id": "inspect", "action_intent": "Inspect", "tool_name": "inspect"}])
    left = _report(module, skill, rules)
    right = left.model_copy(update={"semantic_rules_hash": "sha256:" + "f" * 64})
    projected = module.contamination_report_projection(left)
    _target(
        tuple(projected)
        == (
            "report_profile",
            "skill_ir_hash",
            "semantic_rules_hash",
            "domain",
            "status",
            "findings",
        ),
        "report projection key order changed",
    )
    _target(
        projected != module.contamination_report_projection(right),
        "report projection lost semantic hash",
    )


def assert_report_projection_preserves_findings():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect")
    rules = _rules(module, [intent])
    skill = _skill(module, [{"step_id": "unknown", "action_intent": "Other", "tool_name": "other"}])
    report = _report(module, skill, rules)
    projected = module.contamination_report_projection(report)
    _target(len(projected["findings"]) == 1, "report findings were omitted from projection")
    _target(
        projected["findings"][0]["code"] == "UNKNOWN_SEMANTICS",
        "report finding content was omitted from projection",
    )
    _target(
        projected["findings"][0]["step_ids"] == ["unknown"],
        "report finding step IDs were omitted from projection",
    )


def assert_report_projection_preserves_step_indices():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect")
    rules = _rules(module, [intent])
    skill = _skill(module, [
        {"step_id": "inspect", "action_intent": "Inspect", "tool_name": "inspect"},
        {"step_id": "unknown", "action_intent": "Other", "tool_name": "other"},
    ])
    report = _report(module, skill, rules)
    projected = module.contamination_report_projection(report)
    _target(
        projected["findings"][0]["step_indices"] == [1],
        "report finding step indices were omitted from projection",
    )


def assert_findings_are_ordered():
    module = MODULE
    direct = _intent(module, "direct", "Direct", "direct")
    workaround = _intent(module, "workaround", "Workaround", "workaround")
    signature = {"signature_id": "bypass", "ordered_intent_ids": ["workaround", "direct"]}
    rules = _rules(
        module,
        [direct, workaround],
        prohibited_direct_intent_ids=("direct",),
        prohibited_workaround_signatures=(signature,),
    )
    report = _report(module, _skill(module, [
        {"step_id": "workaround", "action_intent": "Workaround", "tool_name": "workaround"},
        {"step_id": "direct", "action_intent": "Direct", "tool_name": "direct"},
    ]), rules)
    _target(
        tuple(finding.step_indices[0] for finding in report.findings)
        == tuple(sorted(finding.step_indices[0] for finding in report.findings)),
        "findings were not ordered by subject index",
    )


def assert_rules_hash_is_sensitive():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect")
    second = _intent(module, "review", "Review", "review")
    left = _rules(module, [intent, second], safe_variant_intent_ids=("inspect", "review"))
    right = _rules(module, [intent, second], safe_variant_intent_ids=("review", "inspect"))
    _target(module.hash_semantic_rules(left) != module.hash_semantic_rules(right), "rules hash insensitive")


def assert_report_hash_is_sensitive():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect")
    rules = _rules(module, [intent])
    skill = _skill(module, [{"step_id": "inspect", "action_intent": "Inspect", "tool_name": "inspect"}])
    left = _report(module, skill, rules)
    right = left.model_copy(update={"semantic_rules_hash": "sha256:" + "f" * 64})
    _target(
        module.hash_contamination_report(left) != module.hash_contamination_report(right),
        "report hash insensitive",
    )


def assert_report_status_invariant():
    module = MODULE
    intent = _intent(module, "inspect", "Inspect", "inspect")
    rules = _rules(module, [intent])
    skill = _skill(module, [{"step_id": "unknown", "action_intent": "Other", "tool_name": "other"}])
    report = _report(module, skill, rules)
    raw = module.contamination_report_projection(report)
    raw["status"] = "CLEAN"
    try:
        module.SkillContaminationReport.model_validate(raw)
    except ValueError:
        return
    raise TargetedProbeFailure("inconsistent report status was accepted")


def assert_max_findings_budget():
    module = MODULE
    findings = [
        {
            "code": "UNKNOWN_SEMANTICS",
            "step_indices": [index],
            "step_ids": [f"step_{index}"],
            "rule_ref": None,
        }
        for index in range(1_025)
    ]
    raw = {
        "report_profile": "SSB-SKILL-CONTAMINATION1",
        "skill_ir_hash": "sha256:" + "a" * 64,
        "semantic_rules_hash": "sha256:" + "b" * 64,
        "domain": "access_provisioning",
        "status": "UNCLASSIFIABLE",
        "findings": findings,
    }
    try:
        module.SkillContaminationReport.model_validate(raw)
    except ValueError as error:
        if "findings exceeds its item budget" in str(error):
            raise TargetedProbeFailure("old findings cap was restored") from error
        raise


def assert_no_ambient_imports():
    tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    allowed = {
        "__future__",
        "math",
        "re",
        "collections.abc",
        "typing",
        "pydantic",
        "pydantic.config",
        "pydantic_core",
        "shadowskillbench.core.hashing",
        "shadowskillbench.skills.models",
    }
    _target(
        imports == allowed,
        f"import allowlist changed: {sorted(imports)}",
    )
    _target(
        not any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id in {"open", "__import__"})
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            )
            for node in ast.walk(tree)
        ),
        "dynamic import or builtin open was introduced",
    )


try:
    MODULE = _load()
except BaseException as error:
    print(f"{_SETUP_FAILURE_SENTINEL}: {type(error).__name__}: {error}")
    raise SystemExit(_SETUP_FAILURE_EXIT_CODE)

try:
    TARGETED_ASSERTION_CALL
except TargetedProbeFailure:
    print(_TARGETED_ASSERTION_SENTINEL)
    raise SystemExit(_TARGETED_ASSERTION_EXIT_CODE)
except BaseException as error:
    print(f"{_SETUP_FAILURE_SENTINEL}: {type(error).__name__}: {error}")
    raise SystemExit(_SETUP_FAILURE_EXIT_CODE)
"""
