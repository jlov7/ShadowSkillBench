from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.skills import (
    OrderedStep,
    SkillIR,
    hash_skill_ir,
    parse_skill_ir,
    skill_ir_projection,
)

HASH = "sha256:" + "a" * 64


def raw_skill(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "skill_id": "skill_access_review",
        "schema_version": "1.0",
        "domain": "access_provisioning",
        "objective": "Review an access request before provisioning.",
        "applicability": ["Use for a standard access request."],
        "required_inputs": ["An access request identifier."],
        "preconditions": ["The request is visible."],
        "ordered_steps": [
            {
                "step_id": "review_request",
                "action_intent": "Review the requested access.",
                "tool_name": "inspect_request",
                "argument_bindings": {"request_id": "REQ-1", "scope": ["read"]},
                "preconditions": ["The request identifier is supplied."],
                "optional": False,
                "evidence_refs": ["visible request record"],
            }
        ],
        "decision_hints": ["Escalate unusual scope."],
        "verification_steps": ["Confirm the request status."],
        "stop_conditions": ["Stop when the request is unavailable."],
        "escalation_hints": ["Escalate missing approval."],
        "source_trace_ids": ["trace_access_01"],
        "instruction_provenance": {"source": "demonstration"},
        "compiler_manifest_ref": HASH,
    }
    value.update(overrides)
    return value


def test_parse_has_exact_tuples_detaches_and_matches_immutable_schema() -> None:
    raw = raw_skill()
    parsed = parse_skill_ir(raw)
    schema = json.loads(
        (Path(__file__).parents[3] / "schemas" / "skill_ir.schema.json").read_text()
    )

    assert type(parsed.applicability) is tuple
    assert type(parsed.ordered_steps) is tuple
    assert type(parsed.ordered_steps[0].preconditions) is tuple
    assert parsed.ordered_steps[0].argument_bindings == {"request_id": "REQ-1", "scope": ["read"]}
    Draft202012Validator(schema).validate(skill_ir_projection(parsed))
    raw["objective"] = "Changed."
    cast(dict[str, object], cast(list[object], raw["ordered_steps"])[0])["argument_bindings"] = {}
    assert parsed.objective == "Review an access request before provisioning."
    assert parsed.ordered_steps[0].argument_bindings["scope"] == ["read"]


@pytest.mark.parametrize(
    "field",
    [
        "preconditions",
        "decision_hints",
        "verification_steps",
        "stop_conditions",
        "escalation_hints",
    ],
)
def test_all_canonical_fields_are_required_even_when_wire_schema_allows_omission(
    field: str,
) -> None:
    raw = raw_skill()
    del raw[field]
    with pytest.raises(ValueError):
        parse_skill_ir(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"skill_id": "bad"}),
        lambda value: value.update({"schema_version": "1"}),
        lambda value: value.update({"domain": "other"}),
        lambda value: value.update({"compiler_manifest_ref": "sha256:" + "A" * 64}),
        lambda value: value.update({"source_trace_ids": ["trace_a", "trace_a"]}),
        lambda value: value.update({"source_trace_ids": []}),
        lambda value: value.update({"source_trace_ids": ["trace_a"] * 13}),
        lambda value: value.update({"ordered_steps": []}),
        lambda value: value.update(
            {"ordered_steps": [cast(list[object], value["ordered_steps"])[0]] * 2}
        ),
    ],
)
def test_required_identifiers_and_cardinalities_are_strict(mutation: object) -> None:
    raw = raw_skill()
    cast(Any, mutation)(raw)
    with pytest.raises(ValueError):
        parse_skill_ir(raw)


def test_tool_names_may_repeat_and_action_intent_is_not_semantically_classified() -> None:
    raw = raw_skill()
    raw["ordered_steps"] = cast(list[object], raw["ordered_steps"]) + [
        {
            "step_id": "apply_access",
            "action_intent": "A new, unknown operational instruction.",
            "tool_name": "inspect_request",
            "argument_bindings": {},
            "preconditions": [],
            "optional": True,
            "evidence_refs": [],
        }
    ]
    assert len(parse_skill_ir(raw).ordered_steps) == 2


@pytest.mark.parametrize("marker", ["access_compliant_v1", "compliant"])
def test_identifier_fields_reject_exact_treatment_values(marker: str) -> None:
    for field in ("step_id", "tool_name"):
        raw = raw_skill()
        step = cast(dict[str, object], cast(list[object], raw["ordered_steps"])[0])
        step[field] = marker
        with pytest.raises(ValueError):
            parse_skill_ir(raw)


@pytest.mark.parametrize(
    "key",
    [
        "worker_policy_id",
        "semantic_class",
        "contamination_ratio",
        "hidden_benchmark_metadata_ref",
        "hidden_class_manifest_hash",
        "ordered_trace_hashes",
        "source_manifest",
        "hidden_class_manifest",
    ],
)
def test_exact_treatment_keys_are_rejected_but_substrings_are_not(key: str) -> None:
    raw = raw_skill(instruction_provenance={key: "marker"})
    with pytest.raises(ValueError):
        parse_skill_ir(raw)
    assert parse_skill_ir(raw_skill(instruction_provenance={f"safe_{key}": "marker"}))


@pytest.mark.parametrize(
    "value",
    [
        "access_compliant_v1",
        "access_workaround_v1",
        "finance_compliant_v1",
        "finance_workaround_v1",
        "compliant",
        "prohibited_workaround",
    ],
)
def test_exact_treatment_values_are_rejected_but_substrings_are_not(value: str) -> None:
    with pytest.raises(ValueError):
        parse_skill_ir(raw_skill(instruction_provenance={"marker": value}))
    assert parse_skill_ir(raw_skill(instruction_provenance={"marker": f"x_{value}_y"}))


def test_projection_is_complete_order_sensitive_and_hashes_complete_self_excluding_projection() -> (
    None
):
    first = raw_skill()
    second = raw_skill()
    second["required_inputs"] = ["second", "first"]
    first_projection = skill_ir_projection(first)

    assert tuple(first_projection) == (
        "skill_id",
        "schema_version",
        "domain",
        "objective",
        "applicability",
        "required_inputs",
        "preconditions",
        "ordered_steps",
        "decision_hints",
        "verification_steps",
        "stop_conditions",
        "escalation_hints",
        "source_trace_ids",
        "instruction_provenance",
        "compiler_manifest_ref",
    )
    assert hash_skill_ir(first) == sha256_ref(first_projection)
    assert (
        hash_skill_ir(first)
        == "sha256:8d8854c9070ad9317dff1ea572945e6b13ec21282d47894addb7abecf647c461"
    )
    assert hash_skill_ir(first) != hash_skill_ir(second)
    first_projection["objective"] = "mutated"
    assert skill_ir_projection(first)["objective"] != "mutated"


def test_every_projected_field_and_every_array_order_affects_the_skill_hash() -> None:
    mutations: dict[str, object] = {
        "skill_id": "skill_access_changed",
        "schema_version": "1.1",
        "domain": "financial_adjustments",
        "objective": "Changed objective.",
        "applicability": ["Changed applicability."],
        "required_inputs": ["Changed input."],
        "preconditions": ["Changed precondition."],
        "ordered_steps": [
            {
                "step_id": "changed_step",
                "action_intent": "Changed intent.",
                "tool_name": "changed_tool",
                "argument_bindings": {"changed": True},
                "preconditions": ["Changed step precondition."],
                "optional": True,
                "evidence_refs": ["Changed evidence."],
            }
        ],
        "decision_hints": ["Changed decision."],
        "verification_steps": ["Changed verification."],
        "stop_conditions": ["Changed stop."],
        "escalation_hints": ["Changed escalation."],
        "source_trace_ids": ["trace_changed"],
        "instruction_provenance": {"changed": True},
        "compiler_manifest_ref": "sha256:" + "b" * 64,
    }
    baseline = hash_skill_ir(raw_skill())
    for field, changed in mutations.items():
        value = raw_skill(**{field: changed})
        if field == "schema_version":
            # The canonical version is intentionally fixed, so it is sensitive by rejection.
            with pytest.raises(ValueError):
                hash_skill_ir(value)
        else:
            assert hash_skill_ir(value) != baseline, field

    ordered = raw_skill(applicability=["first", "second"])
    reversed_order = raw_skill(applicability=["second", "first"])
    assert hash_skill_ir(ordered) != hash_skill_ir(reversed_order)


def test_models_reject_json_aliases_and_pydantic_forgery() -> None:
    class HostileMapping(dict[str, object]):
        def items(self) -> object:
            raise AssertionError("caller hook invoked")

    with pytest.raises(ValueError):
        parse_skill_ir(HostileMapping(raw_skill()))
    with pytest.raises(ValueError):
        TypeAdapter(SkillIR).validate_json(json.dumps(raw_skill()))
    with pytest.raises(ValueError):
        TypeAdapter(OrderedStep).validate_strings({})

    parsed = parse_skill_ir(raw_skill())
    parsed_again = TypeAdapter(SkillIR).validate_python(parsed)
    step_again = TypeAdapter(OrderedStep).validate_python(parsed.ordered_steps[0])
    assert parsed_again is not parsed and skill_ir_projection(parsed_again) == skill_ir_projection(
        parsed
    )
    assert step_again is not parsed.ordered_steps[0] and step_again == parsed.ordered_steps[0]
    with pytest.raises(ValueError):
        parsed.model_copy(update={"unknown": 1})
    copied = parsed.model_copy()
    assert copied is not parsed and copied == parsed
    forged = SkillIR.model_construct(**object.__getattribute__(parsed, "__dict__"))
    object.__setattr__(forged, "__pydantic_private__", {"forged": True})
    with pytest.raises(ValueError):
        parse_skill_ir(forged)

    forged_step = OrderedStep.model_construct(
        **object.__getattribute__(parsed.ordered_steps[0], "__dict__")
    )
    object.__getattribute__(forged_step, "__dict__")["argument_bindings"] = {"bad": object()}
    forged = SkillIR.model_construct(**object.__getattribute__(parsed, "__dict__"))
    object.__getattribute__(forged, "__dict__")["ordered_steps"] = (forged_step,)
    with pytest.raises(ValueError):
        parse_skill_ir(forged)


def test_legacy_pydantic_ingress_is_rejected_without_parsing_or_file_hooks(tmp_path: Path) -> None:
    raw = raw_skill()
    step = cast(dict[str, object], cast(list[object], raw["ordered_steps"])[0])

    for model, value in ((SkillIR, raw), (OrderedStep, step)):
        with pytest.raises(ValueError):
            model.parse_raw(json.dumps(value))
        with pytest.raises(ValueError):
            model.parse_raw(json.dumps(value).encode("utf-8"))
        with pytest.raises(ValueError):
            model.parse_file(tmp_path / "valid.json")
        assert model.parse_obj(value) == model.model_validate(value)
        assert model.validate(value) == model.model_validate(value)

    class HostilePath:
        def __fspath__(self) -> str:
            raise AssertionError("file hook invoked")

    with pytest.raises(ValueError):
        SkillIR.parse_file(HostilePath())


def test_exact_model_preflight_rejects_forged_cross_root_aliases_and_aggregate_budgets() -> None:
    parsed = parse_skill_ir(raw_skill())
    shared: dict[str, object] = {}
    object.__getattribute__(parsed, "__dict__")["instruction_provenance"] = shared
    object.__getattribute__(parsed.ordered_steps[0], "__dict__")["argument_bindings"] = shared
    with pytest.raises(ValueError):
        parse_skill_ir(parsed)
    with pytest.raises(ValueError):
        TypeAdapter(SkillIR).validate_python(parsed)

    parsed = parse_skill_ir(raw_skill())
    object.__getattribute__(parsed, "__dict__")["instruction_provenance"] = {"values": [0] * 60_000}
    object.__getattribute__(parsed.ordered_steps[0], "__dict__")["argument_bindings"] = {
        "values": [0] * 60_000
    }
    with pytest.raises(ValueError):
        parse_skill_ir(parsed)


def test_legacy_and_python_copy_paths_are_detached_and_strict() -> None:
    parsed = parse_skill_ir(raw_skill())
    for include, exclude in (({"objective"}, None), (None, {"objective"})):
        with pytest.raises(ValueError):
            parsed.copy(include=include, exclude=exclude)
    with pytest.raises(ValueError):
        parsed.copy(update={"unknown": 1})

    shallow = copy.copy(parsed)
    deep = copy.deepcopy(parsed)
    legacy = parsed.copy(update={"objective": "Changed."})
    assert shallow is not parsed and deep is not parsed and legacy.objective == "Changed."
    assert shallow.ordered_steps[0] is not parsed.ordered_steps[0]
    assert shallow.instruction_provenance is not parsed.instruction_provenance
    assert (
        shallow.ordered_steps[0].argument_bindings is not parsed.ordered_steps[0].argument_bindings
    )

    forged = SkillIR.model_construct(**object.__getattribute__(parsed, "__dict__"))
    object.__setattr__(forged, "__pydantic_extra__", {"forged": True})
    with pytest.raises(ValueError):
        copy.copy(forged)

    forged = parse_skill_ir(raw_skill())
    shared: dict[str, object] = {}
    object.__getattribute__(forged, "__dict__")["instruction_provenance"] = shared
    object.__getattribute__(forged.ordered_steps[0], "__dict__")["argument_bindings"] = shared
    for copier in (
        forged.model_copy,
        forged.copy,
        lambda: copy.copy(forged),
        lambda: copy.deepcopy(forged),
    ):
        with pytest.raises(ValueError):
            copier()


def test_oversized_text_domain_and_manifest_fail_before_deeper_validation() -> None:
    with pytest.raises(ValueError):
        parse_skill_ir(raw_skill(objective="x" * 4_097))
    with pytest.raises(ValueError):
        parse_skill_ir(raw_skill(domain="x" * 129))
    with pytest.raises(ValueError):
        parse_skill_ir(raw_skill(compiler_manifest_ref="x" * 72))


def test_hostile_nested_ingress_subclasses_and_invalid_pydantic_state_are_rejected() -> None:
    class HostileList(list[object]):
        def __iter__(self) -> object:
            raise AssertionError("caller hook invoked")

    class SkillSubclass(SkillIR):
        pass

    raw = raw_skill(instruction_provenance=HostileList())
    with pytest.raises(ValueError):
        parse_skill_ir(raw)
    parsed = parse_skill_ir(raw_skill())
    forged = SkillIR.model_construct(**object.__getattribute__(parsed, "__dict__"))
    object.__setattr__(forged, "__pydantic_fields_set__", {"skill_id"})
    with pytest.raises(ValueError):
        parse_skill_ir(forged)
    subclass = SkillSubclass.model_construct(**object.__getattribute__(parsed, "__dict__"))
    with pytest.raises(ValueError):
        parse_skill_ir(subclass)


@pytest.mark.parametrize(
    "provenance",
    [
        {"integer": 2**256},
        {"float": float("inf")},
        {"key" * 43: "value"},
        {"text": "x" * 4_097},
    ],
)
def test_json_budget_and_exact_value_limits_are_enforced(provenance: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_skill_ir(raw_skill(instruction_provenance=provenance))


def test_json_budgets_cycles_and_container_aliases_fail() -> None:
    nested: object = {}
    for _ in range(33):
        nested = {"next": nested}
    with pytest.raises(ValueError):
        parse_skill_ir(raw_skill(instruction_provenance=cast(dict[str, object], nested)))

    aliased: list[object] = []
    with pytest.raises(ValueError):
        parse_skill_ir(raw_skill(instruction_provenance={"left": aliased, "right": aliased}))

    cycle: dict[str, object] = {}
    cycle["cycle"] = cycle
    with pytest.raises(ValueError):
        parse_skill_ir(raw_skill(instruction_provenance=cycle))


def test_json_aggregate_budgets_and_cross_root_aliases_fail_before_field_validation() -> None:
    sixty_thousand = [0] * 60_000
    raw = raw_skill(instruction_provenance={"values": sixty_thousand})
    step = cast(dict[str, object], cast(list[object], raw["ordered_steps"])[0])
    step["argument_bindings"] = {"values": [0] * 60_000}
    with pytest.raises(ValueError):
        parse_skill_ir(raw)

    text_values = ["x" * 4_000] * 150
    raw = raw_skill(instruction_provenance={"values": text_values})
    step = cast(dict[str, object], cast(list[object], raw["ordered_steps"])[0])
    step["argument_bindings"] = {"values": ["x" * 4_000] * 150}
    with pytest.raises(ValueError):
        parse_skill_ir(raw)

    shared_top: list[object] = []
    raw = raw_skill(applicability=shared_top, required_inputs=shared_top)
    with pytest.raises(ValueError):
        parse_skill_ir(raw)

    shared_step: dict[str, object] = cast(
        dict[str, object], cast(list[object], raw_skill()["ordered_steps"])[0]
    )
    raw = raw_skill(ordered_steps=[shared_step, shared_step])
    with pytest.raises(ValueError):
        parse_skill_ir(raw)

    shared_json: dict[str, object] = {}
    raw = raw_skill(instruction_provenance=shared_json)
    step = cast(dict[str, object], cast(list[object], raw["ordered_steps"])[0])
    step["argument_bindings"] = shared_json
    with pytest.raises(ValueError):
        parse_skill_ir(raw)


def test_top_level_cardinality_rejects_before_untrusted_key_iteration() -> None:
    class ExplosiveKey(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("key equality was invoked")

        __hash__ = str.__hash__

    huge = {ExplosiveKey(f"extra_{index}"): None for index in range(20_000)}
    with pytest.raises(ValueError):
        parse_skill_ir(huge)


def test_exact_cardinality_subclassed_keys_fail_before_membership_or_equality_hooks() -> None:
    class ExplosiveKey(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("key equality was invoked")

        __hash__ = str.__hash__

    top_level = raw_skill()
    top_level[ExplosiveKey("skill_id")] = top_level.pop("skill_id")
    with pytest.raises(ValueError):
        parse_skill_ir(top_level)

    nested = raw_skill()
    step = cast(dict[str, object], cast(list[object], nested["ordered_steps"])[0])
    step[ExplosiveKey("step_id")] = step.pop("step_id")
    with pytest.raises(ValueError):
        parse_skill_ir(nested)


def test_model_copy_update_rejects_subclassed_key_before_membership_or_merge() -> None:
    class ExplosiveKey(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("key equality was invoked")

        __hash__ = str.__hash__

    parsed = parse_skill_ir(raw_skill())
    with pytest.raises(ValueError):
        parsed.model_copy(update={ExplosiveKey("objective"): "Changed."})
    assert parsed.model_copy(update={"objective": "Changed."}).objective == "Changed."


def test_skill_model_source_has_only_allowed_imports() -> None:
    import ast

    module = ast.parse(
        (
            Path(__file__).parents[3] / "src" / "shadowskillbench" / "skills" / "models.py"
        ).read_text()
    )
    imports = [node for node in ast.walk(module) if isinstance(node, (ast.Import, ast.ImportFrom))]
    allowed_from = {
        "__future__",
        "collections.abc",
        "typing",
        "pydantic",
        "pydantic.config",
        "pydantic_core",
        "shadowskillbench.core.hashing",
    }
    assert all(
        (isinstance(node, ast.ImportFrom) and node.module in allowed_from)
        or (
            isinstance(node, ast.Import)
            and all(alias.name in {"math", "re"} for alias in node.names)
        )
        for node in imports
    )
