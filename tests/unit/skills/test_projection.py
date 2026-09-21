from __future__ import annotations

import ast
import copy
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.skills import (
    CompilerInput,
    compiler_input_projection,
    compiler_view,
    hash_compiler_input,
)
from shadowskillbench.traces.bundles import (
    DemonstrationBundle,
    GeneratedBundle,
    generate_bundle,
)
from shadowskillbench.traces.models import ActionTrace, action_trace_projection, hash_action_trace


@pytest.fixture(params=["access_provisioning", "financial_adjustments"])
def domain(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


@pytest.fixture(
    params=[Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1")]
)
def ratio(request: pytest.FixtureRequest) -> Decimal:
    return cast(Decimal, request.param)


def test_compiler_view_is_complete_allowlisted_and_ordered(domain: str, ratio: Decimal) -> None:
    generated = generate_bundle(domain, ratio, 12, 220 + int(ratio * 100))
    view = compiler_view(generated.bundle)
    projection = compiler_input_projection(view)

    assert tuple(view.__dict__) == ("projection_profile", "domain", "traces")
    assert view.projection_profile == "SSB-COMPILER-VIEW1"
    assert view.domain == domain
    assert len(view.traces) == 12
    assert tuple(projection) == ("projection_profile", "domain", "traces")
    assert projection["traces"] == [
        {
            "trace_id": trace.trace_id,
            "domain": trace.domain,
            "task_template_id": trace.task_template_id,
            "worker_role": trace.worker_role,
            "world_hash": trace.world_hash,
            "events": [
                {
                    "event_id": event.event_id,
                    "index": event.index,
                    "kind": event.kind,
                    "payload": event.payload,
                }
                for event in trace.events
            ],
            "terminal_state_hash": trace.terminal_state_hash,
            "local_task_outcome": "completed",
        }
        for trace in generated.bundle.traces
    ]
    forbidden = {
        "schema_version",
        "worker_policy_id",
        "narration_mode",
        "hidden_benchmark_metadata_ref",
        "bundle_id",
        "contamination_ratio",
        "ordered_trace_hashes",
        "hidden_class_manifest_hash",
    }
    assert not forbidden.intersection(str(projection))
    assert hash_compiler_input(view) == sha256_ref(projection)


def test_compiler_view_is_detached_and_rejects_generated_and_raw_inputs() -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.5"), 12, 99)
    view = compiler_view(generated.bundle)
    source_payload = generated.bundle.traces[0].events[0].payload
    output_payload = view.traces[0].events[0].payload
    assert output_payload == source_payload and output_payload is not source_payload
    projection = compiler_input_projection(view)
    cast(list[object], projection["traces"])[0] = {}
    assert len(compiler_input_projection(view)["traces"]) == 12
    with pytest.raises(ValueError):
        compiler_view(generated)
    with pytest.raises(ValueError):
        compiler_view(cast(object, {"bundle": generated.bundle}))
    assert type(generated) is GeneratedBundle


def test_exact_bundle_and_compiler_input_classes_are_required() -> None:
    class BundleSubclass(DemonstrationBundle):
        pass

    class InputSubclass(CompilerInput):
        pass

    bundle = generate_bundle("access_provisioning", Decimal("0"), 12, 103).bundle
    bundle_subclass = BundleSubclass.model_construct(
        source_manifest=bundle.source_manifest, traces=bundle.traces
    )
    with pytest.raises(ValueError):
        compiler_view(cast(DemonstrationBundle, bundle_subclass))
    view = compiler_view(bundle)
    input_subclass = InputSubclass.model_construct(
        projection_profile=view.projection_profile, domain=view.domain, traces=view.traces
    )
    with pytest.raises(ValueError):
        compiler_input_projection(cast(CompilerInput, input_subclass))


@pytest.mark.parametrize(
    "key",
    [
        "worker_policy_id",
        "semantic_class",
        "contamination_ratio",
        "hidden_benchmark_metadata_ref",
        "source_manifest",
        "hidden_class_manifest",
        "bundle_id",
        "ordered_trace_hashes",
        "hidden_class_manifest_hash",
        "class_slot_order",
        "final_trace_order",
        "bindings",
        "source_slot",
        "fixture_seed",
        "fixture_variant",
        "trace_hash",
        "compliance_label",
        "policy",
        "policy_id",
        "policy_text",
        "authority",
        "authority_record",
        "authority_decision",
        "heldout",
        "held_out",
        "held_out_case",
        "heldout_case",
        "verifier",
        "verifier_outcome",
        "final_verifier",
        "final_verifier_outcome",
        "cup",
        "cup_outcome",
        "scorer",
        "score",
        "scorer_outcome",
    ],
)
def test_recursive_key_canaries_are_rejected(key: str) -> None:
    payload = {"visible": {key: "canary"}}
    raw = _raw_input(payload)
    with pytest.raises(ValueError):
        CompilerInput.model_validate(raw)


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
def test_recursive_value_canaries_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        CompilerInput.model_validate(_raw_input({"visible": [value]}))


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
@pytest.mark.parametrize("field", ["task_template_id", "worker_role"])
def test_retained_scalar_values_are_leakage_guarded(field: str, value: str) -> None:
    raw = _raw_input({"visible": "safe"})
    cast(dict[str, object], cast(list[object], raw["traces"])[0])[field] = value
    with pytest.raises(ValueError):
        CompilerInput.model_validate(raw)


@pytest.mark.parametrize("field", ["task_template_id", "worker_role"])
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
def test_compiler_view_applies_scalar_guard_at_the_real_source_boundary(
    field: str, value: str
) -> None:
    bundle = _d021_valid_bundle_with_retained_scalar(field, value)
    assert DemonstrationBundle.model_validate(bundle) == bundle
    with pytest.raises(ValueError, match="forbidden leakage value"):
        compiler_view(bundle)


def test_missing_pydantic_state_and_raw_mapping_subclasses_are_rejected() -> None:
    class Raw(dict[str, object]):
        pass

    with pytest.raises(ValueError):
        CompilerInput.model_validate(Raw(_raw_input({"visible": "safe"})))
    view = compiler_view(generate_bundle("access_provisioning", Decimal("0"), 12, 914).bundle)
    object.__delattr__(view, "__pydantic_private__")
    with pytest.raises(ValueError):
        compiler_input_projection(view)


def test_output_models_close_legacy_copy_and_forged_alias_paths() -> None:
    view = compiler_view(generate_bundle("financial_adjustments", Decimal("0"), 12, 31).bundle)
    with pytest.raises(ValueError):
        CompilerInput.model_validate_json("{}")
    with pytest.raises(ValueError):
        CompilerInput.parse_file("anything.json")
    copied = view.model_copy()
    assert copied is not view and copied == view
    shared: dict[str, object] = {}
    object.__getattribute__(view, "__dict__")["traces"] = (view.traces[0],) * 12
    with pytest.raises(ValueError):
        view.model_copy()
    object.__getattribute__(copied, "__dict__")["traces"] = tuple(
        copy.copy(trace) for trace in copied.traces
    )
    object.__getattribute__(copied.traces[0].events[0], "__dict__")["payload"] = shared
    object.__getattribute__(copied.traces[0].events[1], "__dict__")["payload"] = shared
    with pytest.raises(ValueError):
        compiler_input_projection(copied)


def test_output_and_source_field_sets_reject_exact_string_subclasses() -> None:
    class S(str):
        pass

    view = compiler_view(generate_bundle("access_provisioning", Decimal("0"), 12, 911).bundle)
    object.__getattribute__(view, "__pydantic_fields_set__").clear()
    object.__getattribute__(view, "__pydantic_fields_set__").update(
        S(field) for field in ("projection_profile", "domain", "traces")
    )
    with pytest.raises(ValueError):
        compiler_input_projection(view)

    bundle = generate_bundle("financial_adjustments", Decimal("1"), 12, 912).bundle
    object.__getattribute__(bundle, "__pydantic_fields_set__").clear()
    object.__getattribute__(bundle, "__pydantic_fields_set__").update(
        S(field) for field in ("source_manifest", "traces")
    )
    with pytest.raises(ValueError):
        compiler_view(bundle)


def test_broken_d021_manifest_link_is_revalidated_before_projection() -> None:
    bundle = generate_bundle("financial_adjustments", Decimal("0.25"), 12, 913).bundle
    manifest = bundle.source_manifest
    hashes = list(manifest.ordered_trace_hashes)
    hashes[0] = "sha256:" + "0" * 64
    object.__getattribute__(manifest, "__dict__")["ordered_trace_hashes"] = tuple(hashes)
    with pytest.raises(ValueError):
        compiler_view(bundle)


def test_maximum_trace_and_derived_event_identifier_lengths_are_admitted() -> None:
    raw = _raw_input({"visible": "safe"})
    trace = cast(dict[str, object], cast(list[object], raw["traces"])[0])
    trace_id = "trace_" + "a" * 122
    trace["trace_id"] = trace_id
    events = cast(list[dict[str, object]], trace["events"])
    for event in events:
        event["event_id"] = f"event_{trace_id}_{event['index']:06d}"
    parsed = CompilerInput.model_validate(raw)
    assert len(parsed.traces[0].trace_id) == 128
    assert len(parsed.traces[0].events[0].event_id) == 141


def test_per_trace_payload_budgets_do_not_accumulate_across_the_bundle() -> None:
    raw = _raw_input({"visible": "safe"})
    traces = cast(list[dict[str, object]], raw["traces"])
    traces[0]["events"] = cast(list[dict[str, object]], traces[0]["events"])
    traces[1]["events"] = cast(list[dict[str, object]], traces[1]["events"])
    traces[0]["events"][0]["payload"] = {"items": [0] * 60_000}
    traces[1]["events"][0]["payload"] = {"items": [0] * 60_000}
    assert len(CompilerInput.model_validate(raw).traces) == 12


def test_cross_trace_payload_container_alias_is_rejected() -> None:
    raw = _raw_input({"visible": "safe"})
    traces = cast(list[dict[str, object]], raw["traces"])
    shared: dict[str, object] = {"visible": "shared"}
    for trace in traces[:2]:
        events = cast(list[dict[str, object]], trace["events"])
        events[0]["payload"] = shared
    with pytest.raises(ValueError, match="aliased"):
        CompilerInput.model_validate(raw)


def test_projection_import_boundary_is_compiler_safe() -> None:
    source = (
        Path(__file__).parents[3] / "src" / "shadowskillbench" / "skills" / "projection.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert {
        name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names
    } == {"math", "re"}
    imports = {
        node.module: {name.name for name in node.names}
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imports == {
        "__future__": {"annotations"},
        "collections.abc": {"Mapping"},
        "typing": {"Any", "ClassVar", "Literal", "Self", "cast"},
        "pydantic": {
            "BaseModel",
            "ConfigDict",
            "ValidationInfo",
            "field_validator",
            "model_validator",
        },
        "pydantic.config": {"ExtraValues"},
        "pydantic_core": {"core_schema"},
        "shadowskillbench.core.hashing": {"canonical_json_bytes", "sha256_ref"},
        "shadowskillbench.traces.bundles": {"DemonstrationBundle", "SourceBundleManifest"},
        "shadowskillbench.traces.models": {"ActionTrace", "JsonObject", "JsonValue", "TraceEvent"},
    }
    assert not {
        call.func.attr
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"model_dump", "model_dump_json", "dict", "json"}
    }


def _raw_input(payload: dict[str, object]) -> dict[str, object]:
    traces: list[dict[str, object]] = []
    for number in range(12):
        trace_id = f"trace_unit_{number}"
        events: list[dict[str, object]] = [
            {
                "event_id": f"event_{trace_id}_000000",
                "index": 0,
                "kind": "observation",
                "payload": copy.deepcopy(payload),
            }
        ]
        for index, kind in enumerate(("action", "tool_result", "state_delta"), start=1):
            events.append(
                {
                    "event_id": f"event_{trace_id}_{index:06d}",
                    "index": index,
                    "kind": kind,
                    "payload": {"visible": index},
                }
            )
        traces.append(
            {
                "trace_id": trace_id,
                "domain": "access_provisioning",
                "task_template_id": "task_unit",
                "worker_role": "worker",
                "world_hash": "sha256:" + "a" * 64,
                "events": events,
                "terminal_state_hash": "sha256:" + "b" * 64,
                "local_task_outcome": "completed",
            }
        )
    return {
        "projection_profile": "SSB-COMPILER-VIEW1",
        "domain": "access_provisioning",
        "traces": traces,
    }


def _d021_valid_bundle_with_retained_scalar(field: str, value: str) -> DemonstrationBundle:
    original = generate_bundle("access_provisioning", Decimal("0.5"), 12, 781).bundle
    trace_raw = action_trace_projection(original.traces[0])
    trace_raw[field] = value
    altered = ActionTrace.model_validate(trace_raw)
    traces = (altered, *original.traces[1:])
    manifest = original.source_manifest.model_copy(
        update={"ordered_trace_hashes": tuple(hash_action_trace(trace) for trace in traces)}
    )
    return DemonstrationBundle.model_validate({"source_manifest": manifest, "traces": traces})
