from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import pytest
from pydantic import TypeAdapter

from shadowskillbench.traces import BundleDomain, SemanticClass
from shadowskillbench.traces.bundles import (
    GeneratedBundle,
    HiddenClassManifest,
    SourceBundleManifest,
    generate_bundle,
    hash_hidden_class_manifest,
    hash_source_bundle_manifest,
    hidden_class_manifest_projection,
    source_bundle_manifest_projection,
)
from shadowskillbench.traces.models import TraceEvent, hash_action_trace


@pytest.mark.parametrize("domain", ["access_provisioning", "financial_adjustments"])
@pytest.mark.parametrize(
    "ratio, workarounds", [("0", 0), ("0.25", 3), ("0.5", 6), ("0.75", 9), ("1", 12)]
)
def test_generation_has_exact_fixed_composition(domain: str, ratio: str, workarounds: int) -> None:
    generated = generate_bundle(domain, Decimal(ratio), 12, -42)  # type: ignore[arg-type]
    source, hidden = generated.bundle.source_manifest, generated.hidden_class_manifest

    assert len(generated.bundle.traces) == len(hidden.bindings) == 12
    assert source.ordered_trace_hashes == tuple(
        hash_action_trace(trace) for trace in generated.bundle.traces
    )
    assert (
        sum(binding.semantic_class == "prohibited_workaround" for binding in hidden.bindings)
        == workarounds
    )
    assert [binding.source_slot for binding in hidden.bindings] == list(hidden.final_trace_order)
    assert hidden_class_manifest_projection(hidden)["contamination_ratio"] == ratio
    assert source_bundle_manifest_projection(source)["contamination_ratio"] == ratio
    assert source.hidden_class_manifest_hash == hash_hidden_class_manifest(hidden)
    assert hash_source_bundle_manifest(source).startswith("sha256:")
    for trace in generated.bundle.traces:
        assert trace.narration_mode == "none"
        assert trace.local_task_outcome == "completed"
        assert trace.hidden_benchmark_metadata_ref is None
        assert any(event.kind == "action" for event in trace.events)


def test_paired_slots_are_ratio_isolated_and_deterministic() -> None:
    low = generate_bundle("access_provisioning", Decimal("0"), 12, 101)
    high = generate_bundle("access_provisioning", Decimal("1"), 12, 101)

    assert low == generate_bundle("access_provisioning", Decimal("0"), 12, 101)
    assert (
        low.hidden_class_manifest.final_trace_order == high.hidden_class_manifest.final_trace_order
    )
    assert [trace.trace_id for trace in low.bundle.traces] == [
        trace.trace_id for trace in high.bundle.traces
    ]
    assert [binding.fixture_seed for binding in low.hidden_class_manifest.bindings] == [
        binding.fixture_seed for binding in high.hidden_class_manifest.bindings
    ]
    assert [trace.worker_policy_id for trace in low.bundle.traces] != [
        trace.worker_policy_id for trace in high.bundle.traces
    ]


def test_independent_seed_zero_access_vector() -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.5"), 12, 0)

    assert generated.bundle.source_manifest.bundle_id == "bundle_3aea710e07ed4f64d0b5638578ded6fa"
    assert generated.hidden_class_manifest.class_slot_order == (
        7,
        4,
        0,
        6,
        1,
        11,
        5,
        8,
        3,
        9,
        2,
        10,
    )
    assert generated.hidden_class_manifest.final_trace_order == (
        7,
        2,
        3,
        5,
        9,
        8,
        10,
        1,
        6,
        0,
        11,
        4,
    )
    assert generated.hidden_class_manifest.bindings[0].fixture_seed == -3_604_917_875_252_592_624


def test_independent_finance_and_signed_boundary_vectors() -> None:
    finance = generate_bundle("financial_adjustments", Decimal("0.5"), 12, 0)
    minimum = generate_bundle("access_provisioning", Decimal("0.5"), 12, -(2**63))
    access_maximum = generate_bundle("access_provisioning", Decimal("0.5"), 12, 2**63 - 1)
    finance_minimum = generate_bundle("financial_adjustments", Decimal("0.5"), 12, -(2**63))
    finance_maximum = generate_bundle("financial_adjustments", Decimal("0.5"), 12, 2**63 - 1)

    assert finance.bundle.source_manifest.bundle_id == "bundle_9ee58b3b2849050c494a648336e0abb2"
    assert finance.hidden_class_manifest.class_slot_order == (9, 2, 3, 5, 6, 1, 10, 11, 0, 4, 8, 7)
    assert finance.hidden_class_manifest.final_trace_order == (2, 1, 3, 9, 5, 0, 7, 6, 10, 11, 8, 4)
    assert minimum.bundle.source_manifest.bundle_id == "bundle_2a00e3e5c4a2932f7c21422887e96a20"
    assert (
        access_maximum.bundle.source_manifest.bundle_id == "bundle_ce79817f8beb408a39b274ca4a594d12"
    )
    assert (
        finance_minimum.bundle.source_manifest.bundle_id
        == "bundle_e825b97e6bcfb34861962efd3b06066b"
    )
    assert (
        finance_maximum.bundle.source_manifest.bundle_id
        == "bundle_3feea750594ce8721804c25fd358a1c3"
    )


@pytest.mark.parametrize(
    "domain,ratio,count,seed",
    [
        ("wrong", Decimal("0"), 12, 0),
        ("access_provisioning", Decimal("0.1"), 12, 0),
        ("access_provisioning", Decimal("0"), True, 0),
        ("access_provisioning", Decimal("0"), 12, 2**63),
        ("access_provisioning", Decimal("0"), 12, True),
    ],
)
def test_generation_rejects_hostile_inputs(
    domain: object, ratio: object, count: object, seed: object
) -> None:
    with pytest.raises(ValueError):
        generate_bundle(domain, ratio, count, seed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ratio",
    [0, 0.5, "0.5", Decimal("NaN"), Decimal("Infinity"), Decimal("1.0000000000000000")],
)
def test_generation_rejects_noncanonical_or_non_decimal_ratio_ingress(ratio: object) -> None:
    with pytest.raises(ValueError):
        generate_bundle("access_provisioning", ratio, 12, 0)  # type: ignore[arg-type]


def test_equivalent_decimal_is_canonical_and_public_aliases_are_exported() -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.250"), 12, 5)

    assert generated.bundle.source_manifest.contamination_ratio == Decimal("0.25")
    assert BundleDomain is not None
    assert SemanticClass is not None


def test_all_bundle_models_reject_json_strings_subclasses_and_model_copy_bypass() -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.25"), 12, 5)
    values = (
        generated.hidden_class_manifest.bindings[0],
        generated.hidden_class_manifest,
        generated.bundle.source_manifest,
        generated.bundle,
        generated,
    )
    for value in values:
        model = type(value)
        with pytest.raises(ValueError):
            TypeAdapter(model).validate_json("{}")
        with pytest.raises(ValueError):
            TypeAdapter(model).validate_strings({})
        with pytest.raises(ValueError):
            value.model_copy(update={"unknown": 1})

    class BadBinding(type(values[0])):
        pass

    with pytest.raises(ValueError):
        BadBinding.model_validate(values[0])


def test_public_boundaries_reject_fields_set_and_hostile_key_hooks() -> None:
    generated = generate_bundle("financial_adjustments", Decimal("0.5"), 12, 9)
    source = generated.bundle.source_manifest
    object.__setattr__(source, "__pydantic_fields_set__", {"bundle_id"})
    for boundary in (source_bundle_manifest_projection, hash_source_bundle_manifest):
        with pytest.raises(ValueError):
            boundary(source)

    class HostileString(str):
        def __hash__(self) -> int:
            return str.__hash__(self)

        def __eq__(self, other: object) -> bool:
            raise AssertionError("hostile equality invoked")

    raw = dict(object.__getattribute__(generated.hidden_class_manifest, "__dict__"))
    raw[HostileString("bundle_id")] = raw.pop("bundle_id")
    with pytest.raises(ValueError):
        HiddenClassManifest.model_validate(raw)


def test_public_boundaries_reject_constructed_private_and_tampered_values() -> None:
    generated = generate_bundle("financial_adjustments", Decimal("0.5"), 12, 9)
    hidden = generated.hidden_class_manifest
    source = generated.bundle.source_manifest
    object.__setattr__(hidden, "__pydantic_private__", {"evil": 1})
    with pytest.raises(ValueError):
        hidden_class_manifest_projection(hidden)
    with pytest.raises(ValueError):
        GeneratedBundle.model_validate(generated)

    forged = SourceBundleManifest.model_construct(**source.__dict__)
    object.__getattribute__(forged, "__dict__")["bundle_id"] = "bundle_forged"
    with pytest.raises(ValueError):
        source_bundle_manifest_projection(forged)


def test_generated_revalidation_rejects_trace_and_cross_link_forgery() -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.25"), 12, 17)
    raw_hidden = object.__getattribute__(generated.hidden_class_manifest, "__dict__")
    raw_hidden["bindings"] = tuple(reversed(raw_hidden["bindings"]))
    with pytest.raises(ValueError):
        GeneratedBundle.model_validate(generated)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: object.__getattribute__(value.bundle, "__dict__").update(
            {"traces": tuple(reversed(value.bundle.traces))}
        ),
        lambda value: object.__getattribute__(value.bundle, "__dict__").update(
            {"traces": (value.bundle.traces[0],) * 12}
        ),
        lambda value: object.__getattribute__(value.hidden_class_manifest, "__dict__").update(
            {"class_slot_order": tuple(reversed(value.hidden_class_manifest.class_slot_order))}
        ),
        lambda value: object.__getattribute__(
            value.hidden_class_manifest.bindings[0], "__dict__"
        ).update({"fixture_seed": 0}),
        lambda value: object.__getattribute__(
            value.hidden_class_manifest.bindings[0], "__dict__"
        ).update({"fixture_variant": "restricted_no_approval"}),
        lambda value: object.__getattribute__(
            value.hidden_class_manifest.bindings[0], "__dict__"
        ).update({"worker_policy_id": "finance_compliant_v1"}),
        lambda value: object.__getattribute__(value.bundle.traces[0], "__dict__").update(
            {"hidden_benchmark_metadata_ref": "sha256:" + "0" * 64}
        ),
    ],
)
def test_generated_bundle_tamper_matrix_rejects_custody_breaks(
    mutate: Callable[[GeneratedBundle], None],
) -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.5"), 12, 13)
    mutate(generated)
    with pytest.raises(ValueError):
        GeneratedBundle.model_validate(generated)


def _model_data(value: object) -> dict[str, object]:
    return object.__getattribute__(value, "__dict__")


def _wrong_binding_semantic(value: GeneratedBundle) -> None:
    binding = value.hidden_class_manifest.bindings[0]
    semantic = (
        "compliant"
        if binding.semantic_class == "prohibited_workaround"
        else "prohibited_workaround"
    )
    worker = (
        "access_workaround_v1" if semantic == "prohibited_workaround" else "access_compliant_v1"
    )
    _model_data(binding).update(
        {
            "semantic_class": semantic,
            "worker_policy_id": worker,
        }
    )


def _event_subclass(value: GeneratedBundle) -> None:
    event = value.bundle.traces[0].events[0]

    class ForgedEvent(TraceEvent):
        pass

    forged = ForgedEvent.model_construct(**_model_data(event))
    _model_data(value.bundle.traces[0])["events"] = (forged,) + value.bundle.traces[0].events[1:]


def _constructed_event(value: GeneratedBundle) -> None:
    event = value.bundle.traces[0].events[0]
    forged = TraceEvent.model_construct(**_model_data(event))
    _model_data(forged)["payload"] = object()
    _model_data(value.bundle.traces[0])["events"] = (forged,) + value.bundle.traces[0].events[1:]


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            lambda value: _model_data(value.bundle.traces[0]).update(
                {"events": (value.bundle.traces[0].events[0],)}
            ),
            id="zero-action-trace",
        ),
        pytest.param(
            lambda value: _model_data(value.bundle.traces[0]).update({"narration_mode": "neutral"}),
            id="narration-inconsistency",
        ),
        pytest.param(
            lambda value: _model_data(value.bundle.traces[0]).update(
                {"local_task_outcome": "failed"}
            ),
            id="non-completed-outcome",
        ),
        pytest.param(
            lambda value: object.__setattr__(
                value.bundle.traces[0].events[0], "__pydantic_private__", {"forged": True}
            ),
            id="nested-event-private-state",
        ),
        pytest.param(
            lambda value: object.__setattr__(
                value.bundle.traces[0].events[0], "__pydantic_extra__", {"forged": True}
            ),
            id="nested-event-extra-state",
        ),
        pytest.param(
            lambda value: object.__setattr__(
                value.bundle.traces[0].events[0], "__pydantic_fields_set__", {"event_id"}
            ),
            id="nested-event-fields-set",
        ),
        pytest.param(_event_subclass, id="event-subclass"),
        pytest.param(_constructed_event, id="event-model-construct"),
        pytest.param(
            lambda value: _model_data(value.bundle.traces[0]).update(
                {"domain": "financial_adjustments"}
            ),
            id="cross-domain-trace",
        ),
        pytest.param(
            lambda value: _model_data(value.bundle.source_manifest).update(
                {
                    "ordered_trace_hashes": (value.bundle.source_manifest.ordered_trace_hashes[0],)
                    * 12
                }
            ),
            id="duplicate-source-hashes",
        ),
        pytest.param(
            lambda value: _model_data(value.bundle.source_manifest).update(
                {
                    "ordered_trace_hashes": tuple(
                        reversed(value.bundle.source_manifest.ordered_trace_hashes)
                    )
                }
            ),
            id="reordered-source-hashes",
        ),
        pytest.param(
            lambda value: _model_data(value.hidden_class_manifest).update(
                {
                    "final_trace_order": tuple(
                        reversed(value.hidden_class_manifest.final_trace_order)
                    )
                }
            ),
            id="wrong-final-trace-order",
        ),
        pytest.param(
            lambda value: _model_data(value.hidden_class_manifest.bindings[0]).update(
                {"trace_id": "trace_" + "0" * 32}
            ),
            id="binding-trace-id",
        ),
        pytest.param(
            lambda value: _model_data(value.hidden_class_manifest.bindings[0]).update(
                {"trace_hash": "sha256:" + "0" * 64}
            ),
            id="binding-trace-hash",
        ),
        pytest.param(_wrong_binding_semantic, id="binding-semantic"),
        pytest.param(
            lambda value: _model_data(value.bundle.source_manifest).update(
                {"hidden_class_manifest_hash": "sha256:" + "0" * 64}
            ),
            id="source-hidden-cross-link",
        ),
    ],
)
def test_generated_bundle_boundary_rejects_task_19_local_trace_forgery(
    tamper: Callable[[GeneratedBundle], None],
) -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.5"), 12, 19)

    tamper(generated)

    with pytest.raises(ValueError):
        GeneratedBundle.model_validate(generated)


@pytest.mark.parametrize(
    "boundary,is_hidden",
    [
        (hidden_class_manifest_projection, True),
        (hash_hidden_class_manifest, True),
        (source_bundle_manifest_projection, False),
        (hash_source_bundle_manifest, False),
    ],
)
def test_every_public_manifest_projection_hash_rejects_forged_private_state(
    boundary: Callable[[object], object], is_hidden: bool
) -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.5"), 12, 14)
    target = generated.hidden_class_manifest if is_hidden else generated.bundle.source_manifest
    object.__setattr__(target, "__pydantic_private__", {"forged": True})
    with pytest.raises(ValueError):
        boundary(target)


def test_order_rejects_hostile_integer_subclass_before_hashing() -> None:
    generated = generate_bundle("access_provisioning", Decimal("0.5"), 12, 15)
    raw = dict(object.__getattribute__(generated.hidden_class_manifest, "__dict__"))

    class HostileInt(int):
        def __hash__(self) -> int:
            raise AssertionError("hostile integer hash invoked")

    raw["class_slot_order"] = (HostileInt(0),) + tuple(range(1, 12))
    with pytest.raises(ValueError):
        HiddenClassManifest.model_validate(raw)
