from __future__ import annotations

import re
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from shadowskillbench.core import CANONICALIZATION_PROFILE
from shadowskillbench.core.hashing import CanonicalizationError, canonical_json_bytes, sha256_ref


class ExamplePayload(BaseModel):
    labels: list[str]
    active: bool


class AliasedPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    display_name: str = Field(alias="displayName")
    optional: str | None = None


def test_known_pinned_cpython_vectors() -> None:
    assert CANONICALIZATION_PROFILE == "SSB-CJ1"
    assert canonical_json_bytes(None) == b"null"
    assert canonical_json_bytes({}) == b"{}"
    assert canonical_json_bytes({"a": 1}) == b'{"a":1}'
    assert canonical_json_bytes([1, 2]) == b"[1,2]"
    assert canonical_json_bytes(1e-6) == b"1e-06"
    assert canonical_json_bytes(1e-7) == b"1e-07"
    assert canonical_json_bytes(5e-324) == b"5e-324"
    assert sha256_ref(None) == (
        "sha256:74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b"
    )


def test_canonical_json_is_compact_sorted_and_utf8() -> None:
    value = {"z": "caf\u00e9", "a": [True, None]}

    assert canonical_json_bytes(value) == b'{"a":[true,null],"z":"caf\xc3\xa9"}'
    assert sha256_ref(value) == sha256_ref({"a": [True, None], "z": "caf\u00e9"})


def test_array_order_changes_hash() -> None:
    assert sha256_ref(["first", "second"]) != sha256_ref(["second", "first"])


def test_floats_are_finite_deterministic_and_normalize_negative_zero() -> None:
    assert canonical_json_bytes(1.25) == b"1.25"
    assert canonical_json_bytes(-0.0) == b"0.0"
    assert canonical_json_bytes(-0.0) == canonical_json_bytes(0.0)
    assert sha256_ref(-0.0) == sha256_ref(0.0)


def test_boolean_and_integer_have_distinct_hashes() -> None:
    assert sha256_ref(True) != sha256_ref(1)
    assert sha256_ref(False) != sha256_ref(0)


def test_pydantic_models_use_json_mode_dump() -> None:
    model = ExamplePayload(labels=["alpha", "beta"], active=True)

    assert canonical_json_bytes(model) == canonical_json_bytes(
        {"labels": ["alpha", "beta"], "active": True}
    )


def test_pydantic_models_use_the_documented_json_projection_options() -> None:
    model = AliasedPayload(display_name="visible")
    projection = model.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=False,
        exclude_unset=False,
        exclude_defaults=False,
        round_trip=False,
    )

    assert projection == {"displayName": "visible", "optional": None}
    assert canonical_json_bytes(model) == canonical_json_bytes(projection)


def test_unicode_key_order_and_normalization_are_pinned() -> None:
    value = {"\U00010000": "non-bmp", "\ue000": "bmp"}

    assert canonical_json_bytes(value) == b'{"\xee\x80\x80":"bmp","\xf0\x90\x80\x80":"non-bmp"}'
    assert sha256_ref("\u00e9") != sha256_ref("e\u0301")


def test_lone_surrogates_fail_with_a_typed_canonicalization_error() -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes("\ud800")


@pytest.mark.parametrize("container_type", [list, dict])
def test_self_referential_containers_fail_with_a_typed_canonicalization_error(
    container_type: type[list[object]] | type[dict[str, object]],
) -> None:
    value: list[object] | dict[str, object] = container_type()
    if type(value) is list:
        value.append(value)
    else:
        value["self"] = value

    with pytest.raises(CanonicalizationError, match="cyclic"):
        canonical_json_bytes(value)


def test_acyclic_repeated_references_remain_valid() -> None:
    shared = {"values": [1, 2]}
    value = {"left": shared, "right": shared}

    assert canonical_json_bytes(value) == b'{"left":{"values":[1,2]},"right":{"values":[1,2]}}'


@pytest.mark.parametrize(
    "value",
    [
        b"bytes are not JSON",
        {"set": {"not", "JSON"}},
        {"decimal": Decimal("1.0")},
        {"nested": {"bytes": b"not JSON"}},
        float("nan"),
        float("inf"),
        -float("inf"),
        {1: "non-string key"},
        {"tuple": ("not", "a", "JSON array")},
        object(),
    ],
)
def test_noncanonical_or_unsupported_values_are_rejected(value: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes(value)


def test_sha256_reference_has_exact_lowercase_format() -> None:
    reference = sha256_ref({"value": 1})

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", reference)
