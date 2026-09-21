from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref

FINITE_FLOATS = st.floats(allow_nan=False, allow_infinity=False, width=64)
JSON_VALUES = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), FINITE_FLOATS, st.text()),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(), children, max_size=5),
    ),
    max_leaves=20,
)


def reverse_mapping_insertions(value: object) -> object:
    if type(value) is list:
        return [reverse_mapping_insertions(item) for item in value]
    if type(value) is dict:
        items = reversed(list(value.items()))
        return {key: reverse_mapping_insertions(item) for key, item in items}
    return value


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(st.dictionaries(st.text(), JSON_VALUES, max_size=8))
def test_mapping_insertion_order_never_changes_canonical_hash(value: dict[str, object]) -> None:
    reordered = dict(reversed(list(value.items())))

    assert canonical_json_bytes(value) == canonical_json_bytes(reordered)
    assert sha256_ref(value) == sha256_ref(reordered)


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(JSON_VALUES)
def test_nested_mapping_insertion_order_never_changes_canonical_hash(value: object) -> None:
    reordered = reverse_mapping_insertions(value)

    assert canonical_json_bytes(value) == canonical_json_bytes(reordered)
    assert sha256_ref(value) == sha256_ref(reordered)


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(), min_size=2, max_size=8).filter(lambda items: items != items[::-1]))
def test_distinct_array_orders_never_share_canonical_hash(items: list[int]) -> None:
    assert sha256_ref(items) != sha256_ref(items[::-1])
