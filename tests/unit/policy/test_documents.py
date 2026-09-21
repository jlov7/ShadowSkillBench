from __future__ import annotations

import json
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

from shadowskillbench.policy.documents import (
    PolicyClause,
    RenderedPolicy,
    render_matched_handbooks,
    render_policy_card,
)


def clause(
    clause_id: str,
    *,
    domain: Literal["access_provisioning", "financial_adjustments"] = "access_provisioning",
    heading: str | None = None,
    body: str | None = None,
) -> PolicyClause:
    return PolicyClause(
        clause_id=clause_id,
        domain=domain,
        heading=heading or f"Heading {clause_id}",
        body=body or f"Body {clause_id}.",
    )


def test_card_preserves_final_body_lf_and_known_profile_vectors() -> None:
    target = clause("target_1", heading="Access Rule", body="Require approval.\n")

    card = render_policy_card(target)

    assert card.rendered_text == "## Access Rule\nRequire approval.\n"
    assert card.rendered_text.encode("utf-8").hex() == (
        "2323204163636573732052756c650a5265717569726520617070726f76616c2e0a"
    )
    assert card.section_hashes == (
        "sha256:342cc0a4ca16cd4d05b20164f144af3441d46c313846f484c51dfa834757d07c",
    )
    assert card.target_section_hash == card.section_hashes[0]
    assert card.content_multiset_hash == (
        "sha256:016c682b3b1c5884f3d14ff0a636be7fc3ae57f7cf16c24afae925e188a42f3e"
    )
    assert card.rendered_hash == (
        "sha256:74fa3c43c697cf29e4a3d632ab3ba5eb17795eeb09fe1cf1fbe8d60e12db09b9"
    )
    assert card.token_count == 7


def test_ssb_st1_counts_ascii_runs_and_unicode_code_points() -> None:
    card = render_policy_card(clause("target", heading="A_1", body="é\tX-Y"))

    assert card.rendered_text == "## A_1\né\tX-Y"
    assert card.token_count == 7


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clause_id", "bad/id"),
        ("domain", "other"),
        ("heading", " \t\n"),
        ("heading", "one\ntwo"),
        ("body", "\u0000"),
        ("body", "\ud800"),
    ],
)
def test_clause_rejects_invalid_or_rewritten_text(field: str, value: str) -> None:
    values: dict[str, str] = {
        "clause_id": "target",
        "domain": "access_provisioning",
        "heading": "Heading",
        "body": "Body",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        PolicyClause.model_validate(values)


def test_rendered_policy_recomputes_all_derived_fields_and_model_copy() -> None:
    card = render_policy_card(clause("target"))
    payload = card.model_dump()
    payload["rendered_hash"] = "sha256:" + "0" * 64

    with pytest.raises(ValidationError, match="rendered_hash"):
        RenderedPolicy(**payload)
    with pytest.raises(ValidationError, match="token_count"):
        RenderedPolicy(**{**card.model_dump(), "token_count": card.token_count + 1})
    with pytest.raises(ValidationError, match="rendered_hash"):
        card.model_copy(update={"rendered_hash": "sha256:" + "0" * 64})


def test_rendering_detaches_and_revalidates_forged_clause_values() -> None:
    target = clause("target")
    object.__setattr__(target, "body", "\u0000")

    with pytest.raises(ValidationError, match="body"):
        render_policy_card(target)


def test_rendering_rejects_forged_clause_extra_and_missing_key_trees() -> None:
    injected_extra = clause("extra")
    object.__getattribute__(injected_extra, "__dict__")["unexpected"] = "x"
    missing_body = PolicyClause.model_construct(
        clause_id="missing", domain="access_provisioning", heading="Heading"
    )

    with pytest.raises(ValueError, match="key tree"):
        render_policy_card(injected_extra)
    with pytest.raises(ValueError, match="key tree"):
        render_policy_card(missing_body)


def test_rendering_rejects_oversized_forged_clause_key_tree_cardinality_first() -> None:
    target = clause("target")
    raw = object.__getattribute__(target, "__dict__")
    raw.update({f"unexpected_{index}": "x" for index in range(10_000)})

    with pytest.raises(ValueError, match="key tree"):
        render_policy_card(target)


def test_model_copy_rejects_forged_model_key_trees() -> None:
    clause_with_extra = clause("target")
    object.__getattribute__(clause_with_extra, "__dict__")["unexpected"] = "x"
    card_with_extra = render_policy_card(clause("card"))
    object.__getattribute__(card_with_extra, "__dict__")["unexpected"] = "x"

    with pytest.raises(ValueError, match="key tree"):
        clause_with_extra.model_copy()
    with pytest.raises(ValueError, match="key tree"):
        card_with_extra.model_copy(update={})


@pytest.mark.parametrize("model_type", [PolicyClause, RenderedPolicy])
def test_json_and_string_validation_ingress_are_unavailable(
    model_type: type[PolicyClause] | type[RenderedPolicy],
) -> None:
    clause_payload = {
        "clause_id": "target",
        "domain": "access_provisioning",
        "heading": "Heading",
        "body": "Body.",
        "evil": "x",
    }
    card_payload = {**render_policy_card(clause("card")).model_dump(), "evil": "x"}
    payload = clause_payload if model_type is PolicyClause else card_payload

    with pytest.raises(ValueError, match="exact built-in dict"):
        model_type.model_validate_json(json.dumps(payload), extra="allow")
    with pytest.raises(ValueError, match="exact built-in dict"):
        model_type.model_validate_strings(payload, extra="allow")


def test_rendering_and_copy_reject_forged_pydantic_extra_state() -> None:
    target = clause("target")
    card = render_policy_card(clause("card"))
    object.__setattr__(target, "__pydantic_extra__", {"evil": "x"})
    object.__setattr__(card, "__pydantic_extra__", {"evil": "x"})

    with pytest.raises(ValueError, match="extra state"):
        render_policy_card(target)
    with pytest.raises(ValueError, match="extra state"):
        target.model_copy()
    with pytest.raises(ValueError, match="extra state"):
        card.model_copy()


@pytest.mark.parametrize("model_type", [PolicyClause, RenderedPolicy])
def test_direct_mapping_and_constructor_reject_oversized_key_trees_before_validation(
    model_type: type[PolicyClause] | type[RenderedPolicy],
) -> None:
    clause_payload = {
        "clause_id": "target",
        "domain": "access_provisioning",
        "heading": "Heading",
        "body": "Body.",
    }
    card_payload = render_policy_card(clause("card")).model_dump()
    payload = dict(clause_payload if model_type is PolicyClause else card_payload)
    payload.update({f"unexpected_{index}": "x" for index in range(10_000)})

    with pytest.raises(ValueError, match="cardinality"):
        model_type.model_validate(payload)
    with pytest.raises(ValueError, match="cardinality"):
        cast(Any, model_type)(**payload)


def test_matched_handbooks_have_only_the_stated_differences() -> None:
    target = clause("target")
    distractors = [clause("beta"), clause("alpha"), clause("gamma")]

    salient, buried = render_matched_handbooks(target, distractors, seed=17)

    assert salient.document_kind == buried.document_kind == "handbook"
    assert salient.target_position == 0
    assert buried.target_position == len(buried.sections) - 1
    assert salient.sections[0] == target
    assert buried.sections[-1] == target
    assert salient.sections[1:] == buried.sections[:-1]
    assert sorted(section.clause_id for section in salient.sections) == sorted(
        section.clause_id for section in buried.sections
    )
    assert sorted(salient.section_hashes) == sorted(buried.section_hashes)
    assert salient.target_section_hash == buried.target_section_hash
    assert salient.content_multiset_hash == buried.content_multiset_hash
    assert salient.token_count == buried.token_count
    assert salient.rendered_text != buried.rendered_text
    assert salient.rendered_hash != buried.rendered_hash
    assert salient.section_hashes != buried.section_hashes


def test_matched_handbooks_are_deterministic_and_detached_from_input_list() -> None:
    target = clause("target")
    distractors = [clause("first"), clause("second")]
    first = render_matched_handbooks(target, distractors, seed=4)
    distractors[0] = clause("replacement")
    second = render_matched_handbooks(target, [clause("first"), clause("second")], seed=4)

    assert first == second
    with pytest.raises(ValidationError):
        first[0].target_position = 1  # type: ignore[misc]


def test_matched_handbooks_accept_exact_tuple_ingress() -> None:
    salient, buried = render_matched_handbooks(
        clause("target"), (clause("one"), clause("two")), seed=3
    )

    assert salient.sections[1:] == buried.sections[:-1]


class _CustomList(list[PolicyClause]):
    pass


@pytest.mark.parametrize(
    "distractors",
    [
        [],
        _CustomList([clause("distractor")]),
        [clause("target")],
        [clause("one"), clause("one")],
        [clause("finance", domain="financial_adjustments")],
    ],
)
def test_matched_handbooks_reject_invalid_distractor_ingress(
    distractors: list[PolicyClause],
) -> None:
    with pytest.raises((TypeError, ValidationError, ValueError)):
        render_matched_handbooks(clause("target"), distractors, seed=1)


@pytest.mark.parametrize("seed", [True, False, 1.0])
def test_matched_handbooks_require_an_exact_non_bool_integer_seed(seed: Any) -> None:
    with pytest.raises((TypeError, ValidationError, ValueError)):
        render_matched_handbooks(clause("target"), [clause("other")], seed=seed)
