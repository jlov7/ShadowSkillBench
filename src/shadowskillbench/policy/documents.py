from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator
from pydantic.config import ExtraValues

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.engine.models import PathSafeId, Sha256Ref

_POLICY_PROFILE = "SSB-POLICY1"
_TOKEN_PROFILE = "SSB-ST1"
_DOMAINS = frozenset({"access_provisioning", "financial_adjustments"})
_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_ASCII_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_IGNORED_TOKEN_CHARACTERS = frozenset("\t\n\v\f\r ")


class _StrictFrozenPolicyModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="never"
    )
    _field_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, /, **data: Any) -> None:
        payload = _exact_key_tree_payload(
            data,
            field_names=self._field_names,
            field=type(self).__name__,
        )
        super().__init__(**payload)

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        if strict is False or extra not in {None, "forbid"} or from_attributes is True:
            raise ValueError(f"{cls.__name__} requires strict, extra-forbid mapping validation")
        if type(obj) is not dict:
            raise ValueError(f"{cls.__name__} input must be an exact built-in object")
        payload = _exact_key_tree_payload(
            obj,
            field_names=cls._field_names,
            field=cls.__name__,
        )
        return super().model_validate(
            payload,
            strict=True,
            extra="forbid",
            from_attributes=False,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        raise ValueError(f"{cls.__name__} requires an exact built-in dict")

    @classmethod
    def model_validate_strings(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        raise ValueError(f"{cls.__name__} requires an exact built-in dict")

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        if type(deep) is not bool or (update is not None and type(update) is not dict):
            raise ValueError("model_copy requires exact built-in inputs")
        payload = _exact_instance_payload(
            self,
            expected_type=type(self),
            field_names=self._field_names,
            field=type(self).__name__,
        )
        if update is not None:
            payload.update(update)
        return cast(Self, type(self)(**payload))


def _exact_path_safe_id(value: object, *, field: str) -> PathSafeId:
    if type(value) is not str or _PATH_SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact path-safe identifier")
    return cast(PathSafeId, value)


def _exact_domain(value: object) -> Literal["access_provisioning", "financial_adjustments"]:
    if type(value) is not str or value not in _DOMAINS:
        raise ValueError("domain must be access_provisioning or financial_adjustments")
    return cast(Literal["access_provisioning", "financial_adjustments"], value)


def _utf8_text(value: object, *, field: str, single_line: bool = False) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact string")
    if not any(not character.isspace() for character in value):
        raise ValueError(f"{field} must be nonblank")
    if "\x00" in value or (single_line and ("\r" in value or "\n" in value)):
        raise ValueError(f"{field} contains forbidden control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field} must be UTF-8 encodable") from error
    return value


def _exact_sha256_ref(value: object, *, field: str) -> Sha256Ref:
    if type(value) is not str or _SHA256_REF.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase sha256 reference")
    return cast(Sha256Ref, value)


def _exact_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an exact nonnegative integer")
    return value


def _exact_key_tree_payload(
    value: object, *, field_names: tuple[str, ...], field: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{field} key tree must be an exact built-in object")
    tree = cast(dict[object, object], value)
    if len(tree) != len(field_names):
        raise ValueError(f"{field} key tree has an unexpected cardinality")
    for key in tree:
        if type(key) is not str:
            raise ValueError(f"{field} key tree keys must be exact strings")
    for name in field_names:
        if name not in tree:
            raise ValueError(f"{field} key tree is missing a required field")
    return {name: tree[name] for name in field_names}


def _exact_instance_payload(
    value: object,
    *,
    expected_type: type[_StrictFrozenPolicyModel],
    field_names: tuple[str, ...],
    field: str,
) -> dict[str, object]:
    if type(value) is not expected_type:
        raise ValueError(f"{field} must be an exact {expected_type.__name__}")
    if object.__getattribute__(value, "__pydantic_extra__") is not None:
        raise ValueError(f"{field} has forbidden pydantic extra state")
    raw = object.__getattribute__(value, "__dict__")
    _exact_key_tree_payload(raw, field_names=field_names, field=field)
    return {name: object.__getattribute__(value, name) for name in field_names}


class PolicyClause(_StrictFrozenPolicyModel):
    _field_names: ClassVar[tuple[str, ...]] = ("clause_id", "domain", "heading", "body")

    clause_id: PathSafeId
    domain: Literal["access_provisioning", "financial_adjustments"]
    heading: str
    body: str

    @field_validator("clause_id", mode="before")
    @classmethod
    def _validate_clause_id(cls, value: object) -> PathSafeId:
        return _exact_path_safe_id(value, field="clause_id")

    @field_validator("domain", mode="before")
    @classmethod
    def _validate_domain(
        cls, value: object
    ) -> Literal["access_provisioning", "financial_adjustments"]:
        return _exact_domain(value)

    @field_validator("heading", mode="before")
    @classmethod
    def _validate_heading(cls, value: object) -> str:
        return _utf8_text(value, field="heading", single_line=True)

    @field_validator("body", mode="before")
    @classmethod
    def _validate_body(cls, value: object) -> str:
        return _utf8_text(value, field="body")


def _clause_payload(value: object, *, field: str, allow_mapping: bool = False) -> dict[str, object]:
    if type(value) is PolicyClause:
        return _exact_instance_payload(
            value,
            expected_type=PolicyClause,
            field_names=PolicyClause._field_names,
            field=field,
        )
    if allow_mapping and type(value) is dict:
        return _exact_key_tree_payload(
            value,
            field_names=PolicyClause._field_names,
            field=field,
        )
    raise ValueError(f"{field} must be an exact PolicyClause")


def _detached_clause(value: object, *, field: str) -> PolicyClause:
    return PolicyClause.model_validate(_clause_payload(value, field=field))


def _owned_sections(value: object) -> tuple[PolicyClause, ...]:
    if type(value) not in {list, tuple}:
        raise ValueError("sections must be an exact built-in list or tuple")
    return tuple(
        PolicyClause.model_validate(
            _clause_payload(item, field="sections item", allow_mapping=True)
        )
        for item in cast(Sequence[object], value)
    )


def _section_text(section: PolicyClause) -> str:
    return f"## {section.heading}\n{section.body}"


def _section_hash(section: PolicyClause) -> Sha256Ref:
    return cast(
        Sha256Ref,
        sha256_ref({"profile": _POLICY_PROFILE, "section_text": _section_text(section)}),
    )


def _content_multiset_hash(
    sections: tuple[PolicyClause, ...], hashes: tuple[Sha256Ref, ...]
) -> Sha256Ref:
    entries = [
        {"clause_id": section.clause_id, "section_hash": section_hash}
        for section, section_hash in zip(sections, hashes, strict=True)
    ]
    entries.sort(key=lambda entry: cast(str, entry["clause_id"]))
    return cast(
        Sha256Ref,
        sha256_ref({"profile": _POLICY_PROFILE, "sections": entries}),
    )


def _rendered_hash(rendered_text: str) -> Sha256Ref:
    return cast(
        Sha256Ref,
        sha256_ref({"profile": _POLICY_PROFILE, "rendered_text": rendered_text}),
    )


def _ssb_st1_count(text: str) -> int:
    count = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character in _ASCII_WORD:
            count += 1
            index += 1
            while index < len(text) and text[index] in _ASCII_WORD:
                index += 1
        elif character in _IGNORED_TOKEN_CHARACTERS:
            index += 1
        else:
            count += 1
            index += 1
    return count


def _derived_values(
    sections: tuple[PolicyClause, ...], target_clause_id: PathSafeId
) -> tuple[tuple[Sha256Ref, ...], Sha256Ref, Sha256Ref, str, Sha256Ref, int]:
    hashes = tuple(_section_hash(section) for section in sections)
    target_position = next(
        index for index, section in enumerate(sections) if section.clause_id == target_clause_id
    )
    rendered_text = "\n\n".join(_section_text(section) for section in sections)
    return (
        hashes,
        hashes[target_position],
        _content_multiset_hash(sections, hashes),
        rendered_text,
        _rendered_hash(rendered_text),
        _ssb_st1_count(rendered_text),
    )


class RenderedPolicy(_StrictFrozenPolicyModel):
    _field_names: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "rendering_profile",
        "token_count_profile",
        "domain",
        "document_kind",
        "sections",
        "target_clause_id",
        "target_position",
        "section_hashes",
        "target_section_hash",
        "content_multiset_hash",
        "rendered_text",
        "rendered_hash",
        "token_count",
    )

    schema_version: Literal["1.0"]
    rendering_profile: Literal["SSB-POLICY1"]
    token_count_profile: Literal["SSB-ST1"]
    domain: Literal["access_provisioning", "financial_adjustments"]
    document_kind: Literal["card", "handbook"]
    sections: tuple[PolicyClause, ...]
    target_clause_id: PathSafeId
    target_position: int
    section_hashes: tuple[Sha256Ref, ...]
    target_section_hash: Sha256Ref
    content_multiset_hash: Sha256Ref
    rendered_text: str
    rendered_hash: Sha256Ref
    token_count: int

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> Literal["1.0"]:
        if type(value) is not str or value != "1.0":
            raise ValueError("schema_version must be 1.0")
        return "1.0"

    @field_validator("rendering_profile", mode="before")
    @classmethod
    def _validate_rendering_profile(cls, value: object) -> Literal["SSB-POLICY1"]:
        if type(value) is not str or value != _POLICY_PROFILE:
            raise ValueError("rendering_profile must be SSB-POLICY1")
        return _POLICY_PROFILE

    @field_validator("token_count_profile", mode="before")
    @classmethod
    def _validate_token_count_profile(cls, value: object) -> Literal["SSB-ST1"]:
        if type(value) is not str or value != _TOKEN_PROFILE:
            raise ValueError("token_count_profile must be SSB-ST1")
        return _TOKEN_PROFILE

    @field_validator("domain", mode="before")
    @classmethod
    def _validate_domain(
        cls, value: object
    ) -> Literal["access_provisioning", "financial_adjustments"]:
        return _exact_domain(value)

    @field_validator("document_kind", mode="before")
    @classmethod
    def _validate_document_kind(cls, value: object) -> Literal["card", "handbook"]:
        if type(value) is not str or value not in {"card", "handbook"}:
            raise ValueError("document_kind must be card or handbook")
        return cast(Literal["card", "handbook"], value)

    @field_validator("sections", mode="before")
    @classmethod
    def _validate_sections(cls, value: object) -> tuple[PolicyClause, ...]:
        return _owned_sections(value)

    @field_validator("target_clause_id", mode="before")
    @classmethod
    def _validate_target_clause_id(cls, value: object) -> PathSafeId:
        return _exact_path_safe_id(value, field="target_clause_id")

    @field_validator("target_position", "token_count", mode="before")
    @classmethod
    def _validate_nonnegative_int(cls, value: object, info: ValidationInfo) -> int:
        return _exact_nonnegative_int(value, field=cast(str, info.field_name))

    @field_validator("section_hashes", mode="before")
    @classmethod
    def _validate_section_hashes(cls, value: object) -> tuple[Sha256Ref, ...]:
        if type(value) not in {list, tuple}:
            raise ValueError("section_hashes must be an exact built-in list or tuple")
        return tuple(
            _exact_sha256_ref(item, field="section_hashes item")
            for item in cast(Sequence[object], value)
        )

    @field_validator("target_section_hash", "content_multiset_hash", "rendered_hash", mode="before")
    @classmethod
    def _validate_hashes(cls, value: object, info: ValidationInfo) -> Sha256Ref:
        return _exact_sha256_ref(value, field=cast(str, info.field_name))

    @field_validator("rendered_text", mode="before")
    @classmethod
    def _validate_rendered_text(cls, value: object) -> str:
        return _utf8_text(value, field="rendered_text")

    @model_validator(mode="after")
    def _bind_derived_fields(self) -> Self:
        if not self.sections:
            raise ValueError("sections must be nonempty")
        if len({section.clause_id for section in self.sections}) != len(self.sections):
            raise ValueError("sections must have unique clause IDs")
        if any(section.domain != self.domain for section in self.sections):
            raise ValueError("sections must share the rendered policy domain")
        target_matches = [
            index
            for index, section in enumerate(self.sections)
            if section.clause_id == self.target_clause_id
        ]
        if len(target_matches) != 1:
            raise ValueError("target_clause_id must appear exactly once")
        if self.document_kind == "card" and (len(self.sections) != 1 or self.target_position != 0):
            raise ValueError("card requires exactly one target section at zero")
        if self.document_kind == "handbook" and len(self.sections) < 2:
            raise ValueError("handbook requires at least two sections")
        if self.target_position != target_matches[0]:
            raise ValueError("target_position does not bind sections")
        hashes, target_hash, multiset_hash, rendered_text, rendered_hash, token_count = (
            _derived_values(self.sections, self.target_clause_id)
        )
        if self.section_hashes != hashes:
            raise ValueError("section_hashes do not bind sections")
        if self.target_section_hash != target_hash:
            raise ValueError("target_section_hash does not bind target")
        if self.content_multiset_hash != multiset_hash:
            raise ValueError("content_multiset_hash does not bind sections")
        if self.rendered_text != rendered_text:
            raise ValueError("rendered_text does not bind sections")
        if self.rendered_hash != rendered_hash:
            raise ValueError("rendered_hash does not bind rendered_text")
        if self.token_count != token_count:
            raise ValueError("token_count does not bind rendered_text")
        return self


def _make_rendered_policy(
    *,
    document_kind: Literal["card", "handbook"],
    sections: tuple[PolicyClause, ...],
    target: PolicyClause,
) -> RenderedPolicy:
    hashes, target_hash, multiset_hash, rendered_text, rendered_hash, token_count = _derived_values(
        sections, target.clause_id
    )
    target_position = next(
        index for index, section in enumerate(sections) if section.clause_id == target.clause_id
    )
    return RenderedPolicy(
        schema_version="1.0",
        rendering_profile=_POLICY_PROFILE,
        token_count_profile=_TOKEN_PROFILE,
        domain=target.domain,
        document_kind=document_kind,
        sections=sections,
        target_clause_id=target.clause_id,
        target_position=target_position,
        section_hashes=hashes,
        target_section_hash=target_hash,
        content_multiset_hash=multiset_hash,
        rendered_text=rendered_text,
        rendered_hash=rendered_hash,
        token_count=token_count,
    )


def render_policy_card(policy: PolicyClause) -> RenderedPolicy:
    target = _detached_clause(policy, field="policy")
    return _make_rendered_policy(document_kind="card", sections=(target,), target=target)


def _distractor_digest(*, seed: int, clause_id: PathSafeId) -> str:
    return sha256_ref(
        {"profile": _POLICY_PROFILE, "seed": seed, "clause_id": clause_id}
    ).removeprefix("sha256:")


def render_matched_handbooks(
    target: PolicyClause, distractors: Sequence[PolicyClause], seed: int
) -> tuple[RenderedPolicy, RenderedPolicy]:
    if type(distractors) not in {list, tuple}:
        raise ValueError("distractors must be an exact built-in list or tuple")
    if not distractors:
        raise ValueError("distractors must contain at least one clause")
    if type(seed) is not int:
        raise ValueError("seed must be an exact non-bool integer")
    detached_target = _detached_clause(target, field="target")
    detached_distractors = tuple(
        _detached_clause(distractor, field="distractor") for distractor in distractors
    )
    all_sections = (detached_target, *detached_distractors)
    if any(section.domain != detached_target.domain for section in detached_distractors):
        raise ValueError("target and distractors must share one domain")
    if len({section.clause_id for section in all_sections}) != len(all_sections):
        raise ValueError("target and distractors must have unique clause IDs")
    ordered_distractors = tuple(
        sorted(
            detached_distractors,
            key=lambda section: (
                _distractor_digest(seed=seed, clause_id=section.clause_id),
                section.clause_id,
            ),
        )
    )
    salient_sections = (detached_target, *ordered_distractors)
    buried_sections = (*ordered_distractors, detached_target)
    return (
        _make_rendered_policy(
            document_kind="handbook", sections=salient_sections, target=detached_target
        ),
        _make_rendered_policy(
            document_kind="handbook", sections=buried_sections, target=detached_target
        ),
    )
