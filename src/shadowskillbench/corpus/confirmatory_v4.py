"""Forward-only V4 confirmatory corpus construction.

The V4 corpus is intentionally separate from the V1--V3 corpus class.  It
reuses only deterministic local case and bundle construction primitives; it
makes no provider, signing, network, or artifact-writing call.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.corpus.confirmatory import (
    AUTHORITY_CLASSES,
    DOMAINS,
    RATIOS,
    ConfirmatoryBundleSeed,
    ConfirmatoryCase,
    ConfirmatoryHoldError,
    SplitInventory,
    _assert_isolated,
    _build_case,
    _bundle,
    _inventory_for,
)

SCHEMA_VERSION_V4 = "4.0"
CORPUS_SEED_V4 = 104733
DEVELOPMENT_SEEDS_V4 = frozenset({4242, 4243})
STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO = 7
STAGE_A_CASES_PER_DOMAIN = 70
STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN = 22


_AUTHORIZATION_ISSUER = object()
_ISSUED_AUTHORIZATIONS: dict[int, tuple[str, str, str]] = {}


@dataclass(frozen=True, slots=True, init=False)
class ConfirmatoryAuthorizationV4:
    """Opaque issuer-bound V4 custody admission.

    A caller cannot manufacture an authorization by supplying matching hashes:
    every instance is checked against the issuer's in-process receipt registry.
    The production issuer remains deliberately unavailable until the named V4
    custody gates have passed.
    """

    freeze_manifest_hash: str
    anchor_receipt_hash: str
    authorization_hash: str
    _issuer: object = field(repr=False, compare=False)

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("ConfirmatoryAuthorizationV4 is issuer-bound and cannot be constructed")

    @classmethod
    def _issue(
        cls, *, freeze_manifest_hash: str, anchor_receipt_hash: str, issuer: object
    ) -> ConfirmatoryAuthorizationV4:
        if issuer is not _AUTHORIZATION_ISSUER:
            raise ConfirmatoryHoldError("V4 authorization issuer is unavailable")
        if not all(
            type(value) is str and value.startswith("sha256:") and len(value) == 71
            for value in (freeze_manifest_hash, anchor_receipt_hash)
        ):
            raise ConfirmatoryHoldError("V4 authorization hashes are invalid")
        value = object.__new__(cls)
        authorization_hash = sha256_ref(
            {
                "profile": "SSB-CONFIRMATORY-AUTHORIZATION4",
                "freeze_manifest_hash": freeze_manifest_hash,
                "anchor_receipt_hash": anchor_receipt_hash,
            }
        )
        object.__setattr__(value, "freeze_manifest_hash", freeze_manifest_hash)
        object.__setattr__(value, "anchor_receipt_hash", anchor_receipt_hash)
        object.__setattr__(value, "authorization_hash", authorization_hash)
        object.__setattr__(value, "_issuer", issuer)
        _ISSUED_AUTHORIZATIONS[id(value)] = (
            freeze_manifest_hash,
            anchor_receipt_hash,
            authorization_hash,
        )
        return value


def authorization_projection_v4(authorization: ConfirmatoryAuthorizationV4) -> dict[str, str]:
    """Return the exact immutable authorization receipt, or fail closed."""

    validate_confirmatory_authorization_v4(authorization)
    return {
        "profile": "SSB-CONFIRMATORY-AUTHORIZATION4",
        "freeze_manifest_hash": authorization.freeze_manifest_hash,
        "anchor_receipt_hash": authorization.anchor_receipt_hash,
        "authorization_hash": authorization.authorization_hash,
    }


def validate_confirmatory_authorization_v4(authorization: object) -> None:
    if type(authorization) is not ConfirmatoryAuthorizationV4:
        raise ConfirmatoryHoldError("V4 custody authorization is required")
    issued = _ISSUED_AUTHORIZATIONS.get(id(authorization))
    if (
        authorization._issuer is not _AUTHORIZATION_ISSUER
        or issued
        != (
            authorization.freeze_manifest_hash,
            authorization.anchor_receipt_hash,
            authorization.authorization_hash,
        )
        or authorization.authorization_hash
        != sha256_ref(
            {
                "profile": "SSB-CONFIRMATORY-AUTHORIZATION4",
                "freeze_manifest_hash": authorization.freeze_manifest_hash,
                "anchor_receipt_hash": authorization.anchor_receipt_hash,
            }
        )
    ):
        raise ConfirmatoryHoldError("V4 custody authorization was not issued by the V4 issuer")


def authorize_confirmatory_generation_v4(
    manifest_bytes: bytes, detached_sha256_bytes: bytes, anchor_receipt_bytes: bytes
) -> ConfirmatoryAuthorizationV4:
    """Non-authorizing compatibility stub until V4 full custody is implemented."""
    if any(
        type(value) is not bytes
        for value in (manifest_bytes, detached_sha256_bytes, anchor_receipt_bytes)
    ):
        raise ConfirmatoryHoldError("V4 custody inputs must be exact bytes")
    manifest_hash = sha256(manifest_bytes).hexdigest()
    if detached_sha256_bytes != f"{manifest_hash}  freeze_manifest.v4.json\n".encode():
        raise ConfirmatoryHoldError("V4 detached manifest hash does not bind exact bytes")
    try:
        payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryHoldError("V4 freeze manifest is not valid JSON") from error
    canonical = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    if (
        type(payload) is not dict
        or manifest_bytes != canonical
        or payload.get("profile") != "SSB-SCIENTIFIC-PREFREEZE-4"
        or payload.get("schema_version") != "4.0"
        or payload.get("anchor_status") != "HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY"
        or type(payload.get("inputs")) is not list
    ):
        raise ConfirmatoryHoldError("V4 freeze manifest is invalid or is not V4")
    if not anchor_receipt_bytes:
        raise ConfirmatoryHoldError("V4 anchor receipt bytes are required")
    del manifest_hash
    raise ConfirmatoryHoldError(
        "HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY: "
        "scientific prefreeze and arbitrary receipt bytes are not execution authorization"
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryCorpusV4:
    """Validated V4 corpus metadata and deterministic private case custody."""

    schema_version: Literal["4.0"]
    authorization: ConfirmatoryAuthorizationV4
    corpus_seed: int
    bundles: tuple[ConfirmatoryBundleSeed, ...]
    stage_a_cases: tuple[ConfirmatoryCase, ...]
    stage_b_cases: tuple[ConfirmatoryCase, ...]
    inventory: SplitInventory
    content_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION_V4 or type(self.corpus_seed) is not int:
            raise ValueError("V4 corpus schema version or seed is invalid")
        if (
            type(self.authorization) is not ConfirmatoryAuthorizationV4
            or type(self.bundles) is not tuple
            or type(self.stage_a_cases) is not tuple
            or type(self.stage_b_cases) is not tuple
            or type(self.inventory) is not SplitInventory
        ):
            raise ValueError("V4 corpus contents are invalid")
        validate_confirmatory_authorization_v4(self.authorization)
        _validate_layout_v4(self.bundles, self.stage_a_cases, self.stage_b_cases)
        if self.inventory != _inventory_for(self.bundles, self.stage_a_cases, self.stage_b_cases):
            raise ValueError("V4 corpus inventory does not bind generated values")
        if self.content_hash != sha256_ref(self.projection()):
            raise ValueError("V4 corpus content hash does not bind confirmatory corpus")

    def projection(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "freeze_manifest_hash": self.authorization.freeze_manifest_hash,
            "anchor_receipt_hash": self.authorization.anchor_receipt_hash,
            "corpus_seed": self.corpus_seed,
            "bundle_hashes": [bundle.content_hash for bundle in self.bundles],
            "stage_a_hidden_content_hashes": [
                case.hidden_content_hash for case in self.stage_a_cases
            ],
            "stage_b_hidden_content_hashes": [
                case.hidden_content_hash for case in self.stage_b_cases
            ],
        }

    def model_visible_projection(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_seed": self.corpus_seed,
            "bundles": [bundle.projection() for bundle in self.bundles],
            "stage_a_cases": [case.model_visible_projection() for case in self.stage_a_cases],
            "stage_b_cases": [case.model_visible_projection() for case in self.stage_b_cases],
        }


def _validate_layout_v4(
    bundles: tuple[ConfirmatoryBundleSeed, ...],
    stage_a_cases: tuple[ConfirmatoryCase, ...],
    stage_b_cases: tuple[ConfirmatoryCase, ...],
) -> None:
    expected_bundles = len(DOMAINS) * len(RATIOS) * STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO
    expected_a = len(DOMAINS) * STAGE_A_CASES_PER_DOMAIN
    expected_b = len(DOMAINS) * len(AUTHORITY_CLASSES) * STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN
    if (len(bundles), len(stage_a_cases), len(stage_b_cases)) != (
        expected_bundles,
        expected_a,
        expected_b,
    ):
        raise ValueError("V4 corpus does not have preregistered total counts")
    if len({bundle.bundle_id for bundle in bundles}) != len(bundles):
        raise ValueError("V4 bundle ids must be unique")
    if len({case.case_id for case in stage_a_cases + stage_b_cases}) != expected_a + expected_b:
        raise ValueError("V4 case ids must be unique across stages")
    for domain in DOMAINS:
        domain_bundles = [bundle for bundle in bundles if bundle.domain == domain]
        if len(domain_bundles) != len(RATIOS) * STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO or Counter(
            bundle.contamination_ratio for bundle in domain_bundles
        ) != Counter({ratio: STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO for ratio in RATIOS}):
            raise ValueError("V4 Stage A requires seven bundle seeds per ratio and domain")
        domain_a = [case for case in stage_a_cases if case.domain == domain]
        domain_b = [case for case in stage_b_cases if case.domain == domain]
        if len(domain_a) != STAGE_A_CASES_PER_DOMAIN:
            raise ValueError("V4 Stage A requires 70 held-out cases per domain")
        if len(domain_b) != len(AUTHORITY_CLASSES) * STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN:
            raise ValueError("V4 Stage B requires 110 held-out cases per domain")
        if Counter(case.authority_class for case in domain_a) != Counter(
            {authority_class: 4 for authority_class in AUTHORITY_CLASSES}
        ):
            raise ValueError("V4 Stage A authority-class balance is invalid")
        if Counter(case.authority_class for case in domain_b) != Counter(
            {
                authority_class: STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN
                for authority_class in AUTHORITY_CLASSES
            }
        ):
            raise ValueError("V4 Stage B requires 22 cases per authority class and domain")


def generate_confirmatory_corpus_v4(
    *,
    authorization: ConfirmatoryAuthorizationV4,
    corpus_seed: int,
    development_inventory: SplitInventory,
) -> ConfirmatoryCorpusV4:
    """Generate the closed V4 70-bundle/260-case corpus after custody validation."""

    if type(corpus_seed) is not int:
        raise ConfirmatoryHoldError("validated authorization and exact V4 corpus seed are required")
    validate_confirmatory_authorization_v4(authorization)
    if corpus_seed != CORPUS_SEED_V4:
        raise ConfirmatoryHoldError(f"V4 corpus seed must be frozen value {CORPUS_SEED_V4}")
    if type(development_inventory) is not SplitInventory:
        raise ConfirmatoryHoldError("explicit development split inventory is required")
    if not DEVELOPMENT_SEEDS_V4 <= development_inventory.seeds:
        raise ConfirmatoryHoldError(
            "V4 development inventory must cover both frozen seeds 4242 and 4243"
        )
    raise ConfirmatoryHoldError(
        "HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY: "
        "confirmatory corpus generation is not authorized"
    )
    bundles = tuple(
        _bundle(corpus_seed, domain, ratio, ordinal)
        for domain in DOMAINS
        for ratio in RATIOS
        for ordinal in range(STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO)
    )
    stage_a = tuple(
        _build_case("stage_a", corpus_seed, domain, authority_class, ordinal)
        for domain in DOMAINS
        for authority_class in AUTHORITY_CLASSES
        for ordinal in range(4)
    )
    stage_b = tuple(
        _build_case("stage_b", corpus_seed, domain, authority_class, ordinal)
        for domain in DOMAINS
        for authority_class in AUTHORITY_CLASSES
        for ordinal in range(STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN)
    )
    inventory = _inventory_for(bundles, stage_a, stage_b)
    _assert_isolated(development_inventory, inventory)
    projection = {
        "schema_version": SCHEMA_VERSION_V4,
        "freeze_manifest_hash": authorization.freeze_manifest_hash,
        "anchor_receipt_hash": authorization.anchor_receipt_hash,
        "corpus_seed": corpus_seed,
        "bundle_hashes": [bundle.content_hash for bundle in bundles],
        "stage_a_hidden_content_hashes": [case.hidden_content_hash for case in stage_a],
        "stage_b_hidden_content_hashes": [case.hidden_content_hash for case in stage_b],
    }
    return ConfirmatoryCorpusV4(
        SCHEMA_VERSION_V4,
        authorization,
        corpus_seed,
        bundles,
        stage_a,
        stage_b,
        inventory,
        sha256_ref(projection),
    )


def validate_confirmatory_corpus_v4(corpus: ConfirmatoryCorpusV4) -> None:
    if type(corpus) is not ConfirmatoryCorpusV4:
        raise ValueError("corpus must be an exact ConfirmatoryCorpusV4")
    ConfirmatoryCorpusV4(
        corpus.schema_version,
        corpus.authorization,
        corpus.corpus_seed,
        corpus.bundles,
        corpus.stage_a_cases,
        corpus.stage_b_cases,
        corpus.inventory,
        corpus.content_hash,
    )


__all__ = [
    "ConfirmatoryCorpusV4",
    "ConfirmatoryAuthorizationV4",
    "CORPUS_SEED_V4",
    "DEVELOPMENT_SEEDS_V4",
    "SCHEMA_VERSION_V4",
    "STAGE_A_BUNDLE_SEEDS_PER_DOMAIN_RATIO",
    "STAGE_A_CASES_PER_DOMAIN",
    "STAGE_B_CASES_PER_AUTHORITY_CLASS_DOMAIN",
    "authorization_projection_v4",
    "authorize_confirmatory_generation_v4",
    "generate_confirmatory_corpus_v4",
    "validate_confirmatory_corpus_v4",
    "validate_confirmatory_authorization_v4",
]
