from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Literal, cast

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.corpus.development import (
    ACCESS_DOMAIN,
    FINANCE_DOMAIN,
    AuthorityClass,
    DevelopmentCase,
    DevelopmentCorpus,
    _draft_case,
)
from shadowskillbench.traces.bundles import (
    generate_bundle,
    hash_source_bundle_manifest,
)

type Domain = Literal["access_provisioning", "financial_adjustments"]
type Stage = Literal["stage_a", "stage_b"]

DOMAINS: tuple[Domain, ...] = (ACCESS_DOMAIN, FINANCE_DOMAIN)
RATIOS: tuple[Decimal, ...] = (
    Decimal("0"),
    Decimal("0.25"),
    Decimal("0.5"),
    Decimal("0.75"),
    Decimal("1"),
)
AUTHORITY_CLASSES: tuple[AuthorityClass, ...] = tuple(AuthorityClass)
STAGE_A_CASES_PER_DOMAIN = 20
STAGE_B_CASES_PER_DOMAIN = 25
STAGE_A_BUNDLES_PER_DOMAIN = 15
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_LOCATOR = re.compile(r"(?:https://|urn:)[^\s]+\Z")
_OPAQUE_ENTITY = re.compile(r"[a-z][a-z0-9_]*_[0-9a-f]{16}\Z")
_CONFIRMATORY_INPUT = "src/shadowskillbench/corpus/confirmatory.py"
_FREEZE_MANIFEST_PATH = "protocol/freeze_manifest.json"
_DETACHED_FREEZE_MANIFEST_PATH = "protocol/freeze_manifest.sha256"
_ANCHOR_RECEIPT_PATH = "protocol/anchor_receipt.json"


class ConfirmatoryHoldError(ValueError):
    """Raised when custody or split-isolation evidence is insufficient."""


@dataclass(frozen=True, slots=True)
class FreezeWitness:
    manifest_hash: str
    manifest_bytes: bytes


@dataclass(frozen=True, slots=True)
class AnchorReceipt:
    freeze_commit: str
    signed_tag: str
    freeze_manifest_hash: str
    immutable_locator: str
    anchored_at: str
    verification_result: Literal["VERIFIED", "VERIFIED_LOCAL"]
    custody_mode: Literal["EXTERNAL_SIGNED_ANCHOR", "LOCAL_HASH_CUSTODY"]
    receipt_hash: str
    receipt_bytes: bytes

    @property
    def externally_verified(self) -> bool:
        return self.custody_mode == "EXTERNAL_SIGNED_ANCHOR"


@dataclass(frozen=True, slots=True)
class ConfirmatoryAuthorization:
    freeze: FreezeWitness
    receipt: AnchorReceipt
    repository_root: Path = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path):
            raise ConfirmatoryHoldError("confirmatory authorization requires a repository root")
        detached = (
            f"{self.freeze.manifest_hash.removeprefix('sha256:')}  freeze_manifest.json\n".encode()
        )
        verified_freeze = validate_freeze_manifest(self.freeze.manifest_bytes, detached)
        if verified_freeze != self.freeze:
            raise ConfirmatoryHoldError("freeze witness is not the exact validated manifest")
        verified_receipt = validate_anchor_receipt(
            self.receipt.receipt_bytes,
            verified_freeze,
            repository_root=self.repository_root,
        )
        if verified_receipt != self.receipt:
            raise ConfirmatoryHoldError(
                "anchor receipt is not the exact validated verification record"
            )
        if self.freeze.manifest_hash != self.receipt.freeze_manifest_hash:
            raise ConfirmatoryHoldError("anchor receipt does not bind the supplied freeze manifest")


@dataclass(frozen=True, slots=True)
class SplitInventory:
    seeds: frozenset[int]
    entity_ids: frozenset[str]
    value_hashes: frozenset[str]

    def __post_init__(self) -> None:
        if (
            type(self.seeds) is not frozenset
            or not self.seeds
            or any(type(seed) is not int for seed in self.seeds)
        ):
            raise ValueError("inventory seeds must be an exact integer set")
        if (
            type(self.entity_ids) is not frozenset
            or not self.entity_ids
            or any(type(value) is not str or not value for value in self.entity_ids)
        ):
            raise ValueError("inventory entity_ids must be an exact nonempty-string set")
        if (
            type(self.value_hashes) is not frozenset
            or not self.value_hashes
            or any(_SHA256.fullmatch(value) is None for value in self.value_hashes)
        ):
            raise ValueError("inventory value_hashes must be sha256 references")


@dataclass(frozen=True, slots=True)
class ConfirmatoryBundleSeed:
    binding_profile: Literal["SSB-CONFIRMATORY-BUNDLE-SEED2"]
    domain: Domain
    contamination_ratio: Decimal
    seed: int
    bundle_id: str
    source_manifest_hash: str
    content_hash: str

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS or self.contamination_ratio not in RATIOS:
            raise ValueError("bundle domain or contamination ratio is not preregistered")
        if (
            self.binding_profile != "SSB-CONFIRMATORY-BUNDLE-SEED2"
            or type(self.seed) is not int
            or not self.bundle_id.startswith("bundle_")
            or _SHA256.fullmatch(self.source_manifest_hash) is None
        ):
            raise ValueError("bundle seed or id is invalid")
        generated = generate_bundle(self.domain, self.contamination_ratio, 12, self.seed)
        source_manifest = generated.bundle.source_manifest
        if (
            self.bundle_id != source_manifest.bundle_id
            or self.source_manifest_hash != hash_source_bundle_manifest(source_manifest)
        ):
            raise ValueError("bundle seed does not bind its trace source manifest")
        if self.content_hash != sha256_ref(self.projection()):
            raise ValueError("bundle content hash does not bind its deterministic seed")

    def projection(self) -> dict[str, object]:
        return {
            "binding_profile": self.binding_profile,
            "domain": self.domain,
            "contamination_ratio": str(self.contamination_ratio),
            "seed": self.seed,
            "bundle_id": self.bundle_id,
            "source_manifest_hash": self.source_manifest_hash,
        }


@dataclass(frozen=True, slots=True)
class ConfirmatoryCase:
    stage: Stage
    case_id: str
    source: DevelopmentCase
    public_content_hash: str
    hidden_content_hash: str

    def __post_init__(self) -> None:
        if self.stage not in {"stage_a", "stage_b"}:
            raise ValueError("case stage is invalid")
        if type(self.source) is not DevelopmentCase:
            raise ValueError("case source must be an exact generated case")
        if (
            self.case_id != self.source.task_case.case_id
            or self.case_id != self.source.domain_case.case_id
        ):
            raise ValueError("confirmatory case_id must bind the nested execution identities")
        if self.public_content_hash != sha256_ref(self.public_projection()):
            raise ValueError("public content hash does not bind execution inputs")
        if self.hidden_content_hash != sha256_ref(self.hidden_projection()):
            raise ValueError("hidden content hash does not bind scorer custody")

    @property
    def domain(self) -> Domain:
        return cast(Domain, self.source.domain)

    @property
    def seed(self) -> int:
        return self.source.seed

    @property
    def authority_class(self) -> AuthorityClass:
        return self.source.hidden_truth.authority_class

    def public_projection(self) -> dict[str, object]:
        source = self.source.public_projection()
        source["case_id"] = self.case_id
        source["stage"] = self.stage
        return source

    def hidden_projection(self) -> dict[str, object]:
        truth = self.source.hidden_truth
        return {
            "public_content_hash": self.public_content_hash,
            "authority_records": [
                record.model_dump(mode="json") for record in truth.authority_records
            ],
            "authority_query": truth.authority_query.model_dump(mode="json"),
            "authority_class": truth.authority_class.value,
            "expected_decision": truth.expected_decision.value,
            "expected_disposition": truth.expected_disposition.value,
            "expected_reason_code": truth.expected_reason_code.value,
            "authority_set_hash": truth.authority_set_hash,
            "authority_query_hash": truth.authority_query_hash,
            "authority_decision_hash": truth.authority_decision_hash,
        }

    def model_visible_projection(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "case_id": self.case_id,
            "domain": self.domain,
            **self.source.model_visible_projection(),
        }


@dataclass(frozen=True, slots=True)
class ConfirmatoryCorpus:
    schema_version: Literal["1.0"]
    authorization: ConfirmatoryAuthorization
    corpus_seed: int
    bundles: tuple[ConfirmatoryBundleSeed, ...]
    stage_a_cases: tuple[ConfirmatoryCase, ...]
    stage_b_cases: tuple[ConfirmatoryCase, ...]
    inventory: SplitInventory
    content_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or type(self.corpus_seed) is not int:
            raise ValueError("corpus schema version or seed is invalid")
        if type(self.bundles) is not tuple or type(self.stage_a_cases) is not tuple:
            raise ValueError("corpus contents must be exact tuples")
        if type(self.stage_b_cases) is not tuple or type(self.inventory) is not SplitInventory:
            raise ValueError("corpus contents are invalid")
        _validate_layout(self.bundles, self.stage_a_cases, self.stage_b_cases)
        if self.inventory != _inventory_for(self.bundles, self.stage_a_cases, self.stage_b_cases):
            raise ValueError("corpus inventory does not bind generated values")
        if self.content_hash != sha256_ref(self.projection()):
            raise ValueError("corpus content hash does not bind confirmatory corpus")

    def projection(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "freeze_manifest_hash": self.authorization.freeze.manifest_hash,
            "anchor_receipt_hash": self.authorization.receipt.receipt_hash,
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

    def public_artifact_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_seed": self.corpus_seed,
            "bundles": [bundle.projection() for bundle in self.bundles],
            "stage_a_cases": [case.model_visible_projection() for case in self.stage_a_cases],
            "stage_b_cases": [case.model_visible_projection() for case in self.stage_b_cases],
        }

    def hidden_scorer_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "freeze_manifest_hash": self.authorization.freeze.manifest_hash,
            "anchor_receipt_hash": self.authorization.receipt.receipt_hash,
            "case_truth": {
                case.case_id: case.hidden_projection()
                | {"hidden_content_hash": case.hidden_content_hash}
                for case in self.stage_a_cases + self.stage_b_cases
            },
            "content_hash": self.content_hash,
        }


def _canonical_freeze_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _current_generator_hash() -> str:
    try:
        return f"sha256:{sha256(Path(__file__).read_bytes()).hexdigest()}"
    except OSError as error:
        raise ConfirmatoryHoldError(
            "current confirmatory generator bytes are unavailable"
        ) from error


def _strict_json_bytes(value: object, name: str) -> bytes:
    if type(value) is not bytes:
        raise ConfirmatoryHoldError(f"{name} must be exact bytes")
    return cast(bytes, value)


def validate_freeze_manifest(
    manifest_bytes: object, detached_sha256_bytes: object
) -> FreezeWitness:
    """Validate exact phase-one bytes and the detached SHA-256 record."""

    raw_manifest = _strict_json_bytes(manifest_bytes, "freeze manifest")
    raw_detached = _strict_json_bytes(detached_sha256_bytes, "detached manifest hash")
    digest = sha256(raw_manifest).hexdigest()
    if raw_detached != f"{digest}  freeze_manifest.json\n".encode():
        raise ConfirmatoryHoldError("detached manifest hash does not bind exact manifest bytes")
    try:
        payload = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryHoldError("freeze manifest is not valid JSON") from error
    if type(payload) is not dict or raw_manifest != _canonical_freeze_json(payload):
        raise ConfirmatoryHoldError("freeze manifest is not canonical exact bytes")
    if payload.get("schema_version") != "1.0" or payload.get("anchor_status") not in {
        "PENDING_HUMAN_ANCHOR",
        "PENDING_CUSTODY_RECEIPT",
    }:
        raise ConfirmatoryHoldError("freeze manifest is not a phase-one protocol freeze")
    preregistration = payload.get("preregistration_core")
    inputs = payload.get("inputs")
    if (
        type(preregistration) is not dict
        or preregistration.get("path") != "protocol/preregistration.md"
        or type(preregistration.get("sha256")) is not str
        or _SHA256.fullmatch(cast(str, preregistration.get("sha256"))) is None
        or type(inputs) is not list
    ):
        raise ConfirmatoryHoldError("freeze manifest has no valid preregistration core")
    paths: list[str] = []
    for item in inputs:
        if type(item) is not dict or set(item) != {"path", "role", "sha256"}:
            raise ConfirmatoryHoldError("freeze manifest inputs must have the exact custody shape")
        path, role, item_hash = item["path"], item["role"], item["sha256"]
        if type(path) is not str or type(role) is not str or _SHA256.fullmatch(item_hash) is None:
            raise ConfirmatoryHoldError("freeze manifest contains an invalid input")
        paths.append(path)
    generator_entry = next((item for item in inputs if item["path"] == _CONFIRMATORY_INPUT), None)
    if (
        len(paths) != len(set(paths))
        or generator_entry is None
        or generator_entry["sha256"] != _current_generator_hash()
    ):
        raise ConfirmatoryHoldError("freeze manifest does not freeze the confirmatory generator")
    return FreezeWitness(manifest_hash=f"sha256:{digest}", manifest_bytes=raw_manifest)


def validate_anchor_receipt(
    receipt_bytes: object,
    freeze: FreezeWitness,
    *,
    repository_root: Path | None = None,
) -> AnchorReceipt:
    """Validate a private local-hash custody receipt.

    Legacy external receipt JSON is deliberately held until this project has a
    trusted cryptographic verifier for its external anchoring system.  A
    self-asserted locator and ``VERIFIED`` field are not verification.
    """

    raw_receipt = _strict_json_bytes(receipt_bytes, "anchor receipt")
    try:
        payload = json.loads(raw_receipt)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryHoldError("anchor receipt is not valid JSON") from error
    external_required = {
        "freeze_commit",
        "signed_tag",
        "freeze_manifest_hash",
        "immutable_locator",
        "anchored_at",
        "verification_result",
    }
    local_required = {
        "schema_version",
        "custody_mode",
        "freeze_commit",
        "protocol_tag",
        "freeze_manifest_hash",
        "custody_locator",
        "created_at",
        "verification_result",
    }
    if type(payload) is not dict:
        raise ConfirmatoryHoldError("custody receipt must be an object")
    if set(payload) == external_required:
        raise ConfirmatoryHoldError(
            "HOLD_UNVERIFIED_EXTERNAL_ANCHOR: external receipts require trusted "
            "cryptographic verification"
        )
    if set(payload) == local_required:
        if payload["schema_version"] != "2.0" or payload["custody_mode"] != "LOCAL_HASH_CUSTODY":
            raise ConfirmatoryHoldError("local custody receipt profile is invalid")
        custody_mode = "LOCAL_HASH_CUSTODY"
        tag = payload["protocol_tag"]
        locator = payload["custody_locator"]
        anchored_at = payload["created_at"]
        expected_result = "VERIFIED_LOCAL"
    else:
        raise ConfirmatoryHoldError("custody receipt must have an exact supported shape")
    commit = payload["freeze_commit"]
    manifest_hash = payload["freeze_manifest_hash"]
    result = payload["verification_result"]
    if (
        type(commit) is not str
        or _COMMIT.fullmatch(commit) is None
        or type(tag) is not str
        or _TAG.fullmatch(tag) is None
        or type(manifest_hash) is not str
        or manifest_hash != freeze.manifest_hash
        or type(locator) is not str
        or _LOCATOR.fullmatch(locator) is None
        or type(anchored_at) is not str
        or not anchored_at.endswith("Z")
        or result != expected_result
    ):
        raise ConfirmatoryHoldError(
            "custody receipt does not prove the required post-freeze binding"
        )
    try:
        datetime.fromisoformat(anchored_at.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ConfirmatoryHoldError("anchor receipt timestamp is invalid") from error
    _validate_local_hash_custody(
        repository_root=repository_root,
        freeze=freeze,
        receipt_bytes=raw_receipt,
        freeze_commit=commit,
        protocol_tag=tag,
        custody_locator=locator,
    )
    return AnchorReceipt(
        freeze_commit=commit,
        signed_tag=tag,
        freeze_manifest_hash=manifest_hash,
        immutable_locator=locator,
        anchored_at=anchored_at,
        verification_result=cast(Literal["VERIFIED", "VERIFIED_LOCAL"], expected_result),
        custody_mode=cast(Literal["EXTERNAL_SIGNED_ANCHOR", "LOCAL_HASH_CUSTODY"], custody_mode),
        receipt_hash=sha256_ref(payload),
        receipt_bytes=raw_receipt,
    )


def authorize_confirmatory_generation(
    manifest_bytes: object,
    detached_sha256_bytes: object,
    receipt_bytes: object,
    *,
    repository_root: Path | None = None,
) -> ConfirmatoryAuthorization:
    if not isinstance(repository_root, Path):
        raise ConfirmatoryHoldError("confirmatory authorization requires a repository root")
    freeze = validate_freeze_manifest(manifest_bytes, detached_sha256_bytes)
    return ConfirmatoryAuthorization(
        freeze=freeze,
        receipt=validate_anchor_receipt(
            receipt_bytes,
            freeze,
            repository_root=repository_root,
        ),
        repository_root=repository_root,
    )


def _validate_local_hash_custody(
    *,
    repository_root: Path | None,
    freeze: FreezeWitness,
    receipt_bytes: bytes,
    freeze_commit: str,
    protocol_tag: str,
    custody_locator: str,
) -> None:
    if repository_root is None:
        raise ConfirmatoryHoldError("local hash custody requires a repository root")
    if not isinstance(repository_root, Path):
        raise ConfirmatoryHoldError("local hash custody repository root is invalid")
    try:
        if repository_root.is_symlink() or not repository_root.is_dir():
            raise OSError("repository root is not a directory")
        requested_root = repository_root.resolve(strict=True)
    except OSError as error:
        raise ConfirmatoryHoldError("local hash custody repository root is unavailable") from error
    if _git_output(requested_root, "rev-parse", "--is-inside-work-tree") != b"true\n":
        raise ConfirmatoryHoldError("local hash custody repository is not a work tree")
    try:
        actual_root = Path(
            _git_output(requested_root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
        ).resolve(strict=True)
    except (OSError, UnicodeDecodeError) as error:
        raise ConfirmatoryHoldError("local hash custody repository root is unavailable") from error
    if actual_root != requested_root:
        raise ConfirmatoryHoldError("local hash custody requires the repository root")
    if custody_locator != f"urn:git:{freeze_commit}":
        raise ConfirmatoryHoldError("local hash custody locator does not bind the freeze commit")
    if (
        _git_output(actual_root, "rev-parse", "--verify", f"{freeze_commit}^{{commit}}")
        != (freeze_commit + "\n").encode()
    ):
        raise ConfirmatoryHoldError("local hash custody freeze commit is unavailable")
    if (
        _git_output(
            actual_root,
            "rev-parse",
            "--verify",
            f"refs/tags/{protocol_tag}^{{commit}}",
        )
        != (freeze_commit + "\n").encode()
    ):
        raise ConfirmatoryHoldError("local hash custody tag does not resolve to the freeze commit")
    expected_detached = (
        f"{freeze.manifest_hash.removeprefix('sha256:')}  freeze_manifest.json\n".encode()
    )
    for path, expected in (
        (_FREEZE_MANIFEST_PATH, freeze.manifest_bytes),
        (_DETACHED_FREEZE_MANIFEST_PATH, expected_detached),
    ):
        if _git_output(actual_root, "show", f"{freeze_commit}:{path}") != expected:
            raise ConfirmatoryHoldError(f"local hash custody {path} differs from the freeze commit")
    _validate_checked_out_freeze_inputs(actual_root, freeze)
    for path, expected_hash in _freeze_input_hashes(freeze.manifest_bytes):
        committed_bytes = _git_output(actual_root, "show", f"{freeze_commit}:{path}")
        actual_hash = f"sha256:{sha256(committed_bytes).hexdigest()}"
        if actual_hash != expected_hash:
            raise ConfirmatoryHoldError(
                f"local hash custody {path} hash differs from the freeze commit"
            )
        if _checked_out_input_bytes(actual_root, path) != committed_bytes:
            raise ConfirmatoryHoldError(
                f"local hash custody {path} differs from the checked-out freeze input"
            )
    receipt_path = actual_root / _ANCHOR_RECEIPT_PATH
    try:
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise OSError("receipt is not a regular file")
        if receipt_path.read_bytes() != receipt_bytes:
            raise ConfirmatoryHoldError(
                "local hash custody receipt differs from the checked-out file"
            )
    except OSError as error:
        raise ConfirmatoryHoldError("local hash custody receipt is unavailable") from error
    try:
        _git_output(actual_root, "ls-files", "--error-unmatch", "--", _ANCHOR_RECEIPT_PATH)
    except ConfirmatoryHoldError as error:
        raise ConfirmatoryHoldError("local hash custody receipt is dirty or untracked") from error
    if _git_output(
        actual_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        _ANCHOR_RECEIPT_PATH,
    ):
        raise ConfirmatoryHoldError("local hash custody receipt is dirty or untracked")


def _validate_checked_out_freeze_inputs(repository_root: Path, freeze: FreezeWitness) -> None:
    """Fail closed when any manifest input differs from the checked-out byte source."""

    if not isinstance(repository_root, Path):
        raise ConfirmatoryHoldError("checked-out freeze inputs require a repository root")
    try:
        if repository_root.is_symlink() or not repository_root.is_dir():
            raise OSError("repository root is not a directory")
        root = repository_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ConfirmatoryHoldError("checked-out freeze inputs are unavailable") from error
    for path, expected_hash in _freeze_input_hashes(freeze.manifest_bytes):
        actual_hash = f"sha256:{sha256(_checked_out_input_bytes(root, path)).hexdigest()}"
        if actual_hash != expected_hash:
            raise ConfirmatoryHoldError(
                f"checked-out freeze input {path} hash differs from the manifest"
            )


def _git_output(repository_root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repository_root), *args],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
        )
    except OSError as error:
        raise ConfirmatoryHoldError("local hash custody git is unavailable") from error
    if completed.returncode != 0:
        raise ConfirmatoryHoldError("local hash custody git verification failed")
    return completed.stdout


def _checked_out_input_bytes(repository_root: Path, relative: str) -> bytes:
    pure_path = PurePosixPath(relative)
    candidate = repository_root.joinpath(*pure_path.parts)
    try:
        current = repository_root
        for part in pure_path.parts:
            current = current / part
            if current.is_symlink():
                raise ConfirmatoryHoldError(f"local hash custody {relative} is a symbolic link")
        if candidate.is_symlink() or not candidate.is_file():
            raise OSError("freeze input is unavailable")
        candidate.resolve(strict=True).relative_to(repository_root)
        return candidate.read_bytes()
    except ConfirmatoryHoldError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfirmatoryHoldError(
            f"local hash custody {relative} is unavailable from the working tree"
        ) from error


def _freeze_input_hashes(manifest_bytes: bytes) -> tuple[tuple[str, str], ...]:
    try:
        payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryHoldError("local hash custody manifest is invalid") from error
    inputs = payload.get("inputs") if type(payload) is dict else None
    if type(inputs) is not list:
        raise ConfirmatoryHoldError("local hash custody manifest inputs are invalid")
    verified: list[tuple[str, str]] = []
    for item in inputs:
        if type(item) is not dict:
            raise ConfirmatoryHoldError("local hash custody manifest input is invalid")
        path = item.get("path")
        digest = item.get("sha256")
        pure_path = PurePosixPath(path) if type(path) is str else None
        if (
            type(path) is not str
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or pure_path is None
            or pure_path.is_absolute()
            or "\\" in path
            or pure_path.as_posix() != path
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise ConfirmatoryHoldError("local hash custody manifest input is invalid")
        verified.append((path, digest))
    return tuple(verified)


def _opaque(prefix: str, *parts: object) -> str:
    return f"{prefix}_{sha256_ref(list(parts)).removeprefix('sha256:')[:16]}"


def _derived_seed(corpus_seed: int, *parts: object) -> int:
    return int(sha256_ref(["confirmatory", corpus_seed, *parts])[7:23], 16)


def _case_from_source(
    stage: Stage, source: DevelopmentCase, corpus_seed: int, ordinal: int
) -> ConfirmatoryCase:
    case_id = source.task_case.case_id
    public_projection = source.public_projection()
    public_projection["case_id"] = case_id
    public_projection["stage"] = stage
    public_hash = sha256_ref(public_projection)
    truth = source.hidden_truth
    hidden_projection = {
        "public_content_hash": public_hash,
        "authority_records": [record.model_dump(mode="json") for record in truth.authority_records],
        "authority_query": truth.authority_query.model_dump(mode="json"),
        "authority_class": truth.authority_class.value,
        "expected_decision": truth.expected_decision.value,
        "expected_disposition": truth.expected_disposition.value,
        "expected_reason_code": truth.expected_reason_code.value,
        "authority_set_hash": truth.authority_set_hash,
        "authority_query_hash": truth.authority_query_hash,
        "authority_decision_hash": truth.authority_decision_hash,
    }
    hidden_hash = sha256_ref(hidden_projection)
    return ConfirmatoryCase(
        stage=stage,
        case_id=case_id,
        source=source,
        public_content_hash=public_hash,
        hidden_content_hash=hidden_hash,
    )


def _build_case(
    stage: Stage, corpus_seed: int, domain: Domain, authority_class: AuthorityClass, ordinal: int
) -> ConfirmatoryCase:
    source = _draft_case(
        corpus_seed=_derived_seed(corpus_seed, stage, domain, authority_class.value, ordinal),
        domain=domain,
        authority_class=authority_class,
        ordinal=ordinal,
    )
    return _case_from_source(stage, source, corpus_seed, ordinal)


def _bundle(
    corpus_seed: int, domain: Domain, ratio: Decimal, ordinal: int
) -> ConfirmatoryBundleSeed:
    unsigned_seed = _derived_seed(corpus_seed, "bundle", domain, str(ratio), ordinal)
    seed = unsigned_seed if unsigned_seed < 2**63 else unsigned_seed - 2**64
    source_manifest = generate_bundle(domain, ratio, 12, seed).bundle.source_manifest
    bundle_id = source_manifest.bundle_id
    source_manifest_hash = hash_source_bundle_manifest(source_manifest)
    projection = {
        "binding_profile": "SSB-CONFIRMATORY-BUNDLE-SEED2",
        "domain": domain,
        "contamination_ratio": str(ratio),
        "seed": seed,
        "bundle_id": bundle_id,
        "source_manifest_hash": source_manifest_hash,
    }
    return ConfirmatoryBundleSeed(
        "SSB-CONFIRMATORY-BUNDLE-SEED2",
        domain,
        ratio,
        seed,
        bundle_id,
        source_manifest_hash,
        sha256_ref(projection),
    )


def _entity_ids(value: object, key: str | None = None) -> set[str]:
    if type(value) is dict:
        return set().union(*(_entity_ids(item, item_key) for item_key, item in value.items()))
    if type(value) is list:
        return set().union(*(_entity_ids(item, key) for item in value)) if value else set()
    if (
        type(value) is str
        and key is not None
        and (key == "case_id" or key == "world_id" or key.endswith("_id"))
        and _OPAQUE_ENTITY.fullmatch(value) is not None
    ):
        return {value}
    return set()


def _inventory_for(
    bundles: tuple[ConfirmatoryBundleSeed, ...],
    stage_a_cases: tuple[ConfirmatoryCase, ...],
    stage_b_cases: tuple[ConfirmatoryCase, ...],
) -> SplitInventory:
    cases = stage_a_cases + stage_b_cases
    entities = {bundle.bundle_id for bundle in bundles}
    for case in cases:
        entities.add(case.case_id)
        entities.update(_entity_ids(case.model_visible_projection()))
    return SplitInventory(
        seeds=frozenset([bundle.seed for bundle in bundles] + [case.seed for case in cases]),
        entity_ids=frozenset(entities),
        value_hashes=frozenset(
            [bundle.content_hash for bundle in bundles]
            + [case.public_content_hash for case in cases]
            + [case.hidden_content_hash for case in cases]
        ),
    )


def _assert_isolated(development: SplitInventory, confirmatory: SplitInventory) -> None:
    overlaps = {
        "seeds": development.seeds & confirmatory.seeds,
        "entity_ids": development.entity_ids & confirmatory.entity_ids,
        "value_hashes": development.value_hashes & confirmatory.value_hashes,
    }
    present = [name for name, values in overlaps.items() if values]
    if present:
        raise ConfirmatoryHoldError("development/confirmatory split overlap: " + ", ".join(present))


def inventory_from_development(corpus: DevelopmentCorpus) -> SplitInventory:
    """Derive the explicit development split inventory from an exact validated corpus."""

    if type(corpus) is not DevelopmentCorpus:
        raise ValueError("development corpus must be an exact DevelopmentCorpus")
    entities: set[str] = set()
    for case in corpus.cases:
        entities.add(case.case_id)
        entities.update(_entity_ids(case.model_visible_projection()))
    return SplitInventory(
        seeds=frozenset([corpus.seed] + [case.seed for case in corpus.cases]),
        entity_ids=frozenset(entities),
        value_hashes=frozenset(
            [corpus.content_hash]
            + [case.public_content_hash for case in corpus.cases]
            + [case.hidden_content_hash for case in corpus.cases]
        ),
    )


def _validate_layout(
    bundles: tuple[ConfirmatoryBundleSeed, ...],
    stage_a_cases: tuple[ConfirmatoryCase, ...],
    stage_b_cases: tuple[ConfirmatoryCase, ...],
) -> None:
    if len(bundles) != 30 or len(stage_a_cases) != 40 or len(stage_b_cases) != 50:
        raise ValueError("confirmatory corpus does not have preregistered total counts")
    if len({bundle.bundle_id for bundle in bundles}) != len(bundles):
        raise ValueError("confirmatory bundle ids must be unique")
    if len({case.case_id for case in stage_a_cases + stage_b_cases}) != 90:
        raise ValueError("confirmatory case ids must be unique across stages")
    for domain in DOMAINS:
        domain_bundles = [bundle for bundle in bundles if bundle.domain == domain]
        if len(domain_bundles) != STAGE_A_BUNDLES_PER_DOMAIN or Counter(
            bundle.contamination_ratio for bundle in domain_bundles
        ) != Counter({ratio: 3 for ratio in RATIOS}):
            raise ValueError("Stage A requires three bundle seeds per ratio and domain")
        domain_a = [case for case in stage_a_cases if case.domain == domain]
        domain_b = [case for case in stage_b_cases if case.domain == domain]
        if len(domain_a) != STAGE_A_CASES_PER_DOMAIN or len(domain_b) != STAGE_B_CASES_PER_DOMAIN:
            raise ValueError("confirmatory case counts per domain are invalid")
        if Counter(case.authority_class for case in domain_a) != Counter(
            {authority_class: 4 for authority_class in AUTHORITY_CLASSES}
        ):
            raise ValueError("Stage A authority-class balance is invalid")
        if Counter(case.authority_class for case in domain_b) != Counter(
            {authority_class: 5 for authority_class in AUTHORITY_CLASSES}
        ):
            raise ValueError("Stage B requires five cases per authority class and domain")


def generate_confirmatory_corpus(
    *,
    authorization: ConfirmatoryAuthorization,
    corpus_seed: int,
    development_inventory: SplitInventory,
) -> ConfirmatoryCorpus:
    """Build only from already validated custody, without Git, signing, or network I/O."""

    if type(authorization) is not ConfirmatoryAuthorization or type(corpus_seed) is not int:
        raise ConfirmatoryHoldError("validated authorization and exact corpus seed are required")
    if type(development_inventory) is not SplitInventory:
        raise ConfirmatoryHoldError("explicit development split inventory is required")
    bundles = tuple(
        _bundle(corpus_seed, domain, ratio, ordinal)
        for domain in DOMAINS
        for ratio in RATIOS
        for ordinal in range(3)
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
        for ordinal in range(5)
    )
    inventory = _inventory_for(bundles, stage_a, stage_b)
    _assert_isolated(development_inventory, inventory)
    projection = {
        "schema_version": "1.0",
        "freeze_manifest_hash": authorization.freeze.manifest_hash,
        "anchor_receipt_hash": authorization.receipt.receipt_hash,
        "corpus_seed": corpus_seed,
        "bundle_hashes": [bundle.content_hash for bundle in bundles],
        "stage_a_hidden_content_hashes": [case.hidden_content_hash for case in stage_a],
        "stage_b_hidden_content_hashes": [case.hidden_content_hash for case in stage_b],
    }
    return ConfirmatoryCorpus(
        "1.0",
        authorization,
        corpus_seed,
        bundles,
        stage_a,
        stage_b,
        inventory,
        sha256_ref(projection),
    )


def generate_confirmatory_from_files(
    *,
    manifest_path: Path,
    detached_sha256_path: Path,
    anchor_receipt_path: Path,
    repository_root: Path | None = None,
    corpus_seed: int,
    development_inventory: SplitInventory,
) -> ConfirmatoryCorpus:
    """Load only supplied local custody bytes, then run the same fail-closed gate."""

    try:
        paths = (manifest_path, detached_sha256_path, anchor_receipt_path)
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise OSError("custody path is not a regular file")
        authorization = authorize_confirmatory_generation(
            manifest_path.read_bytes(),
            detached_sha256_path.read_bytes(),
            anchor_receipt_path.read_bytes(),
            repository_root=repository_root,
        )
    except OSError as error:
        raise ConfirmatoryHoldError("confirmatory custody files are unavailable") from error
    return generate_confirmatory_corpus(
        authorization=authorization,
        corpus_seed=corpus_seed,
        development_inventory=development_inventory,
    )


def validate_confirmatory_corpus(corpus: ConfirmatoryCorpus) -> None:
    if type(corpus) is not ConfirmatoryCorpus:
        raise ValueError("corpus must be an exact ConfirmatoryCorpus")
    ConfirmatoryCorpus(
        corpus.schema_version,
        corpus.authorization,
        corpus.corpus_seed,
        corpus.bundles,
        corpus.stage_a_cases,
        corpus.stage_b_cases,
        corpus.inventory,
        corpus.content_hash,
    )


def load_model_visible_corpus(public_corpus_path: Path) -> dict[str, object]:
    """Load only the allowlisted public artifact; scorer custody is never accepted here."""

    try:
        if (
            public_corpus_path.name != "confirmatory_public_corpus.json"
            or public_corpus_path.is_symlink()
            or not public_corpus_path.is_file()
        ):
            raise OSError("public corpus path is invalid")
        payload = json.loads(public_corpus_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryHoldError("model-visible public corpus is unavailable") from error
    required = {"schema_version", "corpus_seed", "bundles", "stage_a_cases", "stage_b_cases"}
    if (
        type(payload) is not dict
        or set(payload) != required
        or payload.get("schema_version") != "1.0"
    ):
        raise ConfirmatoryHoldError("model-visible public corpus has an invalid boundary")
    forbidden_keys = {
        "case_truth",
        "hidden_content_hash",
        "authority_records",
        "authority_class",
        "expected_decision",
        "expected_disposition",
        "expected_reason_code",
        "authority_set_hash",
        "authority_query_hash",
        "authority_decision_hash",
    }

    def contains_scorer_truth(value: object) -> bool:
        if type(value) is dict:
            return any(
                type(key) is not str or key in forbidden_keys or contains_scorer_truth(item)
                for key, item in value.items()
            )
        if type(value) is list:
            return any(contains_scorer_truth(item) for item in value)
        return False

    if contains_scorer_truth(payload):
        raise ConfirmatoryHoldError("model-visible public corpus contains scorer-only truth")
    return cast(dict[str, object], payload)


@dataclass(frozen=True, slots=True)
class ConfirmatoryArtifactPaths:
    public_corpus: Path
    hidden_scorer_custody: Path
    manifest: Path
    public_sha256: Path
    hidden_sha256: Path
    manifest_sha256: Path


def write_confirmatory_artifacts(
    corpus: ConfirmatoryCorpus, output_dir: Path
) -> ConfirmatoryArtifactPaths:
    """Exclusively publish separate locked model-visible and scorer-only custody artifacts."""

    if type(corpus) is not ConfirmatoryCorpus:
        raise ConfirmatoryHoldError("only an exact validated confirmatory corpus may be written")
    validate_confirmatory_corpus(corpus)
    try:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise OSError("output directory is invalid")
        public_path = output_dir / "confirmatory_public_corpus.json"
        hidden_path = output_dir / "confirmatory_hidden_scorer_custody.json"
        manifest_path = output_dir / "confirmatory_corpus_manifest.json"
        public_sha256_path = output_dir / "confirmatory_public_corpus.sha256"
        hidden_sha256_path = output_dir / "confirmatory_hidden_scorer_custody.sha256"
        manifest_sha256_path = output_dir / "confirmatory_corpus_manifest.sha256"
        destinations = (
            public_path,
            hidden_path,
            manifest_path,
            public_sha256_path,
            hidden_sha256_path,
            manifest_sha256_path,
        )
        if any(path.exists() or path.is_symlink() for path in destinations):
            raise ConfirmatoryHoldError("refusing to overwrite confirmatory artifacts")
    except OSError as error:
        raise ConfirmatoryHoldError("confirmatory output directory is invalid") from error
    public_payload = canonical_json_bytes(corpus.public_artifact_payload())
    hidden_payload = canonical_json_bytes(corpus.hidden_scorer_payload())
    public_hash = sha256(public_payload).hexdigest()
    hidden_hash = sha256(hidden_payload).hexdigest()
    manifest_payload = canonical_json_bytes(
        {
            "schema_version": "1.0",
            "freeze_manifest_hash": corpus.authorization.freeze.manifest_hash,
            "anchor_receipt_hash": corpus.authorization.receipt.receipt_hash,
            "confirmatory_corpus_content_hash": corpus.content_hash,
            "public_corpus": {
                "path": public_path.name,
                "sha256": f"sha256:{public_hash}",
            },
            "hidden_scorer_custody": {
                "path": hidden_path.name,
                "sha256": f"sha256:{hidden_hash}",
            },
        }
    )
    manifest_hash = sha256(manifest_payload).hexdigest()
    outputs = (
        (public_path, public_payload),
        (hidden_path, hidden_payload),
        (manifest_path, manifest_payload),
        (
            public_sha256_path,
            f"{public_hash}  {public_path.name}\n".encode(),
        ),
        (
            hidden_sha256_path,
            f"{hidden_hash}  {hidden_path.name}\n".encode(),
        ),
        (
            manifest_sha256_path,
            f"{manifest_hash}  {manifest_path.name}\n".encode(),
        ),
    )
    staged: list[tuple[Path, Path]] = []
    written: list[Path] = []
    try:
        for destination, content in outputs:
            with NamedTemporaryFile(
                dir=output_dir, prefix=f".{destination.name}.", delete=False
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                staged.append((Path(temporary.name), destination))
        for temporary, destination in staged:
            os.link(temporary, destination)
            temporary.unlink()
            os.chmod(destination, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            written.append(destination)
    except OSError as error:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        for destination in written:
            try:
                os.chmod(destination, stat.S_IWUSR | stat.S_IRUSR)
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise ConfirmatoryHoldError(
            "could not atomically publish locked confirmatory artifacts"
        ) from error
    return ConfirmatoryArtifactPaths(
        public_path,
        hidden_path,
        manifest_path,
        public_sha256_path,
        hidden_sha256_path,
        manifest_sha256_path,
    )
