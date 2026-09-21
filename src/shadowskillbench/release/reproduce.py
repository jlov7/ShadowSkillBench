"""Offline sealed-artifact verification and deterministic practice rebuilds."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Literal, cast

from shadowskillbench.analysis import (
    SealedAnalysisHold,
    analyze_sealed_confirmatory_artifacts,
    sealed_analysis_receipt_matches,
)
from shadowskillbench.analysis.dataset import AnalysisDataset
from shadowskillbench.analysis.stage_a import StageAAnalysis, StageAInference, StageAModel
from shadowskillbench.analysis.stage_b import StageBAnalysis, StageBInference
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.corpus.confirmatory import (
    ConfirmatoryHoldError,
    authorize_confirmatory_generation,
)
from shadowskillbench.corpus.confirmatory_v4 import authorize_confirmatory_generation_v4
from shadowskillbench.protocol import (
    ClaimsReport,
    PreregistrationReport,
    validate_claims_ledger,
    validate_preregistration,
    validate_release_claims,
)
from shadowskillbench.protocol.scientific_freeze import (
    ScientificFreezeHold,
    derive_scientific_freeze_binding,
)
from shadowskillbench.protocol.scientific_freeze_v4 import (
    ScientificFreezeV4Hold,
    derive_scientific_freeze_binding_v4,
)
from shadowskillbench.reporting import (
    ProtocolEvidence,
    ReportEvidence,
    RepresentativeTrace,
    hash_analysis_dataset,
    render_report,
    render_workbench_json,
    report_status,
)
from shadowskillbench.reporting.selection import SelectionResult, SelectionStatus

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SEALED_PROFILE = "SSB-SEALED-ARTIFACTS-1"
_CONFIRMATORY_SEALED_PROFILE = "SSB-SEALED-ARTIFACTS-2"
_CONFIRMATORY_SEALED_PROFILE_V4 = "SSB-SEALED-ARTIFACTS-3"
_PRACTICE_PROFILE = "SSB-PRACTICE-REBUILD-1"
_CONFIRMATORY_PROFILE = "SSB-CONFIRMATORY-REPRODUCTION1"
_CONFIRMATORY_PROFILE_WITH_CLAIMS = "SSB-CONFIRMATORY-REPRODUCTION2"
_CONFIRMATORY_PROFILE_V4 = "SSB-CONFIRMATORY-REPRODUCTION4"
_CONFIRMATORY_PROFILE_V4_WITH_CLAIMS = "SSB-CONFIRMATORY-REPRODUCTION5"
_VERSION = "1.0"
_DEFAULT_MANIFEST = "artifacts/release/sealed_artifacts_manifest.json"

type ScientificProfile = Literal["V3", "V4"]


class ReproductionHold(ValueError):
    """A truthful, named refusal to reproduce an unsealed release."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    classification: str
    report_sha256: str
    workbench_sha256: str
    output_paths: tuple[Path, ...]


def _hash(data: bytes) -> str:
    return f"sha256:{sha256(data).hexdigest()}"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ReproductionHold("HOLD_DUPLICATE_JSON_KEY", f"{label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReproductionHold("HOLD_INVALID_JSON", label) from error
    if type(value) is not dict:
        raise ReproductionHold("HOLD_INVALID_SCHEMA", f"{label} must be an object")
    return cast(dict[str, object], value)


def _relative(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise ReproductionHold("HOLD_UNSAFE_PATH", label)
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReproductionHold("HOLD_UNSAFE_PATH", label)
    return value


def _safe_file(root: Path, relative: object, *, label: str) -> Path:
    value = _relative(relative, label=label)
    candidate = root.joinpath(*PurePosixPath(value).parts)
    try:
        current = root
        for part in PurePosixPath(value).parts:
            current = current / part
            if current.is_symlink():
                raise ReproductionHold("HOLD_SYMLINK_PATH", value)
        if not candidate.is_file() or candidate.is_symlink():
            raise ReproductionHold("HOLD_MISSING_SEALED_INPUT", value)
        candidate.resolve(strict=True).relative_to(root)
        return candidate
    except ReproductionHold:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ReproductionHold("HOLD_INPUT_IO", value) from error


def _require_sha(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ReproductionHold("HOLD_INVALID_HASH", label)
    return value


def _exact(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ReproductionHold("HOLD_INVALID_SCHEMA", label)
    return cast(dict[str, object], value)


def _read_hashed(root: Path, path: object, digest: object, *, label: str) -> bytes:
    candidate = _safe_file(root, path, label=label)
    try:
        data = candidate.read_bytes()
    except OSError as error:
        raise ReproductionHold("HOLD_INPUT_IO", str(path)) from error
    if _hash(data) != _require_sha(digest, label=label):
        raise ReproductionHold("HOLD_HASH_MISMATCH", str(path))
    return data


def _protocol_relative(root: Path, protocol_path: Path) -> str:
    try:
        candidate = protocol_path if protocol_path.is_absolute() else root / protocol_path
        return _relative(candidate.relative_to(root).as_posix(), label="protocol")
    except ValueError as error:
        raise ReproductionHold(
            "HOLD_UNSAFE_PATH", "protocol must be repository-relative"
        ) from error


def _validate_protocol(root: Path, protocol_path: Path) -> str:
    relative = _protocol_relative(root, protocol_path)
    candidate = _safe_file(root, relative, label="protocol")
    try:
        data = candidate.read_bytes()
    except OSError as error:
        raise ReproductionHold("HOLD_INPUT_IO", relative) from error
    payload = _json_object(data, label="protocol")
    if data != _canonical_bytes(payload):
        raise ReproductionHold("HOLD_NONCANONICAL_JSON", "protocol")
    _exact(
        payload,
        {"anchor_status", "inputs", "preregistration_core", "referents", "schema_version"},
        label="protocol",
    )
    if payload["schema_version"] != _VERSION:
        raise ReproductionHold("HOLD_INVALID_PROTOCOL", "unsupported protocol schema")
    if payload["anchor_status"] not in {
        "PENDING_HUMAN_ANCHOR",
        "PENDING_CUSTODY_RECEIPT",
    }:
        raise ReproductionHold("HOLD_INVALID_PROTOCOL", "practice protocol anchor status")
    if type(payload["inputs"]) is not list or type(payload["referents"]) is not list:
        raise ReproductionHold("HOLD_INVALID_PROTOCOL", "inputs and referents must be lists")
    if not payload["inputs"]:
        raise ReproductionHold("HOLD_INVALID_PROTOCOL", "practice protocol needs frozen inputs")
    input_paths: set[str] = set()
    input_hashes: dict[str, str] = {}
    for item in payload["inputs"]:
        record = _exact(item, {"path", "role", "sha256"}, label="protocol input")
        path = _relative(record["path"], label="protocol input")
        if path in input_paths:
            raise ReproductionHold("HOLD_DUPLICATE_REFERENCE", "protocol input path")
        input_paths.add(path)
        if type(record["role"]) is not str or not cast(str, record["role"]).strip():
            raise ReproductionHold("HOLD_INVALID_PROTOCOL", "protocol input role")
        _read_hashed(root, path, record["sha256"], label="protocol input")
        input_hashes[path] = _require_sha(record["sha256"], label="protocol input")
    preregistration = _exact(
        payload["preregistration_core"], {"path", "sha256"}, label="preregistration core"
    )
    preregistration_path = _relative(preregistration["path"], label="preregistration core")
    if input_hashes.get(preregistration_path) != _require_sha(
        preregistration["sha256"], label="preregistration core"
    ):
        raise ReproductionHold("HOLD_PROTOCOL_REF_MISMATCH", "preregistration core")
    references: set[str] = set()
    for item in payload["referents"]:
        referent = _exact(item, {"reference", "paths"}, label="protocol referent")
        reference = _require_sha(referent["reference"], label="protocol referent")
        paths = referent["paths"]
        if reference in references or type(paths) is not list or not paths:
            raise ReproductionHold("HOLD_DUPLICATE_REFERENCE", "protocol referent")
        resolved_paths = tuple(_relative(path, label="protocol referent") for path in paths)
        if len(resolved_paths) != len(set(resolved_paths)) or any(
            path not in input_hashes for path in resolved_paths
        ):
            raise ReproductionHold("HOLD_PROTOCOL_REF_MISMATCH", "protocol referent")
        references.add(reference)
    return _hash(data)


def _confirmatory_protocol_evidence(
    root: Path,
    protocol_path: Path,
    anchor_receipt: dict[str, object],
) -> ProtocolEvidence:
    receipt_bytes = _read_hashed(
        root,
        anchor_receipt["path"],
        anchor_receipt["sha256"],
        label="anchor receipt",
    )
    try:
        manifest_bytes = _safe_file(
            root, _protocol_relative(root, protocol_path), label="protocol"
        ).read_bytes()
        detached = _safe_file(
            root, "protocol/freeze_manifest.sha256", label="detached manifest"
        ).read_bytes()
        authorization = authorize_confirmatory_generation(
            manifest_bytes,
            detached,
            receipt_bytes,
            repository_root=root,
        )
        manifest = _json_object(authorization.freeze.manifest_bytes, label="protocol")
    except (ConfirmatoryHoldError, OSError, ValueError) as error:
        raise ReproductionHold(
            "HOLD_MISSING_CONFIRMATORY_AUTHORIZATION", "protocol custody"
        ) from error
    prompts: dict[str, str] = {}
    inputs = manifest.get("inputs")
    if type(inputs) is not list:
        raise ReproductionHold("HOLD_INVALID_CONFIRMATORY_AUTHORIZATION", "freeze inputs")
    roles = {"compiler_prompt", "executor_system_prompt"}
    role_aliases = {
        "compiler_prompt": "compiler_prompt",
        "confirmatory_compiler_prompt": "compiler_prompt",
        "executor_system_prompt": "executor_system_prompt",
    }
    for entry in inputs:
        if type(entry) is dict:
            source_role = entry.get("role")
            role = role_aliases.get(source_role) if type(source_role) is str else None
            if role is None:
                continue
            digest = entry.get("sha256")
            if type(digest) is not str or role in prompts:
                raise ReproductionHold("HOLD_INVALID_CONFIRMATORY_AUTHORIZATION", "prompt inputs")
            prompts[role] = digest
    if set(prompts) != roles:
        raise ReproductionHold("HOLD_MISSING_CONFIRMATORY_PROMPT_BINDING", "freeze inputs")
    return ProtocolEvidence(
        protocol_tag=authorization.receipt.signed_tag,
        freeze_manifest_sha256=authorization.freeze.manifest_hash,
        prompt_hashes=tuple(prompts[role] for role in sorted(roles)),
        code_commit=authorization.receipt.freeze_commit,
        reproduce_command=(
            "uv run shadowskillbench reproduce --protocol protocol/freeze_manifest.json"
        ),
        external_anchor_locator=(
            authorization.receipt.immutable_locator
            if authorization.receipt.externally_verified
            else None
        ),
        custody_locator=authorization.receipt.immutable_locator,
        custody_mode=authorization.receipt.custody_mode,
        anchor_receipt_sha256=authorization.receipt.receipt_hash,
        anchor_verified=authorization.receipt.verification_result in {"VERIFIED", "VERIFIED_LOCAL"},
    )


def _confirmatory_protocol_evidence_v4(
    root: Path,
    protocol_path: Path,
    anchor_receipt: dict[str, object],
) -> ProtocolEvidence:
    """Bind a V4 custody receipt to its V4-only manifest and detached digest."""

    relative = _protocol_relative(root, protocol_path)
    if relative != "protocol/freeze_manifest.v4.json":
        raise ReproductionHold("HOLD_CROSS_PROFILE_REPRODUCTION_MISMATCH", "V4 protocol path")
    receipt_bytes = _read_hashed(
        root,
        anchor_receipt["path"],
        anchor_receipt["sha256"],
        label="V4 anchor receipt",
    )
    try:
        manifest_bytes = _safe_file(root, relative, label="V4 protocol").read_bytes()
        detached = _safe_file(
            root, "protocol/freeze_manifest.v4.sha256", label="V4 detached manifest"
        ).read_bytes()
        authorization = authorize_confirmatory_generation_v4(
            manifest_bytes, detached, receipt_bytes
        )
        manifest = _json_object(manifest_bytes, label="V4 protocol")
    except (ConfirmatoryHoldError, OSError, ValueError) as error:
        raise ReproductionHold(
            "HOLD_MISSING_V4_CONFIRMATORY_AUTHORIZATION", "protocol custody"
        ) from error
    inputs = manifest.get("inputs")
    if type(inputs) is not list:
        raise ReproductionHold("HOLD_INVALID_V4_CONFIRMATORY_AUTHORIZATION", "freeze inputs")
    hashes: list[str] = []
    for item in inputs:
        if type(item) is not dict or type(item.get("sha256")) is not str:
            raise ReproductionHold("HOLD_INVALID_V4_CONFIRMATORY_AUTHORIZATION", "freeze inputs")
        digest = item["sha256"]
        assert type(digest) is str
        hashes.append(digest)
    if (
        not hashes
        or len(hashes) != len(set(hashes))
        or any(_SHA256.fullmatch(value) is None for value in hashes)
    ):
        raise ReproductionHold("HOLD_INVALID_V4_CONFIRMATORY_AUTHORIZATION", "freeze inputs")
    return ProtocolEvidence(
        protocol_tag="SSB-PROTOCOL-FREEZE-4",
        freeze_manifest_sha256=authorization.freeze_manifest_hash,
        prompt_hashes=tuple(hashes),
        code_commit="0000000",
        reproduce_command=(
            "uv run shadowskillbench reproduce --protocol protocol/freeze_manifest.v4.json"
        ),
        custody_locator=f"v4-local-custody:{authorization.anchor_receipt_hash}",
        custody_mode="LOCAL_HASH_CUSTODY",
        anchor_receipt_sha256=authorization.anchor_receipt_hash,
        anchor_verified=True,
    )


def _selected_scientific_profile(
    protocol_path: Path, selected: ScientificProfile | None
) -> ScientificProfile:
    if selected is not None:
        if selected not in {"V3", "V4"}:
            raise ReproductionHold("HOLD_INVALID_SCIENTIFIC_PROFILE", "selected profile")
        return selected
    return "V4" if protocol_path.name == "freeze_manifest.v4.json" else "V3"


def _confirmatory_evidence(
    root: Path,
    protocol_path: Path,
    protocol_hash: str,
    anchor_receipt: dict[str, object],
    source: dict[str, object],
    scientific_profile: ScientificProfile = "V3",
) -> ReportEvidence:
    if scientific_profile == "V4":
        return _confirmatory_evidence_v4(root, protocol_path, protocol_hash, anchor_receipt, source)
    profile = source.get("profile")
    source_fields = {
        "profile",
        "schema_version",
        "classification",
        "artifacts_root",
        "analysis_receipt",
        "limitations",
    }
    if profile == _CONFIRMATORY_PROFILE_WITH_CLAIMS:
        source_fields.add("claims_ledger")
    elif profile != _CONFIRMATORY_PROFILE:
        raise ReproductionHold("HOLD_INVALID_CONFIRMATORY_SOURCE", "profile")
    source = _exact(
        source,
        source_fields,
        label="confirmatory rebuild source",
    )
    if source["schema_version"] != _VERSION or source["classification"] != "CONFIRMATORY":
        raise ReproductionHold("HOLD_INVALID_CONFIRMATORY_SOURCE", "profile, version, or class")
    artifacts_root = _relative(source["artifacts_root"], label="confirmatory artifacts root")
    protocol = _confirmatory_protocol_evidence(root, protocol_path, anchor_receipt)
    if protocol.freeze_manifest_sha256 != protocol_hash:
        raise ReproductionHold("HOLD_PROTOCOL_REF_MISMATCH", "confirmatory protocol")
    try:
        manifest_bytes = _safe_file(
            root, _protocol_relative(root, protocol_path), label="protocol"
        ).read_bytes()
        detached_bytes = _safe_file(
            root, "protocol/freeze_manifest.sha256", label="detached manifest"
        ).read_bytes()
        authorization = authorize_confirmatory_generation(
            manifest_bytes,
            detached_bytes,
            _read_hashed(
                root,
                anchor_receipt["path"],
                anchor_receipt["sha256"],
                label="anchor receipt",
            ),
            repository_root=root,
        )
        scientific_freeze = derive_scientific_freeze_binding(
            manifest_bytes=authorization.freeze.manifest_bytes,
            repository_root=root,
        )
    except ScientificFreezeHold as error:
        raise ReproductionHold(error.code, str(error)) from error
    except (ConfirmatoryHoldError, OSError, ValueError) as error:
        raise ReproductionHold(
            "HOLD_MISSING_CONFIRMATORY_AUTHORIZATION", "protocol custody"
        ) from error
    if scientific_freeze.manifest_hash != protocol.freeze_manifest_sha256:
        raise ReproductionHold("HOLD_SCIENTIFIC_FREEZE_MANIFEST_MISMATCH", "protocol custody")
    receipt = _exact(source["analysis_receipt"], {"path", "sha256"}, label="analysis receipt")
    receipt_data = _read_hashed(root, receipt["path"], receipt["sha256"], label="analysis receipt")
    receipt_payload = _json_object(receipt_data, label="analysis receipt")
    if receipt_data != _canonical_bytes(receipt_payload):
        raise ReproductionHold("HOLD_NONCANONICAL_ANALYSIS_RECEIPT", str(receipt["path"]))
    limitations = source["limitations"]
    if (
        type(limitations) is not list
        or not limitations
        or any(type(item) is not str or not item.strip() for item in limitations)
    ):
        raise ReproductionHold("HOLD_INVALID_CONFIRMATORY_SOURCE", "limitations")
    try:
        previous = Path.cwd()
        os.chdir(root)
        try:
            sealed = analyze_sealed_confirmatory_artifacts(
                Path(artifacts_root), scientific_freeze=scientific_freeze
            )
        finally:
            os.chdir(previous)
    except SealedAnalysisHold as error:
        raise ReproductionHold(error.args[0].split(":", 1)[0], str(error)) from error
    if not sealed_analysis_receipt_matches(receipt_payload, sealed):
        raise ReproductionHold("HOLD_CROSS_HASH_ANALYSIS_MISMATCH", "analysis receipt")
    if (
        sealed.runtime_binding.freeze_manifest_hash != protocol.freeze_manifest_sha256
        or sealed.runtime_binding.anchor_receipt_hash != protocol.anchor_receipt_sha256
        or sealed.runtime_binding.plan_hash != sealed.audit.plan_hash
    ):
        raise ReproductionHold("HOLD_RUNTIME_PROTOCOL_BINDING_MISMATCH", "runtime binding")
    preregistration = validate_preregistration(root / "protocol/preregistration.md")
    claims_path = root / "protocol/CLAIMS_LEDGER.yaml"
    if profile == _CONFIRMATORY_PROFILE_WITH_CLAIMS:
        claims_record = _exact(source["claims_ledger"], {"path", "sha256"}, label="claims ledger")
        _read_hashed(
            root,
            claims_record["path"],
            claims_record["sha256"],
            label="claims ledger",
        )
        claims_path = _safe_file(root, claims_record["path"], label="claims ledger")
    claims = validate_claims_ledger(claims_path, root)
    if not preregistration.valid or preregistration.core is None:
        raise ReproductionHold("HOLD_PREREGISTRATION_INVALID", "protocol/preregistration.md")
    if not claims.valid:
        raise ReproductionHold("HOLD_CLAIMS_LEDGER_INVALID", str(claims_path.relative_to(root)))
    protocol = replace(
        protocol,
        experiment_plan_sha256=sealed.audit.plan_hash,
        experiment_runtime_binding_sha256=sha256_ref(sealed.runtime_binding.projection()),
        experiment_audit_report_sha256=sealed.audit.report_hash,
        experiment_audit_status="PASS",
    )
    held_a = RepresentativeTrace(
        "A", SelectionResult("A", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
    )
    held_b = RepresentativeTrace(
        "B", SelectionResult("B", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
    )
    evidence = ReportEvidence(
        dataset=sealed.dataset,
        stage_a=sealed.stage_a,
        stage_b=sealed.stage_b,
        preregistration=preregistration,
        claims=claims,
        protocol=protocol,
        analysis_dataset_sha256=sealed.dataset_sha256,
        representative_traces=(held_a, held_b),
        limitations=tuple(cast(list[str], limitations)),
        exclusion_policy=sealed.exclusion_policy,
        runtime_binding=sealed.runtime_binding,
    )
    if report_status(evidence).value != "confirmatory":
        raise ReproductionHold("HOLD_REPORT_NOT_CONFIRMATORY", "sealed report evidence")
    gate = validate_release_claims(
        claims,
        report_status=report_status(evidence).value,
        experiment_audit_status="PASS",
        experiment_audit_hash=sealed.audit.report_hash,
        analysis_dataset_hash=sealed.dataset_sha256,
        limitations=tuple(cast(list[str], limitations)),
        sealed_analysis=sealed,
        scientific_freeze=scientific_freeze,
    )
    if not gate.passed:
        raise ReproductionHold(gate.status.value, "claims release gate")
    return evidence


def _confirmatory_evidence_v4(
    root: Path,
    protocol_path: Path,
    protocol_hash: str,
    anchor_receipt: dict[str, object],
    source: dict[str, object],
) -> ReportEvidence:
    """Rebuild a V4 release only from V4-bound custody and analysis inputs."""

    source_fields = {
        "profile",
        "schema_version",
        "classification",
        "artifacts_root",
        "analysis_receipt",
        "limitations",
    }
    profile = source.get("profile")
    if profile == _CONFIRMATORY_PROFILE_V4_WITH_CLAIMS:
        source_fields.add("claims_ledger")
    elif profile != _CONFIRMATORY_PROFILE_V4:
        raise ReproductionHold("HOLD_INVALID_V4_CONFIRMATORY_SOURCE", "profile")
    source = _exact(source, source_fields, label="V4 confirmatory rebuild source")
    if source["schema_version"] != "4.0" or source["classification"] != "CONFIRMATORY":
        raise ReproductionHold("HOLD_INVALID_V4_CONFIRMATORY_SOURCE", "profile, version, or class")
    artifacts_root = _relative(source["artifacts_root"], label="V4 confirmatory artifacts root")
    protocol = _confirmatory_protocol_evidence_v4(root, protocol_path, anchor_receipt)
    if protocol.freeze_manifest_sha256 != protocol_hash:
        raise ReproductionHold("HOLD_PROTOCOL_REF_MISMATCH", "V4 confirmatory protocol")
    try:
        manifest_bytes = _safe_file(
            root, "protocol/freeze_manifest.v4.json", label="V4 protocol"
        ).read_bytes()
        scientific_freeze = derive_scientific_freeze_binding_v4(
            manifest_bytes=manifest_bytes, repository_root=root
        )
    except ScientificFreezeV4Hold as error:
        raise ReproductionHold(error.code, str(error)) from error
    if scientific_freeze.manifest_hash != protocol.freeze_manifest_sha256:
        raise ReproductionHold("HOLD_V4_SCIENTIFIC_FREEZE_MANIFEST_MISMATCH", "protocol custody")
    receipt = _exact(source["analysis_receipt"], {"path", "sha256"}, label="V4 analysis receipt")
    receipt_data = _read_hashed(
        root, receipt["path"], receipt["sha256"], label="V4 analysis receipt"
    )
    receipt_payload = _json_object(receipt_data, label="V4 analysis receipt")
    if receipt_data != _canonical_bytes(receipt_payload):
        raise ReproductionHold("HOLD_NONCANONICAL_ANALYSIS_RECEIPT", str(receipt["path"]))
    limitations = source["limitations"]
    if (
        type(limitations) is not list
        or not limitations
        or any(type(item) is not str or not item.strip() for item in limitations)
    ):
        raise ReproductionHold("HOLD_INVALID_V4_CONFIRMATORY_SOURCE", "limitations")
    try:
        previous = Path.cwd()
        os.chdir(root)
        try:
            sealed = analyze_sealed_confirmatory_artifacts(
                Path(artifacts_root), scientific_freeze=scientific_freeze
            )
        finally:
            os.chdir(previous)
    except SealedAnalysisHold as error:
        raise ReproductionHold(error.args[0].split(":", 1)[0], str(error)) from error
    if not sealed_analysis_receipt_matches(receipt_payload, sealed):
        raise ReproductionHold("HOLD_CROSS_HASH_ANALYSIS_MISMATCH", "V4 analysis receipt")
    if (
        sealed.runtime_binding.freeze_manifest_hash != protocol.freeze_manifest_sha256
        or sealed.runtime_binding.anchor_receipt_hash != protocol.anchor_receipt_sha256
        or sealed.runtime_binding.plan_hash != sealed.audit.plan_hash
    ):
        raise ReproductionHold("HOLD_RUNTIME_PROTOCOL_BINDING_MISMATCH", "V4 runtime binding")
    preregistration = validate_preregistration(root / "protocol/preregistration.md")
    claims_path = root / "protocol/CLAIMS_LEDGER.yaml"
    if profile == _CONFIRMATORY_PROFILE_V4_WITH_CLAIMS:
        claims_record = _exact(source["claims_ledger"], {"path", "sha256"}, label="claims ledger")
        _read_hashed(root, claims_record["path"], claims_record["sha256"], label="claims ledger")
        claims_path = _safe_file(root, claims_record["path"], label="claims ledger")
    claims = validate_claims_ledger(claims_path, root)
    if not preregistration.valid or preregistration.core is None:
        raise ReproductionHold("HOLD_PREREGISTRATION_INVALID", "protocol/preregistration.md")
    if not claims.valid:
        raise ReproductionHold("HOLD_CLAIMS_LEDGER_INVALID", str(claims_path.relative_to(root)))
    protocol = replace(
        protocol,
        experiment_plan_sha256=sealed.audit.plan_hash,
        experiment_runtime_binding_sha256=sha256_ref(sealed.runtime_binding.projection()),
        experiment_audit_report_sha256=sealed.audit.report_hash,
        experiment_audit_status="PASS",
    )
    held_a = RepresentativeTrace(
        "A", SelectionResult("A", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
    )
    held_b = RepresentativeTrace(
        "B", SelectionResult("B", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
    )
    evidence = ReportEvidence(
        dataset=sealed.dataset,
        stage_a=sealed.stage_a,
        stage_b=sealed.stage_b,
        preregistration=preregistration,
        claims=claims,
        protocol=protocol,
        analysis_dataset_sha256=sealed.dataset_sha256,
        representative_traces=(held_a, held_b),
        limitations=tuple(cast(list[str], limitations)),
        exclusion_policy=sealed.exclusion_policy,
        runtime_binding=sealed.runtime_binding,
        scientific_freeze=scientific_freeze,
    )
    if report_status(evidence).value != "confirmatory":
        raise ReproductionHold("HOLD_REPORT_NOT_CONFIRMATORY", "V4 sealed report evidence")
    gate = validate_release_claims(
        claims,
        report_status=report_status(evidence).value,
        experiment_audit_status="PASS",
        experiment_audit_hash=sealed.audit.report_hash,
        analysis_dataset_hash=sealed.dataset_sha256,
        limitations=tuple(cast(list[str], limitations)),
        sealed_analysis=sealed,
        scientific_freeze=scientific_freeze,
    )
    if not gate.passed:
        raise ReproductionHold(gate.status.value, "V4 claims release gate")
    return evidence


def _practice_evidence(source: dict[str, object], protocol_hash: str) -> ReportEvidence:
    source = _exact(
        source,
        {"profile", "schema_version", "classification", "protocol", "limitations"},
        label="practice rebuild source",
    )
    if (
        source["profile"] != _PRACTICE_PROFILE
        or source["schema_version"] != _VERSION
        or source["classification"] != "PRACTICE_NOT_EVIDENCE"
    ):
        raise ReproductionHold(
            "HOLD_INVALID_PRACTICE_SOURCE", "profile, version, or classification"
        )
    protocol = _exact(
        source["protocol"],
        {
            "protocol_tag",
            "freeze_manifest_sha256",
            "prompt_hashes",
            "code_commit",
            "reproduce_command",
        },
        label="practice protocol",
    )
    if protocol["freeze_manifest_sha256"] != protocol_hash:
        raise ReproductionHold("HOLD_PROTOCOL_REF_MISMATCH", "practice source freeze manifest")
    prompts = protocol["prompt_hashes"]
    if type(prompts) is not list or not prompts:
        raise ReproductionHold("HOLD_INVALID_PRACTICE_SOURCE", "prompt_hashes")
    prompt_hashes = tuple(_require_sha(value, label="prompt_hashes") for value in prompts)
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise ReproductionHold("HOLD_DUPLICATE_REFERENCE", "prompt_hashes")
    for field in ("protocol_tag", "code_commit", "reproduce_command"):
        if type(protocol[field]) is not str or not cast(str, protocol[field]).strip():
            raise ReproductionHold("HOLD_INVALID_PRACTICE_SOURCE", field)
    limitations = source["limitations"]
    if (
        type(limitations) is not list
        or not limitations
        or any(type(item) is not str or not item.strip() for item in limitations)
    ):
        raise ReproductionHold("HOLD_INVALID_PRACTICE_SOURCE", "limitations")
    dataset = AnalysisDataset(rows=(), exclusions=(), aggregates=())
    stage_a = StageAAnalysis(
        raw_counts=(),
        response_curves=(),
        estimands=(),
        average_marginal_effects=(),
        model=StageAModel("practice", "binomial", "logit", (), (), (), 0, 0, 0, False),
        inference=StageAInference(0, 0, 0.95, "PRACTICE_NOT_EVIDENCE"),
    )
    return ReportEvidence(
        dataset=dataset,
        stage_a=stage_a,
        stage_b=StageBAnalysis(
            (), (), (), (), (), (), StageBInference(0, 0, 0.95, "PRACTICE_NOT_EVIDENCE")
        ),
        preregistration=PreregistrationReport(
            source_path=None, valid=False, core=None, violations=()
        ),
        claims=ClaimsReport(
            ledger_path=Path("protocol/CLAIMS_LEDGER.yaml"),
            version=None,
            valid=False,
            claims=(),
            violations=(),
            renderable_findings=(),
        ),
        protocol=ProtocolEvidence(
            protocol_tag=cast(str, protocol["protocol_tag"]),
            freeze_manifest_sha256=protocol_hash,
            prompt_hashes=prompt_hashes,
            code_commit=cast(str, protocol["code_commit"]),
            reproduce_command=cast(str, protocol["reproduce_command"]),
        ),
        analysis_dataset_sha256=hash_analysis_dataset(dataset),
        representative_traces=(
            RepresentativeTrace(
                "A", SelectionResult("A", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
            ),
            RepresentativeTrace(
                "B", SelectionResult("B", SelectionStatus.HOLD_NO_CANDIDATE, None, ())
            ),
        ),
        limitations=tuple(cast(list[str], limitations)),
    )


def _publish_exclusive(destination: Path, data: bytes) -> None:
    if destination.exists() or destination.is_symlink():
        raise ReproductionHold("HOLD_OUTPUT_EXISTS", str(destination))
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ReproductionHold("HOLD_OUTPUT_DIRECTORY", str(destination.parent))
    with NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise ReproductionHold("HOLD_OUTPUT_EXISTS", str(destination)) from error
    finally:
        temporary.unlink(missing_ok=True)


def reproduce_sealed_artifacts(
    repository_root: Path,
    protocol_path: Path,
    artifacts_manifest_path: Path | None = None,
    *,
    scientific_profile: ScientificProfile | None = None,
) -> ReproductionResult:
    """Verify sealed local inputs and rebuild report/workbench outputs without network access."""
    try:
        root = repository_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ReproductionHold("HOLD_REPOSITORY_ROOT", "repository root is inaccessible") from error
    if root.is_symlink() or not root.is_dir():
        raise ReproductionHold("HOLD_REPOSITORY_ROOT", "repository root is invalid")
    if artifacts_manifest_path is None:
        manifest_relative = _DEFAULT_MANIFEST
    else:
        try:
            manifest_candidate = (
                artifacts_manifest_path
                if artifacts_manifest_path.is_absolute()
                else root / artifacts_manifest_path
            )
            manifest_relative = _relative(
                manifest_candidate.relative_to(root).as_posix(), label="artifacts manifest"
            )
        except ValueError as error:
            raise ReproductionHold("HOLD_UNSAFE_PATH", "artifacts manifest") from error
    manifest_path = _safe_file(root, manifest_relative, label="artifacts manifest")
    try:
        manifest_data = manifest_path.read_bytes()
    except OSError as error:
        raise ReproductionHold("HOLD_INPUT_IO", manifest_relative) from error
    manifest = _json_object(manifest_data, label="sealed artifacts manifest")
    if manifest_data != _canonical_bytes(manifest):
        raise ReproductionHold("HOLD_NONCANONICAL_JSON", "sealed artifacts manifest")
    profile = manifest.get("profile")
    selected_profile = _selected_scientific_profile(protocol_path, scientific_profile)
    if profile == _SEALED_PROFILE:
        if selected_profile != "V3":
            raise ReproductionHold("HOLD_CROSS_PROFILE_REPRODUCTION_MISMATCH", "practice profile")
        manifest = _exact(
            manifest,
            {
                "profile",
                "schema_version",
                "classification",
                "protocol_manifest_sha256",
                "sources",
                "outputs",
            },
            label="sealed artifacts manifest",
        )
        protocol_hash = _validate_protocol(root, protocol_path)
        if manifest["schema_version"] != _VERSION:
            raise ReproductionHold("HOLD_INVALID_SCHEMA", "sealed artifacts manifest profile")
        if manifest["classification"] != "PRACTICE_NOT_EVIDENCE":
            raise ReproductionHold("HOLD_MISSING_ANCHOR_RECEIPT", "practice manifest class")
        expected_role = "practice_rebuild"
        anchor_receipt = None
    elif profile in {_CONFIRMATORY_SEALED_PROFILE, _CONFIRMATORY_SEALED_PROFILE_V4}:
        manifest = _exact(
            manifest,
            {
                "profile",
                "schema_version",
                "classification",
                "protocol_manifest_sha256",
                "anchor_receipt",
                "sources",
                "outputs",
            },
            label="confirmatory sealed artifacts manifest",
        )
        expected_schema_version = "4.0" if profile == _CONFIRMATORY_SEALED_PROFILE_V4 else _VERSION
        expected_profile = "V4" if profile == _CONFIRMATORY_SEALED_PROFILE_V4 else "V3"
        if selected_profile != expected_profile:
            raise ReproductionHold("HOLD_CROSS_PROFILE_REPRODUCTION_MISMATCH", "sealed profile")
        if (
            manifest["schema_version"] != expected_schema_version
            or manifest["classification"] != "CONFIRMATORY"
        ):
            raise ReproductionHold("HOLD_INVALID_SCHEMA", "confirmatory sealed artifacts manifest")
        protocol_hash = _hash(
            _safe_file(root, _protocol_relative(root, protocol_path), label="protocol").read_bytes()
        )
        anchor_receipt = _exact(
            manifest["anchor_receipt"], {"path", "sha256"}, label="anchor receipt"
        )
        expected_role = "confirmatory_rebuild"
    else:
        raise ReproductionHold("HOLD_INVALID_SCHEMA", "sealed artifacts manifest profile")
    if manifest["protocol_manifest_sha256"] != protocol_hash:
        raise ReproductionHold("HOLD_PROTOCOL_REF_MISMATCH", "sealed artifacts manifest")
    sources = manifest["sources"]
    outputs = manifest["outputs"]
    if (
        type(sources) is not list
        or len(sources) != 1
        or type(outputs) is not list
        or len(outputs) != 2
    ):
        raise ReproductionHold("HOLD_INVALID_SCHEMA", "sealed source/output cardinality")
    source = _exact(sources[0], {"path", "sha256", "role"}, label="sealed source")
    if source["role"] != expected_role:
        raise ReproductionHold("HOLD_INVALID_SCHEMA", "sealed source role")
    source_data = _read_hashed(root, source["path"], source["sha256"], label="rebuild source")
    source_payload = _json_object(source_data, label="rebuild source")
    if anchor_receipt is None:
        evidence = _practice_evidence(source_payload, protocol_hash)
    else:
        evidence = _confirmatory_evidence(
            root,
            protocol_path,
            protocol_hash,
            anchor_receipt,
            source_payload,
            selected_profile,
        )
    rebuilt = {
        "report_html": render_report(evidence).encode(),
        "workbench_json": render_workbench_json(evidence),
    }
    expected: dict[str, tuple[Path, str]] = {}
    output_paths: set[Path] = set()
    for record in outputs:
        output = _exact(record, {"kind", "path", "sha256"}, label="sealed output")
        kind = output["kind"]
        if kind not in rebuilt or kind in expected:
            raise ReproductionHold("HOLD_INVALID_SCHEMA", "sealed output kind")
        path = _safe_file_parent(root, output["path"])
        if path in output_paths:
            raise ReproductionHold("HOLD_DUPLICATE_REFERENCE", "sealed output path")
        output_paths.add(path)
        expected[cast(str, kind)] = (path, _require_sha(output["sha256"], label="sealed output"))
    if set(expected) != set(rebuilt):
        raise ReproductionHold("HOLD_INVALID_SCHEMA", "sealed output coverage")
    for kind, data in rebuilt.items():
        if _hash(data) != expected[kind][1]:
            raise ReproductionHold("HOLD_REBUILD_HASH_MISMATCH", kind)
    if any(path.exists() or path.is_symlink() for path, _ in expected.values()):
        raise ReproductionHold("HOLD_OUTPUT_EXISTS", "sealed output already exists")
    for kind in ("report_html", "workbench_json"):
        _publish_exclusive(expected[kind][0], rebuilt[kind])
    return ReproductionResult(
        classification=cast(str, manifest["classification"]),
        report_sha256=expected["report_html"][1],
        workbench_sha256=expected["workbench_json"][1],
        output_paths=(expected["report_html"][0], expected["workbench_json"][0]),
    )


def _safe_file_parent(root: Path, relative: object) -> Path:
    value = _relative(relative, label="output")
    candidate = root.joinpath(*PurePosixPath(value).parts)
    if candidate.parent.is_symlink():
        raise ReproductionHold("HOLD_SYMLINK_PATH", value)
    if candidate.parent == root or not candidate.parent.is_dir():
        raise ReproductionHold("HOLD_OUTPUT_DIRECTORY", value)
    try:
        current = root
        for part in PurePosixPath(value).parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ReproductionHold("HOLD_SYMLINK_PATH", value)
        candidate.parent.resolve(strict=True).relative_to(root)
    except ReproductionHold:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ReproductionHold("HOLD_OUTPUT_DIRECTORY", value) from error
    return candidate
