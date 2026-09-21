from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeGuard

import yaml
from pydantic import BaseModel, ConfigDict
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from shadowskillbench.protocol.claim_estimands import (
    PROFILE_ID,
    ClaimEstimandTarget,
    claim_estimand_mapping,
)
from shadowskillbench.protocol.scientific_freeze import (
    MULTIPLICITY_MEMBERS,
    SCIENTIFIC_FREEZE_INPUTS,
    ScientificFreezeBinding,
)
from shadowskillbench.protocol.scientific_freeze_v4 import (
    MULTIPLICITY_MEMBERS_V4,
    SCIENTIFIC_FREEZE_V4_INPUTS,
    ScientificFreezeV4Binding,
)

if TYPE_CHECKING:
    from shadowskillbench.analysis.sealed import SealedAnalysis

ClaimStatus = Literal["hypothesis", "supported_prior_art", "finding"]
Disposition = Literal["supports", "does_not_support", "mixed", "inconclusive"]

_CLAIM_ID = re.compile(r"(?:PRIOR|EXP)-[0-9]{3}\Z")
_ARTIFACT_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ARTIFACT_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_CLAIM_FIELDS = frozenset(
    {
        "id",
        "wording",
        "status",
        "evidence",
        "evidence_command",
        "artifact",
        "artifact_sha256",
        "finding_disposition",
        "claim_mapping_profile",
        "claim_mapping_id",
        "claim_mapping_sha256",
        "scope",
    }
)
_PRIOR_FORBIDDEN_FIELDS = frozenset(
    {
        "evidence_command",
        "artifact",
        "artifact_sha256",
        "finding_disposition",
        "claim_mapping_profile",
        "claim_mapping_id",
        "claim_mapping_sha256",
    }
)
_AMBIGUOUS_STATUSES = frozenset({"supported", "supported_experimental"})
_MISSING = object()


class ClaimsViolation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    code: str
    location: str
    detail: str
    claim_id: str | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    wording: str
    status: ClaimStatus
    scope: str
    evidence: tuple[str, ...] = ()
    evidence_command: str | None = None
    artifact: str | None = None
    artifact_sha256: str | None = None
    finding_disposition: Disposition | None = None
    claim_mapping_profile: str | None = None
    claim_mapping_id: str | None = None
    claim_mapping_sha256: str | None = None


class ClaimsReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ledger_path: Path
    version: str | None
    valid: bool
    claims: tuple[Claim, ...]
    violations: tuple[ClaimsViolation, ...]
    warnings: tuple[ClaimsViolation, ...] = ()
    renderable_findings: tuple[Claim, ...]

    @property
    def counts(self) -> dict[ClaimStatus, int]:
        return {
            "supported_prior_art": sum(
                claim.status == "supported_prior_art" for claim in self.claims
            ),
            "hypothesis": sum(claim.status == "hypothesis" for claim in self.claims),
            "finding": sum(claim.status == "finding" for claim in self.claims),
        }


class ReleaseClaimsStatus(StrEnum):
    PASS = "PASS"
    HOLD_INVALID_CLAIMS_LEDGER = "HOLD_INVALID_CLAIMS_LEDGER"
    HOLD_REPORT_NOT_CONFIRMATORY = "HOLD_REPORT_NOT_CONFIRMATORY"
    HOLD_EXPERIMENT_AUDIT_NOT_PASS = "HOLD_EXPERIMENT_AUDIT_NOT_PASS"
    HOLD_MISSING_EXPERIMENT_AUDIT_HASH = "HOLD_MISSING_EXPERIMENT_AUDIT_HASH"
    HOLD_MISSING_ANALYSIS_DATASET_HASH = "HOLD_MISSING_ANALYSIS_DATASET_HASH"
    HOLD_NO_EXPERIMENTAL_FINDINGS = "HOLD_NO_EXPERIMENTAL_FINDINGS"
    HOLD_MISSING_SCIENTIFIC_FREEZE = "HOLD_MISSING_SCIENTIFIC_FREEZE"
    HOLD_CLAIM_DISPOSITION_NOT_SUPPORTS = "HOLD_CLAIM_DISPOSITION_NOT_SUPPORTS"
    HOLD_MISSING_CLAIM_ESTIMAND_MAPPING = "HOLD_MISSING_CLAIM_ESTIMAND_MAPPING"
    HOLD_MISSING_SEALED_ANALYSIS_EVIDENCE = "HOLD_MISSING_SEALED_ANALYSIS_EVIDENCE"
    HOLD_CROSS_HASH_SEALED_ANALYSIS_MISMATCH = "HOLD_CROSS_HASH_SEALED_ANALYSIS_MISMATCH"
    HOLD_SEALED_ANALYSIS_EXECUTION_DESIGN_MISMATCH = (
        "HOLD_SEALED_ANALYSIS_EXECUTION_DESIGN_MISMATCH"
    )
    HOLD_CLAIM_DIRECTION_MISMATCH = "HOLD_CLAIM_DIRECTION_MISMATCH"
    HOLD_CLAIM_EFFECT_NOT_MATERIAL = "HOLD_CLAIM_EFFECT_NOT_MATERIAL"
    HOLD_CLAIM_INTERVAL_INCONCLUSIVE = "HOLD_CLAIM_INTERVAL_INCONCLUSIVE"
    HOLD_CLAIM_EXCLUSIONS_REQUIRE_NON_SUPPORT = "HOLD_CLAIM_EXCLUSIONS_REQUIRE_NON_SUPPORT"
    HOLD_INSUFFICIENT_LIMITATIONS = "HOLD_INSUFFICIENT_LIMITATIONS"


class ReleaseClaimsGate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: ReleaseClaimsStatus
    finding_ids: tuple[str, ...]
    limitation_categories: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status is ReleaseClaimsStatus.PASS


class DuplicateYamlKeyError(yaml.YAMLError):
    pass


class StrictSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: StrictSafeLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.YAMLError("mapping keys must be scalar") from error
        if duplicate:
            raise DuplicateYamlKeyError("duplicate YAML mapping key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictSafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _claim_location(claim_id: str | None, field: str | None = None) -> str:
    location = f"claims[{claim_id}]" if claim_id is not None else "claims"
    return f"{location}.{field}" if field is not None else location


def _violation(
    violations: list[ClaimsViolation],
    code: str,
    location: str,
    detail: str,
    claim_id: str | None = None,
) -> None:
    violations.append(
        ClaimsViolation(code=code, location=location, detail=detail, claim_id=claim_id)
    )


def _claim_violation(
    violations: list[ClaimsViolation],
    code: str,
    claim_id: str | None,
    field: str | None,
    detail: str,
) -> None:
    _violation(violations, code, _claim_location(claim_id, field), detail, claim_id)


def _nonblank_string(value: object) -> TypeGuard[str]:
    return type(value) is str and bool(value.strip())


def _safe_artifact_path(
    value: object,
    present: bool,
    repository_root: Path,
    claim_id: str,
    violations: list[ClaimsViolation],
) -> Path | None:
    if not present:
        _claim_violation(
            violations,
            "MISSING_ARTIFACT",
            claim_id,
            "artifact",
            "experimental claims require an artifact path",
        )
        return None
    if type(value) is not str or not value.strip():
        _claim_violation(
            violations,
            "INVALID_ARTIFACT",
            claim_id,
            "artifact",
            "artifact must be a nonblank string",
        )
        return None
    artifact = value
    if (
        artifact != artifact.strip()
        or artifact.startswith("/")
        or "\\" in artifact
        or any(ord(character) < 32 or ord(character) == 127 for character in artifact)
    ):
        _claim_violation(
            violations,
            "UNSAFE_ARTIFACT_PATH",
            claim_id,
            "artifact",
            "artifact path is not repository-relative",
        )
        return None
    parts = artifact.split("/")
    if any(part in {"", ".", ".."} or not _SAFE_ARTIFACT_SEGMENT.fullmatch(part) for part in parts):
        _claim_violation(
            violations,
            "UNSAFE_ARTIFACT_PATH",
            claim_id,
            "artifact",
            "artifact path has an unsafe segment",
        )
        return None
    candidate = repository_root.joinpath(*parts)
    component = repository_root
    try:
        for part in parts:
            component = component / part
            if component.is_symlink():
                _claim_violation(
                    violations,
                    "SYMLINK_ARTIFACT_PATH",
                    claim_id,
                    "artifact",
                    "artifact path contains a symlink component",
                )
                return None
        candidate.resolve(strict=False).relative_to(repository_root)
    except OSError:
        _claim_violation(
            violations,
            "ARTIFACT_IO",
            claim_id,
            "artifact",
            "artifact path resolution failed",
        )
        return None
    except ValueError:
        _claim_violation(
            violations,
            "UNSAFE_ARTIFACT_PATH",
            claim_id,
            "artifact",
            "artifact path escapes repository root",
        )
        return None
    return candidate


def _verify_artifact(
    artifact: Path,
    artifact_sha256: str,
    claim_id: str,
    violations: list[ClaimsViolation],
) -> None:
    try:
        if not artifact.exists():
            _claim_violation(
                violations,
                "MISSING_ARTIFACT",
                claim_id,
                "artifact",
                "artifact file does not exist",
            )
            return
        if artifact.is_symlink() or not artifact.is_file():
            _claim_violation(
                violations,
                "INVALID_ARTIFACT",
                claim_id,
                "artifact",
                "artifact must be a regular non-symlink file",
            )
            return
        actual_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    except OSError:
        _claim_violation(
            violations,
            "ARTIFACT_IO",
            claim_id,
            "artifact",
            "artifact access failed",
        )
        return
    if actual_hash != artifact_sha256:
        _claim_violation(
            violations,
            "ARTIFACT_HASH_MISMATCH",
            claim_id,
            "artifact_sha256",
            "artifact raw-byte hash does not match",
        )


def _validate_existing_artifact(
    artifact: Path, claim_id: str, violations: list[ClaimsViolation]
) -> None:
    try:
        if artifact.exists() and (artifact.is_symlink() or not artifact.is_file()):
            _claim_violation(
                violations,
                "INVALID_ARTIFACT",
                claim_id,
                "artifact",
                "existing artifact must be a regular non-symlink file",
            )
    except OSError:
        _claim_violation(
            violations,
            "ARTIFACT_IO",
            claim_id,
            "artifact",
            "artifact access failed",
        )


def _validated_disposition(value: object) -> Disposition | None:
    if value == "supports":
        return "supports"
    if value == "does_not_support":
        return "does_not_support"
    if value == "mixed":
        return "mixed"
    if value == "inconclusive":
        return "inconclusive"
    return None


def _validate_claim(
    raw_claim: object,
    repository_root: Path,
    violations: list[ClaimsViolation],
) -> Claim | None:
    if type(raw_claim) is not dict:
        _violation(violations, "INVALID_CLAIM_TYPE", "claims", "each claim must be a mapping")
        return None

    raw_id = raw_claim.get("id", _MISSING)
    claim_id = raw_id if _nonblank_string(raw_id) else None
    report_claim_id = claim_id if claim_id is not None and _CLAIM_ID.fullmatch(claim_id) else None
    if claim_id is None:
        _violation(violations, "MISSING_CLAIM_ID", "claims.id", "claim id must be nonblank")
        return None
    if report_claim_id is None:
        _violation(
            violations,
            "INVALID_CLAIM_ID",
            "claims.id",
            "claim id must use PRIOR- or EXP- plus three digits",
        )

    for key in sorted(raw_claim, key=repr):
        if type(key) is not str or key not in _CLAIM_FIELDS:
            _claim_violation(
                violations,
                "UNKNOWN_CLAIM_FIELD",
                report_claim_id,
                "unknown",
                "field is not allowed",
            )

    wording = raw_claim.get("wording", _MISSING)
    if not _nonblank_string(wording):
        _claim_violation(
            violations,
            "MISSING_WORDING",
            report_claim_id,
            "wording",
            "claim wording must be nonblank",
        )
        wording = ""
    scope = raw_claim.get("scope", _MISSING)
    if not _nonblank_string(scope):
        _claim_violation(
            violations,
            "MISSING_SCOPE",
            report_claim_id,
            "scope",
            "claim scope must be nonblank",
        )
        scope = ""

    raw_status = raw_claim.get("status", _MISSING)
    status: ClaimStatus | None
    if type(raw_status) is str and raw_status in _AMBIGUOUS_STATUSES:
        _claim_violation(
            violations,
            "AMBIGUOUS_STATUS",
            report_claim_id,
            "status",
            "supported statuses require explicit lifecycle",
        )
        status = None
    elif raw_status == "hypothesis":
        status = "hypothesis"
    elif raw_status == "supported_prior_art":
        status = "supported_prior_art"
    elif raw_status == "finding":
        status = "finding"
    else:
        _claim_violation(
            violations,
            "INVALID_STATUS",
            report_claim_id,
            "status",
            "claim status is not allowed",
        )
        status = None

    raw_evidence = raw_claim.get("evidence", _MISSING)
    evidence: tuple[str, ...] = ()
    if raw_evidence is _MISSING:
        pass
    elif (
        type(raw_evidence) is list
        and all(_nonblank_string(item) for item in raw_evidence)
        and len(set(raw_evidence)) == len(raw_evidence)
    ):
        evidence = tuple(raw_evidence)
    else:
        _claim_violation(
            violations,
            "INVALID_EVIDENCE",
            report_claim_id,
            "evidence",
            "evidence must be a list of distinct nonblank strings",
        )

    raw_command = raw_claim.get("evidence_command", _MISSING)
    command = raw_command if type(raw_command) is str else None
    artifact_present = "artifact" in raw_claim
    raw_artifact = raw_claim.get("artifact", _MISSING)
    artifact_value = raw_artifact if type(raw_artifact) is str else None
    sha256_present = "artifact_sha256" in raw_claim
    raw_sha256 = raw_claim.get("artifact_sha256", _MISSING)
    artifact_sha256 = raw_sha256 if type(raw_sha256) is str else None
    disposition_present = "finding_disposition" in raw_claim
    raw_disposition = raw_claim.get("finding_disposition", _MISSING)
    finding_disposition = _validated_disposition(raw_disposition)
    mapping_profile_present = "claim_mapping_profile" in raw_claim
    raw_mapping_profile = raw_claim.get("claim_mapping_profile", _MISSING)
    mapping_profile = raw_mapping_profile if type(raw_mapping_profile) is str else None
    mapping_id_present = "claim_mapping_id" in raw_claim
    raw_mapping_id = raw_claim.get("claim_mapping_id", _MISSING)
    mapping_id = raw_mapping_id if type(raw_mapping_id) is str else None
    mapping_sha256_present = "claim_mapping_sha256" in raw_claim
    raw_mapping_sha256 = raw_claim.get("claim_mapping_sha256", _MISSING)
    mapping_sha256 = raw_mapping_sha256 if type(raw_mapping_sha256) is str else None

    if claim_id.startswith("PRIOR-"):
        if status != "supported_prior_art":
            _claim_violation(
                violations,
                "INVALID_NAMESPACE_STATUS",
                report_claim_id,
                "status",
                "PRIOR claims require supported_prior_art",
            )
        if not evidence:
            _claim_violation(
                violations,
                "INVALID_PRIOR_ART",
                report_claim_id,
                "evidence",
                "prior art requires distinct nonempty sources",
            )
        if any(field in raw_claim for field in _PRIOR_FORBIDDEN_FIELDS):
            _claim_violation(
                violations,
                "INVALID_PRIOR_ART",
                report_claim_id,
                None,
                "prior art forbids result metadata",
            )

    if claim_id.startswith("EXP-"):
        if status not in {"hypothesis", "finding"}:
            _claim_violation(
                violations,
                "INVALID_NAMESPACE_STATUS",
                report_claim_id,
                "status",
                "EXP claims require hypothesis or finding",
            )
        expected_command = f"shadowskillbench report claim {claim_id}"
        if command != expected_command:
            _claim_violation(
                violations,
                "INVALID_EVIDENCE_COMMAND",
                report_claim_id,
                "evidence_command",
                "evidence command is not canonical",
            )
        artifact = _safe_artifact_path(
            raw_artifact, artifact_present, repository_root, claim_id, violations
        )
        if artifact is not None:
            _validate_existing_artifact(artifact, claim_id, violations)

        valid_sha256 = False
        if sha256_present:
            if type(raw_sha256) is not str or not _ARTIFACT_SHA256.fullmatch(raw_sha256):
                _claim_violation(
                    violations,
                    "INVALID_ARTIFACT_SHA256",
                    report_claim_id,
                    "artifact_sha256",
                    "artifact sha256 must be lowercase sha256",
                )
            else:
                valid_sha256 = True
        elif status == "finding":
            _claim_violation(
                violations,
                "MISSING_ARTIFACT_SHA256",
                report_claim_id,
                "artifact_sha256",
                "findings require an artifact sha256",
            )

        if status == "finding":
            if not disposition_present:
                _claim_violation(
                    violations,
                    "MISSING_FINDING_DISPOSITION",
                    report_claim_id,
                    "finding_disposition",
                    "findings require a disposition",
                )
            elif finding_disposition is None:
                _claim_violation(
                    violations,
                    "INVALID_FINDING_DISPOSITION",
                    report_claim_id,
                    "finding_disposition",
                    "finding disposition is not allowed",
                )
            mapping = claim_estimand_mapping(claim_id)
            if (
                not mapping_profile_present
                or not _nonblank_string(mapping_profile)
                or not mapping_id_present
                or not _nonblank_string(mapping_id)
                or not mapping_sha256_present
                or type(mapping_sha256) is not str
                or _ARTIFACT_SHA256.fullmatch(mapping_sha256) is None
            ):
                _claim_violation(
                    violations,
                    "MISSING_CLAIM_ESTIMAND_MAPPING",
                    report_claim_id,
                    "claim_mapping",
                    "findings require a canonical claim-estimand mapping binding",
                )
            elif (
                mapping is None
                or mapping_profile != PROFILE_ID
                or mapping_id != mapping.mapping_id
                or mapping_sha256 != mapping.sha256
            ):
                _claim_violation(
                    violations,
                    "INVALID_CLAIM_ESTIMAND_MAPPING",
                    report_claim_id,
                    "claim_mapping",
                    "finding mapping does not bind the canonical claim-estimand profile",
                )
        elif status == "hypothesis" and disposition_present:
            _claim_violation(
                violations,
                "FORBIDDEN_FINDING_DISPOSITION",
                report_claim_id,
                "finding_disposition",
                "hypotheses cannot carry a finding disposition",
            )
        if status == "hypothesis" and (
            mapping_profile_present or mapping_id_present or mapping_sha256_present
        ):
            _claim_violation(
                violations,
                "FORBIDDEN_CLAIM_ESTIMAND_MAPPING",
                report_claim_id,
                "claim_mapping",
                "hypotheses cannot carry a finding mapping binding",
            )

        if artifact is not None and valid_sha256 and artifact_sha256 is not None:
            _verify_artifact(artifact, artifact_sha256, claim_id, violations)

    if status is None:
        return None
    return Claim(
        id=claim_id,
        wording=wording,
        status=status,
        scope=scope,
        evidence=evidence,
        evidence_command=command,
        artifact=artifact_value,
        artifact_sha256=artifact_sha256,
        finding_disposition=finding_disposition,
        claim_mapping_profile=mapping_profile,
        claim_mapping_id=mapping_id,
        claim_mapping_sha256=mapping_sha256,
    )


def _report(
    ledger_path: Path,
    version: str | None,
    claims: list[Claim],
    violations: list[ClaimsViolation],
) -> ClaimsReport:
    sorted_claims = tuple(sorted(claims, key=lambda claim: claim.id))
    sorted_violations = tuple(
        sorted(
            violations,
            key=lambda violation: (
                violation.location,
                violation.code,
                violation.claim_id or "",
                violation.detail,
            ),
        )
    )
    valid = not sorted_violations
    findings = tuple(claim for claim in sorted_claims if claim.status == "finding") if valid else ()
    return ClaimsReport(
        ledger_path=ledger_path,
        version=version,
        valid=valid,
        claims=sorted_claims,
        violations=sorted_violations,
        warnings=(),
        renderable_findings=findings,
    )


def validate_claims_ledger(path: Path, repository_root: Path) -> ClaimsReport:
    violations: list[ClaimsViolation] = []
    claims: list[Claim] = []
    version: str | None = None
    ledger_path = path.resolve(strict=False)
    try:
        root = repository_root.resolve(strict=True)
        if not root.is_dir():
            _violation(
                violations,
                "INVALID_REPOSITORY_ROOT",
                "repository_root",
                "repository root is not a directory",
            )
            return _report(ledger_path, version, claims, violations)
    except OSError:
        _violation(
            violations,
            "INVALID_REPOSITORY_ROOT",
            "repository_root",
            "repository root resolution failed",
        )
        return _report(ledger_path, version, claims, violations)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        _violation(violations, "LEDGER_IO", "ledger", "ledger read failed")
        return _report(ledger_path, version, claims, violations)
    try:
        document = yaml.load(text, Loader=StrictSafeLoader)
    except DuplicateYamlKeyError:
        _violation(violations, "DUPLICATE_YAML_KEY", "root", "YAML mapping keys must be unique")
        return _report(ledger_path, version, claims, violations)
    except yaml.YAMLError:
        _violation(violations, "YAML_PARSE", "root", "YAML parsing failed")
        return _report(ledger_path, version, claims, violations)
    if type(document) is not dict:
        _violation(violations, "INVALID_ROOT", "root", "ledger root must be a mapping")
        return _report(ledger_path, version, claims, violations)
    for key in sorted(document, key=repr):
        if type(key) is not str or key not in {"version", "claims"}:
            _violation(violations, "UNKNOWN_ROOT_FIELD", "root.unknown", "field is not allowed")
    raw_version = document.get("version", _MISSING)
    if type(raw_version) is str:
        version = raw_version
    if version != "1.0":
        _violation(
            violations,
            "INVALID_VERSION",
            "root.version",
            "ledger version must be string 1.0",
        )
    raw_claims = document.get("claims", _MISSING)
    if type(raw_claims) is not list:
        _violation(violations, "INVALID_CLAIMS", "root.claims", "claims must be a list")
        return _report(ledger_path, version, claims, violations)
    seen_ids: set[str] = set()
    for raw_claim in raw_claims:
        claim = _validate_claim(raw_claim, root, violations)
        if claim is None:
            continue
        if claim.id in seen_ids:
            _claim_violation(
                violations, "DUPLICATE_CLAIM_ID", claim.id, "id", "claim id must be unique"
            )
        seen_ids.add(claim.id)
        claims.append(claim)
    return _report(ledger_path, version, claims, violations)


def _release_gate(
    status: ReleaseClaimsStatus,
    findings: tuple[Claim, ...] = (),
    limitation_categories: tuple[str, ...] = (),
) -> ReleaseClaimsGate:
    return ReleaseClaimsGate(
        status=status,
        finding_ids=tuple(claim.id for claim in findings),
        limitation_categories=limitation_categories,
    )


def _validated_release_findings(report: ClaimsReport) -> tuple[Claim, ...] | None:
    expected = tuple(claim for claim in report.claims if claim.status == "finding")
    findings = report.renderable_findings
    if findings != expected or not all(type(claim) is Claim for claim in findings):
        return None
    for claim in findings:
        if (
            not claim.id.startswith("EXP-")
            or not _nonblank_string(claim.wording)
            or not _nonblank_string(claim.scope)
            or not _nonblank_string(claim.artifact)
            or claim.artifact_sha256 is None
            or _ARTIFACT_SHA256.fullmatch(claim.artifact_sha256) is None
            or claim.finding_disposition
            not in {
                "supports",
                "does_not_support",
                "mixed",
                "inconclusive",
            }
            or not _nonblank_string(claim.claim_mapping_profile)
            or not _nonblank_string(claim.claim_mapping_id)
            or claim.claim_mapping_sha256 is None
            or _ARTIFACT_SHA256.fullmatch(claim.claim_mapping_sha256) is None
        ):
            return None
    return findings


type _EvidenceKey = tuple[str, str, str, str, str | None]
type _EffectEvidence = tuple[float, float, float]


def _target_key(target: ClaimEstimandTarget) -> _EvidenceKey:
    return (
        target.estimand_id,
        target.component,
        target.scope,
        target.metric,
        target.authority_class,
    )


def _sealed_claim_evidence(
    sealed_analysis: object,
    experiment_audit_hash: str,
    analysis_dataset_hash: str,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None,
) -> dict[_EvidenceKey, tuple[_EffectEvidence, ...]] | None:
    """Extract the release-only estimand evidence from exact sealed analysis objects."""

    from shadowskillbench.analysis.sealed import SealedAnalysis
    from shadowskillbench.analysis.stage_a import StageAAnalysis, StageAEstimand
    from shadowskillbench.analysis.stage_b import (
        MacroGovernanceEffect,
        RateEstimate,
        StageBAnalysis,
    )
    from shadowskillbench.reporting.report import sealed_analysis_is_release_complete

    if not _valid_scientific_freeze(scientific_freeze):
        return None
    if type(sealed_analysis) is not SealedAnalysis or not sealed_analysis_is_release_complete(
        sealed_analysis
    ):
        return None
    try:
        if (
            type(scientific_freeze) is ScientificFreezeV4Binding
            and sealed_analysis.scientific_freeze != scientific_freeze
        ):
            return None
        if (
            sealed_analysis.audit.report_hash != experiment_audit_hash
            or sealed_analysis.dataset_sha256 != analysis_dataset_hash
            or type(sealed_analysis.stage_a) is not StageAAnalysis
            or type(sealed_analysis.stage_b) is not StageBAnalysis
            or sealed_analysis.runtime_binding.freeze_manifest_hash
            != scientific_freeze.manifest_hash
            or sealed_analysis.stage_a.inference.profile != scientific_freeze.analysis_profile
            or sealed_analysis.stage_b.inference.profile != scientific_freeze.analysis_profile
            or (
                scientific_freeze is not None
                and (
                    sealed_analysis.stage_a.inference.statistics_configuration_sha256
                    != scientific_freeze.statistics_configuration_sha256
                    or sealed_analysis.stage_b.inference.statistics_configuration_sha256
                    != scientific_freeze.statistics_configuration_sha256
                )
            )
        ):
            return None
        values: dict[_EvidenceKey, list[_EffectEvidence]] = {}

        def add(key: _EvidenceKey, estimate: object, ci_low: object, ci_high: object) -> bool:
            if (
                type(estimate) is not float
                or type(ci_low) is not float
                or type(ci_high) is not float
                or not all(isfinite(value) for value in (estimate, ci_low, ci_high))
                or not ci_low <= estimate <= ci_high
            ):
                return False
            values.setdefault(key, []).append((estimate, ci_low, ci_high))
            return True

        for estimand in sealed_analysis.stage_a.estimands:
            if type(estimand) is not StageAEstimand or not add(
                (
                    estimand.estimand,
                    estimand.component,
                    estimand.scope,
                    "completion_under_policy",
                    None,
                ),
                estimand.estimate,
                estimand.ci_low,
                estimand.ci_high,
            ):
                return None
        for rate in sealed_analysis.stage_b.e7_false_enforcement:
            if type(rate) is not RateEstimate or not add(
                (
                    "E7",
                    "rate",
                    rate.domain,
                    rate.metric,
                    rate.authority_class,
                ),
                rate.rate,
                rate.ci_low,
                rate.ci_high,
            ):
                return None
        for effect in sealed_analysis.stage_b.e8_authority_aware_gain:
            if type(effect) is not MacroGovernanceEffect or not add(
                (
                    "E8",
                    "macro_delta",
                    effect.domain,
                    "authority_resolution_correct",
                    None,
                ),
                effect.delta,
                effect.ci_low,
                effect.ci_high,
            ):
                return None
    except (AttributeError, TypeError, ValueError):
        return None
    return {key: tuple(item) for key, item in values.items()}


def _canonical_mapping(claim: Claim):
    mapping = claim_estimand_mapping(claim.id)
    if (
        mapping is None
        or claim.claim_mapping_profile != PROFILE_ID
        or claim.claim_mapping_id != mapping.mapping_id
        or claim.claim_mapping_sha256 != mapping.sha256
    ):
        return None
    return mapping


def _freeze_claim_parameters(
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding,
) -> tuple[float, str, tuple[str, ...], str, str, str]:
    """Return explicit V3/V4 claim parameters without structural fallbacks."""

    if type(scientific_freeze) is ScientificFreezeBinding:
        return (
            scientific_freeze.material_threshold,
            scientific_freeze.multiplicity_family,
            scientific_freeze.multiplicity_members,
            scientific_freeze.multiplicity_scope,
            scientific_freeze.multiplicity_method,
            scientific_freeze.interval_rule,
        )
    if type(scientific_freeze) is ScientificFreezeV4Binding:
        return (
            scientific_freeze.material_threshold,
            "E1-E9-primary-components-16",
            scientific_freeze.multiplicity_members,
            "primary_confirmatory_estimands",
            "bonferroni_simultaneous_bootstrap_intervals",
            "simultaneous_95pct_ci_entirely_beyond_material_threshold",
        )
    raise TypeError("scientific_freeze must be an exact V3 or V4 binding")


def _claim_support_status(
    findings: tuple[Claim, ...],
    sealed_analysis: object,
    experiment_audit_hash: str,
    analysis_dataset_hash: str,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None,
) -> ReleaseClaimsStatus:
    if not _valid_scientific_freeze(scientific_freeze):
        return ReleaseClaimsStatus.HOLD_MISSING_SCIENTIFIC_FREEZE
    (
        material_threshold,
        multiplicity_family,
        multiplicity_members,
        multiplicity_scope,
        multiplicity_method,
        interval_rule,
    ) = _freeze_claim_parameters(scientific_freeze)
    from shadowskillbench.analysis.sealed import SealedAnalysis

    if type(sealed_analysis) is SealedAnalysis and not _sealed_analysis_execution_design_matches(
        sealed_analysis, scientific_freeze
    ):
        return ReleaseClaimsStatus.HOLD_SEALED_ANALYSIS_EXECUTION_DESIGN_MISMATCH
    mappings = tuple(_canonical_mapping(claim) for claim in findings)
    if any(mapping is None for mapping in mappings):
        return ReleaseClaimsStatus.HOLD_MISSING_CLAIM_ESTIMAND_MAPPING
    evidence = _sealed_claim_evidence(
        sealed_analysis,
        experiment_audit_hash,
        analysis_dataset_hash,
        scientific_freeze,
    )
    if evidence is None:
        if sealed_analysis is None:
            return ReleaseClaimsStatus.HOLD_MISSING_SEALED_ANALYSIS_EVIDENCE
        return ReleaseClaimsStatus.HOLD_CROSS_HASH_SEALED_ANALYSIS_MISMATCH
    if any(claim.finding_disposition == "supports" for claim in findings) and getattr(
        getattr(sealed_analysis, "dataset", None), "exclusions", ()
    ):
        return ReleaseClaimsStatus.HOLD_CLAIM_EXCLUSIONS_REQUIRE_NON_SUPPORT
    for claim, mapping in zip(findings, mappings, strict=True):
        assert mapping is not None
        for target in mapping.estimands:
            matches = evidence.get(_target_key(target), ())
            if len(matches) != 1:
                return ReleaseClaimsStatus.HOLD_MISSING_SEALED_ANALYSIS_EVIDENCE
            if claim.finding_disposition != "supports":
                continue
            estimate, ci_low, ci_high = matches[0]
            if (target.expected_direction == "positive" and estimate <= 0.0) or (
                target.expected_direction == "negative" and estimate >= 0.0
            ):
                return ReleaseClaimsStatus.HOLD_CLAIM_DIRECTION_MISMATCH
            if abs(estimate) < target.material_threshold:
                return ReleaseClaimsStatus.HOLD_CLAIM_EFFECT_NOT_MATERIAL
            if (
                target.material_threshold != material_threshold
                or target.multiplicity_family != multiplicity_family
                or target.multiplicity_member not in multiplicity_members
                or target.multiplicity_scope != multiplicity_scope
                or target.multiplicity_method != multiplicity_method
                or target.interval_rule != interval_rule
            ):
                return ReleaseClaimsStatus.HOLD_CLAIM_INTERVAL_INCONCLUSIVE
            interval_is_inconclusive = (
                target.expected_direction == "positive" and ci_low <= target.material_threshold
            ) or (target.expected_direction == "negative" and ci_high >= -target.material_threshold)
            if interval_is_inconclusive:
                return ReleaseClaimsStatus.HOLD_CLAIM_INTERVAL_INCONCLUSIVE
    return ReleaseClaimsStatus.PASS


def _sealed_analysis_execution_design_matches(
    sealed_analysis: SealedAnalysis,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding,
) -> bool:
    """Require the prebuilt sealed analysis to retain the exact frozen design."""

    projection = sealed_analysis.execution_design_projection
    return (
        type(projection) is tuple
        and all(
            type(item) is tuple
            and len(item) == 4
            and type(item[0]) is str
            and all(type(value) is int and value > 0 for value in item[1:])
            for item in projection
        )
        and projection == scientific_freeze.member_execution_designs
        and (
            type(scientific_freeze) is not ScientificFreezeV4Binding
            or sealed_analysis.scientific_freeze == scientific_freeze
        )
    )


def _valid_scientific_freeze(
    binding: object,
) -> TypeGuard[ScientificFreezeBinding | ScientificFreezeV4Binding]:
    if type(binding) is ScientificFreezeV4Binding:
        if (
            type(binding.manifest_hash) is not str
            or _ARTIFACT_SHA256.fullmatch(binding.manifest_hash) is None
            or binding.analysis_profile != "SSB-CONFIRMATORY-STATISTICS3-BONFERRONI"
            or type(binding.statistics_configuration_sha256) is not str
            or _ARTIFACT_SHA256.fullmatch(binding.statistics_configuration_sha256) is None
            or binding.multiplicity_members != MULTIPLICITY_MEMBERS_V4
            or binding.material_threshold != 0.1
            or binding.bootstrap_replicates != 40_000
            or binding.bootstrap_seed != 104_731
            or type(binding.input_hashes) is not tuple
            or type(binding.member_execution_designs) is not tuple
            or tuple(path for path, _ in binding.input_hashes)
            != tuple(item.path for item in SCIENTIFIC_FREEZE_V4_INPUTS)
            or dict(binding.input_hashes).get("config/statistics.v4.yaml")
            != binding.statistics_configuration_sha256
            or len(binding.member_execution_designs) != len(MULTIPLICITY_MEMBERS_V4)
            or tuple(item[0] for item in binding.member_execution_designs)
            != MULTIPLICITY_MEMBERS_V4
        ):
            return False
        return all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is str
            and _ARTIFACT_SHA256.fullmatch(item[1]) is not None
            for item in binding.input_hashes
        ) and all(
            type(item) is tuple
            and len(item) == 4
            and type(item[0]) is str
            and all(type(value) is int and value > 0 for value in item[1:])
            for item in binding.member_execution_designs
        )
    if (
        type(binding) is not ScientificFreezeBinding
        or type(binding.manifest_hash) is not str
        or _ARTIFACT_SHA256.fullmatch(binding.manifest_hash) is None
        or type(binding.input_hashes) is not tuple
        or type(binding.analysis_plan_sha256) is not str
        or _ARTIFACT_SHA256.fullmatch(binding.analysis_plan_sha256) is None
        or type(binding.power_precision_plan_sha256) is not str
        or _ARTIFACT_SHA256.fullmatch(binding.power_precision_plan_sha256) is None
        or type(binding.power_precision_evidence_sha256) is not str
        or _ARTIFACT_SHA256.fullmatch(binding.power_precision_evidence_sha256) is None
        or binding.analysis_profile != "SSB-CONFIRMATORY-STATISTICS3-BONFERRONI"
        or binding.multiplicity_family != "E1-E9-primary-components-16"
        or type(binding.multiplicity_members) is not tuple
        or binding.multiplicity_members != MULTIPLICITY_MEMBERS
        or type(binding.member_execution_designs) is not tuple
        or binding.multiplicity_scope != "primary_confirmatory_estimands"
        or binding.multiplicity_method != "bonferroni_simultaneous_bootstrap_intervals"
        or binding.interval_rule != "simultaneous_95pct_ci_entirely_beyond_material_threshold"
        or type(binding.familywise_confidence_level) is not float
        or binding.familywise_confidence_level != 0.95
        or type(binding.material_threshold) is not float
        or binding.material_threshold != 0.1
        or type(binding.bootstrap_replicates) is not int
        or binding.bootstrap_replicates != 40_000
        or type(binding.bootstrap_seed) is not int
        or binding.bootstrap_seed != 104_731
    ):
        return False
    if any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in binding.input_hashes
    ):
        return False
    if (
        len(binding.member_execution_designs) != len(MULTIPLICITY_MEMBERS)
        or any(
            type(item) is not tuple
            or len(item) != 4
            or type(item[0]) is not str
            or any(type(value) is not int or value < 1 for value in item[1:])
            for item in binding.member_execution_designs
        )
        or tuple(item[0] for item in binding.member_execution_designs) != MULTIPLICITY_MEMBERS
    ):
        return False
    expected_paths = tuple(item.path for item in SCIENTIFIC_FREEZE_INPUTS)
    if tuple(path for path, _ in binding.input_hashes) != expected_paths:
        return False
    hashes = dict(binding.input_hashes)
    if (
        binding.analysis_plan_sha256 != hashes["protocol/analysis_plan.v3.json"]
        or binding.power_precision_plan_sha256 != hashes["protocol/power_precision_plan.v3.json"]
        or binding.power_precision_evidence_sha256
        != hashes["protocol/power_precision_evidence.v3.json"]
    ):
        return False
    return all(_ARTIFACT_SHA256.fullmatch(digest) is not None for _, digest in binding.input_hashes)


def _normalized_limitation(value: str) -> str:
    return " ".join(value.casefold().split())


def _has_any(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)


def _limitation_categories(limitations: object) -> tuple[str, ...] | None:
    if type(limitations) is not tuple or not limitations:
        return None
    if any(type(item) is not str or not item.strip() or "\x00" in item for item in limitations):
        return None
    normalized = tuple(_normalized_limitation(item) for item in limitations)
    if len(set(normalized)) != len(normalized):
        return None
    synthetic_external = any(
        _has_any(item, ("synthetic", "simulat"))
        and _has_any(item, ("domain", "domains"))
        and _has_any(item, ("external", "generaliz", "validity", "real-world", "workplace"))
        for item in normalized
    )
    selected_model = any(
        _has_any(item, ("model", "provider"))
        and _has_any(item, ("selected", "configured", "specific", "version", "scope"))
        for item in normalized
    )
    exclusion_audit = any(
        _has_any(item, ("exclusion", "exclude", "omission"))
        and _has_any(item, ("audit", "bound", "scope", "coverage"))
        for item in normalized
    )
    if not (synthetic_external and selected_model and exclusion_audit):
        return None
    return ("synthetic_external_validity", "selected_model_provider", "exclusion_audit_bounds")


def validate_release_claims(
    report: object,
    *,
    report_status: object,
    experiment_audit_status: object,
    experiment_audit_hash: object,
    analysis_dataset_hash: object,
    limitations: object,
    sealed_analysis: object = None,
    scientific_freeze: ScientificFreezeBinding | ScientificFreezeV4Binding | None = None,
) -> ReleaseClaimsGate:
    """Return a deterministic release claim ceiling from already-validated local evidence."""

    if type(report) is not ClaimsReport or not report.valid:
        return _release_gate(ReleaseClaimsStatus.HOLD_INVALID_CLAIMS_LEDGER)
    findings = _validated_release_findings(report)
    if findings is None:
        return _release_gate(ReleaseClaimsStatus.HOLD_INVALID_CLAIMS_LEDGER)
    if report_status != "confirmatory":
        return _release_gate(ReleaseClaimsStatus.HOLD_REPORT_NOT_CONFIRMATORY)
    if experiment_audit_status != "PASS":
        return _release_gate(ReleaseClaimsStatus.HOLD_EXPERIMENT_AUDIT_NOT_PASS)
    if (
        type(experiment_audit_hash) is not str
        or _ARTIFACT_SHA256.fullmatch(experiment_audit_hash) is None
    ):
        return _release_gate(ReleaseClaimsStatus.HOLD_MISSING_EXPERIMENT_AUDIT_HASH)
    if (
        type(analysis_dataset_hash) is not str
        or _ARTIFACT_SHA256.fullmatch(analysis_dataset_hash) is None
    ):
        return _release_gate(ReleaseClaimsStatus.HOLD_MISSING_ANALYSIS_DATASET_HASH)
    if not findings:
        return _release_gate(ReleaseClaimsStatus.HOLD_NO_EXPERIMENTAL_FINDINGS)
    support_status = _claim_support_status(
        findings,
        sealed_analysis,
        experiment_audit_hash,
        analysis_dataset_hash,
        scientific_freeze,
    )
    if support_status is not ReleaseClaimsStatus.PASS:
        return _release_gate(support_status, findings)
    categories = _limitation_categories(limitations)
    if categories is None:
        return _release_gate(ReleaseClaimsStatus.HOLD_INSUFFICIENT_LIMITATIONS, findings)
    return _release_gate(ReleaseClaimsStatus.PASS, findings, categories)
