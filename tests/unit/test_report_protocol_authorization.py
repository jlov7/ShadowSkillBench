from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from shadowskillbench import cli
from shadowskillbench.analysis.sealed import SealedAnalysisHold
from shadowskillbench.core.hashing import sha256_ref
from shadowskillbench.corpus.confirmatory import _current_generator_hash


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_authorization(
    root: Path, *, local_custody: bool = True, confirmatory_prompt: bool = False
) -> tuple[Path, Path, Path]:
    protocol = root / "protocol"
    protocol.mkdir()
    inputs = [
        {
            "path": "protocol/preregistration.md",
            "role": "preregistration_core",
            "sha256": "sha256:" + "a" * 64,
        },
        {
            "path": "src/shadowskillbench/corpus/confirmatory.py",
            "role": "confirmatory_corpus",
            "sha256": _current_generator_hash(),
        },
        {
            "path": "prompts/confirmatory_skill_compiler.md"
            if confirmatory_prompt
            else "prompts/skill_compiler.md",
            "role": "confirmatory_compiler_prompt" if confirmatory_prompt else "compiler_prompt",
            "sha256": "sha256:" + "b" * 64,
        },
        {
            "path": "prompts/executor_system.md",
            "role": "executor_system_prompt",
            "sha256": "sha256:" + "c" * 64,
        },
    ]
    contents = {
        "protocol/preregistration.md": b"fixture preregistration\n",
        "src/shadowskillbench/corpus/confirmatory.py": (
            Path(__file__)
            .parents[2]
            .joinpath("src", "shadowskillbench", "corpus", "confirmatory.py")
            .read_bytes()
        ),
        str(inputs[2]["path"]): b"fixture compiler prompt\n",
        "prompts/executor_system.md": b"fixture executor prompt\n",
    }
    for entry in inputs:
        path = entry["path"]
        assert type(path) is str
        data = contents[path]
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entry["sha256"] = "sha256:" + sha256(data).hexdigest()
    payload = {
        "anchor_status": "PENDING_HUMAN_ANCHOR",
        "inputs": inputs,
        "preregistration_core": {
            "path": "protocol/preregistration.md",
            "sha256": inputs[0]["sha256"],
        },
        "schema_version": "1.0",
    }
    manifest = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest += b"\n"
    manifest_path = protocol / "freeze_manifest.json"
    manifest_path.write_bytes(manifest)
    detached_path = protocol / "freeze_manifest.sha256"
    digest = sha256(manifest).hexdigest()
    detached_path.write_bytes(f"{digest}  freeze_manifest.json\n".encode())
    if local_custody:
        _git(root, "init")
        _git(root, "config", "user.name", "ShadowSkillBench test")
        _git(root, "config", "user.email", "test@example.invalid")
        _git(root, "add", ".")
        _git(root, "commit", "-m", "freeze")
        freeze_commit = _git(root, "rev-parse", "HEAD")
        protocol_tag = "shadowskillbench-protocol-v1.0.0-local"
        _git(root, "tag", protocol_tag, freeze_commit)
        receipt = {
            "schema_version": "2.0",
            "custody_mode": "LOCAL_HASH_CUSTODY",
            "freeze_commit": freeze_commit,
            "protocol_tag": protocol_tag,
            "freeze_manifest_hash": "sha256:" + digest,
            "custody_locator": "urn:git:" + freeze_commit,
            "created_at": "2026-08-24T12:00:00Z",
            "verification_result": "VERIFIED_LOCAL",
        }
    receipt_path = protocol / "anchor_receipt.json"
    receipt_path.write_bytes(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
    if local_custody:
        _git(root, "add", "protocol/anchor_receipt.json")
        _git(root, "commit", "-m", "record local custody receipt")
    return manifest_path, detached_path, receipt_path


def test_report_protocol_evidence_is_derived_from_verified_files(tmp_path: Path) -> None:
    manifest, _, receipt = _write_authorization(tmp_path)

    evidence = cli._authorized_protocol_evidence(tmp_path)

    assert evidence.freeze_manifest_sha256 == "sha256:" + sha256(manifest.read_bytes()).hexdigest()
    assert evidence.protocol_tag == "shadowskillbench-protocol-v1.0.0-local"
    assert evidence.code_commit is not None
    assert evidence.external_anchor_locator is None
    assert evidence.custody_mode == "LOCAL_HASH_CUSTODY"
    assert evidence.anchor_receipt_sha256 == sha256_ref(json.loads(receipt.read_bytes()))
    inputs = json.loads(manifest.read_bytes())["inputs"]
    prompt_hashes = tuple(
        item["sha256"]
        for item in inputs
        if item["role"] in {"compiler_prompt", "executor_system_prompt"}
    )
    assert evidence.prompt_hashes == prompt_hashes


def test_report_protocol_evidence_accepts_verified_local_confirmatory_custody(
    tmp_path: Path,
) -> None:
    manifest_path, _, _ = _write_authorization(
        tmp_path, local_custody=True, confirmatory_prompt=True
    )

    evidence = cli._authorized_protocol_evidence(tmp_path)

    assert evidence.external_anchor_locator is None
    assert evidence.custody_locator is not None
    assert evidence.custody_locator.startswith("urn:git:")
    assert evidence.custody_mode == "LOCAL_HASH_CUSTODY"
    assert evidence.anchor_verified is True
    inputs = json.loads(manifest_path.read_bytes())["inputs"]
    prompt_hashes = {
        item["role"]: item["sha256"]
        for item in inputs
        if item["role"] in {"confirmatory_compiler_prompt", "executor_system_prompt"}
    }
    assert evidence.prompt_hashes == (
        prompt_hashes["confirmatory_compiler_prompt"],
        prompt_hashes["executor_system_prompt"],
    )
    assert (
        evidence.reproduce_command
        == "uv run shadowskillbench reproduce --protocol protocol/freeze_manifest.json"
    )


@pytest.mark.parametrize("target", ("receipt", "freeze", "detached", "noncanonical_receipt"))
def test_report_protocol_evidence_rejects_tampered_or_cross_bound_custody(
    tmp_path: Path, target: str
) -> None:
    manifest, detached, receipt = _write_authorization(tmp_path)
    if target == "receipt":
        payload = json.loads(receipt.read_bytes())
        payload["freeze_manifest_hash"] = "sha256:" + "0" * 64
        receipt.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    elif target == "noncanonical_receipt":
        receipt.write_bytes(receipt.read_bytes() + b"\n")
    elif target == "freeze":
        manifest.write_bytes(manifest.read_bytes() + b" ")
    else:
        detached.write_bytes(b"0" * 64 + b"  freeze_manifest.json\n")

    expected = (
        "HOLD_NONCANONICAL_CONFIRMATORY_AUTHORIZATION"
        if target == "noncanonical_receipt"
        else "HOLD_INVALID_CONFIRMATORY_AUTHORIZATION"
    )
    with pytest.raises(SealedAnalysisHold, match=expected):
        cli._authorized_protocol_evidence(tmp_path)
