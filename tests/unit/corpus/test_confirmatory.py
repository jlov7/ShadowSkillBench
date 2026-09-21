from __future__ import annotations

import json
import os
import stat
import subprocess
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from shadowskillbench.corpus.confirmatory import (
    AUTHORITY_CLASSES,
    DOMAINS,
    RATIOS,
    ConfirmatoryHoldError,
    SplitInventory,
    _current_generator_hash,
    authorize_confirmatory_generation,
    generate_confirmatory_corpus,
    generate_confirmatory_from_files,
    inventory_from_development,
    load_model_visible_corpus,
    write_confirmatory_artifacts,
)
from shadowskillbench.corpus.development import generate_development_corpus
from shadowskillbench.episodes.models import EpisodeStage, ExperimentCondition
from shadowskillbench.experiments.planner import (
    ConditionBinding,
    HeldOutCaseBinding,
    PlannedEpisode,
)

DEVELOPMENT_SENTINEL = SplitInventory(
    frozenset({-1}),
    frozenset({"development_sentinel"}),
    frozenset({"sha256:" + "0" * 64}),
)
_LOCAL_PREREGISTRATION = b"fixture preregistration\n"


def _manifest_bytes() -> tuple[bytes, bytes]:
    payload = {
        "anchor_status": "PENDING_HUMAN_ANCHOR",
        "inputs": [
            {
                "path": "protocol/preregistration.md",
                "role": "preregistration_core",
                "sha256": "sha256:" + sha256(_LOCAL_PREREGISTRATION).hexdigest(),
            },
            {
                "path": "src/shadowskillbench/corpus/confirmatory.py",
                "role": "confirmatory_corpus",
                "sha256": _current_generator_hash(),
            },
        ],
        "preregistration_core": {
            "path": "protocol/preregistration.md",
            "sha256": "sha256:" + sha256(_LOCAL_PREREGISTRATION).hexdigest(),
        },
        "schema_version": "1.0",
    }
    manifest = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    return manifest, f"{sha256(manifest).hexdigest()}  freeze_manifest.json\n".encode()


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-08-28T12:00:00Z",
            "GIT_COMMITTER_DATE": "2026-08-28T12:00:00Z",
        },
        text=True,
    ).stdout.strip()


def _local_receipt(manifest: bytes, freeze_commit: str, protocol_tag: str) -> bytes:
    return json.dumps(
        {
            "schema_version": "2.0",
            "custody_mode": "LOCAL_HASH_CUSTODY",
            "freeze_commit": freeze_commit,
            "protocol_tag": protocol_tag,
            "freeze_manifest_hash": "sha256:" + sha256(manifest).hexdigest(),
            "custody_locator": "urn:git:" + freeze_commit,
            "created_at": "2026-08-28T12:00:00Z",
            "verification_result": "VERIFIED_LOCAL",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _local_custody_repository(
    repository_root: Path,
    manifest: bytes,
    detached: bytes,
    *,
    commit_receipt: bool = True,
    committed_detached: bytes | None = None,
) -> tuple[bytes, str, str]:
    protocol = repository_root / "protocol"
    protocol.mkdir(parents=True)
    (protocol / "preregistration.md").write_bytes(_LOCAL_PREREGISTRATION)
    confirmatory_path = repository_root / "src" / "shadowskillbench" / "corpus"
    confirmatory_path.mkdir(parents=True)
    confirmatory_path.joinpath("confirmatory.py").write_bytes(
        Path(__file__)
        .parents[3]
        .joinpath("src", "shadowskillbench", "corpus", "confirmatory.py")
        .read_bytes()
    )
    (protocol / "freeze_manifest.json").write_bytes(manifest)
    (protocol / "freeze_manifest.sha256").write_bytes(
        detached if committed_detached is None else committed_detached
    )
    _git(repository_root, "init")
    _git(repository_root, "config", "user.name", "ShadowSkillBench test")
    _git(repository_root, "config", "user.email", "test@example.invalid")
    _git(repository_root, "add", ".")
    _git(repository_root, "commit", "-m", "freeze")
    freeze_commit = _git(repository_root, "rev-parse", "HEAD")
    protocol_tag = "shadowskillbench-protocol-v1.0.0-local"
    _git(repository_root, "tag", protocol_tag, freeze_commit)
    receipt = _local_receipt(manifest, freeze_commit, protocol_tag)
    (protocol / "anchor_receipt.json").write_bytes(receipt)
    if commit_receipt:
        _git(repository_root, "add", "protocol/anchor_receipt.json")
        _git(repository_root, "commit", "-m", "record local custody receipt")
    return receipt, freeze_commit, protocol_tag


def _authorization(repository_root: Path):
    manifest, detached = _manifest_bytes()
    receipt, _, _ = _local_custody_repository(repository_root, manifest, detached)
    return authorize_confirmatory_generation(
        manifest,
        detached,
        receipt,
        repository_root=repository_root,
    )


def _external_receipt(manifest: bytes) -> bytes:
    return json.dumps(
        {
            "freeze_commit": "c" * 40,
            "signed_tag": "shadowskillbench-protocol-v1.0.0",
            "freeze_manifest_hash": "sha256:" + sha256(manifest).hexdigest(),
            "immutable_locator": "https://registry.example.invalid/releases/1",
            "anchored_at": "2026-08-24T12:00:00Z",
            "verification_result": "VERIFIED",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory):
    repository_root = tmp_path_factory.mktemp("confirmatory-custody")
    return generate_confirmatory_corpus(
        authorization=_authorization(repository_root),
        corpus_seed=104729,
        development_inventory=DEVELOPMENT_SENTINEL,
    )


def test_requires_exact_freeze_bytes_detached_hash_and_verified_anchor(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    with pytest.raises(ConfirmatoryHoldError):
        authorize_confirmatory_generation(
            manifest + b" ", detached, b"{}", repository_root=tmp_path
        )
    receipt = json.dumps(
        {
            "freeze_commit": "c" * 40,
            "signed_tag": "release-1",
            "freeze_manifest_hash": "sha256:" + sha256(manifest).hexdigest(),
            "immutable_locator": "https://registry.example.invalid/releases/1",
            "anchored_at": "2026-08-24T12:00:00Z",
            "verification_result": "PENDING",
        }
    ).encode()
    with pytest.raises(ConfirmatoryHoldError):
        authorize_confirmatory_generation(manifest, detached, receipt, repository_root=tmp_path)


def test_legacy_external_receipt_holds_without_cryptographic_verification(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    receipt = _external_receipt(manifest)

    with pytest.raises(ConfirmatoryHoldError, match="HOLD_UNVERIFIED_EXTERNAL_ANCHOR"):
        authorize_confirmatory_generation(manifest, detached, receipt, repository_root=tmp_path)


def test_local_hash_custody_authorizes_without_claiming_external_verification(
    tmp_path: Path,
) -> None:
    manifest, detached = _manifest_bytes()
    receipt, _, _ = _local_custody_repository(tmp_path, manifest, detached)

    authorization = authorize_confirmatory_generation(
        manifest,
        detached,
        receipt,
        repository_root=tmp_path,
    )

    assert authorization.receipt.custody_mode == "LOCAL_HASH_CUSTODY"
    assert authorization.receipt.externally_verified is False


def test_local_hash_custody_rejects_missing_freeze_commit(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    receipt, _, protocol_tag = _local_custody_repository(tmp_path, manifest, detached)
    forged = _local_receipt(manifest, "f" * 40, protocol_tag)
    receipt_path = tmp_path / "protocol" / "anchor_receipt.json"
    receipt_path.write_bytes(forged)
    _git(tmp_path, "add", "protocol/anchor_receipt.json")
    _git(tmp_path, "commit", "-m", "record forged receipt")

    with pytest.raises(ConfirmatoryHoldError, match="git verification failed"):
        authorize_confirmatory_generation(
            manifest,
            detached,
            forged,
            repository_root=tmp_path,
        )


def test_local_hash_custody_rejects_tag_for_a_different_commit(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    _, freeze_commit, _ = _local_custody_repository(tmp_path, manifest, detached)
    wrong_tag = "shadowskillbench-protocol-v1.0.0-other"
    _git(tmp_path, "tag", wrong_tag)
    forged = _local_receipt(manifest, freeze_commit, wrong_tag)
    receipt_path = tmp_path / "protocol" / "anchor_receipt.json"
    receipt_path.write_bytes(forged)
    _git(tmp_path, "add", "protocol/anchor_receipt.json")
    _git(tmp_path, "commit", "-m", "record tag mismatch receipt")

    with pytest.raises(ConfirmatoryHoldError, match="tag does not resolve"):
        authorize_confirmatory_generation(
            manifest,
            detached,
            forged,
            repository_root=tmp_path,
        )


def test_local_hash_custody_rejects_missing_protocol_tag(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    _, freeze_commit, _ = _local_custody_repository(tmp_path, manifest, detached)
    forged = _local_receipt(manifest, freeze_commit, "shadowskillbench-protocol-v1.0.0-missing")
    receipt_path = tmp_path / "protocol" / "anchor_receipt.json"
    receipt_path.write_bytes(forged)
    _git(tmp_path, "add", "protocol/anchor_receipt.json")
    _git(tmp_path, "commit", "-m", "record missing tag receipt")

    with pytest.raises(ConfirmatoryHoldError, match="git verification failed"):
        authorize_confirmatory_generation(
            manifest,
            detached,
            forged,
            repository_root=tmp_path,
        )


def test_local_hash_custody_rejects_manifest_tree_drift(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    _, freeze_commit, protocol_tag = _local_custody_repository(tmp_path, manifest, detached)
    drift_payload = json.loads(manifest)
    drift_payload["inputs"][0]["sha256"] = "sha256:" + "b" * 64
    drift_manifest = (
        json.dumps(drift_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    drift_detached = f"{sha256(drift_manifest).hexdigest()}  freeze_manifest.json\n".encode()
    forged = _local_receipt(drift_manifest, freeze_commit, protocol_tag)
    receipt_path = tmp_path / "protocol" / "anchor_receipt.json"
    receipt_path.write_bytes(forged)
    _git(tmp_path, "add", "protocol/anchor_receipt.json")
    _git(tmp_path, "commit", "-m", "record tree drift receipt")

    with pytest.raises(ConfirmatoryHoldError, match="freeze_manifest.json differs"):
        authorize_confirmatory_generation(
            drift_manifest,
            drift_detached,
            forged,
            repository_root=tmp_path,
        )


def test_local_hash_custody_rejects_detached_hash_tree_drift(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    receipt, _, _ = _local_custody_repository(
        tmp_path,
        manifest,
        detached,
        committed_detached=b"0" * 64 + b"  freeze_manifest.json\n",
    )

    with pytest.raises(ConfirmatoryHoldError, match="freeze_manifest.sha256 differs"):
        authorize_confirmatory_generation(
            manifest,
            detached,
            receipt,
            repository_root=tmp_path,
        )


def test_local_hash_custody_rejects_checked_out_freeze_input_drift(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    receipt, _, _ = _local_custody_repository(tmp_path, manifest, detached)
    (tmp_path / "protocol" / "preregistration.md").write_bytes(b"drift\n")

    with pytest.raises(ConfirmatoryHoldError, match="checked-out freeze input"):
        authorize_confirmatory_generation(
            manifest,
            detached,
            receipt,
            repository_root=tmp_path,
        )


def test_local_hash_custody_rejects_untracked_receipt(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    receipt, _, _ = _local_custody_repository(
        tmp_path,
        manifest,
        detached,
        commit_receipt=False,
    )

    with pytest.raises(ConfirmatoryHoldError, match="dirty or untracked"):
        authorize_confirmatory_generation(
            manifest,
            detached,
            receipt,
            repository_root=tmp_path,
        )


def test_local_hash_custody_rejects_dirty_receipt(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    receipt, _, _ = _local_custody_repository(tmp_path, manifest, detached)
    dirty_payload = json.loads(receipt)
    dirty_payload["created_at"] = "2026-08-29T12:00:00Z"
    dirty_receipt = json.dumps(
        dirty_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    (tmp_path / "protocol" / "anchor_receipt.json").write_bytes(dirty_receipt)

    with pytest.raises(ConfirmatoryHoldError, match="dirty or untracked"):
        authorize_confirmatory_generation(
            manifest,
            detached,
            dirty_receipt,
            repository_root=tmp_path,
        )


def test_local_hash_custody_requires_a_repository_root(tmp_path: Path) -> None:
    manifest, detached = _manifest_bytes()
    receipt = _local_receipt(manifest, "d" * 40, "shadowskillbench-protocol-v1.0.0-local")

    with pytest.raises(ConfirmatoryHoldError, match="requires a repository root"):
        authorize_confirmatory_generation(manifest, detached, receipt)
    with pytest.raises(ConfirmatoryHoldError, match="git verification failed"):
        authorize_confirmatory_generation(
            manifest,
            detached,
            receipt,
            repository_root=tmp_path,
        )


def test_authorization_cannot_be_directly_forged_after_validation(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    with pytest.raises(ConfirmatoryHoldError, match="detached manifest hash"):
        replace(authorization, freeze=replace(authorization.freeze, manifest_bytes=b"{}"))


def test_exact_preregistered_counts_and_deterministic_content(corpus, tmp_path: Path) -> None:
    second = generate_confirmatory_corpus(
        authorization=_authorization(tmp_path),
        corpus_seed=104729,
        development_inventory=DEVELOPMENT_SENTINEL,
    )
    assert second == corpus
    assert len(corpus.bundles) == 30
    assert len(corpus.stage_a_cases) == 40
    assert len(corpus.stage_b_cases) == 50
    for domain in DOMAINS:
        bundles = [bundle for bundle in corpus.bundles if bundle.domain == domain]
        stage_a = [case for case in corpus.stage_a_cases if case.domain == domain]
        stage_b = [case for case in corpus.stage_b_cases if case.domain == domain]
        assert Counter(bundle.contamination_ratio for bundle in bundles) == Counter(
            {ratio: 3 for ratio in RATIOS}
        )
        assert Counter(case.authority_class for case in stage_a) == Counter(
            {item: 4 for item in AUTHORITY_CLASSES}
        )
        assert Counter(case.authority_class for case in stage_b) == Counter(
            {item: 5 for item in AUTHORITY_CLASSES}
        )
        assert sum(bundle.contamination_ratio == Decimal("0.75") for bundle in bundles) == 3


def test_hidden_truth_never_enters_model_visible_projection(corpus) -> None:
    for case in corpus.stage_a_cases + corpus.stage_b_cases:
        visible = repr(case.model_visible_projection())
        assert "authority_records" not in visible
        assert "authority_class" not in visible
        assert "expected_decision" not in visible
        assert "sha256:" not in visible
        assert case.case_id in visible
        assert case.stage in visible


def test_case_identity_binds_directly_to_planner_execution_identity(corpus) -> None:
    case = corpus.stage_a_cases[0]
    assert case.case_id == case.source.task_case.case_id == case.source.domain_case.case_id
    held_out = HeldOutCaseBinding(
        domain=case.domain,
        case_id=case.case_id,
        case_manifest_hash=case.public_content_hash,
        world_hash=case.source.world_hash,
        authority_graph_hash=case.hidden_content_hash,
    )
    condition = ConditionBinding(
        domain=case.domain,
        condition=ExperimentCondition.A0_BARE,
        context_contract_hash="sha256:" + "f" * 64,
        policy_hash=None,
    )
    episode = PlannedEpisode(
        stage=EpisodeStage.CONFIRMATORY_A,
        condition=ExperimentCondition.A0_BARE,
        case=held_out,
        condition_binding=condition,
        repeat_index=1,
    )
    assert episode.case.case_id == case.source.task_case.case_id


def test_explicit_development_split_inventory_blocks_all_overlap_kinds(
    corpus, tmp_path: Path
) -> None:
    inventory = corpus.inventory
    for index, development in enumerate(
        (
            SplitInventory(
                frozenset({-1, next(iter(inventory.seeds))}),
                DEVELOPMENT_SENTINEL.entity_ids,
                DEVELOPMENT_SENTINEL.value_hashes,
            ),
            SplitInventory(
                DEVELOPMENT_SENTINEL.seeds,
                frozenset({"development_sentinel", next(iter(inventory.entity_ids))}),
                DEVELOPMENT_SENTINEL.value_hashes,
            ),
            SplitInventory(
                DEVELOPMENT_SENTINEL.seeds,
                DEVELOPMENT_SENTINEL.entity_ids,
                frozenset({"sha256:" + "0" * 64, next(iter(inventory.value_hashes))}),
            ),
        )
    ):
        with pytest.raises(ConfirmatoryHoldError, match="split overlap"):
            generate_confirmatory_corpus(
                authorization=_authorization(tmp_path / f"custody-{index}"),
                corpus_seed=104729,
                development_inventory=development,
            )


def test_development_inventory_derives_exact_split_boundary(tmp_path: Path) -> None:
    development = inventory_from_development(generate_development_corpus(104729))
    generated = generate_confirmatory_corpus(
        authorization=_authorization(tmp_path),
        corpus_seed=104729,
        development_inventory=development,
    )
    assert not (development.seeds & generated.inventory.seeds)
    assert not (development.entity_ids & generated.inventory.entity_ids)
    assert not (development.value_hashes & generated.inventory.value_hashes)


def test_artifacts_are_exclusive_and_read_only(tmp_path, corpus) -> None:
    artifacts = write_confirmatory_artifacts(corpus, tmp_path)
    public = json.loads(artifacts.public_corpus.read_bytes())
    hidden = json.loads(artifacts.hidden_scorer_custody.read_bytes())
    assert "case_truth" not in public
    assert "case_truth" in hidden
    assert load_model_visible_corpus(artifacts.public_corpus) == public
    with pytest.raises(ConfirmatoryHoldError, match="model-visible public corpus"):
        load_model_visible_corpus(artifacts.hidden_scorer_custody)
    for path, detached in (
        (artifacts.public_corpus, artifacts.public_sha256),
        (artifacts.hidden_scorer_custody, artifacts.hidden_sha256),
        (artifacts.manifest, artifacts.manifest_sha256),
    ):
        digest = sha256(path.read_bytes()).hexdigest()
        assert detached.read_text() == f"{digest}  {path.name}\n"
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert stat.S_IMODE(detached.stat().st_mode) == 0o444
    with pytest.raises(ConfirmatoryHoldError, match="overwrite"):
        write_confirmatory_artifacts(corpus, tmp_path)


def test_missing_real_custody_files_hold_before_generation(tmp_path) -> None:
    with pytest.raises(ConfirmatoryHoldError, match="custody files"):
        generate_confirmatory_from_files(
            manifest_path=tmp_path / "freeze_manifest.json",
            detached_sha256_path=tmp_path / "freeze_manifest.sha256",
            anchor_receipt_path=tmp_path / "anchor_receipt.json",
            corpus_seed=1,
            development_inventory=DEVELOPMENT_SENTINEL,
        )
