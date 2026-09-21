from __future__ import annotations

from shadowskillbench.corpus.development import (
    AuthorityClass,
    generate_development_corpus,
)
from shadowskillbench.experiments import gemma4_recovery
from shadowskillbench.experiments.ollama_gemma4_recovery_profile import (
    recovery_candidate_spec,
)


def test_recovery_precommits_two_fresh_candidate_selections() -> None:
    seen: set[str] = set()
    for candidate in ("primary-26b", "fallback-12b"):
        spec = recovery_candidate_spec(candidate)
        corpus = generate_development_corpus(spec.corpus_seed)
        commitment = gemma4_recovery.recovery_commitment(corpus, candidate)
        screen = gemma4_recovery.select_recovery_cells(commitment, corpus, "screen")
        validation = gemma4_recovery.select_recovery_cells(commitment, corpus, "validation")
        assert len(screen) == 8
        assert len(validation) == 40
        assert all(
            cell.stage == "stage_b"
            for cell in screen
            if cell.condition.value == "B3_DETERMINISTIC_GATE"
        )
        assert {cell.case_id for cell in screen}.isdisjoint({cell.case_id for cell in validation})
        assert all(
            next(
                case for case in corpus.cases if case.case_id == cell.case_id
            ).hidden_truth.authority_class
            is AuthorityClass.PRACTICE_MATCHES_ACTIVE_POLICY
            for cell in validation
        )
        seen.add(commitment["selection"]["corpus"]["content_hash"])
    assert len(seen) == 2


def test_recovery_client_rejects_non_loopback_endpoint() -> None:
    try:
        gemma4_recovery.build_recovery_client(
            "https://example.invalid/api/chat", "SSB_TEST", "primary-26b"
        )
    except gemma4_recovery.Gemma4RecoveryHold:
        pass
    else:
        raise AssertionError("non-loopback endpoint was admitted")
