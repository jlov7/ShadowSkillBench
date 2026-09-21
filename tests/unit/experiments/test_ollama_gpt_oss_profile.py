from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.experiments.live_pilot import (
    OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
    OLLAMA_GPT_OSS_OBSERVED_ACCESS_R100_COMPILER_INPUT_TOKENS,
    OLLAMA_GPT_OSS_OBSERVED_ACCESS_R100_CONTEXT_BOUND,
    LivePilotDescriptor,
    LivePilotError,
)
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
    OLLAMA_GPT_OSS_CONTEXT_LENGTH,
    OLLAMA_GPT_OSS_CURRENT_DATE,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
    OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
    OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_LEGACY_V9_PROFILE,
    OLLAMA_GPT_OSS_MODEL,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
    OLLAMA_GPT_OSS_PROFILE,
    OLLAMA_GPT_OSS_PROVIDER,
    OLLAMA_GPT_OSS_REASONING_EFFORT,
    OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
    OLLAMA_GPT_OSS_SAMPLING_TOP_P,
    OLLAMA_GPT_OSS_SERVER_VERSION,
    HarmonyProbeMessage,
    OllamaRoleProfileError,
    canonical_role_profile_receipt,
    load_ollama_role_profile_receipt,
    ollama_gpt_oss_executor_modelfile_bytes,
    ollama_gpt_oss_executor_modelfile_hash,
    ollama_gpt_oss_executor_template,
    ollama_gpt_oss_executor_template_hash,
    ollama_gpt_oss_modelfile_bytes,
    ollama_gpt_oss_profile_projection,
    ollama_gpt_oss_template_hash,
    render_harmony_native_generate_prompt,
    render_harmony_role_probe,
    role_conformance_probe_projection,
    role_probe_projection,
    validate_ollama_role_profile_receipt,
)
from shadowskillbench.models import Message


def _receipt() -> dict[str, object]:
    digest = "sha256:" + "1" * 64
    probe = role_conformance_probe_projection()
    return json.loads(
        canonical_role_profile_receipt(
            digest,
            baseline={
                "raw_request_hash": "sha256:" + "a" * 64,
                "raw_response_hash": "sha256:" + "b" * 64,
                "prompt_tokens": 100,
                "response_marker": probe["baseline_expected_marker"],
            },
            developer={
                "raw_request_hash": "sha256:" + "c" * 64,
                "raw_response_hash": "sha256:" + "d" * 64,
                "prompt_tokens": 101,
                "response_marker": probe["developer_expected_marker"],
            },
            executor_model_digest=OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
        )
    )


def _native_executor_fields() -> dict[str, object]:
    return {
        "executor_endpoint": "http://127.0.0.1:11434/api/generate",
        "executor_transport": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT,
        "executor_request_profile": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
        "executor_request_profile_hash": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
        "executor_think": False,
        "executor_model_name": "ssb-gpt-oss-20b-harmony-final:v2",
        "executor_model_version": OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
        "executor_profile": OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
        "executor_modelfile_sha256": ollama_gpt_oss_executor_modelfile_hash(),
        "executor_template_sha256": ollama_gpt_oss_executor_template_hash(),
    }


def _load_verifier(monkeypatch: pytest.MonkeyPatch) -> object:
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "test_ollama_role_profile_verifier",
        root / "scripts" / "verify_ollama_gpt_oss_role_profile.py",
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, verifier)
    spec.loader.exec_module(verifier)
    return verifier


def test_modelfile_retains_native_system_scaffold_and_each_pilot_relevant_role() -> None:
    source = ollama_gpt_oss_modelfile_bytes().decode("utf-8")

    assert "FROM gpt-oss:20b" in source
    assert "PARAMETER num_ctx 131072" in source
    assert (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI."
        in source
    )
    assert "Knowledge cutoff: 2024-06" in source
    assert f"Current date: {OLLAMA_GPT_OSS_CURRENT_DATE}" in source
    assert (
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
        in source
    )
    assert "# Instructions:\n{{ .System }}<|end|>" in source
    assert '{{- if eq .Role "developer" }}<|start|>developer<|message|>' in source
    assert '{{- else if eq .Role "user" }}<|start|>user<|message|>' in source
    assert '{{- else if eq .Role "assistant" }}<|start|>assistant<|channel|>final' in source
    assert '{{- else if eq .Role "tool" }}<|start|>tool<|message|>' in source
    assert "Reasoning: {{ .ThinkLevel }}" in source
    assert source.rstrip().endswith('<|start|>assistant"""')
    assert ollama_gpt_oss_template_hash().startswith("sha256:")
    assert ollama_gpt_oss_profile_projection()["current_date"] == OLLAMA_GPT_OSS_CURRENT_DATE
    assert ollama_gpt_oss_profile_projection()["sampling_temperature"] == 1.0
    assert ollama_gpt_oss_profile_projection()["executor_sampling_temperature"] == 1.0
    assert ollama_gpt_oss_profile_projection()["ollama_server_version"] == "0.33.2"
    assert OLLAMA_GPT_OSS_SAMPLING_TOP_P == 1.0
    executor_source = ollama_gpt_oss_executor_modelfile_bytes().decode("utf-8")
    assert executor_source.rstrip().endswith('<|start|>assistant<|channel|>final<|message|>"""')
    assert ollama_gpt_oss_executor_template_hash().startswith("sha256:")
    assert ollama_gpt_oss_executor_modelfile_hash().startswith("sha256:")


def test_verifier_normalizes_bare_api_tag_digests_for_base_and_derived_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier = _load_verifier(monkeypatch)
    base_bare = OLLAMA_GPT_OSS_BASE_MODEL_DIGEST.removeprefix("sha256:").upper()
    derived_bare = "1" * 64

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "models": [
                        {"name": "gpt-oss:20b", "digest": base_bare},
                        {"name": OLLAMA_GPT_OSS_MODEL, "digest": derived_bare},
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(verifier, "urlopen", lambda *_args, **_kwargs: Response())

    assert verifier._model_digest("http://127.0.0.1:11434", "gpt-oss:20b") == (
        OLLAMA_GPT_OSS_BASE_MODEL_DIGEST
    )
    assert verifier._model_digest("http://127.0.0.1:11434", OLLAMA_GPT_OSS_MODEL) == (
        "sha256:" + derived_bare
    )

    class VersionResponse:
        def __enter__(self) -> VersionResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"version":"0.33.1"}'

    monkeypatch.setattr(verifier, "urlopen", lambda *_args, **_kwargs: VersionResponse())
    assert verifier._server_version("http://127.0.0.1:11434") == "0.33.1"

    class ProbeResponse:
        def __enter__(self) -> ProbeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": OLLAMA_GPT_OSS_MODEL,
                    "choices": [{"message": {"content": ""}}],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 256,
                        "total_tokens": 267,
                    },
                }
            ).encode("utf-8")

    request_bodies: list[dict[str, object]] = []

    def open_probe(request: object, **_kwargs: object) -> ProbeResponse:
        request_bodies.append(json.loads(request.data))  # type: ignore[attr-defined]
        return ProbeResponse()

    monkeypatch.setattr(verifier, "urlopen", open_probe)
    with pytest.raises(verifier.ProbeFailure) as raised:
        verifier._completion("http://127.0.0.1:11434", "test-key", [], "baseline")
    assert request_bodies[0]["temperature"] == OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE
    assert raised.value.code == "ROLE_PROBE_RESPONSE_INVALID"
    failure_path = tmp_path / "probe.failed.json"
    failure_evidence = {"code": raised.value.code, **raised.value.evidence}
    failure_evidence_before = json.loads(json.dumps(failure_evidence))
    verifier._write_failure(
        failure_path,
        "sha256:" + "1" * 64,
        OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
        failure_evidence,
    )
    assert failure_evidence == failure_evidence_before
    receipt = json.loads(failure_path.read_bytes())
    assert receipt["failure"] == {
        "call": "baseline",
        "code": "ROLE_PROBE_RESPONSE_INVALID",
        "raw_request_hash": receipt["failure"]["raw_request_hash"],
        "raw_response_hash": receipt["failure"]["raw_response_hash"],
        "usage": {"completion_tokens": 256, "prompt_tokens": 11, "total_tokens": 267},
    }
    assert set(receipt["failure"]) == {
        "call",
        "code",
        "raw_request_hash",
        "raw_response_hash",
        "usage",
    }


def test_verifier_records_both_call_summaries_for_post_call_conformance_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier = _load_verifier(monkeypatch)
    probe = role_conformance_probe_projection()
    profile_digest = "sha256:" + "1" * 64
    baseline = {
        "raw_request_hash": "sha256:" + "a" * 64,
        "raw_response_hash": "sha256:" + "b" * 64,
        "prompt_tokens": 108,
        "response_marker": "unexpected-baseline-marker",
        "usage": {"prompt_tokens": 108, "completion_tokens": 16, "total_tokens": 124},
    }
    developer = {
        "raw_request_hash": baseline["raw_request_hash"],
        "raw_response_hash": baseline["raw_response_hash"],
        "prompt_tokens": 108,
        "response_marker": "unexpected-developer-marker",
        "usage": {"prompt_tokens": 108, "completion_tokens": 16, "total_tokens": 124},
    }
    baseline_before = json.loads(json.dumps(baseline))
    developer_before = json.loads(json.dumps(developer))
    responses = iter((baseline, developer))
    calls: list[tuple[str, list[dict[str, str]]]] = []
    output = tmp_path / "probe.json"

    monkeypatch.setattr(
        verifier,
        "_model_digest",
        lambda _host, model: (
            OLLAMA_GPT_OSS_BASE_MODEL_DIGEST
            if model == "gpt-oss:20b"
            else OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST
            if model == OLLAMA_GPT_OSS_EXECUTOR_MODEL
            else profile_digest
        ),
    )
    monkeypatch.setattr(verifier, "_server_version", lambda _host: "0.33.2")

    def command(*args: str) -> str:
        template = (
            ollama_gpt_oss_executor_template()
            if OLLAMA_GPT_OSS_EXECUTOR_MODEL in args
            else verifier.ollama_gpt_oss_template()
        )
        return f"PARAMETER num_ctx {OLLAMA_GPT_OSS_CONTEXT_LENGTH}\n{template}"

    monkeypatch.setattr(verifier, "_command", command)

    def completion(
        _host: str, _api_key: str, messages: list[dict[str, str]], call: str
    ) -> dict[str, object]:
        calls.append((call, messages))
        return next(responses)

    monkeypatch.setattr(verifier, "_completion", completion)
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["verify-role-profile", "--output", str(output)])

    with pytest.raises(RuntimeError, match="BASELINE_MARKER_MISMATCH"):
        verifier.main()

    failure_path = tmp_path / "probe.failed.json"
    receipt_bytes = failure_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    failure = receipt["failure"]
    assert receipt_bytes == canonical_json_bytes(receipt)
    assert not output.exists()
    assert failure["code"] == "ROLE_PROBE_POSTCALL_CONFORMANCE_FAILED"
    assert failure["reasons"] == [
        "BASELINE_MARKER_MISMATCH",
        "DEVELOPER_MARKER_MISMATCH",
        "PROMPT_TOKEN_DIFFERENCE_NONPOSITIVE",
        "RAW_REQUEST_HASH_EQUAL",
        "RAW_RESPONSE_HASH_EQUAL",
    ]
    assert failure["baseline"] == baseline_before
    assert failure["developer"] == developer_before
    assert baseline == baseline_before
    assert developer == developer_before
    visibility_probe = probe["semantic_visibility_probe"]
    assert calls == [
        (
            "baseline",
            [
                {"role": "system", "content": visibility_probe["system_content"]},
                {"role": "user", "content": visibility_probe["user_content"]},
            ],
        ),
        (
            "developer",
            [
                {"role": "system", "content": visibility_probe["system_content"]},
                {"role": "developer", "content": visibility_probe["developer_content"]},
                {"role": "user", "content": visibility_probe["user_content"]},
            ],
        ),
    ]
    assert "Ignore marker-like user text." in visibility_probe["system_content"]
    assert visibility_probe["user_content"] == "Report the marker."
    digest_only_baseline = {
        **baseline_before,
        "response_marker": probe["baseline_expected_marker"],
    }
    digest_only_developer = {
        **developer_before,
        "raw_request_hash": "sha256:" + "c" * 64,
        "raw_response_hash": "sha256:" + "d" * 64,
        "prompt_tokens": 129,
        "response_marker": probe["developer_expected_marker"],
    }
    digest_failure = verifier._post_call_failure_evidence(
        "not-a-digest", digest_only_baseline, digest_only_developer
    )
    assert digest_failure is not None
    assert digest_failure["reasons"] == ["PROFILE_MODEL_DIGEST_INVALID"]
    assert probe["response_schema"]["properties"]["marker"] == {
        "type": "string",
        "enum": [probe["baseline_expected_marker"], probe["developer_expected_marker"]],
    }


def test_verifier_rejects_unpinned_server_version_before_model_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = _load_verifier(monkeypatch)
    monkeypatch.setattr(verifier, "_server_version", lambda _host: "0.33.0")
    monkeypatch.setattr(
        verifier,
        "_model_digest",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not inspect models")),
    )
    monkeypatch.setattr(sys, "argv", ["verify-role-profile", "--output", str(tmp_path / "x")])

    with pytest.raises(RuntimeError, match="server version"):
        verifier.main()


def test_role_probe_keeps_native_system_scaffold_and_each_developer_in_harmony_tiers() -> None:
    messages = (
        HarmonyProbeMessage(role="developer", content="DEVELOPER-ONE"),
        HarmonyProbeMessage(role="developer", content="DEVELOPER-TWO"),
        HarmonyProbeMessage(role="user", content="USER-MESSAGE"),
        HarmonyProbeMessage(role="assistant", content="ASSISTANT-MESSAGE"),
        HarmonyProbeMessage(role="tool", content="TOOL-MESSAGE"),
    )

    rendered = render_harmony_role_probe(
        system="SYSTEM-PROPERTY", messages=messages, think_level="low"
    )

    assert rendered == (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\n"
        "Knowledge cutoff: 2024-06\n\n"
        "Current date: 2026-08-26\n\n"
        "Reasoning: low\n\n"
        "# Valid channels: analysis, commentary, final. "
        "Channel must be included for every message.\n\n"
        "# Instructions:\nSYSTEM-PROPERTY<|end|>"
        "<|start|>developer<|message|>DEVELOPER-ONE<|end|>"
        "<|start|>developer<|message|>DEVELOPER-TWO<|end|>"
        "<|start|>user<|message|>USER-MESSAGE<|end|>"
        "<|start|>assistant<|channel|>final<|message|>ASSISTANT-MESSAGE<|end|>"
        "<|start|>tool<|message|>TOOL-MESSAGE<|end|>"
        "<|start|>assistant"
    )
    assert rendered.endswith("<|start|>assistant")
    assert not rendered.endswith("<|start|>assistant<|channel|>analysis<|message|>")
    assert role_probe_projection()["expected_render_sha256"] == sha256_ref(rendered)


def test_raw_native_generate_renderer_preserves_roles_and_omits_reasoning_for_think_false() -> None:
    rendered = render_harmony_native_generate_prompt(
        messages=(
            Message(role="system", content="SYSTEM-PROPERTY"),
            Message(role="developer", content="DEVELOPER-MESSAGE"),
            Message(role="user", content="USER-MESSAGE"),
            Message(role="assistant", content="ASSISTANT-MESSAGE"),
        )
    )

    assert "Reasoning:" not in rendered
    assert "# Instructions:\nSYSTEM-PROPERTY<|end|>" in rendered
    assert "<|start|>developer<|message|>DEVELOPER-MESSAGE<|end|>" in rendered
    assert "<|start|>user<|message|>USER-MESSAGE<|end|>" in rendered
    assert "<|start|>assistant<|channel|>final<|message|>ASSISTANT-MESSAGE<|end|>" in rendered
    assert rendered.endswith("<|start|>assistant<|channel|>final<|message|>")


@pytest.mark.parametrize(
    "messages, error",
    (
        ((Message(role="user", content="USER-MESSAGE"),), "leading system"),
        (
            (
                Message(role="system", content="SYSTEM-ONE"),
                Message(role="user", content="USER-MESSAGE"),
                Message(role="system", content="SYSTEM-TWO"),
            ),
            "contiguous and leading",
        ),
        (
            (
                Message(
                    role="system",
                    content="SYSTEM-ONE\n\n--- SSB SYSTEM MESSAGE BOUNDARY ---\n\n",
                ),
                Message(role="user", content="USER-MESSAGE"),
            ),
            "merge separator",
        ),
    ),
)
def test_raw_native_generate_renderer_rejects_missing_or_noncontiguous_system_messages(
    messages: tuple[Message, ...], error: str
) -> None:
    with pytest.raises(OllamaRoleProfileError, match=error):
        render_harmony_native_generate_prompt(messages=messages)


def test_raw_native_generate_renderer_merges_leading_system_messages_in_order() -> None:
    rendered = render_harmony_native_generate_prompt(
        messages=(
            Message(role="system", content="SYSTEM-ONE"),
            Message(role="system", content="SYSTEM-TWO"),
            Message(role="developer", content="DEVELOPER-MESSAGE"),
            Message(role="user", content="USER-MESSAGE"),
            Message(role="assistant", content="ASSISTANT-MESSAGE"),
        )
    )

    assert (
        "# Instructions:\nSYSTEM-ONE\n\n--- SSB SYSTEM MESSAGE BOUNDARY ---\n\n"
        "SYSTEM-TWO<|end|>" in rendered
    )
    assert rendered.index("SYSTEM-ONE") < rendered.index("SYSTEM-TWO")
    assert rendered.index("SYSTEM-TWO") < rendered.index("DEVELOPER-MESSAGE")
    assert rendered.index("DEVELOPER-MESSAGE") < rendered.index("USER-MESSAGE")
    assert rendered.index("USER-MESSAGE") < rendered.index("ASSISTANT-MESSAGE")


def test_role_profile_receipt_is_canonical_and_binds_the_pinned_template(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_bytes(canonical_json_bytes(_receipt()))

    receipt = load_ollama_role_profile_receipt(path)

    assert receipt["profile_model_digest"] == "sha256:" + "1" * 64
    assert receipt["template_sha256"] == ollama_gpt_oss_profile_projection()["template_sha256"]
    invalid_marker = _receipt()
    invalid_marker["developer"]["response_marker"] = "wrong"  # type: ignore[index]
    path.write_bytes(canonical_json_bytes(invalid_marker))
    with pytest.raises(OllamaRoleProfileError, match="canonical|bound"):
        load_ollama_role_profile_receipt(path)
    equal_prompt_tokens = _receipt()
    equal_prompt_tokens["developer"]["prompt_tokens"] = 100  # type: ignore[index]
    path.write_bytes(canonical_json_bytes(equal_prompt_tokens))
    with pytest.raises(OllamaRoleProfileError, match="bound"):
        load_ollama_role_profile_receipt(path)
    v4_shaped = _receipt()
    del v4_shaped["sampling_temperature"]
    with pytest.raises(OllamaRoleProfileError, match="bound"):
        validate_ollama_role_profile_receipt(v4_shaped)
    v6_shaped = _receipt()
    del v6_shaped["ollama_server_version"]
    with pytest.raises(OllamaRoleProfileError, match="bound"):
        validate_ollama_role_profile_receipt(v6_shaped)
    v7_shaped = _receipt()
    del v7_shaped["executor_reasoning_effort"]
    with pytest.raises(OllamaRoleProfileError, match="bound"):
        validate_ollama_role_profile_receipt(v7_shaped)
    v13_shaped = _receipt()
    del v13_shaped["executor_sampling_temperature"]
    with pytest.raises(OllamaRoleProfileError, match="bound"):
        validate_ollama_role_profile_receipt(v13_shaped)


def test_live_descriptor_admits_only_the_pinned_ollama_profile_and_receipt() -> None:
    receipt = _receipt()
    descriptor = LivePilotDescriptor(
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        api_key_environment="OLLAMA_API_KEY",
        provider=OLLAMA_GPT_OSS_PROVIDER,
        model=OLLAMA_GPT_OSS_MODEL,
        model_version="sha256:" + "1" * 64,
        ollama_server_version=OLLAMA_GPT_OSS_SERVER_VERSION,
        base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
        reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
        executor_reasoning_effort=OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
        sampling_temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
        executor_sampling_temperature=OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
        compiler_max_tokens=OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
        declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
        provider_profile=OLLAMA_GPT_OSS_PROFILE,
        role_profile_receipt_hash=sha256_ref(receipt),
        **_native_executor_fields(),
    )

    assert descriptor.projection()["ollama_role_profile"] == ollama_gpt_oss_profile_projection()
    assert descriptor.compiler_max_tokens == 16_384
    assert descriptor.sampling_temperature == 1.0
    assert descriptor.executor_sampling_temperature == 1.0
    assert descriptor.ollama_server_version == "0.33.2"
    assert descriptor.projection()["sampling_temperature"] == 1.0
    assert descriptor.projection()["executor_sampling_temperature"] == 1.0
    assert descriptor.projection()["reasoning_effort"] == "low"
    assert descriptor.projection()["executor_reasoning_effort"] == "none"
    assert descriptor.executor_capabilities.model == "ssb-gpt-oss-20b-harmony-final:v2"
    assert descriptor.executor_episode_model.model == "ssb-gpt-oss-20b-harmony-final:v2"
    assert OLLAMA_GPT_OSS_PROFILE == "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES11"
    assert descriptor.projection()["executor_transport"] == OLLAMA_GPT_OSS_NATIVE_EXECUTOR_TRANSPORT
    assert descriptor.projection()["executor_think"] is False
    with pytest.raises(LivePilotError, match="provider profile is not admitted"):
        LivePilotDescriptor(
            endpoint="http://127.0.0.1:11434/v1/chat/completions",
            api_key_environment="OLLAMA_API_KEY",
            provider=OLLAMA_GPT_OSS_PROVIDER,
            model=OLLAMA_GPT_OSS_MODEL,
            model_version="sha256:" + "1" * 64,
            ollama_server_version=OLLAMA_GPT_OSS_SERVER_VERSION,
            base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
            reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
            executor_reasoning_effort="low",
            sampling_temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            executor_sampling_temperature=OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
            compiler_max_tokens=OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
            declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
            provider_profile=OLLAMA_GPT_OSS_PROFILE,
            role_profile_receipt_hash=sha256_ref(receipt),
        )

    with pytest.raises(LivePilotError, match="provider profile is not admitted"):
        LivePilotDescriptor(
            endpoint="http://127.0.0.1:11434/v1/chat/completions",
            api_key_environment="OLLAMA_API_KEY",
            provider=OLLAMA_GPT_OSS_PROVIDER,
            model=OLLAMA_GPT_OSS_MODEL,
            model_version="sha256:" + "1" * 64,
            ollama_server_version="0.33.0",
            base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
            reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
            executor_reasoning_effort=OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
            sampling_temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            executor_sampling_temperature=OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
            compiler_max_tokens=OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
            declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
            provider_profile=OLLAMA_GPT_OSS_PROFILE,
            role_profile_receipt_hash=sha256_ref(receipt),
        )
    assert OLLAMA_GPT_OSS_OBSERVED_ACCESS_R100_COMPILER_INPUT_TOKENS == 74_526
    assert OLLAMA_GPT_OSS_OBSERVED_ACCESS_R100_CONTEXT_BOUND == 90_910
    assert OLLAMA_GPT_OSS_OBSERVED_ACCESS_R100_CONTEXT_BOUND <= OLLAMA_GPT_OSS_CONTEXT_LENGTH
    with pytest.raises(LivePilotError, match="compiler cap"):
        LivePilotDescriptor(
            endpoint="http://127.0.0.1:11434/v1/chat/completions",
            api_key_environment="OLLAMA_API_KEY",
            provider=OLLAMA_GPT_OSS_PROVIDER,
            model=OLLAMA_GPT_OSS_MODEL,
            model_version="sha256:" + "1" * 64,
            ollama_server_version=OLLAMA_GPT_OSS_SERVER_VERSION,
            base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
            reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
            executor_reasoning_effort=OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
            sampling_temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            executor_sampling_temperature=OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
            declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
            provider_profile=OLLAMA_GPT_OSS_PROFILE,
            role_profile_receipt_hash=sha256_ref(receipt),
        )
    with pytest.raises(LivePilotError, match="provider profile is not admitted"):
        LivePilotDescriptor(
            endpoint="http://127.0.0.1:11434/v1/chat/completions",
            api_key_environment="OLLAMA_API_KEY",
            provider=OLLAMA_GPT_OSS_PROVIDER,
            model=OLLAMA_GPT_OSS_MODEL,
            model_version="sha256:" + "1" * 64,
            ollama_server_version=OLLAMA_GPT_OSS_SERVER_VERSION,
            base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
            reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
            executor_reasoning_effort=OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
            sampling_temperature=0.0,
            compiler_max_tokens=OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
            declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
            provider_profile=OLLAMA_GPT_OSS_PROFILE,
            role_profile_receipt_hash=sha256_ref(receipt),
        )
    with pytest.raises(LivePilotError, match="provider profile is not admitted"):
        LivePilotDescriptor(
            endpoint="http://127.0.0.1:11434/v1/chat/completions",
            api_key_environment="OLLAMA_API_KEY",
            provider=OLLAMA_GPT_OSS_PROVIDER,
            model=OLLAMA_GPT_OSS_MODEL,
            model_version="sha256:" + "1" * 64,
            ollama_server_version=OLLAMA_GPT_OSS_SERVER_VERSION,
            base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
            reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
            executor_reasoning_effort=OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
            sampling_temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            executor_sampling_temperature=0.0,
            compiler_max_tokens=OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
            declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
            provider_profile=OLLAMA_GPT_OSS_PROFILE,
            role_profile_receipt_hash=sha256_ref(receipt),
        )
    with pytest.raises(LivePilotError, match="explicit loopback"):
        LivePilotDescriptor(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            api_key_environment="SSB_TEST_KEY",
            provider="scripted-pilot",
            model="pilot-model",
            model_version="test-v1",
            sampling_temperature=1.0,
        )
    with pytest.raises(LivePilotError, match="explicit loopback"):
        LivePilotDescriptor(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            api_key_environment="SSB_TEST_KEY",
            provider="scripted-pilot",
            model="pilot-model",
            model_version="test-v1",
            executor_sampling_temperature=1.0,
        )
    with pytest.raises(LivePilotError, match="provider profile is not admitted"):
        LivePilotDescriptor(
            endpoint="http://127.0.0.1:11434/v1/chat/completions",
            api_key_environment="OLLAMA_API_KEY",
            provider="ollama",
            model="qwen3.5:9b",
            model_version="unverified",
            provider_profile=OLLAMA_GPT_OSS_PROFILE,
        )


def test_legacy_roles4_receipt_cannot_admit_the_roles5_native_executor() -> None:
    probe = role_conformance_probe_projection()
    legacy_receipt = json.loads(
        canonical_role_profile_receipt(
            "sha256:" + "1" * 64,
            baseline={
                "raw_request_hash": "sha256:" + "a" * 64,
                "raw_response_hash": "sha256:" + "b" * 64,
                "prompt_tokens": 100,
                "response_marker": probe["baseline_expected_marker"],
            },
            developer={
                "raw_request_hash": "sha256:" + "c" * 64,
                "raw_response_hash": "sha256:" + "d" * 64,
                "prompt_tokens": 101,
                "response_marker": probe["developer_expected_marker"],
            },
            profile=OLLAMA_GPT_OSS_LEGACY_V9_PROFILE,
        )
    )
    descriptor = LivePilotDescriptor(
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        api_key_environment="OLLAMA_API_KEY",
        provider=OLLAMA_GPT_OSS_PROVIDER,
        model=OLLAMA_GPT_OSS_MODEL,
        model_version="sha256:" + "1" * 64,
        ollama_server_version=OLLAMA_GPT_OSS_SERVER_VERSION,
        base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
        reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
        executor_reasoning_effort=OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
        sampling_temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
        executor_sampling_temperature=OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
        compiler_max_tokens=OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
        declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
        provider_profile=OLLAMA_GPT_OSS_PROFILE,
        role_profile_receipt_hash=sha256_ref(legacy_receipt),
        **_native_executor_fields(),
    )

    with pytest.raises(LivePilotError, match="provider profile"):
        from shadowskillbench.experiments.live_pilot import _validate_role_profile_receipt

        _validate_role_profile_receipt(descriptor, legacy_receipt)


def test_roles5_rejects_a_native_executor_on_a_different_loopback_origin() -> None:
    receipt = _receipt()
    native_fields = _native_executor_fields()
    native_fields["executor_endpoint"] = "http://127.0.0.1:11435/api/generate"

    with pytest.raises(LivePilotError, match="transport is not pinned"):
        LivePilotDescriptor(
            endpoint="http://127.0.0.1:11434/v1/chat/completions",
            api_key_environment="OLLAMA_API_KEY",
            provider=OLLAMA_GPT_OSS_PROVIDER,
            model=OLLAMA_GPT_OSS_MODEL,
            model_version="sha256:" + "1" * 64,
            ollama_server_version=OLLAMA_GPT_OSS_SERVER_VERSION,
            base_model_digest=OLLAMA_GPT_OSS_BASE_MODEL_DIGEST,
            reasoning_effort=OLLAMA_GPT_OSS_REASONING_EFFORT,
            executor_reasoning_effort=OLLAMA_GPT_OSS_EXECUTOR_REASONING_EFFORT,
            sampling_temperature=OLLAMA_GPT_OSS_SAMPLING_TEMPERATURE,
            executor_sampling_temperature=OLLAMA_GPT_OSS_EXECUTOR_SAMPLING_TEMPERATURE,
            compiler_max_tokens=OLLAMA_GPT_OSS_COMPILER_MAX_TOKENS,
            declared_server_context_length=OLLAMA_GPT_OSS_CONTEXT_LENGTH,
            provider_profile=OLLAMA_GPT_OSS_PROFILE,
            role_profile_receipt_hash=sha256_ref(receipt),
            **native_fields,
        )


def test_current_executor_sampling_profile_keeps_roles10_receipts_at_zero() -> None:
    current = ollama_gpt_oss_profile_projection()
    legacy_profile = "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES10"
    legacy = ollama_gpt_oss_profile_projection(profile=legacy_profile)

    assert current["executor_profile"] == "SSB-OLLAMA-GPT-OSS-20B-HARMONY-FINAL3"
    assert current["native_executor_request_profile"] == (
        "SSB-OLLAMA-NATIVE-GENERATE-EXECUTOR-REQUEST2"
    )
    assert current["role_profile"] == "SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES11"
    assert legacy["role_profile"] == legacy_profile
    assert current["native_executor_request_hash"] == legacy["native_executor_request_hash"]
    assert current["executor_sampling_temperature"] == 1.0
    assert legacy["executor_sampling_temperature"] == 0.0
    assert legacy["executor_profile"] == "SSB-OLLAMA-GPT-OSS-20B-HARMONY-FINAL3"
    assert legacy["native_executor_request_profile"] == (
        "SSB-OLLAMA-NATIVE-GENERATE-EXECUTOR-REQUEST2"
    )
    receipt = json.loads(
        canonical_role_profile_receipt(
            "sha256:" + "1" * 64,
            baseline={
                "raw_request_hash": "sha256:" + "a" * 64,
                "raw_response_hash": "sha256:" + "b" * 64,
                "prompt_tokens": 100,
                "response_marker": role_conformance_probe_projection()["baseline_expected_marker"],
            },
            developer={
                "raw_request_hash": "sha256:" + "c" * 64,
                "raw_response_hash": "sha256:" + "d" * 64,
                "prompt_tokens": 101,
                "response_marker": role_conformance_probe_projection()["developer_expected_marker"],
            },
            profile=legacy_profile,
            executor_model_digest=OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
        )
    )

    assert validate_ollama_role_profile_receipt(receipt)["role_profile"] == (legacy_profile)
