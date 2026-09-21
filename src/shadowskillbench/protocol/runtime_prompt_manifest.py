"""Canonical commitment for the fixed sources of confirmatory executor prompts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from shadowskillbench.episodes.context import _RUNTIME_TEXT
from shadowskillbench.experiments.ollama_gpt_oss_profile import (
    OLLAMA_GPT_OSS_EXECUTOR_MODEL,
    OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
    OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
    OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
)

RUNTIME_PROMPT_MANIFEST_PATH = Path("protocol/commitments/runtime_prompt_manifest.json")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _source_path(path: str) -> Path:
    repository_path = _REPOSITORY_ROOT / path
    if repository_path.is_file():
        return repository_path

    package_relative = Path(path)
    source_prefix = Path("src/shadowskillbench")
    try:
        package_relative = package_relative.relative_to(source_prefix)
        packaged_path = _PACKAGE_ROOT / package_relative
    except ValueError:
        packaged_path = _PACKAGE_ROOT / "_runtime_sources" / package_relative

    if packaged_path.is_file():
        return packaged_path
    raise FileNotFoundError(path)


def _raw_sha256(path: str) -> str:
    return f"sha256:{sha256(_source_path(path).read_bytes()).hexdigest()}"


def runtime_prompt_manifest_bytes() -> bytes:
    """Return the canonical runtime-prompt source manifest bytes."""

    system_path = "prompts/executor_system.md"
    value = {
        "profile": "SSB-CONFIRMATORY-RUNTIME-PROMPT-MANIFEST6",
        "system_prompt": {
            "path": system_path,
            "source_sha256": _raw_sha256(system_path),
            "runtime_content_sha256": f"sha256:{sha256(_RUNTIME_TEXT.encode('utf-8')).hexdigest()}",
        },
        "context_assembler": {
            "path": "src/shadowskillbench/episodes/context.py",
            "source_sha256": _raw_sha256("src/shadowskillbench/episodes/context.py"),
        },
        "native_executor": {
            "adapter_path": "src/shadowskillbench/models/ollama_native.py",
            "adapter_sha256": _raw_sha256("src/shadowskillbench/models/ollama_native.py"),
            "renderer_path": "src/shadowskillbench/experiments/ollama_gpt_oss_profile.py",
            "renderer_sha256": _raw_sha256(
                "src/shadowskillbench/experiments/ollama_gpt_oss_profile.py"
            ),
            "model": OLLAMA_GPT_OSS_EXECUTOR_MODEL,
            "model_digest": OLLAMA_GPT_OSS_EXECUTOR_MODEL_DIGEST,
            "profile": OLLAMA_GPT_OSS_EXECUTOR_PROFILE,
            "request_profile": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE,
            "request_profile_hash": OLLAMA_GPT_OSS_NATIVE_EXECUTOR_REQUEST_PROFILE_HASH,
        },
        "executor_configuration": {
            "path": "config/executor.yaml",
            "source_sha256": _raw_sha256("config/executor.yaml"),
        },
    }
    from shadowskillbench.core.hashing import canonical_json_bytes

    return canonical_json_bytes(value)


def runtime_prompt_manifest_matches(path: Path) -> bool:
    """Return whether ``path`` is the exact current runtime-prompt artifact."""

    try:
        return (
            path.is_file()
            and not path.is_symlink()
            and path.read_bytes() == runtime_prompt_manifest_bytes()
        )
    except OSError:
        return False


__all__ = [
    "RUNTIME_PROMPT_MANIFEST_PATH",
    "runtime_prompt_manifest_bytes",
    "runtime_prompt_manifest_matches",
]
