"""Check the complete repository candidate for public-surface hazards."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()
SKIP_PARTS = frozenset(
    {
        ".git",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)
REQUIRED_FILES = (
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CHANGELOG.md",
)
SENSITIVE_NAMES = re.compile(
    r"(^|/)(\.env(?:\..*)?|\.netrc|\.npmrc|\.pypirc|credentials?(?:\..*)?|"
    r"secrets?(?:\..*)?|service[-_]?account(?:\..*)?|id_(?:rsa|ed25519)|"
    r".*\.(?:key|p12|pfx|pem))$",
    re.IGNORECASE,
)
PRIVATE_REPOSITORY_PATHS = re.compile(
    r"(^|/)(artifacts|evidence|implementation|private-review)(/|$)|"
    r"(^|/)(CODEX_KICKOFF_PROMPT|PROJECT_SITREP|README_FIRST|RELEASE_EVIDENCE|"
    r"REVIEW_START_HERE)\.md$",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN [A-Z0-9 ]+ PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(rb"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bnpm_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bsk_live_[A-Za-z0-9]{20,}\b"),
)
LOCAL_IDENTITY_PATTERNS = (
    re.compile(rb"/(?:Users|home)/[^/\s]+/"),
    re.compile(rb"[A-Z]:\\Users\\[^\\\s]+\\", re.IGNORECASE),
)
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
HTML_SOURCE = re.compile(
    r'<(?:img|source)\b[^>]*\b(?:src|srcset)=["\']([^"\']+)["\']', re.IGNORECASE
)
ARCHIVAL_LINK_ALLOWLIST: frozenset[str] = frozenset()


def is_distribution_path(relative: str) -> bool:
    """Return whether a path is part of the complete public repository."""
    return not SKIP_PARTS.intersection(Path(relative).parts)


def repository_paths(root: Path = ROOT) -> tuple[str, ...]:
    """Return tracked and proposed files, with a filesystem fallback before Git init."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        paths = (item for item in result.stdout.decode().split("\0") if item)
    else:
        paths = (path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    return tuple(sorted(path for path in paths if is_distribution_path(path)))


def payload_findings(root: Path, relative: str, payload: bytes) -> list[str]:
    findings: list[str] = []
    if SENSITIVE_NAMES.search(relative):
        return [f"sensitive-looking repository path: {relative}"]
    if PRIVATE_REPOSITORY_PATHS.search(relative):
        findings.append(f"private or internal repository path: {relative}")
    if relative != SELF:
        for pattern in (*SECRET_PATTERNS, *LOCAL_IDENTITY_PATTERNS):
            if pattern.search(payload):
                findings.append(f"credential or local-identity pattern in: {relative}")
                break
    if relative.lower().endswith((".md", ".markdown")) and relative not in ARCHIVAL_LINK_ALLOWLIST:
        text = payload.decode("utf-8", errors="replace")
        targets = [(target, "Markdown link") for target in MARKDOWN_LINK.findall(text)]
        targets.extend((target, "HTML source") for target in HTML_SOURCE.findall(text))
        for target, kind in targets:
            target = target.strip().strip("<>").split()[0].split("#", 1)[0]
            if not target or re.match(r"(?:[a-z][a-z0-9+.-]*:|//)", target, re.IGNORECASE):
                continue
            resolved = (
                root / target.lstrip("/")
                if target.startswith("/")
                else (root / relative).parent / target
            )
            if not resolved.exists():
                findings.append(f"broken local {kind}: {relative} -> {target}")
    return findings


def self_check() -> int:
    fixture_root = Path(__file__).resolve().parent
    assert payload_findings(fixture_root, "fixture.md", b"[missing](missing.md)") == [
        "broken local Markdown link: fixture.md -> missing.md"
    ]
    assert (
        payload_findings(fixture_root, "fixture.md", b"[external](https://example.com/nope)") == []
    )
    assert payload_findings(fixture_root, ".netrc", b"machine example.invalid") == [
        "sensitive-looking repository path: .netrc"
    ]
    assert payload_findings(fixture_root, "artifacts/result.json", b"{}") == [
        "private or internal repository path: artifacts/result.json"
    ]
    assert payload_findings(fixture_root, "fixture.txt", b"/Users/example/project") == [
        "credential or local-identity pattern in: fixture.txt"
    ]
    assert is_distribution_path("docs/README.md")
    assert is_distribution_path("tests/test_release_tooling.py")
    assert not is_distribution_path("workbench/node_modules/pkg/index.js")
    assert not ARCHIVAL_LINK_ALLOWLIST
    print("PUBLIC_SURFACE_SELF_CHECK_OK")
    return 0


def main() -> int:
    if "--self-check" in sys.argv[1:]:
        return self_check()
    findings: list[str] = []
    paths = repository_paths()
    for required in REQUIRED_FILES:
        if required not in paths:
            findings.append(f"missing governance file: {required}")
    for relative in paths:
        path = ROOT / relative
        try:
            payload = path.read_bytes()
        except OSError as error:
            findings.append(f"unreadable repository path: {relative}: {error}")
            continue
        findings.extend(payload_findings(ROOT, relative, payload))
    if findings:
        print("PUBLIC_SURFACE_SCAN_FAILED")
        print("\n".join(f"- {finding}" for finding in sorted(set(findings))))
        return 1
    print(f"PUBLIC_SURFACE_SCAN_OK repository_files={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
