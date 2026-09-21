from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from shadowskillbench.skills.gate2 import (
    Gate2Error,
    build_gate2_development_report,
    render_gate2_report,
    verify_gate2_report,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and compile the fixed Gate 2 development matrix."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "artifacts" / "development" / "gate2-development-report.json",
        help="JSON report path (default: artifacts/development/gate2-development-report.json)",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=_repo_root() / "prompts" / "skill_compiler.md",
        help="compiler prompt path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = asyncio.run(build_gate2_development_report(prompt_bytes=args.prompt.read_bytes()))
        verify_gate2_report(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_gate2_report(report), encoding="utf-8")
    except (Gate2Error, OSError, RuntimeError, ValueError) as error:
        print(f"Gate 2 failed: {error}", file=sys.stderr)
        return 1
    print(f"Gate 2 passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
