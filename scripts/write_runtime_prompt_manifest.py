"""Generate or check the frozen canonical runtime-prompt source manifest."""

from __future__ import annotations

import argparse

from shadowskillbench.protocol.runtime_prompt_manifest import (
    RUNTIME_PROMPT_MANIFEST_PATH,
    runtime_prompt_manifest_bytes,
    runtime_prompt_manifest_matches,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    target = RUNTIME_PROMPT_MANIFEST_PATH
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(runtime_prompt_manifest_bytes())
        return 0
    if not runtime_prompt_manifest_matches(target):
        raise SystemExit(f"runtime prompt manifest is stale: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
