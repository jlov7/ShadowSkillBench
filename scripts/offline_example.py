"""Compatibility wrapper for the packaged offline access example."""

from shadowskillbench.examples.offline_access import (
    OfflineExampleResult,
    main,
    run_example,
)

__all__ = ["OfflineExampleResult", "main", "run_example"]


if __name__ == "__main__":
    raise SystemExit(main())
