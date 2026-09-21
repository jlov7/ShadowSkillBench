# Contributing to ShadowSkillBench

Thanks for helping improve the project. Contributions should preserve the
synthetic-only v1 boundary, deterministic evidence model, and explicit HOLD
state when research inputs or human authorization are absent.

## Before opening a change

Use the repository-pinned toolchains:

```bash
uv sync --all-groups --frozen --no-editable
make verify
pnpm --dir workbench install --frozen-lockfile
pnpm --dir workbench run check
pnpm --dir workbench run test
```

The workbench build and E2E checks are also run by CI. Do not run provider or
confirmatory commands as part of ordinary development validation.

For the public-surface and dependency gates, run:

```bash
uv lock --check
make public-surface
make sbom SBOM_OUTPUT=/tmp/shadowskillbench.cdx.json
make pack-manifest
make verify-public
pnpm --dir workbench audit --audit-level high
```

The final command consults the package registry and may be unavailable in an
offline environment; the lockfile and local scanner/SBOM checks remain fully
offline and deterministic.

## Change boundaries

- Keep real personal, client, credential, and private-policy data out of the
  repository.
- Do not weaken custody, leakage, protocol-freeze, or claim-ceiling checks to
  make a local command pass.
- Label development fixtures and results as practice evidence.
- Add or update focused tests for behavior changes and preserve the lockfiles.
- Keep protocol identity, preregistration, and confirmatory execution under the
  documented human-approval gates.

## Review checklist

Describe the change, the exact verification commands, and any remaining HOLD
or unavailable evidence. For research-facing changes, identify the affected
claim or estimand and the corresponding protocol or ledger entry.
