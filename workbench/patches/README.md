# Next lint dependency patch

`@next/eslint-plugin-next@16.3.3` uses `fast-glob` only to discover application
roots for `no-html-link-for-pages`. Its dependency chain includes the unpatched
[`braces` stack-exhaustion advisory](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm).

The version-scoped override replaces that dependency with `tinyglobby@0.2.17`,
already used by the workbench's tooling. The accompanying pnpm patch disables
directory expansion and preserves absolute paths, following the
[migration guide](https://superchupu.dev/tinyglobby/migration). The vulnerable
dependency chain is removed. Audit thresholds and exclusions are unchanged.

Qualification covers the current lint configuration, which has no custom
`next.rootDir`, plus the literal, relative, absolute, brace, and array forms
exercised in `tests/next-lint-roots.test.ts`. Glob libraries differ on other
forms, including trailing `/**` patterns. Additional layouts need regression
coverage. The tests also exercise an actual Next rule violation.

Revisit this exact-version patch when updating the lint plugin. Remove it when
upstream removes the vulnerable chain or supplies a compatible fix, then rerun
the workbench tests and full dependency audit.
