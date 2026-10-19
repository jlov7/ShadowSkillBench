// @vitest-environment node

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { Linter, type ESLint } from "eslint";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const nextRequire = createRequire(require.resolve("eslint-config-next"));
const pluginEntry = nextRequire.resolve("@next/eslint-plugin-next");
const nextPlugin = nextRequire("@next/eslint-plugin-next") as ESLint.Plugin;
const { getRootDirs } = require(path.join(path.dirname(pluginEntry), "utils/get-root-dirs.js")) as {
  getRootDirs(context: { cwd: string; settings: { next?: { rootDir: string | string[] } } }): string[];
};

let fixtureRoot: string;

beforeAll(() => {
  const build = path.resolve("../build");
  mkdirSync(build, { recursive: true });
  fixtureRoot = mkdtempSync(path.join(build, "lint-roots-"));
  for (const app of ["web", "admin"]) {
    const pages = path.join(fixtureRoot, "apps", app, "pages");
    mkdirSync(pages, { recursive: true });
    writeFileSync(path.join(pages, "about.js"), "export default function About() {}\n");
  }
  mkdirSync(path.join(fixtureRoot, "apps/web/nested"));
  writeFileSync(path.join(fixtureRoot, "apps/regular-file"), "fixture\n");
});

afterAll(() => {
  if (fixtureRoot) rmSync(fixtureRoot, { recursive: true });
});

function roots(rootDir?: string | string[]) {
  return getRootDirs({
    cwd: fixtureRoot,
    settings: rootDir === undefined ? {} : { next: { rootDir } },
  }).map((dir) => path.resolve(dir)).sort();
}

describe("Next lint root discovery", () => {
  it("keeps the configured cwd when no rootDir is supplied", () => {
    expect(roots()).toEqual([fixtureRoot]);
  });

  it("keeps a literal root without expanding into its descendants", () => {
    expect(roots(path.join(fixtureRoot, "apps/web"))).toEqual([
      path.join(fixtureRoot, "apps/web"),
    ]);
  });

  it("resolves relative, absolute, and brace globs to directories only", () => {
    const expected = ["admin", "web"].map((app) => path.join(fixtureRoot, "apps", app));
    for (const pattern of [
      path.join(fixtureRoot, "apps/*"),
      path.relative(process.cwd(), path.join(fixtureRoot, "apps/*")),
      path.join(fixtureRoot, "apps/{web,admin}"),
    ]) {
      expect(roots(pattern)).toEqual(expected);
    }
  });

  it("supports root arrays and excludes missing roots", () => {
    expect(roots([
      path.join(fixtureRoot, "apps/web"),
      path.join(fixtureRoot, "missing"),
    ])).toEqual([path.join(fixtureRoot, "apps/web")]);
  });

  it("still reports Next rule violations in discovered application roots", () => {
    const config: Linter.Config = {
      files: ["**/*.jsx"],
      languageOptions: {
        parserOptions: { ecmaFeatures: { jsx: true } },
      },
      plugins: { "@next/next": nextPlugin },
      settings: { next: { rootDir: path.join(fixtureRoot, "apps/*") } },
      rules: { "@next/next/no-html-link-for-pages": "error" },
    };
    const linter = new Linter();
    const messages = linter.verify('<a href="/about">About</a>', [config], "component.jsx");
    expect(messages).toHaveLength(1);
    expect(messages[0].ruleId).toBe("@next/next/no-html-link-for-pages");
    expect(messages[0].severity).toBe(2);
    expect(linter.verify('<a href="https://example.com">External</a>', [config], "component.jsx"))
      .toEqual([]);
  });
});
