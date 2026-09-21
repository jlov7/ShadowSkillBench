import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";

const workbenchRoot = resolve(process.cwd());
const script = resolve(workbenchRoot, "scripts/sync-workbench.mjs");
const holdFixture = resolve(workbenchRoot, "data/workbench.hold.json");

function run(source: string) {
  return spawnSync(process.execPath, [script, "--source", source], {
    cwd: workbenchRoot,
    encoding: "utf8",
  });
}

describe("workbench export sync", () => {
  it("accepts a bound export and writes canonical generated data", async () => {
    const directory = await mkdtemp(resolve(tmpdir(), "ssb-workbench-sync-"));
    const source = resolve(directory, "workbench.json");
    try {
      await cp(holdFixture, source);
      const result = run(source);
      expect(result.status).toBe(0);
      const generated = JSON.parse(await readFile(resolve(workbenchRoot, "data/workbench.json"), "utf8"));
      expect(generated.content_hash).toBe("sha256:8f922091969291755d05c6809bc359196daf5427047d16bc58803abd9ea0a0d0");
      expect(generated.schema).toBe("SSB-WORKBENCH-1");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("fails closed for a missing or hash-tampered source", async () => {
    const directory = await mkdtemp(resolve(tmpdir(), "ssb-workbench-sync-"));
    const source = resolve(directory, "workbench.json");
    try {
      const missing = run(resolve(directory, "missing.json"));
      expect(missing.status).toBe(1);
      const payload = JSON.parse(await readFile(holdFixture, "utf8"));
      payload.limitations.push("tampered");
      await writeFile(source, JSON.stringify(payload));
      const tampered = run(source);
      expect(tampered.status).toBe(1);
      expect(tampered.stderr).toContain("HOLD_WORKBENCH_HASH_INVALID");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});
