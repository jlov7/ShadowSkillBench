import { spawnSync } from "node:child_process";
import { readFile, rename, lstat, writeFile, mkdtemp, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = resolve(root, "data/workbench.json");
const required = [
  "schema",
  "schema_version",
  "read_only",
  "claim_ceiling",
  "protocol",
  "preregistration",
  "analysis",
  "figures",
  "summaries",
  "episode_index",
  "episode_traces",
  "claims",
  "limitations",
  "content_hash",
];

function sourceArgument() {
  const index = process.argv.indexOf("--source");
  if (index === -1 || !process.argv[index + 1]) {
    return resolve(root, "../artifacts/reports/workbench.json");
  }
  return resolve(process.cwd(), process.argv[index + 1]);
}

async function rejectSymlink(path, label) {
  try {
    if ((await lstat(path)).isSymbolicLink()) throw new Error(`${label} must not be a symlink`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

async function main() {
  const source = sourceArgument();
  await rejectSymlink(source, "source");
  await rejectSymlink(output, "generated output");
  let sourceBytes;
  let parsed;
  try {
    sourceBytes = await readFile(source);
    parsed = JSON.parse(sourceBytes.toString("utf8"));
  } catch (error) {
    throw new Error(`HOLD_WORKBENCH_SOURCE_INVALID: ${error.message}`);
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("HOLD_WORKBENCH_SCHEMA_INVALID: dataset must be an object");
  }
  const keys = Object.keys(parsed).sort();
  if (parsed.schema !== "SSB-WORKBENCH-1" || parsed.schema_version !== "1.0" || parsed.read_only !== true || keys.join("\n") !== [...required].sort().join("\n")) {
    throw new Error("HOLD_WORKBENCH_SCHEMA_INVALID: unsupported workbench contract");
  }
  const expected = parsed.content_hash;
  if (typeof expected !== "string" || !/^sha256:[0-9a-f]{64}$/.test(expected)) {
    throw new Error("HOLD_WORKBENCH_HASH_INVALID: content_hash does not bind the export");
  }
  const verifier = [
    "import base64, hashlib, json, sys",
    "value = json.load(sys.stdin)",
    "expected = value['content_hash']",
    "unsigned = dict(value)",
    "unsigned.pop('content_hash')",
    "canonical = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')",
    "actual = 'sha256:' + hashlib.sha256(canonical).hexdigest()",
    "if actual != expected: raise SystemExit('content_hash does not bind the export')",
    "payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')",
    "sys.stdout.write(json.dumps({'hash': actual, 'payload': base64.b64encode(payload).decode('ascii')}))",
  ].join("\n");
  const checked = spawnSync(process.env.SSB_PYTHON ?? "python3", ["-c", verifier], {
    input: sourceBytes,
    encoding: "utf8",
  });
  if (checked.status !== 0) {
    throw new Error(`HOLD_WORKBENCH_HASH_INVALID: ${checked.stderr.trim() || "content_hash does not bind the export"}`);
  }
  let verification;
  try {
    verification = JSON.parse(checked.stdout);
  } catch {
    throw new Error("HOLD_WORKBENCH_HASH_INVALID: canonical verifier returned invalid output");
  }
  if (verification.hash !== expected || typeof verification.payload !== "string") {
    throw new Error("HOLD_WORKBENCH_HASH_INVALID: canonical verifier mismatch");
  }
  const payload = Buffer.from(verification.payload, "base64");
  const temporaryDirectory = await mkdtemp(`${dirname(output)}/.workbench-sync-`);
  const temporary = resolve(temporaryDirectory, "workbench.json");
  try {
    await writeFile(temporary, payload, { flag: "wx", mode: 0o644 });
    await rename(temporary, output);
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
  process.stdout.write(`synced ${source} -> ${output} ${expected}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
});
