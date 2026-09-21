# ShadowSkillBench — Protocol Freeze and Custody Runbook

## V4 successor route (scientific prefreeze; current stop)

The V3 9,240-episode design is `NOT_RUN` and underpowered. V4 is the selected
guard design: 59,640 Stage-A episodes plus 18,480 Stage-B episodes (78,120
total), with corpus seed 104733, development seeds 4242/4243, 70 bundles,
70 Stage-A cases per domain, seven bundle seeds per ratio and domain, 22
Stage-B cases per authority class and domain, and three repeats.

Its frozen arm-specific planning simulator has 20,000 replicates, seed 104732,
and family size 16. E4 treats its A1 policy-only control as unbundled. All 16
scenarios pass, with minimum power 0.81185, maximum expected
interval width 0.15378868593653425, and maximum Monte Carlo standard error
0.0027635988990806896. This is a conservative normal-approximation planning
model on the probability-difference scale, not estimator-exact power. The
larger Stage-A matrix is required by the corrected E4 arm-specific covariance
model and does not authorize execution.

V4 remains only a scientific prefreeze. The 2026-09-01 execution followed the
operator order below through the Nemotron screen. Identity and role conformance
passed, and the fresh `screen-v3` root completed 8/8 cells, but its offline audit
returned terminal `HOLD` with receipt
`sha256:ded54d85461aba44d47875f5a26d525ebac3c586f2355639e27c25d2ef9eb3f4`.
All eight cells ended `MODEL_OUTPUT_INVALID`; task completion and completion
under policy were 0/8, successful tool-action cells were 1/8, and three cells
used the full 8,192 output-token cap. Validation, full V4 custody, and the
78,120-episode confirmatory study were not run. A screen HOLD is terminal for
the bundle, so the remaining steps are retained as prospective procedure only:

1. Capture the Nemotron model identity and Ollama version, then write the
   hashed-only identity receipt:

   ```zsh
   OLLAMA_HOST=127.0.0.1:11435
   curl --fail --silent --show-error --request POST "http://${OLLAMA_HOST}/api/show" \\
     --header 'Content-Type: application/json' \\
     --data '{"name":"nemotron-3.5-lightning:30b-mlx"}' \\
     > /private/tmp/ssb-v4-nemotron-show.json
   OLLAMA_HOST=${OLLAMA_HOST} ollama --version > /private/tmp/ssb-v4-ollama-version.txt
   uv run python scripts/record_nemotron_live_show_identity.py \
     --show-json /private/tmp/ssb-v4-nemotron-show.json \
     --ollama-version /private/tmp/ssb-v4-ollama-version.txt \
     --output /private/tmp/ssb-v4-nemotron-identity-receipt.json
   ```

2. Start the isolated Ollama server, write the Nemotron role/structured-output
   receipt, and write the V4 scientific prefreeze manifest and detached digest:

   ```zsh
   OLLAMA_HOST=127.0.0.1:11435 \
   OLLAMA_CONTEXT_LENGTH=131072 \
   OLLAMA_NUM_PARALLEL=1 \
   OLLAMA_MAX_LOADED_MODELS=1 \
   OLLAMA_KEEP_ALIVE=-1 \
   ollama serve

   uv run python scripts/verify_ollama_nemotron_role_profile.py \
     --host http://127.0.0.1:11435 \
     --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
     --output /private/tmp/ssb-v4-nemotron-role-receipt.json \
     --failure-output /private/tmp/ssb-v4-nemotron-role-failure.json

   uv run shadowskillbench protocol freeze-v4
   ```

   `protocol freeze-v4` is a scientific prefreeze only and remains
   `HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY`.

3. Run the one-shot Nemotron seed-4243 capability screen exactly once, then
   audit its fresh output. Do not retry, raise a cap, substitute a case, or
   reuse an older root. The preserved `screen-v2` root is a partial
   zero-provider-call HOLD and must not be overwritten:

   ```zsh
   uv run shadowskillbench pilot capability-run \
     --endpoint http://127.0.0.1:11435/v1/chat/completions \
     --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
     --role-profile-receipt /private/tmp/ssb-v4-nemotron-role-receipt.json \
     --model-identity-receipt /private/tmp/ssb-v4-nemotron-identity-receipt.json \
     --phase screen \
     --output-dir artifacts/development/v4-nemotron-seed-4243-screen-v3 \
     --bundle nemotron
   uv run shadowskillbench pilot capability-audit \
     --phase screen \
     --output-dir artifacts/development/v4-nemotron-seed-4243-screen-v3 \
     --bundle nemotron
   ```

4. Only after a screen PASS, run the separately reserved validation once and
   audit it. A validation HOLD is terminal:

   ```zsh
   uv run shadowskillbench pilot capability-run \
     --endpoint http://127.0.0.1:11435/v1/chat/completions \
     --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
     --role-profile-receipt /private/tmp/ssb-v4-nemotron-role-receipt.json \
     --model-identity-receipt /private/tmp/ssb-v4-nemotron-identity-receipt.json \
     --phase validation \
     --screen-receipt artifacts/development/v4-nemotron-seed-4243-screen-v3/screen-pass-receipt.json \
     --output-dir artifacts/development/v4-nemotron-seed-4243-validation-v3 \
     --bundle nemotron
   uv run shadowskillbench pilot capability-audit \
     --phase validation \
     --output-dir artifacts/development/v4-nemotron-seed-4243-validation-v3 \
     --bundle nemotron
   ```

5. Only after validation PASS may the operator establish full V4 runtime
   freeze and custody. Only after that gate may the V4 corpus, package, and
   run routes be considered; at this checkout their CLIs must HOLD with the
   following fail-closed dispositions:

   ```zsh
   uv run shadowskillbench corpus generate-confirmatory-v4
   uv run shadowskillbench episodes build-confirmatory-package-v4
   uv run shadowskillbench episodes run-confirmatory-v4
   ```

   The corpus route reports
   `HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY`; package construction
   reports `HOLD_CONFIRMATORY_V4_PACKAGE_NOT_IMPLEMENTED`; and execution
   reports `HOLD_CONFIRMATORY_V4_EXECUTION_NOT_IMPLEMENTED`.

There is no V4 research result. The current stop is
`HOLD_NEMOTRON_SCREEN_MODEL_OUTPUT_INVALID`; reserved validation and full
runtime freeze/custody remain unopened. Do not increase the cap, retry this
bundle, reuse an older V3 or failed output root, or interpret the eight-cell
screen as a research comparison.

### Prospective Nemotron indexed-action route

The terminal named-action `nemotron` bundle is preserved unchanged. Its only
successor route is the separately frozen `nemotron-indexed` WIRE4 bundle, using
fresh corpus seed 4244 and fresh screen/validation roots. It retains runtime
seed 4242, the same role and model-identity receipts, 8,192-token cap, one
attempt, and no-retry controls. Before any provider call, use
`--bundle nemotron-indexed` with both receipts; run the fresh eight-cell screen
once, audit it, then run the reserved validation once only after screen PASS.
Either HOLD is terminal for that exact bundle and root.

The following commands are prospective operator procedure only. They reuse the
existing hashed-only Nemotron role and identity receipts, but never reuse the
terminal named-action roots:

```zsh
uv run shadowskillbench pilot capability-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt /private/tmp/ssb-v4-nemotron-role-receipt.json \
  --model-identity-receipt /private/tmp/ssb-v4-nemotron-identity-receipt.json \
  --phase screen \
  --output-dir artifacts/development/v4-nemotron-indexed-seed-4244-screen \
  --bundle nemotron-indexed
uv run shadowskillbench pilot capability-audit \
  --phase screen \
  --output-dir artifacts/development/v4-nemotron-indexed-seed-4244-screen \
  --bundle nemotron-indexed
```

Only if that audit passes, run the separately reserved validation exactly once:

```zsh
uv run shadowskillbench pilot capability-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt /private/tmp/ssb-v4-nemotron-role-receipt.json \
  --model-identity-receipt /private/tmp/ssb-v4-nemotron-identity-receipt.json \
  --phase validation \
  --screen-receipt artifacts/development/v4-nemotron-indexed-seed-4244-screen/screen-pass-receipt.json \
  --output-dir artifacts/development/v4-nemotron-indexed-seed-4244-validation \
  --bundle nemotron-indexed
uv run shadowskillbench pilot capability-audit \
  --phase validation \
  --output-dir artifacts/development/v4-nemotron-indexed-seed-4244-validation \
  --bundle nemotron-indexed
```

### Prospective Nemotron indexed context-32k route

The terminal 131072-context root is not retried or resumed. The prospective
`nemotron-indexed-context32k` route requires fresh V2 role and identity receipts
and binds server context 32768; it is not evidence of a fix. Capture identity
metadata and record the V2 receipt before a provider request, then use the new
root names exactly once:

```zsh
uv run python scripts/verify_ollama_nemotron_context32k_role_profile.py \
  --host http://127.0.0.1:11435 \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --output /private/tmp/ssb-v4-nemotron-context32k-role-receipt.json \
  --failure-output /private/tmp/ssb-v4-nemotron-context32k-role-failure.json
uv run shadowskillbench pilot capability-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt /private/tmp/ssb-v4-nemotron-context32k-role-receipt.json \
  --model-identity-receipt /private/tmp/ssb-v4-nemotron-context32k-identity-receipt.json \
  --phase screen \
  --output-dir artifacts/development/v4-nemotron-indexed-context32768-seed-4245-screen \
  --bundle nemotron-indexed-context32k
uv run shadowskillbench pilot capability-audit \
  --phase screen \
  --output-dir artifacts/development/v4-nemotron-indexed-context32768-seed-4245-screen \
  --bundle nemotron-indexed-context32k
```

If this route also crashes, hold it and choose a separately frozen native-tool
or different-model profile; do not alter the context-32k version in place.

## 1. Pre-freeze preparation

Run from the repository root. Confirm Ollama 0.33.2 and the three frozen model
digests in `config/models.yaml`, then start the isolated single-request server:

```bash
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_CONTEXT_LENGTH=131072 \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_KEEP_ALIVE=-1 \
ollama serve
```

With `SSB_OLLAMA_LOCAL_TOKEN` set to a non-secret local placeholder, generate the
role receipt and native executor descriptor:

```bash
uv run python scripts/verify_ollama_gpt_oss_role_profile.py \
  --host http://127.0.0.1:11435 \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --output protocol/commitments/role_conformance_receipt.json \
  --failure-output /private/tmp/ssb-confirmatory-role-failure.json

uv run shadowskillbench protocol write-confirmatory-run-descriptor \
  --endpoint http://127.0.0.1:11435/api/generate \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt protocol/commitments/role_conformance_receipt.json \
  --output protocol/commitments/run_descriptor.json
```

Generate or verify the deterministic commitments and schemas, then run the full
local gate. The completed preregistration must contain no freeze sentinels.

```bash
uv run shadowskillbench protocol write-confirmatory-design-commitment
uv run python scripts/write_runtime_skill_ir_schema.py
uv run python scripts/write_confirmatory_compiler_wire_schema.py
uv run python scripts/write_runtime_prompt_manifest.py
uv run shadowskillbench protocol validate
uv run ruff format --check . && uv run ruff check .
uv run pyright
uv run pytest -m 'not live' -q
```

The optional two-cell profile check is explicitly exploratory and may not be
reported as confirmatory evidence. It uses exploratory seed 104729; the frozen
confirmatory corpus uses unobserved seed 104730, so no timed cell overlaps the
confirmatory design:

```bash
uv run python scripts/preflight_confirmatory_profile.py \
  --compiler-endpoint http://127.0.0.1:11435/api/generate \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt protocol/commitments/role_conformance_receipt.json \
  --run-descriptor protocol/commitments/run_descriptor.json \
  --output artifacts/confirmatory-readiness/ollama-gpt-oss-20b-profile-v1-preflight
```

## 2. Generate freeze artifacts

```bash
uv run shadowskillbench protocol freeze
```

Required outputs:

```text
protocol/freeze_manifest.json
protocol/freeze_manifest.sha256
protocol/FREEZE_SUMMARY.md
```

`PASS_PROTOCOL_FREEZE` still means `HOLD_PENDING_CUSTODY_RECEIPT`. It is not
authorization to generate the confirmatory corpus, compile the 30 confirmatory
skills, or run any confirmatory episode.

## 3. Pre-run review

Review the preregistration predictions and exclusions, all condition/proof
commitments, role receipt, run descriptor, corpus counts, confound register,
claims ledger, and employer publication policy. Confirm the working tree contains
no unintended artifact roots or raw provider content.

The measured exploratory profile check is a capacity estimate only. On the
reference M4 Max, its compiler calls took 133.895 s and 129.616 s, and its
executor episodes took 5.181 s for five turns and 1.718 s for one turn. At that
observed mix, 9,240 serial local episodes imply about 8.9 hours of generation;
reserve 12–18 hours for corpus variation, 30 compiler calls, staging, I/O, and
audits. This is not a performance guarantee or a research result.

## 4. Commit the freeze

Commit the exact reviewed tree and create an annotated local protocol tag:

```bash
git add protocol docs prompts config src scripts tests
git commit -m "research: freeze ShadowSkillBench v1.0 protocol"
git tag -a shadowskillbench-protocol-v1.0.0-local \
  -m "ShadowSkillBench v1.0 confirmatory protocol"
git show --no-patch shadowskillbench-protocol-v1.0.0-local
```

The freeze commit must include every file named by the manifest. Signing and
external publication are optional release-hardening steps, not execution gates.

## 5. Local custody receipt

Create `protocol/anchor_receipt.json` as canonical JSON with exactly these fields:

```text
schema_version = 2.0
custody_mode = LOCAL_HASH_CUSTODY
freeze_commit
protocol_tag
freeze_manifest_hash
custody_locator = urn:git:<freeze_commit>
created_at
verification_result = VERIFIED_LOCAL
```

`freeze_commit` is the full 40-character commit, `freeze_manifest_hash` is the
`sha256:` value printed by the freeze command, and `created_at` is UTC ending in
`Z`. The receipt is deliberately outside the freeze-manifest input set. This mode
provides local hash custody only; it does not claim a signature, public timestamp,
or independent external verification. The legacy signed external-anchor receipt
remains accepted when publication-grade independent custody is later required.

## 6. Post-custody corpus and skill production

Only after the local custody receipt and an isolated server matching section 1 exist:

```bash
uv run shadowskillbench corpus generate-confirmatory \
  --output-dir artifacts/corpus/confirmatory
uv run shadowskillbench corpus audit --split confirmatory \
  --public-corpus artifacts/corpus/confirmatory/confirmatory_public_corpus.json

uv run shadowskillbench episodes compile-confirmatory-skills \
  --endpoint http://127.0.0.1:11435/api/generate \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt protocol/commitments/role_conformance_receipt.json \
  --output-dir artifacts/experiments/confirmatory-compiler \
  --exclusion-rule protocol/commitments/exclusion_rule.json
```

Compilation is sequential, has no retry, and writes a secret-free failure receipt
before HOLD. Never reuse a failed output root.

## 7. Stage, seal, execute, and audit

The staging command deterministically derives the complete development inventory
from frozen seed 4242; it does not accept an operator-supplied inventory.

```bash
mkdir -p artifacts/experiments

uv run shadowskillbench episodes stage-confirmatory-source \
  --source-dir artifacts/experiments/confirmatory-package-source \
  --compiled-skills-dir artifacts/experiments/confirmatory-compiler/compiled-skills

uv run shadowskillbench episodes build-confirmatory-package \
  --source-dir artifacts/experiments/confirmatory-package-source \
  --package-dir artifacts/experiments/confirmatory-package

uv run shadowskillbench episodes run-confirmatory \
  --stage-a \
  --package-dir artifacts/experiments/confirmatory-package \
  --output-dir artifacts/experiments/confirmatory \
  --endpoint http://127.0.0.1:11435/api/generate \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN
uv run shadowskillbench experiments audit --stage-a

uv run shadowskillbench episodes run-confirmatory \
  --stage-b \
  --package-dir artifacts/experiments/confirmatory-package \
  --output-dir artifacts/experiments/confirmatory \
  --endpoint http://127.0.0.1:11435/api/generate \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN
uv run shadowskillbench experiments audit --stage-b

uv run shadowskillbench analyze
uv run shadowskillbench report build
uv run shadowskillbench reproduce --protocol protocol/freeze_manifest.json
```

Run the stages serially and do not send concurrent work to the isolated server.
An agent outcome such as refusal, invalid action, timeout, or budget exhaustion is
an outcome, not a technical exclusion.

## 8. Amendment procedure

If a genuine defect appears:

1. stop and preserve every artifact;
2. write a protocol amendment;
3. increment the protocol version;
4. regenerate every affected hash;
5. create a new freeze commit, tag, and custody receipt;
6. label amended results exploratory until separately confirmed.

### 2026-08-29 compiler transport amendment

`SSB-CONFIRMATORY-COMPILER3` produced 22 valid skills, then stopped at
`bundle_46317917aef537960e7ef4d1d4743f80` with `MODEL_OUTPUT_INVALID` despite
`finish_reason=stop` (75,960 input tokens and 706 output tokens). The exact
chat-request replay also returned invalid JSON. A native-chat request with low
reasoning remained invalid. In contrast, a native raw `/api/generate` final-
channel diagnostic produced valid output in 333 tokens. Its first diagnostic
call reported a transport-wrapper error after HTTP 200; the corrected second
call is the valid result. A subsequent exact `COMPILER4` projection diagnostic
with `Reasoning: low` and `top_p` omitted also produced valid output, in 304
tokens.

The frozen v15c root and its diagnostic records remain immutable failure and
diagnostic evidence. The decision is `SSB-CONFIRMATORY-COMPILER4`: use only the
native raw `/api/generate` compiler transport; render the Harmony system with
`Reasoning: low`; force the assistant final-channel suffix; and omit `top_p`
from the compiler request. Seed 4243, temperature 1.0, 16,384 maximum tokens,
one attempt, model/digest, prompt, schema, context, and timeout remain fixed.
This amendment requires a new local hash custody receipt before any
confirmatory generation.
