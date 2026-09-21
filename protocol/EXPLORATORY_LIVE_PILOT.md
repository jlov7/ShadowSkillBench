# Exploratory live development pilot

Classification: `EXPLORATORY_LIVE_PILOT_NOT_CONFIRMATORY`

This is a development-only, loopback-model pilot. It does not amend, consume,
or establish the confirmatory preregistration, freeze, anchor, package,
analysis, or release evidence.

## Fixed design

- Seed: `4242`.
- Domains: `access_provisioning` and `financial_adjustments`.
- Cases: the first ten active development cases in each domain.
- Conditions: one `A0_BARE` reference cell per case, plus `A2_SKILL_ONLY`
  at `R0` and `R100` per case.
- Repeats: one execution of each cell.
- Total: `2 domains x 10 cases x (1 A0 + 2 A2 ratios) = 60` cells.
- Immutable order: case ordinal first, then access and financial domain; each
  domain contributes the `A0`, `R0`, `R100` triplet. Therefore `--limit 6`
  is the paired infrastructure/behavior preflight containing one case from
  each domain. It checks only that the pinned provider/profile path can compile
  and execute that prefix; it does not establish a research finding.
- Compilation: exactly four live skills, one for each domain/ratio pair, from
  compiler-visible development bundles using `prompts/skill_compiler.md`, with
  a fixed 16,384-token compiler output cap for the admitted gpt-oss route.
- Execution: one explicit OpenAI-compatible loopback model descriptor is used
  to create separate compiler and episode clients; concurrency is fixed at one.
  The admitted gpt-oss profile uses a fixed 900-second timeout, a 131,072-token
  server context, `temperature=1.0`, `low` compiler reasoning effort, and
  `none` executor reasoning effort. Development fixture seeds remain
  their canonical unsigned 64-bit
  values in the case/world records; the episode manifest and model request use
  the deterministic nonnegative signed-64 low-63-bit mapping required by the
  OpenAI-compatible request contract. A retained 9B operator observation completed 95,084 prompt tokens in
  308.63698 seconds and reached the former 4,096-token output cap in 142.22281
  seconds (450.8598 seconds total). At that observed output rate, 8,192 output
  tokens project to about 593.08 seconds; 900 seconds is the bounded margin.
  Private operator receipts retain these observations, bind them to the failed
  runs' plan/runtime hashes, and explicitly record no raw provider trace,
  compiled skill, or episode result. Those receipts are not part of the public
  software repository.
  A later retained 9B observation reached the 8,192-token cap after 95,081
  prompt tokens in 383.78024 seconds and 8,192 output tokens in 293.46893
  seconds (677.24917 seconds total), then failed as `MODEL_OUTPUT_INVALID`.
  Those retained Qwen attempts remain at their recorded 8,192/900 allowance.
  The gpt-oss route makes one final bounded compiler-cap escalation from 8,192
  to 16,384 tokens after its V4 access `R100` compiler request consumed exactly
  8,192 output tokens in 372.5 seconds and failed as `MODEL_OUTPUT_INVALID`.
  Its observed 74,526-token input plus the final cap is
  `74,526 + 16,384 = 90,910`, within the admitted 131,072-token context (leaving
  40,162 tokens of headroom). If this 16,384-token request exhausts its cap or
  returns invalid output, the local gpt-oss route is `HOLD`. No further
  compiler-cap increase is permitted. The fixed 900-second timeout remains unchanged. The
  compact pilot-only provider wire schema and prompt profile remain
  `SSB-PILOT-WIRE-COMPILER1`. Real traces use six or fewer actions and
  exactly 12 source traces; their aggregate Gate2 witnesses use up to seven
  distinct ordered steps. The wire therefore permits at most seven steps,
  twelve evidence indexes per objective or step, and two flat argument records
  per step; free text is bounded at 192 codepoints, argument text at 96, and
  identifiers at 64. Argument numbers are bounded to absolute value 1,000,000.
  Provider-facing free text is restricted to printable ASCII excluding `"` and
  `\\`, so the fully bounded canonical JSON payload is at most 7,702 bytes,
  including serialization escaping.
  To avoid provider expansion of dynamic objects and long repeated event
  IDs, each argument binding is a finite `{key,value}` record. The provider
  returns only an objective, objective evidence indexes, and ordered steps with
  per-step evidence indexes; indexes address the deterministic flattened
  compiler-input event table. The host supplies all mechanically known SkillIR
  fields and deterministically hydrates the records and indexes to canonical
  evidence IDs before applying the normal exact provenance and source-trace
  checks.
  The prompt profile and pilot wire schema hash are bound into the plan,
  compiled-skill, and runtime custody records; audit repeats the
  wire-to-canonical round trip.
  Episode turns use a separate pilot-only `SSB-PILOT-TURN-WIRE2` schema and
  appended runtime instruction. It is a discriminated `{turn: ...}` object:
  tool calls carry only a zero-based visible-tool index and finite
  argument-index/value records. Argument indexes are zero-based in required
  keys followed by optional keys, and every required key must be bound. The
  host deterministically hydrates the action ID, tool name, and argument keys
  from the visible tool layout. The turn prompt, schema, profile, and per-cell
  visible-tool-layout hash (including required-versus-optional classification)
  are bound through plan, runtime, result, resume, and audit custody. Ordinary
  executor `AgentTurn` requests and confirmatory behavior are unchanged.
  The corresponding private 8,192-token receipt is an operator observation,
  not a raw provider trace or pilot result, and is not published here.
  The context length is an operator declaration, not a property the generic
  OpenAI-compatible endpoint can verify; the operator must verify the provider
  configuration before launch.
- Retries: zero automatic retries. A failed request stops the run; a later
  invocation may resume only from the immutable pilot artifacts.

## Claim ceiling and stop/go

The audit reports raw observed task-completion and Completion Under Policy
counts by domain, condition, and ratio. It makes no confirmatory, causal,
generalization, model-ranking, or release claim.

Stop immediately on a descriptor, protocol, prompt, plan, compiled-skill, or
result custody mismatch; on any non-loopback endpoint; or on any model error.
Proceed only when the explicit operator descriptor is present, its API-key
environment variable is available locally, and the run root remains a separate
`artifacts/exploratory-live-pilot` tree. Credentials are never written to an
artifact.

## Frozen gpt-oss role profile

The only admitted non-scripted pilot provider is
`ollama-gpt-oss-20b-harmony-roles`, using the derived local model
`ssb-gpt-oss-20b-harmony-roles:v1`. Its source model is exactly
`gpt-oss:20b` at
`sha256:17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7`.
The derived model digest is discovered after creation and becomes the runtime
`model_version`; it is not substituted with the base-model digest.

The tracked [Ollama Modelfile](../src/shadowskillbench/experiments/ollama_gpt_oss_20b_harmony.Modelfile)
and [profile implementation](../src/shadowskillbench/experiments/ollama_gpt_oss_profile.py)
are part of the pilot's custody boundary. The profile sets `num_ctx 131072` and
retains the stock gpt-oss Harmony system scaffold (identity, knowledge cutoff,
current date, reasoning, and valid channels) with its current date pinned to
`2026-08-26`, then inserts actual system input under `# Instructions` in that
same Harmony `system` tier. It renders every
developer message in Harmony's `developer` tier and user, assistant, and tool
history in their respective Harmony forms. The compiler's admitted low
reasoning effort renders `Reasoning: low`; executor requests send the explicit
OpenAI-compatible setting `reasoning_effort="none"`. The stock template fallback
is retained only for requests that omit a think setting, which this pilot does
not admit.
The OpenAI-compatible adapter continues to send strict `response_format`
schemas used by the compact compiler and pilot-turn wires. It does not send
native function-tool declarations; the template nevertheless preserves tool
role history, while the current pilot converts its visible tools through the
closed structured pilot-turn wire.

Before any pilot request, an operator must provide an immutable canonical
`OLLAMA_ROLE_CONFORMANCE_RECEIPT1`. The receipt binds the exact base digest,
derived model digest, Modelfile and template hashes, profile name, context,
compiler and executor reasoning efforts, and deterministic template probe. It must also contain two
matched live OpenAI-compatible calls with identical system/user content and
body settings, differing only by one developer message. This is a visibility
probe, not a role-hierarchy preference test: the shared system instruction says
to copy a developer marker if present and otherwise return the fallback marker,
while the shared user message is neutral. The baseline must emit the fallback
marker; the developer case must emit the developer marker; their raw request
and response hashes must differ; and the developer request must have a positive
prompt-token difference. A missing, malformed, stale, or nonconforming receipt
is a `HOLD`, and its hash plus full content are bound into the pilot runtime and
audit custody.
The matched probe uses the same strict response schema for both calls: its
single marker field is enumerated to exactly the system and developer marker
values, so a valid response still reveals which tier won without accepting
irrelevant prose.

Qwen V3/local-stock-template attempts remain quarantined engineering failure
history, not pilot or research evidence. In the observed Ollama 0.33 stock
gpt-oss/Qwen-style route, input developer messages were dropped and actual
system input was mapped to Harmony developer; matched probes had the same 86
prompt tokens with and without a developer message. The retained Qwen receipts
above preserve those facts, including their output-cap and invalid-output
observations. They do not establish model behavior, a comparison, or a research
finding, and the CLI cannot admit them as this profile.

## Frozen gpt-oss sampling profile (V6)

V5 ended at `HOLD_EXPLORATORY_LIVE_PILOT: MODEL_OUTPUT_INVALID` on the finance
R0 compiler call. The request reached HTTP 200 after 65,735 input tokens,
144.930 seconds of prompt evaluation, and 364.505 seconds of generation, and it
consumed exactly 16,384 output tokens, equal to the cap. V5 predates commit
`a664d4c`, so its adapter did not preserve the provider's `finish_reason`.
Two facts are now verified in code. First,
`OpenAICompatibleClient._parse_success` accepts only `finish_reason="stop"`;
`finish_reason="length"` is therefore reported as `MODEL_OUTPUT_INVALID`, and
the exact cap usage identifies output-cap exhaustion rather than schema-invalid
content. Second, V5 pinned compiler `temperature=0.0`, while the Harmony
analysis channel was unconstrained by the final-channel response schema. Greedy
gpt-oss decoding can repeat in analysis without switching to `final`, so the
failure depends on the particular greedy token path rather than prompt length.
This is consistent with finance R0 failing on a 65,735-token prompt while
access R0 and R100 compiled from longer 70,923- and 74,536-token prompts.

V5 contains zero episode results and zero research comparisons. Its root and
all V1-V4 roots and receipts remain immutable failure and custody history,
including the retained
`ollama-gpt-oss-20b-harmony-roles-v4/role-conformance-receipt.json`.
The terminal V5 root remains
`artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v5/`.

The V6 decision freezes `SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES2` as ROLES1 plus
`temperature=1.0` for both compiler and executor requests. The compiler seed
remains 4242 and executor requests retain each manifest seed. Reasoning remains
`low`; the compiler output cap remains 16,384; the timeout remains 900 seconds;
and the server context remains 131,072. The model tag remains
`ssb-gpt-oss-20b-harmony-roles:v1`, and the Modelfile remains unchanged.
OpenAI's gpt-oss recommendation also specifies `top_p=1.0`, recorded by the
profile constant `OLLAMA_GPT_OSS_SAMPLING_TOP_P`. The pinned `SSB-OAI-CHAT1`
request contract has no `top_p` field, so V6 does not change the wire schema or
send it. Ollama's effective default `top_p` is not observable through the
OpenAI-compatible response or role receipt and is therefore documented, not
attested.

If a V6 compiler call returns `finish_reason=length` again, the route is
`HOLD`. The next version must change the reasoning profile or compiler
projection, never the cap. No further compiler-cap increase is permitted.

At `temperature=1.0`, reproducibility depends on Ollama's seeded sampler on
identical hardware and model digest. This exploratory pilot makes no
reproducibility claim beyond hash custody.

## Frozen Ollama runtime profile (V7)

V6 reached HTTP 200 on its first Access R0 compiler request under Ollama
0.33.0. The adapter accepted a schema-valid response after 70,921 input tokens
and 515 output tokens, which proves `finish_reason="stop"` under the pinned
adapter contract. Canonical compiler hydration or binding then returned
`COMPILER_OUTPUT_INVALID`; V6 predates commit `04915db`, so the exact fixed
semantic validation code was not preserved. V6 contains zero compiled skills,
zero episode results, and zero research comparisons. Its root remains
`artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v6/` and
must not be resumed or rewritten.

The locally installed Ollama runtime advanced to 0.33.1 before the next
diagnostic call. V7 freezes `SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES3` as ROLES2
plus Ollama server version 0.33.1. This is the only V7 profile change.
Temperature remains 1.0; top-p remains documented as 1.0 and off the pinned
wire; compiler seed remains 4242 and executor requests retain each manifest
seed; reasoning remains `low`; the compiler cap remains 16,384; timeout remains
900 seconds; context remains 131,072; the compiler prompt and request schema
remain unchanged; and the model, model digests, Modelfile, and `:v1` tag remain
unchanged. The role receipt and runtime projection bind
`ollama_server_version="0.33.1"`, so neither a V6 receipt nor a V6 runtime can
be admitted as V7.

V7 uses the semantic compiler-failure custody added in `04915db`. A compiler
contract HOLD must preserve a fixed stage and validation code without raw
provider content. A cap-exhaustion HOLD still follows the V6 rule: no further
compiler-cap increase is permitted. Any later change to the compiler
projection or reasoning profile requires another version and cannot be folded
into V7.

An executor adapter failure is retained in the episode-result engine artifact,
not admitted as a completed pilot result. Its model-call receipt records the
request/response hashes, attempts, provider finish reason, reported usage,
whether the request cap was exhausted, and no raw provider content. The CLI
prints those safe diagnostics before holding the incomplete pilot.

## Frozen executor final-channel profile (V8)

V7 compiled all four skills, including Finance R0, without a compiler HOLD.
Its first Access A0 episode then returned HTTP 200 with
`finish_reason="stop"`, 655 input tokens, 75 output tokens, and
`output_cap_exhausted=false`, but no meaningful final-channel content. The
Ollama native log reported that the Harmony parser had no reverse mapping for
function name `tool_use`. This identifies an executor routing failure: the
model attempted a Harmony tool/function route instead of returning the
provider-facing final JSON. V7 contains four valid compiled skills, zero
completed episode results, and zero research comparisons. Its root remains
immutable.

V8 changes only the executor prompt projection from
`SSB-PILOT-TURN-WIRE1` to `SSB-PILOT-TURN-WIRE2`. The new prompt explicitly
forbids Harmony tool/function invocation, states that the host executes tools
only after parsing the wire object, and requires a switch from analysis to the
final channel before ending the response. The compact executor schema itself
is unchanged. Ollama version, role profile, temperature, top-p behavior,
reasoning effort, seeds, compiler prompt and artifacts, caps, timeout, context,
model, digests, Modelfile, and concurrency remain unchanged.

## Required V8 operator procedure

Create the local derived profile only after independently confirming the pinned
base digest. This command is operator action; it is not performed by the
benchmark code:

```sh
ollama create ssb-gpt-oss-20b-harmony-roles:v1 \
  -f src/shadowskillbench/experiments/ollama_gpt_oss_20b_harmony.Modelfile
```

The V7 role receipt and four compiled-skill artifacts remain valid because V8
does not change their bound inputs. Copy those immutable artifacts into a
fresh V8 root; do not resume or rewrite V7:

```sh
mkdir -p artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v8/compiled-skills
cp artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v7/role-conformance-receipt.json \
  artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v8/role-conformance-receipt.json
cp artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v7/compiled-skills/*.json \
  artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v8/compiled-skills/
```

Then run the immutable six-cell preflight in that fresh V8 root:

```sh
uv run shadowskillbench pilot live-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v8/role-conformance-receipt.json \
  --limit 6 \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v8
```

The preflight produces infrastructure/behavior observations only. It does not
establish any research result; if it passes, a later explicit immutable prefix
in `1..60` remains exploratory only. Inspect the non-confirmatory raw-count
summary with:

```sh
uv run shadowskillbench pilot audit \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v8
```

If any compiler call raises a model adapter error, the run writes a
secret-free `PILOT_COMPILER_FAILURE_RECEIPT1` record to
`compiled-skills/<domain>-<ratio>.failure-<digest>.json` before the `HOLD`
propagates. The receipt binds the plan and runtime hashes, domain, ratio,
compiler input hash, cap, error code, attempts, request/response hashes, the
provider's reported `finish_reason` and token usage, and whether the reported
output equalled the cap. It records no raw provider trace, content, or
credential. A `finish_reason` of `length` with output equal to the cap is
output-cap exhaustion (the model never reached its `final` channel), not
schema-invalid content; it must not be answered with a further cap increase.

If the adapter accepts a schema-valid response but the compiler's canonical
hydration, binding, rendering, or artifact checks reject it, the run instead
writes a secret-free `PILOT_COMPILER_CONTRACT_FAILURE_RECEIPT1` record to
`compiled-skills/<domain>-<ratio>.contract-failure-<digest>.json`. This receipt
records only a fixed failure stage and validation code, sanitized Pydantic
paths when available, request/response hashes, usage, attempts, and the
adapter-success inference `finish_reason="stop"`; it never records exception
messages or provider content. This is a semantic compiler-contract failure,
not output-cap exhaustion, and requires a versioned compiler projection or
reasoning-profile decision before another pilot attempt.

The preserved V1–V7 receipts and roots remain evidence and must not be
overwritten. V8 changes only the executor prompt projection. It does not
require recreating the explicit `:v1` derived-model tag.

## Frozen split-reasoning profile (V9)

V9 changes exactly one variable from V8: compiler calls retain `low` reasoning
effort while executor calls use `none`. The model tag, Modelfile, base and
derived digests, timeout, context, temperature, top-p behavior, seed, compiler
cap, compiler prompt, wire schemas, and executor prompt projection remain
unchanged. This is recorded as
`SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES4`; a fresh ROLES4 conformance receipt is
required because the profile projection now binds both compiler and executor
reasoning efforts. Its role-conformance probe remains compiler-low.

V9 must use a fresh
`artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v9/` root
and a fresh role-profile receipt. The CLI creates separate local clients for
the compiler and executor, binds both efforts into `runtime.json`, and binds
the executor effort into every result envelope. Audit rejects a result whose
executor reasoning effort does not equal the hashed runtime descriptor. No
Ollama invocation, model creation, or artifact migration is performed by this
repository change.

V8 is terminal executor-routing evidence, not a result: its first attempted
episode returned `finish_reason="stop"` after 707 input tokens and 66 output
tokens with `output_cap_exhausted=false`, but produced no admissible result.
V8 therefore has zero completed episode results and zero research comparisons.
The V9 change does not reinterpret that HOLD or increase any cap.

After manually confirming the existing `:v1` tag and pinned base digest, the
operator records the fresh ROLES4 receipt and runs only the explicit V9 prefix:

```sh
uv run python scripts/verify_ollama_gpt_oss_role_profile.py \
  --host http://127.0.0.1:11435 \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --output artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v9/role-conformance-receipt.json

uv run shadowskillbench pilot live-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v9/role-conformance-receipt.json \
  --limit 6 \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v9
```

V9 is terminal infrastructure evidence. Its first Access A0 executor request
returned HTTP 200 with `finish_reason="stop"`, 702 input tokens, 33 output
tokens, `output_cap_exhausted=false`, and
`before_meaningful_behavior=true`. The isolated Ollama log reported
`thinking = 0` and `truncated = 0`, but the OpenAI-compatible response still
contained no admissible final-channel JSON. V9 therefore has four preserved
compiled skills, zero completed episode results, and zero research
comparisons. Audit was not run because there were no completed results to
audit.

This rules out both an active reasoning loop and output-cap exhaustion for the
V9 executor call; it does not alter the V5 compiler diagnosis. No further
compiler-cap increase is permitted. A subsequent version must make an
explicit, human-approved choice among the provider transport, executor
projection, or derived-model template; V9 must not be resumed or rewritten.

## Frozen native executor transport (V10)

V10 changes exactly one V9 runtime variable: the executor uses Ollama native
`/api/chat` rather than the OpenAI-compatible `/v1/chat/completions` route.
The compiler remains on `/v1/chat/completions` with `reasoning_effort="low"`.
The executor sends canonical request bytes with `stream=false`, `think=false`,
the existing role messages, JSON Schema `format`, `temperature=1.0`, the
existing deterministic seed, `options.num_predict`, `options.num_ctx=131072`,
and `options.top_p=1.0`. It has no OpenAI `response_format`, `max_tokens`, or
`reasoning_effort` field. The native response must carry `done=true`,
`done_reason="stop"`, an assistant JSON object, exact native token counts, and
no non-empty `message.thinking` content.

This is `SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES5`. Its fresh conformance receipt
binds the native executor transport, request profile/hash, `think=false`,
context, and top-p; it cannot be replaced with a preserved ROLES4/V9 receipt.
Model tag and digest, base digest, Ollama `0.33.1`, compiler cap `16384`,
context, timeout `900`, concurrency `1`, schemas, prompts, seed, temperature,
and all V9 facts remain unchanged. V10 does not increase any cap, reinterpret
V9's zero completed results, start Ollama, rewrite a prior root, or authorize
a research claim.

The operator records a new ROLES5 receipt and runs only a new V10 root; never
resume or rewrite a V9 root. Before the live run, copy the four byte-identical
V7 compiled artifacts into the new root so compilation is not re-sampled at
`temperature=1.0`:

```sh
mkdir -p artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v10/compiled-skills
cp artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v7/compiled-skills/*.json \
  artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v10/compiled-skills/

uv run python scripts/verify_ollama_gpt_oss_role_profile.py \
  --host http://127.0.0.1:11435 \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --output artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v10/role-conformance-receipt.json

uv run shadowskillbench pilot live-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v10/role-conformance-receipt.json \
  --limit 6 \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v10

uv run shadowskillbench pilot audit \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v10
```

Any provider, schema, empty-content, non-empty-thinking, custody, or audit
failure is a `HOLD`; retain only hash/usage/finish-reason diagnostics and do
not retry or raise the compiler cap.

V10 is terminal executor-routing evidence. Its original native executor call
returned HTTP 200 with `done=true`, `done_reason="stop"`, 702 input tokens,
33 output tokens, and no admissible content. A controlled identical-request
diagnostic preserved only safe counts: the response body was 313 bytes,
`content_length=0`, and `thinking` was absent. `output_cap_exhausted=false`.
The distinct `think:"low"` falsifier also returned HTTP 200 with
`done_reason="stop"`, 707 input tokens, 66 output tokens,
`thinking_length=82`, and `content_length=0`; it did not reach the output cap.
V10 has zero completed episode results and zero research comparisons.

## Frozen executor final-channel template (V11)

V11 changes exactly one V10 runtime variable: only the executor uses the new
derived model `ssb-gpt-oss-20b-harmony-final:v2`, built from a separately
versioned executor Modelfile whose final template suffix is
`<|start|>assistant<|channel|>final<|message|>`. The compiler remains on the
unchanged `ssb-gpt-oss-20b-harmony-roles:v1` tag and the OpenAI-compatible
route with `reasoning_effort="low"`. Executor native transport remains
`/api/chat`, `think=false`, `SSB-OLLAMA-NATIVE-EXECUTOR-REQUEST1`, temperature
1.0, top-p 1.0, seed, caps, context, timeout, prompts, schemas, and
concurrency remain pinned. This is
`SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES6`; its receipt binds both compiler role
probes and the executor tag, derived digest, Modelfile hash, and template hash.
The executor derived digest is pinned as
`sha256:0ea56576556e5ba38e4fddccf0eaebee959aea5250913c12e6b3d5d80d225790`;
the source Modelfile hash is
`sha256:f585fdf6bc29dc738b3062a3dd566c9fc66b16dff22ed2d58b3aec44bbbd7db9`.

Before a V11 run, create the executor tag from the new, committed executor
Modelfile and record its digest. The operator must use a fresh
`artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v11/` root,
copy the four byte-identical V7 compiled artifacts, then produce a fresh
ROLES6 receipt and run the six-cell prefix. No V10 root may be resumed or
rewritten. Any failure is a HOLD with only hash, count, and finish-reason
custody; do not retry and do not increase the compiler cap.

```sh
ollama create ssb-gpt-oss-20b-harmony-final:v2 \
  -f src/shadowskillbench/experiments/ollama_gpt_oss_20b_harmony_final.Modelfile

mkdir -p artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v11/compiled-skills
cp artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v7/compiled-skills/*.json \
  artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v11/compiled-skills/

uv run python scripts/verify_ollama_gpt_oss_role_profile.py \
  --host http://127.0.0.1:11435 \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --output artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v11/role-conformance-receipt.json

uv run shadowskillbench pilot live-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v11/role-conformance-receipt.json \
  --limit 6 \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v11

uv run shadowskillbench pilot audit \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v11
```

V11 is terminal executor-routing evidence. Its first executor `/api/chat`
request using the final-template V2 model returned HTTP 200 with `done=true`,
`done_reason="stop"`, 705 input tokens, 133 output tokens, and empty content.
It had `before_meaningful_behavior=true` and did not reach the output cap, so
there are zero completed episode results and zero research comparisons.

## Frozen raw native generate transport (V12)

V12 changes exactly one V11 runtime variable: only the executor native
transport and request projection. The compiler remains on the unchanged
OpenAI-compatible `/v1/chat/completions` route with `reasoning_effort="low"`.
The executor retains the V2 derived model, digest, template, `think=false`
semantic, temperature 1.0, top-p 1.0, deterministic seed, `num_ctx=131072`,
`num_predict`, JSON Schema `format`, cap, timeout, prompt, schemas, and
concurrency. It now posts a deterministic raw request to `/api/generate`:
`raw=true`, `stream=false`, the schema, and a V2 Harmony prompt rendered from
the exact ModelRequest messages. Because `think=false`, the renderer omits a
Reasoning line and ends with
`<|start|>assistant<|channel|>final<|message|>`.

This is `SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES7` and
`SSB-OLLAMA-NATIVE-GENERATE-EXECUTOR-REQUEST1`. The raw response must have the
pinned model, `done=true`, `done_reason="stop"`, strict JSON in `response`, and
native prompt/evaluation counts. It retains hash-only custody; raw provider
content is never persisted. The controlled V11 exact-request diagnostic on
this route returned HTTP 200, `done_reason="stop"`, 704 input tokens, 51 output
tokens, `content_length=151`, valid JSON, and zero PilotAgentTurnWire
validation paths. That diagnostic is transport evidence, not an episode result
or research comparison.

Use a fresh
`artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v12/` root,
copy the four byte-identical V7 compiled artifacts, create a fresh ROLES7
receipt, run only the six-cell prefix, then audit. V10 and V11 roots are
immutable audit inputs and must not be resumed or rewritten. Any failure is a
HOLD with hashes, counts, finish reason, and validation paths only; do not
retry and do not increase the compiler cap.

```sh
mkdir -p artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v12/compiled-skills
cp artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v7/compiled-skills/*.json \
  artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v12/compiled-skills/

uv run python scripts/verify_ollama_gpt_oss_role_profile.py \
  --host http://127.0.0.1:11435 \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --output artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v12/role-conformance-receipt.json

uv run shadowskillbench pilot live-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v12/role-conformance-receipt.json \
  --limit 6 \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v12

uv run shadowskillbench pilot audit \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v12
```

V12 completed all six preflight cells on the first fresh run: `completed=6`,
`skipped=0`, `limit=6`. The audit passed with plan hash
`sha256:b15cd69472bda5397bd4ca719c62276b8de1bc9d9432e5eb09805e91146900ec`.
It reported one observation in each of the six domain/condition/ratio groups and
zero task completions and zero completions under policy in every group. Its
claim ceiling remains `raw exploratory counts only; not confirmatory evidence`;
these six cells validate infrastructure and behavior custody, not a research
comparison or finding.

## Full exploratory prefix (V13)

V12 is an immutable six-cell preflight root. Recording its terminal audit
changed this protocol file's hash, so it must not be resumed or rewritten.
V13 extends the explicitly requested exploratory coverage from the six-cell
preflight to the complete fixed 60-cell prefix in a fresh root. This is not a
sampling or model-profile change: ROLES7, compiler and executor models,
temperature 1.0, top-p 1.0, reasoning profiles, seed, caps, timeout, context,
prompts, schemas, transports, and concurrency remain unchanged. The four
compiler artifacts remain byte-identical reusable inputs because their
manifests bind the compiler request contract and input, not a pilot plan hash.

Create a fresh role receipt and V13 root, copy the four validated compiled
skills, run the complete prefix once, and audit it. Any failure is a HOLD; do
not retry, do not rewrite the root, and do not increase the compiler cap.

```sh
mkdir -p artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v13/compiled-skills
cp artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v12/compiled-skills/*.json \
  artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v13/compiled-skills/

uv run python scripts/verify_ollama_gpt_oss_role_profile.py \
  --host http://127.0.0.1:11435 \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --output artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v13/role-conformance-receipt.json

uv run shadowskillbench pilot live-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v13/role-conformance-receipt.json \
  --limit 60 \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v13

uv run shadowskillbench pilot audit \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v13
```

Even with all 60 cells, this run remains exploratory. The audit reports raw
counts only and does not establish a confirmatory research comparison or
finding.

V13 terminated after six completed cells on the seventh cell,
`episode_f13ea326c103653787fd518d7b08ef59`, with
`MODEL_OUTPUT_INVALID`, `finish_reason=stop`, 706 input tokens, 78 output
tokens, `output_cap_exhausted=false`, and
`before_meaningful_behavior=false`. The engine preserved the failed episode
artifact with request and response hashes but no raw provider content. V13 is
therefore a terminal HOLD and must not be resumed.

One controlled diagnostic call is authorized in the quarantined
`artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v13-diagnostic/`
root. It must reconstruct the seventh cell, prove that the request bytes match
the failed request hash, issue those exact bytes once, and retain only hashes,
byte/token counts, the JSON parse result, and PilotAgentTurnWire validation
paths. It must not persist raw provider content. This diagnostic is not an
episode result and cannot change the V13 HOLD.

## Frozen split-sampling profile (V14)

V13 is a terminal HOLD with six completed episode results and no research
comparison. Its seventh executor call returned `MODEL_OUTPUT_INVALID` with
`finish_reason=stop`, 706 input tokens, 78 output tokens,
`output_cap_exhausted=false`, and `before_meaningful_behavior=false`. The
failed raw executor request hash
(`sha256:10e7ae4400709af080627cf2b0ccfa27f2fbbf32c17c503613ebf4c4e6bf4a14`)
was reconstructed exactly for
one quarantined diagnostic: at executor temperature 1.0, the original call was
schema-invalid while the diagnostic returned valid JSON from the same request
bytes with 706 input tokens and 16 output tokens. This establishes
sampling-dependent executor behavior; it is not a compiler-cap or compiler
schema failure.

V14 freezes `SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES8`. It changes exactly one
runtime variable: compiler `sampling_temperature` remains 1.0, while the new
`executor_sampling_temperature` is 0.0. Reasoning effort, caps, timeout,
context, seed, top-p, compiler and executor models, Modelfiles, prompts,
schemas, transports, and request profiles remain pinned. The role receipt,
runtime hash, result envelope, resume validation, and audit bind both values.
The compiler remains at 1.0 because V13's compiler artifacts are valid; only
the executor's temperature is changed to test deterministic greedy decoding
for the fixed raw-native final-channel request.

Use a fresh V14 root. Copy only the four validated compiler artifacts, create
a fresh ROLES8 receipt, run the complete prefix once, and audit it. V13 and its
diagnostic root are immutable and must not be resumed or rewritten. Any
failure is a terminal HOLD: do not retry, do not reuse the root, and do not
increase the compiler cap. No further compiler-cap increase is permitted. The
claim ceiling remains raw exploratory counts only; not confirmatory evidence.

```sh
mkdir -p artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v14/compiled-skills
cp artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v12/compiled-skills/*.json \
  artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v14/compiled-skills/

uv run python scripts/verify_ollama_gpt_oss_role_profile.py \
  --host http://127.0.0.1:11435 \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --output artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v14/role-conformance-receipt.json

uv run shadowskillbench pilot live-run \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --api-key-environment SSB_OLLAMA_LOCAL_TOKEN \
  --role-profile-receipt artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v14/role-conformance-receipt.json \
  --limit 60 \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v14

uv run shadowskillbench pilot audit \
  --output-dir artifacts/exploratory-live-pilot/ollama-gpt-oss-20b-harmony-roles-v14
```

V14 completed the full prefix on its first fresh run: `completed=60`,
`skipped=0`, `limit=60`, with one model call per cell and no retry. The audit
passed with plan hash
`sha256:c91936790f6144499ca4fcca6e00360c6494653e63d32670c003497f50565190`.
Each of the six domain/condition/ratio groups contains 10 observations; every
group reports zero task completions and zero completions under policy. These
are raw exploratory counts only, not a model comparison, benchmark finding, or
confirmatory result. The immutable local artifact root remains untracked; its
sanitized committed custody summary is
`docs/development/V14-exploratory-pilot-receipt.md`.

## Frozen Ollama 0.33.2 runtime profile (V15)

The installed local runtime advanced from Ollama 0.33.1 to 0.33.2 before any
confirmatory model call. V15 freezes
`SSB-OLLAMA-GPT-OSS-20B-HARMONY-ROLES9`; its only runtime change is the Ollama
server version. Compiler temperature remains 1.0, executor temperature remains
0.0, reasoning effort, caps, timeout, context, seeds, top-p, models, model
digests, Modelfiles, prompts, schemas, transports, and request profiles remain
unchanged. A fresh ROLES9 receipt is required. The ROLES8 receipt remains valid
custody for V14 only and cannot authorize a V15 or confirmatory run. No further
compiler-cap increase is permitted.
