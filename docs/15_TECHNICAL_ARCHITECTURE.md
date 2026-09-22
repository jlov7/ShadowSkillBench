# ShadowSkillBench — Technical Architecture and Repository Specification

## 1. Architectural thesis

ShadowSkillBench is an experiment pipeline with a presentation surface—not an
interactive automation platform.

Canonical truth is divided across four isolated layers:

1. **World truth** — deterministic state, tools, policy, and final verifier.
2. **Behavioral evidence** — scripted worker action traces.
3. **Learned artifact** — frozen SkillIR and rendered skill.
4. **Agent execution** — model-visible context, tool trajectory, and completion claim.

No layer may infer the hidden ground truth of another through filesystem access
or shared mutable objects.

## 2. System diagram

```text
Synthetic domain fixtures
        │
        ├── effective authority ───────────────┐
        │                                      │
        ▼                                      ▼
Scripted worker policies                  Hidden verifier
        │                                      │
        ▼                                      │
Action traces ──► bundle composer              │
        │                                      │
        ▼                                      │
Skill compiler ──► SkillIR ──► SKILL.md        │
        │                                      │
        └────────── frozen artifact ─────────┐ │
                                             ▼ ▼
Policy/skill context assembler ──► Agent executor
                                             │
                                             ▼
                                  Tool/action trajectory
                                             │
                                             ▼
                                  Deterministic scoring
                                             │
                         ┌───────────────────┴──────────────┐
                         ▼                                  ▼
                    Analysis store                   Trace/evidence store
                         │                                  │
                         └────────────► report ◄────────────┘
                                             │
                                             ▼
                                  read-only workbench
```

## 3. Layer boundaries

### 3.1 Core layer

Owns:

- canonical JSON/hashing;
- artifact envelopes;
- typed identifiers;
- deterministic time and seeds;
- error taxonomy.

Dependencies:

- standard library;
- Pydantic only.

Must not depend on:

- model providers;
- domains;
- UI;
- statistics.

### 3.2 Engine layer

Owns:

- world state;
- action execution;
- immutable events;
- snapshot/replay;
- atomicity.

Must not know:

- skill conditions;
- model messages;
- contamination ratio;
- policy placement.

### 3.3 Domain plugins

Each plugin implements:

```python
class DomainPlugin(Protocol):
    name: DomainName

    def initial_state(self, case: TaskCase) -> WorldState: ...
    def tool_specs(self) -> tuple[ToolSpec, ...]: ...
    def apply(self, state: WorldState, call: ActionCall) -> TransitionProposal: ...
    def verify(self, state: WorldState, case: TaskCase) -> OutcomeVerdict: ...
    def worker_policies(self) -> Mapping[str, ScriptedWorker]: ...
    def skill_semantics(self) -> SkillSemanticRules: ...
    def authority_fixtures(self, case: TaskCase) -> tuple[AuthorityRecord, ...]: ...
```

`SkillSemanticRules` includes:

- action intents that require approval;
- prohibited workaround signatures;
- required verification intents;
- safe variants;
- escalation conditions.

This enables deterministic contamination analysis without embedding
domain-specific checks in the generic skill module.

### 3.4 Trace layer

Owns:

- action trace schema;
- worker execution;
- bundle composition;
- compiler-safe projection;
- optional narration events.

The hidden benchmark metadata is stored in a separate file tree inaccessible to
the compiler process.

### 3.5 Skill layer

Owns:

- SkillIR;
- compiler prompt;
- model invocation;
- provenance map;
- rendered skill;
- deterministic contamination analysis.

The skill compiler process receives a serialized compiler projection through a
temporary sandbox directory. It does not receive the repository root.

### 3.6 Context and execution layer

Owns:

- condition definitions;
- message-role placement;
- token accounting;
- model execution loop;
- allowlisted tools;
- episode manifest;
- raw model response.

The executor receives only:

- task;
- condition context;
- model-visible tool schemas;
- current observations.

### 3.7 Authority layer

Owns:

- authority records;
- scope;
- effective time;
- supersession;
- exception resolution;
- deterministic gate.

The hidden final verifier and the authority resolver may share pure policy
predicates, but the agent cannot access verifier functions.

### 3.8 Analysis/reporting layer

Owns:

- immutable analysis table;
- estimators;
- confidence intervals;
- figures;
- representative-case selection;
- claims binding;
- HTML/report JSON.

It reads completed episode artifacts only.

### 3.9 Workbench

Reads `artifacts/reports/workbench.json`.

It has:

- no model client;
- no database writes;
- no statistics implementation;
- no endpoint that mutates results.

## 4. Repository layout

```text
shadowskillbench/
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .env.example
├── prompts/
│   ├── skill_compiler.md
│   └── executor_system.md
├── protocol/
│   ├── preregistration.md
│   ├── claims.yaml
│   ├── freeze_manifest.json
│   └── predictions.json
├── src/shadowskillbench/
│   ├── core/
│   ├── engine/
│   ├── domains/
│   │   ├── access/
│   │   └── finance/
│   ├── traces/
│   ├── skills/
│   ├── policy/
│   ├── authority/
│   ├── models/
│   ├── episodes/
│   ├── corpus/
│   ├── experiments/
│   ├── metrics/
│   ├── analysis/
│   ├── reporting/
│   └── cli.py
├── schemas/
├── tests/
│   ├── unit/
│   ├── property/
│   ├── mutation/
│   ├── golden/
│   ├── replay/
│   ├── leakage/
│   ├── confounds/
│   ├── integration/
│   └── e2e/
├── data/
│   ├── development/
│   └── confirmatory/
├── artifacts/
│   ├── traces/
│   ├── skills/
│   ├── episodes/
│   ├── oracle/
│   ├── analysis/
│   └── reports/
├── scripts/
└── workbench/
```

## 5. Identifier conventions

```text
world_access_<seed>
world_finance_<seed>
case_access_<class>_<index>
trace_access_<bundle>_<index>
bundle_access_r75_s02
skill_access_r75_s02
episode_a4_<skill>_<case>_r02
authority_finance_<type>_<index>
```

Identifiers contain domain and experimental structure but never hidden
compliance labels in model-visible artifacts.

## 6. Artifact paths

Canonical artifact path:

```text
artifacts/<type>/<first-two-hash-chars>/<sha256>.json
```

Human-readable indices map semantic IDs to content hashes.

Raw provider payloads are stored separately from normalized episode records.

## 7. Process isolation

Recommended processes:

1. corpus generator;
2. skill compiler worker;
3. executor worker;
4. scorer/auditor;
5. analyzer/reporter.

The compiler and executor are launched with allowlisted input directories.

The scorer runs after the executor terminates and receives hidden case truth.

## 8. Error taxonomy

```text
CONFIGURATION_ERROR
SCHEMA_ERROR
HASH_MISMATCH
PROTOCOL_VIOLATION
LEAKAGE_DETECTED
MODEL_PROVIDER_TRANSIENT
MODEL_PROVIDER_TERMINAL
MODEL_OUTPUT_INVALID
AGENT_BUDGET_EXHAUSTED
AGENT_INVALID_ACTION
ENVIRONMENT_INVARIANT_FAILURE
ANALYSIS_INTEGRITY_FAILURE
```

Only provider/infrastructure failures before meaningful agent behavior may be
technical exclusions.

## 9. Concurrency and resumability

- episode manifest is written before dispatch;
- output writes to temporary path;
- validate and atomically rename on completion;
- exact completed hash is idempotency key;
- worker lease has heartbeat;
- expired lease can be reclaimed;
- model call retries are protocol-defined;
- confirmatory episode content is immutable.

## 10. Configuration

Use typed configuration files:

```text
config/models.yaml
config/compiler.yaml
config/executor.yaml
config/corpora.yaml
config/statistics.yaml
```

Secrets come only from environment variables and never enter manifests.

## 11. Provider abstraction

Primary adapter is OpenAI-compatible but the benchmark core is provider-neutral.

The adapter must expose:

- role capabilities;
- model/version;
- seed support;
- usage;
- raw request/response hashes;
- retry classification.

If a provider cannot represent system and developer tiers, it cannot run the
primary instruction-hierarchy experiment. It may run a separately labeled
replication.

## 12. Reporting contract

`report_data.schema.json` must contain:

- protocol identity;
- corpus summary;
- model configurations;
- estimands;
- confidence intervals;
- raw count references;
- selected illustrative cases;
- claims status;
- limitations;
- artifact hashes.

The workbench consumes this contract only.
