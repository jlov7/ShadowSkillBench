# ShadowSkillBench — Verification and Test Strategy

## 1. Testing objective

The test suite must prove both:

1. the software behaves as specified;
2. the experiment cannot silently become a different experiment.

## 2. Unit tests

Cover every pure predicate and schema:

- state transitions;
- approvals;
- reconciliation;
- authority scope;
- supersession;
- skill rendering;
- context assembly;
- metrics;
- claims validation.

## 3. Property-based tests

### Engine

- replay determinism;
- atomic failure;
- no mutation without event;
- state hash stable under map ordering.

### Demonstration bundles

- always 12 traces;
- exact class composition;
- deterministic shuffle;
- compiler projection never contains hidden labels.

### Authority

- expired record never applies;
- narrower scope never broadens;
- valid supersession removes old authority;
- behavior frequency never changes resolver outcome;
- unresolved equal authority always escalates.

### Policy salience

- identical clause bytes;
- identical content multiset;
- identical token count;
- only section order differs.

## 4. Mutation tests

Deliberately break:

- approval expiry;
- approval scope;
- authority rank;
- supersession edge;
- workaround classifier;
- CuP verifier;
- policy-byte equality;
- system/developer role placement;
- representative-case selector;
- analysis clustering.

Every mutation has a named test that fails.

## 5. Golden tests

Freeze:

- one access compliant trace;
- one access workaround trace;
- one finance compliant trace;
- one finance workaround trace;
- one SkillIR rendering;
- one same-tier context;
- one system-tier context;
- one authority resolution;
- one report fragment.

Golden updates require explicit review.

## 6. Leakage tests

Assert compiler/executor sandboxes cannot read:

- hidden normative labels;
- confirmatory case truth;
- scorer module;
- authority records in conditions that exclude them;
- development notes;
- prior model outcomes.

Inject canary strings into hidden files and scan all model requests.

## 7. Confound tests

### A3 versus A4

- policy bytes equal;
- skill bytes equal;
- task bytes equal;
- tools equal;
- budgets equal;
- only message role differs.

### A3 ordering

- block order exactly balanced within each domain/ratio/seed;
- order recorded;
- no priority wording.

### A5

- handbook token counts equal;
- target clause unchanged;
- only section order differs.

## 8. Statistics tests

Use generated datasets with known effects:

- zero slope;
- negative contamination slope;
- hierarchy interaction;
- cluster dependence;
- missing cell;
- duplicate episode;
- extreme separation.

Verify estimators and confidence intervals behave correctly.

## 9. End-to-end tests

### Offline path

Scripted compiler + scripted executor:

- generate bundle;
- compile skill;
- run all conditions;
- score;
- analyze;
- build report.

### Live smoke

Six cases only:

- one access compliant;
- one access workaround;
- one finance compliant;
- one finance workaround;
- one exception;
- one superseded policy.

No confirmatory run until smoke artifacts pass audit.

## 10. Release gates

```text
G0 schemas
G1 lint/type
G2 unit
G3 property
G4 mutation
G5 leakage
G6 confound
G7 replay
G8 integration
G9 analysis known-effect
G10 report golden
G11 claims
G12 clean reproduction
```

Every gate returns:

- 0 pass;
- 1 finding/failure;
- 2 configuration/could-not-run.
