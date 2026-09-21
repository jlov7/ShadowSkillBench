# Research boundary

## Current claim ceiling

This repository supports a software claim: the synthetic domains, authority
logic, offline example, analysis contracts, packaging, and read-only workbench
can be tested locally.

It does not support a confirmatory research claim. In particular, it does not
establish:

- aggregate model performance;
- a causal effect of skill induction or instruction placement;
- comparative model safety or capability;
- generalization to real organizations, policies, people, or production
  systems; or
- completion of the preregistered provider study.

The committed workbench fixture contains no benchmark episodes, traces, or
metrics. Its status is `HOLD`.

## Missing evidence

One historical A1 policy-only result was not persisted. It remains unknown. The
public candidate does not reconstruct it, substitute another condition, or use
it in an aggregate metric.

Historical exploratory artifacts, private review records, and internal
implementation ledgers are not part of this public repository. Their exclusion
is a publication boundary, not evidence that they passed a confirmatory gate.

Historical receipt assertions remain with the private source and are not part
of the public test suite. Current runtime, packaging, synthetic receipt, and
synthetic-contract tests remain active.

## Protocol status

The files under `protocol/` and the method documents define planned inputs and
checks. The immutable legacy v3 freeze manifest does not match 30 current
inputs in this software revision and refers to four unpublished historical
records: two architecture records and two internal implementation ledgers. The
public repository intentionally cannot reproduce that legacy freeze. The v4
manifest matches all 15 of its current inputs, but
its status is `HOLD_PENDING_NEMOTRON_VALIDATION_AND_FULL_V4_CUSTODY`. Neither
manifest is an anchor receipt, completed run, independent reproduction, or
publication approval.

Any later empirical release must preserve the designated human approval,
custody, missingness, audit, and claim-to-evidence gates. Local engineering
tests cannot replace those gates.
