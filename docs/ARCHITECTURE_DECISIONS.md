# Current architecture decisions

This document summarizes the implementation boundaries that matter to a public
reader. It is not a historical decision diary, a freeze input, or evidence that
a provider study completed.

## Behavior and authority stay separate

Demonstration traces describe actions that occurred. They do not establish
which action was authorized. Policy documents, directives, approvals, waivers,
effective dates, scope, and supersession remain separate inputs to the authority
resolver.

The model-visible condition context and the hidden evaluation boundary are also
separate. An executor cannot read the final verifier or use normative labels
that were withheld from its condition.

## State transitions are deterministic and atomic

Synthetic domains own their state and accept typed actions. A transition either
produces the complete next state or leaves the previous state unchanged. Replay
uses canonical inputs and verifies supplied artifact hashes before accepting a
result.

## Learned skills have a restricted projection

The compiler receives only the declared projection of demonstration bundles.
Held-out cases, hidden labels, current policy, and authority records are outside
that projection. `SkillIR` is schema-validated, canonically serialized, and
rendered deterministically.

## Model adapters are provider-neutral at the runtime boundary

The runtime consumes structured model requests and receipts. Provider-specific
configuration stays in the selected adapter and run descriptor. Ordinary local
verification uses scripted clients and does not require a live endpoint or
credential.

## Provider failures do not become outcomes silently

Adapter retries are bounded and classified. A runner may repeat a whole episode
only for a pre-behavior provider outage allowed by a predeclared exclusion rule
and retry budget. Any model output, action, result, or event counts as meaningful
behavior and prevents that technical retry. Output-invalid, configuration, and
budget failures are not eligible.

## Evidence is content-addressed

Protocol inputs, run descriptors, runtime receipts, analysis inputs, reports,
and workbench exports carry content hashes. Missing or inconsistent evidence
produces an unavailable result or `HOLD`; code does not invent a replacement.

## Reproduction and live replication are different operations

Offline reproduction verifies sealed artifacts and rebuilds deterministic
derived outputs. It does not call a provider. A live replication is a separate
operation with its own model identity, endpoint, request, and custody records;
it is not expected to reproduce provider bytes.

## The workbench is a read-only view

The workbench reads an exported report dataset. It has no provider client,
statistics engine, result-mutation endpoint, or database write path. The
committed public fixture contains no research results and displays `HOLD`.

## Related specifications

- [Authority and evidence model](06_AUTHORITY_AND_EVIDENCE_MODEL.md)
- [Skill induction and execution](07_SKILL_INDUCTION_AND_EXECUTION_SPEC.md)
- [Experiment and statistics](08_EXPERIMENT_AND_STATISTICS_SPEC.md)
- [Technical architecture](15_TECHNICAL_ARCHITECTURE.md)
- [Test strategy](16_TEST_STRATEGY.md)
- [Research boundary](RESEARCH_BOUNDARY.md)
