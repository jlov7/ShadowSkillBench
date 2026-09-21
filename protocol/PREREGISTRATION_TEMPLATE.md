# ShadowSkillBench v1.0 — Preregistration Template

> This file must be completed, validated, committed, signed, and externally
> anchored before confirmatory execution. `REQUIRED_BEFORE_FREEZE` is a
> machine-invalid sentinel.

## Protocol identity

- protocol_version: REQUIRED_BEFORE_FREEZE

## Models

### Skill compiler

- provider: REQUIRED_BEFORE_FREEZE
- model: REQUIRED_BEFORE_FREEZE
- model_version/date: REQUIRED_BEFORE_FREEZE
- system_prompt_hash: REQUIRED_BEFORE_FREEZE
- structured_output_schema_hash: REQUIRED_BEFORE_FREEZE
- temperature: REQUIRED_BEFORE_FREEZE
- seed: REQUIRED_BEFORE_FREEZE
- max_tokens: REQUIRED_BEFORE_FREEZE

### Skill executor

- provider: REQUIRED_BEFORE_FREEZE
- model: REQUIRED_BEFORE_FREEZE
- model_version/date: REQUIRED_BEFORE_FREEZE
- runtime_prompt_hashes: REQUIRED_BEFORE_FREEZE
- temperature: REQUIRED_BEFORE_FREEZE
- seed policy: REQUIRED_BEFORE_FREEZE
- max_turns: REQUIRED_BEFORE_FREEZE
- max_tool_calls: REQUIRED_BEFORE_FREEZE
- max_tokens: REQUIRED_BEFORE_FREEZE

## Domains

- access_provisioning world hash: REQUIRED_BEFORE_FREEZE
- financial_adjustments world hash: REQUIRED_BEFORE_FREEZE

## Demonstration bundles

- primary trace count: 12
- ratios: [0.0, 0.25, 0.5, 0.75, 1.0]
- bundle seeds per domain/ratio: 3
- generator hash: REQUIRED_BEFORE_FREEZE
- trace schema hash: REQUIRED_BEFORE_FREEZE
- narration: disabled in primary Stage A

## Conditions

- A0 Bare: REQUIRED_BEFORE_FREEZE
- A1 PolicyOnlySystem: REQUIRED_BEFORE_FREEZE
- A2 SkillOnly: REQUIRED_BEFORE_FREEZE
- A3 SkillPolicySameTier: REQUIRED_BEFORE_FREEZE
- A4 SkillPolicySystemTier: REQUIRED_BEFORE_FREEZE
- A5 SkillBuriedPolicySameTier: REQUIRED_BEFORE_FREEZE
- B0 SkillOnly: REQUIRED_BEFORE_FREEZE
- B1 FlatPolicySystem: REQUIRED_BEFORE_FREEZE
- B2 AuthorityResolver: REQUIRED_BEFORE_FREEZE
- B3 DeterministicGate: REQUIRED_BEFORE_FREEZE

Confirm:

- A3/A4 policy text is byte-identical: REQUIRED_BEFORE_FREEZE
- A3 block order is counterbalanced: REQUIRED_BEFORE_FREEZE
- A3/A4 total context accounting is logged: REQUIRED_BEFORE_FREEZE
- A5 buried-handbook artifact, target position, and token count are frozen: REQUIRED_BEFORE_FREEZE

## Primary outcomes

- Task Completion
- Completion Under Policy

## Primary Stage A estimands

- E1 Behavioral contamination slope
- E2 Same-tier policy attenuation
- E3 Instruction hierarchy
- E4 Skill interference under policy
- E5 Buried-handbook contextual penalty

## Primary Stage B estimands

- E6 Unsafe imitation
- E7 Flat-policy false enforcement
- E8 Authority-aware gain
- E9 Deterministic enforcement gap

## Figure 1 predictions

Enter point prediction and 80% subjective interval for each condition and ratio.

| Condition | R0 | R25 | R50 | R75 | R100 |
|---|---|---|---|---|---|
| A1 | REQUIRED | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| A2 | REQUIRED | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| A3 | REQUIRED | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| A4 | REQUIRED | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| A5 | REQUIRED | REQUIRED | REQUIRED | REQUIRED | REQUIRED |

- predicted probability of null A4–A3 hierarchy effect: REQUIRED
- predicted probability that B1 over-enforces at least one authority class: REQUIRED
- predicted probability that B2 adds no value over B1: REQUIRED

## Confirmatory corpus

### Stage A

- skill bundles: 30
- held-out cases per domain: 20
- repeats: 3
- planned skill-dependent episodes: 7,200
- planned control episodes: 240
- total: 7,440

### Stage B

- R75 skills per domain: 3
- held-out cases per authority class/domain: 5
- authority classes: 5
- conditions: 4
- repeats: 3
- total: 1,800

## Statistical plan

- primary model: REQUIRED_BEFORE_FREEZE
- clustering: REQUIRED_BEFORE_FREEZE
- bootstrap: REQUIRED_BEFORE_FREEZE
- confidence interval: REQUIRED_BEFORE_FREEZE
- marginal effects: REQUIRED_BEFORE_FREEZE
- per-domain reporting: REQUIRED_BEFORE_FREEZE
- multiple-comparison family: REQUIRED_BEFORE_FREEZE
- material-effect threshold: REQUIRED_BEFORE_FREEZE

## Exclusion rules

Only technical failures that occur before agent behavior begins may qualify.

- provider unavailable before first response: REQUIRED_BEFORE_FREEZE
- malformed provider payload after retry: REQUIRED_BEFORE_FREEZE
- environment hash mismatch: REQUIRED_BEFORE_FREEZE
- runner crash before first action: REQUIRED_BEFORE_FREEZE

The following are outcomes, not exclusions:

- refusal;
- invalid tool call;
- agent timeout;
- budget exhaustion;
- unsafe imitation;
- false enforcement;
- unnecessary escalation;
- wrong completion claim.

## Representative-case selection

### Stage A

1. cases where A2 fails CuP and A4 passes in majority of repeats;
2. median contamination ratio;
3. median action-trace length;
4. lexicographically smallest case ID.

### Stage B

1. cases where B1 and B2 differ;
2. median authority-graph depth;
3. lexicographic case-ID tie-break.

## Freeze statement

After external anchoring:

- no prompt changes;
- no model changes;
- no corpus/generator changes;
- no policy text changes;
- no condition changes;
- no metric changes;
- no exclusion changes;
- no seed replacement after outcomes.

Any amendment creates a new protocol version. Amended results are exploratory
until separately confirmed.
