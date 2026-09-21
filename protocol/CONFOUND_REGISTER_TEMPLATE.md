# ShadowSkillBench — Confound Register Template

| ID | Potential confound | Conditions affected | Prevention/control | Test | Status |
|---|---|---|---|---|---|
| CF-001 | More noncompliant demos also means more total tokens | Ratio sweep | Fix all primary bundles at 12 traces | `test_bundle_composition` | planned |
| CF-002 | Same-tier and system-tier policy wording differs | A3/A4 | Byte-identical policy artifact | `test_context_equivalence` | planned |
| CF-003 | Same-tier block order creates recency effect | A3 | Counterbalance order | `test_order_balance` | planned |
| CF-004 | Buried policy changes content, not salience | A5 | Identical section multiset/tokens | `test_matched_handbooks` | planned |
| CF-005 | Compiler sees hidden policy labels | Induction | Compiler projection + sandbox canaries | leakage gate | planned |
| CF-006 | Executor sees final verifier | All | Separate scorer process | leakage gate | planned |
| CF-007 | Skill compiler and executor model changes co-vary | All | Freeze separately; one executor across conditions | manifests | planned |
| CF-008 | Workaround examples are less successful locally | A2–A5 | Both worker classes achieve local task completion | worker tests | planned |
| CF-009 | Domain cases are not comparable across conditions | All | Exact paired case IDs/state | planner audit | planned |
| CF-010 | Policy hierarchy effect is provider-specific | A3/A4 | State scope; optional later replication | report limitation | planned |
| CF-011 | Authority resolver simply exposes answer | B2 | Resolver returns evidence/status, executor chooses action | tool contract test | planned |
| CF-012 | Deterministic gate blocks everything | B3 | Safe-control precision/false enforcement | Stage B metrics | planned |
