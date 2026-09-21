# Protocol and methods

These documents describe the benchmark design and software contracts. They are
not evidence that a provider run or confirmatory study completed.

Start with:

1. [Research boundary](RESEARCH_BOUNDARY.md) for the current claim ceiling.
2. [Benchmark protocol](03_BENCHMARK_PROTOCOL.md) for conditions, stages, and
   evaluation rules.
3. [Authority and evidence model](06_AUTHORITY_AND_EVIDENCE_MODEL.md) for
   precedence, scope, supersession, approvals, and unresolved conflicts.
4. [Technical architecture](15_TECHNICAL_ARCHITECTURE.md) for process and
   artifact boundaries.
5. [Current architecture decisions](ARCHITECTURE_DECISIONS.md) for a concise
   statement of the implemented boundaries.
6. [Test strategy](16_TEST_STRATEGY.md) for the engineering verification model.

Domain and execution specifications:

- [Access provisioning](04_DOMAIN_ACCESS_PROVISIONING.md)
- [Financial adjustments](05_DOMAIN_FINANCIAL_ADJUSTMENTS.md)
- [Skill induction and execution](07_SKILL_INDUCTION_AND_EXECUTION_SPEC.md)
- [Experiment and statistics](08_EXPERIMENT_AND_STATISTICS_SPEC.md)
- [Security, privacy, and governance](10_SECURITY_PRIVACY_GOVERNANCE.md)
- [Offline worked example](OFFLINE_EXAMPLE.md)

The protocol-freeze runbooks are retained because they describe current
custody mechanics. The legacy v3 manifest has recorded input drift and refers
to four unpublished historical records: two architecture records and two
internal implementation ledgers. The v4 manifest matches its current inputs
but remains in `HOLD` status. These files are operational specifications, not
execution receipts:

- [Protocol freeze runbook](17_PROTOCOL_FREEZE_RUNBOOK.md)
- [Freeze builder runbook](PROTOCOL_FREEZE_RUNBOOK.md)
