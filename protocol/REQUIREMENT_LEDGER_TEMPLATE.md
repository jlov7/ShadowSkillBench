# ShadowSkillBench — Requirement Ledger Template

| Requirement | Source document | Owning module | Primary test | Artifact/evidence | Status |
|---|---|---|---|---|---|
| FR-001 Deterministic engine | PRD | `engine/` | replay/property gates | replay report | not_started |
| FR-002 Action traces | PRD | `traces/` | trace schema/worker tests | trace bundle | not_started |
| FR-003 Fixed-count bundles | PRD | `traces/bundles.py` | composition property | bundle manifest | not_started |
| FR-004 Skill compiler | PRD | `skills/compiler.py` | compiler integration | SkillIR/SKILL.md | not_started |
| FR-005 Frozen executor | PRD | `episodes/executor.py` | paired episode tests | episode artifacts | not_started |
| FR-006 Instruction tiers | PRD | `episodes/context.py` | confound tests | context hashes | not_started |
| FR-007 Policy salience | PRD | `policy/documents.py` | matched handbook test | policy artifacts | not_started |
| FR-008 Authority model | PRD | `authority/` | resolver properties | authority graph | not_started |
| FR-009 Final verifier | PRD | `domains/*/verifier.py` | mutation gates | score record | not_started |
| FR-010 Experiment runner | PRD | `experiments/runner.py` | resume/audit tests | run ledger | not_started |
| FR-011 Preregistration | PRD | `protocol/` | freeze validator | signed manifest | not_started |
| FR-012 Statistics | PRD | `analysis/` | known-effect tests | result tables | not_started |
| FR-013 Claims gate | PRD | `protocol/claims.py` | claims tests | claims report | not_started |
| FR-014 HTML report | PRD | `reporting/report.py` | golden report | report.html | not_started |
| FR-015 Workbench | PRD | `workbench/` | Playwright | screenshots/build | not_started |
