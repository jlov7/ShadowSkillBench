# Security, Privacy, and Governance Specification

## 1. V1 data boundary

All v1 demonstrations, people, companies, policies, approvals, and screenshots
are synthetic.

No real employee or client activity is required.

## 2. Why this matters

Record-to-skill products observe screen content, clicks, typing, and sometimes
narration. Official product guidance warns users not to expose passwords,
secrets, private files, or conversations while recording.

A later real-world version would create surveillance, confidentiality,
employment, privacy, and privilege risks.

## 3. Prohibited v1 behavior

- no capture of real employee screens;
- no client data;
- no credentials;
- no uploading employer policy documents to public services;
- no arbitrary generated code execution;
- no automated consequential external actions;
- no secret values in model-visible manifests;
- no collection of private chain-of-thought.

## 4. Future real-trace requirements

Before any real trace work:

- explicit informed consent;
- purpose limitation;
- employee/works-council review where applicable;
- legal/privacy approval;
- retention schedule;
- local-first capture;
- PII/secret redaction;
- screen-region allowlists;
- role-based access;
- raw-versus-derived artifact separation;
- deletion and subject-access procedures;
- prohibition on productivity scoring from benchmark traces.

## 5. Threat model

### Skill poisoning by ordinary practice

Common unauthorized behavior becomes normative skill instruction.

### Privileged demonstrator leakage

A person with special authority demonstrates an action that ordinary agents
should not generalize.

### Missing approval context

The action is recorded but its external approval is not.

### Stale policy

The supplied document is no longer effective.

### Malicious skill artifact

A skill contains overtly harmful or injected instructions. Adjacent benchmark,
not primary ShadowSkill threat.

### Benchmark leakage

Compiler or executor sees hidden normative labels, held-out tasks, or verifier.

### Instruction-tier confound

Policy is more successful only because it is placed at a more privileged role.
This is measured explicitly, not hidden.

### Over-enforcement

A simplistic system blocks valid exceptions or current practice governed by
newer authority.

## 6. Security controls

- content-addressed artifacts;
- read-only confirmatory corpora;
- allowlisted tools;
- deterministic environment;
- no shell tool for the agent;
- scoped model adapter;
- structured output validation;
- audit log;
- separate scorer process;
- no model access to scorer files;
- CI secret scanning;
- dependency pinning;
- software bill of materials;
- least-privilege file permissions.

## 7. Research-integrity controls

- development/confirmatory separation;
- signed protocol tag;
- external timestamp anchor;
- append-only raw results;
- no replacement seed after outcomes;
- no cherry-picked case;
- all null results retained;
- claims ledger.
