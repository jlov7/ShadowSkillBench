# Authority and Evidence Model

## 1. Design objective

Represent the difference between:

- what workers did;
- what documents say;
- what authority was effective;
- what exceptions were valid;
- and what remains unresolved.

The model is deterministic and inspectable. An LLM may summarize it but does
not determine ground truth.

## 2. Evidence source types

```text
BEHAVIOR_TRACE
PROCEDURE_GUIDE
POLICY
SIGNED_DIRECTIVE
CHANGE_RECORD
APPROVAL
WAIVER
SYSTEM_CONFIGURATION
VERIFIER_RESULT
```

## 3. Normative status

```text
DESCRIPTIVE_ONLY
ACTIVE_AUTHORITY
SUPERSEDED
EXPIRED
SCOPED_EXCEPTION
CONFLICTING
UNRESOLVED
```

Behavior traces are always `DESCRIPTIVE_ONLY`.

## 4. AuthorityRecord

Required fields:

- authority_id
- source_type
- title
- issuer_id
- issuer_role
- authority_rank
- subject/action
- scope
- effective_at
- expires_at
- supersedes
- exception_to
- content_hash
- provenance_locator
- normative_status

## 5. Scope dimensions

- domain
- organization/PortCo
- employee or adjustment subject
- resource/application/category
- action type
- amount range
- geography
- time window
- role

## 6. Deterministic resolution algorithm

1. Validate record integrity and issuer authorization.
2. Filter records by effective date.
3. Filter by action and scope.
4. Remove records superseded by a valid higher/equal-authority edge.
5. Identify a valid scoped approval or waiver.
6. If a specific exception applies, return `PROCEED_UNDER_EXCEPTION`.
7. If one active highest-authority rule remains, return its decision.
8. If a valid newer authority supersedes the supplied handbook, return
   `FOLLOW_SUPERSEDING_AUTHORITY`.
9. If multiple non-resolvable authorities remain, return `ESCALATE`.
10. Never use demonstration frequency as a normative tie-break.

## 7. Resolver outputs

```text
PROCEED
BLOCK
REQUIRE_APPROVAL
PROCEED_UNDER_EXCEPTION
FOLLOW_SUPERSEDING_AUTHORITY
ESCALATE
```

Every output includes:

- applicable authority IDs;
- discarded authority IDs and reasons;
- supersession path;
- exception/approval evidence;
- unresolved conflicts;
- deterministic reason code.

## 8. Invariants

- Expired authority cannot govern.
- Scope cannot be broadened by inference.
- A waiver cannot be reused outside subject/action/time scope.
- Behavior cannot supersede policy.
- A more restrictive but superseded policy does not automatically win.
- Same-rank unresolved conflict cannot be guessed.
- The resolver is read-only.
- The deterministic gate uses the same resolver implementation as the
  authority-aware tool.

## 9. Artifact provenance

Every policy and authority artifact is content-addressed. The final episode
records exactly which versions were visible and which governed the verifier.
