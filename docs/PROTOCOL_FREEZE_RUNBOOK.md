# Protocol-freeze implementation boundary

This companion describes the local freeze builder. The operating procedure remains
authoritative in
[`17_PROTOCOL_FREEZE_RUNBOOK.md`](17_PROTOCOL_FREEZE_RUNBOOK.md).

## What the builder does

`shadowskillbench.protocol.freeze` is offline and deterministic. It validates the
completed `protocol/preregistration.md` and claims ledger, reads the closed input
list, hashes raw bytes, verifies every SHA-256 reference in the preregistration
core against a frozen input, and produces:

- `protocol/freeze_manifest.json` — canonical JSON input inventory;
- `protocol/freeze_manifest.sha256` — detached hash of those exact manifest bytes;
- `protocol/FREEZE_SUMMARY.md` — a deterministic review summary.

It fails closed for an invalid core or claims ledger, missing/non-regular/symlinked
referents, a hash reference with no matching frozen bytes, duplicate inputs, unsafe
paths, an attempted manifest-output/anchor-receipt input, or pre-existing output.
The closed input list covers the protocol and schemas; corpus-development and domain
construction; frozen model, compiler, executor, corpus, and statistics profiles;
the role-conformance receipt and native run descriptor; policy, authority,
source-trace and compiler/executor paths; condition assembly; planner, runner,
audit and metrics; and Stage A/B estimators plus the representative-case selector.

## Two-phase custody boundary

The freeze manifest is phase one. It contains no commit identity, tag, locator,
timestamp, or custody receipt. In particular,
`protocol/anchor_receipt.json` is forbidden as an input so that creating the receipt
cannot mutate the scientific commitment.

After review, the operator performs the local commit-and-tag commands in the
authoritative runbook:

```bash
git commit -m "research: freeze ShadowSkillBench v1.0 protocol"
git tag -a shadowskillbench-protocol-v1.0.0-local \
  -m "ShadowSkillBench v1.0 confirmatory protocol"
git show --no-patch shadowskillbench-protocol-v1.0.0-local
```

No signing key or remote publication is required for private local execution.

## Local-custody checklist

- Confirm the freeze commit contains the reviewed manifest and detached SHA file.
- Create and inspect the ordinary freeze commit and annotated local tag.
- Record the exact commit, protocol tag, manifest SHA-256, `urn:git:` locator, UTC
  timestamp, and `VERIFIED_LOCAL` result in the post-freeze custody receipt.
- Treat the result as locally hash-bound, not signed or independently anchored.
- Keep that receipt outside the manifest input set.
- Do not generate a confirmatory corpus or run confirmatory episodes until the custody
  receipt binds the exact phase-one manifest.
- Do not accept an operator-supplied development split inventory: post-custody staging
  derives it from the frozen seed and generator bytes.

Any defect after this point requires the amendment procedure in the authoritative
runbook and a new protocol version; it does not permit replacing a frozen artifact.
