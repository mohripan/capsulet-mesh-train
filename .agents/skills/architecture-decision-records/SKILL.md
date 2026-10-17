---
name: architecture-decision-records
description: Draft, review, or supersede durable architecture decisions in docs/adr, linking OpenSpec rationale and actual user decisions while preserving accepted history.
license: MIT
---

# Architecture decision records

Adapted from the existing `wshobson/agents` ADR skill; see [upstream provenance](references/upstream.md) and [LICENSE](LICENSE). This repository's OpenSpec workflow governs review and implementation authorization.

## Decide whether a record is useful

Record a choice with lasting architectural, interface, compatibility, numerical, ownership, or failure consequences, especially when alternatives have meaningful trade-offs. Ordinary local details, typo fixes, and routine maintenance do not need an ADR. Keep task-local rationale in the OpenSpec `design.md` regardless.

Read `docs/adr/README.md`, relevant accepted ADRs, `docs/DECISIONS.md`, and the task's OpenSpec first. Follow existing decisions unless the task explicitly changes them. Use [the ADR template](assets/adr.md) when drafting a record; scale detail to the actual decision.

## Write and connect the decision

1. Use `docs/adr/NNNN-short-title.md`, taking the next unused number from the actual files/index. Do not create another decision directory or renumber existing records.
2. State context/constraints, the decision and why, viable alternatives/trade-offs, consequences, and revisit triggers. Link the OpenSpec and issue; link the ADR back from `design.md`.
3. Mark a draft **Proposed**. Use **Accepted** only when an actual user/project decision supports it; cite that authorization. A reviewer verdict alone is not acceptance or implementation approval. An already explicit decision needs no redundant permission request.
4. Preserve accepted decision content as history. A changed decision gets a new record that identifies what it supersedes. Update old status/supersession links and the index without rewriting the original rationale. Use Deprecated or Rejected when those statuses accurately describe the decision.

Use the user's selected OpenSpec review route; read the OpenSpec skill's independent-review procedure if a fresh reviewer was requested. Do not impose a senior-reviewer count, team notifications, separate RFC bureaucracy, or an ADR tool installation. This skill does not authorize messaging others.

## Verify

Check that numbering is unused, statuses match real decisions, local links resolve, alternatives/consequences are concrete, and the OpenSpec/ADR/index point to each other. Do not add a new ADR merely to document use of this skill. Preserve the existing record when the task only needs to explain it.
