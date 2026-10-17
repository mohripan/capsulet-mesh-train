---
name: openspec
description: Plan, review, implement, or resume repository change tasks and GitHub issues through this project's OpenSpec artifacts and user authorization workflow.
---

# OpenSpec

Use `docs/openspecs/<task-name>/` for each repository change task. This is the owner's three-artifact convention; it does not require the OpenSpec CLI. Read `AGENTS.md`, the issue/request, relevant source, existing decisions, and any existing task folder first. Ordinary questions and read-only research do not need their own change folder.

## Draft the task

Choose a stable kebab-case task name. Reuse its folder when resuming; do not create a new spec for each turn or for drafting the spec itself. Start from these assets and replace every template field:

- [proposal.md](assets/proposal.md): why, scope, exclusions, acceptance, review/authorization record.
- [design.md](assets/design.md): approach, contracts, rationale, alternatives, risks, verification.
- [tasks.md](assets/tasks.md): ordered work with an observable outcome and focused live verification for each step.

Keep all three files, with detail proportional to the change. Link the issue when there is one; a direct user task does not require inventing an issue. Inspect/research as needed to make the proposal concrete. Planning and spec revisions may proceed before implementation authorization; code, dependent test/config changes, and installations wait.

## Review and go-ahead

When the artifacts are ready, link them and offer one choice: personal review or an independent sub-agent review. For the latter, read [independent-review.md](references/independent-review.md). Do not launch a reviewer before that route is selected, unless the user already explicitly requested it.

Review is feedback, not implementation authorization. Address findings and obtain the user's go-ahead for the presented scope. Record the actual instruction, date, scope, and any conditions in `proposal.md`. Silence, a reviewer verdict, issue creation, and an older milestone design are not a new task's go-ahead.

**Respect authorization already given.** If the user approves these presented specs, says to proceed/just implement, explicitly waives review, or authorizes implementation after stated review conditions are met, record that instruction and proceed within its scope. Do not ask the same question again or introduce a second approval gate from another planning skill. A review choice alone does not authorize implementation.

If required scope or behavior materially changes, revise the affected artifacts and get direction on that change. Routine implementation choices within the approved contract do not need repeated approval. A resumed task keeps its existing authorization unless the user changes it.

## Implement and verify

Work through `tasks.md`, updating evidence as each step completes. Apply [coding-standards](../coding-standards/SKILL.md) to code changes, [live-verification](../live-verification/SKILL.md) to checks, and [architecture-decision-records](../architecture-decision-records/SKILL.md) when a lasting decision warrants an ADR.

Keep rationale in `design.md` and cross-link durable decisions. Check off a task only after its outcome and required focused verification pass. Record commands/procedures, environment, actual results, and useful artifacts. A blocked required check stays unchecked; distinguish implementation from verification when reporting progress.

At completion, reconcile the three documents with what shipped, summarize evidence and limitations, and update associated issue/dependency status within the task's authorization. Keep the folder at its stable path for future agents. Commit/push follows the user's existing instructions; this skill grants no additional publishing or messaging permission.
