# Design: Repository workflow skills

Status: implementing under the user's 2026-10-17 go-ahead. The [proposal](proposal.md) records scope and authorization; [tasks](tasks.md) tracks implementation and verification.

## Storage and discovery

Use `.agents/skills/` as the canonical repository-local skill root. Each skill has a `SKILL.md` with a discriminating name/description and optional `agents/openai.yaml` metadata. Automatic selection stays enabled. `AGENTS.md` links the local entry points and states the task workflow explicitly, rather than relying exclusively on automatic discovery.

Keep the existing `find-skills` installation, `.claude/` links, and `skills-lock.json` intact. Stage only files owned by this task. A global install or agent-specific symlink installation is not necessary to establish the repository workflow.

## OpenSpec artifacts and lifecycle

Use stable kebab-case task names, reusing an existing folder when resuming the same task. Each folder contains:

- `proposal.md`: problem, scope, exclusions, acceptance criteria, issue/source links, review status, and explicit authorization record.
- `design.md`: technical approach, boundaries, behavior/contracts, alternatives and rationale, risks, verification strategy, and linked ADRs when warranted.
- `tasks.md`: ordered checkboxes connecting implementation steps to observable outcomes, focused verification, and recorded evidence.

Templates should be short prompts, not mandatory long questionnaires. Do not leave unresolved placeholders in a ready-for-review spec. Additional artifacts such as a review report are optional when useful; the three required documents remain the entry point.

The usual lifecycle is draft → review requested → reviewed/revised → user authorized → implementing → verified. A reviewer may find no issues without granting implementation permission. Record the user's actual authorization and its scope; do not manufacture approval from silence, an older project decision, or the creation of a GitHub issue. If the user explicitly says to proceed with the presented specs, waives review, or authorizes implementation conditional on review findings being addressed, preserve that instruction and proceed once its conditions are satisfied without asking the same question again.

Initial repository inspection, issue reads, research, and drafting/revising specs are allowed before the go-ahead. Implementation, dependent code/test changes, installations, and configuration changes wait for it. Material changes to scope or behavior return the changed spec for review; routine implementation details within the approved contract do not create repeated gates.

For a newly introduced repository workflow, these files are the bootstrap spec. Creating them does not require recursively creating another OpenSpec for the same planning work.

## Independent review

After linking the three completed documents, offer two routes: the user reviews them, or a fresh sub-agent reviews them. Choosing the sub-agent route authorizes that review, not implementation by itself.

Use a new agent with `fork_turns="none"` when available. Provide only the task's raw requirements, exact artifact paths, repository location, and necessary issue/source identifiers. Do not send the author's reasoning, desired verdict, prior findings, or conversation history. Give neutral instructions to evaluate correctness, project fit, scope, missing contracts, verification, and unresolved choices. The reviewer may independently read `AGENTS.md`, roadmap/decisions/ADRs, relevant source, the issue, and primary references.

The reviewer is read-only: no implementation, file edits, installs, commits, issue mutations, or messages to others. Its output identifies findings by severity and file/section or line, gives supporting evidence, and distinguishes blockers from suggestions. A no-findings report should state its coverage and limits. The author addresses findings, records the result, and obtains the user's go-ahead unless it was already explicitly given for that reviewed scope.

If a fresh-context agent is unavailable, report that limitation and offer personal review. Do not describe an inherited-context agent or the author's own reread as independent review.

## Coding standards and rationale

Apply the coding skill while implementing approved tasks. Follow the repository's architecture, narrow interfaces, ownership/error contracts, target-scoped CMake conventions, formatting, and meaningful verification. Use clear names and focused units before adding explanatory comments.

Ordinary comment blocks and docstrings in new or changed code have at most two physical lines; adjacent fragments must not be used to bypass the limit. Keep only concise contracts, constraints, or necessary references near code. Put explanations of why an algorithm, ownership model, workaround, or trade-off was chosen in the task's `design.md`. An optional short code reference can point to that section without repeating its rationale. Preserve legal notices, generated content, and tool-required annotations; do not remove meaningful contracts merely to satisfy a line count.

Task-local rationale lives in the OpenSpec. A decision with durable architectural, compatibility, numerical, ownership, or failure consequences also gets a linked ADR in `docs/adr/`. Keep the full explanation in the most appropriate document and cross-link rather than duplicating long passages.

## ADR reuse

Start from the inspected `wshobson/agents` skill at the pinned source revision in the proposal. Inspect its referenced templates, retain the MIT notice and source provenance, and adapt its instructions to this repository. Preserve the purpose and useful decision templates while removing unrelated tool-installation and team-approval prescriptions.

Use the existing sequential `NNNN-title.md` convention, status meanings, and index. A routine local change need not create an ADR. Accepted ADR content remains historical; a changed decision gets a superseding record and appropriate links/status updates. Acceptance must reflect an actual project/user decision. The OpenSpec design links the relevant ADR, and the ADR links back to the task/issue.

## Focused live verification and nightly coverage

The verification skill asks what changed and what observable result would demonstrate it. Choose the smallest credible execution path using real task inputs and the actual relevant boundary. For native code this might be a targeted CTest case or a small actual worker exchange; for bindings, an isolated import and mutation/ownership check. For documentation/skills, validation plus realistic workflow walkthroughs is sufficient. Browser testing is appropriate only when the task has a browser surface.

Fast focused checks may cross a real boundary even if a test framework labels them integration tests. Broad multi-component matrices, complete training runs, and end-to-end suites belong to nightly coverage. An explicit user request or a material unresolved risk can justify an on-demand broader run; the agent should state why and keep it relevant. Do not automatically run unrelated full suites on every small change.

Record the command or procedure, environment, expected observable behavior, actual result, and relevant artifact or metric with the task. Distinguish pass, failure, blocked, deferred, and not run. A blocked required live check leaves its task unverified; never turn a simulation or unavailable GPU into evidence of an actual run. Fix relevant failures and rerun affected checks without silently weakening acceptance criteria.

This task adds the workflow policy, not nightly CI scheduling. Existing native CI currently runs its full suite and container checks on push/PR. Future CI work must explicitly choose the fast subset and schedule the broad suites before claiming this cadence is automated.

## Skill validation

Use the installed skill-creator validator for frontmatter/package integrity. Check local reference/template links and inspect the adapted upstream files. Walk through realistic requests: a fresh issue, resuming an approved task, choosing independent review without a go-ahead, conditional authorization, moving a long rationale comment, a durable ADR decision, and a live-check failure/unavailable environment. Evaluate observable actions and gate behavior; avoid tests that only match prose or headings.

Keep instructions self-contained and concise. The four skills have distinct roles; `openspec` coordinates them rather than duplicating their full guidance. Broader brainstorming may inform a proposal, but it must not add a second conflicting approval workflow after the OpenSpec gate has already been satisfied.
