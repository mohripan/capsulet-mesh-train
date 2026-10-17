# Proposal: Repository workflow skills

Date: 2026-10-17

Status: complete — implemented, verified, and pushed under the user's 2026-10-17 go-ahead.

## Why

The owner wants every repository change task or GitHub issue to start with reviewable specifications, give the owner a choice of personal or independent agent review, and receive explicit implementation authorization. Future agents need this workflow in the repository. Short code comments should describe essential local contracts; explanations of design choices belong in OpenSpec documents.

This proposal applied the new workflow to its own creation. After the three planning documents were presented, the owner explicitly instructed implementation. The four skills and navigation are implemented and verified; runtime code and CI cadence remain unchanged.

## Proposed changes

Create four discoverable, repository-local skills under `.agents/skills/`:

| Skill | Responsibility |
| --- | --- |
| `openspec` | Draft `proposal.md`, `design.md`, and `tasks.md` in `docs/openspecs/<task-name>/`; offer the review choice; track review and explicit user authorization; maintain the task checklist during execution. |
| `coding-standards` | Follow project architecture and formatting conventions; keep ordinary code comment blocks to at most two lines; place rationale in the task's OpenSpec design. |
| `architecture-decision-records` | Adapt the inspected existing ADR skill to the project's numbering, existing records, and OpenSpec review/authorization workflow. |
| `live-verification` | Define and perform focused checks of the changed behavior, record actual evidence, and distinguish those checks from broad nightly integration/end-to-end suites. |

Add concise routing in `AGENTS.md` so the workflow applies even when automatic skill discovery is unavailable. Keep templates and independent-review instructions with the relevant skill. Link the ADR skill to the current `docs/adr/` conventions rather than creating another decision directory.

## Acceptance criteria

1. A new repository change task produces the three requested Markdown documents before implementation. Their length scales with the change; even a small change keeps all three files concise.
2. The agent offers personal review or an independent sub-agent review. A chosen sub-agent starts without parent conversation/history, can inspect project files and research relevant facts, and returns evidence-based findings without implementing.
3. Review completion alone does not authorize implementation. The user must give a go-ahead for the reviewed scope, including an explicitly stated conditional go-ahead if applicable. Silence and a reviewer verdict are not approval.
4. Each implementation task identifies an observable outcome and focused live verification. Its checkbox is completed only after the work and required checks pass, with commands/results recorded.
5. New or changed ordinary comment blocks and docstrings contain at most two lines. Rationale moves into `design.md`, with durable decisions linked to ADRs. Mandatory legal notices and generated annotations are preserved. Existing comments are not rewritten across the repository as incidental cleanup.
6. The ADR skill preserves accepted records, distinguishes proposed from accepted decisions, and records upstream provenance/license. It does not impose unrelated team-review, notification, or tool-installation requirements.
7. Broad integration/end-to-end coverage is a nightly policy; focused real-boundary checks remain available when necessary to verify the task. Skipped, unavailable, and not-yet-scheduled checks are reported accurately.
8. The skills validate as skill packages and pass realistic workflow walkthroughs, including approval-gate and failure cases. Existing local skills/configuration remain intact.

## Scope

This task creates reusable instructions/templates and updates navigation. It does not start #6, refactor existing C++ comments, install the OpenSpec CLI, create a global skill installation, or change the native CI schedule. Turning the nightly policy into concrete CI jobs is separate work; no nightly execution will be claimed before those jobs exist.

The latest owner instruction introduces a per-task review/go-ahead gate. Earlier M1 design acceptance remains useful project context but is not blanket approval for future task-specific OpenSpecs. Previously completed issues and specs remain historical evidence.

## Review and authorization record

- Review route: the owner waived a separate review and directed implementation after the specs were presented.
- Reviewer findings: no independent review was requested or performed. Validation will distinguish manual walkthroughs from independent agent evaluation.
- Implementation authorization: on 2026-10-17, the owner said, "Just implement it." This authorizes the presented workflow-skills scope; no repeated review question is required.
- Commit/push: follows the owner's standing instruction after authorized work and verification are complete, without a co-author trailer.

## Delivery

Implementation commit [35d2c86](https://github.com/mohripan/capsulet-mesh-train/commit/35d2c860c569f114e3955d1da965304a72084cfb) was pushed to `main`; the remote branch was independently checked with `git ls-remote`. It contains the four skills, navigation, and this OpenSpec, with no co-author trailer. [Verification](verification.md) records package checks, actual scratch artifacts, manual workflow cases, and their limits. No independent agent evaluation, runtime change, or nightly scheduling is claimed.

## Research

- [Upstream OpenSpec concepts](https://github.com/Fission-AI/OpenSpec/blob/main/docs/concepts.md) and [change review guidance](https://github.com/Fission-AI/OpenSpec/blob/main/docs/reviewing-changes.md) informed the artifact roles. This repository will use the owner's `docs/openspecs/<task-name>/` convention, without claiming CLI compatibility with upstream's default artifact tree.
- The skills CLI search returned `wshobson/agents@architecture-decision-records` with approximately 20.6K installs. GitHub reported 40,934 repository stars and an MIT license when checked on 2026-10-17. Popularity is context; the source itself was inspected for fit.
- Proposed ADR source: [`architecture-decision-records/SKILL.md` at commit `044b4b21f083948e69ec878a7b07c8d0951a6ffa`](https://github.com/wshobson/agents/blob/044b4b21f083948e69ec878a7b07c8d0951a6ffa/plugins/documentation-generation/skills/architecture-decision-records/SKILL.md). Its two template references and MIT notice were also inspected. Preserve the license and relevant template structure during adaptation; its generic two-senior-reviewer, team-notification, and example migration-time estimates will not become project requirements.
