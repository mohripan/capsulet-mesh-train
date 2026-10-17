# Tasks: Repository workflow skills

Implementation status: in progress. The owner explicitly directed implementation on 2026-10-17 after these specs were presented; see [proposal.md](proposal.md). No independent review is being claimed.

For each task, record actual verification evidence beneath its checkbox before marking it complete. A required check that is blocked or not run leaves the task unverified. Broad integration/end-to-end execution is outside this skill-authoring task.

- [x] **1. Create the OpenSpec skill and its three concise artifact templates.**
  - Outcome: a new task produces `docs/openspecs/<task-name>/{proposal,design,tasks}.md`; resuming reuses the same folder and authorization record.
  - Live verification: validate the skill package and walk through fresh-task and resume examples in a temporary scratch directory; inspect the resulting three artifacts for review readiness and per-task verification steps.
  - Evidence: package validator passed; all three templates were instantiated with concrete fields and verified local links. Resume reused the same scratch task folder and its go-ahead. See [verification](verification.md#actual-scratch-artifacts).

- [x] **2. Add the review-choice and fresh-context review procedure.**
  - Outcome: personal review and independent sub-agent review are both supported; review alone never becomes implementation approval. Conditional user authorization is preserved.
  - Live verification: walk through choosing only review, an explicit later go-ahead, and a conditional go-ahead. Inspect the reviewer prompt for exact artifact paths, raw requirements, no inherited history, neutral criteria, and read-only scope. Use a fresh sub-agent only when that review route has been chosen and tooling is available.
  - Evidence: the reference specifies raw requirements, exact paths, `fork_turns="none"`, neutral criteria, and read-only scope. Manual review/conditional-authorization/resume cases were checked; the current user's direct go-ahead was honored without another gate. No independent reviewer was launched. See [workflow cases](verification.md#manual-workflow-cases).

- [x] **3. Create the coding-standards skill linked to OpenSpec.**
  - Outcome: rationale is captured in `design.md`, ordinary comments/docstrings stay within two lines, and essential contracts/legal notices remain intact.
  - Live verification: apply it to a scratch example with a long rationale comment and a short API contract; inspect both the revised code and matching OpenSpec section. Confirm it does not propose unrelated repository-wide rewrites.
  - Evidence: package validator passed. A scratch header retained its signature and two-line contract while its rationale moved into the generated design; no project source/comment rewrite occurred. See [scratch artifacts](verification.md#actual-scratch-artifacts).

- [x] **4. Import and adapt the existing ADR skill.**
  - Outcome: the pinned upstream skill/templates have preserved provenance/license, use `docs/adr/`, and follow the project's review/authorization rules without extra team or messaging requirements.
  - Live verification: inspect the source/templates/license, validate the adapted package, and draft a temporary ADR linked to a sample OpenSpec. Verify numbering/status against the existing index and that a routine change does not demand an ADR.
  - Evidence: adapted inspected upstream revision `044b4b21f083948e69ec878a7b07c8d0951a6ffa`; retained the exact MIT license and source notes. Package validation and a linked scratch Proposed ADR 0002 passed. No extra reviewer count, notification, or tool-installation prescription remains. See [provenance](verification.md#provenance-and-preservation).

- [x] **5. Create the focused live-verification skill.**
  - Outcome: tasks choose a credible small live check, record actual results, and distinguish broad nightly coverage from completed local proof.
  - Live verification: walk through a native behavior change, a documentation-only change, a failed check, and unavailable hardware/tooling. Confirm a blocked required check stays unverified and that the skill does not claim nightly jobs exist.
  - Evidence: package validator passed. A real broken-link fixture stayed unchecked until repaired/rechecked. Documentation-only checks were performed; native/unavailable-hardware cases were manual procedure walkthroughs, with no unperformed run claimed. Existing CI cadence is explicitly disclosed. See [verification](verification.md).

- [x] **6. Wire repository navigation and verify the skills together.**
  - Outcome: `AGENTS.md` routes future task/issue work through OpenSpec and links the four skills. Existing M1 documents/ADR history and unrelated installed skills remain intact.
  - Live verification: validate all four packages and their references; use a realistic issue request to inspect the complete draft → review → authorized work → focused verification flow. Check the staged diff for unrelated files and confirm that runtime code/CI cadence were not changed.
  - Evidence: `AGENTS.md` routes new work through the four skills; README links the workflow. All four skill packages/metadata and their local references validate. Scope review confirms no runtime or CI configuration changes and excludes pre-existing local skills/configuration. See [verification](verification.md#package-checks).

- [ ] **7. Complete documentation and deliver the authorized change.**
  - Outcome: this OpenSpec records the real review route, authorization, completed work, and verification evidence; commit/push follows the standing main-branch instruction with no co-author trailer.
  - Live verification: inspect the commit contents/message and remote revision. Report whether skill discovery requires the next session/turn and disclose any validation not performed.
  - Evidence: record the commit and actual check outcomes before calling this task complete.
