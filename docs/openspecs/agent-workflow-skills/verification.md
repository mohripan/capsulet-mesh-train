# Workflow skills verification

Date: 2026-10-17

Scope: four repository-local skills, their assets/metadata, navigation, and this OpenSpec. Runtime code and CI configuration were not changed. The user directed implementation after the specs were presented; no independent review was requested or performed.

## Package checks

The installed skill-creator validator accepted all four packages. Commands were run from the repository root, using an isolated environment with Python 3.14.7, PyYAML 6.0.3, and uv 0.12.3:

```powershell
$creator = 'C:\Users\mohri\.codex\skills\.system\skill-creator\scripts'
$names = 'openspec', 'coding-standards', 'live-verification', 'architecture-decision-records'
foreach ($name in $names) {
    uv run --isolated --no-project --with PyYAML==6.0.3 python "$creator/quick_validate.py" ".agents/skills/$name"
}
```

Each returned `Skill is valid!`. The same skill-creator metadata generator produced `agents/openai.yaml`; all four files parse, have valid short-description lengths, and include their own `$skill-name` in the default prompt. Implicit invocation remains enabled by default.

The host Python initially lacked PyYAML. The isolated uv environment resolved that validation dependency without modifying the project's runtime dependencies or the global Python environment.

## Actual scratch artifacts

An author-run walkthrough instantiated the three OpenSpec templates with concrete requirements and verification steps in a temporary directory outside the repository. All template fields were filled. Resuming updated the same folder and retained an explicit go-ahead instead of creating another task or review gate.

The coding example moved a three-line rationale comment into the generated OpenSpec design. The resulting header retained its original function signature and a two-line borrowed-storage contract. No existing project source was rewritten.

The ADR example used the next number after the actual project index, producing a scratch `0002-example-ownership.md` in Proposed status, with links between the record, scratch index, and task design. No project ADR was added or accepted by this exercise.

A deliberately broken local link was detected while the scratch task remained unchecked. After the artifact was corrected and links rechecked, the task was marked complete with actual evidence. This exercised an observable failure/correction path rather than only inspecting template headings.

Final link checks cover the new skill Markdown files, their references/templates, `AGENTS.md`, `README.md`, and this OpenSpec. Metadata and preserved-file checks also pass. The temporary fixture script/artifacts are local validation aids, not a permanent test framework or independent agent evaluation.

## Manual workflow cases

| Input/situation | Checked action or boundary |
| --- | --- |
| Fresh issue/change request | Draft all three artifacts first; inspect/research as needed; implementation waits for the task's go-ahead. |
| User chooses personal review | Present the files and wait; silence does not authorize work. |
| User chooses only sub-agent review | Review procedure uses a fresh context (`fork_turns="none"`), raw requirements, neutral criteria, and read-only scope; implementation still needs a go-ahead. No actual reviewer was launched in this task. |
| Reviewer reports no blockers | A verdict alone is not user authorization. |
| User says review and implement after blockers are resolved | Preserve that conditional authorization; proceed when satisfied without asking again. |
| User says “Just implement” after the specs are presented | Record the explicit go-ahead and implement. This task followed that path. |
| Resume an authorized task | Reuse its folder, authorization, and remaining checklist. |
| Native behavior change | Choose the relevant target/case and an actual changed-boundary check; this walkthrough selected the procedure but did not claim an unperformed native change/test. |
| Documentation/skills-only change | Use package, artifact, and link checks; do not launch unrelated training/end-to-end runs. |
| Required check fails or hardware is unavailable | Keep the task unverified and report the failure/block; a simulation or manual scenario is not real hardware evidence. The unavailable-hardware case was an instruction walkthrough, not a hardware test. |
| Routine local detail versus lasting ownership decision | Keep the routine rationale in OpenSpec; use a linked Proposed ADR for the durable decision, without inventing acceptance. |

These cases were author walkthroughs. They do not establish how an independently spawned model would behave; that evaluation was not requested after the user waived separate review.

## Provenance and preservation

The ADR adaptation uses inspected source/template guidance at upstream revision `044b4b21f083948e69ec878a7b07c8d0951a6ffa`. Its retained MIT license has Git blob hash `326f0a55c96e672fedf9d807ca043c00df05ba0e`, matching the upstream file exactly. The skill includes its source and adaptation notes.

That upstream license has one trailing space on its final line. It is retained verbatim; the staged whitespace check excludes only that exact license file, and its staged blob is checked against the upstream hash. Authored files pass the whitespace check.

The existing `find-skills/SKILL.md` and `skills-lock.json` remained unchanged during the walkthroughs; SHA-256 values recorded locally were `c00eeea0e13e74fe4a9d84ba0a8542205a1b736d65f13134fe1a6647eb14976f` and `0e7ecfa07bc10ff0a197d3837a7e546ebb14883f162521f0691a08fd26c3c245`, respectively. Those files and the existing `.claude/` links are outside the commit scope.

## Limits and delivery

Broad integration/end-to-end coverage is a skill policy, not a newly scheduled job. Existing CI still runs its native/container checks on push/PR. No nightly schedule, Python bindings, runtime change, or global skill installation is claimed here.

The files can be read directly now through `AGENTS.md`; automatic skill discovery may refresh on the next turn/session. Implementation commit `35d2c860c569f114e3955d1da965304a72084cfb` was pushed to `main`; `git ls-remote` confirmed the remote revision, and its contents/message were inspected for scope and the absence of co-author trailers. Task 7 records that completed delivery. This final documentation update does not alter the validated skills or CI configuration.
