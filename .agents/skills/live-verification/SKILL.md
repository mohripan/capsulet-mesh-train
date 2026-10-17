---
name: live-verification
description: Select and execute focused checks of a repository task's changed behavior, record actual evidence in OpenSpec, and distinguish local proof from broad nightly integration or end-to-end coverage.
---

# Focused live verification

Read the task's acceptance criteria and authorized `tasks.md`. For each implementation step, identify the changed behavior and the smallest credible execution that would demonstrate its outcome. Run that check before marking the step complete.

## Choose the right proof

Use real task inputs and the actual relevant boundary. Examples for this repository:

| Change | Focused live verification |
| --- | --- |
| Native numerical behavior | Build the affected target and run the relevant numerical case, checking expected values/tolerance. |
| Transport/lifecycle | Run a small real worker exchange or the affected bounded-failure case; mocks alone do not establish process behavior. |
| Python binding | Build/import the actual extension and verify mutation, rejected inputs, storage lifetime, or Python progress as relevant. |
| Packaging/public C++ interface | Build and run a representative installed consumer when that boundary changed. |
| Skills/docs | Validate package/links as appropriate and walk through realistic actions or artifacts; no unrelated trainer/browser run. |

Keep checks proportional. Prefer an existing targeted command/case; add a focused check when the existing coverage cannot demonstrate the acceptance criterion. Respect required project checks. Do not create tests that merely mirror implementation or match the wording of instructions.

## Broad coverage

Reserve broad integration matrices, full training flows, and end-to-end suites for nightly coverage. A small live check can cross a real boundary even when its framework calls it an integration test. An explicit user request or a material unresolved risk may justify a relevant broader run on demand; state the reason and keep the scope clear.

Check the actual CI configuration before describing coverage as scheduled or completed. This skill does not create a nightly job. The repository's existing workflow may still run broader checks on push/PR until a separately authorized CI change moves them. Record that distinction rather than claiming a schedule that does not exist.

## Record and act on evidence

Under the corresponding task, record:

- Command/procedure, relevant environment/configuration, and expected observable result.
- Actual result, useful metrics/artifact, and pass/fail status.
- Any limitation: blocked, deferred, or not run, including unavailable hardware/tooling.

A required live check that failed or could not run leaves the task unchecked/unverified. Do not turn simulated output, a mock, or one shared GPU into evidence of actual multi-process or distinct-device behavior. Distinguish implementation completion from verified behavior.

Fix relevant failures and rerun affected checks. Do not silently widen tolerances, skip failing acceptance cases, or repeatedly run unrelated suites after sufficient checks pass. If a failure exposes a material design/scope change, update the OpenSpec and get direction on that change while preserving existing authorization for unaffected work.
