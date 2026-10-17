# Independent review: M1 validation OpenSpec

Date: 2026-10-17

Reviewer: `review_m1_validation_spec`, launched with `fork_turns="none"` at the owner's explicit request. It received the raw request/issue #8 acceptance, repository path, exact three artifact paths and the accepted M1 specification identifier. It did not inherit the conversation, author rationale or a requested verdict. It independently inspected repository instructions, contracts/ADRs, native implementation/tests, packaging and trainer verification.

The review and follow-up were read-only: no edits, installations, builds, fault cases, container validation, commits or issue mutations by the reviewer. The author made the revisions below. Review does not grant implementation authorization.

## Initial findings and disposition

| Severity | Finding | Disposition |
| --- | --- | --- |
| Medium, required correction | Sampling only a long-lived controller's descriptors while failed workers exit can miss native worker leaks: process exit closes their descriptors regardless of destructor correctness. | Accepted. Proposal acceptance 4, the design's timing/resource section and the native-fixture task now require each healthy worker to compare its own baseline after native failure and object destruction while still alive. Stable fixture descriptors, mandatory cleanup records and separate controller/target OS cleanup evidence are explicit. |
| Low, suggestion | A supposedly standard-library host runner used an existing toolchain-venv command without clearly locating the CPython 3.12/build-package-dependent full trainer check. | Accepted. The primary command now uses host `python3` (3.12+) with Docker and no host venv/Torch/build packages. The full localhost check runs inside the pinned validation image, where the required runtime/build tools exist, and its evidence is copied out afterwards. |

Author consistency review also distinguished the eight standalone five-second CTest cases from a 20-iteration, long-lived repeat controller: a separate watchdog enforces each iteration's five-second limit, and 110 seconds is only batch containment. Docker failure cleanup now explicitly includes best-effort diagnostics and reserves time for collective stop/kill/removal within one 15-second budget.

## Follow-up review

The same independent reviewer re-read all three revised artifacts and reported: **“No remaining blockers or new material inconsistencies found.”** It confirmed resolution of worker descriptor evidence and runner prerequisites, consistency between per-iteration and batch watchdogs, and the separate Docker execution/cleanup timing domains.

Coverage included the eight fault scenarios, original absolute deadlines, native failure before fallback termination, terminal collective state, installed-wheel isolation, independent post-run reference, bounded resource ownership, fixed numerical tolerances, nightly/manual scope and preservation of existing push/PR coverage.

Limits: this verified the written plan, not runtime feasibility or #8 acceptance. At review delivery the implementation tasks were unchecked and implementation go-ahead was pending. The owner subsequently authorized implementation; see [proposal.md](proposal.md) for the actual authorization and [verification.md](verification.md) for runtime evidence. This review does not become an implementation review or acceptance proof.

## Planning verification

The three artifacts contain no unresolved template fields or TODO/TBD markers; their local links resolve. The revised acceptance, design and tasks agree on worker/controller descriptors, timing, host/container prerequisites and deployment labels. No new production architecture decision is proposed, so existing ADRs 0001–0003 remain sufficient. Git whitespace and final artifact/navigation links are checked before the planning commit. No #8 code, dependency, Docker or workflow changes were made.
