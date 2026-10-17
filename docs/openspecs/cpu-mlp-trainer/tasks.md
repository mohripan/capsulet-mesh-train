# Tasks: Four-process CPU MLP and reference parity

Implementation status: awaiting the review choice and go-ahead in [proposal.md](proposal.md). Commands/files below are planned, not existing or passing checks.

- [ ] **Resolve review and implementation authorization.** Present the three specs and proposed ADR; address personal or fresh independent review and record the owner's actual go-ahead/conditions.
  - Verification: reconcile the review/disposition and presented scope with the actual user instruction. A reviewer verdict alone is not implementation authorization.
  - Evidence: pending owner response; no independent reviewer has been launched for #7.

- [ ] **Add the optional CPU gradient adapter and reproducible runtime.** Define stable parameter layout, retained FP32 staging/view ownership, strict gradient validation, native SUM, explicit averaging and copy-back; add optional dependency metadata/CPU hashes without changing base imports/native dependencies.
  - Live verification: install the actual wheel and CPU-locked Torch; check module paths/versions, Tensor/NumPy shared storage, distinct-gradient native averaging and correct parameter-gradient copy-back. Missing/sparse/wrong dtype/device/shape gradients fail explicitly before synchronization. Check layout mutation rejection and independent trace snapshots.
  - Live verification: the existing base binding environment still imports/works without Torch; no native/core dependency is introduced. Run the existing targeted binding checks if package boundary changes affect them.
  - Evidence: pending implementation/execution; record exact CPU wheel/transitive versions and results.

- [ ] **Prepare deterministic inputs, reference computation and artifact verification.** Save the model/data/teacher/seeds/layout/schedule, bind identity/fingerprints and define per-rank/reference numeric snapshots, final state and report formats.
  - Live verification: verify initial state/dataset digest consistency and the specified disjoint/wrapped batch indices. Run a tiny ordinary combined-batch reference; inspect gradients/parameters at the documented before/after-step boundaries. The verifier rejects missing ranks, mismatched identity/shape, nonfinite data and a perturbed numerical trace, and invalidates old success status.
  - Evidence: pending execution; preserve fixture hashes, failures and numerical metrics in the task verification report.

- [ ] **Implement the standalone worker and local group supervisor.** Execute both native collectives before SGD; support explicit rank/run/endpoints through prepared configuration, rank-tagged output/error capture and bounded owned-worker termination/reaping.
  - Live verification: actual two-worker/three-step training from the installed package matches the independent reference and rank peers with fixed tolerances, correct sequence and trace/final-state artifacts. Exercise the per-rank command separately from the local convenience launcher.
  - Live verification: one worker gets an invalid input configuration path while another waits for startup; check rank-tagged first/secondary errors, nonzero group result, bounded cleanup and no live owned children. Use short cleanup grace and an independent five-second watchdog for that focused case. Check setup/spawn exceptions and interruption cleanup using actual subprocess fixtures.
  - Evidence: pending execution; broad kill/stall/protocol matrices and four-container training remain #8.

- [ ] **Prove the full #7 outcome and wire focused CI.** Add the isolated verifier, CPU dependency cache/container stage and short trainer CI slot; document API/commands and reconcile the specs/ADR.
  - Live verification: `python tools/check_trainer.py --mode smoke --output build/trainer-smoke` and `docker build --target trainer-smoke-test -t meshtrain-trainer-smoke-test:dev .` exercise the actual installed package. Current native/binding jobs remain.
  - Live verification: `python tools/check_trainer.py --mode full --output build/trainer-full` runs four workers, local batch eight and 100 steps once for task acceptance. Require per-step losses/averaged gradients/updated parameters and rank-pair comparisons to pass `atol=1e-6, rtol=1e-5`, finite lower evaluation loss, and full-profile labeling. Save inputs, all metrics/final state, raw logs and environment; investigate failures without widening tolerances.
  - Verification: local links, short ordinary comments/docstrings, relevant formatting, final diff and accurate dependency/platform/CI claims. Record focused vs full acceptance vs deferred coverage; no nightly schedule is claimed.
  - Evidence: pending implementation/execution. Finalize ADR acceptance only against actual owner authorization.

- [ ] **Deliver and update issue navigation.** Commit/push authorized implementation to `main` without co-author trailers; confirm remote/CI, close #7 with reproducible evidence and update #2/#8/root navigation.
  - Verification: read back GitHub state and remote revision, confirm relevant CI and preserve unrelated local files. Keep #8/M1 open for remaining validation/deployment.
  - Evidence: pending implementation, checks and delivery.

## Completion

No training code, dependencies or CI changes have been implemented for #7; no trainer has run. Required failed/unrun checks remain unchecked. Record actual findings, command/environment/numerical/timing evidence, delivery references and limits as work proceeds after go-ahead.
