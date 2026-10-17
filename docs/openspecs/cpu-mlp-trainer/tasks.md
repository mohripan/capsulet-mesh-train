# Tasks: Four-process CPU MLP and reference parity

Implementation status: authorized and in progress; see [proposal.md](proposal.md). Required verification remains unchecked until actual execution passes.

- [x] **Resolve review and implementation authorization.** Present the three specs and proposed ADR; address personal or fresh independent review and record the owner's actual go-ahead/conditions.
  - Verification: reconcile the review/disposition and presented scope with the actual user instruction. A reviewer verdict alone is not implementation authorization.
  - Evidence: owner said “Okay. You can continue!” for the presented #7 scope on 2026-10-17. Fresh independent reviewer `review_cpu_trainer_spec` received no conversation context and reported no blockers. Accepted its finite-SUM overflow check suggestion; implementation proceeded after disposition.

- [x] **Add the optional CPU gradient adapter and reproducible runtime.** Define stable parameter layout, retained FP32 staging/view ownership, strict gradient validation, native SUM, explicit averaging and copy-back; add optional dependency metadata/CPU hashes without changing base imports/native dependencies.
  - Live verification: install the actual wheel and CPU-locked Torch; check module paths/versions, Tensor/NumPy shared storage, distinct-gradient native averaging and correct parameter-gradient copy-back. Missing/sparse/wrong dtype/device/shape gradients fail explicitly before synchronization. Check layout mutation rejection and independent trace snapshots.
  - Live verification: the existing base binding environment still imports/works without Torch; no native/core dependency is introduced. Run the existing targeted binding checks if package boundary changes affect them.
  - Evidence: installed Torch 2.14.1+cpu / NumPy 2.5.4; real distinct-gradient average exactly 1.5 and overflow rejection pass in WSL/container. All ten Torch-free binding checks pass. See [verification.md](verification.md) and the CPU lock.

- [x] **Prepare deterministic inputs, reference computation and artifact verification.** Save the model/data/teacher/seeds/layout/schedule, bind identity/fingerprints and define per-rank/reference numeric snapshots, final state and report formats.
  - Live verification: verify initial state/dataset digest consistency and the specified disjoint/wrapped batch indices. Run a tiny ordinary combined-batch reference; inspect gradients/parameters at the documented before/after-step boundaries. The verifier rejects missing ranks, mismatched identity/shape, nonfinite data and a perturbed numerical trace, and invalidates old success status.
  - Evidence: generator/schedule/identity, independent reference, snapshot and seven artifact-corruption variants pass. Full prepared hashes and all raw comparison records are in [verification.md](verification.md) and `results/`.

- [x] **Implement the standalone worker and local group supervisor.** Execute both native collectives before SGD; support explicit rank/run/endpoints through prepared configuration, rank-tagged output/error capture and bounded owned-worker termination/reaping.
  - Live verification: actual two-worker/three-step training from the installed package matches the independent reference and rank peers with fixed tolerances, correct sequence and trace/final-state artifacts. Exercise the per-rank command separately from the local convenience launcher.
  - Live verification: one worker gets an invalid input configuration path while another waits for startup; check rank-tagged first/secondary errors, nonzero group result, bounded cleanup and no live owned children. Use short cleanup grace and an independent five-second watchdog for that focused case. Check setup/spawn exceptions and interruption cleanup using actual subprocess fixtures.
  - Evidence: standalone installed worker commands pass two-worker/three-step parity. Missing-config/dual-pipe/spawn/interruption fixtures pass independent five-second watchdogs and child reaping in WSL/container; initial eager-import timeout fixed. See [verification.md](verification.md). Broad matrices/four-container training remain #8.

- [x] **Prove the full #7 outcome and wire focused CI.** Add the isolated verifier, CPU dependency cache/container stage and short trainer CI slot; document API/commands and reconcile the specs/ADR.
  - Live verification: `python tools/check_trainer.py --mode smoke --output build/trainer-smoke` and `docker build --target trainer-smoke-test -t meshtrain-trainer-smoke-test:dev .` exercise the actual installed package. Current native/binding jobs remain.
  - Live verification: `python tools/check_trainer.py --mode full --output build/trainer-full` runs four workers, local batch eight and 100 steps once for task acceptance. Require per-step losses/averaged gradients/updated parameters and rank-pair comparisons to pass `atol=1e-6, rtol=1e-5`, finite lower evaluation loss, and full-profile labeling. Save inputs, all metrics/final state, raw logs and environment; investigate failures without widening tolerances.
  - Verification: local links, short ordinary comments/docstrings, relevant formatting, final diff and accurate dependency/platform/CI claims. Record focused vs full acceptance vs deferred coverage; no nightly schedule is claimed.
  - Evidence: container smoke passed 63 records; full four-worker/100-step tool passed 3,090 records with zero failures and evaluation 1.41192746 -> 0.97489858. Seven focused checks pass (16.463 s container; 138.887 s WSL). CI target added; ADR accepted from owner authorization. See [verification.md](verification.md).

- [ ] **Deliver and update issue navigation.** Commit/push authorized implementation to `main` without co-author trailers; confirm remote/CI, close #7 with reproducible evidence and update #2/#8/root navigation.
  - Verification: read back GitHub state and remote revision, confirm relevant CI and preserve unrelated local files. Keep #8/M1 open for remaining validation/deployment.
  - Evidence: pending implementation, checks and delivery.

## Completion

Implementation and required local checks are complete; delivery/remote CI remains pending. See [verification.md](verification.md) for exact commands, numerical/timing evidence, the initial failure and correction, raw artifacts and limits. #8 and M1 remain open.
