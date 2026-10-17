# Tasks: GPU MLP training and independent reference parity

Implementation status: awaiting review choice and the go-ahead recorded in [proposal.md](proposal.md). Commands below are proposed implementation deliverables, not existing verified GPU trainer commands. Keep code/test/dependency/config implementation changes behind that authorization.

- [ ] **Review the concrete task and record authorization.** Complete the owner's selected personal or fresh independent review route, address findings and record the actual go-ahead/conditions. Keep ADR 0005 Proposed until supported by an actual owner decision.
  - Verification: inspect #11/#9, the three artifacts, CPU schema/numerical contracts and #10's tensor/stream/error boundary. A selected independent reviewer starts without inherited conversation and may inspect project material and primary references.
  - Evidence: pending. [Planning checks](verification.md) are not independent review or implementation approval.

- [ ] **Add explicit CUDA execution and versioned prepared artifacts.** Introduce device/memory-mode CLI selection and CUDA schema 2 while preserving CPU defaults/schema 1. Keep saved CPU-generated initialization/data/schedule and validate before device transfer. Isolate CUDA setup and record effective precision/runtime identity.
  - Live verification: actual installed CLI prepares a CPU bundle and a CUDA bundle with expected immutable fingerprints/layout/digests; CUDA selection is explicit. Reject unindexed/invalid/wrapped devices, conflicting settings, wrong schema/context/mode and unavailable CUDA before native admission; CPU base import/CPU trainer still work. Confirm the reference/worker use the same strict FP32/deterministic settings and actual GPU MLP forward/backward completion.
  - Evidence: pending; retain commands, installed paths, config/runtime identities and failure diagnostics.

- [ ] **Implement completed CUDA gradient averaging and ordered worker steps.** Build `CudaGradientSynchronizer` above the existing staging primitive, validate fixed parameters/gradients, retain workspace/storage and use two staged SUMs per training step before SGD.
  - Live verification: real installed one/two-worker known gradients demonstrate exact SUM-then-average, noncontiguous packing, original gradient-storage identity and unchanged sequence/storage on local rejection. Cover missing/replaced/wrong-device/dtype/layout/nonfinite/AD gradients, operation concurrency, default/nondefault-stream completion and post-return consumption. Reduced nonfinite data prevents copy-back/SGD; no public fault-injection switch is required.
  - Live verification: one-rank/three-step GPU training in both modes reaches sequence six and matches its independently computed GPU trajectory. No unsupported input or native/CUDA failure is coerced into a usable gradient.
  - Evidence: pending; record exact calls, sequences, owners, device/mode and errors.

- [ ] **Extend independent reference and strict CUDA verification.** Compute the combined-batch GPU trajectory independently, validate schema-2 completion/digests and compare all steps/rank pairs/final states/recomputed GPU evaluations at the fixed rule. Distinguish CUDA acceptance from CPU M1 full acceptance.
  - Live verification: each two-worker/100-step/local-eight profile (pinned and pageable) matches every comparison at `atol=1e-6, rtol=1e-5` and all evaluation losses strictly decrease; sequence reaches 200 per worker. Keep full per-step errors and first-failure diagnostics.
  - Live verification: corrupt copied tiny bundles with missing/stale/digest/metadata/shape/nonfinite/beyond-tolerance cases, updating hashes where needed to test numerical detection. Each produces failed verification; smoke cannot set CUDA full acceptance and GPU output cannot set CPU `full_acceptance`.
  - Evidence: pending; save raw reports, fixed tolerances, reference identities and input hashes.

- [ ] **Bound real training failure and owned-process cleanup.** Preserve original errors and stop the group without retry. Add an external Torch-free command/process-group controller that contains reference as well as worker initialization/driver waits.
  - Live verification: close a real peer after backward at step 1; both SGD-entry logs show only the completed step-0 update, failed native diagnostics remain useful, no successful completion is written and every owned worker is reaped. Standard-library timeout/later-spawn/interruption-boundary fixtures separately prove controller ownership and cleanup.
  - Live verification: direct-child `prepare`/`worker`/`verify` phases finish or fail within one 180-second profile deadline plus 10-second cleanup bound; record all PIDs/exits/reaping, primary/secondary errors and unknown GPU cleanup. A focused CUDA convenience-`run` smoke also checks its dispatch and nested worker ownership without mistaking parent exit for worker reaping. No deadline extension or retries to rescue failures.
  - Evidence: pending; keep original process/phase logs and distinguish real peer failure from policy injection.

- [ ] **Deliver reproducible GPU timing/memory evidence and CPU compatibility.** Add the installed sdist/wheel runner with the existing hash-locked CUDA runtime, source-before/after identity, disk-backed setup and fresh output. Keep raw per-rank timing/memory/device records and focused regressions.
  - Live verification: proposed `tools/check_cuda_trainer.py --mode full --output DIR [--wheelhouse DIR]` runs both one-rank smoke modes, both two-rank full modes, the nondefault-stream run and focused checks. GPU execution remains local/explicit; one failed or skipped required profile leaves acceptance open.
  - Live verification: all 100 steps remain numerical evidence; five timing warmups leave 380 measured rank-step/760 staged-call records across the two full modes. Validate counts, identities, finite nonnegative synchronized intervals and nested timing labels. Capture baseline/peak allocated/reserved memory, requested buffers, process RSS and actual shared UUIDs without a speedup or device-wide peak claim.
  - Live verification: reject nonempty/low-space outputs and wrong runtime/device explicitly. Run the unchanged #10 focused primitive check if its implementation changes, affected CPU trainer/binding checks for shared code changes, and exact implementation-revision CPU CI. Preserve the existing nightly/manual workflows and CPU/build dependency locks.
  - Evidence: pending; retain setup/import/build/lock/source/wheel/environment hashes, commands and actual GPU/CPU results.

- [ ] **Reconcile architecture, evidence and M2 delivery.** Update trainer/CUDA docs and navigation, reconcile the OpenSpec, accept ADR 0005 only from actual authorization and link raw evidence. Commit/push main without co-author trailers and preserve unrelated local configuration.
  - Verification: local links, complete contracts, two-line ordinary comment/docstring limit, exact tested-source/commit correspondence, remote SHA and all required live/CI evidence. Read back #11 acceptance and #9/milestone-2 child/evidence criteria before closing; keep missing hardware scopes explicit. Leave CPU/M1 evidence unchanged.
  - Evidence: pending. A planning commit is not trainer implementation or milestone completion.

## Completion

Only the planning artifacts exist for this task. No implementation task or hardware check is complete; #11, #9 and milestone 2 remain open. Record delivered revisions, actual results, limits and issue/milestone changes after authorized implementation and verification.
