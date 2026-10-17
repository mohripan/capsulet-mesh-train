# Tasks: Explicit host-staged CUDA tensor adapter

Implementation status: awaiting the go-ahead recorded in [proposal.md](proposal.md). Proposed commands below do not exist yet. This planning task does not authorize code, test/config changes or installation.

- [ ] **Review the concrete OpenSpec and record authorization.** Follow the owner's selected personal or fresh independent review route, address findings, and record the actual implementation instruction/conditions. Keep ADR 0004 Proposed until accepted.
  - Verification: inspect the issue requirements and three artifacts against ADRs 0001–0003. A fresh reviewer must have no inherited conversation and may independently inspect the repository/primary sources. Review alone is not implementation authorization.
  - Evidence: read-only environment/source research is in [preflight.md](preflight.md); review choice and implementation go-ahead are pending.

- [ ] **Prepare a reproducible optional CUDA runtime and installed-package runner.** Add separate CUDA requirements/lock and `tools/check_cuda_adapter.py`, without changing CPU/build locks or making the core discover CUDA/Torch.
  - Live verification: resolve/hash the exact official `2.14.1+cu132` CPython 3.12 Linux x86-64 runtime and dependencies; build the real sdist-derived wheel and install in a fresh isolated environment. Record wheel/source/lock hashes, installed package/extension paths, Torch/NumPy/CUDA component/driver versions and actual GPU identity/capability/architectures. Execute a real CUDA allocation/copy/kernel/completion before accepting runtime availability.
  - Live verification: nonempty output is rejected, missing/incompatible GPU fails explicitly, and no CPU fallback is accepted. A test under the existing CPU-only runtime proves base import/CPU behavior and useful rejection of unavailable CUDA. Preserve logs; do not treat `nvidia-smi` as tensor execution proof.
  - Evidence: pending authorized implementation and actual GPU execution.

- [ ] **Implement strict fixed-workspace CUDA SUM staging and completion.** Add the optional module/timing value with explicit device/count/budget, pinned/pageable storage, retained sharing NumPy view, operation guard and the design's captured-stream waits.
  - Live verification: installed single-rank exact identity for counts 0, 1 and 1025 in both modes. Verify original GPU pointer/storage-offset boundaries, sharing host storage, empty participation/sequence, requested nonempty pin property and raw FP32 identity including special-value bits. Reject invalid argument/dtype/device/shape/layout/autograd/capture/budget without native sequence advance or input mutation.
  - Live verification: current nondefault-stream producer work completes without an external pre-call wait; an explicitly ordered producer/consumer pair gives the same result. After success, query completion and consume from another stream. Repeat workspace reuse and check the caller's device/stream context is restored. Never rely on an unordered-writer race as a test oracle.
  - Evidence: pending implementation and focused hardware checks.

- [ ] **Prove real two-process communication, lifetime and error behavior.** Launch fresh installed-code workers with explicit endpoints/run identity, one adapter each, using the available GPU as a documented shared device.
  - Live verification: both modes, counts 0/1/1025/1048576, repeated calls and native sequence inspection. Exact SUM/no averaging passes exactly; bounded-random results match the independent CPU reference at `atol=1e-6, rtol=1e-5`. The native TCP engine carries every payload; files are evidence after the run, not synchronization for gradients.
  - Live verification: delay a native peer after GPU warmup and demonstrate retained GPU/host owners plus Python progress while the binding waits; reject concurrent use of the same workspace without disrupting the active call. Close/stall a real peer to preserve `EngineError` category/context and unchanged GPU input when native fails before H2D. An independent 45-second controller bound includes worker initialization and teardown, with PID ownership/reaping evidence and no retries.
  - Live verification: explicitly labeled injected copy-failure policy check preserves the primary exception, poisons the adapter and reports secondary close/settling failures. Actual hardware fault/recovery is not claimed. Unknown CUDA cleanup remains unknown and fails acceptance.
  - Evidence: pending real worker/hardware execution; shared-GPU integration is not distinct-GPU scaling.

- [ ] **Record separate staging/native costs and compatibility regressions.** Implement the synchronized timing fields and evidence summaries without a performance pass threshold.
  - Live verification: five warmups then ten measured calls per 1025/1048576 count, pinned/pageable mode and one/two-worker configuration. Keep raw per-rank samples, byte counts, device/mode and stage medians/ranges. Check finite nonnegative times and containment of phase intervals in total; do not relabel binding/native wall time as wire latency or require pinning to win.
  - Live verification: run relevant existing native-only TCP-on/TCP-off consumers, Torch-free installed binding and short CPU trainer checks. CPU CI remains CPU CI; preserve the M1 nightly/manual workflow, and do not claim a GPU nightly runner exists. No #11 full trainer or benchmark baseline is delivered here.
  - Evidence: pending actual timing and regression execution.

- [ ] **Deliver the adapter contract, evidence and dependency handoff.** Reconcile the OpenSpec, accept ADR 0004 only under actual owner authorization, add `docs/CUDA.md`, and link the issue/ADR/verification records.
  - Verification: formatter/comment/docstring limits, local links, exact commands, numerical/timing/raw-error evidence, tested runtime identities and honest hardware limits. Commit/push to main without co-author trailers, confirm remote/exact-revision CPU checks and read back actual GPU proof before closing #10. Update #9/#11 dependency navigation while leaving M2 and trainer parity open.
  - Evidence: pending implementation authorization, all required checks and delivery. Planning can be committed separately without implying those gates passed.

## Completion

Planning only. All implementation tasks remain unchecked. #10 and #9 remain open; #11 still owns GPU training/reference parity. Read [preflight.md](preflight.md) for observed prerequisites and research limits.
