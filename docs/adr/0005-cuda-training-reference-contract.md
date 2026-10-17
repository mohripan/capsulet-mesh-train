# ADR 0005: Explicit CUDA training profile and independent GPU reference

Date: 2026-10-18

Status: Proposed

OpenSpec: [GPU MLP trainer design](../openspecs/cuda-mlp-trainer/design.md)

Issue/source: [#11](https://github.com/mohripan/capsulet-mesh-train/issues/11), under [M2 #9](https://github.com/mohripan/capsulet-mesh-train/issues/9).

## Context

Accepted [ADR 0003](0003-replicated-sgd-training-contract.md) establishes CPU replicated SGD and independent combined-batch evidence. Accepted [ADR 0004](0004-blocking-host-staged-cuda-adapter.md) establishes completed CUDA tensor staging through native host-buffer SUM. The primitive's real hardware checks do not yet establish model/optimizer parity. Only one laptop GPU is available, so functional integration and distinct-GPU performance must stay separate.

## Decision and rationale

Extend the existing trainer with an explicit indexed CUDA mode and separate schema-2 execution/evidence contracts while preserving CPU defaults, schema-1 artifacts and CPU acceptance semantics. Keep an optional CUDA gradient wrapper above the existing staging primitive; the C++ engine remains dependency-free SUM, and Python explicitly averages gradients and loss before SGD.

Use saved CPU-generated initialization/data/schedule, strict eager FP32/deterministic settings and an ordinary GPU combined-batch reference computed after worker exit. The reference never uses worker training values as its expected trajectory. Compare all step losses/averaged gradients/post-SGD parameters, rank pairs, final state and recomputed GPU evaluation at the existing `1e-6` absolute plus `1e-5` relative rule. A mismatch remains failed evidence, without silent tolerance changes.

Give GPU reports a separate CUDA acceptance profile; they cannot satisfy CPU M1 `full_acceptance`. Retain every operation's staging/native timing and explicit memory/runtime/device observations. Require actual two-process shared-GPU training in pinned and pageable modes if it fits; failure/OOM is not an automatic skip or scaling claim. Bound the complete installed run with external process-group supervision, including reference/GPU initialization outside native deadlines. Keep original errors and no-update-after-failure evidence; do not retry a failed group in place.

## Alternatives considered

- A separate CUDA trainer would isolate M1 files but duplicate saved-input, reference, integrity and process-lifecycle logic.
- An abstract training-backend framework could serve future devices and overlap but adds interfaces beyond the concrete second path.
- CPU reference arithmetic or worker-derived expected gradients would weaken the required equivalent-GPU/independent-trajectory proof.
- DDP/NCCL training would bypass our communication implementation; distinct-GPU library baselines remain later work.

## Consequences

Shared code needs focused CPU regressions and explicit schema dispatch. GPU contexts/library workspaces and trace copies can dominate the tiny model; measured allocator bytes and instrumented step time must not be advertised as complete device memory or multi-GPU speed. Full FP32 and completed waits favor a clear correctness contract over overlap. CPU-only native builds and ordinary imports remain independent of CUDA.

## Revisit triggers

Uneven batches, AMP/other dtypes, tied/stateful models, asynchronous buckets, multiple devices per process, Adam/sharding, checkpoint recovery or a larger training example warrant a new focused contract. Preserve accepted CPU/tensor history unless a later decision explicitly supersedes it.

## Decision record

Drafted during #11 planning after the owner's 2026-10-18 request to continue. Review route and implementation go-ahead for this presented scope are pending in the [proposal](../openspecs/cuda-mlp-trainer/proposal.md). No reviewer verdict or earlier #10 authorization accepts this new record.
