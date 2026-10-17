# Proposal: Four-process CPU MLP and reference parity

Date: 2026-10-17

Status: draft for review; implementation has not started.

Source: [issue #7](https://github.com/mohripan/capsulet-mesh-train/issues/7), following the owner's request to continue after #6.

## Problem and outcome

The engine and installed Python package can sum FP32 arrays across real processes. The next deliverable is actual distributed training: four independent workers train on different minibatches, synchronize gradients through our C++ engine, and match an ordinary single-process combined-batch reference. Save enough inputs, traces and final parameters to reproduce and inspect the result.

The accepted [M1 specification](../../superpowers/specs/2026-10-16-cpu-mvp-design.md) fixes the numerical workload. This task makes its adapter, launcher, artifact and verification behavior concrete; [design.md](design.md) records those contracts and [tasks.md](tasks.md) the required live checks.

## Scope

- Add an optional PyTorch gradient adapter with explicit flat-buffer packing, native SUM, division by worker count and gradient copy-back. Keep the base package/core independent of PyTorch.
- Add deterministic input preparation, an independent per-rank worker command, local group supervision, an ordinary combined-batch reference and an artifact verifier.
- Preserve rank-tagged stdout/stderr, failure diagnostics and bounded peer cleanup. Provide the same worker command for later container/host launches.
- Add focused installed-wheel verification and a small CPU trainer CI target, dependency hashes, documentation and [proposed ADR 0003](../../adr/0003-replicated-sgd-training-contract.md).

No transport/collective algorithm change, DDP/FSDP/custom backend, CUDA, transformer, overlap, sharding, checkpoint recovery, cluster scheduling, workflow DAG, benchmark baseline, public package publication or native Windows support is included. Four-container trainer deployment and the broad fault/clean-build matrix remain #8. Nightly scheduling remains separate; this task does not claim an existing nightly job.

## Acceptance

1. Train the specified CPU FP32 `16 -> 32 -> ReLU -> 4` MLP for 100 SGD steps with four workers, local batch eight, learning rate 0.05, mean cross-entropy, no momentum/weight decay/scheduler/mixed precision, one CPU compute thread, and the specified seeds/data formula.
2. Workers and reference load exactly the same saved initialization and recorded input bundle. Each step's global batch is the rank-ordered union of disjoint local slices. The reference computes one ordinary global-batch forward/backward/SGD update independently, without a communicator or worker traces as numerical input.
3. All gradient/loss exchange uses our C++ SUM through the installed binding. Averaging occurs explicitly in Python. Collectives complete successfully before the optimizer step; missing/unsupported gradients or communication failures abort the worker/group.
4. At every step compare each rank's global mean loss, averaged gradients and updated parameters with the reference and every other rank using `abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)`. Save maximum absolute/relative errors and failures; do not widen tolerances. Verify finite, lower final loss on the same fixed evaluation inputs.
5. A fresh-process local launcher assigns explicit identity/endpoints, tags both output streams, returns nonzero for any worker failure and terminates/reaps its remaining workers. The per-rank command works independently from the launcher with prepared inputs and explicit configuration.
6. Save initialization, data/teacher/labels, layout, batch indices, workload/environment, per-rank/reference traces, final model parameters and a verification report. Missing/inconsistent/nonfinite artifacts or an altered numerical trace cannot produce a successful report. These are evidence artifacts, not restartable checkpoints.
7. Prove changed behavior through focused installed-package checks, a two-worker short CI comparison and one full four-worker/100-step acceptance run before closing #7. Keep #8 and M1 open for the remaining validation/deployment work.

## Review and authorization

- Review route: awaiting personal review or the owner's selection of a fresh independent sub-agent review.
- Findings/disposition: author self-review only; no independent review has run for #7.
- User go-ahead: pending for these presented #7 artifacts. The previous independent-review-then-implementation instruction applied to #6; the request to continue authorizes preparing this task's specs.
- Commit/push: follows the owner's standing instruction for completed planning/authorized implementation, to `main` without a co-author trailer.

## Delivery

Draft planning artifacts and proposed ADR only. #7 remains open, implementation tasks are unchecked, and no PyTorch package has been installed or trainer executed for this task. Documentation/API research establishes a proposed approach, not numerical compatibility evidence.

Planning verification on 2026-10-17: local links resolve across the three task artifacts, proposed ADR and ADR index; no unresolved template fields or TODO/TBD markers remain. Author self-review reconciled gradient/loss step boundaries, independent reference computation, rank output ownership, cleanup/watchdog limits and smoke/full acceptance labeling against M1. Git whitespace checks apply before the planning commit. This is document validation, not trainer execution or independent review.
