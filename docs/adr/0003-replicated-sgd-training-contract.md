# ADR 0003: Replicated SGD adapter and independent training evidence

Date: 2026-10-17

Status: Accepted — owner authorized the presented #7 scope on 2026-10-17; independent review completed without blockers. See the [authorization and disposition](../openspecs/cpu-mlp-trainer/proposal.md).

## Context

The native engine implements blocking SUM; its Python binding borrows strict NumPy storage. A real trainer must make loss/gradient normalization and optimizer order explicit, preserve the standalone dependency boundary, and demonstrate parity with an independently computed combined batch. An import test or decreasing distributed loss alone does not establish correct training.

Read the [task design](../openspecs/cpu-mlp-trainer/design.md), [M1 contract](../superpowers/specs/2026-10-16-cpu-mvp-design.md), [ADR 0001](0001-blocking-centralized-fp32-communicator.md) and [ADR 0002](0002-python-buffer-ownership.md).

## Decision

Keep PyTorch optional and its adapter/example in Python, depending on the existing core/binding. Pack dense CPU FP32 gradients in stable named-parameter order into retained contiguous staging storage, expose its detached sharing NumPy view, call our native SUM, explicitly divide by configured worker count, and copy the averages back. Validate the fixed layout/gradient contract before communication and reject nonfinite reduced values before copy-back/SGD. The adapter is synchronous and one per worker; it does not promise DDP hook/buffer semantics.

Each MLP step completes native gradient SUM and loss SUM, with explicit averaging, before SGD. Equal local batches make the mean of local means match the global objective. The independent reference starts from the same saved initialization and computes its own ordinary rank-ordered combined-batch forward/backward/SGD trajectory. It never consumes worker gradients or parameters to generate its expected result.

Prepare immutable versioned input/layout/schedule artifacts and digest them into native application identity. Save every rank's and reference's per-step losses, averaged gradients and updated parameters, complete final state and fixed-set evaluation losses. Compare all elements/rank pairs using the accepted tolerances and publish raw errors. A successful report requires complete finite matching artifacts and the requested profile; the short CI smoke cannot claim full four-worker/100-step acceptance.

Use a fresh-process local supervisor and an independent worker command with explicit configuration. Fail the group, terminate/reap owned workers and preserve rank diagnostics on any failure; do not automatically restart or use incomplete output as a checkpoint. Evidence files are read after training and never replace live C++ communication. Container orchestration, full fault matrices and recoverable checkpoint transactions remain later work.

## Alternatives considered

- DDP or an existing collective: would bypass the engine under test and violate the custom training-path boundary. Framework/backend adapters remain M8.
- Backward hooks/asynchronous buckets now: introduce ordering and lifetime contracts beyond the first blocking trainer; overlap belongs to M4.
- Use a sum of worker gradients as the only reference: can miss normalization/optimizer mistakes and does not establish ordinary combined-batch parity.
- Save only final loss/weights or regenerate inputs solely from seeds: makes per-step divergence and exact initialization/data alignment harder to inspect.
- Retry failed ranks or resume partial artifacts in place: cannot restore consistent collective/optimizer state; group checkpoint recovery belongs to M6.

## Consequences and revisit triggers

Explicit staging adds copies and per-step traces add small demonstration overhead. These costs are intentional correctness instrumentation, not a performance claim. Float32 reduction order and batch execution can produce small differences; the tolerance contract stays fixed, and environment/version limits must be recorded. Evaluation on the prepared dataset demonstrates functional progress, not generalization.

Revisit for uneven batches, unused/sparse gradients, stateful/tied models, additional dtypes/devices, overlap, large trace volumes, sharded optimizers or recovery. Preserve native SUM ownership and distinguish smoke, full trainer, container, physical-host and GPU evidence.
