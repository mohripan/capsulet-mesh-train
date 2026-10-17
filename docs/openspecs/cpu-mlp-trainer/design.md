# Design: Four-process CPU MLP and reference parity

Read [proposal.md](proposal.md) for scope/authorization and [tasks.md](tasks.md) for execution. Preserve the accepted [M1 design](../../superpowers/specs/2026-10-16-cpu-mvp-design.md), [native communicator ADR](../../adr/0001-blocking-centralized-fp32-communicator.md) and [Python ownership ADR](../../adr/0002-python-buffer-ownership.md). [ADR 0003](../../adr/0003-replicated-sgd-training-contract.md) records the durable adapter and training-evidence decisions.

## Architecture and dependency boundaries

Add `meshtrain.torch_adapter` for the optional dense CPU gradient adapter and `meshtrain.training` for the MLP example's input preparation, worker loop, local supervision, reference and verification. Split these responsibilities into focused modules. Keep both the C++ core and `import meshtrain` usable without PyTorch; importing the adapter/trainer explicitly requires the optional dependency. Use standard Python argument parsing, subprocesses and artifact formats, with no orchestration framework.

Add a `training` package extra constrained to PyTorch `>=2.14,<2.15`. Verification initially selects **2.14.1+cpu** on Linux x86-64 / CPython 3.12 from the official CPU wheel index; that wheel's existence/hash was inspected on 2026-10-17. Hash-lock the exact CPU wheel URL and transitive dependencies in separate `requirements/training-cpu.in` / `.lock`, reusing NumPy 2.5.4. A normal dependency range is not evidence that all supported versions/platforms work. Document how to select the CPU build explicitly; the optional metadata must not force PyTorch into the base package.

No C++ API, protocol, native numerical policy or binding conversion behavior needs to change. The native engine remains the only live gradient/loss communication path; Python performs model computation and explicit packing/normalization. Future GPU support, backward overlap and custom backends remain separate designs.

## Adapter contract

`GradientSynchronizer(model, world_size)` owns a stable layout and a contiguous CPU FP32 staging tensor. `synchronize(communicator)` packs current gradients, invokes native SUM through a detached NumPy view, divides by `world_size`, and copies the averages back under `torch.no_grad()`. The caller supplies the same positive integer world size used by its communicator; the example binds both values through the shared manifest/fingerprint. One adapter belongs to one worker and admits no concurrent calls.

Enumerate trainable parameters in sorted `named_parameters()` order. Record every name, shape, dtype, count and flat offset; the example has 676 trainable elements. Retain parameter identities and reject layout/device/dtype changes. Before each call validate all gradients are present, dense strided CPU FP32 with matching shapes and finite values. Reject unsupported input before issuing communication; do not silently substitute zero gradients or skip a parameter. The example has no frozen/tied parameters, stochastic layers or mutable model buffers; this adapter does not promise general DDP buffer/hook semantics.

Packing/unpacking is intentional: copy each detached gradient into the staging tensor, expose `staging.detach().numpy()` with its default `force=False`, retain the tensor/view throughout the call, perform SUM, reject nonfinite reduced values, explicitly divide in FP32, then copy slices into `.grad`. Do not use `numpy(force=True)`, implicit dtype/device conversion, storage rebinding, integer pointers, `.data` mutation, third-party collectives or autograd hooks. Noncontiguous dense gradients may be explicitly flattened/copied while packing; the buffer passed to C++ is always contiguous. No optimizer update or gradient copy-back follows a failed native call.

## Fixed workload and batch semantics

| Setting | Full acceptance value |
| --- | --- |
| Model | CPU FP32 `Linear(16,32,bias=True)`, ReLU, `Linear(32,4,bias=True)` |
| Randomness | Model initialization 1234; independent CPU feature/teacher generators 5678/5679 |
| Inputs | 1024 standard-normal rows of 16 features; standard-normal teacher `16 x 4`; labels `argmax(features @ teacher)` |
| Training | Four ranks, local batch 8, 100 optimizer steps, global batch 32 |
| Optimizer/loss | SGD lr 0.05, momentum/weight decay 0, `foreach=False`, `fused=False`; unweighted mean cross-entropy |
| Execution | One intra-op and inter-op CPU thread set at fresh-process startup; deterministic algorithms enabled; no AMP/scheduler/stochastic or batch-dependent layers |
| Evaluation | All 1024 fixed prepared examples, before/after training, under eval/no-grad |

Using the prepared dataset for evaluation is a functional loss-decrease check, not held-out/generalization evidence. Save the generated teacher, features and labels once instead of relying on each process regenerating equal inputs. Save the initial CPU state dictionary once; workers/reference load it exactly (`map_location="cpu"`, `weights_only=True`). Record generator use, framework/backend versions and thread/determinism settings. No cross-release/platform bitwise reproducibility is promised.

For zero-based step `s` and local offset `j`, rank `r` reads index `(s * (world_size*local_batch_size) + r*local_batch_size + j) % 1024`. Save the actual index table with shape `[steps, world_size, local_batch_size]`. Require positive counts and no within-step duplicated indices for the supported profiles; in particular the global batch cannot exceed the dataset. Across-step reuse/wrap is intentional. Workers see different slices within each step; the dataset is small and may be replicated for this demonstration.

Each worker executes: zero gradients (`set_to_none=True`), local mean-loss forward/backward, native gradient SUM then explicit averaging/copy-back, native one-element loss SUM then explicit averaging, and SGD step. Both collectives complete and their reduced values are checked finite before the update. Sequence is `2*s` for gradients and `2*s+1` for loss; after 100 steps the local next sequence is 200. Snapshot the averaged gradients before the update, the global loss before the update and parameters after it. Trace snapshots must own independent storage and cannot alias later-mutated buffers.

The reference runs its own ordinary forward/backward on the concatenation of each global batch in rank order and one SGD step per batch, from the saved initial state. It never calls a communicator, averages precomputed worker gradients or resets its trajectory to worker parameters. It may share the model/input definition and layout naming, while its numerical computation remains independent. It is run after workers complete, and workers never consume reference outputs.

## Inputs, identity and evidence artifacts

Prepare a fresh output directory, refusing to overwrite a nonempty run. Write versioned JSON configuration/layout, initial state, dataset/teacher/labels and batch-index artifacts. Record SHA-256 digests and canonical workload/layout fields. Derive the native context fingerprint from layout, initial/input/schedule digests, normalization/optimizer settings, world size and relevant framework execution settings. Each worker recomputes/validates these before joining; inconsistent inputs fail explicitly. A fresh UUID run identifier distinguishes each attempt from reusable workload identity.

Each rank writes only its own output directory: numeric loss/gradient/parameter traces (`.npz` without object arrays), final CPU state dictionary, finite evaluation metrics, environment/identity and diagnostics. Controller/reference outputs are separate. Missing, stale, mismatched-shape/count/identity or nonfinite traces fail verification. Collect traces and reports after the workers run; files are not a live gradient or parameter exchange path.

`DIR` in the commands below is the run root. Use `DIR/config.json` for common configuration, `DIR/inputs/` for prepared immutable artifacts, `DIR/ranks/<rank>/` for each worker's outputs/logs, `DIR/reference/` for the ordinary reference and `DIR/verification.json` for the comparison report. A worker always derives its own rank subdirectory from the run root; workers never write a shared trace file.

Write completed artifacts through temporary files and rename, with completion metadata only after the worker has produced its required files. A verifier does not accept an incomplete worker just because some files exist. Its report binds artifact identities/digests and is replaced for every verification attempt; invalidate an old verified status before rerunning. A failed launch/comparison never publishes a successful group result. These measures preserve trustworthy evidence; they do not implement checkpoint commit, resume, retries or artifact reuse across failed training attempts.

## Commands and local process supervision

Provide these installed-package commands through `python -m meshtrain.training`:

| Command | Contract |
| --- | --- |
| `prepare --output DIR [--world-size 4 --steps 100 --local-batch-size 8]` | Prepare the deterministic input bundle/configuration for independent workers; explicit endpoint/run settings are supported for later containers/hosts. |
| `worker --config CONFIG --rank R --output DIR` | Load/validate the prepared configuration, including world size/endpoints/run ID; create its own communicator and train one rank. No local launcher dependency. |
| `verify --output DIR` | Require all completed rank artifacts, compute the independent reference from prepared inputs, compare and write a report; return nonzero on missing/inconsistent inputs or numerical failure. |
| `run --output DIR [--world-size 4 --steps 100 --local-batch-size 8]` | Prepare a fresh local attempt, start independent workers, supervise, then run reference/verification and return the group result. |

Local mode chooses distinct loopback endpoints and a fresh attempt ID, closes its temporary port reservations before spawning and records the selected endpoints. A port-selection race can still cause bounded startup failure; do not silently change membership/configuration or reuse a failed communicator. Prepared remote configurations use explicit advertised endpoints/listen addresses; a local launcher is not a cluster scheduler.

Spawn fresh Python interpreters with argument lists and no shell or forked live communicator. Monitor every PID and drain both stdout/stderr continuously to rank-tagged files/console, including the last lines of workers that fail early. Preserve the first observed actionable error/exit status and report other failures as secondary; do not claim causal ordering from concurrent exits.

On any failed worker, invalid launch, supervision timeout, or user interruption, stop all remaining owned workers. Use a configurable worker-group wall timeout of 120 seconds for this tiny example, separate from native startup/collective timeouts (30 seconds each by default). Default termination grace is five seconds for the group, then force termination and bounded reaping; make grace configurable so focused tests can use a shorter value under an external five-second watchdog. Reap already-started workers if spawning a later rank fails. Successful exit requires all worker exit codes/artifacts and numerical verification, not just rank 0 finishing. M1 has no automatic restart/retry, elasticity or partial optimizer recovery.

## Verifier and reporting

Check each step's global loss, every averaged gradient and every post-update parameter against the independently evolving reference. Also compare every rank pair. Use FP64 error calculation on recorded FP32 values with the existing elementwise threshold `atol=1e-6`, `rtol=1e-5`; require all compared values finite. Report per-step/rank/quantity maximum absolute error and maximum relative error `abs(actual-reference)/max(abs(reference),1e-12)`, plus failure counts and first offending element/name/step. Relative-error reporting is not a separate threshold. Never widen tolerances or omit failed elements.

Check complete named parameter/state keys in final models in addition to flat traces so a packing omission cannot hide in a shared trace-layout helper. Compare initial/final evaluation losses against reference/ranks and require a finite strictly lower final value for every full-run model, without requiring monotonic minibatch losses.

A smoke profile uses two ranks, local batch eight and three steps with the same algorithms/model/seeds. It verifies finite values and parity but is labeled as a smoke result. Only the exact four-rank/100-step profile, full comparisons and evaluation decrease can report the issue's full trainer acceptance as passed. Support limited worker/step/batch overrides for focused tests; unsupported or incompatible configuration fails before launch.

## Rationale, risks and alternatives

- **Blocking adapter with explicit packing (selected):** makes ownership, averaging and optimizer ordering visible while exercising the existing native path. Hook-based bucketing/overlap changes ordering/lifetime and belongs to M4; DDP/backend integration belongs to M8.
- **Independent combined-batch reference (selected):** tests normalization and optimization across processes. Re-summing worker gradients as the only oracle would miss important training mistakes and does not meet #7.
- **Prepared immutable inputs and per-rank traces (selected):** establish exact initialization/schedule and allow offline numerical inspection. Reconstructing inputs only from seeds or saving final loss alone gives weaker evidence. Full recovery/checkpoint protocols remain M6.

CPU batch-size/reduction choices can produce small floating-point differences even with deterministic execution. Keep the accepted tolerances, investigate failures with raw traces, and record actual environment limits. Optional PyTorch dependencies can be large; isolate their cache/runtime from native/base-binding validation. The local launcher must not deadlock on a full output pipe or leave children alive when an exception interrupts setup. The artifact verifier must reject a missing rank, nonfinite data and a deliberately perturbed trace.

Keep ordinary new code comments/docstrings at most two physical lines; rationale belongs here and in the ADR. Routine implementation choices within the reviewed contracts do not add another approval gate.

## Focused verification and CI scope

Add `tools/check_trainer.py --mode smoke|full --output DIR` to build/install the actual wheel in an isolated Linux CPython 3.12 CPU runtime from the locks, check imported paths/versions, and retain evidence under the requested ignored build directory. Keep torch optional and verify base-package import without it using the existing binding environment. In CI add a separate `trainer-smoke-test` Docker target/matrix entry with CPU dependencies cached and hash-checked before offline runtime installation; retain current native/binding jobs.

Focused checks cover actual Torch/NumPy shared storage, packing/averaging/copy-back with distinct gradients, unsupported/missing-gradient rejection, input/schedule fingerprints, reference alignment, trace snapshot independence, tampered/missing/nonfinite artifact rejection, two-worker/three-step parity and launcher cleanup. A launcher fixture supplies an invalid configuration path to one actual worker command while another worker waits to join; require useful rank-tagged diagnostics, nonzero group exit and no surviving owned workers. Additional subprocess fixtures write more than pipe capacity to both output streams and exercise spawn/interruption cleanup. No public fault-injection option is needed.

Run one actual four-worker/100-step comparison before closing #7, capturing all per-step errors, final state/evaluation loss, raw logs and environment. This is the issue's targeted acceptance run; do not defer it to a future nightly job. Push/PR CI keeps the short smoke check. Broad worker-count/failure/training matrices, four-container training and nightly scheduling remain follow-up validation work; current broader native push/PR checks remain as configured.

## Primary references inspected on 2026-10-17

- [PyTorch 2.14 Tensor.numpy](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.numpy.html): the default CPU/detached conversion shares supported storage; forcing conversion can produce a copy.
- [SGD](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html) and [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html): explicit optimizer choices and mean reduction support the selected normalization.
- [Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) and [numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html): deterministic settings do not promise cross-environment or differently batched bitwise equality.
- [torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html), [official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/) and [CPython subprocess management](https://docs.python.org/3.12/library/subprocess.html) informed loading, packaging and owned-child cleanup. Availability/documentation inspection is not runtime compatibility evidence.


Implementation refinement: the installed CLI validates a worker configuration path before importing the optional Torch stack. A real WSL failure fixture showed that eager framework imports could mask the actionable missing-path error behind a short supervisor timeout. The external five-second cleanup watchdog is retained. Runtime dependency installation rewrites only the exact Torch URL reference to its pinned local wheel name after hash-checked download, so offline installation cannot silently fetch that direct URL again.
