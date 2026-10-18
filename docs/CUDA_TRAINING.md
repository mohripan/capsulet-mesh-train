# GPU training through the C++ engine

Issue [#11](https://github.com/mohripan/capsulet-mesh-train/issues/11) adds an explicit CUDA mode to the existing MLP trainer. The reviewed/authorized [OpenSpec](openspecs/cuda-mlp-trainer/proposal.md), [independent review](openspecs/cuda-mlp-trainer/review.md) and [ADR 0005](adr/0005-cuda-training-reference-contract.md) define its contract. Installed full GPU acceptance passed in both memory modes, with independent 100-step reference parity; exact-revision CPU CI/delivery remain pending in the [evidence report](openspecs/cuda-mlp-trainer/verification.md).

The model, features, labels, forward/backward and SGD run on the selected GPU. Our [staging primitive](CUDA.md) copies gradients and loss through pinned or pageable host buffers to the existing C++ TCP SUM. Python explicitly divides both SUMs by worker count. No third-party collective carries training data.

## Install and run

Use Linux/WSL x86-64, CPython 3.12 and the separate `requirements/training-cuda.lock` for Torch `2.14.1+cu132`/NumPy `2.5.4`. The installed-wheel verification command below prepares this runtime; its `runtime/bin/python` can then run another fresh experiment:

```sh
/path/to/cuda-check/runtime/bin/python -I -m meshtrain.training run \
  --device cuda:0 --host-memory pinned --world-size 2 --steps 100 \
  --output /path/to/fresh-training-run
```

Select the logical index after `CUDA_VISIBLE_DEVICES` mapping. Both tiny workers share that device; each records its actual UUID. `--host-memory pageable` explicitly selects the other staging mode. CUDA absence, bad ordinals, conflicting precision controls and allocation failures raise without CPU fallback. Omit the device option for the unchanged [CPU profile](TRAINING.md); host-memory selection applies only to CUDA.

`prepare`, independent `worker` commands and `verify` use the same interface as the CPU trainer. Device/mode are immutable prepared configuration, not worker overrides. Independent deployments copy the complete prepared input bundle before training, supply explicit endpoints and collect each rank's output afterward. Files hold inputs/evidence; the C++ engine carries live gradients and loss. This task establishes local shared-GPU functionality, not GPU containers or physical-host deployment.

Each step completes gradient SUM/averaging and loss SUM/averaging before SGD. Parameters retain the M1 `16 -> 32 -> ReLU -> 4` architecture, saved CPU initialization/data, equal local batches and SGD settings. The library adapter is explicitly imported:

```python
from meshtrain.cuda_gradient import CudaGradientSynchronizer

synchronizer = CudaGradientSynchronizer(model, world_size, device="cuda:0", pin_memory=True)
loss.backward()
staged_sum_timing = synchronizer.synchronize(communicator)
# The application completes its loss checks before optimizer.step().
```

It packs stable named dense FP32 CUDA gradients, uses the blocking staged SUM, validates/reduces/averages, copies back into the original gradient storage and completes the selected stream. Noncontiguous dense gradients are explicitly packed. Its returned timing covers the inner staged SUM; the trainer separately measures complete gradient synchronization. Retain exclusive ownership and order producers onto the configured current stream. Unsupported metadata, autograd/inference/dual gradients, graph capture and overlapping synchronization are rejected.

Native/CUDA failure or reduced overflow prevents a usable result and aborts the job. Copy-back or optimizer failure can leave partial changes; there is no rollback, in-place retry or checkpoint recovery. CUDA execution failure poisons the wrapper, retains owners and records secondary close/stream-cleanup details on the original exception. Native deadlines do not bound driver initialization or GPU waits.

## Precision, reference and artifacts

CUDA CLI processes set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before framework import, reject conflicting values and require TF32 override variables to be unset or `0`. They select the new IEEE FP32 precision APIs consistently, strict deterministic algorithms and no cuDNN benchmarking. No AMP, gradient scaler, TF32 training or compilation is used. The synchronizer itself does not alter global application settings.

CUDA config/report schema 2 preserves the mathematical input/schedule contract and adds device/memory-mode/precision identity. CPU schema 1 and M1 acceptance retain their meaning. Every rank saves step losses, averaged gradients, updated parameters, complete final state and a digest-bound `cuda.json` with execution/device/memory/timing observations.

After workers exit, an independent ordinary GPU reference recomputes the rank-ordered combined-batch trajectory from saved inputs. It uses neither worker training values nor a communicator to generate expected values. All worker/reference and rank-pair comparisons use finite FP32 traces and the unchanged FP64-evaluated rule `abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)`. Final states, final trace snapshots and recomputed GPU evaluations receive separate checks. Deterministic settings do not guarantee bitwise equality across different batch shapes or environments; failed tolerances remain failed evidence.

Only the two-worker/100-step/local-eight CUDA profile can set `cuda_training_acceptance`, and it additionally requires strictly decreased finite fixed-set evaluation loss. GPU output keeps CPU `full_acceptance` false. Smaller runs are smoke/custom parity evidence. Closing #11 requires both memory modes and the focused verification set. Evidence files have no resume semantics; invalid/missing/stale/tampered artifacts fail verification, replacing previous success.

## Reproduce focused acceptance

```sh
build/toolchain-venv/bin/python tools/check_cuda_trainer.py --mode full \
  --output "$HOME/.cache/meshtrain-validation/cuda-training-check"
```

Use a fresh disk-backed Linux output directory with at least 12 GiB free; build/install temporary files stay on that filesystem. Add `--wheelhouse /path/to/cuda-wheels` to reuse collected dependencies; offline installation still verifies lock hashes. The tool builds an sdist-derived wheel, installs fresh, checks source/lock/wheel/runtime/import identity and runs installed code outside the checkout with `-I`.

The outer standard-library controller directly owns `prepare`, concurrent worker and `verify` children. Each profile has one 180-second execution budget and at most 10 seconds for TERM/KILL/reaping. The worker phase is capped at 120 seconds. Native startup/collective deadlines remain 30 seconds. Unknown cleanup fails acceptance. The convenience `run` command has its own worker timeout/grace; that worker timeout does not bound reference or driver work in the parent. Use the external runner for complete acceptance containment.

Checks cover gradient metadata/storage/averaging/concurrency, reduced overflow, default/nondefault streams, one-worker parity in both modes, two-worker full parity in both modes, and a real peer closure that prevents the second SGD update. Corruption fixtures test manifest and numerical detection, including early CLI failures replacing stale success. Separate process fixtures verify timeout, later-spawn failure and interruption registration/cleanup; they do not claim GPU execution.

## Cost and memory observations

Each rank/step retains gradient and scalar-loss staging records, complete gradient-sync and instrumented step wall time, and nested trace-copy time. Staging reports readiness, D2H, binding/native, H2D and total. Native time includes peer waits/TCP/CPU reduction; it is not isolated wire latency. Step time includes checks, synchronization, SGD and diagnostic copies. Nested intervals/medians are not additive.

The first five steps are timing warmup but remain real optimizer/numerical steps. The two full profiles retain 380 measured rank-step and 760 staged-call samples after warmup. Raw rank/step correlations, per-rank medians/ranges and maximum-rank step completion are retained without a speedup threshold.

Memory evidence distinguishes requested model/gradient/workspace/data bytes, per-process Torch allocated/reserved baselines/peaks, Linux RSS high-water marks and device-free observations. Each MLP worker requests 2,708 host-staging bytes; framework contexts/workspaces, allocator rounding, temporaries and trace storage add costs. Requested pinned bytes are not complete locked-page/RSS usage, and independent process peaks cannot be summed into a measured simultaneous device peak. GPU identity/visibility and actual runtime/component/build/driver versions accompany the results.

Two processes sharing one laptop GPU establish functional training. Distinct-GPU scaling/NCCL, physical hosts, GPU containers, native Windows and a GPU nightly runner require their own evidence. Existing hosted CI and scheduled/manual M1 validation remain CPU coverage.
