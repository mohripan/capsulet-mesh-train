# Host-staged CUDA tensor SUM

The optional `meshtrain.cuda_adapter` module stages an ordinary CUDA FP32 tensor through host storage and calls our C++ TCP SUM engine. PyTorch handles copies and streams; the native library still builds without CUDA, PyTorch or Python. The CPU trainer remains unchanged. GPU trainer/reference parity belongs to #11.

Installed GPU acceptance and local CPU regressions passed for [#10](https://github.com/mohripan/capsulet-mesh-train/issues/10). Read the [OpenSpec](openspecs/host-staged-cuda-adapter/proposal.md), [review/disposition](openspecs/host-staged-cuda-adapter/review.md) and [ADR 0004](adr/0004-blocking-host-staged-cuda-adapter.md) for authorization and contracts. The [verification report](openspecs/host-staged-cuda-adapter/verification.md) records actual hardware results, raw samples and current delivery state.

## Using the adapter

Use an installed package with the supported CUDA runtime, and construct the existing native communicator with the same rank/run/endpoints contract on every worker:

```python
import torch
from meshtrain import Communicator
from meshtrain.cuda_adapter import CudaStagingAdapter

adapter = CudaStagingAdapter(1025, device="cuda:0", pin_memory=True)
tensor = torch.ones(1025, device="cuda:0", dtype=torch.float32)
with Communicator(config) as communicator:
    timing = adapter.all_reduce_sum_(tensor, communicator)
print(timing.d2h_ms, timing.native_ms, timing.h2d_ms)
```

`config` is a `meshtrain.RunConfig` as in the [Python API](PYTHON.md). The operation mutates the original GPU storage with SUM and returns a frozen timing record. Training performs its own division by world size. Importing `meshtrain` does not import Torch or initialize CUDA.

Choose a logical indexed device explicitly, accounting for `CUDA_VISIBLE_DEVICES`. The workspace has a fixed element count and a configurable `max_staging_bytes` budget, defaulting to 256 MiB. Its CPU tensor and sharing NumPy view are retained privately. `pin_memory=False` selects pageable storage; requested pinning failure raises, with no silent fallback.

Inputs must be ordinary one-dimensional contiguous strided CUDA FP32 tensors of that count/device, with no reverse/forward AD state, inference-tensor restriction or negative/conjugate view bit. Contiguous subviews are supported. Parameters, tensor subclasses, CPU/meta tensors, sparse/noncontiguous tensors, implicit casts/flattening and CUDA graph capture are rejected. Workspace construction inside inference mode is supported; inference input and dual tensors are rejected even when the caller's mode would otherwise hide the tangent.

## Streams, ownership and failure

Call on the configured device's current stream. Every producer and alias dependency must already be ordered onto it. When a different stream produced the input, use an explicit dependency before the call:

```python
consumer.wait_stream(producer)
with torch.cuda.stream(consumer):
    timing = adapter.all_reduce_sum_(tensor, communicator)
```

The adapter waits for ordered producers, completes D2H before C++ reads host storage, then completes H2D before returning. The result is ready for another stream after success. This first adapter provides a blocking result; it does not overlap communication with backward computation. Keep aliases/storage exclusive for the complete call and do not use the same workspace concurrently.

Local rejection preserves input and native sequence, but peers may already be waiting: the application must abort that job rather than assume rollback. Native errors retain `EngineError` context and native state policy; H2D is skipped if native SUM fails. A CUDA execution failure poisons the workspace, aborts peers by closing the communicator and records secondary cleanup information on the original exception. An H2D failure can occur after native sequence advanced and can leave partial device output. Stop the worker; do not retry or apply an optimizer update.

Native deadlines do not bound CUDA driver initialization or stream waits. If GPU completion cannot be certified, cleanup remains unknown and the worker must terminate through external supervision. The adapter retains failed-operation owners while alive. It does not reset the GPU or promise hardware recovery.

## Reproducing focused checks

Linux/WSL x86-64 CPython 3.12 is the initial target. Use the existing [locked build toolchain](DEVELOPMENT.md#optional-python-312-toolchain-check), GPU access through the existing driver, and a fresh output directory:

```sh
build/toolchain-venv/bin/python tools/check_cuda_adapter.py \
  --output "$HOME/.cache/meshtrain-validation/cuda-check"
```

A disk-backed Linux filesystem for the runtime/output avoids WSL cross-filesystem import overhead and memory-backed `/tmp` limits. Setup requires 12 GiB free there and places child build/install temporary files on that filesystem. The source checkout can remain on the Windows filesystem. The command builds an sdist and derived wheel, creates a new isolated environment, downloads/hash-checks the separate `requirements/training-cuda.lock`, installs offline from those wheels, and executes installed code outside the checkout with `-I`. To reuse an already collected wheelhouse, add `--wheelhouse /path/to/cuda-wheels`; offline installation still checks lock hashes. Nonempty output and unavailable/incompatible GPU execution fail explicitly.

The target runtime is Torch `2.14.1+cu132` and NumPy `2.5.4`. Actual component/driver/device versions, architecture support and executed kernel results must be recorded; the CUDA variant name alone is insufficient. Binary runtime packages do not add `nvcc` or CUDA linkage to our native build. CPU/build locks are independent and unchanged. No driver installation is part of this command.

Checks include single-rank identity/empty/subview and autograd rejection, both memory modes, nondefault streams, real one/two-worker TCP SUM, ownership/GIL/concurrency, native peer-close/stall and explicitly injected copy-error policy. The controller records phases, owned PIDs, exits and reaping; each case has an independent 45-second execution/cleanup bound. GPU fault injection is not used to corrupt the device context.

`validation.json`, per-case/rank JSON and logs, installed-runtime/device preflight, wheel/lock/source hashes, driver details and the complete worker sample records are retained. Five warmups precede ten measured calls for each 1025/1048576 element count, memory mode and one/two-worker configuration. `timing-samples.json` preserves measured rank/iteration samples, and `timing-summary.json` labels pooled per-rank medians/ranges.

| Timing | Synchronized wall boundary |
| --- | --- |
| `ready_wait_ms` | Completion of ordered input work. |
| `d2h_ms` | D2H submission through stream completion. |
| `native_ms` | Python binding/native call, including waits, TCP and CPU reduction. |
| `h2d_ms` | H2D submission through stream completion. |
| `total_ms` | Method entry through completed result, including validation/control overhead. |

These are staging-cost observations, without a required pinned-memory speedup or network-only latency claim. Two processes sharing the laptop GPU are single-GPU functional integration. Hosted CPU CI, native Windows, GPU-container pass-through, physical hosts, distinct GPUs, a GPU nightly runner and full GPU training are separate evidence scopes.
