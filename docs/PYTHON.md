# Python CPU bindings

The optional `meshtrain` package exposes our C++ centralized FP32 SUM engine over Linux TCP. The initial packaging target is Linux CPython 3.12 with NumPy 2.x. Its base installation does not depend on PyTorch. See [CPU training](TRAINING.md) for the optional gradient adapter, worker/launcher and independent reference verifier.

## Build and verify

Run in Linux/WSL from the repository root:

```sh
uv venv --python 3.12 build/toolchain-venv
uv pip sync --require-hashes --python build/toolchain-venv/bin/python requirements/toolchain.lock
build/toolchain-venv/bin/python tools/check_python_bindings.py
```

The check builds an sdist and then its wheel, creates a fresh runtime environment outside the checkout, installs the wheel and hash-locked NumPy, and executes the actual package. It checks configuration, strict array borrowing, storage aliasing/lifetime, Python thread progress, two-process TCP sums, close and bounded failures. The disposable runtime is removed afterwards. Build dependencies are pinned in `requirements/toolchain.lock`; `requirements/python-runtime.lock` pins the NumPy release used for these checks.

For the reproducible container path, run `docker build --target python-bindings-test -t meshtrain-python-bindings-test:dev .`. Docker first prepares the hash-checked NumPy wheel in a cached dependency layer, then installs it offline during verification. CI runs that stage alongside the existing no-Python native target. The earlier `python-toolchain-test` stage remains a manual packaging-toolchain probe. Broad failure/trainer deployment checks use the separate [M1 nightly/manual validation lane](VALIDATION.md).

To retain a wheel and install it into your own environment:

```sh
build/toolchain-venv/bin/python -m build --no-isolation
uv venv --python 3.12 build/python-runtime
uv pip sync --require-hashes --python build/python-runtime/bin/python requirements/python-runtime.lock
uv pip install --no-deps --python build/python-runtime/bin/python dist/meshtrain-0.1.0-cp312-cp312-linux_x86_64.whl
```

The filename above is the initial Linux x86-64 artifact. These are local platform wheels, without a manylinux portability claim or public package publication. Standalone C++ builds still default to `MESHTRAIN_BUILD_PYTHON=OFF`, discover no Python/NumPy, and retain their existing install/export. Manual extension builds opt in with `-DMESHTRAIN_BUILD_PYTHON=ON` and provide the pinned pybind11/CPython development environment. Wheel installation selects only the `python` CMake component, so it does not bundle the C++ SDK.

## Two worker calls

Save this as `your_worker.py`, then invoke it with rank `0` and rank `1` in separate Linux terminals using the installed runtime. Start them within the configured startup timeout. Each rank uses the same ordered endpoints, run identifier and application fingerprint; choose a fresh run identifier for each attempt and ensure the ports are available.

```python
import sys
import numpy as np
from meshtrain import Communicator, Endpoint, RunConfig

rank = int(sys.argv[1])
config = RunConfig(
    rank=rank,
    world_size=2,
    run_id="example-attempt-1",
    context_fingerprint="example-three-elements-v1",
    endpoints=[Endpoint("127.0.0.1", 29500), Endpoint("127.0.0.1", 29501)],
)
values = np.full(3, rank + 1, dtype=np.float32)
with Communicator(config) as communicator:
    communicator.all_reduce_sum(values)
    print(rank, values, communicator.next_sequence)
```

Both ranks receive `[3., 3., 3.]` and sequence `1`. The call returns `None` and mutates `values`; it performs SUM without division. Supply equal counts and issue collectives in the same order on every rank. Empty arrays still participate. World size one preserves the input unchanged while advancing sequence. Endpoints may be routable IPv4 addresses/hostnames; local process/container checks do not establish physical multi-host behavior.

## API and borrowed storage

`Endpoint(host, port)` and keyword-only `RunConfig(...)` expose read-only fields. Configuration copies its endpoint list, and reading `config.endpoints` returns a separate list of endpoint values. Timeouts use integer milliseconds, defaulting to 30000 each; `max_payload_bytes` defaults to 268435456. Integer fields reject booleans and out-of-range values. Native semantic configuration failures raise `EngineError` before sockets open.

`Communicator(config)` blocks until the fixed group joins. `state` reports `ready`, `failed` or `closed`; `next_sequence` starts at zero and advances on local success. `close()` is idempotent and can unblock a call from another Python thread. Context-manager exit closes without suppressing an exception. Each worker constructs its own communicator after process startup; do not fork a live communicator.

`all_reduce_sum(array)` accepts actual writable one-dimensional C-contiguous native-endian float32 NumPy arrays, including valid views/subclasses. Nonempty storage must be aligned for C++ `float`. Lists, memoryviews, tensors, integer pointers, dtype/byte-order conversion, noncontiguous layouts, read-only, scalar and multidimensional arrays are rejected. Empty/singleton contiguity follows NumPy semantics, including their permitted degenerate strides.

The binding never coerces an array or copies a result back from a temporary. It retains the array, its writable export/supporting owner chain, and native communicator for the whole call. It validates/acquires/releases Python-owned objects with the GIL held and releases the GIL around native startup, communication and close. Other Python threads can run while communication waits. The caller must prevent alias reads/writes, resizing, metadata changes and external storage invalidation until the call ends; retention does not make shared-array access thread-safe. See [ADR 0002](adr/0002-python-buffer-ownership.md) and the [OpenSpec rationale](openspecs/python-buffer-bindings/design.md).

## Errors and aborts

Wrong argument kind/dtype raises `TypeError`; unsuitable layout/writability/alignment/representability raises `ValueError`. Buffer acquisition errors retain the exporter's Python exception. These preflight failures leave local sequence/state unchanged and do not start native communication. Other ranks may already be waiting: close/abort the group rather than treating this as a group rollback. Independent peers remain bounded by their native deadlines.

Once admitted, native failures, including the configured payload limit, make the communicator unusable and may leave the buffer partially changed. Do not use that buffer for an optimizer step or retry the collective on the failed communicator. Recreate the complete group with a fresh attempt identifier. An overlapping call raises `invalid_state` without cancelling the first; explicit close preserves `closed` state even if it races with failure. Cancellation can report `peer_closed` or `invalid_state` according to the native interruption point.

`EngineError` is a `RuntimeError` with `code`, `rank`, `run_id`, `operation`, `sequence`, `peer` and `peer_endpoint` attributes. Optional integer fields are `None` when absent. The string includes the contextual native diagnostic; `operation` can name a phase such as `all_reduce_sum.receive_header`. Read these fields rather than parsing diagnostic text. Allocation failures retain Python's normal `MemoryError` mapping. Native deadlines bound communication, while Python scheduling/GIL reacquisition add elapsed time outside that guarantee.

The explicitly imported [CUDA staging adapter](CUDA.md) accepts suitable CUDA tensors through a separate blocking contract and runtime. The optional [CUDA gradient/training layer](CUDA_TRAINING.md) composes it with model packing and explicit averaging. The base package still imports without Torch, and this NumPy API remains a host-buffer boundary.
