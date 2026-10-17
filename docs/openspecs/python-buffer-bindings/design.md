# Design: Safe Python CPU buffer bindings

Read [proposal.md](proposal.md) for scope/authorization and [tasks.md](tasks.md) for execution progress. This extends the accepted [M1 Python contract](../../superpowers/specs/2026-10-16-cpu-mvp-design.md#35-python-boundary) and preserves [ADR 0001](../../adr/0001-blocking-centralized-fp32-communicator.md). [ADR 0002](../../adr/0002-python-buffer-ownership.md) records the accepted Python ownership/failure boundary. [Verification](verification.md) records what shipped and its tested limits.

## Architecture and packaging

Keep `include/meshtrain` and `src` independent of Python. Add binding implementation under `bindings/python`, package exports under `python/meshtrain`, and focused checks under `tests/python`. Python owns configuration conversion and borrowed-buffer validation; the existing C++ communicator owns sockets, collective order, deadlines, reduction, and state.

Add `MESHTRAIN_BUILD_PYTHON`, default OFF. Discover CPython/pybind11 and add the extension only when enabled. The wheel's root `pyproject.toml` opts in and disables native test targets; the extension links the position-independent `meshtrain::core` privately. Install the extension into `meshtrain` using the `python` CMake component, and select only that component for the wheel. Preserve the existing standalone native install/export. A wheel is a Python distribution, not a second C++ SDK.

Use the already verified pybind11 3.1.0 and scikit-build-core 1.1.1, with the existing hash-locked build tools. Initially declare CPython 3.12 (`>=3.12,<3.13`) and NumPy 2.x (`>=2,<3`); pin/hash the concrete NumPy release used for checks in a separate runtime requirements file during implementation. Record its version rather than implying every release in the dependency range was tested. NumPy is a runtime dependency, not a native-core or NumPy-header build prerequisite. Do not publish to PyPI or claim manylinux, native Windows, free-threaded, or subinterpreter support from a local Linux wheel.

## Public Python API

Export `Endpoint`, `RunConfig`, `Communicator`, and `EngineError` from `meshtrain`:

- `Endpoint(host, port)` exposes read-only host/port values.
- `RunConfig(*, rank, world_size, run_id, context_fingerprint, endpoints, listen_address="0.0.0.0", startup_timeout_ms=30000, collective_timeout_ms=30000, max_payload_bytes=268435456)` exposes read-only snapshots of these values. `endpoints` is a sequence of `Endpoint` values copied into native storage; returned endpoint collections cannot mutate that storage. Integer arguments must fit the destination native type; booleans are not numeric configuration values. Semantic checks reuse native validation, without opening sockets.
- `Communicator(config)` copies the native configuration while holding the GIL, then releases it to join TCP and construct the existing communicator. Construction returns only when ready or raises. There are no Python callbacks in startup.
- `comm.all_reduce_sum(array) -> None` performs blocking in-place SUM. `comm.state` returns `"ready"`, `"failed"`, or `"closed"`; `comm.next_sequence` returns an integer starting at zero.
- `comm.close() -> None` is idempotent and may cancel an active call from another thread. `with Communicator(config) as comm` returns that object on entry and closes it on exit without suppressing exceptions. Destruction also closes resources; explicit close is the documented normal use.

The Python handle retains native ownership throughout each call, including when another thread drops application references. A local strong native handle prevents destruction before the active operation exits. The core continues to own its transport exclusively; sharing a Python handle does not permit concurrent collectives.

`EngineError` derives from `RuntimeError`. Its string preserves the native contextual diagnostic; attributes expose `code` as the native category string, `rank`, `run_id`, `operation`, `sequence`, `peer`, and `peer_endpoint`. Unavailable optional integer fields are `None`. Translation happens with the GIL held. Local argument-kind/dtype errors use `TypeError`; invalid shape/layout/writability/alignment/representability uses `ValueError`; failed buffer acquisition preserves the Python exporter error. Native configuration/collective errors remain `EngineError`. Standard allocation failures retain the normal Python `MemoryError` mapping.

## Buffer validation and ownership

1. With the GIL held, require an actual NumPy ndarray (including ndarray views/subclasses with a valid matching export). Do not call `numpy.asarray`, coercing array constructors, `array_t::ensure`, or user `__array__` conversion. Non-NumPy buffers, lists, tensors, and integer addresses are rejected.
2. Check native-endian float32 dtype, exactly one dimension, C contiguity, writability, and representable nonnegative count/byte length. Use contiguity semantics rather than requiring a four-byte stride for an empty or singleton array; NumPy permits arbitrary strides for these contiguous cases. Reject nonempty storage that is null or not aligned for `float` before constructing a native span.
3. Obtain a writable buffer export and verify its descriptor agrees with the accepted ndarray's dtype, count, item size, contiguity, and storage. Revalidate ndarray metadata after acquisition, which may execute a subclass's Python exporter. Hold the raw descriptor in a scoped RAII owner and check dimensionality/length before any shape products; generic buffer-info construction would compute those products before our validation. Keep the ndarray and owning export alive in an outer GIL-held scope. The export retains its supporting object; the ndarray's owner chain retains normal NumPy backing storage. This does not authorize unsafe foreign allocations or aliases that invalidate storage behind NumPy's back.
4. Save only native pointer/count and a strong native communicator handle for the communication scope. Enter a nested `gil_scoped_release` scope, call the existing native operation, then reacquire the GIL before releasing the export, destroying Python references, returning, or translating an exception. Avoid a whole-function release guard around Python validation or cleanup.
5. On success, return `None`; the original allocation contains the sum. The wrapper does not pack, average, copy back, create a result array, or attach autograd behavior. Later training code explicitly packs/unpacks gradients in #7.

The caller prevents reads/writes through all aliases, resizing, shape/dtype/storage mutation, and external invalidation for the duration of a call. GIL release permits Python progress; it does not make concurrent access to borrowed storage safe. A retained owner/export addresses lifetime, not general data-race protection.

## Admission, failure, and threading

Python argument validation/export acquisition occurs before native collective admission. A rejected Python input leaves state and sequence unchanged and does not modify the array. If peers already entered their matching collective, the caller must close/abort the group; the later launcher will stop peers on worker failure, and independently started peers remain bounded by native deadlines. Do not imply that a local Python rejection is a group-wide rollback.

After creating a valid span, call the native communicator without duplicating its payload-limit, state, admission, or deadline machinery. In particular, an excessive configured payload is an admitted native failure and closes the group; do not turn it into a recoverable Python preflight error. Native concurrent-call rejection leaves the active operation intact. Other admitted failures make the communicator failed/unusable; explicit close leaves it closed even when racing with failure. A failed buffer may be partially changed.

Release the GIL during blocking construction, the native collective, and native close. Do not hold a Python-side operation mutex that prevents close from cancelling a waiting call. Perform property/error conversion with the GIL held; native state/sequence access retains its existing synchronization. Native deadlines bound communication; Python thread scheduling and GIL reacquisition are not hard real-time guarantees. Forking after creating a communicator is unsupported; each worker creates its own communicator after process startup.

## Alternatives and consequences

- **Strict NumPy boundary (selected):** directly serves detached CPU tensor views later, exposes storage identity, and has one clear validation/ownership contract. Generic PEP 3118 buffers would widen dtype/exporter behavior and can be designed separately.
- **Automatic conversion/copy-back:** convenient for callers, but can hide copies, truncate dtypes, or update only a temporary. Rejected for an in-place distributed primitive. An explicit packing adapter remains available in #7.
- **Bind the PyTorch C++ tensor API immediately:** increases core integration/dependency scope before basic storage/GIL behavior is proven. NumPy views satisfy the accepted CPU design; a custom backend is M8.

An explicit config object keeps networking identity and units reviewable without exposing mutable live communicator settings. The initial ABI/runtime range is intentionally narrow; expanding supported versions needs actual packaging evidence. Error attributes avoid requiring the trainer to parse diagnostic strings. Keep these reasons in this design/ADR and ordinary new code comments/docstrings at most two physical lines.

## Focused live verification

Add `tools/check_python_bindings.py` as the reproducible entry point. It builds the source distribution and a wheel from that archive using the locked tools, creates a fresh runtime environment, installs the wheel and hash-locked NumPy, and runs isolated subprocess checks outside the source package tree. Assert the imported module lives in that environment and inspect wheel contents. `tests/python` uses the standard library test runner; no new general test framework is needed.

| Boundary | Required observation |
| --- | --- |
| Package/config | Real installed import, version/config/defaults, range/type/semantic failures, fresh-process construction/close, no SDK/test payload in the wheel. |
| Buffers | Reject list/`__array__` object/memoryview, wrong dtype/swapped byte order, read-only, strided/reversed multi-element, misaligned, scalar and 2D arrays; no mutation or sequence advance on rejection. Accept empty/singleton contiguous cases, an offset view and a valid ndarray subclass. Preserve an exporter acquisition error, leaving state/sequence unchanged and allowing subsequent valid use. |
| Native SUM | Two independent processes exchange 0, 1, and 1025 elements; exactly representable fixtures compare exactly at both ranks, base-array sentinels remain unchanged, storage addresses are unchanged, and sequence advances across repeated calls. A one-rank case preserves input bits. |
| Lifetime/GIL | Controlled IPC holds a peer out of startup or its collective. While the native call is waiting, another Python thread must execute a handshake that lets the peer proceed. Drop external references during a collective; weakrefs/finalizers show the view/owner survive, then are collectible after success and failure. |
| Failure/lifecycle | Close from another thread unblocks a waiting call; overlapping calls are rejected without cancelling the first. A one-second collective timeout has an external five-second watchdog. Check error fields, terminal state, refused reuse, explicit close, and repeated group recreation. |

Use control pipes/events and externally enforced watchdogs, not a heartbeat count or unbounded thread join. A stuck GIL test must fail under an independent process watchdog. Always reap workers and close pipes/listeners on failure. Report observed elapsed times and distinguish fixture synchronization from collective timings.

Run native core/communicator contracts and the installed-consumer check after the CMake changes, including TCP-disabled portability. Build the existing no-Python native container target to verify independence; it currently runs its existing seven CTests as part of that target. Add a `python-bindings-test` Docker stage and use it in the existing Python CI matrix slot; retain `python-toolchain-test` as a manual probe. No broad new worker-count/algorithm/training matrix or four-container trainer is added here. Current broader push/PR checks remain as configured; nightly scheduling is a separate task.

Record implementation commands, selected NumPy/tool versions, numerical results, timings, raw output, and limitations in `verification.md` when executed. Required unrun checks stay unchecked. Local Linux/WSL/container evidence does not establish native Windows, physical multi-host, GPU, or PyTorch training support.

The container prepares the hash-checked NumPy wheel in a separate dependency layer before copying task sources, using a BuildKit pip cache. Runtime installation then uses that wheel with indexes disabled. An initial container check exceeded its setup timeout downloading NumPy; separating network preparation from offline verification avoids repeating that download on code/doc changes without weakening collective watchdogs.

## Primary API references inspected on 2026-10-17

- [pybind11 NumPy behavior](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) and [pinned 3.1.0 array implementation](https://github.com/pybind/pybind11/blob/v3.1.0/include/pybind11/numpy.h): array casting can convert; explicitly bypass coercion for this API.
- [pybind11 GIL guidance](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil) and [pinned buffer-info destructor](https://github.com/pybind/pybind11/blob/v3.1.0/include/pybind11/buffer_info.h): export cleanup must be outside the released-GIL scope.
- [CPython 3.12 buffer protocol](https://docs.python.org/3.12/c-api/buffer.html): successful export acquisition retains its supporting object and requires release.
- [NumPy flags](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html): contiguity, writable/aligned storage, and degenerate strides are separate properties.
- [scikit-build-core configuration](https://scikit-build-core.readthedocs.io/en/stable/configuration/index.html) and [pinned 1.1.1 schema](https://github.com/scikit-build/scikit-build-core/blob/v1.1.1/src/scikit_build_core/resources/scikit-build.schema.json): component-selected wheel installation. Documentation inspection informs the design; it is not package execution evidence.
