# ADR 0002: Strict NumPy borrowing at the Python boundary

Date: 2026-10-17

Status: Accepted — the owner authorized implementation after independent review on 2026-10-17; [review and disposition](../openspecs/python-buffer-bindings/review.md) satisfy that condition. This concretizes the accepted M1 NumPy/GIL requirements for the [issue #6 OpenSpec](../openspecs/python-buffer-bindings/proposal.md).

## Context

The native communicator owns collective state and transport but borrows a writable FP32 span. Python integration must preserve in-place mutation and storage lifetime while allowing Python threads to run during waits. Automatic array coercion and incorrectly scoped GIL release can respectively update temporary storage or release Python objects without the GIL.

The [task design](../openspecs/python-buffer-bindings/design.md) specifies packaging, exact API/error behavior, and focused checks. The [M1 Python contract](../superpowers/specs/2026-10-16-cpu-mvp-design.md#35-python-boundary) and [ADR 0001](0001-blocking-centralized-fp32-communicator.md) remain in force.

## Decision

Keep the optional pybind11/scikit-build package dependent on the standalone C++ core. Expose copied, read-only configuration and an owning communicator handle. The default native build never discovers Python/NumPy; the wheel installs only its Python package/component.

Accept actual writable one-dimensional C-contiguous native FP32 NumPy arrays with valid representable lengths and aligned nonempty storage. Validate without array conversion or copy-back. Retain the array, its writable buffer export/supporting owner chain, and native communicator for the complete blocking operation. Empty inputs still participate. SUM mutates the original storage and returns `None`.

Validate and obtain/release Python buffer exports with the GIL held. Release the GIL only in a nested scope containing Python-independent native startup/collective/close work, and reacquire it before Python cleanup or error translation. Do not introduce Python ownership into the core. Callers must prevent concurrent alias access, resizing, and storage invalidation; lifetime retention is not a general lock on the allocation.

Reject unsuitable Python arguments before native collective admission, leaving local sequence/state unchanged. Peers may already be waiting; callers must abort/close rather than assume a collective rollback, with native deadlines providing the independent bound. Once admitted, retain native terminal-failure, payload-limit, concurrency, and close semantics. Preserve native error category/context in a Python `RuntimeError` subclass rather than requiring diagnostic-string parsing.

## Alternatives considered

- Automatic dtype/layout conversion with temporary output: hides copies and can leave the original array unchanged. Explicit training-buffer packing belongs to the later adapter.
- General Python buffer exporters or integer pointers: expands dtype, exporter, and ownership obligations beyond the accepted initial NumPy boundary.
- Holding the GIL throughout communication: simpler scopes, but prevents Python thread progress and Python-triggered cancellation while waiting.
- Reimplementing collective state/error policy in Python: risks divergent sequences, deadlines, and failure semantics; keep the native owner authoritative after admission.
- Requiring the PyTorch C++ tensor library now: adds a framework build dependency before the narrow CPU storage contract is verified; framework backend integration remains later work.

## Consequences and revisit triggers

Callers must explicitly prepare suitable arrays and coordinate aborts after local validation errors. The API does not make borrowed storage safe for concurrent mutation. Focused installed-extension checks must prove aliasing, lifetime, GIL progress and failure cleanup; a successful import alone is insufficient.

CPython 3.12/Linux is the initial packaging target, with concrete tested dependency versions recorded as evidence. Generic buffers, additional Python/platform support, CUDA, asynchronous work, free-threaded Python, and subinterpreters require separate compatibility/ownership decisions. Preserve this record as history if future contracts supersede it.
