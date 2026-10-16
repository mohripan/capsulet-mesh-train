# ADR 0001: Blocking centralized FP32 communicator

Date: 2026-10-16

Status: accepted within the previously approved [M1 specification](../superpowers/specs/2026-10-16-cpu-mvp-design.md), with implementation explicitly requested in [#5](https://github.com/mohripan/capsulet-mesh-train/issues/5).

## Context

TCP startup and exact-byte transfers already exist. The next layer must implement our own all-reduce while remaining usable by optional bindings and future transport adapters. It must make collective order, buffer ownership, memory cost, and failure behavior explicit. Training averages gradients; the communication primitive implements SUM.

## Decision

Add a noncopyable, nonmovable `Communicator` that takes a validated `RunConfig` and exclusive ownership of an already joined `ByteTransport`. Its public header uses standard C++ types and a private implementation. Construction/joining happens before the communicator becomes ready; the TCP adapter remains separately responsible for HELLO/READY. The caller must provide the same group configuration to both layers. No platform handles or Python types cross this interface.

Expose blocking, in-place `all_reduce_sum(std::span<float>)`, idempotent `close()`, state inspection, and the next collective sequence. The caller retains buffer ownership and prevents concurrent access until return. Empty buffers still exchange metadata. One worker is an identity operation with the same validation and sequence rules.

Rank 0 uses its caller buffer as the accumulator. It receives one contribution at a time in ascending rank order into one FP32 scratch vector, adds in FP32, and distributes the result after all contributions arrive. Scratch is at most one local payload, independent of worker count, and is released at the end of the call. Workers send their input and receive the result into that same input storage. There is no hidden division or delegation to another collective library.

Every request/result must match expected operation, source, destination, sequence, count, FP32 dtype, SUM reduction, and byte length before receiving its payload. Sequences begin at zero and advance only after a locally successful operation. Refuse exhaustion instead of wrapping. Successful completion is local completion; the operation is not a transaction guaranteeing that every peer has returned successfully.

Only one collective may run on a communicator. Concurrent-call rejection leaves the active operation intact. Once admitted, any failure, including local buffer-limit validation or allocation failure, closes the transport and leaves the communicator failed. This prevents a rank from continuing after peers have entered a collective it did not complete. Explicit close leaves state closed, including when it races with failure. Destruction requires callers to have joined active operations.

One absolute monotonic deadline covers the entire collective: validation/allocation checks, every header/payload transfer, reduction, and completion. Reduction checks the deadline and explicit close between bounded chunks. Failure propagation uses immediate transport closure; peer EOF or their own original deadlines bound detection. The originating rank retains its contextual error; peers may report closure or timeout. Reserved ERROR frames are not sent in M1, avoiding extra potentially blocked writes during failure. Failed input may be partially modified and must not be used for an optimizer step.

## Alternatives considered

- Merge collective state into the TCP adapter. This couples numerical scheduling to sockets and makes future point-to-point adapters duplicate the collective engine.
- Use free functions with caller-managed sequence and lifetime. This makes accidental reuse after failure and overlapping calls easier; the required state belongs in one owning object.
- Buffer every rank before reducing. This uses memory proportional to worker count times payload size. One receive buffer meets the accepted memory contract.
- Use pairwise/tree accumulation immediately. It can change rounding and belongs to the algorithm comparison milestone. The initial contract requires predictable ascending rank order.
- Return to ready after a local validation error or retry a failed collective. Other ranks may already have changed state; continuing can mix sequences. Recovery creates a new group with a fresh run identifier.
- Send ERROR frames to every peer on failure. A peer may still be writing its contribution or may be unreachable. Closure gives a simpler bounded failure path; richer remote diagnostics can be designed later.

## Consequences and revisit triggers

The root is a bandwidth/computation bottleneck, and ascending rank order can wait behind a slow worker. This implementation establishes correctness rather than optimal scaling. Floating-point addition is order dependent; the tests distinguish exact fixtures, ordered cancellation fixtures, and bounded random numerical error. No cross-algorithm bitwise equivalence is promised.

Ring/tree algorithms, asynchronous work, concurrent progress, device buffers, and elastic membership require focused extensions. Revisit this ADR when those capabilities change ownership, sequencing, numerical order, or failure semantics. Preserve a standalone CPU core throughout.
