# ADR 0004: Blocking host-staged CUDA tensor adapter

Date: 2026-10-17

Status: Accepted — owner authorized implementation after independent review on 2026-10-17; both P2 findings were resolved before runtime acceptance.

OpenSpec: [host-staged CUDA adapter design](../openspecs/host-staged-cuda-adapter/design.md)

Issue/source: [#10](https://github.com/mohripan/capsulet-mesh-train/issues/10), M2's first integration task after completed M1.

## Context

The engine accepts host spans; the installed binding retains strict NumPy storage during blocking native SUM. CUDA tensors require a deliberate device/host ownership and completion boundary. Adding CUDA to the core or accepting integer device pointers would undermine the standalone architecture and existing buffer contract.

The owner wants GPU functionality before advanced CPU optimization, with only one laptop GPU currently available. The initial task needs honest functional evidence and measured copy costs; it does not require efficient direct GPU networking or a custom framework backend. ADRs [0001](0001-blocking-centralized-fp32-communicator.md), [0002](0002-python-buffer-ownership.md) and [0003](0003-replicated-sgd-training-contract.md) remain accepted history.

## Decision and rationale

Add an explicitly imported Python PyTorch adapter with a fixed-count CPU FP32 workspace and sharing NumPy view. Require an explicit indexed CUDA device and ordinary contiguous one-dimensional FP32 tensor without reverse/forward autograd state or inference-tensor restrictions. Allocate an ordinary mutable workspace even when the caller is in inference mode. The caller selects pinned or pageable host storage. Our existing C++ communicator performs SUM exactly once; averaging remains a training-layer operation.

Capture the configured device's current stream. The caller orders producers and aliases onto it. Wait for readiness, finish D2H before native host access, then finish H2D before reporting success or reusing the workspace. Retain owners through completion and reject concurrent workspace use. Do not add asynchronous work handles, graph capture or hidden device selection.

Preserve native errors and admission/sequence semantics. A post-native CUDA copy failure may leave an advanced native sequence and a partial device result: abort the group, poison the adapter and report incomplete/unknown GPU cleanup when synchronization cannot be certified. Do not claim rollback, hardware cancellation or recovery. Separate readiness, D2H, binding/native, H2D and total synchronized wall measurements.

Keep PyTorch/CUDA runtime requirements optional and separately locked. The standalone C++ package, existing NumPy binding and CPU trainer keep their current dependencies and behavior. #11 will separately integrate GPU model/gradient packing/reference parity.

## Alternatives considered

- A C++/LibTorch CUDA extension could manage staging natively, but adds ABI/build dependencies without changing the initial host-staged data path. Revisit for measured overhead or actual CUDA engine work.
- CUDA device buffers/transport or native reduction kernels change core capabilities and require stream-aware scheduling, memory and hardware contracts. They belong to later milestones.
- Existing GPU collective libraries would bypass the algorithms being built. Keep such implementations for separately labeled baselines/adapters.
- Enqueue H2D and return immediately reduces host waiting but imposes output-stream/workspace lifetime obligations on every caller. The first API instead promises completed blocking results; overlap needs its own design.

## Consequences

The adapter adds two visible copies and host waits; a pinned allocation is a bounded optional resource, not an assumed speedup. Input and output data still use our C++ communication path. Explicit completion makes the first integration easier to verify and consume from another stream, at the cost of overlap.

CUDA failures are not bounded by native communication deadlines. External test/process containment remains necessary, and a poisoned GPU context cannot be treated as a recoverable communicator error. Sharing the laptop GPU among worker processes establishes functional integration only; distinct-GPU scaling, Windows and physical-host evidence remain separate.

## Revisit triggers

Measured binding/staging overhead, asynchronous work, backward overlap, native CUDA reduction, direct device transport, additional dtypes/layouts, CUDA graph capture or sharded GPU training may require a new ADR. Preserve the CPU core and accepted history when those contracts evolve.

## Decision record

Proposed during #10 planning. On 2026-10-17 the owner selected independent review and said “And then you can go on and implement it.” [Review/disposition](../openspecs/host-staged-cuda-adapter/review.md) addresses inference/forward-AD rejection, including disabling inference mode for tangent inspection. The explicit owner instruction authorizes implementation after those corrections; the review verdict itself is not approval. The final review confirmed the corrections resolved both findings; implementation/runtime verification remains separate.
