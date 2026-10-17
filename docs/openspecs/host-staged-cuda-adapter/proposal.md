# Proposal: Explicit host-staged CUDA tensor adapter

Date: 2026-10-17

Status: complete; implemented, verified and pushed. #10 is closed; #9/M2 remain open for #11.

Source: [issue #10](https://github.com/mohripan/capsulet-mesh-train/issues/10), under [M2 tracker #9](https://github.com/mohripan/capsulet-mesh-train/issues/9). After M1 completion and the next OpenSpec being identified, the owner said “Okay. Please continue!”. This authorizes preparing the next task's artifacts; it is not approval of an unseen CUDA contract.

## Problem and outcome

The C++ engine and strict NumPy binding currently communicate CPU FP32 storage. A CUDA tensor cannot be passed to that host-byte interface. Add an optional PyTorch adapter that makes device-to-host staging, our native SUM and host-to-device completion explicit. Prove that actual GPU tensors are mutated correctly, with safe storage lifetime and separate transfer/communication measurements.

This task delivers the tensor primitive and its focused hardware checks. GPU training/reference parity and the full training entry point remain [#11](https://github.com/mohripan/capsulet-mesh-train/issues/11).

## Scope

- Add an explicitly imported Python CUDA adapter using the existing installed native communicator. PyTorch owns device copies and host allocations; C++ continues to own transport, sequencing, deadlines and reduction.
- Support ordinary dense, contiguous, one-dimensional CUDA FP32 tensors without autograd tracking, with explicit logical device index and fixed workspace size. Perform SUM without averaging or implicit conversion.
- Support requested pinned host storage by default and explicitly selected pageable storage. Complete the selected CUDA stream before native CPU access and before returning the result. Report synchronized wall times for readiness, each copy, the binding/native call and the complete operation.
- Preserve the standalone core, CPU package/import, CPU trainer, protocol and existing CI. Add a separate hash-locked Linux CPython 3.12 CUDA runtime and a focused installed-wheel verification command using the available single GPU.
- Draft [ADR 0004](../../adr/0004-blocking-host-staged-cuda-adapter.md) for the lasting dependency, stream, ownership and failure decisions. Keep it Proposed until the owner accepts the presented scope.
- Record actual execution evidence and update #10/#9 navigation when earned. Planning may be committed/pushed to main under the standing instruction, without co-author trailers.

Excluded: changes to the C++ host-byte API, LibTorch/CUDA build dependencies in the core, raw device pointers through pybind11, native CUDA reduction kernels, direct GPU networking, NCCL/MPI/Gloo collectives, mixed precision, sparse/noncontiguous packing, autograd hooks, asynchronous work/overlap, CUDA graph capture, automatic retry, driver installation, native Windows, distinct-GPU or physical-host scaling, and #11's trainer implementation. No new GPU CI runner or nightly GPU execution is assumed.

## Acceptance

1. An installed `meshtrain.cuda_adapter.CudaStagingAdapter` stages a fixed-count CUDA FP32 tensor through retained CPU FP32/NumPy storage and calls the existing `Communicator.all_reduce_sum` exactly once. The original GPU allocation contains SUM on success; no other library performs the collective.
2. Pinned and pageable modes are explicit and observable. Both preserve correct completion; requested pinning failure raises rather than silently falling back. The input, host tensor, NumPy owner chain and communicator stay alive for their actual use. Host staging is not read before D2H completion or reused before H2D completion.
3. The call uses the current stream of its configured device, including a nondefault stream. The caller orders every producer/alias onto that stream. On successful return the result copy has completed, permitting use from another stream without an extra dependency on the adapter. This blocking promise does not provide backward overlap.
4. Reject wrong argument kind, device, dtype, shape/count, layout, reverse/forward autograd state, inference input or capture mode before native admission; no implicit cast, flatten, device selection or CPU fallback. Rejection preserves input/primal/tangent and local native sequence. A workspace constructed under inference mode remains ordinary mutable storage. Native admitted failures preserve existing terminal/error-context behavior; CUDA execution failures fail the adapter/job with honest cleanup limits.
5. Real one- and two-process GPU checks exercise both staging modes, counts 0/1/1025/1048576, default/nondefault stream ordering, repeated reuse, retained owners and native error propagation. Two workers may share logical device 0 on this laptop; evidence must say single-GPU functional integration. Zero count still advances the collective sequence. Exact fixtures pass exactly; bounded-random fixtures use unchanged `atol=1e-6, rtol=1e-5` against an independent CPU reference.
6. Publish per-call raw `ready_wait_ms`, `d2h_ms`, `native_ms`, `h2d_ms` and `total_ms`, with counts, byte size, rank, device and memory mode. Times include real completion and are not labeled network-only latency or distinct-GPU speedup. No requirement claims pinned storage is always faster.
7. Build the actual sdist-derived wheel and test it in a fresh isolated GPU runtime. Record source/wheel/lock identities, imported paths, PyTorch/CUDA/driver versions, GPU model/UUID/capability and available architectures. GPU absence or unsupported execution fails the GPU command; a skip cannot close #10.
8. Verify continued native-only/installed-consumer and Torch-free binding/CPU trainer behavior with the relevant existing checks. Keep #10 open until required hardware evidence passes, and leave #9/M2 open for #11 even after #10 completes.

## Review and authorization

- Review route: the owner selected a fresh independent sub-agent on 2026-10-17. `review_cuda_adapter_spec` has no inherited conversation and performs read-only review of the raw issue requirements, artifacts and project material.
- Findings/disposition: the fresh reviewer found two P2 blockers: inference input/workspace behavior and forward-mode AD dual tensors. The design/acceptance/tasks now require early inference/dual rejection and ordinary workspace construction independent of inference mode. The final follow-up confirmed both findings resolved with no remaining material technical inconsistency. See [review.md](review.md); planning/review is not GPU runtime evidence.
- User go-ahead: on 2026-10-17 the owner said “Sub-agent. And then you can go on and implement it.” This authorizes implementation of the presented scope after the selected review and disposition; no second approval gate is required unless scope materially changes.
- Standing delivery instruction: commit/push authorized completed work to main without co-author trailers; preserve unrelated local skills/configuration.

## Delivery

Implementation and installed GPU acceptance passed after the independently reviewed corrections and owner go-ahead. Read [design.md](design.md), [tasks.md](tasks.md), [preflight.md](preflight.md), [review.md](review.md) and Accepted ADR 0004. [Actual runtime evidence](verification.md) includes seven GPU cases, 48 numerical records, 120 measured samples and CPU regressions. Implementation e557542 is pushed without co-author trailers, exact-revision CPU CI passed, and tested-source correspondence was verified. #10 is closed; #9/M2 and GPU trainer parity remain #11.
