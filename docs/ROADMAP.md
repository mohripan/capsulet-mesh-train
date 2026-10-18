# Roadmap

Milestones are durable outcome checkpoints, not a calendar. The [GitHub roadmap tracker](https://github.com/mohripan/capsulet-mesh-train/issues/1) and [milestones](https://github.com/mohripan/capsulet-mesh-train/milestones) track live progress; this document explains scope, dependencies and completion evidence. M1 and M2 are complete: see [CPU acceptance](openspecs/m1-validation/verification.md), [GPU training/CI evidence](openspecs/cuda-mlp-trainer/verification.md) and [build instructions](DEVELOPMENT.md). Continue with M3/#13's focused ring algorithm/measurement OpenSpec. Later work requires its own design/review/go-ahead.

## Milestone map

| ID | Outcome | Prerequisites | Completion evidence |
| --- | --- | --- | --- |
| M1 | Correct four-process CPU trainer | CPU MVP design review | Standalone C++ TCP engine, centralized SUM all-reduce, Python bindings, reference parity, bounded failures, four-container smoke test. |
| M2 | CUDA training through our engine | M1 | Explicit host-staged CUDA path, correct buffer/stream lifetime, one-GPU training parity, and measured transfer costs. |
| M3 | Ring/tree algorithms and CPU communication study | M1; does not block M2 | All algorithms correct for odd worker counts and uneven partitions; reproducible comparable Gloo baseline and ring-model analysis. |
| M4 | CUDA reductions, transfer pipelines, and backward overlap | M2 + M3 | Native CUDA reduction, asynchronous work contracts, ordered gradient buckets, timeline evidence, and correctness under delayed ranks. |
| M5 | Sharded Adam states and public sharding collectives | M3; reuse M2 CUDA support | Public reduce-scatter/all-gather, simplified ZeRO-1 Adam with replicated gradients, optimizer parity, and measured state-memory savings. |
| M6 | Checkpoint recovery of the worker group | M1 for replicated training; M5 for sharded-state coverage | Consistent checkpoint commit, complete state restoration, bounded group retries, and kill/restart tests matching an uninterrupted reference. |
| M7 | Kubernetes deployment and exclusive job admission | M1 + M6; M2 for GPU workload validation | Worker-group lifecycle, explicit resources, whole-group admission, exclusive weighted token pools, and restart-safe accounting. |
| M8 | Framework and transport adapters | M1 + M3; device paths depend on their implementations | An MPI point-to-point transport adapter and a versioned custom PyTorch backend pass conformance and training tests. |
| M9 | Native Windows portability | M1; later features receive separate Windows coverage | Native build/package, Winsock transport, local multi-process collective tests, and CPU training parity. |
| M10 | Physical multi-host and distinct-GPU validation | Relevant algorithms/device paths + external hardware | Raw benchmark data, topology/environment records, fair NCCL comparison on distinct GPUs, and demonstrated operation across physical hosts. |

Suggested first sequence: **M1 -> M2 -> M3 -> M4**. Recovery foundations can start after M1. Other branches should follow their actual dependencies rather than waiting for every earlier milestone number.

## M1: Correct CPU engine and trainer

Completed on 2026-10-17: issues #3–#8, tracker #2 and milestone 1 are closed. [Full manual CI](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39030243246) and [regular CI](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39030238414) passed the final implementation revision. Both 100-step deployment profiles pass 3,090 comparisons each with lower evaluation loss; 28 controlled failure groups and all Docker ownership/cleanup checks pass. [Commands, raw results and limits](openspecs/m1-validation/verification.md).

The first end-to-end result is four independent CPU processes training a deterministic MLP on different minibatches. Our C++ engine performs centralized FP32 SUM all-reduce. The Python training adapter explicitly divides by worker count before applying SGD. Losses, averaged gradients, and updated parameters match a single-process reference using the same combined batches.

The engine accepts general fixed worker counts; four is the main demonstration. Tests include 1, 2, 3, 4, and 5 workers. Endpoint configuration supports separate hosts, although a local/container run establishes only local/container correctness.

Build, transport, collective, bindings, trainer, and integration issues implement the [focused specification](superpowers/specs/2026-10-16-cpu-mvp-design.md). Basic socket failures and hangs must terminate predictably; automatic training recovery belongs to M6.

## M2: CUDA functionality before advanced CPU optimization

Completed on 2026-10-18: #10/#11, tracker #9 and milestone 2 are closed. The [tensor primitive](CUDA.md) and [CUDA trainer](CUDA_TRAINING.md) have independently reviewed/owner-authorized OpenSpecs and Accepted ADRs [0004](adr/0004-blocking-host-staged-cuda-adapter.md)/[0005](adr/0005-cuda-training-reference-contract.md). Both two-worker/100-step memory modes passed 936 independent GPU reference comparisons each, with decreased evaluation loss, 760 measured staged-call samples, resource/failure/reaping evidence and [exact implementation-revision CPU CI](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39115392693). [Raw evidence and limits](openspecs/cuda-mlp-trainer/verification.md): two workers shared one physical GPU. Continue with M3/#13's focused OpenSpec.

Expose a separate CUDA adapter. Initially copy gradients to host buffers, run our C++ collective, and copy the result back. Make all copy costs, buffer ownership, and synchronization explicit. The CPU core remains buildable without CUDA.

The laptop can validate CUDA computations and host-staged integration, including tiny custom-engine runs with multiple processes sharing its GPU if memory permits. Such tests are explicitly functional tests. NCCL multi-rank baselines require distinct GPUs and belong to M10.

Measure staging and synchronization separately from network/reduction time. Completing M2 does not claim efficient device-to-device communication or multi-GPU speedup.

## M3: Algorithms and explanation through measurements

Implement ring all-reduce as reduce-scatter plus all-gather, handling zero-size chunks and message lengths not divisible by worker count. Implement tree reduction plus broadcast for non-power-of-two groups.

Benchmark centralized, ring, and tree with the same rank counts, payload sizes, dtype, warm-up policy, timing boundary, and synchronization. Use Gloo for the initial CPU comparison; MPI is an optional additional baseline after its adapter exists. Store per-rank raw samples and report the maximum rank completion time per iteration, medians, and tail latency.

Study the simplified model:

$$
T_{ring} \approx 2(p-1)\alpha + \frac{2(p-1)}{p}\frac{n}{B}.
$$

Define `n` as the full input-buffer size in bytes and `B` as the assumed effective link bandwidth. Calibrate latency/bandwidth separately, retain held-out measurements, and explain errors from reduction work, host memory, socket behavior, synchronization, binding overhead, topology, and virtualization. Report tree/ring crossover results as observations of the measured environment.

The first communication report can be published from these local results. Later hardware validation extends that report; it does not block an honest local portfolio deliverable.

## M4: GPU communication work and overlap

Introduce explicit asynchronous work handles, progress ownership, completion/error semantics, and buffer lifetime. Implement CUDA FP32 reductions and a measured pinned-host transfer pipeline before pursuing hardware-specific direct paths.

Gradient hooks enqueue ready buckets; every rank issues collectives in a consistent bucket order. Work must complete before the optimizer reads the gradients. Use a tiny transformer to study bucket size, compute/communication overlap, delayed ranks, and cases where additional workers reduce throughput.

One-GPU timelines can verify ordering and functional overlap. They are not distinct-GPU throughput evidence. CUDA IPC, peer access, RDMA, and GPUDirect require separate capability checks and focused designs when the relevant hardware is available.

## M5: Sharded optimizer states

Expose public reduce-scatter and all-gather with an explicit uneven-partition contract. Verify their composition against all-reduce.

The simplified ZeRO-1 implementation keeps parameters and synchronized gradients replicated, assigns each parameter shard to an owner, and stores Adam moments only on that owner. Owners update their parameter shards and distribute the updated parameters. Validate against ordinary Adam with matching hyperparameters and step counters.

Report optimizer-state memory separately from parameters, gradients, communication buffers, and allocator reservations. Gradient sharding is a later research item requiring its own design; it is not part of ZeRO-1 completion.

## M6: Recovery

A checkpoint represents a completed optimizer step of the whole group. Save model and optimizer state, progress, sampler/data position, and per-rank randomness, plus layout/version metadata. Publish a checkpoint manifest only after all required files are complete; ignore incomplete generations during restore.

On worker failure, mark the communicator unusable, stop the group, and let the runtime recreate it with the same world size and a fresh attempt/session identifier. Restore the latest committed checkpoint. Bound retries and preserve the original diagnostic. In-place replacement and elastic resizing are separate future work.

Prove recovery first for replicated training. Extend it to sharded Adam after M5, including missing/corrupt shard cases. A worker killed mid-collective must not leave surviving ranks stuck indefinitely or continue with a partially applied optimizer update.

## M7: Kubernetes and exclusive token pools

Use Kubernetes for placement and resource enforcement. Evaluate existing job-group and admission integrations before choosing manifests/controllers; do not assume a native Pod label implements arbitrary job-token accounting.

The agreed policy is one designated pool and one positive integer cost per complete training job. Capacity is exclusive to that pool, with no borrowing or priority-based preemption. Proposed initial ordering is FIFO within a pool. Reject unknown pools, invalid costs, and costs larger than pool capacity rather than queueing them forever.

Admission must satisfy both the token policy and the physical resource/group-readiness policy. Claim a job's full token cost atomically. Tokens remain occupied until its worker group has stopped; failed/retried attempts must not create duplicate reservations or release capacity while old workers still run. Reconcile state after controller restart and cancellation.

Kueue is a candidate for physical resource quotas and admission checks. The integration design must verify how arbitrary user-defined costs are represented, whether an additional admission component is needed, and how unused quota is prevented from being lent. Token exclusivity alone does not guarantee immediately available GPUs.

## M8: Integrations

Implement an MPI transport using point-to-point communication while retaining our collective algorithms. Keep MPI dependency optional and compare it with the socket transport through the same conformance suite.

Build a custom PyTorch C++ communication backend only after the library contract is established. Pin and test its supported PyTorch versions because the extension interface is experimental. Clearly distinguish our algorithms, third-party transport adapters, and library baselines in configuration and results.

## M9: Native Windows

Port the networking boundary to Winsock and verify native build/package and CPU multi-process behavior. Keep POSIX types out of public interfaces from M1. Linux containers on Windows do not close this milestone. GPU support and backend availability on native Windows require explicit coverage rather than inheriting Linux claims.

## M10: Additional-hardware validation

This milestone requires external hardware access and has no promised completion date. Test distinct GPU devices and physical hosts, record interconnect topology and placement, and compare like-for-like payloads and timing boundaries against NCCL. Do not compare a host-staged path to a device path without exposing the copy and reduction differences.

Publish raw results, scripts, software versions, CPU/GPU placement, effective bandwidth definitions, error bars/tail statistics, and numerical tolerances. Keep unsupported environments and unmeasured claims explicit in the report.

## Completing work

- Close an implementation issue with reproducible commands and evidence against its acceptance criteria.
- Close its milestone tracker only when required child issues and the milestone's completion evidence are satisfied.
- Keep design-only, implemented, and externally validated states distinct.
- For later milestones, complete the focused design issue before implementation tasks that depend on it.
- Update the navigation documents when a dependency, contract, or project boundary changes.
