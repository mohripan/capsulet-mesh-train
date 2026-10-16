# Project decisions

Recorded 2026-10-16 from the initial design discussion.

Focused implementation decisions are recorded in [architecture decision records](adr/README.md), starting with [ADR 0001 — blocking centralized FP32 communicator](adr/0001-blocking-centralized-fp32-communicator.md). This document retains the project-wide agreements.

## Agreed direction

| Decision | Rationale and consequence |
| --- | --- |
| Open-ended side project | No deadline, hours budget, or milestone due dates. Milestones preserve context for the owner and future agents. |
| Library-first C++ architecture | The communication engine is independently usable. Python bindings come before a custom PyTorch backend. |
| Implement the initial transport ourselves | Own raw TCP connections, framing, byte transfers, and collective algorithms. |
| Support both kinds of integration | Allow future transport adapters and framework integrations. An interface is an extension seam, not a claim that every future device/transport already works. |
| Real multi-machine operation is a requirement | Endpoints and worker identity must work across hosts. Localhost and containers are initial test environments. |
| Linux primary; native Windows later | Use Linux/WSL2 and Linux containers first. Track native Windows builds and networking separately. |
| GPU training is the main target | Start on CPU for correctness, then prioritize CUDA integration before orchestration and advanced CPU tuning. |
| Available GPU hardware is one laptop GPU | CUDA integration is locally testable. Distinct multi-GPU and physical multi-host performance require later external hardware access. |
| Kubernetes integration | Use existing orchestration rather than building a cluster scheduler, node agent platform, or workflow engine. |
| Job-token pools are exclusive initially | A job has a pool and positive integer cost. It holds that cost against the pool's capacity for its admitted lifetime. Other pools cannot borrow it. |
| Running jobs are not preempted for priority | Token admission controls concurrency. Guaranteed immediate service still requires actual spare or reserved hardware. |
| Airflow described an analogy | DAG execution is not a project deliverable. |

## Implementation defaults

The user authorized implementation of the CPU specification on 2026-10-16. Its defaults are the M1 contract. Defaults for later subsystems, including FIFO token admission, remain subject to their focused designs. Actual compatibility evidence is recorded in [development instructions](DEVELOPMENT.md).

| Default | Reason |
| --- | --- |
| C++20 and CMake; optional pybind11 package | Keep a modern standalone native core and a thin Python boundary. |
| Python 3.12 as the initial Linux packaging target | Choose one reproducible build environment first; dependency versions are verified and locked in the build issue. The host's Python version does not define the container environment. |
| Dense, contiguous CPU FP32 buffers; SUM collective | A small numerical contract makes failures and reference parity understandable. The trainer performs explicit averaging. |
| Fixed worker membership during a run | Mid-collective membership changes are excluded initially. Checkpoint recovery restarts the worker group. |
| Explicit rank, world size, run identifier, and endpoint list | Support localhost, containers, and separate hosts without relying on a shared local process tree. |
| Blocking centralized all-reduce first | Own the complete communication path before adding concurrency, algorithms, or framework internals. |
| Rank 0 reduces peers in ascending rank order | Predictable reduction order and bounded root scratch memory. |
| Deterministic MLP, synthetic data, and SGD | Make the first training comparison reproducible; a tiny transformer and Adam/sharding follow later. |
| Equal local batch sizes for the first parity test | Mean local gradients averaged across ranks equal the global-batch gradient. Unequal batch sizes require explicit weighting in a later design. |
| Start with bounded failure, add checkpoint recovery | Detect an unusable communicator first; automatic retries belong in the runtime layer. |
| FIFO admission within each exclusive token pool | Keep the initial policy predictable. No borrowing, preemption, duration prediction, or backfilling in the initial admission contract. |

## Resource and recovery distinctions

Token costs are concurrency weights, not a rate limit and not a time-based spending budget. Tokens return when the admitted worker group has stopped. One cost applies to the complete distributed job, not to each worker separately.

Policy admission and physical resource placement are separate gates. Independent token pools do not reserve actual GPUs by themselves. Quota reservation also does not prove that a whole worker group fits the physical node topology. The Kubernetes design must account for both.

The target recovery model is to restore a consistent checkpoint of the failed training job. Completed external preparation artifacts can be reused by a surrounding system. A restart of one Pod alone does not restore the other workers' collective sequence, optimizer state, or data position.

## Revisit triggers

- Add new tensor dtypes or devices through an explicit buffer and numerical contract.
- Redesign group membership before adding elastic worker counts or in-place worker replacement.
- Revisit transport capabilities before adding CUDA IPC, RDMA, or another device transport. A host-byte interface must not silently pretend to support device memory.
- Review fairness before changing FIFO, pool exclusivity, or borrowing policy.
- Add distinct-GPU and physical multi-host evidence before publishing scaling claims.
- Choose and pin supported PyTorch/backend versions when implementing the experimental backend integration.
