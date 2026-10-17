# GitHub work index

GitHub is the source of truth for live status. This index records the initial work breakdown; check the linked issues for subsequent changes.

- [Project roadmap tracker #1](https://github.com/mohripan/capsulet-mesh-train/issues/1)
- [All milestones](https://github.com/mohripan/capsulet-mesh-train/milestones)
- [All issues](https://github.com/mohripan/capsulet-mesh-train/issues)
- [Architecture and milestone outcomes](ROADMAP.md)
- [Decision record](DECISIONS.md)
- [Architecture decision records](adr/README.md)

## Current entry point

M1 is complete: [#3](https://github.com/mohripan/capsulet-mesh-train/issues/3) through [#8](https://github.com/mohripan/capsulet-mesh-train/issues/8), [tracker #2](https://github.com/mohripan/capsulet-mesh-train/issues/2) and [milestone 1](https://github.com/mohripan/capsulet-mesh-train/milestone/1) are closed. The standalone C++ TCP engine, centralized SUM, installed Python bindings and CPU trainer passed complete local clean acceptance and [exact-revision manual CI](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39030243246). Both localhost and four-container 100-step profiles passed 3,090 comparisons each; controlled failures and scoped cleanup passed. See [validation evidence](openspecs/m1-validation/verification.md), [commands](VALIDATION.md) and [accepted training ADR](adr/0003-replicated-sgd-training-contract.md).

Continue with [M2 tracker #9](https://github.com/mohripan/capsulet-mesh-train/issues/9), starting at [#10 — explicit host-staged CUDA adapter](https://github.com/mohripan/capsulet-mesh-train/issues/10). Read its [OpenSpec](openspecs/host-staged-cuda-adapter/proposal.md), [design](openspecs/host-staged-cuda-adapter/design.md), [tasks](openspecs/host-staged-cuda-adapter/tasks.md), [read-only preflight](openspecs/host-staged-cuda-adapter/preflight.md) and [Accepted ADR 0004](adr/0004-blocking-host-staged-cuda-adapter.md). Independent review/corrections and owner authorization are complete. Installed GPU acceptance and local CPU regressions passed; exact-revision CI/delivery remain pending. See [actual verification](openspecs/host-staged-cuda-adapter/verification.md) and [CUDA commands](CUDA.md). GPU training/reference parity remains #11; native Windows and physical multi-host/distinct-GPU evidence are separate open work.

The M1 design gate is satisfied. Later work is labeled `needs-design`, and M10 also carries `hardware-required`. Labels are navigation cues; a design, implementation, or validation gate is complete only with its stated evidence.

## Milestone trackers

| Milestone | Tracker | Required work issues |
| --- | --- | --- |
| [M1 — Correct CPU engine and trainer](https://github.com/mohripan/capsulet-mesh-train/milestone/1) | [#2](https://github.com/mohripan/capsulet-mesh-train/issues/2) | [#3](https://github.com/mohripan/capsulet-mesh-train/issues/3), [#4](https://github.com/mohripan/capsulet-mesh-train/issues/4), [#5](https://github.com/mohripan/capsulet-mesh-train/issues/5), [#6](https://github.com/mohripan/capsulet-mesh-train/issues/6), [#7](https://github.com/mohripan/capsulet-mesh-train/issues/7), [#8](https://github.com/mohripan/capsulet-mesh-train/issues/8) |
| [M2 — CUDA training functionality](https://github.com/mohripan/capsulet-mesh-train/milestone/2) | [#9](https://github.com/mohripan/capsulet-mesh-train/issues/9) | [#10](https://github.com/mohripan/capsulet-mesh-train/issues/10), [#11](https://github.com/mohripan/capsulet-mesh-train/issues/11) |
| [M3 — Collective algorithms and CPU benchmarks](https://github.com/mohripan/capsulet-mesh-train/milestone/3) | [#12](https://github.com/mohripan/capsulet-mesh-train/issues/12) | [#13](https://github.com/mohripan/capsulet-mesh-train/issues/13), [#14](https://github.com/mohripan/capsulet-mesh-train/issues/14), [#15](https://github.com/mohripan/capsulet-mesh-train/issues/15) |
| [M4 — CUDA reductions and backward overlap](https://github.com/mohripan/capsulet-mesh-train/milestone/4) | [#16](https://github.com/mohripan/capsulet-mesh-train/issues/16) | [#17](https://github.com/mohripan/capsulet-mesh-train/issues/17), [#18](https://github.com/mohripan/capsulet-mesh-train/issues/18), [#19](https://github.com/mohripan/capsulet-mesh-train/issues/19) |
| [M5 — Sharding collectives and ZeRO-1 Adam](https://github.com/mohripan/capsulet-mesh-train/milestone/5) | [#20](https://github.com/mohripan/capsulet-mesh-train/issues/20) | [#21](https://github.com/mohripan/capsulet-mesh-train/issues/21), [#22](https://github.com/mohripan/capsulet-mesh-train/issues/22) |
| [M6 — Consistent checkpoints and group recovery](https://github.com/mohripan/capsulet-mesh-train/milestone/6) | [#23](https://github.com/mohripan/capsulet-mesh-train/issues/23) | [#24](https://github.com/mohripan/capsulet-mesh-train/issues/24), [#25](https://github.com/mohripan/capsulet-mesh-train/issues/25), [#26](https://github.com/mohripan/capsulet-mesh-train/issues/26) |
| [M7 — Kubernetes and exclusive token admission](https://github.com/mohripan/capsulet-mesh-train/milestone/7) | [#27](https://github.com/mohripan/capsulet-mesh-train/issues/27) | [#28](https://github.com/mohripan/capsulet-mesh-train/issues/28), [#29](https://github.com/mohripan/capsulet-mesh-train/issues/29), [#30](https://github.com/mohripan/capsulet-mesh-train/issues/30) |
| [M8 — Framework and transport adapters](https://github.com/mohripan/capsulet-mesh-train/milestone/8) | [#31](https://github.com/mohripan/capsulet-mesh-train/issues/31) | [#32](https://github.com/mohripan/capsulet-mesh-train/issues/32), [#33](https://github.com/mohripan/capsulet-mesh-train/issues/33) |
| [M9 — Native Windows portability](https://github.com/mohripan/capsulet-mesh-train/milestone/9) | [#34](https://github.com/mohripan/capsulet-mesh-train/issues/34) | [#35](https://github.com/mohripan/capsulet-mesh-train/issues/35) |
| [M10 — Physical multi-host and multi-GPU validation](https://github.com/mohripan/capsulet-mesh-train/milestone/10) | [#36](https://github.com/mohripan/capsulet-mesh-train/issues/36) | [#37](https://github.com/mohripan/capsulet-mesh-train/issues/37) |

## M1 — Correct CPU engine and trainer

Own the complete CPU communication path: standalone C++ TCP engine, centralized FP32 SUM all-reduce, Python bindings, and a deterministic four-process trainer.

- [#3 — Build the standalone C++ core and reproducible Linux toolchain](https://github.com/mohripan/capsulet-mesh-train/issues/3)
- [#4 — Implement framed TCP transport, group startup, and bounded I/O](https://github.com/mohripan/capsulet-mesh-train/issues/4)
- [#5 — Implement centralized in-place FP32 SUM all-reduce](https://github.com/mohripan/capsulet-mesh-train/issues/5)
- [#6 — Expose safe CPU buffers through an optional pybind11 package](https://github.com/mohripan/capsulet-mesh-train/issues/6)
- [#7 — Prove four-process MLP training against the combined-batch reference](https://github.com/mohripan/capsulet-mesh-train/issues/7)
- [#8 — Validate process failures, clean builds, and the four-container trainer](https://github.com/mohripan/capsulet-mesh-train/issues/8)

## M2 — CUDA training functionality

Make GPU training use our engine through an explicit host-staged CUDA adapter while preserving the optional CPU-only build.

- [#10 — Add an explicit host-staged CUDA tensor adapter](https://github.com/mohripan/capsulet-mesh-train/issues/10)
- [#11 — Verify GPU trainer parity and document single-GPU limits](https://github.com/mohripan/capsulet-mesh-train/issues/11)

## M3 — Collective algorithms and CPU benchmarks

Implement ring and tree collectives and explain their measured behavior against a comparable CPU baseline.

- [#13 — Implement ring all-reduce with uneven and empty chunks](https://github.com/mohripan/capsulet-mesh-train/issues/13)
- [#14 — Implement tree reduction and broadcast for arbitrary worker counts](https://github.com/mohripan/capsulet-mesh-train/issues/14)
- [#15 — Publish comparable CPU benchmarks and ring-model analysis](https://github.com/mohripan/capsulet-mesh-train/issues/15)

## M4 — CUDA reductions and backward overlap

Develop asynchronous progress, native CUDA reductions, pipelined staging, and ordered gradient-bucket overlap.

- [#17 — Define asynchronous work, progress ownership, and ordered buckets](https://github.com/mohripan/capsulet-mesh-train/issues/17)
- [#18 — Implement CUDA FP32 reduction and pipelined host staging](https://github.com/mohripan/capsulet-mesh-train/issues/18)
- [#19 — Measure backward overlap on a tiny transformer](https://github.com/mohripan/capsulet-mesh-train/issues/19)

## M5 — Sharding collectives and ZeRO-1 Adam

Expose reduce-scatter/all-gather and implement simplified Adam optimizer-state sharding with replicated parameters and gradients.

- [#21 — Expose reduce-scatter and all-gather with a partition contract](https://github.com/mohripan/capsulet-mesh-train/issues/21)
- [#22 — Implement simplified ZeRO-1 Adam and measure state memory](https://github.com/mohripan/capsulet-mesh-train/issues/22)

## M6 — Consistent checkpoints and group recovery

Recover an interrupted training job by restarting its fixed-size worker group from the latest committed consistent checkpoint.

- [#24 — Commit and restore complete consistent training checkpoints](https://github.com/mohripan/capsulet-mesh-train/issues/24)
- [#25 — Restart the worker group with bounded retries and fault tests](https://github.com/mohripan/capsulet-mesh-train/issues/25)
- [#26 — Restore complete ZeRO-1 optimizer shards after group failure](https://github.com/mohripan/capsulet-mesh-train/issues/26)

## M7 — Kubernetes and exclusive token admission

Integrate existing Kubernetes orchestration with whole-group resource readiness and exclusive weighted job-token pools.

- [#28 — Design Kubernetes worker groups and physical/token admission](https://github.com/mohripan/capsulet-mesh-train/issues/28)
- [#29 — Deploy and recover distributed training worker groups on Kubernetes](https://github.com/mohripan/capsulet-mesh-train/issues/29)
- [#30 — Enforce exclusive weighted job-token pools and reconcile reservations](https://github.com/mohripan/capsulet-mesh-train/issues/30)

## M8 — Framework and transport adapters

Attach existing libraries through an MPI point-to-point transport and a versioned custom PyTorch communication backend.

- [#32 — Attach an optional MPI point-to-point transport adapter](https://github.com/mohripan/capsulet-mesh-train/issues/32)
- [#33 — Register a versioned custom PyTorch communication backend](https://github.com/mohripan/capsulet-mesh-train/issues/33)

## M9 — Native Windows portability

Build and run the engine natively on Windows through the isolated networking boundary.

- [#35 — Port the socket boundary and validate native Windows training](https://github.com/mohripan/capsulet-mesh-train/issues/35)

## M10 — Physical multi-host and multi-GPU validation

Validate real host/device topology and fair GPU scaling once external hardware is available.

- [#37 — Validate physical hosts and distinct GPUs against NCCL](https://github.com/mohripan/capsulet-mesh-train/issues/37)

## Optional research

- [#38 — Research gradient sharding after ZeRO-1](https://github.com/mohripan/capsulet-mesh-train/issues/38) — outside the required ZeRO-1 gate.

## Keeping this map useful

Every task records its prerequisites, acceptance criteria, and closure evidence. Follow those dependencies rather than issue-number order. Milestone trackers collect required work; the root tracker connects the whole project. Update this index and the roadmap when the work breakdown materially changes.
