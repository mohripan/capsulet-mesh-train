# Project navigation for contributors and agents

## Read before working

1. Read `docs/ROADMAP.md` for milestone outcomes and dependencies.
2. Read `docs/DECISIONS.md` for accepted direction and proposed defaults.
3. Follow `docs/ISSUES.md` to the active GitHub issue and its dependencies.
4. Read the focused design for the milestone being changed. The first is `docs/superpowers/specs/2026-10-16-cpu-mvp-design.md`.

GitHub issues are the source of truth for live work status. Repository documents describe architecture, contracts, and evidence. Keep both consistent when a decision or completion criterion changes.

## Current phase

Planning. The library-first architecture is approved. The detailed CPU MVP specification is a draft for user review; implementation has not started. The current planning work may document the design and publish the agreed roadmap/backlog. Review the focused specification with the user before beginning implementation. Later subsystems need their own focused design before implementation.

## Project boundaries

- C++ is the core: our transport, collective scheduling, and reduction algorithms belong there.
- The first transport uses raw TCP sockets. Put platform-specific networking behind a boundary that can later support Windows and additional transports.
- The standalone library builds without Python, PyTorch, Kubernetes, or CUDA. Bindings and optional adapters depend on the core.
- Our collective path must not delegate to `torch.distributed.all_reduce`, `MPI_Allreduce`, or `ncclAllReduce`. Those belong in clearly identified baselines or explicitly selected adapters.
- GPU training is the destination. CPU-first is a correctness step; basic CUDA integration comes before advanced CPU optimization or Kubernetes integration.
- Kubernetes provides orchestration. Token admission is a separate policy layer; a label or token allocation does not guarantee physical CPU/GPU availability.
- Initial token pools are exclusive, non-preemptive, and charged per complete training job. No borrowing between pools.
- Workflow DAG execution is not part of the agreed project. Airflow was an analogy for retry and artifact reuse.

## Validation and reporting

- Match the same initial parameters, combined batches, loss normalization, and optimizer settings when comparing distributed and reference training.
- Test actual process communication, awkward tensor sizes, odd worker counts, and bounded failure behavior. Pure in-process mocks alone do not establish distributed correctness.
- Label evidence precisely: CPU multi-process, Linux containers, single-GPU integration, native Windows, physical multi-host, and distinct multi-GPU tests are different claims.
- Keep hardware-dependent validation open until the required hardware has been used. Sharing the laptop GPU among processes is not a multi-GPU benchmark or a valid substitute for a distinct-GPU NCCL baseline.
- Record exact commands, environment, raw results, and tolerances when closing an implementation or benchmark issue.
- Milestones are navigation aids with no deadlines. Work from the earliest unmet dependency rather than assuming issue number is execution order.
- Preserve unrelated local changes, including locally installed skills and agent configuration.
