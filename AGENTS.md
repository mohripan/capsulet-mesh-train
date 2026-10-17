# Project navigation for contributors and agents

## Read before working

1. Read `docs/ROADMAP.md` for milestone outcomes and dependencies.
2. Read `docs/DECISIONS.md` for accepted direction and proposed defaults.
3. Follow `docs/ISSUES.md` to the active GitHub issue and its dependencies.
4. Read the focused design for the milestone being changed. The first is `docs/superpowers/specs/2026-10-16-cpu-mvp-design.md`.
5. Follow the OpenSpec workflow below for the specific task. Existing milestone designs remain contracts; new task artifacts live in `docs/openspecs/`.

GitHub issues are the source of truth for live work status. Repository documents describe architecture, contracts, and evidence. Keep both consistent when a decision or completion criterion changes.

## Task workflow and local skills

- Use [OpenSpec](.agents/skills/openspec/SKILL.md) before implementing a repository change task or GitHub issue. Create or resume `docs/openspecs/<task-name>/proposal.md`, `design.md`, and `tasks.md`.
- Present the completed specs and offer personal review or a fresh, context-independent sub-agent review. Launch that reviewer only when selected/requested; it may inspect the project and research facts, but must not inherit the author's conversation or implement changes.
- Wait for the user's implementation go-ahead for the presented scope. Review alone and silence are not approval. Preserve an explicit conditional go-ahead; if the user already says to proceed or just implement the presented specs, record it and continue without another gate. Do not introduce duplicate approval flows from other planning skills.
- Apply [coding standards](.agents/skills/coding-standards/SKILL.md) during implementation. Ordinary comment blocks/docstrings in new or changed code are at most two physical lines; rationale belongs in the task's OpenSpec design. Preserve legal/tool-required content and essential contracts without unrelated comment rewrites.
- Use [live verification](.agents/skills/live-verification/SKILL.md) for each task's changed behavior, recording actual checks before checking off the task. Reserve broad integration/end-to-end suites for nightly coverage, while retaining small real-boundary checks when relevant. Existing native/binding/short-trainer regressions remain on push/PR; #8 added a separate nightly/manual M1 lane. Check the actual workflow before changing or describing its coverage.
- Use [architecture decision records](.agents/skills/architecture-decision-records/SKILL.md) for lasting decisions, cross-linked with OpenSpec. Routine changes do not need a new ADR.

These are repository-local skills. Read their files directly if automatic discovery is unavailable. The owner requested commit/push to `main` after authorized work and verification; do not add co-author trailers. Preserve unrelated local skills and configuration.

## Current phase

M1 is complete for Linux CPU processes and local containers. Issues #3–#8, tracker #2 and milestone 1 are closed after exact-revision full/manual and regular CI passed at `80903ea`. The standalone C++ TCP engine, centralized FP32 SUM, installed bindings and four-worker CPU trainer pass numerical, bounded-failure, resource and deployment acceptance. Read `docs/VALIDATION.md` and `docs/openspecs/m1-validation/verification.md` for commands, raw local/CI results and limits. Read `docs/DEVELOPMENT.md`, `docs/TCP_PROTOCOL.md`, `docs/PYTHON.md`, `docs/TRAINING.md` and ADRs 0001–0003 for current contracts.

Issue #10’s host-staged CUDA adapter is complete: `e557542` passed installed local GPU acceptance and exact-revision CPU CI. Read `docs/CUDA.md`, `docs/openspecs/host-staged-cuda-adapter/verification.md` and Accepted ADR 0004 for the tensor/stream/ownership/failure contract, original results and limits. Two workers shared the laptop GPU; this is functional integration. The C++ core and CPU trainer retain their contracts.

The active planning task is issue #11 under M2 tracker #9: GPU trainer/reference parity. Its focused draft is `docs/openspecs/cuda-mlp-trainer/`, with Proposed ADR 0005. Review route and owner go-ahead for that presented scope are pending; no GPU trainer implementation is claimed. #9 and milestone 2 remain open. Native Windows, GPU-container, physical multi-host/distinct-GPU and GPU-nightly execution need their own actual evidence.

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
- Keep C++ interfaces narrow, ownership explicit, and compiler requirements/warnings scoped to targets. Apply `.clang-format`. Run native contracts and the installed-consumer test after relevant changes.
- Record durable interface, ownership, numerical, and failure decisions in `docs/adr/`, linking the accepted specification and implementation issue. Supersede accepted records when decisions change; keep validation evidence in the corresponding reports/issues.
- The optional Python toolchain probe is not the tensor-binding API; that API belongs to issue #6.
