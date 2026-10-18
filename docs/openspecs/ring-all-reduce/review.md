# Independent review: ring all-reduce

Date: 2026-10-18

Reviewer: fresh `review_ring_spec`, `fork_turns="none"`, read-only. It received raw #13 requirements and exact artifacts, without the author's conversation or desired verdict.

## Result

No blocking findings or required specification changes. The reviewer inspected proposal/design/tasks, Proposed ADR 0006, existing native interfaces/implementation, accepted contracts, actual CI and #12/#13.

Coverage: uneven/empty partitions and phase indices; in-place ownership/scratch; whole-group admission and neighbor startup; two-rank socket aliasing and duplex progress; v2 compatibility/metadata; sequence, cancellation and original deadlines; numerical order; focused process/fault/resource/package/container checks; centralized defaults and follow-up boundaries.

An independent read-only symbolic check passed 1,548 cases (`p=1..12`, `N=0..128`): unpadded coverage, each rank contribution exactly once, identical final chunk order across ranks and aggregate payload count `2*(p-1)*N`. The reviewer checked primary Open MPI send/receive and Linux poll references.

## Disposition and limits

No spec corrections required. The owner's conditional instruction in [proposal.md](proposal.md) authorizes implementation after this review; the review itself grants no approval. ADR 0006 is accepted under that instruction. Symbolic consistency is not runtime correctness or performance: actual native/process/failure/package/container/exact-revision CI evidence remains required.
