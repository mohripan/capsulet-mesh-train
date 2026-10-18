# Proposal: Native ring all-reduce with uneven and empty chunks

Date: 2026-10-18

Status: independently reviewed, owner-authorized and implemented; local verification passed, exact-revision CI/publication pending.

Source: [issue #13](https://github.com/mohripan/capsulet-mesh-train/issues/13), under [M3 tracker #12](https://github.com/mohripan/capsulet-mesh-train/issues/12). M1/#5 and M2 are complete. The owner's “All right. Please continue!” after the M2 handoff starts this next OpenSpec. The previous conditional review/implementation instruction was recorded for #11, not this unpresented scope.

## Problem and outcome

The native engine currently implements centralized SUM through root-only TCP connections. Add our own ring all-reduce, composed internally of reduce-scatter and all-gather, through the host-byte transport boundary. Establish correctness and bounded failure on real Linux processes and a small container group before comparing performance in #15.

The change is architectural: algorithm selection, neighbor connectivity, duplex byte progress and collective metadata must agree. The standalone C++ library remains the core. Centralized remains the default and retains its ascending-rank FP32 arithmetic.

## Scope

- Add an explicit C++ `AllReduceAlgorithm` selection in `RunConfig`, fixed for the communicator's lifetime: `centralized` or `ring`.
- Extend the byte boundary with one synchronous duplex exchange, implemented in Linux TCP using nonblocking I/O and `poll` on the calling thread. Retain exclusive operation admission, ownership and cancellation.
- Keep centralized's star. Ring startup uses root control connections plus separate neighbor data connections; rank 0 never relays another ring rank's gradient payload.
- Introduce wire protocol v2 for both algorithms, with algorithm/phase/step/chunk/full-count metadata and bounded startup identity validation. Mixed v1/v2 workers fail; there is no compatibility negotiation. Existing consumers rebuild against the changed native library.
- Implement uneven partitions and the two ring phases in private C++ components, using at most one largest-chunk FP32 scratch buffer per rank.
- Add focused real-process conformance, malformed-frame, delayed-rank, cancellation/resource and small four-container ring checks. Retain existing native/package/binding/short CPU trainer regression coverage.
- Record the durable contract in Accepted [ADR 0006](../../adr/0006-blocking-ring-all-reduce.md). The [design](design.md) also sets M3 measurement boundaries for follow-up specs.

Excluded: tree implementation (#14), comparative benchmark/calibration/report (#15), automatic algorithm choice, public reduce-scatter/all-gather or ZeRO (M5), Python algorithm selection/trainer artifact changes, CUDA changes/overlap, background progress threads, new transport libraries, recovery, orchestration, native Windows and physical-host/distinct-GPU validation. Existing Python/CPU/CUDA entry points continue to select centralized by default; their historical evidence is not ring evidence.

## Acceptance

1. A rebuilt standalone C++ consumer can explicitly select ring using the existing communicator and TCP factory. No Python, PyTorch, CUDA, MPI or third-party collective dependency is introduced. Invalid enum values fail configuration; unsupported duplex adapters fail explicitly without centralized fallback.
2. Real processes at world sizes 1–5 produce correct in-place FP32 SUM with fixed rank membership. Cover zero total count, empty individual chunks, counts below the world size, uneven counts, repeated calls and messages much larger than requested socket buffers. Only local success increments the collective sequence, once per complete all-reduce.
3. Exact integer-valued fixtures match centralized and an independently generated reference exactly. Seeded bounded random fixtures pass elementwise `abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)` against an FP64 reference, and ring versus centralized with the same threshold. All ring ranks receive the same finite FP32 values. Cancellation fixtures verify the documented ring reduction order rather than claiming cross-algorithm bitwise equality.
4. Headers agree on algorithm, phase, step, chunk, total/local counts, offsets, sequence, ranks, dtype and reduction before the associated payload is received. No padded elements are transferred or reduced. The full logical payload, not just each chunk, must satisfy `max_payload_bytes` and overflow checks.
5. Duplex exchange completes with tiny requested socket buffers, including two ranks sharing one data socket, odd rings and asymmetric/zero byte lengths. It must not depend on all sends fitting in TCP buffers. One original monotonic collective deadline covers all rounds, reductions and completion.
6. Delayed ranks can succeed within that deadline. Missing, mismatched, disconnected, stopped or malicious peers fail within explicit startup/collective deadlines plus a documented test scheduling allowance. Concurrent-call rejection preserves the active operation; explicit close wakes exchange; failed calls cannot be reused. Every owned child is reaped and successful/failed reconstructions leave no socket/listener leaks.
7. Ring data moves directly between neighbors in the four-container check, using advertised rank hostnames and per-rank listeners. Record exact commands, versions, count/world matrix, errors and raw outcomes. Label WSL/local-container evidence accurately.
8. Focused checks, existing affected regressions, installed-consumer/TCP-disabled checks and exact implementation-revision regular CI pass before #13 closes. #12/M3 stays open for #14 and #15. Do not close hardware or benchmark work from functional conformance.

## Review and authorization

- Review route: fresh independent sub-agent `review_ring_spec`, started with `fork_turns="none"`, raw requirements and exact artifact paths, without the author's conversation or desired verdict.
- Findings and disposition: [independent review](review.md) completed without blockers or required changes. Author planning self-check is recorded in [verification.md](verification.md).
- User go-ahead: on 2026-10-18 the owner said “Okay. Let's continue. I guess now we can continue from before. You can use sub-agent and then implement after the review.” This authorizes the presented #13 scope after independent review and resolution of findings, including its explicit wire v2 upgrade. Do not introduce another gate for corrections within this scope.
- ADR 0006 is Accepted after the review condition was satisfied under the owner's actual instruction. Accepted ADRs 0001–0005 remain historical contracts; the new record extends algorithm selection without changing the default centralized arithmetic/lifecycle rules.

## Delivery

Native selection, duplex TCP, validated neighbor startup, wire v2 and unpadded ring phases are implemented. Required local process/fault/resource/package/container checks pass; [actual evidence and limits](verification.md) accompany the [design](design.md), [tasks](tasks.md) and Accepted ADR 0006. Exact implementation-revision CI and publication remain before #13 closes. No performance improvement is claimed.
