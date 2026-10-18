# Design: Native ring all-reduce with uneven and empty chunks

Read [proposal.md](proposal.md) for scope/authorization and [tasks.md](tasks.md) for execution. This is #13's focused M3 algorithm/measurement design, not approval to implement #14/#15.

## Existing contracts and component boundaries

`Communicator` owns a joined `ByteTransport`, admits one blocking all-reduce, and closes the group on any admitted exception. Its sequence advances only on local success. `connect_tcp` currently builds a root star, admits one send or receive at a time, and uses shared RAII socket leases so concurrent close cannot recycle an active descriptor. All of these lifecycle rules remain.

Proposed responsibility split:

| Area | Responsibility |
| --- | --- |
| `include/meshtrain/config.hpp`, `src/config.cpp` | `AllReduceAlgorithm { centralized, ring }`; append a default-centralized field to `RunConfig`; reject unknown values. |
| `include/meshtrain/transport.hpp` | Synchronous exact-byte `exchange_bytes(send_peer, send_span, receive_peer, receive_span, deadline)` contract, using standard types. |
| `src/linux/socket.*` | Duplex progress, partial I/O, readiness, timeout and socket error handling; no collective arithmetic. |
| `src/tcp.cpp`, focused private startup helper(s) | Group join, algorithm agreement, star/control and neighbor socket ownership. No ring scheduling. |
| `src/protocol.*` or focused private ring codec | Versioned, explicitly serialized metadata; pure validation before payload admission. |
| `src/communicator.cpp`, private collective helper(s) | Common admission/deadline/sequence/failure wrapper; dispatch to centralized or ring. |
| Private ring/partition source/header | Checked partition/index calculations, framed phase scheduling and FP32 reduction. No socket descriptors. |
| Native tests and container check | Real communication and controlled faults; no training dependency. |

Keep public interfaces narrow, target-scoped warnings and installed package usage. Avoid turning `communicator.cpp` or `tcp.cpp` into a combined algorithm/network framework: extract the ring schedule and new startup logic along their distinct responsibilities. Internal filenames can follow implementation needs. Rationale stays here; ordinary new/changed code comment blocks remain at most two physical lines.

## Selection and compatibility

Selection is explicit and immutable after construction. `config.all_reduce_algorithm = AllReduceAlgorithm::ring` is supplied to both `connect_tcp(config)` and `Communicator(config, ...)`. Adopting another transport retains the existing caller obligation to use the same joined group configuration. No payload-size heuristic or fallback changes the requested algorithm.

Centralized remains the default, including in bindings whose `RunConfig` constructor does not yet expose the new field. Its root adds ranks in ascending order using one full receive buffer. Ring has its own order and scratch bound. Existing Python/CUDA staging and trainer schemas remain unchanged; the installed default path is reverified after native changes. Exposing ring in Python and changing trainer metadata need a follow-up spec, expected alongside #15's Python timing boundary.

This native library has no stable binary ABI promise. Existing callers recompile. Add the new virtual exchange method with a base implementation that throws `invalid_configuration` for an adapter that has not implemented duplex exchange; it must not perform send-then-receive. Thus existing source implementations of centralized-only adapters remain buildable. A ring call on such an adapter fails explicitly as an admitted, terminal collective failure. World size one requires no exchange and remains valid.

Use protocol v2 for both algorithms. Keep magic and explicit integer/network representation, reject old/unknown versions in the first fixed prefix before waiting for the extended header, and document coordinated worker upgrades. Do not silently accept v1 or change the wire interpretation of v1. This intentionally changes wire interoperability while retaining rebuilt default centralized API behavior. [ADR 0006](../../adr/0006-blocking-ring-all-reduce.md) records these choices; ADR 0001 remains the centralized contract.

## TCP group startup and topology

Every ring rank listens on its own configured endpoint port at its `listen_address`; advertised hosts remain routable IPv4 addresses/hostnames. Centralized retains only the root listener. Generalize the private listener helper to take the local rank's port explicitly instead of always using `endpoints[0]`. World size one opens no sockets.

Use one original startup deadline through the following stages:

1. For ring, each rank binds its listener before joining root. Nonroots connect to root and send group HELLO on separate control sockets, just as the star does. Root validates distinct in-range identities, world, run ID, fingerprint, representation, payload cap and selected algorithm for the whole group.
2. After all HELLOs agree, root sends `RING_CONNECT` on each control connection. It does not send final READY yet. All ring listeners remain open. No worker starts neighbor handshakes before this command.
3. For `p >= 3`, rank `r` connects to successor `(r+1) mod p`, then accepts predecessor `(r-1) mod p`. All listeners already exist, so TCP connection establishment does not require the application to accept first. Do not wait for an application acknowledgment before accepting the incoming edge. There is exactly one incoming edge per listener in a valid launch.
4. For `p == 2`, rank 0 initiates one data connection to rank 1. Rank 1 only accepts it; both directions use that socket. It is separate from their control socket. Do not create two ambiguous connections to the same peer.
5. Once data sockets exist, use duplex exchange for `EDGE_HELLO` headers and bounded identity payloads: send to successor, receive from predecessor (the same peer/socket at `p == 2`). Validate the exact expected neighbor, run/world/fingerprint/algorithm/limit/representation. An edge socket is installed only after validation. The existing application fingerprint is still application supplied, not a cryptographic authentication mechanism.
6. Nonroots send `EDGES_READY` to root. Root checks these plus its own edge completion and then sends final READY to every worker. Only this releases the group to collective operations. Listeners close; retain root control sockets until close and keep data sockets in a distinct peer table. Centralized retains the simpler HELLO/READY star path.

No all-reduce data is relayed through control sockets. Ring has `p` data connections (`p >= 3`) plus `p-1` control connections; root retains O(p) control descriptors, others O(1). This is a deliberate first implementation trade-off, not a constant-degree resource claim for rank 0. A future control-plane redesign requires evidence rather than hiding this cost.

Startup frames/identities have fixed independent bounds, even with a four-byte application cap. Reject unexpected controls, duplicate edges/ranks, self/wrong peers and algorithm disagreements. Any startup exception releases every acquired listener/control/data socket. A missing rank/edge or mixed version is bounded by the original deadline. Closure propagates only through connected edges/control sockets and subsequent operations; there is no idle heartbeat or transactional all-ranks-return guarantee.

## Duplex byte progress and ownership

One `exchange_bytes` call owns one transport operation slot, sends exactly its outgoing span and receives exactly its incoming span before returning. Peers may differ; both may refer to one full-duplex socket. Send/receive spans must be disjoint if nonempty, may have different lengths, and are borrowed until return. Empty directions still validate peer/lifecycle/deadline but need no syscall. Ordinary simultaneous API calls remain rejected; this is not a public asynchronous work handle.

TCP acquires shared leases for both sockets under the lifecycle lock, then progresses both nonblocking directions on the caller's thread. Service each unfinished direction in bounded byte attempts and check the same deadline each loop. On EAGAIN, poll all unfinished directions together; merge POLLIN/POLLOUT when they share a descriptor. Once one direction finishes, continue the other. Handle short I/O, EINTR, spurious readiness, EOF/errors and `MSG_NOSIGNAL` with the existing categories. A zero-byte successful send is not progress. Do not spin on a completed direction or let continuously writable output starve input.

Concurrent close removes ownership and calls shutdown on control/data sockets; leases prevent descriptor reuse until the operation exits. No transport progress thread is created or left to join. A transfer failure closes all local sockets. Invalid peer/concurrent operation rejection follows the existing non-admission behavior; an admitted I/O failure is terminal. Include the actual failed direction's peer/endpoint in diagnostics. The communicator adds its sequence, phase and step without losing that peer context.

The deadlock risk is a cycle of blocking sends when messages exceed socket capacity. Duplex progress breaks that dependency; neither larger buffers nor even/odd ordering is the correctness mechanism. Open MPI's [send/receive documentation](https://docs.open-mpi.org/en/v5.0.0/man-openmpi/man3/MPI_Sendrecv.3.html) discusses the same cyclic scheduling issue. This engine implements the mechanism itself. Linux [socket readiness](https://man7.org/linux/man-pages/man7/socket.7.html) and [poll](https://man7.org/linux/man-pages/man2/poll.2.html) define the OS behavior used here.

## Partitions and exact phase schedule

Let `p` be world size and `N` the full FP32 element count (the model uses `n = 4*N` bytes). Validate `4*N <= max_payload_bytes` with checked arithmetic before allocating scratch or issuing metadata. Partition into `p` contiguous chunks without padding:

```
q = N / p
m = N % p
count(c)  = q + (c < m ? 1 : 0)
offset(c) = c*q + min(c, m)       for 0 <= c < p
```

Checked calculations ensure every range lies in `[0,N]`; their union is exactly the input. Avoid unsigned underflow/wrap in rank subtraction and intermediate `r+p` arithmetic; compute modular indices with a sufficiently wide checked type. Scratch is one FP32 vector of at most `ceil(N/p)` elements (zero for `N == 0`), released at call exit. No full input copies or allocation per round are needed.

For `p > 1`, each rank sends to successor and receives from predecessor. Step `s` ranges from zero through `p-2` in each phase:

| Phase | Outgoing chunk at rank r | Incoming chunk at rank r | Action after complete payload exchange |
| --- | --- | --- | --- |
| Reduce-scatter | `(r-s) mod p` | `(r-s-1) mod p` | Receive into scratch; add `local[incoming] += scratch` in FP32. |
| All-gather | `(r+1-s) mod p` | `(r-s) mod p` | Receive directly into the incoming chunk; no addition. |

Every step first duplex-exchanges fixed headers and validates the incoming header, then duplex-exchanges the exact outgoing/incoming payload lengths. Both spans finish before reduction or the next step. Thus a source chunk is never modified while its send borrows it. In gather the two chunk indices differ for `p > 1`. One header is still exchanged for an empty chunk; zero-size chunks never skip a round. `N == 0` still performs both phases' headers and negotiates the full count/sequence.

After reduce-scatter, rank `r` owns fully reduced chunk `(r+1) mod p`: each traversal added one new rank, exactly `p-1` additions for each element. The next phase forwards that completed chunk and then the completed chunk just received; after `p-1` steps every rank has all chunks. Each chunk is transferred `p-1` times per phase. These are internal phases, not public reduce-scatter/all-gather APIs or optimizer sharding contracts.

For example, at `p=3`, rank 0 sends chunks 0 then 2 and reduces chunks 2 then 1. It then sends completed chunks 1 then 0 and gathers chunks 0 then 2. With `N=2`, counts are `[1,1,0]` and the same schedule still applies. At `p=2`, each phase has one exchange on the shared data socket. At `p=1`, validate and advance once without I/O, preserving every input bit including special FP32 values.

## Wire protocol v2

The 80-byte v2 fixed header retains v1 fields at offsets 0–43 and zero reserved bits at 44–47. Integers remain big-endian; FP32 payloads remain little-endian IEEE binary32. Read/validate the eight-byte magic/version/operation prefix before the remaining fixed header so a live v1 peer cannot force a wait for nonexistent extension bytes.

| Offset | Bytes | Additional field |
| --- | --- | --- |
| 48 | 2 | Algorithm: centralized=1, ring=2 |
| 50 | 2 | Phase: none=0, reduce-scatter=1, all-gather=2 |
| 52 | 4 | Step within the phase |
| 56 | 4 | Chunk index |
| 60 | 4 | Reserved zero |
| 64 | 8 | Full logical element count |
| 72 | 8 | Element offset of this payload in the full input |

Retain operations HELLO=1, READY=2, ALL_REDUCE_REQUEST=3, ALL_REDUCE_RESULT=4, reserved ERROR=5. Add RING_CONNECT=6, EDGE_HELLO=7, EDGES_READY=8, RING_REDUCE=9 and RING_GATHER=10. Control frames require zero sequence/count/dtype/reduction/phase/step/chunk/full-count/offset, but carry the selected algorithm. Root controls have the expected root direction; edge HELLO has the exact neighbor direction. HELLO/EDGE_HELLO reuse the bounded identity payload (22–404 bytes) with algorithm in the header; the other startup controls have zero payload. No ERROR frame is sent on failure.

Centralized data requires phase/step/chunk/offset zero and full-count equal to count. Ring data requires the matching phase operation, algorithm, exact expected step/chunk and deterministic partition range. Each payload must satisfy `bytes == 4*count` with overflow checks. The receiver independently derives expected total count, offset and chunk length from its local buffer and schedule, rejects mismatches before payload receipt, and never allocates from a received untrusted chunk count. Require full logical bytes to satisfy the application cap even when an individual chunk would fit. Zero counts are valid; unknown values/nonzero reserved bits fail.

Keep the golden-byte codec tests and regenerate v2 fixtures deliberately. A pure header decode validates syntax/bounds; startup/collective callers enforce their expected state/direction/sequence. Wrong algorithm must fail root admission rather than deadlocking ranks in different topology setup. The wire document will describe implemented v2 and its upgrade requirement only after authorization/implementation.

## Numerical, lifecycle and failure contracts

Ring adds each chunk at its successive receiving ranks, `local = local + received`, rounding each addition in FP32. This is deterministic for a fixed group/partition/schedule; its order can differ by chunk and from ascending centralized SUM. All ranks obtain the one computed result per chunk. No implicit averaging, compensated summation, arbitrary-dtype support or general bitwise cross-algorithm reproducibility is promised. Use the explicit fixed tolerances on bounded finite random fixtures; report errors instead of widening them. Adversarial cancellation uses an independent FP32 schedule oracle. Multi-rank NaN payload bits and signed-zero ordering are not reproducibility claims; one-rank identity remains bit preserving.

One `collective_timeout` deadline begins after operation admission and covers validation, scratch allocation checks, all headers/payloads, additions and completion. Check deadline/state between bounded addition blocks as in centralized. Local success advances once after both phases; no step increments the logical sequence. Concurrent-call rejection, exhaustion, explicit close, terminal exceptions and partial-buffer ownership retain ADR 0001 semantics. No failed collective is retried, and no optimizer may consume its partial result.

A rank may complete locally before another rank reports a later failure. Immediate neighbor/control closure and each peer's original deadline provide bounds, without claiming consensus or background monitoring. External test controllers add a separate whole-group watchdog and reap only their owned children. Original fault diagnostics remain distinguishable from watchdog cleanup.

## M3 measurement boundary and follow-ups

This issue establishes functional conformance, direct-neighbor payload flow and exact logical byte/round counts. Do not turn its conformance timings into performance results. Under divisible balanced chunks, each rank sends approximately `2*(p-1)/p*n` payload bytes across `2*(p-1)` algorithm rounds. Uneven chunks cause per-rank imbalance; frame bytes, separate header/payload exchanges, reductions and polling are additional costs. At `N < p` many rounds carry only metadata.

The roadmap model `T_ring ≈ 2*(p-1)*alpha + 2*(p-1)/p*n/B` is a hypothesis for later calibration. Use full-input bytes `n`, one-way effective link bandwidth `B`, and a stated per-round startup `alpha`. Our split header/payload exchange may need a different effective startup term; do not fit it away without explanation. The original [Patarasuk/Yuan paper](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf) motivates bandwidth-oriented ring communication and its latency/rounding trade-offs; it does not establish this implementation's performance.

#14 must separately specify deterministic arbitrary-size tree reduction/broadcast and its topology through the same byte/lifecycle boundary. #15 must separately specify explicit algorithm selection at each tested language boundary, centralized/ring/tree versus Gloo with matching ranks/count/dtype/warmup, aligned raw per-rank samples, maximum-rank iteration time, median/tails, independent latency/bandwidth calibration, held-out predictions and version/placement/thread/topology records. Separate native timing from Python overhead; report local/virtualized limits and observed crossover instead of asserting that ring always wins for large messages. M3 remains open until all three issues have evidence.

## Alternatives and risks

- Route ring messages through root: reuses current star but adds a centralized relay bottleneck and does not implement the intended neighbor communication experiment.
- Full mesh: simpler future arbitrary-peer algorithms but O(p²) connections/group and unnecessary endpoint/resource costs for this ring task.
- Neighbor-only rendezvous: saves root control descriptors but needs a separate whole-group identity/readiness protocol. Revisit if measured startup/resource costs justify it.
- Blocking send-then-receive or one I/O thread per direction: the former can deadlock; the latter adds lifetime/cancellation/join machinery before the asynchronous milestone. Caller-driven duplex polling fits the blocking API.
- Preserve v1 and add a second complete codec: avoids a coordinated upgrade but duplicates control/data formats during an evolving pre-stable API. Explicit v2 rejection makes incompatibility visible.

Main risks are topology startup sequencing, same-socket duplex progress, incorrect uneven offsets, lost failure context, and accidentally resetting deadlines per round. The focused checks below target each risk. There are no unresolved hardware requirements to perform local CPU verification; native Windows, physical hosts and distinct GPUs remain outside these claims. Review may revise these defaults before implementation authorization.

## Verification strategy

All commands/outcomes below are requirements, not results. Record actual commands/environment/raw logs and exact tested source revision in [verification.md](verification.md).

| Boundary | Required live observation |
| --- | --- |
| Pure config/codec/partition | Unknown enum/version/metadata rejected; golden v2 bytes and overflow bounds; ranges cover `[0,N]` once; rank/step schedule oracle includes `p=1,2,3,4,5`. |
| Real TCP duplex | Distinct peers and same peer/socket; unequal and empty lengths; multi-MiB spans and requested 4 KiB socket buffers; concurrent rejection, close, EOF, stopped reader, slow drip and original deadline. |
| Startup | Real rank listeners and control/data roles; delayed listener; wrong/mixed algorithm/version/run/context/world/limit/representation; duplicate/wrong edge, absent edge, failure before final READY; all sockets reclaimed. |
| Numerical processes | For both algorithms at `p=1..5`, deduplicated counts `0,1,p-1,p,p+1,2*p+1,1024,1025,131073,1048579`, exact and seeded bounded random fixtures; repeated calls, in-place pointer, per-rank identical ring result and sequence. Large cases exceed requested socket buffers. |
| Ring schedule/bytes | Independent schedule oracle and real transport wrapper verify neighbor destinations, both phases/steps, empty headers, full/chunk metadata and aggregate payload bytes `2*(p-1)*4*N`; central control carries no ring data. No assertion that uneven per-rank bytes are equal. |
| Fault processes | Malformed phase/step/chunk/offset/full-count/local-count/algorithm/dtype/reduction/ranks/sequence, oversized logical buffer, payload EOF/stall/slow drip, kill/stop rank 0 and a nonroot in each phase, close during duplex and rejected concurrent collective. |
| Resource/deadline | At least 20 small successful/failing recreate-and-rebind cycles; descriptor counts before/after destruction while process lives. With one-second native timeouts, externally controlled groups finish/reap within five seconds; successful delayed-rank fixtures stay within the original deadline, cumulative slow steps do not renew it. Record actual elapsed times/categories. |
| Installed/deployment | Standalone warning-clean native build and TCP-off installed consumer; sdist-derived default Python binding/short CPU trainer regressions; a four-container ring group with advertised hostnames, uneven/empty and large cases, bounded controller and cleanup. |

Choose faults with explicit synchronization at the tested phase so the intended path is exercised; do not replace communication with pure mocks. Test doubles may corrupt one real frame or count bytes. Preserve existing centralized rank-order fixtures. Keep the regular native/binding/short-trainer jobs on push/PR; add small ring boundary checks there with bounded runtime. The broad M1 acceptance remains nightly/manual; no broad new end-to-end sweep, GPU runner or performance job is introduced here.
