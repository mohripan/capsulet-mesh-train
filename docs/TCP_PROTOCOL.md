# TCP transport and wire protocol v2

Transport foundation: issue #4; ring/duplex/v2 extension: #13. All workers must use v2; v1 interoperability is intentionally unsupported. Rebuild native consumers and upgrade the complete group together. The public entry point is `meshtrain::connect_tcp(RunConfig, TcpOptions)` in `include/meshtrain/tcp.hpp`, returning an owning `std::unique_ptr<ByteTransport>`. The current adapter requires Linux and IPv4. Its private socket and resolver implementation does not enter public headers. `MESHTRAIN_ENABLE_TCP=OFF` keeps the portable foundation available; `tcp_enabled()` reports the build capability and the disabled factory throws `invalid_configuration`.

## Startup and ownership

With the default centralized algorithm, rank 0 listens at `listen_address` and the root endpoint's port. Other ranks resolve and connect to the advertised root host/port, retrying connection establishment while the original startup deadline remains. Nonroot advertised endpoints are validated and retained for diagnostics; this star does not listen on them. A one-rank group opens no sockets.

Every nonroot sends HELLO. Root checks distinct in-range ranks, protocol version, world size, run identifier, context fingerprint, float representation, payload limit and explicitly selected algorithm. Only after the entire group joins does it send READY to each nonroot. A failed startup releases all accepted sockets and its listener. The listener is also closed after successful startup. A fresh attempt needs a fresh run identifier.

With `AllReduceAlgorithm::ring`, every rank first binds its own advertised endpoint port. Root still admits the entire group over separate control connections, then sends RING_CONNECT. For three or more ranks each worker connects to its successor, accepts its predecessor and exchanges EDGE_HELLO identities. Two ranks use one separate full-duplex data socket initiated by rank 0. Each worker validates its expected neighbor and the complete identity, reports EDGES_READY to root, and waits for final READY. Listeners close after startup. Retained control connections carry no gradient payload; ring data moves directly between neighbors. Root retains O(p) control descriptors and at most two neighbor sockets. Any startup failure releases all acquired sockets/listeners. See [ring API and verification commands](RING.md).

Socket descriptors use move-only RAII. Exact I/O handles short transfers, EINTR, EOF, and EAGAIN through nonblocking sockets and `poll`. Writes use `MSG_NOSIGNAL`; the library does not install a signal handler. `TcpOptions` optionally requests kernel send/receive buffer sizes and controls TCP_NODELAY; zero buffer sizes preserve OS defaults.

Each transport permits one send, receive or duplex exchange operation at a time. A concurrent second operation throws `invalid_state`. An I/O failure permanently closes the group; later calls throw `invalid_state`. Invalid peer selection or concurrent-call rejection does not terminate an otherwise active operation.

`close()` is idempotent and may run concurrently with I/O. It removes connection ownership under a mutex, then shuts down the sockets. An active operation retains a shared socket lease until it returns, preventing descriptor reuse while cancellation wakes `poll`/`recv`/`send`. Callers must join active calls before destroying the transport. The core creates no transport progress thread.

`exchange_bytes(send_peer, outgoing, receive_peer, incoming, deadline)` progresses both exact byte spans on the caller's thread. They may have different sizes or empty directions; nonempty spans must be disjoint. The same peer/socket can serve both directions. Nonblocking sends/receives service both directions in bounded attempts; `poll` watches unfinished directions together and merges flags for a shared descriptor. This avoids cyclic blocking sends without depending on large socket buffers. No background progress thread is introduced. The base adapter method explicitly rejects unsupported duplex exchange; centralized-only adapters remain source-buildable.

## Deadlines and bounds

Startup uses one absolute `steady_clock` deadline across name resolution, connect retries, all group/edge HELLOs, topology setup and every READY. Byte operations take an absolute deadline supplied by their caller. Collective code must pass the same deadline to each transfer in one collective. Partial progress, signals, and retries never renew it. OS scheduling can delay observing expiration; this is a bounded blocking API, not a real-time scheduling guarantee.

Hostnames use `getaddrinfo` in a detached POSIX worker, avoiding a blocking DNS call on the caller's path. If the deadline expires, the call throws while the worker retains its request strings, synchronization state, and eventual result. The worker releases those resources when libc returns. The lookup itself cannot be cancelled; repeated timed-out lookups can temporarily retain threads. Destruction of the library's containing shared object must wait for such workers. Numeric IPv4 addresses skip DNS. No process-wide signals or global resolver pool are installed.

The wire codec validates fixed headers before payload allocation. Data frames are bounded by the agreed application limit, with overflow-safe FP32 count/byte consistency. HELLO has a separate 22–404 byte bound so small application payload limits remain usable. ERROR has a separate 1,024-byte maximum. The raw `ByteTransport` moves exact bytes and does not inspect application framing; the startup layer and communicator own frame validation.

## Header

Every frame starts with exactly 80 bytes. Receive/validate the eight-byte magic/version/operation prefix before reading the remaining header; a v1 prefix fails immediately. All integers are unsigned, explicitly encoded in big-endian network order; native structs are never serialized. Version 2 data payloads are little-endian IEEE-754 binary32. Only compatible hosts are accepted.

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | Magic `0x4D455348` (`MESH`) |
| 4 | 2 | Protocol version `2` |
| 6 | 2 | Operation |
| 8 | 4 | Source rank |
| 12 | 4 | Destination rank |
| 16 | 8 | Collective sequence |
| 24 | 8 | Element count |
| 32 | 8 | Payload byte length |
| 40 | 2 | Dtype |
| 42 | 2 | Reduction |
| 44 | 4 | Reserved, must be zero |
| 48 | 2 | Algorithm: centralized=1, ring=2 |
| 50 | 2 | Phase: none=0, reduce-scatter=1, all-gather=2 |
| 52 | 4 | Step within phase |
| 56 | 4 | Chunk index |
| 60 | 4 | Reserved, must be zero |
| 64 | 8 | Full logical element count |
| 72 | 8 | Element offset of this payload |

Source and destination must differ. Operations are HELLO=1, READY=2, ALL_REDUCE_REQUEST=3, ALL_REDUCE_RESULT=4, reserved ERROR=5, RING_CONNECT=6, EDGE_HELLO=7, EDGES_READY=8, RING_REDUCE=9 and RING_GATHER=10. Unknown operations/algorithms/reserved bits fail.

Controls require zero sequence/count/dtype/reduction/phase/step/chunk/full-count/offset. They carry the selected algorithm. HELLO and EDGES_READY target root; READY/RING_CONNECT originate at root. HELLO/EDGE_HELLO carry bounded identity payloads; other startup controls are empty. Ring-specific controls require ring. ERROR encoding remains reserved; failures propagate through socket closure and original deadlines rather than error-frame writes.

Data requires FP32=1, SUM=1 and `payload_bytes == count * 4`. Both the full count and payload arithmetic are checked; full logical bytes must satisfy the agreed cap even if an individual chunk fits. Centralized metadata requires phase/step/chunk/offset zero and full-count equal to count. Ring metadata requires the corresponding phase/operation and a range within the full input. The communicator independently derives the expected ranks, sequence, phase, step, chunk, full/local counts and offset before receiving payloads. It never allocates scratch from an untrusted received count. Zero-size payloads remain valid.

## Centralized collective exchange

Each nonroot sends ALL_REDUCE_REQUEST with its current sequence, followed by exactly `count * 4` payload bytes. Root receives contributions in ascending rank order and adds them into its own buffer using FP32 arithmetic and one receive scratch buffer. Once every contribution arrives, root sends ALL_REDUCE_RESULT and its summed payload to each nonroot. Empty calls exchange the same headers with zero payload bytes. Sequence begins at zero and advances only after local success; it never wraps.

One original collective deadline covers every transfer and reduction check. A mismatch or other admitted failure makes the communicator unusable; closing its transport wakes peers. Results may be partially modified on failure. Explicit close can race with a collective and leaves state closed. See [ADR 0001](adr/0001-blocking-centralized-fp32-communicator.md) for ownership, alternatives, and consequences.

## Ring collective exchange

Partition N elements into p unpadded contiguous chunks: chunk c has `N/p + (c < N%p)` elements at `c*(N/p) + min(c,N%p)`. Every rank sends to its successor and receives from its predecessor. At step s from 0 through p-2, reduce-scatter sends `(r-s) mod p` and reduces `(r-s-1) mod p`; all-gather sends `(r+1-s) mod p` and receives `(r-s) mod p`. Each step duplex-exchanges prefix, header tail and exact payload under the same collective deadline. Empty chunks still exchange headers.

Reduce-scatter receives into one largest-chunk FP32 scratch vector, then adds into the caller's chunk after both directions finish. Gather receives completed chunks directly into the caller buffer. Scratch is at most `ceil(N/p)` floats per rank. Addition order is fixed by the cyclic schedule and can differ from centralized; no cross-algorithm bitwise equivalence is promised. One rank preserves every input bit without I/O. Sequence advances once after the full all-reduce, never per phase/step. The ownership, concurrent rejection, terminal failure/partial result and local completion rules remain the centralized contract. [ADR 0006](adr/0006-blocking-ring-all-reduce.md) records the extension.

## HELLO / EDGE_HELLO payload

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | World size |
| 4 | 8 | Maximum application payload bytes |
| 12 | 4 | Representation `0x00010020` (little-endian IEEE binary32) |
| 16 | 2 | Run identifier length, 1–128 |
| 18 | 2 | Context fingerprint length, 1–256 |
| 20 | variable | Run identifier then context fingerprint |

Identifiers contain printable ASCII without a terminator. Declared lengths must exactly consume the payload. The fingerprint is supplied by the application; the transport compares it and does not derive model identity. Root closes the group on mismatch. Peer closure or each worker's own deadline bounds failure propagation.

## References and validation

The implementation follows the documented [getaddrinfo ownership](https://www.man7.org/linux/man-pages/man3/getaddrinfo.3.html), [detached thread lifecycle](https://www.man7.org/linux/man-pages/man3/pthread_create.3.html), [poll readiness semantics](https://www.man7.org/linux/man-pages/man2/poll.2.html), and [MSG_NOSIGNAL behavior](https://man7.org/linux/man-pages/man2/sendmsg.2.html). See [development commands and evidence](DEVELOPMENT.md).
