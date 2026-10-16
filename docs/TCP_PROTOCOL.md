# TCP transport and wire protocol v1

Implemented by issue #4. The public entry point is `meshtrain::connect_tcp(RunConfig, TcpOptions)` in `include/meshtrain/tcp.hpp`, returning an owning `std::unique_ptr<ByteTransport>`. The current adapter requires Linux and IPv4. Its private socket and resolver implementation does not enter public headers. `MESHTRAIN_ENABLE_TCP=OFF` keeps the portable foundation available; `tcp_enabled()` reports the build capability and the disabled factory throws `invalid_configuration`.

## Startup and ownership

Rank 0 listens at `listen_address` and the root endpoint's port. Other ranks resolve and connect to the advertised root host/port, retrying connection establishment while the original startup deadline remains. Nonroot advertised endpoints are validated and retained for diagnostics; this star does not listen on them. A one-rank group opens no sockets.

Every nonroot sends HELLO. Root checks distinct in-range ranks, protocol version, world size, run identifier, context fingerprint, float representation, and payload limit. Only after the entire group joins does it send READY to each nonroot. A failed startup releases all accepted sockets and its listener. The listener is also closed after successful startup. A fresh attempt needs a fresh run identifier.

Socket descriptors use move-only RAII. Exact I/O handles short transfers, EINTR, EOF, and EAGAIN through nonblocking sockets and `poll`. Writes use `MSG_NOSIGNAL`; the library does not install a signal handler. `TcpOptions` optionally requests kernel send/receive buffer sizes and controls TCP_NODELAY; zero buffer sizes preserve OS defaults.

Each transport permits one send or receive operation at a time. A concurrent second operation throws `invalid_state`. An I/O failure permanently closes the group; later calls throw `invalid_state`. Invalid peer selection or concurrent-call rejection does not terminate an otherwise active operation.

`close()` is idempotent and may run concurrently with I/O. It removes connection ownership under a mutex, then shuts down the sockets. An active operation retains a shared socket lease until it returns, preventing descriptor reuse while cancellation wakes `poll`/`recv`/`send`. Callers must join active calls before destroying the transport. The core creates no transport progress thread.

## Deadlines and bounds

Startup uses one absolute `steady_clock` deadline across name resolution, connect retries, all HELLOs, and every READY. Byte operations take an absolute deadline supplied by their caller. Collective code must pass the same deadline to each transfer in one collective. Partial progress, signals, and retries never renew it. OS scheduling can delay observing expiration; this is a bounded blocking API, not a real-time scheduling guarantee.

Hostnames use `getaddrinfo` in a detached POSIX worker, avoiding a blocking DNS call on the caller's path. If the deadline expires, the call throws while the worker retains its request strings, synchronization state, and eventual result. The worker releases those resources when libc returns. The lookup itself cannot be cancelled; repeated timed-out lookups can temporarily retain threads. Destruction of the library's containing shared object must wait for such workers. Numeric IPv4 addresses skip DNS. No process-wide signals or global resolver pool are installed.

The wire codec validates fixed headers before payload allocation. Data frames are bounded by the agreed application limit, with overflow-safe FP32 count/byte consistency. HELLO has a separate 22–404 byte bound so small application payload limits remain usable. ERROR has a separate 1,024-byte maximum. The raw `ByteTransport` moves exact bytes and does not inspect application framing; the startup layer and communicator own frame validation.

## Header

Every frame starts with exactly 48 bytes. All integers are unsigned, explicitly encoded in big-endian network order; native structs are never serialized. Version 1 data payloads are little-endian IEEE-754 binary32. Only compatible hosts are accepted.

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | Magic `0x4D455348` (`MESH`) |
| 4 | 2 | Protocol version `1` |
| 6 | 2 | Operation |
| 8 | 4 | Source rank |
| 12 | 4 | Destination rank |
| 16 | 8 | Collective sequence |
| 24 | 8 | Element count |
| 32 | 8 | Payload byte length |
| 40 | 2 | Dtype |
| 42 | 2 | Reduction |
| 44 | 4 | Reserved, must be zero |

Source and destination must differ. Operation values are HELLO=1, READY=2, ALL_REDUCE_REQUEST=3, ALL_REDUCE_RESULT=4, ERROR=5. Unknown operations fail. Control frames require zero sequence/count/dtype/reduction. HELLO requires destination 0; READY requires source 0 and an empty payload. Data frames require dtype FP32=1, reduction SUM=1, and `payload_bytes == count * 4`. Count zero is valid. The communicator enforces expected ranks, operation direction, matching counts, and sequence before receiving payloads. ERROR encoding remains reserved; M1 propagates failures through transport closure and each peer's deadline.

## Centralized collective exchange

Each nonroot sends ALL_REDUCE_REQUEST with its current sequence, followed by exactly `count * 4` payload bytes. Root receives contributions in ascending rank order and adds them into its own buffer using FP32 arithmetic and one receive scratch buffer. Once every contribution arrives, root sends ALL_REDUCE_RESULT and its summed payload to each nonroot. Empty calls exchange the same headers with zero payload bytes. Sequence begins at zero and advances only after local success; it never wraps.

One original collective deadline covers every transfer and reduction check. A mismatch or other admitted failure makes the communicator unusable; closing its transport wakes peers. Results may be partially modified on failure. Explicit close can race with a collective and leaves state closed. See [ADR 0001](adr/0001-blocking-centralized-fp32-communicator.md) for ownership, alternatives, and consequences.

## HELLO payload

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
