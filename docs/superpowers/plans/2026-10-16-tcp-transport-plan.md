# TCP transport: issue #4

The accepted CPU MVP specification authorizes this increment. Keep the native core independent of frameworks.

1. Add an explicit 48-byte protocol header and bounded HELLO payload codec. Validate versions, reserved fields, operation/type/count/length combinations before allocating payloads.
2. Add a private Linux socket owner and exact nonblocking I/O with absolute deadlines, EINTR handling, and MSG_NOSIGNAL. Resolve IPv4 names in an owned detached worker; retain resolver state until libc completion even if the caller times out.
3. Implement fixed-membership star startup: root accepts distinct ranks, validates identity/limits/representation, then sends READY to the complete group. Workers retry root connections within the original startup deadline.
4. Return a framework-free ByteTransport. Serialize data operations, allow concurrent close, and keep a shared socket lease alive until an active operation exits so descriptor reuse cannot corrupt cancellation.
5. Test wire corruption and real subprocess communication, fragmentation, oversized lengths, identity mismatch, missing/stalled peers, signal interruption, cancellation, and repeated cleanup. Verify hostname communication between two Linux containers.
6. Record the wire/concurrency contracts, commands, and limits; verify native/package builds and CI; update issue #4 and push without a co-author trailer.

Implementation review replaced the initially considered GNU `getaddrinfo_a` callback with a detached POSIX resolver worker. GNU enqueue failures have subtle completion-notification behavior; explicit worker ownership keeps failure cleanup straightforward and removes the libanl dependency. A linker-wrapped slow resolver test verifies timeout plus late-result cleanup without changing production interfaces.
