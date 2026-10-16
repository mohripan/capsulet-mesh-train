# Centralized all-reduce: issue #5

The accepted M1 specification and the user's explicit request to proceed authorize this increment. [ADR 0001](../../adr/0001-blocking-centralized-fp32-communicator.md) records its implementation decisions. The referenced writing-plans skill is not installed; this execution plan follows the accepted contract directly.

1. Reuse one portable deadline helper across transport and collectives. Preserve raw error messages when enriching errors with collective sequence/peer context.
2. Add a narrow owning communicator with a private implementation, ready/failed/closed states, single-call admission, ordered FP32 SUM, one root receive buffer, and terminal failure cleanup.
3. Validate every request/result header before payload access. Preserve empty-input participation, one-rank identity, sequence progress, absolute operation deadlines, and concurrent close.
4. Test portable lifecycle/ownership and real TCP subprocesses. Cover all required rank/count combinations with exact and bounded-random fixtures; ordered cancellation; repeated calls; malformed metadata; peer death/stalls; nonrenewing deadlines; and unusable failed communicators.
5. Exercise a four-container native collective, installed consumers, TCP-disabled builds, GCC/Clang warnings as errors, sanitizers, and the existing optional packaging proof.
6. Record commands, numerical errors, timing, and limits. Update ADR navigation and GitHub dependencies, commit/push main without a co-author trailer, and verify CI before closing #5.
