# ADR 0006: Explicit blocking ring all-reduce and duplex host transport

Date: 2026-10-18

Status: Proposed

OpenSpec: [ring all-reduce design](../openspecs/ring-all-reduce/design.md)

Issue/source: [#13](https://github.com/mohripan/capsulet-mesh-train/issues/13), under [M3 #12](https://github.com/mohripan/capsulet-mesh-train/issues/12).

## Context

The standalone engine currently owns centralized FP32 SUM over a Linux TCP star. A real ring requires neighbor connectivity and progress of both byte directions without cyclic blocking sends. Odd worker counts and empty/uneven chunks must preserve sequence, numerical ownership and failure bounds. [ADR 0001](0001-blocking-centralized-fp32-communicator.md) defines the retained default centralized contract; ADRs 0002–0005 retain their binding/training/device contracts.

## Decision and rationale

Propose an explicit immutable native `RunConfig` algorithm choice, defaulting to centralized, and a private ring reduce-scatter/all-gather schedule. Ring partitions the full input contiguously without padding, receives at most one largest chunk in scratch, and uses FP32 addition with its documented per-chunk order. All-reduce remains blocking, in-place SUM with no implicit averaging. Internal phases do not create public sharding APIs.

Extend `ByteTransport` with exact synchronous duplex exchange. A default unsupported implementation preserves source builds of centralized-only adapters and fails explicitly on a ring call. Linux TCP progresses both nonblocking directions on the caller's thread with one operation slot, a shared absolute deadline, RAII leases and concurrent-close cancellation. No background thread or third-party collective is introduced.

Keep star topology for centralized. Ring first validates the complete group over root control connections, then establishes separate successor/predecessor data sockets and releases final READY only after all edges validate. Two ranks share one duplex data socket. Retaining root control sockets costs O(p) root descriptors but prevents ring payload relaying and avoids inventing decentralized rendezvous in the first algorithm extension.

Use explicit wire v2 metadata for algorithm, phase, step, chunk, total count and range, validated before payload receipt. Workers must upgrade together; mixed v1/v2 peers reject rather than negotiate. Rebuilt default centralized consumers retain their algorithm/arithmetic behavior. No stable native ABI or old-wire interoperability is promised.

Common communicator ownership, local sequence advancement, concurrent rejection, one-deadline execution and terminal partial-buffer failures remain ADR 0001 rules. Python configuration/training defaults remain centralized until a separately specified selection boundary is added. The core continues to build independently of Python, CUDA and orchestration.

## Alternatives considered

- Relay ring traffic through the existing star: avoids new connections but creates the root bottleneck the algorithm experiment is meant to remove.
- Full mesh: enables arbitrary peers immediately, with O(p²) connections that ring does not need.
- Neighbor-only group join: reduces root descriptors but adds whole-group identity/readiness machinery beyond the current root-based contract.
- Blocking send-then-receive: can deadlock for large cyclic transfers. Per-direction progress threads avoid that cycle but introduce ownership/join costs ahead of the async milestone.
- A second complete v1/v2 wire stack: retains old worker interoperability at the cost of duplicated formats in an evolving library. Explicit version rejection is easier to audit.
- Automatically select ring for large inputs: needs calibrated evidence from #15 and would make numerical order depend on message size without an explicit policy.

## Consequences

Ring traffic uses direct neighbor data links and has an O(ceil(N/p)) scratch bound per rank. Startup and protocol become more complex, wire compatibility changes, and fixed overhead grows with worker count. Reduction order differs from centralized, so bounded finite random conformance uses fixed numerical tolerances rather than bitwise equality across algorithms.

Real-process backpressure, odd/empty/uneven cases, edge identity, malformed phases, cancellation, deadlines, cleanup and a small container group are required evidence. No speedup, physical-host validity, ring trainer parity, GPU scaling or M3 completion follows merely from implementing this native algorithm. #14/#15 and M5 retain their own scopes.

## Revisit triggers

Measured rendezvous/descriptor costs, new arbitrary-peer algorithms/transports, stable wire/ABI requirements, public reduce-scatter/all-gather, asynchronous work/overlap, new dtypes, or a validated adaptive-selection policy may require a successor decision. Preserve the standalone core and exact ownership/failure contracts.

## Decision record

Drafted with #13's OpenSpec on 2026-10-18. Review route and owner implementation go-ahead for this scope are pending. This proposal supplements ADR 0001 for explicit ring selection; it does not supersede the accepted centralized default. No ring implementation or validation is claimed.
