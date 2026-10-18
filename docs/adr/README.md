# Architecture decision records

Record decisions that establish durable interfaces, ownership, numerical behavior, or failure semantics. Each ADR states its context, decision, alternatives, consequences, and revisit triggers. Implementation evidence belongs in development/validation reports and the associated issue. Preserve accepted records; supersede them with a linked ADR when a decision changes.

The existing [decision record](../DECISIONS.md) retains the initial project agreements. The [CPU MVP specification](../superpowers/specs/2026-10-16-cpu-mvp-design.md) remains the accepted M1 contract.

| ADR | Status | Scope |
| --- | --- | --- |
| [0001 — Blocking centralized FP32 communicator](0001-blocking-centralized-fp32-communicator.md) | Accepted | Issue #5: ownership, reduction order, sequence, deadlines, and terminal failures. |
| [0002 — Strict NumPy borrowing at the Python boundary](0002-python-buffer-ownership.md) | Accepted | Issue #6: optional package, storage ownership, GIL scopes, and Python/native admission. |
| [0003 — Replicated SGD adapter and independent training evidence](0003-replicated-sgd-training-contract.md) | Accepted | Issue #7: explicit gradient averaging, reference/artifact parity, and local group supervision. |
| [0004 — Blocking host-staged CUDA tensor adapter](0004-blocking-host-staged-cuda-adapter.md) | Accepted | Issue #10: optional CUDA staging, completed stream/ownership boundary and separate copy/native costs; independently reviewed and owner-authorized. |
| [0005 — Explicit CUDA training profile and independent GPU reference](0005-cuda-training-reference-contract.md) | Accepted | Issue #11: additive GPU training/evidence schema, strict FP32 policy, reference parity and shared-GPU scope; independently reviewed and owner-authorized. |
| [0006 — Explicit blocking ring all-reduce and duplex host transport](0006-blocking-ring-all-reduce.md) | Accepted | Issue #13: native selection, neighbor topology, duplex progress, uneven chunks and wire v2; independently reviewed and owner-authorized, implemented with local and exact-revision CI evidence. |
