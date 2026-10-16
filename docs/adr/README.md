# Architecture decision records

Record decisions that establish durable interfaces, ownership, numerical behavior, or failure semantics. Each ADR states its context, decision, alternatives, consequences, and revisit triggers. Implementation evidence belongs in development/validation reports and the associated issue. Preserve accepted records; supersede them with a linked ADR when a decision changes.

The existing [decision record](../DECISIONS.md) retains the initial project agreements. The [CPU MVP specification](../superpowers/specs/2026-10-16-cpu-mvp-design.md) remains the accepted M1 contract.

| ADR | Status | Scope |
| --- | --- | --- |
| [0001 — Blocking centralized FP32 communicator](0001-blocking-centralized-fp32-communicator.md) | Accepted | Issue #5: ownership, reduction order, sequence, deadlines, and terminal failures. |
