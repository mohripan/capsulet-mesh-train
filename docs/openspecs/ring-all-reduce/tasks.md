# Tasks: Native ring all-reduce with uneven and empty chunks

Implementation status: independently reviewed and authorized in [proposal.md](proposal.md). Implementation and runtime verification are in progress; actual evidence is recorded in [verification.md](verification.md).

- [x] **Complete selected review and record the actual authorization.**
  - Outcome: resolve blocking findings in all affected artifacts; record review coverage/limits, instruction/date/scope and any conditions. Accept ADR 0006 only when supported by the owner's decision.
  - Verification: inspect cross-links and review disposition; confirm authorization covers the presented scope, including explicit wire v2 compatibility. A conditional go-ahead needs no repeat gate once its review conditions are met.
  - Evidence: [fresh independent review](review.md) found no blockers/required changes; the owner's 2026-10-18 conditional instruction is recorded in the proposal. ADR 0006 accepted under that authorization.

- [x] **Add explicit native selection, duplex boundary and v2 codec contracts.**
  - Outcome: default-centralized enum/config validation, source-compatible unsupported exchange default, v2 fixed prefix/header/control/ring metadata validation and golden fixtures. Keep ordinary comments short.
  - Live verification: build with target warnings as errors; run `core_contracts`, `wire_protocol`, `communicator_contracts` and TCP-disabled package checks. Invalid enums, v1/unknown versions, malformed controls, full-count overflow/cap and zero-length data must have explicit expected categories before payload allocation.
  - Evidence: Core/codec/adoption tests pass in warning-clean GCC 12/15 builds; four TCP-disabled contracts and installed-consumer checks pass. See [actual results](verification.md#actual-results).

- [x] **Implement caller-driven duplex TCP progress with socket leases.**
  - Outcome: one operation admission, exact independent send/receive offsets, both descriptors progressed under one deadline, same-socket flags merged, close/error cleanup and useful peer context.
  - Live verification: real two/three-process byte exchange with requested 4 KiB buffers and multi-MiB, unequal/zero spans; same-socket two-rank case; partial transfers/EINTR, EOF/stall, concurrent rejection and close. Show original-deadline expiry under continuous partial progress and no deadlock on large cyclic exchanges.
  - Evidence: Seven real duplex groups pass, including tiny buffers, same-socket progress, EINTR, partial-progress timeout and close; [raw native output](results/native-ctest.txt).

- [x] **Join and validate the complete ring topology before READY.**
  - Outcome: local-rank listeners, root HELLO/algorithm admission, RING_CONNECT, validated neighbor edges, EDGES_READY/final READY, separate control/data ownership and cleanup. Centralized's star remains its topology.
  - Live verification: actual startup at 1–5 ranks, delayed listener, two-rank single data socket, odd ring, duplicate/missing/wrong edges, mixed algorithm/version and identity mismatches. Inject failures before READY and prove all listeners/control/data sockets are released/rebindable under the original startup deadline.
  - Evidence: Fifteen real startup groups pass, including root/edge identity faults and immediate v1-prefix rejection; [raw native output](results/native-ctest.txt).

- [x] **Implement uneven reduce-scatter and all-gather in the native collective layer.**
  - Outcome: checked unpadded partitions, private schedule, one largest-chunk scratch, framed duplex steps, bounded FP32 addition, unchanged common admission/sequence/failure lifecycle.
  - Live verification: run both algorithms at 1–5 ranks over the design's complete count matrix, exact/random inputs and repeated calls. Independently generate expected SUM and FP32 order fixtures; compare ring/centralized, all ranks and in-place storage using the fixed thresholds. Inspect real frame/byte observations to establish neighbor flow, empty participation and `2*(p-1)*4*N` aggregate payload bytes.
  - Evidence: 188 numerical configurations/584 successful rank calls pass at fixed tolerances; independent FP32 order, direct-neighbor bytes/rounds, live descriptor topology and cancellation fixtures pass. [Results](verification.md#actual-results).

- [x] **Establish ring failure, cancellation and resource conformance.**
  - Outcome: no failed group reuse or sequence increment; explicit close terminal; originating fault remains useful and all owned workers stop/reap.
  - Live verification: corrupt actual frames for phase/step/chunk/offset/count/algorithm/rank/sequence/dtype/reduction mismatches; synchronize kill/stop at both phases for root/nonroot, test slow payload/cumulative deadlines, concurrent collective rejection and close during duplex. Use one-second native deadlines and a five-second whole-group watchdog. Run at least 20 successful/failing recreate/rebind cycles with descriptor counts in living workers.
  - Evidence: 32 metadata corruptions, cumulative/drip/close/cap checks, 20 successful reconstructions, 12 controlled signal cases and 20 failed groups pass. [Original signal records](results/native-signals.json) and [final focused checks](results/final-ring-failures.txt).

- [x] **Verify deployment and compatibility at the installed boundaries.**
  - Outcome: warning-clean standalone builds, relocated consumer, optional Python packaging/default trainer and small ring container group pass. No Python API/schema/CUDA changes.
  - Live verification: run default native CTest excluding broad `m1_integration`, repeat portable CTest with `MESHTRAIN_ENABLE_TCP=OFF`, run sdist-derived binding checks and short CPU trainer, and bounded four-container ring exchange using per-rank hostnames/listeners. Extend only focused regular CI/deployment checks needed for the new boundary; keep nightly/manual broad acceptance separate. Confirm scoped container/network cleanup on success and induced failure.
  - Evidence: Native 24/24, portable 4/4, isolated binding checks, 63 short-trainer comparisons and healthy/failing four-container scoped cleanup pass. [Commands/raw evidence](verification.md#local-environment-and-commands). Exact-revision CI remains the publication task.

- [ ] **Reconcile contracts, publish evidence and close only verified work.**
  - Outcome: document implemented v2 upgrade, explicit C++ selection, topology/duplex/numerical/lifecycle/scratch rules and supported environments; reconcile OpenSpec/ADR/navigation and GitHub #13/#12 status. Keep #14/#15/M3 and hardware claims open.
  - Live verification: links/status/diff checks; commit/push to `main` without co-author trailers; run/wait for regular CI on the exact implementation revision and record its result before closing #13. Confirm remote HEAD and preserve unrelated local configuration.
  - Evidence: pending; tested revision, CI URL, final commands/raw results/limitations and issue reconciliation.

## Completion

Implemented with required local proof. Exact implementation-revision regular CI, publication and issue reconciliation remain in the last task. No benchmark or hardware acceptance beyond the report is claimed.
