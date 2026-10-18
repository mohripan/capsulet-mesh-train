# Verification: Native ring all-reduce

Date: 2026-10-18

Status: complete. Required local verification and exact implementation-revision CI passed; #13 is closed. [Tasks](tasks.md), [review](review.md) and [authorization](proposal.md) retain the actual status.

## Planning baseline and inspection

Planning began at `5c259d3af9f15fa04af6d0444492fb0722c88f0a` on `main`, after M2 closure. `git status --short` showed only unrelated untracked `.agents/skills/find-skills/`, `.claude/` and `skills-lock.json`; preserve them.

Read the repository navigation, M1 focused design, Accepted ADRs/current phase and #12–#15 requirements. Inspected native config/communicator/transport/TCP headers, `src/communicator.cpp`, `src/tcp.cpp`, Linux socket helpers, v1 codec, native conformance/worker fixtures, CMake/install consumer, Python config construction, container entry point and actual `.github/workflows/core.yml`/`m1-validation.yml` coverage.

Observed constraints: current TCP connects only root/nonroots; listeners currently use root's port; the transport's operation lock rejects simultaneous send/receive calls; protocol v1 has a 48-byte header without phase/chunk metadata; Python constructors currently choose default native configuration. The proposed neighbor setup, one duplex operation and deliberate v2 upgrade address these actual boundaries. Existing short regression lanes remain push/PR; broad M1 acceptance remains nightly/manual.

Primary references inspected for uncertain transport/algorithm facts are linked in [design.md](design.md): Linux `poll`/socket readiness, Open MPI's send/receive scheduling explanation, and the authors' ring algorithm paper. These references inform the design; they are not runtime evidence for this engine.

## Planning checks at 00997b8

Author self-check on the PowerShell host used Python 3 from PATH for two one-off read-only checks:

- Scanned all nine changed/new Markdown files, resolving each relative Markdown link against its containing directory: 61 links resolved. All three OpenSpec artifacts and ADR 0006 have no template fields/`TBD`/`TODO` or checked implementation tasks. ADR remains Proposed; navigation and proposal consistently say review/go-ahead pending.
- Symbolically walked the design's exact partition and two-phase formulas for `p=1..9`, `N=0..65`: 594 cases passed. Each chunk contains each rank contribution exactly once after reduction, gather gives every rank every completed chunk, offsets cover the input without gaps/padding, and aggregate payload bytes equal `2*(p-1)*4*N`. This is a mathematical planning check using rank-contribution sets, not a native implementation, numerical test or process result.

`git diff --check` passed before publication. Inspected the actual regular/nightly workflow split and compared the proposed boundary to the current source. Author review explicitly covered two-rank socket aliasing, cyclic backpressure, root/neighbor startup ordering, all-empty participation, version-prefix rejection, full logical count limits, reduction order and terminal sequence behavior. At that planning snapshot review/implementation authorization was pending; the later independent review and owner instruction are recorded below.

Planning publication is a docs-only commit/push to `main` under the standing owner instruction, without co-author trailers. #13 and #12 retain unchecked implementation/completion criteria and link these draft artifacts. Review selection and implementation authorization were the next action at that snapshot.

## Independent review and authorization

Fresh `review_ring_spec` inherited no conversation and returned no blockers or required changes. Its independent 1,548-case symbolic check is planning evidence only. The owner then explicitly authorized implementation after review; [actual wording and scope](proposal.md) and [review limits](review.md) are recorded. Accepted ADR 0006 follows that actual instruction.

## Local environment and commands

Host: Windows PowerShell, with Linux/WSL x86-64 for runtime checks and Docker Desktop engine 29.6.2. WSL: GCC 15.2.0, CMake 4.2.3, Ninja; clang-format 22.1.7 on Windows. The pinned Python-free Debian native image builds with GCC 12.2.0, CMake 3.25.1 and Ninja 1.11.1. Optional installed-package images use pinned Python 3.12.15, NumPy 2.5.4 and (for the CPU trainer) Torch 2.14.1+cpu. No GPU or new runtime dependency was used.

Executed from the project root, using the corresponding Linux shell paths:

```sh
cmake -S . -B build/ring-dev -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/ring-dev -j 8
ctest --test-dir build/ring-dev --output-on-failure -R '^(core_contracts|wire_protocol|communicator_contracts)$'
ctest --test-dir build/ring-dev --output-on-failure -R '^(ring_startup|ring_collectives|installed_consumer)$'
ctest --test-dir build/ring-dev --output-on-failure -R '^ring_failures$' --output-log /mnt/c/Projects/capsulet-mesh-train/build/ring-final-failures.txt
cmake -S . -B build/ring-portable -G Ninja -DCMAKE_BUILD_TYPE=Release -DMESHTRAIN_ENABLE_TCP=OFF -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/ring-portable -j 8
ctest --test-dir build/ring-portable --output-on-failure
docker build --target native-test -t meshtrain-native-test:ring-13-final --progress plain .
docker build --target python-bindings-test -t meshtrain-python-bindings-test:ring-13 --progress plain .
docker build --target trainer-smoke-test -t meshtrain-trainer-smoke-test:ring-13 --progress plain .
python3 tools/check_ring_containers.py meshtrain-native-test:ring-13 --output build/ring-containers-13
python3 tools/check_ring_containers.py meshtrain-native-test:ring-13 --scenario worker-exit --output build/ring-container-failure-13
docker run --rm --init --entrypoint ctest meshtrain-native-test:ring-13-final --test-dir /workspace/build/native --output-on-failure -R '^fault_(startup-root-kill|collective-worker-stop)$'
```

The controller image `meshtrain-native-test:ring-13` is the prior passing native build; `ring-13-final` subsequently separates ring/M1 fault evidence directories. Container commands, resource IDs, image IDs and cleanup are retained in the original external `build/ring-containers-13` / `build/ring-container-failure-13` bundles. Docker builds reused pinned toolchain/dependency layers and rebuilt the changed source; no no-cache claim is made.

## Actual results

| Check | Result and raw evidence |
| --- | --- |
| Native standalone container | 24/24 regular CTest entries passed in 64.09 s; warning-clean Python-free build. [Build](results/native-build.txt), [complete CTest output](results/native-ctest.txt). |
| Both numerical algorithms | 188 world/count/fixture/algorithm configurations, 584 successful rank calls across 1–5 workers. Deduplicated counts are `0,1,p-1,p,p+1,2*p+1,1024,1025,131073,1048579`. All exact, FP64-reference and independently ordered FP32 checks passed. |
| Numerical maxima | Ring versus FP64: `3.0966475606e-7`; centralized versus FP64: `2.6635825634e-7`; ring versus ascending centralized arithmetic: `4.76837158203e-7`. Fixed elementwise `atol=1e-6`, `rtol=1e-5`, no tolerance changes. Every rank matches its algorithm's independent FP32 oracle bit for bit on these fixtures. |
| Round/byte/topology ownership | Observed every ring header/payload operation at direct neighbors, exact phase/step/sequence and unpadded byte counts. Live descriptor deltas match root control plus neighbor data sockets; before/after destruction counts match. Scratch is one largest-chunk vector by implementation inspection, with checked partition/extreme-size tests. |
| Duplex boundary | Seven real process groups passed: two/three-rank cycles, same socket, asymmetric/empty spans, multi-MiB messages, 4 KiB buffer requests, EINTR, concurrent rejection/close, EOF, stalled write and slow-drip original deadlines. |
| Startup | 15 real groups passed: delayed rank, algorithm/run/context/world/limit mismatch, duplicate identity, wrong/missing edge, missing readiness, edge identity/representation/version faults. A v1 peer sends only eight bytes and observes immediate rejection. |
| Collective corruption/lifecycle | 32 frame corruptions across both phases plus delay, cumulative/drip deadlines, closure, concurrent/close, full logical cap, FP32 cancellation order and 20 successful reconstructions passed. Final focused [WSL rerun](results/final-ring-failures.txt) additionally proves a four-byte application cap with real three-rank startup and empty chunks. |
| Controlled native signals | All 12 startup/reduce/gather root/worker kill/stop cases plus 20 failed reconstructions passed. [32 original parsed case records](results/native-signals.json): maximum group elapsed 1006.6 ms, unchanged one-second operation deadline/five-second independent watchdog, living survivor FD equality, reaping and listener rebinding. |
| Portable and installed native | Four TCP-disabled contracts/package checks passed; both algorithms' one-rank identity and relocated consumer work with an explicitly supplied adapter. [Portable CTest](results/portable-ctest.txt); default installed consumer in native output. |
| Python binding | Sdist-derived isolated wheel installation and all ten focused buffer/lifetime/GIL/TCP/failure checks passed. [Original build/check output](results/bindings-build.txt). Default remains centralized. |
| CPU trainer | Installed smoke passed 63 comparisons and focused checks, preserving CPU/schema-1 behavior. [Build/check output](results/trainer-build.txt), [original verification](results/cpu-trainer.json). This is the short regression profile, not renewed 100-step or GPU acceptance. |
| Four containers | Four hostname-addressed native workers pass ring and centralized conformance, empty/uneven/large cases, with no mounts or shared host PID namespace. [Healthy report](results/container-healthy.json), [ownership/cleanup](results/container-healthy-resources.json), rank logs in results. Execution 1.436 s, cleanup 0.794 s; all workers exit zero. |
| Container controller failure | Induced rank-2 exit 23 detected, surviving owned containers stopped/reaped and network removed. [Failure report](results/container-failure.json), [resource record](results/container-failure-resources.json). Execution 2.141 s, cleanup 1.307 s. This checks controller cleanup, not native fault detection. |
| M1 lane preservation | Actual CTest discovery passes nine broad entries and 24 regular entries, including all seven originals. Removing a required original entry is rejected by the selection guard. Ring signal records live separately from M1's exact 28-case collector. Two targeted original M1 signal cases pass [at the updated boundary](results/m1-boundary.txt); broad full M1/nightly acceptance was not rerun. |

## Development findings and corrections

The first v2 regression exposed an old fault fixture that changed `count` without the new matching `total_count`; it was corrected and centralized conformance reran successfully. A gather corruption fixture initially rewrote its operation to the value it already had; it now injects the opposite operation and fails at the intended peer. Prefix/header-tail handling exposed a startup injector that modified the wrong transfer; the injector now targets the actual identity bytes and verifies protocol rejection at the expected rank. These were fixture defects, not waived checks.

A debug conformance attempt with 4 KiB send buffers on both algorithms exceeded the unchanged five-second per-collective deadline on the large centralized fixture. Ring retains backpressure requests; the centralized comparison now uses OS defaults. This is functional conformance and does not claim comparable timing. The dedicated cyclic duplex checks separately exercise small send/receive buffers. No native timeout/tolerance was increased to pass. GCC 12 also found a missing direct `<algorithm>` include hidden by GCC 15's transitive includes; it was added before the passing container build.

Review of the actual nightly collector found two compatibility risks: its exact seven-entry regular-selection count and the shared fault-output directory. The guard now requires the original regular entries while allowing new focused entries, and ring uses `ring-fault-evidence`/`MESHTRAIN_RING_FAULT_OUTPUT`, preserving the nine-entry/28-result broad M1 contract.

## Provenance, publication and limits

[Behavioral source manifest](results/source.json) records 118 raw working-tree files at base `00997b8` plus the implementation. Build/controller records identify actual images; [evidence checksums](results/sha256.json) retain the selected original bytes. Implementation `aa1c13c` is pushed to main without co-author trailers; [raw/commit correspondence](results/source-commit.json) verifies all 118 raw hashes and Git-filtered committed blobs. Local package/container checks preceded final test-only four-byte-cap coverage and CMake formatting; the exact-revision CI below covers that final committed tree.

These results establish Linux CPU processes, WSL and local Linux containers. They do not establish ring Python/training selection, GPU ring behavior, performance superiority, Gloo/MPI/NCCL comparison, calibrated model accuracy, native Windows, physical multi-host or distinct-GPU validity. #14/#15 and M3 remain open. Broad nightly/manual M1 and GPU acceptance retain their existing scopes.

## Exact revision and closure

[Implementation aa1c13cf97efa5eb56e9a5c8d133c36dd59e9abc](https://github.com/mohripan/capsulet-mesh-train/commit/aa1c13cf97efa5eb56e9a5c8d133c36dd59e9abc) passed [regular CI run 39127401157](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39127401157). All three native, binding and short CPU-trainer jobs passed, including original TCP/centralized container checks and healthy/induced-exit ring containers. [Original job/step provenance](results/ci.json) pins the successful run to the implementation SHA. No behavioral source changed during closure documentation; all 118 manifest files still match.

After reading back that evidence, #13 was closed, #12's ring child and #14's dependency were checked, and the root roadmap handoff moved to #14's focused OpenSpec/review/go-ahead. #12/M3 and #14/#15 remain open. M1's stale root checkbox was reconciled against its already-closed tracker; its historical acceptance was not rerun or reclassified. The final documentation/evidence commit follows the standing main/no-co-author instruction and preserves unrelated local skills/configuration.
