# Centralized FP32 SUM validation — 2026-10-16

Issue: [#5](https://github.com/mohripan/capsulet-mesh-train/issues/5). Decision: [ADR 0001](../adr/0001-blocking-centralized-fp32-communicator.md). Build commands and pinned environments: [DEVELOPMENT.md](../DEVELOPMENT.md). Unabridged output from the recorded GCC process run: [centralized-output.txt](2026-10-16-centralized-output.txt).

## Numerical behavior

Each numerical group reuses one communicator for 14 calls: exact and bounded-random fixtures for counts `0, 1, 3, 1024, 1025, 16384, 131073`. The five groups use 1, 2, 3, 4, and 5 workers, yielding 70 numerical configurations. The last size transfers 524,292 payload bytes per rank with a requested 4,096-byte send buffer. Every rank checks its result and successful sequence advancement. Input storage remains owned by the caller.

Exact inputs are small representable integers, and every result matches the unaveraged SUM exactly. Random inputs are deterministic FP32 samples in approximately `[-1, 1]`, generated with `std::mt19937` from rank/iteration seeds. The independent reference accumulates those FP32 inputs in double precision. Acceptance is elementwise `abs(actual - reference) <= 1e-6 + 1e-5 * abs(reference)`.

Recorded maxima, identical across the ranks in each group:

| Workers | Maximum absolute error | Maximum relative error |
| --- | --- | --- |
| 1 | 0 | 0 |
| 2 | `5.96046447754e-8` | `5.9602857814e-8` |
| 3 | `1.78813934326e-7` | `0.00149031296572` |
| 4 | `2.98023223877e-7` | `0.00297619047619` |
| 5 | `4.76837158203e-7` | `0.000932314003356` |

Relative error is reported as `abs(actual - reference) / max(abs(reference), 1e-12)`. Cancellation near zero can make that ratio much larger than `rtol`; it is not an independent pass threshold. All tested elements satisfy the combined absolute/relative tolerance. Tolerances were not widened.

A separate three-rank test requires `(1e20F + -1e20F) + 3.0F == 3.0F`, exercising ascending rank order. Its next call requires `(16777216.0F + 1.0F) + -16777216.0F == 0.0F`, exercising FP32 rounding after each contribution. An empty call then advances the same group's sequence. Twenty fresh two-rank communicator groups preserve each rank's descriptor count.

## Failure behavior

The collective executable passes 30 subprocess groups, including the numerical groups above. Fault fixtures send malformed headers over real joined TCP streams. They cover count mismatch, local payload-limit failure, stale/future sequence, unsupported dtype/reduction, inconsistent/oversized byte lengths, wrong operation/source/destination, and invalid result metadata. Rejected result headers leave the worker's input unchanged. Failed communicators reject reuse and retain the last successful sequence; stale-sequence failure is tested after one completed call.

Representative raw timings for a one-second collective timeout:

```text
failure case=stall_payload rank=0 category=timeout elapsed_ms=1001 sequence=0
failure case=slow_payload rank=0 category=timeout elapsed_ms=1000 sequence=0
failure case=stalled_result rank=1 category=timeout elapsed_ms=1001 sequence=0
failure case=cumulative_deadline rank=0 category=timeout elapsed_ms=1000 sequence=0
```

The cumulative-deadline fixture delays rank 1 by 650 ms and rank 2 by 1,300 ms. Root times out on the original deadline after consuming rank 1, so a per-peer timeout renewal would fail the test. Slow payload progress also does not renew the deadline. The local failure bound is 2,500 ms and the harness gives each complete group five seconds before killing/reaping leftovers. These are controlled correctness timings, not latency benchmarks or real-time scheduling guarantees.

Other groups SIGKILL root or a worker during a collective and require bounded peer closure. The harness explicitly expects only the designated victim's SIGKILL exit. The concurrent-close test waits until the communicator has admitted a receive, rejects a second call without failing the active group, then closes and joins the first call within one second. Explicit close preserves closed state rather than changing it to failed. Portable contracts also verify transport destruction and cleanup after a non-engine exception.

## Builds and containers

```sh
cmake -S . -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/native
ctest --test-dir build/native --output-on-failure
docker build --target native-test -t meshtrain-native-test:dev .
sh tools/check_collective_containers.sh meshtrain-native-test:dev
docker build --target python-toolchain-test -t meshtrain-python-toolchain-test:dev .
```

GCC 15.2.0 Debug and Clang 21.1.8 Release under Ubuntu 26.04/WSL2 pass all seven CTest entries with warnings as errors. The installed consumer verifies relocated public headers/link dependencies and a one-worker collective; the Clang run also uses a custom install include directory. Clang ASan/UBSan passes all six native/wire/communicator/process/resolver entries without reports; the independently configured installed consumer runs in ordinary builds. The TCP-disabled Release build passes all four portable/package entries.

The pinned Debian image uses GCC 12.2.0, CMake 3.25.1, and Ninja 1.11.1, with no Python executable installed. Its actual CTest summary is:

```text
core_contracts          Passed  0.00 sec
wire_protocol           Passed  0.00 sec
communicator_contracts  Passed  0.00 sec
centralized_collectives Passed  6.43 sec
tcp_processes           Passed  5.04 sec
resolver_deadline       Passed  0.05 sec
installed_consumer      Passed  0.58 sec
100% tests passed, 0 tests failed out of 7
Total Test time (real) = 12.10 sec
```

The four-container script starts one worker per container on its own private Docker network. All use Docker DNS name `meshtrain-worker-0` for the root, with no published host ports. Each rank completed 14 calls and exited 0. Actual output:

```text
numerical rank=0 world=4 collectives=14 max_abs=2.98023223877e-07 max_rel=0.00297619047619
numerical rank=1 world=4 collectives=14 max_abs=2.98023223877e-07 max_rel=0.00297619047619
numerical rank=2 world=4 collectives=14 max_abs=2.98023223877e-07 max_rel=0.00297619047619
numerical rank=3 world=4 collectives=14 max_abs=2.98023223877e-07 max_rel=0.00297619047619
PASS four-container centralized SUM: 14 collectives per rank, exact + random fixtures
```

The script removes its four containers and network after completion. CI runs this script plus the earlier two-container byte exchange. The optional pinned Python 3.12.15 toolchain image also passes disposable wheel build, isolated import, native result, and native exception checks.

## Limits

This is our standalone CPU collective, with one root receive scratch buffer and no existing collective library in its execution path. It does not establish Python buffer/GIL behavior, training parity, CUDA, native Windows networking, or physical multi-host/GPU scaling. #6–#8 own the remaining M1 integration gates. Root scratch is evident from the single receive-vector implementation; this report makes no allocator/RSS benchmark claim. Completion is local, and a failed input may be partially modified. See the ADR for recovery and failure-propagation consequences.
