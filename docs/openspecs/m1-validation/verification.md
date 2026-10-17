# M1 validation implementation evidence — 2026-10-17

Status: implementation and complete local acceptance passed. Exact-revision manual CI and delivery remain pending; #8/M1 are not yet complete.

The owner authorized the independently reviewed scope with “Okay. Please implement!” on 2026-10-17. The required worker descriptor correction and runner prerequisites were resolved before implementation. Existing ADRs 0001–0003 remain sufficient: no production API, wire protocol, ownership, numerical or recovery contract has changed.

## Focused native evidence

In Ubuntu/WSL, GCC 15.2.0, Debug/warnings as errors:

```sh
cmake -S . -B build/m1-native-dev -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/m1-native-dev
ctest --test-dir build/m1-native-dev -R 'fault_(startup-root-stop|collective-worker-kill)$' --output-on-failure -V
ctest --test-dir build/m1-native-dev -L m1_integration --output-on-failure -V
```

The two initial cases passed (1.07 s total). All eight cases plus the 20-group repeat test then passed: nine CTest entries, 22.63 s total. Every passing record verified live-survivor descriptors, controller baselines, target signals, child reaping and endpoint reuse; stalled cases observed native timeout while the target was stopped. The final acceptance run additionally checks retained target sockets and runs the executable-local watchdog containment in the pinned native image. Raw development logs are `build/m1-native-focused.log` and `build/m1-native-faults.log`; these are focused development evidence, not the complete clean-run gate.

## Focused Docker evidence

The pinned CPU image built an sdist and its wheel and installed Torch 2.14.1+cpu / NumPy 2.5.4 in `/opt/meshtrain/runtime`. No host Torch was used.

```sh
docker build --target trainer-container -t meshtrain-trainer-container:dev .
python3 tools/check_training_containers.py --image meshtrain-trainer-container:dev --mode smoke --output build/m1-container-smoke-1
python3 tests/validation/checks.py --image meshtrain-trainer-container:dev --output build/m1-controller-checks-final
```

The first four-container smoke passed all 180 comparisons; all four containers exited zero, had no mounts/host PID namespace, and owned-resource cleanup passed in 0.740 s. The controller suite then passed two fresh smoke attempts, real bad-config/kill/spawn/interruption/timeout/cleanup-failure cases, unrelated sentinel preservation, and a finite corrupted gradient with refreshed file metadata rejected by the network-disabled verifier. Original successful artifacts remained unchanged.

An initial controller-suite attempt exposed that `subprocess.TimeoutExpired` from a near-exhausted Docker command budget was not normalized into the group failure type expected by the timeout fixture. The command layer now reports that deadline failure explicitly and preserves cleanup; the complete controller suite passed afterwards. No operation deadline, external watchdog or numerical tolerance was widened.

## Complete local acceptance

```sh
python3 tools/check_m1.py --no-cache --output build/m1-validation
```

Passed on 2026-10-17 under Ubuntu/WSL, host Python 3.14.4 (standard library only), Docker Desktop 4.85.0 / Engine 29.6.2, Linux 6.18.33.2. All five validation targets were built with `--no-cache`; dependency downloads may use the explicitly hash-checked cache. Native images contain GCC 12.2.0, CMake 3.25.1 and Ninja 1.11.1 and no Python. Trainer images use CPython 3.12.15, Torch 2.14.1+cpu and NumPy 2.5.4. No host venv, Torch, wheel, build output or run bundle was consumed.

- TCP-on seven existing entries and relocated consumer, TCP-off four entries/consumer, Torch-free binding package and short CPU trainer checks passed in fresh builds.
- All nine broad CTest entries passed, producing 28 successful native group records: eight named cases plus 20 additional groups. Complete case times were 3.804–1006.220 ms, strictly below 5000 ms. Live-survivor/controller descriptor baselines, reaping, endpoint reuse and stopped-target sockets passed. Missing-phase rejection exited 2; actual watchdog expiry exited 137 and could not claim native cleanup. Selection checks verified nine broad and seven existing entries.
- Existing native two/four-container scripts passed, with UUID-scoped cleanup verified afterwards.
- Both localhost and four-container full profiles passed all 3,090 comparison records each, with zero tolerance failures at unchanged `atol=1e-6, rtol=1e-5`. Largest absolute error was `2.384185791015625e-7`; the relative maximum near zero is not a separate acceptance rule. Evaluation loss decreased from `1.4119274616241455` to `0.9748985767364502` for every rank and the reference. All four ranks completed 200 native collectives and exited zero.
- The full-run controller suite repeated two successful four-container smoke attempts (180 comparisons each), all six real negative fixtures, corrupted-gradient rejection and sentinel preservation. Numerical success with a deliberately failed cleanup was correctly rejected; test-owned recovery preserved the original failed report.
- Every owned helper/worker/network was reclaimed. Four-container full cleanup passed in 0.994 s; no mounts, shared PID namespace, host networking or host ports were used. Inputs were copied before launch and outputs/reference compared after workers stopped.

Selected original evidence is committed in [results/](results/): [native records](results/native-results.json), [case timings](results/native-timings.csv), [four-container numerical report](results/four-container-verification.json), [localhost numerical report](results/localhost-verification.json), [controller results](results/controller-checks.json), [installed identities](results/installed.json), [aggregate manifest](results/local-validation.json), [behavioral source identity](results/local-source.json) and [SHA-256 manifest](results/sha256.json). The full ignored tree `build/m1-validation/` retains inputs, final checkpoints, traces, commands, rank logs, inspect metadata, cleanup records and JUnit; raw driver output is `build/m1-validation-tool.log`.

The clean run used the working tree based on planning revision `8f58455b4504114c367f083f678b5280f50253b9`, with behavioral hash `640ef0143acb3ddaf04227b597ac4691bb32b3d772430af43766b3a26907c588`. After that pass, final code review tightened native PID ownership: assign the returned PID only after successful `posix_spawn`. The affected nine CTest entries passed again (28 groups, `build/m1-native-final.log`). This source change is not falsely attributed to the earlier clean-run hash; the manual CI gate will verify the final implementation revision in full.

## CI and delivery

The new `M1 full CPU validation` workflow supports manual dispatch and a daily UTC schedule; action versions are immutable SHA pins. Existing push/PR native, binding and short-trainer coverage remains. Exact-revision manual execution, downloaded artifact inspection, GitHub issue/milestone reconciliation and final delivery are still pending. A configured schedule is not an executed nightly result.

No new ADR was needed: the test/validation architecture introduces no lasting production contract change. These results establish CPU/Linux local-container correctness, not native Windows, GPU, physical multi-host or scaling performance. Later work retains per-task OpenSpec review and authorization.
