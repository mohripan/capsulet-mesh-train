# M1 validation implementation evidence — 2026-10-17

Status: implementation, complete local acceptance and exact-revision manual CI passed. Evidence was downloaded and inspected; #8, #2 and milestone 1 are closed.

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

The clean run used the working tree based on planning revision `8f58455b4504114c367f083f678b5280f50253b9`, with behavioral hash `640ef0143acb3ddaf04227b597ac4691bb32b3d772430af43766b3a26907c588`. After that pass, final code review tightened native PID ownership: assign the returned PID only after successful `posix_spawn`. The affected nine CTest entries passed again (28 groups, `build/m1-native-final.log`). This source change is not falsely attributed to the earlier clean-run hash; the manual CI gate below verified the final implementation revision in full.

## CI and delivery

Implementation commit [80903ea909fe45b91d9fbeeb9aa8dda74ed64106](https://github.com/mohripan/capsulet-mesh-train/commit/80903ea909fe45b91d9fbeeb9aa8dda74ed64106) was pushed to `main` without a co-author trailer. The remote SHA was confirmed. Both exact-revision workflows completed successfully:

- [Full manual M1 validation, run 39030243246](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39030243246): acceptance, always-run resource reclamation and evidence upload all passed on Ubuntu 24.04. Actual host Python was 3.12.3, Docker Engine 28.0.4, Linux 6.17.0-1022-azure. The behavioral source identity was clean, SHA-256 `4d1ca0c46e0bf4069abd2ef174215455017afaa0d529a7a08518fe4b4b16190f`.
- [Regular native/binding/trainer CI, run 39030238414](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39030238414): all three jobs and the existing two/four-container native exchanges passed.

The full artifact was downloaded with `gh run download 39030243246 --dir build/m1-ci-39030243246` and inspected, rather than accepting the job verdict alone. Both full trainer profiles passed 3,090 comparisons with zero failures, maximum absolute error `2.384185791015625e-7`, unchanged tolerances, the same decreasing evaluation losses as the local run, four zero worker exits and sequence 200. The 28 native groups passed in 4.206–1006.910 ms; positive phase/stopped-socket/live-survivor FD evidence, repeated cleanup and both harness rejection cases passed. All six Docker negative fixtures, two smokes, corruption rejection and sentinel preservation passed again. Every one of 22 recorded resource owners had successful final cleanup and the always-run reclamation step passed; the deliberately failed cleanup remains failed in its original job report, with separate successful recovery. Four-container full cleanup took 0.324 s.

[Artifact 12100521254](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39030243246/artifacts/12100521254), `m1-validation-39030243246-1`, contains 11,506,655 compressed bytes, SHA-256 `19bdd786e844794db571a7b6f316de44cee4f13c6d55352928e44e3be8e4ea4d`; GitHub reports expiry 2026-10-31. Durable selected copies are [CI provenance](results/ci-provenance.json), [aggregate](results/ci-validation.json), [source identity](results/ci-source.json), [native records](results/ci-native-results.json), [timings](results/ci-native-timings.csv), [four-container numerical report](results/ci-four-container-verification.json), [localhost report](results/ci-localhost-verification.json), [installed identities](results/ci-installed.json), [controller checks](results/ci-controller-checks.json) and [resource summary](results/ci-resource-summary.json). [SHA-256 manifest](results/sha256.json) covers the local and CI selected copies. The complete downloaded tree retains rank logs, traces, final checkpoints, input/config bundles and original negative reports.

The new workflow supports manual dispatch and daily `23 18 * * *` UTC execution, with immutable action pins and 14-day failure artifacts. This is an executed manual pass and a configured nightly schedule; no nightly execution is claimed. Git whitespace, task/navigation links, evidence hashes, Python syntax and two-line comment/docstring limits passed; clang-format 22.1.7 accepted the new C++ fixture. No production source was changed by the delivery/evidence reconciliation.

Before closure, issues #3–#7 were read back: all are closed with all acceptance checkboxes satisfied and linked implementation/CI evidence. #8 adds the remaining clean/failure/container outcome proof. After that audit, #8 and #2 were closed; milestone 1 was closed with zero open issues and seven closed issues. Root tracker #1, M2 #9 and #10’s dependency were updated and read back; the next unmet implementation dependency is [#10, host-staged CUDA adapter](https://github.com/mohripan/capsulet-mesh-train/issues/10), under M2 tracker #9. It still needs its own focused OpenSpec/review/go-ahead.

No new ADR was needed: the test/validation architecture introduces no lasting production contract change. These results establish CPU/Linux local-container correctness, not native Windows, GPU, physical multi-host or scaling performance. Later work retains per-task OpenSpec review and authorization.
