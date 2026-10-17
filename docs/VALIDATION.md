# Reproduce M1 CPU validation

Use a Linux/WSL shell, host CPython 3.12 or newer, Docker CLI connected to a Linux daemon, and the checkout. The host runner uses the standard library; Torch, NumPy and build packages are installed inside pinned validation images.

```sh
python3 tools/check_m1.py --no-cache --output build/m1-validation
```

Use a fresh output directory for every attempt. `--no-cache` bypasses Docker build layers; hash-checked dependency/download caches may remain. Native CMake outputs, sdist-derived wheels, isolated package environments and training runs are freshly produced. Pinned base images/direct dependencies and recorded Debian transitive versions establish the tested environment, without a hermetic or bit-identical build claim.

The runner verifies:

- Standalone TCP-on and TCP-off C++ builds, native contracts and relocated installed consumers without Python installed in those images.
- The actual installed binding wheel without Torch, and the short installed CPU trainer checks.
- Eight controlled startup/collective × root/nonroot × SIGKILL/SIGSTOP cases and 20 additional failed groups. Native operation deadlines are one second; each complete group, including worker cleanup and endpoint reuse, must finish within five seconds.
- Descriptor baselines inside every healthy survivor after native object destruction, while it is still alive. Controller counts, signaled-target OS cleanup and child reaping are recorded separately.
- Missing-phase rejection and actual independent-watchdog expiry, existing native two/four-container exchanges, and a four-process/100-step localhost trainer inside one pinned container.
- Four separate trainer containers, two three-step smoke attempts, controller failure/interruption/ownership checks, corrupted-artifact rejection and the full 100-step comparison.

The broad fault tests carry CTest label `m1_integration`. To run them from an existing Linux native build:

```sh
ctest --test-dir build/native -L m1_integration --output-on-failure -V
```

Each standalone case has a five-second CTest timeout. A persistent independent watchdog also enforces five seconds per group in the repeat controller; 110 seconds is only the repeat batch's containment limit. A watchdog kill is a failed test and cannot certify native cleanup. These limits exclude image builds and Docker management, whose separate deadlines are recorded.

## Four-container trainer

```sh
docker build --target trainer-container -t meshtrain-trainer-container:dev .
python3 tools/check_training_containers.py --image meshtrain-trainer-container:dev \
  --mode smoke --output build/container-smoke
python3 tools/check_training_containers.py --image meshtrain-trainer-container:dev \
  --mode full --output build/container-full
```

Each worker executes the installed `meshtrain.training worker` directly, with its own writable filesystem and an explicit rank. A private bridge network resolves `worker-0` through `worker-3`; no host ports, shared run volume, host PID namespace or Docker socket is exposed to workers. The controller copies common prepared inputs before launch, monitors every rank, collects outputs after all workers stop and verifies them in a separate network-disabled helper.

Full mode preserves the accepted four-worker/100-step/local-eight MLP workload and `atol=1e-6, rtol=1e-5` elementwise rule. It requires all 3,090 reference/rank/state/evaluation comparison records, complete finite artifacts, lower final evaluation loss and 200 successful native collectives per rank. Smoke uses four workers/three steps and cannot claim full acceptance. See [training details](TRAINING.md).

The healthy worker group has a 120-second deadline, including startup/imports. Preparation, successful artifact collection and reference verification each have their own 120-second phase budget. Short Docker commands are limited to five seconds or the remaining budget. Failure cleanup has one 15-second budget for collective termination, bounded diagnostics, removal and ownership checks; it does not accumulate Docker's default per-container stop grace. No automatic job retry occurs.

## Evidence and cleanup

`validation.json` is the aggregate result. Component directories preserve source/lock/image/wheel identities, commands, native case JSON/JUnit/logs, rank exit states, input bundles, traces, final parameters and numerical reports. Numerical success cannot override failed cleanup. `resources.json` records a UUID ownership label and intended resources before creation; cleanup queries only that owner and preserves a history of outcomes.

If a controller is interrupted before its cleanup finishes, use its recorded owner manifests:

```sh
python3 tools/cleanup_validation.py --output build/m1-validation
```

This does not prune the daemon or delete unrelated resources. Reclamation appends command evidence and preserves the original failed validation result. Docker unavailability is reported as failed/unknown cleanup. The controller checks deliberately preserve an unrelated sentinel container/network, and reclaim it through a separate test owner afterwards.

## CI and completion evidence

The existing push/PR workflow retains all seven original native entries, native container checks, Torch-free bindings and the short trainer job. It excludes the newly added broad fault label. The separate `M1 full CPU validation` workflow supports manual dispatch and a daily `23 18 * * *` UTC schedule, with selected evidence retained for 14 days even on failure. A configured schedule is not evidence that a nightly execution happened.

Issue #8 acceptance requires a complete local no-layer-cache validation and a manual exact-revision workflow pass. M1 closes only after its child issues and numerical/deployment/failure evidence are reconciled. See the [OpenSpec](openspecs/m1-validation/proposal.md) and its [verification report](openspecs/m1-validation/verification.md) for actual results and remaining limits. Four local containers establish Linux/container correctness, without native Windows, GPU, physical multi-host or scaling claims.
