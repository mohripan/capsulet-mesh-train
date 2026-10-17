# CPU trainer verification — 2026-10-17

Scope: issue #7, local Linux CPU processes through the installed C++ binding. The owner authorized the presented scope; the fresh independent review had no blockers. Its overflow suggestion is implemented and exercised with real native SUM.

## Reproduction and environment

From PowerShell at the repository root:

```powershell
docker build --target trainer-smoke-test -t meshtrain-trainer-smoke-test:dev .
docker create --name meshtrain-trainer-full-20261017 meshtrain-trainer-smoke-test:dev python tools/check_trainer.py --mode full --output build/trainer-full
docker start -a meshtrain-trainer-full-20261017
docker cp meshtrain-trainer-full-20261017:/workspace/build/trainer-full/run build/trainer-container-full/run
```

Create the local parent directory before copying. The stopped container retains tool/build/install logs, wheel/sdist, runtime versions and all run artifacts until removed. Local copies of the full run, logs, runtime/toolchain versions and distributions are in ignored `build/trainer-container-full/`; smoke artifacts remain in the image at `/workspace/build/trainer-smoke/`. The recorded full-run container was removed after copying its evidence.

The pinned Python/Debian image uses CPython 3.12.15, glibc 2.36 and GCC 12.2.0; Docker Desktop 29.6.2 runs Linux kernel `6.18.33.2-microsoft-standard-WSL2`. Build tools: build 1.6.1, CMake 4.4.4, Ninja 1.13.2, pybind11 3.1.0, scikit-build-core 1.1.1. Runtime: Torch 2.14.1+cpu (no CUDA runtime), NumPy 2.5.4, one intra/inter-op thread, deterministic algorithms. Torch reports MKL/AVX2 CPU execution. Exact transitive dependencies are hash-locked in `requirements/training-cpu.lock`.

Both commands build an sdist and its wheel, install the actual wheel into a fresh isolated environment, hash-check CPU dependencies, install them offline and execute installed paths. Full-run wheel SHA-256: `12b6e993e53b85ce4ef961038451e339aea0e16a25cc499b85feacb6404893ec`. The full run and all focused checks exited zero.

## Full four-worker outcome

Four workers, 100 steps, local batch eight, 676 trainable FP32 elements, mean cross-entropy and SGD lr 0.05. Every worker completed 200 native collectives. The reference loaded the saved initialization and computed its own ordinary combined-batch trajectory after workers completed.

| Quantity | Maximum absolute error across all steps/ranks/reference/pairs |
| --- | --- |
| Global mean loss | `2.384185791015625e-7` |
| Averaged gradient | `2.9802322387695312e-8` |
| Updated parameter / complete final state | `5.960464477539063e-8` |
| Rank-pair loss/gradient/parameter errors | `0.0` |

All **3,090** comparison records passed the fixed elementwise `atol=1e-6, rtol=1e-5` rule; failure count **0**, `status=passed`, `profile=full`, `full_acceptance=true`. Evaluation loss for every rank and reference fell from **1.4119274616241455** to **0.9748985767364502**. This evaluates the prepared training dataset, not held-out generalization. Maximum relative gradient error was about 0.08893 near zero; relative error alone is diagnostic and is not the acceptance rule.

Committed raw metrics: [complete verification report](results/full-verification.json), [all 3,000 per-step comparison records as CSV](results/step-errors.csv). The report also binds config/input/rank/reference digests and includes final state/evaluation checks. FP32 trace arrays, saved inputs, final states and raw logs remain in the local run bundle and are reproduced by the command above.

- Attempt: `4b277426beb54fc1b72666341dae5dd4`
- Context: `28167c27f52c408015c3af3ec61d9bba5aa0dfefa563e602b07f94388d97f6e9`
- Report SHA-256: `c4992cf8be0558aaff1142aeea8d2fc552ca699e0f1092f4834037b653a65c1c`
- Initial state SHA-256: `8ebf20030571c27034d36c97a9bc93157889f3781b9377627d60f843d93f004d`
- Dataset SHA-256: `dc51108c2cbcbf751b262395cd4b62016a321df652b1c3f386c2ebf3d678ef07`
- Schedule SHA-256: `f76c4d7847fb68a66e7949c8b609c9853ec0b3eb87b88a75b4576065fef88671`

## Focused checks

The container smoke tool passed its two-worker/three-step run (63 comparison records, full acceptance false). The full tool then passed all seven focused checks in **16.463 seconds**:

- Installed runtime path/version and CPU-only selection; real shared Torch/NumPy storage.
- Missing/sparse/nonfinite gradients, parameter identity/shape mutation, unsupported dtype/meta device, positive world size and noncontiguous packing. Torch itself rejects incompatible gradient dtype/shape assignment; the adapter also validates the accepted tensors before native admission.
- Two real processes with distinct gradients receive exactly 1.5 everywhere. Finite maximum-FP32 contributions overflow native SUM; both adapters reject before copy-back and preserve original gradients. Failed-call buffer mutation cannot leak into parameter gradients. Trace snapshots retain independent storage.
- Prepared generator data, teacher labels, exact wrapped/disjoint indices and 676-element layout; digest validation and nonempty-output refusal.
- Separate installed `prepare`, independent `worker` processes and `verify` commands pass two-worker/three-step parity without the convenience launcher.
- Missing completion, stale identity, wrong trace shape, nonfinite trace, perturbed finite gradient, missing final bias and altered input digest all fail. The finite perturbation has refreshed file hashes, proving numerical comparison also detects it. Prior success status is invalidated in every case.
- Actual subprocess fixtures verify invalid worker configuration while its peer waits for startup, more than pipe capacity on **both** streams (262,144 bytes each), later-spawn failure and interruption. Each fixture must finish under an independent **five-second** watchdog after harness readiness, using 0.1-second group termination grace; every recorded child PID is reaped. Harness imports occur before readiness. No broad fault matrix or public fault-injection switch was added.

WSL CPython 3.12.13 with the same locked CPU runtime independently passed all seven focused checks in **138.887 seconds**, including those same watchdogs. Command:

```sh
build/training-dev/bin/python -I -X faulthandler tests/training/checks.py build/trainer-dev-smoke
```

That runtime installed the built wheel; the initial two-worker/three-step run passed before the focused checks. WSL/NTFS framework imports and dependency installation are substantially slower than the container path. An initial failure fixture hit its two-second group timeout before eager Torch imports exposed the bad path. A parallel full-tool attempt had already built that older wheel: its training passed, but its overall check failed on the same fixture (117.351 s suite); it is not counted as final verification. Moving path validation ahead of optional framework imports fixed the actionable diagnostic and bounded cleanup; no numerical tolerance or external watchdog was widened.

The final wheel was then installed into that isolated WSL full runtime, and both the complete numerical run and focused suite passed: **3,090** comparisons, zero failures, identical evaluation decrease, seven checks in **116.404 seconds**. Retained evidence: `build/trainer-full/run-final/`, `final-training.log`, `final-checks.log`, `final-install.log`.

```sh
uv pip install --python build/trainer-full/runtime/bin/python --no-deps --reinstall build/trainer-final-dist/*.whl
build/trainer-full/runtime/bin/python -I -m meshtrain.training run --output build/trainer-full/run-final
build/trainer-full/runtime/bin/python -I -X faulthandler tests/training/checks.py build/trainer-full/run-final
```

Base-boundary regression: `docker build --target python-bindings-test -t meshtrain-python-bindings-test:dev .` passed all **10** installed binding checks in **2.036 seconds**, including explicit absence of Torch and Torch-free base import. No C++/protocol changes were made. Native regression remains a required existing CI job.

## Limits and delivery

This proves four CPU processes inside one Linux container plus focused WSL process checks. Four separate trainer containers, broad startup/collective kill/stall/count matrices and clean-build consolidation remain #8. Native Windows networking, CUDA, distinct GPUs and physical multi-host performance are untested here. Push/PR CI adds only the short trainer target and retains native/binding checks; there is no nightly schedule.

Implementation [bd155c1](https://github.com/mohripan/capsulet-mesh-train/commit/bd155c1c37cbac023e5773d98600d9ed7e7d75d5) was pushed to `main` without a co-author trailer; remote revision was read back. [Exact-revision CI 39018189654](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39018189654) passed all three jobs: native/container collective checks (45 s), Torch-free bindings (51 s), CPU trainer smoke (118 s). These are CI job elapsed times, not communication benchmarks.

Issue #7 is closed; #2 records it complete, #8's dependency is checked, and #1 points to #8. Readback confirmed #2/M1, #8 and #1 remain open. Unrelated `.agents/skills/find-skills/`, `.claude/` and `skills-lock.json` were preserved. Final source/metadata, short-comment/docstring, local-link, raw JSON/CSV consistency, committed report digest and Git whitespace checks passed. M1 remains open until #8 is complete.
