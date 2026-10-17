# CUDA adapter implementation evidence — 2026-10-17

Status: installed GPU acceptance and local CPU regressions passed. Exact-revision CPU CI/delivery remain pending; #10/#9 remain open.

The owner selected a fresh independent review and authorized implementation afterwards: “Sub-agent. And then you can go on and implement it.” [Review/disposition](review.md) records both P2 findings, the guarded forward-tangent follow-up, and final confirmation of resolution. ADR 0004 is Accepted from that actual owner instruction; the review itself is not approval.

## Implemented boundary

The optional Python adapter owns fixed CPU FP32 staging and a sharing NumPy view. Device copies/stream completion use PyTorch, while the unchanged existing binding/C++ core owns SUM, sequencing and TCP. The input contract rejects inference and forward-AD tensors, including a dual created outside caller inference mode. Workspace allocation explicitly disables inference mode and scopes the configured CUDA device. Failed CUDA execution poisons the workspace and retains failed owners, with original/secondary errors distinguished.

New tests/runner build an installed package, retain source/lock/wheel/runtime/device identities and use fresh owned workers. They cover single/two-worker numerical behavior, modes, streams, lifetime/GIL/concurrency, peer failure, explicit injected copy-error policy and separate timing fields. Implementation alone is not GPU proof.

## CPU regressions

All four commands passed in the existing Docker environment, using pinned native/Python bases and unchanged CPU/build locks:

```sh
docker build --target native-test -t meshtrain-native-test:cuda-10-dev --progress plain .
docker build --target native-portable-test -t meshtrain-native-portable-test:cuda-10-dev --progress plain .
docker build --target python-bindings-test -t meshtrain-python-bindings-test:cuda-10-dev --progress plain .
docker build --target trainer-smoke-test -t meshtrain-trainer-smoke-test:cuda-10-dev --progress plain .
```

Native TCP-on passed all seven existing CTest entries, including relocated consumers. TCP-off passed four entries. Torch-free binding verification passed all ten installed-extension checks. The short CPU trainer passed 63 comparisons and eight focused checks, including explicit rejection of unavailable CUDA from the new optional adapter. No new core/binding C++ dependency or root-package eager Torch import was introduced. Raw logs are `build/cuda-regression-*.log` and `build/cuda-cpu-training-checks.log`.

The test-only controller's real-process timeout, nonzero-exit, later-spawn-failure and interruption checks passed; all tracked children were terminated/reaped, with actual elapsed times below 0.3 seconds in the focused run. Results: `build/cuda-controller-negative-verified/checks.json`. The initial interruption check exposed an unregistered-child window in the factory/ownership boundary. Signal deferral now registers ownership before cancellation is raised, and fixture cleanup also reclaims its own children on failed assertions. That original failed fixture is development failure evidence, not a successful cleanup claim; a scoped check found no still-running child attached to its exact stdout path afterwards. No GPU recovery claim follows from these standard-library process checks.

## Runtime and GPU checks

The separate CUDA lock was generated with:

```sh
uv pip compile --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes requirements/training-cuda.in -o requirements/training-cuda.lock
```

The official Torch `2.14.1+cu132` CPython 3.12 Linux x86-64 wheel is pinned by URL and SHA-256. Its metadata requires CUDA toolkit 13.2.2; individual components have their own versions, to be reported from the installed environment. CPU/build locks remain unchanged.

Initial development execution:

```sh
build/toolchain-venv/bin/python tools/check_cuda_adapter.py --output /tmp/meshtrain-cuda-first
```

The sdist-derived wheel built; runtime libraries are downloading with required hashes. Independent hash-locked package cache prefetches may overlap to avoid serial download delay; all final installation remains offline/hash-checked. This is not a cold-cache claim. The Linux filesystem hosts the isolated runtime, avoiding cross-filesystem import overhead from the Windows checkout.

Source/test refinements were made during that initial dependency setup, so its initial behavioral hash cannot certify the final source. Its component results are development evidence only. A fresh final source-stable run with the collected wheelhouse is required before acceptance; no tolerances, native deadlines or 45-second worker-case bounds are widened by runtime setup.

The first setup completed all downloads but failed during installation: Ubuntu mounts `/tmp` as a 7.7 GiB tmpfs, which filled. The original failure also prevented its final JSON rewrite; the external driver log preserves both errors. Its task-owned files/wheelhouse were moved, with exact path checks, to `/home/mohri/.cache/meshtrain-validation/cuda-first`; the separate cache prefetch directory was preserved alongside it. Nothing outside those owned task directories was reclaimed. Setup now checks 12 GiB free in the output filesystem, places child setup temporaries there, and preserves a primary error if evidence persistence itself fails. The final run uses the large home filesystem and those same hash-checked wheels, without re-downloading or weakening runtime/numerical checks.

## Completed installed GPU acceptance

The final command passed from a fresh output/runtime, using the already downloaded hash-checked wheelhouse:

```sh
build/toolchain-venv/bin/python tools/check_cuda_adapter.py \
  --output /home/mohri/.cache/meshtrain-validation/cuda-final \
  --wheelhouse /home/mohri/.cache/meshtrain-validation/cuda-first/cuda-wheels
```

Environment: Ubuntu 26.04/WSL2, Linux CPython 3.12.13, Torch 2.14.1+cu132, NumPy 2.5.4, Windows KMD 616.87; actual device RTX 5070 Ti Laptop GPU, 12792037376 bytes, capability 12.0. The installed build reports sm_120 among its architectures and executed real arange/square/add/copy/completion checks. Runtime metadata reports CUDA 13.2, cuda-toolkit 13.2.2, runtime/NVRTC 13.2.86, cuda-bindings 13.4.4 and cuDNN 9.24.0.43. [Full preflight/component identities](results/preflight.json) include physical UUID and installed package/extension paths. Both rank processes reported the same physical GPU.

The sdist-derived installed adapter bytes exactly match the working source. The source-stable run was based on planning HEAD `aa75e18fe61269968f1a296ee610e0640e080878`, behavioral SHA-256 `6c3372722f3a6cd11cbb4638a1d0167957e63751a023031d560c2efcc296e8fd`. It checked that identity before hardware execution and afterwards. Wheel SHA-256: `f4d6df6c08b28055fe9c1fdf7335c63400c6cfcc79981c5213297c51602b296c`; CUDA lock SHA-256: `b3b18918b9824a39bdecd3f56d77495a610dd663bec0ba2bd51ccf95c511cc54`.

| Hardware case | Complete elapsed seconds | Worker exits |
| --- | ---: | --- |
| Single-rank contract/streams/AD rejection | 2.141 | 0 |
| Single-rank numerical/timing matrix | 1.538 | 0 |
| Two-rank numerical/timing matrix | 1.947 | 0, 0 |
| Retained ownership/GIL/concurrency | 1.677 | 0, 0 |
| Closed native peer | 1.439 | 0, 0 |
| Stalled native peer | 2.511 | 0, 0 |
| Injected copy/secondary-error policy | 1.231 | 0 |

All seven cases passed, every process was reaped and every complete case stayed below the original 45-second bound. Forty-eight numerical records cover both memory modes, 0/1/1025/1048576 elements and exact/bounded-random fixtures. Exact SUM/no averaging and identity fixtures passed exactly; maximum absolute error against the independent CPU reference was `1.4901161193847656e-8`, with unchanged `atol=1e-6, rtol=1e-5`. Matrix workers completed sequence 76 each; empty participation advanced sequence. The single-rank contract checks passed constructor/device wrap rejection, input metadata, inference/forward-AD including caller inference mode, capture rejection, sharing/pinning/subview boundaries, special FP32 bits and nondefault producer/consumer completion.

The delayed-peer case retained GPU/adapter/host/NumPy owners while native communication waited, allowed 1,191,372 Python counter iterations, rejected concurrent workspace use, returned the exact sum and released transient owners after completion. Peer close reported `peer_closed` with matching rank/run/peer/sequence/endpoint; stall reported `timeout` with the one-second native deadline. The wrapper's observed stall path was 1062.351 ms including staging/control; native sequence stayed zero and GPU input stayed unchanged, with no H2D on failure.

The injected H2D/secondary close errors are explicitly policy-only coverage: the original exception/notes and advanced sequence 1 were retained, the workspace was poisoned and real stream settling completed. Its original failed cleanup record remains false due to the injected close error; separate restoration closed the actual communicator and final worker cleanup passed. No actual poisoned-device recovery is claimed.

## Timing observations and durable evidence

Raw records retain five warmups and ten measured iterations per size/mode/rank configuration, including rank/iteration correlation. There are 120 measured samples. Selected pooled per-rank medians for two workers and 4 MiB payloads are:

| Host memory | D2H ms | Binding/native ms | H2D ms | Total ms |
| --- | ---: | ---: | ---: | ---: |
| Pageable | 0.610 | 1.718 | 0.653 | 3.276 |
| Pinned | 0.227 | 1.677 | 0.279 | 2.285 |

These synchronized wall intervals include framework/control/wait costs. Medians of separate stages need not sum to the median total. Pinning reduced the large-copy costs in this run; for a single rank and 1025 elements its total median was 0.118 ms versus pageable 0.103 ms, so pinning is not a universal performance win. One-rank native timing is identity overhead; two-rank native timing includes peer waits, TCP and CPU reduction. These measurements are shared-single-GPU local functional observations, not network-only latency or distinct-GPU throughput/scaling.

Committed original selected evidence: [aggregate/source/wheel identity](results/validation.json), [all GPU cases and numerical/error records](results/checks.json), [controller lifecycle negatives](results/controller-checks.json), [device/runtime](results/preflight.json), [measured samples](results/timing-samples.json), [CSV](results/timing-samples.csv), [pooled summaries](results/timing-summary.json), [case log](results/checks.txt), [controller log](results/controller-checks.txt), [installed runtime freeze](results/runtime.txt) and [SHA-256 manifest](results/sha256.json). The complete external tree retains per-rank stdout/stderr/preflight, source map, sdist/wheel, dependency/install logs and original process reports. Source/lock/wheel identities are actual recorded values, not reproducible-binary claims.

The low-space preflight and stale/nonempty-output checks also failed explicitly as intended. No operation deadline, outer case bound or numerical tolerance was widened. Exact-revision CPU CI, commit/push and issue handoff are the remaining delivery gates. GPU training/reference parity stays #11; no native Windows, GPU-container, physical-host, distinct-GPU or GPU-nightly execution claim is made.
