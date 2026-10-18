# GPU trainer implementation evidence — 2026-10-18

Status: installed full GPU acceptance and local CPU regressions passed. Exact implementation-revision CPU CI and delivery remain pending; #11/#9/M2 remain open.

The owner selected a fresh independent review and said “I prefer the sub-agent, and then you can go implement it.” [Review/disposition](review.md) records no findings; actual owner authorization accepts ADR 0005 after that review. The implementation preserves native SUM and CPU/schema-1 defaults while adding explicit CUDA execution/schema 2.

## Final installed hardware run

```sh
build/toolchain-venv/bin/python tools/check_cuda_trainer.py --mode full \
  --output /home/mohri/.cache/meshtrain-validation/cuda-trainer-full-01 \
  --wheelhouse /home/mohri/.cache/meshtrain-validation/cuda-first/cuda-wheels
```

The command exited zero after building the real sdist-derived wheel, installing the hash-checked existing CUDA lock in a fresh isolated environment, and running installed code outside the checkout with `-I`. Host log: `build/cuda-trainer-full-01.log`. The external Linux directory retains sdist/wheel/runtime, source/config/input digests, all original NPZ/PT trajectories/states, setup/runtime/driver logs, per-rank stdout/stderr and process/control evidence. It is disk-backed, with more than 900 GiB available at preflight. No driver installation, GPU reset, unrelated job termination or device fallback occurred.

| Identity | Recorded value |
| --- | --- |
| Behavioral source | `011fc41e633950b310e4cb212299be268ca8c5fedf5701eed3c1166d65888a2d`, 106 files, base HEAD `c7925bc` plus the implementation. |
| Installed wheel SHA-256 | `9fb4d52cb58f96d7d1e3844be05c65b8dfbff3f8083f7fc4b4d599e39b6eef4d` |
| CUDA lock SHA-256 | `b3b18918b9824a39bdecd3f56d77495a610dd663bec0ba2bd51ccf95c511cc54` |
| Runtime | Linux/WSL x86-64, CPython 3.12.13, Torch `2.14.1+cu132`, NumPy `2.5.4`; full build/runtime/component records retained. |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12,792,037,376 bytes, capability 12.0, supported `sm_120`. |
| Physical UUID | `8f5c096d-be03-fe0b-17fa-9a35b08e251d`, identical in every worker/reference profile. |
| Policy | Strict deterministic algorithms, no autocast, global/matmul/cuDNN IEEE FP32, no cuDNN benchmarking, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. |

The requested/effective policy and actual GPU MLP forward/backward are recorded per process. CUDA tag 13.2 is not the version of every component: the locked runtime records cuBLAS `13.4.1.3`, CUDA runtime/NVRTC `13.2.86`, cuDNN `9.24.0.43`, and the other original package versions. Windows KMD is 616.87; retain the WSL/UMD banner separately in [driver output](results/driver.txt). The native extension remains a CPU host-buffer module with no Torch/CUDA build linkage.

| Profile | Result | Elapsed seconds | Numerical comparisons |
| --- | --- | ---: | ---: |
| Gradient contract unit groups | Passed, 16 groups plus 6 constructor rejections | 4.580 | — |
| Two-worker known averaging/concurrency/stream dependency | Passed, exact 1.5 then 2.5, sequence 2 | 4.332 | — |
| Two-worker reduced overflow | Passed, unchanged gradients and poisoned reuse | 4.162 | — |
| Real peer-close before second update | Passed | 4.746 | — |
| One-worker pinned, three steps | Passed, exact reference match | 6.941 | 27 |
| One-worker pageable, three steps | Passed, exact reference match | 7.041 | 27 |
| Two-worker nondefault stream, three steps | Passed | 7.393 | 63 |
| Two-worker pinned, 100 steps | Passed, CUDA full acceptance | 8.324 | 936 |
| Two-worker pageable, 100 steps | Passed, CUDA full acceptance | 8.075 | 936 |
| CUDA convenience `run`, three steps | Passed, nested workers reaped by supervisor | 5.303 | 63 |

Both 100-step profiles had maximum absolute comparison error `2.384185791015625e-7`, zero tolerance failures and native sequence 200 per worker. All workers and references decreased fixed-set evaluation loss from `1.411927342414856` to `0.9792857766151428`. The unchanged rule is `abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)`, evaluated in FP64 over finite FP32 traces. The reference independently computes ordinary GPU combined batches and does not consume worker training values. Every rank-pair, per-step loss/gradient/parameter, complete final-state, trace/final-state and recomputed GPU evaluation comparison passed. GPU reports retain CPU `full_acceptance=false`; only the two required full profiles set CUDA acceptance.

Four standard-library controller negative cases passed with owned PIDs reaped, without GPU execution. Fourteen corruption cases passed, including actual runtime-component identity and early schema/environment failures replacing stale success. Both cooperative SIGTERM/SIGINT supervisor fixtures passed without GPU work. Every one of the 10 GPU profile commands and the extra corruption/launcher phases was owned, bounded and cleaned up; profile times were far below the unchanged 180-second execution plus 10-second cleanup limit. There were no retries or tolerance/deadline increases.

The final real-peer fixture produced `peer_closed` at rank 0 and `invalid_state` at rank 1, both at sequence 2. Each actual SGD-entry log contains exactly completed step 0, with no step-1 update or success completion. [Original selected failure/update records](results/peer-failure.json) preserve category, sequence, peer/endpoint and primary error. This is a native connection closure, not a corrupted CUDA context or hardware-recovery test.

## Timing and memory observations

The full profiles retain all 100 numerical steps. Five timing warmup steps leave 380 measured rank-step records and 760 measured staged-call records, tagged by mode/rank/step/gradient-or-loss. Warmup is part of the same real SGD trajectory, with no reset or hidden updates. Selected per-rank medians in milliseconds:

| Memory | Rank | Gradient D2H | Gradient native | Gradient H2D | Gradient staged total | Loss staged total | Instrumented step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pinned | 0 | 0.05177 | 0.04206 | 0.04879 | 0.20473 | 0.34186 | 5.09382 |
| Pinned | 1 | 0.05399 | 0.28786 | 0.07931 | 0.50825 | 0.32887 | 5.19085 |
| Pageable | 0 | 0.06020 | 1.34288 | 0.09633 | 1.61719 | 0.45454 | 5.94628 |
| Pageable | 1 | 0.05700 | 0.10688 | 0.11706 | 0.34637 | 0.37065 | 5.82132 |

These tiny payloads and shared-GPU schedules produce different peer waits per rank; this run does not isolate a causal pinning speedup. The raw records include readiness, complete gradient-sync, nested trace-copy and maximum-rank step completion as well as the shown intervals. Native includes binding, TCP, CPU reduction and peer waits. Stage/step timers and medians are nested/nonadditive; instrumented step time includes correctness/trace costs and is not an uninstrumented throughput benchmark. Separate copy/native stages do not claim isolated wire or reduction-kernel measurements.

Each worker requested 2,708 bytes of host staging, either explicitly pinned or pageable. Parameters/gradients/flat/loss total 8,116 requested device bytes; features/labels add 73,728. Actual per-worker Torch allocator peaks were 67,456,512 allocated and 69,206,016 reserved bytes, reflecting framework/workspace costs beyond those tensors. Linux process RSS high-water marks in the full profiles ranged from 1,793,667,072 to 1,794,396,160 bytes before final artifact emission. Reference memory has separate [original selected observations](results/reference-metrics.json). Requested pinned storage is not total locked pages/RSS, free-device snapshots are not a reservation/peak, and independent process peaks cannot be summed into a measured simultaneous device peak.

Seven additional actual installed CLI negative cases passed for unindexed/wrapped/absent devices, hidden GPU visibility, CPU host-memory misuse, TF32 override and workspace conflict. The new runner also explicitly rejected nonempty output and low-space `/tmp` before setup, with original messages retained in [setup rejections](results/setup-rejections.json). These are expected failures, without an attempted oversized installation or mutation of prior evidence.

## Durable evidence and delivery boundary

Selected original evidence is committed as [source/build/wheel acceptance](results/validation.json), [all profile reports/raw comparisons/worker metrics](results/checks.json), [controller lifecycle negatives](results/controller-checks.json), [rank/step/staged-call samples](results/timing-samples.json), [CSV](results/timing-samples.csv), [per-rank summaries and maximum-rank steps](results/timing-summary.json), [reference metrics](results/reference-metrics.json), [peer failure/update logs](results/peer-failure.json), [installed CLI negatives](results/cli-negatives.json), [runtime freeze](results/runtime.txt), [controller log](results/checks.txt) and [SHA-256 manifest](results/sha256.json). The empty controller log is retained as actually produced; detailed rank logs live in the external bundle.

All 106 behavioral working-tree file hashes still match the full run. Raw CSV retains its original CRLF bytes through a binary text attribute, and raw driver output retains its generated whitespace; their SHA-256 values are preserved rather than normalizing forensic evidence. Source/other report text uses normal repository rules. Exact final Git/source correspondence, commit/push and implementation-revision CPU CI remain required delivery evidence. #11/#9/M2 remain open until that passes. This is two-process functional training on one physical laptop GPU; no GPU-container, native Windows, physical-host, distinct-GPU scaling/NCCL or GPU-nightly execution is claimed.

## Installed development smoke

The first fresh installed `--mode smoke` run passed all 10 cases, 13 corruption cases and both deferred launcher-signal fixtures. Four standard-library controller negative cases passed without GPU execution. Every GPU case completed in 4.442–9.135 seconds with owned processes reaped. One-rank GPU trajectories matched exactly; two-rank/three-step trajectories had maximum absolute error `2.2351741790771484e-8`, at unchanged `atol=1e-6, rtol=1e-5`. Smoke correctly reported `cuda_training_acceptance=false`, with zero post-five-step-warmup timing samples. It is not 100-step acceptance.

Command: `build/toolchain-venv/bin/python tools/check_cuda_trainer.py --mode smoke --output /home/mohri/.cache/meshtrain-validation/cuda-trainer-smoke-01 --wheelhouse /home/mohri/.cache/meshtrain-validation/cuda-first/cuda-wheels`. The sdist-derived wheel was built/installed in a new isolated runtime; source fingerprint `3c3209166df88438eae3055131cccba2fe314c62079f7f259ff1b4124b25c98e` covered 106 behavioral files. Raw setup/run/process logs remain in that external directory; host console log is `build/cuda-trainer-smoke-01.log`.

The real peer-close fixture produced `peer_closed` at rank 0 and `invalid_state` at rank 1, both at sequence 2. Each SGD log contained only completed step 0, with no second update or completion marker. Small follow-up refinements add runtime-component integrity, bit-preserving rejection snapshots, explicit cross-stream dependency averaging and CUDA-specific CLI acceptance output; the fresh full run will test those final bytes separately.

## CPU compatibility

`docker build --target trainer-smoke-test -t meshtrain-trainer-smoke-test:cuda-11-dev --progress plain .` passed 63 numerical comparisons and its focused checks, including CPU-only unavailable-CUDA rejection. `docker build --target python-bindings-test -t meshtrain-python-bindings-test:cuda-11-dev --progress plain .` passed all 10 Torch-free binding checks. Logs: `build/cuda-trainer-cpu-regression.log` and `build/cuda-trainer-binding-regression.log`. Exact final implementation-revision CPU CI remains required; hosted CPU CI does not prove GPU behavior.

## Historical planning record

The following observations were recorded before implementation; they do not substitute for the live evidence above or the required full run.

## Starting state and existing evidence

Started at `496c245a8aa375f9dadcad5e545116eff6efc815` on `main`. The only pre-existing local changes were unrelated untracked `.agents/skills/find-skills/`, `.claude/` and `skills-lock.json`; preserve them. Read #11 and #9 through `gh`: both are open, #10 is checked complete and all #11 acceptance boxes remain unchecked.

Inspected repository instructions, roadmap/decisions/issues, CPU MVP design, accepted ADRs 0001–0004, CPU trainer/adapter/reference/verification/artifacts/launcher, CUDA staging/runner/controller, dependency/package metadata and actual CI workflows. CPU push/PR has native/binding/short-trainer jobs; M1 has scheduled/manual broad CPU validation. No GPU nightly runner is configured.

The [#10 hardware report](../host-staged-cuda-adapter/verification.md) establishes the installed tensor primitive on one RTX 5070 Ti Laptop GPU and two real processes sharing UUID `8f5c096d-be03-fe0b-17fa-9a35b08e251d`. It does not establish GPU model training or the proposed FP32/deterministic policy. Reuse its runtime/lock and preserve its evidence as history.

## Read-only runtime observation

Executed the existing external CUDA environment's Python with `-I` to print Python/Torch/CUDA versions, three precision-setting getters and `torch.cuda.is_initialized()`. It reported CPython `3.12.13`, Torch `2.14.1+cu132`, CUDA tag `13.2`; the global, CUDA matmul and cuDNN precision getters each returned `none`, with CUDA uninitialized (`False`). No settings were changed and no GPU allocation/kernel/training ran. This supports making requested/effective precision explicit rather than treating current defaults as acceptance evidence.

Existing runtime path: `/home/mohri/.cache/meshtrain-validation/cuda-final/runtime/bin/python`. The future delivery runner must still build/install the new wheel and prove source/runtime/device identity; this observation cannot substitute for that check.

## Planning checks

Checked nine changed/new Markdown documents: all 52 local file links resolve, the three OpenSpec artifacts exist, no template placeholders remain, ADR 0005 is Proposed, and all seven implementation tasks remain unchecked. `git diff --check` passed. Changed tracked files are documentation only; no production/test/config/dependency implementation file was changed. No unrelated training suite was run for this documentation task.

The contract self-review walked through CPU schema-1 preservation, successful GPU normalization/order/reference, rejected metadata/nonfinite data, loss failure after gradient synchronization, a real failed peer before SGD, reference initialization stalls, stale success and shared-GPU memory/timing claims. It clarified early precision/environment setup, poisoning for failures outside the inner staging primitive, capture rejection before validation kernels, and direct-child ownership for the complete reference/worker controller. A killed nested supervisor cannot be described as reaping its children.

After these planning checks, the owner selected independent review and conditionally authorized implementation. [Review/disposition](review.md) records that completed route. Remaining implementation/GPU/CI tasks stay unchecked until their actual evidence passes; #11/#9/M2 remain open.
