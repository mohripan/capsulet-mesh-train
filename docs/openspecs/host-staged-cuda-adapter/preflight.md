# CUDA adapter planning preflight — 2026-10-17

Status: read-only planning evidence. No CUDA adapter or GPU runtime was installed, built or executed for this task.

## Repository and dependencies

The working tree started at `1a669838006527bb13fbec96c17d9810fd5bac97` on main, with only the unrelated untracked `.agents/skills/find-skills/`, `.claude/` and `skills-lock.json`; preserve them. M1/#2 and milestone 1 are closed, and #10's #7 dependency is satisfied. Issue #10 and M2/#9 remain open. The final M1 documentation commit's regular [CI run 39031034379](https://github.com/mohripan/capsulet-mesh-train/actions/runs/39031034379) also passed; M1's full proof remains its earlier exact implementation revision.

Inspected `AGENTS.md`, roadmap/decisions/issues, ADRs 0001–0003, the CPU adapter, pybind11 buffer/communicator ownership, package metadata and CPU lock. The existing adapter accepts CPU parameters only. Native `RunConfig` defaults to a 256 MiB payload cap; the binding has no CUDA pointer/tensor API and validates genuine writable one-dimensional FP32 NumPy storage. Preserve those contracts.

## Observed environment

Read-only commands:

```powershell
nvidia-smi
wsl --list --verbose
wsl -d Ubuntu --exec sh -lc 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'
wsl -d Ubuntu --exec sh -lc 'command -v nvcc; python3 --version'
```

Observed Windows and Ubuntu/WSL2 GPU visibility, one NVIDIA GeForce RTX 5070 Ti Laptop GPU with 12227 MiB. Windows NVIDIA-SMI/KMD reports 616.87; WSL's utility banner reports 615.73 and KMD 616.87, with CUDA UMD capability 13.4. These banners do not identify an installed PyTorch CUDA runtime or prove supported kernels. `nvcc` was not found in the WSL command path; no toolkit/driver installation was attempted. Host WSL Python is 3.14.4 and Docker Engine 29.6.2.

The existing `build/training-dev/bin/python -I` reported CPython 3.12.13, Torch `2.14.1+cpu`, `torch.version.cuda=None`, and `torch.cuda.is_available()=False`. This environment is intentionally CPU-only; its result does not show a broken GPU. A separate installed CUDA runtime remains an implementation prerequisite. Docker daemon metadata was inspected, but no GPU container was launched and no GPU pass-through claim is made.

## Primary references inspected

- [PyTorch 2.14 CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda): asynchronous work, stream ordering, backward-stream dependencies and completion-aware measurements.
- [Tensor.copy_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.copy_.html): CPU/GPU copy submission and explicit nondefault-stream obligations. The adapter will use a stronger completed-return contract.
- [Tensor.record_stream](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html): storage lifetime for side-stream use; synchronous completion is the proposed initial boundary.
- [Stream API](https://docs.pytorch.org/docs/stable/generated/torch.Stream.html) and [capture query](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_current_stream_capturing.html): completion/ordering observations and explicit capture rejection.
- [Official CUDA 13.2 Torch wheels](https://download.pytorch.org/whl/cu132/torch/): a `2.14.1+cu132` CPython 3.12 Linux x86-64 wheel is listed. Wheel execution and dependency hashes remain unverified here.
- [PyTorch CUDA packaging announcement](https://dev-discuss.pytorch.org/t/removing-cuda-13-0-builds-from-pytorch-ci-cd-starting-the-week-of-sept-28/3447): informs the proposed CUDA 13.2 binary choice; actual component versions still need recording.
- [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html): Windows-managed GPU driver boundary and WSL constraints.

This research supports the proposed contracts and candidate environment. It is not CUDA correctness, multi-process training, native Windows, physical multi-host or distinct-GPU performance evidence.

## Planning verification

Local checks passed for all task/ADR/navigation links, complete templates, Proposed ADR status, the explicit pending go-ahead and unchecked implementation tasks. `git diff --check` passed. A self-review walked through successful pinned/pageable/empty calls, local rejection, native failure before H2D, post-native CUDA failure and concurrent workspace rejection. It clarified poisoned-workspace reuse, communicator closure before potentially stuck GPU settling, and input reset between timing samples.

These are documentation/contract checks. No implementation tests were run for this planning change, and no new scheduling, GPU runtime or hardware correctness claim follows from them.
