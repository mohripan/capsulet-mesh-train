# GPU trainer planning evidence — 2026-10-18

Status: read-only planning and documentation checks. No #11 implementation, installation or GPU training was performed.

## Starting state and existing evidence

Started at `496c245a8aa375f9dadcad5e545116eff6efc815` on `main`. The only pre-existing local changes were unrelated untracked `.agents/skills/find-skills/`, `.claude/` and `skills-lock.json`; preserve them. Read #11 and #9 through `gh`: both are open, #10 is checked complete and all #11 acceptance boxes remain unchecked.

Inspected repository instructions, roadmap/decisions/issues, CPU MVP design, accepted ADRs 0001–0004, CPU trainer/adapter/reference/verification/artifacts/launcher, CUDA staging/runner/controller, dependency/package metadata and actual CI workflows. CPU push/PR has native/binding/short-trainer jobs; M1 has scheduled/manual broad CPU validation. No GPU nightly runner is configured.

The [#10 hardware report](../host-staged-cuda-adapter/verification.md) establishes the installed tensor primitive on one RTX 5070 Ti Laptop GPU and two real processes sharing UUID `8f5c096d-be03-fe0b-17fa-9a35b08e251d`. It does not establish GPU model training or the proposed FP32/deterministic policy. Reuse its runtime/lock and preserve its evidence as history.

## Read-only runtime observation

Executed the existing external CUDA environment's Python with `-I` to print Python/Torch/CUDA versions, three precision-setting getters and `torch.cuda.is_initialized()`. It reported CPython `3.12.13`, Torch `2.14.1+cu132`, CUDA tag `13.2`; the global, CUDA matmul and cuDNN precision getters each returned `none`, with CUDA uninitialized (`False`). No settings were changed and no GPU allocation/kernel/training ran. This supports making requested/effective precision explicit rather than treating current defaults as acceptance evidence.

Existing runtime path: `/home/mohri/.cache/meshtrain-validation/cuda-final/runtime/bin/python`. The future delivery runner must still build/install the new wheel and prove source/runtime/device identity; this observation cannot substitute for that check.

## Planning checks

Checked nine changed/new Markdown documents: all 52 local file links resolve, the three OpenSpec artifacts exist, no template placeholders/TODO/TBD remain, ADR 0005 is Proposed, and all seven implementation tasks remain unchecked. `git diff --check` passed. Changed tracked files are documentation only; no production/test/config/dependency implementation file was changed. No unrelated training suite was run for this documentation task.

The contract self-review walked through CPU schema-1 preservation, successful GPU normalization/order/reference, rejected metadata/nonfinite data, loss failure after gradient synchronization, a real failed peer before SGD, reference initialization stalls, stale success and shared-GPU memory/timing claims. It clarified early precision/environment setup, poisoning for failures outside the inner staging primitive, capture rejection before validation kernels, and direct-child ownership for the complete reference/worker controller. A killed nested supervisor cannot be described as reaping its children.

The owner has been offered personal review or a fresh independent review with conditional implementation after findings are resolved. No answer or go-ahead has yet been recorded. No sub-agent was launched for this task. Required implementation/GPU checks remain unchecked in [tasks.md](tasks.md); #11/#9/M2 remain open.
