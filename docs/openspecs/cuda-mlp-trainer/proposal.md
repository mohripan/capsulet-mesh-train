# Proposal: GPU MLP training and independent reference parity

Date: 2026-10-18

Status: implemented; installed full GPU acceptance and local CPU regressions passed. Exact-revision CPU CI/delivery pending.

Source: [issue #11](https://github.com/mohripan/capsulet-mesh-train/issues/11), under [M2 tracker #9](https://github.com/mohripan/capsulet-mesh-train/issues/9). The owner said “All right. Please continue!” after #10 was delivered and the next GPU trainer OpenSpec identified. This instruction starts the next planning task; the earlier conditional review/implementation instruction applied to #10.

## Problem and outcome

The installed CUDA tensor adapter works, but no model trains through it yet. Demonstrate that independent GPU workers use our C++ TCP SUM for actual gradients and produce the same learning trajectory as an ordinary single-process GPU reference within fixed tolerances. Retain separate staging costs and honest memory/device evidence.

Keep the existing deterministic `16 -> 32 -> ReLU -> 4` MLP, saved CPU initialization/data/schedule, equal local batches and SGD settings. Extend the existing trainer with an explicitly selected CUDA mode. Preserve CPU defaults, schema-1 artifacts, CPU adapter behavior and the standalone dependency boundary.

## Scope

- Add an optional CUDA gradient-packing/averaging adapter on top of `CudaStagingAdapter`, retaining stable parameter identity, device and layout.
- Extend preparation, independent worker commands, supervision and verification for a versioned CUDA training profile. Both gradient and scalar loss SUM use the staging primitive; SGD follows successful synchronization and finite-value checks.
- Compute an independent combined-batch reference on the selected GPU after worker processes exit. Compare every step's global loss, averaged gradients and updated parameters, all final state entries, rank pairs and recomputed evaluation losses.
- Add a reproducible installed-package GPU check with tiny one-worker and two-worker shared-GPU scenarios, pinned/pageable modes, focused contract/failure checks, raw timings and memory/environment records.
- Document the supported Linux/WSL runtime and the evidence needed to close #11 and M2. Draft [ADR 0005](../../adr/0005-cuda-training-reference-contract.md) for the additive training/artifact contract.

Excluded: new C++ algorithms or CUDA linkage, CPU numerical-contract changes, AMP/TF32 training, Adam or sharding, DDP/FSDP/backend integration, hooks/overlap, checkpoint recovery, GPU-container orchestration, driver installation, native Windows, physical-host or distinct-GPU validation, and NCCL baselines. Existing #10 tensor/stream/error semantics remain in force.

## Acceptance

1. CUDA mode requires an explicit indexed device. Model, features, labels, forward/backward, gradient averaging and SGD execute there; CUDA absence or unsupported execution fails explicitly without CPU fallback. CPU-only imports/builds/default runs retain their current behavior.
2. Each step completes exactly two native SUM calls, gradients then loss, followed by explicit division by world size. No optimizer update uses missing, invalid, nonfinite or failed collective results. One-rank identity and nondefault-stream completion receive live checks.
3. Two workers with different local batches train for 100 steps at local batch eight in each host-memory mode. Every numerical comparison passes `abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)`, evaluated in FP64 over finite FP32 traces. Each worker/reference has strictly lower final fixed-set evaluation loss. Keep the tolerances fixed; diagnose failures rather than silently relaxing them.
4. The reference uses the same saved initialization, rank-ordered combined batches, GPU precision policy, mean cross-entropy and SGD. It computes its own forward/backward/updates and never uses worker gradients or parameters to generate expected training values. Separate integrity/evaluation checks may read worker final states.
5. Versioned artifacts distinguish CPU M1 full acceptance from CUDA training acceptance. Missing, stale, tampered or incomplete output and wrong device/precision/mode/sequence metadata fail verification. A tiny smoke run cannot claim the 100-step CUDA acceptance profile.
6. Record actual GPU UUID/visibility, versions/build settings, deterministic/precision controls, allocated/reserved peaks, explicit host/device buffer bytes and per-step staging/native/total timings. Preserve raw rank/step correlation. Sharing one GPU establishes functional training, with no multi-GPU scaling claim or shared-device NCCL comparison.
7. A real peer failure before the second step's gradient collective prevents that step's SGD and produces nonzero group failure, useful diagnostics and reaped owned workers. An external controller bounds the complete installed run, including GPU reference computation; unknown GPU cleanup cannot become a pass.
8. Required focused GPU checks, affected CPU regressions and exact implementation-revision CPU CI pass before closing #11. Close #9/M2 only after its child/evidence criteria are actually satisfied. Current CPU nightly/manual validation remains configured; no GPU runner is implied.

## Review and authorization

- Review route: fresh independent sub-agent `review_cuda_trainer_spec`, started with `fork_turns="none"` and raw issue requirements/artifact paths, without the author's conversation or desired verdict.
- Findings and disposition: [independent review](review.md) completed without findings; no spec corrections required. Planning self-checks are recorded in [verification.md](verification.md).
- User go-ahead: on 2026-10-18 the owner said “All right. Please continue! I prefer the sub-agent, and then you can go implement it.” This explicitly authorizes implementation of this presented #11 scope after the independent review and resolution of findings. Do not introduce another approval gate for corrections within that scope.
- ADR 0005 is Accepted under the owner's actual conditional instruction after review completed. ADRs 0001–0004 remain accepted history. Required runtime evidence remains pending.

## Delivery

The reviewed/authorized CUDA gradient/training layer, schema-2 GPU reference/integrity checks and installed verification runner are implemented. Both 100-step memory modes and focused checks passed on the shared laptop GPU. [Actual evidence](verification.md) and raw records accompany the [tasks](tasks.md), [review](review.md) and Accepted ADR 0005. Exact implementation-revision CPU CI/delivery remain pending; #11, #9 and milestone 2 remain open.
