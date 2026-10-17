# Proposal: M1 failure, clean-build and four-container validation

Date: 2026-10-17

Status: implementation and complete local acceptance passed; exact-revision manual CI/delivery pending.

Source: [issue #8](https://github.com/mohripan/capsulet-mesh-train/issues/8), after completion of #7. The owner's current request is: “Okay. Lets go next! Please make the OpenSpec and use sub-agent to review.”

## Problem and outcome

The installed trainer already passes local four-worker/100-step parity, and native tests cover many protocol and collective failures. M1 still needs reproducible clean validation, controlled process kill/stall evidence with the accepted five-second bound, and the same trainer in four separate Linux containers. Close these evidence gaps without changing the numerical workload or introducing recovery/orchestration features.

The [accepted M1 specification](../../superpowers/specs/2026-10-16-cpu-mvp-design.md) remains the contract. [Design](design.md) defines the harness, failure timing and deployment; [tasks](tasks.md) defines implementation and live verification.

## Scope

- Add Linux-only native fault fixtures and controlled root/nonroot kill/stall cases for startup and an admitted collective, with structured evidence and repeated resource cleanup checks.
- Add a validation image that installs the real sdist-derived Python wheel and locked CPU dependencies, plus a host-side Docker controller that runs the existing worker in four isolated containers and verifies collected artifacts afterwards.
- Automate native TCP-on/TCP-off installed-consumer checks, Torch-free bindings, local trainer parity and container training from fresh build/runtime directories. Record commands, environments, timing, digests, logs and failure evidence.
- Add a nightly/manual M1 validation workflow and artifact retention. Keep the current push/PR regressions; exclude only the newly added broad fault label from their default native invocation. This task adds that scheduling lane; existing regressions remain.
- Update validation documentation and GitHub navigation after actual acceptance. Review existing ADRs; no new production interface/ownership decision is proposed, so no new ADR is initially needed.

Excluded: new algorithms, performance baselines, CUDA, native Windows, physical multi-host/distinct-GPU claims, Kubernetes, scheduling/admission policy, heartbeat detection, automatic retries, checkpoints, elastic membership, public fault-injection APIs, dependency upgrades and production image size optimization. Existing native shell checks are reused; unrelated refactors and wholesale CI reshuffling are excluded.

## Acceptance

1. One documented Linux/WSL command runs a fresh-directory, no-build-layer-cache M1 validation at the current source revision. It builds/tests standalone TCP-on and TCP-off consumers without Python, installs/tests the actual package without Torch, and runs the locked CPU trainer. Hash-checked dependency caches may be reused and are reported honestly; this is not a hermetic/offline cold-machine claim.
2. Eight controlled native cases cover startup/active collective × target root/nonroot × external SIGKILL/SIGSTOP. Each uses a one-second operation deadline and a five-second overall case bound from first worker spawn through owned-child cleanup/resource checks, including fixture synchronization. Signals act on observed owned PIDs at proven phases; sleeps alone are not phase evidence.
3. Native survivors fail through useful engine errors, rather than passing because a harness kills them first. Failed collectives retain terminal failed state/sequence and reject reuse. Startup failures have no constructed communicator. Expected-failure group statuses are nonzero while the test verdict passes only when the expected failure and cleanup are demonstrated.
4. Each healthy survivor measures its own descriptors before connecting and after failed-constructor/communicator/transport cleanup, before it exits; require baseline restoration. Repeat 20 additional failed groups across the matrix and separately verify controller descriptors, child reaping and listener reuse. Killed-target OS cleanup does not prove native cleanup. Watchdog, timing, resource and unexpected-error failures fail validation; no silent retries or widened bounds.
5. Four separate Linux containers each execute one installed `meshtrain.training worker`, with explicit rank, ordered hostname endpoints and a fresh common run ID. Inputs are copied before launch; ranks have independent writable filesystems. All gradient/loss traffic uses the native TCP engine. Collect outputs only after workers stop and compute the ordinary independent reference afterwards.
6. Run four-container/three-step smoke and the unchanged four-container/100-step/local-eight full profile. The full run requires all per-step/reference/rank-pair, final-state and evaluation checks at `atol=1e-6, rtol=1e-5`, lower finite evaluation loss and 200 successful native collectives per rank. Missing/stale artifacts or a worker failure cannot produce overall success.
7. Prove Docker-controller failure/interruption/setup cleanup with actual owned containers. Preserve rank logs, first observed actionable failure and cleanup results. Remove only this attempt's containers/network; leave unrelated sentinel resources intact. A cleanup failure makes the overall validation fail even if numerical parity passed.
8. Execute one complete manual validation before closing #8; nightly configuration alone is insufficient. Attach/read back the exact-revision CI and raw evidence. Complete #2/M1 and its milestone only after every required child and outcome is verified; then point the roadmap to M2's first unmet dependency.

## Review and authorization

- Review route: the owner's requested fresh independent review completed on 2026-10-17 (`review_m1_validation_spec`, no inherited conversation). See [review.md](review.md) for coverage, findings and follow-up.
- Findings/disposition: corrected the required worker descriptor-evidence gap and clarified host/container prerequisites; also separated repeat-batch watchdogs and Docker cleanup budgets. The reviewer re-read all three revised artifacts and found no remaining blockers or new material inconsistencies.
- User go-ahead: on 2026-10-17 the owner explicitly said “Okay. Please implement!” after the reviewed #8 plan was presented. This authorizes the presented scope, including its manual/nightly validation lane. Independent review and corrections were already complete; no further approval gate is needed.
- Commit/push: standing owner instruction applies to completed authorized work, to `main` without co-author trailers. #8 and M1 remain open until all required acceptance evidence is verified.

## Delivery

The reviewed scope is implemented under the owner's explicit go-ahead. Native fixtures, controller/runtime/clean-run tools and the nightly/manual workflow are written. Focused checks and complete local clean acceptance passed; exact-revision manual CI and artifact inspection remain required before #8/M1 closure. See [actual implementation evidence](verification.md); prior #7 results are context rather than substitutes for #8 acceptance.
