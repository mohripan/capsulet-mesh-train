# Independent CUDA adapter OpenSpec review — 2026-10-17

Reviewer: `review_cuda_adapter_spec`, spawned with `fork_turns="none"`. It received the raw #10 requirements, repository path and three artifact paths, without the author's conversation/reasoning or requested verdict. The owner selected that route and authorized implementation afterwards: “Sub-agent. And then you can go on and implement it.”

## Findings and disposition

1. **P2 blocker — inference tensors and workspace context.** An ordinary contiguous inference tensor can have `requires_grad=False`, but later in-place mutation outside inference mode can fail after SUM has advanced sequence. Constructing workspace inside inference mode can also make it unsuitable for later mutation. Resolved by rejecting inference inputs before any copy/native admission and allocating workspace under `torch.inference_mode(False)`, restoring the caller context. Added focused construction/rejection/sequence checks.
2. **P2 blocker — forward-mode AD dual tensors.** Dual tensors can satisfy the original metadata checks. Copy operations can carry their tangent while native NumPy SUM changes only primal storage. The first correction used `forward_ad.unpack_dual`, but follow-up found that caller inference mode hides its tangent. The revised contract performs that query under `torch.inference_mode(False)`, restores caller mode and rejects before any copy/native admission; verify unchanged primal/tangent/sequence both normally and when an external dual is passed inside inference mode. No silent tangent detach/coercion. The final follow-up confirmed this guarded correction resolved the finding.

The reviewer reproduced both mechanisms with existing CPU Torch 2.14.1 API probes. These observations support the missing-contract findings, not CUDA/collective execution. Primary references: [inference mode](https://docs.pytorch.org/cppdocs/api/autograd/modes.html#inferencemode), [make_dual](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.forward_ad.make_dual.html), [no_grad](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html).

The remainder of the review found the scope consistent with the CPU core/binding contracts and adequate for stream ordering, completed transfers, retained owners, memory modes, containment and separate measurements. It inspected revision `aa75e18`, artifacts/preflight, ADRs 0001–0004, relevant source and actual CI; no files were edited or installations performed. The two corrections enforce the already proposed exclusion of autograd and do not expand task scope.

## Follow-up and limits

The final follow-up confirmed both P2 findings resolved, with no remaining material technical inconsistency. Its nonblocking stale authorization wording in the delivery task was corrected. Review does not grant implementation authorization; the owner's explicit conditional instruction supplies it after review/disposition. GPU compatibility, lifetime, numerical behavior, timing and cleanup remain required implementation evidence.
