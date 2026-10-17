# Proposal: Safe Python CPU buffer bindings

Date: 2026-10-17

Status: implemented and locally verified under the owner's conditional go-ahead; remote delivery in progress.

Source: [issue #6](https://github.com/mohripan/capsulet-mesh-train/issues/6), following the owner's request to continue after the repository workflow skills.

## Problem and outcome

The standalone C++ engine can perform centralized FP32 SUM across real processes. Python cannot yet call that collective. Provide an optional installable package that mutates the caller's NumPy storage, keeps that storage alive throughout the blocking operation, and lets other Python threads progress during native waits. This is the dependency for the PyTorch trainer in #7.

## Scope

- Add a scikit-build-core/pybind11 package named `meshtrain`, importing the extension as `meshtrain._native` and linking the existing core.
- Expose explicit immutable configuration, blocking communicator construction, in-place `all_reduce_sum`, close/context-manager support, state/sequence inspection, and contextual native errors.
- Validate the actual NumPy array without conversion; retain the array/exported buffer and native communicator during calls; release the GIL only around Python-independent native work.
- Add focused installed-package checks, a reproducible Python container target and CI coverage, documentation, and [ADR 0002](../../adr/0002-python-buffer-ownership.md).

The standalone build remains the default and does not discover Python or NumPy. This task excludes a trainer/launcher, PyTorch dependency, arbitrary buffer exporters or pointer arguments, CUDA, asynchronous work, new algorithms, native Windows networking, package publication, and CI nightly scheduling. The disposable toolchain probe remains available; the real package becomes the Python CI validation target.

## Acceptance

1. A wheel built from the source distribution installs and imports in a fresh Linux CPython 3.12 environment outside the checkout. It contains the Python package and extension, without bundling native headers/archive or test fixtures. The independent C++ installation/consumer still works.
2. The binding accepts only writable, one-dimensional, C-contiguous, native-endian FP32 NumPy arrays with valid lengths and suitably aligned nonempty storage. Owning arrays, contiguous offset views, and empty arrays work. Wrong dtype/byte order, array-like conversion, read-only, noncontiguous, misaligned, scalar, and multidimensional inputs fail explicitly without modifying the input or starting a collective.
3. A small real two-process TCP exchange sums exactly representable fixtures in place, including a view whose base and untouched boundary sentinels are checked. Empty calls participate and sequence advances only on success. There is no hidden division or collective delegation.
4. The exporting object, storage owner chain, and native communicator survive until the call returns or raises. Python references and buffer exports are released with the GIL held, including failure paths; repeated calls do not retain finished buffers.
5. Controlled delayed-peer scenarios demonstrate Python thread progress during construction and collective waits, concurrent-call rejection, close during a wait, and deadline-bounded failure. Exceptions preserve native categories/context; admitted native failures leave the communicator unusable and may leave a partially changed input. Concurrent-call rejection leaves the original call active.
6. Focused checks exercise the installed extension rather than a stub or source-tree fallback. Record commands, versions, actual numerical/timing results, and limitations before closing #6. Broad training/container matrices remain later work.

## Review and authorization

- Review route: fresh, context-independent sub-agent review, selected on 2026-10-17. The reviewer starts without inherited conversation and may inspect the repository and primary references; its work is read-only.
- Findings and disposition: [independent review](review.md) found no blockers and one low-priority coverage suggestion. Include valid ndarray-subclass and buffer-acquisition-failure cases in the focused checks.
- User go-ahead: on 2026-10-17, the owner said, "Independent sub-agent review, and then you can go and implement." This authorizes the presented #6 scope after review and disposition, without a second approval question.
- Commit/push: the owner's standing instruction applies to completed planning documents and, after go-ahead and verification, implementation; no co-author trailers.

## Delivery

The optional package, strict buffer boundary, contextual errors and focused verifier are implemented. [Verification](verification.md) records passing package checks in WSL and the pinned Python container, real two-process exact sums, lifetime/GIL/failure evidence, native independence and limits. [Tasks](tasks.md) tracks remote delivery; #6 remains open until that work is completed. ADR 0002 is accepted under the owner's actual conditional go-ahead.

Planning checks on 2026-10-17: local Markdown targets resolved across the three task documents, ADR 0002 and ADR index; no unresolved template fields or TODO/TBD markers remained; the diff passed whitespace checks. Author self-review reconciled pre-admission rejection, admitted failure and concurrent-call rejection with M1/ADR 0001. These are document checks, not live binding/package evidence.
