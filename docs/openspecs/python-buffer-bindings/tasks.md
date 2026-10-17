# Tasks: Safe Python CPU buffer bindings

Implementation status: awaiting review choice and the go-ahead recorded in [proposal.md](proposal.md). All commands below are planned checks, not claims of existing scripts or passing results.

- [ ] **Resolve review and record implementation authorization.** Present the three artifacts and proposed ADR; address the selected personal or independent review, then record the owner's actual go-ahead/conditions.
  - Verification: inspect the review record against the actual instruction and presented scope; a reviewer verdict alone does not complete this gate.
  - Evidence: pending owner response. No independent reviewer has been launched.

- [ ] **Build/install the optional package and expose configuration/lifecycle.** Add the opt-in CMake boundary, Python package metadata/exports, copied read-only config, communicator ownership, state/sequence/close/context-manager API, and contextual errors.
  - Live verification: build an sdist and its wheel with the pinned tools, install/import in an isolated CPython 3.12 environment outside the source tree, inspect wheel files, and exercise valid/invalid configuration plus one-rank construction/closure. The final verification script must reproduce this path.
  - Live verification: configure/build the default core and run `ctest --test-dir build/gcc -R '^(core_contracts|communicator_contracts|installed_consumer)$' --output-on-failure`; check the corresponding portable contracts with TCP disabled. Build `docker build --target native-test -t meshtrain-native-test:dev .` to confirm Python-free operation. Record actual build-directory names/configuration if they differ.
  - Evidence: pending implementation and execution.

- [ ] **Implement strict borrowed-buffer SUM with correct GIL scopes.** Validate actual ndarray/export metadata and alignment without conversion; retain Python/native owners outside the released-GIL scope; delegate only to our native collective.
  - Live verification: installed-extension rejection matrix and empty/singleton/identity cases; rejected inputs stay unchanged with no sequence advance. Check native payload-limit failure remains terminal.
  - Live verification: actual two-process TCP SUM for counts 0, 1, and 1025 across repeated calls, checking exact expected sums, original storage addresses, offset-view base updates, sentinels, and sequence at both ranks.
  - Evidence: pending implementation and execution; record fixture counts and numerical metrics after checks run.

- [ ] **Demonstrate storage lifetime, Python progress, and bounded lifecycle failures.** Add controlled peer-delay fixtures and watchdog cleanup around the installed extension.
  - Live verification: Python-thread handshakes during blocked startup and collective calls; view/base weakrefs/finalizers survive external-reference deletion during communication and are released after success/failure. Repeated group/call cleanup leaves no retained finished inputs or child processes.
  - Live verification: close during a wait, overlapping call rejection followed by success of the original call, and a stalled peer with a one-second operation deadline/five-second external watchdog. Assert contextual exceptions, terminal/refused-reuse behavior, and idempotent close; record actual elapsed times.
  - Evidence: pending implementation and execution. Do not substitute mocks or sleeps alone for communication/thread progress.

- [ ] **Make the focused checks reproducible in CI and finish records.** Add runtime dependency hashes, `tools/check_python_bindings.py`, and the `python-bindings-test` container stage; use the real package check in the Python CI slot while retaining the manual probe. Document the API/commands, reconcile the specs, and finalize ADR 0002 only against actual authorization.
  - Live verification: `python tools/check_python_bindings.py` in the populated Linux CPython 3.12 build environment, then `docker build --target python-bindings-test -t meshtrain-python-bindings-test:dev .`; capture actual package versions, isolated-import paths, numerical/timing output and pass/fail evidence in `verification.md`.
  - Verification: check local links/ADR status/index, formatting and comment lengths; review the final diff for standalone-core independence, no hidden copies/collective delegation, accurate CI scheduling claims, and unchanged unrelated files. Reuse already-passing checks unless changes invalidate them.
  - Delivery: update #6/parent dependency status with reproducible evidence, commit/push authorized implementation to `main` without a co-author trailer, and confirm the remote revision and relevant CI result.
  - Evidence: pending implementation, checks, and delivery. Broad trainer/E2E/nightly scheduling remains separate.

## Completion

No implementation or live package verification has run for #6. The issue remains open. Record completed scope, evidence, commit/CI links and remaining platform/hardware limits here after execution. Required failed, blocked or unrun checks leave their steps unchecked.
