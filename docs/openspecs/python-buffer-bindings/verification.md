# Verification: Python buffer bindings

Date: 2026-10-17. Scope: [issue #6](https://github.com/mohripan/capsulet-mesh-train/issues/6), authorized after [independent specification review](review.md).

## Environments and actual results

| Check | Result |
| --- | --- |
| Ubuntu 26.04 / WSL2, x86-64, GCC 15.2.0, glibc 2.43, CPython 3.12.13 | Sdist-derived wheel, fresh runtime install/import, 10/10 focused Python checks passed in 2.009 s. |
| Pinned Python 3.12.15 / Debian bookworm container, GCC 12.2.0 | Sdist-derived wheel, offline hash-checked NumPy install, 10/10 checks passed in 2.059 s. Includes the final absolute-watchdog refinement. |
| WSL native Debug, warnings as errors | Core/communicator contracts and relocated installed consumer: 3/3 passed, 14.01 s. |
| WSL native Release, TCP disabled, warnings as errors | Corresponding portable contracts and installed consumer: 3/3 passed, 4.66 s. |
| Pinned native Debian container, GCC 12.2.0 | Asserted no Python interpreter installed; native build and existing 7/7 CTests passed, 12.90 s. |

Both Python builds used build 1.6.1, CMake 4.4.4, Ninja 1.13.2, pybind11 3.1.0, scikit-build-core 1.1.1 and NumPy 2.5.4, from separate hash-locked build/runtime requirements. Native WSL checks used system CMake 4.2.3/Ninja 1.13.2. Docker Engine was 29.6.2 on this Windows/WSL machine; the Dockerfile pins base digests and direct apt tool versions.

Raw passing output, with trailing whitespace normalized: [WSL package checks](verification-wsl.txt), [final Python container](verification-container.txt), [Python-free native container](verification-native.txt). WSL package evidence precedes the final harness refinement to one absolute five-second scenario watchdog; the final container checks that refinement against the same binding implementation.

## Commands executed

From Linux/WSL in `/mnt/c/Projects/capsulet-mesh-train`, with the existing Python 3.12 build environment populated from `requirements/toolchain.lock`:

```sh
uv pip compile --python-version 3.12 --generate-hashes requirements/python-runtime.in -o requirements/python-runtime.lock
build/toolchain-venv/bin/python tools/check_python_bindings.py
cmake -S . -B build/gcc -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/gcc
ctest --test-dir build/gcc -R '^(core_contracts|communicator_contracts|installed_consumer)$' --output-on-failure
cmake -S . -B build/no-tcp -G Ninja -DCMAKE_BUILD_TYPE=Release -DMESHTRAIN_ENABLE_TCP=OFF -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/no-tcp
ctest --test-dir build/no-tcp -R '^(core_contracts|communicator_contracts|installed_consumer)$' --output-on-failure
```

Docker commands from PowerShell at the repository root:

```sh
docker build --target native-test -t meshtrain-native-test:dev --progress plain .
docker build --target python-bindings-test -t meshtrain-python-bindings-test:dev --progress plain .
```

Additional checks passed: configured clang-format on binding C++, Python 3.12 AST parsing of five Python/stub files, local Markdown targets, and Git whitespace checks. These complement execution rather than prove runtime behavior.

## Observations

The frontend explicitly built its wheel from the sdist. Each fresh runtime imported the extension from its temporary environment outside the checkout. The wheel contained package exports, type stub/marker, compiled module and distribution metadata/license, with no native SDK or test fixtures. Extension and distribution versions matched at 0.1.0; the package retains the project's Apache-2.0 license.

Two independent spawned processes performed four SUMs per rank at counts `0, 1, 1025, 1`. Input was exactly representable FP32 `arange(count) + rank`; expected SUM was `2 * arange(count) + 1`. Every element matched exactly, maximum absolute error **0.0**. Original pointers, offset-view base updates, untouched boundary sentinels and sequence advancement were checked. One-rank identity preserved signed-zero, NaN-payload, infinity and ordinary-value bits.

Rejection checks covered array-like hooks without conversion, lists/memoryviews/integer addresses, float64/int32/swapped-endian, read-only, strided/reversed multi-element, misaligned, scalar and 2D inputs. Bytes/state/sequence stayed unchanged, followed by valid use. Empty/singleton degenerate contiguous arrays and an ndarray subclass worked. Python-defined acquisition failure preserved `BufferError`; an exporter returning different storage was rejected. Configuration checks covered copied/read-only snapshots, numeric types/ranges/defaults and semantic errors. A native payload-limit failure remained terminal.

Controlled peer withholding required another Python thread to execute a handshake before startup/collective could finish. View/base weakrefs stayed live after external-reference deletion during communication; finalizers ran afterwards. A Python exporter confirmed exactly one release callback after success, timeout and close. The active communicator survived external-reference deletion and was collectible after return. Overlapping-call rejection preserved the original successful call. Twelve additional single-rank create/call/destroy cycles released their input/communicator objects; subprocess groups were joined/closed by the harness.

| Observed timing | WSL | Final container |
| --- | --- | --- |
| Startup Python handshake | 0.052408 s | 0.050314 s |
| Delayed successful collective | 0.056301 s | 0.061624 s |
| Explicit-close scenario | 0.055740 s | 0.057616 s |
| One-second stalled-peer deadline | 1.004040 s | 1.001742 s |

These are correctness timings including a deliberate 50 ms withheld-peer check, Python scheduling and GC; they are not communication benchmarks. The final independent parent enforces an absolute five-second operation watchdog with bounded joins/forced cleanup. Failure checks verified native context fields, terminal state/refused reuse, export cleanup and idempotent close.

## Findings resolved and limits

The first development run failed two assertions: native errors include operation phase names, and cancellation can report `peer_closed` or `invalid_state`. The existing communicator/native cancellation tests confirmed both; expectations were corrected and targeted reruns passed. Numerical thresholds/native failure behavior were unchanged.

The first container built its wheel but exceeded a 240-second setup timeout downloading NumPy. A separate cached dependency layer now prepares the hash-checked wheel, with indexes disabled during runtime installation. Dependency preparation completed and offline verification passed. Communication deadlines/watchdogs were not relaxed.

Evidence establishes local Linux/WSL CPU processes and one Linux container, not PyTorch training parity, four-container Python training, CUDA, native Windows networking, physical multi-host, distinct GPUs, every NumPy 2.x release, or other Python ABIs. #7/#8 and later milestones retain those gates. Existing broader native push/PR checks remain; no nightly schedule was added. Remote delivery/CI references are recorded in [tasks.md](tasks.md) after push.
