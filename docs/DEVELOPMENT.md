# Building the core foundation

The standalone core provides configuration validation, typed contextual errors, checked FP32 payload sizes, and the byte-transport interface. TCP, collectives, tensor bindings, and training are subsequent issues. These checks do not establish distributed-training correctness.

## Native Linux build

Run in a Linux shell, including Ubuntu under WSL2. This checkout is available there at `/mnt/c/Projects/capsulet-mesh-train`.

```sh
cmake -S . -B build/native -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

Requirements: C++20 compiler, CMake 3.24 or newer, and Ninja or another CMake generator. The core does not discover/download Python, PyTorch, CUDA, MPI, or Kubernetes. Tests use the standard library and CTest.

`core_contracts` runs eight groups of membership, endpoint, identity, timeout/limit, payload-overflow, and diagnostic-context checks. `installed_consumer` installs into a temporary build subdirectory, relocates the installation, and builds/runs an independent consumer.

## Consuming the C++ library

```sh
cmake --install build/native --prefix "$PWD/build/install"
cmake -S tests/consumer -B build/consumer -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/consumer
ctest --test-dir build/consumer --output-on-failure
```

Use `find_package(meshtrain 0.1 CONFIG REQUIRED)` and link `meshtrain::core`. C++20 and include paths propagate through that target; our warning flags remain private. `add_subdirectory` is also supported, with native tests defaulting off for subproject builds.

The static library is position-independent so a later extension can link it. Interfaces are evolving during M1; a stable binary ABI is not yet promised.

## Optional Python 3.12 toolchain check

Issue #6 owns the tensor bindings. The current check builds a disposable pybind11 wheel against the actual core and verifies a native result and exception in a fresh isolated Python process. It does not install a project package or implement collectives.

```sh
uv python install 3.12
uv venv --python 3.12 build/toolchain-venv
uv pip sync --require-hashes --python build/toolchain-venv/bin/python requirements/toolchain.lock
build/toolchain-venv/bin/python tools/check_python_toolchain.py
```

The hash-locked tools are build 1.6.1, CMake 4.4.4, Ninja 1.13.2, pybind11 3.1.0, scikit-build-core 1.1.1, and their resolved dependencies. Refresh intentionally through `requirements/toolchain.in` and the generation command in the lock-file header, then repeat the checks.

## Container validation

From the project root in PowerShell or Linux:

```sh
docker build --target native-test -t meshtrain-native-test:dev .
docker build --target python-toolchain-test -t meshtrain-python-toolchain-test:dev .
```

The native target asserts that neither `python` nor `python3` is installed before building/testing. The separate packaging target uses Python 3.12.15. Base-image digests, direct apt tool versions, and Python package hashes are pinned; apt transitive dependencies follow the Debian repositories.

The GitHub workflow builds both targets. These are toolchain/contract checks, not the four-worker-container demonstration required to finish M1.

## Verified environments

Initial local evidence, 2026-10-16:

| Environment | Tools | Result |
| --- | --- | --- |
| Ubuntu 26.04 / WSL2 | GCC 15.2.0, CMake 4.2.3, Ninja 1.13.2 | Debug, warnings as errors; both CTest tests passed. |
| Ubuntu 26.04 / WSL2 | Clang 21.1.8, CMake 4.2.3 | Release, warnings as errors; both tests passed, including a custom install include directory. |
| Pinned Debian container | GCC 12.2.0, CMake 3.25.1, Ninja 1.11.1 | Native-only Release build; both tests passed without Python installed. |
| Pinned Python container | Python 3.12.15, hash-locked tools | Wheel build, isolated import, native result, and exception passed. |
| Ubuntu 26.04 / WSL2 | Python 3.12.13, same locked tools | The same wheel/import/native-result/exception probe passed. |

The pinned container is the reproducible packaging proof. This increment has not validated native Windows networking, distributed training, CUDA compute, or multi-host/device performance.

## Architecture conventions

- Keep standard-library value types/views and narrow interfaces in public headers. POSIX handles belong to the private Linux adapter added in #4.
- Configuration validation performs no DNS/network operations and does not mutate input. Reachability belongs to transport startup.
- Advertised peers use DNS hostnames/IPv4 and nonzero ports. Duplicate-name comparison ignores case and the final dot. Wildcard listening is separate from advertised endpoints; IPv6 is outside M1.
- Run/context identifiers are bounded printable ASCII. Errors retain structured context and escape control bytes in diagnostics.
- CPU buffer views borrow memory. The caller retains ownership and prevents concurrent mutation until completion. Validate size multiplication and limits before allocation or access.
- Pass one absolute monotonic deadline through a transport operation, including retries and partial I/O.
- Prefer RAII and explicit value ownership. Add abstractions at real ownership/platform/integration boundaries; keep framework dependencies and global state out of the core.
- Apply `.clang-format`; initial formatting used clang-format 22.1.7.

Next: [#4 — framed TCP transport and bounded I/O](https://github.com/mohripan/capsulet-mesh-train/issues/4).
