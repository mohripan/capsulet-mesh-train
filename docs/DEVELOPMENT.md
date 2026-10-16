# Building the native engine

The standalone core provides configuration validation, typed contextual errors, checked FP32 payload sizes, framed group startup, and exact-byte TCP communication. Collectives, tensor bindings, and training are subsequent issues. These checks establish transport behavior; training parity remains an M1 completion gate.

## Native Linux build

Run in a Linux shell, including Ubuntu under WSL2. This checkout is available there at `/mnt/c/Projects/capsulet-mesh-train`.

```sh
cmake -S . -B build/native -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

Requirements: C++20 compiler, CMake 3.24 or newer, and Ninja or another CMake generator. TCP defaults on for Linux and uses POSIX threads; the current validated libc is glibc. The core does not discover/download Python, PyTorch, CUDA, MPI, or Kubernetes. Native checks use CTest, with real subprocesses and Linux socket/signal APIs for transport tests.

The five CTest entries are:

- `core_contracts`: eight groups of configuration, payload-overflow, and diagnostic checks.
- `wire_protocol`: golden network-order bytes, corrupt headers, count overflow, HELLO bounds/identifiers/representation, and empty data frames.
- `tcp_processes`: 29 subprocess scenarios covering 1–5 workers, fragmented 524,295-byte transfers with a 4,096-byte requested send buffer, late root, invalid/mismatched startup, duplicate ranks, incomplete membership, stalled reads/writes, slow-drip headers, EINTR, concurrent close/call rejection, and repeated construction. Includes 20 successful reconnects per rank with unchanged descriptor counts and 20 failed startup/rebind attempts.
- `resolver_deadline`: a test-only linker wrapper holds DNS past a 50 ms deadline, then verifies retained inputs and exactly one output release after late completion. Process/container tests use the real resolver.
- `installed_consumer`: installs and relocates the package, then builds/runs an independent consumer, including linkage to the TCP capability function and its native dependencies.

For the foundation without Linux networking, configure with `-DMESHTRAIN_ENABLE_TCP=OFF`; its three portable tests still run. See [TCP protocol and lifetime contracts](TCP_PROTOCOL.md).

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

Then run this in a Linux shell with Docker access (including WSL):

```sh
sh tools/check_tcp_containers.sh meshtrain-native-test:dev
```

It starts two independent containers on a private Docker network. The client resolves the root by hostname and exchanges 524,295 payload bytes in each direction. No loopback address or published host port connects the containers. The script removes its own containers/network on exit.

The native target asserts that neither `python` nor `python3` is installed before building/testing. The separate packaging target uses Python 3.12.15. Base-image digests, direct apt tool versions, and Python package hashes are pinned; apt transitive dependencies follow the Debian repositories.

The GitHub workflow builds both targets and runs the two-container exchange. The four-worker-container trainer demonstration remains in issue #8.

## Verified environments

Transport increment evidence, 2026-10-16:

| Environment | Tools | Result |
| --- | --- | --- |
| Ubuntu 26.04 / WSL2 | GCC 15.2.0, CMake 4.2.3, Ninja 1.13.2 | Debug, warnings as errors; 5/5 CTest entries passed. |
| Ubuntu 26.04 / WSL2 | Clang 21.1.8, CMake 4.2.3 | Release, warnings as errors; 5/5 passed, including a custom install include directory. |
| Ubuntu 26.04 / WSL2 | Clang 21.1.8, AddressSanitizer + UBSan | Debug native/wire/process/resolver tests: 4/4 passed; no sanitizer reports. Installed consumer checked separately above. |
| Ubuntu 26.04 / WSL2 | GCC 15.2.0, TCP disabled | Release, warnings as errors; 3/3 portable/package tests passed. |
| Pinned Debian container | GCC 12.2.0, CMake 3.25.1, Ninja 1.11.1 | Native-only Release build; 5/5 passed without Python installed; CTest total 5.68 s in the recorded run. |
| Two pinned Debian containers | Docker private network and DNS | 524,295 bytes each way; both workers exited 0; temporary resources cleaned up. |
| Pinned Python container | Python 3.12.15, hash-locked tools | Wheel build, isolated import, native result, and exception passed. |
| Ubuntu 26.04 / WSL2 (foundation increment) | Python 3.12.13, same locked tools | Wheel/import/native-result/exception probe passed before the transport increment. |

The pinned container is the reproducible packaging proof. This increment has not validated native Windows networking, distributed training, CUDA compute, or physical multi-host/device performance. Representative raw transport output is recorded in [the validation report](validation/2026-10-16-tcp.md).

To repeat the sanitizer check:

```sh
cmake -S . -B build/sanitized -G Ninja -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug -DMESHTRAIN_WARNINGS_AS_ERRORS=ON \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined'
cmake --build build/sanitized
ctest --test-dir build/sanitized -E installed_consumer --output-on-failure
```

The installed consumer is a separately configured project and does not inherit these ad hoc sanitizer flags. Its relocation/link checks run in the ordinary builds.

## Architecture conventions

- Keep standard-library value types/views and narrow interfaces in public headers. POSIX handles belong to the private Linux adapter.
- Configuration validation performs no DNS/network operations and does not mutate input. Reachability belongs to transport startup.
- Advertised peers use DNS hostnames/IPv4 and nonzero ports. Duplicate-name comparison ignores case and the final dot. Wildcard listening is separate from advertised endpoints; IPv6 is outside M1.
- Run/context identifiers are bounded printable ASCII. Errors retain structured context and escape control bytes in diagnostics.
- CPU buffer views borrow memory. The caller retains ownership and prevents concurrent mutation until completion. Validate size multiplication and limits before allocation or access.
- Pass one absolute monotonic deadline through a transport operation, including retries and partial I/O.
- Prefer RAII and explicit value ownership. Add abstractions at real ownership/platform/integration boundaries; keep framework dependencies and global state out of the core.
- Apply `.clang-format`; initial formatting used clang-format 22.1.7.

Next: [#5 — centralized in-place FP32 SUM all-reduce](https://github.com/mohripan/capsulet-mesh-train/issues/5).
