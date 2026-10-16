# Standalone core foundation: issue #3

The user authorized implementation of the CPU MVP on 2026-10-16 and requested clear C++ architecture. This increment completes [issue #3](https://github.com/mohripan/capsulet-mesh-train/issues/3); TCP behavior remains issue #4.

## Execution

1. Add a C++20 static library with target-scoped requirements, optional tests, and relocatable CMake package exports.
2. Implement configuration validation, contextual typed errors, and overflow-safe FP32 payload validation. Define a minimal synchronous byte-transport interface without OS handles or framework dependencies.
3. Test invalid groups, endpoint configuration, identifiers, deadlines, payload overflow, and diagnostic context. Verify a separate consumer against a relocated installation.
4. Pin the Python 3.12 packaging toolchain, and verify it by compiling a disposable pybind11 extension against the actual core. This is a toolchain check, not the future tensor-binding API.
5. Build with GCC and Clang in WSL, and build/test inside a native-only Linux container without Python. Document verified versions, commands, and boundaries.
6. Synchronize specification/agent status and GitHub acceptance evidence, then commit and push without a co-author trailer.

The writing-plans helper referenced by the brainstorming workflow is not installed in the available skill directories. This execution checklist is written directly from the approved specification and issue dependencies.
