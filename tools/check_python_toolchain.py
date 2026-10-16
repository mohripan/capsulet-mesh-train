"""Compile a disposable Python 3.12 extension against the actual native core.

This verifies the packaging toolchain, C++ linkage, and Python ABI. It does not
install a meshtrain package or expose the future tensor/collective binding API.
Run from an environment populated with requirements/toolchain.lock.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import zipfile


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("The initial packaging target is Python 3.12.")
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="meshtrain-toolchain-") as temporary:
        project = Path(temporary)
        (project / "pyproject.toml").write_text(
            textwrap.dedent('''\
                [build-system]
                requires = ["scikit-build-core==1.1.1", "pybind11==3.1.0"]
                build-backend = "scikit_build_core.build"

                [project]
                name = "meshtrain-toolchain-probe"
                version = "0.0.0"
                requires-python = ">=3.12,<3.13"

                [tool.scikit-build]
                wheel.packages = []
                install.components = ["python"]
            '''), encoding="utf-8",
        )
        # Paths are data in a CMake bracket argument, not executable shell text.
        source_path = root.as_posix()
        if "]=]" in source_path:
            raise SystemExit("Unsupported source path for toolchain probe.")
        (project / "CMakeLists.txt").write_text(
            textwrap.dedent(f'''\
                cmake_minimum_required(VERSION 3.24)
                project(meshtrain_toolchain_probe LANGUAGES CXX)
                set(MESHTRAIN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
                add_subdirectory([=[{source_path}]=] meshtrain-core)
                find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
                find_package(pybind11 CONFIG REQUIRED)
                pybind11_add_module(_meshtrain_toolchain_probe MODULE probe.cpp)
                target_link_libraries(_meshtrain_toolchain_probe PRIVATE meshtrain::core)
                install(TARGETS _meshtrain_toolchain_probe LIBRARY DESTINATION . COMPONENT python)
            '''), encoding="utf-8",
        )
        (project / "probe.cpp").write_text(
            textwrap.dedent('''\
                #include <meshtrain/buffer.hpp>
                #include <pybind11/pybind11.h>
                #include <cstddef>
                PYBIND11_MODULE(_meshtrain_toolchain_probe, module) {
                    module.def("payload_bytes", [](std::size_t count, std::size_t limit) {
                        return meshtrain::checked_payload_bytes(count, limit);
                    });
                }
            '''), encoding="utf-8",
        )
        wheels = project / "wheels"
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheels), str(project)],
            check=True,
        )
        wheel_files = list(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise RuntimeError("Expected exactly one probe wheel.")
        extracted = project / "import-root"
        with zipfile.ZipFile(wheel_files[0]) as wheel:
            wheel.extractall(extracted)
        check = textwrap.dedent('''\
            import sys
            sys.path.insert(0, sys.argv[1])
            import _meshtrain_toolchain_probe as probe
            if probe.payload_bytes(4, 16) != 16:
                raise RuntimeError("Native result mismatch")
            try:
                probe.payload_bytes(5, 16)
            except RuntimeError as error:
                if "invalid_buffer" not in str(error):
                    raise
            else:
                raise RuntimeError("Native error did not cross the extension boundary")
        ''')
        subprocess.run([sys.executable, "-I", "-c", check, str(extracted)], check=True)
    print(f"Python ABI: {sys.version.split()[0]}")
    for package in ("build", "cmake", "ninja", "pybind11", "scikit-build-core"):
        print(f"{package}=={importlib.metadata.version(package)}")
    print("PASS: wheel build, fresh-process import, native result, and native exception")


if __name__ == "__main__":
    main()
