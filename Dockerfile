ARG NATIVE_BASE=debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587
ARG PYTHON_BASE=python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

FROM ${NATIVE_BASE} AS native-toolchain
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++=4:12.2.0-3 cmake=3.25.1-1 ninja-build=1.11.1-2~deb12u1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace

FROM native-toolchain AS native-test
COPY . .
RUN ! command -v python && ! command -v python3 \
    && cmake -S . -B build/native -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DMESHTRAIN_WARNINGS_AS_ERRORS=ON \
    && cmake --build build/native \
    && ctest --test-dir build/native --output-on-failure

FROM ${PYTHON_BASE} AS python-toolchain
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++=4:12.2.0-3 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements/toolchain.lock /tmp/toolchain.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/toolchain.lock
WORKDIR /workspace

FROM python-toolchain AS python-toolchain-test
COPY . .
RUN python tools/check_python_toolchain.py
