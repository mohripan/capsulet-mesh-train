"""Select training policy before importing the optional framework."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import tempfile

from meshtrain._cuda_device import cuda_device_name


def selection(device="cpu", host_memory=None):
    if device == "cpu":
        if host_memory is not None:
            raise ValueError("host-memory selection requires an explicit CUDA device")
        return device, None
    device = cuda_device_name(device)
    mode = "pinned" if host_memory is None else host_memory
    if mode not in ("pinned", "pageable"):
        raise ValueError("host_memory must be pinned or pageable")
    return device, mode


def from_config(config):
    if config["schema"] == 1:
        return "cpu", None
    if config["schema"] != 2:
        raise ValueError("unsupported configuration schema")
    execution = config["context"]["execution"]
    if execution["device"] == "cpu":
        raise ValueError("CUDA schema requires a CUDA device")
    return selection(execution["device"], execution["host_memory"])


def configure_environment(device):
    if device == "cpu":
        return
    cuda_device_name(device)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        raise ValueError("CUDA parity requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    for key in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        if os.environ.get(key, "0") != "0":
            raise ValueError(f"CUDA parity rejects conflicting {key}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def invalidate_verification(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".verification.", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(status="failed", full_acceptance=False, cuda_training_acceptance=False), stream)
        os.replace(temporary, root / "verification.json")
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def deferred_signals():
    received = []
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}

    def record(number, _):
        received.append(number)

    def check():
        if received:
            raise InterruptedError(f"training supervisor received signal {received[0]}")

    try:
        for number in previous:
            signal.signal(number, record)
        yield check
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
