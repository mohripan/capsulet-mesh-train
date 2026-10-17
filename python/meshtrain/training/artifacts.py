from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, lambda stream: stream.write(canonical(value) + b"\n"))


def write_npz(path, **arrays):
    atomic(path, lambda stream: np.savez(stream, **arrays))


def write_state(path, model):
    atomic(path, lambda stream: torch.save({name: value.detach().cpu().clone()
                                          for name, value in model.state_dict().items()}, stream))


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__,
                intra_threads=torch.get_num_threads(), inter_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                torch_config=torch.__config__.show())


def identity(config):
    return {key: config[key] for key in ("schema", "run_id", "context_fingerprint")}
