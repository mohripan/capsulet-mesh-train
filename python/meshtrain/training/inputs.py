from __future__ import annotations

from pathlib import Path
import socket
import uuid

import numpy as np
import torch

from meshtrain import Endpoint, RunConfig
from meshtrain.torch_adapter import parameter_layout
from .artifacts import digest, fingerprint, read_json, write_json, write_npz, write_state
from .workload import contract, execution_contract, load_model, model, schedule, validate_layout


def local_endpoints(count):
    reservations = []
    try:
        for _ in range(count):
            sock = socket.socket()
            reservations.append(sock)
            sock.bind(("127.0.0.1", 0))
        return [dict(host="127.0.0.1", port=sock.getsockname()[1]) for sock in reservations]
    finally:
        for sock in reservations:
            sock.close()


def native_config(config, rank):
    if type(rank) is not int or not 0 <= rank < config["workload"]["world_size"]:
        raise ValueError("rank is outside the configured worker group")
    return RunConfig(rank=rank, world_size=config["workload"]["world_size"],
                     run_id=config["run_id"], context_fingerprint=config["context_fingerprint"],
                     endpoints=[Endpoint(item["host"], item["port"]) for item in config["endpoints"]],
                     listen_address=config["listen_address"], startup_timeout_ms=config["startup_timeout_ms"],
                     collective_timeout_ms=config["collective_timeout_ms"])


def prepare(output, world_size=4, steps=100, local_batch_size=8, endpoints=None, run_id=None,
            listen_address="0.0.0.0", startup_timeout_ms=30000, collective_timeout_ms=30000):
    settings = contract(world_size, steps, local_batch_size)
    root = Path(output).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"output must be a fresh empty directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    network = model()
    write_state(root / "inputs/initial.pt", network)
    features = torch.randn(1024, 16, generator=torch.Generator(device="cpu").manual_seed(5678))
    teacher = torch.randn(16, 4, generator=torch.Generator(device="cpu").manual_seed(5679))
    labels = (features @ teacher).argmax(dim=1)
    write_npz(root / "inputs/data.npz", features=features.numpy(), teacher=teacher.numpy(), labels=labels.numpy())
    write_npz(root / "inputs/schedule.npz", indices=schedule(settings))
    inputs = {name: digest(root / name) for name in ("inputs/initial.pt", "inputs/data.npz", "inputs/schedule.npz")}
    context = dict(workload=settings, layout=parameter_layout(network), inputs=inputs, execution=execution_contract())
    config = dict(schema=1, run_id=run_id or uuid.uuid4().hex, context=context,
                  context_fingerprint=fingerprint(context), workload=settings,
                  endpoints=endpoints if endpoints is not None else local_endpoints(world_size),
                  listen_address=listen_address, startup_timeout_ms=startup_timeout_ms,
                  collective_timeout_ms=collective_timeout_ms)
    validate_config(config)
    write_json(root / "config.json", config)
    return config


def validate_config(config):
    if config["schema"] != 1:
        raise ValueError("unsupported configuration schema")
    settings = config["workload"]
    expected = contract(settings["world_size"], settings["steps"], settings["local_batch_size"])
    context = config["context"]
    if settings != expected or context["workload"] != expected:
        raise ValueError("unsupported workload contract")
    if context["execution"] != execution_contract():
        raise ValueError("execution environment differs from prepared contract")
    if fingerprint(context) != config["context_fingerprint"]:
        raise ValueError("context fingerprint mismatch")
    if len(config["endpoints"]) != settings["world_size"]:
        raise ValueError("endpoint count differs from world size")
    if len({(item["host"], item["port"]) for item in config["endpoints"]}) != settings["world_size"]:
        raise ValueError("duplicate endpoints")
    native_config(config, 0)


def load_inputs(output, config_path=None):
    root = Path(output).resolve()
    config = read_json(config_path or root / "config.json")
    if config != read_json(root / "config.json"):
        raise ValueError("worker configuration differs from run root")
    validate_config(config)
    expected_files = {"inputs/initial.pt", "inputs/data.npz", "inputs/schedule.npz"}
    if set(config["context"]["inputs"]) != expected_files:
        raise ValueError("unexpected input manifest")
    for name, expected in config["context"]["inputs"].items():
        if digest(root / name) != expected:
            raise ValueError(f"input digest mismatch: {name}")
    network = load_model(root)
    validate_layout(network, config["context"]["layout"])
    with np.load(root / "inputs/data.npz", allow_pickle=False) as data:
        features, labels, teacher = (data[key].copy() for key in ("features", "labels", "teacher"))
    with np.load(root / "inputs/schedule.npz", allow_pickle=False) as data:
        indices = data["indices"].copy()
    if (features.shape != (1024, 16) or features.dtype != np.float32 or labels.shape != (1024,)
            or labels.dtype != np.int64 or teacher.shape != (16, 4) or teacher.dtype != np.float32
            or not np.isfinite(features).all() or not np.isfinite(teacher).all()
            or not np.array_equal(labels, (torch.from_numpy(features) @ torch.from_numpy(teacher)).argmax(1).numpy())):
        raise ValueError("invalid prepared dataset")
    if indices.dtype != np.int64 or not np.array_equal(indices, schedule(config["workload"])):
        raise ValueError("invalid prepared batch schedule")
    return root, config, network, torch.from_numpy(features), torch.from_numpy(labels), indices
