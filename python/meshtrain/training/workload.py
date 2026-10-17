from __future__ import annotations

import numpy as np
import torch
from torch import nn

from meshtrain.torch_adapter import parameter_layout


def configure_execution():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def model():
    torch.manual_seed(1234)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(device="cpu", dtype=torch.float32)


def optimizer(network):
    return torch.optim.SGD(network.parameters(), lr=0.05, momentum=0, weight_decay=0,
                           foreach=False, fused=False)


def contract(world_size, steps, local_batch_size):
    for name, value in (("world_size", world_size), ("steps", steps), ("local_batch_size", local_batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if world_size * local_batch_size > 1024:
        raise ValueError("global batch exceeds dataset size")
    return dict(world_size=world_size, steps=steps, local_batch_size=local_batch_size,
                model="linear16-32-relu-linear32-4", dtype="float32", samples=1024,
                init_seed=1234, feature_seed=5678, teacher_seed=5679,
                loss="mean_cross_entropy", optimizer="SGD", lr=0.05, momentum=0,
                weight_decay=0, foreach=False, fused=False,
                gradient_normalization="native_SUM_divide_world_size")


def execution_contract():
    return dict(torch=torch.__version__, numpy=np.__version__, intra_threads=1, inter_threads=1,
                deterministic=True, device="cpu")


def schedule(settings):
    return np.arange(settings["steps"] * settings["world_size"] * settings["local_batch_size"],
                     dtype=np.int64).reshape(settings["steps"], settings["world_size"],
                                             settings["local_batch_size"]) % 1024


def flat_parameters(network):
    return torch.cat([parameter.detach().reshape(-1) for _, parameter in sorted(network.named_parameters())]).numpy().copy()


def flat_gradients(network):
    return torch.cat([parameter.grad.detach().reshape(-1) for _, parameter in sorted(network.named_parameters())]).numpy().copy()


def evaluate(network, features, labels):
    network.eval()
    with torch.no_grad():
        loss = nn.functional.cross_entropy(network(features), labels).item()
    network.train()
    if not np.isfinite(loss):
        raise FloatingPointError("nonfinite evaluation loss")
    return loss


def load_model(root):
    network = model()
    state = torch.load(root / "inputs/initial.pt", map_location="cpu", weights_only=True)
    network.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("nonfinite initial state")
    return network


def validate_layout(network, expected):
    if parameter_layout(network) != expected:
        raise ValueError("model layout differs from prepared layout")
