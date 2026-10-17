from __future__ import annotations

import numpy as np
import torch

from meshtrain import Communicator
from meshtrain.torch_adapter import GradientSynchronizer
from .artifacts import digest, environment, identity, write_json, write_npz, write_state
from .inputs import load_inputs, native_config
from .workload import evaluate, flat_gradients, flat_parameters, optimizer


def finish(directory, config, network, losses, gradients, parameters, before, after, rank, next_sequence):
    write_npz(directory / "trace.npz", losses=np.asarray(losses, dtype=np.float32),
              gradients=np.stack(gradients), parameters=np.stack(parameters))
    write_state(directory / "final.pt", network)
    metadata = dict(**identity(config), rank=rank, steps=config["workload"]["steps"],
                    next_sequence=next_sequence, evaluation_before=before, evaluation_after=after,
                    environment=environment(),
                    files={name: digest(directory / name) for name in ("trace.npz", "final.pt")})
    write_json(directory / "complete.json", metadata)
    return metadata


def train_worker(output, config_path, rank):
    root, config, network, features, labels, indices = load_inputs(output, config_path)
    native = native_config(config, rank)
    directory = root / "ranks" / str(rank)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("complete.json", "trace.npz", "final.pt")):
        raise ValueError(f"rank {rank} already has training artifacts; prepare a fresh run")
    before = evaluate(network, features, labels)
    optim = optimizer(network)
    synchronizer = GradientSynchronizer(network, config["workload"]["world_size"])
    losses, gradients, parameters = [], [], []
    with Communicator(native) as communicator:
        for step, global_indices in enumerate(indices):
            optim.zero_grad(set_to_none=True)
            local = torch.from_numpy(global_indices[rank])
            loss = torch.nn.functional.cross_entropy(network(features[local]), labels[local])
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite local loss")
            loss.backward()
            if communicator.next_sequence != 2 * step:
                raise RuntimeError("unexpected gradient collective sequence")
            synchronizer.synchronize(communicator)
            mean_loss = np.array([loss.detach().item()], dtype=np.float32)
            communicator.all_reduce_sum(mean_loss)
            if not np.isfinite(mean_loss).all():
                raise FloatingPointError("nonfinite reduced loss")
            mean_loss /= config["workload"]["world_size"]
            gradients.append(flat_gradients(network))
            losses.append(float(mean_loss[0]))
            optim.step()
            parameters.append(flat_parameters(network))
            print(f"step={step} loss={mean_loss[0]:.8f} next_sequence={communicator.next_sequence}", flush=True)
        next_sequence = communicator.next_sequence
    after = evaluate(network, features, labels)
    return finish(directory, config, network, losses, gradients, parameters, before, after, rank, next_sequence)
