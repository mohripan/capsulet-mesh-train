from __future__ import annotations

import torch

from .inputs import load_inputs
from .worker import finish
from .workload import evaluate, flat_gradients, flat_parameters, optimizer


def train_reference(output):
    root, config, network, features, labels, indices = load_inputs(output)
    directory = root / "reference"
    (directory / "complete.json").unlink(missing_ok=True)
    optim = optimizer(network)
    before = evaluate(network, features, labels)
    losses, gradients, parameters = [], [], []
    for global_indices in indices:
        batch = torch.from_numpy(global_indices.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(network(features[batch]), labels[batch])
        loss.backward()
        losses.append(loss.detach().item())
        gradients.append(flat_gradients(network))
        optim.step()
        parameters.append(flat_parameters(network))
    after = evaluate(network, features, labels)
    return finish(directory, config, network, losses, gradients, parameters, before, after, "reference", None)
