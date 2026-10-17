"""Blocking dense CPU FP32 gradient synchronization through the native engine."""

from __future__ import annotations

import threading

import torch


def parameter_layout(model):
    layout, offset = [], 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError(f"{name}: expected CPU float32 parameter")
        if parameter.layout != torch.strided:
            raise ValueError(f"{name}: expected dense strided parameter")
        count = parameter.numel()
        layout.append(dict(name=name, shape=list(parameter.shape), dtype="float32", count=count, offset=offset))
        offset += count
    if not offset:
        raise ValueError("model has no nonempty trainable CPU float32 parameters")
    return layout


class GradientSynchronizer:
    def __init__(self, model, world_size):
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        self.model, self.world_size = model, world_size
        self.layout = parameter_layout(model)
        parameters = dict(model.named_parameters())
        self._parameters = tuple(parameters[item["name"]] for item in self.layout)
        self._layout_identity = parameter_layout(model)
        self.staging = torch.empty(sum(item["count"] for item in self.layout), dtype=torch.float32)
        self._view = self.staging.detach().numpy()
        self._lock = threading.Lock()

    def synchronize(self, communicator):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("gradient synchronization is already active")
        try:
            current = dict(self.model.named_parameters())
            if parameter_layout(self.model) != self._layout_identity or any(
                current.get(item["name"]) is not parameter
                for item, parameter in zip(self.layout, self._parameters)
            ):
                raise ValueError("parameter identity or layout changed")
            for item, parameter in zip(self.layout, self._parameters):
                gradient = parameter.grad
                if gradient is None:
                    raise ValueError(f"{item['name']}: missing gradient")
                if (gradient.device.type != "cpu" or gradient.dtype != torch.float32
                        or gradient.layout != torch.strided or gradient.shape != parameter.shape):
                    raise ValueError(f"{item['name']}: expected matching dense CPU float32 gradient")
                if not torch.isfinite(gradient).all().item():
                    raise ValueError(f"{item['name']}: nonfinite gradient")
            with torch.no_grad():
                for item, parameter in zip(self.layout, self._parameters):
                    self.staging.narrow(0, item["offset"], item["count"]).copy_(parameter.grad.detach().reshape(-1))
                communicator.all_reduce_sum(self._view)
                if not torch.isfinite(self.staging).all().item():
                    raise FloatingPointError("nonfinite reduced gradient")
                self.staging.div_(self.world_size)
                for item, parameter in zip(self.layout, self._parameters):
                    parameter.grad.copy_(self.staging.narrow(0, item["offset"], item["count"]).reshape(item["shape"]))
        finally:
            self._lock.release()
