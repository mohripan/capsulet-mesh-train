"""Completed dense CUDA FP32 gradient averaging through native host SUM."""

from __future__ import annotations

import threading

import torch

from . import Communicator, EngineError
from ._cuda_device import cuda_device_name
from .cuda_adapter import CudaStagingAdapter


def _layout(model, device):
    layout, parameters, offset = [], [], 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        if (type(parameter) is not torch.nn.Parameter or parameter.device != device
                or parameter.dtype != torch.float32 or parameter.layout != torch.strided
                or torch.is_inference(parameter)):
            raise ValueError(f"{name}: expected ordinary dense CUDA float32 parameter on {device}")
        count = parameter.numel()
        layout.append(dict(name=name, shape=list(parameter.shape), dtype="float32", count=count, offset=offset))
        parameters.append(parameter)
        offset += count
    if not offset:
        raise ValueError("model has no nonempty trainable CUDA float32 parameters")
    return layout, tuple(parameters)


class CudaGradientSynchronizer:
    def __init__(self, model, world_size, *, device, pin_memory=True, max_staging_bytes=268435456):
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if isinstance(device, torch.device):
            device = str(device)
        self.device = torch.device(cuda_device_name(device))
        self.model, self.world_size = model, world_size
        self.layout, self._parameters = _layout(model, self.device)
        count = sum(item["count"] for item in self.layout)
        self._adapter = CudaStagingAdapter(count, device=self.device, pin_memory=pin_memory,
                                           max_staging_bytes=max_staging_bytes)
        with torch.cuda.device(self.device), torch.inference_mode(False):
            self._flat = torch.empty(count, dtype=torch.float32, device=self.device)
        self._lock = threading.Lock()
        self._failed_owners = None
        self._failure = None

    def _gradients(self, communicator):
        if self._failed_owners is not None:
            raise RuntimeError("CUDA gradient synchronizer is poisoned; terminate the failed worker")
        if type(communicator) is not Communicator:
            raise TypeError("expected an ordinary native meshtrain.Communicator")
        layout, parameters = _layout(self.model, self.device)
        if layout != self.layout or any(a is not b for a, b in zip(parameters, self._parameters)):
            raise ValueError("parameter identity or layout changed")
        gradients = tuple(parameter.grad for parameter in self._parameters)
        for item, gradient in zip(self.layout, gradients):
            if gradient is None:
                raise ValueError(f"{item['name']}: missing gradient")
            if (type(gradient) is not torch.Tensor or gradient.device != self.device
                    or gradient.dtype != torch.float32 or gradient.layout != torch.strided
                    or list(gradient.shape) != item["shape"]):
                raise ValueError(f"{item['name']}: expected matching ordinary dense CUDA float32 gradient")
            if gradient.requires_grad or gradient.is_conj() or gradient.is_neg() or torch.is_inference(gradient):
                raise ValueError(f"{item['name']}: autograd, inference and negative/conjugate gradients are unsupported")
            with torch.inference_mode(False):
                if torch.autograd.forward_ad.unpack_dual(gradient).tangent is not None:
                    raise ValueError(f"{item['name']}: forward-mode gradients are unsupported")
        return gradients

    def _abort(self, communicator, stream, gradients, error):
        self._failed_owners = (communicator, stream, gradients, self._parameters, self._flat, self._adapter)
        failure = dict(native_closed=False, stream_settled=False, cleanup_complete=False, secondary_errors=[])
        self._failure = failure
        try:
            communicator.close()
            failure["native_closed"] = True
        except BaseException as secondary:
            failure["secondary_errors"].append(f"close: {type(secondary).__name__}: {secondary}")
        if stream is not None:
            try:
                stream.synchronize()
                failure["stream_settled"] = True
            except BaseException as secondary:
                failure["secondary_errors"].append(f"stream: {type(secondary).__name__}: {secondary}")
        failure["cleanup_complete"] = failure["native_closed"] and failure["stream_settled"]
        error.add_note(f"CUDA gradient operation failed; cleanup={failure}")

    def synchronize(self, communicator):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("CUDA gradient synchronization is already active")
        try:
            gradients = self._gradients(communicator)
            stream = None
            try:
                with torch.cuda.device(self.device):
                    stream = torch.cuda.current_stream(self.device)
                    capturing = torch.cuda.is_current_stream_capturing()
            except BaseException as error:
                self._abort(communicator, stream, gradients, error)
                raise
            if capturing:
                raise ValueError("CUDA graph capture is unsupported")
            try:
                finite = all(torch.isfinite(gradient).all().item() for gradient in gradients)
            except BaseException as error:
                self._abort(communicator, stream, gradients, error)
                raise
            if not finite:
                raise ValueError("nonfinite local gradient")
            try:
                with torch.cuda.device(self.device), torch.cuda.stream(stream), torch.no_grad():
                    for item, gradient in zip(self.layout, gradients):
                        self._flat.narrow(0, item["offset"], item["count"]).copy_(gradient.detach().reshape(-1))
                    timing = self._adapter.all_reduce_sum_(self._flat, communicator)
                    if not torch.isfinite(self._flat).all().item():
                        raise FloatingPointError("nonfinite reduced gradient")
                    self._flat.div_(self.world_size)
                    if not torch.isfinite(self._flat).all().item():
                        raise FloatingPointError("nonfinite averaged gradient")
                    for item, gradient in zip(self.layout, gradients):
                        gradient.copy_(self._flat.narrow(0, item["offset"], item["count"]).reshape(item["shape"]))
                    stream.synchronize()
            except EngineError:
                raise
            except BaseException as error:
                self._abort(communicator, stream, gradients, error)
                raise
            return timing
        finally:
            self._lock.release()
