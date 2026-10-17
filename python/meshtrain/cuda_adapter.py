"""Blocking CUDA FP32 staging through the native host-buffer SUM engine."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
import threading
import time

import torch

from . import Communicator, EngineError


@dataclass(frozen=True, slots=True)
class CudaReductionTiming:
    count: int
    bytes: int
    device: str
    pin_memory: bool
    ready_wait_ms: float
    d2h_ms: float
    native_ms: float
    h2d_ms: float
    total_ms: float


def _milliseconds(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1_000_000


def _copy(destination: torch.Tensor, source: torch.Tensor, stream, pinned: bool) -> float:
    started = time.perf_counter_ns()
    destination.copy_(source, non_blocking=pinned)
    stream.synchronize()
    return _milliseconds(started)


class CudaStagingAdapter:
    def __init__(self, count: int, *, device: str | torch.device, pin_memory: bool = True,
                 max_staging_bytes: int = 268435456):
        if type(count) is not int or type(max_staging_bytes) is not int:
            raise TypeError("count and max_staging_bytes must be integers, excluding bool")
        if count < 0 or not 0 <= max_staging_bytes <= sys.maxsize or count > sys.maxsize // 4:
            raise ValueError("staging dimensions must be nonnegative and representable")
        if count * 4 > max_staging_bytes:
            raise ValueError("count exceeds max_staging_bytes")
        if type(pin_memory) is not bool:
            raise TypeError("pin_memory must be bool")
        if not isinstance(device, (str, torch.device)):
            raise TypeError("device must be an explicit indexed CUDA device")
        if isinstance(device, str):
            match = re.fullmatch(r"cuda:([0-9]+)", device)
            if match is None or int(match[1]) > 127:
                raise ValueError("device must contain a representable explicit CUDA index")
            selected = torch.device("cuda", int(match[1]))
        else:
            selected = device
        if selected.type != "cuda" or selected.index is None or not 0 <= selected.index <= 127:
            raise ValueError("device must be an explicit indexed CUDA device, such as cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install the supported CUDA runtime and GPU driver")
        if selected.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is out of range")
        with torch.cuda.device(selected), torch.inference_mode(False):
            staging = torch.empty(count, dtype=torch.float32, device="cpu",
                                  pin_memory=pin_memory and count > 0)
            view = staging.detach().numpy()
        if count and staging.is_pinned() != pin_memory:
            raise RuntimeError("host staging allocation did not honor requested pin_memory")
        self._count, self._device, self._pin_memory = count, selected, pin_memory
        self._staging, self._view = staging, view
        self._lock = threading.Lock()
        self._failed_owners = None
        self._failure = None

    def _validate(self, tensor, communicator):
        if self._failed_owners is not None:
            raise RuntimeError("CUDA adapter is poisoned; terminate the failed worker")
        if type(communicator) is not Communicator:
            raise TypeError("communicator must be an ordinary native meshtrain.Communicator")
        if type(tensor) is not torch.Tensor:
            raise TypeError("expected an ordinary torch.Tensor without subclasses")
        if (tensor.device != self._device or tensor.dtype != torch.float32
                or tensor.layout != torch.strided or tensor.ndim != 1
                or tensor.numel() != self._count or not tensor.is_contiguous()):
            raise ValueError("expected matching one-dimensional contiguous CUDA float32 storage")
        if tensor.requires_grad or tensor.is_conj() or tensor.is_neg() or torch.is_inference(tensor):
            raise ValueError("autograd, inference tensors and conjugate/negative views are unsupported")
        with torch.inference_mode(False):
            tangent = torch.autograd.forward_ad.unpack_dual(tensor).tangent
        if tangent is not None:
            raise ValueError("forward-mode AD dual tensors are unsupported")

    def _abort(self, tensor, communicator, stream, phase, error):
        self._failed_owners = (tensor, communicator, stream)
        failure = dict(phase=phase, native_closed=False, stream_settled=False,
                       cleanup_complete=False, secondary_errors=[])
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
        else:
            failure["secondary_errors"].append("captured stream unavailable; CUDA cleanup is unknown")
        failure["cleanup_complete"] = failure["native_closed"] and failure["stream_settled"]
        error.add_note(f"CUDA staging failed in {phase}; cleanup={failure}")

    def all_reduce_sum_(self, tensor: torch.Tensor, communicator: Communicator) -> CudaReductionTiming:
        started = time.perf_counter_ns()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("CUDA staging operation is already active")
        try:
            self._validate(tensor, communicator)
            stream = None
            try:
                with torch.cuda.device(self._device):
                    stream = torch.cuda.current_stream(self._device)
                    capturing = torch.cuda.is_current_stream_capturing()
            except BaseException as error:
                self._abort(tensor, communicator, stream, "stream", error)
                raise
            if capturing:
                raise ValueError("CUDA graph capture is unsupported")
            ready_ms = d2h_ms = h2d_ms = 0.0
            phase = "ready"
            try:
                with torch.cuda.device(self._device), torch.cuda.stream(stream), torch.no_grad():
                    if self._count:
                        phase_started = time.perf_counter_ns()
                        stream.synchronize()
                        ready_ms = _milliseconds(phase_started)
                        phase = "d2h"
                        d2h_ms = _copy(self._staging, tensor, stream, self._pin_memory)
                    phase = "native"
                    phase_started = time.perf_counter_ns()
                    communicator.all_reduce_sum(self._view)
                    native_ms = _milliseconds(phase_started)
                    if self._count:
                        phase = "h2d"
                        h2d_ms = _copy(tensor, self._staging, stream, self._pin_memory)
            except EngineError:
                raise
            except BaseException as error:
                self._abort(tensor, communicator, stream, phase, error)
                raise
            return CudaReductionTiming(self._count, self._count * 4, str(self._device),
                                       self._pin_memory, ready_ms, d2h_ms, native_ms,
                                       h2d_ms, _milliseconds(started))
        finally:
            self._lock.release()
