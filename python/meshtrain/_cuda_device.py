"""Validate CUDA ordinals before Torch can narrow their representation."""

import re


def cuda_device_name(value):
    if not isinstance(value, str):
        raise TypeError("device must be an explicit indexed CUDA string")
    match = re.fullmatch(r"cuda:([0-9]+)", value)
    if match is None or int(match[1]) > 127:
        raise ValueError("expected an explicit representable CUDA device, such as cuda:0")
    return f"cuda:{int(match[1])}"
