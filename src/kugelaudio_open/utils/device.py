"""Device selection utilities for KugelAudio."""

import torch
from typing import Union


def get_device() -> str:
    """Get the best available device string ("cuda", "mps", or "cpu")."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_obj() -> torch.device:
    """Get the best available torch.device object."""
    return torch.device(get_device())


def get_optimal_dtype(device: Union[str, torch.device]) -> torch.dtype:
    """Get the optimal data type for the given device.

    Args:
        device: Device string or object

    Returns:
        torch.bfloat16 for CUDA (Ampere+), torch.float16 for MPS, torch.float32 for CPU
    """
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)
        # Handle "cuda:0" etc.
        if ":" in device_type:
            device_type = device_type.split(":")[0]

    if device_type == "cuda":
        # Default to bfloat16 for CUDA if available (Ampere+)
        # If not, float16 is usually better but codebase used bfloat16.
        # Check if bfloat16 is supported?
        # For simplicity and matching existing code, we use bfloat16 for CUDA.
        return torch.bfloat16
    elif device_type == "mps":
        # MPS supports float16 well.
        return torch.float16

    return torch.float32


def empty_cache(device: Union[str, torch.device] = None):
    """Empty cache for the specified device if applicable."""
    if device is None:
        device = get_device()

    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device).split(":")[0]

    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
