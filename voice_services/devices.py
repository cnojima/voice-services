"""Accelerator detection. CUDA-first: cuda when a CUDA build of torch is installed,
otherwise cpu. An explicit device always wins over "auto"."""
from __future__ import annotations

import sys

import torch

AUTO = "auto"


def is_windows() -> bool:
    return sys.platform == "win32"


def has_cuda() -> bool:
    return torch.cuda.is_available()


def resolve_whisper_device(requested: str) -> str:
    if requested != AUTO:
        return requested
    return "cuda" if has_cuda() else "cpu"


def describe() -> str:
    """One-line summary of what this machine can run on, for startup logs."""
    parts = [sys.platform, f"torch {torch.__version__}"]
    parts.append(f"cuda ({torch.cuda.get_device_name(0)})" if has_cuda() else "cpu only")
    return ", ".join(parts)
