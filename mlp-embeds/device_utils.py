"""Device selection helpers so the same code runs on CPU locally and GPU in prod.

Nothing here hardcodes a device: `resolve_device` picks CUDA when it is actually
available (production GPU box) and falls back to CPU otherwise (local dev here).
"""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Return the device to use.

    - requested None or "auto": use CUDA if available, else CPU.
    - requested "cuda"/"cpu"/etc.: honor it (lets production force a device).
    """
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int, device: torch.device) -> None:
    """Seed RNGs, including the CUDA generators when running on GPU."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move a batch of tensors to the device.

    `non_blocking=True` only helps with pinned CPU memory + CUDA, and is a no-op
    on CPU, so it is safe to always pass.
    """
    return {
        key: value.to(device, non_blocking=True) for key, value in batch.items()
    }
