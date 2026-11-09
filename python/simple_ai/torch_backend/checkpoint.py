"""Atomic checkpoint save/load helpers for resumable PyTorch training."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_FORMAT = "simple-ai-torch-checkpoint"
CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_validation_loss: float | None,
    config: dict[str, Any],
) -> None:
    """Write a complete checkpoint atomically to avoid partial files."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "global_step": int(global_step),
        "best_validation_loss": best_validation_loss,
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
        },
    }

    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore model, optimizer, and optionally RNG state from a trusted file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    # Checkpoints contain Python/NumPy RNG state, so this must only be used
    # with checkpoint files produced by this project or another trusted source.
    payload = torch.load(
        source, map_location=map_location, weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint format.")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version.")

    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        if "optimizer_state" not in payload:
            raise ValueError("Checkpoint does not contain optimizer state.")
        optimizer.load_state_dict(payload["optimizer_state"])

    if restore_rng:
        rng = payload.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu())
        cuda_states = rng.get("cuda")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)

    return {
        "epoch": int(payload.get("epoch", 0)),
        "step_in_epoch": int(payload.get("step_in_epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "best_validation_loss": payload.get("best_validation_loss"),
        "config": payload.get("config", {}),
    }
