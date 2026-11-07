"""Checkpoint mode inference and compatibility checks."""

from pathlib import Path
from typing import Any, Dict


def infer_checkpoint_region_mode(checkpoint: Dict[str, Any]) -> str:
    """Read saved mode metadata or infer legacy mode from ARWM state keys."""
    experiment = checkpoint.get("experiment", {})
    mode = experiment.get("region_attention_mode")
    if mode is None:
        state = checkpoint.get("model", checkpoint)
        keys = set(state)
        prefix = "spatial_encoder.region_attention."
        has_region_projection = any(
            key.startswith(prefix + "region_proj.") for key in keys
        )
        has_adaptive_mlp = any(
            key.startswith(prefix + "region_mlp.") for key in keys
        )
        if not has_region_projection:
            return "off"
        mode = "adaptive" if has_adaptive_mlp else "fixed"
    if mode not in {"off", "fixed", "adaptive"}:
        raise ValueError(
            f"Checkpoint declares unsupported ARWM mode {mode!r}"
        )
    return mode


def validate_checkpoint_region_mode(
    checkpoint: Dict[str, Any],
    expected_mode: str,
    checkpoint_path: str,
) -> str:
    """Raise rather than silently evaluating a checkpoint in another mode."""
    actual_mode = infer_checkpoint_region_mode(checkpoint)
    if actual_mode != expected_mode:
        raise ValueError(
            f"Checkpoint/config ARWM mode mismatch for {Path(checkpoint_path)}: "
            f"checkpoint is {actual_mode!r}, requested model is {expected_mode!r}. "
            "Select a checkpoint trained for the requested mode."
        )
    return actual_mode
