"""Configuration loading helpers shared by training and inference scripts."""

from pathlib import Path
from typing import Any, Dict

import yaml


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config, optionally inheriting one relative base config."""
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    base_path = config.pop("base_config", None)
    if base_path is None:
        return config
    base = load_config(str(path.parent / base_path))
    return _merge_config(base, config)


def get_region_attention_mode(config: Dict[str, Any]) -> str:
    """Resolve explicit modes and legacy enabled booleans consistently."""
    region = config.get("model", {}).get("region_attention", {})
    mode = region.get("mode")
    if mode is None:
        mode = "adaptive" if region.get("enabled", False) else "off"
    if mode not in {"off", "fixed", "adaptive"}:
        raise ValueError(
            f"Invalid model.region_attention.mode: {mode!r}; "
            "expected 'off', 'fixed', or 'adaptive'"
        )
    return mode


def set_region_attention_mode(config: Dict[str, Any], mode: str) -> None:
    """Set the canonical mode and keep the legacy enabled flag synchronized."""
    if mode not in {"off", "fixed", "adaptive"}:
        raise ValueError(
            f"Invalid ARWM mode: {mode!r}; expected 'off', 'fixed', or 'adaptive'"
        )
    region = config.setdefault("model", {}).setdefault("region_attention", {})
    region["mode"] = mode
    region["enabled"] = mode != "off"
