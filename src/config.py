from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    seed: int = 0
    num_samples: int = 256
    num_nodes: int = 24
    feature_dim: int = 16
    train_ratio: float = 0.8
    epochs: int = 20
    lr: float = 1e-3
    rewires: int = 8


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_config(path: str = "configs/default.yaml") -> Config:
    values = Config().__dict__.copy()
    text = Path(path).read_text().splitlines()
    for line in text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key in values:
            values[key] = _parse_value(raw)
    return Config(**values)
