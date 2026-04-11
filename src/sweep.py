from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from .benchmark import run_multi_seed
from .config import Config


def sample_efficiency_sweep(
    cfg: Config,
    sizes: List[int] | None = None,
    seeds: List[int] | None = None,
) -> Dict[int, dict]:
    sizes = sizes or [32, 64, 128, 256, 512]
    seeds = seeds or list(range(5))

    results: Dict[int, dict] = {}
    for size in sizes:
        cfg_size = replace(cfg, num_samples=size)
        results[size] = run_multi_seed(cfg_size, seeds=seeds)
    return results


def rewiring_sweep(
    cfg: Config,
    rewires: List[int] | None = None,
    seeds: List[int] | None = None,
) -> Dict[int, dict]:
    rewires = rewires or [0, 2, 4, 8, 16]
    seeds = seeds or list(range(5))

    results: Dict[int, dict] = {}
    for r in rewires:
        cfg_r = replace(cfg, rewires=r)
        results[r] = run_multi_seed(cfg_r, seeds=seeds)
    return results
