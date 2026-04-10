from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import Config
from .data import generate_anti_topology, generate_homology_task, generate_spurious_topology, train_test_split
from .eval import accuracy
from .metrics import accuracy as metric_accuracy
from .models import GAT, GCN, GraphTransformer, TopoMPNN
from .rewire import rewired_sample


def _models():
    return {
        "gcn": GCN(),
        "gat": GAT(),
        "topo_mpn": TopoMPNN(),
        "transformer": GraphTransformer(),
    }


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_task(samples, cfg: Config, seed: int):
    train_data, test_data = train_test_split(samples, train_ratio=cfg.train_ratio, seed=seed)
    rewired_test = [rewired_sample(s, n_swaps=cfg.rewires, seed=seed + i) for i, s in enumerate(test_data)]

    results = {}
    for model_name, model in _models().items():
        from .train import train

        train(model, train_data, epochs=cfg.epochs, lr=cfg.lr)
        acc = metric_accuracy(model, test_data)
        rob = metric_accuracy(model, rewired_test)
        results[model_name] = {
            "test_acc": acc,
            "rewired_acc": rob,
            "robustness_gap": acc - rob,
        }
    return results


def run_benchmark(cfg: Config, seed: int | None = None):
    seed = cfg.seed if seed is None else seed
    _set_seed(seed)

    tasks = {
        "spurious_topology": generate_spurious_topology(cfg.num_samples, cfg.num_nodes, cfg.feature_dim),
        "homology_task": generate_homology_task(cfg.num_samples, cfg.num_nodes, cfg.feature_dim),
        "anti_topology": generate_anti_topology(cfg.num_samples, cfg.num_nodes, cfg.feature_dim),
    }

    results = {}
    for task_name, samples in tasks.items():
        results[task_name] = _run_task(samples, cfg, seed)

    return results


def run_multi_seed(cfg: Config, seeds: List[int]):
    per_seed = []
    for s in seeds:
        per_seed.append(run_benchmark(cfg, seed=s))

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task in per_seed[0].keys():
        summary[task] = {}
        for model in per_seed[0][task].keys():
            test_vals = [r[task][model]["test_acc"] for r in per_seed]
            rewired_vals = [r[task][model]["rewired_acc"] for r in per_seed]
            gap_vals = [r[task][model]["robustness_gap"] for r in per_seed]
            summary[task][model] = {
                "test_acc_mean": float(np.mean(test_vals)),
                "test_acc_std": float(np.std(test_vals)),
                "rewired_acc_mean": float(np.mean(rewired_vals)),
                "rewired_acc_std": float(np.std(rewired_vals)),
                "robustness_gap_mean": float(np.mean(gap_vals)),
                "robustness_gap_std": float(np.std(gap_vals)),
            }

    return {"per_seed": per_seed, "summary": summary}


def save_results(results, out_path: str = "results/results.json"):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
