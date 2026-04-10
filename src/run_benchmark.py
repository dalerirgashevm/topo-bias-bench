from __future__ import annotations

import json

from .benchmark import run_multi_seed, save_results
from .config import load_config


def main():
    cfg = load_config()
    seeds = [cfg.seed, cfg.seed + 1, cfg.seed + 2]
    results = run_multi_seed(cfg, seeds=seeds)
    save_results(results, out_path="results/results.json")
    print(json.dumps(results["summary"], indent=2))
    print("\nSAVED -> results/results.json")


if __name__ == "__main__":
    main()
