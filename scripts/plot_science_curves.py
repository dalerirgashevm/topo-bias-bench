from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _mean_for(results: dict, task: str = "homology_task", model: str = "topo_mpn") -> float:
    return float(results[task][model]["test_acc_mean"])


def plot_sample_efficiency(science: dict, out_path: str = "results/sample_efficiency.png") -> None:
    sweep = science["sample_efficiency"]
    sizes = sorted(int(k) for k in sweep.keys())
    models = list(next(iter(sweep.values()))["summary"]["homology_task"].keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models:
        vals = [sweep[str(size)]["summary"]["homology_task"][model]["test_acc_mean"] for size in sizes]
        ax.plot(sizes, vals, marker="o", label=model)

    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Sample Efficiency Sweep")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)


def plot_rewiring(science: dict, out_path: str = "results/rewiring_curve.png") -> None:
    sweep = science["rewiring"]
    rewires = sorted(int(k) for k in sweep.keys())
    models = list(next(iter(sweep.values()))["summary"]["homology_task"].keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models:
        vals = [sweep[str(r)]["summary"]["homology_task"][model]["rewired_acc_mean"] for r in rewires]
        ax.plot(rewires, vals, marker="o", label=model)

    ax.set_xlabel("Rewiring strength")
    ax.set_ylabel("Rewired accuracy")
    ax.set_title("Rewiring Robustness Curve")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)


def main(results_path: str = "results/science_results.json") -> None:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    science = json.loads(path.read_text())
    plot_sample_efficiency(science)
    plot_rewiring(science)
    print("saved results/sample_efficiency.png and results/rewiring_curve.png")


if __name__ == "__main__":
    main()
