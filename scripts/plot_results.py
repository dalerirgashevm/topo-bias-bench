from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(results_path: str = "results/results.json") -> None:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    results = json.loads(path.read_text())
    summary = results.get("summary", results)
    tasks = list(summary.keys())
    models = sorted(next(iter(summary.values())).keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(tasks))
    width = 0.2
    offsets = [(-1.5 + i) * width for i in range(len(models))]

    for i, model in enumerate(models):
        vals = [summary[t][model]["test_acc_mean"] if "test_acc_mean" in summary[t][model] else summary[t][model].get("test_acc", 0.0) for t in tasks]
        ax.bar([xi + offsets[i] for xi in x], vals, width=width, label=model)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Topological Bias Benchmark")
    ax.legend()
    plt.tight_layout()
    out = path.parent / "accuracy_by_task.png"
    plt.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
