from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main(results_path: str = "results/results.json", out_path: str = "docs/results_generated.md") -> None:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    raw = json.loads(path.read_text())
    summary: Dict[str, Dict[str, Dict[str, float]]] = raw.get("summary", raw)

    lines = []
    lines.append("# Generated results")
    lines.append("")
    lines.append("This file is generated from `results/results.json`.")
    lines.append("")

    for task, models in summary.items():
        lines.append(f"## {task}")
        lines.append("")
        lines.append("| Model | Clean acc mean | Clean acc std | Rewired acc mean | Rewired acc std | Robustness gap mean | Robustness gap std |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for model, metrics in models.items():
            lines.append(
                f"| {model} | {_fmt(metrics['test_acc_mean'])} | {_fmt(metrics['test_acc_std'])} | "
                f"{_fmt(metrics['rewired_acc_mean'])} | {_fmt(metrics['rewired_acc_std'])} | "
                f"{_fmt(metrics['robustness_gap_mean'])} | {_fmt(metrics['robustness_gap_std'])} |"
            )
        lines.append("")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
