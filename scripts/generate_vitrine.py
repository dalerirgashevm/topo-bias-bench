from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    run([sys.executable, "-m", "src.run_benchmark"])
    run([sys.executable, "scripts/plot_results.py"])
    run([sys.executable, "scripts/render_results_md.py"])
    print("\nVitrine generated: results/results.json, results/accuracy_by_task.png, docs/results_generated.md")


if __name__ == "__main__":
    main()
