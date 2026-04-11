from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.sweep import rewiring_sweep, sample_efficiency_sweep


def main() -> None:
    cfg = load_config()
    science = {
        "sample_efficiency": sample_efficiency_sweep(cfg),
        "rewiring": rewiring_sweep(cfg),
    }

    out = Path("results/science_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(science, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
