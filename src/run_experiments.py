from __future__ import annotations

import json

from .data import generate_anti_topology, generate_homology_task, generate_spurious_topology, train_test_split
from .eval import accuracy
from .models import GAT, GCN, GraphTransformer, TopoMPNN
from .train import train


def run_task(name: str, samples):
    train_data, test_data = train_test_split(samples, seed=42)
    models = {
        "gcn": GCN(),
        "gat": GAT(),
        "topo_mpn": TopoMPNN(),
        "transformer": GraphTransformer(),
    }
    results = {}
    for model_name, model in models.items():
        print(f"\n=== {name} :: {model_name} ===")
        train(model, train_data, epochs=20, lr=1e-3)
        acc = accuracy(model, test_data)
        print(f"test_acc={acc:.4f}")
        results[model_name] = acc
    return results


def main():
    all_results = {}
    all_results["spurious_topology"] = run_task("spurious_topology", generate_spurious_topology())
    all_results["homology_task"] = run_task("homology_task", generate_homology_task())
    all_results["anti_topology"] = run_task("anti_topology", generate_anti_topology())
    print("\nFINAL RESULTS:\n" + json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
