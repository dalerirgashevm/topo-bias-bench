# topo-bias-bench

Benchmarking topological inductive bias in graph learning under controlled OOD conditions.

## What this repo tests

- whether topological priors improve sample efficiency
- whether they improve robustness under rewiring/OOD shift
- whether they help only on tasks that are explicitly topological

## Included tasks

- **spurious topology**: topology looks useful, but the label is not topological
- **homology task**: the label depends on cycle structure
- **anti-topology**: long-range signal without topological signal

## Included models

- GCN baseline
- GAT baseline
- Graph Transformer baseline
- TopoMPNN (topology-aware message passing)

## Quick start

```bash
pip install -r requirements.txt
python -m src.run_experiments
```

## Research note

This is a controlled benchmark, not a claim of new expressivity or AGI.
