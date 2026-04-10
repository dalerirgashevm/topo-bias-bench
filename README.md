# topo-bias-bench

> Benchmarking topological inductive bias in graph learning under controlled OOD conditions.

A compact research benchmark for testing whether topology-aware priors improve sample efficiency, rewiring robustness, and out-of-distribution generalization.

## Why this repo exists

Most claims about topological deep learning sound impressive but collapse into either feature engineering or message passing with a different name. This repo tests the strongest honest version of the idea:

- does a topological prior improve learning efficiency?
- does it survive degree-preserving rewiring?
- does it help only when the target signal is actually structural?

## Included tasks

- **spurious topology** — the graph looks topological, but the label is not
- **homology task** — the label depends on cycle structure
- **anti-topology** — long-range signal without a topological shortcut

## Included models

- **GCN** — graph convolution baseline
- **GAT** — attention baseline
- **GraphTransformer** — global attention baseline
- **TopoMPNN** — topology-aware message passing baseline

## What gets measured

- clean test accuracy
- rewired test accuracy
- robustness gap
- multi-seed mean / std
- spectral probes such as algebraic connectivity and Dirichlet energy

## Quick start

```bash
pip install -r requirements.txt
python -m src.run_benchmark
```

Results are written to:

```text
results/results.json
```

and can be plotted with:

```bash
python scripts/plot_results.py
```

## Repository layout

```text
src/
├── benchmark.py
├── config.py
├── data.py
├── eval.py
├── metrics.py
├── models.py
├── rewire.py
├── run_benchmark.py
├── spectral.py
├── topology.py
└── train.py
```

## Status

This is an intentionally minimal, reproducible benchmark. It is not a claim of new expressivity or AGI.

## Contributing

Open to small, high-signal improvements:
- stronger synthetic tasks
- better baselines
- stronger OOD tests
- cleaner plots
- more robust evaluation
