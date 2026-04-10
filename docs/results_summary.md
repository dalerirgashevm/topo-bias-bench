# Demo benchmark results

These numbers come from a small sanity-check run of the benchmark pipeline.

## Run settings

- seed: `0`
- num_samples: `16`
- num_nodes: `24`
- feature_dim: `16`
- epochs: `1`
- rewires: `2`
- train_ratio: `0.8`

## Results

| Task | Model | Clean acc | Rewired acc | Gap |
|---|---:|---:|---:|---:|
| spurious_topology | GCN | 0.50 | 0.50 | 0.00 |
| spurious_topology | GAT | 0.50 | 0.50 | 0.00 |
| spurious_topology | TopoMPNN | 0.50 | 0.50 | 0.00 |
| spurious_topology | GraphTransformer | 0.50 | 0.50 | 0.00 |
| homology_task | GCN | 1.00 | 1.00 | 0.00 |
| homology_task | GAT | 0.00 | 0.00 | 0.00 |
| homology_task | TopoMPNN | 1.00 | 1.00 | 0.00 |
| homology_task | GraphTransformer | 1.00 | 1.00 | 0.00 |
| anti_topology | GCN | 0.75 | 0.75 | 0.00 |
| anti_topology | GAT | 0.75 | 0.75 | 0.00 |
| anti_topology | TopoMPNN | 0.75 | 0.75 | 0.00 |
| anti_topology | GraphTransformer | 0.75 | 0.75 | 0.00 |

## Interpretation

This is only a sanity check, not a statistical conclusion. The next meaningful step is the full multi-seed run with the default config after the small bug fixes are merged into the source tree.
