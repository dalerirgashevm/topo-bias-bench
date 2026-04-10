from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import numpy as np


@dataclass
class GraphSample:
    x: np.ndarray  # [N, d]
    a: np.ndarray  # [N, N]
    y: int


def _adjacency_from_graph(g: nx.Graph, n: int | None = None) -> np.ndarray:
    a = nx.to_numpy_array(g, nodelist=sorted(g.nodes())).astype(np.float32)
    if n is not None and a.shape[0] != n:
        out = np.zeros((n, n), dtype=np.float32)
        m = min(n, a.shape[0])
        out[:m, :m] = a[:m, :m]
        return out
    return a


def _node_features(g: nx.Graph, d: int = 16) -> np.ndarray:
    n = g.number_of_nodes()
    deg = np.array([g.degree(i) for i in sorted(g.nodes())], dtype=np.float32)
    deg = deg / max(1.0, float(deg.max()))
    feats = [deg[:, None], np.ones((n, 1), dtype=np.float32)]
    while sum(f.shape[1] for f in feats) < d:
        feats.append(np.random.randn(n, 1).astype(np.float32))
    return np.concatenate(feats, axis=1)[:, :d]


def generate_spurious_topology(num_samples: int = 256, n: int = 24, d: int = 16) -> List[GraphSample]:
    """Label is SBM community assignment; extra cycles are spurious."""
    samples: List[GraphSample] = []
    for _ in range(num_samples):
        label = int(np.random.randint(0, 2))
        p_in = 0.25 if label == 1 else 0.12
        p_out = 0.03
        sizes = [n // 2, n - n // 2]
        probs = [[p_in, p_out], [p_out, p_in]]
        g = nx.stochastic_block_model(sizes, probs, seed=int(np.random.randint(1_000_000)))
        # inject extra cycles that should not be predictive
        if np.random.rand() < 0.5:
            for i in range(0, n - 2, 3):
                g.add_edge(i, i + 2)
        samples.append(GraphSample(_node_features(g, d), _adjacency_from_graph(g, n), label))
    return samples


def generate_homology_task(num_samples: int = 256, n: int = 24, d: int = 16) -> List[GraphSample]:
    """Label depends on cycle richness / Betti-1 proxy."""
    samples: List[GraphSample] = []
    for _ in range(num_samples):
        p = float(np.random.uniform(0.06, 0.16))
        g = nx.erdos_renyi_graph(n=n, p=p, seed=int(np.random.randint(1_000_000)))
        if g.number_of_edges() < n:
            for i in range(n - 1):
                g.add_edge(i, (i + 1) % n)
        beta1 = g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)
        label = int(beta1 >= n // 3)
        samples.append(GraphSample(_node_features(g, d), _adjacency_from_graph(g, n), label))
    return samples


def generate_anti_topology(num_samples: int = 256, n: int = 24, d: int = 16) -> List[GraphSample]:
    """Label depends on long-range node feature signal, not topology."""
    samples: List[GraphSample] = []
    for _ in range(num_samples):
        g = nx.random_tree(n, seed=int(np.random.randint(1_000_000)))
        x = _node_features(g, d)
        x[0, 0] = float(np.random.choice([-1.0, 1.0]))
        x[-1, 0] = float(np.random.choice([-1.0, 1.0]))
        label = int((x[0, 0] + x[-1, 0]) > 0)
        samples.append(GraphSample(x, _adjacency_from_graph(g, n), label))
    return samples


def train_test_split(samples: List[GraphSample], train_ratio: float = 0.8, seed: int = 0) -> Tuple[List[GraphSample], List[GraphSample]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    split = int(len(samples) * train_ratio)
    train = [samples[i] for i in idx[:split]]
    test = [samples[i] for i in idx[split:]]
    return train, test
