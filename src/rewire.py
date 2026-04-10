from __future__ import annotations

import networkx as nx
import numpy as np

from .data import GraphSample


def rewired_sample(sample: GraphSample, n_swaps: int = 8, seed: int = 0) -> GraphSample:
    rng = np.random.default_rng(seed)
    a = sample.a.copy()
    g = nx.from_numpy_array(a)

    try:
        if g.number_of_edges() >= 4:
            nx.double_edge_swap(g, nswap=min(n_swaps, g.number_of_edges()), max_tries=50 * n_swaps, seed=int(rng.integers(1_000_000)))
    except Exception:
        pass

    a_new = nx.to_numpy_array(g).astype(np.float32)
    return GraphSample(x=sample.x.copy(), a=a_new, y=sample.y)
