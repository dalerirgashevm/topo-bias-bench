from __future__ import annotations

import networkx as nx
import torch


def graph_laplacian(a: torch.Tensor) -> torch.Tensor:
    a = a.float()
    d = torch.diag(a.sum(dim=1))
    return d - a


def betti_1_proxy(g: nx.Graph) -> int:
    return int(g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g))


def hasse_from_graph(g: nx.Graph) -> nx.DiGraph:
    h = nx.DiGraph()
    h.add_nodes_from(g.nodes())
    for u, v in g.edges():
        h.add_edge(u, v)
        h.add_edge(v, u)
    return h
