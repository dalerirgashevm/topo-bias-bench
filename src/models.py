from __future__ import annotations

import torch
import torch.nn as nn

from .topology import graph_laplacian


class GCN(nn.Module):
    def __init__(self, d_in: int = 16, d_hidden: int = 64, d_out: int = 2):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = a @ x
        h = torch.relu(self.lin1(h))
        h = a @ h
        out = self.lin2(h)
        return out.mean(dim=0)


class GAT(nn.Module):
    def __init__(self, d_in: int = 16, d_hidden: int = 64, d_out: int = 2):
        super().__init__()
        self.q = nn.Linear(d_in, d_hidden)
        self.k = nn.Linear(d_in, d_hidden)
        self.v = nn.Linear(d_in, d_hidden)
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = (q @ k.T) / max(1.0, q.shape[-1] ** 0.5)
        scores = scores.masked_fill(a <= 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        h = attn @ v
        out = self.out(torch.relu(h))
        return out.mean(dim=0)


class GraphTransformer(nn.Module):
    def __init__(self, d_in: int = 16, d_hidden: int = 64, d_out: int = 2, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hidden)
        self.attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x).unsqueeze(0)
        h, _ = self.attn(h, h, h)
        out = self.ff(h.squeeze(0))
        return out.mean(dim=0)


class TopoMPNN(nn.Module):
    def __init__(self, d_in: int = 16, d_hidden: int = 64, d_out: int = 2):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_out)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        l = graph_laplacian(a)
        h = x - self.alpha * (l @ x)
        h = torch.relu(self.lin1(h))
        out = self.lin2(h)
        return out.mean(dim=0)
