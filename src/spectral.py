from __future__ import annotations

import torch


def laplacian(a: torch.Tensor) -> torch.Tensor:
    a = a.float()
    d = torch.diag(a.sum(dim=1))
    return d - a


def algebraic_connectivity(a: torch.Tensor) -> float:
    """Second-smallest eigenvalue of the graph Laplacian."""
    l = laplacian(a)
    evals = torch.linalg.eigvalsh(l)
    if evals.numel() < 2:
        return 0.0
    return float(torch.sort(evals).values[1].item())


def dirichlet_energy(x: torch.Tensor, a: torch.Tensor) -> float:
    """x^T L x averaged per node."""
    l = laplacian(a)
    e = torch.trace(x.T @ l @ x)
    return float((e / max(1, x.shape[0])).item())
