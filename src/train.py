from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .data import GraphSample


def train(model: nn.Module, train_data: Iterable[GraphSample], epochs: int = 20, lr: float = 1e-3) -> None:
    train_data = list(train_data)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total = 0.0
        for sample in train_data:
            x = torch.tensor(sample.x, dtype=torch.float32)
            a = torch.tensor(sample.a, dtype=torch.float32)
            y = torch.tensor([sample.y], dtype=torch.long)

            logits = model(x, a).unsqueeze(0)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        if epoch % 5 == 0:
            print(f"epoch={epoch:03d} loss={total / max(1, len(train_data)):.4f}")
