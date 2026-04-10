from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .data import GraphSample


@torch.no_grad()
def accuracy(model: nn.Module, test_data: Iterable[GraphSample]) -> float:
    test_data = list(test_data)
    model.eval()
    correct = 0
    for sample in test_data:
        x = torch.tensor(sample.x, dtype=torch.float32)
        a = torch.tensor(sample.a, dtype=torch.float32)
        logits = model(x, a)
        pred = int(torch.argmax(logits).item())
        correct += int(pred == sample.y)
    return correct / max(1, len(test_data))
