from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


@dataclass
class ScoringBatchResult:
    sample_ids: List[str]
    values: torch.Tensor


def stack_vectors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not vectors:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.stack([vector.to(torch.float32) for vector in vectors], dim=0)


def normalize_weights(weights: Iterable[float]) -> List[float]:
    values = [float(weight) for weight in weights]
    if not values:
        return []
    total = sum(values)
    if total == 0.0:
        raise ValueError("checkpoint_weights must not sum to zero")
    return [value / total for value in values]