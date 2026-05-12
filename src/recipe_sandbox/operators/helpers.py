from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from recipe_sandbox.schema.types import CanonicalSample


def sample_text_parts(sample: CanonicalSample) -> List[str]:
    parts = [message.content for message in sample.messages if message.content]
    if sample.target.text:
        parts.append(sample.target.text)
    return parts


def sample_to_text(sample: CanonicalSample) -> str:
    return "\n".join(part.strip() for part in sample_text_parts(sample) if part and part.strip())


def resolve_path(obj: Any, path: Optional[str], default: Any = None) -> Any:
    if not path:
        return default

    current = obj
    for segment in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(segment, default)
            continue
        if hasattr(current, segment):
            current = getattr(current, segment)
            continue
        return default
    return current


def set_path(obj: Any, path: str, value: Any) -> None:
    current = obj
    segments = path.split(".")

    for segment in segments[:-1]:
        if isinstance(current, dict):
            if segment not in current or current[segment] is None:
                current[segment] = {}
            current = current[segment]
            continue

        next_value = getattr(current, segment, None)
        if next_value is None:
            next_value = {}
            setattr(current, segment, next_value)
        current = next_value

    last_segment = segments[-1]
    if isinstance(current, dict):
        current[last_segment] = value
        return
    setattr(current, last_segment, value)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot_product requires vectors with the same length")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def generalized_jaccard(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("generalized_jaccard requires vectors with the same length")

    intersection = sum(min(float(a), float(b)) for a, b in zip(left, right))
    union = sum(max(float(a), float(b)) for a, b in zip(left, right))
    if union == 0.0:
        return 0.0
    return intersection / union


def coerce_vector(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, (list, tuple)):
            return [float(item) for item in converted]
    return None


def mean_vector(vectors: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not vectors:
        return None
    size = len(vectors[0])
    if size == 0:
        return []
    accumulator = [0.0] * size
    for vector in vectors:
        if len(vector) != size:
            raise ValueError("mean_vector requires vectors with the same length")
        for index, value in enumerate(vector):
            accumulator[index] += float(value)
    return [value / len(vectors) for value in accumulator]


def score_summary(scores: Iterable[float]) -> Dict[str, float]:
    values = [float(score) for score in scores]
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }