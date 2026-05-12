from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from recipe_sandbox.operators.helpers import resolve_path, sample_to_text
from recipe_sandbox.schema.types import CanonicalSample

# Optional: SAE sparse feature cache for fast MONA-based metrics
try:
    from recipe_sandbox.scoring.sparse_features import SparseFeatureCache
except ImportError:
    SparseFeatureCache = None  # type: ignore


@dataclass
class DataStateVector:
    """Compact state vector for one dataset snapshot (8 scalar dims + per-task scores).

    Dimensions are defined in ``feedback.state_registry.STATE_KEY_REGISTRY``.
    """

    retain_ratio: float
    token_ratio: float
    distribution_drift: float
    score_mean: float
    score_std: float
    mean_varentropy: float
    mean_ifd: float
    cumulative_cost_ratio: float = 0.0  # completed_evaluations / max_evaluations
    score_per_task: Dict[str, float] = field(default_factory=dict)  # per-benchmark MONA mean

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class DataStateComputer:
    """Compute lightweight dataset state vectors for recipe step diagnostics.

    All metrics use a single, deterministic computation path:
    - distribution_drift: SAE SNAR L2 distance (self-computed from reference_dataset)
    - score_mean/std: MONA Jaccard similarity (via metadata)
    - mean_varentropy: aggregated per-sample token-level varentropy
    - mean_ifd: aggregated per-sample Instruction Following Difficulty
    """

    def __init__(
        self,
        reference_dataset: Sequence[CanonicalSample],
        *,
        sparse_cache: Optional["SparseFeatureCache"] = None,
        d_sae: Optional[int] = None,
        score_path: Optional[str] = None,
        score_reducer: str = "max",
    ) -> None:
        self._reference_dataset = list(reference_dataset)
        self._reference_size = max(1, len(self._reference_dataset))
        self._reference_tokens = max(1, self._count_tokens(self._reference_dataset))
        self._sparse_cache = sparse_cache
        self._configured_d_sae = d_sae
        self._score_path = score_path
        self._score_reducer = score_reducer

        # Pre-compute reference means for ratio-based normalization
        self._ref_mean_varentropy = self._raw_mean_varentropy(self._reference_dataset)
        self._ref_mean_ifd = self._raw_mean_ifd(self._reference_dataset)

        # Pre-compute reference SNAR for distribution drift
        self._reference_snar, self._d_sae = self._init_reference_snar()

    def _init_reference_snar(self) -> tuple[Optional[Any], Optional[int]]:
        """Compute reference SNAR vector for drift calculation.

        Priority: reuse from sparse_cache if available, else use the configured
        d_sae from TaskConfig.model.d_sae, else infer a lower bound from the
        reference samples' sparse feature indices.
        """
        if self._sparse_cache is not None:
            return self._sparse_cache.reference_snar, self._sparse_cache.d_sae

        resolved_d_sae = self._configured_d_sae or self._infer_reference_d_sae()
        if not self._reference_dataset or resolved_d_sae is None:
            return None, resolved_d_sae

        from recipe_sandbox.scoring.sparse_features import compute_snar
        return compute_snar(self._reference_dataset, resolved_d_sae), resolved_d_sae

    def _infer_reference_d_sae(self) -> Optional[int]:
        from recipe_sandbox.scoring.sparse_features import get_sparse_features

        max_index = -1
        for sample in self._reference_dataset:
            topk = get_sparse_features(sample)
            if topk is None:
                continue
            indices = topk.get("indices", [])
            if not indices:
                continue
            sample_max = max(int(idx) for idx in indices)
            if sample_max > max_index:
                max_index = sample_max

        if max_index < 0:
            return None
        return max_index + 1

    def compute(self, dataset: Sequence[CanonicalSample]) -> DataStateVector:
        records = list(dataset)
        retain_ratio = self._safe_ratio(len(records), self._reference_size)
        token_ratio = self._safe_ratio(self._count_tokens(records), self._reference_tokens)
        distribution_drift = self._distribution_drift(records)
        score_mean, score_std = self._score_stats(records)
        score_per_task = self._score_stats_per_task(records)
        mean_varentropy = self._mean_varentropy(records)
        mean_ifd = self._mean_ifd(records)

        return DataStateVector(
            retain_ratio=self._round(retain_ratio),
            token_ratio=self._round(token_ratio),
            distribution_drift=self._round(distribution_drift),
            score_mean=self._round(score_mean),
            score_std=self._round(score_std),
            mean_varentropy=self._round(mean_varentropy),
            mean_ifd=self._round(mean_ifd),
            score_per_task={k: self._round(v) for k, v in score_per_task.items()},
        )

    # ------------------------------------------------------------------
    #  Basic counts
    # ------------------------------------------------------------------

    def _count_tokens(self, dataset: Iterable[CanonicalSample]) -> int:
        total = 0
        for sample in dataset:
            for message in sample.messages:
                total += len((message.content or "").split())
            total += len((sample.target.text or "").split())
        return total

    # ------------------------------------------------------------------
    #  Distribution drift (SAE SNAR)
    # ------------------------------------------------------------------

    def _distribution_drift(self, dataset: Sequence[CanonicalSample]) -> float:
        if not dataset or self._reference_snar is None or self._d_sae is None:
            return 0.0
        if self._matches_reference_dataset(dataset):
            return 0.0
        from recipe_sandbox.scoring.sparse_features import compute_snar, sparse_distribution_drift
        current_snar = compute_snar(dataset, self._d_sae)
        return sparse_distribution_drift(self._reference_snar, current_snar)

    def _matches_reference_dataset(self, dataset: Sequence[CanonicalSample]) -> bool:
        if len(dataset) != len(self._reference_dataset):
            return False
        return all(current is reference for current, reference in zip(dataset, self._reference_dataset))

    # ------------------------------------------------------------------
    #  MONA scores
    # ------------------------------------------------------------------

    def _score_stats(self, dataset: Sequence[CanonicalSample]) -> tuple[float, float]:
        if self._sparse_cache is not None:
            mean, std = self._sparse_cache.compute_scores(dataset)
            if mean > 0:
                return mean, std
        values: List[float] = []
        for sample in dataset:
            value = self._extract_score(sample)
            if value is not None:
                values.append(float(value))
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        if len(values) == 1:
            return mean, 0.0
        variance = sum((item - mean) ** 2 for item in values) / len(values)
        return mean, math.sqrt(max(variance, 0.0))

    def _extract_score(self, sample: CanonicalSample) -> Optional[float]:
        if self._score_path:
            found = resolve_path(sample, self._score_path)
            if isinstance(found, (int, float)):
                return float(found)

        mona = resolve_path(sample, "metadata.extra.mona_score")
        if isinstance(mona, (int, float)):
            return float(mona)

        mona_scores = resolve_path(sample, "metadata.extra.mona_scores")
        reduced = self._reduce_multi_scores(mona_scores)
        if reduced is not None:
            return reduced

        similarities = resolve_path(sample, "metadata.extra.similarities")
        reduced = self._reduce_multi_scores(similarities)
        if reduced is not None:
            return reduced

        fallback = resolve_path(sample, "metadata.extra.score")
        if isinstance(fallback, (int, float)):
            return float(fallback)
        return None

    def _score_stats_per_task(self, dataset: Sequence[CanonicalSample]) -> Dict[str, float]:
        """Compute per-task mean MONA similarity from benchmark-aware metadata."""
        per_task_values: Dict[str, List[float]] = {}
        for sample in dataset:
            sims = resolve_path(sample, "metadata.extra.mona_scores")
            if not isinstance(sims, dict):
                sims = resolve_path(sample, "metadata.extra.similarities")
            if not isinstance(sims, dict):
                continue
            for task_name, val in sims.items():
                if isinstance(val, (int, float)):
                    per_task_values.setdefault(task_name, []).append(float(val))
        result: Dict[str, float] = {}
        for task_name, values in per_task_values.items():
            if values:
                result[task_name] = sum(values) / len(values)
        return result

    def _reduce_multi_scores(self, value: Any) -> Optional[float]:
        if not isinstance(value, dict):
            return None
        values = [float(item) for item in value.values() if isinstance(item, (int, float))]
        if not values:
            return None
        if self._score_reducer == "mean":
            return sum(values) / len(values)
        if self._score_reducer == "min":
            return min(values)
        return max(values)

    # ------------------------------------------------------------------
    #  Mean varentropy (token-level LLM logits entropy variance)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_varentropy(sample: CanonicalSample) -> Optional[float]:
        ve = resolve_path(sample, "metadata.extra.varentropy")
        if isinstance(ve, dict):
            score = ve.get("score")
        else:
            score = ve
        if isinstance(score, (int, float)):
            return float(score)
        return None

    def _raw_mean_varentropy(self, dataset: Sequence[CanonicalSample]) -> float:
        values = [v for v in (self._extract_varentropy(s) for s in dataset) if v is not None]
        return sum(values) / len(values) if values else 0.0

    def _mean_varentropy(self, dataset: Sequence[CanonicalSample]) -> float:
        """Normalized varentropy: filtered_mean / reference_mean, clipped to [0,1]."""
        if not dataset:
            return 0.5
        raw = self._raw_mean_varentropy(dataset)
        if self._ref_mean_varentropy <= 0:
            return 0.5
        ratio = raw / self._ref_mean_varentropy
        return min(max(ratio, 0.0), 2.0) / 2.0

    # ------------------------------------------------------------------
    #  Mean IFD (Instruction Following Difficulty)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ifd(sample: CanonicalSample) -> Optional[float]:
        ifd = resolve_path(sample, "metadata.extra.ifd")
        if isinstance(ifd, dict):
            score = ifd.get("score")
        else:
            score = ifd
        if isinstance(score, (int, float)):
            return float(score)
        return None

    def _raw_mean_ifd(self, dataset: Sequence[CanonicalSample]) -> float:
        values = [v for v in (self._extract_ifd(s) for s in dataset) if v is not None]
        return sum(values) / len(values) if values else 0.0

    def _mean_ifd(self, dataset: Sequence[CanonicalSample]) -> float:
        """Normalized IFD: filtered_mean / reference_mean, clipped to [0,1]."""
        if not dataset:
            return 0.5
        raw = self._raw_mean_ifd(dataset)
        if self._ref_mean_ifd <= 0:
            return 0.5
        ratio = raw / self._ref_mean_ifd
        return min(max(ratio, 0.0), 2.0) / 2.0

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _round(self, value: float) -> float:
        return round(float(value), 6)


def vector_delta(current: DataStateVector, baseline: DataStateVector) -> Dict[str, float]:
    """Compute named deltas between two state vectors."""

    left = current.to_dict()
    right = baseline.to_dict()
    result: Dict[str, float] = {}
    for key in left:
        lv, rv = left[key], right[key]
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            result[key] = round(lv - rv, 6)
    return result
