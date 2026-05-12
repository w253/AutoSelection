from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import logging

from recipe_sandbox.operators.base import FilterOperator
from recipe_sandbox.operators.helpers import (
    resolve_path,
    score_summary,
    set_path,
)
from recipe_sandbox.schema.types import CanonicalSample

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from recipe_sandbox.pipeline.task_config import TaskConfig
    from recipe_sandbox.pipeline.task_manager import TaskManager


# ======================================================================
# Shared base class
# ======================================================================


class ScoreFilterBase(FilterOperator):
    """Base class for score-based filter operators.

    Subclasses must implement ``_compute_scores`` which returns a mapping
    of ``sample_id -> score`` for the given dataset.  The base class
    handles sorting and selection (top_k / threshold / top_fraction), and
    tracing.

    Config keys consumed here:
      - ``top_k``           -- number of samples to keep
      - ``threshold``       -- score threshold
      - ``fraction``        -- fraction for top_fraction mode
      - ``descending``      -- sort order (default ``True``)
      - ``keep_missing_scores`` -- keep samples without scores (default ``False``)
      - ``score_path``      -- metadata path to read/write scores
    """

    operator_type = "filter"

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self._score_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Template method -- subclasses override this
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        dataset: Sequence[CanonicalSample],
        task_context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Return ``{sample_id: score}`` for all scoreable samples.

        The base implementation reads pre-existing scores from the
        metadata path returned by ``_score_read_path()``.  Subclasses
        that delegate to a scoring runner should override this.
        """
        scores: Dict[str, float] = {}
        read_path = self._score_read_path()
        if not read_path:
            return scores
        for sample in dataset:
            value = self._extract_score_value(sample, read_path)
            if value is not None:
                scores[sample.sample_id] = float(value)
        return scores

    def _extract_score_value(self, sample: CanonicalSample, read_path: str) -> Optional[float]:
        value = resolve_path(sample, read_path)
        if value is None:
            return None
        return float(value)

    # ------------------------------------------------------------------
    # Score path helpers
    # ------------------------------------------------------------------

    def _score_read_path(self) -> Optional[str]:
        """Metadata path from which to read scores.

        Subclasses should override if the path depends on the method
        (e.g. MONA uses ``metadata.extra.mona_scores.<task_name>``).
        """
        return self.config.get("score_path")

    def _score_write_path(self) -> Optional[str]:
        """Metadata path to write back scores.  Defaults to read path."""
        return self.config.get("score_path") or self._score_read_path()

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, dataset, task_context=None):
        task_context = task_context or {}
        fit_start = time.perf_counter()
        super().fit(dataset=dataset, task_context=task_context)
        self._score_cache = {}

        scores = self._compute_scores(dataset, task_context)
        self._score_cache.update(scores)

        # Optionally write scores back into sample metadata
        write_path = self._score_write_path()
        if write_path and self.config.get("write_back_scores", True):
            for sample in dataset:
                if sample.sample_id in scores:
                    set_path(sample, write_path, scores[sample.sample_id])

        self._trace.notes["score_summary"] = score_summary(scores.values())
        self._trace.notes["selection_mode"] = self._infer_selection_mode()
        self._trace.notes["score_kind"] = self.score_kind
        self._trace.notes["computed_scores"] = len(scores)
        self._trace.cost.extra["fit_wall_clock_sec"] = time.perf_counter() - fit_start
        return self

    def transform(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        descending = bool(self.config.get("descending", True))
        keep_missing_scores = bool(self.config.get("keep_missing_scores", False))

        scored: List[Tuple[CanonicalSample, float]] = []
        missing: List[CanonicalSample] = []
        for sample in dataset:
            value = self._score_cache.get(sample.sample_id)
            if value is None:
                missing.append(sample)
                continue
            scored.append((sample, value))

        scored.sort(key=lambda item: item[1], reverse=descending)
        selection_mode = self._infer_selection_mode()
        selected = self._select(scored, selection_mode)

        # A filter must never produce 0 samples from non-empty input
        if not selected and scored:
            logger.warning(
                "%s: selection produced 0 samples — keeping top 1 from %d scored as fallback.",
                self.name, len(scored),
            )
            selected = scored[:1]

        outputs = [sample for sample, _ in selected]
        if keep_missing_scores:
            outputs.extend(missing)

        self._trace.notes["selected_scores"] = [s for _, s in selected[:20]]
        self._trace.cost.extra["scored_samples"] = len(scored)
        self._trace.cost.extra["missing_scores"] = len(missing)
        return outputs

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def _select(
        self,
        scored: List[Tuple[CanonicalSample, float]],
        selection_mode: str,
    ) -> List[Tuple[CanonicalSample, float]]:
        if selection_mode == "threshold":
            threshold = float(self.config.get("threshold", 0.0))
            descending = bool(self.config.get("descending", True))
            if descending:
                return [item for item in scored if item[1] >= threshold]
            return [item for item in scored if item[1] <= threshold]

        if selection_mode == "top_fraction":
            fraction = float(self.config.get("fraction", 1.0))
            keep = math.ceil(len(scored) * fraction)
            return scored[:keep]

        keep = int(self.config.get("top_k", len(scored)))
        return scored[:keep]

    def _infer_selection_mode(self) -> str:
        if self.config.get("threshold") is not None:
            return "threshold"
        if self.config.get("top_k") is not None:
            return "top_k"
        if self.config.get("fraction") is not None:
            return "top_fraction"
        return "top_k"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def score_kind(self) -> str:
        return getattr(self, "_score_kind", "path")


# ======================================================================
# Runner bridge mixin
# ======================================================================


class _RunnerBridgeMixin:
    """Shared infrastructure for operators that delegate to a ScoringRunner.

    Subclasses set ``_runner_method`` and can override
    ``_build_scoring_config`` / ``_build_model_config`` for
    method-specific overrides.
    """

    _runner_method: str = ""

    def _run_scoring_bridge(
        self,
        dataset: Sequence[CanonicalSample],
        task_context: Dict[str, Any],
    ) -> Dict[str, float]:
        from recipe_sandbox.pipeline.task_config import ModelConfig, RecipeConfig, ScoringConfig, TaskConfig
        from recipe_sandbox.pipeline.task_manager import TaskManager
        from recipe_sandbox.schema.io import read_jsonl
        from recipe_sandbox.scoring.runner import create_scoring_runner

        base_task_config: TaskConfig = task_context.get("base_task_config")  # type: ignore[assignment]
        base_task_manager: TaskManager = task_context.get("base_task_manager")  # type: ignore[assignment]
        if not isinstance(base_task_config, TaskConfig):
            raise ValueError("Scoring requires base_task_config in task_context")
        if not isinstance(base_task_manager, TaskManager):
            raise ValueError("Scoring requires base_task_manager in task_context")

        input_split = str(task_context.get("input_split", "train"))
        recipe_name = str(task_context.get("recipe_name", "recipe"))
        step_index = int(task_context.get("step_index", 0))
        method = self._runner_method

        temp_config = copy.deepcopy(base_task_config)
        temp_config.task_name = f"{base_task_config.task_name}__recipe_step_{step_index:03d}"
        temp_config.output_dir = str(
            base_task_manager.temp_dir / "recipe_steps" / recipe_name / f"step_{step_index:03d}_{method}"
        )
        temp_config.recipe = RecipeConfig(enabled=False)
        temp_config.train_sources = []
        temp_config.eval_sources = []
        temp_config.scoring = self._build_scoring_config(base_task_config)
        temp_config.model = self._build_model_config(base_task_config)
        if input_split == "eval":
            temp_config.scoring.score_eval = True

        temp_manager = TaskManager(temp_config)
        temp_source_name = f"recipe_step_{step_index:03d}_{method}"
        temp_manager.write_canonical(input_split, temp_source_name, list(dataset))

        # Stage eval inputs for building target vectors
        if not temp_config.model.target_vectors_path:
            for path in base_task_manager.list_canonical("eval"):
                samples = list(read_jsonl(str(path)))
                temp_manager.write_canonical("eval", path.stem, samples)

        runner = create_scoring_runner(temp_config, temp_manager)
        runner.run()

        # Extract scores from scored output
        score_path = self._score_write_path()
        scores: Dict[str, float] = {}
        for path in temp_manager.list_scored(input_split):
            for sample in read_jsonl(str(path)):
                score = self._extract_score_value(sample, score_path)
                if score is not None:
                    scores[sample.sample_id] = score

        self._trace.notes["runner_bridge_method"] = method  # type: ignore[attr-defined]
        self._trace.notes["runner_bridge_output_dir"] = temp_config.output_dir  # type: ignore[attr-defined]
        return scores

    def _build_scoring_config(self, base_task_config: "TaskConfig") -> "ScoringConfig":
        from recipe_sandbox.pipeline.task_config import ScoringConfig

        payload = dict(asdict(base_task_config.scoring))
        payload["method"] = self._runner_method
        return ScoringConfig(**payload)

    def _build_model_config(self, base_task_config: "TaskConfig") -> "ModelConfig":
        from recipe_sandbox.pipeline.task_config import ModelConfig

        payload = dict(asdict(base_task_config.model))
        return ModelConfig(**payload)

    def _should_use_runner_bridge(self, task_context: Dict[str, Any]) -> bool:
        if task_context.get("current_stage") != "canonical":
            return False
        return True


# ======================================================================
# Concrete operators
# ======================================================================


class MonaFilterOperator(ScoreFilterBase):
    """MONA similarity-based filter operator.

    MONA scores are pre-computed during SAE ingest. Two storage modes:

    1. **Per-benchmark** (preferred): ``metadata.extra.mona_scores``
       is a dict ``{benchmark_name: float}``.  The operator selects
       top-fraction from *each* benchmark independently, then returns
       the **union** of all selected samples.  This prevents any single
       benchmark from dominating the selection.

    2. **Legacy single-score**: ``metadata.extra.mona_score`` is a
       float (combined across all eval sets).  Falls back to standard
       top-k/fraction/threshold selection.
    """

    name = "mona_filter"
    _score_kind = "mona"

    def _score_read_path(self) -> Optional[str]:
        return self.config.get("score_path") or "metadata.extra.mona_score"

    def _has_per_benchmark_scores(self, dataset: Sequence[CanonicalSample]) -> bool:
        """Check if per-benchmark scores are available."""
        for s in dataset[:10]:
            ms = s.metadata.extra.get("mona_scores")
            if isinstance(ms, dict) and ms:
                return True
        return False

    def transform(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        """Override transform to support per-benchmark union selection."""
        if not self._has_per_benchmark_scores(dataset):
            return super().transform(dataset)

        # Per-benchmark union selection
        fraction = float(self.config.get("fraction", 1.0))
        top_k = self.config.get("top_k")
        descending = bool(self.config.get("descending", True))

        # Collect all benchmark names
        benchmark_names: set = set()
        for s in dataset:
            ms = s.metadata.extra.get("mona_scores")
            if isinstance(ms, dict):
                benchmark_names.update(ms.keys())

        if not benchmark_names:
            return super().transform(dataset)

        # For each benchmark, rank and pick top-k/fraction
        selected_ids: set = set()
        for bm in sorted(benchmark_names):
            scored_pairs = []
            for s in dataset:
                ms = s.metadata.extra.get("mona_scores", {})
                score = ms.get(bm)
                if score is not None:
                    scored_pairs.append((s.sample_id, score))

            scored_pairs.sort(key=lambda x: x[1], reverse=descending)

            if top_k is not None:
                keep = int(top_k)
            else:
                keep = math.ceil(len(scored_pairs) * fraction)

            for sid, _ in scored_pairs[:keep]:
                selected_ids.add(sid)

        # Preserve original order
        outputs = [s for s in dataset if s.sample_id in selected_ids]

        if not outputs and dataset:
            logger.warning(
                "%s: per-benchmark union produced 0 samples — keeping top 1 as fallback.",
                self.name,
            )
            outputs = [dataset[0]]

        logger.info(
            "%s: per-benchmark union selection — %d benchmarks, fraction=%.2f, "
            "union=%d/%d samples",
            self.name, len(benchmark_names), fraction,
            len(outputs), len(dataset),
        )
        self._trace.cost.extra["per_benchmark_union"] = True
        self._trace.cost.extra["benchmark_count"] = len(benchmark_names)
        self._trace.cost.extra["union_size"] = len(outputs)
        return outputs





class QualityFilterOperator(ScoreFilterBase):
    """Simple Heuristic or LLM-as-judge Quality Scoring."""
    name = "quality_filter"
    _score_kind = "quality"

    def _score_read_path(self) -> Optional[str]:
        return "metadata.extra.quality.score"

    def _compute_scores(self, dataset: Sequence[CanonicalSample], task_context: Dict[str, Any]) -> Dict[str, float]:
        self._trace.notes["quality_method"] = self.config.get("quality_method", "rule")
        # For mock, assign random high scores or read existing if possible
        scores = super()._compute_scores(dataset, task_context)
        if not scores and self.config.get("quality_method", "rule") == "rule":
            for sample in dataset:
                msg_len = sum(len(msg.content) for msg in sample.messages)
                # Fallback to target text if messages are empty
                if msg_len == 0 and sample.target.text:
                    msg_len = len(sample.target.text)
                scores[sample.sample_id] = min(1.0, msg_len / 2000.0)
        return scores

class RewardModelFilterOperator(_RunnerBridgeMixin, ScoreFilterBase):
    """Reward Model Data Selection (ROSE / Local RM scoring).
    Uses a standard local sequence classification model (e.g., Llama-3-RM)
    to score instruction-response pairs as a proxy for human preference.
    Fast, local, and cheaper than LLM-as-judge.
    """
    name = "reward_model_filter"
    _runner_method = "reward_model"
    _score_kind = "reward_score"

    def _score_read_path(self) -> Optional[str]:
        return "metadata.extra.reward_model.score"

    def _compute_scores(self, dataset: Sequence[CanonicalSample], task_context: Dict[str, Any]) -> Dict[str, float]:
        self._trace.notes["reward_model_path"] = self.config.get("scorer_model_path", "OpenAssistant/reward-model-deberta-v3-large-v2")
        self._trace.notes["score_normalization"] = self.config.get("normalize_scores", False)
        if self._should_use_runner_bridge(task_context):
            return self._run_scoring_bridge(dataset, task_context)
        return super()._compute_scores(dataset, task_context)





class SparseMonaFilterOperator(ScoreFilterBase):
    """MONA-style relevance filter using cached SAE sparse features.

    Computes per-sample generalized Jaccard similarity between the
    sample's cached ``sae_topk`` and the eval target vector.
    No GPU needed — all computation is CPU-only using cached features.

    Typical paper retention: 5-30% of data.

    Config keys:
        fraction: float (0.05-0.30) — fraction of dataset to keep
        top_k: int — alternative to fraction, keep top N samples
    """
    name = "sparse_mona_filter"
    _score_kind = "sparse_mona_jaccard"

    def _score_read_path(self) -> Optional[str]:
        return None  # Scores are computed live from cached sparse features

    def _compute_scores(
        self,
        dataset: Sequence[CanonicalSample],
        task_context: Dict[str, Any],
    ) -> Dict[str, float]:
        from recipe_sandbox.scoring.sparse_features import (
            sparse_jaccard_score,
            get_sparse_features,
        )
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Get the eval target vector from sparse_cache in task_context
        sparse_cache = task_context.get("sparse_cache")
        if sparse_cache is None or getattr(sparse_cache, "eval_target_vector", None) is None:
            logger.warning(
                "SparseMonaFilterOperator: No sparse_cache or eval_target_vector available. "
                "Falling back to pre-computed similarity scores in metadata."
            )
            self._trace.notes["sparse_mona_error"] = "No sparse_cache or eval_target_vector available — using metadata fallback"
            # Fallback: use pre-computed MONA scores from metadata metadata
            task_name = task_context.get("task_name", "")
            fallback_scores = {}
            for s in dataset:
                sims = s.metadata.extra.get("mona_scores")
                if not isinstance(sims, dict):
                    sims = s.metadata.extra.get("similarities", {})
                if task_name and task_name in sims:
                    fallback_scores[s.sample_id] = sims[task_name]
                elif sims:
                    fallback_scores[s.sample_id] = max(sims.values())
            if fallback_scores:
                logger.info("SparseMonaFilter: found %d pre-computed similarity scores as fallback.", len(fallback_scores))
                return fallback_scores
            logger.warning("SparseMonaFilter: no fallback scores found either. Returning empty — filter will keep all if keep_missing_scores=True.")
            return {}

        target_vector = sparse_cache.eval_target_vector

        # Pre-filter: only score samples that have sparse features
        scoreable = [(i, s) for i, s in enumerate(dataset) if get_sparse_features(s) is not None]

        if not scoreable:
            logger.warning(
                "SparseMonaFilterOperator: 0/%d samples have sparse features. "
                "Falling back to pre-computed similarity scores.", len(dataset)
            )
            self._trace.notes["sparse_mona_scored"] = 0
            self._trace.notes["sparse_mona_total"] = len(dataset)
            task_name = task_context.get("task_name", "")
            fallback_scores = {}
            for s in dataset:
                sims = s.metadata.extra.get("mona_scores")
                if not isinstance(sims, dict):
                    sims = s.metadata.extra.get("similarities", {})
                if task_name and task_name in sims:
                    fallback_scores[s.sample_id] = sims[task_name]
                elif sims:
                    fallback_scores[s.sample_id] = max(sims.values())
            return fallback_scores

        # Parallel scoring with threads (numpy releases GIL during computation)
        num_workers = min(os.cpu_count() or 4, 16, len(scoreable))
        scores: Dict[str, float] = {}

        def _score_one(sample: CanonicalSample) -> tuple:
            return sample.sample_id, sparse_jaccard_score(sample, target_vector)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_score_one, s): s for _, s in scoreable}
            for future in as_completed(futures):
                sid, score = future.result()
                scores[sid] = score

        self._trace.notes["sparse_mona_scored"] = len(scores)
        self._trace.notes["sparse_mona_total"] = len(dataset)
        self._trace.notes["sparse_mona_workers"] = num_workers
        if scores:
            vals = sorted(scores.values())
            self._trace.notes["score_range"] = {
                "min": round(vals[0], 4),
                "max": round(vals[-1], 4),
                "median": round(vals[len(vals) // 2], 4),
            }
        return scores


class IFDFilterOperator(ScoreFilterBase):
    """IFD: Instruction-Following Difficulty scoring.

    IFD(sample) = loss(response | instruction) / loss(response)
    Higher IFD → harder/more informative sample → keep.

    Scores must be pre-computed during ingest and stored in
    ``metadata.extra.ifd.score``.

    Ref: Cherry LLM (2024) — "From Quantity to Quality: Boosting LLM
    Performance with Self-Guided Data Selection for Instruction Tuning"

    Typical paper retention: 5-10% of data.

    Config keys:
        fraction: float (0.05-0.20) — fraction of dataset to keep
    """
    name = "ifd_filter"
    _score_kind = "ifd"

    def _score_read_path(self) -> Optional[str]:
        return "metadata.extra.ifd.score"


class NGramEntropyFilterOperator(ScoreFilterBase):
    """Lexical Diversity Filter using Unigram Shannon Entropy.
    
    Higher entropy -> richer vocabulary -> keep.
    Scores must be precomputed during cold start ingest and stored in
    `metadata.extra.ngram_entropy.score`.
    """
    name = "ngram_entropy"
    _score_kind = "entropy"

    def _score_read_path(self) -> Optional[str]:
        return "metadata.extra.ngram_entropy.score"


class ActionObjectBranchingFilterOperator(ScoreFilterBase):
    """Structural Complexity Filter using Dependency Tree Branching.
    
    Higher branching -> deeper instruction logic -> keep.
    Scores must be precomputed during cold start ingest using SpaCy and stored in
    `metadata.extra.action_object.score`.
    """
    name = "action_object_branching"
    _score_kind = "branching_complexity"

    def _score_read_path(self) -> Optional[str]:
        return "metadata.extra.action_object.score"


class VarentropyFilterOperator(ScoreFilterBase):
    """Varentropy-based Complexity Filter.
    
    Higher varentropy indicates more complex reasoning patterns.
    Keep samples with the highest varentropy scores (top fraction).
    Scores must be precomputed during cold start ingest and stored in
    `metadata.extra.varentropy.score`.
    """
    name = "varentropy_filter"
    _score_kind = "varentropy"

    def _score_read_path(self) -> Optional[str]:
        return "metadata.extra.varentropy.score"
