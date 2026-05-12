"""Evaluator base class and result container.

Every evaluator (mock, LoRA, proxy) implements the same interface so
the search loop can swap them freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    """Result of evaluating a single recipe candidate."""

    dev_score: float  # primary score (e.g. GSM8K accuracy)
    train_cost_gpu_hours: float  # estimated GPU-hours for LoRA fine-tune
    eval_cost_gpu_hours: float = 0.0  # evaluation overhead
    task_scores: Dict[str, float] = field(default_factory=dict)  # per-task breakdown
    extra: Dict[str, Any] = field(default_factory=dict)  # arbitrary metadata


class BaseEvaluator(ABC):
    """Interface that all evaluators must implement."""

    @abstractmethod
    def evaluate(
        self,
        dataset_path: str,
        recipe_name: str,
        *,
        task_names: Optional[List[str]] = None,
    ) -> EvalResult:
        """Evaluate a dataset produced by a recipe.

        Args:
            dataset_path: Path to the JSONL dataset output of the recipe.
            recipe_name: Human-readable recipe identifier for logging.
            task_names: Optional list of eval tasks (e.g. ["gsm8k", "gpqa"]).

        Returns:
            EvalResult with scores and cost estimates.
        """
