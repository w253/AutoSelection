"""Evaluation module for recipe search.

Provides pluggable evaluators (mock, LoRA), GPU/NPU selection, and
cost-aware utility computation.
"""

from recipe_sandbox.evaluation.evaluator_base import BaseEvaluator, EvalResult
from recipe_sandbox.evaluation.utility import UtilityConfig, compute_utility
from recipe_sandbox.evaluation.npu_selector import select_idle_devices

__all__ = [
    "BaseEvaluator",
    "EvalResult",
    "UtilityConfig",
    "compute_utility",
    "select_idle_devices",
]
