from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ScoringBatchResult": ("recipe_sandbox.scoring.base", "ScoringBatchResult"),
    "MonaFeatureExtractor": ("recipe_sandbox.scoring.mona", "MonaFeatureExtractor"),
    "MonaScorer": ("recipe_sandbox.scoring.mona", "MonaScorer"),
    "generalized_jaccard_similarity": ("recipe_sandbox.scoring.mona", "generalized_jaccard_similarity"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'recipe_sandbox.scoring' has no attribute '{name}'")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)