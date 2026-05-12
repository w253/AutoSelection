"""Union Operator — merges current samples with output from another recipe branch.

This transforms the search tree into a DAG by allowing recipes to combine
(A ∪ B) rather than only filter. This is critical because our experiments
show that more data generally leads to better scores, so recovering data
lost by aggressive filtering is valuable.

Usage in recipe:
  {"operator": "union", "params": {"source_recipe": "lhs_seed_1"}}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from recipe_sandbox.operators.base import BaseOperator
from recipe_sandbox.schema.types import CanonicalSample

logger = logging.getLogger(__name__)


class UnionOperator(BaseOperator):
    """Merge current samples with output from another recipe branch.

    Loads dataset.ids.json from the source recipe's output directory,
    filters pool samples to those IDs, and adds them to the current dataset
    (deduplicating by sample_id).

    Config:
        source_recipe (str): Name of a previously-evaluated recipe whose
            output IDs to merge with the current dataset.
    """
    name = "union"
    operator_type = "merge"
    version = "v1"

    def __init__(self, source_recipe: str = "", **config: Any) -> None:
        super().__init__(source_recipe=source_recipe, **config)
        self.source_recipe = source_recipe

    def transform(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        """Merge current dataset with samples from source_recipe."""
        if not self.source_recipe:
            logger.warning("UnionOperator: no source_recipe specified, returning dataset unchanged.")
            self._trace.notes["union_effective"] = False
            self._trace.notes["union_reason"] = "no_source_recipe"
            return list(dataset)

        task_context = self._runtime_context

        # Find the recipes directory from task context
        base_manager = task_context.get("base_task_manager")
        if base_manager is None:
            logger.error("UnionOperator: no task manager in context, cannot locate source recipe.")
            self._trace.notes["union_effective"] = False
            self._trace.notes["union_reason"] = "no_task_manager"
            return list(dataset)

        if hasattr(base_manager, "recipe_dataset_ids_path"):
            source_ids_path = Path(base_manager.recipe_dataset_ids_path(self.source_recipe))
        elif hasattr(base_manager, "root"):
            source_ids_path = Path(base_manager.root) / "recipes" / self.source_recipe / "dataset.ids.json"
        elif hasattr(base_manager, "output_dir"):
            source_ids_path = Path(base_manager.output_dir) / "recipes" / self.source_recipe / "dataset.ids.json"
        else:
            logger.error(
                "UnionOperator: base_task_manager cannot resolve recipe dataset paths "
                "(type=%s).", type(base_manager).__name__
            )
            self._trace.notes["union_effective"] = False
            self._trace.notes["union_reason"] = f"invalid_manager_type:{type(base_manager).__name__}"
            return list(dataset)

        if not source_ids_path.exists():
            logger.warning(
                "UnionOperator: source recipe IDs not found at %s. "
                "Returning dataset unchanged.", source_ids_path
            )
            self._trace.notes["union_effective"] = False
            self._trace.notes["union_reason"] = f"ids_not_found:{source_ids_path}"
            return list(dataset)

        # Load source IDs
        with open(source_ids_path, "r") as f:
            source_ids = set(json.load(f))

        if not source_ids:
            logger.warning("UnionOperator: source recipe '%s' has empty IDs.", self.source_recipe)
            self._trace.notes["union_effective"] = False
            self._trace.notes["union_reason"] = "empty_source_ids"
            return list(dataset)

        # Get pool samples from context (the full hydrated pool)
        pool_samples = task_context.get("pool_samples")
        if pool_samples is None:
            logger.warning("UnionOperator: no pool_samples in context. Cannot perform union.")
            self._trace.notes["union_effective"] = False
            self._trace.notes["union_reason"] = "no_pool_samples"
            return list(dataset)

        # Current IDs for deduplication
        current_ids = {s.sample_id for s in dataset}

        # Find new samples: in source but not in current
        new_samples = []
        for s in pool_samples:
            if s.sample_id in source_ids and s.sample_id not in current_ids:
                new_samples.append(s)

        merged = list(dataset) + new_samples
        self._trace.notes["union_effective"] = len(new_samples) > 0
        self._trace.notes["union_source_recipe"] = self.source_recipe
        self._trace.notes["union_source_ids_count"] = len(source_ids)
        self._trace.notes["union_added_count"] = len(new_samples)
        self._trace.notes["union_merged_total"] = len(merged)
        logger.info(
            "UnionOperator: merged %d current + %d new from '%s' = %d total "
            "(source had %d IDs, %d were new)",
            len(dataset), len(new_samples), self.source_recipe,
            len(merged), len(source_ids), len(new_samples),
        )
        return merged
