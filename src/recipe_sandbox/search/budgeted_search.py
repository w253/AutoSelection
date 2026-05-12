"""
Budgeted Recipe Search Loop

Phase 3 autonomous recipe search:
1. Evaluate Seed Recipes (cold start)
2. Iterate:
   - Observe Best Recipe + Diagnoses
   - Action LLM proposes Next Candidate
   - Execute & Evaluate Pipeline (via pluggable Evaluator)
   - Update Pareto front / Best utility until Budget is exhausted
3. Write JSON Lines search log
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from recipe_sandbox.evaluation.evaluator_base import BaseEvaluator, EvalResult
from recipe_sandbox.evaluation.utility import UtilityConfig, compute_utility
from recipe_sandbox.pipeline.recipe_executor import RecipeDataBus, RecipeExecutor
from recipe_sandbox.pipeline.task_config import RecipeConfig, RecipeStepConfig
from recipe_sandbox.pipeline.task_manager import TaskManager
from recipe_sandbox.agents.action_llm import ActionLLMGenerator


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
#  Data Containers
# -----------------------------------------------------------------------

@dataclass
class SearchCandidate:
    recipe: RecipeConfig
    score: float = 0.0
    cost: float = 0.0
    utility: float = 0.0
    step_traces: Optional[List[Dict]] = None
    eval_result: Optional[EvalResult] = None
    output_samples: int = 0
    iteration: int = 0
    parent_name: str = ""
    eval_mode: str = "full"
    predicted_score: float = 0.0
    state_features: Optional[Any] = None  # 18D feature vector
    parent_visits: int = 0
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    trajectory_id: int = 0
    proposal_index: int = 0
    origin_type: str = "candidate"

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "recipe_name": self.recipe.recipe_name,
            "steps": [
                {"operator": s.operator, "params": s.params}
                for s in (self.recipe.steps or [])
                if s.enabled
            ],
            "score": self.score,
            "predicted_score": self.predicted_score,
            "cost": self.cost,
            "utility": self.utility,
            "iteration": self.iteration,
            "parent_name": self.parent_name,
            "eval_mode": self.eval_mode,
            "output_samples": self.output_samples,
            "per_benchmark": self.eval_result.task_scores if self.eval_result else {},
            "eval_extra": self.eval_result.extra if self.eval_result else {},
            "parent_visits": self.parent_visits,
            "cost_breakdown": self.cost_breakdown,
            "trajectory_id": self.trajectory_id,
            "proposal_index": self.proposal_index,
            "origin_type": self.origin_type,
        }


# -----------------------------------------------------------------------
#  Search Controller
# -----------------------------------------------------------------------

class BudgetedRecipeSearch:
    """Central orchestrator for autonomous recipe search."""

    def __init__(
        self,
        manager: TaskManager,
        recipe_executor: RecipeExecutor,
        action_generator: ActionLLMGenerator,
        evaluator: BaseEvaluator,
        *,
        budget_gpu_hours: float = 10.0,
        utility_config: Optional[UtilityConfig] = None,
        search_log_path: Optional[str] = None,
    ):
        self.manager = manager
        self.executor = recipe_executor
        self.action_generator = action_generator
        self.evaluator = evaluator

        self.budget_gpu_hours = budget_gpu_hours
        self.utility_config = utility_config or UtilityConfig()

        self.history: List[SearchCandidate] = []
        self.cumulative_cost = 0.0
        self._iteration = 0
        self._cost_breakdown_totals: Dict[str, float] = {
            "llm": 0.0,
            "pipeline": 0.0,
            "evaluation": 0.0,
        }

        self._log_path = Path(search_log_path) if search_log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    #  Core evaluate step
    # ------------------------------------------------------------------

    def _evaluate_candidate(self, recipe: RecipeConfig) -> SearchCandidate:
        """Run pipeline → evaluate → log."""
        self._iteration += 1
        logger.info(
            "\n[%d | %.2f/%.2fh] Evaluating: %s",
            self._iteration,
            self.cumulative_cost,
            self.budget_gpu_hours,
            recipe.recipe_name,
        )

        # 1. Pipeline execution (applies operators, records state vectors)
        pipeline_start = time.perf_counter()
        result = self.executor.run(recipe)
        pipeline_cost = self._seconds_to_hours(time.perf_counter() - pipeline_start)
        self._charge_budget_hours(pipeline_cost, "pipeline")

        # Guard: if recipe produced 0 samples, skip expensive eval
        if result.output_samples == 0:
            logger.warning(
                "Recipe '%s' produced 0 samples — skipping LoRA eval, assigning score=0.",
                recipe.recipe_name,
            )
            candidate = SearchCandidate(
                recipe=recipe,
                score=0.0,
                cost=pipeline_cost,
                utility=0.0,
                step_traces=result.step_traces,
                iteration=self._iteration,
                output_samples=0,
                cost_breakdown={"pipeline_hours": pipeline_cost},
            )
            self.history.append(candidate)
            self._write_log_entry(candidate)
            return candidate

        # 2. Evaluation
        final_state = result.final_state  # Dict from the executor
        dataset_path = str(getattr(result, "output_path", ""))
        eval_result = self.evaluator.evaluate(
            dataset_path=dataset_path,
            recipe_name=recipe.recipe_name,
            state_vector=final_state,
        )

        # 3. Cost-aware utility
        step_cost = eval_result.train_cost_gpu_hours + eval_result.eval_cost_gpu_hours
        utility = compute_utility(
            dev_score=eval_result.dev_score,
            search_cost=self.cumulative_cost + step_cost,
            train_cost=eval_result.train_cost_gpu_hours,
            config=self.utility_config,
        )

        self._charge_budget_hours(step_cost, "evaluation")

        candidate = SearchCandidate(
            recipe=recipe,
            score=eval_result.dev_score,
            cost=pipeline_cost + step_cost,
            utility=utility,
            step_traces=result.step_traces,
            eval_result=eval_result,
            iteration=self._iteration,
            output_samples=result.output_samples,
            cost_breakdown={
                "pipeline_hours": pipeline_cost,
                "evaluation_hours": step_cost,
            },
        )
        self.history.append(candidate)
        self._write_log_entry(candidate)
        return candidate

    # ------------------------------------------------------------------
    #  Main search loop
    # ------------------------------------------------------------------

    def search(self) -> SearchCandidate:
        """Run the full budgeted search."""
        logger.info("Starting Budgeted Search (budget=%.1fh)...", self.budget_gpu_hours)

        # --- 1. Cold Start: Seed Nodes ---
        seed_baseline = RecipeConfig(
            enabled=True,
            recipe_name="seed_baseline",
            input_split="train",
            input_stage="canonical",
            steps=[],
        )
        seed_clean_dedup = RecipeConfig(
            enabled=True,
            recipe_name="seed_clean_dedup",
            input_split="train",
            input_stage="canonical",
            steps=[
                RecipeStepConfig(
                    operator="semantic_dedup",
                    params={"strategy": "minhash", "jaccard_threshold": 0.85},
                    enabled=True,
                    name="dedup",
                ),
            ],
        )

        self._evaluate_candidate(seed_baseline)
        self._evaluate_candidate(seed_clean_dedup)

        best = self._get_best()
        logger.info(
            "Cold start done. Best: %s (utility=%.2f, score=%.2f)",
            best.recipe.recipe_name,
            best.utility,
            best.score,
        )

        # --- 2. LLM-driven search loop ---
        while self.cumulative_cost < self.budget_gpu_hours:
            logger.info(
                "--- Iteration %d (cost=%.2f/%.2fh) ---",
                self._iteration + 1,
                self.cumulative_cost,
                self.budget_gpu_hours,
            )

            try:
                next_recipe = self.action_generator.propose_next_recipe(
                    current_recipe=best.recipe,
                    diagnoses=[],
                    score=best.score,
                    cost=self.cumulative_cost,
                )
            except Exception as exc:
                logger.error("Action LLM failed: %s. Stopping search.", exc)
                break

            candidate = self._evaluate_candidate(next_recipe)

            if candidate.utility > best.utility:
                logger.info(
                    "★ New best! %s → utility %.2f (was %.2f)",
                    candidate.recipe.recipe_name,
                    candidate.utility,
                    best.utility,
                )
                best = candidate
            else:
                logger.info(
                    "No improvement (%.2f ≤ %.2f). Continuing.",
                    candidate.utility,
                    best.utility,
                )

        logger.info(
            "Search finished. Best recipe: %s (utility=%.2f, score=%.2f, total_cost=%.2fh)",
            best.recipe.recipe_name,
            best.utility,
            best.score,
            self.cumulative_cost,
        )
        return best

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _get_best(self) -> SearchCandidate:
        return max(self.history, key=lambda c: c.score)

    def get_pareto_front(self) -> List[SearchCandidate]:
        """Return Pareto-optimal candidates (score vs cost)."""
        sorted_by_score = sorted(self.history, key=lambda c: c.score, reverse=True)
        pareto: List[SearchCandidate] = []
        min_cost = float("inf")
        for c in sorted_by_score:
            if c.cost < min_cost:
                pareto.append(c)
                min_cost = c.cost
        return pareto

    def _write_log_entry(self, candidate: SearchCandidate) -> None:
        if not self._log_path:
            return
        entry = candidate.to_log_dict()
        entry["total_cost_hours"] = round(self.cumulative_cost, 4)
        entry["total_cost_breakdown"] = self._current_cost_breakdown()
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _ensure_cost_trackers(self) -> None:
        if not hasattr(self, "_cost_breakdown_totals"):
            self._cost_breakdown_totals = {
                "llm": 0.0,
                "pipeline": 0.0,
                "evaluation": 0.0,
            }

    def _seconds_to_hours(self, seconds: float) -> float:
        return max(0.0, float(seconds)) / 3600.0

    def _charge_budget_hours(self, hours: float, category: str) -> None:
        hours = max(0.0, float(hours))
        if hours == 0.0:
            return
        self._ensure_cost_trackers()
        self.cumulative_cost += hours
        self._cost_breakdown_totals[category] = self._cost_breakdown_totals.get(category, 0.0) + hours

    def _current_cost_breakdown(self) -> Dict[str, float]:
        self._ensure_cost_trackers()
        return {
            key: round(value, 6)
            for key, value in self._cost_breakdown_totals.items()
            if value > 0
        }
