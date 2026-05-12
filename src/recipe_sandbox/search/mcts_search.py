import json
import logging
import random
import time
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from recipe_sandbox.search.budgeted_search import BudgetedRecipeSearch, SearchCandidate
from recipe_sandbox.evaluation.evaluator_base import BaseEvaluator
from recipe_sandbox.evaluation.utility import UtilityConfig
from recipe_sandbox.pipeline.recipe_executor import RecipeExecutor
from recipe_sandbox.pipeline.task_manager import TaskManager
from recipe_sandbox.agents.action_llm import ActionLLMGenerator
from recipe_sandbox.agents.feedback_llm import FeedbackLLM
from recipe_sandbox.agents.selection_llm import SelectionLLM
from recipe_sandbox.pipeline.task_config import RecipeConfig, RecipeStepConfig
from recipe_sandbox.feedback.benchmark_suggest import BenchmarkSuggestor

from recipe_sandbox.surrogate.lhs_sampler import generate_lhs_seeds
from recipe_sandbox.surrogate.model import ANOVARegressor
from recipe_sandbox.search import family_for_operator
from recipe_sandbox.search.operator_policy import resolve_operator_space

logger = logging.getLogger(__name__)

class MCTSSearchLoop(BudgetedRecipeSearch):
    """MCTS-inspired search loop using ANOVARegressor GP and LLM heuristics.
    
    1. Generates deterministic staged warmup recipes.
    2. Builds a candidate pool via ActionLLM (incorporating Diagnosis feedback).
    3. Batch-executes candidate pipelines to get state vectors (for LLM context).
    4. Scores candidates via ANOVARegressor (recipe encoding → utility) computing UCB.
    5. SelectionLLM picks one candidate for full evaluation.
    """
    
    def __init__(
        self,
        manager: TaskManager,
        recipe_executor: RecipeExecutor,
        action_generator: ActionLLMGenerator,
        evaluator: BaseEvaluator,
        catalog_path: str,
        *,
        budget_gpu_hours: float = 10.0,
        utility_config: Optional[UtilityConfig] = None,
        search_log_path: Optional[str] = None,
        k_exploration: float = 1.0,
        n_lhs_seeds: int = 8,
        pool_size: int = 0,
        feedback_llm: Optional[FeedbackLLM] = None,
        selection_llm: Optional[SelectionLLM] = None,
        thinking_logger: Optional[Any] = None,
        stagnation_patience: int = 3,
    ):
        super().__init__(
            manager, recipe_executor, action_generator, evaluator,
            budget_gpu_hours=budget_gpu_hours,
            utility_config=utility_config,
            search_log_path=search_log_path
        )
        if feedback_llm is None:
            raise ValueError("feedback_llm is required in the final MCTS search path.")
        if selection_llm is None:
            raise ValueError("selection_llm is required in the final MCTS search path.")
        self.stagnation_patience: int = stagnation_patience
        registry = getattr(self.executor, "registry", None)
        operator_order = resolve_operator_space(registry.names()) if registry is not None else None
        self.surrogate = ANOVARegressor(operator_order=operator_order)
        self.catalog_path = catalog_path
        self.feedback_llm = feedback_llm
        self.selection_llm = selection_llm
        self.k_exploration = k_exploration
        self.n_lhs_seeds = n_lhs_seeds
        self.pool_size = pool_size
        self.thinking_logger = thinking_logger
        self._unexplored_pool: List[RecipeConfig] = []
        self._parent_rotation_idx: int = 0
        self._benchmark_suggestor = BenchmarkSuggestor()
        self._input_source_names = self.executor.input_source_names("canonical", "train")
        # Trajectory tracking
        self._current_trajectory_id: int = 0
        self._global_best_score: float = float("-inf")
        self._stagnation_count: int = 0

    def _all_recipe_names(self) -> set[str]:
        names = {candidate.recipe.recipe_name for candidate in self.history}
        names.update(recipe.recipe_name for recipe in self._unexplored_pool)
        return names

    def _name_recipe(
        self,
        recipe: RecipeConfig,
        *,
        origin_type: str,
        proposal_index: int = 0,
    ) -> RecipeConfig:
        if origin_type == "restart_seed":
            base_name = f"traj{self._current_trajectory_id}_seed"
        elif origin_type == "fallback":
            base_name = f"traj{self._current_trajectory_id}_i{self._iteration + 1:03d}_fb{proposal_index:02d}"
        elif origin_type == "candidate":
            base_name = f"traj{self._current_trajectory_id}_i{self._iteration + 1:03d}_c{proposal_index:02d}"
        else:
            base_name = recipe.recipe_name

        existing_names = self._all_recipe_names()
        name = base_name
        suffix = 2
        while name in existing_names:
            name = f"{base_name}_{suffix}"
            suffix += 1

        recipe.recipe_name = name
        recipe._origin_type = origin_type
        recipe._proposal_index = proposal_index
        return recipe

    # ------------------------------------------------------------------
    #  Search history context for LLM prompts
    # ------------------------------------------------------------------

    def _render_search_history(self, last_n: int = 10) -> str:
        """Render recent search history as compact text for LLM context."""
        if not self.history:
            return ""

        recent = self.history[-last_n:]
        parent_scores = {c.recipe.recipe_name: c.score for c in self.history}

        lines = [f"=== SEARCH HISTORY (last {min(last_n, len(self.history))} iterations) ==="]
        for c in recent:
            parent_name = getattr(c, "parent_name", "")
            marker = "★"

            # Build name with parent lineage
            if parent_name:
                display_name = f"{parent_name}→{c.recipe.recipe_name.split('_')[-1]}"
            else:
                display_name = c.recipe.recipe_name

            # Score display
            score_str = f"score={c.score:.2f}%, {c.output_samples} samples"
            if parent_name and parent_name in parent_scores:
                delta = c.score - parent_scores[parent_name]
                score_str += f", Δ={delta:+.2f}"
            tag = "VERIFIED"

            # Compact operator summary
            ops = []
            for s in (c.recipe.steps or []):
                if s.enabled:
                    key_params = ", ".join(
                        f"{k}={v}" for k, v in list(s.params.items())[:2]
                    )
                    ops.append(f"{s.operator}({key_params})" if key_params else s.operator)
            ops_str = "[" + ", ".join(ops) + "]" if ops else "[]"

            lines.append(
                f"iter={c.iteration} {marker} {display_name} ({score_str}) "
                f"— {tag} {ops_str}"
            )

        # Trend detection: quantity vs quality correlation
        verified = [c for c in self.history if getattr(c, "eval_mode", "full") == "full"]
        if len(verified) >= 3:
            samples_list = [c.output_samples for c in verified]
            scores_list = [c.score for c in verified]
            corr = np.corrcoef(samples_list, scores_list)[0, 1]
            if corr > 0.3:
                lines.append("Trend: more data → better scores. Consider relaxing filters.")
            elif corr < -0.3:
                lines.append("Trend: quality filtering helps. Consider tighter filters.")
            else:
                lines.append("Trend: no clear quantity-quality pattern. Explore both directions.")

        best = self._get_best()
        lines.append(f"Best: {best.recipe.recipe_name} at {best.score:.2f}%")

        credit_summary = self._render_operator_credit_guidance()
        if credit_summary:
            lines.append(credit_summary)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Feedback history for FeedbackLLM
    # ------------------------------------------------------------------

    def _build_feedback_history(self) -> list:
        """Build history dicts from SearchCandidate history for FeedbackLLM."""
        result = []
        for c in self.history:
            operators = []
            params = {}
            for s in (c.recipe.steps or []):
                if s.enabled:
                    operators.append(s.operator)
                    params[s.operator] = dict(s.params) if s.params else {}

            per_benchmark = {}
            if c.eval_result and c.eval_result.task_scores:
                per_benchmark = c.eval_result.task_scores
            else:
                # Fallback: try raw_metrics in eval_result.extra
                extra = (c.eval_result.extra if c.eval_result else {}) or {}
                raw_metrics = extra.get("raw_metrics", {})
                if raw_metrics:
                    per_benchmark = raw_metrics.get("task_scores", {})

            state_vector = None
            if c.step_traces:
                state_vector = c.step_traces[-1].get("state_after")

            result.append({
                "recipe_name": c.recipe.recipe_name,
                "operators": operators,
                "params": params,
                "score": c.score,
                "per_benchmark": per_benchmark,
                "output_samples": c.output_samples,
                "state_vector": state_vector,
                "eval_mode": getattr(c, "eval_mode", "full"),
            })
        return result

    # ------------------------------------------------------------------
    #  Rich context for SelectionLLM
    # ------------------------------------------------------------------

    def _build_selection_context(self, parent: 'SearchCandidate') -> dict:
        """Build rich structured context for SelectionLLM decision-making.

        Returns a dict with per-benchmark history, state vectors, and parent details
        that supplements the basic search_history text.
        """
        # Per-benchmark history table
        verified = [c for c in self.history if getattr(c, "eval_mode", "full") == "full"]
        history_entries = []
        for c in verified:
            per_bm = {}
            if c.eval_result and c.eval_result.task_scores:
                per_bm = c.eval_result.task_scores
            else:
                extra = (c.eval_result.extra if c.eval_result else {}) or {}
                raw = extra.get("raw_metrics", {})
                if raw:
                    per_bm = raw.get("task_scores", {})

            state = None
            if c.step_traces:
                state = c.step_traces[-1].get("state_after")

            history_entries.append({
                "name": c.recipe.recipe_name,
                "score": c.score,
                "per_benchmark": per_bm,
                "samples": c.output_samples,
                "state": state,
                "iteration": c.iteration,
            })

        # Parent details
        parent_state = None
        if parent.step_traces:
            parent_state = parent.step_traces[-1].get("state_after")
        parent_per_bm = {}
        if parent.eval_result and parent.eval_result.task_scores:
            parent_per_bm = parent.eval_result.task_scores

        return {
            "history_entries": history_entries,
            "parent": {
                "name": parent.recipe.recipe_name,
                "score": parent.score,
                "per_benchmark": parent_per_bm,
                "samples": parent.output_samples,
                "state_vector": parent_state,
            },
        }

    # ------------------------------------------------------------------
    #  Trajectory-based parent selection
    # ------------------------------------------------------------------

    def _select_expansion_parent(self) -> Optional[SearchCandidate]:
        """Return the globally best verified candidate as expansion parent.

        Always selects the highest-scoring verified candidate across all
        trajectories, so the search always expands from the best known recipe.
        """
        verified = [
            c for c in self.history
            if getattr(c, "eval_mode", "full") == "full"
        ]
        if not verified:
            logger.warning("No verified nodes remain for expansion.")
            return None

        selected = max(verified, key=lambda c: (c.score, c.iteration))

        logger.info(
            "Selected global-best parent '%s' (score=%.2f, iter=%d)",
            selected.recipe.recipe_name,
            selected.score,
            selected.iteration,
        )
        return selected

    # ------------------------------------------------------------------
    #  Trajectory restart on stagnation
    # ------------------------------------------------------------------

    def _recipe_step_map(self, recipe: RecipeConfig) -> Dict[str, Dict[str, Any]]:
        return {
            step.operator: dict(step.params)
            for step in (recipe.steps or [])
            if step.enabled
        }

    def _extract_operator_changes(
        self,
        parent_recipe: RecipeConfig,
        child_recipe: RecipeConfig,
    ) -> Dict[str, List[str]]:
        parent_steps = self._recipe_step_map(parent_recipe)
        child_steps = self._recipe_step_map(child_recipe)

        added = sorted(op for op in child_steps if op not in parent_steps)
        removed = sorted(op for op in parent_steps if op not in child_steps)
        param_changed = sorted(
            op
            for op in child_steps
            if op in parent_steps and child_steps[op] != parent_steps[op]
        )
        return {
            "added": added,
            "removed": removed,
            "param_changed": param_changed,
        }

    def _build_operator_credit_summary(self) -> Dict[str, Any]:
        verified = [
            candidate
            for candidate in self.history
            if getattr(candidate, "eval_mode", "full") == "full"
        ]
        by_name = {candidate.recipe.recipe_name: candidate for candidate in verified}
        operator_credit: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"score_credit": 0.0, "total_credit": 0.0, "count": 0.0}
        )
        operator_pair_credit: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"score_credit": 0.0, "total_credit": 0.0, "count": 0.0}
        )

        transitions = 0
        for child in verified:
            if not child.parent_name:
                continue
            parent = by_name.get(child.parent_name)
            if parent is None:
                continue

            changes = self._extract_operator_changes(parent.recipe, child.recipe)
            positive_targets = sorted(set(changes["added"] + changes["param_changed"]))
            removed_targets = sorted(set(changes["removed"]))
            if not positive_targets and not removed_targets:
                continue

            delta_score = child.score - parent.score
            delta_total = delta_score
            transitions += 1

            if positive_targets:
                share = 1.0 / len(positive_targets)
                for operator_name in positive_targets:
                    stats = operator_credit[operator_name]
                    stats["score_credit"] += delta_score * share
                    stats["total_credit"] += delta_total * share
                    stats["count"] += 1.0

            if removed_targets:
                share = 1.0 / len(removed_targets)
                for operator_name in removed_targets:
                    stats = operator_credit[operator_name]
                    stats["score_credit"] -= delta_score * share
                    stats["total_credit"] -= delta_total * share
                    stats["count"] += 1.0

            active_pair_targets = [
                operator_name
                for operator_name in positive_targets
                if operator_name not in {"truncate_samples", "union"}
            ]
            if len(active_pair_targets) >= 2:
                pair_share = 1.0 / len(list(combinations(active_pair_targets, 2)))
                for pair in combinations(active_pair_targets, 2):
                    normalized_pair = tuple(sorted(pair))
                    stats = operator_pair_credit[normalized_pair]
                    stats["score_credit"] += delta_score * pair_share
                    stats["total_credit"] += delta_total * pair_share
                    stats["count"] += 1.0

        ranked_operators = sorted(
            operator_credit.items(),
            key=lambda item: (item[1]["total_credit"], item[1]["score_credit"]),
            reverse=True,
        )
        ranked_pairs = sorted(
            operator_pair_credit.items(),
            key=lambda item: (item[1]["total_credit"], item[1]["score_credit"]),
            reverse=True,
        )
        return {
            "transitions": transitions,
            "operator_credit": operator_credit,
            "operator_pair_credit": operator_pair_credit,
            "ranked_operators": ranked_operators,
            "ranked_pairs": ranked_pairs,
            "top_positive_operators": [item for item in ranked_operators if item[1]["total_credit"] > 0][:5],
            "top_negative_operators": [item for item in ranked_operators if item[1]["total_credit"] < 0][-5:],
            "top_positive_pairs": [item for item in ranked_pairs if item[1]["total_credit"] > 0][:5],
        }

    def _render_operator_credit_guidance(self) -> str:
        credit = self._build_operator_credit_summary()
        if not credit["transitions"]:
            return ""

        lines = ["=== VERIFIED OPERATOR CREDIT ==="]
        if credit["top_positive_operators"]:
            positives = ", ".join(
                f"{name}(credit={stats['total_credit']:.2f}, n={int(stats['count'])})"
                for name, stats in credit["top_positive_operators"][:3]
            )
            lines.append(f"Top positive operators: {positives}")
        if credit["top_negative_operators"]:
            negatives = ", ".join(
                f"{name}(credit={stats['total_credit']:.2f}, n={int(stats['count'])})"
                for name, stats in credit["top_negative_operators"][-3:]
            )
            lines.append(f"Top negative operators: {negatives}")
        if credit["top_positive_pairs"]:
            pairs = ", ".join(
                f"{pair[0]}+{pair[1]}(credit={stats['total_credit']:.2f})"
                for pair, stats in credit["top_positive_pairs"][:3]
            )
            lines.append(f"Top positive pairs: {pairs}")
        return "\n".join(lines)

    def _generate_restart_seed(self) -> RecipeConfig:
        """Build a restart recipe through the LLM restart operator path."""
        rng = random.Random()
        available = self._available_search_operators()
        llm_steps = self._generate_llm_restart_steps(available)
        if llm_steps:
            steps = list(llm_steps)
            if self.pool_size > 0 and (not steps or steps[0].operator != "truncate_samples"):
                total = rng.randint(max(1, self.pool_size // 10), self.pool_size)
                steps = [
                    RecipeStepConfig(
                        operator="truncate_samples",
                        params={"total_samples": total},
                        enabled=True,
                        name="restart_truncate",
                    ),
                    *steps,
                ]
            traj_id = self._current_trajectory_id
            recipe = RecipeConfig(
                enabled=True,
                recipe_name=f"traj{traj_id}_seed",
                input_split="train",
                input_stage="canonical",
                steps=steps,
            )
            recipe._origin_type = "restart_seed"
            recipe._proposal_index = 0
            return recipe
        verified = self._top_verified_candidates(limit=5)
        credit_summary = self._build_operator_credit_summary()
        chosen_motif = self._sample_restart_motif(verified, available, rng, credit_summary)

        steps: List[RecipeStepConfig] = []
        if self.pool_size > 0:
            total = rng.randint(max(1, self.pool_size // 10), self.pool_size)
            steps.append(RecipeStepConfig(
                operator="truncate_samples",
                params={"total_samples": total},
                enabled=True,
                name="restart_truncate",
            ))

        historical_examples = self._collect_restart_examples(verified, chosen_motif)
        for op in chosen_motif:
            steps.append(
                RecipeStepConfig(
                    operator=op,
                    params=self._sample_restart_params(op, historical_examples.get(op, []), rng),
                    enabled=True,
                    name=f"restart_{op}",
                )
            )

        action_generator = getattr(self, "action_generator", None)
        if len(steps) > 1 and action_generator is not None and hasattr(action_generator, "tune_restart_params"):
            tune_start = time.perf_counter()
            steps = action_generator.tune_restart_params(
                steps,
                historical_examples=historical_examples,
            )
            self._charge_budget_hours(
                self._seconds_to_hours(time.perf_counter() - tune_start),
                "llm",
            )

        traj_id = self._current_trajectory_id
        recipe = RecipeConfig(
            enabled=True,
            recipe_name=f"traj{traj_id}_seed",
            input_split="train",
            input_stage="canonical",
            steps=steps,
        )
        recipe._origin_type = "restart_seed"
        recipe._proposal_index = 0
        return recipe

    def _generate_llm_restart_steps(self, available: set[str]) -> List[RecipeStepConfig]:
        action_generator = getattr(self, "action_generator", None)
        if action_generator is None or not hasattr(action_generator, "propose_restart_steps"):
            return []

        verified = self._top_verified_candidates(limit=5)
        credit_summary = self._build_operator_credit_summary()
        historical_examples = self._collect_restart_examples(
            verified,
            [name for name, _ in credit_summary.get("top_positive_operators", [])[:3]],
        )
        try:
            return action_generator.propose_restart_steps(
                allowed_operators=available,
                historical_examples=historical_examples,
                search_history=self._render_search_history(),
                credit_summary=credit_summary,
                pool_size=self.pool_size,
            )
        except Exception as exc:
            logger.warning(
                "LLM restart operator selection failed, falling back to credit-based restart: %s",
                exc,
            )
            return []

    def _top_verified_candidates(self, limit: int = 5) -> List[SearchCandidate]:
        verified = [
            c for c in self.history
            if getattr(c, "eval_mode", "full") == "full" and c.score > 0
        ]
        return sorted(
            verified,
            key=lambda candidate: candidate.score,
            reverse=True,
        )[:limit]

    def _sample_restart_motif(
        self,
        verified: List[SearchCandidate],
        available: set[str],
        rng: random.Random,
        credit_summary: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        credit_summary = credit_summary or self._build_operator_credit_summary()
        positive_pairs = [
            (pair, stats)
            for pair, stats in credit_summary.get("top_positive_pairs", [])
            if all(operator_name in available for operator_name in pair)
        ]
        if positive_pairs:
            pairs = [pair for pair, _ in positive_pairs]
            weights = [stats["total_credit"] for _, stats in positive_pairs]
            return list(rng.choices(pairs, weights=weights, k=1)[0])

        positive_operators = [
            (operator_name, stats)
            for operator_name, stats in credit_summary.get("top_positive_operators", [])
            if operator_name in available and operator_name not in {"truncate_samples", "union"}
        ]
        if positive_operators:
            operators = [operator_name for operator_name, _ in positive_operators]
            weights = [stats["total_credit"] for _, stats in positive_operators]
            return [rng.choices(operators, weights=weights, k=1)[0]]

        motif_weights: Dict[Tuple[str, ...], float] = defaultdict(float)
        for candidate in verified:
            motif = tuple(
                step.operator
                for step in (candidate.recipe.steps or [])
                if step.enabled
                and step.operator in available
                and step.operator not in {"truncate_samples", "union"}
            )
            if motif:
                motif_weights[motif] += max(candidate.score, 0.0)

        if motif_weights:
            motifs = list(motif_weights.keys())
            weights = [motif_weights[motif] for motif in motifs]
            return list(rng.choices(motifs, weights=weights, k=1)[0])

        fallback = [
            op for op in ["mona_filter", "ifd_filter", "ngram_entropy", "varentropy_filter"]
            if op in available
        ]
        return [rng.choice(fallback)] if fallback else []

    def _collect_restart_examples(
        self,
        verified: List[SearchCandidate],
        chosen_motif: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        examples: Dict[str, List[Dict[str, Any]]] = {op: [] for op in chosen_motif}
        for candidate in verified:
            for step in (candidate.recipe.steps or []):
                if not step.enabled or step.operator not in examples:
                    continue
                examples[step.operator].append(
                    {
                        "recipe_name": candidate.recipe.recipe_name,
                        "score": candidate.score,
                        "params": dict(step.params),
                    }
                )
        return examples

    def _sample_restart_params(
        self,
        operator_name: str,
        examples: List[Dict[str, Any]],
        rng: random.Random,
    ) -> Dict[str, Any]:
        if examples:
            weights = [
                max(float(example.get("score", 0.0)), 0.0)
                for example in examples
            ]
            chosen = rng.choices(examples, weights=weights, k=1)[0]
            return dict(chosen.get("params", {}))
        if operator_name in {
            "mona_filter",
            "ifd_filter",
            "ngram_entropy",
            "action_object_branching",
            "varentropy_filter",
        }:
            return {"fraction": round(rng.uniform(0.15, 0.85), 2)}
        return {}

    def _start_new_trajectory(self) -> None:
        """Start a new trajectory after stagnation is detected.

        Generates a random restart seed, evaluates it immediately, and
        resets trajectory bookkeeping so the next iteration expands from
        this new seed.
        """
        self._current_trajectory_id += 1
        self._stagnation_count = 0
        logger.info(
            "=== RESTART: starting trajectory %d (stagnation_patience=%d reached) ===",
            self._current_trajectory_id,
            self.stagnation_patience,
        )

    # ------------------------------------------------------------------
    #  Search tree persistence
    # ------------------------------------------------------------------

    def _save_search_tree(self):
        """Persist search tree structure for visualization."""
        tree_path = self._log_path.parent / "search_tree.json" if self._log_path else None
        if not tree_path:
            return
        nodes = []
        for c in self.history:
            eval_mode = getattr(c, "eval_mode", "full")
            predicted_score = getattr(c, "predicted_score", 0.0)
            # Include state vector for user analysis
            state = None
            if c.step_traces:
                state = c.step_traces[-1].get("state_after")
            nodes.append({
                "id": c.recipe.recipe_name,
                "parent": getattr(c, "parent_name", "") or None,
                "trajectory_id": getattr(c, "trajectory_id", 0),
                "proposal_index": getattr(c, "proposal_index", 0),
                "origin_type": getattr(c, "origin_type", "candidate"),
                "iteration": c.iteration,
                "eval_mode": eval_mode,
                "actual_score": c.score,
                "predicted_score": predicted_score,
                "score": c.score,
                "utility": c.utility,
                "output_samples": c.output_samples,
                "operators": [s.operator for s in c.recipe.steps if s.enabled],
                "params": {s.operator: s.params for s in c.recipe.steps if s.enabled},
                "state_vector": state,
            })
        wall_clock_hours = round(
            (time.time() - getattr(self, "_search_start_time", time.time())) / 3600, 4
        )
        tree = {
            "metadata": {
                "search_policy": "llm_only_with_batch_state_context",
                "feedback_policy": "llm_history_summary",
                "stagnation_patience": getattr(self, "stagnation_patience", 3),
                "restart_policy": "llm_stagnation_triggered",
                "total_trajectories": getattr(self, "_current_trajectory_id", 0) + 1,
                "cumulative_cost_hours": round(self.cumulative_cost, 4),
                "wall_clock_hours": wall_clock_hours,
                "cost_breakdown": self._current_cost_breakdown(),
                "parent_selection_policy": "global_best_score",
                "trajectory_restart_policy": "stagnation_triggered_restart",
                "backtrack_policy": "no_node_level_backtrack",
                "operator_credit_guidance": {
                    "top_positive_operators": [
                        {"operator": name, **stats}
                        for name, stats in self._build_operator_credit_summary()["top_positive_operators"][:3]
                    ],
                    "top_negative_operators": [
                        {"operator": name, **stats}
                        for name, stats in self._build_operator_credit_summary()["top_negative_operators"][-3:]
                    ],
                    "top_positive_pairs": [
                        {"operators": list(pair), **stats}
                        for pair, stats in self._build_operator_credit_summary()["top_positive_pairs"][:3]
                    ],
                },
            },
            "nodes": nodes,
        }
        with open(tree_path, "w") as f:
            json.dump(tree, f, indent=2)

    def _available_search_operators(self) -> set[str]:
        registry = getattr(getattr(self, "executor", None), "registry", None)
        operators = registry.names() if registry is not None else ()
        return set(resolve_operator_space(operators))

    def _persist_warmup_recipes(self, recipes: List[RecipeConfig]) -> None:
        if not self._log_path:
            return
        payload = []
        for recipe in recipes:
            payload.append(
                {
                    "recipe_name": recipe.recipe_name,
                    "steps": [
                        {"operator": step.operator, "params": dict(step.params)}
                        for step in (recipe.steps or [])
                        if step.enabled
                    ],
                }
            )
        warmup_path = self._log_path.parent / "warmup_recipes.json"
        with open(warmup_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        logger.info("Persisted %d warmup recipes → %s", len(payload), warmup_path)

    def _load_persisted_warmup_recipes(self) -> List[RecipeConfig]:
        if not self._log_path:
            return []
        warmup_path = self._log_path.parent / "warmup_recipes.json"
        if not warmup_path.exists():
            return []

        with open(warmup_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        recipes: List[RecipeConfig] = []
        for item in payload:
            steps = [
                RecipeStepConfig(
                    operator=step["operator"],
                    params=dict(step.get("params", {})),
                    enabled=True,
                    name=step["operator"],
                )
                for step in item.get("steps", [])
            ]
            recipes.append(
                RecipeConfig(
                    enabled=True,
                    recipe_name=item["recipe_name"],
                    input_split="train",
                    input_stage="canonical",
                    steps=steps,
                )
            )
            recipes[-1]._origin_type = "warmup"
            recipes[-1]._proposal_index = 0
        logger.info("Loaded %d persisted warmup recipes ← %s", len(recipes), warmup_path)
        return recipes

    def _prepare_warmup_recipes(self) -> List[RecipeConfig]:
        recipes = self._load_persisted_warmup_recipes()
        if recipes:
            return recipes

        recipes = generate_lhs_seeds(
            self.catalog_path,
            n_samples=self.n_lhs_seeds,
            pool_size=self.pool_size,
            allowed_operators=tuple(self._available_search_operators()),
        )
        for recipe in recipes:
            recipe._origin_type = "warmup"
            recipe._proposal_index = 0
        self._persist_warmup_recipes(recipes)
        return recipes

    def _run_warmup_phase(self) -> None:
        warmup_recipes = self._prepare_warmup_recipes()
        completed_names = {
            candidate.recipe.recipe_name
            for candidate in self.history
            if getattr(candidate, "eval_mode", "full") == "full"
        }
        pending_recipes = [
            recipe for recipe in warmup_recipes
            if recipe.recipe_name not in completed_names
        ]

        if completed_names:
            logger.info(
                "PHASE 1 Resume: %d/%d warmup recipes already evaluated; %d pending.",
                len(warmup_recipes) - len(pending_recipes),
                len(warmup_recipes),
                len(pending_recipes),
            )
        else:
            logger.info("PHASE 1: staged warmup (%d recipes)...", len(warmup_recipes))

        if not pending_recipes:
            return

        warmup_results = self._execute_candidates_pipelines(pending_recipes)
        valid_warmup = [pr for pr in warmup_results if pr["valid"]]

        for pipeline_result in valid_warmup:
            if self.cumulative_cost >= self.budget_gpu_hours:
                logger.warning("Agent consumed budget during warm-up!")
                break
            try:
                self._evaluate_candidate(
                    recipe=pipeline_result["recipe"],
                    pipeline_result=pipeline_result,
                    parent_name="",
                    predicted_utility=0.0,
                )
            except Exception as exc:
                logger.error(
                    "Warmup full eval failed for '%s': %s. Continuing with remaining warmup recipes.",
                    pipeline_result["recipe"].recipe_name,
                    exc,
                )

        self._update_surrogate()

    # ------------------------------------------------------------------
    #  Batch pipeline execution for all candidates
    # ------------------------------------------------------------------

    def _execute_candidates_pipelines(
        self, candidates: List[RecipeConfig],
        max_workers: int = 3,
    ) -> List[Dict[str, Any]]:
        """Execute pipeline for each candidate to get state vectors (cheap, no training).

        Runs up to ``max_workers`` recipes in parallel via ThreadPoolExecutor.

        Returns a list of dicts, one per candidate:
            {
                "recipe": RecipeConfig,
                "initial_state": dict,
                "final_state": dict,
                "step_traces": list,
                "output_samples": int,
                "output_path": str,
                "valid": bool,  # False if pipeline failed or produced 0 samples
            }
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_one(index: int, recipe: RecipeConfig) -> tuple[int, Dict[str, Any]]:
            logger.info(
                "Pipeline exec %d/%d: executing '%s' (%d step(s))",
                index, len(candidates), recipe.recipe_name,
                len(recipe.steps or []),
            )
            try:
                start = time.perf_counter()
                result = self.executor.run(recipe)
                elapsed_sec = time.perf_counter() - start
                logger.info(
                    "Pipeline exec %d/%d complete: '%s' -> %d samples",
                    index, len(candidates), recipe.recipe_name,
                    result.output_samples,
                )
                return index, {
                    "recipe": recipe,
                    "initial_state": getattr(result, "initial_state", None) or {},
                    "final_state": result.final_state or {},
                    "step_traces": result.step_traces,
                    "output_samples": result.output_samples,
                    "output_path": str(getattr(result, "output_path", "")),
                    "pipeline_wall_clock_sec": elapsed_sec,
                    "valid": result.output_samples > 0,
                }
            except Exception as exc:
                logger.error(
                    "Pipeline exec %d/%d failed for '%s': %s",
                    index, len(candidates), recipe.recipe_name, exc,
                )
                return index, {
                    "recipe": recipe,
                    "initial_state": {},
                    "final_state": {},
                    "step_traces": [],
                    "output_samples": 0,
                    "output_path": "",
                    "pipeline_wall_clock_sec": 0.0,
                    "valid": False,
                }

        if not candidates:
            return []

        actual_workers = min(max_workers, len(candidates))
        results: List[Optional[Dict[str, Any]]] = [None] * len(candidates)
        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = {
                pool.submit(_run_one, i + 1, recipe): i
                for i, recipe in enumerate(candidates)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[futures[future]] = result

        batch_hours = self._seconds_to_hours(time.perf_counter() - batch_start)
        self._charge_budget_hours(batch_hours, "pipeline")
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    #  Budget cost tracking in state vectors
    # ------------------------------------------------------------------

    def _patch_cost_ratio(self, pipeline_result: Dict[str, Any]) -> None:
        """Inject cumulative_cost_ratio into the final state of step traces."""
        cost_ratio = round(
            self.cumulative_cost / self.budget_gpu_hours if self.budget_gpu_hours > 0 else 0.0,
            6,
        )
        step_traces = pipeline_result.get("step_traces", [])
        if step_traces:
            state_after = step_traces[-1].get("state_after")
            if isinstance(state_after, dict):
                state_after["cumulative_cost_ratio"] = cost_ratio
        final_state = pipeline_result.get("final_state")
        if isinstance(final_state, dict):
            final_state["cumulative_cost_ratio"] = cost_ratio

    # ------------------------------------------------------------------
    #  Full evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        recipe: RecipeConfig,
        pipeline_result: Dict[str, Any],
        parent_name: str,
        predicted_utility: float = 0.0,
    ) -> SearchCandidate:
        """Run full training + evaluation for a pipeline-validated recipe."""
        self._iteration += 1

        # Inject cumulative_cost_ratio into the final state vector
        self._patch_cost_ratio(pipeline_result)

        # Guard: 0-sample recipe
        if not pipeline_result["valid"]:
            pipeline_cost = self._seconds_to_hours(pipeline_result.get("pipeline_wall_clock_sec", 0.0))
            logger.warning("Recipe '%s' produced 0 samples — score=0, skipping eval.", recipe.recipe_name)
            candidate = SearchCandidate(
                recipe=recipe, score=0.0, cost=pipeline_cost, utility=0.0,
                step_traces=pipeline_result["step_traces"],
                iteration=self._iteration, parent_name=parent_name,
                eval_mode="full", output_samples=0, predicted_score=0.0,
                cost_breakdown={"pipeline_hours": pipeline_cost},
                proposal_index=int(getattr(recipe, "_proposal_index", 0)),
                origin_type=str(getattr(recipe, "_origin_type", "candidate")),
            )
            self.history.append(candidate)
            self._write_log_entry(candidate)
            return candidate

        pipeline_cost = self._seconds_to_hours(pipeline_result.get("pipeline_wall_clock_sec", 0.0))
        dataset_path = pipeline_result["output_path"]
        final_state = pipeline_result["final_state"]
        eval_result = self.evaluator.evaluate(
            dataset_path=dataset_path,
            recipe_name=recipe.recipe_name,
            state_vector=final_state,
        )

        step_cost = eval_result.train_cost_gpu_hours + eval_result.eval_cost_gpu_hours
        utility = eval_result.dev_score  # utility = score (no cost penalty)
        self._charge_budget_hours(step_cost, "evaluation")

        candidate = SearchCandidate(
            recipe=recipe,
            score=eval_result.dev_score,
            cost=pipeline_cost + step_cost,
            utility=utility,
            step_traces=pipeline_result["step_traces"],
            eval_result=eval_result,
            iteration=self._iteration,
            parent_name=parent_name,
            eval_mode="full",
            output_samples=pipeline_result["output_samples"],
            predicted_score=predicted_utility,
            cost_breakdown={
                "pipeline_hours": pipeline_cost,
                "evaluation_hours": step_cost,
            },
            proposal_index=int(getattr(recipe, "_proposal_index", 0)),
            origin_type=str(getattr(recipe, "_origin_type", "candidate")),
        )
        self.history.append(candidate)
        self._write_log_entry(candidate)

        # Update GP surrogate with all verified observations
        self._update_surrogate()

        logger.info(
            "[FULL EVAL] %s: score=%.2f%%, cost=%.4fh",
            recipe.recipe_name, eval_result.dev_score, step_cost,
        )
        return candidate

    # ------------------------------------------------------------------
    #  Surrogate helpers
    # ------------------------------------------------------------------

    def _update_surrogate(self):
        """Fit ANOVARegressor on all verified (full eval) history."""
        verified = [c for c in self.history if getattr(c, "eval_mode", "full") == "full"]
        if not verified:
            return
        recipes = [c.recipe for c in verified]
        scores = [c.score for c in verified]
        self.surrogate.fit(recipes, scores)
        
    def resume_from_log(self, log_path: str) -> None:
        """Parse historical search log to reconstruct search state."""
        logger.info(f"Resuming search from log: {log_path}")
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_cost = 0.0
        max_traj_id = 0
        for line in lines:
            if not line.strip(): continue
            record = json.loads(line)
            if "recipe_name" not in record:
                continue
            if record.get("eval_mode", "full") != "full":
                logger.info("Skipping non-full historical record during resume: %s", record["recipe_name"])
                continue
            self._iteration += 1
            # Reconstruct recipe with steps (not just empty)
            steps = []
            for s in record.get("steps", []):
                steps.append(RecipeStepConfig(
                    operator=s.get("operator", ""),
                    params=s.get("params", {}),
                    enabled=True,
                    name=s.get("operator", "unknown"),
                ))
            recipe = RecipeConfig(
                recipe_name=record["recipe_name"],
                steps=steps,
                input_split="train",
                input_stage="canonical",
                enabled=True,
            )
            step_cost = record.get("cost", 0.0)
            if "total_cost_hours" in record:
                total_cost = max(total_cost, float(record.get("total_cost_hours", 0.0)))
            else:
                total_cost += step_cost
            traj_id = record.get("trajectory_id", 0)
            max_traj_id = max(max_traj_id, traj_id)
            candidate = SearchCandidate(
                recipe=recipe,
                score=record.get("score", 0.0),
                cost=step_cost,
                utility=record.get("utility", 0.0),
                iteration=record.get("iteration", self._iteration),
                parent_name=record.get("parent_name", ""),
                eval_mode=record.get("eval_mode", "full"),
                parent_visits=record.get("parent_visits", 0),
                cost_breakdown=dict(record.get("cost_breakdown", {})),
                trajectory_id=traj_id,
                proposal_index=record.get("proposal_index", 0),
                origin_type=record.get("origin_type", "candidate"),
            )
            self.history.append(candidate)
        
        self.cumulative_cost = total_cost
        self._current_trajectory_id = max_traj_id
        if self.history:
            # Restore global best score and stagnation state
            all_verified = [
                c for c in self.history
                if getattr(c, "eval_mode", "full") == "full"
            ]
            if all_verified:
                self._global_best_score = max(c.score for c in all_verified)
                # Count trailing non-improvements (across all candidates, not just current trajectory)
                stag = 0
                for c in reversed(sorted(all_verified, key=lambda x: x.iteration)):
                    if c.score < self._global_best_score:
                        stag += 1
                    else:
                        break
                self._stagnation_count = stag

            logger.info(
                "Loaded %d past iterations (cost=%.2fh, trajectory=%d, stagnation=%d/%d). Best: %s",
                len(self.history), self.cumulative_cost,
                self._current_trajectory_id, self._stagnation_count,
                self.stagnation_patience, self._get_best().recipe.recipe_name,
            )
            self._update_surrogate()
        
    def search(self) -> SearchCandidate:
        """Run the MCTS budgeted search loop."""
        self._search_start_time = time.time()
        logger.info(f"Starting Surrogate MCTS Search (budget={self.budget_gpu_hours:.1f}h)")
        
        # --- 1. Cold Start: staged warmup with batch pipeline execution ---
        self._run_warmup_phase()
        if not self.history:
            raise RuntimeError("Warmup produced no valid candidate (all pipelines failed or produced 0 samples).")
        
        best = self._get_best()
        self._global_best_score = best.score
        logger.info(
            "Cold start Phase completed. Baseline best: %s (score=%.2f)",
            best.recipe.recipe_name, best.score
        )

        # --- 2. Surrogate Search Loop (recipe encoding → score) ---
        logger.info("PHASE 2: ANOVARegressor Search (batch pipeline → GP recipe encoding → SelectionLLM)...")
        
        unexplored_pool = self._unexplored_pool
        
        while self.cumulative_cost < self.budget_gpu_hours:
            # Each iteration should execute only the freshly expanded batch.
            unexplored_pool.clear()

            verified_count = sum(1 for c in self.history if getattr(c, "eval_mode", "full") == "full")
            current_iter = self._iteration + 1
            logger.info(
                "--- Iteration %d (cost=%.2f/%.2fh, GP points=%d) ---",
                current_iter, self.cumulative_cost, self.budget_gpu_hours,
                verified_count,
            )
            
            # Sync iteration counter to LLM agents for thinking log
            self.action_generator._iteration = current_iter
            self.feedback_llm._iteration = current_iter
            self.selection_llm._iteration = current_iter
            
            # 0. Select parent for expansion (verified only)
            parent = self._select_expansion_parent()
            if parent is None:
                logger.warning("No verified parent remains. Ending search.")
                break

            # 1. Action LLM (Right Brain) expands the node with search history context.
            llm_succeeded = False
            for _llm_attempt in range(2):
                try:
                    # Extract parent's final state vector for LLM context
                    parent_state = None
                    if parent.step_traces:
                        last_trace = parent.step_traces[-1]
                        parent_state = last_trace.get("state_after")

                    evaluated_recipes = [
                        c.recipe.recipe_name for c in self.history
                        if getattr(c, "eval_mode", "full") == "full"
                    ]

                    # Get experiment insights from feedback LLM
                    experiment_insights = ""
                    if len(self.history) >= 3:
                        feedback_history = self._build_feedback_history()
                        feedback_start = time.perf_counter()
                        experiment_insights = self.feedback_llm.summarize_patterns(feedback_history)
                        self._charge_budget_hours(
                            self._seconds_to_hours(time.perf_counter() - feedback_start),
                            "llm",
                        )
                    operator_credit_guidance = self._render_operator_credit_guidance()
                    if operator_credit_guidance:
                        experiment_insights = (
                            f"{experiment_insights}\n\n{operator_credit_guidance}".strip()
                            if experiment_insights
                            else operator_credit_guidance
                        )

                    # Benchmark diagnostic analysis for Action LLM
                    benchmark_analysis = ""
                    if parent_state and len(self.history) >= 1:
                        best_state_vec = {}
                        best_task_scores_dict = {}
                        if best.step_traces:
                            best_state_vec = best.step_traces[-1].get("state_after", {}) or {}
                        if best.eval_result and best.eval_result.task_scores:
                            best_task_scores_dict = best.eval_result.task_scores

                        parent_task_scores_dict = {}
                        if parent.eval_result and parent.eval_result.task_scores:
                            parent_task_scores_dict = parent.eval_result.task_scores

                        recipe_ops = [
                            {"operator": s.operator, "params": s.params}
                            for s in parent.recipe.steps if s.enabled
                        ]
                        benchmark_analysis = self._benchmark_suggestor.analyze_for_action(
                            parent_state=parent_state or {},
                            best_state=best_state_vec,
                            parent_task_scores=parent_task_scores_dict,
                            best_task_scores=best_task_scores_dict,
                            current_recipe_ops=recipe_ops,
                        )

                    action_start = time.perf_counter()
                    candidate_pool = self.action_generator.propose_candidate_pool(
                        current_recipe=parent.recipe,
                        diagnoses=[],
                        score=parent.score,
                        cost=self.cumulative_cost,
                        n_candidates=5,
                        search_history=self._render_search_history(),
                        pool_size=self.pool_size,
                        state_vector=parent_state,
                        evaluated_recipes=evaluated_recipes,
                        experiment_insights=experiment_insights,
                        benchmark_analysis=benchmark_analysis,
                        available_operators=self._available_search_operators(),
                        pool_source_count=len(self._input_source_names),
                    )
                    self._charge_budget_hours(
                        self._seconds_to_hours(time.perf_counter() - action_start),
                        "llm",
                    )
                    for idx, recipe in enumerate(candidate_pool, start=1):
                        self._name_recipe(
                            recipe,
                            origin_type="candidate",
                            proposal_index=idx,
                        )
                    # Tag parent lineage on each proposed recipe
                    for r in candidate_pool:
                        r._parent_name = parent.recipe.recipe_name  # temporary attr
                    unexplored_pool.extend(candidate_pool)
                    llm_succeeded = True
                    break
                except Exception as exc:
                    logger.error("Action LLM attempt %d failed: %s", _llm_attempt + 1, exc)
            
            if not llm_succeeded:
                logger.warning("All LLM attempts failed. Generating random mutation of parent.")
                # Fallback: generate a simple mutation by re-using LHS sampler
                try:
                    fallback = generate_lhs_seeds(
                        self.catalog_path, n_samples=1, pool_size=self.pool_size,
                        allowed_operators=tuple(self._available_search_operators()),
                    )
                    for idx, fb in enumerate(fallback, start=1):
                        self._name_recipe(
                            fb,
                            origin_type="fallback",
                            proposal_index=idx,
                        )
                        fb._parent_name = parent.recipe.recipe_name
                    unexplored_pool.extend(fallback)
                except Exception as fb_exc:
                    logger.error("Fallback LHS generation also failed: %s", fb_exc)
                
            if not unexplored_pool:
                logger.warning("No unvisited candidates left. Ending search.")
                break

            # 2. Pipeline pre-execution for all ActionLLM proposals
            candidates_to_run = list(unexplored_pool)
            pipeline_results = self._execute_candidates_pipelines(candidates_to_run)
            if self.cumulative_cost >= self.budget_gpu_hours:
                logger.warning("Budget exhausted after pipeline execution; stopping before further evaluation.")
                self._save_search_tree()
                break

            valid_indices = [
                i for i, pr in enumerate(pipeline_results)
                if pr["valid"]
            ]
            if not valid_indices:
                logger.warning("All candidates produced 0 samples or failed. Clearing pool.")
                unexplored_pool.clear()
                continue

            # 3. ANOVARegressor: predict (μ, σ) on the current ActionLLM proposal set
            current_recipes: List[RecipeConfig] = []
            current_pipeline_results: List[Dict[str, Any]] = []

            for i in valid_indices:
                pr = pipeline_results[i]
                current_recipes.append(pr["recipe"])
                current_pipeline_results.append(pr)

            if not current_recipes:
                logger.warning("No valid candidates for UCB. Ending search.")
                break

            mu, sigma = self.surrogate.predict(current_recipes)
            ucb_scores = mu + self.k_exploration * sigma
            ucb_ranking = np.argsort(ucb_scores)[::-1]

            # 4. Candidate selection via SelectionLLM (llm_only main path)
            best_idx = int(ucb_ranking[0])
            if len(current_recipes) > 1:
                candidate_states = [pr.get("final_state", {}) for pr in current_pipeline_results]
                benchmark_comparison = ""
                if parent_state and candidate_states:
                    benchmark_comparison = self._benchmark_suggestor.compare_candidates(
                        parent_state=parent_state or {},
                        candidate_states=candidate_states,
                        candidate_recipes=current_recipes,
                    )
                selection_start = time.perf_counter()
                sel_result = self.selection_llm.select_candidate(
                    candidates=current_recipes,
                    mu=[float(x) for x in mu],
                    sigma=[float(x) for x in sigma],
                    ucb_scores=[float(x) for x in ucb_scores],
                    search_history=self._render_search_history(),
                    experiment_insights=experiment_insights,
                    best_score=best.score,
                    best_name=best.recipe.recipe_name,
                    budget_remaining=self.budget_gpu_hours - self.cumulative_cost,
                    budget_total=self.budget_gpu_hours,
                    n_iterations=len(self.history),
                    pool_size=self.pool_size,
                    selection_context=self._build_selection_context(parent),
                    candidate_states=candidate_states,
                    benchmark_comparison=benchmark_comparison,
                )
                self._charge_budget_hours(
                    self._seconds_to_hours(time.perf_counter() - selection_start),
                    "llm",
                )

                llm_ranking_local = sel_result.ranking  # 0-based within current proposal set
                winner_local = llm_ranking_local[0] if llm_ranking_local else 0
                if 0 <= winner_local < len(current_recipes):
                    best_idx = winner_local
                logger.info(
                    "SelectionLLM (llm_only): chose candidate %d (%s, confidence=%s)",
                    winner_local + 1, sel_result.rationale, sel_result.confidence,
                )

            selected_recipe = current_recipes[best_idx]
            selected_mu = float(mu[best_idx])
            selected_sigma = float(sigma[best_idx])

            if self.cumulative_cost >= self.budget_gpu_hours:
                logger.warning("Budget exhausted before candidate evaluation. Ending search.")
                self._save_search_tree()
                break

            logger.info(
                "Selected: '%s' (μ=%.2f, σ=%.2f, UCB=%.2f, eval=FULL)",
                selected_recipe.recipe_name, selected_mu, selected_sigma,
                float(ucb_scores[best_idx]),
            )

            # 5. Execute decision on the selected fresh proposal
            for ui, ur in enumerate(unexplored_pool):
                if ur is selected_recipe:
                    unexplored_pool.pop(ui)
                    break

            selected_pipeline_result = current_pipeline_results[best_idx]

            parent_name = getattr(selected_recipe, '_parent_name', parent.recipe.recipe_name)
            candidate = self._evaluate_candidate(
                recipe=selected_recipe,
                pipeline_result=selected_pipeline_result,
                parent_name=parent_name,
                predicted_utility=selected_mu,
            )
            
            # Tag trajectory membership
            candidate.trajectory_id = self._current_trajectory_id

            # 6. Posterior update + persist
            self._update_surrogate()
            self._save_search_tree()
            
            # Update empirical best + global stagnation tracking
            if candidate.score > best.score:
                logger.info("★ New best! %s → score %.2f", candidate.recipe.recipe_name, candidate.score)
                best = candidate
            else:
                logger.info("No improvement (%.2f ≤ %.2f). Continuing.", candidate.score, best.score)

            if candidate.score > self._global_best_score:
                self._global_best_score = candidate.score
                self._stagnation_count = 0
            else:
                self._stagnation_count += 1
                logger.info(
                    "Global stagnation: %d/%d (best=%.2f)",
                    self._stagnation_count,
                    self.stagnation_patience,
                    self._global_best_score,
                )

            # Check for trajectory restart
            if self._stagnation_count >= self.stagnation_patience:
                if self.cumulative_cost < self.budget_gpu_hours:
                    self._start_new_trajectory()
                    # Evaluate restart seed immediately
                    restart_seed = self._generate_restart_seed()
                    logger.info("Restart seed: '%s'", restart_seed.recipe_name)
                    restart_results = self._execute_candidates_pipelines([restart_seed])
                    restart_pr = restart_results[0]
                    if restart_pr["valid"] and self.cumulative_cost < self.budget_gpu_hours:
                        restart_candidate = self._evaluate_candidate(
                            recipe=restart_seed,
                            pipeline_result=restart_pr,
                            parent_name="",
                            predicted_utility=0.0,
                        )
                        restart_candidate.trajectory_id = self._current_trajectory_id
                        if restart_candidate.score > self._global_best_score:
                            self._global_best_score = restart_candidate.score
                            self._stagnation_count = 0
                        if restart_candidate.score > best.score:
                            best = restart_candidate
                            logger.info("★ Restart seed is new best! score=%.2f", best.score)
                        self._save_search_tree()
                    else:
                        logger.warning("Restart seed '%s' invalid or budget exhausted.", restart_seed.recipe_name)
                
        logger.info(
            "Search Exhausted Budget. Best: %s (score=%.2f, cost=%.2fh)",
            best.recipe.recipe_name, best.score, self.cumulative_cost
        )
        return best
