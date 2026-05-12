"""Selection LLM — Strategic candidate selection via reasoning model.

This module adds an LLM-based ranking step on the current ActionLLM proposal set.
After the Gaussian Process surrogate computes scores for the current candidates,
the SelectionLLM reviews that same proposal set with full search context and
uses chain-of-thought reasoning to rank the most promising choices.

Design rationale:
  - GP+UCB provides a solid quantitative prior but is limited by its encoding
    recipe-level features. It cannot capture operator synergies, feedback
    patterns, or strategic considerations.
  - The reasoning model adds qualitative judgement: operator compatibility,
    risk mitigation, exploration/exploitation balance, and alignment with
    historical experiment insights.
  - The active workflow presents the full ActionLLM proposal set to the LLM
    instead of letting GP-based truncation decide candidate visibility.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from recipe_sandbox.agents.base import LLMClient, ReasoningResponse
from recipe_sandbox.pipeline.task_config import LLMConfig, RecipeConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from recipe_sandbox.agents.thinking_logger import ThinkingLogger

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of the SelectionLLM's candidate ranking."""
    selected_index: int          # 0-based index into the presented candidate list
    confidence: str              # "high", "medium", "low"
    rationale: str               # One-sentence explanation
    thinking: str                # Full chain-of-thought reasoning (for logging)
    raw_response: str            # Raw LLM response
    ranking: List[int] = None    # type: ignore[assignment]  # Full ranking, best-first (0-based)

    def __post_init__(self) -> None:
        if self.ranking is None:
            self.ranking = [self.selected_index]


class SelectionLLM:
    """Uses a reasoning model to strategically select the best candidate."""

    def __init__(
        self,
        llm_config: LLMConfig,
        *,
        temperature: float = 0.6,
        thinking_logger: Optional["ThinkingLogger"] = None,
    ):
        """
        Args:
            llm_config: LLM API configuration (should point to thinking model).
            temperature: Sampling temperature for the reasoning model.
            thinking_logger: Optional logger for capturing reasoning traces.
        """
        self.client = LLMClient(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            model=llm_config.model,
        )
        self.temperature = temperature
        self.thinking_logger = thinking_logger
        self._iteration: int = 0

    def select_candidate(
        self,
        candidates: List[RecipeConfig],
        mu: List[float],
        sigma: List[float],
        ucb_scores: List[float],
        *,
        search_history: str = "",
        experiment_insights: str = "",
        best_score: float = 0.0,
        best_name: str = "",
        budget_remaining: float = 0.0,
        budget_total: float = 0.0,
        n_iterations: int = 0,
        pool_size: int = 0,
        selection_context: Optional[Dict[str, Any]] = None,
        candidate_states: Optional[List[Dict[str, float]]] = None,
        benchmark_comparison: str = "",
    ) -> SelectionResult:
        """Rank the current ActionLLM proposal set and select the best candidate.

        Args:
            candidates: Candidate recipes from the current ActionLLM proposal set.
            mu: GP predicted mean scores for each candidate.
            sigma: GP predicted uncertainties for each candidate.
            ucb_scores: UCB values (mu + k*sigma) for each candidate.
            search_history: Formatted search history string.
            experiment_insights: FeedbackLLM qualitative insights.
            best_score: Current best verified score (%).
            best_name: Name of the current best recipe.
            budget_remaining: Remaining full-evaluation budget.
            budget_total: Total full-evaluation budget.
            n_iterations: Number of completed iterations.
            pool_size: Total training data pool size.
            selection_context: Rich structured context dict.
            candidate_states: List of state vectors (dicts) for each candidate
                after pipeline execution. Used for richer decision-making.

        Returns:
            SelectionResult with chosen candidate index and reasoning.
        """
        n_present = len(candidates)
        if n_present == 0:
            return SelectionResult(
                selected_index=0, confidence="low",
                rationale="No candidates available",
                thinking="", raw_response="",
            )

        # Build the prompt
        prompt = self._build_prompt(
            candidates=candidates,
            mu=mu,
            sigma=sigma,
            ucb_scores=ucb_scores,
            search_history=search_history,
            experiment_insights=experiment_insights,
            best_score=best_score,
            best_name=best_name,
            budget_remaining=budget_remaining,
            budget_total=budget_total,
            n_iterations=n_iterations,
            pool_size=pool_size,
            selection_context=selection_context,
            candidate_states=(candidate_states or []),
            benchmark_comparison=benchmark_comparison,
        )

        try:
            resp: ReasoningResponse = self.client.chat_with_reasoning(
                prompt, temperature=self.temperature,
            )

            result = self._parse_response(resp, n_present)
            logger.info(
                "SelectionLLM chose candidate %d/%d (confidence=%s): %s",
                result.selected_index + 1, n_present,
                result.confidence, result.rationale,
            )
            if self.thinking_logger:
                self.thinking_logger.log(
                    "selection", self._iteration, result.thinking, resp.answer,
                    prompt_summary=f"select from {n_present} candidates",
                    extra={
                        "selected_index": result.selected_index,
                        "confidence": result.confidence,
                    },
                )
            if result.thinking:
                logger.debug("SelectionLLM thinking:\n%s", result.thinking)

            return result

        except Exception as e:
            logger.error("SelectionLLM failed: %s. Falling back to UCB rank 1.", e)
            return SelectionResult(
                selected_index=0, confidence="low",
                rationale=f"LLM fallback due to error: {e}",
                thinking="", raw_response="",
            )

    def _build_prompt(
        self,
        candidates: List[RecipeConfig],
        mu: List[float],
        sigma: List[float],
        ucb_scores: List[float],
        search_history: str,
        experiment_insights: str,
        best_score: float,
        best_name: str,
        budget_remaining: float,
        budget_total: float,
        n_iterations: int,
        pool_size: int,
        selection_context: Optional[Dict[str, Any]] = None,
        candidate_states: Optional[List[Dict[str, float]]] = None,
        benchmark_comparison: str = "",
    ) -> str:
        """Construct the selection prompt with full context."""

        budget_pct = (budget_remaining / budget_total * 100) if budget_total > 0 else 0
        phase = "early exploration (prioritize diversity)" if n_iterations < 6 else \
                "mid search (balance exploration and exploitation)" if n_iterations < 12 else \
                "late refinement (prioritize exploitation near best)"

        # Build candidate table
        candidate_rows: List[str] = []
        _states = candidate_states or []
        gp_order = sorted(range(len(ucb_scores)), key=lambda idx: ucb_scores[idx], reverse=True)
        gp_rank_by_idx = {idx: rank + 1 for rank, idx in enumerate(gp_order)}
        for i, (recipe, m, s, u) in enumerate(zip(candidates, mu, sigma, ucb_scores)):
            ops_desc = self._render_recipe_ops(recipe)
            row = (
                f"### Candidate {i} (GP Rank {gp_rank_by_idx.get(i, i + 1)})\n"
                f"  Name: {recipe.recipe_name}\n"
                f"  Operators: {ops_desc}\n"
                f"  GP Predicted Utility (μ): {m:.4f}\n"
                f"  GP Uncertainty (σ): {s:.4f}\n"
                f"  UCB Score (μ + κ·σ): {u:.4f}\n"
            )
            # Append data state metrics if available (from pre-executed pipeline)
            if i < len(_states) and _states[i]:
                sv = _states[i]
                from recipe_sandbox.feedback.state_registry import get_active_keys
                state_keys = get_active_keys()
                state_parts = [f"{k}={sv[k]:.4f}" for k in state_keys if k in sv]
                if state_parts:
                    row += f"  Data State (post-pipeline): {', '.join(state_parts)}\n"
                # Per-task MONA scores
                spt = sv.get("score_per_task")
                if isinstance(spt, dict) and spt:
                    spt_str = ", ".join(f"{t}={v:.4f}" for t, v in sorted(spt.items()))
                    row += f"  MONA Per-Task: {spt_str}\n"
            candidate_rows.append(row)
        candidates_text = "\n".join(candidate_rows)

        # Insights section
        insights_section = ""
        if experiment_insights:
            insights_section = f"""
## EXPERIMENT INSIGHTS (patterns discovered from history)
{experiment_insights}
"""

        # History section (compact text)
        history_section = ""
        if search_history:
            history_section = f"""
## SEARCH HISTORY (recent iterations, compact)
{search_history}
"""

        # Rich per-benchmark + state vector history
        detailed_history_section = ""
        if selection_context and selection_context.get("history_entries"):
            entries = selection_context["history_entries"]
            # Collect all benchmark names dynamically from history
            all_benchmarks: List[str] = []
            for e in entries:
                for bm in (e.get("per_benchmark") or {}):
                    if bm not in all_benchmarks:
                        all_benchmarks.append(bm)
            if not all_benchmarks:
                all_benchmarks = ["gpqa", "gsm8k"]  # fallback
            bm_headers = " | ".join(bm.upper() for bm in all_benchmarks)
            bm_sep = " | ".join("------" for _ in all_benchmarks)
            rows = [f"| Iter | Recipe | Score | {bm_headers} | Samples | retain | drift | mona_mean | mona_per_task |"]
            rows.append(f"|------|--------|-------|{bm_sep}|---------|--------|-------|-----------|---------------|")
            for e in entries:
                pbm = e.get("per_benchmark", {})
                bm_vals = " | ".join(
                    f"{pbm.get(bm, 0):.1f}%" if pbm.get(bm) else "N/A"
                    for bm in all_benchmarks
                )
                sv = e.get("state", {}) or {}
                retain = f"{sv.get('retain_ratio', 0):.3f}" if sv else "N/A"
                drift = f"{sv.get('distribution_drift', 0):.3f}" if sv else "N/A"
                sc_mean = f"{sv.get('score_mean', 0):.3f}" if sv else "N/A"
                spt = sv.get("score_per_task", {}) if sv else {}
                spt_str = ", ".join(f"{t}={v:.3f}" for t, v in sorted(spt.items())) if spt else "N/A"
                rows.append(
                    f"| {e.get('iteration', '?')} | {e['name'][:30]} | {e['score']:.2f}% "
                    f"| {bm_vals} | {e.get('samples', '?')} "
                    f"| {retain} | {drift} | {sc_mean} | {spt_str} |"
                )
            detailed_history_section = f"""
## DETAILED EXPERIMENT HISTORY (per-benchmark + data state metrics)
{chr(10).join(rows)}

State metric definitions & interpretation guide:
- retain_ratio: fraction of data retained after filtering (1.0 = no filtering). Too low (< 0.05) risks catastrophic data loss.
- token_ratio: fraction of tokens retained. Reflects how filtering affects the total training signal.
- distribution_drift: SAE feature activation drift from original data. Higher → more distributional shift (risky if > 0.3).
- score_mean: average MONA quality score of retained samples (aggregate across benchmarks). Higher → data more relevant to eval tasks overall.
- score_per_task: per-benchmark MONA similarity (e.g. gpqa=0.39, gsm8k=0.28). Use this to evaluate trade-offs between benchmarks — a recipe that boosts one but hurts another may need re-balancing.
- score_std: standard deviation of MONA scores. Higher → heterogeneous relevance (some samples very relevant, others not).
- mean_varentropy: normalized token-level varentropy — data complexity/reasoning difficulty. 0.5 = same as original, >0.5 = more complex.
- mean_ifd: normalized Instruction Following Difficulty. 0.5 = same as original, >0.5 = harder instructions.

Operator reference (available filtering methods):
- mona_filter(fraction): Task-relevance filter using MONA similarity. Higher score_mean → more relevant data. Lower fraction → higher purity but fewer samples. Effective at 5-30% retention.
- semdedup(num_clusters, cosine_threshold): Semantic deduplication. Removes near-duplicate samples. Lower cosine_threshold → more aggressive. Primarily reduces distribution_drift.
- ifd_filter(fraction): Instruction-Following Difficulty filter. Keeps challenging, information-dense samples. Very aggressive: 5-20% retention typical.
- ngram_entropy(fraction): Lexical diversity filter. Keeps high-entropy (vocabulary-rich) samples. Safe complementary filter, 20-80% retention.
- action_object_branching(fraction): Dependency tree complexity filter. ⚠️ Historical data shows negative signal — use with extreme caution.
"""
            # Per-benchmark trend summary
            benchmark_trends = self._compute_benchmark_trends(entries)
            if benchmark_trends:
                detailed_history_section += f"\n{benchmark_trends}\n"

        # Parent context
        parent_section = ""
        if selection_context and selection_context.get("parent"):
            p = selection_context["parent"]
            p_ops = ""
            parent_state_str = ""
            pbm = p.get("per_benchmark", {})
            pbm_str = ", ".join(f"{k}={v:.1f}%" for k, v in pbm.items()) if pbm else "N/A"
            sv = p.get("state_vector", {}) or {}
            if sv:
                from recipe_sandbox.feedback.state_registry import get_active_keys as _gak
                key_metrics = _gak()
                parent_state_str = ", ".join(
                    f"{k}={sv.get(k, 0):.4f}" for k in key_metrics if k in sv
                )
            parent_section = f"""
## EXPANSION PARENT (candidates are mutations of this recipe)
- Name: {p['name']}
- Score: {p['score']:.2f}% (per benchmark: {pbm_str})
- Output samples: {p.get('samples', '?')}
- State vector: {parent_state_str if parent_state_str else 'N/A'}
"""

        # Benchmark comparison from BenchmarkSuggestor
        benchmark_comp_section = ""
        if benchmark_comparison:
            benchmark_comp_section = f"""
## BENCHMARK DIAGNOSTIC COMPARISON
{benchmark_comparison}
"""

        prompt = f"""You are a strategic advisor for an automated data selection search system. Your task is to select the SINGLE most promising candidate recipe for real evaluation.

Real evaluation is expensive (requires full model training + benchmark evaluation). Your selection directly impacts the search efficiency.

## SEARCH STATE
- Total data pool: {pool_size:,} samples
- Iterations completed: {n_iterations}
- Evaluation budget remaining: {budget_remaining:.0f} / {budget_total:.0f} full evals ({budget_pct:.0f}%)
- Current best score: {best_score:.2f}% (recipe: {best_name})
- Search phase: {phase}
{parent_section}
{detailed_history_section}
{history_section}
{insights_section}
## CANDIDATES (current ActionLLM proposal set with GP scores)
NOTE: The GP surrogate predicts expected utility from a recipe encoding (operator presence + parameters).
The candidates below are the current ActionLLM proposals; GP scores are advisory signals, not a visibility filter.
Each candidate's pipeline has been pre-executed to obtain data state metrics (shown below for reference).
UCB = μ + κ·σ balances expected performance with exploration value.

{candidates_text}
{benchmark_comp_section}

## SELECTION CRITERIA
Consider these factors carefully:

1. **Per-Task MONA Scores (PRIMARY SIGNAL)**:
   - `score_per_task` shows how relevant the filtered data is to EACH benchmark (e.g., bbh=0.43, gpqa=0.40, gsm8k=0.39).
   - A candidate whose per-task MONA scores improve across multiple benchmarks is a strong positive signal, even if retain_ratio drops.
   - Compare each candidate's score_per_task against the parent's — look for improvements on weak benchmarks.
   - If a candidate improves some benchmarks but hurts others, weigh the magnitude and importance of each.
   - score_mean is the aggregate; score_per_task is the breakdown. Always prioritize the per-task view for decision-making.

2. **Exploration vs Exploitation Trade-off**:
   - Early search: prefer high σ (uncertain, novel) candidates to gather information.
   - Late search: prefer high μ (confident, high-performing) candidates to refine the best.
   - Current phase: {phase}.

3. **Data Quantity Risk**:
   - Recipes that aggressively filter data (many filtering steps or low fractions) risk producing too few samples.
   - Historical evidence shows extreme filtering (< 5K samples from {pool_size:,}) often fails catastrophically.
   - Union operators can recover data volume — these are safer exploration choices.
   - Refer to the per-benchmark history to see how sample count correlates with each benchmark.

4. **Operator Synergies & Redundancy**:
   - Multiple filtering operators in sequence compound data loss multiplicatively.
   - Operators from the same family (e.g., two dedup methods) are often redundant.
   - Complementary operators (e.g., quality filter + diversity sampling) tend to work well together.

5. **Feedback Alignment**:
   - Does this candidate address the patterns identified in experiment insights?
   - Does it avoid strategies that have been shown to fail?

6. **State Vector Patterns**:
   - High retain_ratio (> 0.5) with good score_mean tends to perform well.
   - High distribution_drift (> 0.3) indicates risky distributional shift.
   - The parent's state vector shows the data profile that candidates will modify.

7. **GP Model Limitations**:
   - The GP has only {n_iterations} training points — predictions carry significant uncertainty.
   - Don't blindly trust UCB rankings, especially when scores are close.
   - Your qualitative reasoning about operator interactions can add value beyond the GP.

## OUTPUT FORMAT
After thorough reasoning, output a **full ranking** of all presented candidates as a JSON object.
The ranking list must contain ALL candidate indices (0-based) sorted from most promising to least:

```json
{{
  "ranking": [<best_idx>, <2nd_idx>, ..., <worst_idx>],
  "confidence": "<high|medium|low>",
  "rationale": "<one-sentence explanation of why your top choice was chosen>"
}}
```

Think carefully before answering. Consider each candidate's strengths and risks."""

        return prompt

    @staticmethod
    def _compute_benchmark_trends(entries: List[Dict[str, Any]]) -> str:
        """Compute per-benchmark trend summary from history entries."""
        # Collect per-benchmark score trajectories
        bm_series: Dict[str, List[float]] = {}
        for e in entries:
            pbm = e.get("per_benchmark", {})
            for bm_name, val in pbm.items():
                if isinstance(val, (int, float)) and val > 0:
                    bm_series.setdefault(bm_name, []).append(float(val))
        if not bm_series:
            return ""

        lines = ["## BENCHMARK TRENDS"]
        for bm_name, vals in sorted(bm_series.items()):
            if len(vals) < 2:
                lines.append(f"- {bm_name}: {vals[0]:.1f}% (single data point)")
                continue
            trajectory = " → ".join(f"{v:.1f}%" for v in vals)
            overall = "↑" if vals[-1] > vals[0] else ("↓" if vals[-1] < vals[0] else "→")
            last_dir = "↑" if vals[-1] > vals[-2] else ("↓" if vals[-1] < vals[-2] else "→")
            lines.append(f"- {bm_name}: {trajectory}  (overall {overall}, last {last_dir})")
        return "\n".join(lines)

    def _render_recipe_ops(self, recipe: RecipeConfig) -> str:
        """Render recipe operators into a compact readable string."""
        if not recipe.steps:
            return "(empty — use full dataset)"
        parts: List[str] = []
        for step in recipe.steps:
            if not step.enabled:
                continue
            params_str = ", ".join(f"{k}={v}" for k, v in (step.params or {}).items())
            parts.append(f"{step.operator}({params_str})" if params_str else step.operator)
        return " → ".join(parts) if parts else "(empty — use full dataset)"

    def _parse_response(self, resp: ReasoningResponse, n_candidates: int) -> SelectionResult:
        """Parse the LLM response to extract the candidate ranking."""
        answer = resp.answer

        selected_index = 0
        confidence = "medium"
        rationale = ""
        ranking: List[int] = []

        # Try to find a JSON block containing "ranking" (new format)
        json_match = re.search(r"\{[^{}]*\"ranking\"[^{}]*\}", answer, re.DOTALL)
        if not json_match:
            # Fallback: try old "selected_index" format
            json_match = re.search(r"\{[^{}]*\"selected_index\"[^{}]*\}", answer, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group(0))

                # Extract ranking list (new format)
                raw_ranking = parsed.get("ranking")
                if isinstance(raw_ranking, list):
                    ranking = [
                        int(x) for x in raw_ranking
                        if isinstance(x, (int, float)) and 0 <= int(x) < n_candidates
                    ]

                # Backward compat: extract selected_index if no ranking
                if not ranking:
                    idx = parsed.get("selected_index", 0)
                    if isinstance(idx, int) and 0 <= idx < n_candidates:
                        selected_index = idx

                confidence = parsed.get("confidence", "medium")
                rationale = parsed.get("rationale", "")
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse SelectionLLM JSON, using fallback.")
        else:
            # Fallback: look for a number pattern like "Candidate 2" or "index: 2"
            num_match = re.search(r"(?:candidate|index|选择)\s*[:：]?\s*(\d+)", answer, re.IGNORECASE)
            if num_match:
                idx = int(num_match.group(1))
                if 0 <= idx < n_candidates:
                    selected_index = idx
            rationale = answer[:200] if answer else "No structured response"

        # Derive selected_index from ranking if available
        if ranking:
            selected_index = ranking[0]
        else:
            # No ranking from LLM — seed with selected_index as best
            ranking = [selected_index]

        # Ensure ranking covers all candidates.
        seen = set(ranking)
        for i in range(n_candidates):
            if i not in seen:
                ranking.append(i)
                seen.add(i)

        return SelectionResult(
            selected_index=selected_index,
            confidence=confidence,
            rationale=rationale,
            thinking=resp.thinking,
            raw_response=resp.raw,
            ranking=ranking,
        )
