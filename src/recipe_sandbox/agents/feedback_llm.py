"""Feedback LLM — Summarizes qualitative patterns from experiment history.

This module analyzes the search history and produces concise qualitative
findings that are injected into the Action LLM prompt. This helps the
Action LLM avoid repeating failed strategies and build on successful patterns.

Called every N MCTS iterations to balance insight freshness vs API cost.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from recipe_sandbox.agents.base import LLMClient
from recipe_sandbox.pipeline.task_config import LLMConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from recipe_sandbox.agents.thinking_logger import ThinkingLogger

logger = logging.getLogger(__name__)


class FeedbackLLM:
    """Summarizes experiment history into qualitative insights for Action LLM."""

    def __init__(
        self,
        llm_config: LLMConfig,
        *,
        call_interval: int = 3,
        temperature: float = 0.3,
        thinking_logger: Optional["ThinkingLogger"] = None,
    ):
        """
        Args:
            llm_config: LLM API configuration.
            call_interval: Only regenerate insights every N calls (cache between).
            temperature: Low temperature for analytical, consistent output.
            thinking_logger: Optional logger for capturing reasoning traces.
        """
        self.client = LLMClient(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            model=llm_config.model,
        )
        self.call_interval = call_interval
        self.temperature = temperature
        self.thinking_logger = thinking_logger
        self._cached_insights: str = ""
        self._call_count: int = 0
        self._last_history_len: int = 0
        self._iteration: int = 0

    def summarize_patterns(
        self,
        history: List[Dict[str, Any]],
        force: bool = False,
    ) -> str:
        """Analyze experiment history and produce qualitative insights.

        Args:
            history: List of dicts, each with keys:
                - recipe_name (str)
                - operators (list of str)
                - params (dict of operator -> params)
                - score (float, aggregated %)
                - per_benchmark (dict of benchmark -> score)
                - output_samples (int)
                - state_vector (dict or None)
                - eval_mode (str: "full" or "proxy")
            force: If True, regenerate even if interval not reached.

        Returns:
            String of 3-5 qualitative findings, or cached version.
        """
        self._call_count += 1

        # Only regenerate if interval reached or history grew significantly
        if not force and self._cached_insights:
            if self._call_count % self.call_interval != 0:
                if len(history) <= self._last_history_len:
                    return self._cached_insights

        # Need at least 3 data points for meaningful patterns
        verified = [h for h in history if h.get("eval_mode", "full") == "full"]
        if len(verified) < 3:
            return ""

        self._last_history_len = len(history)

        # Build compact history table for LLM
        table_lines = ["| # | Recipe | Operators | Samples | Score | Per-Benchmark |"]
        table_lines.append("|---|--------|-----------|---------|-------|---------------|")
        for i, h in enumerate(verified):
            ops = ", ".join(h.get("operators", []))
            samples = h.get("output_samples", "?")
            score = h.get("score", 0.0)
            per_bench = h.get("per_benchmark", {})
            bench_str = ", ".join(f"{k}={v:.1f}%" for k, v in per_bench.items()) if per_bench else "N/A"
            table_lines.append(
                f"| {i+1} | {h.get('recipe_name', '?')} | {ops} | {samples} | {score:.2f}% | {bench_str} |"
            )
        table = "\n".join(table_lines)

        prompt = f"""You are a data science experiment analyst. Analyze the following experiment history from an automated data selection search system.

The system is searching for the best data filtering recipe to train an LLM. Each row is one experiment where data was filtered differently and a model was trained and evaluated.

{table}

KEY CONTEXT:
- Higher scores are better (aggregated accuracy across benchmarks)
- "Operators" are data filtering/mixing steps applied sequentially
- "Samples" is the number of training samples after filtering
- The pool has {verified[-1].get('output_samples', 90000) * 3 if verified else 90000} total samples

TASK: Produce exactly 3-5 concise, actionable findings. Each finding should be:
1. A specific observation (not vague)
2. Backed by data from the table
3. Actionable (suggests what to try or avoid)

Format each finding as a numbered line. Be direct and quantitative.
These are HYPOTHESES based on limited data, not proven facts.

Example format:
1. More data consistently helps: recipe_A (12K samples, 22.2%) > recipe_C (3K samples, 18.5%). Avoid aggressive filtering.
2. operator_X at rate 0.3 hurts benchmark_Y: recipe_B dropped from 15% to 0.9%. Try higher rates or skip it.
"""

        try:
            resp = self.client.chat_with_reasoning(prompt, temperature=self.temperature)
            response = resp.answer
            if self.thinking_logger:
                self.thinking_logger.log(
                    "feedback", self._iteration, resp.thinking, resp.answer,
                    prompt_summary=f"summarize_patterns: {len(verified)} experiments",
                )
            # Extract just the numbered findings
            lines = response.strip().split("\n")
            findings = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    findings.append(line)

            if findings:
                self._cached_insights = "\n".join(findings)
            else:
                # Fallback: use full response
                self._cached_insights = response.strip()

            logger.info(
                "FeedbackLLM generated %d insights from %d experiments",
                len(findings), len(verified),
            )
        except Exception as e:
            logger.error("FeedbackLLM call failed: %s", e)
            # Keep cached insights on failure
            if not self._cached_insights:
                self._cached_insights = ""

        return self._cached_insights

    def get_cached_insights(self) -> str:
        """Return the most recent cached insights without making an API call."""
        return self._cached_insights
