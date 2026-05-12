"""ThinkingLogger — Persistent JSONL logger for LLM reasoning traces.

Records the full thinking/reasoning process from all LLM agents
(Action, Feedback, Selection) to enable post-hoc experiment analysis.

Output: ``<output_dir>/thinking_log.jsonl`` with one JSON object per LLM call.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ThinkingLogger:
    """Thread-safe JSONL logger for LLM thinking traces."""

    def __init__(self, log_dir: str) -> None:
        self._log_path = os.path.join(log_dir, "thinking_log.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        self._lock = threading.Lock()
        logger.info("ThinkingLogger → %s", self._log_path)

    def log(
        self,
        agent: str,
        iteration: int,
        thinking: str,
        answer: str,
        *,
        prompt_summary: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one thinking record to the JSONL log.

        Parameters
        ----------
        agent : str
            Agent name, e.g. ``"selection"``, ``"action"``, ``"feedback"``.
        iteration : int
            Current MCTS search iteration.
        thinking : str
            The model's chain-of-thought / reasoning content.
        answer : str
            The model's final answer (after reasoning).
        prompt_summary : str, optional
            A brief summary of what was asked (for context).
        extra : dict, optional
            Any additional metadata (confidence, selected_index, etc.).
        """
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent": agent,
            "iteration": iteration,
            "thinking": thinking,
            "answer": answer,
            "prompt_summary": prompt_summary,
        }
        if extra:
            entry["extra"] = extra

        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @property
    def path(self) -> str:
        return self._log_path
