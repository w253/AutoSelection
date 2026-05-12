"""Global State Vector Key Registry.

Single source of truth for all state vector dimension metadata.
LLM prompts and benchmark_suggest read from this registry so adding/removing a
dimension only requires changes here + state_vector.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StateKeyMeta:
    """Metadata for one state vector dimension."""

    key: str
    display_name: str
    description: str       # English description injected into LLM prompts
    range_hint: str        # e.g. "[0, 1]"
    direction: str         # "higher_better" | "lower_better" | "neutral"


# Ordered registry — iteration order = canonical dimension order
STATE_KEY_REGISTRY: Dict[str, StateKeyMeta] = {}


def _register(meta: StateKeyMeta) -> None:
    STATE_KEY_REGISTRY[meta.key] = meta


# -----------------------------------------------------------------------
#  Register all active state keys (order matters)
# -----------------------------------------------------------------------

_register(StateKeyMeta(
    key="retain_ratio",
    display_name="Data Retain Ratio",
    description=(
        "Fraction of original samples retained after filtering "
        "(1.0 = all kept, 0.0 = all removed). "
        "Very low values indicate overly aggressive filtering."
    ),
    range_hint="[0, 1]",
    direction="neutral",
))

_register(StateKeyMeta(
    key="token_ratio",
    display_name="Token Retain Ratio",
    description=(
        "Fraction of original tokens retained — reflects training signal volume. "
        "Can diverge from retain_ratio when filtering removes long/short samples selectively."
    ),
    range_hint="[0, 1]",
    direction="neutral",
))

_register(StateKeyMeta(
    key="distribution_drift",
    display_name="Distribution Drift",
    description=(
        "SAE feature activation pattern drift from original data "
        "(L2 norm of SNAR delta, normalized by sqrt(D)). "
        "0 = identical distribution, higher = more shifted. "
        "Values > 0.3 suggest significant distributional change."
    ),
    range_hint="[0, inf), typically [0, 0.5]",
    direction="lower_better",
))

_register(StateKeyMeta(
    key="score_mean",
    display_name="MONA Score Mean",
    description=(
        "Mean task-relevance score (MONA Jaccard similarity to benchmark targets) "
        "across all samples. Higher = data more relevant to target benchmarks. "
        "Also captures semantic coverage — if coverage drops, score_mean drops."
    ),
    range_hint="[0, 1]",
    direction="higher_better",
))

_register(StateKeyMeta(
    key="score_std",
    display_name="MONA Score Std",
    description=(
        "Standard deviation of MONA scores across samples. "
        "High = heterogeneous relevance (mix of very relevant and irrelevant data), "
        "Low = uniform quality."
    ),
    range_hint="[0, 1]",
    direction="neutral",
))

_register(StateKeyMeta(
    key="mean_varentropy",
    display_name="Mean Varentropy",
    description=(
        "Normalized mean token-level varentropy (LLM output logits entropy variance). "
        "Measures data complexity and reasoning difficulty. "
        "0.5 = same complexity as original data, "
        ">0.5 = filtered data is more complex, "
        "<0.5 = simpler/easier samples remain."
    ),
    range_hint="[0, 1]",
    direction="neutral",
))

_register(StateKeyMeta(
    key="mean_ifd",
    display_name="Mean IFD",
    description=(
        "Normalized mean Instruction Following Difficulty "
        "(ratio of full-text loss to response-only loss). "
        "Measures how challenging the instructions are to follow. "
        "0.5 = same as original, >0.5 = harder instructions, <0.5 = easier."
    ),
    range_hint="[0, 1]",
    direction="neutral",
))

_register(StateKeyMeta(
    key="cumulative_cost_ratio",
    display_name="Eval Budget Used",
    description="Fraction of total evaluation budget consumed so far.",
    range_hint="[0, 1]",
    direction="lower_better",
))


# -----------------------------------------------------------------------
#  Utility functions
# -----------------------------------------------------------------------

def get_active_keys() -> List[str]:
    """Return ordered list of currently active state key names."""
    return list(STATE_KEY_REGISTRY.keys())


def get_prompt_descriptions() -> str:
    """Generate LLM-prompt-ready description block for all active dimensions."""
    lines = []
    for meta in STATE_KEY_REGISTRY.values():
        lines.append(f"- {meta.key}: {meta.description}")
    return "\n".join(lines)


def get_key_meta(key: str) -> Optional[StateKeyMeta]:
    """Look up metadata for a single state key."""
    return STATE_KEY_REGISTRY.get(key)
