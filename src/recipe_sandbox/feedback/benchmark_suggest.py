"""Benchmark Diagnostic Suggestor — structured analysis for Action & Selection LLMs.

Provides two main entry-points:
  * ``analyze_for_action()`` — produces a diagnostic text for the Action LLM
    that explains per-benchmark performance deltas, full state-vector changes,
    operator mechanisms and actionable hypotheses.
  * ``compare_candidates()`` — produces a per-candidate comparison text for the
    Selection LLM showing how each candidate's state vector and per-benchmark
    MONA similarities changed relative to the parent.

Design principles:
  1. **No LLM calls** — pure rule-based analysis using built-in knowledge about
     operators and state-vector dimensions.
  2. **Informational, not prescriptive** — provides structured context so the
     LLM can reason about trade-offs.  The LLM makes the final decision.
  3. **Graceful degradation** — returns empty string when insufficient data
     (cold start, missing fields).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
#  Operator Knowledge Base
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ParamEffect:
    """Describes what happens when a parameter increases."""
    increase: str
    decrease: str


@dataclass(frozen=True)
class _OperatorKnowledge:
    mechanism: str
    param_effects: Dict[str, _ParamEffect]
    benchmark_link: str  # "direct" | "indirect"
    state_impact: str    # how this operator typically affects state vector


OPERATOR_KNOWLEDGE: Dict[str, _OperatorKnowledge] = {
    "mona_filter": _OperatorKnowledge(
        mechanism=(
            "Generalized Jaccard similarity to eval target vectors. "
            "Selects top-{fraction} most relevant samples. "
            "Per-benchmark union mode: independently selects top-{fraction} for each "
            "benchmark, then returns the union — so actual retention may exceed {fraction}. "
            "Top-1 is the most similar sample; as fraction grows, more distant (less "
            "relevant) samples are included."
        ),
        param_effects={
            "fraction": _ParamEffect(
                increase="more samples retained, lower avg MONA similarity, higher coverage",
                decrease="fewer but more task-relevant samples, risk of losing diversity",
            ),
        },
        benchmark_link="direct",
        state_impact=(
            "↑fraction → retain_ratio↑, score_mean↓, "
            "score_per_task per-bm↓, distribution_drift may↓ (closer to original)"
        ),
    ),
    "ifd_filter": _OperatorKnowledge(
        mechanism=(
            "Instruction-Following Difficulty filter. Keeps top-{fraction} hardest "
            "samples (highest loss(response|instruction) / loss(response) ratio). "
            "Higher IFD means the response is harder to produce given the instruction."
        ),
        param_effects={
            "fraction": _ParamEffect(
                increase="includes easier samples — may help simple benchmarks (GSM8K arithmetic)",
                decrease="only hardest samples — may help reasoning benchmarks (GPQA) but risks noise",
            ),
        },
        benchmark_link="indirect",
        state_impact=(
            "↑fraction → retain_ratio↑, score_std may↑ (wider quality range), "
            "mean_ifd may↑ (more diverse difficulties)"
        ),
    ),
    "ngram_entropy": _OperatorKnowledge(
        mechanism=(
            "Lexical diversity filter. Keeps top-{fraction} samples with highest "
            "unigram Shannon entropy — i.e. richest vocabulary. "
            "Low-entropy samples have repetitive or formulaic text."
        ),
        param_effects={
            "fraction": _ParamEffect(
                increase="includes lower-diversity samples — may reduce generalization",
                decrease="only most lexically diverse — improves diversity but may lose domain-specific patterns",
            ),
        },
        benchmark_link="indirect",
        state_impact=(
            "↑fraction → retain_ratio↑, mean_varentropy may↑ (less diverse filtering), "
            "distribution_drift relatively stable"
        ),
    ),
    "varentropy_filter": _OperatorKnowledge(
        mechanism=(
            "Reasoning complexity filter. Keeps top-{fraction} samples with highest "
            "varentropy score — indicating more complex reasoning patterns. "
            "Varentropy measures prediction variance/uncertainty during generation."
        ),
        param_effects={
            "fraction": _ParamEffect(
                increase="includes simpler reasoning samples — may help code/factual benchmarks",
                decrease="only most complex reasoning — may help GPQA/BBH but hurt simple tasks",
            ),
        },
        benchmark_link="indirect",
        state_impact=(
            "↑fraction → retain_ratio↑, may shift distribution toward simpler content"
        ),
    ),
    "action_object_branching": _OperatorKnowledge(
        mechanism=(
            "Structural complexity filter based on dependency tree branching. "
            "Keeps top-{fraction} samples with highest branching factor. "
            "Measures instruction structural complexity via SpaCy parse trees."
        ),
        param_effects={
            "fraction": _ParamEffect(
                increase="includes structurally simpler instructions",
                decrease="only most complex structures — similar effect to IFD",
            ),
        },
        benchmark_link="indirect",
        state_impact=(
            "↑fraction → retain_ratio↑, similar to IFD effects on data composition"
        ),
    ),
    "semdedup": _OperatorKnowledge(
        mechanism=(
            "Semantic deduplication via K-means clustering on SAE sparse features. "
            "Within each cluster, removes samples with cosine similarity ≥ threshold. "
            "num_clusters controls granularity; cosine_threshold controls aggressiveness."
        ),
        param_effects={
            "num_clusters": _ParamEffect(
                increase="finer-grained clusters, preserves more intra-cluster diversity",
                decrease="coarser clusters, more aggressive cross-topic dedup",
            ),
            "cosine_threshold": _ParamEffect(
                increase="more aggressive dedup (removes more 'similar' samples)",
                decrease="keeps more similar variants, less dedup",
            ),
        },
        benchmark_link="indirect",
        state_impact=(
            "↑cosine_threshold → distribution_drift may↓, retain_ratio↓, "
            "mean_ifd may change (losing useful variations)"
        ),
    ),
}


# ---------------------------------------------------------------------------
#  State Vector Interpretation Rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StateInterpretation:
    description: str
    good_direction: str  # "higher" | "lower" | "context"
    improve_msg: str = ""


STATE_INTERPRETATIONS: Dict[str, _StateInterpretation] = {
    "retain_ratio": _StateInterpretation(
        description="Fraction of original samples retained (1.0 = all kept)",
        good_direction="higher",
        improve_msg="Relax filter fractions or remove filter steps",
    ),
    "token_ratio": _StateInterpretation(
        description="Fraction of original tokens retained (reflects training signal volume)",
        good_direction="higher",
        improve_msg="Relax filtering to preserve more training tokens",
    ),
    "distribution_drift": _StateInterpretation(
        description="SAE feature activation drift from original data (L2 norm of SNAR delta). 0 = identical",
        good_direction="lower",
        improve_msg="Check if a specific filter is selecting too narrowly",
    ),
    "score_mean": _StateInterpretation(
        description="Mean MONA relevance score (aggregate across benchmarks). Higher = more task-relevant data",
        good_direction="context",
        improve_msg="Balance: very high score_mean may indicate over-filtering to only the most relevant samples",
    ),
    "score_std": _StateInterpretation(
        description="Std dev of MONA scores. High = heterogeneous relevance, Low = uniform quality",
        good_direction="context",
        improve_msg="High std → some very relevant, some not. Low std → consistent but possibly narrow",
    ),
    "mean_varentropy": _StateInterpretation(
        description="Normalized mean token-level varentropy (0.5 = same as original, >0.5 = more complex)",
        good_direction="context",
        improve_msg="Extreme values indicate the filtering is biasing data complexity. 0.5 is neutral.",
    ),
    "mean_ifd": _StateInterpretation(
        description="Normalized mean Instruction Following Difficulty (0.5 = same as original, >0.5 = harder)",
        good_direction="context",
        improve_msg="Extreme values indicate the filtering is biasing instruction difficulty. 0.5 is neutral.",
    ),
    "cumulative_cost_ratio": _StateInterpretation(
        description="Fraction of total evaluation budget consumed so far",
        good_direction="lower",
    ),
}


# ---------------------------------------------------------------------------
#  Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    """Per-benchmark performance comparison."""
    benchmark: str
    current_eval: Optional[float] = None
    best_eval: Optional[float] = None
    eval_delta: float = 0.0
    current_mona_sim: Optional[float] = None
    best_mona_sim: Optional[float] = None
    mona_delta: float = 0.0
    trend: str = "unknown"  # "improving" / "declining" / "stable" / "unknown"


@dataclass
class StateDelta:
    """One dimension's change between two state vectors."""
    metric: str
    current: float
    reference: float
    delta: float
    interpretation: str = ""
    risk: bool = False


# ---------------------------------------------------------------------------
#  BenchmarkSuggestor
# ---------------------------------------------------------------------------

class BenchmarkSuggestor:
    """Rule-based diagnostic analysis engine for MCTS recipe search."""

    # Threshold for "declining" vs "stable" benchmark performance
    DECLINE_THRESHOLD_PP = 2.0  # percentage points
    IMPROVE_THRESHOLD_PP = 1.0

    def analyze_for_action(
        self,
        *,
        parent_state: Dict[str, Any],
        best_state: Dict[str, Any],
        parent_task_scores: Optional[Dict[str, float]] = None,
        best_task_scores: Optional[Dict[str, float]] = None,
        current_recipe_ops: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate structured diagnostic text for the Action LLM.

        Args:
            parent_state: Full state_vector dict of the parent node
                          (includes score_per_task, retain_ratio, etc.)
            best_state: Full state_vector dict of the best-so-far node
            parent_task_scores: Parent's eval_result.task_scores (actual eval %)
            best_task_scores: Best node's eval_result.task_scores
            current_recipe_ops: List of dicts with 'operator' and 'params' keys

        Returns:
            Formatted text ready for prompt injection. Empty string if
            insufficient data.
        """
        if not parent_state and not best_state:
            return ""

        sections: List[str] = []
        sections.append("=== BENCHMARK DIAGNOSTIC ANALYSIS ===\n")

        # Part A: Performance Report
        perf = self._build_performance_report(
            parent_task_scores or {},
            best_task_scores or {},
            parent_state.get("score_per_task", {}),
            best_state.get("score_per_task", {}),
        )
        if perf:
            sections.append(perf)

        # Part B: State Vector Analysis
        sv_analysis = self._build_state_vector_analysis(parent_state, best_state)
        if sv_analysis:
            sections.append(sv_analysis)

        # Part C: Operator Impact + Hypotheses
        if current_recipe_ops:
            impact = self._build_operator_impact(
                current_recipe_ops,
                parent_state.get("score_per_task", {}),
                best_state.get("score_per_task", {}),
                parent_state,
                best_state,
            )
            if impact:
                sections.append(impact)

        if len(sections) <= 1:
            return ""
        return "\n".join(sections)

    def compare_candidates(
        self,
        *,
        parent_state: Dict[str, Any],
        candidate_states: List[Dict[str, Any]],
        candidate_recipes: Optional[List[Any]] = None,
    ) -> str:
        """Generate per-candidate comparison text for the Selection LLM.

        Args:
            parent_state: Parent node's full state_vector dict
            candidate_states: List of state_vector dicts, one per candidate
            candidate_recipes: Optional list of recipe configs (for op descriptions)

        Returns:
            Formatted text with per-candidate delta analysis.
        """
        if not parent_state or not candidate_states:
            return ""

        parent_spt = parent_state.get("score_per_task", {})
        lines: List[str] = ["## Benchmark & State Comparison vs Parent\n"]

        for i, cs in enumerate(candidate_states):
            if not cs:
                continue

            # Recipe description
            recipe_desc = ""
            if candidate_recipes and i < len(candidate_recipes):
                r = candidate_recipes[i]
                if hasattr(r, "steps"):
                    ops = [f"{s.operator}({_format_params(s.params)})" for s in r.steps if s.enabled]
                    recipe_desc = " + ".join(ops) if ops else "no-op"

            lines.append(f"### Candidate {i}" + (f": {recipe_desc}" if recipe_desc else ""))

            # State vector key metric deltas (candidate = current, parent = reference)
            sv_deltas = self._compute_state_deltas(cs, parent_state)
            if sv_deltas:
                lines.append("  State Changes vs Parent:")
                for sd in sv_deltas:
                    symbol = self._delta_symbol(sd.metric, sd.delta)
                    lines.append(
                        f"    {sd.metric}: {sd.reference:.3f} → {sd.current:.3f} "
                        f"({sd.delta:+.3f} {symbol})"
                    )

            # Per-benchmark MONA similarity deltas
            cand_spt = cs.get("score_per_task", {})
            bm_keys = sorted(set(parent_spt) | set(cand_spt))
            if bm_keys:
                lines.append("  Benchmark MONA Similarity Changes:")
                for bm in bm_keys:
                    p_val = parent_spt.get(bm)
                    c_val = cand_spt.get(bm)
                    if p_val is not None and c_val is not None:
                        delta = c_val - p_val
                        symbol = "✓" if delta > 0.005 else ("⚠" if delta < -0.005 else "~")
                        lines.append(
                            f"    {bm}: {p_val:.4f} → {c_val:.4f} ({delta:+.4f} {symbol})"
                        )
                    elif c_val is not None:
                        lines.append(f"    {bm}: N/A → {c_val:.4f}")

            lines.append("")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ------------------------------------------------------------------
    #  Part A: Performance Report
    # ------------------------------------------------------------------

    def _build_performance_report(
        self,
        parent_scores: Dict[str, float],
        best_scores: Dict[str, float],
        parent_spt: Dict[str, float],
        best_spt: Dict[str, float],
    ) -> str:
        benchmarks = sorted(set(parent_scores) | set(best_scores) | set(parent_spt) | set(best_spt))
        if not benchmarks:
            return ""

        reports: List[BenchmarkReport] = []
        for bm in benchmarks:
            cur_eval = parent_scores.get(bm)
            bst_eval = best_scores.get(bm)
            eval_delta = (cur_eval - bst_eval) if (cur_eval is not None and bst_eval is not None) else 0.0
            cur_sim = parent_spt.get(bm)
            bst_sim = best_spt.get(bm)
            mona_delta = (cur_sim - bst_sim) if (cur_sim is not None and bst_sim is not None) else 0.0

            if eval_delta < -self.DECLINE_THRESHOLD_PP:
                trend = "declining"
            elif eval_delta > self.IMPROVE_THRESHOLD_PP:
                trend = "improving"
            elif cur_eval is not None:
                trend = "stable"
            else:
                trend = "unknown"

            reports.append(BenchmarkReport(
                benchmark=bm,
                current_eval=cur_eval, best_eval=bst_eval, eval_delta=eval_delta,
                current_mona_sim=cur_sim, best_mona_sim=bst_sim, mona_delta=mona_delta,
                trend=trend,
            ))

        lines = ["## Performance Report"]
        header = "| Benchmark | Eval(curr) | Eval(best) | Eval Δ | Status | MONA Sim(curr) | MONA Sim(best) | Sim Δ |"
        sep = "|-----------|------------|------------|--------|--------|----------------|----------------|-------|"
        lines.extend([header, sep])

        for r in reports:
            status_icon = {"declining": "⚠ declining", "improving": "✓ improving",
                           "stable": "~ stable", "unknown": "? no eval"}[r.trend]
            e_cur = f"{r.current_eval:.1f}%" if r.current_eval is not None else "N/A"
            e_bst = f"{r.best_eval:.1f}%" if r.best_eval is not None else "N/A"
            e_d = f"{r.eval_delta:+.1f}" if r.current_eval is not None and r.best_eval is not None else "N/A"
            s_cur = f"{r.current_mona_sim:.4f}" if r.current_mona_sim is not None else "N/A"
            s_bst = f"{r.best_mona_sim:.4f}" if r.best_mona_sim is not None else "N/A"
            s_d = f"{r.mona_delta:+.4f}" if r.current_mona_sim is not None and r.best_mona_sim is not None else "N/A"
            lines.append(f"| {r.benchmark} | {e_cur} | {e_bst} | {e_d} | {status_icon} | {s_cur} | {s_bst} | {s_d} |")

        # Summary
        declining = [r for r in reports if r.trend == "declining"]
        improving = [r for r in reports if r.trend == "improving"]
        if declining:
            weak = ", ".join(f"{r.benchmark} ({r.eval_delta:+.1f}pp)" for r in declining)
            lines.append(f"\nWeakest: {weak}")
        if improving:
            strong = ", ".join(f"{r.benchmark} ({r.eval_delta:+.1f}pp)" for r in improving)
            lines.append(f"Strongest: {strong}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Part B: State Vector Analysis
    # ------------------------------------------------------------------

    def _build_state_vector_analysis(
        self,
        parent_state: Dict[str, Any],
        best_state: Dict[str, Any],
    ) -> str:
        deltas = self._compute_state_deltas(parent_state, best_state, include_interpretation=True)
        if not deltas:
            return ""

        lines = ["\n## State Vector Analysis (current vs best)"]
        header = "| Metric | Current | Best | Δ | Interpretation |"
        sep = "|--------|---------|------|---|----------------|"
        lines.extend([header, sep])

        for sd in deltas:
            interp = sd.interpretation
            if sd.risk:
                interp = f"⚠ {interp}"
            elif sd.delta != 0.0:
                symbol = self._delta_symbol(sd.metric, sd.delta)
                if symbol == "✓":
                    interp = f"✓ {interp}"
            lines.append(
                f"| {sd.metric} | {sd.current:.4f} | {sd.reference:.4f} | {sd.delta:+.4f} | {interp} |"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Part C: Operator Impact + Hypotheses
    # ------------------------------------------------------------------

    def _build_operator_impact(
        self,
        recipe_ops: List[Dict[str, Any]],
        parent_spt: Dict[str, float],
        best_spt: Dict[str, float],
        parent_state: Dict[str, Any],
        best_state: Dict[str, Any],
    ) -> str:
        lines = ["\n## Operator Impact Analysis (current recipe)\n"]
        hypotheses: List[str] = []

        for op_dict in recipe_ops:
            op_name = op_dict.get("operator", "")
            params = op_dict.get("params", {})
            knowledge = OPERATOR_KNOWLEDGE.get(op_name)
            if not knowledge:
                continue

            param_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
            lines.append(f"### {op_name} ({param_str})")
            lines.append(f"- Mechanism: {knowledge.mechanism}")

            # Parameter-specific trade-offs
            for pname, pval in params.items():
                pe = knowledge.param_effects.get(pname)
                if pe:
                    lines.append(f"- {pname}={pval}: increase → {pe.increase}; decrease → {pe.decrease}")

            lines.append(f"- State impact pattern: {knowledge.state_impact}")

            # MONA filter specific: per-benchmark similarity analysis
            if op_name == "mona_filter" and parent_spt and best_spt:
                declining_bms = []
                for bm in sorted(set(parent_spt) & set(best_spt)):
                    delta = parent_spt[bm] - best_spt[bm]
                    if delta < -0.01:
                        declining_bms.append((bm, delta))
                        lines.append(
                            f"- {bm} MONA sim dropped {delta:+.4f} vs best → "
                            f"current fraction may be too aggressive for {bm}"
                        )
                if declining_bms and "fraction" in params:
                    frac = params["fraction"]
                    bm_list = ", ".join(bm for bm, _ in declining_bms)
                    hypotheses.append(
                        f"MONA filter fraction={frac}: {bm_list} MONA similarity declined. "
                        f"Options: (a) increase fraction to retain more relevant samples for these benchmarks, "
                        f"(b) check if other filters are removing samples that MONA would have kept."
                    )

            # Retain ratio risk
            cur_retain = parent_state.get("retain_ratio", 1.0)
            best_retain = best_state.get("retain_ratio", 1.0)
            if cur_retain < best_retain * 0.6 and op_name in ("ifd_filter", "ngram_entropy",
                                                                "varentropy_filter", "action_object_branching"):
                frac = params.get("fraction", "?")
                hypotheses.append(
                    f"{op_name}(fraction={frac}): retain_ratio={cur_retain:.3f} is much lower than "
                    f"best ({best_retain:.3f}). This operator may be filtering too aggressively. "
                    f"Consider increasing fraction or removing this step."
                )

            lines.append("")

        if hypotheses:
            lines.append("## Key Hypotheses (for LLM to evaluate)\n")
            for i, h in enumerate(hypotheses, 1):
                lines.append(f"{i}. {h}")

        return "\n".join(lines) if len(lines) > 2 else ""

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _compute_state_deltas(
        self,
        current: Dict[str, Any],
        reference: Dict[str, Any],
        include_interpretation: bool = False,
    ) -> List[StateDelta]:
        """Compare two state dicts on key scalar dimensions."""
        key_metrics = [
            "retain_ratio", "token_ratio",
            "distribution_drift", "score_mean", "score_std",
            "mean_varentropy", "mean_ifd",
        ]
        deltas: List[StateDelta] = []
        for m in key_metrics:
            c_val = current.get(m)
            r_val = reference.get(m)
            if c_val is None or r_val is None:
                continue
            if not isinstance(c_val, (int, float)) or not isinstance(r_val, (int, float)):
                continue
            d = float(c_val) - float(r_val)
            interp = ""

            if include_interpretation:
                si = STATE_INTERPRETATIONS.get(m)
                if si:
                    interp = self._interpret_delta(m, float(c_val), d, si)

            deltas.append(StateDelta(
                metric=m, current=float(c_val), reference=float(r_val),
                delta=d, interpretation=interp, risk=False,
            ))
        return deltas

    @staticmethod
    def _interpret_delta(metric: str, current: float, delta: float, si: _StateInterpretation) -> str:
        if abs(delta) < 0.001:
            return "stable"
        direction = "increased" if delta > 0 else "decreased"
        if si.good_direction == "higher":
            quality = "improved" if delta > 0 else "worsened"
        elif si.good_direction == "lower":
            quality = "improved" if delta < 0 else "worsened"
        else:
            quality = "changed"
        return f"{direction} ({quality})"

    @staticmethod
    def _delta_symbol(metric: str, delta: float) -> str:
        """Return ✓/⚠/~ based on whether delta is in the good direction."""
        if abs(delta) < 0.005:
            return "~"
        si = STATE_INTERPRETATIONS.get(metric)
        if not si or si.good_direction == "context":
            return "~"
        if si.good_direction == "higher":
            return "✓" if delta > 0 else "⚠"
        else:
            return "✓" if delta < 0 else "⚠"


# ---------------------------------------------------------------------------
#  Utility
# ---------------------------------------------------------------------------

def _format_params(params: Dict[str, Any]) -> str:
    """Compact parameter formatting."""
    if not params:
        return ""
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
