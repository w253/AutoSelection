"""
LLM Action Generator for Autonomous Recipe Search.

This module parses the operator catalog and provides an LLM-based Action Generator
that takes the current recipe and catalog bounds to autonomously
propose a mutated RecipeConfig for the next search step.
"""

from __future__ import annotations

import json
import logging
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from recipe_sandbox.agents.base import LLMClient
from recipe_sandbox.pipeline.task_config import LLMConfig, RecipeConfig, RecipeStepConfig
from recipe_sandbox.search import OPERATOR_FAMILIES, family_for_operator

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from recipe_sandbox.agents.thinking_logger import ThinkingLogger


logger = logging.getLogger(__name__)


def _normalize_operator_name(
    operator_name: Optional[str],
    *,
    available_operators: Optional[Set[str]] = None,
) -> Optional[str]:
    if operator_name != "varentropy_mix":
        return operator_name
    if available_operators is not None and "varentropy_mix" not in available_operators and "truncate_samples" in available_operators:
        return "truncate_samples"
    return operator_name


def _extract_json(raw: str) -> Any:
    """Robustly extract a JSON object or array from LLM output that may
    contain surrounding prose, markdown fences, or other noise."""
    text = raw.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Fast path: the whole text is valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON code block inside markdown fences
    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Greedy bracket matching: find the first '[' or '{' and its matching closer
    for opener, closer in [("[", "]"), ("{", "}")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("No valid JSON found in LLM response", raw, 0)


class OperatorCatalog:
    """Parses and manages the operator parameter space from v2 YAML catalog."""

    def __init__(self, catalog_path: str):
        self.catalog_path = Path(catalog_path)
        self.families = self._load_catalog()
        
    def _load_catalog(self) -> Dict[str, Any]:
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Operator catalog not found: {self.catalog_path}")
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("families", {})
        
    def get_operator_schema(self, family_name: str, operator_name: str) -> Dict[str, Any]:
        """Returns the parameter schema for a specific operator."""
        family = self.families.get(family_name, {})
        operators = family.get("operators", {})
        return operators.get(operator_name, {})

    def get_param_bounds(self, operator_name: str) -> Dict[str, Dict[str, Any]]:
        """Get parameter bounds for an operator by scanning all families."""
        for family_name, family_data in self.families.items():
            operators = family_data.get("operators", {})
            if operator_name in operators:
                return operators[operator_name].get("params", {})
        return {}

    def clamp_params(self, operator_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp LLM-generated parameters to catalog-defined bounds."""
        bounds = self.get_param_bounds(operator_name)
        if not bounds:
            return params
        clamped = dict(params)
        for p_name, p_spec in bounds.items():
            if p_name not in clamped:
                continue
            p_range = p_spec.get("range")
            p_type = p_spec.get("type", "any")
            val = clamped[p_name]
            if p_range and isinstance(val, (int, float)) and len(p_range) == 2:
                low, high = p_range
                original = val
                val = max(low, min(high, val))
                if p_type == "int":
                    val = int(round(val))
                if val != original:
                    logger.debug("Clamped %s.%s: %s → %s (range=%s)", operator_name, p_name, original, val, p_range)
                clamped[p_name] = val
        return clamped

    def to_prompt_schema(self, allowed_operators: Optional[Set[str]] = None) -> str:
        """Renders the catalog as a compact prompt string for the LLM."""
        lines = ["# OPERATOR CATALOG (Available Families, Methods, & Hyperparameters)"]
        for family_name, family_data in self.families.items():
            lines.append(f"\n## Family: {family_name}")
            operators = family_data.get("operators", {})
            for op_name, op_data in operators.items():
                if allowed_operators is not None and op_name not in allowed_operators:
                    continue
                desc = op_data.get("description", "")
                lines.append(f"  ### Operator: {op_name}")
                if desc:
                    lines.append(f"      Description: {desc}")
                params = op_data.get("params", {})
                if params:
                    lines.append(f"      Tunable Parameters:")
                    for p_name, p_spec in params.items():
                        p_type = p_spec.get("type", "any")
                        p_default = p_spec.get("default")
                        p_range = p_spec.get("range")
                        p_choices = p_spec.get("choices")
                        p_desc = p_spec.get("description", "")
                        detail = f"        - {p_name} (type={p_type}, default={p_default}"
                        if p_range:
                            detail += f", range={p_range}"
                        if p_choices:
                            detail += f", choices={p_choices}"
                        detail += f") {p_desc}"
                        lines.append(detail)
        return "\n".join(lines)


class ActionLLMGenerator:
    """LLM wrapper that generates the next recipe based on diagnoses."""

    def __init__(self, llm_config: LLMConfig, catalog_path: str,
                 registered_operators: Optional[Set[str]] = None,
                 thinking_logger: Optional["ThinkingLogger"] = None):
        self.client = LLMClient(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            model=llm_config.model,
        )
        self.catalog = OperatorCatalog(catalog_path)
        self.registered_operators = registered_operators
        self.thinking_logger = thinking_logger
        self._iteration: int = 0

    def _render_current_recipe(self, recipe: RecipeConfig) -> str:
        steps = []
        for step in recipe.steps:
            if step.enabled:
                steps.append(f"  - {step.operator} (params: {json.dumps(step.params)})")
        if not steps:
            return "Current Recipe is EMPTY."
        return "\n".join(steps)

    def propose_next_recipe(
        self,
        current_recipe: RecipeConfig,
        diagnoses: List[Any],
        score: float,
        cost: float,
    ) -> RecipeConfig:
        """Ask the LLM to propose a new recipe mutating the current one."""
        effective_available_operators = self.registered_operators

        prompt = f"""You are an expert Data-Centric AI Search Controller optimizing a dataset processing recipe.

YOUR GOAL: Propose a SINGLE mutated recipe configuration that maximizes the performance score. Minor modifications generally work better than wild leaps.

{self.catalog.to_prompt_schema(effective_available_operators)}

=== CURRENT STATE ===
Current Recipe:
{self._render_current_recipe(current_recipe)}

Current Metric Score: {score:.4f}

=== INSTRUCTIONS ===
1. Analyze the current recipe and its performance.
2. Select operators and hyperparameters ONLY from the OPERATOR CATALOG.
3. Your output MUST be a valid JSON object representing the next recipe steps. Do NOT include markdown blocks (` ```json `), just raw JSON!
4. Format:
{{
  "steps": [
    {{
      "operator": "operator_name",
      "params": {{"param1": "value", "param2": 123}}
    }}
  ]
}}
"""
        logger.info("Querying LLM Action Generator for next recipe proposal...")
        resp = self.client.chat_with_reasoning(prompt, temperature=0.7)
        response = resp.answer
        if self.thinking_logger:
            self.thinking_logger.log(
                "action", self._iteration, resp.thinking, resp.answer,
                prompt_summary=f"propose_next: score={score:.2f}, cost={cost:.2f}",
            )
        
        try:
            parsed = _extract_json(response)
            steps = parsed.get("steps", [])
        except json.JSONDecodeError:
            logger.warning("First LLM parse failed, retrying with lower temperature...")
            resp2 = self.client.chat_with_reasoning(prompt, temperature=0.3)
            response = resp2.answer
            if self.thinking_logger:
                self.thinking_logger.log(
                    "action", self._iteration, resp2.thinking, resp2.answer,
                    prompt_summary="propose_next retry (temp=0.3)",
                )
            try:
                parsed = _extract_json(response)
                steps = parsed.get("steps", [])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Action LLM output after retry. Raw response:\n{response}")
                raise ValueError("Action LLM generated invalid JSON.") from e
            
        new_steps = []
        for i, step_dict in enumerate(steps):
            op = _normalize_operator_name(
                step_dict.get("operator"),
                available_operators=effective_available_operators,
            )
            params = step_dict.get("params", {})
            if op:
                if effective_available_operators and op not in effective_available_operators:
                    logger.warning("LLM proposed unavailable operator '%s' — skipping step.", op)
                    continue
                params = self.catalog.clamp_params(op, params)
                new_steps.append(
                    RecipeStepConfig(
                        step_type="auto",
                        operator=op,
                        params=params,
                        enabled=True,
                        name=f"{op}_step_{i+1}"
                    )
                )
        
        # We spawn a new configuration, maintaining the rest (name, split)
        new_recipe = RecipeConfig(
            enabled=True,
            recipe_name=f"{current_recipe.recipe_name}_mutated",
            input_split=current_recipe.input_split,
            input_stage=current_recipe.input_stage,
            steps=new_steps,
            task_context=current_recipe.task_context,
        )
        return new_recipe

    def propose_candidate_pool(
        self,
        current_recipe: RecipeConfig,
        diagnoses: List[Any],
        score: float,
        cost: float,
        n_candidates: int = 5,
        search_history: str = "",
        pool_size: int = 0,
        state_vector: Optional[Dict[str, float]] = None,
        experiment_insights: str = "",
        evaluated_recipes: Optional[List[str]] = None,
        benchmark_analysis: str = "",
        available_operators: Optional[Set[str]] = None,
        pool_source_count: Optional[int] = None,
    ) -> List[RecipeConfig]:
        """Ask the LLM to propose a pool of distinct new recipes mutating the current one."""
        effective_available_operators = available_operators or self.registered_operators

        registered_note = ""
        if effective_available_operators:
            registered_note = (
                "\n\n⚠️ IMPORTANT: Only the following operators are currently IMPLEMENTED and available for use: "
                + ", ".join(sorted(effective_available_operators))
                + ". Do NOT use any other operators from the catalog."
            )

        history_section = ""
        if search_history:
            history_section = f"""
=== SEARCH HISTORY ===
{search_history}

Use the history above to understand what worked and what didn't. Avoid repeating configurations that scored poorly. Try to build on successful patterns.
"""

        # Pool context and bidirectional search guidance
        pool_section = ""
        if pool_size > 0:
            ts_low = max(5000, int(pool_size * 0.10))
            ts_high = max(ts_low + 1000, int(pool_size * 0.60))
            pool_section = f"""
=== POOL CONTEXT ===
Total pool size: {pool_size:,} samples
Recommended total_samples range for truncate_samples: {ts_low:,} — {ts_high:,}

=== SEARCH STRATEGY (CRITICAL) ===
The search is BIDIRECTIONAL — you can BOTH tighten AND relax filters:
- If recent recipes that kept FEWER samples scored WORSE → RELAX filters (increase fraction, increase total_samples, remove filter steps)
- If recent recipes that kept MORE samples scored WORSE → TIGHTEN filters (decrease fraction, decrease total_samples, add filter steps)
- Do NOT always reduce data. The optimal recipe balances quality filtering with data quantity.
- At least 1 of your {n_candidates} proposals should try RELAXING filters compared to the current recipe.
- At least 1 should try TIGHTENING. The rest should explore moderately.
"""
            if pool_source_count is not None and pool_source_count <= 1:
                pool_section += (
                    "\nSingle-source pool detected: use truncate_samples when you need "
                    "a hard sample-count cap.\n"
                )

        # State vector section with metric explanations
        state_section = ""
        if state_vector:
            from recipe_sandbox.feedback.state_registry import STATE_KEY_REGISTRY
            sv_lines = []
            for k, v in state_vector.items():
                meta = STATE_KEY_REGISTRY.get(k)
                desc = meta.description if meta else ""
                if isinstance(v, (int, float)):
                    val_str = f"{v:.4f}"
                elif isinstance(v, dict):
                    val_str = ", ".join(f"{bk}: {bv:.4f}" if isinstance(bv, (int, float)) else f"{bk}: {bv}" for bk, bv in v.items())
                else:
                    val_str = str(v)
                sv_lines.append(f"  {k}: {val_str}  — {desc}" if desc else f"  {k}: {val_str}")
            state_section = "\n=== STATE VECTOR (current recipe metrics) ===\n" + "\n".join(sv_lines) + "\n"

        # Union operator section
        union_section = ""
        if evaluated_recipes:
            union_section = f"""
=== UNION OPERATOR ===
The 'union' operator merges current filtered data with output from a previously-evaluated recipe.
This increases data count by adding samples from the source recipe that are not in the current dataset.

Available source recipes for union: {', '.join(evaluated_recipes)}

Usage: Add a step {{"operator": "union", "params": {{"source_recipe": "recipe_name"}}}}
"""

        insights_section = ""
        if experiment_insights:
            insights_section = f"""
=== EXPERIMENT INSIGHTS (from historical pattern analysis) ===
{experiment_insights}

Use these insights to guide your proposals. Avoid repeating strategies that have been shown to fail.
These are hypotheses based on limited data — treat them as strong suggestions, not absolute rules.
"""

        # Benchmark diagnostic analysis (from BenchmarkSuggestor)
        benchmark_section = ""
        if benchmark_analysis:
            benchmark_section = f"\n{benchmark_analysis}\n"

        prompt = f"""You are an expert Data-Centric AI Search Controller optimizing a dataset processing recipe.

YOUR GOAL: Propose {n_candidates} DISTINCT mutated recipe configurations that resolve current risks and explore different valid subspaces. Some should be conservative, some more aggressive.

{self.catalog.to_prompt_schema(effective_available_operators)}{registered_note}

=== CURRENT STATE ===
Current Recipe:
{self._render_current_recipe(current_recipe)}

Current Metric Score: {score:.4f}
{state_section}{benchmark_section}
{pool_section}{history_section}{insights_section}{union_section}=== INSTRUCTIONS ===
1. Analyze the current recipe, state vector, and search history.
2. Select operators and hyperparameters ONLY from the OPERATOR CATALOG.
3. Your output MUST be a valid JSON array of objects representing the {n_candidates} recipes. Do NOT include markdown blocks (` ```json `), just raw JSON!
4. Format:
[
  {{
    "steps": [
      {{
        "operator": "operator_name",
        "params": {{"param1": "value", "param2": 123}}
      }}
    ]
  }},
  ... (up to {n_candidates} distinct configurations)
]
"""
        logger.info(f"Querying LLM Action Generator for {n_candidates} recipe proposals...")
        resp = self.client.chat_with_reasoning(prompt, temperature=0.9)
        response = resp.answer
        if self.thinking_logger:
            self.thinking_logger.log(
                "action", self._iteration, resp.thinking, resp.answer,
                prompt_summary=f"propose_pool: {n_candidates} candidates",
            )
        
        try:
            parsed_list = _extract_json(response)
            if not isinstance(parsed_list, list):
                if "recipes" in parsed_list:
                    parsed_list = parsed_list["recipes"]
                else:
                    parsed_list = [parsed_list]
        except json.JSONDecodeError:
            logger.warning("First LLM pool parse failed, retrying with lower temperature...")
            resp2 = self.client.chat_with_reasoning(prompt, temperature=0.3)
            response = resp2.answer
            if self.thinking_logger:
                self.thinking_logger.log(
                    "action", self._iteration, resp2.thinking, resp2.answer,
                    prompt_summary="propose_pool retry (temp=0.3)",
                )
            try:
                parsed_list = _extract_json(response)
                if not isinstance(parsed_list, list):
                    if "recipes" in parsed_list:
                        parsed_list = parsed_list["recipes"]
                    else:
                        parsed_list = [parsed_list]
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Action LLM output after retry. Raw response:\n{response}")
                raise ValueError("Action LLM generated invalid JSON.") from e
            
        pool = []
        for v_idx, parsed in enumerate(parsed_list):
            steps = parsed.get("steps", [])
            new_steps = []
            for i, step_dict in enumerate(steps):
                op = _normalize_operator_name(
                    step_dict.get("operator"),
                    available_operators=effective_available_operators,
                )
                params = step_dict.get("params", {})
                if op:
                    if effective_available_operators and op not in effective_available_operators:
                        logger.warning("LLM proposed unregistered operator '%s' — skipping step.", op)
                        continue
                    params = self.catalog.clamp_params(op, params)
                    new_steps.append(
                        RecipeStepConfig(
                            step_type="auto",
                            operator=op,
                            params=params,
                            enabled=True,
                            name=f"{op}_step_{i+1}"
                        )
                    )
            
            if not new_steps:
                continue
                
            new_recipe = RecipeConfig(
                enabled=True,
                recipe_name=f"candidate_{v_idx+1}",
                input_split=current_recipe.input_split,
                input_stage=current_recipe.input_stage,
                steps=new_steps,
                task_context=current_recipe.task_context,
            )
            pool.append(new_recipe)
            
        if not pool:
            logger.warning("LLM returned 0 valid recipes in pool. Padding with current.")
            pool.append(
                RecipeConfig(
                    enabled=True,
                    recipe_name=current_recipe.recipe_name,
                    input_split=current_recipe.input_split,
                    input_stage=current_recipe.input_stage,
                    steps=[
                        RecipeStepConfig(
                            step_type=step.step_type,
                            operator=step.operator,
                            params=dict(step.params),
                            enabled=step.enabled,
                            name=step.name,
                        )
                        for step in (current_recipe.steps or [])
                    ],
                    task_context=current_recipe.task_context,
                )
            )
             
        return pool[:n_candidates]

    def tune_restart_params(
        self,
        fixed_steps: List[RecipeStepConfig],
        *,
        historical_examples: Dict[str, List[Dict[str, Any]]],
    ) -> List[RecipeStepConfig]:
        """Let the LLM tune params for a fixed restart motif without changing structure."""
        if len(fixed_steps) <= 1:
            return fixed_steps

        tunable_steps = fixed_steps[1:]
        allowed_operators = {step.operator for step in tunable_steps}
        prompt = f"""You are tuning parameters for a FIXED restart recipe motif.

The operator structure is locked. You MUST keep the same operator order and names.
You are ONLY allowed to fill or adjust parameter values for the non-truncate steps.

{self.catalog.to_prompt_schema(allowed_operators)}

=== FIXED RESTART STEPS ===
{json.dumps([{"operator": step.operator, "params": step.params} for step in tunable_steps], ensure_ascii=False, indent=2)}

=== HISTORICAL SUCCESSFUL EXAMPLES ===
{json.dumps(historical_examples, ensure_ascii=False, indent=2)}

=== RULES ===
1. Do NOT add, remove, or reorder operators.
2. Do NOT change operator names.
3. Keep parameter values within catalog bounds.
4. Prefer historically successful ranges and values when possible.
5. Return raw JSON only, in this exact format:
[
  {{"operator": "ifd_filter", "params": {{"fraction": 0.35}}}}
]
"""
        resp = self.client.chat_with_reasoning(prompt, temperature=0.3)
        if self.thinking_logger:
            self.thinking_logger.log(
                "action", self._iteration, resp.thinking, resp.answer,
                prompt_summary="restart_param_tune",
            )

        try:
            parsed_steps = _extract_json(resp.answer)
            if not isinstance(parsed_steps, list):
                raise ValueError("restart param tuning must return a JSON array")
        except Exception as exc:
            logger.warning("Restart param tuning failed, keeping historical defaults: %s", exc)
            return fixed_steps

        tuned_steps: List[RecipeStepConfig] = [fixed_steps[0]]
        for index, base_step in enumerate(tunable_steps):
            if index >= len(parsed_steps):
                tuned_steps.append(base_step)
                continue
            raw_step = parsed_steps[index] or {}
            op = _normalize_operator_name(
                raw_step.get("operator"),
                available_operators=allowed_operators,
            )
            if op != base_step.operator:
                tuned_steps.append(base_step)
                continue
            params = self.catalog.clamp_params(op, raw_step.get("params", {}))
            merged_params = dict(base_step.params)
            merged_params.update(params)
            tuned_steps.append(
                RecipeStepConfig(
                    step_type=base_step.step_type,
                    operator=base_step.operator,
                    params=merged_params,
                    enabled=base_step.enabled,
                    name=base_step.name,
                )
            )
        return tuned_steps

    def propose_restart_steps(
        self,
        *,
        allowed_operators: Set[str],
        historical_examples: Dict[str, List[Dict[str, Any]]],
        search_history: str,
        credit_summary: Dict[str, Any],
        pool_size: int,
    ) -> List[RecipeStepConfig]:
        """Let the thinking model choose restart operators from search evidence."""

        nontruncate_allowed = {
            op for op in allowed_operators
            if op not in {"truncate_samples", "union"}
        }
        if not nontruncate_allowed:
            return []

        positive_ops = [
            {
                "operator": name,
                **stats,
            }
            for name, stats in credit_summary.get("top_positive_operators", [])
            if name in nontruncate_allowed
        ]
        positive_pairs = [
            {
                "operators": list(pair),
                **stats,
            }
            for pair, stats in credit_summary.get("top_positive_pairs", [])
            if all(name in nontruncate_allowed for name in pair)
        ]
        prompt = f"""You are choosing a restart operator motif for trajectory search.

Use the search evidence below to select a small restart motif that is promising.
You must choose between 1 and 3 NON-TRUNCATE operators.

{self.catalog.to_prompt_schema(nontruncate_allowed)}

=== SEARCH HISTORY ===
{search_history}

=== POSITIVE OPERATOR SIGNALS ===
{json.dumps(positive_ops, ensure_ascii=False, indent=2)}

=== POSITIVE PAIR SIGNALS ===
{json.dumps(positive_pairs, ensure_ascii=False, indent=2)}

=== HISTORICAL SUCCESSFUL EXAMPLES ===
{json.dumps(historical_examples, ensure_ascii=False, indent=2)}

=== RULES ===
1. Select only operators from the allowed catalog above.
2. Prefer operators and combinations supported by the evidence.
3. Do not use truncate_samples or union here; truncate is handled separately.
4. Keep parameters within catalog bounds.
5. Return raw JSON only in this exact format:
[
  {{"operator": "mona_filter", "params": {{"fraction": 0.5}}}},
  {{"operator": "ngram_entropy", "params": {{"fraction": 0.4}}}}
]

Pool size reference: {pool_size}
"""
        resp = self.client.chat_with_reasoning(prompt, temperature=0.3)
        if self.thinking_logger:
            self.thinking_logger.log(
                "action", self._iteration, resp.thinking, resp.answer,
                prompt_summary="restart_operator_select",
            )

        parsed = _extract_json(resp.answer)
        if not isinstance(parsed, list):
            raise ValueError("restart operator selection must return a JSON array")

        steps: List[RecipeStepConfig] = []
        seen: Set[str] = set()
        for index, raw_step in enumerate(parsed[:3], start=1):
            raw_step = raw_step or {}
            op = _normalize_operator_name(
                raw_step.get("operator"),
                available_operators=nontruncate_allowed,
            )
            if not op or op in seen or op not in nontruncate_allowed:
                continue
            params = self.catalog.clamp_params(op, raw_step.get("params", {}))
            steps.append(
                RecipeStepConfig(
                    operator=op,
                    params=params,
                    enabled=True,
                    name=f"restart_{op}_{index}",
                )
            )
            seen.add(op)
        if not steps:
            raise ValueError("restart operator selection returned no valid steps")
        return steps
