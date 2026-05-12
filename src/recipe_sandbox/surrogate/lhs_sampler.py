import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats.qmc import LatinHypercube

from recipe_sandbox.pipeline.task_config import RecipeConfig, RecipeStepConfig
from recipe_sandbox.search.operator_policy import OFFICIAL_OPERATOR_ORDER, resolve_lhs_operator_space

logger = logging.getLogger(__name__)

_TRUNCATION_OP = "truncate_samples"

# ── GP-aligned operator encoding layout ──
# Each operator occupies a fixed slice in the 17D encoding vector.
# Layout mirrors `_encode_base()` in surrogate/model.py exactly.
#
# Operator                  dims  indices   param semantics
# ─────────────────────────────────────────────────────────
# truncate_samples           2    [0, 1]    enabled, total_samples/100000
# mona_filter                2    [2, 3]    enabled, fraction
# ifd_filter                 2    [4, 5]    enabled, fraction
# ngram_entropy              2    [6, 7]    enabled, fraction
# action_object_branching    2    [8, 9]    enabled, fraction
# varentropy_filter          2   [10,11]    enabled, fraction
# semantic_dedup             2   [12,13]    enabled, jaccard_threshold
# semdedup                   3   [14,15,16] enabled, num_clusters/10000, cosine_threshold
#
# Total: 17 dimensions

_GP_BASE_OPERATOR_ORDER: Tuple[str, ...] = tuple(
    op for op in OFFICIAL_OPERATOR_ORDER if op != "union"
)

_BASE_DIM = 17  # total GP base dimensions

# Operator → (start_index, n_dims)
_OP_LAYOUT: Dict[str, Tuple[int, int]] = {}
_idx = 0
for _op in _GP_BASE_OPERATOR_ORDER:
    _nd = 3 if _op == "semdedup" else 2
    _OP_LAYOUT[_op] = (_idx, _nd)
    _idx += _nd
assert _idx == _BASE_DIM, f"Layout mismatch: {_idx} != {_BASE_DIM}"

# Parameter ranges for decoding LHS [0,1] → real params
# For each operator: list of (param_name, lo, hi) for dimensions after the enabled bit
_PARAM_RANGES: Dict[str, List[Tuple[str, float, float]]] = {
    "truncate_samples":        [("total_samples_frac", 0.1, 1.0)],
    "mona_filter":             [("fraction", 0.05, 0.95)],
    "ifd_filter":              [("fraction", 0.05, 0.95)],
    "ngram_entropy":           [("fraction", 0.05, 0.95)],
    "action_object_branching": [("fraction", 0.05, 0.95)],
    "varentropy_filter":       [("fraction", 0.05, 0.95)],
    "semantic_dedup":          [("jaccard_threshold", 0.5, 0.99)],
    "semdedup":                [("num_clusters_frac", 0.01, 0.05),
                                ("cosine_threshold", 0.3, 0.95)],
}

# Target retention zones for warmup only
_RETENTION_ZONES: Tuple[Tuple[float, float], ...] = (
    (0.60, 0.70),   # moderate
    (0.30, 0.50),   # aggressive
    (0.05, 0.15),   # exploratory
)
_ZONE_NAMES: Tuple[str, ...] = ("mod", "agg", "exp")


def _decode_lhs_point(
    point: np.ndarray,
    pool_size: int,
    allowed_ops: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    """Decode a single 17D LHS point in [0,1]^17 into a list of recipe steps.

    For each operator in the official order:
      - point[enabled_dim] > 0.5 → operator is enabled
      - remaining dims are linearly mapped to parameter ranges
    Only operators in *allowed_ops* can be enabled.
    """
    steps: List[Dict[str, Any]] = []

    for op_name in _GP_BASE_OPERATOR_ORDER:
        start, n_dims = _OP_LAYOUT[op_name]
        enabled_val = point[start]

        if op_name not in allowed_ops:
            continue
        if enabled_val <= 0.5:
            continue

        params: Dict[str, Any] = {}
        ranges = _PARAM_RANGES[op_name]
        for dim_offset, (pname, lo, hi) in enumerate(ranges, start=1):
            raw = float(point[start + dim_offset])
            real_val = lo + raw * (hi - lo)
            params[pname] = real_val

        # Convert internal param names to operator-specific params
        if op_name == "truncate_samples":
            total_samples = int(round(max(1, pool_size * params["total_samples_frac"])))
            steps.append({"operator": op_name, "params": {"total_samples": total_samples}})
        elif op_name == "semantic_dedup":
            threshold = round(params["jaccard_threshold"], 2)
            steps.append({"operator": op_name, "params": {
                "strategy": "minhash", "jaccard_threshold": threshold,
            }})
        elif op_name == "semdedup":
            num_clusters = max(100, int(round(pool_size * params["num_clusters_frac"])))
            cosine_threshold = round(params["cosine_threshold"], 2)
            steps.append({"operator": op_name, "params": {
                "cosine_threshold": cosine_threshold, "num_clusters": num_clusters,
            }})
        else:
            frac = round(max(0.05, min(0.95, params["fraction"])), 2)
            steps.append({"operator": op_name, "params": {"fraction": frac}})

    return steps


def generate_lhs_recipes(
    n_samples: int = 10,
    seed: int = 42,
    pool_size: int = 0,
    allowed_operators: Optional[Sequence[str]] = None,
    min_enabled: int = 2,
    max_enabled: int = 4,
) -> List[RecipeConfig]:
    """Generate recipes via Latin Hypercube Sampling in the 17D GP encoding space.

    Each recipe corresponds to a unique, well-spread point in the continuous
    search space.  The LHS guarantees that no two recipes share the same
    stratum in any dimension, providing maximal coverage.

    Args:
        n_samples: Number of recipes to generate.
        seed: Random seed for reproducibility.
        pool_size: Training pool size (used to set truncate_samples).
        allowed_operators: Operators to include (defaults to full LHS space).
        min_enabled: Minimum number of enabled operators per recipe.
        max_enabled: Maximum number of enabled operators per recipe.
    """
    if allowed_operators is not None:
        allowed_ops = tuple(
            op for op in _GP_BASE_OPERATOR_ORDER
            if op in set(allowed_operators)
        )
    else:
        lhs_space = resolve_lhs_operator_space(tuple(_GP_BASE_OPERATOR_ORDER))
        allowed_ops = tuple(op for op in _GP_BASE_OPERATOR_ORDER if op in set(lhs_space))

    # Generate LHS samples in [0,1]^17
    sampler = LatinHypercube(d=_BASE_DIM, seed=seed)
    raw_points = sampler.random(n=n_samples)

    recipes: List[RecipeConfig] = []
    for index, point in enumerate(raw_points):
        steps = _decode_lhs_point(point, pool_size, allowed_ops)

        # Enforce minimum: force-enable operators closest to threshold
        if len(steps) < min_enabled:
            candidates: List[Tuple[float, str]] = []
            for op_name in _GP_BASE_OPERATOR_ORDER:
                if op_name not in allowed_ops:
                    continue
                start, _ = _OP_LAYOUT[op_name]
                if point[start] <= 0.5:
                    candidates.append((point[start], op_name))
            candidates.sort(key=lambda x: -x[0])
            forced_point = point.copy()
            for _, op_name in candidates[:min_enabled - len(steps)]:
                start, _ = _OP_LAYOUT[op_name]
                forced_point[start] = 0.75
            steps = _decode_lhs_point(forced_point, pool_size, allowed_ops)

        # Enforce maximum: keep only the top-N operators by enabled_val
        if len(steps) > max_enabled:
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for step in steps:
                start, _ = _OP_LAYOUT[step["operator"]]
                scored.append((point[start], step))
            scored.sort(key=lambda x: -x[0])
            steps = [s for _, s in scored[:max_enabled]]

        recipe_steps = [
            RecipeStepConfig(
                operator=str(step["operator"]),
                params=dict(step["params"]),
                enabled=True,
                name=f"lhs_{index + 1}_{si}",
            )
            for si, step in enumerate(steps, start=1)
        ]
        recipe = RecipeConfig(
            enabled=True,
            recipe_name=f"lhs_recipe_{index + 1:02d}",
            input_split="train",
            input_stage="canonical",
            steps=recipe_steps,
        )
        recipes.append(recipe)

        logger.info(
            "LHS %d/%d: %d ops → %s",
            index + 1, n_samples, len(steps),
            [(s["operator"], s["params"]) for s in steps],
        )

    return recipes


# ── Legacy warmup seeds (three-zone retention targeting) ──

def _resolve_ops(
    allowed_operators: Optional[Sequence[str]],
) -> Tuple[Tuple[str, ...], bool]:
    if allowed_operators is None:
        fast_ops = ("varentropy_filter", "mona_filter", "ifd_filter", "ngram_entropy")
        return fast_ops, True

    lhs_space = resolve_lhs_operator_space(tuple(allowed_operators))
    allow_truncate = _TRUNCATION_OP in lhs_space
    filter_ops = tuple(op for op in lhs_space if op != _TRUNCATION_OP)
    if not filter_ops:
        raise ValueError("LHS/random generation requires at least one filter operator.")
    return filter_ops, allow_truncate


def _build_step(
    operator: str,
    raw_ratio: float,
    *,
    pool_size: int,
) -> Dict[str, Any]:
    if operator == _TRUNCATION_OP:
        if pool_size <= 0:
            raise ValueError("truncate_samples warmup requires a positive pool_size")
        total_samples = int(round(max(1, min(pool_size, pool_size * raw_ratio))))
        return {"operator": operator, "params": {"total_samples": total_samples}}

    if operator == "semantic_dedup":
        threshold = round(max(0.5, min(0.99, raw_ratio)), 2)
        return {"operator": operator, "params": {"strategy": "minhash", "jaccard_threshold": threshold}}

    if operator == "semdedup":
        cosine_threshold = round(max(0.3, min(0.95, raw_ratio)), 2)
        num_clusters = max(100, int(round(pool_size * 0.01))) if pool_size > 0 else 500
        return {"operator": operator, "params": {"cosine_threshold": cosine_threshold, "num_clusters": num_clusters}}

    frac = round(max(0.05, min(0.95, raw_ratio)), 2)
    return {"operator": operator, "params": {"fraction": frac}}


def generate_lhs_seeds(
    catalog_path: str,
    n_samples: int = 10,
    seed: int = 42,
    pool_size: int = 0,
    allowed_operators: Optional[Sequence[str]] = None,
) -> List[RecipeConfig]:
    """Generate warmup recipes targeting three retention zones.

    This is the warmup-specific generator used by the MCTS search loop.
    For baseline experiments, use ``generate_lhs_recipes`` instead.
    """
    del catalog_path
    rng = random.Random(seed)
    filter_ops, allow_truncate = _resolve_ops(allowed_operators)
    logger.info("Generating %d staged warmup recipes.", n_samples)

    recipes: List[RecipeConfig] = []
    for index in range(n_samples):
        zone_idx = index % len(_RETENTION_ZONES)
        lo, hi = _RETENTION_ZONES[zone_idx]
        zone_name = _ZONE_NAMES[zone_idx]

        target_retain = rng.uniform(lo, hi)

        if pool_size > 0 and allow_truncate:
            n_filters = rng.randint(1, min(3, len(filter_ops)))
            sampled_filters = rng.sample(filter_ops, n_filters)
            ops = [_TRUNCATION_OP] + sampled_filters
        else:
            n_steps = rng.randint(2, min(4, len(filter_ops)))
            ops = rng.sample(filter_ops, n_steps)

        n_steps = len(ops)
        base_frac = target_retain ** (1.0 / n_steps)
        steps: List[Dict[str, Any]] = []
        for op in ops:
            raw_ratio = base_frac * rng.uniform(0.92, 1.08)
            steps.append(_build_step(op, raw_ratio, pool_size=pool_size))

        logger.info(
            "Warmup %d/%d [%s]: target_retain=%.2f, ops=%s",
            index + 1, n_samples, zone_name, target_retain,
            [(s["operator"], s["params"]) for s in steps],
        )

        recipe_steps = [
            RecipeStepConfig(
                operator=str(step["operator"]),
                params=dict(step["params"]),
                enabled=True,
                name=f"warmup_{zone_name}_{si}",
            )
            for si, step in enumerate(steps, start=1)
        ]
        recipes.append(RecipeConfig(
            enabled=True,
            recipe_name=f"warmup_{zone_name}_{index + 1}",
            input_split="train",
            input_stage="canonical",
            steps=recipe_steps,
        ))

    return recipes
