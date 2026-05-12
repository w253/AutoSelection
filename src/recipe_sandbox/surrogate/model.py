import numpy as np
import logging
from typing import Dict, List, Optional, Sequence, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

from recipe_sandbox.pipeline.task_config import RecipeConfig, RecipeStepConfig
from recipe_sandbox.search.operator_policy import OFFICIAL_OPERATOR_ORDER

logger = logging.getLogger(__name__)

_GP_BASE_OPERATOR_ORDER: Tuple[str, ...] = tuple(
    op for op in OFFICIAL_OPERATOR_ORDER if op != "union"
)


class ANOVARegressor:
    """Surrogate Model mapping Recipe -> Expected Utility (mu, sigma).
    
    Uses a Gaussian Process with Matern kernel to capture low-order interactions
    (Main effects + pairwise interactions) efficiently, estimating both the 
    mean predicted score and the epistemic uncertainty for UCB-based MCTS.
    """
    
    def __init__(self):
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, 
            alpha=1e-2,
            normalize_y=True
        )
        self.is_fitted = False
        self._known_recipes: Dict[str, RecipeConfig] = {}
        
    def _encode_recipes(self, recipes: List[RecipeConfig]) -> np.ndarray:
        """Encode recipes via union-aware fusion over the official operator space."""
        lookup = dict(self._known_recipes)
        lookup.update({recipe.recipe_name: recipe for recipe in recipes if recipe.recipe_name})

        return self._encode_with_union_features(
            recipes,
            lookup,
            encoder=self._encode_base,
        )

    def _enabled_steps(
        self,
        recipe: RecipeConfig,
        *,
        exclude_union: bool = False,
    ) -> List[RecipeStepConfig]:
        steps = []
        for step in recipe.steps or []:
            if not step.enabled:
                continue
            if exclude_union and step.operator == "union":
                continue
            steps.append(step)
        return steps

    def _recipe_without_union(self, recipe: RecipeConfig) -> RecipeConfig:
        return RecipeConfig(
            enabled=recipe.enabled,
            recipe_name=recipe.recipe_name,
            input_split=recipe.input_split,
            input_stage=recipe.input_stage,
            steps=[
                RecipeStepConfig(
                    step_type=step.step_type,
                    operator_ref=step.operator_ref,
                    operator=step.operator,
                    params=dict(step.params),
                    enabled=step.enabled,
                    name=step.name,
                )
                for step in self._enabled_steps(recipe, exclude_union=True)
            ],
            task_context=dict(recipe.task_context),
        )

    def _find_union_source_name(self, recipe: RecipeConfig) -> Optional[str]:
        for step in self._enabled_steps(recipe):
            if step.operator != "union":
                continue
            source_recipe = step.params.get("source_recipe")
            if not source_recipe:
                return None
            return str(source_recipe)
        return None

    def _encode_with_union_features(
        self,
        recipes: Sequence[RecipeConfig],
        lookup: Dict[str, RecipeConfig],
        *,
        encoder,
    ) -> np.ndarray:
        encodings = []
        for recipe in recipes:
            base_vec = encoder(self._recipe_without_union(recipe))
            source_vec = np.zeros_like(base_vec)
            diff_vec = np.zeros_like(base_vec)
            max_vec = np.zeros_like(base_vec)
            union_enabled = 0.0

            source_name = self._find_union_source_name(recipe)
            if source_name:
                union_enabled = 1.0
                source_recipe = lookup.get(source_name)
                if source_recipe is not None:
                    source_vec = encoder(self._recipe_without_union(source_recipe))
                    diff_vec = np.abs(base_vec - source_vec)
                    max_vec = np.maximum(base_vec, source_vec)

            encodings.append(
                np.concatenate(
                    (base_vec, source_vec, diff_vec, max_vec, np.array([union_enabled], dtype=float))
                )
            )
        return np.vstack(encodings) if encodings else np.empty((0, 0), dtype=float)

    def _encode_base(self, recipe: RecipeConfig) -> np.ndarray:
        """Encode the official operator space (excluding union, which is handled via fusion).

        Dimensions (17 total):
          truncate_samples:        enabled(0/1), total_samples/100000
          mona_filter:             enabled(0/1), fraction
          ifd_filter:              enabled(0/1), fraction
          ngram_entropy:           enabled(0/1), fraction
          action_object_branching: enabled(0/1), fraction
          varentropy_filter:       enabled(0/1), fraction
          semantic_dedup:          enabled(0/1), jaccard_threshold
          semdedup:                enabled(0/1), num_clusters/10000, cosine_threshold
        """

        op_map = {step.operator: step.params for step in recipe.steps if step.enabled}
        vec: list[float] = []

        for operator_name in _GP_BASE_OPERATOR_ORDER:
            if operator_name == "truncate_samples":
                if operator_name in op_map:
                    total_samples = float(op_map[operator_name].get("total_samples", 10000))
                    vec.extend([1.0, max(0.0, min(1.0, total_samples / 100000.0))])
                else:
                    vec.extend([0.0, 1.0])
            elif operator_name == "semantic_dedup":
                if operator_name in op_map:
                    threshold = float(op_map[operator_name].get("jaccard_threshold", 0.8))
                    vec.extend([1.0, threshold])
                else:
                    vec.extend([0.0, 1.0])
            elif operator_name == "semdedup":
                if operator_name in op_map:
                    num_clusters = float(op_map[operator_name].get("num_clusters", 1000))
                    cosine_threshold = float(op_map[operator_name].get("cosine_threshold", 0.95))
                    vec.extend([1.0, num_clusters / 10000.0, cosine_threshold])
                else:
                    vec.extend([0.0, 0.1, 1.0])
            else:
                if operator_name in op_map:
                    vec.extend([1.0, float(op_map[operator_name].get("fraction", 0.5))])
                else:
                    vec.extend([0.0, 1.0])

        return np.array(vec, dtype=float)

    def fit(self, recipes: List[RecipeConfig], utilities: List[float]):
        """Fit the GP model to historical data."""
        if not recipes:
            return
        self._known_recipes.update({recipe.recipe_name: recipe for recipe in recipes if recipe.recipe_name})
            
        X = self._encode_recipes(recipes)
        y = np.array(utilities)
        
        logger.info(f"Fitting GaussianProcess Surrogate Model on {len(X)} samples.")
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, recipes: List[RecipeConfig]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict expected utility and uncertainty for candidates.
        
        Returns: 
            mu (np.ndarray): Mean predicted utility.
            sigma (np.ndarray): Standard deviation (uncertainty).
        """
        if not self.is_fitted:
            logger.warning("Surrogate Model not fitted yet, returning 0 expected utility with 1.0 uncertainty.")
            return np.zeros(len(recipes)), np.ones(len(recipes))
            
        X = self._encode_recipes(recipes)
        mu, sigma = self.model.predict(X, return_std=True)
        return mu, sigma
