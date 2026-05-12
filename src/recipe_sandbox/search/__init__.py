"""Recipe action space.

Defines the three-layer action space for recipe mutations:
  1. Family enable / disable
  2. Method switching within a family
  3. Parameter adjustment within a method
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
#  Operator Family Registry (canonical family → operator names)
# ---------------------------------------------------------------------------

OPERATOR_FAMILIES: Dict[str, List[str]] = {
    "source_mixing": ["source_mix", "truncate_samples", "varentropy_mix"],
    "task_relevance_selection": [
        "mona_filter",
        "ifd_filter",
        "ngram_entropy",
        "action_object_branching",
        "varentropy_filter",
    ],
    "dedup_or_redundancy_control": ["semantic_dedup", "semdedup"],
    "set_operations": ["union"],
}

OPERATOR_TO_FAMILY: Dict[str, str] = {}
for _family, _ops in OPERATOR_FAMILIES.items():
    for _op in _ops:
        OPERATOR_TO_FAMILY[_op] = _family


def family_for_operator(operator_name: str) -> str:
    return OPERATOR_TO_FAMILY.get(operator_name, "unknown")


# ---------------------------------------------------------------------------
#  Action Type
# ---------------------------------------------------------------------------

class ActionType(Enum):
    ENABLE_FAMILY = "enable_family"
    DISABLE_FAMILY = "disable_family"
    CHANGE_METHOD = "change_method"
    ADJUST_PARAM = "adjust_param"


# ---------------------------------------------------------------------------
#  Recipe Action
# ---------------------------------------------------------------------------

@dataclass
class RecipeAction:
    """One atomic recipe mutation."""

    action_type: ActionType
    target_family: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Hashable identity for dedup and masking."""
        return f"{self.action_type.value}::{self.target_family}::{sorted(self.details.items())}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target_family": self.target_family,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
#  Structured Recipe (family-level view)
# ---------------------------------------------------------------------------

@dataclass
class FamilyConfig:
    """Configuration of one operator family within a recipe."""

    family: str
    enabled: bool = True
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredRecipe:
    """Recipe as r = (r1, r2, r3, r4, r5), one slot per family."""

    families: List[FamilyConfig] = field(default_factory=list)
    name: str = ""

    def get_family(self, family_name: str) -> Optional[FamilyConfig]:
        for fc in self.families:
            if fc.family == family_name:
                return fc
        return None

    def is_enabled(self, family_name: str) -> bool:
        fc = self.get_family(family_name)
        return fc.enabled if fc else False


# ---------------------------------------------------------------------------
#  Parameter Perturbation Ranges
# ---------------------------------------------------------------------------

DEFAULT_PARAM_RANGES: Dict[str, Dict[str, Any]] = {
    "source_mix": {
        "total_samples": {"type": "int", "step": 500, "min": 100, "max": 100000},
    },
    "truncate_samples": {
        "total_samples": {"type": "int", "step": 500, "min": 100, "max": 100000},
    },
    "mona_filter": {
        "top_k": {"type": "int", "step": 500, "min": 50, "max": 50000},
    },
    "ifd_filter": {
        "fraction": {"type": "float", "step": 0.05, "min": 0.05, "max": 1.0},
    },
    "semantic_dedup": {
        "similarity_threshold": {"type": "float", "step": 0.05, "min": 0.5, "max": 1.0},
    },
    "varentropy_filter": {
        "fraction": {"type": "float", "step": 0.05, "min": 0.05, "max": 1.0},
    },
}


# ---------------------------------------------------------------------------
#  Action Enumeration
# ---------------------------------------------------------------------------

def enumerate_actions(
    recipe: StructuredRecipe,
    *,
    param_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[RecipeAction]:
    """Enumerate all candidate actions for a given recipe state."""

    ranges = param_ranges or DEFAULT_PARAM_RANGES
    actions: List[RecipeAction] = []

    for family_name, methods in OPERATOR_FAMILIES.items():
        fc = recipe.get_family(family_name)

        if fc and fc.enabled:
            # Can disable
            actions.append(RecipeAction(
                action_type=ActionType.DISABLE_FAMILY,
                target_family=family_name,
            ))
            # Can switch method
            for alt_method in methods:
                if alt_method != fc.method:
                    actions.append(RecipeAction(
                        action_type=ActionType.CHANGE_METHOD,
                        target_family=family_name,
                        details={"from": fc.method, "to": alt_method},
                    ))
            # Can adjust params
            method_ranges = ranges.get(fc.method, {})
            for param_name, spec in method_ranges.items():
                current_val = fc.params.get(param_name)
                step = spec.get("step", 1)
                lo = spec.get("min", float("-inf"))
                hi = spec.get("max", float("inf"))
                # Up
                if current_val is not None:
                    new_up = current_val + step
                    if new_up <= hi:
                        actions.append(RecipeAction(
                            action_type=ActionType.ADJUST_PARAM,
                            target_family=family_name,
                            details={"param": param_name, "direction": "up", "from": current_val, "to": new_up},
                        ))
                    new_down = current_val - step
                    if new_down >= lo:
                        actions.append(RecipeAction(
                            action_type=ActionType.ADJUST_PARAM,
                            target_family=family_name,
                            details={"param": param_name, "direction": "down", "from": current_val, "to": new_down},
                        ))
        else:
            # Can enable (with each method)
            for method in methods:
                actions.append(RecipeAction(
                    action_type=ActionType.ENABLE_FAMILY,
                    target_family=family_name,
                    details={"method": method},
                ))

    return actions
