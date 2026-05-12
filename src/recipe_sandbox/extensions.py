from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from types import ModuleType
from typing import Any, List, Optional

from recipe_sandbox.operators.registry import OperatorRegistry

logger = logging.getLogger(__name__)


def parse_extension_modules(value: Optional[str | Iterable[str]]) -> List[str]:
    """Normalize comma-separated extension module config."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = list(value)
    return [str(item).strip() for item in raw_items if str(item).strip()]


def load_extensions(
    module_names: str | Iterable[str] | None,
    *,
    registry: Optional[OperatorRegistry] = None,
) -> List[Any]:
    """Import extension modules and return recipe execution hooks.

    Supported module-level functions/variables:

    - register_operators(registry): register custom BaseOperator classes.
    - register(registry): legacy alias for register_operators.
    - get_recipe_hooks(): return one hook or an iterable of hooks.
    - RECIPE_HOOKS: one hook or an iterable of hooks.
    """
    hooks: List[Any] = []
    for module_name in parse_extension_modules(module_names):
        module = importlib.import_module(module_name)
        _register_module_operators(module, registry)
        hooks.extend(_load_module_hooks(module))
        logger.info("Loaded recipe_sandbox extension module: %s", module_name)
    return hooks


def _register_module_operators(
    module: ModuleType,
    registry: Optional[OperatorRegistry],
) -> None:
    if registry is None:
        return

    register_operators = getattr(module, "register_operators", None)
    if callable(register_operators):
        register_operators(registry)
        return

    register = getattr(module, "register", None)
    if callable(register):
        register(registry)


def _load_module_hooks(module: ModuleType) -> List[Any]:
    get_recipe_hooks = getattr(module, "get_recipe_hooks", None)
    if callable(get_recipe_hooks):
        return _coerce_hooks(get_recipe_hooks())

    recipe_hooks = getattr(module, "RECIPE_HOOKS", None)
    return _coerce_hooks(recipe_hooks)


def _coerce_hooks(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise TypeError("Recipe hooks must be objects, not strings.")
    if isinstance(value, Iterable):
        return list(value)
    return [value]
