from __future__ import annotations

import importlib
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable
from types import ModuleType
from typing import Any, Dict, List, Optional

import yaml

from recipe_sandbox.schema.types import CanonicalSample
from recipe_sandbox.operators.registry import OperatorRegistry

logger = logging.getLogger(__name__)


@dataclass
class ExtensionPrecomputeResult:
    module_name: str
    summary: Dict[str, Any] = field(default_factory=dict)


def parse_extension_modules(value: Optional[str | Iterable[str]]) -> List[str]:
    """Normalize comma-separated extension module config."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = list(value)
    return [str(item).strip() for item in raw_items if str(item).strip()]


def import_extension_modules(module_names: str | Iterable[str] | None) -> List[ModuleType]:
    return [
        importlib.import_module(module_name)
        for module_name in parse_extension_modules(module_names)
    ]


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
    for module in import_extension_modules(module_names):
        _register_module_operators(module, registry)
        hooks.extend(_load_module_hooks(module))
        logger.info("Loaded recipe_sandbox extension module: %s", module.__name__)
    return hooks


def run_extension_precomputations(
    module_names: str | Iterable[str] | None,
    *,
    samples: List[CanonicalSample],
    context: Optional[Dict[str, Any]] = None,
) -> List[ExtensionPrecomputeResult]:
    """Run optional cold-start precomputations supplied by extensions.

    Extension modules can expose ``precompute_features(samples, context)``.
    The function may mutate sample.metadata.extra in place and can return a
    small summary dict for logging.
    """
    results: List[ExtensionPrecomputeResult] = []
    ctx = context or {}
    for module in import_extension_modules(module_names):
        precompute = getattr(module, "precompute_features", None)
        if not callable(precompute):
            continue
        summary = precompute(samples=samples, context=ctx)
        if summary is None:
            summary = {}
        if not isinstance(summary, dict):
            summary = {"result": summary}
        results.append(
            ExtensionPrecomputeResult(module_name=module.__name__, summary=dict(summary))
        )
        logger.info(
            "Extension precompute complete: %s summary=%s",
            module.__name__,
            summary,
        )
    return results


def materialize_extension_operator_catalog(
    base_catalog_path: str,
    output_path: str,
    module_names: str | Iterable[str] | None,
) -> str:
    """Merge extension operator prompt metadata into a run-local catalog.

    Extension modules can expose ``OPERATOR_CATALOG_PATCH`` or
    ``get_operator_catalog_patch()``.  The patch uses the same top-level shape
    as ``examples/recipes/operator_catalog.yaml``.
    """
    modules = import_extension_modules(module_names)
    patches = [_get_catalog_patch(module) for module in modules]
    patches = [patch for patch in patches if patch]
    if not patches:
        return base_catalog_path

    base_path = Path(base_catalog_path)
    with base_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle) or {}
    if not isinstance(catalog, dict):
        raise ValueError(f"Operator catalog must be a mapping: {base_catalog_path}")

    merged = deepcopy(catalog)
    for patch in patches:
        _deep_merge(merged, patch)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False, allow_unicode=True)
    logger.info(
        "Materialized extension operator catalog with %d patch(es): %s",
        len(patches),
        destination,
    )
    return str(destination)


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


def _get_catalog_patch(module: ModuleType) -> Dict[str, Any]:
    get_patch = getattr(module, "get_operator_catalog_patch", None)
    if callable(get_patch):
        patch = get_patch()
    else:
        patch = getattr(module, "OPERATOR_CATALOG_PATCH", None)
    if patch is None:
        return {}
    if not isinstance(patch, dict):
        raise TypeError(f"Operator catalog patch from {module.__name__} must be a dict.")
    return patch


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base
