from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_recipe_catalog(path: str | Path) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Operator catalog must be a YAML mapping.")
    payload.setdefault("operators", {})
    return payload


def resolve_recipe_with_catalog(path: str | Path, recipe_payload: Dict[str, Any]) -> Dict[str, Any]:
    catalog = load_recipe_catalog(path)
    operators = dict(catalog.get("operators") or {})
    resolved = deepcopy(dict(recipe_payload))
    operator_overrides = dict(resolved.get("operators") or {})
    steps = _resolve_steps_from_operator_map(operators, operator_overrides)
    resolved["steps"] = steps
    resolved["operators"] = {
        key: deepcopy(operator_overrides.get(key) or {"enabled": False})
        for key in operators
    }
    return resolved


def _resolve_steps_from_operator_map(
    catalog_operators: Dict[str, Dict[str, Any]],
    operator_overrides: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    steps: list[Dict[str, Any]] = []
    unknown = sorted(set(operator_overrides) - set(catalog_operators))
    if unknown:
        raise ValueError(f"Unknown operators in recipe: {', '.join(unknown)}")

    for operator_name, operator_spec in catalog_operators.items():
        default_enabled = bool(operator_spec.get("enabled", False))
        defaults = {
            key: deepcopy(value)
            for key, value in operator_spec.items()
            if key not in {"enabled"}
        }
        override = deepcopy(dict(operator_overrides.get(operator_name) or {}))
        enabled = bool(override.pop("enabled", default_enabled))
        if not enabled:
            continue
        _validate_override_keys(defaults, override, section=operator_name)
        steps.append(
            {
                "step_type": "operator",
                "operator_ref": operator_name,
                "operator": operator_name,
                "name": operator_name,
                "enabled": True,
                "params": {**defaults, **override},
            }
        )
    return steps


def _validate_override_keys(defaults: Dict[str, Any], override: Dict[str, Any], *, section: str) -> None:
    if not defaults and override:
        raise ValueError(f"{section} does not allow parameter overrides")
    unknown = sorted(set(override) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown override keys for {section}: {', '.join(unknown)}")
