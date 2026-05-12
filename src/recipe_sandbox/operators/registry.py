from __future__ import annotations

from typing import Dict, Type

from recipe_sandbox.operators.base import BaseOperator


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: Dict[str, Type[BaseOperator]] = {}

    def register(self, operator_cls: Type[BaseOperator]) -> None:
        name = operator_cls.name
        if name in self._operators:
            raise ValueError(f"operator '{name}' is already registered")
        self._operators[name] = operator_cls

    def create(self, name: str, **config):
        if name not in self._operators:
            raise KeyError(f"operator '{name}' is not registered")
        return self._operators[name](**config)

    def get(self, name: str) -> Type[BaseOperator]:
        if name not in self._operators:
            raise KeyError(f"operator '{name}' is not registered")
        return self._operators[name]

    def list(self) -> Dict[str, Type[BaseOperator]]:
        return dict(self._operators)
