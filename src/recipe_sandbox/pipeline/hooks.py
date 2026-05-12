from __future__ import annotations

import logging
from typing import Any, Iterable, List

logger = logging.getLogger(__name__)


class RecipeHookManager:
    """Dispatch optional lifecycle hooks around recipe execution.

    Hook objects can implement any subset of these keyword-only methods:

    - before_recipe(recipe, bus, state)
    - after_recipe(recipe, result)
    - before_step(recipe, step, step_index, operator, bus, state_before, step_context)
    - after_step(recipe, step, step_index, operator, bus_before, bus_after, step_trace)
    - on_step_error(recipe, step, step_index, bus, error)
    """

    def __init__(self, hooks: Iterable[Any] | None = None) -> None:
        self._hooks: List[Any] = list(hooks or [])

    @property
    def hooks(self) -> List[Any]:
        return list(self._hooks)

    def before_recipe(self, **kwargs: Any) -> None:
        self._dispatch("before_recipe", **kwargs)

    def after_recipe(self, **kwargs: Any) -> None:
        self._dispatch("after_recipe", **kwargs)

    def before_step(self, **kwargs: Any) -> None:
        self._dispatch("before_step", **kwargs)

    def after_step(self, **kwargs: Any) -> None:
        self._dispatch("after_step", **kwargs)

    def on_step_error(self, **kwargs: Any) -> None:
        self._dispatch("on_step_error", **kwargs)

    def _dispatch(self, method_name: str, **kwargs: Any) -> None:
        for hook in self._hooks:
            callback = getattr(hook, method_name, None)
            if callback is None:
                continue
            callback(**kwargs)


class LoggingRecipeHook:
    """Tiny example hook useful for extension smoke tests."""

    def before_step(
        self,
        *,
        step_index: int,
        operator: Any,
        bus: Any,
        **_: Any,
    ) -> None:
        logger.info(
            "Hook before_step: #%d %s on %d samples",
            step_index,
            getattr(operator, "name", operator.__class__.__name__),
            len(getattr(bus, "samples", [])),
        )
