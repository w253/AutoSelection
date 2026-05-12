from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence

from recipe_sandbox.operators.types import OperatorCost, OperatorStats, OperatorTrace
from recipe_sandbox.schema.types import CanonicalSample


class BaseOperator(ABC):
    name: str = "base_operator"
    operator_type: str = "base"
    version: str = "v1"

    def __init__(self, **config: Any) -> None:
        self.config = config
        self._trace = OperatorTrace(
            operator_name=self.name,
            operator_type=self.operator_type,
            operator_version=self.version,
            config=dict(config),
        )
        self._fitted = False
        self._runtime_context: Dict[str, Any] = {}

    def fit(
        self,
        dataset: Sequence[CanonicalSample],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> "BaseOperator":
        ctx = task_context or {}
        self._runtime_context = ctx
        self._trace.notes["fit_task_context"] = self._sanitize_trace_payload(ctx)
        self._fitted = True
        self._trace.fitted = True
        return self

    @abstractmethod
    def transform(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        raise NotImplementedError

    def estimate_cost(self) -> Dict[str, Any]:
        return self._trace.cost.to_dict()

    def get_trace(self) -> Dict[str, Any]:
        return self._trace.to_dict()

    def resolve_output_stage(self, input_stage: str) -> str:
        return input_stage

    def reset_trace(self) -> None:
        self._trace = OperatorTrace(
            operator_name=self.name,
            operator_type=self.operator_type,
            operator_version=self.version,
            config=dict(self.config),
        )
        self._fitted = False
        self._runtime_context = {}

    def apply(
        self,
        dataset: Sequence[CanonicalSample],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> List[CanonicalSample]:
        if not self._fitted:
            self.fit(dataset=dataset, task_context=task_context)

        input_samples = list(dataset)
        self._trace.stats.input_samples = len(input_samples)
        self._trace.stats.input_tokens = self._count_tokens(input_samples)

        start = time.perf_counter()
        output_samples = self.transform(input_samples)
        elapsed = time.perf_counter() - start

        self._trace.stats.output_samples = len(output_samples)
        self._trace.stats.output_tokens = self._count_tokens(output_samples)
        self._trace.cost.wall_clock_sec = elapsed

        return output_samples

    def _count_tokens(self, dataset: Iterable[CanonicalSample]) -> int:
        total = 0
        for sample in dataset:
            for message in sample.messages:
                total += len(message.content.split())
            if sample.target.text:
                total += len(sample.target.text.split())
        return total

    def _sanitize_trace_payload(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, CanonicalSample):
            return {
                "sample_id": value.sample_id,
                "source_name": value.source_name,
                "task_type": value.metadata.task_type.value,
            }
        if isinstance(value, dict):
            return {
                str(key): self._sanitize_trace_payload(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            items = [self._sanitize_trace_payload(item) for item in value]
            if len(items) > 8:
                return {
                    "count": len(items),
                    "preview": items[:8],
                }
            return items
        if callable(value):
            return getattr(value, "__name__", value.__class__.__name__)
        return repr(value)

    @property
    def fitted(self) -> bool:
        return self._fitted


class MixOperator(BaseOperator):
    operator_type = "mix"


class FilterOperator(BaseOperator):
    operator_type = "filter"


class DedupOperator(BaseOperator):
    operator_type = "dedup"


class TokenCleaner(BaseOperator):
    operator_type = "clean"
