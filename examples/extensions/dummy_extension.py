from __future__ import annotations

import logging
from typing import Any, Dict, List

from recipe_sandbox.operators.base import FilterOperator
from recipe_sandbox.operators.helpers import sample_to_text
from recipe_sandbox.schema.types import CanonicalSample

logger = logging.getLogger(__name__)

FEATURE_KEY = "example_length"
OPERATOR_NAME = "example_length_filter"


class ExampleLengthFilter(FilterOperator):
    """Small extension operator used by tests and onboarding docs."""

    name = OPERATOR_NAME
    version = "v1"

    def transform(self, dataset):
        max_chars = int(self.config.get("max_chars", 4096))
        keep_missing = bool(self.config.get("keep_missing", True))
        kept = []
        for sample in dataset:
            feature = sample.metadata.extra.get(FEATURE_KEY, {})
            length = feature.get("chars")
            if length is None:
                if not keep_missing:
                    continue
                length = len(sample_to_text(sample))
            if int(length) <= max_chars:
                kept.append(sample)
        return kept


class ExampleAuditHook:
    def after_step(self, *, step_index: int, operator: Any, bus_after: Any, **_: Any) -> None:
        logger.info(
            "ExampleAuditHook: step=%d operator=%s output_samples=%d",
            step_index,
            getattr(operator, "name", operator.__class__.__name__),
            len(getattr(bus_after, "samples", [])),
        )


def register_operators(registry) -> None:
    registry.register(ExampleLengthFilter)


def precompute_features(
    *,
    samples: List[CanonicalSample],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    del context
    for sample in samples:
        sample.metadata.extra[FEATURE_KEY] = {
            "chars": len(sample_to_text(sample)),
        }
    return {
        "feature_key": FEATURE_KEY,
        "samples": len(samples),
    }


RECIPE_HOOKS = [ExampleAuditHook()]


OPERATOR_CATALOG_PATCH = {
    "families": {
        "example_extensions": {
            "operators": {
                OPERATOR_NAME: {
                    "description": (
                        "Example extension operator that keeps samples whose "
                        "precomputed text length is below max_chars."
                    ),
                    "params": {
                        "max_chars": {
                            "type": "int",
                            "default": 4096,
                            "range": [1, 100000],
                            "description": "Maximum combined prompt/target character length to keep.",
                        },
                        "keep_missing": {
                            "type": "bool",
                            "default": True,
                            "description": "Compute length on the fly if cold-start precompute did not run.",
                        },
                    },
                },
            },
        },
    },
}
