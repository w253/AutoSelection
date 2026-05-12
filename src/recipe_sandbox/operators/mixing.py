from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Sequence

from recipe_sandbox.operators.base import MixOperator
from recipe_sandbox.schema.types import CanonicalSample


class SourceMixOperator(MixOperator):
    name = "source_mix"

    def transform(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        source_weights = dict(self.config.get("source_weights") or {})
        shuffle_within_source = bool(self.config.get("shuffle_within_source", False))
        seed = self.config.get("seed")
        allow_oversample = bool(self.config.get("allow_oversample", False))

        grouped: Dict[str, List[CanonicalSample]] = defaultdict(list)
        for sample in dataset:
            grouped[sample.source_name].append(sample)

        active_sources = list(source_weights) if source_weights else list(grouped)
        active_sources = [source for source in active_sources if source in grouped]
        if not active_sources:
            self._trace.notes["selected_by_source"] = {}
            return []

        if not source_weights:
            source_weights = {source: 1.0 for source in active_sources}

        total_weight = sum(float(source_weights[source]) for source in active_sources)
        normalized_weights = {
            source: float(source_weights[source]) / total_weight for source in active_sources
        }

        available_total = sum(len(grouped[source]) for source in active_sources)
        requested_total = self.config.get("total_samples")
        total_samples = int(requested_total) if requested_total is not None else available_total
        if not allow_oversample:
            total_samples = min(total_samples, available_total)

        quotas = {source: math.floor(total_samples * normalized_weights[source]) for source in active_sources}
        remainder = total_samples - sum(quotas.values())
        ranked_remainders = sorted(
            active_sources,
            key=lambda source: total_samples * normalized_weights[source] - quotas[source],
            reverse=True,
        )
        for source in ranked_remainders:
            if remainder == 0:
                break
            quotas[source] += 1
            remainder -= 1

        if not allow_oversample:
            capped = {source: min(quotas[source], len(grouped[source])) for source in active_sources}
            shortfall = total_samples - sum(capped.values())
            quotas = capped

            while shortfall > 0:
                progressed = False
                for source in sorted(active_sources, key=lambda item: normalized_weights[item], reverse=True):
                    if quotas[source] < len(grouped[source]):
                        quotas[source] += 1
                        shortfall -= 1
                        progressed = True
                    if shortfall == 0:
                        break
                if not progressed:
                    break

        randomizer = random.Random(seed)
        outputs: List[CanonicalSample] = []
        selected_by_source: Dict[str, int] = {}
        for source in active_sources:
            candidates = list(grouped[source])
            if shuffle_within_source:
                randomizer.shuffle(candidates)

            quota = quotas[source]
            if allow_oversample and quota > len(candidates) and candidates:
                picked = [candidates[index % len(candidates)] for index in range(quota)]
            else:
                picked = candidates[:quota]
            outputs.extend(picked)
            selected_by_source[source] = len(picked)

        self._trace.notes["source_weights"] = normalized_weights
        self._trace.notes["selected_by_source"] = selected_by_source
        self._trace.cost.extra["requested_total_samples"] = requested_total
        self._trace.cost.extra["output_total_samples"] = len(outputs)
        return outputs


class TruncateSamplesOperator(SourceMixOperator):
    """Hard-cap the dataset to a fixed number of samples.

    This keeps the existing mix-style input/output contract (same params and trace
    shape through ``SourceMixOperator``) while making the operator intent explicit:
    it is a sample-count cap with randomized selection, not an entropy-aware mix.
    """

    name = "truncate_samples"

    def transform(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        if "shuffle_within_source" not in self.config:
            self.config = {**self.config, "shuffle_within_source": True}
        self._trace.notes["selection_strategy"] = "truncate_samples"
        self._trace.notes["random_sampling"] = True
        return super().transform(dataset)


class VarentropyMixOperator(SourceMixOperator):
    """Varentropy-Aware Mix (VAM): Data Mixture via Reasoning Complexity.
    Instead of uniform or arbitrary weights, domains are weighted proportionally
    to their average response varentropy (variance of token entropy).
    High expected varentropy = domain requires complex multi-path reasoning.

    Ref: LogitScope & Entropix (2024) - Varentropy as a marker for Cognitive Uncertainty.
    """

    name = "varentropy_mix"

    def transform(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        source_varentropy = defaultdict(list)
        for sample in dataset:
            ve = sample.metadata.extra.get("varentropy", {}).get("score")
            if ve is not None:
                source_varentropy[sample.source_name].append(ve)

        if not source_varentropy:
            self._trace.notes["warning"] = "No varentropy scores found in dataset. Falling back to uniform mixing."
            return super().transform(dataset)

        avg_varentropy = {
            src: sum(ves) / len(ves)
            for src, ves in source_varentropy.items() if ves
        }

        temperature = self.config.get("temperature", 1.0)
        if temperature <= 0:
            self._trace.notes["warning"] = f"Invalid temperature={temperature}, clamping to 0.01"
            temperature = 0.01

        source_weights = {}
        for src, avg_ve in avg_varentropy.items():
            try:
                source_weights[src] = math.exp(avg_ve / temperature)
            except OverflowError:
                source_weights[src] = 1e10

        self._trace.notes["varentropy_hyperparameters"] = {"temperature": temperature}
        self._trace.notes["average_varentropy"] = avg_varentropy
        self._trace.notes["computed_weights"] = source_weights

        self.config = {**self.config, "source_weights": source_weights}
        return super().transform(dataset)
