from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from recipe_sandbox.operators.helpers import resolve_path, sample_to_text
from recipe_sandbox.pipeline.recipe_executor import RecipeExecutionResult, RecipeExecutor
from recipe_sandbox.pipeline.task_config import ExperimentSearchConfig, RecipeConfig, RecipeStepConfig, TaskConfig
from recipe_sandbox.pipeline.task_manager import TaskManager
from recipe_sandbox.schema.types import CanonicalSample


@dataclass
class ExperimentMetrics:
    mean_score: float = 0.0
    coverage_ratio: float = 1.0
    source_entropy: float = 0.0
    format_integrity: float = 1.0
    wall_clock_sec: float = 0.0
    output_samples: int = 0
    input_samples: int = 0
    token_delta: int = 0
    objective_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisRecord:
    diagnosis_code: str
    severity: str
    confidence: float
    attributed_step: str
    supporting_metrics: Dict[str, Any] = field(default_factory=dict)
    delta_evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateRecord:
    recipe_name: str
    recipe_config: Dict[str, Any]
    metrics: ExperimentMetrics
    manifest_path: str
    trace_path: str
    diagnoses: List[DiagnosisRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        payload["diagnoses"] = [item.to_dict() for item in self.diagnoses]
        return payload


@dataclass
class ExperimentSearchResult:
    search_name: str
    selected_recipe_name: str
    selected_recipe_path: str
    manifest_path: str
    trace_path: str
    diagnoses_path: str
    rounds: List[Dict[str, Any]]


class ExperimentSearchController:
    def __init__(
        self,
        config: TaskConfig,
        manager: TaskManager,
        recipe_executor: Optional[RecipeExecutor] = None,
    ) -> None:
        self.config = config
        self.manager = manager
        self.recipe_executor = recipe_executor or RecipeExecutor(config, manager)

    def run(
        self,
        search_config: Optional[ExperimentSearchConfig] = None,
    ) -> ExperimentSearchResult:
        search = search_config or self.config.search
        if not search.enabled:
            raise ValueError("Experiment search requested, but search config is not enabled")

        baseline_candidates = self._baseline_candidates(search)
        identity = baseline_candidates[0]
        seen_signatures = {self._recipe_signature(candidate) for candidate in baseline_candidates}
        rounds: List[Dict[str, Any]] = []

        round_zero_results = self._evaluate_candidates(baseline_candidates, search, round_index=0)
        best = max(round_zero_results, key=lambda item: item.metrics.objective_score)
        rounds.append(
            {
                "round_index": 0,
                "mode": "baseline",
                "selected_recipe_name": best.recipe_name,
                "candidates": [candidate.to_dict() for candidate in round_zero_results],
            }
        )

        stagnation = 0
        current = best
        identity_result = next(
            item
            for item in round_zero_results
            if not (item.recipe_config.get("steps") or [])
        )
        diagnoses_output: List[Dict[str, Any]] = []

        for round_index in range(1, max(1, int(search.max_rounds))):
            diagnoses = self._diagnose_candidate(current, reference=identity_result, search=search)
            diagnoses_output.append(
                {
                    "round_index": round_index,
                    "recipe_name": current.recipe_name,
                    "diagnoses": [item.to_dict() for item in diagnoses],
                }
            )
            if self._should_stop(diagnoses, stagnation=stagnation, round_index=round_index, search=search):
                break

            proposals = self._generate_local_proposals(current, diagnoses, search=search)
            filtered_proposals: List[RecipeConfig] = []
            for proposal in proposals:
                signature = self._recipe_signature(proposal)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                filtered_proposals.append(proposal)
                if len(filtered_proposals) >= int(search.max_candidates_per_round):
                    break
            if not filtered_proposals:
                break

            proposal_results = self._evaluate_candidates(filtered_proposals, search, round_index=round_index)
            winner = max(proposal_results + [current], key=lambda item: item.metrics.objective_score)
            improved = winner.recipe_name != current.recipe_name and (
                winner.metrics.objective_score - current.metrics.objective_score
            ) >= float(search.min_gain)
            if improved:
                current = winner
                best = max([best, winner], key=lambda item: item.metrics.objective_score)
                stagnation = 0
            else:
                stagnation += 1

            rounds.append(
                {
                    "round_index": round_index,
                    "mode": "local_revision",
                    "selected_recipe_name": current.recipe_name,
                    "candidates": [candidate.to_dict() for candidate in proposal_results],
                    "diagnoses": [item.to_dict() for item in diagnoses],
                }
            )

        final_recipe_payload = dict(best.recipe_config)
        manifest_payload = {
            "search_name": search.search_name,
            "selected_recipe_name": best.recipe_name,
            "rounds": len(rounds),
            "max_rounds": search.max_rounds,
            "input_stage": search.input_stage,
            "input_split": search.input_split,
        }
        trace_payload = {
            "search_name": search.search_name,
            "rounds": rounds,
            "selected_recipe_name": best.recipe_name,
            "selected_metrics": best.metrics.to_dict(),
        }
        diagnoses_payload = {
            "search_name": search.search_name,
            "records": diagnoses_output,
        }

        manifest_path = self.manager.write_search_manifest(search.search_name, manifest_payload)
        trace_path = self.manager.write_search_trace(search.search_name, trace_payload)
        diagnoses_path = self.manager.write_search_diagnoses(search.search_name, diagnoses_payload)
        selected_recipe_path = self.manager.write_search_recipe(search.search_name, final_recipe_payload)
        return ExperimentSearchResult(
            search_name=search.search_name,
            selected_recipe_name=best.recipe_name,
            selected_recipe_path=str(selected_recipe_path),
            manifest_path=str(manifest_path),
            trace_path=str(trace_path),
            diagnoses_path=str(diagnoses_path),
            rounds=rounds,
        )

    def _baseline_candidates(self, search: ExperimentSearchConfig) -> List[RecipeConfig]:
        score_path = self._score_path(search)
        return [
            self._recipe_config(search, "identity", []),
            self._recipe_config(
                search,
                "mix_only",
                [RecipeStepConfig(operator="source_mix", name="mix", params={"shuffle_within_source": True, "seed": 13})],
            ),
            self._recipe_config(search, "filter_only", [self._default_filter_step(search, score_path)]),
            self._recipe_config(
                search,
                "dedup_only",
                [RecipeStepConfig(operator="semantic_dedup", name="dedup", params={"strategy": "exact"})],
            ),
            self._recipe_config(
                search,
                "clean_only",
                [RecipeStepConfig(operator="rule_based_cleaner", name="clean", params={"normalize_whitespace": True})],
            ),
            self._recipe_config(
                search,
                "mix_filter",
                [
                    RecipeStepConfig(operator="source_mix", name="mix", params={"shuffle_within_source": True, "seed": 13}),
                    self._default_filter_step(search, score_path),
                ],
            ),
        ]

    def _default_filter_step(self, search: ExperimentSearchConfig, score_path: Optional[str]) -> RecipeStepConfig:
        if score_path:
            return RecipeStepConfig(
                operator="path_filter",
                name="filter",
                params={
                    "score_path": score_path,
                    "fraction": 0.5,
                },
            )
        method = str(self.config.scoring.method or "mona")
        operator_name = "mona_filter"
        return RecipeStepConfig(
            operator=operator_name,
            name="filter",
            params={
                "fraction": 0.5,
            },
        )

    def _recipe_config(self, search: ExperimentSearchConfig, suffix: str, steps: List[RecipeStepConfig]) -> RecipeConfig:
        return RecipeConfig(
            enabled=True,
            recipe_name=f"{search.search_name}_{suffix}",
            input_split=search.input_split,
            input_stage=search.input_stage,
            steps=steps,
            task_context={"search_name": search.search_name},
        )

    def _evaluate_candidates(
        self,
        candidates: Sequence[RecipeConfig],
        search: ExperimentSearchConfig,
        *,
        round_index: int,
    ) -> List[CandidateRecord]:
        records: List[CandidateRecord] = []
        for candidate in candidates:
            recipe_name = f"{search.search_name}_r{round_index:02d}_{candidate.recipe_name}"
            candidate = copy.deepcopy(candidate)
            candidate.recipe_name = recipe_name
            if candidate.steps:
                result = self.recipe_executor.run(
                    recipe_config=candidate,
                    task_context_override={"search_round": round_index},
                )
                samples = self._load_recipe_samples(result.recipe_name)
                metrics = self._compute_metrics(samples, result, search)
                trace_path = result.trace_path
                manifest_path = result.manifest_path
            else:
                result, samples = self._run_identity_candidate(candidate)
                metrics = self._compute_metrics(samples, result, search)
                trace_path = result.trace_path
                manifest_path = result.manifest_path
            records.append(
                CandidateRecord(
                    recipe_name=recipe_name,
                    recipe_config=asdict(candidate),
                    metrics=metrics,
                    manifest_path=manifest_path,
                    trace_path=trace_path,
                )
            )
        return records

    def _run_identity_candidate(self, candidate: RecipeConfig) -> tuple[RecipeExecutionResult, List[CanonicalSample]]:
        samples = self.recipe_executor._load_input_samples(candidate.input_stage, candidate.input_split)  # noqa: SLF001
        output_path = self.manager.write_recipe_dataset(candidate.recipe_name, samples)
        trace_payload = {
            "task_name": self.config.task_name,
            "recipe_name": candidate.recipe_name,
            "input_stage": candidate.input_stage,
            "input_split": candidate.input_split,
            "input_samples": len(samples),
            "output_samples": len(samples),
            "total_wall_clock_sec": 0.0,
            "steps": [],
        }
        manifest_payload = {
            "task_name": self.config.task_name,
            "recipe_name": candidate.recipe_name,
            "input_stage": candidate.input_stage,
            "input_split": candidate.input_split,
            "input_samples": len(samples),
            "output_samples": len(samples),
            "num_steps": 0,
            "sources": dict(sorted(Counter(sample.source_name for sample in samples).items())),
            "task_types": dict(sorted(Counter(sample.metadata.task_type.value for sample in samples).items())),
            "step_names": [],
        }
        trace_path = self.manager.write_recipe_trace(candidate.recipe_name, trace_payload)
        manifest_path = self.manager.write_recipe_manifest(candidate.recipe_name, manifest_payload)
        result = RecipeExecutionResult(
            recipe_name=candidate.recipe_name,
            input_samples=len(samples),
            output_samples=len(samples),
            output_path=str(output_path),
            manifest_path=str(manifest_path),
            trace_path=str(trace_path),
            step_traces=[],
        )
        return result, samples

    def _load_recipe_samples(self, recipe_name: str) -> List[CanonicalSample]:
        from recipe_sandbox.schema.io import read_jsonl

        return list(read_jsonl(str(self.manager.recipe_dataset_path(recipe_name))))

    def _compute_metrics(
        self,
        samples: Sequence[CanonicalSample],
        result: RecipeExecutionResult,
        search: ExperimentSearchConfig,
    ) -> ExperimentMetrics:
        input_samples = max(1, int(result.input_samples))
        output_samples = int(result.output_samples)
        score_path = self._score_path(search)
        scores: List[float] = []
        for sample in samples:
            value = resolve_path(sample, score_path) if score_path else None
            if value is not None:
                scores.append(float(value))
        mean_score = sum(scores) / len(scores) if scores else 0.0
        coverage_ratio = output_samples / input_samples
        source_entropy = self._source_entropy(samples)
        format_integrity = self._format_integrity(samples)
        wall_clock_sec = sum(float(step["trace"]["cost"].get("wall_clock_sec") or 0.0) for step in result.step_traces)
        token_delta = sum(
            int(step["trace"]["stats"].get("token_delta") or 0)
            for step in result.step_traces
        )
        objective_score = (
            mean_score
            + 0.05 * source_entropy
            + 0.05 * format_integrity
            - 0.12 * max(0.0, 0.7 - coverage_ratio)
            - 0.01 * wall_clock_sec
        )
        return ExperimentMetrics(
            mean_score=round(mean_score, 6),
            coverage_ratio=round(coverage_ratio, 6),
            source_entropy=round(source_entropy, 6),
            format_integrity=round(format_integrity, 6),
            wall_clock_sec=round(wall_clock_sec, 6),
            output_samples=output_samples,
            input_samples=input_samples,
            token_delta=token_delta,
            objective_score=round(objective_score, 6),
        )

    def _diagnose_candidate(
        self,
        candidate: CandidateRecord,
        *,
        reference: CandidateRecord,
        search: ExperimentSearchConfig,
    ) -> List[DiagnosisRecord]:
        recipe_steps = list(candidate.recipe_config.get("steps") or [])
        step_names = [str(step.get("name") or step.get("operator") or "") for step in recipe_steps]
        diagnoses: List[DiagnosisRecord] = []
        score_delta = candidate.metrics.mean_score - reference.metrics.mean_score
        coverage_drop = reference.metrics.coverage_ratio - candidate.metrics.coverage_ratio
        entropy_drop = reference.metrics.source_entropy - candidate.metrics.source_entropy
        format_drop = reference.metrics.format_integrity - candidate.metrics.format_integrity
        cost_ratio = 0.0
        if reference.metrics.wall_clock_sec > 0:
            cost_ratio = (candidate.metrics.wall_clock_sec - reference.metrics.wall_clock_sec) / reference.metrics.wall_clock_sec
        elif candidate.metrics.wall_clock_sec > 0:
            cost_ratio = 1.0

        if score_delta < -float(search.min_gain):
            diagnoses.append(
                DiagnosisRecord(
                    diagnosis_code="LOW_TASK_FIT",
                    severity="high",
                    confidence=0.8,
                    attributed_step=step_names[-1] if step_names else "identity",
                    supporting_metrics={"mean_score": candidate.metrics.mean_score, "reference_score": reference.metrics.mean_score},
                    delta_evidence={"score_delta": round(score_delta, 6)},
                    suggested_actions=["tighten filter", "rollback", "stop"],
                )
            )
        if any("filter" in step.get("operator", "") for step in recipe_steps) and coverage_drop > 0.2 and score_delta < float(search.min_gain):
            diagnoses.append(
                DiagnosisRecord(
                    diagnosis_code="OVER_FILTER",
                    severity="medium",
                    confidence=0.75,
                    attributed_step=self._first_matching_step(step_names, recipe_steps, "filter"),
                    supporting_metrics={"coverage_ratio": candidate.metrics.coverage_ratio, "reference_coverage": reference.metrics.coverage_ratio},
                    delta_evidence={"coverage_drop": round(coverage_drop, 6), "score_delta": round(score_delta, 6)},
                    suggested_actions=["relax filter", "rollback"],
                )
            )
        if any("dedup" in step.get("operator", "") for step in recipe_steps) and (coverage_drop > 0.15 or entropy_drop > 0.1):
            diagnoses.append(
                DiagnosisRecord(
                    diagnosis_code="OVER_DEDUP",
                    severity="medium",
                    confidence=0.72,
                    attributed_step=self._first_matching_step(step_names, recipe_steps, "dedup"),
                    supporting_metrics={"coverage_ratio": candidate.metrics.coverage_ratio, "source_entropy": candidate.metrics.source_entropy},
                    delta_evidence={"coverage_drop": round(coverage_drop, 6), "entropy_drop": round(entropy_drop, 6)},
                    suggested_actions=["relax dedup", "rollback"],
                )
            )
        if any("clean" in step.get("operator", "") for step in recipe_steps) and format_drop > 0.05:
            diagnoses.append(
                DiagnosisRecord(
                    diagnosis_code="FORMAT_DAMAGE",
                    severity="medium",
                    confidence=0.7,
                    attributed_step=self._first_matching_step(step_names, recipe_steps, "clean"),
                    supporting_metrics={"format_integrity": candidate.metrics.format_integrity},
                    delta_evidence={"format_drop": round(format_drop, 6)},
                    suggested_actions=["relax cleaning", "rollback"],
                )
            )
        if cost_ratio > float(search.max_cost_increase_ratio) and score_delta < float(search.min_gain):
            diagnoses.append(
                DiagnosisRecord(
                    diagnosis_code="HIGH_COST_LOW_GAIN",
                    severity="high",
                    confidence=0.85,
                    attributed_step=step_names[-1] if step_names else "identity",
                    supporting_metrics={"wall_clock_sec": candidate.metrics.wall_clock_sec, "reference_wall_clock_sec": reference.metrics.wall_clock_sec},
                    delta_evidence={"cost_ratio": round(cost_ratio, 6), "score_delta": round(score_delta, 6)},
                    suggested_actions=["rollback", "stop"],
                )
            )
        return diagnoses

    def _generate_local_proposals(
        self,
        current: CandidateRecord,
        diagnoses: Sequence[DiagnosisRecord],
        *,
        search: ExperimentSearchConfig,
    ) -> List[RecipeConfig]:
        base = RecipeConfig(**{**current.recipe_config, "steps": [RecipeStepConfig(**step) for step in current.recipe_config.get("steps", [])]})
        proposals: List[RecipeConfig] = []
        if not diagnoses and len(base.steps) < 2 and not any("filter" in step.operator for step in base.steps):
            score_path = self._score_path(search)
            base_plus_filter = copy.deepcopy(base)
            base_plus_filter.steps.append(self._default_filter_step(search, score_path))
            base_plus_filter.recipe_name = f"{search.search_name}_add_filter"
            proposals.append(base_plus_filter)
            return proposals

        for diagnosis in diagnoses:
            if diagnosis.diagnosis_code == "OVER_FILTER":
                relaxed = self._mutate_filter_fraction(base, delta=0.1, suffix="relax_filter")
                if relaxed is not None:
                    proposals.append(relaxed)
            elif diagnosis.diagnosis_code == "OVER_DEDUP":
                rollback = self._remove_operator(base, "dedup", suffix="rollback_dedup")
                if rollback is not None:
                    proposals.append(rollback)
            elif diagnosis.diagnosis_code == "FORMAT_DAMAGE":
                rollback = self._remove_operator(base, "clean", suffix="rollback_clean")
                if rollback is not None:
                    proposals.append(rollback)
            elif diagnosis.diagnosis_code == "LOW_TASK_FIT":
                tightened = self._mutate_filter_fraction(base, delta=-0.1, suffix="tighten_filter")
                if tightened is not None:
                    proposals.append(tightened)
                elif not any("filter" in step.operator for step in base.steps):
                    plus_filter = copy.deepcopy(base)
                    plus_filter.steps.append(self._default_filter_step(search, self._score_path(search)))
                    plus_filter.recipe_name = f"{search.search_name}_add_filter"
                    proposals.append(plus_filter)
        return proposals

    def _mutate_filter_fraction(self, recipe: RecipeConfig, *, delta: float, suffix: str) -> Optional[RecipeConfig]:
        updated = copy.deepcopy(recipe)
        for step in updated.steps:
            if "filter" not in step.operator:
                continue
            fraction = float(step.params.get("fraction", 0.8))
            next_fraction = min(0.95, max(0.5, fraction + delta))
            if math.isclose(next_fraction, fraction):
                return None
            step.params["fraction"] = next_fraction
            updated.recipe_name = f"{recipe.recipe_name}_{suffix}"
            return updated
        return None

    def _remove_operator(self, recipe: RecipeConfig, operator_token: str, *, suffix: str) -> Optional[RecipeConfig]:
        remaining = [step for step in recipe.steps if operator_token not in step.operator]
        if len(remaining) == len(recipe.steps):
            return None
        updated = copy.deepcopy(recipe)
        updated.steps = remaining
        updated.recipe_name = f"{recipe.recipe_name}_{suffix}"
        return updated

    def _should_stop(
        self,
        diagnoses: Sequence[DiagnosisRecord],
        *,
        stagnation: int,
        round_index: int,
        search: ExperimentSearchConfig,
    ) -> bool:
        if round_index >= int(search.max_rounds):
            return True
        if stagnation >= int(search.stagnation_rounds):
            return True
        if any(item.diagnosis_code == "HIGH_COST_LOW_GAIN" and "stop" in item.suggested_actions for item in diagnoses):
            return True
        return False

    def _score_path(self, search: ExperimentSearchConfig) -> Optional[str]:
        if search.score_path:
            return search.score_path
        method = str(self.config.scoring.method or "mona")
        task_label = search.task_label or self.config.task_name
        if method == "mona":
            return f"metadata.extra.mona_scores.{task_label}"
        if method == "less":
            return "metadata.extra.less.influence_score"
        return None

    def _recipe_signature(self, recipe: RecipeConfig) -> str:
        payload = asdict(recipe)
        payload["recipe_name"] = "__signature__"
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def _source_entropy(self, samples: Sequence[CanonicalSample]) -> float:
        if not samples:
            return 0.0
        counts = Counter(sample.source_name for sample in samples)
        total = sum(counts.values()) or 1
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability + 1e-12)
        return entropy

    def _format_integrity(self, samples: Sequence[CanonicalSample]) -> float:
        if not samples:
            return 0.0
        valid = 0
        for sample in samples:
            has_messages = bool(sample.messages) and all(message.content.strip() for message in sample.messages)
            has_target = sample.target.text is None or bool((sample.target.text or "").strip())
            valid += 1 if has_messages and has_target else 0
        return valid / len(samples)

    def _first_matching_step(
        self,
        step_names: Sequence[str],
        recipe_steps: Sequence[Dict[str, Any]],
        token: str,
    ) -> str:
        for name, step in zip(step_names, recipe_steps):
            if token in str(step.get("operator", "")):
                return name or str(step.get("operator", token))
        return token
