from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

from recipe_sandbox.operators import (
    BaseOperator,
    SourceMixOperator,
    TruncateSamplesOperator,
    VarentropyMixOperator,
    MonaFilterOperator,
    QualityFilterOperator,
    SparseMonaFilterOperator,
    IFDFilterOperator,
    SemanticDedupOperator,
    SemDeDupOperator,
    UnionOperator,
    OperatorRegistry,
    ScoreFilterBase,
    NGramEntropyFilterOperator,
    ActionObjectBranchingFilterOperator,
    VarentropyFilterOperator,
)
from recipe_sandbox.pipeline.hooks import RecipeHookManager
from recipe_sandbox.pipeline.task_config import RecipeConfig, RecipeStepConfig, TaskConfig
from recipe_sandbox.pipeline.task_manager import TaskManager
from recipe_sandbox.schema.io import read_jsonl
from recipe_sandbox.schema.types import CanonicalSample
from recipe_sandbox.feedback.state_vector import DataStateComputer, DataStateVector, vector_delta

MIN_VIABLE_SAMPLES = 500


def build_default_operator_registry(
    extra_operators: Optional[Iterable[Type[BaseOperator]]] = None,
) -> OperatorRegistry:
    registry = OperatorRegistry()
    registry.register_many([
        # F1: Source Mixing
        SourceMixOperator,
        TruncateSamplesOperator,
        VarentropyMixOperator,
        # F2: Task Relevance Selection
        MonaFilterOperator,
        SparseMonaFilterOperator,
        # F3: Quality Selection
        QualityFilterOperator,
        IFDFilterOperator,
        NGramEntropyFilterOperator,
        ActionObjectBranchingFilterOperator,
        # F6: Complexity Selection
        VarentropyFilterOperator,
        # F4: Dedup / Redundancy Control
        SemanticDedupOperator,
        SemDeDupOperator,
        # F5: Set Operations
        UnionOperator,
    ])
    if extra_operators:
        registry.register_many(extra_operators)
    return registry


@dataclass
class RecipeDataBus:
    samples: List[CanonicalSample]
    stage: str
    split: str
    task_context: Dict[str, Any]


@dataclass
class RecipeExecutionResult:
    recipe_name: str
    input_samples: int
    output_samples: int
    output_path: str
    manifest_path: str
    trace_path: str
    step_traces: List[Dict[str, Any]]
    initial_state: Optional[Dict[str, Any]] = None
    final_state: Optional[Dict[str, Any]] = None


class RecipeExecutor:
    def __init__(
        self,
        config: TaskConfig,
        manager: TaskManager,
        registry: Optional[OperatorRegistry] = None,
        sparse_cache: Optional[Any] = None,
        cached_train_samples: Optional[List[CanonicalSample]] = None,
        hooks: Optional[Sequence[Any]] = None,
    ) -> None:
        self.config = config
        self.manager = manager
        self.registry = registry or build_default_operator_registry()
        self._sparse_cache = sparse_cache  # SparseFeatureCache or None
        self._cached_train_samples = cached_train_samples  # in-memory samples with all numeric features
        self.hooks = RecipeHookManager(hooks)

    def input_source_names(self, stage: str = "canonical", split: str = "train") -> List[str]:
        samples = self._load_input_samples(stage, split)
        return sorted({sample.source_name for sample in samples})

    def run(
        self,
        recipe_config: Optional[RecipeConfig] = None,
        task_context_override: Optional[Dict[str, Any]] = None,
    ) -> RecipeExecutionResult:
        recipe = recipe_config or self.config.recipe
        if recipe is None or not recipe.enabled:
            raise ValueError("Recipe execution requested, but recipe config is not enabled")
        if not recipe.steps:
            raise ValueError("Recipe execution requested, but no recipe steps were configured")

        samples = self._load_input_samples(recipe.input_stage, recipe.input_split)
        input_samples = len(samples)
        task_context = {
            "task_name": self.config.task_name,
            "recipe_name": recipe.recipe_name,
            "input_split": recipe.input_split,
            "base_task_config": self.config,
            "base_task_manager": self.manager,
            "sparse_cache": self._sparse_cache,
            "pool_samples": samples,
            **dict(recipe.task_context),
        }
        if task_context_override:
            task_context.update(task_context_override)
        bus = RecipeDataBus(
            samples=samples,
            stage=recipe.input_stage,
            split=recipe.input_split,
            task_context=task_context,
        )
        state_computer = DataStateComputer(
            reference_dataset=samples,
            sparse_cache=self._sparse_cache,
            d_sae=self.config.model.d_sae if self.config.model else None,
        )
        initial_state = state_computer.compute(samples)
        self.hooks.before_recipe(recipe=recipe, bus=bus, state=initial_state)

        step_traces: List[Dict[str, Any]] = []
        self.manager.log(
            f"Executing recipe '{recipe.recipe_name}' with {len(recipe.steps)} configured step(s) on {input_samples} samples"
        )
        self.manager.log(f"Initial state vector: {_format_state_vector(initial_state)}")

        for index, step in enumerate(recipe.steps, start=1):
            if not step.enabled:
                continue
            bus, initial_state = self._apply_step(
                bus=bus,
                recipe=recipe,
                step=step,
                step_index=index,
                step_traces=step_traces,
                state_computer=state_computer,
                initial_state=initial_state,
            )
            if len(bus.samples) < MIN_VIABLE_SAMPLES:
                self.manager.log(
                    f"WARNING: Step {index} ({step.operator}) left only "
                    f"{len(bus.samples)} samples (< MIN_VIABLE={MIN_VIABLE_SAMPLES}). "
                    f"Halting remaining filters."
                )
                break
        output_path = self.manager.write_recipe_dataset_ids(recipe.recipe_name, bus.samples)

        final_state = state_computer.compute(bus.samples)
        initial_state_payload = step_traces[0]["state_before"] if step_traces else initial_state.to_dict()

        trace_payload = self._build_trace_payload(recipe, input_samples, bus.samples, step_traces)
        trace_path = self.manager.write_recipe_trace(recipe.recipe_name, trace_payload)
        manifest_payload = self._build_manifest_payload(recipe, input_samples, bus.samples, step_traces)
        manifest_path = self.manager.write_recipe_manifest(recipe.recipe_name, manifest_payload)

        self.manager.log(
            f"Recipe '{recipe.recipe_name}' completed: {input_samples} -> {len(bus.samples)} samples"
        )
        self.manager.log(f"Final state vector: {_format_state_vector(final_state)}")

        result = RecipeExecutionResult(
            recipe_name=recipe.recipe_name,
            input_samples=input_samples,
            output_samples=len(bus.samples),
            output_path=str(output_path),
            manifest_path=str(manifest_path),
            trace_path=str(trace_path),
            step_traces=step_traces,
            initial_state=initial_state_payload,
            final_state=final_state.to_dict(),
        )
        self.hooks.after_recipe(recipe=recipe, result=result)
        return result

    def _load_input_samples(self, stage: str, split: str, manager: Optional[TaskManager] = None) -> List[CanonicalSample]:
        # Fast path: return in-memory cached samples (already have all numeric features)
        if self._cached_train_samples is not None and stage == "canonical" and split == "train":
            self.manager.log(f"Loaded {len(self._cached_train_samples)} canonical samples from in-memory cache (no disk read)")
            return list(self._cached_train_samples)

        active_manager = manager or self.manager
        if stage == "canonical":
            files = active_manager.list_canonical(split)
        elif stage == "scored":
            files = active_manager.list_scored(split)
        else:
            raise ValueError(f"Unsupported recipe input_stage: {stage}")

        if not files:
            raise ValueError(
                f"No {stage} files found for split '{split}'. "
                "Run the prerequisite pipeline stages first."
            )

        samples: List[CanonicalSample] = []
        for path in files:
            loaded = list(read_jsonl(str(path)))
            self.manager.log(f"Loaded {len(loaded)} {stage} samples from {path}")
            samples.extend(loaded)

        return samples

    def _apply_step(
        self,
        *,
        bus: RecipeDataBus,
        recipe: RecipeConfig,
        step: RecipeStepConfig,
        step_index: int,
        step_traces: List[Dict[str, Any]],
        state_computer: DataStateComputer,
        initial_state: DataStateVector,
    ) -> tuple[RecipeDataBus, DataStateVector]:
        try:
            return self._apply_step_impl(
                bus=bus,
                recipe=recipe,
                step=step,
                step_index=step_index,
                step_traces=step_traces,
                state_computer=state_computer,
                initial_state=initial_state,
            )
        except Exception as error:
            self.hooks.on_step_error(
                recipe=recipe,
                step=step,
                step_index=step_index,
                bus=bus,
                error=error,
            )
            raise

    def _apply_step_impl(
        self,
        *,
        bus: RecipeDataBus,
        recipe: RecipeConfig,
        step: RecipeStepConfig,
        step_index: int,
        step_traces: List[Dict[str, Any]],
        state_computer: DataStateComputer,
        initial_state: DataStateVector,
    ) -> tuple[RecipeDataBus, DataStateVector]:
        operator_name = step.resolved_operator_name
        step_label = step.name or operator_name
        operator = self.registry.create(operator_name, **step.resolved_operator_config)
        step_context = {
            **dict(bus.task_context),
            "current_stage": bus.stage,
            "recipe_name": recipe.recipe_name,
            "input_split": bus.split,
            "step_index": step_index,
            "step_name": step_label,
        }
        if isinstance(operator, ScoreFilterBase):
            operator.fit(bus.samples, task_context=step_context)
            if not step_traces:
                initial_state = state_computer.compute(bus.samples)
        state_before = state_computer.compute(bus.samples)
        self.manager.log(
            f"Recipe step {step_index}: {step_label} ({operator_name}) on {len(bus.samples)} {bus.stage} samples"
        )
        self.hooks.before_step(
            recipe=recipe,
            step=step,
            step_index=step_index,
            operator=operator,
            bus=bus,
            state_before=state_before,
            step_context=step_context,
        )

        outputs = operator.apply(bus.samples, task_context=step_context)
        state_after = state_computer.compute(outputs)
        next_stage = operator.resolve_output_stage(bus.stage)
        trace = operator.get_trace()
        step_trace = {
            "step_index": step_index,
            "step_name": step_label,
            "step_type": trace.get("operator_type", step.resolved_step_type),
            "operator": operator_name,
            "state_before": state_before.to_dict(),
            "state_after": state_after.to_dict(),
            "delta_loc": vector_delta(state_after, state_before),
            "delta_glob": vector_delta(state_after, initial_state),
            "trace": trace,
        }
        step_traces.append(step_trace)
        self.manager.log(
            f"Recipe step {step_index} completed: {trace['stats']['input_samples']} -> {trace['stats']['output_samples']}"
        )
        self.manager.log(f"  State BEFORE: {_format_state_vector(state_before)}")
        self.manager.log(f"  State AFTER:  {_format_state_vector(state_after)}")
        self.manager.log(f"  Δ_loc:  {_format_delta(step_trace['delta_loc'])}")
        self.manager.log(f"  Δ_glob: {_format_delta(step_trace['delta_glob'])}")
        next_bus = RecipeDataBus(
            samples=outputs,
            stage=next_stage,
            split=bus.split,
            task_context=dict(bus.task_context),
        )
        self.hooks.after_step(
            recipe=recipe,
            step=step,
            step_index=step_index,
            operator=operator,
            bus_before=bus,
            bus_after=next_bus,
            step_trace=step_trace,
        )
        return next_bus, initial_state

    def _build_trace_payload(
        self,
        recipe: RecipeConfig,
        input_samples: int,
        output_samples: List[CanonicalSample],
        step_traces: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        total_wall_clock = sum(
            float(step["trace"]["cost"].get("wall_clock_sec") or 0.0) for step in step_traces
        )
        return {
            "task_name": self.config.task_name,
            "recipe_name": recipe.recipe_name,
            "input_stage": recipe.input_stage,
            "input_split": recipe.input_split,
            "input_samples": input_samples,
            "output_samples": len(output_samples),
            "initial_state": step_traces[0]["state_before"] if step_traces else None,
            "final_state": step_traces[-1]["state_after"] if step_traces else None,
            "total_wall_clock_sec": total_wall_clock,
            "steps": step_traces,
        }

    def _build_manifest_payload(
        self,
        recipe: RecipeConfig,
        input_samples: int,
        output_samples: List[CanonicalSample],
        step_traces: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_counts = Counter(sample.source_name for sample in output_samples)
        task_types = Counter(sample.metadata.task_type.value for sample in output_samples)
        return {
            "task_name": self.config.task_name,
            "recipe_name": recipe.recipe_name,
            "input_stage": recipe.input_stage,
            "input_split": recipe.input_split,
            "input_samples": input_samples,
            "output_samples": len(output_samples),
            "num_steps": len(step_traces),
            "sources": dict(sorted(source_counts.items())),
            "task_types": dict(sorted(task_types.items())),
            "step_names": [step["step_name"] for step in step_traces],
        }


# ---------------------------------------------------------------------------
#  Formatting helpers for log output
# ---------------------------------------------------------------------------

def _format_state_vector(state: DataStateVector) -> str:
    """One-line compact representation of a state vector."""
    d = state.to_dict()
    parts = []
    for k, v in d.items():
        if isinstance(v, dict):
            inner = ", ".join(f"{sk}={sv:.4f}" for sk, sv in v.items())
            parts.append(f"{k}={{{inner}}}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return "{" + ", ".join(parts) + "}"


def _format_delta(delta: Dict[str, float]) -> str:
    """One-line compact delta, showing only non-zero changes."""
    parts = []
    for k, v in delta.items():
        if isinstance(v, dict):
            inner_changed = {sk: sv for sk, sv in v.items() if abs(sv) > 1e-8}
            if inner_changed:
                inner = ", ".join(f"{sk}={sv:+.6f}" for sk, sv in inner_changed.items())
                parts.append(f"{k}={{{inner}}}")
        elif isinstance(v, (int, float)) and abs(v) > 1e-8:
            parts.append(f"{k}={v:+.6f}")
    if not parts:
        return "{no change}"
    return "{" + ", ".join(parts) + "}"
