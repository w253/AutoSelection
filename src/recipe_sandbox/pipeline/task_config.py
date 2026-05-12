"""Task configuration dataclasses for pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LLMConfig:
    """Configuration for the LLM used by AgentMapper."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    n_sample: int = 5
    max_retries: int = 2


@dataclass
class DataSourceConfig:
    """Describes one input data file."""

    path: str
    source_name: str
    format: str = "auto"  # "auto" | "canonical" | "alpaca" | "sharegpt" | ...
    mapping_code: Optional[str] = None  # path to cached mapping code

    @property
    def is_auto(self) -> bool:
        return self.format == "auto"

    @property
    def is_canonical(self) -> bool:
        return self.format == "canonical"


@dataclass
class ModelConfig:
    """Model + SAE paths for MONA scoring."""

    model_path: str
    sae_path: Optional[str] = None
    target_vectors_path: Optional[str] = None
    d_sae: Optional[int] = None
    device: Optional[str] = None
    device_map: Optional[str] = None
    torch_dtype: str = "bfloat16"
    hf_home: Optional[str] = None
    max_length: int = 2048
    hidden_state_index: int = -2


@dataclass
class ScoringConfig:
    """Parameters for the scoring step."""

    method: str = "mona"  # "mona" | "less" | ...
    store_feature: bool = False
    max_samples: Optional[int] = None
    batch_size: int = 8
    max_eval_samples: Optional[int] = 128  # per-task cap for target vector building
    devices: Optional[List[str]] = None
    parallel_workers: Optional[int] = None
    shard_size: Optional[int] = None
    show_progress: bool = True
    progress_interval: int = 100
    score_eval: bool = False
    eval_sample_seed: int = 42


@dataclass
class RecipeStepConfig:
    """One executable step inside a structured data recipe."""

    step_type: str = "auto"  # "auto" | "operator" | "scoring"
    operator_ref: Optional[str] = None
    operator: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    name: Optional[str] = None

    @property
    def resolved_step_type(self) -> str:
        if self.step_type != "auto":
            return self.step_type
        return "operator"

    @property
    def resolved_operator_name(self) -> str:
        name = self.operator or self.operator_ref
        if not name:
            raise ValueError("operator step requires operator")
        return str(name)

    @property
    def resolved_operator_config(self) -> Dict[str, Any]:
        return dict(self.params)


@dataclass
class RecipeConfig:
    """Recipe execution config for post-scoring data processing."""

    enabled: bool = False
    recipe_name: str = "default_recipe"
    input_split: str = "train"
    input_stage: str = "canonical"  # "scored" | "canonical"
    steps: List[RecipeStepConfig] = field(default_factory=list)
    task_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSearchConfig:
    """Configuration for experiment-first recipe search."""

    enabled: bool = False
    search_name: str = "default_search"
    input_split: str = "train"
    input_stage: str = "scored"  # "scored" | "canonical"
    max_rounds: int = 3
    max_candidates_per_round: int = 6
    stagnation_rounds: int = 1
    min_gain: float = 0.01
    max_cost_increase_ratio: float = 0.5
    score_path: Optional[str] = None
    task_label: Optional[str] = None
    allow_knowledge_baseline: bool = False


@dataclass
class TaskConfig:
    """Top-level task configuration.

    A *task* is one complete run: ingest data → convert → score → output.
    """

    task_name: str
    output_dir: str

    train_sources: List[DataSourceConfig] = field(default_factory=list)
    eval_sources: List[DataSourceConfig] = field(default_factory=list)

    model: ModelConfig = field(default_factory=lambda: ModelConfig(model_path=""))
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    recipe: RecipeConfig = field(default_factory=RecipeConfig)
    search: ExperimentSearchConfig = field(default_factory=ExperimentSearchConfig)
    recipe_catalog_path: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TaskConfig":
        config_path = Path(path)
        raw = _load_structured_config(config_path)
        return cls.from_dict(raw, base_dir=config_path.parent)

    @classmethod
    def from_dict(cls, data: dict, base_dir: Optional[Path] = None) -> "TaskConfig":
        from recipe_sandbox.pipeline.recipe_catalog import resolve_recipe_with_catalog

        recipe_payload: Dict[str, Any] = {}
        recipe_path = data.get("recipe_path")
        if recipe_path:
            resolved_recipe_path = Path(recipe_path)
            if not resolved_recipe_path.is_absolute() and base_dir is not None:
                resolved_recipe_path = base_dir / resolved_recipe_path
            recipe_payload.update(_load_structured_config(resolved_recipe_path))

        recipe_catalog_path = data.get("recipe_catalog_path")
        resolved_catalog_path: Optional[Path] = None
        if recipe_catalog_path:
            resolved_catalog_path = Path(recipe_catalog_path)
            if not resolved_catalog_path.is_absolute() and base_dir is not None:
                resolved_catalog_path = base_dir / resolved_catalog_path

        recipe_payload.update(data.get("recipe", {}))
        if resolved_catalog_path is not None and recipe_payload:
            recipe_payload = resolve_recipe_with_catalog(resolved_catalog_path, recipe_payload)

        recipe_payload_for_dataclass = dict(recipe_payload)
        recipe_payload_for_dataclass.pop("operators", None)

        return cls(
            task_name=data["task_name"],
            output_dir=data["output_dir"],
            train_sources=[DataSourceConfig(**s) for s in data.get("train_sources", [])],
            eval_sources=[DataSourceConfig(**s) for s in data.get("eval_sources", [])],
            model=ModelConfig(**{"model_path": "", **data.get("model", {})}),
            scoring=ScoringConfig(**data.get("scoring", {})),
            llm=LLMConfig(**data.get("llm", {})),
            recipe=RecipeConfig(
                **{
                    **recipe_payload_for_dataclass,
                    "steps": [RecipeStepConfig(**step) for step in recipe_payload_for_dataclass.get("steps", [])],
                }
            ),
            search=ExperimentSearchConfig(**data.get("search", {})),
            recipe_catalog_path=str(resolved_catalog_path) if resolved_catalog_path is not None else None,
            extra=data.get("extra", {}),
        )


def _load_structured_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Structured config at {path} must be a mapping/object")
    return payload
