"""PipelineOrchestrator — top-level workflow orchestrator for pipeline tasks.

The PipelineOrchestrator drives the entire workflow:
    1. Ingest: convert all data sources to canonical format (batch AgentMapper)
    2. Execute the configured recipe pipeline
    3. Output: save recipe datasets and intermediate artifacts

Usage:
    config = TaskConfig.load("task.json")
    orchestrator = PipelineOrchestrator(config)
    orchestrator.run()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from recipe_sandbox.pipeline.experiment_search import ExperimentSearchController
from recipe_sandbox.pipeline.recipe_executor import RecipeExecutor
from recipe_sandbox.pipeline.task_config import DataSourceConfig, TaskConfig
from recipe_sandbox.pipeline.task_manager import TaskManager
from recipe_sandbox.scoring.runner import ScoringRunner, create_scoring_runner
from recipe_sandbox.schema.types import CanonicalSample


class PipelineOrchestrator:
    """Orchestrates multi-source data ingestion and recipe-driven execution.

    Parameters
    ----------
    config : TaskConfig
        The task configuration.
    llm_client : object, optional
        LLM client for AgentMapper (object with ``.chat()`` or callable).
        If None, one is built from ``config.llm``.
    """

    def __init__(
        self,
        config: TaskConfig,
        llm_client: Any = None,
    ) -> None:
        self.config = config
        self.manager = TaskManager(config)
        self._llm_client = llm_client
        self._step_hooks: Dict[str, List[Callable]] = {}
        self._scoring_runner: Optional[ScoringRunner] = None
        self._recipe_executor: Optional[RecipeExecutor] = None
        self._search_controller: Optional[ExperimentSearchController] = None

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for a pipeline event.

        Events: before_ingest, after_ingest, before_score, after_score,
            before_recipe, after_recipe, before_source, after_source
        """
        self._step_hooks.setdefault(event, []).append(callback)

    def _emit(self, event: str, **kwargs: Any) -> None:
        for cb in self._step_hooks.get(event, []):
            cb(**kwargs)

    def run(self) -> None:
        """Run the full pipeline: ingest -> recipe pipeline -> finalize."""
        self.manager.save_config()
        self.manager.log(f"=== Task '{self.config.task_name}' started ===")

        self._emit("before_ingest")
        self._ingest_all("train", self.config.train_sources)
        self._ingest_all("eval", self.config.eval_sources)
        self._emit("after_ingest")

        if self.config.search.enabled:
            if self.config.search.input_stage == "scored":
                self._emit("before_score")
                self._score_all()
                self._emit("after_score")
            self._emit("before_recipe")
            self._run_search()
            self._emit("after_recipe")
        elif self.config.recipe.enabled and self.config.recipe.steps:
            self._emit("before_recipe")
            self._run_recipe()
            self._emit("after_recipe")
        else:
            self._emit("before_score")
            self._score_all()
            self._emit("after_score")

        self.manager.log(f"=== Task '{self.config.task_name}' completed ===")
        self.manager.log(self.manager.summary())
        self.manager.flush_log()

    def run_ingest_only(self) -> None:
        """Run only the ingestion step (convert sources to canonical)."""
        self.manager.save_config()
        self.manager.log(f"=== Ingest-only for '{self.config.task_name}' ===")
        self._emit("before_ingest")
        self._ingest_all("train", self.config.train_sources)
        self._ingest_all("eval", self.config.eval_sources)
        self._emit("after_ingest")
        self.manager.flush_log()

    def run_score_only(self) -> None:
        """Run only the scoring step (assumes canonical data exists)."""
        self.manager.log(f"=== Score-only for '{self.config.task_name}' ===")
        self._score_all()
        self.manager.flush_log()

    def run_recipe_only(self) -> None:
        """Run only the recipe execution step (assumes prerequisite data already exists)."""
        self.manager.log(f"=== Recipe-only for '{self.config.task_name}' ===")
        self._emit("before_recipe")
        self._run_recipe()
        self._emit("after_recipe")
        self.manager.flush_log()

    def run_search_only(self) -> None:
        """Run only the experiment-first recipe search step."""
        self.manager.log(f"=== Search-only for '{self.config.task_name}' ===")
        self._emit("before_recipe")
        self._run_search()
        self._emit("after_recipe")
        self.manager.flush_log()

    def _ingest_all(self, split: str, sources: List[DataSourceConfig]) -> None:
        if not sources:
            self.manager.log(f"No {split} sources configured, skipping ingest.")
            return

        self.manager.log(f"Ingesting {len(sources)} {split} source(s)...")
        for src in sources:
            self._emit("before_source", split=split, source=src)
            samples = self._ingest_source(src)
            self.manager.write_canonical(split, src.source_name, samples)
            if samples:
                self.manager.log(f"Sample preview ({src.source_name}):\n{samples[0].pretty}")
            self._emit("after_source", split=split, source=src, count=len(samples))

    def _ingest_source(self, src: DataSourceConfig) -> List[CanonicalSample]:
        """Convert one data source to canonical samples."""
        if src.is_canonical:
            return self._load_canonical(src.path)

        if src.is_auto:
            return self._load_auto(src)

        return self._load_with_converter(src)

    def _load_canonical(self, path: str) -> List[CanonicalSample]:
        from recipe_sandbox.schema.io import read_jsonl

        self.manager.log(f"Loading canonical data from {path}")
        return list(read_jsonl(path))

    def _load_auto(self, src: DataSourceConfig) -> List[CanonicalSample]:
        from recipe_sandbox.agents import AgentMapper
        from recipe_sandbox.schema.io import sample_from_dict

        llm_cfg = self.config.llm
        mapper = AgentMapper(
            llm_client=self._llm_client,
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            model=llm_cfg.model,
            max_retries=llm_cfg.max_retries,
        )

        cached_path = src.mapping_code
        if cached_path is None:
            auto_cache = self.manager.mapping_path(src.source_name)
            if auto_cache.exists():
                cached_path = str(auto_cache)

        if cached_path is not None:
            self.manager.log(f"Loading cached mapping code from {cached_path}")
            mapper.load_mapping_code(cached_path)

        self.manager.log(f"Auto-mapping {src.path} (source: {src.source_name})")
        mapped_dicts = list(
            mapper.read_jsonl(
                src.path,
                source_name=src.source_name,
                n_sample=llm_cfg.n_sample,
            )
        )
        samples = [sample_from_dict(d) for d in mapped_dicts]
        self.manager.log(f"Mapped sample preview:\n{samples[0].pretty if samples else 'No samples mapped.'}")
        if mapper.mapping_code is not None:
            save_to = str(self.manager.mapping_path(src.source_name))
            mapper.save_mapping_code(save_to)
            self.manager.log(f"Mapping code cached -> {save_to}")

        return samples

    def _load_with_converter(self, src: DataSourceConfig) -> List[CanonicalSample]:
        import json as _json

        from recipe_sandbox.converters.alpaca import AlpacaConverter
        from recipe_sandbox.converters.normalized import NormalizedConverter
        from recipe_sandbox.converters.openai_chat import OpenAIChatConverter
        from recipe_sandbox.converters.sharegpt import ShareGPTConverter

        converters = {
            "alpaca": AlpacaConverter,
            "sharegpt": ShareGPTConverter,
            "openai_chat": OpenAIChatConverter,
            "normalized": NormalizedConverter,
        }

        cls = converters.get(src.format)
        if cls is None:
            raise ValueError(f"Unknown format '{src.format}'. Use 'auto' for LLM-based mapping.")

        converter = cls()
        self.manager.log(f"Converting {src.path} with {converter.name}")

        samples = []
        with open(src.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = _json.loads(line)
                    samples.append(
                        converter.convert_record(
                            record=record,
                            source_name=src.source_name,
                            raw_path=src.path,
                        )
                    )
        return samples

    def _score_all(self) -> None:
        """Score all canonical data files."""
        self._get_scoring_runner().run()

    def _run_recipe(self) -> None:
        """Execute configured recipe steps on scored or canonical data."""
        if self._recipe_executor is None:
            self._recipe_executor = RecipeExecutor(self.config, self.manager)
        self._recipe_executor.run()

    def _run_search(self) -> None:
        """Execute experiment-first recipe search."""
        if self._search_controller is None:
            self._search_controller = ExperimentSearchController(
                self.config,
                self.manager,
                recipe_executor=self._recipe_executor or RecipeExecutor(self.config, self.manager),
            )
        self._search_controller.run()

    def _build_scoring_jobs(self, files_to_score: List[tuple[str, str]], devices: List[str]) -> List[dict]:
        """Expose scoring job planning for tests and orchestration inspection."""
        return self._get_scoring_runner()._build_scoring_jobs(files_to_score, devices)

    def _partition_scoring_jobs(self, jobs: List[dict], devices: List[str]) -> List[tuple[str, List[dict]]]:
        """Expose scoring job partitioning for tests and orchestration inspection."""
        return self._get_scoring_runner()._partition_scoring_jobs(jobs, devices)

    def _get_scoring_runner(self) -> ScoringRunner:
        if self._scoring_runner is None:
            self._scoring_runner = create_scoring_runner(self.config, self.manager)
        return self._scoring_runner
