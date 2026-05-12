"""TaskManager — structured output directory and artifact management.

Directory layout for a task:
    <output_dir>/
        task_config.json          # frozen task config
        log.txt                   # runtime log
        mappings/                 # cached AgentMapper mapping code files
            <source_name>.py
        canonical/                # canonical JSONL (after conversion)
            train/
                <source_name>.jsonl
            eval/
                <source_name>.jsonl
        scored/                   # scored output
            train/
                <source_name>.jsonl
                <source_name>.pt
            eval/
                <source_name>.jsonl
                <source_name>.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from recipe_sandbox.pipeline.task_config import TaskConfig
from recipe_sandbox.schema.types import CanonicalSample


class TaskManager:
    """Manages output directory structure and intermediate artifacts for a task."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self.root = Path(config.output_dir)
        self._ensure_dirs()
        self._log_lines: List[str] = []

    # ---- directory layout --------------------------------------------------

    @property
    def mappings_dir(self) -> Path:
        return self.root / "mappings"

    @property
    def canonical_dir(self) -> Path:
        return self.root / "canonical"

    @property
    def scored_dir(self) -> Path:
        return self.root / "scored"

    @property
    def recipes_dir(self) -> Path:
        return self.root / "recipes"

    @property
    def recipe_data_dir(self) -> Path:
        return self.recipes_dir

    @property
    def searches_dir(self) -> Path:
        return self.root / "searches"

    @property
    def temp_dir(self) -> Path:
        return self.root / "tmp"

    def canonical_path(self, split: str, source_name: str) -> Path:
        return self.canonical_dir / split / f"{_safe_name(source_name)}.jsonl"

    def scored_path(self, split: str, source_name: str, ext: str = ".jsonl") -> Path:
        return self.scored_dir / split / f"{_safe_name(source_name)}{ext}"

    def scored_shard_path(
        self,
        split: str,
        source_name: str,
        shard_index: int,
        ext: str = ".jsonl",
    ) -> Path:
        path = self.temp_dir / "scored_shards" / split / f"{_safe_name(source_name)}.part{shard_index:05d}{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def target_vector_shard_path(self, task_name: str) -> Path:
        path = self.temp_dir / "target_vectors" / f"{_safe_name(task_name)}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def recipe_dir(self, recipe_name: str) -> Path:
        return self.recipe_data_dir / _safe_name(recipe_name)

    def recipe_dataset_path(self, recipe_name: str) -> Path:
        return self.recipe_dir(recipe_name) / "dataset.jsonl"

    def recipe_manifest_path(self, recipe_name: str) -> Path:
        return self.recipe_dir(recipe_name) / "manifest.json"

    def recipe_trace_path(self, recipe_name: str) -> Path:
        return self.recipe_dir(recipe_name) / "trace.json"

    def search_dir(self, search_name: str) -> Path:
        return self.searches_dir / _safe_name(search_name)

    def search_manifest_path(self, search_name: str) -> Path:
        return self.search_dir(search_name) / "manifest.json"

    def search_trace_path(self, search_name: str) -> Path:
        return self.search_dir(search_name) / "trace.json"

    def search_diagnoses_path(self, search_name: str) -> Path:
        return self.search_dir(search_name) / "diagnoses.json"

    def search_recipe_path(self, search_name: str) -> Path:
        return self.search_dir(search_name) / "final_recipe.json"

    def mapping_path(self, source_name: str) -> Path:
        path = self.mappings_dir / f"{_safe_name(source_name)}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def target_vectors_path(self) -> Path:
        """Path for auto-built target vectors file."""
        return self.root / "target_vectors.pt"

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.mappings_dir.mkdir(parents=True, exist_ok=True)

    # ---- config persistence ------------------------------------------------

    def save_config(self) -> None:
        self.config.save(str(self.root / "task_config.json"))

    # ---- logging -----------------------------------------------------------

    def log(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        self._log_lines.append(line)
        print(line)

    def flush_log(self) -> None:
        log_path = self.root / "log.txt"
        with open(log_path, "a", encoding="utf-8") as f:
            for line in self._log_lines:
                f.write(line + "\n")
        self._log_lines.clear()

    # ---- data I/O helpers --------------------------------------------------

    def write_canonical(
        self, split: str, source_name: str, samples: List[CanonicalSample]
    ) -> Path:
        from recipe_sandbox.schema.io import write_jsonl

        path = self.canonical_path(split, source_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(str(path), samples)
        self.log(f"Wrote {len(samples)} canonical samples → {path}")
        return path

    def read_canonical(self, split: str, source_name: str) -> List[CanonicalSample]:
        from recipe_sandbox.schema.io import read_jsonl

        path = self.canonical_path(split, source_name)
        samples = list(read_jsonl(str(path)))
        self.log(f"Read {len(samples)} canonical samples ← {path}")
        return samples

    def write_scored(
        self, split: str, source_name: str, samples: List[CanonicalSample]
    ) -> Path:
        from recipe_sandbox.schema.io import write_scored_jsonl

        path = self.scored_path(split, source_name, ext=".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_scored_jsonl(str(path), samples)
        self.log(f"Wrote {len(samples)} scored samples → {path}")
        return path

    def list_canonical(self, split: str) -> List[Path]:
        d = self.canonical_dir / split
        return sorted(d.glob("*.jsonl")) if d.exists() else []

    def list_scored(self, split: str) -> List[Path]:
        d = self.scored_dir / split
        return sorted(d.glob("*.jsonl")) if d.exists() else []

    def recipe_dataset_ids_path(self, recipe_name: str) -> Path:
        return self.recipe_dir(recipe_name) / "dataset.ids.json"

    def write_recipe_dataset(self, recipe_name: str, samples: List[CanonicalSample]) -> Path:
        from recipe_sandbox.schema.io import write_jsonl

        path = self.recipe_dataset_path(recipe_name)
        write_jsonl(str(path), samples)
        self.log(f"Wrote {len(samples)} recipe samples → {path}")
        return path

    def write_recipe_dataset_ids(self, recipe_name: str, samples: List[CanonicalSample]) -> Path:
        path = self.recipe_dataset_ids_path(recipe_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = [s.sample_id for s in samples]
        path.write_text(json.dumps(ids, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Wrote {len(ids)} recipe sample IDs → {path}")
        return path

    def write_recipe_manifest(self, recipe_name: str, payload: Dict[str, Any]) -> Path:
        path = self.recipe_manifest_path(recipe_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Wrote recipe manifest → {path}")
        return path

    def write_recipe_trace(self, recipe_name: str, payload: Dict[str, Any]) -> Path:
        path = self.recipe_trace_path(recipe_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Wrote recipe trace → {path}")
        return path

    def write_search_manifest(self, search_name: str, payload: Dict[str, Any]) -> Path:
        path = self.search_manifest_path(search_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Wrote search manifest → {path}")
        return path

    def write_search_trace(self, search_name: str, payload: Dict[str, Any]) -> Path:
        path = self.search_trace_path(search_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Wrote search trace → {path}")
        return path

    def write_search_diagnoses(self, search_name: str, payload: Dict[str, Any]) -> Path:
        path = self.search_diagnoses_path(search_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Wrote search diagnoses → {path}")
        return path

    def write_search_recipe(self, search_name: str, payload: Dict[str, Any]) -> Path:
        path = self.search_recipe_path(search_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Wrote final search recipe → {path}")
        return path

    # ---- summary -----------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Task: {self.config.task_name}",
            f"Output: {self.root}",
            f"Train sources: {len(self.config.train_sources)}",
            f"Eval sources: {len(self.config.eval_sources)}",
            f"Canonical train files: {len(self.list_canonical('train'))}",
            f"Canonical eval files: {len(self.list_canonical('eval'))}",
            f"Scored train files: {len(self.list_scored('train'))}",
            f"Scored eval files: {len(self.list_scored('eval'))}",
            f"Recipe runs: {len(list(self.recipes_dir.glob('*')))}",
            f"Search runs: {len(list(self.searches_dir.glob('*')))}",
        ]
        return "\n".join(lines)


def _safe_name(name: str) -> str:
    """Sanitize source name for use as a filename."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_").replace("..", "_")
