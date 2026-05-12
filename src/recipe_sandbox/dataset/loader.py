import json
from pathlib import Path
from typing import Iterable, Iterator

from recipe_sandbox.schema.io import read_jsonl
from recipe_sandbox.schema.types import CanonicalSample


class CanonicalDatasetLoader:
    def load_samples(self, path: str) -> Iterator[CanonicalSample]:
        return read_jsonl(path)

    def load_manifest(self, path: str) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def load_all(self, samples_path: str) -> list[CanonicalSample]:
        return list(self.load_samples(samples_path))
