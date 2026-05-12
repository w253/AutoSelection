from abc import ABC, abstractmethod
from typing import Any

from recipe_sandbox.schema.types import CanonicalSample


class BaseConverter(ABC):
    name: str = "base_converter"
    version: str = "v1"

    @abstractmethod
    def convert_record(self, record: Any, source_name: str, raw_path: str) -> CanonicalSample:
        raise NotImplementedError
