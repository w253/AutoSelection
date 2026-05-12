from typing import Any, Dict

from recipe_sandbox.converters.base import BaseConverter
from recipe_sandbox.schema.io import sample_from_dict
from recipe_sandbox.schema.types import CanonicalSample


class NormalizedConverter(BaseConverter):
    name = "normalized_converter"
    version = "v1"

    def convert_record(self, record: Dict[str, Any], source_name: str, raw_path: str) -> CanonicalSample:
        sample = sample_from_dict(record)
        if not sample.source_name:
            sample.source_name = source_name
        if not sample.provenance.raw_path:
            sample.provenance.raw_path = raw_path
        return sample
