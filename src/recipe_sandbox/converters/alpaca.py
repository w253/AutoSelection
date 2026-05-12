from typing import Any, Dict

from recipe_sandbox.converters.base import BaseConverter
from recipe_sandbox.schema.enums import Role
from recipe_sandbox.schema.types import CanonicalSample, Message, Provenance, SampleMetadata, Target
from recipe_sandbox.utils.hashing import stable_md5
from recipe_sandbox.utils.text import join_non_empty


class AlpacaConverter(BaseConverter):
    name = "alpaca_converter"
    version = "v1"

    def convert_record(self, record: Dict[str, Any], source_name: str, raw_path: str) -> CanonicalSample:
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        output_text = record.get("output", "")

        user_text = join_non_empty([instruction, input_text])
        sample_id = stable_md5(
            {
                "source_name": source_name,
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }
        )

        return CanonicalSample(
            sample_id=sample_id,
            source_name=source_name,
            messages=[Message(role=Role.USER, content=user_text)],
            target=Target(text=output_text),
            metadata=SampleMetadata(
                has_multiturn=False,
                has_system_prompt=False,
                tool_use=False,
                format_tag="alpaca",
            ),
            raw_fingerprint=sample_id,
            provenance=Provenance(
                raw_path=raw_path,
                raw_format="alpaca",
                raw_record_id=str(record.get("id")) if record.get("id") is not None else None,
                converter_name=self.name,
                converter_version=self.version,
            ),
        )
