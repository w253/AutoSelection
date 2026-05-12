from typing import Any, Dict, List

from recipe_sandbox.converters.base import BaseConverter
from recipe_sandbox.schema.enums import Role
from recipe_sandbox.schema.types import CanonicalSample, Message, Provenance, SampleMetadata, Target
from recipe_sandbox.utils.hashing import stable_md5


_ROLE_MAP = {
    "system": Role.SYSTEM,
    "human": Role.USER,
    "user": Role.USER,
    "gpt": Role.ASSISTANT,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
}


class ShareGPTConverter(BaseConverter):
    name = "sharegpt_converter"
    version = "v1"

    def convert_record(self, record: Dict[str, Any], source_name: str, raw_path: str) -> CanonicalSample:
        conversations: List[Dict[str, Any]] = record.get("conversations", [])
        if not conversations:
            raise ValueError("sharegpt record 缺少 conversations")

        messages = []
        for item in conversations[:-1]:
            mapped_role = _ROLE_MAP.get(str(item.get("from", "")).lower())
            if mapped_role is None:
                continue
            messages.append(Message(role=mapped_role, content=item.get("value", "")))

        last = conversations[-1]
        target_text = last.get("value", "")

        if not messages:
            first_role = _ROLE_MAP.get(str(last.get("from", "")).lower(), Role.USER)
            messages = [Message(role=first_role, content=target_text)]
            target_text = record.get("target", target_text)

        sample_id = stable_md5({"source_name": source_name, "conversations": conversations})

        return CanonicalSample(
            sample_id=sample_id,
            source_name=source_name,
            messages=messages,
            target=Target(text=target_text),
            metadata=SampleMetadata(
                has_multiturn=len(conversations) > 2,
                has_system_prompt=any(message.role == Role.SYSTEM for message in messages),
                tool_use=any(message.role == Role.TOOL for message in messages),
                format_tag="sharegpt",
            ),
            raw_fingerprint=sample_id,
            provenance=Provenance(
                raw_path=raw_path,
                raw_format="sharegpt",
                raw_record_id=str(record.get("id")) if record.get("id") is not None else None,
                converter_name=self.name,
                converter_version=self.version,
            ),
        )
