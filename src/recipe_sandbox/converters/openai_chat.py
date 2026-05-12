from typing import Any, Dict, List

from recipe_sandbox.converters.base import BaseConverter
from recipe_sandbox.schema.enums import Role
from recipe_sandbox.schema.types import CanonicalSample, Message, Provenance, SampleMetadata, Target
from recipe_sandbox.utils.hashing import stable_md5


class OpenAIChatConverter(BaseConverter):
    name = "openai_chat_converter"
    version = "v1"

    def convert_record(self, record: Dict[str, Any], source_name: str, raw_path: str) -> CanonicalSample:
        messages_raw: List[Dict[str, Any]] = record.get("messages", [])
        if not messages_raw:
            raise ValueError("openai-chat record 缺少 messages")

        messages = []
        for item in messages_raw[:-1]:
            messages.append(
                Message(
                    role=Role(item["role"]),
                    content=item.get("content", ""),
                    tool_name=item.get("name"),
                    metadata={k: v for k, v in item.items() if k not in {"role", "content", "name"}},
                )
            )

        last = messages_raw[-1]
        target_text = last.get("content", "")
        if not messages:
            messages = [Message(role=Role.USER, content=record.get("prompt", ""))]

        sample_id = stable_md5({"source_name": source_name, "messages": messages_raw})
        return CanonicalSample(
            sample_id=sample_id,
            source_name=source_name,
            messages=messages,
            target=Target(text=target_text),
            metadata=SampleMetadata(
                has_multiturn=len(messages_raw) > 2,
                has_system_prompt=any(m.role == Role.SYSTEM for m in messages),
                tool_use=any(m.role == Role.TOOL for m in messages),
                format_tag="openai_chat",
            ),
            raw_fingerprint=sample_id,
            provenance=Provenance(
                raw_path=raw_path,
                raw_format="openai_chat",
                raw_record_id=str(record.get("id")) if record.get("id") is not None else None,
                converter_name=self.name,
                converter_version=self.version,
            ),
        )
