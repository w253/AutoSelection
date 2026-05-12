from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from recipe_sandbox.schema.enums import Role, Split, TaskType

# Max characters to show per message content in pretty output
_PRETTY_CONTENT_LIMIT = 200


@dataclass
class Message:
    role: Role
    content: str
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Target:
    text: Optional[str] = None
    structured: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Provenance:
    raw_path: Optional[str] = None
    raw_format: Optional[str] = None
    raw_record_id: Optional[str] = None
    converter_name: Optional[str] = None
    converter_version: str = "v1"


@dataclass
class SampleMetadata:
    task_type: TaskType = TaskType.OTHER
    source_split: Split = Split.UNSPECIFIED
    language: Optional[str] = None
    format_tag: Optional[str] = None
    tool_use: bool = False
    has_multiturn: bool = False
    has_system_prompt: bool = False
    quality_flags: List[str] = field(default_factory=list)
    license: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _truncate(text: str, limit: int = _PRETTY_CONTENT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... ({len(text)} chars)"


@dataclass
class CanonicalSample:
    sample_id: str
    source_name: str
    messages: List[Message]
    target: Target
    metadata: SampleMetadata = field(default_factory=SampleMetadata)
    tags: List[str] = field(default_factory=list)
    canonical_version: str = "v1"
    raw_fingerprint: Optional[str] = None
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def pretty(self) -> str:
        """Return a human-readable summary of this sample."""
        lines: List[str] = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Sample ID   : {self.sample_id}")
        lines.append(f"  Source      : {self.source_name}")
        lines.append(f"  Task Type   : {self.metadata.task_type.value}")
        lines.append(f"  Split       : {self.metadata.source_split.value}")
        if self.tags:
            lines.append(f"  Tags        : {', '.join(self.tags)}")
        flags = []
        if self.metadata.has_system_prompt:
            flags.append("system_prompt")
        if self.metadata.has_multiturn:
            flags.append("multiturn")
        if self.metadata.tool_use:
            flags.append("tool_use")
        if flags:
            lines.append(f"  Flags       : {', '.join(flags)}")
        lines.append(f"  Messages ({len(self.messages)}):")
        for i, msg in enumerate(self.messages):
            role_tag = f"[{msg.role.value.upper()}]"
            content = _truncate(msg.content.replace("\n", "\\n"))
            lines.append(f"    {i + 1}. {role_tag:12s} {content}")
        if self.target.text is not None:
            lines.append(f"  Target      : {_truncate(self.target.text)}")
        if self.target.structured:
            lines.append(f"  Target(struct): {_truncate(str(self.target.structured))}")
        # Show similarity scores if present
        sims = self.metadata.extra.get("mona_scores")
        if not sims:
            sims = self.metadata.extra.get("similarities")
        if sims:
            sim_parts = [f"{k}={v:.4f}" for k, v in sims.items()]
            lines.append(f"  Similarities: {', '.join(sim_parts)}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)
