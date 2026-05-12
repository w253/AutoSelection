import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

from recipe_sandbox.schema.enums import Role, Split, TaskType
from recipe_sandbox.schema.types import CanonicalSample, Message, Provenance, SampleMetadata, Target

# Keys in metadata.extra that contain large vectors (SAE features, embeddings,
# hidden states, etc.).  These are already persisted in companion .pt files so
# there is no need to duplicate them inside the human-readable JSONL output.
# Stripping them keeps scored / recipe JSONL files orders of magnitude smaller.
_HEAVY_EXTRA_KEYS = frozenset({
    "feature",            # MONA SAE feature vector (d_sae floats)
    "embedding",          # generic embedding vector
    "activations",        # raw activations
    "hidden_states",      # intermediate hidden states
    "sae_features",       # alternative SAE feature key
    "representations",    # LESS representation vectors
})

_NUMERIC_EXTRA_KEYS = frozenset({
    "sae_topk",
    "ifd",
    "varentropy",
    "ngram_entropy",
    "action_object",
    "mona_score",
    "mona_scores",
    "similarities",
})


def sample_from_dict(data: dict) -> CanonicalSample:
    metadata = data.get("metadata") or {}
    target = data.get("target") or {}
    messages_in = data.get("messages") or []
    return CanonicalSample(
        sample_id=data.get("sample_id") or data.get("id") or "unknown_id",
        source_name=data.get("source_name") or data.get("dataset") or "unknown_source",
        messages=[
            Message(
                role=Role(message["role"]),
                content=message["content"],
                tool_name=message.get("tool_name"),
                tool_call_id=message.get("tool_call_id"),
                metadata=message.get("metadata") or {},
            )
            for message in messages_in
        ],
        target=Target(
            text=target.get("text"),
            structured=target.get("structured") or {},
        ),
        metadata=SampleMetadata(
            task_type=TaskType(metadata.get("task_type", TaskType.OTHER.value)),
            source_split=Split(metadata.get("source_split", Split.UNSPECIFIED.value)),
            language=metadata.get("language"),
            format_tag=metadata.get("format_tag"),
            tool_use=metadata.get("tool_use", False),
            has_multiturn=metadata.get("has_multiturn", False),
            has_system_prompt=metadata.get("has_system_prompt", False),
            quality_flags=metadata.get("quality_flags", []),
            license=metadata.get("license"),
            extra=metadata.get("extra", {}),
        ),
        tags=data.get("tags", []),
        canonical_version=data.get("canonical_version", "v1"),
        raw_fingerprint=data.get("raw_fingerprint"),
        provenance=Provenance(**data.get("provenance", {})),
    )


def read_jsonl(path: str) -> Iterator[CanonicalSample]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield sample_from_dict(json.loads(line))


def _strip_heavy_fields(d: dict) -> dict:
    """Remove large vector fields from a serialized sample dict *in-place*.

    This prevents multi-MB feature vectors from bloating JSONL output.
    The original ``metadata.extra`` dict on the *in-memory* CanonicalSample
    is **not** mutated — only the dict produced by ``to_dict()`` is cleaned.
    """
    extra = d.get("metadata", {}).get("extra")
    if extra is None:
        return d
    for key in _HEAVY_EXTRA_KEYS:
        extra.pop(key, None)
    # Also drop any list/dict value whose length exceeds a reasonable scalar
    # threshold — catches unexpected large blobs added by future scorers.
    _drop = [k for k, v in extra.items() if isinstance(v, (list, tuple)) and len(v) > 256]
    for k in _drop:
        extra.pop(k, None)
    return d


def strip_numeric_extras(sample: CanonicalSample) -> CanonicalSample:
    filtered_extra = {
        key: value
        for key, value in sample.metadata.extra.items()
        if key not in _NUMERIC_EXTRA_KEYS
    }
    return replace(sample, metadata=replace(sample.metadata, extra=filtered_extra))


def iter_samples_without_numeric_extras(samples: Iterable[CanonicalSample]) -> Iterator[CanonicalSample]:
    for sample in samples:
        yield strip_numeric_extras(sample)


def write_jsonl(path: str, samples: Iterable[CanonicalSample], *, strip_heavy: bool = True) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for sample in samples:
            d = sample.to_dict()
            if strip_heavy:
                _strip_heavy_fields(d)
            handle.write(json.dumps(d, ensure_ascii=False) + "\n")


def write_scored_jsonl(path: str, samples: Iterable[CanonicalSample]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(scored_sample_to_dict(sample), ensure_ascii=False) + "\n")


def scored_sample_to_dict(sample: CanonicalSample) -> dict:
    similarities = sample.metadata.extra.get("mona_scores")
    if not isinstance(similarities, dict):
        similarities = sample.metadata.extra.get("similarities", {})
    payload = sample.to_dict()

    # Strip heavy vector fields — they are already in the companion .pt file.
    _strip_heavy_fields(payload)

    payload["messages"] = [
        {
            "role": message.role.value,
            "content": message.content,
            "tool_name": message.tool_name,
            "tool_call_id": message.tool_call_id,
            "metadata": message.metadata,
        }
        for message in sample.messages
        if message.content
    ]
    payload["text"] = "\n".join(
        part for part in [
            *(message.content.strip() for message in sample.messages if message.content and message.content.strip()),
            sample.target.text.strip() if sample.target.text and sample.target.text.strip() else None,
        ] if part
    )
    payload["similarities"] = similarities
    return payload


def materialize_dataset(ids_path: str, canonical_files: List[str], output_jsonl: str) -> None:
    """Read a JSON array of sample IDs, stream through canonical_files, and write matched samples to output."""
    with open(ids_path, "r", encoding="utf-8") as f:
        target_ids = set(json.load(f))

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    found = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for cfile in canonical_files:
            with open(cfile, "r", encoding="utf-8") as infile:
                for line in infile:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("sample_id") in target_ids:
                            _strip_heavy_fields(record)
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            found += 1
                    except Exception:
                        pass
    return found


def read_jsonl_auto(
    path: str,
    source_name: str = "unknown",
    *,
    llm_client: Any = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o",
    mapping_code_path: Optional[str] = None,
    save_mapping_to: Optional[str] = None,
    n_sample: int = 5,
) -> Iterator[CanonicalSample]:
    """Read an arbitrary JSONL file by using an LLM agent to auto-map fields.

    This function handles JSONL files whose keys do not match the canonical
    schema. It uses an LLM to inspect sample records and generate a mapping
    function, then applies that mapping to every record.

    Parameters
    ----------
    path : str
        Path to the input JSONL file.
    source_name : str
        Name tag for the data source.
    llm_client : object, optional
        An object with a ``.chat(prompt) -> str`` method, or a callable.
        If not given, an OpenAI-compatible client is created from
        *api_key*, *base_url*, and *model*.
    api_key / base_url / model : str
        Used to construct a default LLM client when *llm_client* is None.
    mapping_code_path : str, optional
        If given, load a previously generated mapping code file instead of
        calling the LLM. Useful for reuse / deterministic pipelines.
    save_mapping_to : str, optional
        If given, save the generated mapping code to this path for reuse.
    n_sample : int
        Number of records to sample for LLM inference (default 5).

    Yields
    ------
    CanonicalSample
    """
    from recipe_sandbox.agents import AgentMapper

    mapper = AgentMapper(
        llm_client=llm_client,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    if mapping_code_path is not None:
        mapper.load_mapping_code(mapping_code_path)

    for mapped_dict in mapper.read_jsonl(path, source_name=source_name, n_sample=n_sample):
        yield sample_from_dict(mapped_dict)

    if save_mapping_to is not None and mapper.mapping_code is not None:
        mapper.save_mapping_code(save_mapping_to)
