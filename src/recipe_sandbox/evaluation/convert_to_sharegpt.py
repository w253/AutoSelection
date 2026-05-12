"""Convert CanonicalSample JSONL to LlamaFactory ShareGPT format.

CanonicalSample format:
  {"sample_id": "...", "messages": [{"role": "user", "content": "..."}], "target": {"text": "..."}, ...}

LlamaFactory ShareGPT format:
  {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
"""

import json
import sys
from pathlib import Path


def convert_canonical_to_sharegpt(input_path: str, output_path: str, max_samples: int = None):
    """Convert CanonicalSample JSONL → LlamaFactory ShareGPT JSON."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if not line.strip():
                continue
            sample = json.loads(line)

            conversations = []
            
            def _add_turn(role: str, text: str):
                if not text: return
                if conversations and conversations[-1]["from"] == role:
                    conversations[-1]["value"] += "\n" + text
                else:
                    conversations.append({"from": role, "value": text})

            # Add message turns
            for msg in sample.get("messages", []):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "human"):
                    _add_turn("human", content)
                elif role in ("assistant", "gpt"):
                    _add_turn("gpt", content)
                elif role == "system":
                    _add_turn("system", content)

            # Add target as assistant response
            target = sample.get("target", {})
            target_text = target.get("text", "") if isinstance(target, dict) else str(target)
            if target_text:
                _add_turn("gpt", target_text)

            if len(conversations) >= 2:  # Need at least human + gpt
                records.append({"conversations": conversations})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    print(f"Converted {len(records)} samples: {input_path} → {output_path}")
    return len(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    convert_canonical_to_sharegpt(args.input, args.output, args.max_samples)
