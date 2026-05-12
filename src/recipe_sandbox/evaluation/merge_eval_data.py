"""Merge multiple benchmark eval datasets into a single JSONL.

Each record gets a ``data_source`` field so that downstream evaluation
can dispatch to the correct prompt builder / scorer while benefiting
from a single vLLM inference pass.

Usage (CLI):
    python merge_eval_data.py \\
        --inputs gpqa:/data/eval/gpqa_extended.jsonl,gsm8k:/data/eval/gsm8k_test.jsonl \\
        --output /data/eval/eval_merged.jsonl

Usage (API):
    from recipe_sandbox.evaluation.merge_eval_data import (
        merge_eval_datasets, load_merged_eval_data,
    )
    merge_eval_datasets(
        {"gpqa": "gpqa.jsonl", "gsm8k": "gsm8k.jsonl"},
        "eval_merged.jsonl",
    )
    grouped = load_merged_eval_data("eval_merged.jsonl")
    # grouped == {"gpqa": [...], "gsm8k": [...]}
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


def merge_eval_datasets(
    task_data_paths: Dict[str, str],
    output_path: str,
) -> int:
    """Merge multiple benchmark JSONL files into one with ``data_source``.

    Args:
        task_data_paths: Mapping of benchmark name → JSONL file path.
        output_path: Where to write the merged JSONL.

    Returns:
        Total number of records written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for task_name, data_path in task_data_paths.items():
            if not os.path.isfile(data_path):
                raise FileNotFoundError(
                    f"Data file not found for '{task_name}': {data_path}"
                )
            with open(data_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record["data_source"] = task_name
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
    print(f"Merged {total} records from {list(task_data_paths.keys())} → {output_path}")
    return total


def load_merged_eval_data(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load a merged JSONL and group records by ``data_source``.

    Returns:
        Dict mapping data_source name → list of records.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ds = record.get("data_source", "unknown")
            grouped[ds].append(record)
    return dict(grouped)


def _parse_inputs(raw: str) -> Dict[str, str]:
    """Parse 'gpqa:/path/gpqa.jsonl,gsm8k:/path/gsm8k.jsonl'."""
    tasks: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid spec '{part}'. Expected name:/path/to/data.jsonl"
            )
        name, path = part.split(":", 1)
        tasks[name.strip()] = path.strip()
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple eval JSONL files with data_source tagging",
    )
    parser.add_argument(
        "--inputs", type=str, required=True,
        help="Comma-separated name:path pairs, e.g. gpqa:gpqa.jsonl,gsm8k:gsm8k.jsonl",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the merged JSONL",
    )
    args = parser.parse_args()

    task_data_paths = _parse_inputs(args.inputs)
    merge_eval_datasets(task_data_paths, args.output)


if __name__ == "__main__":
    main()
