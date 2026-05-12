"""Run unified vLLM evaluation across a user-specified set of checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


RECIPE_SANDBOX_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DATA_DIR = RECIPE_SANDBOX_ROOT / "data" / "eval"
DEFAULT_UNIFIED_EVAL_SCRIPT = (
    RECIPE_SANDBOX_ROOT / "src" / "recipe_sandbox" / "evaluation" / "unified_eval_vllm.py"
)

TASK_FILE_MAP: Dict[str, str] = {
    "gpqa": "gpqa_main.jsonl",
    "gsm8k": "gsm8k_test.jsonl",
    "bbh": "bbh_test.jsonl",
    "mmlu": "mmlu_test.jsonl",
    "graphwiz": "GraphWiz_test.jsonl",
    "nlgraph_yesno": "NLgraph_test.jsonl",
}

TASK_ALIASES: Dict[str, str] = {
    "graph": "graphwiz",
    "graph_yesno": "nlgraph_yesno",
}


def parse_tasks(raw: str) -> List[str]:
    """Parse a comma-separated task list."""
    tasks = [
        TASK_ALIASES.get(task.strip(), task.strip())
        for task in raw.split(",")
        if task.strip()
    ]
    if not tasks:
        raise argparse.ArgumentTypeError("At least one task must be specified.")
    unknown = [task for task in tasks if task not in TASK_FILE_MAP]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown tasks: {unknown}. Supported: {sorted(TASK_FILE_MAP)}"
        )
    return tasks


def build_eval_tasks_spec(eval_data_dir: Path, tasks: List[str]) -> str:
    """Build the unified_eval_vllm.py --eval_tasks string."""
    parts: List[str] = []
    for task in tasks:
        data_path = Path(TASK_FILE_MAP[task])
        if not data_path.is_absolute():
            data_path = eval_data_dir / data_path
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing eval data for '{task}': {data_path}")
        parts.append(f"{task}:{data_path}")
    return ",".join(parts)


def sanitize_checkpoint_name(index: int, checkpoint: Path) -> str:
    """Create a stable output directory name for one checkpoint."""
    if checkpoint.name == "adapter" and checkpoint.parent.name:
        name_source = checkpoint.parent.name
    else:
        name_source = checkpoint.name
    safe_name = name_source.replace("/", "_")
    return f"{index:03d}_{safe_name}"


def load_metrics(metrics_path: Path) -> Dict[str, object]:
    """Load eval_metrics.json."""
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_accelerator_env_var() -> str:
    """Return the accelerator visibility env var used in this environment."""
    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        return "ASCEND_RT_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"


def resolve_device_ids(raw: str | None, num_shards: int) -> List[str]:
    """Resolve one device id per shard."""
    if num_shards <= 1:
        return []

    if raw:
        device_ids = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        env_var = get_accelerator_env_var()
        visible = os.environ.get(env_var, "")
        device_ids = [part.strip() for part in visible.split(",") if part.strip()]

    if len(device_ids) < num_shards:
        raise ValueError(
            f"Need at least {num_shards} device ids for sharded eval, got {device_ids!r}."
        )
    return device_ids[:num_shards]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loop over checkpoints and run unified_eval_vllm.py for each one.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Optional base model path. If set, checkpoints are treated as LoRA adapters.",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=str(DEFAULT_EVAL_DATA_DIR),
        help="Directory containing eval JSONL files.",
    )
    parser.add_argument(
        "--tasks",
        type=parse_tasks,
        default=parse_tasks("gpqa,gsm8k,bbh,mmlu,graphwiz,nlgraph_yesno"),
        help=(
            "Comma-separated task names. Supported: gpqa,gsm8k,bbh,mmlu,"
            "graphwiz,nlgraph_yesno. Legacy aliases graph and graph_yesno "
            "are accepted."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for per-checkpoint outputs.",
    )
    parser.add_argument(
        "--unified_eval_script",
        type=str,
        default=str(DEFAULT_UNIFIED_EVAL_SCRIPT),
        help="Path to unified_eval_vllm.py.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Run N independent eval shard processes and aggregate results.",
    )
    parser.add_argument(
        "--device_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated physical device ids for shard mode. "
            "Defaults to the current visible device list."
        ),
    )
    args = parser.parse_args()

    eval_data_dir = Path(args.eval_data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    unified_eval_script = Path(args.unified_eval_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not unified_eval_script.is_file():
        raise FileNotFoundError(f"unified_eval_vllm.py not found: {unified_eval_script}")

    task_spec = build_eval_tasks_spec(eval_data_dir, args.tasks)
    device_ids = resolve_device_ids(args.device_ids, args.num_shards)
    accelerator_env_var = get_accelerator_env_var()
    summary_rows: List[Dict[str, object]] = []

    for index, checkpoint_raw in enumerate(args.checkpoints):
        checkpoint = Path(checkpoint_raw).expanduser().resolve()
        run_name = sanitize_checkpoint_name(index, checkpoint)
        run_output_dir = output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        if args.base_model_path:
            model_path = str(Path(args.base_model_path).expanduser().resolve())
            lora_path = str(checkpoint)
        else:
            model_path = str(checkpoint)
            lora_path = None

        print(f"[{index + 1}/{len(args.checkpoints)}] Evaluating {checkpoint}")
        if args.num_shards > 1:
            processes: List[subprocess.Popen] = []
            for shard_id, device_id in enumerate(device_ids):
                cmd = [
                    sys.executable,
                    str(unified_eval_script),
                    "--model_path",
                    model_path,
                    "--eval_tasks",
                    task_spec,
                    "--output_dir",
                    str(run_output_dir),
                    "--batch_size",
                    str(args.batch_size),
                    "--temperature",
                    str(args.temperature),
                    "--max_tokens",
                    str(args.max_tokens),
                    "--max_num_seqs",
                    str(args.max_num_seqs),
                    "--gpu_memory_utilization",
                    str(args.gpu_memory_utilization),
                    "--shard_id",
                    str(shard_id),
                    "--num_shards",
                    str(args.num_shards),
                ]
                if lora_path is not None:
                    cmd.extend(["--lora_path", lora_path])

                env = os.environ.copy()
                env[accelerator_env_var] = device_id
                print(f"  shard {shard_id}: {accelerator_env_var}={device_id}")
                processes.append(subprocess.Popen(cmd, env=env))

            shard_returncodes = [proc.wait() for proc in processes]
            shard_failed = [code for code in shard_returncodes if code != 0]
            if shard_failed:
                completed = subprocess.CompletedProcess(
                    args=[],
                    returncode=shard_failed[0],
                )
            else:
                aggregate_cmd = [
                    sys.executable,
                    str(unified_eval_script),
                    "--eval_tasks",
                    task_spec,
                    "--output_dir",
                    str(run_output_dir),
                    "--aggregate",
                    "--num_shards",
                    str(args.num_shards),
                ]
                completed = subprocess.run(aggregate_cmd, check=False)
        else:
            cmd = [
                sys.executable,
                str(unified_eval_script),
                "--model_path",
                model_path,
                "--eval_tasks",
                task_spec,
                "--output_dir",
                str(run_output_dir),
                "--batch_size",
                str(args.batch_size),
                "--temperature",
                str(args.temperature),
                "--max_tokens",
                str(args.max_tokens),
                "--tensor_parallel_size",
                str(args.tensor_parallel_size),
                "--pipeline_parallel_size",
                str(args.pipeline_parallel_size),
                "--max_num_seqs",
                str(args.max_num_seqs),
                "--gpu_memory_utilization",
                str(args.gpu_memory_utilization),
                "--merged",
            ]
            if lora_path is not None:
                cmd.extend(["--lora_path", lora_path])
            completed = subprocess.run(cmd, check=False)

        row: Dict[str, object] = {
            "checkpoint": str(checkpoint),
            "output_dir": str(run_output_dir),
            "returncode": completed.returncode,
        }
        metrics_path = run_output_dir / "eval_metrics.json"
        if completed.returncode == 0 and metrics_path.is_file():
            metrics = load_metrics(metrics_path)
            row["aggregated_score"] = metrics.get("aggregated_score", 0.0)
            row["task_scores"] = metrics.get("task_scores", {})
        summary_rows.append(row)

    summary_path = output_dir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_json_path = output_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    print(f"Saved checkpoint summary to {summary_path}")


if __name__ == "__main__":
    main()
