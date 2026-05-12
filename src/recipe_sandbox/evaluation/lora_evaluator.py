"""LoRA Evaluator — Training + Evaluation Orchestration Agent.

Design informed by TokenCleaning (UCSC-REAL) experiment patterns.

Pipeline per search step:
  1. LlamaFactory CLI → LoRA SFT (llamafactory-cli train config.yaml)
  2. (Option A) vLLM native LoRA inference (enable_lora=True, no merge)
     (Option B) merge_lora → vLLM inference on merged model
  3. GPQA eval → accuracy

Key decisions:
  - Training: use llamafactory-cli (don't rewrite HF Trainer)
  - Inference: vLLM supports LoRA adapters natively — no merge needed
    via LLM(enable_lora=True) + LoRARequest(lora_local_path=adapter_dir)
  - Merge is optional fallback (scripts/merge_lora.py pattern from TokenCleaning)

Default local layout:
  - LLM: recipe_sandbox/models/base_model
  - Eval data: recipe_sandbox/data/eval/gpqa_main.jsonl
  - SAE features -> sparse storage
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from recipe_sandbox.evaluation.evaluator_base import BaseEvaluator, EvalResult
from recipe_sandbox.evaluation.npu_selector import select_idle_devices

logger = logging.getLogger(__name__)

RECIPE_SANDBOX_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_MODEL_PATH = str(RECIPE_SANDBOX_ROOT / "models" / "base_model")
DEFAULT_EVAL_DATA_PATH = str(RECIPE_SANDBOX_ROOT / "data" / "eval" / "gpqa_main.jsonl")


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _detect_device_type_from_env() -> str:
    """Detect device type from environment variables **without** importing torch.

    This avoids triggering ``torch_npu._C._npu_init()`` in the parent
    process, which would corrupt NPU state for child subprocesses launched
    via ``subprocess.Popen``.
    """
    if os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        return "npu"
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        return "gpu"
    # Probe /dev for Ascend devices
    if any(Path(f"/dev/davinci{i}").exists() for i in range(8)):
        return "npu"
    return "gpu"


def _detect_device_count_from_env() -> int:
    """Return device count from env vars without importing torch."""
    ascend = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
    if ascend:
        return len([x for x in ascend.split(",") if x.strip()])
    cuda = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if cuda:
        return len([x for x in cuda.split(",") if x.strip()])
    # Probe /dev for Ascend devices
    count = sum(1 for i in range(128) if Path(f"/dev/davinci{i}").exists())
    if count > 0:
        return count
    return 1


# -----------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------

@dataclass
class LoRATrainConfig:
    """LlamaFactory SFT parameters (supports LoRA and full fine-tuning)."""

    base_model_path: str = DEFAULT_BASE_MODEL_PATH
    template: str = "qwen"

    # Fine-tuning method: "lora" or "full"
    finetuning_type: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str = "all"

    # Training
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    cutoff_len: int = 1024

    # Efficiency
    bf16: bool = True
    flash_attn: str = "fa2"
    gradient_checkpointing: bool = False
    deepspeed: str = ""  # path to deepspeed config JSON, empty = disabled
    fsdp_config: str = ""  # path to accelerate FSDP config YAML, empty = disabled

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2

    # Pilot mode (fast iteration for search)
    pilot_max_steps: int = 50  # legacy, kept for compatibility
    pilot_num_epochs: float = 2.0  # preferred: train for N epochs in pilot mode

    @property
    def is_full_finetune(self) -> bool:
        return self.finetuning_type == "full"


@dataclass
class LoRAEvalConfig:
    """vLLM evaluation parameters."""

    eval_data_path: str = DEFAULT_EVAL_DATA_PATH
    eval_tasks: Optional[Dict[str, str]] = None  # {"gpqa": "/path/gpqa.jsonl", "gsm8k": "/path/gsm8k.jsonl"}
    batch_size: int = 64
    temperature: float = 0.7
    max_tokens: int = 4096
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1  # vLLM 0.8 sync LLM only supports PP=1
    max_num_seqs: int = 128  # concurrent sequences in vLLM scheduler
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 16384
    # vLLM LoRA mode: 'native' (no merge) or 'merge' (merge then serve)
    mode: str = "native"
    # Merged batching: all benchmarks in a single inference pass
    merged_eval: bool = True


# -----------------------------------------------------------------------
#  Script Generator
# -----------------------------------------------------------------------

class TrainEvalScriptGenerator:
    """Generates bash scripts for LlamaFactory training + vLLM evaluation.

    Follows TokenCleaning patterns but uses:
      - llamafactory-cli instead of custom HF Trainer
      - vLLM native LoRA serving instead of requiring merge
    """

    def __init__(
        self,
        workspace_dir: str,
        train_config: Optional[LoRATrainConfig] = None,
        eval_config: Optional[LoRAEvalConfig] = None,
    ):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.train_config = train_config or LoRATrainConfig()
        self.eval_config = eval_config or LoRAEvalConfig()

    # ----- Training (LlamaFactory) -----

    def _prepare_llamafactory_dataset(self, dataset_path: str) -> str:
        """Convert canonical JSONL to ShareGPT format and create dataset_info.json.

        LlamaFactory requires:
          1. A JSON file in ShareGPT format (list of {"conversations": [...]})
          2. A dataset_info.json in the same dir that maps dataset name → file
        Returns the dataset name (stem) to use in YAML config.
        """
        from recipe_sandbox.evaluation.convert_to_sharegpt import convert_canonical_to_sharegpt

        src = Path(dataset_path)
        sharegpt_path = src.parent / f"{src.stem}_sharegpt.json"
        convert_canonical_to_sharegpt(str(src), str(sharegpt_path))

        # Write dataset_info.json next to the data file
        dataset_name = f"{src.stem}_sharegpt"
        info_path = src.parent / "dataset_info.json"
        info = {
            dataset_name: {
                "file_name": sharegpt_path.name,
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
            }
        }
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("ShareGPT data → %s  |  dataset_info → %s", sharegpt_path, info_path)
        return dataset_name

    def generate_llamafactory_yaml(
        self,
        recipe_name: str,
        dataset_path: str,
        output_dir: str,
        *,
        pilot: bool = True,
    ) -> str:
        """Generate LlamaFactory YAML config.

        Automatically converts canonical JSONL → ShareGPT format.
        """
        tc = self.train_config

        # Convert data to LlamaFactory-compatible ShareGPT format
        dataset_name = self._prepare_llamafactory_dataset(dataset_path)
        dataset_dir = str(Path(dataset_path).parent)

        config = {
            # Model
            "model_name_or_path": tc.base_model_path,
            "template": tc.template,
            "trust_remote_code": True,

            # Method
            "stage": "sft",
            "do_train": True,
            "finetuning_type": tc.finetuning_type,

            # Dataset — use converted ShareGPT data
            "dataset_dir": dataset_dir,
            "dataset": dataset_name,
            "cutoff_len": tc.cutoff_len,
            "preprocessing_num_workers": 8,

            # Training
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": tc.per_device_train_batch_size,
            "gradient_accumulation_steps": tc.gradient_accumulation_steps,
            "learning_rate": tc.learning_rate,
            "lr_scheduler_type": tc.lr_scheduler_type,
            "warmup_ratio": tc.warmup_ratio,
            "bf16": tc.bf16,
            "flash_attn": tc.flash_attn,
            "gradient_checkpointing": tc.gradient_checkpointing,

            # Logging & saving
            "logging_steps": tc.logging_steps,
            "save_steps": tc.save_steps,
            "save_total_limit": tc.save_total_limit,
            "report_to": "none",
        }

        # LoRA-specific params (skip for full fine-tuning)
        if not tc.is_full_finetune:
            config["lora_rank"] = tc.lora_rank
            config["lora_alpha"] = tc.lora_alpha
            config["lora_dropout"] = tc.lora_dropout
            config["lora_target"] = tc.lora_target

        # Full fine-tuning: always enable gradient checkpointing + deepspeed
        if tc.is_full_finetune:
            config["gradient_checkpointing"] = True
            if tc.deepspeed:
                config["deepspeed"] = tc.deepspeed

        if pilot:
            # Use epochs instead of fixed steps to ensure model fits regardless of dataset size
            config["num_train_epochs"] = tc.pilot_num_epochs if hasattr(tc, 'pilot_num_epochs') and tc.pilot_num_epochs else tc.num_train_epochs
        else:
            config["num_train_epochs"] = tc.num_train_epochs

        yaml_path = str(self.workspace / "recipes" / recipe_name / f"train_{recipe_name}.yaml")
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info("LlamaFactory YAML → %s", yaml_path)
        return yaml_path

    def generate_train_script(
        self,
        recipe_name: str,
        yaml_path: str,
        device_ids: Optional[List[int]] = None,
        device_type: str = "gpu",
    ) -> str:
        """Generate bash script: llamafactory-cli train.

        When ``fsdp_config`` is set on the train config, wraps the command
        with ``accelerate launch`` for FSDP distributed training.
        """
        tc = self.train_config
        id_str = ",".join(str(g) for g in device_ids) if device_ids else "0"
        num_devices = len(device_ids) if device_ids else 1
        script_path = str(self.workspace / "recipes" / recipe_name / f"run_train_{recipe_name}.sh")
        Path(script_path).parent.mkdir(parents=True, exist_ok=True)

        if device_type == "npu":
            env_line = f"export ASCEND_RT_VISIBLE_DEVICES={id_str}"
            device_label = f"NPUs: {id_str}"
        else:
            env_line = f"export CUDA_VISIBLE_DEVICES={id_str}"
            device_label = f"GPUs: {id_str}"

        if tc.fsdp_config:
            train_cmd = (
                f"accelerate launch --config_file {tc.fsdp_config} "
                f"--num_processes {num_devices} "
                f"-m llamafactory.train.tuner {yaml_path}"
            )
        else:
            train_cmd = f"llamafactory-cli train {yaml_path}"

        script = f"""#!/bin/bash
# LlamaFactory LoRA Training — {recipe_name}
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
set -euo pipefail

{env_line}
echo "[TRAIN] Recipe: {recipe_name} | {device_label}"
echo "[TRAIN] Config: {yaml_path}"

{train_cmd}

echo "[TRAIN] Done: {recipe_name}"
"""
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        return script_path

    # ----- Evaluation (vLLM) -----

    def generate_eval_script(
        self,
        recipe_name: str,
        adapter_dir: str,
        eval_output_dir: str,
        device_ids: Optional[List[int]] = None,
        device_type: str = "gpu",
    ) -> str:
        """Generate bash script: vLLM evaluation.

        Supports:
          - Merged batching (default): all benchmarks in a single inference pass
          - Sequential (legacy): each benchmark evaluated separately
          - LoRA native serving or merge-then-serve
        """
        ec = self.eval_config
        tc = self.train_config
        gpqa_eval_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "gpqa_eval_vllm.py",
        )
        unified_eval_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "unified_eval_vllm.py",
        )
        script_path = str(self.workspace / "recipes" / recipe_name / f"run_eval_{recipe_name}.sh")
        Path(script_path).parent.mkdir(parents=True, exist_ok=True)

        if not device_ids:
            device_ids = [0]

        tp_size = ec.tensor_parallel_size
        pp_size = ec.pipeline_parallel_size if ec.pipeline_parallel_size > 1 else 1
        devices_per_eval_worker = max(tp_size * pp_size, 1)
        if len(device_ids) < devices_per_eval_worker:
            raise ValueError(
                f"Evaluation requires at least {devices_per_eval_worker} device(s) "
                f"for TP={tp_size}, PP={pp_size}, but only {len(device_ids)} provided: {device_ids}"
            )

        # Expose enough devices for a single TP × PP worker in non-sharded mode.
        eval_device_ids = device_ids[:devices_per_eval_worker]
        eval_devs = ",".join(str(d) for d in eval_device_ids)

        # ----- Merged batching mode (recommended) -----
        if ec.merged_eval and ec.eval_tasks and len(ec.eval_tasks) >= 1:
            tasks_arg = ",".join(f"{k}:{v}" for k, v in ec.eval_tasks.items())

            if tc.is_full_finetune:
                model_arg = f"--model_path {adapter_dir}"
                lora_arg = ""
            elif ec.mode == "native":
                model_arg = f"--model_path {tc.base_model_path}"
                lora_arg = f"\\\n    --lora_path {adapter_dir}"
            else:
                # Merge mode: merge first, then evaluate merged model
                merged_dir = f"{adapter_dir}_merged"
                merge_script_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "merge_lora.py",
                )
                merge_cmd = f'''echo "[EVAL] Merging LoRA adapter"
python {merge_script_path} \\
    --base_model_name_or_path {tc.base_model_path} \\
    --lora_model_name_or_path {adapter_dir} \\
    --output_dir {merged_dir} \\
    --save_tokenizer --use_fast_tokenizer
'''
                model_arg = f"--model_path {merged_dir}"
                lora_arg = ""

            merge_prefix = ""
            if ec.mode == "merge" and not tc.is_full_finetune:
                merge_prefix = merge_cmd + "\n"

            # --- Parallel sharded eval: split across device groups of TP × PP ---
            num_shards = len(device_ids) // devices_per_eval_worker
            leftover_devices = len(device_ids) % devices_per_eval_worker
            if leftover_devices:
                logger.warning(
                    "Ignoring %d leftover eval device(s) because TP=%d and PP=%d require groups of %d: %s",
                    leftover_devices,
                    tp_size,
                    pp_size,
                    devices_per_eval_worker,
                    device_ids[num_shards * devices_per_eval_worker:],
                )
            if num_shards > 1:
                shard_cmds = []
                for sid in range(num_shards):
                    shard_device_ids = device_ids[
                        sid * devices_per_eval_worker:(sid + 1) * devices_per_eval_worker
                    ]
                    shard_devs = ",".join(str(d) for d in shard_device_ids)
                    env_var = "ASCEND_RT_VISIBLE_DEVICES" if device_type == "npu" else "CUDA_VISIBLE_DEVICES"
                    shard_cmds.append(
                        f'{env_var}={shard_devs} python {unified_eval_script} \\\n'
                        f'    {model_arg} {lora_arg}\\\n'
                        f'    --eval_tasks {tasks_arg} \\\n'
                        f'    --output_dir {eval_output_dir} \\\n'
                        f'    --batch_size {ec.batch_size} \\\n'
                        f'    --temperature {ec.temperature} \\\n'
                        f'    --max_tokens {ec.max_tokens} \\\n'
                        f'    --tensor_parallel_size {tp_size} \\\n'
                        f'    --max_num_seqs {ec.max_num_seqs} \\\n'
                        f'    --gpu_memory_utilization {ec.gpu_memory_utilization} \\\n'
                        f'    --merged \\\n'
                        f'    --shard_id {sid} --num_shards {num_shards} &'
                    )

                all_shard_cmds = "\n\n".join(shard_cmds)
                aggregate_cmd = (
                    f'python {unified_eval_script} \\\n'
                    f'    --eval_tasks {tasks_arg} \\\n'
                    f'    --output_dir {eval_output_dir} \\\n'
                    f'    --aggregate --num_shards {num_shards}'
                )

                eval_body = f'''
echo "[EVAL] Parallel sharded eval ({num_shards} GPUs, TP={tp_size})"
{merge_prefix}{all_shard_cmds}

echo "[EVAL] Waiting for all shards to complete..."
wait

echo "[EVAL] Aggregating shard results..."
{aggregate_cmd}'''
            else:
                # Single worker: use standard merged mode
                pp_arg = f"\\\n    --pipeline_parallel_size {pp_size}" if pp_size > 1 else ""

                eval_cmd = f'''python {unified_eval_script} \\
    {model_arg} {lora_arg}\\
    --eval_tasks {tasks_arg} \\
    --output_dir {eval_output_dir} \\
    --batch_size {ec.batch_size} \\
    --temperature {ec.temperature} \\
    --max_tokens {ec.max_tokens} \\
    --tensor_parallel_size {tp_size} \\
    --max_num_seqs {ec.max_num_seqs} \\
    --gpu_memory_utilization {ec.gpu_memory_utilization} \\
    --merged{pp_arg}'''

                eval_body = f'''
echo "[EVAL] Merged batching (TP={tp_size}, PP={pp_size}, max_num_seqs={ec.max_num_seqs})"
{merge_prefix}{eval_cmd}'''

        # ----- Legacy: multi-benchmark sequential -----
        elif ec.eval_tasks:
            tasks_arg = ",".join(f"{k}:{v}" for k, v in ec.eval_tasks.items())
            pp_arg = f"    --pipeline_parallel_size {pp_size} \\\n" if pp_size > 1 else ""
            gpu_mem_trailing = f"{chr(32)}{chr(92)}" if pp_arg else ""

            if tc.is_full_finetune:
                eval_body = f'''
echo "[EVAL] Full fine-tuned model eval (TP={tp_size}, PP={pp_size})"
python {unified_eval_script} \\
    --model_path {adapter_dir} \\
    --eval_tasks {tasks_arg} \\
    --output_dir {eval_output_dir} \\
    --batch_size {ec.batch_size} \\
    --temperature {ec.temperature} \\
    --max_tokens {ec.max_tokens} \\
    --tensor_parallel_size {tp_size} \\
    --gpu_memory_utilization {ec.gpu_memory_utilization}{gpu_mem_trailing}
{pp_arg}'''
            elif ec.mode == "native":
                eval_body = f'''
echo "[EVAL] Sequential multi-benchmark eval (TP={tp_size}, PP={pp_size})"
python {unified_eval_script} \\
    --model_path {tc.base_model_path} \\
    --lora_path {adapter_dir} \\
    --eval_tasks {tasks_arg} \\
    --output_dir {eval_output_dir} \\
    --batch_size {ec.batch_size} \\
    --temperature {ec.temperature} \\
    --max_tokens {ec.max_tokens} \\
    --tensor_parallel_size {tp_size} \\
    --gpu_memory_utilization {ec.gpu_memory_utilization}{gpu_mem_trailing}
{pp_arg}'''
            else:
                merged_dir = f"{adapter_dir}_merged"
                merge_script_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "merge_lora.py",
                )
                eval_body = f'''
echo "[EVAL] Merging LoRA adapter → merged model"
python {merge_script_path} \\
    --base_model_name_or_path {tc.base_model_path} \\
    --lora_model_name_or_path {adapter_dir} \\
    --output_dir {merged_dir} \\
    --save_tokenizer --use_fast_tokenizer

echo "[EVAL] Sequential eval (TP={tp_size}, PP={pp_size})"
python {unified_eval_script} \\
    --model_path {merged_dir} \\
    --eval_tasks {tasks_arg} \\
    --output_dir {eval_output_dir} \\
    --batch_size {ec.batch_size} \\
    --temperature {ec.temperature} \\
    --max_tokens {ec.max_tokens} \\
    --tensor_parallel_size {tp_size} \\
    --gpu_memory_utilization {ec.gpu_memory_utilization}{gpu_mem_trailing}
{pp_arg}'''

        # ----- Single benchmark (GPQA only) -----
        elif tc.is_full_finetune:
            pp_arg = f"    --pipeline_parallel_size {pp_size} \\\n" if pp_size > 1 else ""
            gpu_mem_trailing = f"{chr(32)}{chr(92)}" if pp_arg else ""
            eval_body = f'''
echo "[EVAL] Full fine-tuned model eval (TP={tp_size}, PP={pp_size})"
python {gpqa_eval_script} \\
    --model_path {adapter_dir} \\
    --eval_data {ec.eval_data_path} \\
    --output_dir {eval_output_dir} \\
    --batch_size {ec.batch_size} \\
    --temperature {ec.temperature} \\
    --max_tokens {ec.max_tokens} \\
    --tensor_parallel_size {tp_size} \\
    --gpu_memory_utilization {ec.gpu_memory_utilization}{gpu_mem_trailing}
{pp_arg}'''
        elif ec.mode == "native":
            pp_arg = f"    --pipeline_parallel_size {pp_size} \\\n" if pp_size > 1 else ""
            gpu_mem_trailing = f"{chr(32)}{chr(92)}" if pp_arg else ""
            eval_body = f'''
echo "[EVAL] vLLM native LoRA serving (TP={tp_size}, PP={pp_size})"
python {gpqa_eval_script} \\
    --model_path {tc.base_model_path} \\
    --lora_path {adapter_dir} \\
    --eval_data {ec.eval_data_path} \\
    --output_dir {eval_output_dir} \\
    --batch_size {ec.batch_size} \\
    --temperature {ec.temperature} \\
    --max_tokens {ec.max_tokens} \\
    --tensor_parallel_size {tp_size} \\
    --gpu_memory_utilization {ec.gpu_memory_utilization}{gpu_mem_trailing}
{pp_arg}'''
        else:
            merged_dir = f"{adapter_dir}_merged"
            merge_script_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "merge_lora.py",
            )
            pp_arg = f"    --pipeline_parallel_size {pp_size} \\\n" if pp_size > 1 else ""
            gpu_mem_trailing = f"{chr(32)}{chr(92)}" if pp_arg else ""
            eval_body = f'''
echo "[EVAL] Merging LoRA adapter → merged model"
python {merge_script_path} \\
    --base_model_name_or_path {tc.base_model_path} \\
    --lora_model_name_or_path {adapter_dir} \\
    --output_dir {merged_dir} \\
    --save_tokenizer --use_fast_tokenizer

echo "[EVAL] Evaluating merged model (TP={tp_size}, PP={pp_size})"
python {gpqa_eval_script} \\
    --model_path {merged_dir} \\
    --eval_data {ec.eval_data_path} \\
    --output_dir {eval_output_dir} \\
    --batch_size {ec.batch_size} \\
    --temperature {ec.temperature} \\
    --max_tokens {ec.max_tokens} \\
    --tensor_parallel_size {tp_size} \\
    --gpu_memory_utilization {ec.gpu_memory_utilization}{gpu_mem_trailing}
{pp_arg}'''

        # For parallel sharded mode, each shard manages its own CUDA_VISIBLE_DEVICES,
        # so we only set the global env for non-parallel modes.
        shard_count = len(device_ids) // devices_per_eval_worker
        is_parallel_sharded = (
            ec.merged_eval and ec.eval_tasks and len(ec.eval_tasks) >= 1
            and shard_count > 1
        )

        if device_type == "npu":
            env_line = f"export ASCEND_RT_VISIBLE_DEVICES={eval_devs}" if not is_parallel_sharded else "# Per-shard device assignment (see below)"
            device_label = (
                f"NPU: {','.join(str(d) for d in device_ids[:shard_count * devices_per_eval_worker])} "
                f"(shards={shard_count}, devices_per_shard={devices_per_eval_worker}, TP={tp_size}, PP={pp_size})"
                if is_parallel_sharded
                else f"NPU: {eval_devs} (TP={tp_size}, PP={pp_size})"
            )
        else:
            env_line = f"export CUDA_VISIBLE_DEVICES={eval_devs}" if not is_parallel_sharded else "# Per-shard CUDA_VISIBLE_DEVICES (see below)"
            device_label = (
                f"GPU: {','.join(str(d) for d in device_ids[:shard_count * devices_per_eval_worker])} "
                f"(shards={shard_count}, devices_per_shard={devices_per_eval_worker}, TP={tp_size}, PP={pp_size})"
                if is_parallel_sharded
                else f"GPU: {eval_devs} (TP={tp_size}, PP={pp_size})"
            )

        script = f"""#!/bin/bash
# vLLM Evaluation — {recipe_name} (mode={ec.mode}, merged={ec.merged_eval})
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
set -euo pipefail

{env_line}
echo "[EVAL] Recipe: {recipe_name} | {device_label}"
{eval_body}
echo "[EVAL] Done: {recipe_name}"
"""
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        return script_path

    # ----- Full Pipeline -----

    def generate_full_pipeline(
        self,
        recipe_name: str,
        dataset_path: str,
        adapter_output_dir: str,
        eval_output_dir: str,
        device_ids: Optional[List[int]] = None,
        device_type: str = "gpu",
        *,
        pilot: bool = True,
    ) -> Dict[str, str]:
        """Generate all scripts for train → eval pipeline."""

        yaml_path = self.generate_llamafactory_yaml(
            recipe_name, dataset_path, adapter_output_dir, pilot=pilot,
        )
        train_sh = self.generate_train_script(
            recipe_name, yaml_path, device_ids, device_type,
        )
        eval_sh = self.generate_eval_script(
            recipe_name, adapter_output_dir, eval_output_dir, device_ids, device_type,
        )

        # Combined run-all
        run_all_path = str(self.workspace / "recipes" / recipe_name / f"run_all_{recipe_name}.sh")
        Path(run_all_path).parent.mkdir(parents=True, exist_ok=True)
        ec = self.eval_config
        if ec.eval_tasks:
            metrics_path = f"{eval_output_dir}/eval_metrics.json"
        else:
            metrics_path = f"{eval_output_dir}/gpqa_metrics.json"
        script = f"""#!/bin/bash
# Full Pipeline: {recipe_name}
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
set -euo pipefail

echo "===== STEP 1: LoRA Training ====="
bash {train_sh}

echo ""
echo "===== Wait for training processes to exit ====="
sleep 5

echo ""
echo "===== STEP 2: Evaluation ====="
bash {eval_sh}

echo ""
echo "===== Pipeline Complete ====="
if [ -f "{metrics_path}" ]; then
    echo "Metrics:"
    cat {metrics_path}
else
    echo "WARNING: metrics file not found at {metrics_path}"
fi
"""
        with open(run_all_path, "w") as f:
            f.write(script)
        os.chmod(run_all_path, 0o755)

        return {
            "yaml_config": yaml_path,
            "train_script": train_sh,
            "eval_script": eval_sh,
            "run_all_script": run_all_path,
            "adapter_dir": adapter_output_dir,
            "eval_output_dir": eval_output_dir,
            "metrics_path": metrics_path,
        }


# -----------------------------------------------------------------------
#  LoRA Evaluator (BaseEvaluator)
# -----------------------------------------------------------------------

class LoRAEvaluator(BaseEvaluator):
    """Evaluator integrating LlamaFactory + vLLM.

    Usage:
        evaluator = LoRAEvaluator(workspace_dir="runs/search_01")
        result = evaluator.evaluate(dataset_path, recipe_name)

        # result.extra contains script paths
        # User runs: bash result.extra['run_all_script']
        # Then: final = evaluator.read_metrics(result.extra['eval_output_dir'])
    """

    def __init__(
        self,
        workspace_dir: str,
        train_config: Optional[LoRATrainConfig] = None,
        eval_config: Optional[LoRAEvalConfig] = None,
        *,
        pilot: bool = True,
        auto_run: bool = False,
        gpu_ids: Optional[List[int]] = None,
        cleanup_checkpoints: bool = True,
    ):
        self.workspace = Path(workspace_dir)
        self.generator = TrainEvalScriptGenerator(
            workspace_dir, train_config, eval_config,
        )
        self.pilot = pilot
        self.auto_run = auto_run
        self._gpu_ids = gpu_ids
        self._run_counter = 0
        self._cleanup_checkpoints_enabled = cleanup_checkpoints

    def _get_device_ids(self) -> tuple[List[int], str]:
        """Return (device_ids, device_type).

        Uses environment-variable detection instead of ``torch.npu.is_available()``
        to avoid initialising NPU in the parent process.  That init would corrupt
        NPU state for child subprocesses (the training workers).
        """
        device_type = _detect_device_type_from_env()

        if self._gpu_ids:
            return self._gpu_ids, device_type

        # Detect device count from env without importing torch
        n = _detect_device_count_from_env()
        if n > 0:
            return list(range(n)), device_type

        # Fallback: try idle-device selector
        devices = select_idle_devices(min_free_mb=8000, max_util_pct=30.0)
        if not devices or devices == ["cpu"]:
            return [0], "gpu"
        dt = "npu" if devices[0].startswith("npu") else "gpu"
        ids = [int(d.split(":")[-1]) for d in devices]
        return ids, dt

    def evaluate(
        self,
        dataset_path: str,
        recipe_name: str,
        *,
        task_names: Optional[List[str]] = None,
        state_vector: Optional[Dict[str, float]] = None,
    ) -> EvalResult:
        """Generate train+eval scripts. Optionally auto-execute."""
        self._run_counter += 1
        run_id = f"{recipe_name}_r{self._run_counter}"
        # All artifacts for this run live under recipes/<run_id>/
        run_dir = str(self.workspace / "recipes" / run_id)
        adapter_dir = os.path.join(run_dir, "adapter")
        eval_dir = os.path.join(run_dir, "eval_results")
        os.makedirs(run_dir, exist_ok=True)
        recipe_data_dir = Path(run_dir) / "recipe_data"
        device_ids, device_type = self._get_device_ids()

        logger.info("[LoRAEval] %s | %s=%s | pilot=%s", run_id, device_type.upper(), device_ids, self.pilot)

        # Agent Memory Pattern: Materialize dataset if an ID list is provided
        actual_dataset_path = dataset_path
        if dataset_path.endswith(".ids.json"):
            # Guard: check if the ID list is empty → return zero-score result
            import json as _json
            import shutil as _shutil
            with open(dataset_path) as _f:
                ids_data = _json.load(_f)
            if not ids_data:
                logger.warning("Empty .ids.json at %s — returning zero-score EvalResult.", dataset_path)
                return EvalResult(dev_score=0.0, train_cost_gpu_hours=0.0, eval_cost_gpu_hours=0.0)

            source_recipe_dir = Path(dataset_path).parent
            recipe_data_dir.mkdir(parents=True, exist_ok=True)
            for artifact_name in ("dataset.ids.json", "manifest.json", "trace.json"):
                artifact_path = source_recipe_dir / artifact_name
                if artifact_path.exists():
                    _shutil.copy2(artifact_path, recipe_data_dir / artifact_name)

            from recipe_sandbox.schema.io import materialize_dataset
            canonical_files = []
            # Search for canonical train JSONL in known locations
            search_dirs = [
                self.workspace / "canonical" / "train",
                self.workspace.parent / "canonical" / "train",
            ]
            for sdir in search_dirs:
                if sdir.exists():
                    found = [str(p) for p in sdir.glob("*.jsonl")]
                    if found:
                        canonical_files = found
                        break
            
            if not canonical_files:
                raise ValueError(f"Cannot materialize {dataset_path}: no canonical train files found in {[str(d) for d in search_dirs]}")
            actual_dataset_path = str(Path(adapter_dir) / f"{run_id}_hydrated.jsonl")
            logger.info("Materializing ID list -> %s", actual_dataset_path)
            materialize_dataset(dataset_path, canonical_files, actual_dataset_path)

        paths = self.generator.generate_full_pipeline(
            recipe_name=run_id,
            dataset_path=actual_dataset_path,
            adapter_output_dir=adapter_dir,
            eval_output_dir=eval_dir,
            device_ids=device_ids,
            device_type=device_type,
            pilot=self.pilot,
        )

        if self.auto_run:
            result = self._execute_and_read(paths, run_id, device_ids=device_ids, device_type=device_type)
            if self._cleanup_checkpoints_enabled:
                self._cleanup_checkpoints(adapter_dir)
            else:
                logger.info("[Cleanup] Skipped checkpoint cleanup for %s (cleanup_checkpoints=False)", adapter_dir)
            return result

        logger.info("[LoRAEval] Scripts ready. Run:\n  bash %s", paths["run_all_script"])
        return EvalResult(
            dev_score=0.0,
            train_cost_gpu_hours=0.0,
            extra={"evaluator": "lora", "status": "scripts_generated", "run_id": run_id, **paths},
        )

    def _cleanup_checkpoints(self, adapter_dir: str) -> None:
        """Remove intermediate checkpoints and large temp files after eval.

        Keeps only the final adapter weights (adapter_model.safetensors,
        adapter_config.json) to save disk space. Deletes:
          - checkpoint-* directories (optimizer.pt, rng_state, etc.)
          - Hydrated JSONL and ShareGPT JSON files (already consumed)
          - training_args.bin, trainer_state.json (not needed for inference)
        """
        import shutil
        adapter_path = Path(adapter_dir)
        if not adapter_path.exists():
            return

        freed_bytes = 0
        # 1. Delete checkpoint-* subdirectories
        for ckpt_dir in sorted(adapter_path.glob("checkpoint-*")):
            if ckpt_dir.is_dir():
                size = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
                shutil.rmtree(ckpt_dir)
                freed_bytes += size
                logger.info("[Cleanup] Removed checkpoint: %s (%.1fMB)", ckpt_dir.name, size / 1024**2)

        # 2. Delete hydrated data files (already consumed by training)
        for pattern in ("*_hydrated.jsonl", "*_sharegpt.json", "dataset_info.json"):
            for f in adapter_path.glob(pattern):
                size = f.stat().st_size
                f.unlink()
                freed_bytes += size
                logger.info("[Cleanup] Removed temp data: %s (%.1fMB)", f.name, size / 1024**2)

        # 3. Delete optimizer/training snapshots not needed for vLLM inference
        for fname in ("training_args.bin", "trainer_state.json", "trainer_log.jsonl",
                       "train_results.json", "all_results.json"):
            p = adapter_path / fname
            if p.exists():
                freed_bytes += p.stat().st_size
                p.unlink()

        if freed_bytes > 0:
            logger.info("[Cleanup] Total freed: %.1fMB in %s", freed_bytes / 1024**2, adapter_dir)

    def _execute_and_read(
        self,
        paths: Dict,
        run_id: str,
        *,
        device_ids: Optional[List[int]] = None,
        device_type: str = "gpu",
    ) -> EvalResult:
        """Execute pipeline and read metrics with real-time log streaming.

        IMPORTANT: gpu_cleanup functions (``ensure_devices_free``,
        ``release_all_gpu_memory``) are imported **lazily** and ONLY when
        actually needed.  Importing ``gpu_cleanup`` eagerly would trigger
        ``torch.npu.is_available()`` → ``_npu_init()`` in the parent process,
        which corrupts NPU state for child subprocesses on Ascend hardware.
        """
        import subprocess as sp

        skip_device_cleanup = _env_enabled("RECIPE_SANDBOX_SKIP_DEVICE_CLEANUP")
        skip_current_cleanup = _env_enabled("RECIPE_SANDBOX_SKIP_PRELAUNCH_CLEANUP")

        # Kill stale processes on target devices before launching
        if device_ids and not skip_device_cleanup:
            from recipe_sandbox.evaluation.gpu_cleanup import ensure_devices_free
            ensure_devices_free(device_ids, device_type)
        elif device_ids:
            logger.info(
                "[LoRAEval] Skipping ensure_devices_free for %s=%s "
                "(RECIPE_SANDBOX_SKIP_DEVICE_CLEANUP=1)",
                device_type.upper(),
                device_ids,
            )

        # Release current process's device memory (SAE ingest, etc.)
        if not skip_current_cleanup:
            from recipe_sandbox.evaluation.gpu_cleanup import release_all_gpu_memory
            release_all_gpu_memory(wait_seconds=1.0)
        else:
            logger.info(
                "[LoRAEval] Skipping prelaunch device cleanup "
                "(RECIPE_SANDBOX_SKIP_PRELAUNCH_CLEANUP=1)"
            )

        start = time.time()
        logger.info("[LoRAEval] Executing: bash %s", paths["run_all_script"])

        process = sp.Popen(
            ["bash", paths["run_all_script"]],
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )

        output_lines: List[str] = []
        for line in process.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
            logger.info("[%s] %s", run_id, line)

        returncode = process.wait()
        elapsed_h = (time.time() - start) / 3600.0

        # Post-run cleanup (safe to import gpu_cleanup here — training is done)
        try:
            from recipe_sandbox.evaluation.gpu_cleanup import (
                ensure_devices_free,
                release_all_gpu_memory,
            )
            if device_ids:
                ensure_devices_free(device_ids, device_type)
            release_all_gpu_memory(wait_seconds=1.0)
        except Exception as exc:
            logger.warning("[LoRAEval] Post-run cleanup failed (non-fatal): %s", exc)

        if returncode != 0:
            tail = "\n".join(output_lines[-50:])
            logger.warning("[LoRAEval] %s exited with rc=%d, checking for metrics anyway ...", run_id, returncode)
            # Try to salvage metrics even if script had a non-zero exit code
            # (e.g. trailing syntax error after eval already wrote results)
            salvaged = self.read_metrics(paths["eval_output_dir"], elapsed_h)
            if salvaged.dev_score > 0.0:
                logger.info("[LoRAEval] Salvaged valid score %.4f from %s despite rc=%d",
                            salvaged.dev_score, run_id, returncode)
                salvaged.extra["returncode"] = returncode
                salvaged.extra["status"] = "completed_with_warnings"
                return salvaged
            logger.error("[LoRAEval] FAILED %s (rc=%d), no salvageable metrics:\n%s",
                         run_id, returncode, tail)
            return EvalResult(
                dev_score=0.0,
                train_cost_gpu_hours=elapsed_h,
                extra={"evaluator": "lora", "status": "failed", "returncode": returncode,
                        "tail": tail[-2000:]},
            )

        logger.info("[LoRAEval] %s completed in %.2f hours", run_id, elapsed_h)
        return self.read_metrics(paths["eval_output_dir"], elapsed_h)

    def read_metrics(self, eval_output_dir: str, train_cost_hours: float = 0.0) -> EvalResult:
        """Read metrics from a completed run.

        Checks for unified eval_metrics.json first (multi-benchmark),
        then falls back to gpqa_metrics.json (single-benchmark).
        """
        unified_path = os.path.join(eval_output_dir, "eval_metrics.json")
        gpqa_path = os.path.join(eval_output_dir, "gpqa_metrics.json")

        if os.path.exists(unified_path):
            with open(unified_path) as f:
                metrics = json.load(f)
            task_scores = metrics.get("task_scores", {})
            agg = metrics.get("aggregated_score", 0.0)
            return EvalResult(
                dev_score=agg * 100.0,
                train_cost_gpu_hours=train_cost_hours,
                eval_cost_gpu_hours=0.01,
                task_scores=task_scores,
                extra={"evaluator": "lora", "status": "completed", "raw_metrics": metrics},
            )
        elif os.path.exists(gpqa_path):
            with open(gpqa_path) as f:
                metrics = json.load(f)
            accuracy = metrics.get("accuracy", 0.0)
            return EvalResult(
                dev_score=accuracy * 100.0,
                train_cost_gpu_hours=train_cost_hours,
                eval_cost_gpu_hours=0.01,
                task_scores={"gpqa_extended": accuracy},
                extra={"evaluator": "lora", "status": "completed", "raw_metrics": metrics},
            )
        else:
            logger.warning("No metrics file found in %s", eval_output_dir)
            return EvalResult(
                dev_score=0.0, train_cost_gpu_hours=train_cost_hours,
                extra={"evaluator": "lora", "status": "metrics_not_found"},
            )
