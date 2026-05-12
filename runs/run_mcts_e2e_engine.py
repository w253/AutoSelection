#!/usr/bin/env python
"""MCTS AutoSelection v2 — Fully Automatic E2E Engine.

Pipeline:
  1. (Optional) Ingest raw data via AgentMapper if --task_config is provided.
  2. SAE Feature Ingestion (extracts sparse activations over the canonical datasets).
  3. MCTS Search Loop:
       - Uses LHS for cold start.
       - Uses GP Surrogate Model (Left Brain) for UCB evaluation.
       - Uses LLM ActionGenerator (Right Brain) for candidate proposal.
       - Selects candidates with SelectionLLM.
       - Evaluates candidates via full fine-tuning + vLLM.
       - Supports RESUME FROM CHECKPOINT (search_log.jsonl).
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import os
import random
import shutil
import numpy as np
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from recipe_sandbox.evaluation.lora_evaluator import (
    LoRAEvaluator,
    LoRATrainConfig,
    LoRAEvalConfig,    
)
from recipe_sandbox.pipeline.pipeline_orchestrator import PipelineOrchestrator
from recipe_sandbox.pipeline.recipe_executor import RecipeExecutor, build_default_operator_registry
from recipe_sandbox.pipeline.task_config import TaskConfig, ModelConfig
from recipe_sandbox.pipeline.task_manager import TaskManager
from recipe_sandbox.search.mcts_search import MCTSSearchLoop
from recipe_sandbox.search.operator_policy import resolve_operator_space
from recipe_sandbox.agents.action_llm import ActionLLMGenerator, LLMConfig
from recipe_sandbox.agents.feedback_llm import FeedbackLLM
from recipe_sandbox.agents.selection_llm import SelectionLLM
from recipe_sandbox.extensions import load_extensions, parse_extension_modules
from recipe_sandbox.schema.io import iter_samples_without_numeric_extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mcts_e2e")
_CACHE_SAVE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-writer")


RECIPE_SANDBOX_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = RECIPE_SANDBOX_ROOT / "data"
DEFAULT_MODEL_DIR = RECIPE_SANDBOX_ROOT / "models"
DEFAULT_BASE_MODEL = str(DEFAULT_MODEL_DIR / "base_model")
DEFAULT_SAE_PATH = str(DEFAULT_MODEL_DIR / "sae" / "layers.27")
_DEFAULT_CANONICAL_DATA = DEFAULT_DATA_DIR / "train3" / "merged_data.jsonl"
if not _DEFAULT_CANONICAL_DATA.is_file() and _DEFAULT_CANONICAL_DATA.with_suffix(
    _DEFAULT_CANONICAL_DATA.suffix + ".bak"
).is_file():
    _DEFAULT_CANONICAL_DATA = _DEFAULT_CANONICAL_DATA.with_suffix(
        _DEFAULT_CANONICAL_DATA.suffix + ".bak"
    )
CANONICAL_DATA = str(_DEFAULT_CANONICAL_DATA)
DEFAULT_EVAL_DATA = str(DEFAULT_DATA_DIR / "target_vector_samples")
DEFAULT_EVAL_NORM_DIR = ""


BENCHMARK_NAME_ALIASES = {
    "gpqa_ext_98": "gpqa",
    "gpqa_extended": "gpqa",
    "gsm8k_train_100": "gsm8k",
    "gsm8k_test": "gsm8k",
    "bbh_few_shot": "bbh",
    "bbh_test": "bbh",
    "mmlu_val": "mmlu",
    "mmlu_test": "mmlu",
}


def _setup_file_logging(log_dir: str) -> None:
    """Add a file handler to the root logger for persistent experiment logs."""
    log_path = os.path.join(log_dir, "engine.log")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    logger.info("File logging enabled → %s", log_path)

def _merge_jsonl_files(source_files: List[str], output_path: str) -> int:
    """Concatenate multiple JSONL files into a single output file. Returns total line count."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for src in source_files:
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        total += 1
    logger.info("Merged %d lines from %d files → %s", total, len(source_files), output_path)
    return total


def _normalize_benchmark_name(raw_name: str) -> str:
    stem = Path(raw_name).stem
    return BENCHMARK_NAME_ALIASES.get(stem, stem)


def _detect_all_accelerator_ids() -> tuple[List[int], str]:
    """Detect accelerator IDs preferring env-var probing over torch init.

    For the MCTS engine this is less critical (torch will be imported for
    SAE ingest anyway), but keeping it consistent avoids surprising
    NPU-init side effects if called before ingest.
    """
    from recipe_sandbox.evaluation.lora_evaluator import (
        _detect_device_count_from_env,
        _detect_device_type_from_env,
    )

    device_type = _detect_device_type_from_env()
    n = _detect_device_count_from_env()
    if n > 0:
        return list(range(n)), device_type
    return [0], "gpu"


def _restore_numeric_features_from_npz(samples, npz_path: str) -> int:
    """Restore pre-computed scores from npz into sample metadata.
    
    The pool.jsonl stores only text data. Numerical features live in train_combined.npz.
    This function re-hydrates samples by matching sample_id.
    """
    data = np.load(npz_path, allow_pickle=True)
    scores_array = data["scores"] if "scores" in data.files else None
    sample_ids_array = data.get("sample_ids")
    indices_array = data.get("indices")
    values_array = data.get("values")
    ngram_arr = data.get("ngram_entropy")
    act_arr = data.get("action_object")
    ifd_arr = data.get("ifd")
    ve_arr = data.get("varentropy")

    # Detect per-benchmark MONA score arrays (keys like mona_bm_gpqa, mona_bm_gsm8k, ...)
    per_bm_arrays: Dict[str, Any] = {}
    for key in data.files:
        if key.startswith("mona_bm_"):
            bm_stem = key[len("mona_bm_"):]
            per_bm_arrays[bm_stem] = data[key]
    
    loaded_count = 0
    sparse_restored_count = 0

    def _restore_sparse_topk(sample, ii: int) -> None:
        nonlocal sparse_restored_count
        if indices_array is None or values_array is None:
            return
        if ii >= len(indices_array) or ii >= len(values_array):
            return
        idxs = np.asarray(indices_array[ii], dtype=np.int32)
        vals = np.asarray(values_array[ii], dtype=np.float32)
        if idxs.size == 0:
            return
        sample.metadata.extra["sae_topk"] = {
            "indices": idxs.tolist(),
            "values": vals.tolist(),
        }
        sparse_restored_count += 1

    if sample_ids_array is not None:
        cache_map = {sid: i for i, sid in enumerate(sample_ids_array)}
        for s in samples:
            if s.sample_id in cache_map:
                i = cache_map[s.sample_id]
                _restore_sparse_topk(s, i)
                if scores_array is not None:
                    s.metadata.extra["mona_score"] = float(scores_array[i])
                
                def _restore(arr, key, ii=i):
                    if arr is not None and not np.isnan(arr[ii]):
                        if key not in s.metadata.extra:
                            s.metadata.extra[key] = {}
                        s.metadata.extra[key]["score"] = float(arr[ii])
                
                _restore(ngram_arr, "ngram_entropy")
                _restore(act_arr, "action_object")
                _restore(ifd_arr, "ifd")
                _restore(ve_arr, "varentropy")

                # Per-benchmark MONA scores
                if per_bm_arrays:
                    mona_scores = {}
                    for bm_stem, bm_arr in per_bm_arrays.items():
                        val = float(bm_arr[i])
                        if not np.isnan(val):
                            mona_scores[bm_stem] = val
                    if mona_scores:
                        s.metadata.extra["mona_scores"] = mona_scores

                loaded_count += 1
    else:
        for i, s in enumerate(samples):
            _restore_sparse_topk(s, i)
            if scores_array is not None and i < len(scores_array):
                s.metadata.extra["mona_score"] = float(scores_array[i])
                loaded_count += 1
    
    if per_bm_arrays:
        logger.info(
            "Restored numeric features for %d/%d samples from npz "
            "(sparse_topk=%d, incl. %d per-benchmark MONA scores)",
            loaded_count,
            len(samples),
            sparse_restored_count,
            len(per_bm_arrays),
        )
    else:
        logger.info(
            "Restored numeric features for %d/%d samples from npz (sparse_topk=%d)",
            loaded_count,
            len(samples),
            sparse_restored_count,
        )
    return loaded_count


def _save_npz_cache(npz_path: str, save_kwargs: Dict[str, Any]) -> None:
    start = time.time()
    logger.info("Background cache save started → %s", npz_path)
    np.savez_compressed(npz_path, **save_kwargs)
    logger.info("Background cache save finished in %.2fs → %s", time.time() - start, npz_path)


def _await_pending_cache_save(cache_future: Optional[Future], npz_path: Path) -> None:
    if cache_future is None:
        return
    if not cache_future.done():
        logger.info("Waiting for pending cache save to finish → %s", npz_path)
    cache_future.result()


def _parse_bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _sample_has_all_benchmark_scores(sample: Any, benchmark_names: List[str]) -> bool:
    if not benchmark_names:
        return True
    mona_scores = sample.metadata.extra.get("mona_scores")
    if not isinstance(mona_scores, dict):
        return False
    return all(name in mona_scores for name in benchmark_names)


def _summarize_train_cache_readiness(samples: List[Any], benchmark_names: List[str]) -> Dict[str, Any]:
    total = len(samples)
    sparse_ready = sum(1 for sample in samples if sample.metadata.extra.get("sae_topk"))
    score_ready = sum(
        1 for sample in samples if _sample_has_all_benchmark_scores(sample, benchmark_names)
    )
    return {
        "total": total,
        "sparse_ready": sparse_ready,
        "score_ready": score_ready,
        "benchmarks": list(benchmark_names),
    }


def _stage_eval_canonical_files(
    pool_eval_dir: Path,
    eval_canonical_files: List[str],
    target_benchmark_by_file: Dict[str, str],
) -> List[str]:
    """Refresh working eval shards without deleting live source files."""

    resolved_sources = {}
    for eval_file in eval_canonical_files:
        src = Path(eval_file)
        try:
            resolved_sources[src.resolve()] = src
        except FileNotFoundError:
            resolved_sources[src] = src

    for stale_eval in pool_eval_dir.glob("*.jsonl"):
        stale_key = stale_eval.resolve()
        if stale_key in resolved_sources:
            continue
        stale_eval.unlink()
        logger.info("Removed stale canonical eval shard → %s", stale_eval)

    staged_files: List[str] = []
    for eval_file in eval_canonical_files:
        src = Path(eval_file)
        dst = pool_eval_dir / src.name
        try:
            same_path = src.resolve() == dst.resolve()
        except FileNotFoundError:
            same_path = False

        benchmark_name = target_benchmark_by_file.get(str(src))
        if same_path:
            staged_files.append(str(dst))
            if benchmark_name:
                target_benchmark_by_file[str(dst)] = benchmark_name
            continue

        shutil.copy2(str(src), str(dst))
        staged_files.append(str(dst))
        if benchmark_name:
            target_benchmark_by_file[str(dst)] = benchmark_name
    return staged_files

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Ingestion Config (New Feature from run_task.py integration)
    parser.add_argument("--task_config", type=str, default=None, help="Path to task config JSON file for raw data mapping/ingest.")
    parser.add_argument("--raw_train_data", type=str, default="", help="Comma separated paths to raw train data")
    parser.add_argument(
        "--target_vector_data",
        type=str,
        default="",
        help="Comma separated paths or directories used only to build target vectors. "
             "If set, this takes precedence over --eval_data for MONA prototype construction.",
    )
    
    # Search
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--n_lhs_seeds", type=int, default=3)
    
    # Data Mode
    parser.add_argument("--data_path", type=str, default=CANONICAL_DATA)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--eval_data", type=str, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--eval_data_norm_dir", type=str, default=DEFAULT_EVAL_NORM_DIR)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--resume", action="store_true", help="Resume from search_log.jsonl in output_dir")
    
    # Model / SAE
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sae_path", type=str, default=DEFAULT_SAE_PATH)
    parser.add_argument("--sae_top_k", type=int, default=192)
    parser.add_argument("--sae_batch_size", type=int, default=2)
    parser.add_argument("--sae_max_length", type=int, default=512)
    parser.add_argument("--sae_device", type=str, default=None)
    parser.add_argument("--sae_cache_from", type=str, default="",
                        help="Copy sae_caches from a previous run directory to avoid re-computing SAE features.")
    parser.add_argument("--cpu_max_workers", type=int, default=None, help="Max CPU worker threads for ngram/ACT heuristics. Defaults to os.cpu_count().")
    
    # LLM Generator
    parser.add_argument("--llm_api_key", type=str, default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--llm_base_url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--llm_model", type=str, default=os.getenv("LLM_MODEL", "gpt-4o"))
    parser.add_argument("--thinking_model", type=str, default=os.getenv("THINKING_MODEL", ""),
                        help="Model endpoint for reasoning/thinking LLM (used for all 3 LLM agents). "
                             "If empty, falls back to --llm_model.")
    parser.add_argument("--operator_catalog", type=str, default=os.getenv("OPERATOR_CATALOG", "examples/recipes/operator_catalog.yaml"),
                        help="YAML catalog used by the LLM proposer to describe available operators.")
    parser.add_argument("--extension_modules", type=str, default=os.getenv("RECIPE_SANDBOX_EXTENSIONS", os.getenv("EXTENSION_MODULES", "")),
                        help="Comma-separated Python modules that register custom operators/hooks.")
    
    # Full fine-tuning / Eval Params
    parser.add_argument("--deepspeed", type=str, default="",
                        help="Path to DeepSpeed config JSON for full fine-tuning. Empty = auto-detect.")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--train_gpu_ids", type=str, default=None)
    
    # VLLM Eval
    parser.add_argument("--vllm_mode", type=str, default="native", choices=["native", "merge"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_tasks", type=str, default="",
                        help="Multi-benchmark eval tasks: gpqa:/path/gpqa.jsonl,gsm8k:/path/gsm8k.jsonl")
    
    parser.add_argument(
        "--cleanup_checkpoints",
        type=_parse_bool_arg,
        default=True,
        help="Whether to delete recipe checkpoints/temp training artifacts after eval completes.",
    )

    parser.add_argument("--stagnation_patience", type=int, default=3,
                        help="Number of non-improving iterations before starting a new trajectory (default: 3)")

    return parser.parse_args()


def run_e2e():
    args = parse_args()
    
    if not args.output_dir:
        if args.resume:
            raise ValueError("Safety check: --resume cannot be used without explicitly setting --output_dir. Please provide the exact directory.")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"runs/e2e_mcts_{timestamp}")
    else:
        out_dir = Path(args.output_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logging(str(out_dir))
    logger.info("Initializing Workspace in %s", out_dir)
    
    # 1. Pipeline Orchestrator Raw Data Ingest 
    data_path = args.data_path
    
    # Pre-copy canonical/mappings from a previous run to avoid re-ingesting via LLM
    _has_precopied_canonical = False
    if args.sae_cache_from:
        import shutil
        src_run = Path(args.sae_cache_from)
        for subdir in ["canonical", "mappings"]:
            src_sub = src_run / subdir
            dst_sub = out_dir / subdir
            if src_sub.exists() and not dst_sub.exists():
                shutil.copytree(str(src_sub), str(dst_sub))
                logger.info("Copied %s/ from previous run → %s", subdir, dst_sub)
                if subdir == "canonical":
                    _has_precopied_canonical = True
    
    target_benchmark_by_file: Dict[str, str] = {}

    if args.task_config or args.raw_train_data or args.target_vector_data:
        from recipe_sandbox.pipeline.task_config import DataSourceConfig
        
        if args.task_config:
            logger.info("TaskConfig detected: Initiating Ingest-Only Flow using PipelineOrchestrator...")
            cfg = TaskConfig.load(args.task_config)
        else:
            logger.info("Raw data flags detected: Building TaskConfig dynamically for Auto Ingestion...")
            
            def _expand_path(raw_arg: str, prefix: str) -> List[DataSourceConfig]:
                sources = []
                for p in raw_arg.split(","):
                    p = p.strip()
                    if not p: continue
                    path_obj = Path(p)
                    if path_obj.is_dir():
                        for i, f in enumerate(sorted(path_obj.glob("*.json*"))):
                            sources.append(DataSourceConfig(path=str(f), source_name=f"{prefix}_{i}", format="auto"))
                    else:
                        sources.append(DataSourceConfig(path=str(path_obj), source_name=f"{prefix}_{len(sources)}", format="auto"))
                return sources

            train_sources = _expand_path(args.raw_train_data, "train_raw")
            eval_sources = _expand_path(args.target_vector_data, "target_raw")
            
            cfg = TaskConfig(
                task_name=out_dir.name,
                output_dir=str(out_dir),
                train_sources=train_sources,
                eval_sources=eval_sources,
                model=ModelConfig(model_path=args.base_model, sae_path=args.sae_path),
                llm=LLMConfig(api_key=args.llm_api_key, base_url=args.llm_base_url, model=args.llm_model)
            )

        if args.target_vector_data:
            logger.info("Overriding canonical eval/target-vector sources from --target_vector_data: %s", args.target_vector_data)
            cfg.eval_sources = _expand_path(args.target_vector_data, "target_raw")

        # Skip LLM-based ingestion if canonical was pre-copied from previous run
        if _has_precopied_canonical:
            logger.info("Canonical data pre-copied from %s — skipping orchestrator ingestion.", args.sae_cache_from)
            # Still need to save the task_config for downstream use
            manager_tmp = TaskManager(cfg)
            manager_tmp.save_config()
        else:
            orchestrator = PipelineOrchestrator(cfg, llm_client=None)
            orchestrator.run_ingest_only()
        
        # Collect canonical file paths (keep separate, one per source)
        _mgr = manager_tmp if _has_precopied_canonical else orchestrator.manager
        train_canonical_files = []
        if cfg.train_sources:
            for s in cfg.train_sources:
                cp = _mgr.canonical_path("train", s.source_name)
                if cp.exists():
                    train_canonical_files.append(str(cp))
            logger.info("Train canonical files (%d): %s", len(train_canonical_files), train_canonical_files)
            
        eval_canonical_files = []
        if cfg.eval_sources:
            for s in cfg.eval_sources:
                cp = _mgr.canonical_path("eval", s.source_name)
                if cp.exists():
                    eval_canonical_files.append(str(cp))
                    target_benchmark_by_file[str(cp)] = _normalize_benchmark_name(Path(s.path).stem)
            logger.info(
                "Target-vector canonical files (%d): %s",
                len(eval_canonical_files),
                eval_canonical_files,
            )
            
    else:
        # Single file mode (legacy)
        train_canonical_files = [data_path] if data_path else []
        eval_canonical_files = []
        target_source_arg = args.target_vector_data or args.eval_data
        if target_source_arg:
            for p in str(target_source_arg).split(","):
                p = p.strip()
                if not p:
                    continue
                path_obj = Path(p)
                if path_obj.is_dir():
                    for f in sorted(path_obj.glob("*.json*")):
                        eval_canonical_files.append(str(f))
                        target_benchmark_by_file[str(f)] = _normalize_benchmark_name(f.stem)
                else:
                    eval_canonical_files.append(str(path_obj))
                    target_benchmark_by_file[str(path_obj)] = _normalize_benchmark_name(path_obj.stem)
    
    # Fallback: if orchestrator path produced no train files, use --data_path
    if not train_canonical_files and data_path:
        logger.info("No train canonical from orchestrator; falling back to --data_path: %s", data_path)
        train_canonical_files = [data_path]
    
    # Detect d_sae early so downstream MonaScorer can use it
    detected_d_sae = None
    if args.sae_path:
        from recipe_sandbox.scoring.sparse_features import _detect_d_sae
        detected_d_sae = _detect_d_sae(args.sae_path)
        logger.info("Detected d_sae = %d from %s", detected_d_sae, args.sae_path)
    
    # ── Directory Layout ──────────────────────────────────────────────
    # canonical/         ← Active working pool used by RecipeExecutor
    #   train/pool.jsonl
    #   eval/*.jsonl
    # sae_caches/        ← Per-source + unified SAE feature caches
    # ─────────────────────────────────────────────────────────────────

    # The TaskConfig for RecipeExecutor now uses the run root directly.
    task_config = TaskConfig(
        task_name=out_dir.name,
        output_dir=str(out_dir),
        model=ModelConfig(model_path=args.base_model, sae_path=args.sae_path, d_sae=detected_d_sae)
    )
    manager = TaskManager(task_config)
    
    # 3. Load train data: from search pool if exists (contains SAE metadata), else from canonical/
    from recipe_sandbox.schema.io import read_jsonl, write_jsonl
    
    pool_train_dir = out_dir / "canonical" / "train"
    pool_train_path = pool_train_dir / "pool.jsonl"
    all_train_samples = []
    required_benchmark_names = sorted(set(target_benchmark_by_file.values()))
    
    # ---- Step 1: Load samples (text from pool.jsonl, or raw canonical) ----
    if pool_train_path.exists():
        logger.info("Found existing working pool at %s. Loading text-only pool.", pool_train_path)
        all_train_samples = list(read_jsonl(str(pool_train_path)))
        logger.info("Total loaded from pool: %d samples", len(all_train_samples))
    else:
        logger.info("No prior working pool found. Initializing from canonical sources...")
        for tidx, train_file in enumerate(train_canonical_files):
            samples = list(read_jsonl(train_file))
            if args.subsample and args.subsample < len(samples):
                samples = samples[:args.subsample]
                logger.info("[Load %d/%d] %s: took first %d samples (subsample)", tidx + 1, len(train_canonical_files), Path(train_file).name, len(samples))
            else:
                logger.info("[Load %d/%d] %s: %d samples", tidx + 1, len(train_canonical_files), Path(train_file).name, len(samples))
            all_train_samples.extend(samples)
        logger.info("Total initial load: %d samples from %d source(s)", len(all_train_samples), len(train_canonical_files))
    
    # ---- Step 2: Cache directory + optional copy from prior run ----
    sae_cache_dir = out_dir / "sae_caches"
    sae_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.sae_cache_from:
        import shutil
        src_cache = Path(args.sae_cache_from) / "sae_caches"
        if src_cache.exists():
            for npz_file in src_cache.glob("*.npz"):
                dst = sae_cache_dir / npz_file.name
                if not dst.exists():
                    shutil.copy2(str(npz_file), str(dst))
                    logger.info("Copied SAE cache: %s → %s", npz_file.name, dst)
        else:
            logger.warning("--sae_cache_from specified but %s does not exist. Will compute SAE.", src_cache)

    # ---- Step 3: Unified restore from train_combined.npz ----
    sparse_cache = None
    pending_train_cache_save: Optional[Future] = None
    skip_sae = False
    train_npz_path = sae_cache_dir / "train_combined.npz"

    if train_npz_path.exists() and all_train_samples:
        _restore_numeric_features_from_npz(all_train_samples, str(train_npz_path))
        cache_status = _summarize_train_cache_readiness(all_train_samples, required_benchmark_names)
        logger.info(
            "Cache restore: sparse_topk=%d/%d, per-benchmark MONA=%d/%d, benchmarks=%s",
            cache_status["sparse_ready"],
            cache_status["total"],
            cache_status["score_ready"],
            cache_status["total"],
            cache_status["benchmarks"],
        )
        if (cache_status["total"] > 0
                and cache_status["sparse_ready"] == cache_status["total"]
                and cache_status["score_ready"] == cache_status["total"]):
            skip_sae = True
            logger.info(
                "Cache is complete (sparse features + per-benchmark MONA present). "
                "Skipping SAE ingest and MONA recomputation."
            )
    elif not train_npz_path.exists() and args.sae_cache_from:
        logger.warning(
            "SAE caches copied from %s but train_combined.npz is MISSING "
            "(prior run likely interrupted during train ingest). "
            "Will re-compute SAE features + MONA scores.", args.sae_cache_from
        )

    # ---- Step 4: Build SparseFeatureCache from in-memory samples (skip_sae path) ----
    if args.sae_path and skip_sae and sparse_cache is None and all_train_samples:
        from recipe_sandbox.scoring.sparse_features import SparseFeatureCache, has_sparse_features
        sparse_ready = sum(1 for s in all_train_samples if has_sparse_features(s))
        if sparse_ready == len(all_train_samples):
            if detected_d_sae is None:
                raise ValueError(f"d_sae is unavailable for sparse cache build: {args.sae_path}")
            sparse_cache = SparseFeatureCache.from_samples(all_train_samples, d_sae=detected_d_sae)
            logger.info("SparseFeatureCache built from restored samples (%d samples, d_sae=%d)",
                        len(all_train_samples), detected_d_sae)
        else:
            logger.warning(
                "Only %d/%d samples have sparse features; distribution_drift unavailable.",
                sparse_ready, len(all_train_samples),
            )

    # ---- Step 5: SAE ingest (only when not skip_sae) ----
    if args.sae_path and not skip_sae:
        from recipe_sandbox.scoring.sparse_features import (
            ingest_sparse_features, SparseFeatureCache,
            has_sparse_features, compute_mean_activation,
        )
        if detected_d_sae is None:
            raise ValueError(f"d_sae is unavailable for SAE ingest: {args.sae_path}")
        d_sae = detected_d_sae

        # 5a. Target vectors for MONA (per-benchmark only)
        # IMPORTANT: these files are prototype-only sources and must not be
        # the held-out benchmark eval files used by LoRAEvaluator.
        from recipe_sandbox.scoring.mona import _sample_canonical_samples, DEFAULT_TARGET_VECTOR_MAX_SAMPLES
        eval_target_vectors_per_bm: Dict[str, Any] = {}  # stem → target_vector
        for eidx, eval_file in enumerate(eval_canonical_files):
            partial_target = None
            eval_npz = sae_cache_dir / f"eval_{eidx}_{Path(eval_file).stem}.npz"
            eval_samples = list(read_jsonl(eval_file))
            
            # IMPORTANT: Subsample eval set for target vector identically to MonaScorer
            # Scale up for large eval sets (>10K items → double the default)
            _max_tv_samples = DEFAULT_TARGET_VECTOR_MAX_SAMPLES
            if len(eval_samples) > 10000:
                _max_tv_samples = DEFAULT_TARGET_VECTOR_MAX_SAMPLES * 2
            eval_samples = _sample_canonical_samples(
                eval_samples,
                max_samples=_max_tv_samples,
                sample_seed=42,
                sample_label=Path(eval_file).stem,
            )
            
            if eval_npz.exists():
                logger.info("[Eval %d/%d] SAE cache exists, loading: %s", eidx + 1, len(eval_canonical_files), eval_npz)
                data = np.load(eval_npz)
                partial_target = data["target_vector"]
            else:
                cached = sum(1 for s in eval_samples if has_sparse_features(s))
                if cached < len(eval_samples) * 0.9:
                    logger.info("[Eval %d/%d] SAE ingest on %d samples from %s ...", eidx + 1, len(eval_canonical_files), len(eval_samples), Path(eval_file).name)
                    ingest_sparse_features(
                        eval_samples, model_path=args.base_model, sae_path=args.sae_path,
                        top_k=args.sae_top_k, batch_size=args.sae_batch_size,
                        max_length=args.sae_max_length, device=args.sae_device, show_progress=True,
                        max_workers=1, compute_cpu_heuristics=False, compute_ifd=False, cpu_max_workers=args.cpu_max_workers,
                    )
                    
                    partial_target = compute_mean_activation(eval_samples, d_sae)
                    np.savez_compressed(eval_npz, target_vector=partial_target)
                    logger.info("Saved single dense target_vector to %s", eval_npz)
                    
                    write_jsonl(eval_file, iter_samples_without_numeric_extras(eval_samples))
                    logger.info(
                        "[Eval %d/%d] SAE ingest done, text-only canonical file updated.",
                        eidx + 1,
                        len(eval_canonical_files),
                    )
                    try:
                        from recipe_sandbox.evaluation.gpu_cleanup import release_all_gpu_memory
                        release_all_gpu_memory(wait_seconds=2.0)
                    except Exception:
                        pass
            
                if partial_target is None:
                    partial_target = compute_mean_activation(eval_samples, d_sae)
            
            eval_stem = target_benchmark_by_file.get(
                str(eval_file),
                _normalize_benchmark_name(Path(eval_file).stem),
            )
            eval_target_vectors_per_bm[eval_stem] = partial_target
            del eval_samples

        if eval_target_vectors_per_bm:
            logger.info(
                "Prepared %d per-benchmark target vector(s): %s",
                len(eval_target_vectors_per_bm),
                sorted(eval_target_vectors_per_bm.keys()),
            )
        # 5b. Train SAE ingest
        cached = sum(1 for s in all_train_samples if has_sparse_features(s))
        if cached < len(all_train_samples) * 0.9:
            logger.info("SAE ingest on %d train samples (top_k=%d) ...", len(all_train_samples), args.sae_top_k)
            ingest_sparse_features(
                all_train_samples, model_path=args.base_model, sae_path=args.sae_path,
                top_k=args.sae_top_k, batch_size=args.sae_batch_size,
                max_length=args.sae_max_length, device=args.sae_device, show_progress=True,
                cpu_max_workers=args.cpu_max_workers,
            )
            try:
                from recipe_sandbox.evaluation.gpu_cleanup import release_all_gpu_memory
                release_all_gpu_memory(wait_seconds=2.0)
            except Exception:
                pass
        
        # 5c. Calculate MONA score — per-benchmark only
        if eval_target_vectors_per_bm:
            from recipe_sandbox.scoring.sparse_features import compute_mona_score_arrays

            logger.info(
                "Computing per-benchmark MONA jaccard scores for %d samples against %d target vector(s) via vectorized batches...",
                len(all_train_samples),
                len(eval_target_vectors_per_bm),
            )
            score_arrays = compute_mona_score_arrays(all_train_samples, eval_target_vectors_per_bm)

            logger.info(
                "Writing per-benchmark MONA scores for %d benchmark(s)...",
                len(eval_target_vectors_per_bm),
            )
            for bm_stem in sorted(eval_target_vectors_per_bm.keys()):
                bm_scores = score_arrays[bm_stem]
                for sample, score in zip(all_train_samples, bm_scores):
                    sample.metadata.extra.setdefault("mona_scores", {})[bm_stem] = float(score)
                logger.info("  %s: done (mean=%.4f)", bm_stem, float(bm_scores.mean()) if bm_scores.size else 0.0)
            logger.info("Per-benchmark MONA scores complete.")
            
        # 5d. Save train_combined.npz
        train_npz = sae_cache_dir / "train_combined.npz"
        indices_list = []
        values_list = []
        sample_ids_list = []
        ngram_entropy_list = []
        action_object_list = []
        ifd_list = []
        varentropy_list = []
        per_bm_scores: Dict[str, list] = {}
        for bm_stem in sorted(eval_target_vectors_per_bm.keys()):
            per_bm_scores[bm_stem] = []

        for s in all_train_samples:
            topk = s.metadata.extra.get("sae_topk")
            if topk:
                indices_list.append(np.array(topk["indices"], dtype=np.int32))
                values_list.append(np.array(topk["values"], dtype=np.float32))
            else:
                indices_list.append(np.array([], dtype=np.int32))
                values_list.append(np.array([], dtype=np.float32))
            sample_ids_list.append(s.sample_id)
            ngram_entropy_list.append(s.metadata.extra.get("ngram_entropy", {}).get("score", float('nan')))
            action_object_list.append(s.metadata.extra.get("action_object", {}).get("score", float('nan')))
            ifd_list.append(s.metadata.extra.get("ifd", {}).get("score", float('nan')))
            varentropy_list.append(s.metadata.extra.get("varentropy", {}).get("score", float('nan')))
            ms = s.metadata.extra.get("mona_scores", {})
            for bm_stem in per_bm_scores:
                per_bm_scores[bm_stem].append(ms.get(bm_stem, float('nan')))

        save_kwargs = dict(
            indices=np.array(indices_list, dtype=object),
            values=np.array(values_list, dtype=object),
            sample_ids=np.array(sample_ids_list, dtype=str),
            ngram_entropy=np.array(ngram_entropy_list, dtype=np.float32),
            action_object=np.array(action_object_list, dtype=np.float32),
            ifd=np.array(ifd_list, dtype=np.float32),
            varentropy=np.array(varentropy_list, dtype=np.float32),
        )
        for bm_stem, bm_vals in per_bm_scores.items():
            save_kwargs[f"mona_bm_{bm_stem}"] = np.array(bm_vals, dtype=np.float32)

        pending_train_cache_save = _CACHE_SAVE_EXECUTOR.submit(
            _save_npz_cache,
            str(train_npz),
            save_kwargs,
        )
        logger.info(
            "Scheduled async cache save for train vectors, per-benchmark MONA scores, heuristics, and sample_ids → %s",
            train_npz,
        )

        # Build SparseFeatureCache from freshly ingested samples
        sparse_cache = SparseFeatureCache.from_samples(all_train_samples, d_sae=d_sae)
        logger.info("SparseFeatureCache built from ingested samples (%d samples, d_sae=%d)",
                    len(all_train_samples), d_sae)
        
    elif skip_sae:
        logger.info("SAE ingest skipped because the cache is complete.")
    
    # ---- Step 6: Write pool (text-only) ----
    pool_train_dir.mkdir(parents=True, exist_ok=True)
    pool_train_path = str(pool_train_dir / "pool.jsonl")
    for stale_jsonl in pool_train_dir.glob("*.jsonl"):
        if stale_jsonl.name != "pool.jsonl":
            stale_jsonl.unlink()
            logger.info("Removed stale canonical train shard → %s", stale_jsonl)

    write_jsonl(pool_train_path, iter_samples_without_numeric_extras(all_train_samples))
    logger.info("Working pool: %d train samples (text-only) → %s", len(all_train_samples), pool_train_path)
    
    # Also copy eval into the working canonical tree for MONA operator access
    pool_eval_dir = out_dir / "canonical" / "eval"
    pool_eval_dir.mkdir(parents=True, exist_ok=True)
    eval_canonical_files = _stage_eval_canonical_files(
        pool_eval_dir,
        eval_canonical_files,
        target_benchmark_by_file,
    )
    
    # 4. Executor and Registry
    registry = build_default_operator_registry()
    extension_modules = parse_extension_modules(args.extension_modules)
    extension_hooks = load_extensions(extension_modules, registry=registry)
    if extension_modules:
        logger.info(
            "Loaded %d extension module(s), %d recipe hook(s), %d registered operator(s)",
            len(extension_modules),
            len(extension_hooks),
            len(registry.names()),
        )

    if pending_train_cache_save is not None:
        logger.info(
            "Continuing without waiting for train_combined.npz save; "
            "search uses in-memory samples and sparse cache."
        )

    executor = RecipeExecutor(task_config, manager, registry,
                              sparse_cache=sparse_cache,
                              cached_train_samples=all_train_samples,
                              hooks=extension_hooks)
    
    # 5. LLM Agents (Action, Feedback, Selection) + ThinkingLogger
    catalog_path = args.operator_catalog
    thinking_model = args.thinking_model or args.llm_model
    
    # Initialize ThinkingLogger for recording LLM reasoning traces
    from recipe_sandbox.agents.thinking_logger import ThinkingLogger
    thinking_logger = ThinkingLogger(str(out_dir))
    
    allowed_search_operators = set(resolve_operator_space(
        registry.names(),
    ))
    pool_size = len(all_train_samples)
    logger.info("Pool size for MCTS/generation: %d samples", pool_size)

    # Use thinking model for all LLM agents when available
    llm_conf = LLMConfig(api_key=args.llm_api_key, base_url=args.llm_base_url, model=thinking_model)
    agent = ActionLLMGenerator(
        llm_config=llm_conf, catalog_path=catalog_path,
        registered_operators=allowed_search_operators,
        thinking_logger=thinking_logger,
    )
    feedback = FeedbackLLM(llm_conf, call_interval=3, thinking_logger=thinking_logger)
    selection = SelectionLLM(llm_conf, temperature=0.6, thinking_logger=thinking_logger)
    logger.info(
        "LLM agents initialized: model=%s (thinking=%s), feedback_llm=%s, selection_llm=%s, thinking_log=%s",
        thinking_model,
        bool(args.thinking_model),
        "enabled",
        "enabled",
        thinking_logger.path,
    )
    
    # 6. Physical Evaluator (full fine-tuning)
    # Parse eval_tasks if provided
    eval_tasks_dict = None
    if args.eval_tasks:
        eval_tasks_dict = {}
        for task_spec in args.eval_tasks.split(","):
            task_spec = task_spec.strip()
            if ":" in task_spec:
                name, path = task_spec.split(":", 1)
                eval_tasks_dict[name.strip()] = path.strip()
        logger.info("Multi-benchmark eval tasks: %s", eval_tasks_dict)
    
    logger.info("Initializing full fine-tuning evaluator for the hardware loop...")

    ds_config = args.deepspeed
    if not ds_config:
        ds_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "src", "recipe_sandbox", "evaluation", "ds_zero2.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ds_zero2.json"),
        ]
        for candidate in ds_candidates:
            if os.path.isfile(candidate):
                ds_config = os.path.abspath(candidate)
                break
        if ds_config:
            logger.info("Full fine-tuning: using DeepSpeed config %s", ds_config)
        else:
            logger.warning("Full fine-tuning: no DeepSpeed config found. Training may OOM on multi-GPU.")

    model_lower = args.base_model.lower()
    if "qwen" in model_lower:
        template = "qwen"
    elif "llama-3" in model_lower or "llama3" in model_lower:
        template = "llama3"
    elif "llama-2" in model_lower or "llama2" in model_lower:
        template = "llama2"
    elif "mistral" in model_lower:
        template = "mistral"
    elif "gemma" in model_lower:
        template = "gemma"
    else:
        template = "default"
    logger.info("Auto-detected chat template: %s (from model path %s)", template, args.base_model)

    train_cfg = LoRATrainConfig(
        base_model_path=args.base_model,
        template=template,
        finetuning_type="full",
        deepspeed=ds_config or "",
        num_train_epochs=args.num_epochs,
        pilot_num_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
    )
    gpu_ids = [int(x.strip()) for x in args.train_gpu_ids.split(",")] if args.train_gpu_ids else None
    train_cfg.train_gpu_ids = gpu_ids

    # The pipeline script runs training then eval sequentially, so the same
    # devices can be safely reused for evaluation.
    detected_ids, detected_device_type = _detect_all_accelerator_ids()
    eval_gpu_ids = gpu_ids or detected_ids
    logger.info("Evaluator devices: type=%s ids=%s", detected_device_type, eval_gpu_ids)

    eval_cfg = LoRAEvalConfig(
        eval_data_path=args.eval_data,
        eval_tasks=eval_tasks_dict,
        batch_size=args.eval_batch_size,
        mode=args.vllm_mode,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    evaluator = LoRAEvaluator(
        str(out_dir),
        train_cfg,
        eval_cfg,
        auto_run=True,
        gpu_ids=eval_gpu_ids,
        cleanup_checkpoints=args.cleanup_checkpoints,
    )
    
    # 7. MCTS Engine Execution
    log_path = str(out_dir / "search_log.jsonl")

    mcts_engine = MCTSSearchLoop(
        manager=manager,
        recipe_executor=executor,
        action_generator=agent,
        evaluator=evaluator,
        catalog_path=catalog_path,
        budget_gpu_hours=args.budget,
        search_log_path=log_path,
        k_exploration=1.5,
        n_lhs_seeds=args.n_lhs_seeds,
        pool_size=pool_size,
        feedback_llm=feedback,
        selection_llm=selection,
        thinking_logger=thinking_logger,
        stagnation_patience=args.stagnation_patience,
    )
    
    if args.resume and os.path.exists(log_path):
        logger.warning(f"Resuming requested. Loading breakpoint from {log_path}")
        mcts_engine.resume_from_log(log_path)
    elif not args.resume and os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        logger.info("Auto-detected existing search_log.jsonl — auto-resuming.")
        mcts_engine.resume_from_log(log_path)
        
    best_candidate = mcts_engine.search()
    _await_pending_cache_save(pending_train_cache_save, train_npz_path)
    
    logger.info("=" * 70)
    logger.info("  MCTS SEARCH COMPLETE")
    logger.info("  Ultimate Pareto Optimal Recipe: %s", best_candidate.recipe.recipe_name)
    logger.info("  Score: %.2f | Utility: %.2f", best_candidate.score, best_candidate.utility)
    logger.info("  Check logs at %s", out_dir)
    logger.info("=" * 70)

if __name__ == "__main__":
    run_e2e()
