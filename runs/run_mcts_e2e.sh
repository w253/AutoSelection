#!/bin/bash
# ================================================================
#  Surrogate MCTS AutoSelection v2 — Fully Automatic E2E Engine
#
#  Pipeline: 
#    1. (Optional) Raw data ingest via LLM AgentMapper 
#    2. SAE feature extraction
#    3. LLM-guided MCTS Data Selection Loop
#    4. Evaluated natively via SFT -> VLLM (GPQA, GSM8K, BBH, MMLU)
#
#  Usage:
#    cd /path/to/AutoSelection/recipe_sandbox
#    bash runs/run_mcts_e2e.sh
#
#  Resume an aborted physical run directly from JSONL Search Log:
#    OUTPUT_DIR=runs/e2e_mcts_XXX RESUME=1 bash runs/run_mcts_e2e.sh
#
#  Provide raw ingestion pipeline task_config json:
#    TASK_CONFIG=examples/mona_task.json bash runs/run_mcts_e2e.sh
# ================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Environment Configuration (override via env vars) ----
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models}"
BASE_MODEL="${BASE_MODEL:-${MODEL_DIR}/base_model}"
SAE_PATH="${SAE_PATH:-${MODEL_DIR}/sae/layers.27}"
TARGET_VECTOR_DATA="${TARGET_VECTOR_DATA:-${DATA_DIR}/target_vector_samples}"

# Multi-benchmark eval tasks (format: benchmark:path,benchmark:path)
EVAL_TASKS="${EVAL_TASKS:-gpqa:${DATA_DIR}/eval/gpqa_main.jsonl,gsm8k:${DATA_DIR}/eval/gsm8k_test.jsonl,bbh:${DATA_DIR}/eval/bbh_test.jsonl,mmlu:${DATA_DIR}/eval/mmlu_test.jsonl}"

N_LHS_SEEDS="${N_LHS_SEEDS:-3}"
BUDGET="${BUDGET:-20.0}"
NUM_EPOCHS="${NUM_EPOCHS:-3.0}"
SAE_CACHE_FROM="${SAE_CACHE_FROM:-}"
CPU_MAX_WORKERS="${CPU_MAX_WORKERS:-16}"
SAE_TOP_K="${SAE_TOP_K:-192}"
CLEANUP_CHECKPOINTS="${CLEANUP_CHECKPOINTS:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME="${RESUME:-0}"
TASK_CONFIG="${TASK_CONFIG:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OPERATOR_CATALOG="${OPERATOR_CATALOG:-examples/recipes/operator_catalog.yaml}"
EXTENSION_MODULES="${EXTENSION_MODULES:-${RECIPE_SANDBOX_EXTENSIONS:-}}"

STAGNATION_PATIENCE="${STAGNATION_PATIENCE:-4}"    # steps before trajectory restart

# Required LLM Backend for MCTS Proposer
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
# Thinking/reasoning model for strategic decisions (Feedback, Action, Selection LLMs)
export THINKING_MODEL="${THINKING_MODEL:-${LLM_MODEL}}"
export RECIPE_SANDBOX_EXTENSIONS="${EXTENSION_MODULES}"

# --- Data Ingestion (Raw Data mapping -> Canonical) ---
DEFAULT_TRAIN_DATA="${DATA_DIR}/train3/merged_data.jsonl"
if [ ! -f "${DEFAULT_TRAIN_DATA}" ] && [ -f "${DEFAULT_TRAIN_DATA}.bak" ]; then
    DEFAULT_TRAIN_DATA="${DEFAULT_TRAIN_DATA}.bak"
fi
RAW_TRAIN_DATA="${RAW_TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"

echo "================================================================"
echo "  MCTS Active Search Engine v2 — Fully Autonomous E2E"
echo "  Date: $(date)"
echo "  Base model: ${BASE_MODEL}"
echo "  SAE: ${SAE_PATH}"
echo "  Target Vector Source: ${TARGET_VECTOR_DATA}"
echo "  Agent: ${LLM_MODEL} @ ${OPENAI_BASE_URL}"
echo "  Thinking Model: ${THINKING_MODEL}"
echo "  Budget: ${BUDGET}h | LHS Seeds: ${N_LHS_SEEDS}"
echo "  Operator Catalog: ${OPERATOR_CATALOG}"
echo "  Eval TP: ${TENSOR_PARALLEL_SIZE}"
if [ -n "${EXTENSION_MODULES}" ]; then
    echo "  Extensions: ${EXTENSION_MODULES}"
fi
if [ -n "${TASK_CONFIG}" ]; then
    echo "  Ingestion Source: ${TASK_CONFIG} (via PipelineOrchestrator)"
fi
echo "================================================================"
echo ""

EXTRA_ARGS=()
if [ -n "${OUTPUT_DIR}" ]; then
    EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [ "${RESUME}" = "1" ]; then
    EXTRA_ARGS+=(--resume)
    echo ">>> RESUME ENABLED: MCTS Will restore GP weights and unvisited node pool from ${OUTPUT_DIR}/search_log.jsonl"
fi
if [ -n "${TASK_CONFIG}" ]; then
    EXTRA_ARGS+=(--task_config "${TASK_CONFIG}")
fi
if [ -n "${RAW_TRAIN_DATA}" ]; then
    EXTRA_ARGS+=(--data_path "${RAW_TRAIN_DATA}")
fi
if [ -n "${TARGET_VECTOR_DATA}" ]; then
    EXTRA_ARGS+=(--target_vector_data "${TARGET_VECTOR_DATA}")
fi
if [ -n "${CPU_MAX_WORKERS}" ]; then
    EXTRA_ARGS+=(--cpu_max_workers "${CPU_MAX_WORKERS}")
fi
if [ -n "${SAE_CACHE_FROM}" ]; then
    EXTRA_ARGS+=(--sae_cache_from "${SAE_CACHE_FROM}")
fi
if [ -n "${DEEPSPEED_CONFIG}" ]; then
    EXTRA_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi
if [ -n "${EXTENSION_MODULES}" ]; then
    EXTRA_ARGS+=(--extension_modules "${EXTENSION_MODULES}")
fi

# Point to Canonical logic  
# Determine log file path for tee
if [ -n "${OUTPUT_DIR}" ]; then
    LOG_DIR="${OUTPUT_DIR}"
else
    LOG_DIR="runs"
fi
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log"

python runs/run_mcts_e2e_engine.py \
    --budget "${BUDGET}" \
    --n_lhs_seeds "${N_LHS_SEEDS}" \
    --base_model "${BASE_MODEL}" \
    --sae_path "${SAE_PATH}" \
    --eval_tasks "${EVAL_TASKS}" \
    --thinking_model "${THINKING_MODEL}" \
    --operator_catalog "${OPERATOR_CATALOG}" \
    --sae_top_k "${SAE_TOP_K}" \
    --sae_batch_size 4 \
    --sae_max_length 2048 \
    --eval_batch_size 64 \
    --num_epochs "${NUM_EPOCHS}" \
    --train_batch_size 4 \
    --gradient_accumulation 8 \
    --cutoff_len 2048 \
    --vllm_mode native \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization 0.85 \
    --cleanup_checkpoints "${CLEANUP_CHECKPOINTS}" \
    --stagnation_patience "${STAGNATION_PATIENCE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Done! The Pareto Front trace and optimal candidate logic should be finalized."
