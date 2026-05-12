#!/bin/bash
# Run OOD evaluation for GraphWiz and NLgraph yes/no tasks.
#
# Required:
#   MODEL_PATH=/path/to/base_or_full_model bash runs/run_ood_eval.sh
#
# Optional LoRA native serving:
#   MODEL_PATH=/path/to/base_model LORA_PATH=/path/to/adapter bash runs/run_ood_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
EVAL_DIR="${EVAL_DIR:-${DATA_DIR}/eval}"
GRAPH_DATA="${GRAPH_DATA:-${EVAL_DIR}/GraphWiz_test.jsonl}"
GRAPH_YESNO_DATA="${GRAPH_YESNO_DATA:-${EVAL_DIR}/NLgraph_test.jsonl}"

MODEL_PATH="${MODEL_PATH:-}"
LORA_PATH="${LORA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/ood_eval_$(date +%Y%m%d_%H%M%S)}"

BATCH_SIZE="${BATCH_SIZE:-64}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: set MODEL_PATH=/path/to/base_or_full_model before running OOD eval." >&2
    exit 2
fi

if [ ! -f "${GRAPH_DATA}" ]; then
    echo "ERROR: GraphWiz file not found: ${GRAPH_DATA}" >&2
    exit 2
fi

if [ ! -f "${GRAPH_YESNO_DATA}" ]; then
    echo "ERROR: NLgraph yes/no file not found: ${GRAPH_YESNO_DATA}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"

EVAL_TASKS="graph:${GRAPH_DATA},graph_yesno:${GRAPH_YESNO_DATA}"
LORA_ARGS=()
if [ -n "${LORA_PATH}" ]; then
    LORA_ARGS=(--lora_path "${LORA_PATH}")
fi

python src/recipe_sandbox/evaluation/unified_eval_vllm.py \
    --model_path "${MODEL_PATH}" \
    "${LORA_ARGS[@]}" \
    --eval_tasks "${EVAL_TASKS}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --max_tokens "${MAX_TOKENS}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --pipeline_parallel_size "${PIPELINE_PARALLEL_SIZE}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --merged

echo "OOD evaluation outputs saved to ${OUTPUT_DIR}"
