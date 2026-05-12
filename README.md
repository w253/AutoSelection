# Recipe Sandbox

This directory contains the final AutoSelection MCTS pipeline and evaluation
entrypoints.

## 1. Directory Layout

```text
recipe_sandbox/
  data/
    train3/merged_data.jsonl              # default training pool
    target_vector_samples/*.jsonl         # target-vector data for MONA/SAE scoring
    eval/*.jsonl                          # in-domain and OOD evaluation sets
  examples/recipes/operator_catalog.yaml  # operator search space
  runs/
    run_mcts_e2e.sh                       # main MCTS search entrypoint
    run_mcts_e2e_engine.py                # Python engine used by the shell entrypoint
    run_ood_eval.sh                       # OOD evaluation entrypoint
    run_multi_ckpt_eval.py                # multi-checkpoint evaluation helper
  src/recipe_sandbox/                     # package source code
```


## 2. Environment Setup

Use Python 3.10 or newer. Install the accelerator-specific `torch`, `vllm`,
and `LLaMA-Factory` builds that match your CUDA/NPU cluster.

Example:

```bash
conda create -n autosel python=3.10 -y
conda activate autosel
pip install -U pip

pip install numpy scipy scikit-learn pyyaml tqdm pandas pyarrow datasets openai
pip install torch transformers accelerate peft deepspeed vllm llamafactory
```

For Ascend/NPU environments, also install the matching `torch-npu` package and
set `ASCEND_RT_VISIBLE_DEVICES` as needed. For CUDA environments, set
`CUDA_VISIBLE_DEVICES` as usual.

Make sure these commands are available before running the full pipeline:

```bash
python -c "import sklearn, torch, transformers, vllm"
llamafactory-cli --help
```

## 3. Model and LLM Configuration

By default the scripts expect:

```text
recipe_sandbox/models/base_model
recipe_sandbox/models/sae/layers.27
```

You can override them with environment variables:

```bash
export BASE_MODEL=/path/to/base_model
export SAE_PATH=/path/to/sae/layers.27
```

The MCTS agents use an OpenAI-compatible LLM endpoint:

```bash
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENAI_API_KEY=your_api_key
export LLM_MODEL=your_model_name
export THINKING_MODEL=${LLM_MODEL}
```

`THINKING_MODEL` is used by Action, Feedback, and Selection LLM agents. If it is
not set, it defaults to `LLM_MODEL`.

## 4. Data Setup

Default paths are relative to `recipe_sandbox/`:

```text
data/train3/merged_data.jsonl
data/target_vector_samples/gpqa_ext_98.jsonl
data/target_vector_samples/gsm8k_train_100.jsonl
data/target_vector_samples/bbh_few_shot.jsonl
data/target_vector_samples/mmlu_val.jsonl
data/eval/gpqa_main.jsonl
data/eval/gsm8k_test.jsonl
data/eval/bbh_test.jsonl
data/eval/mmlu_test.jsonl
data/eval/GraphWiz_test.jsonl
data/eval/NLgraph_test.jsonl
```

Training data should be JSONL in canonical chat format. Each line should contain
at least a `messages` list with `{role, content}` objects. Optional fields such
as `sample_id`, `source_name`, `target`, `metadata`, and `tags` are supported.
See `src/recipe_sandbox/schema/canonical_schema.yaml` for the full schema.

To use another training pool:

```bash
RAW_TRAIN_DATA=/path/to/train.jsonl bash runs/run_mcts_e2e.sh
```

To use another data root:

```bash
DATA_DIR=/path/to/data bash runs/run_mcts_e2e.sh
```

## 5. Run Main MCTS Search

```bash
cd /path/to/AutoSelection/recipe_sandbox

export BASE_MODEL=/path/to/base_model
export SAE_PATH=/path/to/sae/layers.27
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENAI_API_KEY=your_api_key
export LLM_MODEL=your_model_name

bash runs/run_mcts_e2e.sh
```

Common overrides:

```bash
BUDGET=20.0 \
N_LHS_SEEDS=3 \
NUM_EPOCHS=3.0 \
STAGNATION_PATIENCE=4 \
TENSOR_PARALLEL_SIZE=1 \
bash runs/run_mcts_e2e.sh
```

If using a custom DeepSpeed config:

```bash
DEEPSPEED_CONFIG=/path/to/ds_zero2.json bash runs/run_mcts_e2e.sh
```

Resume a previous run:

```bash
OUTPUT_DIR=runs/e2e_mcts_YYYYMMDD_HHMMSS RESUME=1 bash runs/run_mcts_e2e.sh
```

Reuse SAE/canonical cache from a previous run:

```bash
SAE_CACHE_FROM=runs/e2e_mcts_YYYYMMDD_HHMMSS bash runs/run_mcts_e2e.sh
```

Main outputs are written under `runs/e2e_mcts_*` or the `OUTPUT_DIR` you set:

```text
engine.log
experiment_*.log
search_log.jsonl
search_tree.json
thinking_logs/
recipes/
canonical/
sae_caches/
```

## 6. Run OOD Evaluation

OOD evaluation uses:

```text
data/eval/GraphWiz_test.jsonl
data/eval/NLgraph_test.jsonl
```

Evaluate a full model:

```bash
cd /path/to/AutoSelection/recipe_sandbox
MODEL_PATH=/path/to/full_model bash runs/run_ood_eval.sh
```

Evaluate a LoRA adapter with a base model:

```bash
MODEL_PATH=/path/to/base_model \
LORA_PATH=/path/to/adapter \
bash runs/run_ood_eval.sh
```

Common OOD overrides:

```bash
BATCH_SIZE=64 \
MAX_TOKENS=4096 \
TENSOR_PARALLEL_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.9 \
OUTPUT_DIR=runs/ood_eval_custom \
MODEL_PATH=/path/to/model \
bash runs/run_ood_eval.sh
```

## 7. Evaluate Multiple Checkpoints

Evaluate full checkpoints:

```bash
python runs/run_multi_ckpt_eval.py \
  --checkpoints /path/to/ckpt1 /path/to/ckpt2 \
  --output_dir runs/multi_ckpt_eval
```

Evaluate LoRA adapters against one base model:

```bash
python runs/run_multi_ckpt_eval.py \
  --base_model_path /path/to/base_model \
  --checkpoints /path/to/adapter1 /path/to/adapter2 \
  --tasks gpqa,gsm8k,bbh,mmlu,graph,graph_yesno \
  --output_dir runs/multi_ckpt_eval
```

For sharded evaluation:

```bash
python runs/run_multi_ckpt_eval.py \
  --checkpoints /path/to/ckpt \
  --tasks gpqa,gsm8k,bbh,mmlu,graph,graph_yesno \
  --num_shards 4 \
  --device_ids 0,1,2,3 \
  --output_dir runs/multi_ckpt_eval
```

## 8. Quick Checks

```bash
bash -n runs/run_mcts_e2e.sh
bash -n runs/run_ood_eval.sh
python -m compileall -q runs src
```

If `run_mcts_e2e_engine.py --help` fails with `ModuleNotFoundError:
No module named 'sklearn'`, install `scikit-learn` in the active environment.

