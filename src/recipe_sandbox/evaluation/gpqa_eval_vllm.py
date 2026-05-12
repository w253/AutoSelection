"""GPQA Evaluation Pipeline with vLLM Backend.

Multiple-choice evaluation for GPQA Extended using vLLM chat API.
Adapted from gsm8k_eval_vllm.py with the key change: uses chat mode
to properly apply template tokens.

Data format (JSONL):
  {"question": "...", "correct_answer": "...", "incorrect_answers": ["...", "...", "..."]}
"""

from __future__ import annotations

import json
import os
import random
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """You are an expert assistant. Answer the following multiple choice question by selecting the correct option (A, B, C, or D).

Instructions:
1. Read the question carefully.
2. Consider each option.
3. Respond with ONLY the letter of the correct answer (A, B, C, or D) on the last line.
4. Format: put your final answer after "Answer:" on the last line.

Example format:
[Your reasoning]
Answer: B"""


def load_gpqa_data(path: str) -> List[Dict[str, Any]]:
    """Load GPQA Extended JSONL and shuffle options."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Build shuffled options
            options = [item["correct_answer"]] + item["incorrect_answers"]
            random.seed(hash(item["question"]) % (2**31))
            random.shuffle(options)
            correct_idx = options.index(item["correct_answer"])
            correct_letter = chr(ord("A") + correct_idx)

            data.append({
                "question": item["question"],
                "options": options,
                "correct_letter": correct_letter,
                "correct_answer": item["correct_answer"],
                "explanation": item.get("explanation", ""),
            })
    return data


def build_mc_prompt(question: str, options: List[str]) -> str:
    """Build multiple-choice prompt string."""
    lines = [f"Question: {question}", ""]
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


def build_chat_messages(question: str, options: List[str]) -> List[Dict[str, str]]:
    """Build chat messages for vLLM chat API."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_mc_prompt(question, options)},
    ]


def extract_answer_letter(text: str) -> str:
    """Extract the answer letter (A/B/C/D) from model output."""
    text = text.strip()

    # Pattern 1: "Answer: X"
    match = re.search(r"Answer:\s*([A-Da-d])", text)
    if match:
        return match.group(1).upper()

    # Pattern 2: Last single letter on its own line
    for line in reversed(text.split("\n")):
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()

    # Pattern 3: "The answer is X"
    match = re.search(r"[Tt]he answer is\s*\(?([A-Da-d])\)?", text)
    if match:
        return match.group(1).upper()

    # Pattern 4: First capital letter A-D at start of response
    match = re.match(r"^\s*\(?([A-Da-d])\)?[\.\)\s]", text)
    if match:
        return match.group(1).upper()

    return ""


def evaluate_gpqa_vllm(
    model_path: str,
    eval_data_path: str,
    output_dir: str,
    *,
    lora_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    temperature: float = 0.1,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> Dict[str, Any]:
    """Run GPQA evaluation with vLLM chat API.

    If lora_path is provided, uses vLLM native LoRA serving
    (enable_lora=True + LoRARequest) — no model merging needed.

    Returns metrics dict with accuracy etc.
    """
    from vllm import LLM, SamplingParams

    # Load data
    print(f"Loading GPQA data from {eval_data_path}...")
    data = load_gpqa_data(eval_data_path)
    if max_samples:
        data = data[:max_samples]
    print(f"Total samples: {len(data)}")

    # Load model (with or without LoRA)
    use_lora = lora_path is not None and os.path.isdir(lora_path)
    print(f"Loading model from {model_path}...")
    if use_lora:
        print(f"  LoRA adapter: {lora_path} (native vLLM serving)")

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=16384,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=use_lora,
        max_lora_rank=64 if use_lora else None,
    )
    if pipeline_parallel_size > 1:
        llm_kwargs["pipeline_parallel_size"] = pipeline_parallel_size
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Build chat messages
    print("Building chat messages...")
    all_messages = [
        build_chat_messages(item["question"], item["options"]) for item in data
    ]

    # Batch inference with vLLM chat
    print(f"Running batch inference ({len(all_messages)} samples, batch={batch_size})...")
    all_outputs = []
    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(all_messages), batch_size), desc="GPQA eval")
    except ImportError:
        iterator = range(0, len(all_messages), batch_size)

    # Prepare LoRA request if using native LoRA
    lora_request = None
    if use_lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name="recipe_adapter",
            lora_int_id=1,
            lora_local_path=lora_path,
        )

    for i in iterator:
        batch = all_messages[i : i + batch_size]
        if lora_request is not None:
            outputs = llm.chat(batch, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.chat(batch, sampling_params)
        all_outputs.extend(outputs)

    # Score
    correct = 0
    results = []
    for idx, (item, output) in enumerate(zip(data, all_outputs)):
        generated = output.outputs[0].text.strip()
        predicted = extract_answer_letter(generated)
        is_correct = predicted == item["correct_letter"]
        if is_correct:
            correct += 1

        results.append({
            "index": idx,
            "question": item["question"][:200],
            "correct_letter": item["correct_letter"],
            "correct_answer": item["correct_answer"],
            "predicted": predicted,
            "correct": is_correct,
            "generated": generated[:500],
        })

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0

    metrics = {
        "benchmark": "gpqa_extended",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "model_path": model_path,
        "temperature": temperature,
        "backend": "vllm",
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "gpqa_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(output_dir, "gpqa_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nGPQA Evaluation Complete!")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results → {output_dir}")
    return metrics


def main():
    default_eval_data = Path(__file__).resolve().parents[3] / "data" / "eval" / "gpqa_main.jsonl"
    parser = argparse.ArgumentParser(description="GPQA evaluation with vLLM")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter for native vLLM LoRA serving")
    parser.add_argument(
        "--eval_data",
        type=str,
        default=str(default_eval_data),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    evaluate_gpqa_vllm(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        lora_path=args.lora_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
