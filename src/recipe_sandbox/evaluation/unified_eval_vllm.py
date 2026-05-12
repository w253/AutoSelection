"""Unified Evaluation Pipeline with vLLM Backend.

Supports GPQA, GSM8K, BBH, MMLU, MBPP, Math-500, GraphWiz
(excluding topology), and an additional graph yes/no benchmark
with a single vLLM model initialization. Load the model once, evaluate
each benchmark sequentially, output per-task scores and an aggregated metric.

Usage:
    python unified_eval_vllm.py \
        --model_path /path/to/model \
        --eval_tasks gpqa:/path/gpqa.jsonl,gsm8k:/path/gsm8k.jsonl,bbh:/path/bbh_test.jsonl \
        --output_dir /path/to/output

Data formats (JSONL):
  GPQA:  {"question": "...", "correct_answer": "...", "incorrect_answers": ["...", "...", "..."]}
  GSM8K: {"question": "...", "answer": "text with #### <number>"}
  BBH:   {"input": "...", "target": "...", "task": "...", "split": "test"}
  MMLU:  {"question": "...", "subject": "...", "choices": [...], "answer": 1.0, "split": "test"}
  MBPP:  {"text": "...", "code": "...", "task_id": N, "test_list": [...], ...}
  Graph: {"input_prompt": "...", "answer": "...", "task": "..."}  # topology excluded
  Graph yes/no: {"question": "...", "answer": "...", "task": "...", ...}
"""

from __future__ import annotations

import json
import os
import random
import re
import argparse
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import multiprocessing
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# GPQA helpers (adapted from gpqa_eval_vllm.py)
# ---------------------------------------------------------------------------

GPQA_SYSTEM_PROMPT = """You are an expert assistant. Answer the following multiple choice question by selecting the correct option (A, B, C, or D).

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
            data.append(prepare_gpqa_item(item))
    return data


def prepare_gpqa_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a single GPQA record (shuffle options, compute correct letter)."""
    options = [item["correct_answer"]] + item["incorrect_answers"]
    random.seed(hash(item["question"]) % (2**31))
    random.shuffle(options)
    correct_idx = options.index(item["correct_answer"])
    correct_letter = chr(ord("A") + correct_idx)
    return {
        "question": item["question"],
        "options": options,
        "correct_letter": correct_letter,
        "correct_answer": item["correct_answer"],
        "explanation": item.get("explanation", ""),
    }


def build_mc_prompt(question: str, options: List[str]) -> str:
    """Build multiple-choice prompt string."""
    lines = [f"Question: {question}", ""]
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


def build_gpqa_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for a single GPQA item with 3-shot few-shot examples."""
    return [
        {"role": "system", "content": GPQA_SYSTEM_PROMPT},
        # Few-shot example 1 (Biology)
        {"role": "user", "content": "Question: Which of the following is NOT a function of the cell membrane?\n\nA. Selective permeability\nB. Protein synthesis\nC. Cell signaling\nD. Cell adhesion"},
        {"role": "assistant", "content": "The cell membrane has multiple functions including selective permeability (controlling what enters/exits), cell signaling (receiving signals), and cell adhesion (connecting to other cells). Protein synthesis occurs at ribosomes, not at the cell membrane.\nAnswer: B"},
        # Few-shot example 2 (Physics)
        {"role": "user", "content": "Question: What is the SI unit of electrical resistance?\n\nA. Volt\nB. Ampere\nC. Ohm\nD. Watt"},
        {"role": "assistant", "content": "Electrical resistance is measured in Ohms (Ω), named after Georg Ohm. Volts measure potential difference, Amperes measure current, and Watts measure power.\nAnswer: C"},
        # Few-shot example 3 (Chemistry)
        {"role": "user", "content": "Question: Which element has the highest electronegativity?\n\nA. Oxygen\nB. Nitrogen\nC. Fluorine\nD. Chlorine"},
        {"role": "assistant", "content": "Fluorine has the highest electronegativity of all elements (3.98 on the Pauling scale). It is the most electronegative element because it has a small atomic radius and high effective nuclear charge.\nAnswer: C"},
        # Actual question
        {"role": "user", "content": build_mc_prompt(item["question"], item["options"])},
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


def score_gpqa(data: List[Dict[str, Any]], outputs: list) -> tuple:
    """Score GPQA outputs. Returns (results_list, metrics_dict)."""
    correct = 0
    results = []
    for idx, (item, output) in enumerate(zip(data, outputs)):
        generated = output.outputs[0].text.strip()
        predicted = extract_answer_letter(generated)
        is_correct = predicted == item["correct_letter"]
        if is_correct:
            correct += 1

        results.append({
            "index": idx,
            "question": item["question"],
            "correct_letter": item["correct_letter"],
            "correct_answer": item["correct_answer"],
            "predicted": predicted,
            "correct": is_correct,
            "generated": generated,
        })

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "benchmark": "gpqa",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    return results, metrics


# ---------------------------------------------------------------------------
# GSM8K helpers
# ---------------------------------------------------------------------------

GSM8K_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following math problem step by step. "
    "Show your work, then provide the final answer after '#### ' on the last line."
)


def load_gsm8k_data(path: str) -> List[Dict[str, Any]]:
    """Load GSM8K JSONL data."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(prepare_gsm8k_item(item))
    return data


def prepare_gsm8k_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a single GSM8K record (extract ground truth answer)."""
    ground_truth = extract_gsm8k_answer(item["answer"])
    return {
        "question": item["question"],
        "raw_answer": item["answer"],
        "ground_truth": ground_truth,
    }


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer from GSM8K format.

    Looks for '####' followed by a number in the text.
    Handles: integers, decimals, commas, negative numbers.
    Returns None if no valid answer found.
    """
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None

    # Fallback: try last occurrence of #### with any number-like content
    matches = re.findall(r"####\s*(-?[\d,]+\.?\d*)", text)
    if matches:
        num_str = matches[-1].replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass

    return None


def build_gsm8k_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for a single GSM8K item with 3-shot few-shot examples."""
    return [
        {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
        # Few-shot example 1
        {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?"},
        {"role": "assistant", "content": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. That leaves 16 - 7 = 9 eggs to sell. At $2 each, she makes 9 × $2 = $18.\n#### 18"},
        # Few-shot example 2
        {"role": "user", "content": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"},
        {"role": "assistant", "content": "The robe takes 2 bolts of blue fiber. It takes half that much white fiber, so 2 / 2 = 1 bolt of white fiber. In total, it takes 2 + 1 = 3 bolts.\n#### 3"},
        # Few-shot example 3
        {"role": "user", "content": "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"},
        {"role": "assistant", "content": "Josh spent $80,000 + $50,000 = $130,000 total. The repairs increased the value by 150%, so the increase is $80,000 × 1.5 = $120,000. The new value is $80,000 + $120,000 = $200,000. His profit is $200,000 - $130,000 = $70,000.\n#### 70000"},
        # Actual question
        {"role": "user", "content": item["question"]},
    ]


def score_gsm8k(data: List[Dict[str, Any]], outputs: list) -> tuple:
    """Score GSM8K outputs. Returns (results_list, metrics_dict)."""
    correct = 0
    results = []
    for idx, (item, output) in enumerate(zip(data, outputs)):
        generated = output.outputs[0].text.strip()
        predicted = extract_gsm8k_answer(generated)
        ground_truth = item["ground_truth"]

        is_correct = False
        if predicted is not None and ground_truth is not None:
            # Compare as floats with tolerance for floating-point imprecision
            is_correct = abs(predicted - ground_truth) < 1e-6
        if is_correct:
            correct += 1

        results.append({
            "index": idx,
            "question": item["question"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "generated": generated,
        })

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "benchmark": "gsm8k",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    return results, metrics


# ---------------------------------------------------------------------------
# BBH helpers
# ---------------------------------------------------------------------------

BBH_SYSTEM_PROMPT = """You are an expert reasoning assistant. Answer the following question step by step, then provide your final answer.

Instructions:
1. Think through the problem carefully.
2. Show your reasoning.
3. On the last line, write your final answer after "Answer:" exactly matching the expected format.

Example format:
[Your reasoning]
Answer: (B)"""


def load_bbh_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(prepare_bbh_item(item))
    return data


def prepare_bbh_item(item):
    return {
        "input": item["input"],
        "target": item["target"],
        "task": item.get("task", "unknown"),
    }


def build_bbh_messages(item):
    # 3-shot examples
    return [
        {"role": "system", "content": BBH_SYSTEM_PROMPT},
        # Few-shot 1: boolean
        {"role": "user", "content": "not ( True ) and ( True ) is"},
        {"role": "assistant", "content": "not ( True ) evaluates to False. False and ( True ) evaluates to False.\nAnswer: False"},
        # Few-shot 2: disambiguation
        {"role": "user", "content": "In the following sentences, explain the antecedent of the pronoun.\nSentence: The nurse notified the patient that his shift would be ending in an hour.\nOptions:\n(A) The nurse's shift\n(B) The patient's shift\n(C) Ambiguous"},
        {"role": "assistant", "content": "The pronoun 'his' could refer to either the nurse or the patient. However, given the context of notifying about a shift ending, 'his' most likely refers to the nurse's shift.\nAnswer: (A)"},
        # Few-shot 3: simple math/logic
        {"role": "user", "content": "If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.\nFind the sum of all the multiples of 3 or 5 below 20."},
        {"role": "assistant", "content": "Multiples of 3 below 20: 3, 6, 9, 12, 15, 18\nMultiples of 5 below 20: 5, 10, 15\nCombined (no duplicates): 3, 5, 6, 9, 10, 12, 15, 18\nSum = 3+5+6+9+10+12+15+18 = 78\nAnswer: 78"},
        # Actual question
        {"role": "user", "content": item["input"]},
    ]


def _normalize_bbh_answer(text):
    """Normalize BBH answer for comparison."""
    text = text.strip()
    # Remove trailing periods
    text = text.rstrip('.')
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def score_bbh(data, outputs):
    correct = 0
    results = []
    for idx, (item, output) in enumerate(zip(data, outputs)):
        generated = output.outputs[0].text.strip()

        # Extract answer after "Answer:"
        predicted = ""
        match = re.search(r'Answer:\s*(.+?)(?:\n|$)', generated, re.IGNORECASE)
        if match:
            predicted = match.group(1).strip()
        else:
            # Fallback: last line
            lines = [l.strip() for l in generated.split('\n') if l.strip()]
            if lines:
                predicted = lines[-1]

        target = item["target"]
        is_correct = _normalize_bbh_answer(predicted) == _normalize_bbh_answer(target)

        # Also try case-insensitive comparison
        if not is_correct:
            is_correct = _normalize_bbh_answer(predicted).lower() == _normalize_bbh_answer(target).lower()

        if is_correct:
            correct += 1

        results.append({
            "index": idx,
            "input": generated,
            "target": target,
            "predicted": predicted,
            "correct": is_correct,
            "task": item["task"],
            "generated": generated,
        })

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "benchmark": "bbh",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    return results, metrics


# ---------------------------------------------------------------------------
# MMLU helpers
# ---------------------------------------------------------------------------

MMLU_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the following multiple choice question by selecting the correct option (A, B, C, or D).

Instructions:
1. Read the question carefully.
2. Consider each option.
3. Respond with your reasoning, then provide the letter of the correct answer after "Answer:" on the last line.

Example format:
[Your reasoning]
Answer: B"""


def load_mmlu_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(prepare_mmlu_item(item))
    return data


def prepare_mmlu_item(item):
    answer_idx = int(item["answer"])
    correct_letter = chr(ord("A") + answer_idx)
    return {
        "question": item["question"],
        "subject": item.get("subject", "unknown"),
        "choices": item["choices"],
        "correct_letter": correct_letter,
    }


def build_mmlu_messages(item):
    # Reuse the build_mc_prompt function that already exists for GPQA
    mc_text = build_mc_prompt(item["question"], item["choices"])
    return [
        {"role": "system", "content": MMLU_SYSTEM_PROMPT},
        # Few-shot 1
        {"role": "user", "content": "Question: What is the capital of France?\n\nA. London\nB. Berlin\nC. Paris\nD. Madrid"},
        {"role": "assistant", "content": "Paris is the capital and largest city of France.\nAnswer: C"},
        # Few-shot 2
        {"role": "user", "content": "Question: Which planet is known as the Red Planet?\n\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn"},
        {"role": "assistant", "content": "Mars is commonly known as the Red Planet due to its reddish appearance caused by iron oxide on its surface.\nAnswer: B"},
        # Few-shot 3
        {"role": "user", "content": "Question: What is the powerhouse of the cell?\n\nA. Nucleus\nB. Ribosome\nC. Mitochondria\nD. Golgi apparatus"},
        {"role": "assistant", "content": "Mitochondria are often called the 'powerhouse of the cell' because they generate most of the cell's supply of ATP, the main energy currency.\nAnswer: C"},
        # Actual question
        {"role": "user", "content": mc_text},
    ]


def score_mmlu(data, outputs):
    # Reuse extract_answer_letter from GPQA (already handles A-D extraction)
    correct = 0
    results = []
    for idx, (item, output) in enumerate(zip(data, outputs)):
        generated = output.outputs[0].text.strip()
        predicted = extract_answer_letter(generated)
        is_correct = predicted == item["correct_letter"]
        if is_correct:
            correct += 1
        results.append({
            "index": idx,
            "question": item["question"],
            "subject": item["subject"],
            "correct_letter": item["correct_letter"],
            "predicted": predicted,
            "correct": is_correct,
            "generated": generated,
        })
    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "benchmark": "mmlu",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    return results, metrics


# ---------------------------------------------------------------------------
# MBPP helpers
# ---------------------------------------------------------------------------

MBPP_SYSTEM_PROMPT = """You are an expert Python programmer. Write a Python function to solve the given problem.

Instructions:
1. Read the problem description carefully.
2. Write a clean, correct Python function.
3. Only output the Python code, no explanations or markdown formatting.
4. Do not include test cases in your code."""


def load_mbpp_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(prepare_mbpp_item(item))
    return data


def prepare_mbpp_item(item):
    if "messages" in item:
        messages = item.get("messages") or []
        metadata = item.get("metadata") or {}
        extra = metadata.get("extra") or {}

        prompt_text = ""
        reference_code = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content", "")
            if role == "user" and not prompt_text:
                prompt_text = content
            elif role == "assistant" and not reference_code:
                reference_code = content

        target = item.get("target") or {}
        if not reference_code:
            reference_code = target.get("text", "")

        task_id = item.get("task_id")
        if task_id is None:
            sample_id = item.get("sample_id")
            if isinstance(sample_id, str) and sample_id.isdigit():
                task_id = int(sample_id)
            else:
                task_id = sample_id if sample_id is not None else -1

        return {
            "text": prompt_text,
            "code": reference_code,
            "task_id": task_id,
            "test_setup_code": extra.get("test_setup_code", ""),
            "test_list": extra.get("test_list", []),
            "challenge_test_list": extra.get("challenge_test_list", []),
        }

    return {
        "text": item["text"],
        "code": item.get("code", ""),
        "task_id": item.get("task_id", -1),
        "test_setup_code": item.get("test_setup_code", ""),
        "test_list": item.get("test_list", []),
        "challenge_test_list": item.get("challenge_test_list", []),
    }


def build_mbpp_messages(item):
    return [
        {"role": "system", "content": MBPP_SYSTEM_PROMPT},
        # Few-shot 1
        {"role": "user", "content": "Write a python function to find the maximum of two numbers."},
        {"role": "assistant", "content": "def max_of_two(a, b):\n    if a > b:\n        return a\n    return b"},
        # Few-shot 2
        {"role": "user", "content": "Write a function to check if a number is even."},
        {"role": "assistant", "content": "def is_even(n):\n    return n % 2 == 0"},
        # Few-shot 3
        {"role": "user", "content": "Write a function to reverse a string."},
        {"role": "assistant", "content": "def reverse_string(s):\n    return s[::-1]"},
        # Actual question
        {"role": "user", "content": item["text"]},
    ]


def _mbpp_exec_worker(code_str, return_dict):
    """Execute code in an isolated namespace."""
    try:
        exec_globals = {}
        exec(code_str, exec_globals)
        return_dict['success'] = True
        return_dict['error'] = None
    except Exception:
        return_dict['success'] = False
        return_dict['error'] = traceback.format_exc()


def _execute_code_with_timeout(code_str, timeout=5):
    """Run code in a separate process with timeout."""
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=_mbpp_exec_worker, args=(code_str, return_dict))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Timeout: execution exceeded time limit."
    return return_dict.get('success', False), return_dict.get('error', 'Unknown error')


def _extract_python_code(text):
    """Extract Python code from LLM output, handling markdown code blocks."""
    text = text.strip()
    # Try to extract from markdown code block
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no code block, assume the entire response is code
    # Remove any leading/trailing non-code text
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'return ', '#')) or \
           line.startswith((' ', '\t')) or stripped == '' or \
           any(stripped.startswith(kw) for kw in ['if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'except', 'with ', 'raise ']):
            in_code = True
            code_lines.append(line)
        elif in_code and stripped:
            code_lines.append(line)
    if code_lines:
        return '\n'.join(code_lines)
    return text


def score_mbpp(data, outputs):
    correct = 0
    results = []
    for idx, (item, output) in enumerate(zip(data, outputs)):
        generated = output.outputs[0].text.strip()
        extracted_code = _extract_python_code(generated)

        # Assemble code: test_setup + generated code + test assertions
        test_setup = item.get("test_setup_code", "") or ""
        test_assertions = "\n".join(item.get("test_list", []))
        full_code = f"{test_setup}\n{extracted_code}\n\n{test_assertions}"

        success, error = _execute_code_with_timeout(full_code, timeout=5)

        if success:
            correct += 1

        results.append({
            "index": idx,
            "task_id": item["task_id"],
            "text": item["text"][:200],
            "correct": success,
            "error": (error or "")[:300] if not success else None,
            "generated": generated,
        })

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "benchmark": "mbpp",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    return results, metrics


# ---------------------------------------------------------------------------
# Math-500 helpers
# ---------------------------------------------------------------------------

MATH500_SYSTEM_PROMPT = """You are a careful competition math assistant.

Solve the problem step by step. On the last line, write the final answer
after "Answer:" using only the final mathematical expression or value.

Rules:
1. Do not copy the problem statement.
2. Do not include \\boxed{} in the final line.
3. If the answer is a number, expression, tuple, or short text label,
   output exactly that after "Answer:".

Example format:
[Your reasoning]
Answer: 5/12"""


def _strip_matching_outer_delimiters(text: str) -> str:
    pairs = {"(": ")", "[": "]", "{": "}"}
    changed = True
    while changed and len(text) >= 2:
        changed = False
        left = text[0]
        right = pairs.get(left)
        if right != text[-1]:
            break
        depth = 0
        balanced = True
        for index, char in enumerate(text):
            if char == left:
                depth += 1
            elif char == right:
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    balanced = False
                    break
        if balanced and depth == 0:
            text = text[1:-1].strip()
            changed = True
    return text


def _strip_latex_boxing(text: str) -> str:
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for prefix in ("\\boxed{", "\\fbox{"):
            if text.startswith(prefix) and text.endswith("}"):
                inner = text[len(prefix):-1]
                depth = 0
                balanced = True
                for char in inner:
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth < 0:
                            balanced = False
                            break
                if balanced and depth == 0:
                    text = inner.strip()
                    changed = True
    patterns = [
        r"\\boxed\{([^{}]+)\}",
        r"\\fbox\{([^{}]+)\}",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_text = re.sub(pattern, r"\1", text)
            if new_text != text:
                text = new_text
                changed = True
    return text.strip()


def _extract_answer_line(text: str, prefix_pattern: str) -> str:
    matches = re.findall(prefix_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].strip()
    return ""


def _normalize_fraction_string(text: str) -> str:
    compact = text.replace(" ", "")
    if re.fullmatch(r"[-+]?\d+/\d+", compact):
        try:
            return str(Fraction(compact))
        except (ValueError, ZeroDivisionError):
            return compact
    if re.fullmatch(r"[-+]?\d+", compact):
        try:
            return str(int(compact))
        except ValueError:
            return compact
    if re.fullmatch(r"[-+]?\d+\.\d+", compact):
        try:
            value = float(compact)
            if value.is_integer():
                return str(int(value))
            return format(value, ".12g")
        except ValueError:
            return compact
    return compact


def _normalize_math500_answer(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.strip("`")
    normalized = normalized.strip("$")
    normalized = normalized.rstrip(".;,")
    normalized = _strip_latex_boxing(normalized)
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace("\\tfrac", "\\frac")
    normalized = normalized.replace("\\dfrac", "\\frac")
    normalized = normalized.replace("\\cdot", "*")
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("\\pi", "pi")
    normalized = normalized.replace("π", "pi")
    normalized = re.sub(r"\^\{?\\circ\}?", "deg", normalized)
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\operatorname\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", normalized)
    normalized = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", normalized)
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = normalized.strip()
    normalized = _strip_matching_outer_delimiters(normalized) if normalized.startswith("{") else normalized
    normalized = "".join(normalized.split())
    normalized = normalized.lower()
    return normalized


def extract_math500_answer(text: str) -> str:
    """Extract the final Math-500 answer from model output."""
    stripped = text.strip()
    answer = _extract_answer_line(stripped, r"Answer:\s*(.+)$")
    if answer:
        return _strip_latex_boxing(answer)

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", stripped)
    if boxed_matches:
        return boxed_matches[-1].strip()

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ""

    final_line = lines[-1]
    final_line = re.sub(r"^(final answer|answer)\s*[:：-]?\s*", "", final_line, flags=re.IGNORECASE)
    return _strip_latex_boxing(final_line.strip())


def load_math500_data(path: str) -> List[Dict[str, Any]]:
    """Load Math-500 JSONL data."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(prepare_math500_item(item))
    return data


def prepare_math500_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a single Math-500 record."""
    answer = item["answer"].strip()
    return {
        "problem": item["problem"],
        "answer": answer,
        "normalized_answer": _normalize_math500_answer(answer),
        "solution": item.get("solution", ""),
        "subject": item.get("subject", "unknown"),
        "level": item.get("level"),
        "unique_id": item.get("unique_id", ""),
    }


def build_math500_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for a single Math-500 item with non-leaking few-shot examples."""
    return [
        {"role": "system", "content": MATH500_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Simplify the expression 3(2x - 5) + 4.",
        },
        {
            "role": "assistant",
            "content": "Expand first: 3(2x - 5) = 6x - 15. Then add 4 to get 6x - 11.\nAnswer: 6x - 11",
        },
        {
            "role": "user",
            "content": "A right triangle has legs of length 6 and 8. What is the length of the hypotenuse?",
        },
        {
            "role": "assistant",
            "content": "By the Pythagorean theorem, the hypotenuse is sqrt(6^2 + 8^2) = sqrt(36 + 64) = sqrt(100) = 10.\nAnswer: 10",
        },
        {
            "role": "user",
            "content": "Find the coordinates of the midpoint of the segment joining (2, 5) and (8, 1).",
        },
        {
            "role": "assistant",
            "content": "The midpoint is ((2 + 8)/2, (5 + 1)/2) = (5, 3).\nAnswer: (5, 3)",
        },
        {"role": "user", "content": item["problem"]},
    ]


def score_math500(data: List[Dict[str, Any]], outputs: list) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Score Math-500 outputs using normalized exact match."""
    correct = 0
    results: List[Dict[str, Any]] = []
    subject_totals: Dict[str, int] = defaultdict(int)
    subject_correct: Dict[str, int] = defaultdict(int)

    for idx, (item, output) in enumerate(zip(data, outputs)):
        generated = output.outputs[0].text.strip()
        predicted_raw = extract_math500_answer(generated)
        predicted_normalized = _normalize_math500_answer(predicted_raw)
        target_normalized = item["normalized_answer"]

        is_correct = predicted_normalized == target_normalized
        if not is_correct:
            numeric_pred = _normalize_fraction_string(predicted_normalized)
            numeric_target = _normalize_fraction_string(target_normalized)
            is_correct = numeric_pred == numeric_target

        if is_correct:
            correct += 1
            subject_correct[item["subject"]] += 1
        subject_totals[item["subject"]] += 1

        results.append({
            "index": idx,
            "unique_id": item["unique_id"],
            "subject": item["subject"],
            "level": item["level"],
            "answer": item["answer"],
            "predicted": predicted_raw,
            "normalized_answer": target_normalized,
            "normalized_predicted": predicted_normalized,
            "correct": is_correct,
            "generated": generated,
        })

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    subject_accuracy = {
        subject: subject_correct[subject] / count
        for subject, count in subject_totals.items()
    }
    metrics = {
        "benchmark": "math500",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "subject_accuracy": subject_accuracy,
    }
    return results, metrics


# ---------------------------------------------------------------------------
# GraphWiz helpers
# ---------------------------------------------------------------------------

GRAPH_EXCLUDED_TASKS = {"topology"}
GRAPH_BINARY_TASKS = {"bipartite", "connectivity", "cycle", "hamilton", "substructure"}
GRAPH_NUMERIC_TASKS = {"flow", "shortest", "triangle"}

GRAPH_BINARY_SYSTEM_PROMPT = """You are a graph reasoning assistant.

Solve the graph problem carefully. On the last line, output only:
Answer: Yes
or
Answer: No"""

GRAPH_NUMERIC_SYSTEM_PROMPT = """You are a graph reasoning assistant.

Solve the graph problem carefully. On the last line, output only:
Answer: <integer>"""

GRAPH_YESNO_SYSTEM_PROMPT = """You are a graph reasoning assistant.

Solve the graph problem carefully. On the last line, output only:
Answer: Yes
or
Answer: No"""

GRAPH_FEW_SHOT_EXAMPLES: Dict[str, List[Dict[str, str]]] = {
    "bipartite": [
        {
            "role": "user",
            "content": (
                "Determine whether or not a graph is bipartite. In a directed graph, (i->j) means that node i "
                "and node j are connected with a directed edge from node i to node j. Given a graph, you need "
                "to output Yes or No, indicating whether the graph is bipartite. Q: The nodes are numbered from "
                "0 to 3, and the edges are: (0->1) (1->2) (2->3). Is this graph bipartite?"
            ),
        },
        {
            "role": "assistant",
            "content": "Ignoring edge direction for bipartiteness, this graph is a simple path and can be split into two sets.\nAnswer: Yes",
        },
    ],
    "connectivity": [
        {
            "role": "user",
            "content": (
                "Determine whether two nodes are connected in an undirected graph. In an undirected graph, "
                "(i,j) means that node i and node j are connected with an undirected edge. Given a graph and a "
                "pair of nodes, you need to output Yes or No, indicating whether the node i and node j are "
                "connected. Q: The nodes are numbered from 0 to 4, and the edges are: (0, 1) (1, 2) (3, 4). "
                "Is there a path between node 0 and node 4?"
            ),
        },
        {
            "role": "assistant",
            "content": "Nodes 0, 1, and 2 form one component, while 3 and 4 form another. There is no path from 0 to 4.\nAnswer: No",
        },
    ],
    "cycle": [
        {
            "role": "user",
            "content": (
                "Determine whether or not there is a cycle in an undirected graph. In an undirected graph, "
                "(i,j) means that node i and node j are connected with an undirected edge. Given a graph, you "
                "need to output Yes or No, indicating whether there is a cycle in the graph. Q: The nodes are "
                "numbered from 0 to 2, and the edges are: (0, 1) (1, 2) (2, 0). Is there a cycle in this graph?"
            ),
        },
        {
            "role": "assistant",
            "content": "The three edges form a triangle, so the graph contains a cycle.\nAnswer: Yes",
        },
    ],
    "flow": [
        {
            "role": "user",
            "content": (
                "Find the maximum flow between two nodes in a directed graph. In a directed graph, (i->j,k) "
                "means that node i and node j are connected with a directed edge from node i to node j with "
                "weight k. Given a graph and a pair of nodes, you need to output the maximum flow between the "
                "two nodes. Q: The nodes are numbered from 0 to 2, and the edges are: (0->1,3) (1->2,2) "
                "(0->2,1). What is the maximum flow from node 0 to node 2?"
            ),
        },
        {
            "role": "assistant",
            "content": "One unit can go directly from 0 to 2, and two units can go through 0->1->2. The total maximum flow is 3.\nAnswer: 3",
        },
    ],
    "hamilton": [
        {
            "role": "user",
            "content": (
                "Determine whether or not there is a Hamiltonian path in an undirected graph. In an undirected "
                "graph, (i,j) means that node i and node j are connected with an undirected edge. Given a graph, "
                "you need to output Yes or No, indicating whether there is a Hamiltonian path in the graph. "
                "Q: The nodes are numbered from 0 to 3, and the edges are: (0, 1) (1, 2) (2, 3). Is there a "
                "Hamiltonian path in this graph?"
            ),
        },
        {
            "role": "assistant",
            "content": "The path 0-1-2-3 visits every node exactly once, so a Hamiltonian path exists.\nAnswer: Yes",
        },
    ],
    "shortest": [
        {
            "role": "user",
            "content": (
                "Find the shortest path between two nodes in an undirected graph. In an undirected graph, "
                "(i,j,k) means that node i and node j are connected with an undirected edge with weight k. "
                "Given a graph and a pair of nodes, you need to output the shortest path between the two nodes. "
                "Q: The nodes are numbered from 0 to 2, and the edges are: (0,1,4) (1,2,3) (0,2,10). Give the "
                "weight of the shortest path from node 0 to node 2."
            ),
        },
        {
            "role": "assistant",
            "content": "Going through node 1 has total weight 4 + 3 = 7, which is smaller than 10.\nAnswer: 7",
        },
    ],
    "substructure": [
        {
            "role": "user",
            "content": (
                "Determine if a smaller graph is present as an exact match within a larger graph. In a directed "
                "graph, (i->j) means that node i and node j are connected with a directed edge from node i to "
                "node j. Given a graph G and a subgraph G', you need to output Yes or No, indicating whether "
                "subgraph G' is present within the directed graph G. Q: The nodes of graph G are numbered from "
                "0 to 3, and the edges are: (0->1) (1->2) (0->2) (2->3). The nodes of subgraph G' are numbered "
                "from a to c, and the edges are: (a->b) (b->c). Is subgraph G' present within graph G as a "
                "direct substructure?"
            ),
        },
        {
            "role": "assistant",
            "content": "Graph G contains directed chains such as 0->1->2 and 1->2->3, so the subgraph is present.\nAnswer: Yes",
        },
    ],
    "triangle": [
        {
            "role": "user",
            "content": (
                "Find the maximum sum of the weights of three interconnected nodes. In an undirected graph, "
                "[i, k] means that node i has the weight k. (i,j) means that node i and node j are connected "
                "with an undirected edge. Given a graph, you need to output the maximum sum of the weights of "
                "three interconnected nodes. Q: The nodes are numbered from 0 to 3, weights of nodes are: "
                "[0, 5] [1, 4] [2, 3] [3, 10], and the edges are: (0, 1) (1, 2) (0, 2) (1, 3) (0, 3). "
                "What is the maximum sum of the weights of three nodes?"
            ),
        },
        {
            "role": "assistant",
            "content": "Nodes 0, 1, and 3 are all pairwise connected, and their weights sum to 5 + 4 + 10 = 19.\nAnswer: 19",
        },
    ],
}


def extract_graph_answer(text: str, task: str) -> str:
    """Extract the final GraphWiz answer from model output."""
    stripped = text.strip()
    answer = _extract_answer_line(stripped, r"Answer:\s*(.+)$")
    if not answer:
        answer = _extract_answer_line(stripped, r"###\s*(.+)$")
    if not answer:
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        answer = lines[-1] if lines else ""
    answer = answer.strip().strip(".")

    if task in GRAPH_BINARY_TASKS:
        lowered = answer.lower()
        if "yes" in lowered:
            return "yes"
        if "no" in lowered:
            return "no"
    if task in GRAPH_NUMERIC_TASKS:
        match = re.search(r"-?\d+", answer)
        if match:
            return match.group(0)
    return answer


def load_graph_data(path: str) -> List[Dict[str, Any]]:
    """Load GraphWiz JSONL data, excluding topology."""
    data: List[Dict[str, Any]] = []
    excluded = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("task") in GRAPH_EXCLUDED_TASKS:
                excluded += 1
                continue
            data.append(prepare_graph_item(item))
    if excluded:
        print(f"GraphWiz: excluded {excluded} topology samples from evaluation.")
    return data


def prepare_graph_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a single GraphWiz record."""
    task = item["task"]
    return {
        "task": task,
        "input_prompt": item["input_prompt"],
        "answer": item["answer"],
        "ground_truth": extract_graph_answer(item["answer"], task),
        "index": item.get("index"),
    }


def build_graph_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for a single GraphWiz item."""
    task = item["task"]
    system_prompt = GRAPH_BINARY_SYSTEM_PROMPT if task in GRAPH_BINARY_TASKS else GRAPH_NUMERIC_SYSTEM_PROMPT
    few_shot = GRAPH_FEW_SHOT_EXAMPLES[task]
    return [
        {"role": "system", "content": system_prompt},
        *few_shot,
        {"role": "user", "content": item["input_prompt"]},
    ]


def score_graph(data: List[Dict[str, Any]], outputs: list) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Score GraphWiz outputs, excluding topology."""
    correct = 0
    results: List[Dict[str, Any]] = []
    task_totals: Dict[str, int] = defaultdict(int)
    task_correct: Dict[str, int] = defaultdict(int)

    for idx, (item, output) in enumerate(zip(data, outputs)):
        generated = output.outputs[0].text.strip()
        predicted = extract_graph_answer(generated, item["task"])
        ground_truth = item["ground_truth"]
        is_correct = predicted == ground_truth

        if is_correct:
            correct += 1
            task_correct[item["task"]] += 1
        task_totals[item["task"]] += 1

        results.append({
            "index": idx,
            "original_index": item["index"],
            "task": item["task"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "generated": generated,
            "input_prompt": item["input_prompt"],
        })

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    task_accuracy = {
        task: task_correct[task] / count
        for task, count in task_totals.items()
    }
    metrics = {
        "benchmark": "graph",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "task_accuracy": task_accuracy,
        "excluded_tasks": sorted(GRAPH_EXCLUDED_TASKS),
    }
    return results, metrics


def extract_graph_yesno_label(text: str) -> str:
    """Extract a yes/no label from free-form text."""
    stripped = text.strip()
    answer = _extract_answer_line(stripped, r"Answer:\s*(.+)$")
    if not answer:
        answer = stripped
    lowered = answer.lower()
    if re.search(r"\byes\b", lowered):
        return "yes"
    if re.search(r"\bno\b", lowered):
        return "no"
    return ""


def load_graph_yesno_data(path: str) -> List[Dict[str, Any]]:
    """Load NLgraph JSONL data and keep only samples with yes/no answers."""
    data_path = Path(path)
    data: List[Dict[str, Any]] = []
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("index", index)
            rows.append(item)

    for item in rows:
        prepared = prepare_graph_yesno_item(item)
        if prepared is not None:
            data.append(prepared)
    return data


def prepare_graph_yesno_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Prepare a single yes/no graph record."""
    ground_truth = extract_graph_yesno_label(str(item["answer"]))
    if not ground_truth:
        return None
    return {
        "task": item["task"],
        "question": item["question"],
        "answer": item["answer"],
        "ground_truth": ground_truth,
        "index": item.get("index"),
        "difficulty": item.get("difficulty"),
    }


def build_graph_yesno_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for yes/no graph items."""
    return [
        {"role": "system", "content": GRAPH_YESNO_SYSTEM_PROMPT},
        {"role": "user", "content": item["question"]},
    ]


def score_graph_yesno(
    data: List[Dict[str, Any]],
    outputs: list,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Score yes/no graph outputs."""
    correct = 0
    results: List[Dict[str, Any]] = []
    task_totals: Dict[str, int] = defaultdict(int)
    task_correct: Dict[str, int] = defaultdict(int)

    for idx, (item, output) in enumerate(zip(data, outputs)):
        generated = output.outputs[0].text.strip()
        predicted = extract_graph_yesno_label(generated)
        ground_truth = item["ground_truth"]
        is_correct = predicted == ground_truth

        task_totals[item["task"]] += 1
        if is_correct:
            correct += 1
            task_correct[item["task"]] += 1

        results.append({
            "index": idx,
            "original_index": item["index"],
            "difficulty": item["difficulty"],
            "task": item["task"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "generated": generated,
            "question": item["question"],
        })

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "benchmark": "graph_yesno",
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "task_accuracy": {
            task: task_correct[task] / count
            for task, count in task_totals.items()
        },
    }
    return results, metrics


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY = {
    "gpqa": {
        "load": load_gpqa_data,
        "prepare_item": prepare_gpqa_item,
        "build_messages": build_gpqa_messages,
        "score": score_gpqa,
    },
    "gsm8k": {
        "load": load_gsm8k_data,
        "prepare_item": prepare_gsm8k_item,
        "build_messages": build_gsm8k_messages,
        "score": score_gsm8k,
    },
    "bbh": {
        "load": load_bbh_data,
        "prepare_item": prepare_bbh_item,
        "build_messages": build_bbh_messages,
        "score": score_bbh,
    },
    "mmlu": {
        "load": load_mmlu_data,
        "prepare_item": prepare_mmlu_item,
        "build_messages": build_mmlu_messages,
        "score": score_mmlu,
    },
    "mbpp": {
        "load": load_mbpp_data,
        "prepare_item": prepare_mbpp_item,
        "build_messages": build_mbpp_messages,
        "score": score_mbpp,
    },
    "math500": {
        "load": load_math500_data,
        "prepare_item": prepare_math500_item,
        "build_messages": build_math500_messages,
        "score": score_math500,
    },
    "graph": {
        "load": load_graph_data,
        "prepare_item": prepare_graph_item,
        "build_messages": build_graph_messages,
        "score": score_graph,
    },
    "graph_yesno": {
        "load": load_graph_yesno_data,
        "prepare_item": prepare_graph_yesno_item,
        "build_messages": build_graph_yesno_messages,
        "score": score_graph_yesno,
    },
}


# ---------------------------------------------------------------------------
# Helpers for data loading and message building (shared by all eval modes)
# ---------------------------------------------------------------------------

def _load_eval_data(
    eval_tasks: Dict[str, str],
    merged_data_path: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load evaluation data from eval_tasks or a pre-merged JSONL.

    Returns per_task_data: {task_name: [prepared_items]}.
    """
    per_task_data: Dict[str, List[Dict[str, Any]]] = {}

    if merged_data_path and os.path.isfile(merged_data_path):
        from recipe_sandbox.evaluation.merge_eval_data import load_merged_eval_data
        raw_grouped = load_merged_eval_data(merged_data_path)
        for task_name, raw_items in raw_grouped.items():
            if task_name not in BENCHMARK_REGISTRY:
                print(f"WARNING: Unknown data_source '{task_name}' in merged file, skipping.")
                continue
            prepare = BENCHMARK_REGISTRY[task_name]["prepare_item"]
            per_task_data[task_name] = [prepare(item) for item in raw_items]
    else:
        if not eval_tasks:
            raise ValueError("eval_tasks is empty and no merged_data_path provided.")
        for task_name, data_path in eval_tasks.items():
            if task_name not in BENCHMARK_REGISTRY:
                raise ValueError(
                    f"Unknown task '{task_name}'. "
                    f"Supported: {list(BENCHMARK_REGISTRY.keys())}"
                )
            per_task_data[task_name] = BENCHMARK_REGISTRY[task_name]["load"](data_path)

    return per_task_data


def _build_all_messages(
    per_task_data: Dict[str, List[Dict[str, Any]]],
) -> tuple:
    """Build flat message list and item_mapping from per_task_data.

    Returns (all_messages, item_mapping) where item_mapping[i] = (task_name, idx_within_task).
    """
    all_messages: List[List[Dict[str, str]]] = []
    item_mapping: List[tuple] = []

    for task_name, items in per_task_data.items():
        build_fn = BENCHMARK_REGISTRY[task_name]["build_messages"]
        for idx, item in enumerate(items):
            all_messages.append(build_fn(item))
            item_mapping.append((task_name, idx))

    return all_messages, item_mapping


class _MockRequestOutput:
    """Minimal mock of vLLM RequestOutput for scoring shard-aggregated text."""

    class _Out:
        def __init__(self, text: str):
            self.text = text

    def __init__(self, text: str):
        self.outputs = [self._Out(text)]


# ---------------------------------------------------------------------------
# Parallel sharded evaluation (all GPUs, all benchmarks)
# ---------------------------------------------------------------------------

def evaluate_shard(
    model_path: str,
    eval_tasks: Dict[str, str],
    output_dir: str,
    shard_id: int,
    num_shards: int,
    *,
    lora_path: Optional[str] = None,
    merged_data_path: Optional[str] = None,
    batch_size: int = 64,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_num_seqs: int = 128,
    gpu_memory_utilization: float = 0.9,
) -> str:
    """Run vLLM inference on one shard of eval data (single GPU).

    Saves raw generated text to ``{output_dir}/shard_{shard_id}_outputs.jsonl``.
    Each line: {"global_idx": N, "task_name": "...", "task_idx": M, "text": "..."}
    """
    from vllm import LLM, SamplingParams

    # 1. Load all data and build messages
    per_task_data = _load_eval_data(eval_tasks, merged_data_path)
    all_messages, item_mapping = _build_all_messages(per_task_data)
    total = len(all_messages)

    # 2. Interleaved sharding: item i → shard (i % num_shards)
    shard_indices = [i for i in range(total) if i % num_shards == shard_id]
    shard_messages = [all_messages[i] for i in shard_indices]
    shard_mapping = [item_mapping[i] for i in shard_indices]

    print(f"[Shard {shard_id}/{num_shards}] {len(shard_messages)}/{total} items")

    if not shard_messages:
        print(f"[Shard {shard_id}] Empty shard, nothing to do.")
        os.makedirs(output_dir, exist_ok=True)
        shard_path = os.path.join(output_dir, f"shard_{shard_id}_outputs.jsonl")
        Path(shard_path).touch()
        return shard_path

    # 3. Init vLLM on this GPU (TP=1, single device)
    use_lora = lora_path is not None and os.path.isdir(lora_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=16384,
        max_num_seqs=max_num_seqs,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=use_lora,
        max_lora_rank=64 if use_lora else None,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    lora_request = None
    if use_lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("recipe_adapter", 1, lora_path)

    # 4. Run inference
    all_outputs = []
    for i in range(0, len(shard_messages), batch_size):
        batch = shard_messages[i : i + batch_size]
        if lora_request is not None:
            outputs = llm.chat(batch, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.chat(batch, sampling_params)
        all_outputs.extend(outputs)

    # 5. Save raw outputs
    os.makedirs(output_dir, exist_ok=True)
    shard_path = os.path.join(output_dir, f"shard_{shard_id}_outputs.jsonl")
    with open(shard_path, "w", encoding="utf-8") as f:
        for global_idx, output, (task_name, task_idx) in zip(
            shard_indices, all_outputs, shard_mapping
        ):
            entry = {
                "global_idx": global_idx,
                "task_name": task_name,
                "task_idx": task_idx,
                "text": output.outputs[0].text if output.outputs else "",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[Shard {shard_id}] Done — {len(all_outputs)} outputs → {shard_path}")
    return shard_path


def aggregate_shard_results(
    output_dir: str,
    eval_tasks: Dict[str, str],
    num_shards: int,
    *,
    merged_data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read shard outputs, score per-benchmark, write eval_metrics.json."""

    # 1. Load original eval data for scoring
    per_task_data = _load_eval_data(eval_tasks, merged_data_path)

    # 2. Read all shard output files
    all_entries = []
    for sid in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{sid}_outputs.jsonl")
        if not os.path.isfile(shard_path):
            print(f"WARNING: Missing shard file {shard_path}")
            continue
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))

    # Sort by global_idx to restore original order
    all_entries.sort(key=lambda x: x["global_idx"])

    # 3. Group generated text by (task_name, task_idx) for scoring
    task_texts: Dict[str, Dict[int, str]] = {t: {} for t in per_task_data}
    for entry in all_entries:
        tn = entry["task_name"]
        ti = entry["task_idx"]
        if tn in task_texts:
            task_texts[tn][ti] = entry["text"]

    # 4. Score each benchmark
    task_scores: Dict[str, float] = {}
    os.makedirs(output_dir, exist_ok=True)

    for task_name, items in per_task_data.items():
        texts_map = task_texts.get(task_name, {})
        # Build mock outputs in original item order
        mock_outputs = []
        for idx in range(len(items)):
            text = texts_map.get(idx, "")
            mock_outputs.append(_MockRequestOutput(text))

        if len(texts_map) != len(items):
            print(f"WARNING: {task_name} has {len(texts_map)} shard outputs "
                  f"but {len(items)} items — scoring may be partial")

        score_fn = BENCHMARK_REGISTRY[task_name]["score"]
        results, metrics = score_fn(items, mock_outputs)

        # Save per-benchmark results
        results_path = os.path.join(output_dir, f"{task_name}_results.jsonl")
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        metrics_path = os.path.join(output_dir, f"{task_name}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        task_scores[task_name] = metrics["accuracy"]
        print(f"  {task_name}: {metrics['accuracy']:.4f} "
              f"({metrics['correct']}/{metrics['total_samples']})")

    # 5. Aggregate and save
    aggregated_score = (
        sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    )
    total_items = sum(len(v) for v in per_task_data.values())

    aggregated_metrics = {
        "task_scores": task_scores,
        "aggregated_score": aggregated_score,
        "tasks_evaluated": list(task_scores.keys()),
        "eval_mode": "parallel_sharded",
        "num_shards": num_shards,
        "total_samples": total_items,
    }

    agg_path = os.path.join(output_dir, "eval_metrics.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2, ensure_ascii=False)

    # Clean up shard files
    for sid in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{sid}_outputs.jsonl")
        if os.path.isfile(shard_path):
            os.remove(shard_path)

    print(f"\n{'='*60}")
    print("Parallel Sharded Evaluation Summary")
    print(f"{'='*60}")
    for task_name in task_scores:
        print(f"  {task_name:>8s}: {task_scores[task_name]:.4f}")
    print(f"  {'aggregate':>8s}: {aggregated_score:.4f}")
    print(f"\nResults → {output_dir}")

    return aggregated_metrics


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _run_benchmark(
    benchmark_name: str,
    data_path: str,
    llm,
    sampling_params,
    lora_request,
    batch_size: int,
    output_dir: str,
) -> Dict[str, Any]:
    """Run a single benchmark and save its results. Returns metrics dict."""
    if benchmark_name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Supported: {list(BENCHMARK_REGISTRY.keys())}"
        )

    reg = BENCHMARK_REGISTRY[benchmark_name]

    # Load data
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark_name}")
    print(f"{'='*60}")
    print(f"Loading data from {data_path}...")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = reg["load"](data_path)
    if len(data) == 0:
        print(f"WARNING: 0 samples loaded for {benchmark_name}, skipping.")
        metrics = {
            "benchmark": benchmark_name,
            "total_samples": 0,
            "correct": 0,
            "accuracy": 0.0,
        }
        return metrics

    print(f"Loaded {len(data)} samples")

    # Build chat messages
    all_messages = [reg["build_messages"](item) for item in data]

    # Batch inference
    print(f"Running inference ({len(all_messages)} samples, batch={batch_size})...")
    all_outputs = []
    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(all_messages), batch_size), desc=f"{benchmark_name} eval")
    except ImportError:
        iterator = range(0, len(all_messages), batch_size)

    for i in iterator:
        batch = all_messages[i : i + batch_size]
        if lora_request is not None:
            outputs = llm.chat(batch, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.chat(batch, sampling_params)
        all_outputs.extend(outputs)

    # Score
    results, metrics = reg["score"](data, all_outputs)

    # Save per-benchmark results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{benchmark_name}_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(output_dir, f"{benchmark_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"{benchmark_name} accuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total_samples']})")
    return metrics


def evaluate_unified(
    model_path: str,
    eval_tasks: Dict[str, str],  # {"gpqa": "/path/gpqa.jsonl", "gsm8k": "/path/gsm8k.jsonl"}
    output_dir: str,
    *,
    lora_path: Optional[str] = None,
    batch_size: int = 32,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> Dict[str, Any]:
    """Run unified evaluation across multiple benchmarks.

    Loads the vLLM model once, then evaluates each benchmark sequentially.
    Returns the aggregated metrics dict.
    """
    from vllm import LLM, SamplingParams

    if not eval_tasks:
        raise ValueError("eval_tasks is empty — nothing to evaluate.")

    # Validate task names up front
    for task_name in eval_tasks:
        if task_name not in BENCHMARK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Supported: {list(BENCHMARK_REGISTRY.keys())}"
            )

    # Single model init
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
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    # Prepare LoRA request
    lora_request = None
    if use_lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name="recipe_adapter",
            lora_int_id=1,
            lora_local_path=lora_path,
        )

    # Evaluate each benchmark sequentially
    task_scores: Dict[str, float] = {}
    tasks_evaluated: List[str] = []

    for task_name, data_path in eval_tasks.items():
        metrics = _run_benchmark(
            benchmark_name=task_name,
            data_path=data_path,
            llm=llm,
            sampling_params=sampling_params,
            lora_request=lora_request,
            batch_size=batch_size,
            output_dir=output_dir,
        )
        task_scores[task_name] = metrics["accuracy"]
        tasks_evaluated.append(task_name)

    # Aggregated score = mean of all task accuracies
    aggregated_score = (
        sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    )

    aggregated_metrics = {
        "task_scores": task_scores,
        "aggregated_score": aggregated_score,
        "model_path": model_path,
        "tasks_evaluated": tasks_evaluated,
    }

    # Save aggregated metrics
    os.makedirs(output_dir, exist_ok=True)
    agg_path = os.path.join(output_dir, "eval_metrics.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("Unified Evaluation Summary")
    print(f"{'='*60}")
    for task_name in tasks_evaluated:
        print(f"  {task_name:>8s}: {task_scores[task_name]:.4f}")
    print(f"  {'aggregate':>8s}: {aggregated_score:.4f}")
    print(f"\nResults → {output_dir}")
    return aggregated_metrics


def evaluate_merged(
    model_path: str,
    eval_tasks: Dict[str, str],
    output_dir: str,
    *,
    lora_path: Optional[str] = None,
    merged_data_path: Optional[str] = None,
    batch_size: int = 64,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_num_seqs: int = 128,
    gpu_memory_utilization: float = 0.9,
) -> Dict[str, Any]:
    """Run merged evaluation — all benchmarks in a single vLLM inference pass.

    Instead of running each benchmark sequentially, this function:
      1. Loads all benchmark data (from merged JSONL or separate files)
      2. Builds chat messages for all items at once (each with its own
         system prompt and few-shot examples based on data_source)
      3. Submits everything to vLLM in a single inference call
      4. Splits outputs by data_source and scores each benchmark separately
      5. Returns aggregated metrics

    This maximises GPU utilisation by keeping the vLLM scheduler
    continuously fed with requests from all benchmarks simultaneously.

    Args:
        model_path: Path to the base model.
        eval_tasks: {"gpqa": "/path/gpqa.jsonl", "gsm8k": "/path/gsm8k.jsonl"}.
        output_dir: Directory for result files.
        lora_path: Optional LoRA adapter path.
        merged_data_path: Optional pre-merged JSONL (overrides eval_tasks for loading).
        batch_size: vLLM chat batch size (for progress bar chunking only).
        temperature: Sampling temperature.
        max_tokens: Maximum generation tokens.
        tensor_parallel_size: vLLM tensor parallelism.
        pipeline_parallel_size: vLLM pipeline parallelism (use >1 with multi-GPU).
        max_num_seqs: Maximum concurrent sequences in the vLLM scheduler.
        gpu_memory_utilization: Fraction of GPU memory for KV cache.

    Returns:
        Aggregated metrics dict with per-task and overall scores.
    """
    from vllm import LLM, SamplingParams

    # ------------------------------------------------------------------
    # 1. Load & prepare all benchmark data
    # ------------------------------------------------------------------
    per_task_data = _load_eval_data(eval_tasks, merged_data_path)

    total_items = sum(len(v) for v in per_task_data.values())
    if total_items == 0:
        raise ValueError("No evaluation data loaded.")

    # ------------------------------------------------------------------
    # 2. Build all chat messages and track mapping
    # ------------------------------------------------------------------
    all_messages, item_mapping = _build_all_messages(per_task_data)

    print(f"\n{'='*60}")
    print("Merged Evaluation")
    print(f"{'='*60}")
    for task_name, items in per_task_data.items():
        print(f"  {task_name}: {len(items)} samples")
    print(f"  Total: {total_items} samples")

    # ------------------------------------------------------------------
    # 3. Initialize vLLM with optimised parameters
    # ------------------------------------------------------------------
    use_lora = lora_path is not None and os.path.isdir(lora_path)
    print(f"\nLoading model from {model_path}...")
    if use_lora:
        print(f"  LoRA adapter: {lora_path} (native vLLM serving)")
    print(f"  TP={tensor_parallel_size}, PP={pipeline_parallel_size}, "
          f"max_num_seqs={max_num_seqs}")

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=16384,
        max_num_seqs=max_num_seqs,
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
        top_p=0.95,
        max_tokens=max_tokens,
    )

    lora_request = None
    if use_lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name="recipe_adapter",
            lora_int_id=1,
            lora_local_path=lora_path,
        )

    # ------------------------------------------------------------------
    # 4. Single merged inference pass
    # ------------------------------------------------------------------
    print(f"\nRunning merged inference ({total_items} samples)...")
    all_outputs = []
    try:
        from tqdm import tqdm
        iterator = tqdm(
            range(0, len(all_messages), batch_size),
            desc="merged eval",
            total=(len(all_messages) + batch_size - 1) // batch_size,
        )
    except ImportError:
        iterator = range(0, len(all_messages), batch_size)

    for i in iterator:
        batch = all_messages[i : i + batch_size]
        if lora_request is not None:
            outputs = llm.chat(batch, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.chat(batch, sampling_params)
        all_outputs.extend(outputs)

    # ------------------------------------------------------------------
    # 5. Split outputs by data_source and score each benchmark
    # ------------------------------------------------------------------
    # Group outputs back to their benchmark
    task_outputs: Dict[str, List] = {t: [] for t in per_task_data}
    for output, (task_name, _idx) in zip(all_outputs, item_mapping):
        task_outputs[task_name].append(output)

    task_scores: Dict[str, float] = {}
    os.makedirs(output_dir, exist_ok=True)

    for task_name, items in per_task_data.items():
        outputs = task_outputs[task_name]
        score_fn = BENCHMARK_REGISTRY[task_name]["score"]
        results, metrics = score_fn(items, outputs)

        # Save per-benchmark results
        results_path = os.path.join(output_dir, f"{task_name}_results.jsonl")
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        metrics_path = os.path.join(output_dir, f"{task_name}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        task_scores[task_name] = metrics["accuracy"]
        print(f"  {task_name}: {metrics['accuracy']:.4f} "
              f"({metrics['correct']}/{metrics['total_samples']})")

    # ------------------------------------------------------------------
    # 6. Aggregate and save
    # ------------------------------------------------------------------
    aggregated_score = (
        sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    )

    aggregated_metrics = {
        "task_scores": task_scores,
        "aggregated_score": aggregated_score,
        "model_path": model_path,
        "tasks_evaluated": list(task_scores.keys()),
        "eval_mode": "merged",
        "pipeline_parallel_size": pipeline_parallel_size,
        "max_num_seqs": max_num_seqs,
        "total_samples": total_items,
    }

    agg_path = os.path.join(output_dir, "eval_metrics.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Merged Evaluation Summary")
    print(f"{'='*60}")
    for task_name in task_scores:
        print(f"  {task_name:>8s}: {task_scores[task_name]:.4f}")
    print(f"  {'aggregate':>8s}: {aggregated_score:.4f}")
    print(f"\nResults → {output_dir}")
    return aggregated_metrics


def run_parallel_eval(
    model_path: str,
    eval_tasks: Dict[str, str],
    output_dir: str,
    *,
    lora_path: Optional[str] = None,
    batch_size: int = 32,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    available_gpu_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run benchmarks in parallel on separate GPUs.

    Each benchmark gets its own GPU and subprocess.
    Falls back to sequential evaluation if parallel fails.
    """
    import subprocess
    import sys

    if not eval_tasks or len(eval_tasks) < 2:
        return evaluate_unified(
            model_path=model_path, eval_tasks=eval_tasks, output_dir=output_dir,
            lora_path=lora_path, batch_size=batch_size, temperature=temperature,
            max_tokens=max_tokens, tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if available_gpu_ids is None:
        vis_env = _accelerator_env_var()
        visible = os.environ.get(vis_env, "")
        if visible:
            available_gpu_ids = [int(x.strip()) for x in visible.split(",")]
        else:
            try:
                import torch
                try:
                    import torch_npu  # noqa: F401
                except Exception:
                    pass
                npu = getattr(torch, "npu", None)
                if vis_env == "ASCEND_RT_VISIBLE_DEVICES" and npu and hasattr(npu, "device_count"):
                    available_gpu_ids = list(range(int(npu.device_count())))
                else:
                    available_gpu_ids = list(range(torch.cuda.device_count() or 8))
            except Exception:
                available_gpu_ids = list(range(8))

    n_gpus = len(available_gpu_ids)
    task_list = list(eval_tasks.items())

    this_script = os.path.abspath(__file__)
    os.makedirs(output_dir, exist_ok=True)

    processes = []
    for i, (task_name, data_path) in enumerate(task_list):
        gpu_id = available_gpu_ids[i % n_gpus]
        task_spec = f"{task_name}:{data_path}"

        cmd = [
            sys.executable, this_script,
            "--model_path", model_path,
            "--eval_tasks", task_spec,
            "--output_dir", output_dir,
            "--batch_size", str(batch_size),
            "--temperature", str(temperature),
            "--max_tokens", str(max_tokens),
            "--tensor_parallel_size", str(tensor_parallel_size),
            "--gpu_memory_utilization", str(gpu_memory_utilization),
            "--gpu_id", str(gpu_id),
        ]
        if lora_path:
            cmd.extend(["--lora_path", lora_path])

        env = dict(os.environ)
        env[_accelerator_env_var()] = str(gpu_id)

        print(f"[PARALLEL] Launching {task_name} on device {gpu_id}")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append((task_name, gpu_id, proc))

    failed = []
    for task_name, gpu_id, proc in processes:
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            print(f"[PARALLEL] {task_name} FAILED (GPU {gpu_id}, rc={proc.returncode})")
            if stdout:
                print(stdout.decode("utf-8", errors="replace")[-2000:])
            failed.append(task_name)
        else:
            print(f"[PARALLEL] {task_name} completed (GPU {gpu_id})")

    if failed:
        print(f"[PARALLEL] {len(failed)} task(s) failed: {failed}. Falling back to sequential.")
        return evaluate_unified(
            model_path=model_path, eval_tasks=eval_tasks, output_dir=output_dir,
            lora_path=lora_path, batch_size=batch_size, temperature=temperature,
            max_tokens=max_tokens, tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    # Merge results: each subprocess wrote {task_name}_metrics.json
    task_scores = {}
    for task_name, data_path in task_list:
        metrics_path = os.path.join(output_dir, f"{task_name}_metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            task_scores[task_name] = m.get("accuracy", 0.0)
        else:
            print(f"[PARALLEL] WARNING: {metrics_path} not found")
            task_scores[task_name] = 0.0

    aggregated_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    aggregated_metrics = {
        "task_scores": task_scores,
        "aggregated_score": aggregated_score,
        "model_path": model_path,
        "tasks_evaluated": list(task_scores.keys()),
        "eval_mode": "parallel",
    }

    agg_path = os.path.join(output_dir, "eval_metrics.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Parallel Evaluation Summary")
    print(f"{'='*60}")
    for task_name in task_scores:
        print(f"  {task_name:>8s}: {task_scores[task_name]:.4f}")
    print(f"  {'aggregate':>8s}: {aggregated_score:.4f}")

    return aggregated_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_eval_tasks(raw: str) -> Dict[str, str]:
    """Parse 'gpqa:/path/gpqa.jsonl,gsm8k:/path/gsm8k.jsonl' into a dict."""
    tasks: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid task spec '{part}'. Expected format: name:/path/to/data.jsonl"
            )
        # Split on first colon only so paths like /data/... work
        name, path = part.split(":", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise argparse.ArgumentTypeError(
                f"Invalid task spec '{part}'. Both name and path are required."
            )
        tasks[name] = path
    return tasks


def _accelerator_env_var() -> str:
    try:
        import torch
        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass
        npu = getattr(torch, "npu", None)
        if npu and hasattr(npu, "is_available") and npu.is_available() and not torch.cuda.is_available():
            return "ASCEND_RT_VISIBLE_DEVICES"
    except Exception:
        pass
    return "CUDA_VISIBLE_DEVICES"


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Merged mode (recommended) — single inference pass:\n"
            "  python unified_eval_vllm.py \\\n"
            "    --model_path /models/qwen \\\n"
            "    --eval_tasks gpqa:/data/gpqa.jsonl,gsm8k:/data/gsm8k.jsonl \\\n"
            "    --output_dir ./results --merged\n\n"
            "  # Parallel sharded mode — split across 8 GPUs:\n"
            "  for i in $(seq 0 7); do\n"
            "    CUDA_VISIBLE_DEVICES=$i python unified_eval_vllm.py \\\n"
            "      --model_path /models/qwen \\\n"
            "      --eval_tasks gpqa:/data/gpqa.jsonl,gsm8k:/data/gsm8k.jsonl \\\n"
            "      --output_dir ./results --merged \\\n"
            "      --shard_id $i --num_shards 8 &\n"
            "  done\n"
            "  wait\n"
            "  python unified_eval_vllm.py \\\n"
            "    --eval_tasks gpqa:/data/gpqa.jsonl,gsm8k:/data/gsm8k.jsonl \\\n"
            "    --output_dir ./results --aggregate --num_shards 8\n\n"
            "  # Sequential mode (legacy):\n"
            "  python unified_eval_vllm.py \\\n"
            "    --model_path /models/qwen \\\n"
            "    --eval_tasks gpqa:/data/gpqa.jsonl \\\n"
            "    --output_dir ./results"
        ),
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the base model (not required for --aggregate)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter for native vLLM LoRA serving")
    parser.add_argument("--eval_tasks", type=str, required=True,
                        help="Comma-separated task:path pairs, e.g. "
                             "gpqa:/path/gpqa.jsonl,gsm8k:/path/gsm8k.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output files")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                        help="Pipeline parallel size (>1 for multi-GPU pipeline)")
    parser.add_argument("--max_num_seqs", type=int, default=128,
                        help="Max concurrent sequences in vLLM scheduler")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="Specific accelerator ID to use. Sets CUDA_VISIBLE_DEVICES or ASCEND_RT_VISIBLE_DEVICES.")
    parser.add_argument("--merged", action="store_true", default=False,
                        help="Merged batching: all benchmarks in a single inference pass")
    parser.add_argument("--merged_data_path", type=str, default=None,
                        help="Path to pre-merged JSONL (with data_source field)")
    # Parallel sharded eval
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Shard index for parallel eval (0-based)")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of shards for parallel eval")
    parser.add_argument("--aggregate", action="store_true", default=False,
                        help="Aggregate shard results (no inference, just scoring)")
    # Legacy flag kept for backward compatibility
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="(Legacy) Run benchmarks in parallel on separate GPUs")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ[_accelerator_env_var()] = str(args.gpu_id)

    eval_tasks = parse_eval_tasks(args.eval_tasks)

    # --- Aggregate mode: no inference, just combine shard outputs and score ---
    if args.aggregate:
        if args.num_shards is None:
            parser.error("--aggregate requires --num_shards")
        aggregate_shard_results(
            eval_tasks=eval_tasks,
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            merged_data_path=args.merged_data_path,
        )
        return

    # All inference modes require --model_path
    if args.model_path is None:
        parser.error("--model_path is required for inference modes")

    # --- Shard mode: run inference on one interleaved shard ---
    if args.shard_id is not None and args.num_shards is not None:
        evaluate_shard(
            model_path=args.model_path,
            eval_tasks=eval_tasks,
            output_dir=args.output_dir,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            lora_path=args.lora_path,
            merged_data_path=args.merged_data_path,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    elif args.merged or args.merged_data_path:
        # --- Merged mode: single GPU, all benchmarks in one pass ---
        evaluate_merged(
            model_path=args.model_path,
            eval_tasks=eval_tasks,
            output_dir=args.output_dir,
            lora_path=args.lora_path,
            merged_data_path=args.merged_data_path,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    elif args.parallel and len(eval_tasks) >= 2 and args.gpu_id is None:
        run_parallel_eval(
            model_path=args.model_path,
            eval_tasks=eval_tasks,
            output_dir=args.output_dir,
            lora_path=args.lora_path,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        evaluate_unified(
            model_path=args.model_path,
            eval_tasks=eval_tasks,
            output_dir=args.output_dir,
            lora_path=args.lora_path,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )


if __name__ == "__main__":
    main()
