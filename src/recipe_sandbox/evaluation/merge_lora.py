"""Merge LoRA adapter into base model.

Adapted from TokenCleaning (UCSC-REAL/TokenCleaning/scripts/merge_lora.py).
Simplified: no QLoRA dequantization (we use bf16 LoRA).
"""

import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_dir: str,
    *,
    save_tokenizer: bool = True,
    use_fast_tokenizer: bool = True,
    dtype: str = "bfloat16",
) -> None:
    """Merge LoRA adapter into base model and save."""

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    print(f"Loading tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=True,
    )

    # Resize embeddings if needed (Llama-3 pattern from TokenCleaning)
    if len(tokenizer) > base_model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing embeddings: {base_model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))

    print(f"Loading LoRA adapter: {lora_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging LoRA weights...")
    merged_model = lora_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model → {output_dir}")
    merged_model.save_pretrained(output_dir)

    if save_tokenizer:
        print(f"Saving tokenizer → {output_dir}")
        tokenizer.save_pretrained(output_dir)

    print("Merge complete!")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_tokenizer", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    args = parser.parse_args()

    merge_lora(
        base_model_path=args.base_model_name_or_path,
        lora_path=args.lora_model_name_or_path,
        output_dir=args.output_dir,
        save_tokenizer=args.save_tokenizer,
        use_fast_tokenizer=args.use_fast_tokenizer,
    )


if __name__ == "__main__":
    main()
