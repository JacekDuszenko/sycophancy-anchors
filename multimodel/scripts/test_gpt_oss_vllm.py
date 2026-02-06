#!/usr/bin/env python3
"""
Quick test script to verify GPT-OSS-20B works with vLLM.

Tests:
1. Model loading with MXFP4
2. Chat template formatting
3. Generation with special tokens preserved
4. CoT extraction

Usage:
    VLLM_WORKER_MULTIPROC_METHOD=spawn python test_gpt_oss_vllm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM, SamplingParams


def main():
    print("=" * 80)
    print("GPT-OSS-20B vLLM Test")
    print("=" * 80)

    model_id = "openai/gpt-oss-20b"

    print(f"\n[1] Loading model: {model_id}")
    try:
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            quantization="mxfp4",
        )
        print("    Model loaded successfully!")
    except Exception as e:
        print(f"    FAILED to load model: {e}")
        return

    tokenizer = llm.get_tokenizer()

    # Print special tokens
    print("\n[2] Special tokens:")
    print(f"    BOS: {repr(tokenizer.bos_token)}")
    print(f"    EOS: {repr(tokenizer.eos_token)}")
    print(f"    PAD: {repr(tokenizer.pad_token)}")
    if hasattr(tokenizer, 'additional_special_tokens'):
        print(f"    Additional: {tokenizer.additional_special_tokens[:5]}...")

    # Simple test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Answer briefly."},
    ]

    print("\n[3] Applying chat template...")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"    Prompt:\n{'-'*40}\n{prompt}\n{'-'*40}")

    print("\n[4] Generating response...")
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=512,
    )

    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text

    print(f"\n[5] Raw response:\n{'-'*40}\n{response}\n{'-'*40}")

    # Check for channel tokens
    print("\n[6] Token analysis:")
    has_analysis = "<|channel|>analysis" in response
    has_final = "<|channel|>final" in response
    has_end = "<|end|>" in response
    print(f"    Has analysis channel: {has_analysis}")
    print(f"    Has final channel: {has_final}")
    print(f"    Has end token: {has_end}")

    # Try extraction
    print("\n[7] Testing extraction...")
    from generation.cot_utils import extract_cot_and_answer
    from config.model_registry import get_model_config

    config = get_model_config("gpt-oss-20b")
    cot, answer, has_marker = extract_cot_and_answer(response, config)

    print(f"    Has marker: {has_marker}")
    print(f"    CoT: {cot[:100] if cot else '(empty)'}...")
    print(f"    Answer: {answer[:100] if answer else '(empty)'}...")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
