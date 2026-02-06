#!/usr/bin/env python3
"""
Exploratory script for openai/gpt-oss-20b model.

This script tests:
1. Model loading with MXFP4 quantization (built into weights)
2. Memory usage on A100
3. Generation of one sample from the ARC dataset
4. Output format and potential CoT markers

The model is a MoE with 21B total params / 3.6B active params.
Should fit in ~16GB VRAM with MXFP4 quantization.

CRITICAL: MXFP4 requires specific dependencies:
    pip install -U transformers torch
    pip install triton>=3.4.0
    pip install kernels

Usage:
    python explore_gpt_oss_20b.py
"""

import gc
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import load_arc_dataset


def check_mxfp4_dependencies():
    """Check if MXFP4 quantization dependencies are available."""
    print("\n[0] Checking MXFP4 dependencies...")

    issues = []

    # Check Triton
    try:
        import triton
        triton_version = triton.__version__
        print(f"    Triton: {triton_version}")
        # Parse version - need >= 3.4.0
        parts = triton_version.split('.')
        major, minor = int(parts[0]), int(parts[1])
        if major < 3 or (major == 3 and minor < 4):
            issues.append(f"Triton {triton_version} < 3.4.0 (required for MXFP4)")
    except ImportError:
        issues.append("Triton not installed (required for MXFP4)")
        print("    Triton: NOT INSTALLED")

    # Check kernels package
    try:
        import kernels
        print(f"    kernels: installed")
    except ImportError:
        issues.append("kernels package not installed (required for MXFP4)")
        print("    kernels: NOT INSTALLED")

    # Check torch
    print(f"    torch: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA version: {torch.version.cuda}")

    if issues:
        print("\n    WARNING: MXFP4 will NOT work due to missing dependencies:")
        for issue in issues:
            print(f"      - {issue}")
        print("\n    To fix, run:")
        print("      pip install triton>=3.4.0 kernels")
        return False

    print("    All MXFP4 dependencies satisfied!")
    return True


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"available": False}

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    # Get total memory
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "max_allocated_gb": round(max_allocated, 2),
        "free_gb": round(total - reserved, 2),
    }


def log_memory(stage: str, output_log: list):
    """Log memory at a specific stage."""
    mem = get_gpu_memory_info()
    entry = {"stage": stage, "timestamp": datetime.now().isoformat(), **mem}
    output_log.append(entry)
    print(f"[{stage}] GPU Memory: {mem.get('allocated_gb', 'N/A')} GB allocated, "
          f"{mem.get('reserved_gb', 'N/A')} GB reserved, {mem.get('free_gb', 'N/A')} GB free")
    return entry


def build_chat_messages(sample: dict) -> list:
    """Build chat messages from sample's conversation history."""
    history = sample.get("conversation_history", [])
    if not history:
        question = sample["question"]
        choices = sample.get("choices", {})
        user_answer = sample.get("user_answer", sample.get("distractor", ""))

        if isinstance(choices, dict) and "label" in choices and "text" in choices:
            choices_text = "\n".join(
                f"{l}. {t}" for l, t in zip(choices["label"], choices["text"])
            )
        else:
            choices_text = ""

        content = f"{question}\n{choices_text}\nI think it is {user_answer}, right?"
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]

    return [{"role": "system", "content": "You are a helpful assistant."}] + history


def main():
    output_log = []
    output_dir = Path(__file__).parent.parent.parent / "gpt_oss_exploration"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GPT-OSS-20B Exploration Script")
    print("=" * 80)

    # Check dependencies first
    mxfp4_available = check_mxfp4_dependencies()

    if not mxfp4_available:
        print("\n" + "=" * 80)
        print("ABORTING: MXFP4 dependencies not met.")
        print("Without MXFP4, the model requires ~40GB VRAM (bf16 fallback)")
        print("Install dependencies first:")
        print("  pip install triton>=3.4.0 kernels")
        print("=" * 80)

        # Save diagnostic info
        with open(output_dir / "dependency_check.json", "w") as f:
            json.dump({
                "mxfp4_available": False,
                "timestamp": datetime.now().isoformat(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            }, f, indent=2)
        return

    # Log initial state
    log_memory("initial", output_log)

    # Load dataset first (minimal memory)
    print("\n[1] Loading ARC dataset...")
    dataset = load_arc_dataset()
    sample = dataset[0]  # Get first sample
    print(f"    Loaded {len(dataset)} samples, using sample 0")
    print(f"    Question: {sample['question'][:100]}...")
    print(f"    Correct answer: {sample['correct_answer']}")
    print(f"    User answer (distractor): {sample.get('user_answer', sample.get('distractor', 'N/A'))}")

    log_memory("after_dataset_load", output_log)

    # Load model
    print("\n[2] Loading GPT-OSS-20B model...")
    print("    Model: openai/gpt-oss-20b (MoE, 21B total / 3.6B active params)")
    print("    Quantization: MXFP4 (built into weights)")

    model_id = "openai/gpt-oss-20b"
    start_time = time.time()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("    Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        log_memory("after_tokenizer_load", output_log)

        print("    Loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",  # Let model decide dtype
            device_map="auto",   # Automatic device placement
            trust_remote_code=True,
        )
        model.eval()

        load_time = time.time() - start_time
        print(f"    Model loaded in {load_time:.1f} seconds")

        log_memory("after_model_load", output_log)

        # Log model info
        model_info = {
            "model_id": model_id,
            "load_time_seconds": round(load_time, 2),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "model_type": type(model).__name__,
            "dtype": str(next(model.parameters()).dtype),
            "config": {
                "hidden_size": getattr(model.config, "hidden_size", None),
                "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
                "num_attention_heads": getattr(model.config, "num_attention_heads", None),
                "vocab_size": getattr(model.config, "vocab_size", None),
                "num_experts": getattr(model.config, "num_local_experts",
                                      getattr(model.config, "num_experts", None)),
            }
        }
        print(f"\n    Model info:")
        print(f"      - Parameters: {model_info['num_parameters']:,}")
        print(f"      - Type: {model_info['model_type']}")
        print(f"      - Dtype: {model_info['dtype']}")
        print(f"      - Hidden size: {model_info['config']['hidden_size']}")
        print(f"      - Layers: {model_info['config']['num_hidden_layers']}")
        print(f"      - Experts: {model_info['config']['num_experts']}")

    except Exception as e:
        error_info = {
            "stage": "model_loading",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        output_log.append(error_info)
        print(f"\n    ERROR loading model: {e}")

        # Save partial results
        with open(output_dir / "exploration_results.json", "w") as f:
            json.dump({
                "status": "failed",
                "error": str(e),
                "memory_log": output_log,
            }, f, indent=2)

        return

    # Format prompt using chat template
    print("\n[3] Formatting prompt...")
    messages = build_chat_messages(sample)

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"    Prompt length: {len(prompt)} chars")
        print(f"    Prompt preview:\n{'-'*40}\n{prompt[:500]}...\n{'-'*40}")
    except Exception as e:
        print(f"    WARNING: Chat template failed ({e}), using fallback")
        # Fallback: simple concatenation
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt += "\nassistant:"

    # Generate response
    print("\n[4] Generating response...")
    gen_start = time.time()

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        log_memory("before_generation", output_log)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_time = time.time() - gen_start
        log_memory("after_generation", output_log)

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract only the generated part (after prompt)
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=False)

        # Count tokens
        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        tokens_per_second = output_tokens / gen_time if gen_time > 0 else 0

        print(f"    Generation completed in {gen_time:.1f}s")
        print(f"    Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"    Speed: {tokens_per_second:.1f} tokens/second")

        # Analyze response for potential CoT markers
        print("\n[5] Analyzing response for CoT markers...")
        potential_markers = [
            "</think>", "<|end_of_thought|>", "</reasoning>",
            "<|answer|>", "Final answer:", "Therefore,"
        ]
        found_markers = []
        for marker in potential_markers:
            if marker.lower() in generated_text.lower():
                found_markers.append(marker)

        print(f"    Found potential CoT markers: {found_markers if found_markers else 'None'}")

        # Check special tokens in tokenizer
        print("\n    Special tokens in tokenizer:")
        special_tokens = tokenizer.special_tokens_map
        for name, token in special_tokens.items():
            print(f"      - {name}: {token}")

        # Also check additional special tokens
        if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
            print(f"      - additional_special_tokens: {tokenizer.additional_special_tokens}")

        # Print generated response
        print("\n[6] Generated response:")
        print("=" * 60)
        print(generated_text[:3000])
        if len(generated_text) > 3000:
            print(f"\n... [truncated, total {len(generated_text)} chars]")
        print("=" * 60)

        generation_info = {
            "status": "success",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_time_seconds": round(gen_time, 2),
            "tokens_per_second": round(tokens_per_second, 2),
            "found_cot_markers": found_markers,
            "special_tokens": special_tokens,
        }

    except Exception as e:
        print(f"    ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        generation_info = {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        generated_text = ""

    # Final memory check
    log_memory("final", output_log)

    # Save all results
    print("\n[7] Saving results...")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "model_info": model_info if 'model_info' in dir() else {},
        "sample": {
            "question": sample["question"],
            "correct_answer": sample["correct_answer"],
            "user_answer": sample.get("user_answer", sample.get("distractor", "")),
        },
        "prompt": prompt,
        "generated_text": generated_text,
        "generation_info": generation_info,
        "memory_log": output_log,
    }

    output_file = output_dir / "exploration_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved to: {output_file}")

    # Also save raw response to text file for easy viewing
    response_file = output_dir / "generated_response.txt"
    with open(response_file, "w") as f:
        f.write(f"Prompt:\n{'-'*40}\n{prompt}\n\n")
        f.write(f"Response:\n{'-'*40}\n{generated_text}")
    print(f"    Raw response saved to: {response_file}")

    print("\n" + "=" * 80)
    print("Exploration complete!")
    print("=" * 80)

    # Memory summary
    if output_log:
        initial = next((e for e in output_log if e["stage"] == "initial"), {})
        final = next((e for e in output_log if e["stage"] == "final"), {})
        peak = next((e for e in output_log if e["stage"] == "after_generation"), {})

        print("\nMemory Summary:")
        print(f"  GPU: {initial.get('device_name', 'N/A')}")
        print(f"  Total VRAM: {initial.get('total_gb', 'N/A')} GB")
        print(f"  Peak allocated: {peak.get('max_allocated_gb', 'N/A')} GB")
        print(f"  Final allocated: {final.get('allocated_gb', 'N/A')} GB")


if __name__ == "__main__":
    main()
