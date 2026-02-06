#!/usr/bin/env python3
"""
Prepare Llama-8B data for multimodel pipeline activation extraction.

This script:
1. Creates llama_multimodel directory structure
2. Copies correct samples from prod_run_v1 (is_sycophantic=False)
3. Copies sycophantic samples from syco_base (is_sycophantic=True)
4. Ensures is_sycophantic field is set in all base.json files
5. Creates rollouts_done.flag files where needed
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm


def copy_sample_with_sycophantic_flag(
    src_dir: Path,
    dst_dir: Path,
    is_sycophantic: bool,
) -> bool:
    """Copy a sample directory and set is_sycophantic flag in base.json."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy base.json with is_sycophantic field added
    src_base = src_dir / "base.json"
    if not src_base.exists():
        return False

    with open(src_base, "r") as f:
        base_data = json.load(f)

    # Check for anchor_analysis
    if "anchor_analysis" not in base_data or not base_data["anchor_analysis"]:
        return False

    # Set is_sycophantic
    base_data["is_sycophantic"] = is_sycophantic

    # Normalize anchor_analysis fields:
    # - syco_base uses "accuracy_increase" for sycophantic samples
    # - prod_run uses "accuracy_drop" for correct samples
    # Both measure the causal impact of the sentence, so we normalize to "accuracy_drop"
    for anchor in base_data["anchor_analysis"]:
        if "accuracy_increase" in anchor and "accuracy_drop" not in anchor:
            anchor["accuracy_drop"] = anchor["accuracy_increase"]
        if "is_sycophantic_anchor" in anchor and "is_anchor" not in anchor:
            anchor["is_anchor"] = anchor["is_sycophantic_anchor"]

    dst_base = dst_dir / "base.json"
    with open(dst_base, "w") as f:
        json.dump(base_data, f, indent=2)

    # Copy sample.json if exists
    src_sample = src_dir / "sample.json"
    if src_sample.exists():
        shutil.copy2(src_sample, dst_dir / "sample.json")

    # Copy rollouts.jsonl if exists
    src_rollouts = src_dir / "rollouts.jsonl"
    if src_rollouts.exists():
        shutil.copy2(src_rollouts, dst_dir / "rollouts.jsonl")

    # Create rollouts_done.flag (needed by activation extractor)
    (dst_dir / "rollouts_done.flag").touch()

    return True


def main():
    base_dir = Path("/home/jacekduszenko/Workspace/keyword_probe_experiment/thought-anchors")

    # Source directories
    prod_run_dir = base_dir / "prod_run_v1" / "DeepSeek-R1-Distill-Llama-8B"
    syco_base_dir = base_dir / "syco_base" / "DeepSeek-R1-Distill-Llama-8B"

    # Output directory
    output_dir = base_dir / "llama_multimodel" / "deepseek-llama-8b" / "base_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preparing Llama-8B Multimodel Data")
    print("=" * 60)

    # Collect samples
    correct_samples = []
    sycophantic_samples = []

    # Step 1: Find correct samples with anchor_analysis from prod_run_v1
    print("\nScanning prod_run_v1 for correct samples with anchor_analysis...")
    for sample_dir in sorted(prod_run_dir.glob("arc_conv_*")):
        base_path = sample_dir / "base.json"
        if not base_path.exists():
            continue

        try:
            with open(base_path, "r") as f:
                data = json.load(f)

            is_correct = data.get("is_correct", False)
            has_anchor = "anchor_analysis" in data and len(data.get("anchor_analysis", [])) > 0

            if is_correct and has_anchor:
                correct_samples.append(sample_dir)
        except Exception as e:
            print(f"  Error reading {sample_dir}: {e}")

    print(f"Found {len(correct_samples)} correct samples with anchor_analysis")

    # Step 2: Find sycophantic samples from syco_base
    print("\nScanning syco_base for sycophantic samples...")
    for sample_dir in sorted(syco_base_dir.glob("arc_conv_*")):
        base_path = sample_dir / "base.json"
        if not base_path.exists():
            continue

        try:
            with open(base_path, "r") as f:
                data = json.load(f)

            # syco_base samples are sycophantic (is_correct=False)
            is_correct = data.get("is_correct", True)  # default True so we skip if not found
            has_anchor = "anchor_analysis" in data and len(data.get("anchor_analysis", [])) > 0

            if not is_correct and has_anchor:
                sycophantic_samples.append(sample_dir)
        except Exception as e:
            print(f"  Error reading {sample_dir}: {e}")

    print(f"Found {len(sycophantic_samples)} sycophantic samples with anchor_analysis")

    # Check for overlap (same sample_id in both)
    correct_ids = set(s.name for s in correct_samples)
    syco_ids = set(s.name for s in sycophantic_samples)
    overlap = correct_ids & syco_ids
    print(f"\nOverlapping sample IDs: {len(overlap)}")

    # For overlapping samples, prioritize syco_base (has the wrong-answer version)
    # But we need both for the probe - so we'll skip overlaps from prod_run
    if overlap:
        print(f"Will skip overlapping samples from prod_run_v1 (keep syco_base versions)")
        correct_samples = [s for s in correct_samples if s.name not in overlap]
        print(f"Adjusted correct samples: {len(correct_samples)}")

    # Step 3: Copy samples
    print("\nCopying samples to output directory...")

    copied_correct = 0
    copied_syco = 0

    # Copy correct samples (is_sycophantic=False)
    print("  Copying correct samples...")
    for sample_dir in tqdm(correct_samples, desc="Correct"):
        dst_dir = output_dir / sample_dir.name
        if copy_sample_with_sycophantic_flag(sample_dir, dst_dir, is_sycophantic=False):
            copied_correct += 1

    # Copy sycophantic samples (is_sycophantic=True)
    print("  Copying sycophantic samples...")
    for sample_dir in tqdm(sycophantic_samples, desc="Sycophantic"):
        dst_dir = output_dir / sample_dir.name
        if copy_sample_with_sycophantic_flag(sample_dir, dst_dir, is_sycophantic=True):
            copied_syco += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Correct (non-sycophantic) samples: {copied_correct}")
    print(f"Sycophantic samples: {copied_syco}")
    print(f"Total samples: {copied_correct + copied_syco}")

    # Create activations directory
    activations_dir = output_dir.parent / "activations"
    activations_dir.mkdir(exist_ok=True)
    print(f"\nActivations directory created: {activations_dir}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run activation extraction:")
    print("   VLLM_WORKER_MULTIPROC_METHOD=spawn python multimodel/scripts/run_pipeline.py \\")
    print("     --model deepseek-llama-8b --output-dir llama_multimodel --stage 3")
    print("")
    print("2. Run threshold sweep analysis:")
    print("   python scripts/threshold_sweep_analysis.py deepseek-llama-8b")
    print("")
    print("3. Run text baseline comparison:")
    print("   python scripts/text_baseline_comparison.py deepseek-llama-8b")


if __name__ == "__main__":
    main()
