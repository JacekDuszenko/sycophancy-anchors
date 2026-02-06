#!/usr/bin/env python3
"""Main entry point for the multi-model sycophancy experiment pipeline."""

import argparse
import sys
import gc
import torch
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multimodel.config.model_registry import (
    MODEL_CONFIGS,
    MODEL_ORDER,
    get_model_config,
)
from multimodel.config.experiment_config import ExperimentConfig
from multimodel.data.dataset_loader import DatasetLoader
from multimodel.generation.cot_token_discovery import discover_cot_token
from multimodel.generation.base_generator import BaseGenerator
from multimodel.generation.rollout_generator import RolloutGenerator
from multimodel.generation.judge_rollouts import RolloutJudge
from multimodel.extraction.activation_extractor import ActivationExtractor
from multimodel.experiments.experiment_runner import (
    ExperimentRunner,
    generate_cross_model_summary,
)
from multimodel.output.csv_exporter import export_all_results
from multimodel.checkpointing.checkpoint_manager import CheckpointManager


def clear_gpu_memory():
    """Clear GPU memory between stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_stage_0(experiment_config: ExperimentConfig) -> bool:
    """Stage 0: Download dataset.

    Args:
        experiment_config: Experiment configuration

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("STAGE 0: Dataset Download")
    print("=" * 60)

    try:
        loader = DatasetLoader(cache_dir=experiment_config.cache_dir)
        data = loader.load()
        print(f"Loaded {len(data)} samples from dataset")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def run_stage_1(model_name: str, experiment_config: ExperimentConfig, filter_questions_file: str = None) -> bool:
    """Stage 1: Generate base responses until target syco/nonsyco counts are met.

    Args:
        model_name: Model identifier
        experiment_config: Experiment configuration
        filter_questions_file: Optional path to file with question IDs to filter

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print(f"STAGE 1: Base Generation ({model_name})")
    print(f"Target: {experiment_config.target_syco_count} syco + {experiment_config.target_nonsyco_count} nonsyco")
    print("=" * 60)

    model_config = get_model_config(model_name)
    experiment_config.ensure_dirs(model_name)

    checkpoint_mgr = CheckpointManager(
        experiment_config.get_base_data_dir(model_name)
    )

    if checkpoint_mgr.is_stage_complete("base_generation"):
        print("Base generation already complete, skipping...")
        return True

    if model_config.cot_end_token is None:
        print("Discovering CoT end token...")
        discover_cot_token(model_config)
        clear_gpu_memory()

    try:
        loader = DatasetLoader(cache_dir=experiment_config.cache_dir)
        samples = loader.load()

        # Filter samples if filter file provided
        if filter_questions_file:
            with open(filter_questions_file, 'r') as f:
                filter_ids = set(line.strip() for line in f if line.strip())
            # Extract question indices from IDs like "arc_conv_0002"
            filter_indices = set()
            for qid in filter_ids:
                try:
                    idx = int(qid.split('_')[-1])
                    filter_indices.add(idx)
                except ValueError:
                    pass
            # Keep only filtered samples but preserve original indices in the sample
            filtered_samples = []
            for i, s in enumerate(samples):
                if i in filter_indices:
                    s = dict(s)  # Copy to avoid modifying original
                    s['_original_idx'] = i  # Store original index
                    filtered_samples.append(s)
            samples = filtered_samples
            print(f"Filtered to {len(samples)} difficult questions from {len(filter_indices)} IDs")

        generator = BaseGenerator(model_config, experiment_config)
        start_idx = checkpoint_mgr.get_last_sample_idx() + 1
        generator.generate_batch(samples[start_idx:], start_idx=start_idx)
        generator.cleanup()
        clear_gpu_memory()

        checkpoint_mgr.mark_stage_complete("base_generation")
        return True

    except Exception as e:
        print(f"Error in base generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_stage_2(model_name: str, experiment_config: ExperimentConfig) -> bool:
    """Stage 2: Generate rollouts.

    Args:
        model_name: Model identifier
        experiment_config: Experiment configuration

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print(f"STAGE 2: Rollout Generation ({model_name})")
    print("=" * 60)

    model_config = get_model_config(model_name)

    checkpoint_mgr = CheckpointManager(
        experiment_config.get_base_data_dir(model_name)
    )

    if checkpoint_mgr.is_stage_complete("rollout_generation"):
        print("Rollout generation already complete, skipping...")
        return True

    try:
        generator = RolloutGenerator(model_config, experiment_config)
        generator.generate_for_all_samples()
        generator.cleanup()
        clear_gpu_memory()

        checkpoint_mgr.mark_stage_complete("rollout_generation")
        return True

    except Exception as e:
        print(f"Error in rollout generation: {e}")
        return False


def run_stage_2_judge(model_name: str, experiment_config: ExperimentConfig) -> bool:
    """Stage 2.5: Judge rollouts using Llama 8B distill.

    This is a separate stage to avoid GPU OOM when loading both generation
    and judge models simultaneously.

    Args:
        model_name: Model identifier
        experiment_config: Experiment configuration

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print(f"STAGE 2.5: Judging Rollouts ({model_name})")
    print("=" * 60)

    model_config = get_model_config(model_name)

    checkpoint_mgr = CheckpointManager(
        experiment_config.get_base_data_dir(model_name)
    )

    if checkpoint_mgr.is_stage_complete("rollout_judging"):
        print("Rollout judging already complete, skipping...")
        return True

    try:
        judge = RolloutJudge(model_config, experiment_config)
        judged_count = judge.judge_all_samples()
        judge.cleanup()
        clear_gpu_memory()

        print(f"Judged {judged_count} samples")
        checkpoint_mgr.mark_stage_complete("rollout_judging")
        return True

    except Exception as e:
        print(f"Error in rollout judging: {e}")
        return False


def run_stage_3(model_name: str, experiment_config: ExperimentConfig) -> bool:
    """Stage 3: Extract activations.

    Args:
        model_name: Model identifier
        experiment_config: Experiment configuration

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print(f"STAGE 3: Activation Extraction ({model_name})")
    print("=" * 60)

    model_config = get_model_config(model_name)

    checkpoint_mgr = CheckpointManager(
        experiment_config.get_base_data_dir(model_name)
    )

    if checkpoint_mgr.is_stage_complete("activation_extraction"):
        print("Activation extraction already complete, skipping...")
        return True

    try:
        extractor = ActivationExtractor(model_config, experiment_config)
        metadata = extractor.extract_all()
        extractor.cleanup()
        clear_gpu_memory()

        print(f"Extracted activations for {metadata['num_samples']} samples")
        checkpoint_mgr.mark_stage_complete("activation_extraction")
        return True

    except Exception as e:
        print(f"Error in activation extraction: {e}")
        return False


def run_stage_3_5(model_name: str, experiment_config: ExperimentConfig) -> bool:
    """Stage 3.5: Extract trajectory activations for prob_ratio regression.

    Args:
        model_name: Model identifier
        experiment_config: Experiment configuration

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print(f"STAGE 3.5: Trajectory Activation Extraction ({model_name})")
    print("=" * 60)

    model_config = get_model_config(model_name)

    checkpoint_mgr = CheckpointManager(
        experiment_config.get_base_data_dir(model_name)
    )

    if checkpoint_mgr.is_stage_complete("trajectory_extraction"):
        print("Trajectory extraction already complete, skipping...")
        return True

    try:
        extractor = ActivationExtractor(model_config, experiment_config)
        metadata = extractor.extract_trajectory_all()
        extractor.cleanup()
        clear_gpu_memory()

        print(f"Extracted trajectory activations for {metadata['num_samples']} samples ({metadata['num_sentences']} sentences)")
        checkpoint_mgr.mark_stage_complete("trajectory_extraction")
        return True

    except Exception as e:
        print(f"Error in trajectory extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_stage_4(model_name: str, experiment_config: ExperimentConfig) -> bool:
    """Stage 4: Run experiments.

    Args:
        model_name: Model identifier
        experiment_config: Experiment configuration

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print(f"STAGE 4: Running Experiments ({model_name})")
    print("=" * 60)

    model_config = get_model_config(model_name)

    checkpoint_mgr = CheckpointManager(
        experiment_config.get_base_data_dir(model_name)
    )

    if checkpoint_mgr.is_stage_complete("experiments"):
        print("Experiments already complete, skipping...")
        return True

    try:
        runner = ExperimentRunner(model_config, experiment_config)
        summary = runner.run_all()

        print(f"\nExperiment Summary for {model_name}:")
        print(f"  Samples: {summary.n_samples}")
        print(f"  Sycophantic: {summary.n_sycophantic}")
        print(f"  Best pairwise accuracy: {summary.best_pairwise_accuracy:.4f}")
        print(f"  Best emergence accuracy: {summary.best_emergence_accuracy:.4f}")
        print(f"  Best regressor R2: {summary.best_regressor_r2:.4f}")
        if summary.best_prob_ratio_r2 > 0:
            print(f"  Best prob_ratio R2: {summary.best_prob_ratio_r2:.4f}")

        checkpoint_mgr.mark_stage_complete("experiments")
        return True

    except Exception as e:
        print(f"Error in experiments: {e}")
        return False


def run_pipeline_for_model(
    model_name: str,
    experiment_config: ExperimentConfig,
) -> bool:
    """Run the full pipeline for a single model.

    Args:
        model_name: Model identifier
        experiment_config: Experiment configuration

    Returns:
        True if all stages successful
    """
    print("\n" + "#" * 60)
    print(f"# Starting pipeline for: {model_name}")
    print("#" * 60)

    success = True

    if not run_stage_0(experiment_config):
        return False

    if not run_stage_1(model_name, experiment_config):
        success = False
        print(f"Stage 1 failed for {model_name}")

    # Stage 2: Rollout generation
    if success and not run_stage_2(model_name, experiment_config):
        success = False
        print(f"Stage 2 failed for {model_name}")

    # Stage 2.5: Judge rollouts (separate stage for memory efficiency)
    if success and not run_stage_2_judge(model_name, experiment_config):
        success = False
        print(f"Stage 2.5 (judging) failed for {model_name}")

    # Stage 3: Activation extraction
    if success and not run_stage_3(model_name, experiment_config):
        success = False
        print(f"Stage 3 failed for {model_name}")

    # Stage 3.5: Trajectory activation extraction (for prob_ratio regression)
    if success and not run_stage_3_5(model_name, experiment_config):
        # Don't fail pipeline if trajectory extraction fails (it's optional)
        print(f"Stage 3.5 (trajectory) failed for {model_name}, continuing...")

    # Stage 4: Run experiments
    if success and not run_stage_4(model_name, experiment_config):
        success = False
        print(f"Stage 4 failed for {model_name}")

    if success:
        print(f"\n✓ Pipeline completed successfully for {model_name}")
    else:
        print(f"\n✗ Pipeline failed for {model_name}")

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multi-model sycophancy experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to run experiments on",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run on all models in sequence",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["0", "1", "2", "2.5", "3", "3.5", "4"],
        help="Run only a specific stage (0=dataset, 1=base, 2=rollouts, 2.5=judge, 3=activations, 3.5=trajectory, 4=experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="multimodel_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-syco",
        type=int,
        default=101,
        help="Target number of sycophantic samples (default: 101)",
    )
    parser.add_argument(
        "--n-nonsyco",
        type=int,
        default=410,
        help="Target number of non-sycophantic samples (default: 410)",
    )
    parser.add_argument(
        "--rollouts-per-sentence",
        type=int,
        default=20,
        help="Number of rollouts per sentence (default: 20)",
    )
    parser.add_argument(
        "--base-batch-size",
        type=int,
        default=256,
        help="Batch size for base generation (default: 256)",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=128,
        help="Batch size for rollout generation (default: 128)",
    )
    parser.add_argument(
        "--activation-batch-size",
        type=int,
        default=64,
        help="Batch size for activation extraction (default: 64)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="vLLM max concurrent sequences (default: 256, use 512 for H200)",
    )
    parser.add_argument(
        "--no-probes",
        action="store_true",
        help="Skip probe generation during rollouts (faster, but no trajectory data)",
    )
    parser.add_argument(
        "--filter-questions",
        type=str,
        default=None,
        help="Path to file with question IDs to filter (one per line, e.g., arc_conv_0002)",
    )

    args = parser.parse_args()

    if not args.model and not args.all_models:
        parser.error("Must specify --model or --all-models")

    total_target = args.n_syco + args.n_nonsyco
    print(f"Target samples: {args.n_syco} sycophantic + {args.n_nonsyco} non-sycophantic = {total_target} total")

    experiment_config = ExperimentConfig(
        output_base_dir=Path(args.output_dir),
        target_syco_count=args.n_syco,
        target_nonsyco_count=args.n_nonsyco,
        rollouts_per_sentence=args.rollouts_per_sentence,
        base_gen_batch_size=args.base_batch_size,
        rollout_batch_size=args.rollout_batch_size,
        activation_batch_size=args.activation_batch_size,
        max_num_seqs=args.max_num_seqs,
        generate_probes=not args.no_probes,
    )
    experiment_config.output_base_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = MODEL_ORDER if args.all_models else [args.model]
    completed_models = []

    for model_name in models_to_run:
        if args.stage is not None:
            stage_funcs = {
                "0": lambda m: run_stage_0(experiment_config),
                "1": lambda m: run_stage_1(m, experiment_config, args.filter_questions),
                "2": lambda m: run_stage_2(m, experiment_config),
                "2.5": lambda m: run_stage_2_judge(m, experiment_config),
                "3": lambda m: run_stage_3(m, experiment_config),
                "3.5": lambda m: run_stage_3_5(m, experiment_config),
                "4": lambda m: run_stage_4(m, experiment_config),
            }
            success = stage_funcs[args.stage](model_name)
        else:
            success = run_pipeline_for_model(model_name, experiment_config)

        if success:
            completed_models.append(model_name)

        clear_gpu_memory()

    # Generate cross-model summary if multiple models
    if len(completed_models) > 1:
        print("\n" + "=" * 60)
        print("Generating Cross-Model Summary")
        print("=" * 60)
        generate_cross_model_summary(experiment_config, completed_models)

        # Export combined CSV files
        print("\nExporting combined CSV files...")
        output_paths = export_all_results(experiment_config, completed_models)
        for name, path in output_paths.items():
            print(f"  {name}: {path}")

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Completed models: {', '.join(completed_models)}")
    print(f"Results saved to: {experiment_config.output_base_dir}")


if __name__ == "__main__":
    main()
