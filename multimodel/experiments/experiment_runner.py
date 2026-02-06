"""Orchestrate all experiments for a model."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from ..config.model_registry import ModelConfig
from ..config.experiment_config import ExperimentConfig
from .pairwise_discriminability import PairwiseDiscriminability
from .sycophancy_emergence import SycophancyEmergence
from .causal_impact_regressor import CausalImpactRegressor
from .prob_ratio_regressor import ProbRatioRegressor


@dataclass
class ExperimentSummary:
    """Summary of all experiments for a model."""
    model_name: str
    n_samples: int
    n_sycophantic: int
    n_non_sycophantic: int
    best_pairwise_accuracy: float
    best_emergence_accuracy: float
    best_regressor_r2: float
    best_prob_ratio_r2: float = 0.0  # New field for prob_ratio regressor


class ExperimentRunner:
    """Run all experiments for a model."""

    def __init__(
        self,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
    ):
        """Initialize experiment runner.

        Args:
            model_config: Model configuration
            experiment_config: Experiment configuration
        """
        self.model_config = model_config
        self.experiment_config = experiment_config

    def _load_metadata(self) -> Dict[str, Any]:
        """Load activation metadata."""
        metadata_path = (
            self.experiment_config.get_activations_dir(self.model_config.name)
            / "metadata.json"
        )
        with open(metadata_path, "r") as f:
            return json.load(f)

    def run_pairwise(self) -> List[Dict[str, Any]]:
        """Run pairwise discriminability experiment.

        Returns:
            List of result dictionaries
        """
        print("\n=== Running Pairwise Discriminability ===")
        exp = PairwiseDiscriminability(self.model_config, self.experiment_config)
        results = exp.run()
        output_path = exp.save_results(results)
        print(f"Saved to {output_path}")

        return [asdict(r) for r in results]

    def run_emergence(self) -> List[Dict[str, Any]]:
        """Run sycophancy emergence experiment with divergence analysis.

        Returns:
            List of result dictionaries
        """
        print("\n=== Running Sycophancy Emergence ===")
        exp = SycophancyEmergence(self.model_config, self.experiment_config)
        results, divergence_points = exp.run()
        emergence_path, divergence_path = exp.save_results(results, divergence_points)
        print(f"Saved emergence to {emergence_path}")
        if divergence_path:
            print(f"Saved divergence analysis to {divergence_path}")
            for dp in divergence_points:
                print(f"  Layer {dp.layer}: divergence at {dp.tokens_before_anchor} tokens before anchor")

        return [asdict(r) for r in results]

    def run_regressor(self) -> List[Dict[str, Any]]:
        """Run causal impact regressor experiment.

        Returns:
            List of result dictionaries
        """
        print("\n=== Running Causal Impact Regressor ===")
        exp = CausalImpactRegressor(self.model_config, self.experiment_config)
        results = exp.run()
        output_path = exp.save_results(results)
        print(f"Saved to {output_path}")

        return [asdict(r) for r in results]

    def run_prob_ratio_regressor(self) -> List[Dict[str, Any]]:
        """Run prob_ratio regressor experiment (predicting model's belief from activations).

        Returns:
            List of result dictionaries, or empty list if trajectory data not available
        """
        print("\n=== Running Prob Ratio Regressor ===")
        
        # Check if trajectory data exists
        act_dir = self.experiment_config.get_activations_dir(self.model_config.name)
        if not (act_dir / "trajectory.safetensors").exists():
            print("Trajectory activations not found. Skipping prob_ratio regressor.")
            print("Run with --extract-trajectory to generate trajectory activations.")
            return []

        exp = ProbRatioRegressor(self.model_config, self.experiment_config)
        results = exp.run()
        output_path = exp.save_results(results)
        print(f"Saved to {output_path}")

        return [asdict(r) for r in results]

    def run_all(self) -> ExperimentSummary:
        """Run all experiments and generate summary.

        Returns:
            ExperimentSummary
        """
        metadata = self._load_metadata()

        # Run all experiments
        pairwise_results = self.run_pairwise()
        emergence_results = self.run_emergence()
        regressor_results = self.run_regressor()
        prob_ratio_results = self.run_prob_ratio_regressor()

        # Extract best metrics
        best_pairwise = max(
            (r["mean_accuracy"] for r in pairwise_results),
            default=0.5,
        )
        best_emergence = max(
            (r["accuracy"] for r in emergence_results),
            default=0.5,
        )
        best_regressor = max(
            (r["r2_score"] for r in regressor_results if r.get("model_type") == "mlp"),
            default=0.0,
        )
        best_prob_ratio = max(
            (r["r2_score"] for r in prob_ratio_results if r.get("model_type") == "mlp"),
            default=0.0,
        )

        n_anchors = metadata.get("num_anchors", metadata.get("num_samples", 0))
        n_samples = metadata.get("num_samples", 0)
        n_syco = metadata.get("num_sycophantic_samples", metadata.get("num_sycophantic", 0))
        
        summary = ExperimentSummary(
            model_name=self.model_config.name,
            n_samples=n_anchors,
            n_sycophantic=n_syco,
            n_non_sycophantic=n_anchors - n_syco,
            best_pairwise_accuracy=best_pairwise,
            best_emergence_accuracy=best_emergence,
            best_regressor_r2=best_regressor,
            best_prob_ratio_r2=best_prob_ratio,
        )

        # Save summary
        self._save_summary(summary, pairwise_results, emergence_results, regressor_results, prob_ratio_results)

        return summary

    def _save_summary(
        self,
        summary: ExperimentSummary,
        pairwise: List[Dict[str, Any]],
        emergence: List[Dict[str, Any]],
        regressor: List[Dict[str, Any]],
        prob_ratio: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Save experiment summary to JSON.

        Args:
            summary: Overall summary
            pairwise: Pairwise results
            emergence: Emergence results
            regressor: Regressor results
            prob_ratio: Prob ratio regressor results (optional)
        """
        output_dir = self.experiment_config.get_model_output_dir(self.model_config.name)
        output_path = output_dir / "summary.json"

        data = {
            "summary": asdict(summary),
            "pairwise_results": pairwise,
            "emergence_results": emergence,
            "regressor_results": regressor,
        }
        
        if prob_ratio:
            data["prob_ratio_results"] = prob_ratio

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved summary to {output_path}")


def generate_cross_model_summary(
    experiment_config: ExperimentConfig,
    model_names: List[str],
) -> Dict[str, Any]:
    """Generate cross-model summary.

    Args:
        experiment_config: Experiment configuration
        model_names: List of model names to include

    Returns:
        Cross-model summary dictionary
    """
    summaries = []

    for model_name in model_names:
        summary_path = (
            experiment_config.get_model_output_dir(model_name) / "summary.json"
        )
        if summary_path.exists():
            with open(summary_path, "r") as f:
                data = json.load(f)
                summaries.append(data["summary"])

    cross_summary = {
        "models": summaries,
        "comparison": {
            "best_pairwise_model": max(
                summaries, key=lambda x: x["best_pairwise_accuracy"]
            )["model_name"] if summaries else None,
            "best_emergence_model": max(
                summaries, key=lambda x: x["best_emergence_accuracy"]
            )["model_name"] if summaries else None,
            "best_regressor_model": max(
                summaries, key=lambda x: x["best_regressor_r2"]
            )["model_name"] if summaries else None,
        },
    }

    # Save cross-model summary
    output_path = experiment_config.output_base_dir / "all_models_summary.json"
    with open(output_path, "w") as f:
        json.dump(cross_summary, f, indent=2)

    print(f"\nSaved cross-model summary to {output_path}")

    return cross_summary
