"""CSV export utilities for experiment results."""

import csv
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config.model_registry import MODEL_ORDER
from ..config.experiment_config import ExperimentConfig


class CSVExporter:
    """Export experiment results to CSV format."""

    def __init__(self, experiment_config: ExperimentConfig):
        """Initialize CSV exporter.

        Args:
            experiment_config: Experiment configuration
        """
        self.experiment_config = experiment_config

    def export_pairwise_combined(
        self,
        model_names: Optional[List[str]] = None,
    ) -> Path:
        """Export combined pairwise results for all models.

        Args:
            model_names: List of model names (defaults to all)

        Returns:
            Path to output file
        """
        model_names = model_names or MODEL_ORDER
        output_path = self.experiment_config.output_base_dir / "combined_pairwise.csv"

        rows = []
        for model_name in model_names:
            results_path = (
                self.experiment_config.get_results_dir(model_name)
                / "pairwise_results.csv"
            )
            if results_path.exists():
                with open(results_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)

        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return output_path

    def export_emergence_combined(
        self,
        model_names: Optional[List[str]] = None,
    ) -> Path:
        """Export combined emergence results for all models.

        Args:
            model_names: List of model names (defaults to all)

        Returns:
            Path to output file
        """
        model_names = model_names or MODEL_ORDER
        output_path = self.experiment_config.output_base_dir / "combined_emergence.csv"

        rows = []
        for model_name in model_names:
            results_path = (
                self.experiment_config.get_results_dir(model_name)
                / "emergence_results.csv"
            )
            if results_path.exists():
                with open(results_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)

        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return output_path

    def export_regressor_combined(
        self,
        model_names: Optional[List[str]] = None,
    ) -> Path:
        """Export combined regressor results for all models.

        Args:
            model_names: List of model names (defaults to all)

        Returns:
            Path to output file
        """
        model_names = model_names or MODEL_ORDER
        output_path = self.experiment_config.output_base_dir / "combined_regressor.csv"

        rows = []
        for model_name in model_names:
            results_path = (
                self.experiment_config.get_results_dir(model_name)
                / "regressor_results.csv"
            )
            if results_path.exists():
                with open(results_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)

        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return output_path

    def export_summary_table(
        self,
        model_names: Optional[List[str]] = None,
    ) -> Path:
        """Export summary table with key metrics per model.

        Args:
            model_names: List of model names (defaults to all)

        Returns:
            Path to output file
        """
        import json

        model_names = model_names or MODEL_ORDER
        output_path = self.experiment_config.output_base_dir / "summary_table.csv"

        rows = []
        for model_name in model_names:
            summary_path = (
                self.experiment_config.get_model_output_dir(model_name)
                / "summary.json"
            )
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    data = json.load(f)
                    summary = data.get("summary", {})
                    rows.append({
                        "model_name": summary.get("model_name", model_name),
                        "n_samples": summary.get("n_samples", 0),
                        "n_sycophantic": summary.get("n_sycophantic", 0),
                        "n_non_sycophantic": summary.get("n_non_sycophantic", 0),
                        "best_pairwise_accuracy": summary.get("best_pairwise_accuracy", 0),
                        "best_emergence_accuracy": summary.get("best_emergence_accuracy", 0),
                        "best_regressor_r2": summary.get("best_regressor_r2", 0),
                    })

        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return output_path


def export_all_results(
    experiment_config: ExperimentConfig,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Export all combined results.

    Args:
        experiment_config: Experiment configuration
        model_names: List of model names (defaults to all)

    Returns:
        Dictionary mapping result type to output path
    """
    exporter = CSVExporter(experiment_config)

    return {
        "pairwise": exporter.export_pairwise_combined(model_names),
        "emergence": exporter.export_emergence_combined(model_names),
        "regressor": exporter.export_regressor_combined(model_names),
        "summary": exporter.export_summary_table(model_names),
    }
