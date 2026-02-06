"""Sycophancy emergence experiment - token trajectory divergence analysis."""

import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from ..config.model_registry import ModelConfig
from ..config.experiment_config import ExperimentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EmergenceResult:
    layer: int
    position: int
    position_label: str
    accuracy: float
    n_samples: int


@dataclass
class DivergencePoint:
    layer: int
    tokens_before_anchor: int
    accuracy_at_divergence: float
    anchor_accuracy: float


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


class SycophancyEmergence:

    DIVERGENCE_THRESHOLD = 0.6

    def __init__(
        self,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
    ):
        self.model_config = model_config
        self.experiment_config = experiment_config

    def _load_activations(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, int, List[int]]:
        act_dir = self.experiment_config.get_activations_dir(self.model_config.name)
        act_path = act_dir / "emergence.safetensors"
        metadata_path = act_dir / "metadata.json"

        tensors = load_file(str(act_path))

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        labels = tensors["is_sycophantic"].numpy()
        accuracy_drops = tensors["accuracy_drop"].numpy()
        activations = tensors["activations"].to(DEVICE).float()
        
        window_size = metadata["window_size"]
        target_layers = metadata["target_layers"]

        return activations, labels, accuracy_drops, window_size, target_layers

    def _get_position_labels(self, window_size: int) -> List[str]:
        labels = []
        for i in range(window_size):
            offset = window_size - 1 - i
            if offset > 0:
                labels.append(f"anchor-{offset}")
            else:
                labels.append("anchor_end")
        return labels

    def _gpu_standard_scaler(self, X_train: torch.Tensor, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True) + 1e-8
        return (X_train - mean) / std, (X_test - mean) / std

    def _compute_class_weights(self, y: torch.Tensor) -> torch.Tensor:
        labels, counts = torch.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(labels)
        weights = n_samples / (n_classes * counts.float())
        weight_tensor = torch.ones(2, device=DEVICE)
        for i, label in enumerate(labels):
            weight_tensor[label.long()] = weights[i]
        return weight_tensor

    def _train_probe(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
        model = LinearProbe(X_train.shape[1]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=10)

        model.train()
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            return loss

        for _ in range(5):
            optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            return torch.argmax(model(X_test), dim=1)

    def _train_probe_cv(
        self,
        X: torch.Tensor,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> Tuple[float, float]:
        y_cpu = y.astype(int)
        unique, counts = np.unique(y_cpu, return_counts=True)
        if len(unique) < 2 or np.min(counts) < 2:
            return 0.5, 0.0
        if np.min(counts) < n_splits:
            n_splits = max(2, np.min(counts))

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.experiment_config.random_seed)
        
        accuracies = []
        for train_idx, test_idx in cv.split(np.zeros(len(y_cpu)), y_cpu):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = torch.tensor(y_cpu[train_idx], device=DEVICE)
            y_test = y_cpu[test_idx]

            X_train, X_test = self._gpu_standard_scaler(X_train, X_test)
            weights = self._compute_class_weights(y_train)
            y_pred = self._train_probe(X_train, y_train, X_test, weights)

            acc = balanced_accuracy_score(y_test, y_pred.cpu().numpy())
            accuracies.append(acc)

        return float(np.mean(accuracies)), float(np.std(accuracies))

    def _find_divergence_point(
        self,
        accuracies: List[float],
        threshold: float = None,
    ) -> Optional[int]:
        if threshold is None:
            threshold = self.DIVERGENCE_THRESHOLD
            
        for i, acc in enumerate(accuracies):
            if acc >= threshold:
                return i
        return None

    def run(self) -> Tuple[List[EmergenceResult], List[DivergencePoint]]:
        activations, labels, accuracy_drops, window_size, target_layers = self._load_activations()
        position_labels = self._get_position_labels(window_size)

        n_anchors, n_layers, n_positions, hidden_size = activations.shape

        print(f"Emergence analysis: {n_anchors} anchors, {n_layers} layers, {n_positions} positions")

        results = []
        divergence_points = []

        total_probes = n_layers * n_positions
        pbar = tqdm(total=total_probes, desc="Training emergence probes")
        
        for layer_i in range(n_layers):
            layer_idx = target_layers[layer_i]
            layer_accuracies = []

            for pos in range(n_positions):
                X = activations[:, layer_i, pos, :]

                accuracy, std = self._train_probe_cv(X, labels)
                layer_accuracies.append(accuracy)

                pos_label = position_labels[pos] if pos < len(position_labels) else f"pos_{pos}"
                
                results.append(
                    EmergenceResult(
                        layer=layer_idx,
                        position=pos,
                        position_label=pos_label,
                        accuracy=accuracy,
                        n_samples=n_anchors,
                    )
                )
                pbar.update(1)

            div_idx = self._find_divergence_point(layer_accuracies)
            if div_idx is not None:
                tokens_before = window_size - 1 - div_idx
                divergence_points.append(
                    DivergencePoint(
                        layer=layer_idx,
                        tokens_before_anchor=tokens_before,
                        accuracy_at_divergence=layer_accuracies[div_idx],
                        anchor_accuracy=layer_accuracies[-1] if layer_accuracies else 0.5,
                    )
                )

        pbar.close()
        return results, divergence_points

    def save_results(self, results: List[EmergenceResult], divergence_points: List[DivergencePoint] = None) -> Tuple[Path, Optional[Path]]:
        output_dir = self.experiment_config.get_results_dir(self.model_config.name)
        emergence_path = output_dir / "emergence_results.csv"

        with open(emergence_path, "w") as f:
            f.write("model_name,layer,position,position_label,accuracy,n_samples\n")
            for r in results:
                f.write(
                    f"{self.model_config.name},{r.layer},{r.position},"
                    f"{r.position_label},{r.accuracy:.4f},{r.n_samples}\n"
                )

        divergence_path = None
        if divergence_points:
            divergence_path = output_dir / "divergence_analysis.csv"
            with open(divergence_path, "w") as f:
                f.write("model_name,layer,tokens_before_anchor,accuracy_at_divergence,anchor_accuracy\n")
                for dp in divergence_points:
                    f.write(
                        f"{self.model_config.name},{dp.layer},{dp.tokens_before_anchor},"
                        f"{dp.accuracy_at_divergence:.4f},{dp.anchor_accuracy:.4f}\n"
                    )

        return emergence_path, divergence_path
