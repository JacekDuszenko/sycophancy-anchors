"""Pairwise discriminability experiment."""

import json
from pathlib import Path
from typing import List, Tuple
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
class PairwiseResult:
    pair_name: str
    layer: int
    mean_accuracy: float
    std_accuracy: float
    n_runs: int
    n_class_0: int
    n_class_1: int


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


class PairwiseDiscriminability:

    def __init__(
        self,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
    ):
        self.model_config = model_config
        self.experiment_config = experiment_config

    def _load_activations(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[int]]:
        act_dir = self.experiment_config.get_activations_dir(self.model_config.name)
        act_path = act_dir / "experiments.safetensors"
        metadata_path = act_dir / "metadata.json"

        tensors = load_file(str(act_path))
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        labels = tensors["is_sycophantic"].numpy()
        accuracy_drops = tensors["accuracy_drop"].numpy()
        activations = tensors["activations"].to(DEVICE).float()
        target_layers = metadata["target_layers"]

        return activations, labels, accuracy_drops, target_layers

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

    def _run_classification(
        self,
        X: torch.Tensor,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> Tuple[float, float]:
        y_cpu = y.astype(int)
        unique, counts = np.unique(y_cpu, return_counts=True)
        if len(unique) < 2 or np.min(counts) < n_splits:
            if len(unique) < 2 or np.min(counts) < 2:
                return 0.5, 0.0
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

    def run(self) -> List[PairwiseResult]:
        activations, labels, accuracy_drops, target_layers = self._load_activations()

        results = []
        n_layers = activations.shape[1]

        high_drop_mask = accuracy_drops >= 0.3
        is_syco_anchor = labels & high_drop_mask
        is_correct_anchor = (~labels) & high_drop_mask
        is_neutral = ~(is_syco_anchor | is_correct_anchor)

        print(f"Anchor counts: syco={is_syco_anchor.sum()}, correct={is_correct_anchor.sum()}, neutral={is_neutral.sum()}")

        for layer_i in tqdm(range(n_layers), desc="Pairwise probes"):
            layer_idx = target_layers[layer_i]
            X = activations[:, layer_i, 0, :]

            mask = is_syco_anchor | is_correct_anchor
            if mask.sum() >= 10:
                X_pair = X[mask]
                y_pair = is_syco_anchor[mask].astype(int)

                if len(np.unique(y_pair)) == 2:
                    mean_acc, std_acc = self._run_classification(X_pair, y_pair)
                    results.append(
                        PairwiseResult(
                            pair_name="syco_vs_correct",
                            layer=layer_idx,
                            mean_accuracy=mean_acc,
                            std_accuracy=std_acc,
                            n_runs=5,
                            n_class_0=int((y_pair == 0).sum()),
                            n_class_1=int((y_pair == 1).sum()),
                        )
                    )

            mask = is_syco_anchor | is_neutral
            if mask.sum() >= 10:
                X_pair = X[mask]
                y_pair = is_syco_anchor[mask].astype(int)

                if len(np.unique(y_pair)) == 2:
                    mean_acc, std_acc = self._run_classification(X_pair, y_pair)
                    results.append(
                        PairwiseResult(
                            pair_name="syco_vs_neutral",
                            layer=layer_idx,
                            mean_accuracy=mean_acc,
                            std_accuracy=std_acc,
                            n_runs=5,
                            n_class_0=int((y_pair == 0).sum()),
                            n_class_1=int((y_pair == 1).sum()),
                        )
                    )

            mask = is_correct_anchor | is_neutral
            if mask.sum() >= 10:
                X_pair = X[mask]
                y_pair = is_correct_anchor[mask].astype(int)

                if len(np.unique(y_pair)) == 2:
                    mean_acc, std_acc = self._run_classification(X_pair, y_pair)
                    results.append(
                        PairwiseResult(
                            pair_name="correct_vs_neutral",
                            layer=layer_idx,
                            mean_accuracy=mean_acc,
                            std_accuracy=std_acc,
                            n_runs=5,
                            n_class_0=int((y_pair == 0).sum()),
                            n_class_1=int((y_pair == 1).sum()),
                        )
                    )

        return results

    def save_results(self, results: List[PairwiseResult]) -> Path:
        output_dir = self.experiment_config.get_results_dir(self.model_config.name)
        output_path = output_dir / "pairwise_results.csv"

        with open(output_path, "w") as f:
            f.write("model_name,layer,pair_name,mean_accuracy,std_accuracy,n_runs,n_class_0,n_class_1\n")
            for r in results:
                f.write(
                    f"{self.model_config.name},{r.layer},{r.pair_name},"
                    f"{r.mean_accuracy:.4f},{r.std_accuracy:.4f},{r.n_runs},{r.n_class_0},{r.n_class_1}\n"
                )

        return output_path
