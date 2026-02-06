"""Causal impact regressor experiment - predicting accuracy_drop from activations."""

import json
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

from ..config.model_registry import ModelConfig
from ..config.experiment_config import ExperimentConfig


RANDOM_SEED = 42
DEFAULT_HIDDEN_DIMS = [256, 64]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RegressorResult:
    model_type: str
    layer: int
    r2_score: float
    r2_std: float
    rmse: float
    rmse_std: float
    n_samples: int


class LinearRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = DEFAULT_HIDDEN_DIMS

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CausalImpactRegressor:

    def __init__(
        self,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
        hidden_dims: List[int] = None,
        n_folds: int = 5,
    ):
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS
        self.n_folds = n_folds

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        act_dir = self.experiment_config.get_activations_dir(self.model_config.name)
        act_path = act_dir / "experiments.safetensors"
        metadata_path = act_dir / "metadata.json"

        tensors = load_file(str(act_path))

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        activations = tensors["activations"].to(DEVICE).float()
        accuracy_drops = tensors["accuracy_drop"].to(DEVICE).float()
        target_layers = metadata["target_layers"]

        return activations, accuracy_drops, target_layers

    def _gpu_standard_scaler(self, X_train: torch.Tensor, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True) + 1e-8
        return (X_train - mean) / std, (X_test - mean) / std

    def _train_linear_lbfgs(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        model = LinearRegressor(X_train.shape[1]).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.LBFGS(model.parameters(), lr=1, max_iter=50, history_size=10)

        model.train()
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            return loss

        for _ in range(10):
            optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            return model(X_test)

    def _train_mlp_lbfgs(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        model = MLPRegressor(X_train.shape[1], self.hidden_dims).to(DEVICE)
        criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            optimizer.step()
        
        optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, history_size=10)
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            return loss

        for _ in range(5):
            optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            return model(X_test)

    def _run_cv_for_layer(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        mean_r2_scores = []
        mean_rmse_scores = []
        linear_r2_scores = []
        linear_rmse_scores = []
        mlp_r2_scores = []
        mlp_rmse_scores = []

        X_cpu = X.cpu().numpy()
        y_cpu = y.cpu().numpy()

        for train_idx, val_idx in kf.split(X_cpu):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            X_train_scaled, X_val_scaled = self._gpu_standard_scaler(X_train, X_val)

            mean_pred = y_train.mean()
            mean_preds = torch.full_like(y_val, mean_pred)
            mean_r2_scores.append(r2_score(y_val.cpu().numpy(), mean_preds.cpu().numpy()))
            mean_rmse_scores.append(np.sqrt(mean_squared_error(y_val.cpu().numpy(), mean_preds.cpu().numpy())))

            linear_preds = self._train_linear_lbfgs(X_train_scaled, y_train, X_val_scaled)
            linear_r2_scores.append(r2_score(y_val.cpu().numpy(), linear_preds.cpu().numpy()))
            linear_rmse_scores.append(np.sqrt(mean_squared_error(y_val.cpu().numpy(), linear_preds.cpu().numpy())))

            mlp_preds = self._train_mlp_lbfgs(X_train_scaled, y_train, X_val_scaled)
            mlp_r2_scores.append(r2_score(y_val.cpu().numpy(), mlp_preds.cpu().numpy()))
            mlp_rmse_scores.append(np.sqrt(mean_squared_error(y_val.cpu().numpy(), mlp_preds.cpu().numpy())))

        return (
            np.mean(mean_r2_scores), np.std(mean_r2_scores),
            np.mean(mean_rmse_scores), np.std(mean_rmse_scores),
            np.mean(linear_r2_scores), np.std(linear_r2_scores),
            np.mean(linear_rmse_scores), np.std(linear_rmse_scores),
            np.mean(mlp_r2_scores), np.std(mlp_r2_scores),
            np.mean(mlp_rmse_scores), np.std(mlp_rmse_scores),
        )

    def run(self) -> List[RegressorResult]:
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)

        print(f"Loading data for {self.model_config.name}...")
        activations, accuracy_drops, target_layers = self._load_data()

        print(f"Dataset statistics:")
        print(f"  Total anchors: {len(accuracy_drops)}")
        print(f"  Activations shape: {activations.shape}")
        print(f"  Target (accuracy_drop) range: [{accuracy_drops.min():.3f}, {accuracy_drops.max():.3f}]")
        print(f"  Target mean: {accuracy_drops.mean():.3f}, std: {accuracy_drops.std():.3f}")

        if len(accuracy_drops) < 20:
            print("Not enough samples for regression")
            return []

        results = []
        n_layers = activations.shape[1]

        for layer_i in tqdm(range(n_layers), desc="Running regressor per layer"):
            layer_idx = target_layers[layer_i]
            
            X = activations[:, layer_i, 0, :]
            y = accuracy_drops

            (mean_r2, mean_r2_std, mean_rmse, mean_rmse_std,
             linear_r2, linear_r2_std, linear_rmse, linear_rmse_std,
             mlp_r2, mlp_r2_std, mlp_rmse, mlp_rmse_std) = self._run_cv_for_layer(X, y)

            results.append(RegressorResult(
                model_type="mean_baseline",
                layer=layer_idx,
                r2_score=float(mean_r2),
                r2_std=float(mean_r2_std),
                rmse=float(mean_rmse),
                rmse_std=float(mean_rmse_std),
                n_samples=len(y),
            ))
            results.append(RegressorResult(
                model_type="linear_regression",
                layer=layer_idx,
                r2_score=float(linear_r2),
                r2_std=float(linear_r2_std),
                rmse=float(linear_rmse),
                rmse_std=float(linear_rmse_std),
                n_samples=len(y),
            ))
            results.append(RegressorResult(
                model_type="mlp",
                layer=layer_idx,
                r2_score=float(mlp_r2),
                r2_std=float(mlp_r2_std),
                rmse=float(mlp_rmse),
                rmse_std=float(mlp_rmse_std),
                n_samples=len(y),
            ))

            print(f"  Layer {layer_idx}: MLP R²={mlp_r2:.4f}±{mlp_r2_std:.4f}, Linear R²={linear_r2:.4f}±{linear_r2_std:.4f}")

        print("\n" + "=" * 70)
        print("SUMMARY BY LAYER")
        print("=" * 70)
        print(f"{'Layer':<10} {'Model':<18} {'R²':>15} {'RMSE':>15}")
        print("-" * 60)
        for r in results:
            if r.model_type == "mlp":
                print(f"{r.layer:<10} {r.model_type:<18} {r.r2_score:>7.4f} ± {r.r2_std:<5.4f} {r.rmse:>7.4f} ± {r.rmse_std:<5.4f}")
        print("-" * 60)

        best_mlp = max([r for r in results if r.model_type == "mlp"], key=lambda x: x.r2_score)
        print(f"\nBest layer: {best_mlp.layer} with R²={best_mlp.r2_score:.4f}")

        return results

    def save_results(self, results: List[RegressorResult]) -> Path:
        output_dir = self.experiment_config.get_results_dir(self.model_config.name)
        output_path = output_dir / "regressor_results.csv"

        with open(output_path, "w") as f:
            f.write("model_name,layer,model_type,r2_score,r2_std,rmse,rmse_std,n_samples\n")
            for r in results:
                f.write(
                    f"{self.model_config.name},{r.layer},{r.model_type},"
                    f"{r.r2_score:.4f},{r.r2_std:.4f},{r.rmse:.4f},{r.rmse_std:.4f},{r.n_samples}\n"
                )

        return output_path
