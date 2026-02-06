#!/usr/bin/env python3
"""
Threshold Sweep Analysis for Probe Accuracy.

Runs probe classification at multiple thresholds to demonstrate that
probe accuracy remains high even when including more subtle cases
(lower accuracy_drop thresholds).

This addresses the "cherry-picking" critique by showing accuracy across:
δ = {0.1, 0.2, 0.3, 0.4, 0.5}
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


def gpu_standard_scaler(X_train: torch.Tensor, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def compute_class_weights(y: torch.Tensor) -> torch.Tensor:
    labels, counts = torch.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(labels)
    weights = n_samples / (n_classes * counts.float())
    weight_tensor = torch.ones(2, device=DEVICE)
    for i, label in enumerate(labels):
        weight_tensor[label.long()] = weights[i]
    return weight_tensor


def train_probe(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
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


def balanced_subsample(X: torch.Tensor, y: np.ndarray, rng: np.random.RandomState) -> Tuple[torch.Tensor, np.ndarray]:
    """Downsample the majority class to match the minority class."""
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min_count, replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def run_classification(X: torch.Tensor, y: np.ndarray, n_splits: int = 5, random_seed: int = 42, n_subsamples: int = 10) -> Tuple[float, float]:
    y_cpu = y.astype(int)
    unique, counts = np.unique(y_cpu, return_counts=True)
    if len(unique) < 2 or np.min(counts) < n_splits:
        if len(unique) < 2 or np.min(counts) < 2:
            return 0.5, 0.0
        n_splits = max(2, np.min(counts))

    # Check class imbalance ratio
    imbalance_ratio = counts.max() / counts.min()
    use_subsampling = imbalance_ratio > 1.5

    if use_subsampling:
        # Run multiple balanced subsamples and average
        all_accuracies = []
        for sub_i in range(n_subsamples):
            rng = np.random.RandomState(random_seed + sub_i)
            X_bal, y_bal = balanced_subsample(X, y_cpu, rng)
            n_splits_sub = min(n_splits, np.min(np.unique(y_bal, return_counts=True)[1]))
            if n_splits_sub < 2:
                continue
            cv = StratifiedKFold(n_splits=n_splits_sub, shuffle=True, random_state=random_seed + sub_i)
            for train_idx, test_idx in cv.split(np.zeros(len(y_bal)), y_bal):
                X_train, X_test = X_bal[train_idx], X_bal[test_idx]
                y_train = torch.tensor(y_bal[train_idx], device=DEVICE)
                y_test = y_bal[test_idx]
                X_train, X_test = gpu_standard_scaler(X_train, X_test)
                weights = compute_class_weights(y_train)
                y_pred = train_probe(X_train, y_train, X_test, weights)
                acc = balanced_accuracy_score(y_test, y_pred.cpu().numpy())
                all_accuracies.append(acc)
        if not all_accuracies:
            return 0.5, 0.0
        return float(np.mean(all_accuracies)), float(np.std(all_accuracies))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    accuracies = []

    for train_idx, test_idx in cv.split(np.zeros(len(y_cpu)), y_cpu):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = torch.tensor(y_cpu[train_idx], device=DEVICE)
        y_test = y_cpu[test_idx]

        X_train, X_test = gpu_standard_scaler(X_train, X_test)
        weights = compute_class_weights(y_train)
        y_pred = train_probe(X_train, y_train, X_test, weights)

        acc = balanced_accuracy_score(y_test, y_pred.cpu().numpy())
        accuracies.append(acc)

    return float(np.mean(accuracies)), float(np.std(accuracies))


def load_activations(act_dir: Path) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[int]]:
    act_path = act_dir / "experiments.safetensors"
    metadata_path = act_dir / "metadata.json"

    print(f"Loading activations from: {act_path}")
    tensors = load_file(str(act_path))

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    labels = tensors["is_sycophantic"].numpy()
    accuracy_drops = tensors["accuracy_drop"].numpy()
    activations = tensors["activations"].to(DEVICE).float()
    target_layers = metadata["target_layers"]

    print(f"Loaded {len(labels)} samples, {activations.shape[1]} layers")
    print(f"Labels distribution: sycophantic={labels.sum()}, non-sycophantic={(~labels.astype(bool)).sum()}")

    return activations, labels, accuracy_drops, target_layers


def run_threshold_sweep(
    activations: torch.Tensor,
    labels: np.ndarray,
    accuracy_drops: np.ndarray,
    target_layers: List[int],
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
) -> Dict[float, Dict]:
    """Run probe classification at multiple thresholds."""

    results = {}
    n_layers = activations.shape[1]

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Running at threshold δ = {threshold}")
        print(f"{'='*60}")

        high_drop_mask = accuracy_drops >= threshold
        is_syco_anchor = labels.astype(bool) & high_drop_mask
        is_correct_anchor = (~labels.astype(bool)) & high_drop_mask

        n_syco = is_syco_anchor.sum()
        n_correct = is_correct_anchor.sum()
        n_total = n_syco + n_correct
        pct_included = n_total / len(labels) * 100

        print(f"Anchors: syco={n_syco}, correct={n_correct}, total={n_total} ({pct_included:.1f}% of all samples)")

        if n_syco < 5 or n_correct < 5:
            print(f"Skipping threshold {threshold}: not enough samples")
            continue

        layer_results = []
        best_acc = 0
        best_layer = None

        for layer_i in range(n_layers):
            layer_idx = target_layers[layer_i]
            X = activations[:, layer_i, 0, :]  # [samples, hidden_dim]

            mask = is_syco_anchor | is_correct_anchor
            X_pair = X[mask]
            y_pair = is_syco_anchor[mask].astype(int)

            if len(np.unique(y_pair)) < 2:
                continue

            mean_acc, std_acc = run_classification(X_pair, y_pair)
            layer_results.append({
                "layer": layer_idx,
                "accuracy": mean_acc,
                "std": std_acc,
            })

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_layer = layer_idx

        # Compute mean across all layers
        mean_across_layers = np.mean([r["accuracy"] for r in layer_results])
        std_across_layers = np.std([r["accuracy"] for r in layer_results])

        results[threshold] = {
            "n_syco_anchors": int(n_syco),
            "n_correct_anchors": int(n_correct),
            "n_total": int(n_total),
            "pct_included": pct_included,
            "best_accuracy": best_acc,
            "best_layer": best_layer,
            "mean_accuracy": mean_across_layers,
            "std_accuracy": std_across_layers,
            "layer_results": layer_results,
        }

        print(f"Best layer {best_layer}: accuracy = {best_acc:.3f}")
        print(f"Mean across layers: {mean_across_layers:.3f} ± {std_across_layers:.3f}")

    return results


def plot_threshold_sweep(results: Dict[float, Dict], output_path: Path, model_name: str):
    """Generate the threshold vs accuracy plot."""

    thresholds = sorted(results.keys())
    best_accs = [results[t]["best_accuracy"] for t in thresholds]
    mean_accs = [results[t]["mean_accuracy"] for t in thresholds]
    std_accs = [results[t]["std_accuracy"] for t in thresholds]
    n_samples = [results[t]["n_total"] for t in thresholds]
    pct_included = [results[t]["pct_included"] for t in thresholds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy vs Threshold
    ax1.errorbar(thresholds, mean_accs, yerr=std_accs,
                 fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=10,
                 color='#2E86AB', label='Mean across layers')
    ax1.plot(thresholds, best_accs, 's--', linewidth=2, markersize=10,
             color='#A23B72', label='Best layer')

    ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Random baseline (50%)')
    ax1.axhline(y=0.7, color='green', linestyle=':', linewidth=2, alpha=0.7, label='70% threshold')

    ax1.set_xlabel('Importance Threshold (δ)', fontsize=14)
    ax1.set_ylabel('Probe Accuracy (Balanced)', fontsize=14)
    ax1.set_title(f'Probe Accuracy vs Threshold\n{model_name}', fontsize=16)
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xticks(thresholds)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add sample counts as annotations
    for i, (t, n, pct) in enumerate(zip(thresholds, n_samples, pct_included)):
        ax1.annotate(f'n={n}\n({pct:.0f}%)',
                    xy=(t, mean_accs[i]),
                    xytext=(0, 20),
                    textcoords='offset points',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Plot 2: Sample size vs Threshold
    ax2.bar(thresholds, n_samples, width=0.08, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Importance Threshold (δ)', fontsize=14)
    ax2.set_ylabel('Number of Anchor Samples', fontsize=14)
    ax2.set_title('Sample Size at Each Threshold', fontsize=16)
    ax2.set_xticks(thresholds)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (t, n, pct) in enumerate(zip(thresholds, n_samples, pct_included)):
        ax2.annotate(f'{pct:.1f}%',
                    xy=(t, n),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def print_summary_table(results: Dict[float, Dict]):
    """Print a formatted summary table."""

    print("\n" + "="*80)
    print("SUMMARY: Threshold Sweep Results")
    print("="*80)
    print(f"{'Threshold':<12} {'N Samples':<12} {'% Data':<10} {'Best Acc':<12} {'Mean Acc':<12} {'Best Layer':<12}")
    print("-"*80)

    for threshold in sorted(results.keys()):
        r = results[threshold]
        print(f"{threshold:<12.2f} {r['n_total']:<12} {r['pct_included']:<10.1f} "
              f"{r['best_accuracy']:<12.3f} {r['mean_accuracy']:<12.3f} {r['best_layer']:<12}")

    print("="*80)

    # Key finding
    min_threshold = min(results.keys())
    min_result = results[min_threshold]
    print(f"\n** KEY FINDING **")
    print(f"At δ={min_threshold} (including {min_result['pct_included']:.1f}% of samples),")
    print(f"probe accuracy = {min_result['best_accuracy']:.1%}")

    if min_result['best_accuracy'] >= 0.7:
        print(f"→ Accuracy remains above 70% even with subtle cases!")
        print(f"→ This REFUTES the 'cherry-picking' critique.")
    else:
        print(f"→ Accuracy drops below 70% for subtle cases.")


def main():
    # Default to qwen-7b (largest dataset)
    base_dir = Path("/home/jacekduszenko/Workspace/keyword_probe_experiment/thought-anchors")

    # Available models with activations
    available_models = {
        "deepseek-qwen-7b": base_dir / "multimodel_seven/multimodel_prod/deepseek-qwen-7b/activations",
        "deepseek-qwen-1.5b": base_dir / "qwen_multimodel/deepseek-qwen-1.5b/activations",
        "deepseek-llama-8b": base_dir / "llama_multimodel/deepseek-llama-8b/activations",
        "falcon-h1r-7b": base_dir / "falcon/falcon-h1r-7b/activations",
    }

    # Check command line args for model selection
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "deepseek-qwen-7b"  # Default

    if model_name not in available_models:
        print(f"Available models: {list(available_models.keys())}")
        sys.exit(1)

    act_dir = available_models[model_name]
    print(f"Using model: {model_name}")
    print(f"Activations dir: {act_dir}")

    # Load data
    activations, labels, accuracy_drops, target_layers = load_activations(act_dir)

    # Run threshold sweep
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = run_threshold_sweep(activations, labels, accuracy_drops, target_layers, thresholds)

    # Save results
    output_dir = base_dir / "analysis_results"
    output_dir.mkdir(exist_ok=True)

    # Save JSON results
    json_path = output_dir / f"threshold_sweep_{model_name}.json"
    with open(json_path, "w") as f:
        # Convert for JSON serialization
        json_results = {}
        for k, v in results.items():
            json_results[str(k)] = v
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate plot
    plot_path = output_dir / f"threshold_sweep_{model_name}.png"
    plot_threshold_sweep(results, plot_path, model_name)

    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
