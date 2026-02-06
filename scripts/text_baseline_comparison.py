#!/usr/bin/env python3
"""
Text Baseline Comparison: Activation Probe vs Text-Only Features.

Addresses the critique: "Your probe just detects the word 'user' in sentences."

We compare:
1. Activation Probe (linear probe on hidden states)
2. Bag-of-Words Logistic Regression (TF-IDF on sentence text)
3. Simple "contains 'user'" heuristic
4. N-gram features (unigrams + bigrams)

If Activation Probe >> Text-Only, we prove we're detecting internal state,
not surface-level vocabulary.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def balanced_subsample(X, y: np.ndarray, rng: np.random.RandomState):
    """Downsample the majority class to match the minority class.
    X can be a torch.Tensor, numpy array, or list."""
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min_count, replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    rng.shuffle(indices)

    if isinstance(X, list):
        return [X[i] for i in indices], y[indices]
    return X[indices], y[indices]


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


def train_activation_probe(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
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


def run_activation_probe(X: torch.Tensor, y: np.ndarray, n_splits: int = 5, random_seed: int = 42, n_subsamples: int = 10) -> Tuple[float, float]:
    """Run activation probe with cross-validation and balanced subsampling."""
    y_cpu = y.astype(int)
    unique, counts = np.unique(y_cpu, return_counts=True)
    if len(unique) < 2 or np.min(counts) < n_splits:
        return 0.5, 0.0

    imbalance_ratio = counts.max() / counts.min()
    use_subsampling = imbalance_ratio > 1.5

    if use_subsampling:
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
                y_train_t = torch.tensor(y_bal[train_idx], device=DEVICE)
                y_test = y_bal[test_idx]
                X_train, X_test = gpu_standard_scaler(X_train, X_test)
                weights = compute_class_weights(y_train_t)
                y_pred = train_activation_probe(X_train, y_train_t, X_test, weights)
                acc = balanced_accuracy_score(y_test, y_pred.cpu().numpy())
                all_accuracies.append(acc)
        if not all_accuracies:
            return 0.5, 0.0
        return float(np.mean(all_accuracies)), float(np.std(all_accuracies))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    accuracies = []

    for train_idx, test_idx in cv.split(np.zeros(len(y_cpu)), y_cpu):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_t = torch.tensor(y_cpu[train_idx], device=DEVICE)
        y_test = y_cpu[test_idx]

        X_train, X_test = gpu_standard_scaler(X_train, X_test)
        weights = compute_class_weights(y_train_t)
        y_pred = train_activation_probe(X_train, y_train_t, X_test, weights)

        acc = balanced_accuracy_score(y_test, y_pred.cpu().numpy())
        accuracies.append(acc)

    return float(np.mean(accuracies)), float(np.std(accuracies))


def run_text_baseline(texts: List[str], y: np.ndarray, method: str = "tfidf", n_splits: int = 5, random_seed: int = 42, n_subsamples: int = 10) -> Tuple[float, float]:
    """Run text-based classification with cross-validation and balanced subsampling."""
    y_cpu = y.astype(int)
    unique, counts = np.unique(y_cpu, return_counts=True)
    if len(unique) < 2 or np.min(counts) < n_splits:
        return 0.5, 0.0

    imbalance_ratio = counts.max() / counts.min()
    use_subsampling = imbalance_ratio > 1.5

    if use_subsampling:
        all_accuracies = []
        for sub_i in range(n_subsamples):
            rng = np.random.RandomState(random_seed + sub_i)
            texts_bal, y_bal = balanced_subsample(texts, y_cpu, rng)
            n_splits_sub = min(n_splits, np.min(np.unique(y_bal, return_counts=True)[1]))
            if n_splits_sub < 2:
                continue
            cv = StratifiedKFold(n_splits=n_splits_sub, shuffle=True, random_state=random_seed + sub_i)
            for train_idx, test_idx in cv.split(np.zeros(len(y_bal)), y_bal):
                texts_train = [texts_bal[i] for i in train_idx]
                texts_test = [texts_bal[i] for i in test_idx]
                y_train = y_bal[train_idx]
                y_test = y_bal[test_idx]

                if method == "tfidf":
                    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                elif method == "tfidf_ngram":
                    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
                elif method == "tfidf_full":
                    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
                else:
                    vectorizer = TfidfVectorizer(max_features=5000)

                X_train = vectorizer.fit_transform(texts_train)
                X_test = vectorizer.transform(texts_test)

                clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_seed + sub_i)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                acc = balanced_accuracy_score(y_test, y_pred)
                all_accuracies.append(acc)
        if not all_accuracies:
            return 0.5, 0.0
        return float(np.mean(all_accuracies)), float(np.std(all_accuracies))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    accuracies = []

    for train_idx, test_idx in cv.split(np.zeros(len(y_cpu)), y_cpu):
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]
        y_train = y_cpu[train_idx]
        y_test = y_cpu[test_idx]

        if method == "tfidf":
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        elif method == "tfidf_ngram":
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        elif method == "tfidf_full":
            vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        else:
            vectorizer = TfidfVectorizer(max_features=5000)

        X_train = vectorizer.fit_transform(texts_train)
        X_test = vectorizer.transform(texts_test)

        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = balanced_accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    return float(np.mean(accuracies)), float(np.std(accuracies))


def run_keyword_heuristic(texts: List[str], y: np.ndarray, keywords: List[str]) -> Tuple[float, float]:
    """Run simple keyword presence heuristic."""
    y_cpu = y.astype(int)

    # Predict sycophantic (1) if any keyword is present
    y_pred = np.array([
        1 if any(kw.lower() in text.lower() for kw in keywords) else 0
        for text in texts
    ])

    acc = balanced_accuracy_score(y_cpu, y_pred)

    # Also compute recall for each class
    tp = ((y_pred == 1) & (y_cpu == 1)).sum()
    fn = ((y_pred == 0) & (y_cpu == 1)).sum()
    tn = ((y_pred == 0) & (y_cpu == 0)).sum()
    fp = ((y_pred == 1) & (y_cpu == 0)).sum()

    recall_syco = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_correct = tn / (tn + fp) if (tn + fp) > 0 else 0

    return float(acc), 0.0  # No std for heuristic


def analyze_keyword_distribution(texts: List[str], labels: np.ndarray, keywords: List[str]):
    """Analyze how often keywords appear in each class."""
    syco_texts = [t for t, l in zip(texts, labels) if l]
    correct_texts = [t for t, l in zip(texts, labels) if not l]

    print("\n" + "="*60)
    print("KEYWORD DISTRIBUTION ANALYSIS")
    print("="*60)

    for kw in keywords:
        syco_count = sum(1 for t in syco_texts if kw.lower() in t.lower())
        correct_count = sum(1 for t in correct_texts if kw.lower() in t.lower())

        syco_pct = syco_count / len(syco_texts) * 100 if syco_texts else 0
        correct_pct = correct_count / len(correct_texts) * 100 if correct_texts else 0
        ratio = syco_pct / correct_pct if correct_pct > 0 else float('inf')

        print(f"'{kw}': syco={syco_pct:.1f}%, correct={correct_pct:.1f}%, ratio={ratio:.1f}x")


def load_data(act_dir: Path) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[str], List[int]]:
    """Load activations and metadata."""
    act_path = act_dir / "experiments.safetensors"
    metadata_path = act_dir / "metadata.json"

    print(f"Loading from: {act_dir}")
    tensors = load_file(str(act_path))

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    labels = tensors["is_sycophantic"].numpy()
    accuracy_drops = tensors["accuracy_drop"].numpy()
    activations = tensors["activations"].to(DEVICE).float()
    target_layers = metadata["target_layers"]

    # Extract sentence texts
    texts = [anchor["sentence_text"] for anchor in metadata["anchors"]]

    print(f"Loaded {len(labels)} samples")
    print(f"Labels: syco={labels.sum()}, correct={(~labels.astype(bool)).sum()}")

    return activations, labels, accuracy_drops, texts, target_layers


def run_comparison(
    activations: torch.Tensor,
    labels: np.ndarray,
    accuracy_drops: np.ndarray,
    texts: List[str],
    target_layers: List[int],
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
) -> Dict[float, Dict]:
    """Run full comparison at each threshold."""

    results = {}

    for threshold in thresholds:
        print(f"\n{'='*70}")
        print(f"THRESHOLD δ = {threshold}")
        print(f"{'='*70}")

        # Filter to anchors above threshold
        high_drop_mask = accuracy_drops >= threshold
        is_syco_anchor = labels.astype(bool) & high_drop_mask
        is_correct_anchor = (~labels.astype(bool)) & high_drop_mask
        mask = is_syco_anchor | is_correct_anchor

        n_syco = is_syco_anchor.sum()
        n_correct = is_correct_anchor.sum()
        n_total = mask.sum()
        pct_data = n_total / len(labels) * 100

        print(f"Samples: syco={n_syco}, correct={n_correct}, total={n_total} ({pct_data:.1f}%)")

        if n_syco < 10 or n_correct < 10:
            print("Skipping: not enough samples")
            continue

        # Prepare data
        y = is_syco_anchor[mask].astype(int)
        filtered_texts = [texts[i] for i in range(len(texts)) if mask[i]]

        # Analyze keyword distribution
        analyze_keyword_distribution(filtered_texts, y, ["user", "they", "question", "answer"])

        # 1. Activation Probe (best layer)
        print("\n--- Activation Probe ---")
        best_act_acc = 0
        best_layer = None
        for layer_i in range(activations.shape[1]):
            X = activations[:, layer_i, 0, :][mask]
            acc, std = run_activation_probe(X, y)
            if acc > best_act_acc:
                best_act_acc = acc
                best_layer = target_layers[layer_i]
        print(f"Best layer {best_layer}: {best_act_acc:.1%}")

        # 2. TF-IDF Bag of Words
        print("\n--- TF-IDF (Unigram) ---")
        tfidf_acc, tfidf_std = run_text_baseline(filtered_texts, y, method="tfidf")
        print(f"Accuracy: {tfidf_acc:.1%} (±{tfidf_std:.1%})")

        # 3. TF-IDF with N-grams
        print("\n--- TF-IDF (Unigram + Bigram) ---")
        ngram_acc, ngram_std = run_text_baseline(filtered_texts, y, method="tfidf_ngram")
        print(f"Accuracy: {ngram_acc:.1%} (±{ngram_std:.1%})")

        # 4. TF-IDF Full (trigrams, no stop words)
        print("\n--- TF-IDF (Full: 1-3 grams, no stopword removal) ---")
        full_acc, full_std = run_text_baseline(filtered_texts, y, method="tfidf_full")
        print(f"Accuracy: {full_acc:.1%} (±{full_std:.1%})")

        # 5. Simple "user" keyword heuristic
        print("\n--- Keyword Heuristic ('user') ---")
        user_acc, _ = run_keyword_heuristic(filtered_texts, y, ["user"])
        print(f"Accuracy: {user_acc:.1%}")

        # 6. Extended keyword heuristic
        print("\n--- Keyword Heuristic ('user', 'they', 'their') ---")
        extended_acc, _ = run_keyword_heuristic(filtered_texts, y, ["user", "they", "their"])
        print(f"Accuracy: {extended_acc:.1%}")

        # Store results
        results[threshold] = {
            "n_syco": int(n_syco),
            "n_correct": int(n_correct),
            "n_total": int(n_total),
            "pct_data": pct_data,
            "activation_probe": best_act_acc,
            "activation_layer": best_layer,
            "tfidf_unigram": tfidf_acc,
            "tfidf_ngram": ngram_acc,
            "tfidf_full": full_acc,
            "keyword_user": user_acc,
            "keyword_extended": extended_acc,
        }

        # Summary
        print("\n" + "-"*50)
        print("COMPARISON SUMMARY")
        print("-"*50)
        print(f"Activation Probe:     {best_act_acc:.1%}")
        print(f"TF-IDF (best):        {max(tfidf_acc, ngram_acc, full_acc):.1%}")
        print(f"Keyword Heuristic:    {max(user_acc, extended_acc):.1%}")
        print(f"")
        gap = best_act_acc - max(tfidf_acc, ngram_acc, full_acc)
        print(f"Activation vs Text:   {gap:+.1%} ({'PROBE WINS' if gap > 0 else 'TEXT WINS'})")

    return results


def plot_comparison(results: Dict[float, Dict], output_path: Path, model_name: str):
    """Generate comparison plot."""

    thresholds = sorted(results.keys())

    activation_accs = [results[t]["activation_probe"] for t in thresholds]
    tfidf_accs = [results[t]["tfidf_unigram"] for t in thresholds]
    ngram_accs = [results[t]["tfidf_ngram"] for t in thresholds]
    full_accs = [results[t]["tfidf_full"] for t in thresholds]
    keyword_accs = [results[t]["keyword_user"] for t in thresholds]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(thresholds))
    width = 0.15

    bars1 = ax.bar(x - 2*width, activation_accs, width, label='Activation Probe', color='#2E86AB', edgecolor='black')
    bars2 = ax.bar(x - width, tfidf_accs, width, label='TF-IDF (unigram)', color='#A23B72', edgecolor='black')
    bars3 = ax.bar(x, ngram_accs, width, label='TF-IDF (n-gram)', color='#F18F01', edgecolor='black')
    bars4 = ax.bar(x + width, full_accs, width, label='TF-IDF (full)', color='#C73E1D', edgecolor='black')
    bars5 = ax.bar(x + 2*width, keyword_accs, width, label="'user' heuristic", color='#3A6B35', edgecolor='black')

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Random (50%)')

    ax.set_xlabel('Threshold (δ)', fontsize=14)
    ax.set_ylabel('Balanced Accuracy', fontsize=14)
    ax.set_title(f'Activation Probe vs Text Baselines\n{model_name}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f'δ={t}' for t in thresholds])
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def print_final_summary(results: Dict[float, Dict]):
    """Print publication-ready summary table."""

    print("\n" + "="*90)
    print("FINAL SUMMARY: Activation Probe vs Text Baselines")
    print("="*90)
    print(f"{'Threshold':<10} {'N':<8} {'Activation':<12} {'TF-IDF':<12} {'Keyword':<12} {'Gap':<12}")
    print("-"*90)

    for threshold in sorted(results.keys()):
        r = results[threshold]
        best_text = max(r["tfidf_unigram"], r["tfidf_ngram"], r["tfidf_full"])
        gap = r["activation_probe"] - best_text

        print(f"{threshold:<10.2f} {r['n_total']:<8} {r['activation_probe']:<12.1%} "
              f"{best_text:<12.1%} {r['keyword_user']:<12.1%} {gap:+.1%}")

    print("="*90)

    # Key finding
    avg_gap = np.mean([
        results[t]["activation_probe"] - max(results[t]["tfidf_unigram"], results[t]["tfidf_ngram"], results[t]["tfidf_full"])
        for t in results.keys()
    ])

    print(f"\n** KEY FINDING **")
    print(f"Average gap (Activation - Text): {avg_gap:+.1%}")
    if avg_gap > 0.05:
        print(f"The activation probe outperforms text baselines by {avg_gap:.0%} on average.")
        print(f"This proves we detect INTERNAL STATE, not just surface vocabulary.")
    elif avg_gap > 0:
        print(f"Small advantage for activation probe. Consider additional analysis.")
    else:
        print(f"WARNING: Text baseline performs comparably or better!")


def main():
    base_dir = Path("/home/jacekduszenko/Workspace/keyword_probe_experiment/thought-anchors")

    available_models = {
        "deepseek-qwen-7b": base_dir / "multimodel_seven/multimodel_prod/deepseek-qwen-7b/activations",
        "deepseek-qwen-1.5b": base_dir / "qwen_multimodel/deepseek-qwen-1.5b/activations",
        "deepseek-llama-8b": base_dir / "llama_multimodel/deepseek-llama-8b/activations",
        "falcon-h1r-7b": base_dir / "falcon/falcon-h1r-7b/activations",
    }

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "deepseek-qwen-7b"

    if model_name not in available_models:
        print(f"Available models: {list(available_models.keys())}")
        sys.exit(1)

    act_dir = available_models[model_name]

    # Load data
    activations, labels, accuracy_drops, texts, target_layers = load_data(act_dir)

    # Run comparison
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = run_comparison(activations, labels, accuracy_drops, texts, target_layers, thresholds)

    # Save results
    output_dir = base_dir / "analysis_results"
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f"text_baseline_{model_name}.json"
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate plot
    plot_path = output_dir / f"text_baseline_{model_name}.png"
    plot_comparison(results, plot_path, model_name)

    # Final summary
    print_final_summary(results)


if __name__ == "__main__":
    main()
