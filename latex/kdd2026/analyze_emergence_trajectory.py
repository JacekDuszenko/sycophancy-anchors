"""
Analyze emergence trajectory shapes across all models.
Investigates: linearity, inflection points, rate of change, etc.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt

# Define paths to emergence results for each model
BASE_PATH = Path("/home/jacekduszenko/Workspace/keyword_probe_experiment/thought-anchors")

MODEL_CONFIGS = {
    "Llama-8B": {
        "path": None,  # We'll need to find this or use the figure data
        "best_layer": 28,
    },
    "Falcon-7B": {
        "path": BASE_PATH / "falcon-results" / "emergence_results.csv",
        "best_layer": 34,
    },
    "Qwen-7B": {
        "path": BASE_PATH / "qwen7b-results" / "emergence_results.csv",
        "best_layer": 21,
    },
    "Qwen-1.5B": {
        "path": BASE_PATH / "qwen1.5b-results" / "emergence_results.csv",
        "best_layer": 21,
    },
}

def load_emergence_data(path, best_layer):
    """Load emergence data for the best layer."""
    df = pd.read_csv(path)
    df_layer = df[df['layer'] == best_layer].copy()
    df_layer = df_layer.sort_values('position')
    return df_layer

def analyze_trajectory(model_name, df):
    """Analyze the trajectory shape for a model."""
    positions = df['position'].values
    accuracies = df['accuracy'].values

    # Basic stats
    start_acc = accuracies[0]
    end_acc = accuracies[-1]
    emergence = end_acc - start_acc

    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(positions, accuracies)
    r_squared = r_value ** 2

    # Check for acceleration (quadratic fit)
    coeffs = np.polyfit(positions, accuracies, 2)
    quad_pred = np.polyval(coeffs, positions)
    ss_res_quad = np.sum((accuracies - quad_pred) ** 2)
    ss_tot = np.sum((accuracies - np.mean(accuracies)) ** 2)
    r_squared_quad = 1 - (ss_res_quad / ss_tot)

    # Curvature (positive = accelerating, negative = decelerating)
    curvature = coeffs[0]

    # Find where accuracy first significantly exceeds start (> 2 std above start)
    std_start = np.std(accuracies[:5])  # std of first 5 positions
    threshold = start_acc + 2 * std_start
    first_sig_idx = np.where(accuracies > threshold)[0]
    first_sig_pos = first_sig_idx[0] if len(first_sig_idx) > 0 else None

    # Rate of change in last 5 vs first 5 tokens
    rate_first5 = (accuracies[4] - accuracies[0]) / 5
    rate_last5 = (accuracies[-1] - accuracies[-5]) / 5
    acceleration_ratio = rate_last5 / rate_first5 if rate_first5 != 0 else np.inf

    return {
        'model': model_name,
        'start_acc': start_acc,
        'end_acc': end_acc,
        'emergence_pp': emergence * 100,
        'linear_r2': r_squared,
        'quad_r2': r_squared_quad,
        'curvature': curvature,
        'curvature_sign': 'accelerating' if curvature > 0 else 'decelerating',
        'first_sig_pos': first_sig_pos,
        'rate_first5': rate_first5 * 100,  # pp per token
        'rate_last5': rate_last5 * 100,
        'acceleration_ratio': acceleration_ratio,
    }

def main():
    results = []
    all_data = {}

    for model_name, config in MODEL_CONFIGS.items():
        if config['path'] is None or not config['path'].exists():
            print(f"Skipping {model_name} - no data file found")
            continue

        df = load_emergence_data(config['path'], config['best_layer'])
        analysis = analyze_trajectory(model_name, df)
        results.append(analysis)
        all_data[model_name] = df

        print(f"\n{'='*50}")
        print(f"Model: {model_name} (Layer {config['best_layer']})")
        print(f"{'='*50}")
        print(f"Start accuracy: {analysis['start_acc']*100:.1f}%")
        print(f"End accuracy: {analysis['end_acc']*100:.1f}%")
        print(f"Emergence: +{analysis['emergence_pp']:.1f} pp")
        print(f"Linear R²: {analysis['linear_r2']:.3f}")
        print(f"Quadratic R²: {analysis['quad_r2']:.3f}")
        print(f"Curvature: {analysis['curvature']:.6f} ({analysis['curvature_sign']})")
        print(f"First significant position: {analysis['first_sig_pos']}")
        print(f"Rate first 5 tokens: {analysis['rate_first5']:.3f} pp/token")
        print(f"Rate last 5 tokens: {analysis['rate_last5']:.3f} pp/token")
        print(f"Acceleration ratio (last/first): {analysis['acceleration_ratio']:.2f}x")

    # Summary across models
    print(f"\n{'='*50}")
    print("SUMMARY ACROSS MODELS")
    print(f"{'='*50}")

    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))

        # Key findings
        print("\n--- Key Findings ---")
        avg_linear_r2 = df_results['linear_r2'].mean()
        avg_quad_r2 = df_results['quad_r2'].mean()
        print(f"Average linear R²: {avg_linear_r2:.3f}")
        print(f"Average quadratic R²: {avg_quad_r2:.3f}")

        if avg_quad_r2 > avg_linear_r2 + 0.05:
            print("→ Trajectory is notably non-linear (quadratic fits better)")
        else:
            print("→ Trajectory is approximately linear")

        # Curvature pattern
        curvatures = df_results['curvature'].values
        if all(c > 0 for c in curvatures):
            print("→ All models show ACCELERATING emergence (rate increases near anchor)")
        elif all(c < 0 for c in curvatures):
            print("→ All models show DECELERATING emergence (rate decreases near anchor)")
        else:
            print("→ Mixed curvature patterns across models")

        # Acceleration ratio
        avg_accel = df_results['acceleration_ratio'].mean()
        print(f"Average acceleration ratio: {avg_accel:.2f}x")
        if avg_accel > 1.5:
            print("→ Sycophancy signal strengthens significantly in final tokens")

    # Plot trajectories
    if all_data:
        plt.figure(figsize=(10, 6))
        for model_name, df in all_data.items():
            positions = df['position'].values
            accuracies = df['accuracy'].values * 100
            plt.plot(positions, accuracies, marker='o', markersize=3, label=model_name)

        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
        plt.xlabel('Token Position (0 = anchor-29, 29 = anchor end)')
        plt.ylabel('Probe Accuracy (%)')
        plt.title('Emergence Trajectories Across Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('emergence_trajectory_analysis.png', dpi=150)
        print(f"\nPlot saved to emergence_trajectory_analysis.png")

if __name__ == "__main__":
    main()
