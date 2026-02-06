#!/usr/bin/env python3
"""
Generate publication-quality figure showing sycophancy emergence during reasoning.
Shows probe accuracy increasing from prompt-end (near chance) to anchor (72.9%).
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
})

# Data from TOKEN_TRAJECTORY_ANALYSIS.md
# Positions relative to anchor
positions = [-29, -28, -27, -26, -25, -24, -23, -22, -21, -20,
             -19, -18, -17, -16, -15, -14, -13, -12, -11, -10,
             -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]

accuracies = [0.604, 0.596, 0.602, 0.584, 0.571, 0.579, 0.562, 0.583, 0.576, 0.580,
              0.591, 0.620, 0.647, 0.619, 0.644, 0.663, 0.686, 0.646, 0.650, 0.645,
              0.649, 0.656, 0.631, 0.631, 0.633, 0.607, 0.604, 0.654, 0.693, 0.729]

std_devs = [0.033, 0.037, 0.022, 0.058, 0.035, 0.041, 0.062, 0.039, 0.047, 0.081,
            0.055, 0.041, 0.050, 0.050, 0.056, 0.029, 0.058, 0.023, 0.053, 0.068,
            0.070, 0.056, 0.043, 0.050, 0.022, 0.035, 0.041, 0.026, 0.053, 0.039]

# Prompt-end data point (shown separately)
prompt_end_acc = 0.551
prompt_end_std = 0.040

fig, ax = plt.subplots(figsize=(9, 5))

# Plot the main trajectory line
positions_arr = np.array(positions)
accuracies_arr = np.array(accuracies)
std_arr = np.array(std_devs)

# Fill between for confidence interval
ax.fill_between(positions_arr, accuracies_arr - std_arr, accuracies_arr + std_arr,
                alpha=0.2, color='#2980b9')

# Main line
ax.plot(positions_arr, accuracies_arr, 'o-', color='#2980b9', linewidth=2,
        markersize=5, label='CoT tokens', markeredgecolor='white', markeredgewidth=0.5)

# Highlight anchor point
ax.scatter([0], [0.729], s=150, c='#e74c3c', marker='*', zorder=10,
           edgecolors='white', linewidths=1.5, label='Anchor')

# Prompt-end point (shown to the left with a gap)
prompt_x = -35
ax.errorbar([prompt_x], [prompt_end_acc], yerr=[prompt_end_std],
            fmt='D', color='#27ae60', markersize=10, capsize=6,
            markeredgecolor='white', markeredgewidth=1.5,
            label='Prompt-end', elinewidth=2, capthick=2)

# Chance line
ax.axhline(y=0.5, color='#95a5a6', linestyle='--', linewidth=1.5,
           label='Chance (50%)', zorder=0)

# Add annotation for prompt-end
ax.annotate(f'{prompt_end_acc:.1%}', xy=(prompt_x, prompt_end_acc + prompt_end_std + 0.02),
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#27ae60')

# Add annotation for anchor
ax.annotate(f'{0.729:.1%}', xy=(0, 0.729 + 0.03),
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#e74c3c')

# Arrow showing the emergence
ax.annotate('', xy=(0, 0.72), xytext=(prompt_x, 0.56),
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.5,
                           connectionstyle='arc3,rad=0.15'))
ax.text(-17, 0.52, 'Sycophancy emerges\nduring reasoning\n(+17.8 pp)',
        ha='center', va='top', fontsize=10, fontweight='bold', color='#8e44ad',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#8e44ad', linewidth=1.5, alpha=0.95))

# Add vertical separator between prompt-end and CoT
ax.axvline(x=-32, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Labels
ax.set_xlabel('Token Position Relative to Anchor', fontweight='bold')
ax.set_ylabel('Balanced Accuracy', fontweight='bold')

# X-axis formatting
ax.set_xlim(-38, 3)
ax.set_ylim(0.45, 0.82)

# Custom x-ticks
xticks = [-35, -29, -20, -10, 0]
xticklabels = ['Prompt\nEnd', '-29', '-20', '-10', 'Anchor']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

# Grid
ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='gray')
ax.set_axisbelow(True)

# Legend
ax.legend(loc='lower right', framealpha=0.95)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/sycophancy_emergence.pdf', format='pdf')
plt.savefig('figures/sycophancy_emergence.png', format='png')
print("Saved figures/sycophancy_emergence.pdf and .png")
