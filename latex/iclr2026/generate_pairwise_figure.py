#!/usr/bin/env python3
"""
Generate publication-quality pairwise classification figure for ICLR paper.
Shows the asymmetry: sycophantic anchors are distinctive, correct anchors are not.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects

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
    'axes.linewidth': 0.8,
})

# Data from experiments
categories = ['Sycophantic\nvs Correct', 'Sycophantic\nvs Neutral', 'Correct\nvs Neutral']
short_labels = ['Syc. vs Corr.', 'Syc. vs Neut.', 'Corr. vs Neut.']
accuracies = [0.846, 0.775, 0.640]
std_devs = [0.007, 0.011, 0.006]

# Modern color palette - warm for sycophancy-related, cool gray for indistinguishable
colors = ['#E74C3C', '#E67E22', '#BDC3C7']  # Red, Orange, Silver
edge_colors = ['#C0392B', '#D35400', '#95A5A6']

fig, ax = plt.subplots(figsize=(6.5, 4.2))

# Subtle gradient background
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

x = np.arange(len(categories))
width = 0.6

# Create bars with rounded edges effect via zorder layering
bars = ax.bar(x, accuracies, width, yerr=std_devs,
              color=colors, edgecolor=edge_colors, linewidth=2,
              capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5, 'color': '#555555'},
              zorder=3)

# Add subtle shadow effect
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    shadow = plt.Rectangle((bar.get_x() + 0.02, 0.5),
                           bar.get_width(), acc - 0.5,
                           color='gray', alpha=0.15, zorder=1)
    ax.add_patch(shadow)

# Add value labels on bars with white outline for readability
for bar, acc, std in zip(bars, accuracies, std_devs):
    height = bar.get_height()
    text = ax.annotate(f'{acc:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.018),
                ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='#2C3E50')
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# Chance line with label
ax.axhline(y=0.5, color='#7F8C8D', linestyle='--', linewidth=1.5,
           label='Chance', zorder=2, alpha=0.8)
ax.text(2.55, 0.505, '50%', fontsize=9, color='#7F8C8D', va='bottom', ha='left')

# Elegant bracket for "Highly Detectable" region
bracket_y = 0.91
bracket_color = '#27AE60'
ax.plot([0, 0, 1, 1], [bracket_y - 0.02, bracket_y, bracket_y, bracket_y - 0.02],
        color=bracket_color, linewidth=2, solid_capstyle='round')
ax.text(0.5, bracket_y + 0.015, 'Highly Detectable', ha='center', va='bottom',
        fontsize=10, fontweight='bold', color=bracket_color,
        fontstyle='italic')

# Asymmetry annotation - cleaner arrow (positioned to the right of the bar)
gap_x = 2.42
ax.annotate('', xy=(gap_x, 0.775), xytext=(gap_x, 0.640),
            arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=2,
                           shrinkA=2, shrinkB=2, mutation_scale=12))
ax.text(gap_x + 0.08, 0.707, '20.6 pp\ngap', ha='left', va='center',
        fontsize=9, fontweight='bold', color='#8E44AD',
        linespacing=0.9)

# Add "Near Chance" label for the third bar
ax.text(2, 0.58, 'Near\nChance', ha='center', va='top',
        fontsize=9, color='#7F8C8D', fontstyle='italic',
        linespacing=0.9)

# Formatting
ax.set_ylabel('Balanced Accuracy', fontweight='bold', color='#2C3E50')
ax.set_xticks(x)
ax.set_xticklabels(categories, color='#2C3E50')
ax.set_ylim(0.45, 0.98)
ax.set_xlim(-0.5, 3.0)

# Subtle horizontal grid only
ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#BDC3C7', zorder=0)
ax.set_axisbelow(True)

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDC3C7')
ax.spines['bottom'].set_color('#BDC3C7')
ax.tick_params(colors='#2C3E50')

# Y-axis formatting
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_yticklabels(['50%', '60%', '70%', '80%', '90%'])

plt.tight_layout()
plt.savefig('figures/pairwise_accuracy.pdf', format='pdf', facecolor='white', edgecolor='none')
plt.savefig('figures/pairwise_accuracy.png', format='png', facecolor='white', edgecolor='none')
print("Saved figures/pairwise_accuracy.pdf and .png")
