#!/usr/bin/env python3
"""
Generate publication-quality causal impact regressor figure for ICLR paper.
Shows a beautiful trajectory plot of predicted vs actual confidence over sentences.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as path_effects

# Set up publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# Load predictions data
df = pd.read_csv('/home/jacekduszenko/Workspace/keyword_probe_experiment/thought-anchors/causal_impact_data_predictions.csv')

# Get trajectory for a good sample (arc_conv_0323 - lowest error, interesting trajectory)
sample_id = 'arc_conv_0323'
sample_df = df[df['sample_id'] == sample_id].sort_values('sentence_idx')

# Create figure
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

sentence_idx = sample_df['sentence_idx'].values
actual = sample_df['actual_prob_ratio'].values
predicted = sample_df['predicted_prob_ratio'].values

# Fill regions for sycophantic (negative) vs correct (positive)
ax.axhspan(-10, 0, alpha=0.08, color='#E74C3C', zorder=0)
ax.axhspan(0, 10, alpha=0.08, color='#27AE60', zorder=0)

# Add region labels
ax.text(0.5, 5.8, 'Favors Correct Answer', fontsize=9, color='#27AE60',
        fontstyle='italic', alpha=0.8, ha='left')
ax.text(0.5, -2.5, 'Favors User\'s Wrong Answer', fontsize=9, color='#E74C3C',
        fontstyle='italic', alpha=0.8, ha='left')

# Zero line
ax.axhline(y=0, color='#7F8C8D', linestyle='-', linewidth=1, alpha=0.5, zorder=1)

# Plot actual trajectory
ax.plot(sentence_idx, actual, 'o-', color='#2980B9', linewidth=2.5, markersize=6,
        label='Actual $\\log \\frac{P(\\mathrm{correct})}{P(\\mathrm{wrong})}$',
        markeredgecolor='white', markeredgewidth=1, zorder=3)

# Plot predicted trajectory
ax.plot(sentence_idx, predicted, 's--', color='#E74C3C', linewidth=2, markersize=5,
        label='Predicted from activations',
        markeredgecolor='white', markeredgewidth=0.8, zorder=3, alpha=0.85)

# Add R² annotation
r2 = 0.742
rmse = 1.90
textstr = f'$R^2 = {r2:.2f}$\nRMSE $= {rmse:.2f}$'
props = dict(boxstyle='round,pad=0.4', facecolor='white',
             edgecolor='#8E44AD', linewidth=1.5, alpha=0.95)
ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        fontweight='bold', color='#8E44AD', bbox=props)

# Labels and formatting
ax.set_xlabel('Sentence Index in Reasoning Trace', fontweight='bold', color='#2C3E50')
ax.set_ylabel('Confidence Ratio', fontweight='bold', color='#2C3E50')
ax.set_xlim(-0.5, max(sentence_idx) + 0.5)
ax.set_ylim(-4, 7)

# Grid
ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#BDC3C7', zorder=0)
ax.xaxis.grid(True, linestyle='-', alpha=0.2, color='#BDC3C7', zorder=0)
ax.set_axisbelow(True)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDC3C7')
ax.spines['bottom'].set_color('#BDC3C7')
ax.tick_params(colors='#2C3E50')

# Legend
ax.legend(loc='lower right', framealpha=0.95, edgecolor='#BDC3C7')

plt.tight_layout()
plt.savefig('figures/causal_impact_regressor.pdf', format='pdf',
            facecolor='white', edgecolor='none')
plt.savefig('figures/causal_impact_regressor.png', format='png',
            facecolor='white', edgecolor='none')
print("Saved figures/causal_impact_regressor.pdf and .png")
