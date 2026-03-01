#!/usr/bin/env python3
"""
Generate dual-axis Pareto figure: feasibility rate + mean reduction vs k.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from k-factor ablation table
k_values =      [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]
feasibility =   [76.5, 71.4, 66.7, 61.9, 38.8, 24.2, 18.7, 14.2, 7.1]
mean_reduction = [18.7, 19.9, 21.3, 22.4, 23.5, 25.5, 26.1, 26.0, 25.8]

fig, ax1 = plt.subplots(figsize=(7, 4.5))

# Left axis: feasibility rate (blue)
color1 = '#2166ac'
ax1.set_xlabel('Uncertainty factor $k$', fontsize=12)
ax1.set_ylabel('Feasibility rate (%)', color=color1, fontsize=12)
line1 = ax1.plot(k_values, feasibility, 'o-', color=color1, linewidth=2, 
                  markersize=7, label='Feasibility rate', zorder=3)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 85)
ax1.set_xlim(-0.15, 3.15)

# Right axis: mean reduction (red)
ax2 = ax1.twinx()
color2 = '#b2182b'
ax2.set_ylabel('Mean reduction among feasible (%)', color=color2, fontsize=12)
line2 = ax2.plot(k_values, mean_reduction, 's--', color=color2, linewidth=2, 
                  markersize=7, label='Mean reduction', zorder=3)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(15, 30)

# Highlight operating point k=1.0
ax1.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7, zorder=1)
ax1.annotate('$k = 1.0$\n(operating point)',
             xy=(1.0, 38.8), xytext=(1.6, 55),
             fontsize=10, ha='left',
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                       edgecolor='gray', alpha=0.9))

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=10, framealpha=0.9)

# Grid
ax1.grid(True, alpha=0.3, linestyle='-')

plt.title('Conservatism–Yield Pareto Frontier', fontsize=13, pad=12)
plt.tight_layout()

out_path = 'figures/fig_pareto_dual_axis.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved to {out_path}")

# Also save PDF for LaTeX
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved PDF to {out_path.replace('.png', '.pdf')}")
