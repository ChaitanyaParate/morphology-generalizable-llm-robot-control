#!/usr/bin/env python3
"""Generate matplotlib figures for the seminar report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = "/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/report_figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'figure.dpi': 200,
})

# ============================================================================
# Figure 6: Training Reward Curve
# ============================================================================
steps = np.array([2, 50, 100, 200, 300, 400, 500, 600, 700, 750, 800, 900, 1000,
                  1100, 1200, 1300, 1400, 1500, 1540, 1550, 1570, 1590, 1680, 1700,
                  1720, 1740, 1760, 1800, 1820, 1900, 1960, 1970, 1975]) * 1024

rewards = np.array([808, 850, 870, 880, 882, 900, 920, 950, 960, 977, 1050, 1100, 1120,
                    1193, 1634, 1661, 1653, 1595, 1416, 889, 610, 650, 588, 845,
                    762, 716, 673, 656, 629, 624, 578, 782, 812])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(steps/1e6, rewards, color='#2196F3', linewidth=1.5, label='Episode Reward')
ax.fill_between(steps/1e6, rewards*0.9, rewards*1.05, alpha=0.15, color='#2196F3')

# Phase annotations
ax.axvspan(0, 0.3, alpha=0.08, color='green', label='Phase 1: Survival')
ax.axvspan(0.3, 1.0, alpha=0.08, color='orange', label='Phase 2: Active Balance')
ax.axvspan(1.0, 1.54, alpha=0.08, color='blue', label='Phase 3: Reward Tuning')
ax.axvspan(1.54, 2.0, alpha=0.08, color='red', label='Phase 4-5: Anti-Crouch')

ax.axvline(x=1.54, color='red', linestyle='--', alpha=0.7)
ax.annotate('Anti-Crouch\nGating Applied', xy=(1.54, 1400), fontsize=9,
            ha='center', color='red', fontweight='bold')
ax.axvline(x=1.2, color='blue', linestyle='--', alpha=0.5)
ax.annotate('Phase 4:\nSafe Exploration', xy=(1.2, 1650), fontsize=8,
            ha='center', color='blue')

ax.set_xlabel('Training Steps (Millions)', fontsize=13)
ax.set_ylabel('Episode Reward', fontsize=13)
ax.set_title('Figure 6: Training Reward Curve — GNN PPO (2M Steps)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.set_xlim(0, 2.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_reward_curve.png', bbox_inches='tight')
plt.close()
print("✅ Figure 6 saved")

# ============================================================================
# Figure 7: Explained Variance Progression
# ============================================================================
steps_ev = np.array([2, 100, 200, 300, 500, 700, 800, 1000, 1200, 1400, 1500,
                     1550, 1570, 1590, 1680, 1700, 1720, 1800, 1900, 1960, 1975]) * 1024

ev_vals = np.array([0.00, 0.05, 0.12, 0.24, 0.29, 0.29, 0.34, 0.35, 0.02, 0.08,
                    0.18, 0.37, 0.56, 0.56, 0.74, 0.61, 0.64, 0.82, 0.79, 0.71, 0.72])

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(steps_ev/1e6, ev_vals, color='#4CAF50', linewidth=2, marker='o', markersize=4, label='Explained Variance')
ax.fill_between(steps_ev/1e6, 0, ev_vals, alpha=0.15, color='#4CAF50')

ax.axhline(y=0.6, color='green', linestyle=':', alpha=0.6, label='Good EV threshold (0.6)')
ax.axhline(y=0.3, color='orange', linestyle=':', alpha=0.6, label='Moderate EV threshold (0.3)')

ax.axvline(x=1.2, color='blue', linestyle='--', alpha=0.5)
ax.annotate('Reward landscape\nchanged — Critic\nre-anchoring', xy=(1.2, 0.05),
            fontsize=8, ha='center', color='blue')
ax.axvline(x=1.54, color='red', linestyle='--', alpha=0.5)
ax.annotate('Anti-Crouch\nCritic recovers', xy=(1.65, 0.55),
            fontsize=8, ha='center', color='red')

ax.set_xlabel('Training Steps (Millions)', fontsize=13)
ax.set_ylabel('Explained Variance', fontsize=13)
ax.set_title('Figure 7: Explained Variance Progression Over Training', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 2.05)
ax.set_ylim(-0.05, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_explained_variance.png', bbox_inches='tight')
plt.close()
print("✅ Figure 7 saved")

# ============================================================================
# Figure 8: GNN vs MLP Parameter Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 4))

models = ['MLP Policy\n(256-dim hidden)', 'GNN Policy\n(48-dim hidden)']
params = [200000, 29566]
colors = ['#EF5350', '#4CAF50']

bars = ax.barh(models, params, color=colors, height=0.5, edgecolor='white', linewidth=1.5)
ax.bar_label(bars, labels=['~200,000 params', '29,566 params'], padding=8, fontsize=12, fontweight='bold')

ax.annotate('85% fewer\nparameters', xy=(115000, 0.5), fontsize=14, fontweight='bold',
            color='#1565C0', ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2),
            xytext=(115000, 1.2))

ax.set_xlabel('Number of Parameters', fontsize=13)
ax.set_title('Figure 8: GNN vs MLP Parameter Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0, 280000)
ax.grid(True, alpha=0.2, axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}/fig8_parameter_comparison.png', bbox_inches='tight')
plt.close()
print("✅ Figure 8 saved")

# ============================================================================
# Figure: Forward velocity progression (bonus useful chart)
# ============================================================================
steps_fwd = np.array([2, 100, 300, 500, 700, 800, 1000, 1200, 1400, 1500,
                      1550, 1590, 1700, 1720, 1800, 1960, 1970, 1975]) * 1024

fwd_vals = np.array([0.002, 0.003, 0.008, 0.009, 0.012, 0.016, 0.012, 0.014,
                     0.003, 0.009, 0.012, 0.015, 0.025, 0.020, 0.014, 0.012, 0.021, 0.023])

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(steps_fwd/1e6, fwd_vals, color='#FF9800', linewidth=2, marker='s', markersize=4, label='Forward Velocity (m/s)')
ax.fill_between(steps_fwd/1e6, 0, fwd_vals, alpha=0.15, color='#FF9800')

ax.axhline(y=0.02, color='green', linestyle=':', alpha=0.6, label='Walking threshold (0.02 m/s)')
ax.axvline(x=1.54, color='red', linestyle='--', alpha=0.5)
ax.annotate('Anti-Crouch →\nLegs can swing!', xy=(1.65, 0.024), fontsize=9, ha='center', color='red', fontweight='bold')

ax.set_xlabel('Training Steps (Millions)', fontsize=13)
ax.set_ylabel('Forward Velocity (m/s)', fontsize=13)
ax.set_title('Forward Velocity Progression Over Training', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 2.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_forward_velocity.png', bbox_inches='tight')
plt.close()
print("✅ Forward velocity chart saved")

print(f"\n📁 All figures saved to {OUT}/")
print("   fig6_reward_curve.png")
print("   fig7_explained_variance.png")
print("   fig8_parameter_comparison.png")
print("   fig_forward_velocity.png")
