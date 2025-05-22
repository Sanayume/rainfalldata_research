import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects

# 1. Data Preparation (Same as before)
data = {
    'Probability Threshold': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    'Accuracy': [0.9404, 0.9401, 0.9389, 0.9371, 0.9348, 0.9316, 0.9274],
    'POD': [0.9493, 0.9426, 0.9356, 0.9282, 0.9202, 0.9114, 0.9014],
    'FAR': [0.0460, 0.0404, 0.0357, 0.0314, 0.0274, 0.0238, 0.0207],
    'CSI': [0.9077, 0.9066, 0.9043, 0.9011, 0.8970, 0.8916, 0.8846],
    'FP': [29917, 25949, 22618, 19642, 16922, 14529, 12414],
    'FN': [33114, 37458, 42037, 46911, 52090, 57843, 64379]
}
df_threshold = pd.DataFrame(data)
df_threshold['Threshold_str'] = df_threshold['Probability Threshold'].apply(lambda x: f"{x:.2f}")

# 2. Styling (Same as before, focusing on light theme, flashy colors)
plt.style.use('seaborn-v0_8-white')
plt.rcParams['figure.facecolor'] = '#FDFEFE'
plt.rcParams['axes.facecolor'] = '#FDFEFE'

TEXT_COLOR = "#2C3E50"
TITLE_COLOR = "#E74C3C"
LABEL_COLOR = "#34495E"
MARKER_EDGE_COLOR = "#5D6D7E"
AXIS_LINE_COLOR = '#444444' # Color for the axis lines themselves

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Comic Sans MS', 'Arial Rounded MT Bold', 'Calibri', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['figure.titlesize'] = 40
plt.rcParams['axes.edgecolor'] = AXIS_LINE_COLOR # Set default spine color

num_thresholds = len(df_threshold)
palette = sns.color_palette("turbo", num_thresholds)

marker_effect = [
    path_effects.Stroke(linewidth=2, foreground='#FFFFFF', alpha=0.7),
    path_effects.Stroke(linewidth=1, foreground=MARKER_EDGE_COLOR, alpha=0.8),
    path_effects.Normal()
]

rate_metrics = ['Accuracy', 'POD', 'FAR', 'CSI']
count_metrics = ['FP', 'FN']

# --- Generating Individual "Flashy" Vertical Lollipop Plots ---
for metric in rate_metrics + count_metrics:
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(df_threshold)) * 2
    metric_values = df_threshold[metric].values

    for i in range(len(df_threshold)):
        stem_base = 0
        if metric in count_metrics and metric_values.min() > 0:
             stem_base = metric_values.min() * 0.95
        ax.hlines(y=y_pos[i], xmin=stem_base, xmax=metric_values[i],
                  color=palette[i % len(palette)],
                  linewidth=3.5, alpha=0.7, zorder=1)

    ax.scatter(metric_values, y_pos,
               s=350, c=[palette[i % len(palette)] for i in range(len(df_threshold))],
               edgecolors=MARKER_EDGE_COLOR, linewidths=1.5, alpha=0.9, zorder=2,
               path_effects=marker_effect if metric not in ['FP', 'FN'] else None)

    for i in range(len(df_threshold)):
        x_val = metric_values[i]
        text_align = 'left' if x_val >= 0 else 'right'
        x_range = df_threshold[metric].max() - (df_threshold[metric].min() if metric in count_metrics else 0)
        x_offset_factor = 0.015
        x_offset = x_range * x_offset_factor if x_range > 0 else 0.01
        text_val_str = f"{x_val:.3f}" if metric in rate_metrics else f"{x_val:,.0f}"
        ax.text(x_val + (x_offset if x_val >=0 else -x_offset), y_pos[i], text_val_str,
                color=TEXT_COLOR, ha=text_align, va='center',
                fontsize=24, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_threshold['Threshold_str'], color=LABEL_COLOR, fontsize=30)
    ax.set_ylabel("Probability Threshold", color=LABEL_COLOR, fontsize=30, fontweight='medium')
    ax.set_xlabel(f"{metric} Value" if metric in rate_metrics else f"{metric} Count",
                  color=LABEL_COLOR, fontsize=24, fontweight='medium') # Clarified x-axis label
    fig.suptitle(f"{metric} vs. Probability Threshold (Vertical)",
                 color=TITLE_COLOR, fontsize=24, fontweight='bold', y=0.97)

    if metric in rate_metrics:
        ax.set_xlim(min(0.0, df_threshold[metric].min() * 0.9 if df_threshold[metric].min() < 0 else 0),
                    df_threshold[metric].max() * 1.15 if df_threshold[metric].max() > 0 else 0.1)
    else:
        min_val_counts = df_threshold[metric].min()
        max_val_counts = df_threshold[metric].max()
        padding = (max_val_counts - min_val_counts) * 0.15 if (max_val_counts - min_val_counts) > 0 else 100
        ax.set_xlim(min_val_counts - padding *0.3 , max_val_counts + padding)

    # --- Crucial Change Here ---
    # Remove grid lines
    ax.grid(False)

    # Keep bottom and left axis lines (spines), remove top and right
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    # Alternatively, more direct control:
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(True)
    # ax.spines['left'].set_color(AXIS_LINE_COLOR) # Ensure consistent color
    # ax.spines['left'].set_linewidth(1.2)        # Optional: set linewidth
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['bottom'].set_color(AXIS_LINE_COLOR)
    # ax.spines['bottom'].set_linewidth(1.2)

    # Ensure tick marks are visible if spines are kept
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, color=TEXT_COLOR, labelcolor=TEXT_COLOR)


    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(f'flashy_vertical_lollipop_with_axes_{metric}.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

print("Flashy vertical lollipop plots with axis lines (no grid) generated!")