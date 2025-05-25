import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects

# 1. Data Preparation (Same as before)
data = {
    'Probability Threshold': [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    'Accuracy': [0.9228, 0.9404, 0.9459, 0.9471, 0.9456, 0.9420, 0.9366],
    'POD': [0.9876, 0.9764, 0.9660, 0.9558, 0.9447, 0.9319, 0.9169],
    'FAR': [0.1024, 0.0695, 0.0526, 0.0416, 0.0335, 0.0270, 0.0210],
    'CSI': [0.8876, 0.9100, 0.9168, 0.9177, 0.9147, 0.9084, 0.8992],
    'FP': [73571, 47621, 35019, 27111, 21408, 16881, 12864],
    'FN': [8073, 15424, 22198, 28850, 36132, 44456, 54238]
}
df_threshold = pd.DataFrame(data)
df_threshold['Threshold_str'] = df_threshold['Probability Threshold'].apply(lambda x: f"{x:.2f}")

# 2. New color palette
IMG_PALETTE = {
    'light_green': '#B8DBB3',
    'medium_green': '#72B063', # Green for accuracy in target bar chart
    'muted_blue': '#719AAC',   # Muted blue for csi in target bar chart
    'orange': '#E29135',       # Orange for pod in target bar chart
    'light_blue': '#94C6CD',   # Light blue for 1-FAR in target bar chart
    'dark_blue': '#4A5F7E'
}

# 3. Styling with new colors
plt.style.use('seaborn-v0_8-white')
plt.rcParams['figure.facecolor'] = '#FDFEFE'
plt.rcParams['axes.facecolor'] = '#FDFEFE'

TEXT_COLOR = IMG_PALETTE['dark_blue']
TITLE_COLOR = IMG_PALETTE['orange']
LABEL_COLOR = IMG_PALETTE['dark_blue']
MARKER_EDGE_COLOR = IMG_PALETTE['dark_blue']
AXIS_LINE_COLOR = IMG_PALETTE['dark_blue']

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Comic Sans MS', 'Arial Rounded MT Bold', 'Calibri', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['figure.titlesize'] = 40
plt.rcParams['axes.edgecolor'] = AXIS_LINE_COLOR

num_thresholds = len(df_threshold)
# Instead of using turbo palette, create a custom palette from our colors
color_values = list(IMG_PALETTE.values())
palette = color_values * (num_thresholds // len(color_values) + 1)
palette = palette[:num_thresholds]

marker_effect = [
    path_effects.Stroke(linewidth=2, foreground='#FFFFFF', alpha=0.7),
    path_effects.Stroke(linewidth=1, foreground=MARKER_EDGE_COLOR, alpha=0.8),
    path_effects.Normal()
]

rate_metrics = ['Accuracy', 'POD', 'FAR', 'CSI']
count_metrics = ['FP', 'FN']

# --- Generating Individual "Flashy" Vertical Lollipop Plots with new colors ---
for metric in rate_metrics + count_metrics:
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(df_threshold)) * 2
    metric_values = df_threshold[metric].values

    for i in range(len(df_threshold)):
        stem_base = 0
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
                  color=LABEL_COLOR, fontsize=24, fontweight='medium')
    fig.suptitle(f"{metric} vs. Probability Threshold (Vertical)",
                 color=TITLE_COLOR, fontsize=24, fontweight='bold', y=0.97)

    if metric in rate_metrics:
        ax.set_xlim(min(0.0, df_threshold[metric].min() * 0.9 if df_threshold[metric].min() < 0 else 0),
                    df_threshold[metric].max() * 1.15 if df_threshold[metric].max() > 0 else 0.1)
    else:
        min_val_counts = df_threshold[metric].min()
        max_val_counts = df_threshold[metric].max()

        padding_right_factor = 0.15 # Factor for padding on the right of max value
        
        # Set x-axis limits to start exactly at 0 (or slightly before for label visibility)
        # and extend beyond max_val_counts
        # We want the visual y-axis (left spine) to be at x=0.
        
        # Ensure a small buffer on the left of 0 if we want '0' label not to be cut.
        # However, for lollipop starting at 0, having xlim start at 0 is visually clean.
        # If tick labels overlap, we adjust tick padding or label formatting.
        current_xlim_left = 0
        current_xlim_right = max_val_counts * (1 + padding_right_factor)
        
        if max_val_counts == 0 : # Handle case where all FP/FN are 0
             current_xlim_right = 10 # Arbitrary small range if all values are 0

        ax.set_xlim(current_xlim_left, current_xlim_right)

    # Remove grid lines
    ax.grid(False)

    # Keep bottom and left axis lines (spines), remove top and right
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    # For FP/FN plots, ensure the left spine (our Y-axis) is positioned at x=0
    if metric in count_metrics:
        ax.spines['left'].set_position('zero')
        # Optionally, if the bottom spine feels redundant now that Y-axis is at x=0
        # ax.spines['bottom'].set_visible(False)
        # ax.tick_params(axis='x', which='both', bottom=False, labelbottom=True) # if bottom spine hidden

    # Ensure tick marks are visible if spines are kept
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, 
                   color=AXIS_LINE_COLOR, labelcolor=TEXT_COLOR)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, 
                   color=AXIS_LINE_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(f'flashy_vertical_lollipop_with_axes_{metric}.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

print("Flashy vertical lollipop plots with new color palette generated!")