import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import MinMaxScaler # For normalization
from pandas.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D # For 3D plot
from matplotlib.lines import Line2D # For custom legends

# --- Define the IMG_PALETTE (from evolution.py) ---
IMG_PALETTE = {
    'light_green': '#B8DBB3',
    'medium_green': '#72B063',
    'muted_blue': '#719AAC',
    'orange': '#E29135',
    'light_blue': '#94C6CD',
    'dark_blue': '#4A5F7E'
}

# --- Data Preparation (Same as before) ---
data = {
    'model_name_raw': [
        'K-Nearest Neighbors (Tuned on Subset)',
        'Support Vector Machine (Tuned on Subset)',
        'Random Forest',
        'LightGBM',
        'Gaussian Naive Bayes (Default)',
        'xgboost(defautl)' # Original spelling
    ],
    'accuracy': [0.7917, 0.8021, 0.8408, 0.8366, 0.7019, 0.8819],
    'pod':      [0.7839, 0.7496, 0.8378, 0.8221, 0.5799, 0.8880],
    'far':      [0.1308, 0.0819, 0.1001, 0.0929, 0.0909, 0.0819],
    'csi':      [0.7012, 0.7026, 0.7665, 0.7582, 0.5481, 0.8228],
    'tp':       [516971, 494322, 552553, 542149, 382432, np.nan],
    'fn':       [142520, 165169, 106938, 117342, 277059, 73133],
    'fp':       [77795,  44111,  61430,  55529,  38237,  51763],
    'tn':       [320429, 354113, 336794, 342695, 359987, np.nan]
}

df_full = pd.DataFrame(data)

def clean_model_name(name):
    name = name.replace('(Tuned on Subset)', '').replace('(Default)', '').strip()
    if name == 'xgboost(defautl)':
        return 'XGBoost'
    return name

df_full['model_name'] = df_full['model_name_raw'].apply(clean_model_name)
df_full = df_full.set_index('model_name')

metrics_to_compare = ['accuracy', 'pod', 'far', 'csi', 'fn', 'fp']
df_compare = df_full[metrics_to_compare].copy()

# --- Global Plotting Aesthetics (Inspired by evolution.py with Even Larger Fonts) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Verdana', 'Arial', 'Geneva', 'Lucid', 'Helvetica', 'Avant Garde', 'sans-serif']

# Even Larger Font sizes
plt.rcParams['axes.titlesize'] = 24 # Main plot titles
plt.rcParams['axes.labelsize'] = 20 # Axis labels (X, Y)
plt.rcParams['xtick.labelsize'] = 16 # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 16 # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 16 # Legend text
plt.rcParams['legend.title_fontsize'] = 18 # Legend title
plt.rcParams['figure.titlesize'] = 26 # For suptitle if used

# Colors from IMG_PALETTE for text elements
plt.rcParams['text.color'] = IMG_PALETTE['dark_blue']
plt.rcParams['axes.labelcolor'] = IMG_PALETTE['dark_blue']
plt.rcParams['xtick.color'] = IMG_PALETTE['dark_blue']
plt.rcParams['ytick.color'] = IMG_PALETTE['dark_blue']
plt.rcParams['axes.titlecolor'] = IMG_PALETTE['medium_green']

# Spines and Facecolor
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.edgecolor'] = IMG_PALETTE['dark_blue']
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# Grid style
plt.rcParams['grid.color'] = IMG_PALETTE['light_green']
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.alpha'] = 0.75

# Savefig defaults
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


# --- 1. Heatmap of All 6 Metrics ---
scaler = MinMaxScaler()
df_normalized_for_heatmap = df_compare.copy()
df_normalized_for_heatmap[metrics_to_compare] = scaler.fit_transform(df_compare[metrics_to_compare])

df_color_scaled = df_normalized_for_heatmap.copy()
for metric_lower_is_better in ['far', 'fn', 'fp']:
    df_color_scaled[metric_lower_is_better] = 1 - df_color_scaled[metric_lower_is_better]

plt.figure(figsize=(14, 10)) # Adjusted size for larger fonts
cmap_heatmap = sns.light_palette(IMG_PALETTE['medium_green'], as_cmap=True)

ax_heatmap = sns.heatmap(
    df_color_scaled,
    annot=df_compare,
    fmt=".4g",
    cmap=cmap_heatmap,
    linewidths=2, # Slightly thicker
    linecolor='white',
    cbar=True,
    annot_kws={"size": 14, "weight": "normal", "color": IMG_PALETTE['dark_blue']} # Larger annotations
)
ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontweight='bold', color=IMG_PALETTE['dark_blue'], fontsize=16) # Larger y-tick labels
ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=30, ha='right', fontweight='bold', color=IMG_PALETTE['dark_blue'], fontsize=16) # Larger x-tick labels
plt.title('Comprehensive Model Performance Heatmap\n(Colors: Normalized, Annotations: Original Values)', pad=25, fontweight='bold', color=IMG_PALETTE['medium_green']) # Increased pad

cbar = ax_heatmap.collections[0].colorbar
cbar.set_label('Normalized Performance (Greener is Better)',
                  fontsize=18, fontweight='bold', color=IMG_PALETTE['dark_blue'], labelpad=20) # Larger cbar label, increased pad
cbar.ax.tick_params(labelsize=14, colors=IMG_PALETTE['dark_blue']) # Larger cbar ticks
ax_heatmap.set_ylabel("")
ax_heatmap.tick_params(axis='x', colors=IMG_PALETTE['dark_blue'])
ax_heatmap.tick_params(axis='y', colors=IMG_PALETTE['dark_blue'])

plt.tight_layout(pad=2.5) # Increased pad
plt.savefig('model_performance_heatmap.png')
plt.show()

# --- 2. Parallel Coordinates Plot ---
df_parallel = df_compare.copy()
scaler_parallel = MinMaxScaler()
df_parallel[metrics_to_compare] = scaler_parallel.fit_transform(df_parallel[metrics_to_compare])
df_parallel_plot = df_parallel.reset_index()

plt.figure(figsize=(17, 9)) # Adjusted size
num_models_parallel = len(df_parallel_plot['model_name'].unique())
parallel_plot_colors_options = [
    IMG_PALETTE['medium_green'], IMG_PALETTE['orange'], IMG_PALETTE['muted_blue'],
    IMG_PALETTE['light_blue'], IMG_PALETTE['dark_blue'], IMG_PALETTE['light_green']
]
parallel_plot_line_colors = [parallel_plot_colors_options[i % len(parallel_plot_colors_options)] for i in range(num_models_parallel)]

pc_plot_ax = plt.gca()
parallel_coordinates(
    df_parallel_plot, 'model_name', ax=pc_plot_ax,
    color=parallel_plot_line_colors,
    linewidth=3.0, alpha=0.9 # Thicker lines
)
for line in pc_plot_ax.get_lines()[num_models_parallel:]:
    line.set_color(IMG_PALETTE['dark_blue'])
    line.set_linewidth(2.0) # Thicker axis lines
    line.set_alpha(0.85)

plt.title('Parallel Coordinates Plot of Normalized Model Metrics', pad=25, fontweight='bold', color=IMG_PALETTE['medium_green'])
plt.ylabel('Normalized Metric Value (0 to 1)', fontweight='bold', color=IMG_PALETTE['dark_blue'])
plt.xticks(fontweight='bold', color=IMG_PALETTE['dark_blue'])
plt.yticks(fontweight='normal', color=IMG_PALETTE['dark_blue'])
pc_plot_ax.tick_params(axis='x', colors=IMG_PALETTE['dark_blue'], labelsize=16) # Larger ticks
pc_plot_ax.tick_params(axis='y', colors=IMG_PALETTE['dark_blue'], labelsize=16) # Larger ticks

legend = plt.legend(title='Models', bbox_to_anchor=(1.04, 1), loc='upper left', # Slightly adjusted anchor
                        fancybox=True, shadow=False, frameon=True,
                        edgecolor=IMG_PALETTE['dark_blue'], framealpha=0.9)
plt.setp(legend.get_texts(), color=IMG_PALETTE['dark_blue'], fontsize=16) # Larger legend text
plt.setp(legend.get_title(), color=IMG_PALETTE['dark_blue'], fontweight='bold', fontsize=18) # Larger legend title

plt.tight_layout(rect=[0, 0, 0.82, 1], pad=2.5) # Adjust rect for legend, increased pad
plt.savefig('model_performance_parallel_coordinates.png')
plt.show()
# ... (rest of print statements)


# --- 3. Enhanced Grouped Bar Chart ---
rate_metrics = ['accuracy', 'pod', 'csi']
df_rates_for_bar = df_compare[rate_metrics].copy()
df_rates_for_bar['1-FAR'] = 1 - df_compare['far']
metrics_for_enhanced_bar = ['accuracy', 'pod', 'csi', '1-FAR']
df_rates_for_bar = df_rates_for_bar[metrics_for_enhanced_bar]
df_rates_melted = df_rates_for_bar.reset_index().melt(id_vars='model_name', var_name='Metric', value_name='Value')

bar_chart_palette_map = {
    'accuracy': IMG_PALETTE['medium_green'],
    'pod': IMG_PALETTE['orange'],
    'csi': IMG_PALETTE['muted_blue'],
    '1-FAR': IMG_PALETTE['light_blue']
}
ordered_bar_palette = [bar_chart_palette_map[metric] for metric in metrics_for_enhanced_bar]

plt.figure(figsize=(18, 10)) # Adjusted size
ax_bar = sns.barplot(x='model_name', y='Value', hue='Metric', data=df_rates_melted,
                     palette=ordered_bar_palette, edgecolor=IMG_PALETTE['dark_blue'], linewidth=0.8, # Slightly thicker edge
                     hue_order=metrics_for_enhanced_bar, alpha=0.9)

for p in ax_bar.patches:
    ax_bar.annotate(format(p.get_height(), '.3f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 10), # Increased offset for larger fonts
                   textcoords = 'offset points',
                   fontsize=12, color=IMG_PALETTE['dark_blue'], fontweight='normal') # Larger annotations

plt.title('Key Model Performance Metrics Comparison', pad=25, fontweight='bold', color=IMG_PALETTE['medium_green'])
plt.xlabel('Model Name', fontweight='bold', color=IMG_PALETTE['dark_blue'])
plt.ylabel('Metric Value (Higher is Better)', fontweight='bold', color=IMG_PALETTE['dark_blue'])
plt.xticks(rotation=30, ha='right', fontweight='bold', color=IMG_PALETTE['dark_blue'])
plt.yticks(fontweight='normal', color=IMG_PALETTE['dark_blue'])
ax_bar.tick_params(axis='x', colors=IMG_PALETTE['dark_blue'], labelsize=16) # Larger ticks
ax_bar.tick_params(axis='y', colors=IMG_PALETTE['dark_blue'], labelsize=16) # Larger ticks

plt.ylim(0, df_rates_melted['Value'].max() * 1.20 if not df_rates_melted.empty else 1.20) # Adjusted for annotation space

legend = plt.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left',
                        fancybox=True, shadow=False, frameon=True,
                        edgecolor=IMG_PALETTE['dark_blue'], framealpha=0.9)
plt.setp(legend.get_texts(), color=IMG_PALETTE['dark_blue'], fontsize=16) # Larger legend text
plt.setp(legend.get_title(), color=IMG_PALETTE['dark_blue'], fontweight='bold', fontsize=18) # Larger legend title

plt.grid(axis='y', linestyle=':', alpha=0.75, color=IMG_PALETTE['light_green'])
plt.tight_layout(rect=[0, 0, 0.86, 1], pad=2.5) # Adjust rect for legend, increased pad
plt.savefig('model_performance_enhanced_bar.png')
plt.show()


# --- 4. 3D Bar Chart ---
metrics_3d = ['accuracy', 'pod', 'csi']
df_3d = df_compare[metrics_3d].copy()

fig_3d = plt.figure(figsize=(16, 11)) # Adjusted size
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_facecolor('#f8f9fa')

x_labels = df_3d.index.to_list()
num_models = len(x_labels)
x_pos = np.arange(num_models)

colors_3d_list = [IMG_PALETTE['medium_green'], IMG_PALETTE['orange'], IMG_PALETTE['muted_blue']]

for i, metric in enumerate(metrics_3d):
    y_values = df_3d[metric].values
    y_pos_metric = np.full(num_models, i)
    z_pos = np.zeros(num_models)
    dx = dy = 0.5 # Keep bar size manageable
    dz = y_values

    ax_3d.bar3d(x_pos - dx/2, y_pos_metric - dy/2, z_pos, dx, dy, dz,
                color=colors_3d_list[i], alpha=0.88, # Slightly less transparent
                edgecolor=IMG_PALETTE['dark_blue'], linewidth=0.6) # Thicker edge
    for j in range(num_models):
         ax_3d.text(x_pos[j], y_pos_metric[j], dz[j] + 0.03, f'{dz[j]:.3f}', # Adjusted offset
                    color=IMG_PALETTE['dark_blue'], ha='center', va='bottom', fontsize=10) # Larger text

# Increased labelpad for Model Name to avoid overlap
ax_3d.set_xlabel('Model Name', labelpad=25, fontweight='bold', color=IMG_PALETTE['dark_blue'], fontsize=18)
ax_3d.set_ylabel('Metric Type', labelpad=15, fontweight='bold', color=IMG_PALETTE['dark_blue'], fontsize=18)
ax_3d.set_zlabel('Metric Value', labelpad=10, fontweight='bold', color=IMG_PALETTE['dark_blue'], fontsize=18)
ax_3d.set_title('3D Bar Chart of Key Performance Metrics', pad=30, fontweight='bold', color=IMG_PALETTE['medium_green'], fontsize=24) # Larger title

ax_3d.set_xticks(x_pos)
ax_3d.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=13, fontweight='bold', color=IMG_PALETTE['dark_blue']) # Larger tick labels
ax_3d.set_yticks(np.arange(len(metrics_3d)))
ax_3d.set_yticklabels(metrics_3d, fontsize=13, fontweight='bold', color=IMG_PALETTE['dark_blue']) # Larger tick labels
ax_3d.tick_params(axis='z', labelsize=13, colors=IMG_PALETTE['dark_blue']) # Larger tick labels
ax_3d.set_zlim(0, 1)

ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False
ax_3d.xaxis.pane.set_edgecolor(IMG_PALETTE['light_green'])
ax_3d.yaxis.pane.set_edgecolor(IMG_PALETTE['light_green'])
ax_3d.zaxis.pane.set_edgecolor(IMG_PALETTE['light_green'])
ax_3d.grid(color=IMG_PALETTE['light_green'], linestyle=':')

legend_elements = [Line2D([0], [0], color=colors_3d_list[i], lw=6, label=metric) for i, metric in enumerate(metrics_3d)] # Even Thicker legend lines
legend = ax_3d.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.18, 1.0), # Adjusted anchor for larger text
                      fancybox=True, shadow=False, frameon=True,
                      edgecolor=IMG_PALETTE['dark_blue'], framealpha=0.9)
plt.setp(legend.get_texts(), color=IMG_PALETTE['dark_blue'], fontsize=16) # Larger legend text
plt.setp(legend.get_title(), color=IMG_PALETTE['dark_blue'], fontweight='bold', fontsize=18) # Larger legend title

ax_3d.view_init(elev=28, azim=-60) # Adjusted view for label spacing
plt.tight_layout(pad=2.5) # Increased pad
plt.savefig('model_performance_3d_bar.png')
plt.show()


# --- 5. Styled Table (using Pandas Styler) ---
styled_df = df_compare.style \
    .set_caption("Model Performance Metrics Summary") \
    .set_table_styles([
        {'selector': 'caption',
         'props': [('color', IMG_PALETTE['medium_green']),
                   ('font-size', '20px'), # Larger caption
                   ('font-weight', 'bold'),
                   ('padding-bottom', '18px')]},
        {'selector': 'th',
         'props': [('color', IMG_PALETTE['dark_blue']),
                   ('background-color', IMG_PALETTE['light_green'] + 'A0'),
                   ('font-weight', 'bold'), ('font-size', '16px'), # Larger header font
                   ('border', f"1.5px solid {IMG_PALETTE['medium_green']}")]},
        {'selector': 'td',
         'props': [('color', IMG_PALETTE['dark_blue']), ('font-size', '15px'), # Larger data font
                   ('border', f"1px solid {IMG_PALETTE['light_green']}")]},
        {'selector': 'tr:hover td',
         'props': [('background-color', IMG_PALETTE['light_blue'] + '50')]}
    ]) \
    .format({
        'accuracy': "{:.4f}", 'pod': "{:.4f}", 'far': "{:.4f}", 'csi': "{:.4f}",
        'fn': "{:,.0f}", 'fp': "{:,.0f}"
    }) \
    .background_gradient(subset=['accuracy', 'pod', 'csi'], cmap=sns.light_palette(IMG_PALETTE['medium_green'], as_cmap=True), text_color_threshold=0.55) \
    .background_gradient(subset=['far', 'fn', 'fp'], cmap=sns.light_palette(IMG_PALETTE['orange'], as_cmap=True, reverse=True), text_color_threshold=0.55)

try:
    from IPython.display import display
    print("\n--- Styled Table (for Jupyter/HTML environments) ---")
    display(styled_df)
    with open("styled_model_performance_table.html", "w") as f:
        f.write(styled_df.to_html())
    print("Styled table saved as styled_model_performance_table.html")
except ImportError:
    print("\n--- Pandas Styler Table (raw data, as IPython is not available) ---")
    print(df_compare)
    print("Consider running in Jupyter Notebook to see the styled table.")

print("\nAll plots generated and saved with even larger font sizes and 3D label adjustment!")