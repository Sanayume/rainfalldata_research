import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 添加3D多边形支持

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import rcParams

# Set style for the plots - 使用通用字体，避免找不到Arial字体的警告
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Verdana', 'Geneva', 'Lucid', 'Helvetica', 'Avant Garde', 'sans-serif']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (12, 9)

# 添加额外的样式设置，提高美观度
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# --- New Color Palette from the Image ---
IMG_PALETTE = {
    'light_green': '#B8DBB3',
    'medium_green': '#72B063',
    'muted_blue': '#719AAC',
    'orange': '#E29135',
    'light_blue': '#94C6CD',
    'dark_blue': '#4A5F7E'
}
# --- End of New Color Palette ---

# Data preparation
final_plot_data_values = [
    # --- Baseline Products ---
    {'Model_Name': 'CHIRPS',    'Original_Identifier': 'Baseline_CHIRPS',   'Type': 'Baseline Product', 'Accuracy': 0.5906, 'POD': 0.4413, 'FAR': 0.1913, 'CSI': 0.3996},
    {'Model_Name': 'CMORPH',    'Original_Identifier': 'Baseline_CMORPH',   'Type': 'Baseline Product', 'Accuracy': 0.6334, 'POD': 0.5168, 'FAR': 0.1763, 'CSI': 0.4654},
    {'Model_Name': 'PERSIANN',  'Original_Identifier': 'Baseline_PERSIANN', 'Type': 'Baseline Product', 'Accuracy': 0.6498, 'POD': 0.6387, 'FAR': 0.2437, 'CSI': 0.5297},
    {'Model_Name': 'GSMAP',     'Original_Identifier': 'Baseline_GSMAP',    'Type': 'Baseline Product', 'Accuracy': 0.7353, 'POD': 0.6108, 'FAR': 0.0606, 'CSI': 0.5876},
    {'Model_Name': 'IMERG',     'Original_Identifier': 'Baseline_IMERG',    'Type': 'Baseline Product', 'Accuracy': 0.7145, 'POD': 0.7073, 'FAR': 0.1935, 'CSI': 0.6047},
    {'Model_Name': 'SM2RAIN',   'Original_Identifier': 'Baseline_SM2RAIN',  'Type': 'Baseline Product', 'Accuracy': 0.6930, 'POD': 0.9033, 'FAR': 0.3072, 'CSI': 0.6450},

    # --- Feature Engineering Iterations ---
    {'Model_Name': 'FeatV1', 'Original_Identifier': 'XGB_V5.1_T0.5', 'Type': 'FeatEng Iteration', 'Accuracy': 0.8526, 'POD': 0.8410, 'FAR': 0.0823, 'CSI': 0.7820},
    {'Model_Name': 'FeatV2', 'Original_Identifier': 'XGB_V5_T0.5',   'Type': 'FeatEng Iteration', 'Accuracy': 0.8585, 'POD': 0.8322, 'FAR': 0.0687, 'CSI': 0.7841},
    {'Model_Name': 'FeatV3', 'Original_Identifier': 'XGB_V3_T0.5',   'Type': 'FeatEng Iteration', 'Accuracy': 0.8597, 'POD': 0.8337, 'FAR': 0.0681, 'CSI': 0.7859},
    {'Model_Name': 'FeatV4', 'Original_Identifier': 'XGB_V2_T0.5',   'Type': 'FeatEng Iteration', 'Accuracy': 0.8767, 'POD': 0.8520, 'FAR': 0.0572, 'CSI': 0.8101},
    {'Model_Name': 'FeatV5', 'Original_Identifier': 'XGB_V4_T0.5',   'Type': 'FeatEng Iteration', 'Accuracy': 0.8777, 'POD': 0.8529, 'FAR': 0.0564, 'CSI': 0.8115},
    {'Model_Name': 'FeatV6', 'Original_Identifier': 'XGB_V1_Default_T0.5', 'Type': 'FeatEng Iteration', 'Accuracy': 0.8819, 'POD': 0.8880, 'FAR': 0.0819, 'CSI': 0.8228},

    # --- Optimized Model ---
    # Note: Changed 'opt_trial_50' references in code logic to 'opt_trial_322' to match this data entry.
    {'Model_Name': 'opt_trial_322', 'Original_Identifier': 'XGB_V1_Optuna_T0.5', 'Type': 'XGBoost Optimized', 'Accuracy': 0.9456, 'POD': 0.9447, 'FAR': 0.0335, 'CSI': 0.9147},
]

# Convert to DataFrame
df = pd.DataFrame(final_plot_data_values)

# Color settings (original, not used directly in functions after refactor)
colors = {
    'Baseline Product': 'gray',
    'FeatEng Iteration': 'royalblue',
    'XGBoost Optimized': 'crimson'
}

markers = {
    'Baseline Product': 'o',
    'FeatEng Iteration': 's',
    'XGBoost Optimized': '*'
}

# Create separate figures instead of subplots
def create_performance_evolution(fig, df):
    """创建性能演变线图"""
    metrics = ['Accuracy', 'POD', 'CSI', 'FAR']

    if fig is None:
        fig, axs = plt.subplots(2, 2, figsize=(16, 14), sharex=True)
        axs = axs.flatten()
    else:
        fig.set_position([0.1, 0.55, 0.8, 0.35])
        _, axs = plt.subplots(2, 2, figsize=(16, 14), sharex=True)
        axs = axs.flatten()

    # New image-based配色方案
    colors_plot = {
        'Baseline Product': IMG_PALETTE['muted_blue'],
        'FeatEng Iteration': IMG_PALETTE['medium_green'],
        'XGBoost Optimized': IMG_PALETTE['orange']
    }
    
    markers_new = { # Kept original marker styles
        'Baseline Product': 'o',
        'FeatEng Iteration': 'D',
        'XGBoost Optimized': '*'
    }

    feat_eng_df = df[df['Type'] == 'FeatEng Iteration']
    opt_df = df[df['Type'] == 'XGBoost Optimized']

    # The optimized model name from data (assuming one optimized model)
    optimized_model_name = opt_df['Model_Name'].iloc[0] if not opt_df.empty else 'opt_trial_322'


    for i, metric in enumerate(metrics):
        ax = axs[i]
        ax.set_facecolor('#f8f9fa')

        for idx, row in df.iterrows():
            ax.scatter(row['Model_Name'], row[metric],
                        c=colors_plot[row['Type']],
                        marker=markers_new[row['Type']],
                        s=200 if row['Type'] == 'XGBoost Optimized' else 120,
                        zorder=3,
                        edgecolor='black' if row['Type'] == 'XGBoost Optimized' else 'white',
                        linewidth=2 if row['Type'] == 'XGBoost Optimized' else 1,
                        alpha=0.9,
                        )

        if not feat_eng_df.empty:
            ax.plot(feat_eng_df['Model_Name'], feat_eng_df[metric],
                    c=colors_plot['FeatEng Iteration'], linestyle='-', linewidth=3, zorder=2, alpha=0.8)

        if not opt_df.empty and not feat_eng_df.empty:
            stage6_data = feat_eng_df[feat_eng_df['Model_Name'] == 'FeatV6']
            if not stage6_data.empty:
                connect_x = ['FeatV6', optimized_model_name] # Use actual optimized model name
                connect_y = [stage6_data[metric].values[0], opt_df[metric].values[0]]
                ax.plot(connect_x, connect_y, c=colors_plot['XGBoost Optimized'], linestyle='-.', linewidth=3, zorder=2, alpha=0.8)

        # Annotations for certain points
        # Using optimized_model_name for consistency
        for model_label in ['FeatV1', 'FeatV6', optimized_model_name]:
            model_data = df[df['Model_Name'] == model_label]
            if not model_data.empty:
                value = model_data[metric].values[0]
                # Determine color based on type or specific name
                model_type = model_data['Type'].values[0]
                color = colors_plot.get(model_type, IMG_PALETTE['medium_green']) # Default to medium_green
                if model_label == optimized_model_name:
                    color = colors_plot['XGBoost Optimized']
                elif model_type == 'FeatEng Iteration':
                     color = colors_plot['FeatEng Iteration']


                ax.annotate(f'{value:.4f}',
                           (model_label, value),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=14, fontweight='bold', color=color)

        title_text = f'{metric}\n(Higher is better)' if metric != 'FAR' else f'{metric}\n(Lower is better)'
        ax.set_ylabel(title_text, fontsize=20, fontweight='bold', labelpad=35, color='#424242')

        ymin = max(0, df[metric].min() - 0.05)
        ymax = min(1, df[metric].max() + 0.05)
        ax.set_ylim(ymin, ymax)
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        ax.grid(True, linestyle=':', alpha=0.6, color='#E0E0E0')
        ax.tick_params(axis='y', which='major', labelsize=18, colors='#424242')
        ax.tick_params(axis='x', which='major', labelsize=22, colors='#424242')

        x_labels = ax.get_xticklabels()
        for label in x_labels:
            model_name_tick = label.get_text()
            model_type_tick_series = df[df['Model_Name'] == model_name_tick]['Type']
            if not model_type_tick_series.empty:
                model_type_tick = model_type_tick_series.values[0]
                label.set_color(colors_plot[model_type_tick])
                label.set_fontweight('bold')
            
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#424242')
        ax.spines['left'].set_color('#424242')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        arrow_color_ax = colors_plot['FeatEng Iteration']
        arrow_props = dict(facecolor=arrow_color_ax, edgecolor=arrow_color_ax,
                           width=1.5, headwidth=10, headlength=12,
                           shrinkA=0, shrinkB=0)

        ax.annotate('', xy=(xlims[1], ylims[0]), xytext=(-arrow_props['headlength'], 0),
                    textcoords='offset points', arrowprops=arrow_props, xycoords='data', clip_on=False)
        ax.annotate('', xy=(xlims[0], ylims[1]), xytext=(0, -arrow_props['headlength']),
                    textcoords='offset points', arrowprops=arrow_props, xycoords='data', clip_on=False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_plot['Baseline Product'], 
               markersize=15, label='Baseline Products', markeredgecolor='white', markeredgewidth=1),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=colors_plot['FeatEng Iteration'], 
               markersize=15, label='Feature Engineering', markeredgecolor='white', markeredgewidth=1),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=colors_plot['XGBoost Optimized'], 
               markersize=20, label='Optimized Model (50 trials)', markeredgecolor='black', markeredgewidth=1.5), # Kept "50 trials" as per original
        Line2D([0], [0], color=colors_plot['FeatEng Iteration'], linestyle='-', linewidth=3, label='FeatEng Evolution'),
        Line2D([0], [0], color=colors_plot['XGBoost Optimized'], linestyle='-.', linewidth=3, label='Optimization Improvement')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.08),
              fancybox=True, shadow=True, ncol=3, fontsize=18,
              framealpha=0.95, edgecolor='#424242', borderpad=1.2, handletextpad=1.0)

    fig.suptitle('Model Performance Evolution Across Development Stages', 
                 fontsize=26, x= 0.56, y=0.92, fontweight='bold', color=IMG_PALETTE['medium_green']) # Changed title color

    plt.subplots_adjust(left=0.18, bottom=0.20, right=0.95, top=0.88, wspace=0.45, hspace=0.10)
    return fig

def create_radar_chart(ax, df):
    """创建雷达图比较"""
    opt_df = df[df['Type'] == 'XGBoost Optimized']
    optimized_model_name = opt_df['Model_Name'].iloc[0] if not opt_df.empty else 'opt_trial_322'

    selected_models = ['IMERG', 'SM2RAIN', 'FeatV1', 'FeatV6', optimized_model_name]
    radar_df = df[df['Model_Name'].isin(selected_models)].copy()

    metrics = ['Accuracy', 'POD', '1-FAR', 'CSI']
    radar_df['1-FAR'] = 1 - radar_df['FAR']

    # New image-based配色方案 for radar
    radar_model_colors = {
        'IMERG': IMG_PALETTE['muted_blue'],
        'SM2RAIN': IMG_PALETTE['dark_blue'], 
        'FeatV1': IMG_PALETTE['light_green'],
        'FeatV6': IMG_PALETTE['medium_green'],
        optimized_model_name: IMG_PALETTE['orange']
    }
    
    line_styles = {
        'IMERG': ':', 'SM2RAIN': '--', 'FeatV1': '-.', 'FeatV6': '-',
        optimized_model_name: '-'
    }
    line_widths = {
        'IMERG': 2.0, 'SM2RAIN': 2.0, 'FeatV1': 2.0, 'FeatV6': 2.5,
        optimized_model_name: 3.0
    }
    display_names = {
        'IMERG': 'IMERG (Baseline)', 'SM2RAIN': 'SM2RAIN (Baseline)',
        'FeatV1': 'Initial FeatEng (V1)', 'FeatV6': 'Final FeatEng (V6)',
        optimized_model_name: 'Optimized Model'
    }

    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_facecolor('#f8f9fa')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=20, fontweight='bold', color='#424242')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18, color='#424242')
    ax.grid(True, linestyle=':', alpha=0.6, color='#E0E0E0')

    for model_key in selected_models: # Use model_key to iterate selected_models
        if model_key not in radar_df['Model_Name'].values: continue # Skip if model not in data
        model_data = radar_df[radar_df['Model_Name'] == model_key]
        if model_data.empty: continue # Should not happen if selected_models are in radar_df
        
        values = model_data[metrics].values.flatten().tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=line_widths[model_key], linestyle=line_styles[model_key],
                color=radar_model_colors[model_key], label=display_names[model_key], alpha=0.9)
        ax.fill(angles, values, alpha=0.15, color=radar_model_colors[model_key])

    ax.spines['polar'].set_visible(False)
    
    legend_elements_radar = []
    for model_key in selected_models:
        if model_key in display_names: # Ensure key exists
            legend_elements_radar.append(
                Line2D([0], [0], color=radar_model_colors[model_key], linestyle=line_styles[model_key], 
                      linewidth=line_widths[model_key], label=display_names[model_key])
            )
    
    ax.legend(handles=legend_elements_radar, loc='upper right', bbox_to_anchor=(0.15, 0.15),
             fancybox=True, shadow=True, fontsize=18, framealpha=0.95, 
             edgecolor='#424242', borderpad=1.2)

    ax.set_title('Radar Chart Model Comparison', 
                fontsize=26, fontweight='bold', color=IMG_PALETTE['medium_green'], pad=20, # Changed title color
                bbox=dict(boxstyle="round,pad=0.6", fc=IMG_PALETTE['light_blue'], ec=IMG_PALETTE['medium_green'], alpha=0.8)) # Changed bbox colors

    plt.figtext(0.5, 0.02, 'Note: 1-FAR is used to make all metrics follow "higher is better" principle',
               ha='center', fontsize=18, style='italic', color='#424242',
               bbox=dict(boxstyle="round,pad=0.3", fc=IMG_PALETTE['light_blue'], ec=IMG_PALETTE['medium_green'], alpha=0.7)) # Changed bbox colors

    return ax

def create_heatmap(ax, df):
    """创建热力图"""
    opt_df = df[df['Type'] == 'XGBoost Optimized']
    optimized_model_name = opt_df['Model_Name'].iloc[0] if not opt_df.empty else 'opt_trial_322'

    model_order = [
        'CHIRPS', 'CMORPH', 'PERSIANN', 'GSMAP', 'IMERG', 'SM2RAIN',
        'FeatV1', 'FeatV6',
        optimized_model_name 
    ]
    metrics = ['Accuracy', 'POD', 'FAR', 'CSI']
    
    heatmap_df = df[df['Model_Name'].isin(model_order)].copy()
    # Ensure correct order if some models are missing from df
    heatmap_df['order'] = pd.Categorical(heatmap_df['Model_Name'], categories=model_order, ordered=True)
    heatmap_df = heatmap_df.sort_values('order')
        
    heatmap_data = pd.DataFrame(index=heatmap_df['Model_Name']) # Use already sorted model names
    for metric in metrics:
        heatmap_data[metric] = heatmap_df.set_index('Model_Name')[metric].reindex(heatmap_df['Model_Name'])


    heatmap_norm = heatmap_data.copy()
    for col in metrics:
        min_val = heatmap_data[col].min()
        max_val = heatmap_data[col].max()
        if max_val == min_val: # Avoid division by zero if all values are same
             heatmap_norm[col] = 0.5 # or 0 or 1, depending on desired representation
        elif col == 'FAR':
            heatmap_norm[col] = (max_val - heatmap_data[col]) / (max_val - min_val)
        else:
            heatmap_norm[col] = (heatmap_data[col] - min_val) / (max_val - min_val)


    ax.set_facecolor('#f8f9fa')
    
    # Changed cmap to be green-based, as "greener is better"
    cmap_heatmap = sns.light_palette(IMG_PALETTE['medium_green'], as_cmap=True) 
    
    sns.heatmap(heatmap_norm, annot=heatmap_data, fmt=".4f", cmap=cmap_heatmap,
                linewidths=2, linecolor='white', cbar=True, ax=ax,
                annot_kws={"size": 18, "weight": "bold", "color": "#424242"})
    
    ax.set_ylabel('', fontsize=0)
    ax.set_xlabel('', fontsize=0)
    
    y_labels = ax.get_yticklabels()
    for label in y_labels:
        model_name_heatmap = label.get_text()
        if model_name_heatmap in ['CHIRPS', 'CMORPH', 'PERSIANN', 'GSMAP', 'IMERG', 'SM2RAIN']:
            label.set_color(IMG_PALETTE['muted_blue']) # Baseline color
        elif model_name_heatmap in ['FeatV1', 'FeatV6']:
            label.set_color(IMG_PALETTE['medium_green']) # FeatEng color
        elif model_name_heatmap == optimized_model_name:
            label.set_color(IMG_PALETTE['orange']) # Optimized color
        label.set_rotation(0)
        label.set_fontweight('bold')
        label.set_fontsize(20)
    
    x_labels = ax.get_xticklabels()
    for label in x_labels:
        label.set_fontweight('bold')
        label.set_fontsize(20)
        label.set_color('#424242')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18, colors='#424242')
    cbar.set_label('Normalized Performance\n(greener is better for all metrics)', 
                  fontsize=20, fontweight='bold', color='#424242', labelpad=15)
    
    ax.set_title('Model Performance Metrics Comparison', 
                fontsize=26, fontweight='bold', color=IMG_PALETTE['medium_green'], pad=20) # Changed title color
    
    ax.annotate('Note: For FAR, color scale is inverted (greener = better performance)',
               xy=(0.5, -0.15), xycoords='axes fraction',
               ha='center', va='center', fontsize=18, style='italic', color='#424242',
               bbox=dict(boxstyle="round,pad=0.3", fc=IMG_PALETTE['light_blue'], ec=IMG_PALETTE['medium_green'], alpha=0.7)) # Changed bbox colors
    
    return ax


def create_visualization():
    """创建独立的图表而不是使用subplot"""

    print("正在创建性能演变线图...")
    evolution_fig = create_performance_evolution(None, df)
    evolution_fig.savefig('performance_evolution.png', dpi=300, bbox_inches='tight')

    # Note: The original create_visualization did not call create_radar_chart.
    # If you want to generate it, you can add:
    # print("正在创建雷达图...")
    # radar_fig = plt.figure(figsize=(12, 12)) # Adjust size as needed
    # ax_radar = radar_fig.add_subplot(111, polar=True)
    # create_radar_chart(ax_radar, df)
    # radar_fig.savefig('model_metrics_radar.png', dpi=300, bbox_inches='tight')


    print("正在创建热力图...")
    heatmap_fig = plt.figure(figsize=(16, 12)) # Adjusted figsize for better label fit
    ax_heatmap = heatmap_fig.add_subplot(111)
    create_heatmap(ax_heatmap, df)
    # plt.tight_layout(pad=2.0) # tight_layout might conflict with suptitle/legend, or bbox_inches='tight'
    heatmap_fig.savefig('model_metrics_heatmap.png', dpi=300, bbox_inches='tight')


    print("所有图表已创建完成并保存。")
    plt.show()


if __name__ == "__main__":
    create_visualization()