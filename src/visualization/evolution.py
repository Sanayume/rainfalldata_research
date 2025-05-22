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
    {'Model_Name': 'opt_trial_50',  'Original_Identifier': 'XGB_V1_Optuna_T0.5', 'Type': 'XGBoost Optimized', 'Accuracy': 0.9389, 'POD': 0.9356, 'FAR': 0.0357, 'CSI': 0.9043},
    #{'Model_Name': 'opt_trial_300',  'Original_Identifier': 'XGB_V1_Optuna_T0.5', 'Type': 'XGBoost Optimized', 'Accuracy': 0.9389, 'POD': 0.9356, 'FAR': 0.0357, 'CSI': 0.9043}
]

# Convert to DataFrame
df = pd.DataFrame(final_plot_data_values)

# Color settings
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

    # 如果没有提供figure对象，创建一个新的
    if fig is None:
        fig, axs = plt.subplots(2, 2, figsize=(16, 14), sharex=True)
        axs = axs.flatten()
    else:
        fig.set_position([0.1, 0.55, 0.8, 0.35])
        _, axs = plt.subplots(2, 2, figsize=(16, 14), sharex=True)
        axs = axs.flatten()

    # Google风格配色方案
    colors_new = {
        'Baseline Product': '#616161',  # 深灰色
        'FeatEng Iteration': '#4285F4',  # 谷歌蓝
        'XGBoost Optimized': '#EA4335'   # 谷歌红
    }
    
    # 标记样式
    markers_new = {
        'Baseline Product': 'o',
        'FeatEng Iteration': 'D',  # 钻石形
        'XGBoost Optimized': '*'
    }

    # Get feature engineering stages for line connection
    feat_eng_df = df[df['Type'] == 'FeatEng Iteration']
    opt_df = df[df['Type'] == 'XGBoost Optimized']

    for i, metric in enumerate(metrics):
        ax = axs[i]
        
        # 设置背景颜色为浅灰色，增强可读性
        ax.set_facecolor('#f8f9fa')

        # Plot points for each model
        for idx, row in df.iterrows():
            ax.scatter(row['Model_Name'], row[metric],
                        c=colors_new[row['Type']],
                        marker=markers_new[row['Type']],
                        s=200 if row['Type'] == 'XGBoost Optimized' else 120,
                        zorder=3,
                        edgecolor='black' if row['Type'] == 'XGBoost Optimized' else 'white',
                        linewidth=2 if row['Type'] == 'XGBoost Optimized' else 1,
                        alpha=0.9,
                        )

        # Connect feature engineering stages with blue line
        if not feat_eng_df.empty:
            ax.plot(feat_eng_df['Model_Name'], feat_eng_df[metric],
                    c='#4285F4', linestyle='-', linewidth=3, zorder=2, alpha=0.8)

        # Connect FeatV6 to XGB_Optimized with red dashed line
        if not opt_df.empty and not feat_eng_df.empty:
            # Find the FeatV6 data
            stage6_data = feat_eng_df[feat_eng_df['Model_Name'] == 'FeatV6']
            if not stage6_data.empty:
                connect_x = ['FeatV6', 'opt_trial_50']
                connect_y = [stage6_data[metric].values[0], opt_df[metric].values[0]]
                ax.plot(connect_x, connect_y, c='#EA4335', linestyle='-.', linewidth=3, zorder=2, alpha=0.8)

        # Annotations for certain points without text boxes
        for model in ['FeatV1', 'FeatV6', 'opt_trial_50']:
            model_data = df[df['Model_Name'] == model]
            if not model_data.empty:
                value = model_data[metric].values[0]
                color = '#EA4335' if model == 'opt_trial_50' else '#4285F4'
                
                ax.annotate(f'{value:.4f}',
                           (model, value),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=14, fontweight='bold', color=color)

        # Set titles and labels with improved styling
        title_text = ''
        if metric == 'FAR':
            title_text = f'{metric}\n(Lower is better)'
        else:
            title_text = f'{metric}\n(Higher is better)'
        
        ax.set_ylabel(title_text, fontsize=20, fontweight='bold', labelpad=35, color='#424242')

        # Set y-axis limits with padding
        if metric == 'FAR':
            ymin = max(0, df[metric].min() - 0.05)
            ymax = min(1, df[metric].max() + 0.05)
        else:
            ymin = max(0, df[metric].min() - 0.05)
            ymax = min(1, df[metric].max() + 0.05)
        ax.set_ylim(ymin, ymax)
        ylims = ax.get_ylim() # 获取更新后的ylims
        xlims = ax.get_xlim() # 获取更新后的xlims

        # Format grid and ticks
        ax.grid(True, linestyle=':', alpha=0.6, color='#E0E0E0')
        ax.tick_params(axis='y', which='major', labelsize=18, colors='#424242')
        ax.tick_params(axis='x', which='major', labelsize=22, colors='#424242')

        # 为X轴标签上色
        x_labels = ax.get_xticklabels()
        for label in x_labels:
            model_name = label.get_text()
            model_type = df[df['Model_Name'] == model_name]['Type'].values[0] if model_name in df['Model_Name'].values else ''
            if model_type:
                label.set_color(colors_new[model_type])
                label.set_fontweight('bold')
            
        # Rotate x-tick labels with improved alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 添加带箭头的坐标轴
        # 隐藏默认坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#424242')
        ax.spines['left'].set_color('#424242')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        arrow_color = colors_new['FeatEng Iteration'] # Google Blue for arrows
        
        # 定义箭头属性
        # headlength 和 headwidth 控制箭头头部的大小 (单位: points)
        # width 控制箭头线的宽度 (单位: points)
        arrow_props = dict(facecolor=arrow_color, edgecolor=arrow_color,
                           width=1.5, headwidth=10, headlength=12, # 调整了头部大小
                           shrinkA=0, shrinkB=0) # shrinkA/B=0 确保箭头紧密连接

        # x_ticks_pos = ax.get_xticks() # 不再直接使用ticks位置来确定箭头末端

        # X-轴箭头: 尖端在x轴的末端（最大x值）和y轴的起点（最小y值）
        # 尾部从尖端向左偏移 headlength points
        ax.annotate('',
                    xy=(xlims[1], ylims[0]),  # 箭头尖端 (数据坐标: x轴最大值, y轴最小值)
                    xytext=(-arrow_props['headlength'], 0),  # 箭头尾部相对尖端的偏移 (points)
                    textcoords='offset points',  # xytext 的坐标系
                    arrowprops=arrow_props,
                    xycoords='data',  # xy 的坐标系
                    clip_on=False)

        # Y-轴箭头: 尖端在x轴的起点（最小x值）和y轴的末端（最大y值）
        # 尾部从尖端向下偏移 headlength points
        ax.annotate('',
                    xy=(xlims[0], ylims[1]),  # 箭头尖端 (数据坐标: x轴最小值, y轴最大值)
                    xytext=(0, -arrow_props['headlength']),  # 箭头尾部相对尖端的偏移 (points)
                    textcoords='offset points',  # xytext 的坐标系
                    arrowprops=arrow_props,
                    xycoords='data',  # xy 的坐标系
                    clip_on=False)

    # Create a custom legend with improved styling
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_new['Baseline Product'], 
               markersize=15, label='Baseline Products', markeredgecolor='white', markeredgewidth=1),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=colors_new['FeatEng Iteration'], 
               markersize=15, label='Feature Engineering', markeredgecolor='white', markeredgewidth=1),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=colors_new['XGBoost Optimized'], 
               markersize=20, label='Optimized Model (50 trials)', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], color=colors_new['FeatEng Iteration'], linestyle='-', linewidth=3, label='FeatEng Evolution'),
        Line2D([0], [0], color=colors_new['XGBoost Optimized'], linestyle='-.', linewidth=3, label='Optimization Improvement')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.06),
              fancybox=True, shadow=True, ncol=3, fontsize=18,
              framealpha=0.95, edgecolor='#424242', borderpad=1.2, handletextpad=1.0)

    # 将总图标题移到最上面，使用谷歌蓝
    fig.suptitle('Model Performance Evolution Across Development Stages', 
                 fontsize=26, y=0.98, fontweight='bold', color='#4285F4',
                 bbox=dict(boxstyle="round,pad=0.6", fc='#E8F0FE', ec='#4285F4', alpha=0.8))

    # 调整subplots_adjust参数以获得更好的布局
    plt.subplots_adjust(left=0.18, bottom=0.20, right=0.95, top=0.88, wspace=0.45, hspace=0.10)

    return fig

def create_radar_chart(ax, df):
    """创建雷达图比较"""
    # 选择要在雷达图中显示的模型
    selected_models = ['IMERG', 'SM2RAIN', 'FeatV1', 'FeatV6', 'opt_trial_50']
    radar_df = df[df['Model_Name'].isin(selected_models)].copy()

    # 定义指标并创建1-FAR列
    metrics = ['Accuracy', 'POD', '1-FAR', 'CSI']
    radar_df['1-FAR'] = 1 - radar_df['FAR']

    # Google风格配色方案
    colors_new = {
        'IMERG': '#616161',       # 深灰色 (Baseline)
        'SM2RAIN': '#616161',     # 深灰色 (Baseline)
        'FeatV1': '#4285F4',      # 谷歌蓝 (FeatEng初始)
        'FeatV6': '#4285F4',      # 谷歌蓝 (FeatEng最终)
        'opt_trial_50': '#EA4335' # 谷歌红 (优化模型)
    }
    
    # 定义不同模型的线条样式
    line_styles = {
        'IMERG': ':',
        'SM2RAIN': '--',
        'FeatV1': '-.',
        'FeatV6': '-',
        'opt_trial_50': '-'
    }
    
    # 线宽
    line_widths = {
        'IMERG': 2.0,
        'SM2RAIN': 2.0,
        'FeatV1': 2.0,
        'FeatV6': 2.5,
        'opt_trial_50': 3.0
    }
    
    # 模型显示名称
    display_names = {
        'IMERG': 'IMERG (Baseline)',
        'SM2RAIN': 'SM2RAIN (Baseline)',
        'FeatV1': 'Initial FeatEng (V1)',
        'FeatV6': 'Final FeatEng (V6)',
        'opt_trial_50': 'Optimized Model'
    }

    # 指标数量
    N = len(metrics)

    # 各指标角度（弧度制）
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 设置背景颜色为浅灰色，增强可读性
    ax.set_facecolor('#f8f9fa')
    
    # 设置图表
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=20, fontweight='bold', color='#424242')

    # 设置指标范围和网格
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18, color='#424242')
    
    # 添加网格线优化
    ax.grid(True, linestyle=':', alpha=0.6, color='#E0E0E0')

    # 绘制每个模型
    for model in selected_models:
        model_data = radar_df[radar_df['Model_Name'] == model]
        values = model_data[metrics].values.flatten().tolist()
        values += values[:1]  # 闭合图形

        # 绘制线条
        ax.plot(angles, values, linewidth=line_widths[model], linestyle=line_styles[model],
                color=colors_new[model], label=display_names[model], alpha=0.9)
        
        # 填充区域
        ax.fill(angles, values, alpha=0.15, color=colors_new[model])

    # 美化雷达图框架
    # 隐藏框架线条
    ax.spines['polar'].set_visible(False)
    
    # 创建一个带样式的图例
    legend_elements = []
    for model in selected_models:
        legend_elements.append(
            Line2D([0], [0], color=colors_new[model], linestyle=line_styles[model], 
                  linewidth=line_widths[model], label=display_names[model])
        )
    
    # 添加图例
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.15, 0.15),
             fancybox=True, shadow=True, fontsize=18, framealpha=0.95, 
             edgecolor='#424242', borderpad=1.2)

    # 添加标题
    ax.set_title('Radar Chart Model Comparison', 
                fontsize=26, fontweight='bold', color='#4285F4', pad=20,
                bbox=dict(boxstyle="round,pad=0.6", fc='#E8F0FE', ec='#4285F4', alpha=0.8))

    # 添加说明文本
    plt.figtext(0.5, 0.02, 'Note: 1-FAR is used to make all metrics follow "higher is better" principle',
               ha='center', fontsize=18, style='italic', color='#424242',
               bbox=dict(boxstyle="round,pad=0.3", fc="#E8F0FE", ec="#4285F4", alpha=0.7))

    return ax

def create_heatmap(ax, df):
    """创建热力图"""
    # 选择要显示的模型
    model_order = [
        'CHIRPS', 'CMORPH', 'PERSIANN', 'GSMAP', 'IMERG', 'SM2RAIN',  # Baseline
        'FeatV1', 'FeatV6',  # Feature Engineering
        'opt_trial_50'  # Optimized
    ]
    metrics = ['Accuracy', 'POD', 'FAR', 'CSI']
    
    # 筛选并排序数据
    heatmap_df = df[df['Model_Name'].isin(model_order)].copy()
    heatmap_df['order'] = heatmap_df['Model_Name'].map({m: i for i, m in enumerate(model_order)})
    heatmap_df = heatmap_df.sort_values('order')
    
    # 准备热图数据
    heatmap_data = pd.DataFrame(index=heatmap_df['Model_Name'])
    for metric in metrics:
        heatmap_data[metric] = heatmap_df.set_index('Model_Name')[metric]

    # 为了可视化效果，创建归一化数据
    heatmap_norm = heatmap_data.copy()
    for col in metrics:
        if col == 'FAR':  # FAR是越低越好
            heatmap_norm[col] = (heatmap_data[col].max() - heatmap_data[col]) / (heatmap_data[col].max() - heatmap_data[col].min())
        else:  # 其他指标是越高越好
            heatmap_norm[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())

    # 设置背景颜色
    ax.set_facecolor('#f8f9fa')
    
    # 定义颜色映射
    cmap = sns.diverging_palette(10, 150, as_cmap=True)  # 使用蓝到红的渐变色
    
    # 绘制热图
    sns.heatmap(heatmap_norm, annot=heatmap_data, fmt=".4f", cmap=cmap,
                linewidths=2, linecolor='white', cbar=True, ax=ax,
                annot_kws={"size": 18, "weight": "bold", "color": "#424242"})
    
    # 设置坐标轴标签样式
    ax.set_ylabel('', fontsize=0)  # 不显示y轴标签
    ax.set_xlabel('', fontsize=0)  # 不显示x轴标签
    
    # 自定义y轴标签颜色
    y_labels = ax.get_yticklabels()
    for label in y_labels:
        model_name = label.get_text()
        if model_name in ['CHIRPS', 'CMORPH', 'PERSIANN', 'GSMAP', 'IMERG', 'SM2RAIN']:
            label.set_color('#EA4335')  # 红色代表基线模型
        elif model_name in ['FeatV1', 'FeatV6']:
            label.set_color('#4285F4')  # 蓝色代表特征工程模型
        elif model_name == 'opt_trial_50':
            label.set_color('#616161')  # 灰色代表优化模型

        label.set_rotation(0)
        label.set_fontweight('bold')
        label.set_fontsize(20)
    
    # 自定义x轴标签样式
    x_labels = ax.get_xticklabels()
    for label in x_labels:
        label.set_fontweight('bold')
        label.set_fontsize(20)
        label.set_color('#424242')
    
    # 修改colorbar样式
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18, colors='#424242')
    cbar.set_label('Normalized Performance\n(greener is better for all metrics)', 
                  fontsize=20, fontweight='bold', color='#424242', labelpad=15)
    
    # 设置标题
    ax.set_title('Model Performance Metrics Comparison', 
                fontsize=26, fontweight='bold', color='#4285F4', pad=20)
    
    # 添加说明注释
    ax.annotate('Note: For FAR, color scale is inverted (greener = better performance)',
               xy=(0.5, -0.15), xycoords='axes fraction',
               ha='center', va='center', fontsize=18, style='italic', color='#424242',
               bbox=dict(boxstyle="round,pad=0.3", fc="#E8F0FE", ec="#4285F4", alpha=0.7))
    
    return ax


def create_visualization():
    """创建独立的图表而不是使用subplot"""

    # 1. 创建性能演变线图
    print("正在创建性能演变线图...")
    evolution_fig = create_performance_evolution(None, df)
    evolution_fig.savefig('performance_evolution.png', dpi=300, bbox_inches='tight')


    # 3. 创建热力图
    print("正在创建热力图...")
    heatmap_fig = plt.figure(figsize=(12, 10))
    ax_heatmap = heatmap_fig.add_subplot(111)
    create_heatmap(ax_heatmap, df)
    plt.tight_layout(pad=2.0)
    plt.savefig('model_metrics_heatmap.png', dpi=300, bbox_inches='tight')


    print("所有图表已创建完成并保存。")
    plt.show()


if __name__ == "__main__":
    create_visualization()