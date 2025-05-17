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
    {'Model_Name': 'FeatEng_Stage1', 'Original_Identifier': 'XGB_V5.1_T0.5', 'Type': 'FeatEng Iteration', 'Accuracy': 0.8526, 'POD': 0.8410, 'FAR': 0.0823, 'CSI': 0.7820},
    {'Model_Name': 'FeatEng_Stage2', 'Original_Identifier': 'XGB_V5_T0.5',   'Type': 'FeatEng Iteration', 'Accuracy': 0.8585, 'POD': 0.8322, 'FAR': 0.0687, 'CSI': 0.7841},
    {'Model_Name': 'FeatEng_Stage3', 'Original_Identifier': 'XGB_V3_T0.5',   'Type': 'FeatEng Iteration', 'Accuracy': 0.8597, 'POD': 0.8337, 'FAR': 0.0681, 'CSI': 0.7859},
    {'Model_Name': 'FeatEng_Stage4', 'Original_Identifier': 'XGB_V2_T0.5',   'Type': 'FeatEng Iteration', 'Accuracy': 0.8767, 'POD': 0.8520, 'FAR': 0.0572, 'CSI': 0.8101},
    {'Model_Name': 'FeatEng_Stage5', 'Original_Identifier': 'XGB_V4_T0.5',   'Type': 'FeatEng Iteration', 'Accuracy': 0.8777, 'POD': 0.8529, 'FAR': 0.0564, 'CSI': 0.8115},
    {'Model_Name': 'FeatEng_Stage6', 'Original_Identifier': 'XGB_V1_Default_T0.5', 'Type': 'FeatEng Iteration', 'Accuracy': 0.8819, 'POD': 0.8880, 'FAR': 0.0819, 'CSI': 0.8228},

    # --- Optimized Model ---
    {'Model_Name': 'XGB_Optimized',  'Original_Identifier': 'XGB_V1_Optuna_T0.5', 'Type': 'XGBoost Optimized', 'Accuracy': 0.9389, 'POD': 0.9356, 'FAR': 0.0357, 'CSI': 0.9043}
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
def create_visualization():
    """创建独立的图表而不是使用subplot"""
    
    # 1. 创建性能演变线图
    print("正在创建性能演变线图...")
    evolution_fig = create_performance_evolution(None, df)
    evolution_fig.savefig('performance_evolution.png', dpi=300, bbox_inches='tight')
    
    # 2. 创建雷达图
    print("正在创建雷达图...")
    radar_fig = plt.figure(figsize=(10, 9))
    ax_radar = radar_fig.add_subplot(111, polar=True)
    create_radar_chart(ax_radar, df)
    plt.tight_layout(pad=2.0)
    plt.savefig('radar_chart_comparison.png', dpi=300, bbox_inches='tight')
    
    # 3. 创建热力图
    print("正在创建热力图...")
    heatmap_fig = plt.figure(figsize=(12, 10))
    ax_heatmap = heatmap_fig.add_subplot(111)
    create_heatmap(ax_heatmap, df)
    plt.tight_layout(pad=2.0)
    plt.savefig('model_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    
    # 4. 创建3D性能图
    print("正在创建3D性能图...")
    fig_3d = plt.figure(figsize=(14, 12))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    create_3d_performance(ax_3d, df)
    plt.tight_layout(pad=2.0)
    plt.savefig('3d_performance_visualization.png', dpi=300, bbox_inches='tight')
    
    print("所有图表已创建完成并保存。")
    plt.show()

def create_performance_evolution(fig, df):
    """创建性能演变线图"""
    metrics = ['Accuracy', 'POD', 'CSI', 'FAR']
    
    # 如果没有提供figure对象，创建一个新的
    if fig is None:
        fig, axs = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
        axs = axs.flatten()
    else:
        fig.set_position([0.1, 0.55, 0.8, 0.35])
        _, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        axs = axs.flatten()
    
    # Get feature engineering stages for line connection
    feat_eng_df = df[df['Type'] == 'FeatEng Iteration']
    opt_df = df[df['Type'] == 'XGBoost Optimized']
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        
        # Plot points for each model
        for idx, row in df.iterrows():
            ax.scatter(row['Model_Name'], row[metric], 
                        c=colors[row['Type']], 
                        marker=markers[row['Type']], 
                        s=150 if row['Type'] == 'XGBoost Optimized' else 80,
                        zorder=3,
                        edgecolor='black' if row['Type'] == 'XGBoost Optimized' else None,
                        linewidth=1.5 if row['Type'] == 'XGBoost Optimized' else 0)
        
        # Connect feature engineering stages with blue line
        if not feat_eng_df.empty:
            ax.plot(feat_eng_df['Model_Name'], feat_eng_df[metric], 
                    c='royalblue', linestyle='-', linewidth=2, zorder=2)
            
        # Connect FeatEng_Stage6 to XGB_Optimized with red dashed line
        if not opt_df.empty and not feat_eng_df.empty:
            # Find the FeatEng_Stage6 data
            stage6_data = feat_eng_df[feat_eng_df['Model_Name'] == 'FeatEng_Stage6']
            if not stage6_data.empty:
                connect_x = ['FeatEng_Stage6', 'XGB_Optimized']
                connect_y = [stage6_data[metric].values[0], opt_df[metric].values[0]]
                ax.plot(connect_x, connect_y, c='crimson', linestyle='-.', linewidth=2, zorder=2)
        
        # Annotations for certain points
        for model in ['FeatEng_Stage1', 'FeatEng_Stage6', 'XGB_Optimized']:
            model_data = df[df['Model_Name'] == model]
            if not model_data.empty:
                ax.annotate(f'{model_data[metric].values[0]:.4f}',
                           (model, model_data[metric].values[0]),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold')
        
        # Set titles and labels
        if metric == 'FAR':
            ax.set_title(f'{metric} (Lower is better)', fontsize=12)
        else:
            ax.set_title(f'{metric} (Higher is better)', fontsize=12)
        
        # Set y-axis limits with padding
        if metric == 'FAR':
            ymin = max(0, df[metric].min() - 0.05)
            ymax = min(1, df[metric].max() + 0.05)
        else:
            ymin = max(0, df[metric].min() - 0.05)
            ymax = min(1, df[metric].max() + 0.05)
        ax.set_ylim(ymin, ymax)
        
        # Format grid and ticks
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Rotate x-tick labels
        plt.setp(ax.get_xticklabels(), rotation=75, ha='right')
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Create a custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Baseline Products'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='royalblue', markersize=10, label='Feature Engineering'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='crimson', markersize=15, label='Optimized Model'),
        Line2D([0], [0], color='royalblue', linestyle='-', linewidth=2, label='FeatEng Evolution'),
        Line2D([0], [0], color='crimson', linestyle='-.', linewidth=2, label='Optimization Improvement')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              fancybox=True, shadow=True, ncol=5)
    
    fig.suptitle('Model Performance Evolution Across Development Stages', fontsize=16, y=0.98)
    
    return fig

def create_radar_chart(ax, df):
    """创建雷达图比较"""
    # 选择要在雷达图中显示的模型
    selected_models = ['FeatEng_Stage1', 'FeatEng_Stage6', 'XGB_Optimized', 'SM2RAIN', 'GSMAP']
    radar_df = df[df['Model_Name'].isin(selected_models)].copy()
    
    # 定义指标并创建1-FAR列
    metrics = ['Accuracy', 'POD', '1-FAR', 'CSI']
    radar_df['1-FAR'] = 1 - radar_df['FAR']
    
    # 指标数量
    N = len(metrics)
    
    # 各指标角度（弧度制）
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 设置图表
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    
    # 设置指标范围
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    
    # 定义不同模型的颜色
    model_colors = {
        'FeatEng_Stage1': '#4169E1',  # 皇家蓝
        'FeatEng_Stage6': '#00008B',  # 深蓝色
        'XGB_Optimized': '#DC143C',   # 深红色
        'SM2RAIN': '#FF8C00',         # 深橙色
        'GSMAP': '#228B22'            # 森林绿
    }
    
    # 定义不同模型的线条样式
    model_lines = {
        'FeatEng_Stage1': '--',
        'FeatEng_Stage6': '-',
        'XGB_Optimized': '-',
        'SM2RAIN': ':',
        'GSMAP': '-.'
    }
    
    # 绘制每个模型
    for model in selected_models:
        model_data = radar_df[radar_df['Model_Name'] == model]
        values = model_data[metrics].values.flatten().tolist()
        values += values[:1]  # 闭合图形
        
        # 绘制线条
        ax.plot(angles, values, linewidth=2.5, linestyle=model_lines[model], 
                color=model_colors[model], label=model, alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=model_colors[model])
    
    # 添加网格线优化
    ax.grid(True, color='gray', linestyle='--', alpha=0.5)
    
    # 添加标注和标题
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1),
              fancybox=True, shadow=True, framealpha=0.7)
    ax.set_title('Radar Chart Comparison of Key Models', fontsize=14, fontweight='bold', pad=20)
    
    # 添加总标题
    plt.figtext(0.5, 0.02, 'Radar Chart displays normalized metrics (higher is better for all axes)', 
               ha='center', fontsize=10, style='italic',
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
    
    return ax

def create_heatmap(ax, df):
    """创建热力图"""
    # 修复使用pivot方法的问题 - 不使用None作为columns参数
    metrics = ['Accuracy', 'POD', 'FAR', 'CSI']
    
    # 直接创建一个新的DataFrame而不是使用pivot方法
    heatmap_data = pd.DataFrame(index=df['Model_Name'])
    for metric in metrics:
        heatmap_data[metric] = df.set_index('Model_Name')[metric]
    
    # Create a copy of the data for normalization
    heatmap_norm = heatmap_data.copy()
    
    # Normalize the data (0-1 scale) based on min-max values
    # For FAR, lower is better, so invert the normalization
    for col in metrics:
        if col == 'FAR':
            heatmap_norm[col] = (heatmap_data[col].max() - heatmap_data[col]) / (heatmap_data[col].max() - heatmap_data[col].min())
        else:
            heatmap_norm[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
    
    # 美化热图显示
    cmap = sns.diverging_palette(10, 150, as_cmap=True)
    sns.heatmap(heatmap_norm, annot=heatmap_data, fmt=".4f", cmap=cmap, 
                linewidths=.5, linecolor='white', cbar=True, ax=ax,
                annot_kws={"size": 9, "weight": "bold"})
    
    # 设置更美观的标题和标签
    ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')
    
    # 美化y轴标签
    plt.setp(ax.get_yticklabels(), rotation=0, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    
    # 添加关于FAR归一化的说明
    ax.annotate('Note: For FAR, color scale is inverted (green = lower FAR)', 
               xy=(0.5, -0.1), xycoords='axes fraction',
               ha='center', va='center', fontsize=10, style='italic',
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
    
    return ax

def create_3d_performance(ax, df):
    """创建3D性能可视化"""
    # 选择所有模型进行3D可视化 - 修改为包含所有模型
    plot_3d_df = df.copy()
    
    # 提取数据
    x = plot_3d_df['POD']
    y = 1 - plot_3d_df['FAR']  # 使用1-FAR以获得更直观的视觉效果
    z = plot_3d_df['CSI']
    
    # 定义不同类型模型的颜色
    color_dict = {
        'Baseline Product': '#808080',       # 灰色
        'FeatEng Iteration': '#4169E1',      # 皇家蓝
        'XGBoost Optimized': '#DC143C'       # 深红色
    }
    
    # 定义不同类型的标记
    marker_dict = {
        'Baseline Product': 'o',
        'FeatEng Iteration': 's',
        'XGBoost Optimized': '*'
    }
    
    # 创建散点图
    for i, row in plot_3d_df.iterrows():
        marker = marker_dict[row['Type']]
        color = color_dict[row['Type']]
        size = 150 if row['Type'] == 'XGBoost Optimized' else (80 if row['Type'] == 'FeatEng Iteration' else 60)
        
        # 绘制点
        ax.scatter(row['POD'], 1-row['FAR'], row['CSI'], 
                  c=color, marker=marker, s=size, label=row['Model_Name'],
                  edgecolor='black' if row['Type'] == 'XGBoost Optimized' else None,
                  linewidth=1.5 if row['Type'] == 'XGBoost Optimized' else 0,
                  alpha=0.8)
        
        # 添加文本标签
        ax.text(row['POD'], 1-row['FAR'], row['CSI'], row['Model_Name'], 
                fontsize=8, ha='center', va='bottom')
        
        # 添加垂直于各平面的虚线来显示点的位置
        # 垂直于XY平面的线
        ax.plot([row['POD'], row['POD']], [1-row['FAR'], 1-row['FAR']], [0, row['CSI']], 
                color=color, linestyle=':', linewidth=0.8, alpha=0.4)
        
        # 垂直于YZ平面的线
        ax.plot([0, row['POD']], [1-row['FAR'], 1-row['FAR']], [row['CSI'], row['CSI']], 
                color=color, linestyle=':', linewidth=0.8, alpha=0.4)
        
        # 垂直于XZ平面的线
        ax.plot([row['POD'], row['POD']], [0, 1-row['FAR']], [row['CSI'], row['CSI']], 
                color=color, linestyle=':', linewidth=0.8, alpha=0.4)
    
    # 设置标签和标题
    ax.set_xlabel('POD', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('1-FAR', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('CSI', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('3D Performance Visualization of All Models', fontsize=16, fontweight='bold')
    
    # 设置轴范围 - 突出右上角区域
    ax.set_xlim(0.4, 1)
    ax.set_ylim(0.6, 1)
    ax.set_zlim(0.3, 1)
    
    # 添加关于K-Fold AUC的注释
    ax.text2D(0.05, 0.95, "K-Fold Average AUC ≈ 0.984 for XGB_Optimized", 
              transform=ax.transAxes, fontsize=11, 
              bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 移除图例中的重复标签
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # 手动创建模型类型的图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Baseline Products'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='royalblue', markersize=8, label='Feature Engineering'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='crimson', markersize=12, label='Optimized Model')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.85),
              fancybox=True, shadow=True)
    
    # 设置视角以突出右上角位置的点 - 调整为更好地显示最佳点
    ax.view_init(elev=20, azim=120)
    
    # 美化3D图表
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 提高轴线和刻度的可见度
    ax.xaxis.line.set_color('black')
    ax.yaxis.line.set_color('black')
    ax.zaxis.line.set_color('black')
    
    try:
        # 添加参考平面 - 用浅色透明平面指示位置
        # XY平面
        xx, yy = np.meshgrid(np.linspace(0.4, 1, 2), np.linspace(0.6, 1, 2))
        zz = np.zeros(xx.shape)
        ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1)
        
        # YZ平面
        yy, zz = np.meshgrid(np.linspace(0.6, 1, 2), np.linspace(0.3, 1, 2))
        xx = np.zeros(yy.shape)
        ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1)
        
        # XZ平面
        xx, zz = np.meshgrid(np.linspace(0.4, 1, 2), np.linspace(0.3, 1, 2))
        yy = np.zeros(xx.shape)
        ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1)
        
        # 添加一个指示"更好"方向的箭头
        ax.quiver3D(0.45, 0.7, 0.4, 0.2, 0.1, 0.2, 
                  color='green', arrow_length_ratio=0.1, alpha=0.6)
        ax.text(0.65, 0.8, 0.6, "Better", color='green', fontweight='bold')
    except Exception as e:
        print(f"注意: 绘制3D参考面时出现错误: {str(e)}")
        # 使用替代方法创建参考平面
        try:
            # XY平面
            verts = [[(0.4, 0.6, 0), (1, 0.6, 0), (1, 1, 0), (0.4, 1, 0)]]
            poly = Poly3DCollection(verts, alpha=0.1, facecolor='gray')
            ax.add_collection3d(poly)
            
            # YZ平面
            verts = [[(0, 0.6, 0.3), (0, 1, 0.3), (0, 1, 1), (0, 0.6, 1)]]
            poly = Poly3DCollection(verts, alpha=0.1, facecolor='gray')
            ax.add_collection3d(poly)
            
            # XZ平面
            verts = [[(0.4, 0, 0.3), (1, 0, 0.3), (1, 0, 1), (0.4, 0, 1)]]
            poly = Poly3DCollection(verts, alpha=0.1, facecolor='gray')
            ax.add_collection3d(poly)
            
            # 使用较简单的箭头方法
            ax.text(0.65, 0.8, 0.6, "Better →", color='green', fontweight='bold')
        except Exception as e:
            print(f"注意: 创建参考多边形时出现错误: {str(e)}")
    
    return ax

if __name__ == "__main__":
    create_visualization()