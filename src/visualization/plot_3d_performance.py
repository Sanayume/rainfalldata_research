"""
绘制模型性能的3D可视化图
此文件专注于创建更详细的3D性能视图，展示不同模型在POD、FAR和CSI三个关键指标上的表现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Verdana', 'Geneva', 'Lucid', 'Helvetica', 'Avant Garde', 'sans-serif']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (14, 12)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 数据定义 - 与主文件相同的数据
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

# 颜色和标记定义
color_dict = {
    'Baseline Product': '#808080',       # 灰色
    'FeatEng Iteration': '#4169E1',      # 皇家蓝
    'XGBoost Optimized': '#DC143C'       # 深红色
}

marker_dict = {
    'Baseline Product': 'o',
    'FeatEng Iteration': 's', 
    'XGBoost Optimized': '*'
}

def create_3d_plot(save_path=None, rotate_view=False, show_plot=True):
    """
    创建交互式3D性能图表
    
    参数:
    save_path : str, optional
        保存图像的路径，如果为None则不保存
    rotate_view : bool, optional
        是否创建多角度视图并保存
    show_plot : bool, optional
        是否显示图表
    """
    # 将数据转换为DataFrame
    df = pd.DataFrame(final_plot_data_values)
    
    # 创建图形和坐标轴
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制每个点以及投影线
    for i, row in df.iterrows():
        # 获取点的属性
        model_type = row['Type']
        color = color_dict[model_type]
        marker = marker_dict[model_type]
        
        # 设置点的大小
        size = 200 if model_type == 'XGBoost Optimized' else (100 if model_type == 'FeatEng Iteration' else 80)
        
        # 计算点的坐标，对于FAR使用1-FAR使得值越高越好
        x, y, z = row['POD'], 1-row['FAR'], row['CSI']
        
        # 绘制3D散点
        ax.scatter(x, y, z, 
                  c=color, marker=marker, s=size, 
                  edgecolor='black' if model_type == 'XGBoost Optimized' else None,
                  linewidth=1.5 if model_type == 'XGBoost Optimized' else 0,
                  alpha=0.9)
        
        # 添加点的标签
        label_offset = 0.01  # 调整标签位置
        ax.text(x, y, z+label_offset, row['Model_Name'], 
                fontsize=9, ha='center', va='bottom', 
                fontweight='bold' if model_type == 'XGBoost Optimized' else 'normal')
        
        # 添加投影线到底面
        alpha_projection = 0.3  # 投影线的透明度
        lw_projection = 0.8     # 投影线的宽度
        
        # 到XY面(z=0)的投影线
        ax.plot([x, x], [y, y], [0, z], 
                color=color, linestyle=':', linewidth=lw_projection, alpha=alpha_projection)
        
        # 到YZ面(x=0)的投影线
        ax.plot([0, x], [y, y], [z, z], 
                color=color, linestyle=':', linewidth=lw_projection, alpha=alpha_projection)
        
        # 到XZ面(y=0)的投影线
        ax.plot([x, x], [0, y], [z, z], 
                color=color, linestyle=':', linewidth=lw_projection, alpha=alpha_projection)
    
    # 添加坐标平面用于参考
    xlim, ylim, zlim = (0.4, 1), (0.6, 1), (0.3, 1)  # 设置轴的范围
    
    # 创建参考平面
    try:
        # XY平面(Z=0)
        xx, yy = np.meshgrid(np.linspace(*xlim, 2), np.linspace(*ylim, 2))
        zz = np.zeros(xx.shape)
        ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.1)
        
        # XZ平面(Y=0)
        xx, zz = np.meshgrid(np.linspace(*xlim, 2), np.linspace(*zlim, 2))
        yy = np.zeros(xx.shape)
        ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.1)
        
        # YZ平面(X=0)
        yy, zz = np.meshgrid(np.linspace(*ylim, 2), np.linspace(*zlim, 2))
        xx = np.zeros(yy.shape)
        ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.1)
    except Exception as e:
        print(f"警告: 无法创建参考平面: {str(e)}")
        # 备用方法：使用多边形
        verts = [
            [(xlim[0], ylim[0], 0), (xlim[1], ylim[0], 0), (xlim[1], ylim[1], 0), (xlim[0], ylim[1], 0)],  # XY平面
            [(xlim[0], 0, zlim[0]), (xlim[1], 0, zlim[0]), (xlim[1], 0, zlim[1]), (xlim[0], 0, zlim[1])],  # XZ平面
            [(0, ylim[0], zlim[0]), (0, ylim[1], zlim[0]), (0, ylim[1], zlim[1]), (0, ylim[0], zlim[1])]   # YZ平面
        ]
        for v in verts:
            poly = Poly3DCollection([v], alpha=0.1, facecolor='lightgray')
            ax.add_collection3d(poly)
    
    # 标记最优方向
    ax.quiver(xlim[0]+0.05, ylim[0]+0.05, zlim[0]+0.05, 
             0.2, 0.1, 0.2, color='green', arrow_length_ratio=0.1)
    ax.text(xlim[0]+0.25, ylim[0]+0.15, zlim[0]+0.25, 
           "Better Performance", color='green', fontweight='bold')
    
    # 设置轴标签
    ax.set_xlabel('POD', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('1-FAR', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('CSI', fontsize=14, fontweight='bold', labelpad=10)
    
    # 设置轴范围
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # 设置轴刻度
    ax.set_xticks(np.arange(xlim[0], xlim[1]+0.1, 0.1))
    ax.set_yticks(np.arange(ylim[0], ylim[1]+0.1, 0.1))
    ax.set_zticks(np.arange(zlim[0], zlim[1]+0.1, 0.1))
    
    # 美化坐标轴
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    
    # 添加标题
    plt.title('3D Performance Visualization of Rainfall Models', fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Baseline Products'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='royalblue', markersize=10, label='Feature Engineering'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='crimson', markersize=15, label='Optimized Model')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
             fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
    
    # 添加文字注解
    ax.text2D(0.02, 0.98, "K-Fold Average AUC ≈ 0.984 for XGB_Optimized", 
             transform=ax.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 调整视角以突出最佳点
    ax.view_init(elev=25, azim=135)
    
    # 保存具有多个视角的图像
    if rotate_view and save_path:
        base_name = os.path.splitext(save_path)[0]
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            ax.view_init(elev=30, azim=angle)
            plt.savefig(f"{base_name}_angle_{angle}.png")
    
    # 如果提供了保存路径，保存图像
    if save_path:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    # 显示图像
    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

def main():
    """主函数"""
    # 定义保存路径
    save_dir = "d:\\desktop\\gitclone\\rainfalldata_research\\results\\figures"
    # 确保目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 创建并保存3D图表
    save_path = os.path.join(save_dir, "rainfall_model_3d_performance.png")
    create_3d_plot(save_path=save_path, rotate_view=False)
    
    print("完成!")

if __name__ == "__main__":
    main()
