import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import MinMaxScaler # For normalization
from pandas.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D # For 3D plot
import os # For saving figures to a specific path

# 导入你的绘图类
from academic_plotter import AcademicStylePlotter # 假设类保存在 academic_plotter.py

# --- 初始化绘图器 ---
# 选择一个基础字号，9pt 或 10pt 比较常见
plotter = AcademicStylePlotter(font_family='Arial', base_font_size=9)

# --- 定义图像保存路径 ---
FIGURE_SAVE_DIR = "comparison_figures"
if not os.path.exists(FIGURE_SAVE_DIR):
    os.makedirs(FIGURE_SAVE_DIR)

# --- Data Preparation (Same as before) ---
data = {
    'model_name_raw': [
        'K-Nearest Neighbors (Tuned on Subset)',
        'Support Vector Machine (Tuned on Subset)',
        'Random Forest',
        'LightGBM',
        'Gaussian Naive Bayes (Default)',
        'xgboost(defautl)'
    ],
    'accuracy': [0.7917, 0.8021, 0.8408, 0.8366, 0.7019, 0.8819],
    'pod':      [0.7839, 0.7496, 0.8378, 0.8221, 0.5799, 0.8880],
    'far':      [0.1308, 0.0819, 0.1001, 0.0929, 0.0909, 0.0819],
    'csi':      [0.7012, 0.7026, 0.7665, 0.7582, 0.5481, 0.8228],
    'fn':       [142520, 165169, 106938, 117342, 277059, 73133],
    'fp':       [77795,  44111,  61430,  55529,  38237,  51763],
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

# --- 全局绘图设置已经被 AcademicStylePlotter 处理 ---
# sns.set_theme(style="whitegrid") # plotter._apply_base_style() 会覆盖或提供类似效果
# plt.rcParams... 这些现在由 plotter 类管理


# --- 1. Heatmap of All 6 Metrics ---
scaler = MinMaxScaler()
df_normalized_for_heatmap = df_compare.copy()
df_normalized_for_heatmap[metrics_to_compare] = scaler.fit_transform(df_compare[metrics_to_compare])

df_color_scaled = df_normalized_for_heatmap.copy()
for metric_lower_is_better in ['far', 'fn', 'fp']:
    df_color_scaled[metric_lower_is_better] = 1 - df_color_scaled[metric_lower_is_better]

fig1, ax1 = plt.subplots(figsize=(10, 7)) # 调整figsize
# 使用 plotter 获取色板，例如一个蓝-绿-黄的发散色板可能比 viridis 更适合"好坏"
# 或者一个从红(差)到绿(好)的色板
cmap_heatmap = plotter.get_colormap('correlation') # 'correlation' 是 红-黄-绿

sns.heatmap(
    df_color_scaled,
    annot=df_compare,
    fmt=".4g",
    cmap=cmap_heatmap, # 使用 plotter 的色板
    linewidths=.5,
    linecolor='gray',
    cbar_kws={'label': 'Normalized Metric (Higher is "Better" Color)'},
    ax=ax1 # 指定ax
)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=plotter.base_font_size)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right', fontsize=plotter.base_font_size)
plotter.set_title_for_subplot(ax1, 'Comprehensive Model Performance Heatmap', fontsize_offset=2)
fig1.suptitle('Colors: Normalized, Annotations: Original Values', fontsize=plotter.base_font_size, y=0.93) # 副标题
plt.tight_layout(rect=[0, 0, 1, 0.92]) # 为标题留出空间
plotter.save_figure(fig1, os.path.join(FIGURE_SAVE_DIR, 'model_performance_heatmap_styled.png'), dpi=300)
plt.show()


# --- 2. Parallel Coordinates Plot ---
df_parallel = df_compare.copy()
scaler_parallel = MinMaxScaler()
df_parallel[metrics_to_compare] = scaler_parallel.fit_transform(df_parallel[metrics_to_compare])
df_parallel_plot = df_parallel.reset_index()

fig2, ax2 = plt.subplots(figsize=(12, 6)) # 调整figsize
num_models_par = len(df_parallel_plot['model_name'].unique())
# 获取定性色板
parallel_colors = plotter.get_colormap('qualitative_prod_lines', n_colors=num_models_par) # 使用类中定义的，或tab10

parallel_coordinates(df_parallel_plot, 'model_name', ax=ax2, color=parallel_colors, linewidth=2, alpha=0.85)
plotter.set_title_for_subplot(ax2, 'Parallel Coordinates of Normalized Model Metrics', fontsize_offset=2)
ax2.set_ylabel('Normalized Metric Value (0 to 1)')
ax2.tick_params(axis='x', rotation=15, labelsize=plotter.base_font_size-1)
# ax2.grid(True, alpha=0.6, linestyle=':') # plotter类已设置网格
ax2.legend(title='Models', bbox_to_anchor=(1.03, 1), loc='upper left', frameon=True) # 保持图例边框
plt.tight_layout(rect=[0, 0, 0.83, 0.95])
plotter.save_figure(fig2, os.path.join(FIGURE_SAVE_DIR, 'model_performance_parallel_coordinates_styled.png'), dpi=300)
plt.show()
# (Note remains the same)


# --- 3. Enhanced Grouped Bar Chart ---
rate_metrics = ['accuracy', 'pod', 'csi']
df_rates_for_bar = df_compare[rate_metrics].copy()
df_rates_for_bar['1-FAR'] = 1 - df_compare['far'] # Higher is better
metrics_for_enhanced_bar = ['accuracy', 'pod', 'csi', '1-FAR']
df_rates_for_bar = df_rates_for_bar[metrics_for_enhanced_bar]
df_rates_melted = df_rates_for_bar.reset_index().melt(id_vars='model_name', var_name='Metric', value_name='Value')

fig3, ax3 = plt.subplots(figsize=(13, 7)) # 调整figsize
# 获取美观的调色板
bar_palette = plotter.get_colormap('error_types_stack', n_colors=len(metrics_for_enhanced_bar)) # 或其他定性色板

sns.barplot(x='model_name', y='Value', hue='Metric', data=df_rates_melted, palette=bar_palette, edgecolor='black', linewidth=0.7, ax=ax3)

for p in ax3.patches:
    ax3.annotate(format(p.get_height(), '.3f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 8), # 调整偏移
                   textcoords = 'offset points',
                   fontsize=plotter.base_font_size - 1.5, color='black')

plotter.set_title_for_subplot(ax3, 'Key Model Performance Metrics Comparison', fontsize_offset=3)
ax3.set_xlabel('Model Name')
ax3.set_ylabel('Metric Value (Higher is Better)')
ax3.tick_params(axis='x', rotation=25, labelsize=plotter.base_font_size) # 调整X轴标签
ax3.set_ylim(0, df_rates_melted['Value'].max() * 1.18 if not df_rates_melted.empty else 1.18)
ax3.legend(title='Metric', title_fontsize=plotter.base_font_size, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
# ax3.grid(axis='y', linestyle='--', alpha=0.7) # plotter类已设置网格
plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plotter.save_figure(fig3, os.path.join(FIGURE_SAVE_DIR, 'model_performance_enhanced_bar_styled.png'), dpi=300)
plt.show()


# --- 4. 3D Bar Chart ---
metrics_3d = ['accuracy', 'pod', 'csi']
df_3d = df_compare[metrics_3d].copy()

fig4 = plt.figure(figsize=(12, 9)) # 调整figsize
ax4 = fig4.add_subplot(111, projection='3d')

x_labels_3d = df_3d.index.to_list()
num_models_3d = len(x_labels_3d)
num_metrics_3d_count = len(metrics_3d)
x_pos_3d = np.arange(num_models_3d)
y_pos_metric_3d = np.arange(num_metrics_3d_count) # Y轴代表指标类型

# 获取3D柱状图颜色
colors_3d_bars = plotter.get_colormap('qualitative_prod_lines', n_colors=num_metrics_3d_count) # 例如

_x, _y = np.meshgrid(x_pos_3d, y_pos_metric_3d)
x, y = _x.ravel(), _y.ravel()

top = df_3d.values.ravel() # 所有值拉平，顺序要对应 x,y
bottom = np.zeros_like(top)
width = depth = 0.6 # 柱子的宽度和深度

# bar3d 颜色参数需要一个列表，每个柱子一个颜色
bar_colors_list = []
for i in range(num_metrics_3d_count): # 每个指标一种颜色
    for _ in range(num_models_3d): # 每个模型都用这种颜色
        bar_colors_list.append(colors_3d_bars[i])

ax4.bar3d(x - width/2, y - depth/2, bottom, width, depth, top, color=bar_colors_list, alpha=0.9, edgecolor='black', linewidth=0.3)

# 添加数值标签
# for xi, yi, zi, val in zip(x, y, top, top):
#     ax4.text(xi, yi, zi + 0.02, f'{val:.3f}', color='black', ha='center', va='bottom', fontsize=plotter.base_font_size - 2)


ax4.set_xlabel('Model Name', labelpad=12)
ax4.set_ylabel('Metric Type', labelpad=12)
ax4.set_zlabel('Metric Value', labelpad=8)
plotter.set_title_for_subplot(ax4, '3D Bar Chart of Key Performance Metrics', fontsize_offset=3) # 使用类方法设置标题

ax4.set_xticks(x_pos_3d)
ax4.set_xticklabels(x_labels_3d, rotation=15, ha='right', fontsize=plotter.base_font_size-1)
ax4.set_yticks(y_pos_metric_3d)
ax4.set_yticklabels(metrics_3d, fontsize=plotter.base_font_size-1)
ax4.set_zlim(0, 1)

legend_elements_3d = [plt.Rectangle((0, 0), 1, 1, color=colors_3d_bars[i], label=metric, ec='black') for i, metric in enumerate(metrics_3d)]
ax4.legend(handles=legend_elements_3d, loc='center left', bbox_to_anchor=(1.05, 0.85)) # 调整图例位置

ax4.view_init(elev=25, azim=-55) # 调整视角
plt.tight_layout(rect=[0,0,0.9,0.95])
plotter.save_figure(fig4, os.path.join(FIGURE_SAVE_DIR, 'model_performance_3d_bar_styled.png'), dpi=300)
plt.show()


# --- 5. Styled Table (Pandas Styler - 基本保持不变，因为这不是matplotlib的图) ---
# 但我们可以确保字体大小和颜色与我们的风格一致
styled_df = df_compare.style \
    .set_caption("Model Performance Metrics Summary") \
    .set_table_styles([
        {'selector': 'caption',
         'props': [('color', 'black'),
                   ('font-size', f'{plotter.base_font_size + 5}pt'), # 增大标题字号
                   ('font-weight', 'bold'),
                   ('padding-bottom', '10px'),
                   ('font-family', plotter.font_family)]},
        {'selector': 'th', # 表头
         'props': [('font-size', f'{plotter.base_font_size + 2}pt'), # 增大表头字号
                   ('font-family', plotter.font_family),
                   ('background-color', '#f0f0f0'), # 浅灰表头
                   ('border', '1px solid #cccccc')]},
        {'selector': 'td', # 单元格
         'props': [('font-size', f'{plotter.base_font_size + 1}pt'), # 增大单元格字号
                   ('font-family', plotter.font_family),
                   ('border', '1px solid #dddddd')]},
        {'selector': 'tr:hover td', # 鼠标悬停行
         'props': [('background-color', '#f5f5f5')]}
    ]) \
    .format({
        'accuracy': "{:.4f}", 'pod': "{:.4f}", 'far': "{:.4f}", 'csi': "{:.4f}",
        'fn': "{:,.0f}", 'fp': "{:,.0f}"
    }) \
    .background_gradient(subset=['accuracy', 'pod', 'csi'], cmap=plotter.get_colormap('sca')) \
    .background_gradient(subset=['far', 'fn', 'fp'], cmap=plotter.get_colormap('rmse').reversed()) # rmse是绿-黄-红，反转后红是低值(好)

try:
    from IPython.display import display, HTML
    print("\n--- Styled Table (for Jupyter/HTML environments) ---")
    display(HTML(styled_df.to_html())) # 使用 to_html()
    # To save as HTML
    with open(os.path.join(FIGURE_SAVE_DIR, "styled_model_performance_table.html"), "w", encoding="utf-8") as f:
        f.write(styled_df.to_html())
    print(f"Styled table saved as {os.path.join(FIGURE_SAVE_DIR, 'styled_model_performance_table.html')}")
except ImportError:
    print("\n--- Pandas Styler Table (raw data, as IPython is not available) ---")
    print(df_compare)

print(f"\nAll plots generated and saved in '{FIGURE_SAVE_DIR}' directory!")