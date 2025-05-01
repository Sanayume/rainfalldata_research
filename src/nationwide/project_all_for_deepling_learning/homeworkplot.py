import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置样式参数
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 12

# 数据集 1: λ1=435nm, Φ1=4mm
x1 = [-2, -1.8, -1.6, -1.4,-1.238, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
y1 = [-2.2, -2.0, -1.9, -1.5, 0, 1.1, 8.8, 20.1, 38.2, 63.6, 97.1, 135.3, 195.3, 26.8*10,35.2*10, 44.4*10, 52.9*10, 61.1*10, 66.4*10]

# 数据集 2: λ1=435nm, Φ2=2mm
x2 = [-2.0, -1.8, -1.6, -1.4,-1.266, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5,1.8, 2.0]
y2 = [-1.0, -0.5, -0.3, -0.2, 0,     0.1, 2.0, 5.0, 9.4, 15.9, 24.4, 34.4,51.8,71.9,93.8,116.5,141.2,174.2,183.1] 

# 数据集 3: λ2=405nm, Φ1=4mm
x3 = [-2.0, -1.8, -1.6,-1.485, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
y3 = [1.6, 1.4, -0.9, 0, 2.0, 12.1, 29.1, 51.2, 77.6, 107.5, 139.9,174.0, 24.0*10, 30.6*10, 37.7*10, 45.1*10, 52.8*10, 60.8*10, 66.4*10]

# 创建具有现代风格的图表
fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor='white')

# 定义更好的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

# 用增强的样式绘制数据
p1 = ax.plot(x1, y1, marker='o', linestyle='-', linewidth=2.5, markersize=8, 
          color=colors[0], label='λ=435nm, Φ=4mm', alpha=0.9)
p2 = ax.plot(x2, y2, marker='s', linestyle='--', linewidth=2.5, markersize=8, 
          color=colors[1], label='λ=435nm, Φ=2mm', alpha=0.9)
p3 = ax.plot(x3, y3, marker='^', linestyle=':', linewidth=2.5, markersize=8, 
          color=colors[2], label='λ=405nm, Φ=4mm', alpha=0.9)

# 添加标签和标题，并改进格式
ax.set_xlabel("Voltage (V)", fontsize=14, fontweight='bold')
ax.set_ylabel("Current (μA)", fontsize=14, fontweight='bold')
ax.set_title("Photoelectric Effect Current-Voltage Characteristic", fontsize=16, fontweight='bold', pad=20)

# 增强网格外观
ax.grid(True, linestyle='--', alpha=0.7)

# 设置坐标轴样式
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 将Y轴移动到X=0的位置
ax.spines['left'].set_position(('data', 0))

# 确保Y轴标签可见
ax.tick_params(width=1.5, length=5, labelsize=12)
ax.tick_params(axis='y', pad=8)  # 增加Y轴标签与轴的距离

# 使Y轴标签在数据前面可见
for label in ax.get_yticklabels():
    label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.7))

# 添加图例并改进样式
legend = ax.legend(loc='best', frameon=True, fontsize=12, 
               facecolor='white', edgecolor='gray', framealpha=0.9)
legend.get_frame().set_linewidth(1.5)

# 为所有数据点添加坐标标注
for i, (x_data, y_data, color) in enumerate(zip([x1, x2, x3], [y1, y2, y3], colors)):
    for j, (x, y) in enumerate(zip(x_data, y_data)):
        # 确定坐标标注的位置 (轮换不同方向以避免重叠)
        pos = j % 4
        if pos == 0:
            xytext = (5, 5)
        elif pos == 1:
            xytext = (-5, 5)
        elif pos == 2:
            xytext = (-5, -5)
        else:
            xytext = (5, -5)
            
        # 标注坐标，字体更大
        ax.annotate(f'({x:.1f}, {y:.1f})', 
                   (x, y), 
                   textcoords="offset points",
                   xytext=xytext, 
                   ha='center',
                   fontsize=9,  # 增大字体大小
                   color=color,  # 与对应线条相同的颜色
                   alpha=0.9)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()