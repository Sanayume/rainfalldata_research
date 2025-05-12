from loaddata import mydata
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
mydata = mydata()
X, Y = mydata.get_basin_point_data(2)
output_dir = "results/value_characteristics_fp_fn"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

FAR_index = []
POD_index = []

#既然是关注误报漏报,那么肯定要对小降雨事件敏感，所以降雨分类阈值应该要小些
threshold = 0.1

for i in range(X.shape[0]):
    product = mydata.get_products()[i]
    print(f"Processing product: {product}")

    X_data = X[i, :, :]

    X_is_rain = np.where(X_data > threshold, 1, 0)
    Y_is_rain = np.where(Y > threshold, 1, 0)

    fp = ((X_is_rain == 1) & (Y_is_rain == 0))
    fn = ((X_is_rain == 0) & (Y_is_rain == 1))
    tp = ((X_is_rain == 1) & (Y_is_rain == 1))
    tn = ((X_is_rain == 0) & (Y_is_rain == 0))

    POD = tp / (tp + fn)
    FAR = fp / (fp + tp)

    FAR_index.append(FAR)
    POD_index.append(POD)


FAR_index = np.array(FAR_index)
POD_index = np.array(POD_index)

#保存index
np.save(os.path.join(output_dir, "FAR_index.npy"), FAR_index)
np.save(os.path.join(output_dir, "POD_index.npy"), POD_index)

POD_value = np.where(POD_index == 1, X, np.nan)
FAR_value = np.where(FAR_index == 1, X, np.nan)

# 合并POD和FAR的分布图，使用不同颜色区分
plt.figure(figsize=(12, 8))

# 准备数据，去除NaN值
pod_data = POD_value.flatten()
far_data = FAR_value.flatten()
pod_data = pod_data[~np.isnan(pod_data)]
far_data = far_data[~np.isnan(far_data)]

# 计算统计信息用于标注
pod_mean = np.mean(pod_data) if len(pod_data) > 0 else np.nan
pod_median = np.median(pod_data) if len(pod_data) > 0 else np.nan
far_mean = np.mean(far_data) if len(far_data) > 0 else np.nan
far_median = np.median(far_data) if len(far_data) > 0 else np.nan

# 绘制直方图
bins = np.linspace(0, max(np.max(pod_data) if len(pod_data) > 0 else 0, 
                          np.max(far_data) if len(far_data) > 0 else 0), 100)
plt.hist(pod_data, bins=bins, alpha=0.6, color='blue', label=f'POD (均值={pod_mean:.4f}, 中位数={pod_median:.4f})')
plt.hist(far_data, bins=bins, alpha=0.6, color='red', label=f'FAR (均值={far_mean:.4f}, 中位数={far_median:.4f})')

# 添加垂直线标记均值和中位数
plt.axvline(pod_mean, color='blue', linestyle='dashed', linewidth=1.5, label='_POD均值')
plt.axvline(pod_median, color='blue', linestyle='dotted', linewidth=1.5, label='_POD中位数')
plt.axvline(far_mean, color='red', linestyle='dashed', linewidth=1.5, label='_FAR均值')
plt.axvline(far_median, color='red', linestyle='dotted', linewidth=1.5, label='_FAR中位数')

# 添加图表标题和标签
plt.title(f"POD和FAR值分布对比 (阈值={threshold}mm/d)", fontsize=14, fontweight='bold')
plt.xlabel("值", fontsize=12)
plt.ylabel("频数", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=10)

# 添加文本注释显示更多统计信息
pod_stats = f"POD统计:\n数量: {len(pod_data)}\n最大值: {np.max(pod_data):.4f}\n最小值: {np.min(pod_data):.4f}\n标准差: {np.std(pod_data):.4f}"
far_stats = f"FAR统计:\n数量: {len(far_data)}\n最大值: {np.max(far_data):.4f}\n最小值: {np.min(far_data):.4f}\n标准差: {np.std(far_data):.4f}"

plt.annotate(pod_stats, xy=(0.02, 0.95), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.8),
             fontsize=9, verticalalignment='top')
             
plt.annotate(far_stats, xy=(0.02, 0.65), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", alpha=0.8),
             fontsize=9, verticalalignment='top')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "POD_FAR_distribution_comparison.png"), dpi=300)
plt.show()

#关于POD_value和FAR_value的各种统计特征, 
print(f"POD_value的统计特征:")
print(f"  平均值: {np.nanmean(POD_value):.4f}")
print(f"  中位数: {np.nanmedian(POD_value):.4f}")
print(f"  标准差: {np.nanstd(POD_value):.4f}")

# 计算 POD_value 的众数
# 原先使用 np.bincount 的方法不适用于浮点型数据或包含 NaN 的数据。
# 我们改用 scipy.stats.mode 来寻找最频繁出现的值，并正确处理 NaN。
_pod_values_flattened_for_mode = POD_value.flatten()
_pod_values_flattened_for_mode = _pod_values_flattened_for_mode[~np.isnan(_pod_values_flattened_for_mode)]
_mode_result_pod = stats.mode(_pod_values_flattened_for_mode, nan_policy='omit', keepdims=True)

if len(_pod_values_flattened_for_mode) > 0 and _mode_result_pod.mode.size > 0:
    _actual_pod_mode = _mode_result_pod.mode[0]
    print(f"  众数: {_actual_pod_mode:.4f}")
else:
    print(f"  众数: 无法计算（可能全为NaN或数据集为空）")

print(f"  方差: {np.nanvar(POD_value):.4f}")

# 安全计算偏度和峰度，确保有足够的非NaN数据
if len(_pod_values_flattened_for_mode) > 2:  # 需要至少3个点才能计算偏度和峰度
    print(f"  偏度: {stats.skew(_pod_values_flattened_for_mode):.4f}")
    print(f"  峰度: {stats.kurtosis(_pod_values_flattened_for_mode):.4f}")
else:
    print(f"  偏度: 无法计算（数据不足）")
    print(f"  峰度: 无法计算（数据不足）")

print(f"  最大值: {np.nanmax(POD_value):.4f}")
print(f"  最小值: {np.nanmin(POD_value):.4f}")

# 安全获取最大值和最小值的位置
if not np.all(np.isnan(POD_value)):
    max_index = np.nanargmax(POD_value.flatten())
    min_index = np.nanargmin(POD_value.flatten())
    print(f"  最大值位置: {np.unravel_index(max_index, POD_value.shape)}")
    print(f"  最小值位置: {np.unravel_index(min_index, POD_value.shape)}")
else:
    print(f"  最大值位置: 无法确定（全为NaN）")
    print(f"  最小值位置: 无法确定（全为NaN）")

print(f"FAR_value的统计特征:")
print(f"  平均值: {np.nanmean(FAR_value):.4f}")
print(f"  中位数: {np.nanmedian(FAR_value):.4f}")
print(f"  标准差: {np.nanstd(FAR_value):.4f}")

# 计算 FAR_value 的众数
_far_values_flattened_for_mode = FAR_value.flatten()
_far_values_flattened_for_mode = _far_values_flattened_for_mode[~np.isnan(_far_values_flattened_for_mode)]
_mode_result_far = stats.mode(_far_values_flattened_for_mode, nan_policy='omit', keepdims=True)

if len(_far_values_flattened_for_mode) > 0 and _mode_result_far.mode.size > 0:
    _actual_far_mode = _mode_result_far.mode[0]
    print(f"  众数: {_actual_far_mode:.4f}")
else:
    print(f"  众数: 无法计算（可能全为NaN或数据集为空）")

print(f"  方差: {np.nanvar(FAR_value):.4f}")

# 安全计算偏度和峰度，确保有足够的非NaN数据
if len(_far_values_flattened_for_mode) > 2:  # 需要至少3个点才能计算偏度和峰度
    print(f"  偏度: {stats.skew(_far_values_flattened_for_mode):.4f}")
    print(f"  峰度: {stats.kurtosis(_far_values_flattened_for_mode):.4f}")
else:
    print(f"  偏度: 无法计算（数据不足）")
    print(f"  峰度: 无法计算（数据不足）")

print(f"  最大值: {np.nanmax(FAR_value):.4f}")
print(f"  最小值: {np.nanmin(FAR_value):.4f}")

# 安全获取最大值和最小值的位置
if not np.all(np.isnan(FAR_value)):
    max_index = np.nanargmax(FAR_value.flatten())
    min_index = np.nanargmin(FAR_value.flatten())
    print(f"  最大值位置: {np.unravel_index(max_index, FAR_value.shape)}")
    print(f"  最小值位置: {np.unravel_index(min_index, FAR_value.shape)}")
else:
    print(f"  最大值位置: 无法确定（全为NaN）")
    print(f"  最小值位置: 无法确定（全为NaN）")

# 计算POD_value和FAR_value的协方差
if len(_pod_values_flattened_for_mode) > 1 and len(_far_values_flattened_for_mode) > 1:
    covariance = np.cov(_pod_values_flattened_for_mode, _far_values_flattened_for_mode)[0, 1]
    print(f"POD_value和FAR_value的协方差: {covariance:.4f}")
else:
    print(f"POD_value和FAR_value的协方差: 无法计算（数据不足）")

# 计算POD_value和FAR_value的皮尔逊相关系数
if len(_pod_values_flattened_for_mode) > 1 and len(_far_values_flattened_for_mode) > 1:
    correlation = np.corrcoef(_pod_values_flattened_for_mode, _far_values_flattened_for_mode)[0, 1]
    print(f"POD_value和FAR_value的皮尔逊相关系数: {correlation:.4f}")
else:
    print(f"POD_value和FAR_value的皮尔逊相关系数: 无法计算（数据不足）")






