from loaddata import mydata # 假设 loaddata.py 在同一目录或可访问
import numpy as np
import os
import time
from scipy.ndimage import generic_filter # 导入用于空间计算的模块

# --- 1. 数据加载和初始准备 ---
print("--- Step 1: Data Loading (Yangtze - Spatial Grid) ---")
start_load = time.time()
ALL_DATA = mydata()
# 加载长江流域的空间网格数据 (basin_mask_value=2 代表长江流域)
# X_raw 形状: (产品数, 时间, 纬度, 经度)
# Y_raw 形状: (时间, 纬度, 经度)
X_raw, Y_raw = ALL_DATA.get_basin_spatial_data(basin_mask_value=2)
product_names = ALL_DATA.get_products()
n_products, nday, n_lat, n_lon = X_raw.shape # 获取维度信息
print(f"Initial X_raw shape: {X_raw.shape}")
print(f"Initial Y_raw shape: {Y_raw.shape}")
print(f"Product names: {product_names}")
print(f"Spatial dimensions: n_lat={n_lat}, n_lon={n_lon}")
end_load = time.time()
print(f"Data loading finished in {end_load - start_load:.2f} seconds.")

# --- 2. 数据整合与重塑 ---
print("\n--- Step 2: Reshaping ---")
# 将 X 转置为 (时间, 纬度, 经度, 产品数)
X = np.transpose(X_raw, (1, 2, 3, 0)).astype(np.float32)
del X_raw # 释放内存
Y = Y_raw.astype(np.float32) # 确保正确的数据类型
del Y_raw # 释放内存
print(f"Transposed X shape: {X.shape}") # (1827, n_lat, n_lon, 6)
print(f"Y shape: {Y.shape}")   # (1827, n_lat, n_lon)

# --- 3. 处理时间依赖性 ---
print("\n--- Step 3: Time Alignment ---")
# 根据特征确定 max_lookback (窗口=15, 滞后=3)
max_lookback = 30 # 保持30, 对于窗口=15, 滞后=3 足够
# nday, n_lat, n_lon, n_products 已在前面定义

valid_time_range = slice(max_lookback, nday)
n_valid_days = nday - max_lookback

# 长江数据不需要基于 Y NaNs 计算 valid_mask
print(f"Max lookback: {max_lookback} days")
print(f"Valid time range: Day index {max_lookback} to {nday-1}")

# 对齐 X 和 Y
X_aligned = X[valid_time_range] # 形状: (n_valid_days, n_lat, n_lon, n_products)
Y_aligned = Y[valid_time_range] # 形状: (n_valid_days, n_lat, n_lon)
print(f"Aligned X shape: {X_aligned.shape}")
print(f"Aligned Y shape: {Y_aligned.shape}")

# --- 4. 特征工程 (增强 - 适用于网格数据) ---
print("\n--- Step 4: Feature Engineering (Yangtze - Spatial Grid) ---")
start_feat = time.time()
features_dict = {}
epsilon = 1e-6
RAIN_THR = 0.1 # 计数降雨产品的阈值

# 安全除法辅助函数
def safe_divide(numerator, denominator, default=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / (denominator + epsilon)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

# --- 4.1 基础特征 ---
print("  Calculating basic features...")
features_dict['raw_values'] = X_aligned # 形状: (n_valid_days, n_lat, n_lon, n_products)

# --- 4.2 多产品一致性/差异特征 ---
print("  Calculating multi-product stats (current time)...")
# 沿产品维度 (axis=3) 计算统计量
features_dict['product_mean'] = np.nanmean(X_aligned, axis=3, keepdims=True).astype(np.float32)
features_dict['product_std'] = np.nanstd(X_aligned, axis=3, keepdims=True).astype(np.float32)
features_dict['product_median'] = np.nanmedian(X_aligned, axis=3, keepdims=True).astype(np.float32)
product_max = np.nanmax(X_aligned, axis=3, keepdims=True).astype(np.float32)
product_min = np.nanmin(X_aligned, axis=3, keepdims=True).astype(np.float32)
features_dict['product_max'] = product_max
features_dict['product_min'] = product_min
features_dict['product_range'] = (product_max - product_min).astype(np.float32)
X_rain = (np.nan_to_num(X_aligned, nan=0.0) > RAIN_THR)
features_dict['rain_product_count'] = np.sum(X_rain, axis=3, keepdims=True).astype(np.float32)

# --- 4.3 时间演化特征 ---
# 4.3.1 周期性
print("  Calculating periodicity features...")
days_in_year = 365.25
day_index_original = np.arange(nday, dtype=np.float32)
day_of_year = day_index_original % days_in_year
sin_time = np.sin(2 * np.pi * day_of_year / days_in_year).astype(np.float32)
cos_time = np.cos(2 * np.pi * day_of_year / days_in_year).astype(np.float32)
# 切片和扩展: (n_valid_days,) -> (n_valid_days, 1, 1, 1) -> 广播到 (n_valid_days, n_lat, n_lon, 1)
sin_time_aligned = sin_time[valid_time_range, np.newaxis, np.newaxis, np.newaxis] * np.ones((1, n_lat, n_lon, 1), dtype=np.float32)
cos_time_aligned = cos_time[valid_time_range, np.newaxis, np.newaxis, np.newaxis] * np.ones((1, n_lat, n_lon, 1), dtype=np.float32)
features_dict['sin_day'] = sin_time_aligned
features_dict['cos_day'] = cos_time_aligned
# 季节 One-Hot
month = (day_of_year // 30.4375).astype(int) % 12 + 1
season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0} # 0:冬, 1:春, 2:夏, 3:秋
season = np.array([season_map[m] for m in month])
seasons_onehot = np.zeros((nday, 4), dtype=np.float32)
seasons_onehot[np.arange(nday), season] = 1
# 切片和扩展: (n_valid_days, 4) -> (n_valid_days, 1, 1, 4) -> 广播到 (n_valid_days, n_lat, n_lon, 4)
season_aligned = seasons_onehot[valid_time_range, np.newaxis, np.newaxis, :] * np.ones((1, n_lat, n_lon, 1), dtype=np.float32)
features_dict['season_onehot'] = season_aligned

# 4.3.2 滞后特征 (深化)
print("  Calculating deepened lag features...")
lag_data_cache = {}
lag_mean_cache = {}
lag_std_cache = {}
lag_rain_count_cache = {}

for lag in [1, 2, 3]:
    print(f"    Lag {lag}...")
    lag_slice = slice(max_lookback - lag, nday - lag)
    lag_data = X[lag_slice] # 形状: (n_valid_days, n_lat, n_lon, n_products)
    lag_data_cache[lag] = lag_data
    features_dict[f'lag_{lag}_values'] = lag_data

    lag_mean = np.nanmean(lag_data, axis=3, keepdims=True).astype(np.float32) # 产品维度 axis=3
    lag_std = np.nanstd(lag_data, axis=3, keepdims=True).astype(np.float32)   # 产品维度 axis=3
    lag_rain = (np.nan_to_num(lag_data, nan=0.0) > RAIN_THR)
    lag_rain_count = np.sum(lag_rain, axis=3, keepdims=True).astype(np.float32) # 产品维度 axis=3

    lag_mean_cache[lag] = lag_mean
    lag_std_cache[lag] = lag_std
    lag_rain_count_cache[lag] = lag_rain_count

    features_dict[f'lag_{lag}_mean'] = lag_mean
    features_dict[f'lag_{lag}_std'] = lag_std
    features_dict[f'lag_{lag}_rain_count'] = lag_rain_count

if 1 in lag_mean_cache and 2 in lag_mean_cache:
    features_dict['lag_1_2_mean_diff'] = (lag_mean_cache[1] - lag_mean_cache[2]).astype(np.float32)
if 2 in lag_mean_cache and 3 in lag_mean_cache:
    features_dict['lag_2_3_mean_diff'] = (lag_mean_cache[2] - lag_mean_cache[3]).astype(np.float32)

# 4.3.3 差异特征 (t - (t-1))
print("  Calculating difference features (t - t-1)...")
prev_data = lag_data_cache[1] # t-1
features_dict['diff_1_values'] = (X_aligned - prev_data).astype(np.float32)
features_dict['diff_1_mean'] = features_dict['product_mean'] - lag_mean_cache[1]
features_dict['diff_1_std'] = features_dict['product_std'] - lag_std_cache[1]
print("    Deleting raw lag data cache...")
del lag_data_cache, prev_data

# 4.3.4 移动窗口特征 (深化)
print("  Calculating deepened moving window features...")
# 首先计算完整时间序列上的产品平均值
product_mean_full = np.nanmean(X, axis=3, keepdims=True).astype(np.float32) # 形状: (nday, n_lat, n_lon, 1)
for window in [3, 7, 15]:
    print(f"    Window {window} (on product mean)...")
    window_mean = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
    window_std = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
    window_max = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
    window_min = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
    for i in range(n_valid_days):
        current_original_idx = i + max_lookback
        # window_data_mean_prod 形状: (window, n_lat, n_lon, 1)
        window_data_mean_prod = product_mean_full[current_original_idx - window : current_original_idx]
        window_mean[i] = np.nanmean(window_data_mean_prod, axis=0) # 沿时间窗口 (axis=0) 计算
        window_std[i] = np.nanstd(window_data_mean_prod, axis=0)
        window_max[i] = np.nanmax(window_data_mean_prod, axis=0)
        window_min[i] = np.nanmin(window_data_mean_prod, axis=0)
    features_dict[f'window_{window}_mean'] = window_mean.astype(np.float32)
    features_dict[f'window_{window}_std'] = window_std.astype(np.float32)
    features_dict[f'window_{window}_max'] = window_max.astype(np.float32)
    features_dict[f'window_{window}_min'] = window_min.astype(np.float32)
    features_dict[f'window_{window}_range'] = (window_max - window_min).astype(np.float32)
del product_mean_full # 清理

# 每个产品的移动窗口统计 (例如 window=7, GSMAP, PERSIANN)
window = 7
print(f"    Window {window} (per product - GSMAP, PERSIANN)...")
gsmap_idx = product_names.index("GSMAP")
persiann_idx = product_names.index("PERSIANN")
window_mean_gsmap = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
window_std_persiann = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
for i in range(n_valid_days):
    current_original_idx = i + max_lookback
    # window_data_X 形状: (window, n_lat, n_lon, n_products)
    window_data_X = X[current_original_idx - window : current_original_idx]
    # 对 GSMAP 计算时间均值 (axis=0), 结果形状 (n_lat, n_lon)
    mean_gsmap_result = np.nanmean(window_data_X[:, :, :, gsmap_idx], axis=0)
    # 对 PERSIANN 计算时间标准差 (axis=0), 结果形状 (n_lat, n_lon)
    std_persiann_result = np.nanstd(window_data_X[:, :, :, persiann_idx], axis=0)
    # 分配到目标数组, 添加特征维度
    window_mean_gsmap[i] = mean_gsmap_result[..., np.newaxis]
    window_std_persiann[i] = std_persiann_result[..., np.newaxis]
features_dict[f'window_{window}_mean_GSMAP'] = window_mean_gsmap.astype(np.float32)
features_dict[f'window_{window}_std_PERSIANN'] = window_std_persiann.astype(np.float32)


# --- 4.4 空间背景特征 (深化 - 适用于网格数据) ---
print("  Calculating deepened spatial features (for Grid Data)...")
# 辅助函数，用于在单个2D切片上应用空间滤波器
def apply_spatial_filter(data_2d_slice, filter_func, footprint_size):
    if np.all(np.isnan(data_2d_slice)):
        return np.full_like(data_2d_slice, np.nan)
    return generic_filter(
        data_2d_slice,
        filter_func,
        footprint=np.ones((footprint_size, footprint_size)),
        mode='constant',
        cval=np.nan
    )

# 4.4.1 3x3 邻域
print("    Calculating 3x3 neighborhood statistics...")
spatial_mean_3x3 = np.zeros_like(X_aligned, dtype=np.float32)
spatial_std_3x3 = np.zeros_like(X_aligned, dtype=np.float32)
spatial_max_3x3 = np.zeros_like(X_aligned, dtype=np.float32)

for t in range(n_valid_days):
    for p in range(n_products):
        data_slice = X_aligned[t, :, :, p]
        spatial_mean_3x3[t, :, :, p] = apply_spatial_filter(data_slice, np.nanmean, 3)
        spatial_std_3x3[t, :, :, p] = apply_spatial_filter(data_slice, np.nanstd, 3)
        spatial_max_3x3[t, :, :, p] = apply_spatial_filter(data_slice, np.nanmax, 3)

features_dict['spatial_mean_3x3'] = spatial_mean_3x3
features_dict['spatial_std_3x3'] = spatial_std_3x3
features_dict['spatial_max_3x3'] = spatial_max_3x3
features_dict['spatial_center_diff_3x3'] = (features_dict['raw_values'] - spatial_mean_3x3).astype(np.float32)

# 4.4.2 5x5 邻域
print("    Calculating 5x5 neighborhood statistics...")
spatial_mean_5x5 = np.zeros_like(X_aligned, dtype=np.float32)
spatial_std_5x5 = np.zeros_like(X_aligned, dtype=np.float32)
spatial_max_5x5 = np.zeros_like(X_aligned, dtype=np.float32)

for t in range(n_valid_days):
    for p in range(n_products):
        data_slice = X_aligned[t, :, :, p]
        spatial_mean_5x5[t, :, :, p] = apply_spatial_filter(data_slice, np.nanmean, 5)
        spatial_std_5x5[t, :, :, p] = apply_spatial_filter(data_slice, np.nanstd, 5)
        spatial_max_5x5[t, :, :, p] = apply_spatial_filter(data_slice, np.nanmax, 5)

features_dict['spatial_mean_5x5'] = spatial_mean_5x5
features_dict['spatial_std_5x5'] = spatial_std_5x5
features_dict['spatial_max_5x5'] = spatial_max_5x5
features_dict['spatial_center_diff_5x5'] = (features_dict['raw_values'] - spatial_mean_5x5).astype(np.float32)

# 4.4.3 空间梯度
print("    Calculating spatial gradients...")
grad_mag_gsmap = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
grad_mag_persiann = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
grad_mag_mean = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)

gsmap_idx = product_names.index("GSMAP")
persiann_idx = product_names.index("PERSIANN")

for t in range(n_valid_days):
    # GSMAP 梯度
    data_slice_gsmap = X_aligned[t, :, :, gsmap_idx]
    if not np.all(np.isnan(data_slice_gsmap)):
        gy, gx = np.gradient(data_slice_gsmap)
        grad_mag_gsmap[t, :, :, 0] = np.sqrt(gy**2 + gx**2)
    else:
        grad_mag_gsmap[t, :, :, 0] = np.nan
    # PERSIANN 梯度
    data_slice_persiann = X_aligned[t, :, :, persiann_idx]
    if not np.all(np.isnan(data_slice_persiann)):
        gy, gx = np.gradient(data_slice_persiann)
        grad_mag_persiann[t, :, :, 0] = np.sqrt(gy**2 + gx**2)
    else:
        grad_mag_persiann[t, :, :, 0] = np.nan
    # 平均产品梯度 (使用之前计算的 product_mean)
    data_slice_mean = features_dict['product_mean'][t, :, :, 0] # product_mean 形状 (..., 1)
    if not np.all(np.isnan(data_slice_mean)):
        gy, gx = np.gradient(data_slice_mean)
        grad_mag_mean[t, :, :, 0] = np.sqrt(gy**2 + gx**2)
    else:
        grad_mag_mean[t, :, :, 0] = np.nan

features_dict['gradient_magnitude_GSMAP'] = grad_mag_gsmap
features_dict['gradient_magnitude_PERSIANN'] = grad_mag_persiann
features_dict['gradient_magnitude_mean'] = grad_mag_mean

# --- 在空间计算后删除 X_aligned ---
print("    Deleting X_aligned...")
del X_aligned # 删除别名

# --- 4.5 低强度信号特征 (增强) ---
print("  Calculating enhanced low intensity features...")
# 4.5.1 阈值接近度
features_dict['threshold_proximity'] = np.abs(features_dict['product_mean'] - RAIN_THR).astype(np.float32)

# 4.5.2 变异系数及其滞后版本
cv = safe_divide(features_dict['product_std'], features_dict['product_mean'])
features_dict['coef_of_variation'] = cv.astype(np.float32)
cv_lag1 = safe_divide(lag_std_cache[1], lag_mean_cache[1])
features_dict['lag_1_coef_of_variation'] = cv_lag1.astype(np.float32)
print("    Deleting lag stats caches...")
del lag_mean_cache, lag_std_cache, lag_rain_count_cache, cv_lag1

# 4.5.3 条件不确定性
low_intensity_std = np.where(
    features_dict['product_mean'] < 1.0,
    features_dict['product_std'],
    0.0
).astype(np.float32)
features_dict['low_intensity_std'] = low_intensity_std

# 4.5.4 强度分箱 (均值和计数)
mean_values = features_dict['product_mean'] # 形状 (n_valid_days, n_lat, n_lon, 1)
intensity_bins_mean = np.zeros((n_valid_days, n_lat, n_lon, 4), dtype=np.float32)
intensity_bins_mean[:, :, :, 0] = (mean_values <= 0.1).squeeze(axis=-1)
intensity_bins_mean[:, :, :, 1] = ((mean_values > 0.1) & (mean_values <= 0.5)).squeeze(axis=-1)
intensity_bins_mean[:, :, :, 2] = ((mean_values > 0.5) & (mean_values <= 1.0)).squeeze(axis=-1)
intensity_bins_mean[:, :, :, 3] = (mean_values > 1.0).squeeze(axis=-1)
features_dict['intensity_bins_mean'] = intensity_bins_mean

count_values = features_dict['rain_product_count'] # 形状 (n_valid_days, n_lat, n_lon, 1)
intensity_bins_count = np.zeros((n_valid_days, n_lat, n_lon, 4), dtype=np.float32)
intensity_bins_count[:, :, :, 0] = (count_values == 0).squeeze(axis=-1)
intensity_bins_count[:, :, :, 1] = (count_values == 1).squeeze(axis=-1)
intensity_bins_count[:, :, :, 2] = (count_values == 2).squeeze(axis=-1)
intensity_bins_count[:, :, :, 3] = (count_values >= 3).squeeze(axis=-1)
features_dict['intensity_bins_count'] = intensity_bins_count

# --- 4.6 交互特征 (增强) ---
print("  Calculating enhanced interaction features...")
features_dict['std_season_interaction'] = (features_dict['product_std'] * np.abs(features_dict['sin_day'])).astype(np.float32)
features_dict['low_intense_high_uncertain'] = (low_intensity_std * features_dict['coef_of_variation']).astype(np.float32)
features_dict['rain_count_std_interaction'] = (features_dict['rain_product_count'] * features_dict['product_std']).astype(np.float32)

print("    Calculating std_x_spatial_diff interaction...")
# 使用已计算的 spatial_center_diff_3x3
# spatial_center_diff_3x3 形状: (n_valid_days, n_lat, n_lon, n_products)
# 我们需要其在产品维度上的平均值
mean_spatial_center_diff_3x3 = np.nanmean(features_dict['spatial_center_diff_3x3'], axis=3, keepdims=True)
features_dict['std_x_spatial_diff'] = (features_dict['product_std'] * np.abs(mean_spatial_center_diff_3x3)).astype(np.float32)
del mean_spatial_center_diff_3x3

print("    Calculating std_x_diff_1_mean interaction...")
features_dict['std_x_diff_1_mean'] = (features_dict['product_std'] * np.abs(features_dict['diff_1_mean'])).astype(np.float32)

print("    Deleting remaining intermediate feature variables...")
del cv, low_intensity_std
# 删除大的空间差异数组
if 'spatial_center_diff_3x3' in features_dict:
    print("    Deleting spatial_center_diff_3x3...")
    del features_dict['spatial_center_diff_3x3']
if 'spatial_center_diff_5x5' in features_dict:
     print("    Deleting spatial_center_diff_5x5...")
     del features_dict['spatial_center_diff_5x5']

end_feat = time.time()
print(f"Feature engineering finished in {end_feat - start_feat:.2f} seconds.")


# --- 5. 连接特征并生成名称 ---
print("\n--- Step 5: Concatenating Features & Naming (Yangtze v2 - Spatial) ---")
start_concat = time.time()
# 通过连接 features_dict 中的数组来构建特征矩阵
features_list_final = []
feature_names = []

# 使用与 national turn2.py 相同的命名逻辑
for name, feat_array in features_dict.items():
    if feat_array is None or not isinstance(feat_array, np.ndarray): continue
    try:
        n_cols = feat_array.shape[3] # 轴 3 是特征维度 (时间, 纬度, 经度, 特征)
        features_list_final.append(feat_array)
    except IndexError:
         print(f"Warning: Feature '{name}' has unexpected shape {feat_array.shape}. Skipping.")
         continue

    base_name = name
    if n_cols == 1:
        feature_names.append(base_name)
    # 调整最初具有产品维度的特征的命名
    elif base_name == 'raw_values' or \
         (base_name.startswith('lag_') and '_values' in base_name) or \
         base_name == 'diff_1_values' or \
         base_name.startswith('spatial_mean_') or \
         base_name.startswith('spatial_std_') or \
         base_name.startswith('spatial_max_'):
        if n_cols == n_products:
            for i in range(n_cols): feature_names.append(f"{base_name}_{product_names[i]}")
        else:
            print(f"Warning: Mismatch in columns for {base_name}. Expected {n_products}, got {n_cols}. Using index.")
            for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    elif base_name == 'season_onehot': # 形状 (..., 4)
        for i in range(n_cols): feature_names.append(f"{base_name}_{i}") # season_onehot_0, _1, _2, _3
    elif base_name == 'intensity_bins_mean': # 形状 (..., 4)
        bin_labels = ['<=0.1', '0.1-0.5', '0.5-1.0', '>1.0']
        if n_cols == len(bin_labels):
             for i in range(n_cols): feature_names.append(f"{base_name}_{bin_labels[i]}")
        else:
             print(f"Warning: Mismatch in columns for {base_name}. Expected {len(bin_labels)}, got {n_cols}. Using index.")
             for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    elif base_name == 'intensity_bins_count': # 形状 (..., 4)
        bin_labels = ['0', '1', '2', '>=3']
        if n_cols == len(bin_labels):
             for i in range(n_cols): feature_names.append(f"{base_name}_{bin_labels[i]}")
        else:
             print(f"Warning: Mismatch in columns for {base_name}. Expected {len(bin_labels)}, got {n_cols}. Using index.")
             for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    else: # 后备方案
        if n_cols != 1: # 对于其他多列特征，如果未明确处理
             print(f"Warning: Unhandled multi-column feature naming for '{base_name}' with {n_cols} columns. Using index.")
             for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
        # else: 单列特征已在开头处理

# 沿特征轴 (axis=3) 连接
if not features_list_final:
    raise ValueError("No features were generated or added to the list for concatenation.")
X_features = np.concatenate(features_list_final, axis=3).astype(np.float32)
del features_list_final, features_dict # 释放内存

total_features = X_features.shape[3] # 特征维度现在是 axis=3
print(f"Concatenated features shape: {X_features.shape}") # (n_valid_days, n_lat, n_lon, total_features)
print(f"Total calculated feature columns: {total_features}")
print(f"Length of feature_names list: {len(feature_names)}")
if len(feature_names) != total_features:
     print(f"FATAL: Mismatch between calculated total features ({total_features}) and feature name list length ({len(feature_names)})!")
     exit()

# 为长江 v2 明确定义输出目录和文件名
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features_spatial") # 目录名更改以反映空间数据
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
X_flat_filename = os.path.join(OUTPUT_DIR, "X_Yangtsu_flat_spatial_features_v4.npy") # 添加 _spatial 和 _v4
Y_flat_filename = os.path.join(OUTPUT_DIR, "Y_Yangtsu_flat_spatial_target_v4.npy") # 添加 _spatial 和 _v4
feature_names_filename = os.path.join(OUTPUT_DIR, "feature_names_yangtsu_spatial_v4.txt") # 添加 _spatial 和 _v4

# 保存特征名称
print(f"Saving feature names to {feature_names_filename}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(feature_names_filename, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")
end_concat = time.time()
print(f"Concatenation and naming finished in {end_concat - start_concat:.2f} seconds.")

# --- 6. 展平数据以供模型输入 ---
print("\n--- Step 6: Flattening Data (Yangtze v4 - Spatial) ---")
start_flat = time.time()
# 重塑 X: (n_valid_days, n_lat, n_lon, n_total_features) -> (n_valid_days * n_lat * n_lon, n_total_features)
n_samples = n_valid_days * n_lat * n_lon
X_flat = X_features.reshape(n_samples, total_features)
del X_features # 释放内存

# 重塑 Y: (n_valid_days, n_lat, n_lon) -> (n_valid_days * n_lat * n_lon,)
Y_flat = Y_aligned.reshape(n_samples)
del Y_aligned # 释放内存

print(f"Flattened X shape: {X_flat.shape}")
print(f"Flattened Y shape: {Y_flat.shape}")

# 处理特征计算中可能引入的 NaN (尤其是空间特征)
X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0) # 用0填充NaN和inf
# Y_flat 中的 NaN 通常表示原始数据中的无效点，应谨慎处理或在模型训练前过滤
if np.isnan(Y_flat).any():
    print(f"Warning: NaNs found in flattened Y target data! Count: {np.isnan(Y_flat).sum()}. These usually correspond to areas outside the basin mask in the original spatial grid.")
    # Y_flat = np.nan_to_num(Y_flat, nan=-9999.0) # 或者根据模型需要处理

end_flat = time.time()
print(f"Flattening finished in {end_flat - start_flat:.2f} seconds.")

# --- 7. 保存数据 ---
print("\n--- Step 7: Saving Flattened Data (Yangtze v4 - Spatial) ---")
start_save = time.time()
np.save(X_flat_filename, X_flat)
np.save(Y_flat_filename, Y_flat)
print(f"Flattened features saved to: {X_flat_filename}")
print(f"Flattened target saved to: {Y_flat_filename}")
end_save = time.time()
print(f"Saving finished in {end_save - start_save:.2f} seconds.")

print(f"\nTotal processing time: {time.time() - start_load:.2f} seconds")
print("Data processing complete.")

