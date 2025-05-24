from loaddata import mydata # 假设 loaddata.py 在同一目录或可访问
import numpy as np
import os
import time
from scipy.ndimage import generic_filter, maximum_filter # 导入用于空间滤波的函数
import bottleneck as bn # 导入 bottleneck
from joblib import Parallel, delayed # 导入 joblib

# --- 1. 数据加载和初始准备 ---
print("--- Step 1: Data Loading (Yangtze with Spatial Grid) ---")
start_load = time.time()
ALL_DATA = mydata()
# 加载长江流域的空间网格数据 (basin_mask_value=2 代表长江流域)
# X_raw_spatial: (产品数, 时间, 纬度, 经度)
# Y_raw_spatial: (时间, 纬度, 经度)
X_raw_spatial, Y_raw_spatial = ALL_DATA.get_basin_spatial_data(basin_mask_value=2)

product_names = ALL_DATA.get_products()
n_products, nday, n_lat, n_lon = X_raw_spatial.shape # 获取维度信息
print(f"Initial X_raw_spatial shape: {X_raw_spatial.shape}")
print(f"Initial Y_raw_spatial shape: {Y_raw_spatial.shape}")
print(f"Product names: {product_names}")
print(f"Grid dimensions: Lat={n_lat}, Lon={n_lon}")
end_load = time.time()
print(f"Data loading finished in {end_load - start_load:.2f} seconds.")

# --- 2. 数据整合与重塑 ---
print("\n--- Step 2: Reshaping ---")
# 将 X_raw_spatial 转置为 (时间, 纬度, 经度, 产品数)
X = np.transpose(X_raw_spatial, (1, 2, 3, 0)).astype(np.float32)
del X_raw_spatial # 释放内存
Y = Y_raw_spatial.astype(np.float32) # 确保正确的数据类型
del Y_raw_spatial # 释放内存
print(f"Transposed X shape: {X.shape}") # (nday, n_lat, n_lon, n_products)
print(f"Y shape: {Y.shape}")   # (nday, n_lat, n_lon)

# --- 3. 处理时间依赖性 ---
print("\n--- Step 3: Time Alignment ---")
max_lookback = 30 # 保留30天回顾
# nday, n_lat, n_lon, n_products 已在前面定义

valid_time_range = slice(max_lookback, nday)
n_valid_days = nday - max_lookback

# 长江数据不需要基于 Y NaNs 的 valid_mask 计算
print(f"Max lookback: {max_lookback} days")
print(f"Valid time range: Day index {max_lookback} to {nday-1}")

# 对齐 X 和 Y
X_aligned = X[valid_time_range] # Shape: (n_valid_days, n_lat, n_lon, n_products)
Y_aligned = Y[valid_time_range] # Shape: (n_valid_days, n_lat, n_lon)
print(f"Aligned X shape: {X_aligned.shape}")
print(f"Aligned Y shape: {Y_aligned.shape}")

start_feat = time.time()
features_dict = {}
epsilon = 1e-6
RAIN_THR = 0.1 # 计数降雨产品的阈值

# --- 4. 特征工程 ---
# --- 4.1 基础特征 ---
print("  Calculating basic features...")
features_dict['raw_values'] = X_aligned # 原始值

# --- 4.2 多产品一致性/差异特征 (简化) ---
print("  Calculating simplified multi-product stats (current time)...")
X_rain = (np.nan_to_num(X_aligned, nan=0.0) > RAIN_THR)
# 在产品维度 (axis=3) 上求和
features_dict['rain_product_count'] = np.sum(X_rain, axis=3, keepdims=True).astype(np.float32)
del X_rain

# --- 4.3 时间演化特征 (简化 + sin/cos 日期) ---
# 4.3.1 周期性 (重新添加 sin/cos 日期)
print("  Calculating periodicity features (Season + Sin/Cos Day)...")
days_in_year = 365.25
day_index_original = np.arange(nday, dtype=np.float32)
day_of_year = day_index_original % days_in_year
# --- 重新添加 sin/cos 计算 ---
sin_time = np.sin(2 * np.pi * day_of_year / days_in_year).astype(np.float32)
cos_time = np.cos(2 * np.pi * day_of_year / days_in_year).astype(np.float32)

# 切片并扩展: (n_valid_days,) -> (n_valid_days, 1, 1, 1) -> 广播到 (n_valid_days, n_lat, n_lon, 1)
sin_time_aligned = sin_time[valid_time_range, np.newaxis, np.newaxis, np.newaxis] * np.ones((1, n_lat, n_lon, 1), dtype=np.float32)
cos_time_aligned = cos_time[valid_time_range, np.newaxis, np.newaxis, np.newaxis] * np.ones((1, n_lat, n_lon, 1), dtype=np.float32)
features_dict['sin_day'] = sin_time_aligned
features_dict['cos_day'] = cos_time_aligned

# --- 保留季节 One-Hot ---
month = (day_of_year // 30.4375).astype(int) % 12 + 1
season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0} # 定义季节映射
season = np.array([season_map[m] for m in month])
seasons_onehot = np.zeros((nday, 4), dtype=np.float32)
seasons_onehot[np.arange(nday), season] = 1
# 切片并扩展: (n_valid_days, 4) -> (n_valid_days, 1, 1, 4) -> 广播到 (n_valid_days, n_lat, n_lon, 4)
season_aligned_expanded = seasons_onehot[valid_time_range, np.newaxis, np.newaxis, :]
season_aligned = season_aligned_expanded * np.ones((1, n_lat, n_lon, 1), dtype=np.float32)
features_dict['season_onehot'] = season_aligned
del day_index_original, day_of_year, month, season, seasons_onehot, season_aligned_expanded, season_aligned, sin_time, cos_time, sin_time_aligned, cos_time_aligned # 清理变量

# 4.3.2 滞后特征 (简化)
print("  Calculating simplified lag features...")
lag_data_cache = {} # 缓存原始滞后数据
lag_std_cache = {} # 缓存滞后标准差
lag_rain_count_cache = {} # 缓存滞后降雨计数

for lag in [1, 2, 3]:
    print(f"    Lag {lag}...")
    lag_slice = slice(max_lookback - lag, nday - lag)
    lag_data = X[lag_slice] # Shape: (n_valid_days, n_lat, n_lon, n_products)
    lag_data_cache[lag] = lag_data
    features_dict[f'lag_{lag}_values'] = lag_data # 原始滞后值

    # 计算并缓存滞后统计量 (仅标准差和降雨计数)
    lag_std = np.nanstd(lag_data, axis=3, keepdims=True).astype(np.float32) # 在产品维度 (axis=3) 计算
    lag_rain = (np.nan_to_num(lag_data, nan=0.0) > RAIN_THR)
    lag_rain_count = np.sum(lag_rain, axis=3, keepdims=True).astype(np.float32) # 在产品维度 (axis=3) 计算

    lag_std_cache[lag] = lag_std
    lag_rain_count_cache[lag] = lag_rain_count

    features_dict[f'lag_{lag}_std'] = lag_std # 滞后标准差
    features_dict[f'lag_{lag}_rain_count'] = lag_rain_count # 滞后降雨产品计数

# 4.3.3 差异特征 (简化)
print("  Calculating difference features (Values only)...")
prev_data = lag_data_cache[1] # t-1 时刻数据 (已切片)
features_dict['diff_1_values'] = (X_aligned - prev_data).astype(np.float32)
print("    Deleting raw lag data cache...")
del lag_data_cache, prev_data

# 4.3.4 滑动窗口特征 (使用 bottleneck 优化)
print("  Calculating moving window features (on product mean) using Bottleneck...")
product_mean_full_squeezed = np.nanmean(X, axis=3).astype(np.float32) # Shape: (nday, n_lat, n_lon)

for window in [3, 7, 15]:
    print(f"    Window {window} (on product mean)...")
    full_window_mean = bn.move_mean(product_mean_full_squeezed, window=window, axis=0, min_count=1)
    full_window_std = bn.move_std(product_mean_full_squeezed, window=window, axis=0, min_count=1)
    full_window_max = bn.move_max(product_mean_full_squeezed, window=window, axis=0, min_count=1)
    full_window_min = bn.move_min(product_mean_full_squeezed, window=window, axis=0, min_count=1)

    window_mean_aligned = full_window_mean[valid_time_range][..., np.newaxis]
    window_std_aligned = full_window_std[valid_time_range][..., np.newaxis]
    window_max_aligned = full_window_max[valid_time_range][..., np.newaxis]
    window_min_aligned = full_window_min[valid_time_range][..., np.newaxis]

    features_dict[f'window_{window}_mean'] = window_mean_aligned.astype(np.float32)
    features_dict[f'window_{window}_std'] = window_std_aligned.astype(np.float32)
    features_dict[f'window_{window}_max'] = window_max_aligned.astype(np.float32)
    features_dict[f'window_{window}_min'] = window_min_aligned.astype(np.float32)
    features_dict[f'window_{window}_range'] = (window_max_aligned - window_min_aligned).astype(np.float32)
del product_mean_full_squeezed

window = 7 # 特定窗口大小的特定产品特征
print(f"    Window {window} (per product - GSMAP, PERSIANN) using Bottleneck...")
gsmap_idx = product_names.index("GSMAP")
persiann_idx = product_names.index("PERSIANN")

X_gsmap_full = X[:, :, :, gsmap_idx].astype(np.float32) # (nday, n_lat, n_lon)
X_persiann_full = X[:, :, :, persiann_idx].astype(np.float32) # (nday, n_lat, n_lon)

full_window_mean_gsmap = bn.move_mean(X_gsmap_full, window=window, axis=0, min_count=1)
full_window_std_persiann = bn.move_std(X_persiann_full, window=window, axis=0, min_count=1)

features_dict[f'window_{window}_mean_GSMAP'] = full_window_mean_gsmap[valid_time_range][..., np.newaxis].astype(np.float32)
features_dict[f'window_{window}_std_PERSIANN'] = full_window_std_persiann[valid_time_range][..., np.newaxis].astype(np.float32)
del X_gsmap_full, X_persiann_full

# --- 4.4 空间上下文特征 (实现 5x5 和梯度) ---
print("  Calculating spatial features (5x5 neighborhood and Gradients)...")

# 4.4.2 5x5 邻域统计 (使用 maximum_filter 并为其他保留 generic_filter, 考虑并行化)
print("    Calculating 5x5 neighborhood statistics...")
spatial_mean_5x5_all = np.full((n_valid_days, n_lat, n_lon, n_products), np.nan, dtype=np.float32)
spatial_std_5x5_all = np.full((n_valid_days, n_lat, n_lon, n_products), np.nan, dtype=np.float32)
spatial_max_5x5_all = np.full((n_valid_days, n_lat, n_lon, n_products), np.nan, dtype=np.float32)

def calculate_spatial_stats_for_timestep(t_idx, current_X_aligned_t, num_products):
    s_mean_t = np.full((n_lat, n_lon, num_products), np.nan, dtype=np.float32)
    s_std_t = np.full((n_lat, n_lon, num_products), np.nan, dtype=np.float32)
    s_max_t = np.full((n_lat, n_lon, num_products), np.nan, dtype=np.float32)
    if (t_idx + 1) % (n_valid_days // 10 if n_valid_days > 10 else 1) == 0:
        print(f"      Processing 5x5 features for time step {t_idx + 1}/{n_valid_days}")
    for p_idx in range(num_products):
        data_slice_2d = current_X_aligned_t[:, :, p_idx]
        if np.all(np.isnan(data_slice_2d)):
            s_mean_t[:, :, p_idx] = np.nan
            s_std_t[:, :, p_idx] = np.nan
            s_max_t[:, :, p_idx] = np.nan
            continue
        s_mean_t[:, :, p_idx] = generic_filter(data_slice_2d, np.nanmean, size=5, mode='constant', cval=np.nan)
        s_std_t[:, :, p_idx] = generic_filter(data_slice_2d, np.nanstd, size=5, mode='constant', cval=np.nan)
        s_max_t[:, :, p_idx] = maximum_filter(data_slice_2d, size=5, mode='constant', cval=np.nan) # 使用 maximum_filter
    return t_idx, s_mean_t, s_std_t, s_max_t

results = Parallel(n_jobs=-1)(delayed(calculate_spatial_stats_for_timestep)(t, X_aligned[t], n_products) for t in range(n_valid_days))

for t_idx, s_mean_t, s_std_t, s_max_t in results:
    spatial_mean_5x5_all[t_idx] = s_mean_t
    spatial_std_5x5_all[t_idx] = s_std_t
    spatial_max_5x5_all[t_idx] = s_max_t

features_dict['spatial_mean_5x5'] = spatial_mean_5x5_all
features_dict['spatial_std_5x5'] = spatial_std_5x5_all
features_dict['spatial_max_5x5'] = spatial_max_5x5_all
features_dict['spatial_center_diff_5x5'] = (features_dict['raw_values'] - spatial_mean_5x5_all).astype(np.float32)

# 4.4.3 空间梯度 (向量化)
print("    Calculating spatial gradients (vectorized)...")
gradient_magnitude_gsmap = np.full((n_valid_days, n_lat, n_lon, 1), np.nan, dtype=np.float32)
gradient_magnitude_persiann = np.full((n_valid_days, n_lat, n_lon, 1), np.nan, dtype=np.float32)
gradient_magnitude_mean_prod = np.full((n_valid_days, n_lat, n_lon, 1), np.nan, dtype=np.float32)

X_gsmap_all_t = X_aligned[:, :, :, gsmap_idx]
valid_gsmap_mask = ~np.all(np.isnan(X_gsmap_all_t), axis=(1,2))
if np.any(valid_gsmap_mask):
    grad_lat_gsmap, grad_lon_gsmap = np.gradient(np.nan_to_num(X_gsmap_all_t[valid_gsmap_mask]), axis=(1, 2))
    gradient_magnitude_gsmap[valid_gsmap_mask, :, :, 0] = np.sqrt(grad_lat_gsmap**2 + grad_lon_gsmap**2)

X_persiann_all_t = X_aligned[:, :, :, persiann_idx]
valid_persiann_mask = ~np.all(np.isnan(X_persiann_all_t), axis=(1,2))
if np.any(valid_persiann_mask):
    grad_lat_persiann, grad_lon_persiann = np.gradient(np.nan_to_num(X_persiann_all_t[valid_persiann_mask]), axis=(1, 2))
    gradient_magnitude_persiann[valid_persiann_mask, :, :, 0] = np.sqrt(grad_lat_persiann**2 + grad_lon_persiann**2)

X_aligned_mean_across_products = np.nanmean(X_aligned, axis=3)
valid_mean_prod_mask = ~np.all(np.isnan(X_aligned_mean_across_products), axis=(1,2))
if np.any(valid_mean_prod_mask):
    grad_lat_mean, grad_lon_mean = np.gradient(np.nan_to_num(X_aligned_mean_across_products[valid_mean_prod_mask]), axis=(1, 2))
    gradient_magnitude_mean_prod[valid_mean_prod_mask, :, :, 0] = np.sqrt(grad_lat_mean**2 + grad_lon_mean**2)

features_dict['gradient_magnitude_GSMAP'] = gradient_magnitude_gsmap
features_dict['gradient_magnitude_PERSIANN'] = gradient_magnitude_persiann
features_dict['gradient_magnitude_mean'] = gradient_magnitude_mean_prod
del X_aligned_mean_across_products, gradient_magnitude_gsmap, gradient_magnitude_persiann, gradient_magnitude_mean_prod
del spatial_mean_5x5_all

# --- 4.5 低强度信号特征 (简化) ---
print("  Calculating simplified low intensity features...")
print("    Deleting lag stats caches...")
del lag_std_cache, lag_rain_count_cache # 删除滞后统计缓存
count_values = features_dict['rain_product_count'] # Shape: (n_valid_days, n_lat, n_lon, 1)
intensity_bins_count = np.zeros((n_valid_days, n_lat, n_lon, 4), dtype=np.float32)
intensity_bins_count[:, :, :, 0] = (count_values == 0).squeeze(axis=-1)
intensity_bins_count[:, :, :, 1] = (count_values == 1).squeeze(axis=-1)
intensity_bins_count[:, :, :, 2] = (count_values == 2).squeeze(axis=-1)
intensity_bins_count[:, :, :, 3] = (count_values >= 3).squeeze(axis=-1)
features_dict['intensity_bins_count'] = intensity_bins_count
del count_values, intensity_bins_count

end_feat = time.time()
print(f"Feature engineering finished in {end_feat - start_feat:.2f} seconds.")

# --- 5. 连接特征并生成名称 ---
print("\n--- Step 5: Concatenating Features & Naming (Yangtze v5 with Spatial) ---")
start_concat = time.time()
features_list_final = []
feature_names = []

# 使用与 national turn4.py 相同的命名逻辑
for name, feat_array in features_dict.items():
    if feat_array is None or not isinstance(feat_array, np.ndarray): continue
    try:
        # 特征通道维度现在是 axis 3
        n_cols = feat_array.shape[3]
        features_list_final.append(feat_array)
    except IndexError:
         print(f"Warning: Feature '{name}' has unexpected shape {feat_array.shape} (expected 4 dims). Skipping.")
         continue

    base_name = name
    if n_cols == 1:
        feature_names.append(base_name)
    elif base_name == 'raw_values' or \
         (base_name.startswith('lag_') and '_values' in base_name) or \
         base_name == 'diff_1_values' or \
         base_name.startswith('spatial_std_5x5') or \
         base_name.startswith('spatial_max_5x5') or \
         base_name.startswith('spatial_mean_5x5') or \
         base_name == 'spatial_center_diff_5x5':
        if n_cols == n_products:
            for i in range(n_cols): feature_names.append(f"{base_name}_{product_names[i]}")
        else:
            print(f"Warning: Mismatch in columns for {base_name}. Expected {n_products}, got {n_cols}. Using index.")
            for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    elif base_name == 'season_onehot':
        for i in range(n_cols): feature_names.append(f"{base_name}_{i}") # 0, 1, 2, 3 for seasons
    elif base_name == 'intensity_bins_count':
        bin_labels = ['0', '1', '2', '>=3']
        if n_cols == len(bin_labels):
             for i in range(n_cols): feature_names.append(f"{base_name}_{bin_labels[i]}")
        else:
             print(f"Warning: Mismatch in columns for {base_name}. Expected {len(bin_labels)}, got {n_cols}. Using index.")
             for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    else: # 回退机制
        if n_cols != 1: # 确保不会为单列特征重复打印警告
             print(f"Warning: Unhandled multi-column feature naming for '{base_name}' (shape {feat_array.shape}). Using index.")
             for i in range(n_cols): feature_names.append(f"{base_name}_{i}")

if not features_list_final:
    raise ValueError("No features were generated or added to the list for concatenation.")

total_features = len(feature_names) # 应该与 X_features.shape[3] 一致
print(f"Total calculated feature columns: {total_features}")
print(f"Length of feature_names list: {len(feature_names)}")
if len(feature_names) != sum(feat.shape[3] if feat.ndim == 4 else 1 for feat in features_list_final):
     print(f"FATAL: Mismatch between calculated total features and feature name list length!")
     # exit()

# 为长江 v5 定义输出目录和文件名
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features_spatial_v5") # 新目录名
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
X_features_memmap_filename = os.path.join(OUTPUT_DIR, "X_Yangtsu_features_concatenated_v5.mmap") # 临时 memmap 文件
X_flat_filename = os.path.join(OUTPUT_DIR, "X_Yangtsu_flat_features_spatial_v5.npy")
Y_flat_filename = os.path.join(OUTPUT_DIR, "Y_Yangtsu_flat_target_spatial_v5.npy")
feature_names_filename = os.path.join(OUTPUT_DIR, "feature_names_yangtsu_spatial_v5.txt")

# 保存特征名称
print(f"Saving feature names to {feature_names_filename}")
with open(feature_names_filename, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")

current_n_valid_days, current_n_lat, current_n_lon = features_list_final[0].shape[0], features_list_final[0].shape[1], features_list_final[0].shape[2]

print(f"Creating memory-mapped file for X_features at: {X_features_memmap_filename}")
X_features_shape = (current_n_valid_days, current_n_lat, current_n_lon, total_features)

calculated_total_features = sum(feat.shape[3] if feat.ndim == 4 else 1 for feat in features_list_final)
if total_features != calculated_total_features:
    print(f"Warning: Mismatch in total_features. From feature_names: {total_features}, from sum of shapes: {calculated_total_features}. Using sum of shapes.")
    total_features = calculated_total_features
    X_features_shape = (current_n_valid_days, current_n_lat, current_n_lon, total_features)

X_features_mmap = np.memmap(X_features_memmap_filename, dtype=np.float32, mode='w+', shape=X_features_shape)

print("Concatenating features into memory-mapped array...")
current_col_offset = 0
for i, feat_array in enumerate(features_list_final):
    print(f"  Concatenating feature group {i+1}/{len(features_list_final)}, shape: {feat_array.shape}")
    num_feat_channels = feat_array.shape[3] if feat_array.ndim == 4 else 1
    if feat_array.ndim == 3: # 例如单个通道的特征被存储为 (days, lat, lon)
        X_features_mmap[:, :, :, current_col_offset:current_col_offset+1] = feat_array[:,:,:,np.newaxis]
    else: # 已经是 (days, lat, lon, channels)
        X_features_mmap[:, :, :, current_col_offset:current_col_offset+num_feat_channels] = feat_array
    current_col_offset += num_feat_channels
    del features_list_final[i] # 尝试逐步释放内存

del features_list_final # 确保列表被清空

print(f"Concatenated features shape (mmap): {X_features_mmap.shape}")
end_concat = time.time()
print(f"Concatenation and naming finished in {end_concat - start_concat:.2f} seconds.")

# --- 6. 展平数据以供模型输入 ---
print("\n--- Step 6: Flattening Data (Yangtze v5 with Spatial) ---")
start_flat = time.time()

n_samples = X_features_mmap.shape[0] * X_features_mmap.shape[1] * X_features_mmap.shape[2]

print(f"Saving flattened X_features (from mmap) to: {X_flat_filename}")

X_flat_mmap_filename = os.path.join(OUTPUT_DIR, "X_Yangtsu_flat_features_spatial_v5_temp.mmap")
X_flat_mmap = np.memmap(X_flat_mmap_filename, dtype=np.float32, mode='w+', shape=(n_samples, total_features))

current_row_offset = 0
num_pixels_per_timestep = X_features_mmap.shape[1] * X_features_mmap.shape[2]
for t_idx in range(X_features_mmap.shape[0]):
    if (t_idx + 1) % (X_features_mmap.shape[0] // 20 if X_features_mmap.shape[0] > 20 else 1) == 0:
        print(f"  Flattening time step {t_idx+1}/{X_features_mmap.shape[0]}")
    time_slice_data = X_features_mmap[t_idx, :, :, :] # Shape (n_lat, n_lon, total_features)
    time_slice_flat = time_slice_data.reshape(num_pixels_per_timestep, total_features)
    X_flat_mmap[current_row_offset : current_row_offset + num_pixels_per_timestep, :] = time_slice_flat
    current_row_offset += num_pixels_per_timestep

print(f"Saving final X_flat from temporary mmap to {X_flat_filename}")
np.save(X_flat_filename, X_flat_mmap)
del X_flat_mmap # 删除 mmap 对象
os.remove(X_flat_mmap_filename) # 删除临时的 mmap 文件

del X_features_mmap # 删除 mmap 对象

Y_flat = Y_aligned.reshape(n_samples) # Y 通常较小，可以直接 reshape
del Y_aligned # 释放内存

print(f"Flattened X shape: ({n_samples}, {total_features})") # 从 mmap 形状获取
print(f"Flattened Y shape: {Y_flat.shape}")

print("Loading X_flat to handle NaNs (this might take a while and memory)...")
X_flat_loaded = np.load(X_flat_filename, mmap_mode='r+') # 以可写内存映射模式打开
print("Applying np.nan_to_num to X_flat...")
try:
    temp_X_flat = np.array(X_flat_loaded) # 尝试加载到内存
    np.nan_to_num(temp_X_flat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.save(X_flat_filename, temp_X_flat)
    del temp_X_flat
except MemoryError:
    print("MemoryError during nan_to_num on full X_flat. Consider chunked NaN handling if this occurs.")
del X_flat_loaded

if np.isnan(Y_flat).any():
    print("Warning: NaNs found in flattened Y target data! This should not happen if source Y is clean.")

# --- 7. 保存数据 ---
print("\n--- Step 7: Saving Flattened Data (Yangtze v5 with Spatial) ---")
start_save = time.time()
np.save(Y_flat_filename, Y_flat)
print(f"Flattened features saved to: {X_flat_filename}")
print(f"Flattened target saved to: {Y_flat_filename}")
end_save = time.time()
print(f"Saving finished in {end_save - start_save:.2f} seconds.")

print(f"\nTotal processing time: {time.time() - start_load:.2f} seconds")
print("Data processing complete.")
