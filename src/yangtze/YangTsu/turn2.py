from loaddata import mydata # 假设 loaddata.py 在同一目录或可访问
import numpy as np
import os
import time
import scipy.ndimage # 导入scipy进行空间特征计算

# --- 1. 数据加载和初始准备 ---
print("--- Step 1: Data Loading (Yangtze) ---")
start_load = time.time()
ALL_DATA = mydata()
# 加载长江流域的空间数据 (n_products, time, lat, lon) 和 (time, lat, lon)
# X_raw, Y_raw, _, _ = ALL_DATA.yangtsu() # 旧的加载方式，可能返回点数据
X_raw, Y_raw = ALL_DATA.get_basin_spatial_data(basin_mask_value=2) # 假设长江流域的掩码值为2

product_names = ALL_DATA.get_products()
n_products, nday, n_lat, n_lon = X_raw.shape # X_raw: (产品数, 时间, 纬度, 经度)
print(f"Initial X_raw shape: {X_raw.shape}") # 应该是 (产品数, 时间, 纬度, 经度)
print(f"Initial Y_raw shape: {Y_raw.shape}") # 应该是 (时间, 纬度, 经度)
print(f"Product names: {product_names}")
print(f"Grid dimensions: Lat={n_lat}, Lon={n_lon}")
end_load = time.time()
print(f"Data loading finished in {end_load - start_load:.2f} seconds.")

# --- 2. 数据整合与重塑 ---
print("\n--- Step 2: Reshaping ---")
# 将 X 转置为 (时间, 纬度, 经度, 产品数)
X = np.transpose(X_raw, (1, 2, 3, 0)).astype(np.float32)
del X_raw # 释放内存
Y = Y_raw.astype(np.float32) # 确保正确的dtype
del Y_raw # 释放内存
print(f"Transposed X shape: {X.shape}") # (时间, 纬度, 经度, 产品数)
print(f"Y shape: {Y.shape}")   # (时间, 纬度, 经度)

# --- 3. 处理时间依赖性 ---
print("\n--- Step 3: Time Alignment ---")
max_lookback = 30 # 保留30天回顾
# nday, n_lat, n_lon, n_products 已在前面定义

valid_time_range = slice(max_lookback, nday)
n_valid_days = nday - max_lookback

# 长江数据不需要基于Y NaNs的valid_mask计算
print(f"Max lookback: {max_lookback} days")
print(f"Valid time range: Day index {max_lookback} to {nday-1}")

# 对齐 X 和 Y
X_aligned = X[valid_time_range] # Shape: (有效天数, 纬度, 经度, 产品数)
Y_aligned = Y[valid_time_range] # Shape: (有效天数, 纬度, 经度)
print(f"Aligned X shape: {X_aligned.shape}")
print(f"Aligned Y shape: {Y_aligned.shape}")

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
features_dict['raw_values'] = X_aligned # Shape: (有效天数, 纬度, 经度, 产品数)

# --- 4.2 多产品一致性/差异特征 ---
print("  Calculating multi-product stats (Rain Count + Disagreement)...")
X_rain = (np.nan_to_num(X_aligned, nan=0.0) > RAIN_THR)
features_dict['rain_product_count'] = np.sum(X_rain, axis=3, keepdims=True).astype(np.float32) # axis=3 是产品维度
del X_rain

# 计算产品维度 (axis=3) 的统计量
product_mean = np.nanmean(X_aligned, axis=3, keepdims=True).astype(np.float32)
product_std = np.nanstd(X_aligned, axis=3, keepdims=True).astype(np.float32)
product_max = np.nanmax(X_aligned, axis=3, keepdims=True).astype(np.float32)
product_min = np.nanmin(X_aligned, axis=3, keepdims=True).astype(np.float32)

features_dict['coef_of_variation'] = safe_divide(product_std, product_mean).astype(np.float32)
features_dict['product_range'] = (product_max - product_min).astype(np.float32)

# --- 4.3 时间演化特征 (简化) ---
print("  Calculating periodicity features (Season only)...")
days_in_year = 365.25
day_index_original = np.arange(nday, dtype=np.float32)
day_of_year = day_index_original % days_in_year
month = (day_of_year // 30.4375).astype(int) % 12 + 1
season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0} # 0:冬, 1:春, 2:夏, 3:秋
season = np.array([season_map[m] for m in month])
seasons_onehot = np.zeros((nday, 4), dtype=np.float32)
seasons_onehot[np.arange(nday), season] = 1
# 切片并扩展: (有效天数, 4) -> (有效天数, 1, 1, 4) -> 广播到 (有效天数, 纬度, 经度, 4)
season_aligned = seasons_onehot[valid_time_range, np.newaxis, np.newaxis, :] * np.ones((1, n_lat, n_lon, 1), dtype=np.float32)
features_dict['season_onehot'] = season_aligned
del day_index_original, day_of_year, month, season, seasons_onehot, season_aligned

print("  Calculating simplified lag features...")
lag_data_cache = {}
lag_std_cache = {}
lag_rain_count_cache = {}
for lag in [1, 2, 3]:
    print(f"    Lag {lag}...")
    lag_slice = slice(max_lookback - lag, nday - lag)
    lag_data = X[lag_slice] # Shape: (有效天数, 纬度, 经度, 产品数)
    lag_data_cache[lag] = lag_data
    features_dict[f'lag_{lag}_values'] = lag_data
    lag_std = np.nanstd(lag_data, axis=3, keepdims=True).astype(np.float32) # axis=3 是产品维度
    lag_rain = (np.nan_to_num(lag_data, nan=0.0) > RAIN_THR)
    lag_rain_count = np.sum(lag_rain, axis=3, keepdims=True).astype(np.float32) # axis=3 是产品维度
    lag_std_cache[lag] = lag_std
    lag_rain_count_cache[lag] = lag_rain_count
    features_dict[f'lag_{lag}_std'] = lag_std
    features_dict[f'lag_{lag}_rain_count'] = lag_rain_count

print("  Calculating difference features (Values only)...")
prev_data = lag_data_cache[1] # t-1 (已切片)
features_dict['diff_1_values'] = (X_aligned - prev_data).astype(np.float32)
print("    Deleting raw lag data cache...")
del lag_data_cache, prev_data

print("  Calculating moving window features (on product mean)...")
# 首先计算完整时间序列的产品平均值
product_mean_full = np.nanmean(X, axis=3, keepdims=True).astype(np.float32) # Shape: (总天数, 纬度, 经度, 1)
for window in [3, 7, 15]:
    print(f"    Window {window} (on product mean)...")
    window_mean = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
    window_std = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
    window_max = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
    window_min = np.zeros((n_valid_days, n_lat, n_lon, 1), dtype=np.float32)
    for i in range(n_valid_days):
        current_original_idx = i + max_lookback
        # window_data_mean_prod shape: (窗口大小, 纬度, 经度, 1)
        window_data_mean_prod = product_mean_full[current_original_idx - window : current_original_idx]
        window_mean[i] = np.nanmean(window_data_mean_prod, axis=0) # 沿时间轴(axis=0)聚合
        window_std[i] = np.nanstd(window_data_mean_prod, axis=0)
        window_max[i] = np.nanmax(window_data_mean_prod, axis=0)
        window_min[i] = np.nanmin(window_data_mean_prod, axis=0)
    features_dict[f'window_{window}_mean'] = window_mean.astype(np.float32)
    features_dict[f'window_{window}_std'] = window_std.astype(np.float32)
    features_dict[f'window_{window}_max'] = window_max.astype(np.float32)
    features_dict[f'window_{window}_min'] = window_min.astype(np.float32)
    features_dict[f'window_{window}_range'] = (window_max - window_min).astype(np.float32)
del product_mean_full

# --- 4.4 空间背景特征 (简化 - 仅5x5和梯度) ---
print("  Calculating spatial features (5x5 and Gradients)...")
# X_aligned shape: (有效天数, 纬度, 经度, 产品数)
# 4.4.2 5x5邻域特征
print("    Calculating 5x5 neighborhood features...")
footprint_5x5 = np.ones((5, 5), dtype=bool) # 5x5的卷积核/足迹

spatial_mean_5x5 = np.full_like(X_aligned, np.nan, dtype=np.float32)
spatial_std_5x5 = np.full_like(X_aligned, np.nan, dtype=np.float32)
spatial_max_5x5 = np.full_like(X_aligned, np.nan, dtype=np.float32)

for t in range(n_valid_days):
    if t % (n_valid_days // 10 + 1) == 0 : print(f"      5x5 features: processing day {t+1}/{n_valid_days}")
    for p in range(n_products):
        data_slice = X_aligned[t, :, :, p] # 当前时间和产品的2D空间切片 (纬度, 经度)
        if np.all(np.isnan(data_slice)): # 如果整个切片都是NaN，则跳过以避免警告
            spatial_mean_5x5[t, :, :, p] = np.nan
            spatial_std_5x5[t, :, :, p] = np.nan
            spatial_max_5x5[t, :, :, p] = np.nan
            continue
        
        # 使用 generic_filter 计算均值、标准差、最大值，处理边界NaN值
        # mode='constant', cval=np.nan 表示用NaN填充边界外区域
        # uniform_filter 对于均值更快，但 generic_filter 更通用
        spatial_mean_5x5[t, :, :, p] = scipy.ndimage.uniform_filter(data_slice, size=5, mode='constant', cval=np.nan)
        # spatial_mean_5x5[t, :, :, p] = scipy.ndimage.generic_filter(data_slice, np.nanmean, footprint=footprint_5x5, mode='constant', cval=np.nan)
        spatial_std_5x5[t, :, :, p] = scipy.ndimage.generic_filter(data_slice, np.nanstd, footprint=footprint_5x5, mode='constant', cval=np.nan)
        spatial_max_5x5[t, :, :, p] = scipy.ndimage.generic_filter(data_slice, np.nanmax, footprint=footprint_5x5, mode='constant', cval=np.nan)

features_dict['spatial_mean_5x5'] = spatial_mean_5x5
features_dict['spatial_std_5x5'] = spatial_std_5x5
features_dict['spatial_max_5x5'] = spatial_max_5x5
features_dict['spatial_center_diff_5x5'] = (features_dict['raw_values'] - spatial_mean_5x5).astype(np.float32)
# del spatial_mean_5x5 # spatial_mean_5x5 仍然被 spatial_center_diff_5x5 使用，在features_dict中删除

# 4.4.3 空间梯度特征
print("    Calculating spatial gradient features...")
# 初始化梯度特征数组
grad_mag_gsmap = np.full((n_valid_days, n_lat, n_lon, 1), np.nan, dtype=np.float32)
grad_mag_persiann = np.full((n_valid_days, n_lat, n_lon, 1), np.nan, dtype=np.float32)
grad_mag_mean_prod = np.full((n_valid_days, n_lat, n_lon, 1), np.nan, dtype=np.float32)

# 获取特定产品索引
try:
    gsmap_idx = product_names.index('GSMAP')
    persiann_idx = product_names.index('PERSIANN')
except ValueError:
    print("Warning: GSMAP or PERSIANN not found in product_names. Gradient features for them will be NaN.")
    gsmap_idx = -1 # 无效索引
    persiann_idx = -1 # 无效索引

product_mean_squeezed = product_mean.squeeze(axis=3) # Shape: (有效天数, 纬度, 经度)

for t in range(n_valid_days):
    if t % (n_valid_days // 10 + 1) == 0 : print(f"      Gradient features: processing day {t+1}/{n_valid_days}")
    # GSMAP 梯度
    if gsmap_idx != -1:
        data_slice_gsmap = X_aligned[t, :, :, gsmap_idx]
        if not np.all(np.isnan(data_slice_gsmap)):
            gy, gx = np.gradient(data_slice_gsmap) # 默认沿所有轴计算
            grad_mag_gsmap[t, :, :, 0] = np.sqrt(gy**2 + gx**2)
    
    # PERSIANN 梯度
    if persiann_idx != -1:
        data_slice_persiann = X_aligned[t, :, :, persiann_idx]
        if not np.all(np.isnan(data_slice_persiann)):
            gy, gx = np.gradient(data_slice_persiann)
            grad_mag_persiann[t, :, :, 0] = np.sqrt(gy**2 + gx**2)
            
    # 产品平均值的梯度
    data_slice_mean = product_mean_squeezed[t, :, :]
    if not np.all(np.isnan(data_slice_mean)):
        gy, gx = np.gradient(data_slice_mean)
        grad_mag_mean_prod[t, :, :, 0] = np.sqrt(gy**2 + gx**2)

features_dict['gradient_magnitude_GSMAP'] = grad_mag_gsmap
features_dict['gradient_magnitude_PERSIANN'] = grad_mag_persiann
features_dict['gradient_magnitude_mean'] = grad_mag_mean_prod # 基于产品平均值的梯度

# --- 4.5 低强度/模糊性特征 (增强) ---
print("  Calculating low intensity / ambiguity features...")
features_dict['threshold_proximity'] = np.abs(product_mean - RAIN_THR).astype(np.float32) # product_mean shape (有效天数, 纬度, 经度, 1)

# 报告低强度、非零降雨 (0 < rain <= 0.5) 的产品比例
low_range_mask = (X_aligned > epsilon) & (X_aligned <= 0.5) # Shape (有效天数, 纬度, 经度, 产品数)
low_range_count = np.sum(low_range_mask, axis=3, keepdims=True).astype(np.float32) # axis=3 是产品维度
features_dict['fraction_products_low_range'] = (low_range_count / n_products).astype(np.float32)
del low_range_mask, low_range_count

# 基于降雨产品计数的强度分箱
count_values = features_dict['rain_product_count'] # Shape (有效天数, 纬度, 经度, 1)
intensity_bins_count = np.zeros((n_valid_days, n_lat, n_lon, 4), dtype=np.float32)
intensity_bins_count[:, :, :, 0] = (count_values == 0).squeeze(axis=-1) # squeeze 最后一个维度
intensity_bins_count[:, :, :, 1] = (count_values == 1).squeeze(axis=-1)
intensity_bins_count[:, :, :, 2] = (count_values == 2).squeeze(axis=-1)
intensity_bins_count[:, :, :, 3] = (count_values >= 3).squeeze(axis=-1)
features_dict['intensity_bins_count'] = intensity_bins_count
del count_values, intensity_bins_count

print("    Deleting intermediate variables...")
del product_mean, product_std, product_max, product_min # 这些已经存入 features_dict 或用于计算其他特征
del lag_std_cache, lag_rain_count_cache
del X_aligned # 删除别名

end_feat = time.time()
print(f"Feature engineering finished in {end_feat - start_feat:.2f} seconds.")

# --- 5. 连接特征并生成名称 ---
print("\n--- Step 5: Concatenating Features & Naming (Yangtze v2) ---")
start_concat = time.time()
features_list_final = []
feature_names = []

# 使用与 national turn5.py 相同的命名逻辑
for name, feat_array in features_dict.items():
    if feat_array is None or not isinstance(feat_array, np.ndarray): continue
    try:
        # 对于空间数据，特征维度是 axis=3
        n_cols = feat_array.shape[3] 
        features_list_final.append(feat_array)
    except IndexError:
         print(f"Warning: Feature '{name}' has unexpected shape {feat_array.shape}. Skipping.")
         continue

    base_name = name
    if n_cols == 1:
        feature_names.append(base_name)
    # 调整那些原本具有产品维度的特征的命名
    elif base_name == 'raw_values' or \
         (base_name.startswith('lag_') and '_values' in base_name) or \
         base_name == 'diff_1_values' or \
         base_name.startswith('spatial_mean_5x5') or \
         base_name.startswith('spatial_std_5x5') or \
         base_name.startswith('spatial_max_5x5') or \
         base_name == 'spatial_center_diff_5x5':
        if n_cols == n_products:
            for i in range(n_cols): feature_names.append(f"{base_name}_{product_names[i]}")
        else:
            print(f"Warning: Mismatch in columns for {base_name}. Expected {n_products}, got {n_cols}. Using index.")
            for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    elif base_name == 'season_onehot': # 季节onehot编码有4列
        for i in range(n_cols): feature_names.append(f"{base_name}_{i}") # e.g., season_onehot_0, season_onehot_1
    elif base_name == 'intensity_bins_count': # 强度分箱有4列
        bin_labels = ['0', '1', '2', '>=3']
        if n_cols == len(bin_labels):
             for i in range(n_cols): feature_names.append(f"{base_name}_{bin_labels[i]}")
        else:
             print(f"Warning: Mismatch in columns for {base_name}. Expected {len(bin_labels)}, got {n_cols}. Using index.")
             for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    else: # 后备方案
        if n_cols != 1: # 确保不会为单列特征重复添加索引
             print(f"Warning: Unhandled multi-column feature naming for '{base_name}'. Using index.")
             for i in range(n_cols): feature_names.append(f"{base_name}_{i}")

# 沿特征轴 (axis=3) 连接
if not features_list_final:
    raise ValueError("No features were generated or added to the list for concatenation.")
X_features = np.concatenate(features_list_final, axis=3).astype(np.float32)
del features_list_final, features_dict # 释放内存

total_features = X_features.shape[3] # 特征数量在最后一个维度
print(f"Concatenated features shape: {X_features.shape}") # (有效天数, 纬度, 经度, 总特征数)
print(f"Total calculated feature columns: {total_features}")
print(f"Length of feature_names list: {len(feature_names)}")
if len(feature_names) != total_features:
     print(f"FATAL: Mismatch between calculated total features ({total_features}) and feature name list length ({len(feature_names)})!")
     # exit() # 暂时注释掉exit以便调试

# 为长江v2明确定义输出目录和文件名
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features_spatial_v2") # 修改目录名以区分
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
X_flat_filename = os.path.join(OUTPUT_DIR, "X_Yangtsu_flat_features_spatial_v2.npy")
Y_flat_filename = os.path.join(OUTPUT_DIR, "Y_Yangtsu_flat_target_spatial_v2.npy")
feature_names_filename = os.path.join(OUTPUT_DIR, "feature_names_yangtsu_spatial_v2.txt")

# 保存特征名称
print(f"Saving feature names to {feature_names_filename}")
os.makedirs(OUTPUT_DIR, exist_ok=True) # 再次确保目录存在
with open(feature_names_filename, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")
end_concat = time.time()
print(f"Concatenation and naming finished in {end_concat - start_concat:.2f} seconds.")

# --- 6. 展平数据以供模型输入 ---
print("\n--- Step 6: Flattening Data (Yangtze spatial v2) ---")
start_flat = time.time()
# 重塑 X: (有效天数, 纬度, 经度, 总特征数) -> (有效天数 * 纬度 * 经度, 总特征数)
n_samples = n_valid_days * n_lat * n_lon
X_flat = X_features.reshape(n_samples, total_features)
del X_features # 释放内存

# 重塑 Y: (有效天数, 纬度, 经度) -> (有效天数 * 纬度 * 经度,)
Y_flat = Y_aligned.reshape(n_samples)
del Y_aligned # 释放内存

print(f"Flattened X shape: {X_flat.shape}")
print(f"Flattened Y shape: {Y_flat.shape}")

# 处理特征计算中可能引入的NaN (尤其是空间特征的边界)
X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0) # 用0填充NaN/inf
# Y_flat 中的NaN通常表示该点/时间的数据无效，模型训练时可能需要特殊处理或移除
if np.isnan(Y_flat).any():
    print(f"Warning: NaNs found in flattened Y target data! Count: {np.sum(np.isnan(Y_flat))}. These samples might be excluded during training.")
    # Y_flat = np.nan_to_num(Y_flat, nan=-9999) # 或者用一个特殊值标记，取决于模型如何处理

end_flat = time.time()
print(f"Flattening finished in {end_flat - start_flat:.2f} seconds.")

# --- 7. 保存数据 ---
print("\n--- Step 7: Saving Flattened Data (Yangtze spatial v2) ---")
start_save = time.time()
np.save(X_flat_filename, X_flat)
np.save(Y_flat_filename, Y_flat)
print(f"Flattened features saved to: {X_flat_filename}")
print(f"Flattened target saved to: {Y_flat_filename}")
end_save = time.time()
print(f"Saving finished in {end_save - start_save:.2f} seconds.")

print(f"\nTotal processing time: {time.time() - start_load:.2f} seconds")
print("Data processing complete.")
