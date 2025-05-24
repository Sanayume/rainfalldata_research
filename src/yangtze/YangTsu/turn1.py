from loaddata import mydata # 假设 loaddata.py 在同一目录或可访问
import numpy as np
import os
import time
from scipy.ndimage import generic_filter # For median filter (目前未使用，但保留)

# --- 1. 数据加载和初始准备 ---
# --- 1. Data Loading and Initial Preparation ---
data_loader = mydata()
# 从 loaddata 获取指定流域的空间格点数据 (产品, 时间, 纬度, 经度) 和目标数据 (时间, 纬度, 经度)
# 假设 basin_mask_value=2 对应长江流域 (原 yangtsu 方法暗示)
X_spatial, Y_spatial = data_loader.get_basin_spatial_data(basin_mask_value=2)
product_names = data_loader.get_products() # ["CMORPH", "CHIRPS", ...]

print(f"Initial X_spatial shape (products, time, lat, lon): {X_spatial.shape}")
print(f"Initial Y_spatial shape (time, lat, lon): {Y_spatial.shape}")
print(f"Product names: {product_names}")

# --- 2. 数据整合与重塑 ---
# --- 2. Data Integration & Reshaping ---
print("\n--- Step 2: Reshaping ---")
# 将 X 的维度从 (产品, 时间, 纬度, 经度) 转换为 (时间, 纬度, 经度, 产品)
X = np.transpose(X_spatial, (1, 2, 3, 0)).astype(np.float32)
Y = Y_spatial.astype(np.float32) # 确保 Y 的数据类型正确
del X_spatial, Y_spatial # 释放原始数据内存

print(f"Transposed X shape (time, lat, lon, product): {X.shape}")
print(f"Y shape (time, lat, lon): {Y.shape}")

# --- 3. 应用掩码和处理时间依赖性 ---
# --- 3. Apply Mask & Handle Time Dependency ---
print("\n--- Step 3: Masking & Time Alignment ---")
max_lookback = 30 # 保留30天作为最大回溯期，用于计算滑动窗口等特征
nday, nlat, nlon, nproduct = X.shape

# 定义有效的时间范围，排除掉 max_lookback 之前的数据
valid_time_range = slice(max_lookback, nday)
n_valid_days = nday - max_lookback

# 基于目标变量 Y 创建完整数据的有效性掩码 (非 NaN 的点为 True)
# Y 的形状是 (时间, 纬度, 经度)
valid_mask_full = ~np.isnan(Y)
# 获取在有效时间范围内的掩码
valid_mask = valid_mask_full[valid_time_range] # 形状: (n_valid_days, nlat, nlon)

# 计算有效样本总数 (在有效时间内，Y 不为 NaN 的格点数)
n_valid_samples = np.sum(valid_mask)
print(f"Max lookback: {max_lookback} days")
print(f"Valid time range: Day index {max_lookback} to {nday-1}")
print(f"Shape of valid mask (aligned): {valid_mask.shape}")
print(f"Total valid samples for training/evaluation: {n_valid_samples}")

# 对齐 Y 数据到有效时间范围
Y_aligned = Y[valid_time_range] # 形状: (n_valid_days, nlat, nlon)
print(f"Aligned Y shape: {Y_aligned.shape}")

# --- 4. 特征工程 ---
# --- 4. Feature Engineering ---
print("\n--- Step 4: Feature Engineering ---")
start_feat = time.time()
features_dict = {} # 用于存储所有生成的特征
epsilon = 1e-6 # 用于安全除法的小常数
RAIN_THR = 0.1 # 定义降雨阈值

# 安全除法辅助函数
def safe_divide(numerator, denominator, default=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / (denominator + epsilon)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

# --- 4.1 基础特征 ---
# --- 4.1 Basic Features ---
print("  Calculating basic features...")
# 获取在有效时间范围内的 X 数据
X_valid_time = X[valid_time_range] # 形状: (n_valid_days, nlat, nlon, nproduct)
features_dict['raw_values'] = X_valid_time

# --- 4.2 多产品一致性/差异特征 ---
# --- 4.2 Multi-product Consistency/Difference Features ---
print("  Calculating multi-product stats (current time)...")
# 计算当前时刻，降雨量超过阈值的产品数量
X_rain = (np.nan_to_num(X_valid_time, nan=0.0) > RAIN_THR)
features_dict['rain_product_count'] = np.sum(X_rain, axis=3, keepdims=True).astype(np.float32) #沿产品维度求和
del X_rain

# --- 4.3 时间演化特征 ---
# --- 4.3 Temporal Evolution Features ---
print("  Calculating periodicity features (Season only)...")
days_in_year = 365.25
day_index_original = np.arange(nday, dtype=np.float32) # 原始数据的每日索引
day_of_year = day_index_original % days_in_year
month = (day_of_year // 30.4375).astype(int) % 12 + 1 # 近似月份
season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0} # 季节映射: 0-冬, 1-春, 2-夏, 3-秋
season = np.array([season_map[m] for m in month])
seasons_onehot = np.zeros((nday, 4), dtype=np.float32) # (原始总天数, 4个季节)
seasons_onehot[np.arange(nday), season] = 1
# 扩展季节特征到 (天, 纬度, 经度, 4个季节) 以匹配 X 的维度结构
season_expanded = np.reshape(seasons_onehot, (nday, 1, 1, 4)) * np.ones((1, nlat, nlon, 1), dtype=np.float32)
features_dict['season_onehot'] = season_expanded[valid_time_range] # 取有效时间范围
del day_index_original, day_of_year, month, season, seasons_onehot, season_expanded

print("  Calculating lag features...")
lag_data_cache = {} # 缓存滞后数据
for lag in [1, 2, 3]: # 计算滞后1, 2, 3天的数据
    print(f"    Lag {lag}...")
    # 注意：这里是从原始X中取滞后数据，所以索引要对应好
    lag_slice = slice(max_lookback - lag, nday - lag)
    lag_data = X[lag_slice] # 形状: (n_valid_days, nlat, nlon, nproduct)
    lag_data_cache[lag] = lag_data
    features_dict[f'lag_{lag}_values'] = lag_data

print("  Calculating difference features...")
current_data = X_valid_time # 当前有效时间的数据
prev_data = lag_data_cache[1] # 滞后1天的数据
features_dict['diff_1_values'] = (current_data - prev_data).astype(np.float32)
del lag_data_cache, prev_data # 清理缓存

print("  Calculating moving window features...")
# 计算各产品在有效时间内的局部均值，用于后续窗口计算的参考（如果需要）
# 此处 product_mean_local 未在后续窗口计算中直接使用，但计算逻辑保留
product_mean_local = np.nanmean(X_valid_time, axis=3, keepdims=True).astype(np.float32) # (n_valid_days, nlat, nlon, 1)
for window in [3, 7, 15]: # 滑动窗口大小
    print(f"    Window {window}...")
    # 初始化存储窗口统计特征的数组
    window_mean = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    window_std = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    window_max = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    window_min = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    # 遍历有效天数，计算每个时间点的回溯窗口特征
    for i in range(n_valid_days):
        current_original_idx = i + max_lookback # 当前有效天在原始数据中的索引
        # 从原始X中提取窗口数据 (window_size, nlat, nlon, nproduct)
        window_data_X = X[current_original_idx - window : current_original_idx]
        # 首先计算窗口内每个时间点、每个格点上所有产品的均值 (window_size, nlat, nlon, 1)
        window_data_mean_prod = np.nanmean(window_data_X, axis=3, keepdims=True)
        # 然后计算这个产品均值在时间窗口上的统计量
        window_mean[i] = np.nanmean(window_data_mean_prod, axis=0) # (nlat, nlon, 1)
        window_std[i] = np.nanstd(window_data_mean_prod, axis=0)
        window_max[i] = np.nanmax(window_data_mean_prod, axis=0)
        window_min[i] = np.nanmin(window_data_mean_prod, axis=0)
    features_dict[f'window_{window}_mean'] = window_mean.astype(np.float32)
    features_dict[f'window_{window}_std'] = window_std.astype(np.float32)
    features_dict[f'window_{window}_max'] = window_max.astype(np.float32)
    features_dict[f'window_{window}_min'] = window_min.astype(np.float32)
    features_dict[f'window_{window}_range'] = (window_max - window_min).astype(np.float32)
del product_mean_local

print("  Calculating spatial features...")
# 初始化存储空间特征的数组，形状为 (n_valid_days, nlat, nlon, nproduct)
# 因为空间特征是针对每个产品独立计算的
spatial_mean_5x5 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_std_5x5 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_max_5x5 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)

# 遍历格点，计算5x5邻域的空间统计特征
# 边界像素(2个像素宽)不计算，保留为NaN
for r_idx in range(2, nlat - 2): # r_idx for row/latitude
    for c_idx in range(2, nlon - 2): # c_idx for col/longitude
        # 提取5x5邻域数据，形状 (n_valid_days, 5, 5, nproduct)
        neighborhood = X_valid_time[:, r_idx-2:r_idx+3, c_idx-2:c_idx+3, :]
        # 计算邻域内空间统计量，结果形状 (n_valid_days, nproduct)
        spatial_mean_5x5[:, r_idx, c_idx, :] = np.nanmean(neighborhood, axis=(1, 2))
        spatial_std_5x5[:, r_idx, c_idx, :] = np.nanstd(neighborhood, axis=(1, 2))
        spatial_max_5x5[:, r_idx, c_idx, :] = np.nanmax(neighborhood, axis=(1, 2))

features_dict['spatial_mean_5x5'] = spatial_mean_5x5.astype(np.float32)
features_dict['spatial_std_5x5'] = spatial_std_5x5.astype(np.float32)
features_dict['spatial_max_5x5'] = spatial_max_5x5.astype(np.float32)
# del spatial_mean_5x5 # 保留，因为它是字典中的一个值，后续会用到

end_feat = time.time()
print(f"Feature engineering finished in {end_feat - start_feat:.2f} seconds.")

# --- 5. 扁平化数据以供模型输入 ---
# --- 5. Flattening Data for Model Input ---
print("\n--- Step 5: Flattening Data ---")
start_flat = time.time()
# 计算总特征数量 (所有特征字典中数组的最后一个维度之和)
total_features = sum(feat.shape[3] for feat in features_dict.values())
print(f"Total number of calculated feature columns: {total_features}")

feature_names = [] # 存储特征名称
for name, feat_array in features_dict.items():
    n_cols = feat_array.shape[3] # 当前特征数组的通道数/产品数
    base_name = name
    if n_cols == 1: # 单通道特征
        feature_names.append(base_name)
    # 对于原始值、滞后值、差分值、空间特征，它们是每个产品一个特征
    elif base_name == 'raw_values' or base_name.startswith('lag_') or \
         base_name.startswith('diff_1_values') or base_name.startswith('spatial_'):
        for i in range(n_cols): # n_cols 应该是 nproduct
            feature_names.append(f"{base_name}_{product_names[i]}")
    elif base_name == 'season_onehot': # 季节one-hot编码有4个特征
        for i in range(n_cols): # n_cols 应该是 4
            feature_names.append(f"{base_name}_{i}")
    else: # 其他多通道特征（如果存在）
        for i in range(n_cols):
            feature_names.append(f"{base_name}_{i}")

print(f"Length of feature_names list: {len(feature_names)}")
if len(feature_names) != total_features:
    print(f"FATAL: Mismatch between calculated total features ({total_features}) and feature name list length ({len(feature_names)})!")
    # Consider exiting or raising an error if this happens
    # exit()

# 定义输出目录和文件名
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features_gridded_input")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
X_flat_filename = os.path.join(OUTPUT_DIR, "X_flat_features_v2_gridded.npy")
Y_flat_filename = os.path.join(OUTPUT_DIR, "Y_flat_target_v2_gridded.npy")
feature_names_filename = os.path.join(OUTPUT_DIR, "feature_names_v2_gridded.txt")

print(f"Saving feature names to {feature_names_filename}")
with open(feature_names_filename, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")

# 扁平化后的 X 和 Y 的形状
# n_valid_samples 是 (有效时间范围内的天数 * 有效纬度数 * 有效经度数) 中 Y 非 NaN 的总点数
X_flat_shape = (int(n_valid_samples), int(total_features))
Y_flat_shape = (int(n_valid_samples),)

print(f"Creating memory-mapped file: {X_flat_filename} with shape {X_flat_shape}")
os.makedirs(os.path.dirname(X_flat_filename), exist_ok=True) # 确保目录存在
X_flat_mmap = np.lib.format.open_memmap(X_flat_filename, mode='w+', dtype=np.float32, shape=X_flat_shape)

print(f"Creating memory-mapped file: {Y_flat_filename} with shape {Y_flat_shape}")
os.makedirs(os.path.dirname(Y_flat_filename), exist_ok=True) # 确保目录存在
Y_flat_mmap = np.lib.format.open_memmap(Y_flat_filename, mode='w+', dtype=np.float32, shape=Y_flat_shape)

print("Starting flattening process...")
current_mmap_idx = 0 # 当前写入 memmap 的索引
# 遍历有效天数
for t in range(n_valid_days):
    # 获取当天的空间掩码 (nlat, nlon)
    day_mask = valid_mask[t]
    # 获取当天有效的空间格点索引 (lat_indices, lon_indices)
    day_valid_indices = np.where(day_mask)
    n_day_valid = len(day_valid_indices[0]) # 当天有效的格点数

    if n_day_valid == 0: # 如果当天没有有效格点，则跳过
        continue

    # 初始化当天扁平化的 X 特征数组
    day_X_flat = np.zeros((n_day_valid, total_features), dtype=np.float32)
    col_idx = 0 # 当前特征列的起始索引
    # 遍历特征字典中的每个特征数组
    for name, feat_array in features_dict.items():
        n_feat_channels = feat_array.shape[3] # 当前特征的通道数
        # 提取当前特征在当天、有效格点的数据
        # feat_array[t] 的形状是 (nlat, nlon, n_feat_channels)
        # feat_day_valid 的形状将是 (n_day_valid, n_feat_channels)
        feat_day_valid = feat_array[t][day_valid_indices]

        # 如果特征只有一个通道且提取后是一维的，则增加一个维度以匹配 (n_day_valid, 1)
        if n_feat_channels == 1 and feat_day_valid.ndim == 1:
            feat_day_valid = feat_day_valid[:, np.newaxis]
        
        # 处理可能的 NaN/inf 值
        feat_day_valid = np.nan_to_num(feat_day_valid, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 将提取并处理后的特征数据填入当天的扁平化 X 数组
        day_X_flat[:, col_idx : col_idx + n_feat_channels] = feat_day_valid
        col_idx += n_feat_channels

    # 提取当天有效格点的 Y 值
    # Y_aligned[t] 的形状是 (nlat, nlon)
    # day_Y_flat 的形状将是 (n_day_valid,)
    day_Y_flat = Y_aligned[t][day_valid_indices]
    day_Y_flat = np.nan_to_num(day_Y_flat, nan=0.0, posinf=0.0, neginf=0.0) # 以防万一，Y也处理下

    # 计算写入 memmap 的起止索引
    write_start_idx = current_mmap_idx
    write_end_idx = current_mmap_idx + n_day_valid

    # 防止写入超出 memmap 预设大小 (理论上不应发生，如果 n_valid_samples 计算正确)
    if write_end_idx > X_flat_shape[0]:
        print(f"Warning: Attempting to write beyond mmap capacity. Truncating. Day {t}, n_day_valid {n_day_valid}")
        write_end_idx = X_flat_shape[0]
        n_day_valid_adjusted = X_flat_shape[0] - write_start_idx
        if n_day_valid_adjusted <= 0:
            print(f"Warning: No space left in mmap. Skipping remaining data from day {t}.")
            continue
        day_X_flat = day_X_flat[:n_day_valid_adjusted]
        day_Y_flat = day_Y_flat[:n_day_valid_adjusted]
        n_day_valid = n_day_valid_adjusted # 更新实际写入的样本数

    # 将当天扁平化后的数据写入 memmap 文件
    X_flat_mmap[write_start_idx:write_end_idx] = day_X_flat
    Y_flat_mmap[write_start_idx:write_end_idx] = day_Y_flat
    current_mmap_idx = write_end_idx # 更新下一个写入位置

    # 打印进度
    if (t + 1) % 50 == 0 or t == n_valid_days - 1:
        progress_percent = (t + 1) / n_valid_days * 100
        # 原始天数索引是 t + max_lookback
        print(f"  Processed day {t+max_lookback}/{nday-1} (valid day {t+1}/{n_valid_days}). Samples written: {current_mmap_idx}/{X_flat_shape[0]} ({progress_percent:.1f}%)")

# 刷新 memmap 缓存到磁盘
X_flat_mmap.flush()
Y_flat_mmap.flush()
del X_flat_mmap # 删除 memmap 对象，关闭文件
del Y_flat_mmap
del features_dict # 清理特征字典，释放内存

end_flat = time.time()
print(f"\nFlattening process complete in {end_flat - start_flat:.2f} seconds.")
if current_mmap_idx != X_flat_shape[0]:
    print(f"Warning: Final written sample count ({current_mmap_idx}) does not match calculated total valid samples ({X_flat_shape[0]}).")
else:
    print(f"Successfully wrote {current_mmap_idx} samples.")

print(f"Flattened features saved to: {X_flat_filename}")
print(f"Flattened target saved to: {Y_flat_filename}")
print("Data processing complete.")
