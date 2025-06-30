#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的单独特征生成系统
生成所有详细可用的特征，每个特征单独保存为.npy文件
空间特征基于真实空间数据计算，保持空间意义
"""

import numpy as np
import os
import time
import warnings
from scipy import ndimage
from scipy.stats import skew, kurtosis
from loaddata import mydata

# 配置
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
MAX_LOOKBACK = 30  # 最大回看天数
EPSILON = 1e-6     # 防止除零
RAIN_THR = 0.1     # 降雨阈值

def safe_divide(numerator, denominator, default=0.0):
    """安全除法，避免除零"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / (denominator + EPSILON)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

def save_feature(feature_data, feature_name, description=""):
    """保存特征到npy文件"""
    filepath = os.path.join(OUTPUT_DIR, f"{feature_name}.npy")
    np.save(filepath, feature_data.astype(np.float32))
    print(f"  Saved: {feature_name}.npy {feature_data.shape} - {description}")

def calculate_spatial_gradients(data_2d):
    """计算二维数据的空间梯度"""
    if len(data_2d.shape) != 2:
        raise ValueError("Input must be 2D array")
    
    # 使用Sobel算子计算梯度
    grad_x = ndimage.sobel(data_2d, axis=1)  # 经度方向梯度
    grad_y = ndimage.sobel(data_2d, axis=0)  # 纬度方向梯度
    
    # 梯度幅度
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 梯度方向（弧度）
    gradient_direction = np.arctan2(grad_y, grad_x)
    
    return grad_x, grad_y, gradient_magnitude, gradient_direction

def calculate_neighborhood_stats(data_2d, window_size):
    """计算邻域统计量"""
    if len(data_2d.shape) != 2:
        raise ValueError("Input must be 2D array")
    
    # 使用滑动窗口计算统计量
    half_window = window_size // 2
    padded = np.pad(data_2d, half_window, mode='constant', constant_values=np.nan)
    
    rows, cols = data_2d.shape
    mean_result = np.zeros_like(data_2d)
    std_result = np.zeros_like(data_2d)
    max_result = np.zeros_like(data_2d)
    min_result = np.zeros_like(data_2d)
    
    for i in range(rows):
        for j in range(cols):
            window = padded[i:i+window_size, j:j+window_size]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                mean_result[i, j] = np.mean(valid_window)
                std_result[i, j] = np.std(valid_window)
                max_result[i, j] = np.max(valid_window)
                min_result[i, j] = np.min(valid_window)
            else:
                mean_result[i, j] = np.nan
                std_result[i, j] = np.nan
                max_result[i, j] = np.nan
                min_result[i, j] = np.nan
    
    return mean_result, std_result, max_result, min_result

def main():
    print("=== 开始生成完整的单独特征库 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. 加载数据
    print("\n1. 加载长江流域数据...")
    start_time = time.time()
    ALL_DATA = mydata()
    
    # 加载空间数据用于真实空间特征计算
    X_spatial, Y_spatial = ALL_DATA.get_basin_spatial_data(basin_mask_value=2)
    print(f"空间数据形状: X_spatial {X_spatial.shape}, Y_spatial {Y_spatial.shape}")
    
    # 加载点数据用于非空间特征计算
    X_points, Y_points = ALL_DATA.get_basin_point_data(basin_mask_value=2)
    print(f"点数据形状: X_points {X_points.shape}, Y_points {Y_points.shape}")
    
    product_names = ALL_DATA.get_products()
    print(f"产品列表: {product_names}")
    print(f"数据加载耗时: {time.time() - start_time:.2f}秒")
    
    n_products, n_days, n_lat, n_lon = X_spatial.shape
    _, _, n_points = X_points.shape
    
    # 处理时间依赖性
    valid_time_slice = slice(MAX_LOOKBACK, n_days)
    n_valid_days = n_days - MAX_LOOKBACK
    
    print(f"\n有效时间范围: {MAX_LOOKBACK} 到 {n_days-1} (共{n_valid_days}天)")
    
    # ===========================
    # 2. 基础原始特征
    # ===========================
    print("\n2. 生成基础原始特征...")
    
    # 2.1 空间数据的原始值
    for i, product in enumerate(product_names):
        # 完整时间序列的空间数据
        save_feature(X_spatial[i], f"raw_spatial_{product}", 
                    f"{product}产品完整空间数据 (time, lat, lon)")
        
        # 有效时间范围的空间数据
        save_feature(X_spatial[i, valid_time_slice], f"raw_spatial_{product}_valid", 
                    f"{product}产品有效时间空间数据 (valid_time, lat, lon)")
    
    # 2.2 点数据的原始值
    for i, product in enumerate(product_names):
        # 完整时间序列的点数据
        save_feature(X_points[i], f"raw_points_{product}", 
                    f"{product}产品完整点数据 (time, points)")
        
        # 有效时间范围的点数据
        save_feature(X_points[i, valid_time_slice], f"raw_points_{product}_valid", 
                    f"{product}产品有效时间点数据 (valid_time, points)")
    
    # 2.3 目标变量
    save_feature(Y_spatial, "target_spatial", "CHM目标变量空间数据 (time, lat, lon)")
    save_feature(Y_spatial[valid_time_slice], "target_spatial_valid", "CHM目标变量有效时间空间数据")
    save_feature(Y_points, "target_points", "CHM目标变量点数据 (time, points)")
    save_feature(Y_points[valid_time_slice], "target_points_valid", "CHM目标变量有效时间点数据")
    
    # ===========================
    # 3. 多产品协同特征（基于点数据）
    # ===========================
    print("\n3. 生成多产品协同特征...")
    
    # 转换为 (time, points, products) 便于计算
    X_points_reorder = np.transpose(X_points, (1, 2, 0))  # (time, points, products)
    X_points_valid = X_points_reorder[valid_time_slice]
    
    # 3.1 当前时刻多产品统计
    save_feature(np.nanmean(X_points_valid, axis=2), "multi_product_mean", 
                "多产品均值 (valid_time, points)")
    save_feature(np.nanstd(X_points_valid, axis=2), "multi_product_std", 
                "多产品标准差 (valid_time, points)")
    save_feature(np.nanmedian(X_points_valid, axis=2), "multi_product_median", 
                "多产品中位数 (valid_time, points)")
    save_feature(np.nanmax(X_points_valid, axis=2), "multi_product_max", 
                "多产品最大值 (valid_time, points)")
    save_feature(np.nanmin(X_points_valid, axis=2), "multi_product_min", 
                "多产品最小值 (valid_time, points)")
    save_feature(np.nanmax(X_points_valid, axis=2) - np.nanmin(X_points_valid, axis=2), 
                "multi_product_range", "多产品极差 (valid_time, points)")
    
    # 3.2 降雨产品计数
    rain_mask = X_points_valid > RAIN_THR
    save_feature(np.sum(rain_mask, axis=2).astype(np.float32), "rain_product_count", 
                "指示降雨的产品数量 (valid_time, points)")
    
    # 3.3 变异系数
    mean_vals = np.nanmean(X_points_valid, axis=2)
    std_vals = np.nanstd(X_points_valid, axis=2)
    cv = safe_divide(std_vals, mean_vals)
    save_feature(cv, "multi_product_cv", "多产品变异系数 (valid_time, points)")
    
    # 3.4 产品间相关性特征
    for i, prod1 in enumerate(product_names):
        for j, prod2 in enumerate(product_names):
            if i < j:  # 只计算上三角，避免重复
                corr_map = np.zeros((n_valid_days, n_points))
                for t in range(n_valid_days):
                    for p in range(n_points):
                        if t >= 30:  # 需要足够的历史数据计算相关性
                            x1 = X_points_valid[max(0, t-30):t+1, p, i]
                            x2 = X_points_valid[max(0, t-30):t+1, p, j]
                            valid_mask = ~(np.isnan(x1) | np.isnan(x2))
                            if np.sum(valid_mask) > 10:
                                corr_map[t, p] = np.corrcoef(x1[valid_mask], x2[valid_mask])[0, 1]
                            else:
                                corr_map[t, p] = 0.0
                        else:
                            corr_map[t, p] = 0.0
                
                save_feature(corr_map, f"correlation_{prod1}_{prod2}", 
                           f"{prod1}与{prod2}滚动相关性 (valid_time, points)")
    
    # ===========================
    # 4. 时序动态特征
    # ===========================
    print("\n4. 生成时序动态特征...")
    
    # 4.1 周期性特征
    days_in_year = 365.25
    day_index = np.arange(n_days, dtype=np.float32)
    day_of_year = day_index % days_in_year
    
    # 年内日周期
    sin_day = np.sin(2 * np.pi * day_of_year / days_in_year)
    cos_day = np.cos(2 * np.pi * day_of_year / days_in_year)
    save_feature(sin_day[valid_time_slice], "sin_day_of_year", "年内日周期正弦 (valid_time,)")
    save_feature(cos_day[valid_time_slice], "cos_day_of_year", "年内日周期余弦 (valid_time,)")
    
    # 季节特征
    month = (day_of_year // 30.4375).astype(int) % 12 + 1
    season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    season = np.array([season_map[m] for m in month])
    
    # 季节独热编码
    for s in range(4):
        season_onehot = (season == s).astype(np.float32)
        save_feature(season_onehot[valid_time_slice], f"season_onehot_{s}", 
                    f"季节{s}独热编码 (valid_time,)")
    
    # 4.2 滞后特征
    print("  生成滞后特征...")
    for lag in [1, 2, 3, 7, 15]:
        if lag <= MAX_LOOKBACK:
            for i, product in enumerate(product_names):
                # 点数据滞后
                lag_data = X_points[i, MAX_LOOKBACK-lag:n_days-lag]
                save_feature(lag_data, f"lag_{lag}_points_{product}", 
                           f"{product}滞后{lag}天点数据 (valid_time, points)")
                
                # 空间数据滞后（选择性保存，避免文件过大）
                if lag <= 3:  # 只保存短期滞后的空间数据
                    lag_spatial = X_spatial[i, MAX_LOOKBACK-lag:n_days-lag]
                    save_feature(lag_spatial, f"lag_{lag}_spatial_{product}", 
                               f"{product}滞后{lag}天空间数据 (valid_time, lat, lon)")
    
    # 4.3 多产品统计量的滞后
    multi_mean = np.nanmean(X_points_reorder, axis=2)  # (time, points)
    multi_std = np.nanstd(X_points_reorder, axis=2)
    
    for lag in [1, 2, 3]:
        if lag <= MAX_LOOKBACK:
            save_feature(multi_mean[MAX_LOOKBACK-lag:n_days-lag], f"lag_{lag}_multi_product_mean", 
                        f"多产品均值滞后{lag}天 (valid_time, points)")
            save_feature(multi_std[MAX_LOOKBACK-lag:n_days-lag], f"lag_{lag}_multi_product_std", 
                        f"多产品标准差滞后{lag}天 (valid_time, points)")
    
    # 4.4 差分特征
    print("  生成差分特征...")
    for i, product in enumerate(product_names):
        # 一阶差分
        diff_1 = X_points_valid[1:] - X_points_valid[:-1]
        save_feature(np.concatenate([np.zeros((1, n_points)), diff_1], axis=0), 
                    f"diff_1_points_{product}", f"{product}一阶差分 (valid_time, points)")
        
        # 二阶差分
        if n_valid_days > 2:
            diff_2 = diff_1[1:] - diff_1[:-1]
            save_feature(np.concatenate([np.zeros((2, n_points)), diff_2], axis=0), 
                        f"diff_2_points_{product}", f"{product}二阶差分 (valid_time, points)")
    
    # 多产品统计量差分
    diff_mean = multi_mean[valid_time_slice][1:] - multi_mean[valid_time_slice][:-1]
    save_feature(np.concatenate([np.zeros((1, n_points)), diff_mean], axis=0), 
                "diff_1_multi_product_mean", "多产品均值一阶差分 (valid_time, points)")
    
    diff_std = multi_std[valid_time_slice][1:] - multi_std[valid_time_slice][:-1]
    save_feature(np.concatenate([np.zeros((1, n_points)), diff_std], axis=0), 
                "diff_1_multi_product_std", "多产品标准差一阶差分 (valid_time, points)")
    
    # 4.5 滑动窗口统计特征
    print("  生成滑动窗口特征...")
    for window in [3, 7, 15, 30]:
        if window <= MAX_LOOKBACK:
            for i, product in enumerate(product_names):
                # 计算滑动统计量
                rolling_mean = np.zeros((n_valid_days, n_points))
                rolling_std = np.zeros((n_valid_days, n_points))
                rolling_max = np.zeros((n_valid_days, n_points))
                rolling_min = np.zeros((n_valid_days, n_points))
                rolling_sum = np.zeros((n_valid_days, n_points))
                
                for t in range(n_valid_days):
                    start_idx = max(0, MAX_LOOKBACK + t - window)
                    end_idx = MAX_LOOKBACK + t + 1
                    window_data = X_points[i, start_idx:end_idx]
                    
                    rolling_mean[t] = np.nanmean(window_data, axis=0)
                    rolling_std[t] = np.nanstd(window_data, axis=0)
                    rolling_max[t] = np.nanmax(window_data, axis=0)
                    rolling_min[t] = np.nanmin(window_data, axis=0)
                    rolling_sum[t] = np.nansum(window_data, axis=0)
                
                save_feature(rolling_mean, f"rolling_{window}d_mean_{product}", 
                           f"{product} {window}天滑动均值 (valid_time, points)")
                save_feature(rolling_std, f"rolling_{window}d_std_{product}", 
                           f"{product} {window}天滑动标准差 (valid_time, points)")
                save_feature(rolling_max, f"rolling_{window}d_max_{product}", 
                           f"{product} {window}天滑动最大值 (valid_time, points)")
                save_feature(rolling_min, f"rolling_{window}d_min_{product}", 
                           f"{product} {window}天滑动最小值 (valid_time, points)")
                save_feature(rolling_sum, f"rolling_{window}d_sum_{product}", 
                           f"{product} {window}天滑动累计 (valid_time, points)")
    
    # ===========================
    # 5. 真实空间特征
    # ===========================
    print("\n5. 生成真实空间特征...")
    
    # 5.1 空间梯度特征
    print("  计算空间梯度特征...")
    for i, product in enumerate(product_names):
        grad_x_all = np.zeros((n_valid_days, n_lat, n_lon))
        grad_y_all = np.zeros((n_valid_days, n_lat, n_lon))
        grad_mag_all = np.zeros((n_valid_days, n_lat, n_lon))
        grad_dir_all = np.zeros((n_valid_days, n_lat, n_lon))
        
        for t in range(n_valid_days):
            actual_t = MAX_LOOKBACK + t
            data_2d = X_spatial[i, actual_t]
            
            # 只对非全NaN的数据计算梯度
            if not np.all(np.isnan(data_2d)):
                grad_x, grad_y, grad_mag, grad_dir = calculate_spatial_gradients(
                    np.nan_to_num(data_2d, nan=0.0)
                )
                
                grad_x_all[t] = grad_x
                grad_y_all[t] = grad_y
                grad_mag_all[t] = grad_mag
                grad_dir_all[t] = grad_dir
        
        save_feature(grad_x_all, f"gradient_x_{product}", 
                    f"{product}经度方向梯度 (valid_time, lat, lon)")
        save_feature(grad_y_all, f"gradient_y_{product}", 
                    f"{product}纬度方向梯度 (valid_time, lat, lon)")
        save_feature(grad_mag_all, f"gradient_magnitude_{product}", 
                    f"{product}梯度幅度 (valid_time, lat, lon)")
        save_feature(grad_dir_all, f"gradient_direction_{product}", 
                    f"{product}梯度方向 (valid_time, lat, lon)")
    
    # 5.2 邻域统计特征
    print("  计算邻域统计特征...")
    for window_size in [3, 5, 7]:
        for i, product in enumerate(product_names):
            neighbor_mean_all = np.zeros((n_valid_days, n_lat, n_lon))
            neighbor_std_all = np.zeros((n_valid_days, n_lat, n_lon))
            neighbor_max_all = np.zeros((n_valid_days, n_lat, n_lon))
            neighbor_min_all = np.zeros((n_valid_days, n_lat, n_lon))
            
            for t in range(n_valid_days):
                actual_t = MAX_LOOKBACK + t
                data_2d = X_spatial[i, actual_t]
                
                if not np.all(np.isnan(data_2d)):
                    n_mean, n_std, n_max, n_min = calculate_neighborhood_stats(data_2d, window_size)
                    
                    neighbor_mean_all[t] = n_mean
                    neighbor_std_all[t] = n_std
                    neighbor_max_all[t] = n_max
                    neighbor_min_all[t] = n_min
            
            save_feature(neighbor_mean_all, f"neighbor_{window_size}x{window_size}_mean_{product}", 
                        f"{product} {window_size}x{window_size}邻域均值 (valid_time, lat, lon)")
            save_feature(neighbor_std_all, f"neighbor_{window_size}x{window_size}_std_{product}", 
                        f"{product} {window_size}x{window_size}邻域标准差 (valid_time, lat, lon)")
            save_feature(neighbor_max_all, f"neighbor_{window_size}x{window_size}_max_{product}", 
                        f"{product} {window_size}x{window_size}邻域最大值 (valid_time, lat, lon)")
            save_feature(neighbor_min_all, f"neighbor_{window_size}x{window_size}_min_{product}", 
                        f"{product} {window_size}x{window_size}邻域最小值 (valid_time, lat, lon)")
    
    # 5.3 空间聚集性特征
    print("  计算空间聚集性特征...")
    for i, product in enumerate(product_names):
        # 空间方差（每个时刻的空间变异性）
        spatial_var = np.nanvar(X_spatial[i, valid_time_slice], axis=(1, 2))
        save_feature(spatial_var, f"spatial_variance_{product}", 
                    f"{product}空间方差 (valid_time,)")
        
        # 空间偏度和峰度
        spatial_skew = np.zeros(n_valid_days)
        spatial_kurt = np.zeros(n_valid_days)
        
        for t in range(n_valid_days):
            actual_t = MAX_LOOKBACK + t
            flat_data = X_spatial[i, actual_t].flatten()
            valid_data = flat_data[~np.isnan(flat_data)]
            if len(valid_data) > 3:
                spatial_skew[t] = skew(valid_data)
                spatial_kurt[t] = kurtosis(valid_data)
        
        save_feature(spatial_skew, f"spatial_skewness_{product}", 
                    f"{product}空间偏度 (valid_time,)")
        save_feature(spatial_kurt, f"spatial_kurtosis_{product}", 
                    f"{product}空间峰度 (valid_time,)")
    
    # ===========================
    # 6. 弱信号增强特征
    # ===========================
    print("\n6. 生成弱信号增强特征...")
    
    # 6.1 阈值距离特征
    for threshold in [0.05, 0.1, 0.2, 0.5]:
        for i, product in enumerate(product_names):
            distance_to_threshold = np.abs(X_points_valid[..., i] - threshold)
            save_feature(distance_to_threshold, f"distance_to_threshold_{threshold}_{product}", 
                        f"{product}距离阈值{threshold}的距离 (valid_time, points)")
    
    # 6.2 低强度条件特征
    low_intensity_mask = np.nanmean(X_points_valid, axis=2) < 0.5
    
    # 低强度下的产品间标准差
    conditional_std = np.nanstd(X_points_valid, axis=2)
    conditional_std[~low_intensity_mask] = 0.0
    save_feature(conditional_std, "low_intensity_conditional_std", 
                "低强度条件下的产品间标准差 (valid_time, points)")
    
    # 低强度下的变异系数
    conditional_cv = cv.copy()
    conditional_cv[~low_intensity_mask] = 0.0
    save_feature(conditional_cv, "low_intensity_conditional_cv", 
                "低强度条件下的变异系数 (valid_time, points)")
    
    # 6.3 降雨强度分箱特征
    mean_rainfall = np.nanmean(X_points_valid, axis=2)
    
    # 基于均值的强度分箱
    intensity_bins = np.digitize(mean_rainfall, bins=[0.1, 0.5, 1.0, 5.0, 10.0])
    for bin_idx in range(6):  # 0-5 bins
        bin_onehot = (intensity_bins == bin_idx).astype(np.float32)
        save_feature(bin_onehot, f"intensity_bin_{bin_idx}_mean", 
                    f"基于均值的强度分箱{bin_idx} (valid_time, points)")
    
    # 基于降雨产品数量的分箱
    rain_count = np.sum(X_points_valid > RAIN_THR, axis=2)
    count_bins = np.digitize(rain_count, bins=[1, 2, 3, 4, 5])
    for bin_idx in range(6):
        bin_onehot = (count_bins == bin_idx).astype(np.float32)
        save_feature(bin_onehot, f"intensity_bin_{bin_idx}_count", 
                    f"基于产品数量的强度分箱{bin_idx} (valid_time, points)")
    
    # ===========================
    # 7. 高阶交互特征
    # ===========================
    print("\n7. 生成高阶交互特征...")
    
    # 7.1 产品间乘性交互
    for i, prod1 in enumerate(product_names):
        for j, prod2 in enumerate(product_names):
            if i < j:
                interaction = X_points_valid[..., i] * X_points_valid[..., j]
                save_feature(interaction, f"interaction_multiply_{prod1}_{prod2}", 
                           f"{prod1}与{prod2}乘性交互 (valid_time, points)")
    
    # 7.2 统计量与周期性交互
    sin_day_expanded = sin_day[valid_time_slice][:, np.newaxis]
    cos_day_expanded = cos_day[valid_time_slice][:, np.newaxis]
    
    std_sin_interaction = std_vals * sin_day_expanded
    std_cos_interaction = std_vals * cos_day_expanded
    save_feature(std_sin_interaction, "interaction_std_sin_day", 
                "产品标准差与年内日周期正弦交互 (valid_time, points)")
    save_feature(std_cos_interaction, "interaction_std_cos_day", 
                "产品标准差与年内日周期余弦交互 (valid_time, points)")
    
    # 7.3 条件统计量交互
    low_intensity_std_cv_interaction = conditional_std * conditional_cv
    save_feature(low_intensity_std_cv_interaction, "interaction_low_intensity_std_cv", 
                "低强度标准差与变异系数交互 (valid_time, points)")
    
    # 7.4 降雨数量与统计量交互
    rain_count_std_interaction = rain_count.astype(np.float32) * std_vals
    save_feature(rain_count_std_interaction, "interaction_rain_count_std", 
                "降雨产品数量与标准差交互 (valid_time, points)")
    
    # ===========================
    # 8. 高级统计特征
    # ===========================
    print("\n8. 生成高级统计特征...")
    
    # 8.1 分位数特征
    for q in [0.25, 0.75, 0.9, 0.95]:
        quantile_vals = np.nanquantile(X_points_valid, q, axis=2)
        save_feature(quantile_vals, f"multi_product_quantile_{int(q*100)}", 
                    f"多产品{int(q*100)}%分位数 (valid_time, points)")
    
    # 8.2 极值比例特征
    for threshold in [0.1, 1.0, 5.0]:
        extreme_ratio = np.mean(X_points_valid > threshold, axis=2)
        save_feature(extreme_ratio, f"extreme_ratio_above_{threshold}", 
                    f"超过{threshold}mm的产品比例 (valid_time, points)")
    
    # 8.3 变化幅度特征
    daily_range = np.nanmax(X_points_valid, axis=2) - np.nanmin(X_points_valid, axis=2)
    save_feature(daily_range, "daily_product_range", 
                "日产品间极差 (valid_time, points)")
    
    # 8.4 一致性指标
    # 产品间最大最小比值
    max_vals = np.nanmax(X_points_valid, axis=2)
    min_vals = np.nanmin(X_points_valid, axis=2)
    consistency_ratio = safe_divide(min_vals, max_vals, default=1.0)
    save_feature(consistency_ratio, "product_consistency_ratio", 
                "产品一致性比值(最小/最大) (valid_time, points)")
    
    # ===========================
    # 9. 目标相关特征
    # ===========================
    print("\n9. 生成目标相关特征...")
    
    # 9.1 与历史目标的相关性
    Y_points_valid = Y_points[valid_time_slice]
    
    for lag in [1, 3, 7]:
        if lag <= MAX_LOOKBACK:
            Y_lag = Y_points[MAX_LOOKBACK-lag:n_days-lag]
            save_feature(Y_lag, f"target_lag_{lag}", 
                        f"目标变量滞后{lag}天 (valid_time, points)")
    
    # 9.2 目标变量统计特征
    for window in [3, 7, 15]:
        if window <= MAX_LOOKBACK:
            target_rolling_mean = np.zeros((n_valid_days, n_points))
            target_rolling_std = np.zeros((n_valid_days, n_points))
            
            for t in range(n_valid_days):
                start_idx = max(0, MAX_LOOKBACK + t - window)
                end_idx = MAX_LOOKBACK + t + 1
                window_target = Y_points[start_idx:end_idx]
                
                target_rolling_mean[t] = np.nanmean(window_target, axis=0)
                target_rolling_std[t] = np.nanstd(window_target, axis=0)
            
            save_feature(target_rolling_mean, f"target_rolling_{window}d_mean", 
                        f"目标变量{window}天滑动均值 (valid_time, points)")
            save_feature(target_rolling_std, f"target_rolling_{window}d_std", 
                        f"目标变量{window}天滑动标准差 (valid_time, points)")
    
    # ===========================
    # 10. 元信息特征
    # ===========================
    print("\n10. 生成元信息特征...")
    
    # 10.1 数据质量特征
    nan_count = np.sum(np.isnan(X_points_valid), axis=2)
    save_feature(nan_count, "data_quality_nan_count", 
                "每天每点的NaN数量 (valid_time, points)")
    
    valid_count = np.sum(~np.isnan(X_points_valid), axis=2)
    save_feature(valid_count, "data_quality_valid_count", 
                "每天每点的有效数据数量 (valid_time, points)")
    
    # 10.2 时间索引特征
    time_index = np.arange(n_valid_days, dtype=np.float32)
    save_feature(time_index, "time_index", "时间索引 (valid_time,)")
    
    # 10.3 归一化时间特征
    normalized_time = time_index / (n_valid_days - 1)
    save_feature(normalized_time, "normalized_time", "归一化时间 (valid_time,)")
    
    print(f"\n=== 特征生成完成 ===")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 生成特征清单
    feature_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
    feature_files.sort()
    
    print(f"\n生成的特征文件数量: {len(feature_files)}")
    print("特征文件列表已保存到: feature_list.txt")
    
    with open(os.path.join(OUTPUT_DIR, "feature_list.txt"), 'w', encoding='utf-8') as f:
        f.write("# 生成的特征文件列表\n")
        f.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 总文件数: {len(feature_files)}\n\n")
        for feature_file in feature_files:
            f.write(f"{feature_file}\n")

if __name__ == "__main__":
    main()