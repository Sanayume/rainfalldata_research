#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级统计特征生成器
生成分位数、极值、弱信号增强等高级统计特征
"""

import numpy as np
import os
import time
import warnings
from loaddata import mydata

# 抑制warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 配置
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
MAX_LOOKBACK = 30
EPSILON = 1e-6
RAIN_THR = 0.1

def safe_divide(numerator, denominator, default=0.0):
    """安全除法，避免除零"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / (denominator + EPSILON)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

def safe_nanmean(arr, axis=None, default=0.0):
    """安全的nanmean，处理全NaN情况"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.nanmean(arr, axis=axis)
        if np.isscalar(result):
            return default if np.isnan(result) else result
        else:
            return np.nan_to_num(result, nan=default)

def safe_nanstd(arr, axis=None, default=0.0):
    """安全的nanstd，处理全NaN情况"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.nanstd(arr, axis=axis)
        if np.isscalar(result):
            return default if np.isnan(result) else result
        else:
            return np.nan_to_num(result, nan=default)

def save_feature(feature_data, feature_name, description=""):
    """保存特征到npy文件"""
    filepath = os.path.join(OUTPUT_DIR, f"{feature_name}.npy")
    np.save(filepath, feature_data.astype(np.float32))
    print(f"  Saved: {feature_name}.npy {feature_data.shape} - {description}")

def main():
    print("=== 生成高级统计特征 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 加载数据
    print("\n1. 加载长江流域数据...")
    start_time = time.time()
    ALL_DATA = mydata()
    
    # 加载点数据
    X_points, Y_points = ALL_DATA.get_basin_point_data(basin_mask_value=2)
    product_names = ALL_DATA.get_products()
    
    print(f"点数据形状: X_points {X_points.shape}, Y_points {Y_points.shape}")
    print(f"数据加载耗时: {time.time() - start_time:.2f}秒")
    
    n_products, n_days, n_points = X_points.shape
    
    # 处理时间依赖性
    valid_time_slice = slice(MAX_LOOKBACK, n_days)
    n_valid_days = n_days - MAX_LOOKBACK
    
    # 转换为 (time, points, products) 便于计算
    X_points_reorder = np.transpose(X_points, (1, 2, 0))
    X_points_valid = X_points_reorder[valid_time_slice]
    Y_points_valid = Y_points[valid_time_slice]
    
    print(f"有效数据形状: {X_points_valid.shape}")
    
    # 生成高级统计特征
    print("\n2. 生成高级统计特征...")
    
    # 2.1 分位数特征
    print("  生成分位数特征...")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    for q in quantiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantile_vals = np.nanquantile(X_points_valid, q, axis=2)
            quantile_vals = np.nan_to_num(quantile_vals, nan=0.0)
            save_feature(quantile_vals, f"multi_product_quantile_{int(q*100)}", 
                        f"多产品{int(q*100)}%分位数 (valid_time, points)")
    
    # 各产品的分位数
    for i, product in enumerate(product_names):
        for q in [0.25, 0.75, 0.95]:
            product_data = X_points_valid[:, :, i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 计算时间维度上的分位数（每个点的历史分位数）
                rolling_quantile = np.zeros_like(product_data)
                window = 30  # 30天窗口
                
                for t in range(len(product_data)):
                    start_idx = max(0, t - window)
                    for p in range(min(100, n_points)):  # 限制点数
                        window_data = product_data[start_idx:t+1, p]
                        valid_data = window_data[~np.isnan(window_data)]
                        if len(valid_data) > 5:
                            rolling_quantile[t, p] = np.quantile(valid_data, q)
                
                save_feature(rolling_quantile, f"rolling_quantile_{int(q*100)}_{product}", 
                           f"{product}滚动{int(q*100)}%分位数 (valid_time, points)")
    
    # 2.2 极值特征
    print("  生成极值特征...")
    
    # 极值比例特征
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for threshold in thresholds:
        extreme_ratio = np.mean(X_points_valid > threshold, axis=2)
        save_feature(extreme_ratio, f"extreme_ratio_above_{threshold}", 
                    f"超过{threshold}mm的产品比例 (valid_time, points)")
        
        extreme_count = np.sum(X_points_valid > threshold, axis=2)
        save_feature(extreme_count, f"extreme_count_above_{threshold}", 
                    f"超过{threshold}mm的产品数量 (valid_time, points)")
    
    # 极值持续性
    for threshold in [1.0, 5.0]:
        extreme_mask = X_points_valid > threshold
        extreme_persistence = np.zeros((n_valid_days, n_points))
        
        for t in range(1, n_valid_days):
            # 当前时刻有极值且前一时刻也有极值的产品比例
            current_extreme = extreme_mask[t]
            prev_extreme = extreme_mask[t-1]
            persistence = np.logical_and(current_extreme, prev_extreme)
            extreme_persistence[t] = np.mean(persistence, axis=1)
        
        save_feature(extreme_persistence, f"extreme_persistence_{threshold}", 
                    f"极值{threshold}mm持续性 (valid_time, points)")
    
    # 2.3 变化幅度特征
    print("  生成变化幅度特征...")
    
    # 日变化幅度
    daily_range = np.nanmax(X_points_valid, axis=2) - np.nanmin(X_points_valid, axis=2)
    save_feature(daily_range, "daily_product_range", 
                "日产品间极差 (valid_time, points)")
    
    # 相对变化幅度
    daily_mean = safe_nanmean(X_points_valid, axis=2)
    relative_range = safe_divide(daily_range, daily_mean)
    save_feature(relative_range, "relative_daily_range", 
                "日相对变化幅度 (valid_time, points)")
    
    # 变异系数
    daily_std = safe_nanstd(X_points_valid, axis=2)
    cv = safe_divide(daily_std, daily_mean)
    save_feature(cv, "coefficient_of_variation", "变异系数 (valid_time, points)")
    
    # 2.4 异常值检测特征
    print("  生成异常值检测特征...")
    
    # Z-score异常值
    for i, product in enumerate(product_names):
        product_data = X_points_valid[:, :, i]
        
        # 计算滚动均值和标准差
        rolling_mean = np.zeros_like(product_data)
        rolling_std = np.zeros_like(product_data)
        window = 30
        
        for t in range(len(product_data)):
            start_idx = max(0, t - window)
            rolling_mean[t] = safe_nanmean(product_data[start_idx:t+1], axis=0)
            rolling_std[t] = safe_nanstd(product_data[start_idx:t+1], axis=0)
        
        # 计算Z-score
        z_score = safe_divide(product_data - rolling_mean, rolling_std)
        
        # 异常值标识（|Z-score| > 2）
        anomaly_flag = (np.abs(z_score) > 2).astype(np.float32)
        save_feature(anomaly_flag, f"anomaly_zscore_{product}", 
                    f"{product} Z-score异常值标识 (valid_time, points)")
    
    # 2.5 弱信号增强特征
    print("  生成弱信号增强特征...")
    
    # 阈值距离特征
    distance_thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    for threshold in distance_thresholds:
        # 计算到阈值的最小距离
        min_distance = np.min(np.abs(X_points_valid - threshold), axis=2)
        save_feature(min_distance, f"min_distance_to_threshold_{threshold}", 
                    f"到阈值{threshold}的最小距离 (valid_time, points)")
        
        # 在阈值附近的产品数量
        tolerance = threshold * 0.1  # 10%容差
        near_threshold = np.sum(np.abs(X_points_valid - threshold) <= tolerance, axis=2)
        save_feature(near_threshold, f"near_threshold_count_{threshold}", 
                    f"阈值{threshold}附近的产品数量 (valid_time, points)")
    
    # 低强度条件特征
    low_intensity_threshold = 0.5
    low_intensity_mask = daily_mean < low_intensity_threshold
    
    # 低强度下的产品间标准差
    conditional_std = daily_std.copy()
    conditional_std[~low_intensity_mask] = 0.0
    save_feature(conditional_std, "low_intensity_conditional_std", 
                "低强度条件下的产品间标准差 (valid_time, points)")
    
    # 低强度下的变异系数
    conditional_cv = cv.copy()
    conditional_cv[~low_intensity_mask] = 0.0
    save_feature(conditional_cv, "low_intensity_conditional_cv", 
                "低强度条件下的变异系数 (valid_time, points)")
    
    # 2.6 强度分箱特征
    print("  生成强度分箱特征...")
    
    # 基于均值的强度分箱
    intensity_bins_mean = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]
    intensity_categories = np.digitize(daily_mean, bins=intensity_bins_mean[:-1])
    
    for bin_idx in range(len(intensity_bins_mean) - 1):
        bin_onehot = (intensity_categories == bin_idx).astype(np.float32)
        save_feature(bin_onehot, f"intensity_bin_{bin_idx}_mean", 
                    f"基于均值的强度分箱{bin_idx} (valid_time, points)")
    
    # 基于最大值的强度分箱
    daily_max = np.nanmax(X_points_valid, axis=2)
    intensity_categories_max = np.digitize(daily_max, bins=intensity_bins_mean[:-1])
    
    for bin_idx in range(len(intensity_bins_mean) - 1):
        bin_onehot = (intensity_categories_max == bin_idx).astype(np.float32)
        save_feature(bin_onehot, f"intensity_bin_{bin_idx}_max", 
                    f"基于最大值的强度分箱{bin_idx} (valid_time, points)")
    
    # 基于降雨产品数量的分箱
    rain_count = np.sum(X_points_valid > RAIN_THR, axis=2)
    count_bins = [0, 1, 2, 3, 4, 5, 6]
    count_categories = np.digitize(rain_count, bins=count_bins[:-1])
    
    for bin_idx in range(len(count_bins) - 1):
        bin_onehot = (count_categories == bin_idx).astype(np.float32)
        save_feature(bin_onehot, f"rain_count_bin_{bin_idx}", 
                    f"降雨产品数量分箱{bin_idx} (valid_time, points)")
    
    # 2.7 一致性和不确定性特征
    print("  生成一致性和不确定性特征...")
    
    # 产品间一致性指标
    # 最大最小比值
    daily_max = np.nanmax(X_points_valid, axis=2)
    daily_min = np.nanmin(X_points_valid, axis=2)
    consistency_ratio = safe_divide(daily_min, daily_max, default=1.0)
    save_feature(consistency_ratio, "product_consistency_ratio", 
                "产品一致性比值(最小/最大) (valid_time, points)")
    
    # 产品间分歧度
    disagreement = safe_divide(daily_std, daily_mean + 0.1)
    save_feature(disagreement, "product_disagreement", 
                "产品分歧度 (valid_time, points)")
    
    # 不确定性指标（基于熵）
    # 简化的熵计算
    entropy = np.zeros((n_valid_days, n_points))
    for t in range(n_valid_days):
        for p in range(min(100, n_points)):  # 限制点数
            values = X_points_valid[t, p, :]
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 1:
                # 简单的离散化熵
                hist, _ = np.histogram(valid_values, bins=5, density=True)
                hist = hist[hist > 0]  # 去除零值
                entropy[t, p] = -np.sum(hist * np.log(hist + EPSILON))
    
    save_feature(entropy, "product_entropy", "产品不确定性熵 (valid_time, points)")
    
    # 2.8 目标相关的高级特征
    print("  生成目标相关的高级特征...")
    
    # 目标变量的分位数特征
    for q in [0.1, 0.25, 0.75, 0.9]:
        target_rolling_quantile = np.zeros_like(Y_points_valid)
        window = 30
        
        for t in range(len(Y_points_valid)):
            start_idx = max(0, t - window)
            for p in range(min(100, n_points)):  # 限制点数
                window_data = Y_points_valid[start_idx:t+1, p]
                valid_data = window_data[~np.isnan(window_data)]
                if len(valid_data) > 5:
                    target_rolling_quantile[t, p] = np.quantile(valid_data, q)
        
        save_feature(target_rolling_quantile, f"target_rolling_quantile_{int(q*100)}", 
                    f"目标变量滚动{int(q*100)}%分位数 (valid_time, points)")
    
    # 目标变量异常值
    target_rolling_mean = np.zeros_like(Y_points_valid)
    target_rolling_std = np.zeros_like(Y_points_valid)
    window = 30
    
    for t in range(len(Y_points_valid)):
        start_idx = max(0, t - window)
        target_rolling_mean[t] = safe_nanmean(Y_points_valid[start_idx:t+1], axis=0)
        target_rolling_std[t] = safe_nanstd(Y_points_valid[start_idx:t+1], axis=0)
    
    target_z_score = safe_divide(Y_points_valid - target_rolling_mean, target_rolling_std)
    target_anomaly = (np.abs(target_z_score) > 2).astype(np.float32)
    save_feature(target_anomaly, "target_anomaly_zscore", 
                "目标变量Z-score异常值标识 (valid_time, points)")
    
    print(f"\n=== 高级统计特征生成完成 ===")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    
    # 统计生成的特征
    advanced_features = [f for f in os.listdir(OUTPUT_DIR) if any(f.startswith(prefix) for prefix in [
        'multi_product_quantile_', 'rolling_quantile_', 'extreme_', 'daily_', 'relative_', 
        'coefficient_', 'anomaly_', 'min_distance_', 'near_threshold_', 'low_intensity_', 
        'intensity_bin_', 'rain_count_bin_', 'product_consistency_', 'product_disagreement', 
        'product_entropy', 'target_rolling_', 'target_anomaly_'
    ]) and f.endswith('.npy')]
    print(f"生成的高级统计特征数量: {len(advanced_features)}")

if __name__ == "__main__":
    main()