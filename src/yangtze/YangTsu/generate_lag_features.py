#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滞后特征生成器
生成各种滞后特征，捕捉时间依赖性
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
    print("=== 生成滞后特征 ===")
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
    
    print(f"有效时间范围: {MAX_LOOKBACK} 到 {n_days-1} (共{n_valid_days}天)")
    
    # 转换为 (time, points, products) 便于计算
    X_points_reorder = np.transpose(X_points, (1, 2, 0))
    
    # 生成滞后特征
    print("\n2. 生成滞后特征...")
    
    # 2.1 单个产品的滞后特征
    print("  生成单产品滞后特征...")
    lag_days = [1, 2, 3, 5, 7, 10, 15, 21, 30]
    
    for lag in lag_days:
        if lag <= MAX_LOOKBACK:
            print(f"    处理滞后{lag}天...")
            for i, product in enumerate(product_names):
                # 点数据滞后
                lag_data = X_points[i, MAX_LOOKBACK-lag:n_days-lag]
                save_feature(lag_data, f"lag_{lag}_points_{product}", 
                           f"{product}滞后{lag}天点数据 (valid_time, points)")
    
    # 2.2 多产品统计量的滞后
    print("  生成多产品统计量滞后特征...")
    
    # 先计算多产品统计量
    multi_mean = safe_nanmean(X_points_reorder, axis=2)  # (time, points)
    multi_std = safe_nanstd(X_points_reorder, axis=2)
    multi_max = np.nanmax(X_points_reorder, axis=2)
    multi_min = np.nanmin(X_points_reorder, axis=2)
    
    # 降雨产品计数
    rain_count = np.sum(X_points_reorder > 0.1, axis=2).astype(np.float32)
    
    multi_stats = {
        'mean': multi_mean,
        'std': multi_std,
        'max': multi_max,
        'min': multi_min,
        'rain_count': rain_count
    }
    
    for lag in [1, 2, 3, 5, 7, 15]:
        if lag <= MAX_LOOKBACK:
            print(f"    处理多产品统计量滞后{lag}天...")
            for stat_name, stat_data in multi_stats.items():
                lag_stat = stat_data[MAX_LOOKBACK-lag:n_days-lag]
                save_feature(lag_stat, f"lag_{lag}_multi_product_{stat_name}", 
                           f"多产品{stat_name}滞后{lag}天 (valid_time, points)")
    
    # 2.3 目标变量的滞后
    print("  生成目标变量滞后特征...")
    for lag in [1, 2, 3, 7, 15]:
        if lag <= MAX_LOOKBACK:
            Y_lag = Y_points[MAX_LOOKBACK-lag:n_days-lag]
            save_feature(Y_lag, f"lag_{lag}_target", 
                        f"目标变量滞后{lag}天 (valid_time, points)")
    
    # 2.4 滞后差值特征
    print("  生成滞后差值特征...")
    
    # 计算不同滞后期之间的差值
    lag_pairs = [(1, 2), (1, 3), (2, 3), (3, 7), (7, 15)]
    
    for lag1, lag2 in lag_pairs:
        if lag1 <= MAX_LOOKBACK and lag2 <= MAX_LOOKBACK:
            print(f"    计算滞后{lag1}天与滞后{lag2}天的差值...")
            
            # 多产品均值差值
            mean_lag1 = multi_mean[MAX_LOOKBACK-lag1:n_days-lag1]
            mean_lag2 = multi_mean[MAX_LOOKBACK-lag2:n_days-lag2]
            
            # 对齐时间维度（取较短的）
            min_len = min(len(mean_lag1), len(mean_lag2))
            diff_lag = mean_lag1[:min_len] - mean_lag2[:min_len]
            
            save_feature(diff_lag, f"lag_diff_{lag1}_{lag2}_multi_product_mean", 
                        f"滞后{lag1}天与{lag2}天多产品均值差值 (valid_time, points)")
    
    # 2.5 滞后比值特征
    print("  生成滞后比值特征...")
    
    epsilon = 1e-6
    for lag in [1, 3, 7]:
        if lag <= MAX_LOOKBACK:
            # 当前值与滞后值的比值
            current_mean = multi_mean[valid_time_slice]
            lag_mean = multi_mean[MAX_LOOKBACK-lag:n_days-lag]
            
            min_len = min(len(current_mean), len(lag_mean))
            ratio = current_mean[:min_len] / (lag_mean[:min_len] + epsilon)
            ratio = np.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.1)
            
            save_feature(ratio, f"lag_ratio_current_{lag}_multi_product_mean", 
                        f"当前与滞后{lag}天多产品均值比值 (valid_time, points)")
    
    # 2.6 移动平均滞后特征
    print("  生成移动平均滞后特征...")
    
    # 计算移动平均，然后生成滞后
    window_sizes = [3, 7, 15]
    
    for window in window_sizes:
        if window <= MAX_LOOKBACK:
            print(f"    计算{window}天移动平均...")
            
            # 计算多产品均值的移动平均
            moving_avg = np.zeros_like(multi_mean)
            for t in range(len(multi_mean)):
                start_idx = max(0, t - window + 1)
                moving_avg[t] = safe_nanmean(multi_mean[start_idx:t+1], axis=0)
            
            # 生成移动平均的滞后特征
            for lag in [1, 3, 7]:
                if lag <= MAX_LOOKBACK:
                    ma_lag = moving_avg[MAX_LOOKBACK-lag:n_days-lag]
                    save_feature(ma_lag, f"lag_{lag}_ma_{window}_multi_product_mean", 
                               f"滞后{lag}天的{window}天移动平均 (valid_time, points)")
    
    # 2.7 关键产品的深度滞后特征
    print("  生成关键产品深度滞后特征...")
    
    # 选择关键产品进行更详细的滞后分析
    key_products = ['GSMAP', 'IMERG', 'SM2RAIN']
    detailed_lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
    
    for product in key_products:
        if product in product_names:
            print(f"    处理{product}的详细滞后特征...")
            product_idx = product_names.index(product)
            
            for lag in detailed_lags:
                if lag <= MAX_LOOKBACK:
                    lag_data = X_points[product_idx, MAX_LOOKBACK-lag:n_days-lag]
                    save_feature(lag_data, f"lag_{lag}_detailed_{product}", 
                               f"{product}详细滞后{lag}天 (valid_time, points)")
    
    # 2.8 组合滞后特征
    print("  生成组合滞后特征...")
    
    # 多个滞后期的组合统计
    for product in ['GSMAP', 'IMERG']:
        if product in product_names:
            product_idx = product_names.index(product)
            
            # 滞后1-7天的平均值
            lag_values = []
            for lag in range(1, 8):
                if lag <= MAX_LOOKBACK:
                    lag_data = X_points[product_idx, MAX_LOOKBACK-lag:n_days-lag]
                    # 对齐到最短长度
                    if len(lag_values) == 0:
                        min_len = len(lag_data)
                    else:
                        min_len = min(min_len, len(lag_data))
                    lag_values.append(lag_data)
            
            if lag_values:
                # 裁剪到统一长度
                lag_values = [lv[:min_len] for lv in lag_values]
                lag_stack = np.stack(lag_values, axis=0)  # (n_lags, time, points)
                
                # 计算统计量
                lag_mean = safe_nanmean(lag_stack, axis=0)
                lag_std = safe_nanstd(lag_stack, axis=0)
                lag_max = np.nanmax(lag_stack, axis=0)
                lag_min = np.nanmin(lag_stack, axis=0)
                
                save_feature(lag_mean, f"lag_1to7_mean_{product}", 
                           f"{product}滞后1-7天平均值 (valid_time, points)")
                save_feature(lag_std, f"lag_1to7_std_{product}", 
                           f"{product}滞后1-7天标准差 (valid_time, points)")
                save_feature(lag_max, f"lag_1to7_max_{product}", 
                           f"{product}滞后1-7天最大值 (valid_time, points)")
                save_feature(lag_min, f"lag_1to7_min_{product}", 
                           f"{product}滞后1-7天最小值 (valid_time, points)")
    
    print(f"\n=== 滞后特征生成完成 ===")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    
    # 统计生成的特征
    lag_features = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('lag_') and f.endswith('.npy')]
    print(f"生成的滞后特征数量: {len(lag_features)}")

if __name__ == "__main__":
    main()