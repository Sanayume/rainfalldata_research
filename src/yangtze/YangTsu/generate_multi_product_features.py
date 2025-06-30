#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多产品协同特征生成器
生成产品间的统计关系和一致性特征
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
    print("=== 生成多产品协同特征 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 加载数据
    print("\n1. 加载长江流域数据...")
    start_time = time.time()
    ALL_DATA = mydata()
    
    # 只加载点数据用于多产品统计
    X_points, Y_points = ALL_DATA.get_basin_point_data(basin_mask_value=2)
    product_names = ALL_DATA.get_products()
    
    print(f"点数据形状: X_points {X_points.shape}, Y_points {Y_points.shape}")
    print(f"产品列表: {product_names}")
    print(f"数据加载耗时: {time.time() - start_time:.2f}秒")
    
    n_products, n_days, n_points = X_points.shape
    
    # 处理时间依赖性
    valid_time_slice = slice(MAX_LOOKBACK, n_days)
    n_valid_days = n_days - MAX_LOOKBACK
    
    # 转换为 (time, points, products) 便于计算
    X_points_reorder = np.transpose(X_points, (1, 2, 0))  # (time, points, products)
    X_points_valid = X_points_reorder[valid_time_slice]
    
    print(f"有效数据形状: {X_points_valid.shape}")
    
    # 生成多产品协同特征
    print("\n2. 生成多产品协同特征...")
    
    # 2.1 当前时刻多产品统计
    print("  计算多产品基础统计量...")
    save_feature(safe_nanmean(X_points_valid, axis=2), "multi_product_mean", 
                "多产品均值 (valid_time, points)")
    save_feature(safe_nanstd(X_points_valid, axis=2), "multi_product_std", 
                "多产品标准差 (valid_time, points)")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_feature(np.nanmedian(X_points_valid, axis=2), "multi_product_median", 
                    "多产品中位数 (valid_time, points)")
        save_feature(np.nanmax(X_points_valid, axis=2), "multi_product_max", 
                    "多产品最大值 (valid_time, points)")
        save_feature(np.nanmin(X_points_valid, axis=2), "multi_product_min", 
                    "多产品最小值 (valid_time, points)")
        
        max_vals = np.nanmax(X_points_valid, axis=2)
        min_vals = np.nanmin(X_points_valid, axis=2)
        save_feature(max_vals - min_vals, "multi_product_range", 
                    "多产品极差 (valid_time, points)")
    
    # 2.2 降雨产品计数
    print("  计算降雨产品计数...")
    rain_mask = X_points_valid > RAIN_THR
    save_feature(np.sum(rain_mask, axis=2).astype(np.float32), "rain_product_count", 
                "指示降雨的产品数量 (valid_time, points)")
    
    # 2.3 变异系数
    print("  计算变异系数...")
    mean_vals = safe_nanmean(X_points_valid, axis=2)
    std_vals = safe_nanstd(X_points_valid, axis=2)
    cv = safe_divide(std_vals, mean_vals)
    save_feature(cv, "multi_product_cv", "多产品变异系数 (valid_time, points)")
    
    # 2.4 产品间相关性特征
    print("  计算产品间相关性...")
    correlation_window = 30  # 30天滚动窗口计算相关性
    
    for i, prod1 in enumerate(product_names):
        for j, prod2 in enumerate(product_names):
            if i < j:  # 只计算上三角，避免重复
                print(f"    计算 {prod1} vs {prod2}")
                corr_map = np.zeros((n_valid_days, n_points))
                
                for t in range(n_valid_days):
                    for p in range(min(100, n_points)):  # 限制点数减少计算量
                        if t >= correlation_window:
                            start_idx = t - correlation_window
                            x1 = X_points_valid[start_idx:t+1, p, i]
                            x2 = X_points_valid[start_idx:t+1, p, j]
                            valid_mask = ~(np.isnan(x1) | np.isnan(x2))
                            if np.sum(valid_mask) > 10:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    corr_coef = np.corrcoef(x1[valid_mask], x2[valid_mask])
                                    if not np.isnan(corr_coef[0, 1]):
                                        corr_map[t, p] = corr_coef[0, 1]
                
                save_feature(corr_map, f"correlation_{prod1}_{prod2}", 
                           f"{prod1}与{prod2}滚动相关性 (valid_time, points)")
    
    # 2.5 一致性指标
    print("  计算一致性指标...")
    # 产品间最大最小比值
    max_vals = np.nanmax(X_points_valid, axis=2)
    min_vals = np.nanmin(X_points_valid, axis=2)
    consistency_ratio = safe_divide(min_vals, max_vals, default=1.0)
    save_feature(consistency_ratio, "product_consistency_ratio", 
                "产品一致性比值(最小/最大) (valid_time, points)")
    
    # 2.6 产品分歧度特征
    print("  计算产品分歧度...")
    # 标准化后的标准差
    normalized_std = safe_divide(std_vals, mean_vals + 1.0)  # 加1避免小值的影响
    save_feature(normalized_std, "product_disagreement", 
                "产品分歧度 (valid_time, points)")
    
    # 2.7 极值产品识别
    print("  计算极值产品特征...")
    # 识别最大值和最小值来自哪个产品（安全处理全NaN情况）
    max_product_idx = np.zeros((n_valid_days, n_points), dtype=np.float32)
    min_product_idx = np.zeros((n_valid_days, n_points), dtype=np.float32)
    
    for t in range(n_valid_days):
        for p in range(n_points):
            values = X_points_valid[t, p, :]
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                valid_values = values[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                max_product_idx[t, p] = valid_indices[np.argmax(valid_values)]
                min_product_idx[t, p] = valid_indices[np.argmin(valid_values)]
            else:
                max_product_idx[t, p] = -1  # 表示无有效数据
                min_product_idx[t, p] = -1
    
    save_feature(max_product_idx, "max_product_index", 
                "最大值产品索引 (valid_time, points)")
    save_feature(min_product_idx, "min_product_index", 
                "最小值产品索引 (valid_time, points)")
    
    # 2.8 产品权重特征
    print("  计算产品权重特征...")
    # 基于历史性能的简单权重（这里用方差倒数作为权重）
    weights = np.zeros_like(X_points_valid)
    for i in range(n_products):
        product_var = np.nanvar(X_points_valid[:, :, i], axis=0, keepdims=True)
        product_weight = safe_divide(1.0, product_var + EPSILON)
        weights[:, :, i] = product_weight
    
    # 归一化权重
    weight_sum = np.nansum(weights, axis=2, keepdims=True)
    normalized_weights = safe_divide(weights, weight_sum)
    
    # 加权平均
    weighted_mean = np.nansum(X_points_valid * normalized_weights, axis=2)
    save_feature(weighted_mean, "weighted_multi_product_mean", 
                "加权多产品均值 (valid_time, points)")
    
    print(f"\n=== 多产品协同特征生成完成 ===")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    
    # 统计生成的特征
    multi_features = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(('multi_product_', 'rain_product_', 'correlation_', 'product_', 'weighted_', 'max_product_', 'min_product_')) and f.endswith('.npy')]
    print(f"生成的多产品特征数量: {len(multi_features)}")

if __name__ == "__main__":
    main()