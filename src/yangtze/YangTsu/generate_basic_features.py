#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础原始特征生成器
生成各产品的原始数据和目标变量
"""

import numpy as np
import os
import time
from loaddata import mydata

# 配置
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
MAX_LOOKBACK = 30

def save_feature(feature_data, feature_name, description=""):
    """保存特征到npy文件"""
    filepath = os.path.join(OUTPUT_DIR, f"{feature_name}.npy")
    np.save(filepath, feature_data.astype(np.float32))
    print(f"  Saved: {feature_name}.npy {feature_data.shape} - {description}")

def main():
    print("=== 生成基础原始特征 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 加载数据
    print("\n1. 加载长江流域数据...")
    start_time = time.time()
    ALL_DATA = mydata()
    
    # 加载空间数据和点数据
    X_spatial, Y_spatial = ALL_DATA.get_basin_spatial_data(basin_mask_value=2)
    X_points, Y_points = ALL_DATA.get_basin_point_data(basin_mask_value=2)
    product_names = ALL_DATA.get_products()
    
    print(f"空间数据形状: X_spatial {X_spatial.shape}, Y_spatial {Y_spatial.shape}")
    print(f"点数据形状: X_points {X_points.shape}, Y_points {Y_points.shape}")
    print(f"产品列表: {product_names}")
    print(f"数据加载耗时: {time.time() - start_time:.2f}秒")
    
    n_products, n_days, n_lat, n_lon = X_spatial.shape
    _, _, n_points = X_points.shape
    
    # 处理时间依赖性
    valid_time_slice = slice(MAX_LOOKBACK, n_days)
    n_valid_days = n_days - MAX_LOOKBACK
    print(f"有效时间范围: {MAX_LOOKBACK} 到 {n_days-1} (共{n_valid_days}天)")
    
    # 生成基础特征
    print("\n2. 生成基础原始特征...")
    
    # 2.1 空间数据的原始值
    print("  生成空间原始特征...")
    for i, product in enumerate(product_names):
        # 完整时间序列的空间数据
        save_feature(X_spatial[i], f"raw_spatial_{product}", 
                    f"{product}产品完整空间数据 (time, lat, lon)")
        
        # 有效时间范围的空间数据
        save_feature(X_spatial[i, valid_time_slice], f"raw_spatial_{product}_valid", 
                    f"{product}产品有效时间空间数据 (valid_time, lat, lon)")
    
    # 2.2 点数据的原始值
    print("  生成点原始特征...")
    for i, product in enumerate(product_names):
        # 完整时间序列的点数据
        save_feature(X_points[i], f"raw_points_{product}", 
                    f"{product}产品完整点数据 (time, points)")
        
        # 有效时间范围的点数据
        save_feature(X_points[i, valid_time_slice], f"raw_points_{product}_valid", 
                    f"{product}产品有效时间点数据 (valid_time, points)")
    
    # 2.3 目标变量
    print("  生成目标变量特征...")
    save_feature(Y_spatial, "target_spatial", "CHM目标变量空间数据 (time, lat, lon)")
    save_feature(Y_spatial[valid_time_slice], "target_spatial_valid", "CHM目标变量有效时间空间数据")
    save_feature(Y_points, "target_points", "CHM目标变量点数据 (time, points)")
    save_feature(Y_points[valid_time_slice], "target_points_valid", "CHM目标变量有效时间点数据")
    
    print(f"\n=== 基础特征生成完成 ===")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    
    # 统计生成的特征
    basic_features = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(('raw_', 'target_')) and f.endswith('.npy')]
    print(f"生成的基础特征数量: {len(basic_features)}")

if __name__ == "__main__":
    main()