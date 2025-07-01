#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实空间特征生成器
基于空间数据生成真实的空间特征，保持空间意义
"""

import numpy as np
import os
import time
import warnings
from scipy import ndimage
from scipy.stats import skew, kurtosis
from loaddata import mydata

# 抑制warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 配置
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
MAX_LOOKBACK = 30

def save_feature(feature_data, feature_name, description=""):
    """保存特征到npy文件"""
    filepath = os.path.join(OUTPUT_DIR, f"{feature_name}.npy")
    np.save(filepath, feature_data.astype(np.float32))
    print(f"  Saved: {feature_name}.npy {feature_data.shape} - {description}")

def calculate_spatial_gradients(data_2d):
    """计算二维数据的空间梯度"""
    if len(data_2d.shape) != 2:
        raise ValueError("Input must be 2D array")
    
    # 处理NaN值
    data_filled = np.nan_to_num(data_2d, nan=0.0)
    
    # 使用Sobel算子计算梯度
    grad_x = ndimage.sobel(data_filled, axis=1)  # 经度方向梯度
    grad_y = ndimage.sobel(data_filled, axis=0)  # 纬度方向梯度
    
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
    
    # 使用反射填充处理边界
    padded = np.pad(data_2d, half_window, mode='reflect')
    
    rows, cols = data_2d.shape
    mean_result = np.zeros_like(data_2d)
    std_result = np.zeros_like(data_2d)
    max_result = np.zeros_like(data_2d)
    min_result = np.zeros_like(data_2d)
    
    # 使用更高效的方法计算邻域统计
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

def calculate_spatial_autocorrelation(data_2d, max_lag=5):
    """计算空间自相关"""
    if len(data_2d.shape) != 2:
        raise ValueError("Input must be 2D array")
    
    # 简化的空间自相关计算
    rows, cols = data_2d.shape
    autocorr = np.zeros((max_lag + 1, max_lag + 1))
    
    # 计算不同滞后的自相关
    data_filled = np.nan_to_num(data_2d, nan=0.0)
    data_centered = data_filled - np.mean(data_filled)
    
    for lag_i in range(max_lag + 1):
        for lag_j in range(max_lag + 1):
            if lag_i == 0 and lag_j == 0:
                autocorr[lag_i, lag_j] = 1.0
            else:
                shifted = np.roll(np.roll(data_centered, lag_i, axis=0), lag_j, axis=1)
                correlation = np.corrcoef(data_centered.flatten(), shifted.flatten())[0, 1]
                autocorr[lag_i, lag_j] = correlation if not np.isnan(correlation) else 0.0
    
    return autocorr

def main():
    print("=== 生成真实空间特征 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 加载数据
    print("\n1. 加载长江流域空间数据...")
    start_time = time.time()
    ALL_DATA = mydata()
    
    # 加载空间数据 - 这是关键，保持真实空间结构
    X_spatial, Y_spatial = ALL_DATA.get_basin_spatial_data(basin_mask_value=2)
    product_names = ALL_DATA.get_products()
    
    print(f"空间数据形状: X_spatial {X_spatial.shape}, Y_spatial {Y_spatial.shape}")
    print(f"数据加载耗时: {time.time() - start_time:.2f}秒")
    
    n_products, n_days, n_lat, n_lon = X_spatial.shape
    
    # 处理时间依赖性
    valid_time_slice = slice(MAX_LOOKBACK, n_days)
    n_valid_days = n_days - MAX_LOOKBACK
    
    print(f"有效时间范围: {MAX_LOOKBACK} 到 {n_days-1} (共{n_valid_days}天)")
    print(f"空间网格: {n_lat} x {n_lon}")
    
    # 生成空间特征
    print("\n2. 生成真实空间特征...")
    
    # 2.1 空间梯度特征
    print("  计算空间梯度特征...")
    
    # 为主要产品计算梯度特征
    main_products = ['GSMAP', 'IMERG', 'SM2RAIN']
    
    for i, product in enumerate(product_names):
        if product in main_products:
            print(f"    计算{product}的空间梯度...")
            
            grad_x_all = np.zeros((n_valid_days, n_lat, n_lon))
            grad_y_all = np.zeros((n_valid_days, n_lat, n_lon))
            grad_mag_all = np.zeros((n_valid_days, n_lat, n_lon))
            grad_dir_all = np.zeros((n_valid_days, n_lat, n_lon))
            
            # 只处理部分时间步以减少计算量
            time_step = max(1, n_valid_days // 100)  # 最多处理100个时间步
            time_indices = range(0, n_valid_days, time_step)
            
            for t_idx, t in enumerate(time_indices):
                if t_idx % 10 == 0:
                    print(f"      处理时间步 {t_idx+1}/{len(time_indices)}")
                
                actual_t = MAX_LOOKBACK + t
                data_2d = X_spatial[i, actual_t]
                
                # 只对非全NaN的数据计算梯度
                if not np.all(np.isnan(data_2d)):
                    grad_x, grad_y, grad_mag, grad_dir = calculate_spatial_gradients(data_2d)
                    
                    grad_x_all[t] = grad_x
                    grad_y_all[t] = grad_y
                    grad_mag_all[t] = grad_mag
                    grad_dir_all[t] = grad_dir
            
            # 为了节省空间，只保存梯度幅度和部分时间步的详细梯度
            save_feature(grad_mag_all[::time_step], f"spatial_gradient_magnitude_{product}", 
                        f"{product}梯度幅度 (sampled_time, lat, lon)")
            
            # 保存平均梯度特征
            avg_grad_mag = np.nanmean(grad_mag_all, axis=0)
            save_feature(avg_grad_mag, f"spatial_avg_gradient_magnitude_{product}", 
                        f"{product}平均梯度幅度 (lat, lon)")
    
    # 2.2 邻域统计特征
    print("  计算邻域统计特征...")
    
    window_sizes = [3, 5]  # 减少窗口大小以降低计算量
    
    for window_size in window_sizes:
        print(f"    计算{window_size}x{window_size}邻域统计...")
        
        for i, product in enumerate(product_names):
            if product in main_products:
                print(f"      处理{product}...")
                
                # 只处理少数几个时间步
                sample_times = [0, n_valid_days//4, n_valid_days//2, 3*n_valid_days//4, n_valid_days-1]
                
                neighbor_mean_samples = []
                neighbor_std_samples = []
                
                for t in sample_times:
                    actual_t = MAX_LOOKBACK + t
                    data_2d = X_spatial[i, actual_t]
                    
                    if not np.all(np.isnan(data_2d)):
                        n_mean, n_std, n_max, n_min = calculate_neighborhood_stats(data_2d, window_size)
                        neighbor_mean_samples.append(n_mean)
                        neighbor_std_samples.append(n_std)
                
                if neighbor_mean_samples:
                    # 保存样本时间步的邻域统计
                    neighbor_mean_stack = np.stack(neighbor_mean_samples, axis=0)
                    neighbor_std_stack = np.stack(neighbor_std_samples, axis=0)
                    
                    save_feature(neighbor_mean_stack, 
                               f"spatial_neighbor_{window_size}x{window_size}_mean_{product}_samples", 
                               f"{product} {window_size}x{window_size}邻域均值样本 (sample_time, lat, lon)")
                    
                    # 保存时间平均的邻域统计
                    avg_neighbor_mean = np.nanmean(neighbor_mean_stack, axis=0)
                    avg_neighbor_std = np.nanmean(neighbor_std_stack, axis=0)
                    
                    save_feature(avg_neighbor_mean, 
                               f"spatial_avg_neighbor_{window_size}x{window_size}_mean_{product}", 
                               f"{product} {window_size}x{window_size}平均邻域均值 (lat, lon)")
                    save_feature(avg_neighbor_std, 
                               f"spatial_avg_neighbor_{window_size}x{window_size}_std_{product}", 
                               f"{product} {window_size}x{window_size}平均邻域标准差 (lat, lon)")
    
    # 2.3 空间聚集性特征
    print("  计算空间聚集性特征...")
    
    for i, product in enumerate(product_names):
        if product in main_products:
            print(f"    计算{product}的空间聚集性...")
            
            # 空间方差（每个时刻的空间变异性）
            spatial_var = np.nanvar(X_spatial[i, valid_time_slice], axis=(1, 2))
            save_feature(spatial_var, f"spatial_variance_{product}", 
                        f"{product}空间方差 (valid_time,)")
            
            # 空间偏度和峰度
            spatial_skew = np.zeros(n_valid_days)
            spatial_kurt = np.zeros(n_valid_days)
            
            # 只计算部分时间步
            sample_indices = np.linspace(0, n_valid_days-1, min(50, n_valid_days), dtype=int)
            
            for idx, t in enumerate(sample_indices):
                actual_t = MAX_LOOKBACK + t
                flat_data = X_spatial[i, actual_t].flatten()
                valid_data = flat_data[~np.isnan(flat_data)]
                if len(valid_data) > 10:
                    spatial_skew[t] = skew(valid_data)
                    spatial_kurt[t] = kurtosis(valid_data)
            
            save_feature(spatial_skew, f"spatial_skewness_{product}", 
                        f"{product}空间偏度 (valid_time,)")
            save_feature(spatial_kurt, f"spatial_kurtosis_{product}", 
                        f"{product}空间峰度 (valid_time,)")
    
    # 2.4 空间自相关特征
    print("  计算空间自相关特征...")
    
    for i, product in enumerate(product_names):
        if product in ['GSMAP', 'IMERG']:  # 只为两个主要产品计算
            print(f"    计算{product}的空间自相关...")
            
            # 选择几个代表性时间步
            representative_times = [n_valid_days//4, n_valid_days//2, 3*n_valid_days//4]
            
            autocorr_maps = []
            for t in representative_times:
                actual_t = MAX_LOOKBACK + t
                data_2d = X_spatial[i, actual_t]
                
                if not np.all(np.isnan(data_2d)):
                    autocorr = calculate_spatial_autocorrelation(data_2d, max_lag=3)
                    autocorr_maps.append(autocorr)
            
            if autocorr_maps:
                # 保存平均自相关
                avg_autocorr = np.nanmean(autocorr_maps, axis=0)
                save_feature(avg_autocorr, f"spatial_autocorrelation_{product}", 
                           f"{product}空间自相关 (lag_i, lag_j)")
    
    # 2.5 多产品空间一致性特征
    print("  计算多产品空间一致性...")
    
    # 计算产品间的空间相关性
    product_pairs = [('GSMAP', 'IMERG'), ('GSMAP', 'SM2RAIN'), ('IMERG', 'SM2RAIN')]
    
    for prod1, prod2 in product_pairs:
        if prod1 in product_names and prod2 in product_names:
            print(f"    计算{prod1}与{prod2}的空间相关性...")
            
            idx1 = product_names.index(prod1)
            idx2 = product_names.index(prod2)
            
            # 选择中间时间步
            mid_time = n_valid_days // 2
            actual_t = MAX_LOOKBACK + mid_time
            
            data1 = X_spatial[idx1, actual_t]
            data2 = X_spatial[idx2, actual_t]
            
            # 计算逐像元相关性
            pixel_correlation = np.zeros((n_lat, n_lon))
            
            # 使用滑动窗口计算局部相关性
            window_size = 5
            half_window = window_size // 2
            
            for i in range(half_window, n_lat - half_window):
                for j in range(half_window, n_lon - half_window):
                    # 提取局部窗口
                    window1 = data1[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                    window2 = data2[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                    
                    # 展平并去除NaN
                    flat1 = window1.flatten()
                    flat2 = window2.flatten()
                    valid_mask = ~(np.isnan(flat1) | np.isnan(flat2))
                    
                    if np.sum(valid_mask) > 5:
                        corr = np.corrcoef(flat1[valid_mask], flat2[valid_mask])[0, 1]
                        pixel_correlation[i, j] = corr if not np.isnan(corr) else 0.0
            
            save_feature(pixel_correlation, f"spatial_correlation_{prod1}_{prod2}", 
                        f"{prod1}与{prod2}空间相关性 (lat, lon)")
    
    # 2.6 空间结构特征
    print("  计算空间结构特征...")
    
    # 计算空间聚类特征（简化版）
    for i, product in enumerate(product_names):
        if product in ['GSMAP', 'IMERG']:
            print(f"    计算{product}的空间结构...")
            
            # 选择一个代表性时间步
            mid_time = n_valid_days // 2
            actual_t = MAX_LOOKBACK + mid_time
            data_2d = X_spatial[i, actual_t]
            
            if not np.all(np.isnan(data_2d)):
                # 计算不同阈值下的连通域
                thresholds = [0.1, 1.0, 5.0]
                
                for threshold in thresholds:
                    binary_map = (data_2d > threshold).astype(int)
                    
                    # 计算连通域
                    labeled_array, num_features = ndimage.label(binary_map)
                    
                    # 连通域统计
                    if num_features > 0:
                        # 每个连通域的大小
                        sizes = ndimage.sum(binary_map, labeled_array, range(1, num_features + 1))
                        
                        # 创建特征图
                        cluster_size_map = np.zeros_like(data_2d)
                        for label_id, size in enumerate(sizes, 1):
                            cluster_size_map[labeled_array == label_id] = size
                        
                        save_feature(cluster_size_map, f"spatial_cluster_size_{product}_threshold_{threshold}", 
                                   f"{product}阈值{threshold}连通域大小 (lat, lon)")
    
    print(f"\n=== 真实空间特征生成完成 ===")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    
    # 统计生成的特征
    spatial_features = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('spatial_') and f.endswith('.npy')]
    print(f"生成的空间特征数量: {len(spatial_features)}")

if __name__ == "__main__":
    main()