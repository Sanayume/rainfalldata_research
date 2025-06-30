#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的单独特征生成系统
分批生成特征，避免超时，处理NaN值警告
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

def generate_temporal_features(X_points, Y_points, product_names, n_days, n_points, valid_time_slice):
    """生成时序特征"""
    print("\n=== 生成时序特征 ===")
    
    n_valid_days = n_days - MAX_LOOKBACK
    X_points_reorder = np.transpose(X_points, (1, 2, 0))  # (time, points, products)
    X_points_valid = X_points_reorder[valid_time_slice]
    
    # 周期性特征
    print("  生成周期性特征...")
    days_in_year = 365.25
    day_index = np.arange(n_days, dtype=np.float32)
    day_of_year = day_index % days_in_year
    
    sin_day = np.sin(2 * np.pi * day_of_year / days_in_year)
    cos_day = np.cos(2 * np.pi * day_of_year / days_in_year)
    save_feature(sin_day[valid_time_slice], "sin_day_of_year", "年内日周期正弦")
    save_feature(cos_day[valid_time_slice], "cos_day_of_year", "年内日周期余弦")
    
    # 季节特征
    month = (day_of_year // 30.4375).astype(int) % 12 + 1
    season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    season = np.array([season_map[m] for m in month])
    
    for s in range(4):
        season_onehot = (season == s).astype(np.float32)
        save_feature(season_onehot[valid_time_slice], f"season_onehot_{s}", f"季节{s}独热编码")
    
    # 滞后特征
    print("  生成滞后特征...")
    for lag in [1, 2, 3, 7]:
        if lag <= MAX_LOOKBACK:
            for i, product in enumerate(product_names):
                lag_data = X_points[i, MAX_LOOKBACK-lag:n_days-lag]
                save_feature(lag_data, f"lag_{lag}_points_{product}", 
                           f"{product}滞后{lag}天点数据")
    
    # 多产品统计量的滞后
    multi_mean = safe_nanmean(X_points_reorder, axis=2)
    multi_std = safe_nanstd(X_points_reorder, axis=2)
    
    for lag in [1, 2, 3]:
        if lag <= MAX_LOOKBACK:
            save_feature(multi_mean[MAX_LOOKBACK-lag:n_days-lag], f"lag_{lag}_multi_product_mean", 
                        f"多产品均值滞后{lag}天")
            save_feature(multi_std[MAX_LOOKBACK-lag:n_days-lag], f"lag_{lag}_multi_product_std", 
                        f"多产品标准差滞后{lag}天")
    
    # 差分特征
    print("  生成差分特征...")
    diff_mean = np.diff(multi_mean[valid_time_slice], axis=0, prepend=0)
    diff_std = np.diff(multi_std[valid_time_slice], axis=0, prepend=0)
    save_feature(diff_mean, "diff_1_multi_product_mean", "多产品均值一阶差分")
    save_feature(diff_std, "diff_1_multi_product_std", "多产品标准差一阶差分")

def generate_advanced_features(X_points, product_names, valid_time_slice):
    """生成高级特征"""
    print("\n=== 生成高级特征 ===")
    
    X_points_reorder = np.transpose(X_points, (1, 2, 0))
    X_points_valid = X_points_reorder[valid_time_slice]
    
    # 分位数特征
    print("  生成分位数特征...")
    for q in [0.25, 0.75, 0.9]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantile_vals = np.nanquantile(X_points_valid, q, axis=2)
            quantile_vals = np.nan_to_num(quantile_vals, nan=0.0)
            save_feature(quantile_vals, f"multi_product_quantile_{int(q*100)}", 
                        f"多产品{int(q*100)}%分位数")
    
    # 阈值特征
    print("  生成阈值特征...")
    for threshold in [0.1, 0.5, 1.0]:
        extreme_ratio = np.mean(X_points_valid > threshold, axis=2)
        save_feature(extreme_ratio, f"extreme_ratio_above_{threshold}", 
                    f"超过{threshold}mm的产品比例")
    
    # 强度分箱
    print("  生成强度分箱特征...")
    mean_rainfall = safe_nanmean(X_points_valid, axis=2)
    intensity_bins = np.digitize(mean_rainfall, bins=[0.1, 0.5, 1.0, 5.0])
    
    for bin_idx in range(5):
        bin_onehot = (intensity_bins == bin_idx).astype(np.float32)
        save_feature(bin_onehot, f"intensity_bin_{bin_idx}", f"强度分箱{bin_idx}")

def generate_interaction_features(X_points, valid_time_slice):
    """生成交互特征"""
    print("\n=== 生成交互特征 ===")
    
    X_points_reorder = np.transpose(X_points, (1, 2, 0))
    X_points_valid = X_points_reorder[valid_time_slice]
    
    # 简化的交互特征
    mean_vals = safe_nanmean(X_points_valid, axis=2)
    std_vals = safe_nanstd(X_points_valid, axis=2)
    
    # 时间相关交互
    n_valid_days = X_points_valid.shape[0]
    time_trend = np.arange(n_valid_days, dtype=np.float32)[:, np.newaxis]
    
    mean_time_interaction = mean_vals * time_trend
    std_time_interaction = std_vals * time_trend
    
    save_feature(mean_time_interaction, "interaction_mean_time", "均值与时间趋势交互")
    save_feature(std_time_interaction, "interaction_std_time", "标准差与时间趋势交互")
    
    # 统计量交互
    mean_std_interaction = mean_vals * std_vals
    save_feature(mean_std_interaction, "interaction_mean_std", "均值与标准差交互")

def generate_rolling_features(X_points, product_names, valid_time_slice):
    """生成滑动窗口特征"""
    print("\n=== 生成滑动窗口特征 ===")
    
    n_products, n_days, n_points = X_points.shape
    n_valid_days = n_days - MAX_LOOKBACK
    
    # 只为主要产品生成滑动特征，减少计算量
    main_products = ['GSMAP', 'IMERG', 'SM2RAIN']
    
    for window in [3, 7, 15]:
        if window <= MAX_LOOKBACK:
            print(f"    处理{window}天窗口...")
            for i, product in enumerate(product_names):
                if product in main_products:
                    rolling_mean = np.zeros((n_valid_days, n_points))
                    rolling_std = np.zeros((n_valid_days, n_points))
                    
                    for t in range(n_valid_days):
                        start_idx = max(0, MAX_LOOKBACK + t - window)
                        end_idx = MAX_LOOKBACK + t + 1
                        window_data = X_points[i, start_idx:end_idx]
                        
                        rolling_mean[t] = safe_nanmean(window_data, axis=0)
                        rolling_std[t] = safe_nanstd(window_data, axis=0)
                    
                    save_feature(rolling_mean, f"rolling_{window}d_mean_{product}", 
                               f"{product} {window}天滑动均值")
                    save_feature(rolling_std, f"rolling_{window}d_std_{product}", 
                               f"{product} {window}天滑动标准差")

def main():
    print("=== 优化的单独特征生成系统 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 检查已完成的基础特征
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
    print(f"已存在特征文件: {len(existing_files)}个")
    
    # 如果基础特征已存在，跳过数据加载
    if len(existing_files) > 20:
        print("检测到已有基础特征，跳过重复生成...")
        
        # 从现有文件中获取数据维度信息
        sample_file = os.path.join(OUTPUT_DIR, "raw_points_GSMAP.npy")
        if os.path.exists(sample_file):
            sample_data = np.load(sample_file)
            n_days, n_points = sample_data.shape
            n_valid_days = n_days - MAX_LOOKBACK
            valid_time_slice = slice(MAX_LOOKBACK, n_days)
            product_names = ['CMORPH', 'CHIRPS', 'SM2RAIN', 'IMERG', 'GSMAP', 'PERSIANN']
            
            print(f"数据维度: {n_days}天, {n_points}个点")
            
            # 加载基础数据用于高级特征计算
            print("加载基础数据用于高级特征计算...")
            ALL_DATA = mydata()
            X_points, Y_points = ALL_DATA.get_basin_point_data(basin_mask_value=2)
        else:
            print("未找到样本文件，重新加载数据...")
            # 完整数据加载流程
            start_time = time.time()
            ALL_DATA = mydata()
            X_spatial, Y_spatial = ALL_DATA.get_basin_spatial_data(basin_mask_value=2)
            X_points, Y_points = ALL_DATA.get_basin_point_data(basin_mask_value=2)
            product_names = ALL_DATA.get_products()
            print(f"数据加载耗时: {time.time() - start_time:.2f}秒")
            
            n_products, n_days, n_points = X_points.shape
            valid_time_slice = slice(MAX_LOOKBACK, n_days)
    else:
        print("开始完整特征生成...")
        start_time = time.time()
        ALL_DATA = mydata()
        X_spatial, Y_spatial = ALL_DATA.get_basin_spatial_data(basin_mask_value=2)
        X_points, Y_points = ALL_DATA.get_basin_point_data(basin_mask_value=2)
        product_names = ALL_DATA.get_products()
        print(f"数据加载耗时: {time.time() - start_time:.2f}秒")
        
        n_products, n_days, n_points = X_points.shape
        valid_time_slice = slice(MAX_LOOKBACK, n_days)
        
        # 基础特征已由第一个脚本生成，这里跳过
        print("基础特征生成已完成，跳过...")
    
    # 生成高级特征
    print(f"\n开始生成高级特征...")
    
    # 1. 时序特征
    generate_temporal_features(X_points, Y_points, product_names, n_days, n_points, valid_time_slice)
    
    # 2. 高级统计特征
    generate_advanced_features(X_points, product_names, valid_time_slice)
    
    # 3. 交互特征
    generate_interaction_features(X_points, valid_time_slice)
    
    # 4. 滑动窗口特征
    generate_rolling_features(X_points, product_names, valid_time_slice)
    
    # 生成最终特征清单
    feature_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
    feature_files.sort()
    
    print(f"\n=== 特征生成完成 ===")
    print(f"总特征文件数量: {len(feature_files)}")
    
    # 保存特征清单
    with open(os.path.join(OUTPUT_DIR, "feature_list.txt"), 'w', encoding='utf-8') as f:
        f.write("# 生成的特征文件列表\n")
        f.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 总文件数: {len(feature_files)}\n\n")
        
        # 按类别分组
        categories = {
            'raw_': '原始数据特征',
            'target_': '目标变量特征',
            'multi_product_': '多产品协同特征',
            'rain_product_': '降雨产品特征',
            'sin_': '周期性特征',
            'cos_': '周期性特征',
            'season_': '季节特征',
            'lag_': '滞后特征',
            'diff_': '差分特征',
            'rolling_': '滑动窗口特征',
            'quantile_': '分位数特征',
            'extreme_': '极值特征',
            'intensity_': '强度特征',
            'interaction_': '交互特征'
        }
        
        for prefix, category_name in categories.items():
            category_files = [f for f in feature_files if f.startswith(prefix)]
            if category_files:
                f.write(f"\n## {category_name} ({len(category_files)}个)\n")
                for file_name in category_files:
                    f.write(f"{file_name}\n")
        
        # 其他特征
        other_files = []
        for file_name in feature_files:
            if not any(file_name.startswith(prefix) for prefix in categories.keys()):
                other_files.append(file_name)
        
        if other_files:
            f.write(f"\n## 其他特征 ({len(other_files)}个)\n")
            for file_name in other_files:
                f.write(f"{file_name}\n")
    
    print("特征清单已保存到: feature_list.txt")

if __name__ == "__main__":
    main()