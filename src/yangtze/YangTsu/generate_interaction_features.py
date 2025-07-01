#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互特征生成器
生成特征间的交互和组合特征
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
    print("=== 生成交互特征 ===")
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
    
    print(f"有效数据形状: {X_points_valid.shape}")
    
    # 预计算基础统计量
    print("\n2. 预计算基础统计量...")
    multi_mean = safe_nanmean(X_points_valid, axis=2)
    multi_std = safe_nanstd(X_points_valid, axis=2)
    multi_max = np.nanmax(X_points_valid, axis=2)
    multi_min = np.nanmin(X_points_valid, axis=2)
    rain_count = np.sum(X_points_valid > 0.1, axis=2).astype(np.float32)
    
    # 时序特征
    days_in_year = 365.25
    day_index = np.arange(MAX_LOOKBACK, n_days, dtype=np.float32)
    day_of_year = day_index % days_in_year
    sin_day = np.sin(2 * np.pi * day_of_year / days_in_year)
    cos_day = np.cos(2 * np.pi * day_of_year / days_in_year)
    
    # 生成交互特征
    print("\n3. 生成交互特征...")
    
    # 3.1 产品间乘性交互
    print("  生成产品间乘性交互特征...")
    
    # 主要产品组合
    key_product_pairs = [
        ('GSMAP', 'IMERG'),
        ('GSMAP', 'SM2RAIN'),
        ('IMERG', 'SM2RAIN'),
        ('GSMAP', 'CMORPH'),
        ('IMERG', 'CHIRPS')
    ]
    
    for prod1, prod2 in key_product_pairs:
        if prod1 in product_names and prod2 in product_names:
            idx1 = product_names.index(prod1)
            idx2 = product_names.index(prod2)
            
            data1 = X_points_valid[:, :, idx1]
            data2 = X_points_valid[:, :, idx2]
            
            # 乘性交互
            interaction_multiply = data1 * data2
            save_feature(interaction_multiply, f"interaction_multiply_{prod1}_{prod2}", 
                        f"{prod1}与{prod2}乘性交互 (valid_time, points)")
            
            # 几何平均
            geometric_mean = np.sqrt(np.abs(data1 * data2))
            save_feature(geometric_mean, f"interaction_geometric_mean_{prod1}_{prod2}", 
                        f"{prod1}与{prod2}几何平均 (valid_time, points)")
            
            # 差异交互
            diff_interaction = np.abs(data1 - data2)
            save_feature(diff_interaction, f"interaction_abs_diff_{prod1}_{prod2}", 
                        f"{prod1}与{prod2}绝对差异 (valid_time, points)")
            
            # 比值交互
            ratio_interaction = safe_divide(data1, data2, default=1.0)
            save_feature(ratio_interaction, f"interaction_ratio_{prod1}_{prod2}", 
                        f"{prod1}与{prod2}比值交互 (valid_time, points)")
    
    # 3.2 统计量与时序特征交互
    print("  生成统计量与时序特征交互...")
    
    # 扩展时序特征到所有点
    sin_day_expanded = sin_day[:, np.newaxis]  # (time, 1)
    cos_day_expanded = cos_day[:, np.newaxis]  # (time, 1)
    
    # 统计量与周期性交互
    interactions_temporal = [
        (multi_mean, "mean"),
        (multi_std, "std"),
        (multi_max, "max"),
        (rain_count, "rain_count")
    ]
    
    for stat_data, stat_name in interactions_temporal:
        # 与正弦交互
        sin_interaction = stat_data * sin_day_expanded
        save_feature(sin_interaction, f"interaction_{stat_name}_sin_day", 
                    f"{stat_name}与年内日周期正弦交互 (valid_time, points)")
        
        # 与余弦交互
        cos_interaction = stat_data * cos_day_expanded
        save_feature(cos_interaction, f"interaction_{stat_name}_cos_day", 
                    f"{stat_name}与年内日周期余弦交互 (valid_time, points)")
    
    # 3.3 时间趋势交互
    print("  生成时间趋势交互特征...")
    
    # 线性时间趋势
    time_trend = np.arange(n_valid_days, dtype=np.float32)[:, np.newaxis]
    normalized_time = time_trend / (n_valid_days - 1)
    
    # 统计量与时间趋势交互
    for stat_data, stat_name in interactions_temporal:
        time_interaction = stat_data * normalized_time
        save_feature(time_interaction, f"interaction_{stat_name}_time_trend", 
                    f"{stat_name}与时间趋势交互 (valid_time, points)")
        
        # 二次时间交互
        quadratic_time_interaction = stat_data * (normalized_time ** 2)
        save_feature(quadratic_time_interaction, f"interaction_{stat_name}_quadratic_time", 
                    f"{stat_name}与二次时间趋势交互 (valid_time, points)")
    
    # 3.4 条件统计量交互
    print("  生成条件统计量交互特征...")
    
    # 低强度条件下的交互
    low_intensity_threshold = 0.5
    low_intensity_mask = multi_mean < low_intensity_threshold
    
    # 低强度下的标准差与变异系数交互
    cv = safe_divide(multi_std, multi_mean)
    low_intensity_std_cv = multi_std * cv
    low_intensity_std_cv[~low_intensity_mask] = 0.0
    save_feature(low_intensity_std_cv, "interaction_low_intensity_std_cv", 
                "低强度条件下标准差与变异系数交互 (valid_time, points)")
    
    # 高强度条件下的交互
    high_intensity_threshold = 2.0
    high_intensity_mask = multi_mean > high_intensity_threshold
    
    high_intensity_mean_max = multi_mean * multi_max
    high_intensity_mean_max[~high_intensity_mask] = 0.0
    save_feature(high_intensity_mean_max, "interaction_high_intensity_mean_max", 
                "高强度条件下均值与最大值交互 (valid_time, points)")
    
    # 3.5 降雨事件相关交互
    print("  生成降雨事件相关交互特征...")
    
    # 降雨产品数量与统计量交互
    rain_count_interactions = [
        (multi_mean, "mean"),
        (multi_std, "std"),
        (cv, "cv")
    ]
    
    for stat_data, stat_name in rain_count_interactions:
        rain_stat_interaction = rain_count * stat_data
        save_feature(rain_stat_interaction, f"interaction_rain_count_{stat_name}", 
                    f"降雨产品数量与{stat_name}交互 (valid_time, points)")
    
    # 极值事件交互
    extreme_threshold = 5.0
    extreme_mask = multi_max > extreme_threshold
    extreme_indicator = extreme_mask.astype(np.float32)
    
    # 极值指示与其他特征交互
    extreme_interactions = [
        (multi_std, "std"),
        (rain_count, "rain_count"),
        (multi_mean, "mean")
    ]
    
    for stat_data, stat_name in extreme_interactions:
        extreme_interaction = extreme_indicator * stat_data
        save_feature(extreme_interaction, f"interaction_extreme_{stat_name}", 
                    f"极值事件与{stat_name}交互 (valid_time, points)")
    
    # 3.6 滞后交互特征
    print("  生成滞后交互特征...")
    
    # 当前与滞后统计量交互
    lag_days = [1, 3, 7]
    
    for lag in lag_days:
        if lag <= MAX_LOOKBACK:
            # 计算滞后统计量
            lag_mean = safe_nanmean(X_points_reorder[MAX_LOOKBACK-lag:n_days-lag], axis=2)
            lag_std = safe_nanstd(X_points_reorder[MAX_LOOKBACK-lag:n_days-lag], axis=2)
            
            # 对齐维度
            min_len = min(len(multi_mean), len(lag_mean))
            current_mean_aligned = multi_mean[:min_len]
            current_std_aligned = multi_std[:min_len]
            lag_mean_aligned = lag_mean[:min_len]
            lag_std_aligned = lag_std[:min_len]
            
            # 当前与滞后均值交互
            current_lag_mean_interaction = current_mean_aligned * lag_mean_aligned
            save_feature(current_lag_mean_interaction, f"interaction_current_lag{lag}_mean", 
                        f"当前与滞后{lag}天均值交互 (valid_time, points)")
            
            # 当前与滞后变化交互
            mean_change = current_mean_aligned - lag_mean_aligned
            std_change = current_std_aligned - lag_std_aligned
            change_interaction = mean_change * std_change
            save_feature(change_interaction, f"interaction_mean_std_change_lag{lag}", 
                        f"均值与标准差{lag}天变化交互 (valid_time, points)")
    
    # 3.7 多产品复合交互
    print("  生成多产品复合交互特征...")
    
    # 三元产品交互（选择主要产品）
    if all(prod in product_names for prod in ['GSMAP', 'IMERG', 'SM2RAIN']):
        gsmap_idx = product_names.index('GSMAP')
        imerg_idx = product_names.index('IMERG')
        sm2rain_idx = product_names.index('SM2RAIN')
        
        gsmap_data = X_points_valid[:, :, gsmap_idx]
        imerg_data = X_points_valid[:, :, imerg_idx]
        sm2rain_data = X_points_valid[:, :, sm2rain_idx]
        
        # 三元几何平均
        triple_geometric_mean = np.power(np.abs(gsmap_data * imerg_data * sm2rain_data), 1/3)
        save_feature(triple_geometric_mean, "interaction_triple_geometric_mean_GSI", 
                    "GSMAP-IMERG-SM2RAIN三元几何平均 (valid_time, points)")
        
        # 三元方差
        triple_data = np.stack([gsmap_data, imerg_data, sm2rain_data], axis=2)
        triple_std = safe_nanstd(triple_data, axis=2)
        save_feature(triple_std, "interaction_triple_std_GSI", 
                    "GSMAP-IMERG-SM2RAIN三元标准差 (valid_time, points)")
    
    # 3.8 非线性交互特征
    print("  生成非线性交互特征...")
    
    # 平方交互
    mean_squared = multi_mean ** 2
    save_feature(mean_squared, "interaction_mean_squared", 
                "多产品均值平方 (valid_time, points)")
    
    # 对数交互（处理小值）
    log_mean = np.log(multi_mean + 1.0)  # 加1避免log(0)
    save_feature(log_mean, "interaction_log_mean", 
                "多产品均值对数 (valid_time, points)")
    
    # 指数交互（限制范围避免溢出）
    exp_std = np.exp(np.clip(multi_std, 0, 5))  # 限制在合理范围
    save_feature(exp_std, "interaction_exp_std", 
                "多产品标准差指数 (valid_time, points)")
    
    # 3.9 组合一致性交互
    print("  生成组合一致性交互特征...")
    
    # 最大最小比值与标准差交互
    max_min_ratio = safe_divide(multi_min, multi_max, default=1.0)
    consistency_std_interaction = max_min_ratio * multi_std
    save_feature(consistency_std_interaction, "interaction_consistency_std", 
                "一致性比值与标准差交互 (valid_time, points)")
    
    # 变异系数与降雨数量交互
    cv_rain_count_interaction = cv * rain_count
    save_feature(cv_rain_count_interaction, "interaction_cv_rain_count", 
                "变异系数与降雨数量交互 (valid_time, points)")
    
    # 3.10 复合条件交互
    print("  生成复合条件交互特征...")
    
    # 季节-强度复合条件
    month = (day_of_year // 30.4375).astype(int) % 12 + 1
    season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    season = np.array([season_map[m] for m in month])
    
    # 夏季高强度交互
    summer_mask = (season == 2)  # 夏季
    high_intensity_mask = multi_mean > 1.0
    summer_high_intensity = summer_mask[:, np.newaxis] & high_intensity_mask
    
    summer_high_interaction = summer_high_intensity.astype(np.float32) * multi_std
    save_feature(summer_high_interaction, "interaction_summer_high_intensity_std", 
                "夏季高强度条件下的标准差 (valid_time, points)")
    
    # 冬季低强度交互
    winter_mask = (season == 0)  # 冬季
    low_intensity_mask = multi_mean < 0.5
    winter_low_intensity = winter_mask[:, np.newaxis] & low_intensity_mask
    
    winter_low_interaction = winter_low_intensity.astype(np.float32) * cv
    save_feature(winter_low_interaction, "interaction_winter_low_intensity_cv", 
                "冬季低强度条件下的变异系数 (valid_time, points)")
    
    print(f"\n=== 交互特征生成完成 ===")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    
    # 统计生成的特征
    interaction_features = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('interaction_') and f.endswith('.npy')]
    print(f"生成的交互特征数量: {len(interaction_features)}")

if __name__ == "__main__":
    main()