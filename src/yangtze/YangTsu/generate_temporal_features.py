#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序动态特征生成器
生成周期性、季节性、趋势等时序特征
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

def save_feature(feature_data, feature_name, description=""):
    """保存特征到npy文件"""
    filepath = os.path.join(OUTPUT_DIR, f"{feature_name}.npy")
    np.save(filepath, feature_data.astype(np.float32))
    print(f"  Saved: {feature_name}.npy {feature_data.shape} - {description}")

def main():
    print("=== 生成时序动态特征 ===")
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
    
    # 生成时序特征
    print("\n2. 生成时序动态特征...")
    
    # 2.1 周期性特征
    print("  生成周期性特征...")
    days_in_year = 365.25
    day_index = np.arange(n_days, dtype=np.float32)
    day_of_year = day_index % days_in_year
    
    # 年内日周期
    sin_day = np.sin(2 * np.pi * day_of_year / days_in_year)
    cos_day = np.cos(2 * np.pi * day_of_year / days_in_year)
    save_feature(sin_day[valid_time_slice], "sin_day_of_year", "年内日周期正弦 (valid_time,)")
    save_feature(cos_day[valid_time_slice], "cos_day_of_year", "年内日周期余弦 (valid_time,)")
    
    # 月周期
    sin_month = np.sin(2 * np.pi * day_of_year / 30.4375)
    cos_month = np.cos(2 * np.pi * day_of_year / 30.4375)
    save_feature(sin_month[valid_time_slice], "sin_month_cycle", "月周期正弦 (valid_time,)")
    save_feature(cos_month[valid_time_slice], "cos_month_cycle", "月周期余弦 (valid_time,)")
    
    # 季节特征
    print("  生成季节特征...")
    month = (day_of_year // 30.4375).astype(int) % 12 + 1
    season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    season = np.array([season_map[m] for m in month])
    
    # 季节独热编码
    season_names = ['winter', 'spring', 'summer', 'autumn']
    for s in range(4):
        season_onehot = (season == s).astype(np.float32)
        save_feature(season_onehot[valid_time_slice], f"season_onehot_{s}", 
                    f"季节{season_names[s]}独热编码 (valid_time,)")
    
    # 月份独热编码
    print("  生成月份特征...")
    for m in range(1, 13):
        month_onehot = (month == m).astype(np.float32)
        save_feature(month_onehot[valid_time_slice], f"month_onehot_{m}", 
                    f"月份{m}独热编码 (valid_time,)")
    
    # 2.2 时间趋势特征
    print("  生成时间趋势特征...")
    
    # 线性时间趋势
    time_trend = np.arange(n_valid_days, dtype=np.float32)
    normalized_time = time_trend / (n_valid_days - 1)
    save_feature(time_trend, "time_index", "时间索引 (valid_time,)")
    save_feature(normalized_time, "normalized_time", "归一化时间 (valid_time,)")
    
    # 二次时间趋势
    quadratic_time = normalized_time ** 2
    save_feature(quadratic_time, "quadratic_time", "二次时间趋势 (valid_time,)")
    
    # 2.3 差分特征
    print("  生成差分特征...")
    
    # 转换为 (time, points, products) 便于计算
    X_points_reorder = np.transpose(X_points, (1, 2, 0))
    X_points_valid = X_points_reorder[valid_time_slice]
    
    # 多产品统计量
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multi_mean = np.nanmean(X_points_valid, axis=2)
        multi_std = np.nanstd(X_points_valid, axis=2)
    
    # 一阶差分
    diff_1_mean = np.diff(multi_mean, axis=0, prepend=multi_mean[0:1])
    diff_1_std = np.diff(multi_std, axis=0, prepend=multi_std[0:1])
    save_feature(diff_1_mean, "diff_1_multi_product_mean", "多产品均值一阶差分 (valid_time, points)")
    save_feature(diff_1_std, "diff_1_multi_product_std", "多产品标准差一阶差分 (valid_time, points)")
    
    # 二阶差分
    diff_2_mean = np.diff(diff_1_mean, axis=0, prepend=diff_1_mean[0:1])
    save_feature(diff_2_mean, "diff_2_multi_product_mean", "多产品均值二阶差分 (valid_time, points)")
    
    # 2.4 变化率特征
    print("  生成变化率特征...")
    
    # 相对变化率
    relative_change_mean = np.zeros_like(multi_mean)
    relative_change_mean[1:] = (multi_mean[1:] - multi_mean[:-1]) / (multi_mean[:-1] + 0.01)
    save_feature(relative_change_mean, "relative_change_multi_product_mean", 
                "多产品均值相对变化率 (valid_time, points)")
    
    # 2.5 周期内统计特征
    print("  生成周期内统计特征...")
    
    # 一周内的统计（7天周期）
    week_cycle = day_index % 7
    for day_in_week in range(7):
        day_mask = (week_cycle == day_in_week).astype(np.float32)
        save_feature(day_mask[valid_time_slice], f"day_of_week_{day_in_week}", 
                    f"一周内第{day_in_week}天 (valid_time,)")
    
    # 2.6 累积特征
    print("  生成累积特征...")
    
    # 累积时间（从开始到现在的天数）
    cumulative_days = np.arange(1, n_valid_days + 1, dtype=np.float32)
    save_feature(cumulative_days, "cumulative_days", "累积天数 (valid_time,)")
    
    # 年内累积天数
    year_start_indices = []
    current_year_start = 0
    for i in range(1, n_valid_days):
        if day_of_year[MAX_LOOKBACK + i] < day_of_year[MAX_LOOKBACK + i - 1]:
            year_start_indices.append(current_year_start)
            current_year_start = i
    
    days_in_current_year = np.zeros(n_valid_days, dtype=np.float32)
    current_start = 0
    for i in range(n_valid_days):
        if i in year_start_indices:
            current_start = i
        days_in_current_year[i] = i - current_start + 1
    
    save_feature(days_in_current_year, "days_in_current_year", "年内累积天数 (valid_time,)")
    
    # 2.7 周期强度特征
    print("  生成周期强度特征...")
    
    # 基于历史数据的季节强度
    season_intensity = np.zeros(n_valid_days, dtype=np.float32)
    for s in range(4):
        season_mask = season[valid_time_slice] == s
        if np.any(season_mask):
            # 计算该季节的历史平均降雨强度
            season_data = multi_mean[season_mask]
            if len(season_data) > 0:
                season_avg = np.nanmean(season_data)
                season_intensity[season_mask] = season_avg
    
    save_feature(season_intensity, "season_intensity", "季节强度特征 (valid_time, points)")
    
    print(f"\n=== 时序动态特征生成完成 ===")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    
    # 统计生成的特征
    temporal_features = [f for f in os.listdir(OUTPUT_DIR) if any(f.startswith(prefix) for prefix in [
        'sin_', 'cos_', 'season_', 'month_', 'time_', 'normalized_', 'quadratic_', 
        'diff_', 'relative_', 'day_of_', 'cumulative_', 'days_in_'
    ]) and f.endswith('.npy')]
    print(f"生成的时序特征数量: {len(temporal_features)}")

if __name__ == "__main__":
    main()