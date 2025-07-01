
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立特征文件展平脚本

功能:
1. 遍历所有独立的特征 .npy 文件。
2. 智能地将不同维度的特征统一到 (总样本数,) 的一维格式。
3. 处理时间切片、空间掩码应用、广播、平铺等对齐问题。
4. 将处理后的数据覆盖保存回原始文件。
"""

import numpy as np
import os
import json
import time
from loadfeatures import FeatureLoader
from scipy.io import loadmat

# --- 配置 ---
FEATURES_SOURCE_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"

# 排除无法有意义地展平到 (总样本数,) 格式的特征
EXCLUDE_PREFIXES = (
    'spatial_neighbor',  # 采样时间维度的空间特征
    'spatial_gradient_magnitude', # 采样时间维度的空间特征
    'spatial_autocorrelation', # 高度聚合的非时间/空间特征
    # 'spatial_cluster_size' # 现在尝试展平这类特征，所以从排除列表中移除
)

# 硬编码统一后的目标 shape 和总样本数
TARGET_TIME_STEPS = 1797
TARGET_POINTS = 2920
TARGET_SHAPE = (TARGET_TIME_STEPS, TARGET_POINTS)
TOTAL_SAMPLES = TARGET_TIME_STEPS * TARGET_POINTS

def get_static_spatial_mask():
    """
    加载并返回一个静态的空间掩码 (144, 256)，用于处理原始空间特征。
    """
    try:
        mask_path = "/mnt/f/rainfalldata/data/processed/nationwide/masks/combined_china_basin_mask.mat"
        mask_mat = loadmat(mask_path)
        mask_data = None
        if 'data' in mask_mat:
            mask_data = mask_mat['data']
        elif 'mask' in mask_mat:
            mask_data = mask_mat['mask']
        else:
            raise KeyError("Mask variable not found in MASK file. Expected 'data' or 'mask'.")
        
        spatial_mask = (mask_data == 2) # 假设2是长江流域
        if np.sum(spatial_mask) != TARGET_POINTS:
            print(f"警告: 静态空间掩码的有效点数({np.sum(spatial_mask)})与目标点数({TARGET_POINTS})不匹配。请检查掩码文件。")

        return spatial_mask
    except Exception as e:
        print(f"错误: 无法加载静态空间掩码: {e}")
        return None

def flatten_features():
    """
    主函数，执行特征文件的展平流程。
    """
    start_time = time.time()
    print(f"=== 开始展平独立特征文件 ===")
    print(f"特征目录: {FEATURES_SOURCE_DIR}")

    # 1. 初始化特征加载器
    print("\n1. 初始化 FeatureLoader...")
    loader = FeatureLoader(FEATURES_SOURCE_DIR)

    # 2. 获取静态空间掩码
    static_spatial_mask = get_static_spatial_mask()
    if static_spatial_mask is None:
        return
    
    print(f"  目标总样本数: {TOTAL_SAMPLES}")

    # 3. 遍历所有特征文件并进行处理
    print("\n2. 正在处理和展平所有特征文件...")
    processed_count = 0
    skipped_count = 0

    for i, fname in enumerate(sorted(loader.feature_files)):
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(loader.feature_files)}...")

        # 检查是否为排除列表中的特征
        if any(fname.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            print(f"  跳过排除特征 '{fname}' (不适合展平)。")
            skipped_count += 1
            continue

        # 尝试加载特征，如果文件已经被展平，则直接使用
        data = np.load(os.path.join(FEATURES_SOURCE_DIR, fname))
        
        # 如果文件已经被展平，并且维度正确，则跳过处理
        if data.ndim == 1 and data.shape[0] == TOTAL_SAMPLES:
            print(f"  特征 '{fname}' 已展平，跳过。")
            processed_count += 1
            continue

        processed_data_2d = None # 目标是 (time, points) 形状
        
        # --- 核心逻辑：维度判断与转换到 (time, points) ---
        # Case A: 已经是 (time, points) 且点数一致 (e.g., 1797, 2920)
        if data.ndim == 2 and data.shape[0] == TARGET_TIME_STEPS and data.shape[1] == TARGET_POINTS:
            processed_data_2d = data
        # Case B: 原始点数据 (e.g., 1827, 2920) - 需要时间切片
        elif data.ndim == 2 and data.shape[0] == 1827 and data.shape[1] == TARGET_POINTS:
            processed_data_2d = data[30:, :] # 应用 MAX_LOOKBACK 切片
        # Case C: 全局时间特征 (e.g., 1797,) - 需要广播
        elif data.ndim == 1 and data.shape[0] == TARGET_TIME_STEPS:
            processed_data_2d = np.broadcast_to(data.reshape(TARGET_TIME_STEPS, 1), TARGET_SHAPE)
        # Case D: 静态空间特征 (e.g., 144, 256) - 需要掩码和平铺
        # 包含 spatial_cluster_size_*.npy
        elif data.ndim == 2 and data.shape == static_spatial_mask.shape:
            flattened_data = data[static_spatial_mask]
            processed_data_2d = np.tile(flattened_data.reshape(1, TARGET_POINTS), (TARGET_TIME_STEPS, 1))
        # Case E: 原始空间数据 (e.g., 1827, 144, 256) 或 _valid 原始空间数据 (e.g., 1797, 144, 256)
        elif data.ndim == 3 and data.shape[1:] == static_spatial_mask.shape:
            temp_list = []
            for t_idx in range(data.shape[0]):
                slice_2d = data[t_idx, :, :]
                temp_list.append(slice_2d[static_spatial_mask])
            processed_data_spatial_flattened = np.array(temp_list) # Shape (original_time, 2920)
            
            # 根据原始时间维度进行切片
            if processed_data_spatial_flattened.shape[0] == 1827: # 原始数据
                processed_data_2d = processed_data_spatial_flattened[30:, :] # 应用 MAX_LOOKBACK 切片
            elif processed_data_spatial_flattened.shape[0] == TARGET_TIME_STEPS: # _valid 数据
                processed_data_2d = processed_data_spatial_flattened
            else:
                print(f"  警告: 特征 '{fname}' 的时间维度({data.shape[0]})不符合预期。跳过。")
                skipped_count += 1
                continue
        else:
            print(f"  警告: 跳过维度不兼容的特征 '{fname}'，shape: {data.shape}。")
            skipped_count += 1
            continue

        # 最终展平并保存
        if processed_data_2d is not None:
            final_flattened_data = processed_data_2d.flatten()
            if final_flattened_data.shape[0] == TOTAL_SAMPLES:
                np.save(os.path.join(FEATURES_SOURCE_DIR, fname), final_flattened_data.astype(np.float32))
                print(f"  已处理并保存: {fname} -> {final_flattened_data.shape}")
                processed_count += 1
            else:
                print(f"  错误: 特征 '{fname}' 最终 shape 不匹配: {final_flattened_data.shape} vs ({TOTAL_SAMPLES},)。跳过。")
                skipped_count += 1
        else:
            skipped_count += 1

    print(f"\n=== 独立特征文件展平完成 ===")
    print(f"成功处理并保存了 {processed_count} 个特征文件。")
    print(f"跳过了 {skipped_count} 个特征文件。")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    flatten_features()
