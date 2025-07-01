#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征矩阵构建脚本 (总装车间)

功能:
1. 加载所有独立的特征 .npy 文件。
2. 智能地将不同维度的特征统一到 (样本数, 特征数) 的标准格式。
3. 处理时间广播、空间平铺等对齐问题。
4. 将所有处理好的特征堆叠成一个巨大的、可直接用于模型训练的二维矩阵。
5. 保存最终的特征矩阵 (X_matrix.npy)、目标向量 (y_vector.npy) 和特征名称列表 (final_feature_names.json)。
"""

import numpy as np
import os
import json
import time
from loadfeatures import FeatureLoader

# --- 配置 ---
FEATURES_SOURCE_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/model_predict"

# --- 特征选择 ---
EXCLUDE_PREFIXES = (
    'raw_',
    'target_spatial',
    'spatial_neighbor',
    'spatial_gradient_magnitude',
    'spatial_cluster_size',
    'spatial_autocorrelation'  # Exclude non-compatible spatial features
)

def get_unified_mask_and_indices(loader):
    """
    生成统一的空间掩码和点数据的索引，解决空间数据和点数据不一致的问题。
    基准：以空间数据为准。
    """
    print("  正在生成统一的掩码和索引...")
    try:
        raw_spatial_map = loader.load_feature('raw_spatial_GSMAP')
        if raw_spatial_map is None: raise FileNotFoundError("raw_spatial_GSMAP.npy")
        
        target_points_valid = loader.load_feature('target_points_valid')
        if target_points_valid is None: raise FileNotFoundError("target_points_valid.npy")

        # 基准掩码来自于空间数据
        spatial_mask = ~np.isnan(raw_spatial_map[0])
        num_spatial_points = np.sum(spatial_mask)
        num_point_features = target_points_valid.shape[1]

        point_indices = None
        # 如果点数据的列数和空间数据的有效点数不一致
        if num_spatial_points != num_point_features:
            print(f"警告: 空间点数({num_spatial_points})与特征点数({num_point_features})不匹配。将以空间数据为基准进行统一。")
            # 我们假设点数据的前 N (num_spatial_points) 列对应于空间数据中的有效点
            point_indices = np.arange(num_spatial_points)
        else:
            print(f"  数据一致，有效点数: {num_spatial_points}")

        # 最终的目标 shape 以空间数据为准
        TARGET_SHAPE = (target_points_valid.shape[0], num_spatial_points)
        print(f"  已统一基准，目标 Shape: {TARGET_SHAPE}")
        return spatial_mask, point_indices, TARGET_SHAPE

    except Exception as e:
        print(f"错误: 无法生成统一掩码: {e}")
        return None, None, None

def build_matrix():
    """
    主函数，执行特征矩阵的构建流程。
    """
    start_time = time.time()
    print(f"=== 开始构建特征矩阵 ===")
    print(f"源目录: {FEATURES_SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("\n1. 初始化 FeatureLoader...")
    loader = FeatureLoader(FEATURES_SOURCE_DIR)

    spatial_mask, point_indices, TARGET_SHAPE = get_unified_mask_and_indices(loader)
    if spatial_mask is None:
        return

    selected_files = [f for f in loader.feature_files if not f.startswith(EXCLUDE_PREFIXES)]
    print(f"选择了 {len(selected_files)} 个特征用于构建矩阵。")

    print("\n2. 正在处理和对齐所有特征...")
    processed_features = []
    final_feature_names = []

    for i, fname in enumerate(sorted(selected_files)):
        data = loader.load_feature(fname)
        if data is None: continue

        processed_data = None
        
        # --- 核心逻辑：维度判断与转换 ---
        # Case A: 已经是统一后的点数据格式 (e.g., from a previous run)
        if data.ndim == 2 and data.shape == TARGET_SHAPE:
            processed_data = data
        # Case B: 原始点数据 (e.g., 1797, 2943)
        elif data.ndim == 2 and data.shape[0] == TARGET_SHAPE[0] and data.shape[1] > TARGET_SHAPE[1]:
            if point_indices is not None:
                processed_data = data[:, point_indices]
            else: # Should not happen if logic is correct
                processed_data = data[:, :TARGET_SHAPE[1]]
        # Case C: 全局时间特征 (1797,)
        elif data.ndim == 1 and data.shape[0] == TARGET_SHAPE[0]:
            processed_data = np.broadcast_to(data.reshape(TARGET_SHAPE[0], 1), TARGET_SHAPE)
        # Case D: 静态空间特征 (144, 256)
        elif data.ndim == 2 and data.shape == spatial_mask.shape:
            flattened_data = data[spatial_mask]
            processed_data = np.tile(flattened_data.reshape(1, TARGET_SHAPE[1]), (TARGET_SHAPE[0], 1))
        else:
            # Skip incompatible features
            if data.shape != TARGET_SHAPE:
                 print(f"  警告: 跳过维度不兼容的特征 '{fname}'，shape: {data.shape}")
            continue

        processed_features.append(processed_data)
        final_feature_names.append(fname.replace('.npy', ''))

    if not processed_features:
        print("错误: 没有可以处理的特征，程序终止。")
        return

    print(f"\n3. 成功处理并对齐了 {len(processed_features)} 个特征。")
    print("正在将所有特征堆叠为最终矩阵...")
    stacked_3d_matrix = np.stack(processed_features, axis=-1)
    final_X_matrix = stacked_3d_matrix.reshape(-1, len(final_feature_names))
    print(f"  最终特征矩阵 X shape: {final_X_matrix.shape}")

    print("正在处理目标变量 y...")
    target_data = loader.load_feature('target_points_valid')
    if target_data is not None:
        if point_indices is not None:
            final_y_vector = target_data[:, point_indices].flatten()
        else:
            final_y_vector = target_data.flatten()
        print(f"  最终目标向量 y shape: {final_y_vector.shape}")

        print("\n4. 正在保存最终产出物...")
        np.save(os.path.join(OUTPUT_DIR, "X_matrix.npy"), final_X_matrix)
        np.save(os.path.join(OUTPUT_DIR, "y_vector.npy"), final_y_vector)
        with open(os.path.join(OUTPUT_DIR, "final_feature_names.json"), 'w') as f:
            json.dump(final_feature_names, f, indent=2)
        print("所有文件保存成功！")

    end_time = time.time()
    print(f"\n=== 构建完成，总耗时: {end_time - start_time:.2f} 秒 ===")

if __name__ == "__main__":
    build_matrix()