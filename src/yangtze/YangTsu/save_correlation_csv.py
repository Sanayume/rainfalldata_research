#!/usr/bin/env python3
"""
保存相关性矩阵为CSV格式，便于查看和分析
"""

import os
import numpy as np
import pandas as pd

INPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/feature_analysis_corrected"
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/feature_analysis_corrected"

def save_correlation_matrix_csv():
    """
    将相关性矩阵保存为CSV格式
    """
    print("Loading correlation matrix and feature names...")
    
    # 加载数据
    corr_matrix_path = os.path.join(INPUT_DIR, 'corrected_correlation_matrix.npy')
    feature_names_path = os.path.join(INPUT_DIR, 'valid_feature_names.txt')
    
    if not os.path.exists(corr_matrix_path):
        print("Error: Correlation matrix file not found!")
        return
    
    if not os.path.exists(feature_names_path):
        print("Error: Feature names file not found!")
        return
    
    # 加载相关性矩阵
    corr_matrix = np.load(corr_matrix_path)
    print(f"Loaded correlation matrix: {corr_matrix.shape}")
    
    # 加载特征名称
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip().replace('.npy', '') for line in f.readlines()]
    
    print(f"Loaded {len(feature_names)} feature names")
    
    # 创建DataFrame
    print("Creating correlation DataFrame...")
    corr_df = pd.DataFrame(
        corr_matrix, 
        index=feature_names, 
        columns=feature_names
    )
    
    # 保存完整矩阵CSV（可能很大）
    print("Saving full correlation matrix to CSV...")
    full_csv_path = os.path.join(OUTPUT_DIR, 'correlation_matrix_full.csv')
    corr_df.to_csv(full_csv_path, float_format='%.6f')
    
    file_size_mb = os.path.getsize(full_csv_path) / 1024 / 1024
    print(f"Full matrix saved: {full_csv_path} ({file_size_mb:.1f} MB)")
    
    return corr_df

def save_high_correlation_pairs_detailed():
    """
    保存高相关性特征对的详细信息
    """
    print("Creating detailed high correlation pairs...")
    
    # 加载数据
    corr_matrix_path = os.path.join(INPUT_DIR, 'corrected_correlation_matrix.npy')
    feature_names_path = os.path.join(INPUT_DIR, 'valid_feature_names.txt')
    
    corr_matrix = np.load(corr_matrix_path)
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip().replace('.npy', '') for line in f.readlines()]
    
    # 创建高相关性对列表
    high_corr_data = []
    
    print("Extracting high correlation pairs...")
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    for i in range(len(feature_names)):
        if i % 50 == 0:
            print(f"  Progress: {i+1}/{len(feature_names)}")
        
        for j in range(i + 1, len(feature_names)):
            corr_val = corr_matrix[i, j]
            abs_corr = abs(corr_val)
            
            # 只保存绝对相关性 > 0.3 的对
            if abs_corr > 0.3:
                # 确定相关性级别
                level = "moderate"
                for threshold in sorted(thresholds, reverse=True):
                    if abs_corr > threshold:
                        level = f"very_high_{threshold}" if threshold >= 0.9 else f"high_{threshold}"
                        break
                
                high_corr_data.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': corr_val,
                    'abs_correlation': abs_corr,
                    'correlation_level': level,
                    'feature1_category': categorize_feature(feature_names[i]),
                    'feature2_category': categorize_feature(feature_names[j])
                })
    
    # 创建DataFrame并排序
    high_corr_df = pd.DataFrame(high_corr_data)
    high_corr_df = high_corr_df.sort_values('abs_correlation', ascending=False)
    
    # 保存不同阈值的文件
    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95]:
        subset_df = high_corr_df[high_corr_df['abs_correlation'] > threshold]
        if len(subset_df) > 0:
            filename = f'high_correlation_pairs_gt_{threshold}.csv'
            filepath = os.path.join(OUTPUT_DIR, filename)
            subset_df.to_csv(filepath, index=False, float_format='%.6f')
            print(f"Saved {len(subset_df)} pairs with |r| > {threshold}: {filename}")
    
    # 保存所有moderate以上的相关性
    moderate_path = os.path.join(OUTPUT_DIR, 'correlation_pairs_gt_0.3.csv')
    high_corr_df.to_csv(moderate_path, index=False, float_format='%.6f')
    print(f"Saved {len(high_corr_df)} pairs with |r| > 0.3: correlation_pairs_gt_0.3.csv")
    
    return high_corr_df

def categorize_feature(feature_name):
    """
    简单的特征分类
    """
    fname = feature_name.lower()
    
    if any(x in fname for x in ['raw_', 'target_']):
        return 'basic'
    elif any(x in fname for x in ['cos_', 'sin_', 'day_', 'month_', 'season_', 'time']):
        return 'temporal'
    elif any(x in fname for x in ['correlation_', 'multi_product_', 'consistency', 'disagreement']):
        return 'multi_product'
    elif 'lag_' in fname:
        return 'lag'
    elif 'spatial_' in fname:
        return 'spatial'
    elif any(x in fname for x in ['quantile_', 'anomaly_', 'extreme_', 'rolling_', 'intensity_']):
        return 'statistical'
    elif 'interaction_' in fname:
        return 'interaction'
    else:
        return 'other'

def create_correlation_summary():
    """
    创建相关性矩阵的摘要统计
    """
    print("Creating correlation summary...")
    
    # 加载数据
    corr_matrix_path = os.path.join(INPUT_DIR, 'corrected_correlation_matrix.npy')
    feature_names_path = os.path.join(INPUT_DIR, 'valid_feature_names.txt')
    
    corr_matrix = np.load(corr_matrix_path)
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip().replace('.npy', '') for line in f.readlines()]
    
    # 计算每个特征的平均绝对相关性
    feature_summary = []
    
    for i, fname in enumerate(feature_names):
        if i % 50 == 0:
            print(f"  Processing feature {i+1}/{len(feature_names)}")
        
        # 该特征与其他所有特征的相关性（排除自己）
        correlations = np.concatenate([corr_matrix[i, :i], corr_matrix[i, i+1:]])
        abs_correlations = np.abs(correlations)
        
        feature_summary.append({
            'feature_name': fname,
            'feature_category': categorize_feature(fname),
            'mean_abs_correlation': np.mean(abs_correlations),
            'max_abs_correlation': np.max(abs_correlations),
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations),
            'high_corr_count_gt_0.5': np.sum(abs_correlations > 0.5),
            'high_corr_count_gt_0.7': np.sum(abs_correlations > 0.7),
            'high_corr_count_gt_0.9': np.sum(abs_correlations > 0.9),
            'redundancy_score': np.sum(abs_correlations > 0.95)  # 冗余度评分
        })
    
    # 创建DataFrame并排序
    summary_df = pd.DataFrame(feature_summary)
    
    # 按平均绝对相关性排序
    summary_sorted = summary_df.sort_values('mean_abs_correlation', ascending=False)
    summary_path = os.path.join(OUTPUT_DIR, 'feature_correlation_summary.csv')
    summary_sorted.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"Feature summary saved: feature_correlation_summary.csv")
    
    # 按冗余度排序
    redundancy_sorted = summary_df.sort_values('redundancy_score', ascending=False)
    redundancy_path = os.path.join(OUTPUT_DIR, 'feature_redundancy_ranking.csv')
    redundancy_sorted.to_csv(redundancy_path, index=False, float_format='%.6f')
    print(f"Redundancy ranking saved: feature_redundancy_ranking.csv")
    
    return summary_df

def save_correlation_matrix_chunks():
    """
    将大矩阵分块保存为多个较小的CSV文件
    """
    print("Saving correlation matrix in chunks...")
    
    # 加载数据
    corr_matrix_path = os.path.join(INPUT_DIR, 'corrected_correlation_matrix.npy')
    feature_names_path = os.path.join(INPUT_DIR, 'valid_feature_names.txt')
    
    corr_matrix = np.load(corr_matrix_path)
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip().replace('.npy', '') for line in f.readlines()]
    
    # 创建chunks目录
    chunks_dir = os.path.join(OUTPUT_DIR, 'correlation_matrix_chunks')
    os.makedirs(chunks_dir, exist_ok=True)
    
    # 分块大小
    chunk_size = 100
    n_features = len(feature_names)
    
    print(f"Splitting {n_features}x{n_features} matrix into {chunk_size}x{chunk_size} chunks...")
    
    chunk_info = []
    
    for i in range(0, n_features, chunk_size):
        for j in range(0, n_features, chunk_size):
            i_end = min(i + chunk_size, n_features)
            j_end = min(j + chunk_size, n_features)
            
            # 提取子矩阵
            chunk_matrix = corr_matrix[i:i_end, j:j_end]
            chunk_row_names = feature_names[i:i_end]
            chunk_col_names = feature_names[j:j_end]
            
            # 创建DataFrame
            chunk_df = pd.DataFrame(
                chunk_matrix,
                index=chunk_row_names,
                columns=chunk_col_names
            )
            
            # 保存
            chunk_filename = f'correlation_chunk_{i}_{j}_to_{i_end-1}_{j_end-1}.csv'
            chunk_path = os.path.join(chunks_dir, chunk_filename)
            chunk_df.to_csv(chunk_path, float_format='%.6f')
            
            chunk_info.append({
                'filename': chunk_filename,
                'row_start': i,
                'row_end': i_end - 1,
                'col_start': j,
                'col_end': j_end - 1,
                'shape': f"({i_end-i}, {j_end-j})"
            })
            
            print(f"  Saved chunk [{i}:{i_end}, {j}:{j_end}] -> {chunk_filename}")
    
    # 保存chunk索引
    chunk_index_df = pd.DataFrame(chunk_info)
    chunk_index_path = os.path.join(chunks_dir, 'chunk_index.csv')
    chunk_index_df.to_csv(chunk_index_path, index=False)
    
    print(f"Saved {len(chunk_info)} chunks to: {chunks_dir}")
    print(f"Chunk index saved: chunk_index.csv")

def main():
    """
    主函数
    """
    print("=" * 80)
    print("SAVING CORRELATION MATRIX IN MULTIPLE FORMATS")
    print("=" * 80)
    
    try:
        # 1. 保存完整相关性矩阵CSV
        print("\n1. Saving full correlation matrix as CSV...")
        corr_df = save_correlation_matrix_csv()
        
        # 2. 保存高相关性特征对详细信息
        print("\n2. Saving detailed high correlation pairs...")
        high_corr_df = save_high_correlation_pairs_detailed()
        
        # 3. 创建特征相关性摘要
        print("\n3. Creating feature correlation summary...")
        summary_df = create_correlation_summary()
        
        # 4. 分块保存矩阵
        print("\n4. Saving matrix in chunks...")
        save_correlation_matrix_chunks()
        
        print("\n" + "=" * 80)
        print("SUMMARY OF SAVED FILES:")
        print("=" * 80)
        
        files_created = [
            "correlation_matrix_full.csv - Complete 415x415 correlation matrix",
            "high_correlation_pairs_gt_0.95.csv - Highly redundant feature pairs", 
            "high_correlation_pairs_gt_0.9.csv - Very high correlation pairs",
            "high_correlation_pairs_gt_0.8.csv - High correlation pairs",
            "high_correlation_pairs_gt_0.7.csv - Moderately high correlation pairs",
            "high_correlation_pairs_gt_0.5.csv - Moderate correlation pairs",
            "correlation_pairs_gt_0.3.csv - All meaningful correlations",
            "feature_correlation_summary.csv - Per-feature correlation statistics",
            "feature_redundancy_ranking.csv - Features ranked by redundancy",
            "correlation_matrix_chunks/ - Matrix split into 100x100 chunks"
        ]
        
        for file_desc in files_created:
            print(f"✓ {file_desc}")
        
        print(f"\nAll files saved to: {OUTPUT_DIR}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    main()