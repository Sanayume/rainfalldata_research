#!/usr/bin/env python3
"""
验证特征分析结果的正确性
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

FEATURES_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/feature_verification"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_specific_features_for_verification():
    """加载一些应该相关的特征进行验证"""
    print("Loading specific features for verification...")
    
    # 选择一些明显应该相关的特征
    test_features = [
        'raw_points_CHIRPS.npy',
        'raw_points_CHIRPS_valid.npy',
        'lag_1_points_CHIRPS.npy',
        'multi_product_mean.npy',
        'multi_product_max.npy',
        'multi_product_min.npy',
        'target_points_valid.npy',
        'correlation_CHIRPS_GSMAP.npy',
        'correlation_CHIRPS_IMERG.npy'
    ]
    
    feature_data = {}
    
    for fname in test_features:
        try:
            fpath = os.path.join(FEATURES_DIR, fname)
            if os.path.exists(fpath):
                data = np.load(fpath)
                print(f"Loaded {fname}: shape={data.shape}, dtype={data.dtype}")
                
                # 随机采样
                sample_size = min(5000, data.shape[0])
                indices = np.random.choice(data.shape[0], sample_size, replace=False)
                sampled_data = data[indices]
                
                # 基本统计
                print(f"  Mean: {np.nanmean(sampled_data):.6f}")
                print(f"  Std: {np.nanstd(sampled_data):.6f}")
                print(f"  Min: {np.nanmin(sampled_data):.6f}")
                print(f"  Max: {np.nanmax(sampled_data):.6f}")
                print(f"  Non-zero ratio: {np.mean(sampled_data != 0):.3f}")
                print(f"  NaN ratio: {np.mean(np.isnan(sampled_data)):.3f}")
                print()
                
                feature_data[fname] = sampled_data
                
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
    
    return feature_data

def manual_correlation_check(feature_data):
    """手动计算一些相关性进行验证"""
    print("Manual correlation verification:")
    print("=" * 50)
    
    feature_names = list(feature_data.keys())
    
    # 检查一些预期相关的特征对
    expected_correlations = [
        ('raw_points_CHIRPS.npy', 'raw_points_CHIRPS_valid.npy'),
        ('raw_points_CHIRPS.npy', 'lag_1_points_CHIRPS.npy'),
        ('multi_product_mean.npy', 'multi_product_max.npy'),
        ('multi_product_mean.npy', 'multi_product_min.npy'),
        ('raw_points_CHIRPS.npy', 'target_points_valid.npy')
    ]
    
    results = []
    
    for feat1, feat2 in expected_correlations:
        if feat1 in feature_data and feat2 in feature_data:
            data1 = feature_data[feat1]
            data2 = feature_data[feat2]
            
            # 移除NaN值
            mask = ~(np.isnan(data1) | np.isnan(data2))
            clean_data1 = data1[mask]
            clean_data2 = data2[mask]
            
            if len(clean_data1) > 100:
                try:
                    corr, p_value = pearsonr(clean_data1, clean_data2)
                    print(f"{feat1.replace('.npy', '')}")
                    print(f"vs {feat2.replace('.npy', '')}")
                    print(f"Correlation: {corr:.6f} (p={p_value:.6f})")
                    print(f"Valid samples: {len(clean_data1)}")
                    print()
                    
                    results.append({
                        'feature1': feat1,
                        'feature2': feat2,
                        'correlation': corr,
                        'p_value': p_value,
                        'n_samples': len(clean_data1)
                    })
                    
                except Exception as e:
                    print(f"Error calculating correlation: {e}")
    
    return results

def check_data_preprocessing():
    """检查数据预处理是否正确"""
    print("Checking data preprocessing...")
    
    # 加载一个简单特征检查
    fname = 'multi_product_mean.npy'
    fpath = os.path.join(FEATURES_DIR, fname)
    
    if os.path.exists(fpath):
        data = np.load(fpath)
        print(f"File: {fname}")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Memory usage: {data.nbytes / 1024 / 1024:.2f} MB")
        
        # 检查前100个值
        print(f"First 10 values: {data[:10]}")
        print(f"Last 10 values: {data[-10:]}")
        
        # 检查异常值
        finite_mask = np.isfinite(data)
        print(f"Finite values: {np.sum(finite_mask)} / {len(data)} ({np.sum(finite_mask)/len(data)*100:.1f}%)")
        
        if np.sum(finite_mask) > 0:
            finite_data = data[finite_mask]
            print(f"Min (finite): {np.min(finite_data)}")
            print(f"Max (finite): {np.max(finite_data)}")
            print(f"Mean (finite): {np.mean(finite_data):.6f}")
            print(f"Std (finite): {np.std(finite_data):.6f}")
        
        # 检查是否有重复值
        unique_values = len(np.unique(data[finite_mask]))
        print(f"Unique values: {unique_values}")
        
        return True
    else:
        print(f"File {fname} not found")
        return False

def load_saved_correlation_matrix():
    """加载并检查保存的相关性矩阵"""
    matrix_path = "/mnt/f/rainfalldata/results/yangtze/feature_analysis_full/correlation_matrix.npy"
    names_path = "/mnt/f/rainfalldata/results/yangtze/feature_analysis_full/feature_names.txt"
    
    if os.path.exists(matrix_path) and os.path.exists(names_path):
        print("Checking saved correlation matrix...")
        
        corr_matrix = np.load(matrix_path)
        with open(names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Matrix shape: {corr_matrix.shape}")
        print(f"Feature names count: {len(feature_names)}")
        
        # 检查对角线
        diagonal = np.diag(corr_matrix)
        print(f"Diagonal min: {np.min(diagonal)}")
        print(f"Diagonal max: {np.max(diagonal)}")
        print(f"Non-1 diagonal values: {np.sum(diagonal != 1.0)}")
        
        # 检查非对角线元素
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diagonal = corr_matrix[mask]
        finite_off_diag = off_diagonal[np.isfinite(off_diagonal)]
        
        print(f"Off-diagonal finite values: {len(finite_off_diag)} / {len(off_diagonal)}")
        if len(finite_off_diag) > 0:
            print(f"Off-diagonal min: {np.min(finite_off_diag)}")
            print(f"Off-diagonal max: {np.max(finite_off_diag)}")
            print(f"Off-diagonal mean: {np.mean(finite_off_diag):.6f}")
            print(f"Off-diagonal std: {np.std(finite_off_diag):.6f}")
            
            # 检查高相关性
            high_corr = finite_off_diag[np.abs(finite_off_diag) > 0.5]
            print(f"Correlations > 0.5: {len(high_corr)}")
            if len(high_corr) > 0:
                print(f"Highest correlations: {np.sort(np.abs(high_corr))[-10:]}")
        
        return corr_matrix, feature_names
    else:
        print("Saved correlation matrix not found")
        return None, None

def main():
    """主函数"""
    print("=" * 60)
    print("VERIFICATION OF FEATURE ANALYSIS RESULTS")
    print("=" * 60)
    
    # 1. 检查数据预处理
    print("\n1. Data preprocessing check:")
    print("-" * 30)
    preprocessing_ok = check_data_preprocessing()
    
    # 2. 加载特定特征进行验证
    print("\n2. Loading specific features:")
    print("-" * 30)
    feature_data = load_specific_features_for_verification()
    
    # 3. 手动计算相关性
    print("\n3. Manual correlation check:")
    print("-" * 30)
    manual_results = manual_correlation_check(feature_data)
    
    # 4. 检查保存的相关性矩阵
    print("\n4. Saved correlation matrix check:")
    print("-" * 30)
    corr_matrix, feature_names = load_saved_correlation_matrix()
    
    # 5. 总结
    print("\n5. SUMMARY:")
    print("-" * 30)
    print(f"Preprocessing check: {'PASS' if preprocessing_ok else 'FAIL'}")
    print(f"Features loaded: {len(feature_data)}")
    print(f"Manual correlations calculated: {len(manual_results)}")
    
    if manual_results:
        correlations = [r['correlation'] for r in manual_results]
        print(f"Manual correlation range: {min(correlations):.6f} to {max(correlations):.6f}")
        
        # 找出最高相关性
        max_corr_idx = np.argmax([abs(c) for c in correlations])
        max_result = manual_results[max_corr_idx]
        print(f"Highest manual correlation: {max_result['correlation']:.6f}")
        print(f"  Between: {max_result['feature1']} and {max_result['feature2']}")
    
    # 保存验证结果
    if manual_results:
        df = pd.DataFrame(manual_results)
        df.to_csv(os.path.join(OUTPUT_DIR, 'manual_correlation_verification.csv'), index=False)
        print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()