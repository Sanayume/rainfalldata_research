#!/usr/bin/env python3
"""
修正的特征分析脚本：保持数据对应关系进行正确的相关性计算
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FEATURES_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/feature_analysis_corrected"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_features_synchronized(feature_names, sample_size=20000, random_seed=42):
    """
    同步加载特征数据，保持数据点的对应关系
    """
    print(f"Loading {len(feature_names)} features with synchronized sampling...")
    
    # 设置随机种子确保可重复性
    np.random.seed(random_seed)
    
    # 首先确定有效的样本索引
    total_samples = 5247240
    if sample_size >= total_samples:
        sample_indices = np.arange(total_samples)
    else:
        sample_indices = np.random.choice(total_samples, sample_size, replace=False)
        sample_indices = np.sort(sample_indices)  # 保持顺序
    
    print(f"Selected {len(sample_indices)} synchronized sample indices")
    
    feature_data = {}
    valid_features = []
    
    for i, fname in enumerate(feature_names):
        if i % 50 == 0:
            print(f"  Progress: {i+1}/{len(feature_names)}")
        
        try:
            fpath = os.path.join(FEATURES_DIR, fname)
            if not os.path.exists(fpath):
                print(f"    Skip {fname}: file not found")
                continue
            
            # 加载完整数据
            data = np.load(fpath)
            
            if data.shape[0] != total_samples:
                print(f"    Skip {fname}: shape mismatch {data.shape}")
                continue
            
            # 使用相同的样本索引进行采样
            sampled_data = data[sample_indices]
            
            # 数据质量检查
            if np.all(np.isnan(sampled_data)):
                print(f"    Skip {fname}: all NaN")
                continue
            
            # 处理无限值
            finite_mask = np.isfinite(sampled_data)
            if np.sum(finite_mask) < len(sampled_data) * 0.9:  # 至少90%有限值
                print(f"    Skip {fname}: too many infinite values ({np.sum(finite_mask)}/{len(sampled_data)})")
                continue
            
            # 用中位数替换NaN和inf
            if np.sum(~finite_mask) > 0:
                median_val = np.nanmedian(sampled_data[finite_mask])
                if np.isnan(median_val):
                    median_val = 0.0
                sampled_data = np.where(finite_mask, sampled_data, median_val)
            
            feature_data[fname] = sampled_data
            valid_features.append(fname)
            
        except Exception as e:
            print(f"    Failed to load {fname}: {e}")
            continue
    
    print(f"Successfully loaded {len(valid_features)} valid features")
    return feature_data, valid_features, sample_indices

def calculate_correlation_matrix_correct(feature_data, chunk_size=100):
    """
    正确计算相关性矩阵 - 使用同步采样的数据
    """
    feature_names = list(feature_data.keys())
    n_features = len(feature_names)
    
    print(f"Calculating {n_features}x{n_features} correlation matrix (corrected method)...")
    
    # 初始化相关性矩阵
    corr_matrix = np.eye(n_features)
    
    # 分块计算以节省内存
    for i in range(0, n_features, chunk_size):
        i_end = min(i + chunk_size, n_features)
        print(f"  Processing chunk [{i}:{i_end}] of {n_features}")
        
        for ii in range(i, i_end):
            data1 = feature_data[feature_names[ii]]
            
            for jj in range(ii + 1, n_features):  # 只计算上三角矩阵
                data2 = feature_data[feature_names[jj]]
                
                try:
                    # 检查数据变异性
                    if np.std(data1) < 1e-10 or np.std(data2) < 1e-10:
                        corr_val = 0.0
                    else:
                        corr_val, _ = pearsonr(data1, data2)
                        if np.isnan(corr_val):
                            corr_val = 0.0
                    
                    corr_matrix[ii, jj] = corr_val
                    corr_matrix[jj, ii] = corr_val
                    
                except Exception:
                    corr_matrix[ii, jj] = 0.0
                    corr_matrix[jj, ii] = 0.0
    
    return corr_matrix

def verify_correlations_manually(feature_data):
    """
    手动验证一些预期的相关性
    """
    print("Manual verification of expected correlations:")
    print("=" * 50)
    
    verification_pairs = [
        ('raw_points_CHIRPS.npy', 'raw_points_CHIRPS_valid.npy'),
        ('multi_product_mean.npy', 'multi_product_max.npy'),
        ('multi_product_mean.npy', 'multi_product_min.npy'), 
        ('raw_points_CHIRPS.npy', 'lag_1_points_CHIRPS.npy'),
        ('target_points_valid.npy', 'multi_product_mean.npy')
    ]
    
    verification_results = []
    
    for feat1, feat2 in verification_pairs:
        if feat1 in feature_data and feat2 in feature_data:
            data1 = feature_data[feat1]
            data2 = feature_data[feat2]
            
            try:
                corr, p_val = pearsonr(data1, data2)
                print(f"{feat1.replace('.npy', '')[:30]}")
                print(f"vs {feat2.replace('.npy', '')[:30]}")
                print(f"Correlation: {corr:.6f} (p={p_val:.2e})")
                print(f"Sample size: {len(data1)}")
                print()
                
                verification_results.append({
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': corr,
                    'p_value': p_val
                })
                
            except Exception as e:
                print(f"Error: {e}")
    
    return verification_results

def analyze_correlation_distribution(corr_matrix, feature_names):
    """
    分析相关性分布
    """
    print("Analyzing correlation distribution...")
    
    # 提取上三角矩阵（排除对角线）
    upper_triangle = np.triu(corr_matrix, k=1)
    correlations = upper_triangle[upper_triangle != 0]
    correlations = correlations[np.isfinite(correlations)]
    
    # 统计信息
    stats = {
        'count': len(correlations),
        'mean': np.mean(correlations),
        'std': np.std(correlations),
        'min': np.min(correlations),
        'max': np.max(correlations),
        'abs_max': np.max(np.abs(correlations))
    }
    
    # 高相关性分析
    high_corr_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    high_corr_counts = {}
    
    for threshold in high_corr_thresholds:
        count = np.sum(np.abs(correlations) > threshold)
        high_corr_counts[threshold] = count
        stats[f'abs_corr_gt_{threshold}'] = count
    
    print(f"Correlation statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    return stats, correlations

def find_highly_correlated_pairs(corr_matrix, feature_names, threshold=0.7):
    """
    找出高度相关的特征对
    """
    print(f"Finding feature pairs with |correlation| > {threshold}...")
    
    high_corr_pairs = []
    
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            corr_val = corr_matrix[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append({
                    'feature1': feature_names[i].replace('.npy', ''),
                    'feature2': feature_names[j].replace('.npy', ''),
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val)
                })
    
    # 按绝对相关性排序
    high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    print(f"Found {len(high_corr_pairs)} highly correlated pairs")
    return high_corr_pairs

def create_corrected_visualizations(corr_matrix, feature_names, correlations, high_corr_pairs):
    """
    创建修正后的可视化
    """
    print("Creating corrected visualizations...")
    
    # 1. 相关性分布直方图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(correlations, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.5)
    plt.axvline(x=np.mean(correlations), color='green', linestyle='--', 
                label=f'Mean: {np.mean(correlations):.4f}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Correlations')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. 绝对相关性分布
    plt.subplot(2, 2, 2)
    abs_correlations = np.abs(correlations)
    plt.hist(abs_correlations, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(x=np.mean(abs_correlations), color='red', linestyle='--',
                label=f'Mean: {np.mean(abs_correlations):.4f}')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Absolute Correlations')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 3. 高相关性阈值统计
    plt.subplot(2, 2, 3)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    counts = [np.sum(abs_correlations > t) for t in thresholds]
    plt.bar(thresholds, counts, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Correlation Threshold')
    plt.ylabel('Number of Pairs')
    plt.title('High Correlation Pairs by Threshold')
    plt.grid(axis='y', alpha=0.3)
    
    # 4. 采样相关性矩阵热力图
    plt.subplot(2, 2, 4)
    if len(feature_names) > 50:
        # 随机选择50个特征进行显示
        indices = np.random.choice(len(feature_names), 50, replace=False)
        sample_matrix = corr_matrix[np.ix_(indices, indices)]
        sample_names = [feature_names[i].replace('.npy', '')[:10] for i in indices]
    else:
        sample_matrix = corr_matrix
        sample_names = [f.replace('.npy', '')[:10] for f in feature_names]
    
    im = plt.imshow(sample_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, shrink=0.8)
    plt.title(f'Correlation Matrix (Sample of {len(sample_names)} features)')
    plt.xticks(range(len(sample_names)), sample_names, rotation=45, ha='right')
    plt.yticks(range(len(sample_names)), sample_names)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'corrected_correlation_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 高相关性特征对条形图
    if high_corr_pairs:
        plt.figure(figsize=(14, max(8, len(high_corr_pairs[:20]) * 0.5)))
        top_pairs = high_corr_pairs[:20]
        
        correlations_plot = [pair['correlation'] for pair in top_pairs]
        labels = [f"{pair['feature1'][:15]}...\nvs\n{pair['feature2'][:15]}..." 
                 for pair in top_pairs]
        
        colors = ['red' if abs(c) > 0.9 else 'orange' if abs(c) > 0.7 else 'yellow' 
                 for c in correlations_plot]
        
        y_pos = range(len(top_pairs))
        plt.barh(y_pos, correlations_plot, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(y_pos, labels)
        plt.xlabel('Correlation Coefficient')
        plt.title(f'Top {len(top_pairs)} Highly Correlated Feature Pairs')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'high_correlation_pairs.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def generate_corrected_report(stats, verification_results, high_corr_pairs, n_features):
    """
    生成修正后的分析报告
    """
    report = []
    report.append("# Corrected Feature Correlation Analysis Report")
    report.append("# Yangtze River Basin Rainfall Prediction\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Analysis method: Synchronized sampling (preserving data correspondence)\n")
    
    report.append("## 1. Overview")
    report.append(f"- Total features analyzed: {n_features}")
    report.append(f"- Sample size: {stats['count'] * 2 + n_features} total correlations")
    report.append(f"- Unique feature pairs: {stats['count']}\n")
    
    report.append("## 2. Correlation Statistics")
    report.append(f"- Mean correlation: {stats['mean']:.6f}")
    report.append(f"- Standard deviation: {stats['std']:.6f}")
    report.append(f"- Minimum correlation: {stats['min']:.6f}")
    report.append(f"- Maximum correlation: {stats['max']:.6f}")
    report.append(f"- Maximum absolute correlation: {stats['abs_max']:.6f}\n")
    
    report.append("## 3. High Correlation Analysis")
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    for threshold in thresholds:
        key = f'abs_corr_gt_{threshold}'
        if key in stats:
            report.append(f"- Pairs with |r| > {threshold}: {stats[key]}")
    report.append("")
    
    if verification_results:
        report.append("## 4. Manual Verification Results")
        for i, result in enumerate(verification_results, 1):
            report.append(f"{i}. {result['feature1'].replace('.npy', '')} vs {result['feature2'].replace('.npy', '')}")
            report.append(f"   Correlation: {result['correlation']:.6f} (p={result['p_value']:.2e})")
        report.append("")
    
    if high_corr_pairs:
        report.append("## 5. Top Highly Correlated Feature Pairs")
        for i, pair in enumerate(high_corr_pairs[:10], 1):
            report.append(f"{i}. {pair['feature1']} vs {pair['feature2']}")
            report.append(f"   Correlation: {pair['correlation']:.6f}")
        report.append("")
    
    report.append("## 6. Recommendations")
    if stats['abs_max'] > 0.95:
        report.append("- Consider removing highly redundant features (|r| > 0.95)")
    if stats.get('abs_corr_gt_0.8', 0) > 20:
        report.append("- Review features with |r| > 0.8 for potential redundancy")
    if stats['abs_max'] < 0.1:
        report.append("- Features show good independence - no obvious redundancy")
    
    report_text = '\n'.join(report)
    
    with open(os.path.join(OUTPUT_DIR, 'corrected_analysis_report.md'), 'w') as f:
        f.write(report_text)
    
    return report_text

def main():
    """
    主函数 - 执行修正后的特征分析
    """
    print("=" * 80)
    print("CORRECTED FEATURE CORRELATION ANALYSIS")
    print("Key improvement: Synchronized sampling preserving data correspondence")
    print("=" * 80)
    
    # 1. 加载特征信息
    print("\n1. Loading feature information...")
    csv_path = os.path.join(FEATURES_DIR, "features_list.csv")
    features_df = pd.read_csv(csv_path)
    flattened_features = features_df[features_df['shape'] == '(5247240,)']
    feature_names = flattened_features['feature_file_name'].tolist()
    
    print(f"Found {len(feature_names)} flattened features")
    
    # 2. 同步加载特征数据
    print("\n2. Loading features with synchronized sampling...")
    feature_data, valid_features, sample_indices = load_features_synchronized(
        feature_names, sample_size=20000, random_seed=42
    )
    
    if len(valid_features) < 10:
        print("ERROR: Too few valid features for meaningful analysis")
        return
    
    # 3. 手动验证预期相关性
    print("\n3. Manual verification of expected correlations...")
    verification_results = verify_correlations_manually(feature_data)
    
    # 4. 计算完整相关性矩阵
    print("\n4. Calculating corrected correlation matrix...")
    corr_matrix = calculate_correlation_matrix_correct(feature_data, chunk_size=50)
    
    # 5. 分析相关性分布
    print("\n5. Analyzing correlation distribution...")
    stats, correlations = analyze_correlation_distribution(corr_matrix, valid_features)
    
    # 6. 找出高相关性特征对
    print("\n6. Finding highly correlated feature pairs...")
    high_corr_pairs = find_highly_correlated_pairs(corr_matrix, valid_features, threshold=0.5)
    
    # 7. 创建可视化
    print("\n7. Creating visualizations...")
    create_corrected_visualizations(corr_matrix, valid_features, correlations, high_corr_pairs)
    
    # 8. 生成报告
    print("\n8. Generating corrected analysis report...")
    report = generate_corrected_report(stats, verification_results, high_corr_pairs, len(valid_features))
    
    # 9. 保存结果
    print("\n9. Saving results...")
    
    # 保存数据
    np.save(os.path.join(OUTPUT_DIR, 'corrected_correlation_matrix.npy'), corr_matrix)
    np.save(os.path.join(OUTPUT_DIR, 'sample_indices.npy'), sample_indices)
    
    with open(os.path.join(OUTPUT_DIR, 'valid_feature_names.txt'), 'w') as f:
        for name in valid_features:
            f.write(name + '\n')
    
    # 保存统计信息
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'correlation_statistics.csv'), index=False)
    
    if verification_results:
        verification_df = pd.DataFrame(verification_results)
        verification_df.to_csv(os.path.join(OUTPUT_DIR, 'verification_results.csv'), index=False)
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df.to_csv(os.path.join(OUTPUT_DIR, 'high_correlation_pairs.csv'), index=False)
    
    print(f"\nCorrected analysis completed!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Key findings:")
    print(f"  - Analyzed {len(valid_features)} features")
    print(f"  - Max absolute correlation: {stats['abs_max']:.6f}")
    print(f"  - High correlation pairs (|r| > 0.5): {len(high_corr_pairs)}")
    
    return feature_data, corr_matrix, stats, report

if __name__ == "__main__":
    feature_data, corr_matrix, stats, report = main()