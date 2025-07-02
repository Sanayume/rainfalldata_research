#!/usr/bin/env python3
"""
完整特征分析脚本：分析所有415个已展平特征的相关性和统计属性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置字体 - 使用系统默认英文字体避免中文显示问题
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FEATURES_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/feature_analysis_full"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_flattened_features():
    """加载所有已展平特征的信息"""
    csv_path = os.path.join(FEATURES_DIR, "features_list.csv")
    df = pd.read_csv(csv_path)
    
    # 筛选出shape为(5247240,)的可比较特征
    flattened_features = df[df['shape'] == '(5247240,)'].copy()
    
    print(f"Total features: {len(df)}")
    print(f"Flattened features: {len(flattened_features)}")
    print(f"Non-flattened features: {len(df) - len(flattened_features)}")
    
    return flattened_features

def load_feature_batch(feature_names, batch_size=50, sample_size=10000):
    """批量加载特征数据进行分析"""
    print(f"Loading {len(feature_names)} features in batches...")
    
    all_feature_data = {}
    valid_features = []
    
    for i in range(0, len(feature_names), batch_size):
        batch = feature_names[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(feature_names)-1)//batch_size + 1}: {len(batch)} features")
        
        for j, fname in enumerate(batch):
            if (i + j) % 20 == 0:
                print(f"    Progress: {i+j+1}/{len(feature_names)}")
            
            try:
                fpath = os.path.join(FEATURES_DIR, fname)
                data = np.load(fpath)
                
                if data.shape[0] != 5247240:
                    print(f"    Skip {fname}: shape mismatch {data.shape}")
                    continue
                
                # 随机采样以减少内存使用
                indices = np.random.choice(data.shape[0], min(sample_size, data.shape[0]), replace=False)
                sampled_data = data[indices]
                
                # 检查数据质量
                if np.all(np.isnan(sampled_data)):
                    print(f"    Skip {fname}: all NaN")
                    continue
                
                # 替换 inf 和 -inf
                sampled_data = np.where(np.isinf(sampled_data), np.nan, sampled_data)
                
                all_feature_data[fname] = sampled_data
                valid_features.append(fname)
                
            except Exception as e:
                print(f"    Failed to load {fname}: {e}")
                continue
    
    print(f"Successfully loaded {len(valid_features)} valid features")
    return all_feature_data, valid_features

def calculate_correlation_matrix_chunked(feature_data, chunk_size=100):
    """分块计算大型相关性矩阵"""
    feature_names = list(feature_data.keys())
    n_features = len(feature_names)
    
    print(f"Calculating {n_features}x{n_features} correlation matrix...")
    
    # 预处理数据：移除NaN
    processed_data = {}
    for fname in feature_names:
        data = feature_data[fname]
        # 用中位数填充NaN值
        median_val = np.nanmedian(data)
        if np.isnan(median_val):
            median_val = 0.0
        processed_data[fname] = np.where(np.isnan(data), median_val, data)
    
    # 初始化相关性矩阵
    corr_matrix = np.eye(n_features)  # 对角线设为1
    
    # 分块计算
    for i in range(0, n_features, chunk_size):
        for j in range(i, n_features, chunk_size):
            i_end = min(i + chunk_size, n_features)
            j_end = min(j + chunk_size, n_features)
            
            print(f"  Computing chunk [{i}:{i_end}, {j}:{j_end}]")
            
            for ii in range(i, i_end):
                for jj in range(max(j, ii), j_end):  # 只计算上三角矩阵
                    try:
                        data1 = processed_data[feature_names[ii]]
                        data2 = processed_data[feature_names[jj]]
                        
                        if ii == jj:
                            corr_matrix[ii, jj] = 1.0
                        else:
                            # 检查数据变异性
                            if np.std(data1) < 1e-10 or np.std(data2) < 1e-10:
                                corr_matrix[ii, jj] = 0.0
                                corr_matrix[jj, ii] = 0.0
                            else:
                                corr, _ = pearsonr(data1, data2)
                                if not np.isnan(corr):
                                    corr_matrix[ii, jj] = corr
                                    corr_matrix[jj, ii] = corr
                                else:
                                    corr_matrix[ii, jj] = 0.0
                                    corr_matrix[jj, ii] = 0.0
                    except Exception as e:
                        corr_matrix[ii, jj] = 0.0
                        corr_matrix[jj, ii] = 0.0
                        continue
    
    return corr_matrix, feature_names

def calculate_basic_statistics(feature_data):
    """计算基本统计量"""
    stats_list = []
    
    print("Calculating basic statistics...")
    for i, (fname, data) in enumerate(feature_data.items()):
        if i % 50 == 0:
            print(f"  Progress: {i+1}/{len(feature_data)}")
        
        try:
            # 移除inf值
            clean_data = data[np.isfinite(data)]
            
            stats = {
                'feature': fname.replace('.npy', ''),
                'mean': np.nanmean(clean_data) if len(clean_data) > 0 else 0,
                'std': np.nanstd(clean_data) if len(clean_data) > 0 else 0,
                'min': np.nanmin(clean_data) if len(clean_data) > 0 else 0,
                'max': np.nanmax(clean_data) if len(clean_data) > 0 else 0,
                'median': np.nanmedian(clean_data) if len(clean_data) > 0 else 0,
                'q25': np.nanpercentile(clean_data, 25) if len(clean_data) > 0 else 0,
                'q75': np.nanpercentile(clean_data, 75) if len(clean_data) > 0 else 0,
                'variance': np.nanvar(clean_data) if len(clean_data) > 0 else 0,
                'non_zero_ratio': np.mean(clean_data != 0) if len(clean_data) > 0 else 0,
                'finite_ratio': len(clean_data) / len(data) if len(data) > 0 else 0,
                'unique_values': len(np.unique(clean_data)) if len(clean_data) > 0 else 0
            }
            stats_list.append(stats)
        except Exception as e:
            print(f"    Error calculating stats for {fname}: {e}")
            continue
    
    return pd.DataFrame(stats_list)

def identify_redundant_features(corr_matrix, feature_names, threshold=0.95):
    """识别高度相关的冗余特征"""
    redundant_pairs = []
    
    print(f"Identifying redundant features with |correlation| > {threshold}...")
    
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_val = corr_matrix[i, j]
            if not np.isnan(corr_val) and abs(corr_val) > threshold:
                redundant_pairs.append({
                    'feature1': feature_names[i].replace('.npy', ''),
                    'feature2': feature_names[j].replace('.npy', ''),
                    'correlation': corr_val
                })
    
    print(f"Found {len(redundant_pairs)} highly correlated feature pairs")
    return pd.DataFrame(redundant_pairs)

def create_enhanced_visualizations(stats_df, corr_matrix, feature_names, redundant_df):
    """创建增强的可视化图表"""
    
    print("Creating visualizations...")
    
    # 1. 特征统计分布
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 方差分布
    axes[0, 0].hist(stats_df['variance'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Feature Variance Distribution')
    axes[0, 0].set_xlabel('Variance')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_yscale('log')
    
    # 非零比例分布
    axes[0, 1].hist(stats_df['non_zero_ratio'], bins=50, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Non-zero Ratio Distribution')
    axes[0, 1].set_xlabel('Non-zero Ratio')
    axes[0, 1].set_ylabel('Number of Features')
    
    # 唯一值数量分布
    axes[0, 2].hist(stats_df['unique_values'], bins=50, alpha=0.7, color='orange')
    axes[0, 2].set_title('Unique Values Distribution')
    axes[0, 2].set_xlabel('Number of Unique Values')
    axes[0, 2].set_ylabel('Number of Features')
    axes[0, 2].set_xscale('log')
    
    # 均值分布
    axes[1, 0].hist(stats_df['mean'], bins=50, alpha=0.7, color='pink')
    axes[1, 0].set_title('Feature Mean Distribution')
    axes[1, 0].set_xlabel('Mean Value')
    axes[1, 0].set_ylabel('Number of Features')
    
    # 标准差分布
    axes[1, 1].hist(stats_df['std'], bins=50, alpha=0.7, color='lightcoral')
    axes[1, 1].set_title('Feature Standard Deviation Distribution')
    axes[1, 1].set_xlabel('Standard Deviation')
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].set_yscale('log')
    
    # 有限值比例分布
    axes[1, 2].hist(stats_df['finite_ratio'], bins=50, alpha=0.7, color='gold')
    axes[1, 2].set_title('Finite Values Ratio Distribution')
    axes[1, 2].set_xlabel('Finite Ratio')
    axes[1, 2].set_ylabel('Number of Features')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_statistics_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 相关性矩阵热力图（采样显示）
    if len(feature_names) > 100:
        # 对于大矩阵，只显示前100个特征
        sample_indices = np.random.choice(len(feature_names), min(100, len(feature_names)), replace=False)
        sample_corr = corr_matrix[np.ix_(sample_indices, sample_indices)]
        sample_names = [feature_names[i].replace('.npy', '')[:15] for i in sample_indices]
    else:
        sample_corr = corr_matrix
        sample_names = [f.replace('.npy', '')[:15] for f in feature_names]
    
    plt.figure(figsize=(16, 14))
    mask = np.isnan(sample_corr)
    sns.heatmap(sample_corr, mask=mask, cmap='RdBu_r', center=0, 
                square=True, annot=False, cbar_kws={"shrink": .8},
                xticklabels=sample_names, yticklabels=sample_names)
    plt.title(f'Feature Correlation Matrix (Sample of {len(sample_names)} features)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 高相关性特征对
    if len(redundant_df) > 0:
        plt.figure(figsize=(12, 8))
        redundant_df_sorted = redundant_df.reindex(redundant_df['correlation'].abs().sort_values(ascending=False).index)
        top_redundant = redundant_df_sorted.head(20)
        
        y_pos = range(len(top_redundant))
        colors = ['red' if abs(x) > 0.99 else 'orange' if abs(x) > 0.97 else 'yellow' 
                 for x in top_redundant['correlation']]
        
        plt.barh(y_pos, top_redundant['correlation'], color=colors)
        plt.yticks(y_pos, [f"{row['feature1'][:20]}...\nvs\n{row['feature2'][:20]}..." 
                          for _, row in top_redundant.iterrows()])
        plt.xlabel('Correlation Coefficient')
        plt.title('Top 20 Highly Correlated Feature Pairs')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'top_redundant_features_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 相关性分布
    upper_triangle = np.triu(corr_matrix, k=1)
    correlations = upper_triangle[upper_triangle != 0]
    correlations = correlations[~np.isnan(correlations)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(correlations, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=0.95, color='red', linestyle='--', label='Threshold: 0.95')
    plt.axvline(x=-0.95, color='red', linestyle='--')
    plt.axvline(x=np.mean(correlations), color='green', linestyle='-', label=f'Mean: {np.mean(correlations):.3f}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Correlations')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(stats_df, redundant_df, corr_matrix, feature_names):
    """生成综合分析报告"""
    report = []
    report.append("# Complete Feature Analysis Report - Yangtze River Basin Rainfall Prediction\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## 1. Overview\n")
    report.append(f"- Total analyzed features: {len(stats_df)}")
    report.append(f"- Sample size per feature: 10,000 random samples")
    report.append(f"- Total possible correlations: {len(feature_names) * (len(feature_names) - 1) // 2}\n")
    
    report.append("## 2. Data Quality Assessment\n")
    report.append("### 2.1 Basic Statistics")
    report.append(f"- Mean variance: {stats_df['variance'].mean():.6f}")
    report.append(f"- Mean non-zero ratio: {stats_df['non_zero_ratio'].mean():.3f}")
    report.append(f"- Mean finite ratio: {stats_df['finite_ratio'].mean():.3f}")
    report.append(f"- Average unique values per feature: {stats_df['unique_values'].mean():.1f}\n")
    
    # 识别问题特征
    low_variance_features = stats_df[stats_df['variance'] < 1e-10]
    high_zero_features = stats_df[stats_df['non_zero_ratio'] < 0.01]
    low_finite_features = stats_df[stats_df['finite_ratio'] < 0.95]
    
    report.append("### 2.2 Problematic Features")
    report.append(f"- Low variance features (< 1e-10): {len(low_variance_features)}")
    report.append(f"- High sparsity features (< 1% non-zero): {len(high_zero_features)}")
    report.append(f"- Low finite ratio features (< 95% finite): {len(low_finite_features)}\n")
    
    # 相关性分析
    upper_triangle = np.triu(corr_matrix, k=1)
    correlations = upper_triangle[upper_triangle != 0]
    correlations = correlations[~np.isnan(correlations)]
    
    report.append("## 3. Correlation Analysis\n")
    report.append(f"- Valid correlation pairs: {len(correlations)}")
    report.append(f"- Mean correlation: {np.mean(correlations):.3f}")
    report.append(f"- Std correlation: {np.std(correlations):.3f}")
    report.append(f"- Max correlation: {np.max(np.abs(correlations)):.3f}")
    report.append(f"- High correlation pairs (|r| > 0.95): {len(redundant_df)}")
    if len(redundant_df) > 0:
        report.append(f"- Very high correlation pairs (|r| > 0.99): {len(redundant_df[redundant_df['correlation'].abs() > 0.99])}")
    else:
        report.append(f"- Very high correlation pairs (|r| > 0.99): 0")
    report.append("")
    
    if len(redundant_df) > 0:
        report.append("### 3.1 Top 10 Highly Correlated Feature Pairs")
        top_pairs = redundant_df.reindex(redundant_df['correlation'].abs().sort_values(ascending=False).index).head(10)
        for i, (_, row) in enumerate(top_pairs.iterrows(), 1):
            report.append(f"{i}. {row['feature1']} vs {row['feature2']}: r={row['correlation']:.4f}")
        report.append("")
    
    report.append("## 4. Recommendations\n")
    
    if len(redundant_df) > 10:
        report.append("- Consider removing highly redundant features to reduce multicollinearity")
    if len(low_variance_features) > 0:
        report.append("- Remove low variance features to improve model efficiency")
    if len(high_zero_features) > 5:
        report.append("- Consider log transformation or binning for highly sparse features")
    if len(low_finite_features) > 0:
        report.append("- Investigate and clean features with infinite/NaN values")
    
    report.append(f"\n## 5. Feature Categories Distribution")
    feature_categories = categorize_features_simple([f['feature'] for f in stats_df.to_dict('records')])
    for category, count in feature_categories.items():
        report.append(f"- {category}: {count} features")
    
    # 保存报告
    with open(os.path.join(OUTPUT_DIR, 'comprehensive_feature_analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def categorize_features_simple(feature_names):
    """简单的特征分类"""
    categories = {
        'Basic': 0, 'Temporal': 0, 'Multi-product': 0, 
        'Lag': 0, 'Spatial': 0, 'Statistical': 0, 'Interaction': 0
    }
    
    for fname in feature_names:
        if any(x in fname for x in ['raw_', 'target_']):
            categories['Basic'] += 1
        elif any(x in fname for x in ['cos_', 'sin_', 'day_', 'month_', 'season_', 'time', 'diff_']):
            categories['Temporal'] += 1
        elif any(x in fname for x in ['correlation_', 'multi_product_', 'consistency', 'disagreement']):
            categories['Multi-product'] += 1
        elif 'lag_' in fname:
            categories['Lag'] += 1
        elif 'spatial_' in fname:
            categories['Spatial'] += 1
        elif any(x in fname for x in ['quantile_', 'anomaly_', 'extreme_', 'rolling_', 'intensity_']):
            categories['Statistical'] += 1
        elif 'interaction_' in fname:
            categories['Interaction'] += 1
        else:
            categories['Basic'] += 1
    
    return categories

def main():
    """主函数"""
    print("=" * 80)
    print("COMPLETE FEATURE ANALYSIS - YANGTZE RIVER BASIN RAINFALL PREDICTION")
    print("=" * 80)
    
    # 设置随机种子确保可复现性
    np.random.seed(42)
    
    # 1. 加载所有特征信息
    print("\n1. Loading feature information...")
    features_df = load_all_flattened_features()
    
    # 2. 批量加载特征数据
    print("\n2. Loading feature data...")
    feature_names = features_df['feature_file_name'].tolist()
    feature_data, valid_features = load_feature_batch(feature_names, batch_size=50, sample_size=10000)
    
    if len(valid_features) < 10:
        print("ERROR: Not enough valid features for analysis")
        return
    
    # 3. 计算基本统计量
    print("\n3. Calculating basic statistics...")
    stats_df = calculate_basic_statistics(feature_data)
    
    # 4. 计算完整相关性矩阵
    print("\n4. Calculating correlation matrix...")
    corr_matrix, feature_names_valid = calculate_correlation_matrix_chunked(feature_data, chunk_size=50)
    
    # 5. 识别冗余特征
    print("\n5. Identifying redundant features...")
    redundant_df = identify_redundant_features(corr_matrix, feature_names_valid, threshold=0.95)
    
    # 6. 创建可视化
    print("\n6. Creating visualizations...")
    create_enhanced_visualizations(stats_df, corr_matrix, feature_names_valid, redundant_df)
    
    # 7. 生成综合报告
    print("\n7. Generating comprehensive report...")
    report = generate_comprehensive_report(stats_df, redundant_df, corr_matrix, feature_names_valid)
    
    # 8. 保存结果
    print("\n8. Saving results...")
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'complete_feature_statistics.csv'), index=False)
    if len(redundant_df) > 0:
        redundant_df.to_csv(os.path.join(OUTPUT_DIR, 'redundant_feature_pairs.csv'), index=False)
    
    # 保存相关性矩阵
    np.save(os.path.join(OUTPUT_DIR, 'correlation_matrix.npy'), corr_matrix)
    with open(os.path.join(OUTPUT_DIR, 'feature_names.txt'), 'w') as f:
        for name in feature_names_valid:
            f.write(name + '\n')
    
    print(f"\nAnalysis completed successfully!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Analyzed {len(valid_features)} features")
    print(f"Found {len(redundant_df)} highly correlated pairs")
    
    return stats_df, redundant_df, corr_matrix, report

if __name__ == "__main__":
    stats_df, redundant_df, corr_matrix, report = main()