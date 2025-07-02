#!/usr/bin/env python3
"""
特征分析脚本：评估长江流域降雨预测特征库中特征的质量和相关性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FEATURES_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/feature_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_features_info():
    """加载特征信息CSV文件"""
    csv_path = os.path.join(FEATURES_DIR, "features_list.csv")
    df = pd.read_csv(csv_path)
    
    # 筛选出shape为(5247240,)的可比较特征
    flattened_features = df[df['shape'] == '(5247240,)'].copy()
    
    print(f"总特征数: {len(df)}")
    print(f"已展平特征数: {len(flattened_features)}")
    print(f"未展平特征数: {len(df) - len(flattened_features)}")
    
    return flattened_features

def categorize_features(features_df):
    """将特征按类别分组"""
    categories = {
        '基础特征': [],
        '时序特征': [],
        '多产品协同': [],
        '滞后特征': [],
        '空间特征': [],
        '高级统计': [],
        '交互特征': []
    }
    
    for _, row in features_df.iterrows():
        fname = row['feature_file_name']
        desc = row['feature_description']
        
        if any(x in fname for x in ['raw_', 'target_']):
            categories['基础特征'].append(fname)
        elif any(x in fname for x in ['cos_', 'sin_', 'day_', 'month_', 'season_', 'time', 'diff_']):
            categories['时序特征'].append(fname)
        elif any(x in fname for x in ['correlation_', 'multi_product_', 'consistency', 'disagreement']):
            categories['多产品协同'].append(fname)
        elif 'lag_' in fname:
            categories['滞后特征'].append(fname)
        elif 'spatial_' in fname:
            categories['空间特征'].append(fname)
        elif any(x in fname for x in ['quantile_', 'anomaly_', 'extreme_', 'rolling_', 'intensity_']):
            categories['高级统计'].append(fname)
        elif 'interaction_' in fname:
            categories['交互特征'].append(fname)
        else:
            categories['基础特征'].append(fname)
    
    return categories

def load_sample_features(feature_names, sample_size=50000):
    """加载特征数据的样本进行分析"""
    print(f"加载 {len(feature_names)} 个特征的样本数据...")
    
    feature_data = {}
    valid_features = []
    
    for i, fname in enumerate(feature_names[:20]):  # 先分析前20个特征
        if i % 5 == 0:
            print(f"  处理进度: {i+1}/{min(20, len(feature_names))}")
        
        try:
            fpath = os.path.join(FEATURES_DIR, fname)
            data = np.load(fpath)
            
            if data.shape[0] != 5247240:
                print(f"  跳过 {fname}: shape不匹配 {data.shape}")
                continue
            
            # 随机采样
            indices = np.random.choice(data.shape[0], min(sample_size, data.shape[0]), replace=False)
            sampled_data = data[indices]
            
            # 检查数据质量
            if np.all(np.isnan(sampled_data)) or np.all(sampled_data == 0):
                print(f"  跳过 {fname}: 全为NaN或0")
                continue
            
            feature_data[fname] = sampled_data
            valid_features.append(fname)
            
        except Exception as e:
            print(f"  加载 {fname} 失败: {e}")
            continue
    
    print(f"成功加载 {len(valid_features)} 个有效特征")
    return feature_data, valid_features

def calculate_basic_statistics(feature_data):
    """计算基本统计量"""
    stats_list = []
    
    for fname, data in feature_data.items():
        stats = {
            'feature': fname.replace('.npy', ''),
            'mean': np.nanmean(data),
            'std': np.nanstd(data),
            'min': np.nanmin(data),
            'max': np.nanmax(data),
            'median': np.nanmedian(data),
            'q25': np.nanpercentile(data, 25),
            'q75': np.nanpercentile(data, 75),
            'skewness': calculate_skewness(data),
            'variance': np.nanvar(data),
            'non_zero_ratio': np.mean(data != 0),
            'nan_ratio': np.mean(np.isnan(data))
        }
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)

def calculate_skewness(data):
    """计算偏度"""
    try:
        from scipy.stats import skew
        return skew(data[~np.isnan(data)])
    except:
        return np.nan

def calculate_correlation_matrix(feature_data):
    """计算相关性矩阵"""
    feature_names = list(feature_data.keys())
    n_features = len(feature_names)
    
    # Pearson相关性
    pearson_corr = np.full((n_features, n_features), np.nan)
    
    print("计算Pearson相关性矩阵...")
    for i in range(n_features):
        for j in range(i, n_features):
            try:
                data1 = feature_data[feature_names[i]]
                data2 = feature_data[feature_names[j]]
                
                # 移除NaN值
                mask = ~(np.isnan(data1) | np.isnan(data2))
                if np.sum(mask) > 100:  # 至少需要100个有效值
                    corr, _ = pearsonr(data1[mask], data2[mask])
                    pearson_corr[i, j] = corr
                    pearson_corr[j, i] = corr
            except:
                continue
    
    return pearson_corr, feature_names

def identify_redundant_features(corr_matrix, feature_names, threshold=0.95):
    """识别高度相关的冗余特征"""
    redundant_pairs = []
    
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) > threshold:
                redundant_pairs.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': corr_matrix[i, j]
                })
    
    return pd.DataFrame(redundant_pairs)

def create_visualizations(stats_df, corr_matrix, feature_names, redundant_df):
    """创建可视化图表"""
    
    # 1. 特征分布统计
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 方差分布
    axes[0, 0].hist(stats_df['variance'], bins=20, alpha=0.7)
    axes[0, 0].set_title('特征方差分布')
    axes[0, 0].set_xlabel('方差')
    axes[0, 0].set_ylabel('特征数量')
    
    # 非零比例分布
    axes[0, 1].hist(stats_df['non_zero_ratio'], bins=20, alpha=0.7)
    axes[0, 1].set_title('特征非零比例分布')
    axes[0, 1].set_xlabel('非零比例')
    axes[0, 1].set_ylabel('特征数量')
    
    # 偏度分布
    skew_data = stats_df['skewness'].dropna()
    axes[1, 0].hist(skew_data, bins=20, alpha=0.7)
    axes[1, 0].set_title('特征偏度分布')
    axes[1, 0].set_xlabel('偏度')
    axes[1, 0].set_ylabel('特征数量')
    
    # NaN比例分布
    axes[1, 1].hist(stats_df['nan_ratio'], bins=20, alpha=0.7)
    axes[1, 1].set_title('特征缺失值比例分布')
    axes[1, 1].set_xlabel('NaN比例')
    axes[1, 1].set_ylabel('特征数量')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_statistics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 相关性矩阵热力图
    plt.figure(figsize=(12, 10))
    mask = np.isnan(corr_matrix)
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0, 
                square=True, annot=False, cbar_kws={"shrink": .8})
    plt.title('特征相关性矩阵热力图')
    plt.xticks(range(len(feature_names)), [f.replace('.npy', '')[:20] for f in feature_names], rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), [f.replace('.npy', '')[:20] for f in feature_names], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 高相关性特征对
    if len(redundant_df) > 0:
        plt.figure(figsize=(10, 6))
        redundant_df_sorted = redundant_df.reindex(redundant_df['correlation'].abs().sort_values(ascending=False).index)
        top_redundant = redundant_df_sorted.head(10)
        
        y_pos = range(len(top_redundant))
        plt.barh(y_pos, top_redundant['correlation'])
        plt.yticks(y_pos, [f"{row['feature1'][:15]}...\nvs\n{row['feature2'][:15]}..." 
                          for _, row in top_redundant.iterrows()])
        plt.xlabel('相关系数')
        plt.title('Top 10 高相关性特征对')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'top_redundant_features.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(stats_df, redundant_df, categories):
    """生成分析报告"""
    report = []
    report.append("# 长江流域降雨预测特征分析报告\n")
    report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## 1. 特征概览\n")
    report.append(f"- 总分析特征数: {len(stats_df)}")
    for cat, features in categories.items():
        if features:
            report.append(f"- {cat}: {len(features)}个")
    report.append("")
    
    report.append("## 2. 数据质量评估\n")
    report.append("### 2.1 基本统计量")
    report.append(f"- 平均方差: {stats_df['variance'].mean():.6f}")
    report.append(f"- 平均非零比例: {stats_df['non_zero_ratio'].mean():.3f}")
    report.append(f"- 平均缺失值比例: {stats_df['nan_ratio'].mean():.6f}")
    report.append("")
    
    # 识别问题特征
    low_variance_features = stats_df[stats_df['variance'] < 1e-10]
    high_nan_features = stats_df[stats_df['nan_ratio'] > 0.1]
    constant_features = stats_df[stats_df['std'] < 1e-10]
    
    report.append("### 2.2 问题特征识别")
    report.append(f"- 低方差特征(< 1e-10): {len(low_variance_features)}个")
    report.append(f"- 高缺失值特征(> 10%): {len(high_nan_features)}个")
    report.append(f"- 常数特征(std < 1e-10): {len(constant_features)}个")
    report.append("")
    
    report.append("## 3. 相关性分析\n")
    report.append(f"- 高相关性特征对数(|r| > 0.95): {len(redundant_df)}")
    if len(redundant_df) > 0:
        report.append("- Top 5 最高相关性:")
        for _, row in redundant_df.head(5).iterrows():
            report.append(f"  - {row['feature1']} vs {row['feature2']}: r={row['correlation']:.3f}")
    report.append("")
    
    report.append("## 4. 建议\n")
    if len(redundant_df) > 5:
        report.append("- 考虑移除高度相关的冗余特征以减少多重共线性")
    if len(low_variance_features) > 0:
        report.append("- 考虑移除低方差特征以提高模型效率")
    if len(high_nan_features) > 0:
        report.append("- 检查高缺失值特征的数据质量")
    
    # 保存报告
    with open(os.path.join(OUTPUT_DIR, 'feature_analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def main():
    """主函数"""
    print("=" * 60)
    print("长江流域降雨预测特征分析")
    print("=" * 60)
    
    # 1. 加载特征信息
    print("\n1. 加载特征信息...")
    features_df = load_features_info()
    categories = categorize_features(features_df)
    
    # 2. 加载样本数据
    print("\n2. 加载特征样本数据...")
    feature_names = features_df['feature_file_name'].tolist()
    feature_data, valid_features = load_sample_features(feature_names)
    
    if len(valid_features) == 0:
        print("错误: 没有有效的特征数据可以分析")
        return
    
    # 3. 计算基本统计量
    print("\n3. 计算基本统计量...")
    stats_df = calculate_basic_statistics(feature_data)
    
    # 4. 计算相关性矩阵
    print("\n4. 计算相关性矩阵...")
    corr_matrix, feature_names_valid = calculate_correlation_matrix(feature_data)
    
    # 5. 识别冗余特征
    print("\n5. 识别冗余特征...")
    redundant_df = identify_redundant_features(corr_matrix, feature_names_valid, threshold=0.95)
    
    # 6. 创建可视化
    print("\n6. 生成可视化图表...")
    create_visualizations(stats_df, corr_matrix, feature_names_valid, redundant_df)
    
    # 7. 生成报告
    print("\n7. 生成分析报告...")
    report = generate_report(stats_df, redundant_df, categories)
    
    # 8. 保存结果
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_statistics.csv'), index=False)
    if len(redundant_df) > 0:
        redundant_df.to_csv(os.path.join(OUTPUT_DIR, 'redundant_features.csv'), index=False)
    
    print(f"\n分析完成! 结果保存在: {OUTPUT_DIR}")
    print("\n生成的文件:")
    print("- feature_statistics.csv: 特征统计量")
    print("- redundant_features.csv: 冗余特征对")
    print("- feature_analysis_report.md: 分析报告")
    print("- *.png: 可视化图表")
    
    return stats_df, redundant_df, report

if __name__ == "__main__":
    stats_df, redundant_df, report = main()