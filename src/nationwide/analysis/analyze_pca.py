"""
PCA降维分析和可视化脚本
分析三天降水数据的主成分分析结果
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 创建输出目录
os.makedirs('f:/rainfalldata/figures/pca_analysis', exist_ok=True)

def load_and_prepare_data():
    """加载数据并准备三天特征"""
    print("加载数据...")
    DATAFILE = {
        "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat",
        "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat",
        "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat",
        "IMERG": "IMERGdata/IMERG_2016_2020.mat",
        "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat",
        "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat",
        "CHM": "CHMdata/CHM_2016_2020.mat",
        "MASK": "mask.mat",
    }
    
    DATAS = {}
    for key, filepath in DATAFILE.items():
        if key == "MASK":
            DATAS[key] = loadmat(filepath)["mask"]
        else:
            DATAS[key] = loadmat(filepath)["data"].astype(np.float32)
    
    # 准备特征矩阵
    MASK = DATAS["MASK"]
    PRODUCT = DATAS.copy()
    PRODUCT.pop("MASK")
    CHM_DATA = PRODUCT.pop("CHM")
    
    feature_names = []
    for product in PRODUCT.keys():
        for day in ["today", "yesterday", "day_before"]:
            feature_names.append(f"{product}_{day}")
    
    return DATAS, PRODUCT, CHM_DATA, MASK, feature_names

def create_cross_features(X, feature_names):
    """创建交叉特征和三交叉特征"""
    print("\n创建交叉特征...")
    n_features = X.shape[1]
    n_products = len(feature_names) // 3  # 每天的产品数量
    
    # 创建两两交叉特征
    pair_cross_features = []
    pair_feature_names = []
    
    # 1. 同一产品不同天之间的交叉
    for p in range(n_products):
        today_idx = p
        yesterday_idx = p + n_products
        day_before_idx = p + 2 * n_products
        
        # 今天和昨天
        pair_cross_features.append(X[:, today_idx] * X[:, yesterday_idx])
        pair_feature_names.append(f"{feature_names[today_idx]}*{feature_names[yesterday_idx]}")
        
        # 今天和前天
        pair_cross_features.append(X[:, today_idx] * X[:, day_before_idx])
        pair_feature_names.append(f"{feature_names[today_idx]}*{feature_names[day_before_idx]}")
        
        # 昨天和前天
        pair_cross_features.append(X[:, yesterday_idx] * X[:, day_before_idx])
        pair_feature_names.append(f"{feature_names[yesterday_idx]}*{feature_names[day_before_idx]}")
    
    # 2. 同一天不同产品之间的交叉
    for day_offset in range(3):  # 对每一天
        start_idx = day_offset * n_products
        for i in range(n_products):
            for j in range(i+1, n_products):
                idx1 = start_idx + i
                idx2 = start_idx + j
                pair_cross_features.append(X[:, idx1] * X[:, idx2])
                pair_feature_names.append(f"{feature_names[idx1]}*{feature_names[idx2]}")
    
    # 创建三交叉特征
    triple_cross_features = []
    triple_feature_names = []
    
    # 1. 同一产品三天的交叉
    for p in range(n_products):
        today_idx = p
        yesterday_idx = p + n_products
        day_before_idx = p + 2 * n_products
        
        triple_cross_features.append(X[:, today_idx] * X[:, yesterday_idx] * X[:, day_before_idx])
        triple_feature_names.append(f"{feature_names[today_idx]}*{feature_names[yesterday_idx]}*{feature_names[day_before_idx]}")
    
    # 2. 同一天三个不同产品的交叉（限制组合数量）
    for day_offset in range(3):
        start_idx = day_offset * n_products
        for i in range(n_products-2):
            for j in range(i+1, i+3):  # 只取相邻的两个产品
                for k in range(j+1, j+2):  # 再取一个相邻的产品
                    if k < n_products:
                        idx1 = start_idx + i
                        idx2 = start_idx + j
                        idx3 = start_idx + k
                        triple_cross_features.append(X[:, idx1] * X[:, idx2] * X[:, idx3])
                        triple_feature_names.append(f"{feature_names[idx1]}*{feature_names[idx2]}*{feature_names[idx3]}")
    
    # 合并所有特征
    X_cross = np.column_stack([X] + pair_cross_features + triple_cross_features)
    all_feature_names = feature_names + pair_feature_names + triple_feature_names
    
    print(f"原始特征数: {len(feature_names)}")
    print(f"两交叉特征数: {len(pair_feature_names)}")
    print(f"三交叉特征数: {len(triple_feature_names)}")
    print(f"总特征数: {X_cross.shape[1]}")
    
    return X_cross, all_feature_names, triple_feature_names

def analyze_pca(X, feature_names):
    """详细的PCA分析"""
    print("\n执行PCA分析...")
    
    # 创建交叉特征
    X_cross, all_feature_names, triple_feature_names = create_cross_features(X, feature_names)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cross)
    
    # 执行PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 1. 绘制方差解释比例
    plt.figure(figsize=(10, 6))
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained_var_ratio)
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% 方差')
    plt.grid(True)
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差比例')
    plt.title('PCA累计解释方差比例')
    plt.legend()
    plt.savefig('f:/rainfalldata/figures/pca_analysis/variance_explained.png')
    plt.close()
    
    # 2. 绘制碎石图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_) + 1), 
             pca.explained_variance_, 'bo-')
    plt.xlabel('主成分序号')
    plt.ylabel('特征值')
    plt.title('PCA碎石图')
    plt.grid(True)
    plt.savefig('f:/rainfalldata/figures/pca_analysis/scree_plot.png')
    plt.close()
    
    # 修改特征贡献分析部分以处理交叉特征
    n_components = np.argmax(cumsum >= 0.95) + 1
    print(f"\n解释95%方差需要的主成分数量: {n_components}")
    
    # 分析前5个主成分的特征贡献
    n_show = min(5, len(all_feature_names))
    
    # 创建特征贡献热力图（只显示top 30个特征）
    top_features = 30
    loadings = pd.DataFrame(
        pca.components_[:n_show].T,
        columns=[f'PC{i+1}' for i in range(n_show)],
        index=all_feature_names
    )
    
    # 计算每个特征的总体重要性
    feature_importance = np.abs(loadings.values).mean(axis=1)
    top_indices = np.argsort(feature_importance)[-top_features:]
    
    # 绘制热力图
    plt.figure(figsize=(15, 10))
    sns.heatmap(loadings.iloc[top_indices], cmap='RdBu_r', center=0)
    plt.title('前30个最重要特征在前5个主成分中的贡献')
    plt.tight_layout()
    plt.savefig('f:/rainfalldata/figures/pca_analysis/top_feature_loadings.png')
    plt.close()
    
    # 分析特征类型的重要性
    n_original = len(feature_names)
    n_pair_cross = len(all_feature_names) - len(triple_feature_names) - n_original
    
    importance_by_type = {
        'original': np.mean(feature_importance[:n_original]),
        'pair_cross': np.mean(feature_importance[n_original:n_original+n_pair_cross]),
        'triple_cross': np.mean(feature_importance[n_original+n_pair_cross:])
    }
    
    # 绘制特征类型重要性对比
    plt.figure(figsize=(8, 6))
    plt.bar(importance_by_type.keys(), importance_by_type.values())
    plt.title('不同类型特征的平均重要性')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('f:/rainfalldata/figures/pca_analysis/feature_type_importance.png')
    plt.close()
    
    # 打印前20个最重要的特征
    print("\n前20个最重要的特征:")
    top20_idx = np.argsort(feature_importance)[-20:]
    for idx in reversed(top20_idx):
        print(f"{all_feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    return pca, n_components, importance_by_type

def main():
    # 加载数据
    DATAS, PRODUCT, CHM_DATA, MASK, feature_names = load_and_prepare_data()
    
    # 准备样本数据(使用部分数据进行分析)
    print("\n准备样本数据...")
    nlat, nlon, ntime = CHM_DATA.shape
    sample_size = 100000  # 使用10万个样本进行分析
    
    features = []
    for t in range(3, ntime):  # 从第三天开始
        for i in range(nlat):
            for j in range(nlon):
                if MASK[i,j] == 1:
                    feature = []
                    # 收集三天的特征
                    for day_offset in range(3):
                        current_day = t - day_offset
                        for product in PRODUCT:
                            value = DATAS[product][i,j,current_day]
                            feature.append(float(value) if not np.isnan(value) else 0.0)
                    features.append(feature)
                    
                    if len(features) >= sample_size:
                        break
            if len(features) >= sample_size:
                break
        if len(features) >= sample_size:
            break
    
    X = np.array(features)
    print(f"分析数据形状: {X.shape}")
    
    # 执行PCA分析
    pca, n_components, importance_by_type = analyze_pca(X, feature_names)
    
    # 保存分析结果
    results = {
        'n_components_95': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'feature_importance_by_type': importance_by_type,
        'feature_names': feature_names,
        'components': pca.components_.tolist()
    }
    
    import json
    with open('f:/rainfalldata/results/pca_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nPCA分析完成！结果已保存至figures/pca_analysis/目录")
    print(f"95%方差解释所需的主成分数量: {n_components}")
    print(f"第一主成分解释方差比例: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"前三个主成分累计解释方差比例: {np.sum(pca.explained_variance_ratio_[:3]):.4f}")
    
    # 打印特征类型重要性
    print("\n特征类型重要性分析:")
    for feat_type, importance in importance_by_type.items():
        print(f"{feat_type}: {importance:.4f}")

if __name__ == "__main__":
    main()
