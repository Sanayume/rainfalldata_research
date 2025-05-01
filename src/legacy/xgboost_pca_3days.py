"""
使用前三天数据的PCA优化版XGBoost降水预测训练脚本
包含PCA降维、特征工程、早停策略和阈值优化
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                           f1_score, recall_score, precision_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
import gc

# 设置常量
N_DAYS = 3  # 使用的历史天数
BATCH_SIZE = 50000  # 分批处理的大小
PCA_COMPONENTS = 20  # PCA保留的组件数
RAIN_THRESHOLD = 0.1  # 降雨阈值

# 定义数据文件路径
DATAFILE = {
    'CHM': './data/CHM.mat',  # 降水数据
    'CLW': './data/CLW.mat',  # 云水数据
    'RH': './data/RH.mat',    # 相对湿度
    'TMP': './data/TMP.mat',  # 温度
    'MASK': './data/MASK.mat' # 掩码
}

# 定义产品字典
PRODUCT = {
    'CLW': 'Cloud Liquid Water',
    'RH': 'Relative Humidity',
    'TMP': 'Temperature'
}

# ...existing code for directory creation and warnings...

def load_and_preprocess_data(DATAFILE):
    """加载并预处理数据，转换为float32类型"""
    print("加载数据...")
    DATAS = {}
    for key, filepath in DATAFILE.items():
        try:
            if key == "MASK":
                DATAS[key] = loadmat(filepath)["mask"]
            else:
                # 直接加载为float32
                DATAS[key] = loadmat(filepath)["data"].astype(np.float32)
            print(f"成功加载 {key}: 形状 {DATAS[key].shape}")
        except Exception as e:
            print(f"加载 {key} 失败: {str(e)}")
    return DATAS

def create_multi_day_features(DATAS, PRODUCT, MASK, N_DAYS):
    """创建多天的特征矩阵，使用生成器分批处理"""
    nlat, nlon, ntime = DATAS[list(PRODUCT.keys())[0]].shape
    valid_point = MASK == 1
    n_valid = np.sum(valid_point)
    
    # 从第N_DAYS天开始，确保有足够的历史数据
    for t in range(N_DAYS, ntime):
        features = []
        labels = []
        
        for i in range(nlat):
            for j in range(nlon):
                if MASK[i,j] == 1:
                    feature = []
                    # 收集过去N_DAYS天的数据
                    for day in range(N_DAYS):
                        current_t = t - day
                        for product in PRODUCT:
                            value = DATAS[product][i,j,current_t]
                            feature.append(float(value) if not np.isnan(value) else 0.0)
                    
                    features.append(feature)
                    labels.append(float(DATAS['CHM'][i,j,t] > 0))
                    
                    # 达到批处理大小时yield
                    if len(features) == BATCH_SIZE:
                        yield np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)
                        features = []
                        labels = []
        
        # 处理最后一批
        if features:
            yield np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)

def create_interaction_features_with_pca(X, scaler=None, pca=None, training=False):
    """创建交互特征并使用PCA降维"""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    n_products = X.shape[1] // N_DAYS
    interaction_features = []
    
    # 1. 同一产品不同天之间的交互
    for p in range(n_products):
        for d1 in range(N_DAYS):
            for d2 in range(d1+1, N_DAYS):
                idx1 = p + d1 * n_products
                idx2 = p + d2 * n_products
                interaction = X_scaled[:, idx1] * X_scaled[:, idx2]
                interaction_features.append(interaction.reshape(-1, 1))

    # 2. 同一天不同产品之间的交互
    for day in range(N_DAYS):
        offset = day * n_products
        for i in range(n_products):
            for j in range(i+1, n_products):
                idx1 = offset + i
                idx2 = offset + j
                interaction = X_scaled[:, idx1] * X_scaled[:, idx2]
                interaction_features.append(interaction.reshape(-1, 1))

    # 合并所有交互特征
    X_interaction = np.hstack(interaction_features)
    
    # PCA降维
    if pca is None:
        pca = PCA(n_components=PCA_COMPONENTS)
        X_pca = pca.fit_transform(X_interaction)
    else:
        X_pca = pca.transform(X_interaction)

    # 组合原始特征和PCA降维后的交互特征
    X_combined = np.hstack([X_scaled, X_pca])
    
    if training:
        return X_combined, scaler, pca
    return X_combined

def train_model_with_cv(X_train, y_train, X_test, y_test):
    """使用交叉验证训练模型"""
    # 计算类别权重
    neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
    adjusted_ratio = min(neg_pos_ratio, 1.5)

    # 创建训练集和验证集
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 创建模型
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=adjusted_ratio,
        tree_method='hist',  # 使用直方图算法加速
def main():
    # 加载数据
    DATAS = load_and_preprocess_data(DATAFILE)
    
    # 获取掩码数据
    MASK = DATAS['MASK']
    
    # 创建特征生成器
    feature_generator = create_multi_day_features(DATAS, PRODUCT, MASK, N_DAYS)
    model.fit(
        X_train_part, y_train_part,
        eval_set=eval_set,
        eval_metric=['logloss', 'auc'],
        early_stopping_rounds=20,
        verbose=100
    )

    return model

def main():
    # 加载数据
    DATAS = load_and_preprocess_data(DATAFILE)
    
    # 创建特征生成器
    feature_generator = create_multi_day_features(DATAS, PRODUCT, MASK, N_DAYS)
    
    # 收集所有特征
    X_all = []
    y_all = []
    print("\n生成特征...")
    for i, (X_batch, y_batch) in enumerate(feature_generator):
        X_all.append(X_batch)
        y_all.append(y_batch)
        if (i+1) % 10 == 0:
            print(f"已处理 {(i+1)*BATCH_SIZE} 个样本")
    
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    
    # 分割训练集和测试集
    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    
    # 创建并应用特征工程
    print("\n应用特征工程和PCA降维...")
    X_train_final, scaler, pca = create_interaction_features_with_pca(
        X_train, training=True
    )
    X_test_final = create_interaction_features_with_pca(
        X_test, scaler=scaler, pca=pca
    )
    
    # 训练模型
    model = train_model_with_cv(X_train_final, y_train, X_test_final, y_test)
    
    # 模型评估和结果保存
    # ...existing code for model evaluation and results saving...

if __name__ == "__main__":
    main()
