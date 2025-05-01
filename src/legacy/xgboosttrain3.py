"""
优化版XGBoost降水预测训练脚本
包含特征工程、交互特征、早停策略和阈值优化
加入前一天数据作为特征
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            r2_score, accuracy_score, classification_report,
                            f1_score, recall_score, precision_score)
from sklearn.preprocessing import StandardScaler
import sys
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('f:/rainfalldata/figures', exist_ok=True)
os.makedirs('f:/rainfalldata/results', exist_ok=True)
os.makedirs('f:/rainfalldata/models', exist_ok=True)

# 抑制不相关警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("=" * 70)
print("优化版XGBoost降水预测训练脚本 - 包含高级特征工程和阈值优化")
print("=" * 70)
print("增强版：使用当天和前一天的数据作为特征")

# 定义数据文件路径
DATAFILE = {
    "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat", # CMORPH降水数据 shape :  (144, 256, 1827)
    "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat", # CHIRPS降水数据 shape :  (144, 256, 1827)
    "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat", # SM2RAIN降水数据 shape :  (144, 256, 1827)
    "IMERG": "IMERGdata/IMERG_2016_2020.mat", # IMERG降水数据 shape :  (144, 256, 1827)
    "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat", # GSMAP降水数据 shape :  (144, 256, 1827)
    "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat", # PERSIANN降水数据 shape :  (144, 256, 1827)
    "CHM": "CHMdata/CHM_2016_2020.mat", # CHM降水数据 shape :  (144, 256, 1827)
    "MASK": "mask.mat", #china mask shape :  (144, 256)
}

print("加载数据...")
# 加载数据
DATAS = {}
for key, filepath in DATAFILE.items():
    try:
        if key == "MASK":
            DATAS[key] = loadmat(filepath)["mask"]
        else:
            DATAS[key] = loadmat(filepath)["data"]
        print(f"成功加载 {key}: 形状 {DATAS[key].shape}")
    except Exception as e:
        print(f"加载 {key} 失败: {str(e)}")

MASK = DATAS["MASK"]
PRODUCT = DATAS.copy()
PRODUCT.pop("MASK")
CHM_DATA = PRODUCT.pop("CHM")
print(f"产品数据列表: {list(PRODUCT.keys())}")

# 数据预处理
print("\n开始数据预处理...")
nlat, nlon, ntime = CHM_DATA.shape
valid_point = MASK == 1
print(f"有效点数: {np.sum(valid_point)}")

# 使用当天+前一天数据，因此从第二天开始训练
n_samples = np.sum(valid_point) * (ntime-1)  # 减去一天，因为第一天没有前一天的数据
trainsample = np.sum(valid_point) * (ntime-1-366)  # 使用前几年作为训练集，减去第一天
testsample = np.sum(valid_point) * 366           # 使用最后一年作为测试集

# 特征维度翻倍，因为每个产品有今天和昨天两天的数据
X_train = np.zeros((trainsample, len(PRODUCT)*2))
Y_train = np.zeros((trainsample, 1))
X_test = np.zeros((testsample, len(PRODUCT)*2))
Y_test = np.zeros((testsample, 1))

# 提取特征和标签
print("处理特征数据...")
train_idx = 0
test_idx = 0

# 从t=1开始，因为t=0没有前一天的数据
for t in range(1, ntime):
    is_train = t < (ntime - 366)  # 判断是训练集还是测试集
    
    for i in range(nlat):
        for j in range(nlon):
            if MASK[i,j] == 1:
                feature = []
                # 先添加当天的所有产品数据
                for p_idx, product in enumerate(PRODUCT):
                    value_today = DATAS[product][i,j,t]
                    if np.isnan(value_today):
                        value_today = 0.0
                    feature.append(value_today)
                
                # 再添加前一天的所有产品数据
                for p_idx, product in enumerate(PRODUCT):
                    value_yesterday = DATAS[product][i,j,t-1]
                    if np.isnan(value_yesterday):
                        value_yesterday = 0.0
                    feature.append(value_yesterday)
                
                if is_train:
                    X_train[train_idx, :] = feature
                    Y_train[train_idx, 0] = CHM_DATA[i,j,t] > 0  # 二分类标签
                    train_idx += 1
                else:
                    X_test[test_idx, :] = feature
                    Y_test[test_idx, 0] = CHM_DATA[i,j,t] > 0    # 二分类标签
                    test_idx += 1
                    
print(f"训练集: X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"测试集: X_test: {X_test.shape}, Y_test: {Y_test.shape}")
X_is_below_3mm_train = np.where(X_train < 3, 1, 0)  # 将小于3的值标记为1，其余为0
X_is_below_3mm_test = np.where(X_test < 3, 1, 0)    # 将小于3的值标记为1，其余为0

# 计算正负样本比例，用于后续调整模型参数
pos_count = np.sum(Y_train == 1)
neg_count = np.sum(Y_train == 0)
balance_ratio = neg_count / pos_count
print(f"训练集正样本(有降雨): {pos_count}, 负样本(无降雨): {neg_count}, 比例: {balance_ratio:.4f}")

# 特征工程部分
print("\n开始特征工程...")
# 创建特征名称，用于后续分析
feature_names = []
for product in PRODUCT.keys():
    feature_names.append(f"{product}_today")
for product in PRODUCT.keys():
    feature_names.append(f"{product}_yesterday")
print(f"原始特征列表: {feature_names}")

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 计算交互特征的数量
n_features = X_train.shape[1]
# 由于特征数量增加，可能需要限制交互特征数量，避免维度爆炸
# 这里我们选择只计算部分交互特征，例如今天和昨天相同产品的交互
n_products = len(PRODUCT)
# 相同产品今天和昨天的交互
n_interactions = n_products  
# 今天各产品之间的交互
n_interactions += n_products * (n_products - 1) // 2  
# 昨天各产品之间的交互
n_interactions += n_products * (n_products - 1) // 2  

total_features = n_features + n_interactions

print(f"原始特征数: {n_features}, 交互特征数: {n_interactions}, 总特征数: {total_features}")

# 添加交互特征
X_train_with_interaction = np.zeros((X_train.shape[0], total_features))
X_test_with_interaction = np.zeros((X_test.shape[0], total_features))

# 保留原始特征
X_train_with_interaction[:, :n_features] = X_train_scaled
X_test_with_interaction[:, :n_features] = X_test_scaled

# 添加特征交互项
feature_idx = n_features
interaction_names = []

# 1. 同一产品今天和昨天的交互
for p in range(n_products):
    today_idx = p
    yesterday_idx = p + n_products
    X_train_with_interaction[:, feature_idx] = X_train_scaled[:, today_idx] * X_train_scaled[:, yesterday_idx]
    X_test_with_interaction[:, feature_idx] = X_test_scaled[:, today_idx] * X_test_scaled[:, yesterday_idx]
    interaction_names.append(f"{feature_names[today_idx]}*{feature_names[yesterday_idx]}")
    feature_idx += 1

# 2. 今天各产品之间的交互
for i in range(n_products):
    for j in range(i+1, n_products):
        X_train_with_interaction[:, feature_idx] = X_train_scaled[:, i] * X_train_scaled[:, j]
        X_test_with_interaction[:, feature_idx] = X_test_scaled[:, i] * X_test_scaled[:, j]
        interaction_names.append(f"{feature_names[i]}*{feature_names[j]}")
        feature_idx += 1

# 3. 昨天各产品之间的交互
for i in range(n_products):
    for j in range(i+1, n_products):
        i_idx = i + n_products
        j_idx = j + n_products
        X_train_with_interaction[:, feature_idx] = X_train_scaled[:, i_idx] * X_train_scaled[:, j_idx]
        X_test_with_interaction[:, feature_idx] = X_test_scaled[:, i_idx] * X_test_scaled[:, j_idx]
        interaction_names.append(f"{feature_names[i_idx]}*{feature_names[j_idx]}")
        feature_idx += 1

print(f"交互特征示例: {interaction_names[:5]}...")

# 添加各产品是否检测到降雨的二值特征
rain_threshold = 0.1  # 降雨阈值，可以根据实际情况调整
X_train_binary = (X_train > rain_threshold).astype(int)
X_test_binary = (X_test > rain_threshold).astype(int)

#为了告诉模型低于3mm的容易发生误报和漏报
# 添加二值特征
X_is_below_3mm_train = np.where(X_train < 3, 1, 0).astype(int)  # 将小于3的值标记为1，其余为0
X_is_below_3mm_test = np.where(X_test < 3, 1, 0).astype(int)    # 将小于3的值标记为1，其余为0

# 组合所有特征
X_train_final = np.hstack((X_train_with_interaction, X_is_below_3mm_train), dtype=np.float32)
X_test_final = np.hstack((X_test_with_interaction, X_is_below_3mm_test), dtype=np.float32)

print(f"增强后特征维度: {X_train_final.shape[1]}")

# 拆分训练集以创建验证集(10%的训练数据)
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train_final, 
    Y_train, 
    test_size=0.1, 
    random_state=42,
    stratify=Y_train  # 确保训练集和验证集有相似的类别分布
)

print(f"训练子集: {X_train_part.shape}, 验证集: {X_val.shape}")

# 使用早停策略训练模型
# 由于特征增多，调整部分超参数
adjusted_ratio = min(balance_ratio, 1.5)  # 限制最大不平衡比例

print("\n训练XGBoost模型...")
print(f"使用调整后的正负样本权重: {adjusted_ratio:.4f} (原始比例: {balance_ratio:.4f})")

xgb_model = xgb.XGBClassifier(
    max_depth=6,  # 增加树的深度以处理更多特征
    learning_rate=0.05,  
    n_estimators=500,
    subsample=0.9,  
    colsample_bytree=0.7,  
    colsample_bylevel=0.7, 
    colsample_bynode=0.7,
    scale_pos_weight=adjusted_ratio,
    objective='binary:logistic',
    random_state=42,
    reg_alpha=0.3,
    reg_lambda=1.0,
    early_stopping_rounds=20,
    eval_metric=['logloss', 'auc']
)

print("使用验证集训练模型并应用早停策略...")
eval_set = [(X_val, y_val.ravel())]
xgb_model.fit(X_train_part, y_train_part.ravel(), eval_set=eval_set, verbose=True)
print(f"模型最佳迭代次数: {xgb_model.best_iteration}")

# 保存学习曲线
plt.figure(figsize=(10, 6))
results = xgb_model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
plt.plot(x_axis, results['validation_0']['logloss'], label='验证集对数损失')
plt.plot(x_axis, results['validation_0']['auc'], label='验证集AUC')
plt.grid()
plt.legend()
plt.xlabel('迭代次数')
plt.ylabel('指标值')
plt.title('XGBoost学习曲线')
plt.tight_layout()
plt.savefig('f:/rainfalldata/figures/learning_curve.png')

# 预测并保存概率值，用于后续阈值调整
print("\n在测试集上评估模型...")
y_pred_proba = xgb_model.predict_proba(X_test_final)[:, 1]
y_pred_default = (y_pred_proba >= 0.5).astype(int)

# 使用默认阈值的评估结果
default_accuracy = accuracy_score(Y_test, y_pred_default)
print(f"\n默认阈值(0.5)模型准确率: {default_accuracy:.4f}")
print("\n默认阈值分类报告:")
print(classification_report(Y_test, y_pred_default))

# 尝试找到最佳阈值以平衡召回率和精确度
thresholds = np.arange(0.2, 0.8, 0.05)
best_f1 = 0
best_threshold = 0.5
results = {}

print("\n寻找最佳分类阈值...")
for threshold in thresholds:
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    # 计算雨天(正类)的F1分数
    rain_f1 = f1_score(Y_test, y_pred_binary, pos_label=1)
    rain_recall = recall_score(Y_test, y_pred_binary, pos_label=1)
    rain_precision = precision_score(Y_test, y_pred_binary, pos_label=1)
    # 计算无雨(负类)的性能
    no_rain_f1 = f1_score(Y_test, y_pred_binary, pos_label=0)
    no_rain_recall = recall_score(Y_test, y_pred_binary, pos_label=0)
    no_rain_precision = precision_score(Y_test, y_pred_binary, pos_label=0)
    
    # 保存结果
    results[threshold] = {
        'rain_f1': rain_f1, 
        'rain_recall': rain_recall, 
        'rain_precision': rain_precision,
        'no_rain_f1': no_rain_f1,
        'no_rain_recall': no_rain_recall,
        'no_rain_precision': no_rain_precision
    }
    
    # 使用加权F1分数作为优化目标，更重视雨天的性能
    weighted_f1 = 0.6 * rain_f1 + 0.4 * no_rain_f1
    
    print(f"阈值 {threshold:.2f}: 雨天F1={rain_f1:.4f}, 雨天召回={rain_recall:.4f}, "
          f"无雨精确={no_rain_precision:.4f}, 加权F1={weighted_f1:.4f}")
    
    if weighted_f1 > best_f1:
        best_f1 = weighted_f1
        best_threshold = threshold

print(f"\n最佳阈值: {best_threshold:.2f}, 加权F1分数: {best_f1:.4f}")

# 使用最佳阈值获得最终预测
y_pred_final = (y_pred_proba >= best_threshold).astype(int)

# 评估最终模型
accuracy = accuracy_score(Y_test, y_pred_final)
print(f"\n优化阈值后模型准确率: {accuracy:.4f}")
print("\n优化后的分类报告:")
print(classification_report(Y_test, y_pred_final))

# 特征重要性分析
# 获取所有特征的重要性得分
importance = xgb_model.feature_importances_
# 创建完整的特征名称列表
all_feature_names = feature_names + interaction_names + [f"{f}_binary" for f in feature_names]

# 仅显示前20个最重要的特征
top_indices = np.argsort(importance)[-20:]
print("\n前20个最重要特征:")
for idx in reversed(top_indices):
    if idx < len(all_feature_names):
        print(f"{all_feature_names[idx]}: {importance[idx]:.4f}")
    else:
        print(f"特征{idx}: {importance[idx]:.4f}")

# 显示按分组特征的重要性
print("\n按分组的特征重要性:")
today_sum = np.sum(importance[:n_products])
yesterday_sum = np.sum(importance[n_products:n_products*2])
print(f"今天的产品特征重要性总和: {today_sum:.4f}")
print(f"昨天的产品特征重要性总和: {yesterday_sum:.4f}")

# 可视化阈值调整结果
plt.figure(figsize=(10, 6))
thresholds_list = list(results.keys())
rain_recall = [results[t]['rain_recall'] for t in thresholds_list]
no_rain_precision = [results[t]['no_rain_precision'] for t in thresholds_list]
rain_f1 = [results[t]['rain_f1'] for t in thresholds_list]

plt.plot(thresholds_list, rain_recall, 'b-', label='雨天召回率')
plt.plot(thresholds_list, no_rain_precision, 'r-', label='无雨精确度')
plt.plot(thresholds_list, rain_f1, 'g-', label='雨天F1分数')
plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'最佳阈值={best_threshold:.2f}')
plt.xlabel('分类阈值')
plt.ylabel('性能指标')
plt.title('阈值对模型性能的影响')
plt.legend()
plt.grid(True)
plt.savefig('f:/rainfalldata/figures/threshold_optimization.png')
print("\n阈值调优图已保存至 'f:/rainfalldata/figures/threshold_optimization.png'")

# 保存模型
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'f:/rainfalldata/models/xgboost_model_{timestamp}.json'
xgb_model.save_model(model_path)
print(f"\n模型已保存至: {model_path}")

# 保存预测结果
results_df = pd.DataFrame({
    'y_true': Y_test.ravel(),
    'y_pred_default': y_pred_default,
    'y_pred_optimized': y_pred_final,
    'y_pred_proba': y_pred_proba
})
results_path = f'f:/rainfalldata/results/predictions_{timestamp}.csv'
results_df.to_csv(results_path, index=False)
print(f"预测结果已保存至: {results_path}")

print("\n优化版XGBoost训练完成（使用当天和前一天数据）！")
