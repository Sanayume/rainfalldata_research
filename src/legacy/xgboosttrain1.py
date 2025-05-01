"""
优化版XGBoost降水预测训练脚本
包含特征工程、交互特征、早停策略、阈值优化和五倍交叉验证
加入前一天数据作为特征
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
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
print("优化版XGBoost降水预测训练脚本 - 包含高级特征工程、阈值优化和五倍交叉验证")
print("=" * 70)
print("增强版：使用当天和前一天的数据作为特征")

# 定义数据文件路径
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

# 在数据加载后立即转换为float32
for key in PRODUCT.keys():
    DATAS[key] = DATAS[key].astype(np.float32)
CHM_DATA = CHM_DATA.astype(np.float32)

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
X_train = np.zeros((trainsample, len(PRODUCT)*2), dtype=np.float32)
Y_train = np.zeros((trainsample, 1), dtype=np.float32)
X_test = np.zeros((testsample, len(PRODUCT)*2), dtype=np.float32)
Y_test = np.zeros((testsample, 1), dtype=np.float32)

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

# 优化特征工程部分，减少内存使用
def create_interaction_features(X, scaler=None):
    """分批处理创建交互特征"""
    batch_size = 100000  # 根据可用内存调整批次大小
    n_samples = X.shape[0]
    n_products = len(PRODUCT)
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # 计算输出特征维度
    n_interactions = (n_products +  # 同一产品今天和昨天的交互
                     n_products * (n_products - 1) // 2 +  # 今天产品间交互
                     n_products * (n_products - 1) // 2)   # 昨天产品间交互
    
    total_features = X.shape[1] + n_interactions
    X_with_interaction = np.zeros((n_samples, total_features), dtype=np.float32)
    
    # 分批处理
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_scaled = X_scaled[start_idx:end_idx]
        
        # 复制原始特征
        X_with_interaction[start_idx:end_idx, :X.shape[1]] = batch_scaled
        
        feature_idx = X.shape[1]
        
        # 添加交互特征
        for p in range(n_products):
            today_idx = p
            yesterday_idx = p + n_products
            X_with_interaction[start_idx:end_idx, feature_idx] = (
                batch_scaled[:, today_idx] * batch_scaled[:, yesterday_idx]
            )
            feature_idx += 1
            
        # 今天产品间交互
        for i in range(n_products):
            for j in range(i+1, n_products):
                X_with_interaction[start_idx:end_idx, feature_idx] = (
                    batch_scaled[:, i] * batch_scaled[:, j]
                )
                feature_idx += 1
        
        # 昨天产品间交互
        for i in range(n_products):
            for j in range(i+1, n_products):
                i_idx = i + n_products
                j_idx = j + n_products
                X_with_interaction[start_idx:end_idx, feature_idx] = (
                    batch_scaled[:, i_idx] * batch_scaled[:, j_idx]
                )
                feature_idx += 1
    
    return X_with_interaction, scaler

# 使用优化后的特征工程函数
print("\n开始特征工程(优化内存使用)...")
X_train_final, scaler = create_interaction_features(X_train)
X_test_final, _ = create_interaction_features(X_test, scaler)

# 添加二值特征
rain_threshold = 0.1
X_train_binary = (X_train > rain_threshold).astype(np.float32)
X_test_binary = (X_test > rain_threshold).astype(np.float32)

# 使用列连接代替hstack以节省内存
X_train_final = np.column_stack([X_train_final, X_train_binary])
X_test_final = np.column_stack([X_test_final, X_test_binary])

# 清理不需要的大数组
del X_train_binary, X_test_binary
del X_train, X_test
import gc
gc.collect()

# 五倍交叉验证设置
print("\n开始五倍交叉验证训练...")
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# 存储每一折的性能指标
cv_results = {
    'train_accuracy': [], 
    'val_accuracy': [],
    'train_f1': [], 
    'val_f1': [],
    'best_iteration': [],
    'best_threshold': []
}

# 存储每一折的模型
fold_models = []
fold_thresholds = []

# 使用早停策略训练模型
# 由于特征增多，调整部分超参数
adjusted_ratio = min(balance_ratio, 1.5)  # 限制最大不平衡比例

print(f"\n正负样本调整权重: {adjusted_ratio:.4f} (原始比例: {balance_ratio:.4f})")

# 开始5折交叉验证训练
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_final)):
    print(f"\n{'='*30} 第 {fold+1}/{n_folds} 折训练 {'='*30}")
    
    # 使用索引切片而不是复制数据
    xgb_train = xgb.DMatrix(X_train_final[train_idx], label=Y_train[train_idx])
    xgb_val = xgb.DMatrix(X_train_final[val_idx], label=Y_train[val_idx])
    
    # 其余代码保持不变...

    # 创建XGBoost模型
    xgb_model = xgb.XGBClassifier(
        max_depth=6,  
        learning_rate=0.05,  
        n_estimators=500,
        subsample=0.9,  
        colsample_bytree=0.7,
        colsample_bylevel=0.7, 
        colsample_bynode=0.7,
        scale_pos_weight=adjusted_ratio,
        objective='binary:logistic',
        random_state=42+fold,  # 每一折使用不同的随机种子
        reg_alpha=0.3,
        reg_lambda=1.0,
        early_stopping_rounds=20,
        eval_metric=['logloss', 'auc']
    )

    # 训练模型
    print(f"第 {fold+1} 折 - 使用验证集训练模型并应用早停策略...")
    X_train_fold = X_train_final[train_idx]
    y_train_fold = Y_train[train_idx]
    X_val_fold = X_train_final[val_idx]
    y_val_fold = Y_train[val_idx]
    
    eval_set = [(X_val_fold, y_val_fold.ravel())]
    xgb_model.fit(X_train_fold, y_train_fold.ravel(), eval_set=eval_set, verbose=False)
    
    # 记录最佳迭代次数
    best_iter = xgb_model.best_iteration
    cv_results['best_iteration'].append(best_iter)
    print(f"第 {fold+1} 折 - 最佳迭代次数: {best_iter}")
    
    # 评估训练集性能
    y_train_pred = xgb_model.predict(X_train_fold)
    train_acc = accuracy_score(y_train_fold, y_train_pred)
    train_f1 = f1_score(y_train_fold, y_train_pred, pos_label=1)
    cv_results['train_accuracy'].append(train_acc)
    cv_results['train_f1'].append(train_f1)
    
    # 评估验证集性能
    y_val_pred = xgb_model.predict(X_val_fold)
    val_acc = accuracy_score(y_val_fold, y_val_pred)
    val_f1 = f1_score(y_val_fold, y_val_pred, pos_label=1)
    cv_results['val_accuracy'].append(val_acc)
    cv_results['val_f1'].append(val_f1)
    
    # 优化阈值(在验证集上)
    print(f"\n第 {fold+1} 折 - 寻找最佳分类阈值...")
    y_val_proba = xgb_model.predict_proba(X_val_fold)[:, 1]
    thresholds = np.arange(0.2, 0.8, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_val_binary = (y_val_proba >= threshold).astype(int)
        rain_f1 = f1_score(y_val_fold, y_val_binary, pos_label=1)
        rain_recall = recall_score(y_val_fold, y_val_binary, pos_label=1)
        no_rain_precision = precision_score(y_val_fold, y_val_binary, pos_label=0)
        
        # 使用加权F1分数作为优化目标
        weighted_f1 = 0.6 * rain_f1 + 0.4 * no_rain_precision
        
        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            best_threshold = threshold
    
    cv_results['best_threshold'].append(best_threshold)
    fold_thresholds.append(best_threshold)
    print(f"第 {fold+1} 折 - 最佳阈值: {best_threshold:.2f}")
    
    # 使用最佳阈值重新评估验证集
    y_val_optimized = (y_val_proba >= best_threshold).astype(int)
    val_opt_acc = accuracy_score(y_val_fold, y_val_optimized)
    val_opt_f1 = f1_score(y_val_fold, y_val_optimized, pos_label=1)
    print(f"第 {fold+1} 折 - 优化阈值后验证集准确率: {val_opt_acc:.4f}, F1: {val_opt_f1:.4f}")
    
    # 打印详细分类报告
    print(f"\n第 {fold+1} 折 - 验证集分类报告:")
    print(classification_report(y_val_fold, y_val_optimized))
    
    # 保存此折的模型
    fold_models.append(xgb_model)
    
    # 每一折训练后绘制学习曲线
    plt.figure(figsize=(8, 5))
    results = xgb_model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    plt.plot(x_axis, results['validation_0']['logloss'], label='验证集对数损失')
    plt.plot(x_axis, results['validation_0']['auc'], label='验证集AUC')
    plt.grid()
    plt.legend()
    plt.xlabel('迭代次数')
    plt.ylabel('指标值')
    plt.title(f'第 {fold+1} 折 XGBoost学习曲线')
    plt.tight_layout()
    plt.savefig(f'f:/rainfalldata/figures/learning_curve_fold{fold+1}.png')
    plt.close()

# 打印交叉验证结果摘要
print("\n" + "="*50)
print("五折交叉验证结果摘要:")
print("="*50)
print(f"平均训练集准确率: {np.mean(cv_results['train_accuracy']):.4f} ± {np.std(cv_results['train_accuracy']):.4f}")
print(f"平均验证集准确率: {np.mean(cv_results['val_accuracy']):.4f} ± {np.std(cv_results['val_accuracy']):.4f}")
print(f"平均训练集F1分数: {np.mean(cv_results['train_f1']):.4f} ± {np.std(cv_results['train_f1']):.4f}")
print(f"平均验证集F1分数: {np.mean(cv_results['val_f1']):.4f} ± {np.std(cv_results['val_f1']):.4f}")
print(f"平均最佳迭代次数: {np.mean(cv_results['best_iteration']):.1f}")
print(f"平均最佳阈值: {np.mean(cv_results['best_threshold']):.2f}")

# 可视化交叉验证结果
plt.figure(figsize=(10, 6))
fold_nums = list(range(1, n_folds+1))
plt.plot(fold_nums, cv_results['train_accuracy'], 'b-', label='训练准确率')
plt.plot(fold_nums, cv_results['val_accuracy'], 'r-', label='验证准确率')
plt.plot(fold_nums, cv_results['train_f1'], 'g--', label='训练F1')
plt.plot(fold_nums, cv_results['val_f1'], 'y--', label='验证F1')
plt.xlabel('折数')
plt.ylabel('性能指标')
plt.title('交叉验证性能指标')
plt.legend()
plt.grid(True)
plt.savefig('f:/rainfalldata/figures/cv_performance.png')
plt.close()

# 选择最佳模型和阈值进行最终评估
# 这里我们选择验证集F1分数最高的模型
best_fold_idx = np.argmax(cv_results['val_f1'])
best_model = fold_models[best_fold_idx]
best_overall_threshold = fold_thresholds[best_fold_idx]

print(f"\n选择第 {best_fold_idx+1} 折的模型作为最终模型 (验证F1={cv_results['val_f1'][best_fold_idx]:.4f})")
print(f"最终选择的阈值: {best_overall_threshold:.2f}")

# 在测试集上评估最终模型
print("\n在测试集上评估最终模型...")
y_test_proba = best_model.predict_proba(X_test_final)[:, 1]
y_test_pred = (y_test_proba >= best_overall_threshold).astype(int)

# 评估最终模型
test_accuracy = accuracy_score(Y_test, y_test_pred)
test_f1 = f1_score(Y_test, y_test_pred, pos_label=1)
test_recall = recall_score(Y_test, y_test_pred, pos_label=1)
test_precision = precision_score(Y_test, y_test_pred, pos_label=1)

print(f"\n测试集最终结果:")
print(f"准确率: {test_accuracy:.4f}")
print(f"F1分数: {test_f1:.4f}")
print(f"召回率: {test_recall:.4f}")
print(f"精确度: {test_precision:.4f}")

print("\n测试集详细分类报告:")
print(classification_report(Y_test, y_test_pred))

# 使用集成方法获得更好的预测结果
print("\n使用所有交叉验证模型进行集成预测...")
ensemble_proba = np.zeros(Y_test.shape[0])

# 获取所有模型的平均预测概率
for model in fold_models:
    ensemble_proba += model.predict_proba(X_test_final)[:, 1]
ensemble_proba /= n_folds

# 使用平均阈值进行最终预测
avg_threshold = np.mean(fold_thresholds)
ensemble_pred = (ensemble_proba >= avg_threshold).astype(int)

# 评估集成模型
ensemble_accuracy = accuracy_score(Y_test, ensemble_pred)
ensemble_f1 = f1_score(Y_test, ensemble_pred, pos_label=1)
ensemble_recall = recall_score(Y_test, ensemble_pred, pos_label=1)
ensemble_precision = precision_score(Y_test, ensemble_pred, pos_label=1)

print(f"\n测试集集成模型结果 (阈值={avg_threshold:.2f}):")
print(f"集成准确率: {ensemble_accuracy:.4f}")
print(f"集成F1分数: {ensemble_f1:.4f}")
print(f"集成召回率: {ensemble_recall:.4f}")
print(f"集成精确度: {ensemble_precision:.4f}")

print("\n集成模型分类报告:")
print(classification_report(Y_test, ensemble_pred))

# 保存最终模型和集成结果
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 保存所有单个模型
for i, model in enumerate(fold_models):
    model_path = f'f:/rainfalldata/models/xgboost_fold{i+1}_{timestamp}.json'
    model.save_model(model_path)

# 保存预测结果
results_df = pd.DataFrame({
    'y_true': Y_test.ravel(),
    'y_pred_best_model': y_test_pred,
    'y_pred_ensemble': ensemble_pred,
    'y_proba_best_model': y_test_proba,
    'y_proba_ensemble': ensemble_proba
})
results_path = f'f:/rainfalldata/results/cv_predictions_{timestamp}.csv'
results_df.to_csv(results_path, index=False)
print(f"\n预测结果已保存至: {results_path}")

# 保存交叉验证结果
cv_summary = pd.DataFrame(cv_results)
cv_summary.index = [f"Fold_{i+1}" for i in range(n_folds)]
cv_summary_path = f'f:/rainfalldata/results/cv_summary_{timestamp}.csv'
cv_summary.to_csv(cv_summary_path)
print(f"交叉验证结果已保存至: {cv_summary_path}")

print("\n优化版XGBoost交叉验证训练完成！")

