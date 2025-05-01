"""
优化版XGBoost降水预测训练脚本
包含特征工程、交互特征、早停策略、阈值优化和五倍交叉验证
加入前一天数据作为特征
内存优化版本与GPU加速支持
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, classification_report,
                            f1_score, recall_score, precision_score)
from sklearn.preprocessing import StandardScaler
import sys
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
import gc  # 垃圾回收
from tqdm import tqdm  # 进度条

# 检查是否可以使用GPU
try:
    import cupy as cp
    HAS_GPU = True
    print("已检测到GPU支持，将使用GPU加速")
except ImportError:
    HAS_GPU = False
    cp = np
    print("未检测到cupy或cudf，将使用CPU模式")

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('f:/rainfalldata/figures', exist_ok=True)
os.makedirs('f:/rainfalldata/results', exist_ok=True)
os.makedirs('f:/rainfalldata/models', exist_ok=True)

# 抑制不相关警告
warnings.filterwarnings('ignore')

print("=" * 70)
print("优化版XGBoost降水预测训练脚本 - 内存优化版 + GPU加速")
print("=" * 70)
print("增强版：使用当天和前一天的数据作为特征")

# 配置使用较少内存的数据类型
DTYPE = np.float32  # 使用float32替代float64，减少一半内存使用

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
            # 加载时立即转为float32以节省内存
            DATAS[key] = loadmat(filepath)["data"].astype(DTYPE)
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
valid_point_count = np.sum(valid_point)
print(f"有效点数: {valid_point_count}")

# 使用当天+前一天数据，因此从第二天开始训练
trainsample = valid_point_count * (ntime-1-366)  # 使用前几年作为训练集，减去第一天
testsample = valid_point_count * 366            # 使用最后一年作为测试集

# 内存优化: 分批次处理数据，避免一次性生成过大的数组
print("将使用批处理方式构建特征...")

# 计算特征数量
n_products = len(PRODUCT)
n_features_per_day = n_products  # 每天的特征数量
n_binary_features = n_features_per_day * 2  # 二值特征数量（今天和昨天）
n_features = n_features_per_day * 2  # 今天和昨天的原始特征

# 使用函数生成特征，避免占用过多内存
def generate_data_batch(start_idx, end_idx, data_type="train"):
    """批量生成特征数据"""
    # 确定要处理的时间范围
    if data_type == "train":
        time_start = 1  # 从第二天开始
        time_end = ntime - 366  # 训练集结束点
    else:  # test
        time_start = ntime - 366  # 测试集开始点
        time_end = ntime  # 到最后一天
        
    batch_size = end_idx - start_idx
    # 只创建当前批次所需要的数据
    X_batch = np.zeros((batch_size, n_features), dtype=DTYPE)
    y_batch = np.zeros((batch_size, 1), dtype=DTYPE)
    
    batch_idx = 0
    for t in range(time_start, time_end):
        for i in range(nlat):
            for j in range(nlon):
                if MASK[i,j] == 1:
                    # 只处理当前批次范围内的数据
                    if batch_idx >= start_idx and batch_idx < end_idx:
                        # 提取特征
                        feature = []
                        # 当天的产品数据
                        for product in PRODUCT:
                            value_today = DATAS[product][i,j,t]
                            feature.append(float('0.0') if np.isnan(value_today) else float(value_today))
                        
                        # 前一天的产品数据
                        for product in PRODUCT:
                            value_yesterday = DATAS[product][i,j,t-1]
                            feature.append(float('0.0') if np.isnan(value_yesterday) else float(value_yesterday))
                            
                        # 保存到批次数组
                        X_batch[batch_idx-start_idx] = feature
                        y_batch[batch_idx-start_idx, 0] = float(CHM_DATA[i,j,t] > 0)
                        
                    batch_idx += 1
                    if batch_idx >= end_idx:
                        return X_batch, y_batch
    
    return X_batch, y_batch

# 批处理参数
BATCH_SIZE = 100000  # 每批处理的样本数，根据可用内存调整

# 特征工程函数，对批次数据进行特征工程
def add_features_to_batch(X_batch, scaler=None):
    """对批次数据添加特征，节省内存"""
    # 标准化
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_batch)
    else:
        X_scaled = scaler.transform(X_batch)
    
    # 计算需要的交互特征
    n_products = len(PRODUCT)
    
    # 分配结果数组 (原始特征 + 交互特征)
    n_interactions = n_products + (n_products * (n_products - 1))
    result = np.zeros((X_batch.shape[0], n_features + n_interactions), dtype=DTYPE)
    
    # 复制原始标准化特征
    result[:, :n_features] = X_scaled
    
    # 添加特征交互项
    feature_idx = n_features
    
    # 1. 同一产品今天和昨天的交互
    for p in range(n_products):
        today_idx = p
        yesterday_idx = p + n_products
        result[:, feature_idx] = X_scaled[:, today_idx] * X_scaled[:, yesterday_idx]
        feature_idx += 1
    
    # 2. 今天各产品之间的交互
    for i in range(n_products):
        for j in range(i+1, n_products):
            result[:, feature_idx] = X_scaled[:, i] * X_scaled[:, j]
            feature_idx += 1
    
    # 3. 昨天各产品之间的交互
    for i in range(n_products):
        for j in range(i+1, n_products):
            i_idx = i + n_products
            j_idx = j + n_products
            result[:, feature_idx] = X_scaled[:, i_idx] * X_scaled[:, j_idx]
            feature_idx += 1
    
    # 添加二值特征
    rain_threshold = 0.1
    binary_features = (X_batch > rain_threshold).astype(DTYPE)
    
    # 将特征与二值特征组合
    final_result = np.hstack((result, binary_features))
    return final_result, scaler

# 准备XGBoost参数
def get_xgb_params(balance_ratio):
    """获取XGBoost参数"""
    adjusted_ratio = min(balance_ratio, 1.5)  # 限制最大不平衡比例
    
    params = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'colsample_bynode': 0.7,
        'scale_pos_weight': adjusted_ratio,
        'objective': 'binary:logistic',
        'random_state': 42,
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 20,
        'eval_metric': ['logloss', 'auc'],
    }
    
    # 如果有GPU，启用GPU加速
    if HAS_GPU:
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
        params['predictor'] = 'gpu_predictor'
    
    return params

# 准备特征名称，主要用于后续的特征重要性分析
feature_names = []
for product in PRODUCT.keys():
    feature_names.append(f"{product}_today")
for product in PRODUCT.keys():
    feature_names.append(f"{product}_yesterday")

# 构建交互特征名称
interaction_names = []
n_products = len(PRODUCT)
product_names = list(PRODUCT.keys())

# 1. 同一产品今天和昨天的交互
for p in range(n_products):
    today_name = f"{product_names[p]}_today"
    yesterday_name = f"{product_names[p]}_yesterday"
    interaction_names.append(f"{today_name}*{yesterday_name}")

# 2. 今天各产品之间的交互
for i in range(n_products):
    for j in range(i+1, n_products):
        name_i = f"{product_names[i]}_today"
        name_j = f"{product_names[j]}_today"
        interaction_names.append(f"{name_i}*{name_j}")

# 3. 昨天各产品之间的交互
for i in range(n_products):
    for j in range(i+1, n_products):
        name_i = f"{product_names[i]}_yesterday"
        name_j = f"{product_names[j]}_yesterday"
        interaction_names.append(f"{name_i}*{name_j}")

print(f"特征数量: 原始={n_features}, 交互={len(interaction_names)}, 二值={n_binary_features}")

# 计算训练集正负样本比例
# 我们需要一个小样本来确定比例
sample_train_X, sample_train_y = generate_data_batch(0, min(10000, trainsample), "train")
pos_count = np.sum(sample_train_y == 1)
neg_count = np.sum(sample_train_y == 0)
balance_ratio = float(neg_count) / float(max(1, pos_count))
print(f"训练集采样正负比例: {balance_ratio:.4f} (基于{len(sample_train_y)}个样本)")
del sample_train_X, sample_train_y
gc.collect()

# 获取XGBoost参数
xgb_params = get_xgb_params(balance_ratio)

# 初始化StandardScaler
scaler = StandardScaler()

# 首先拟合scaler - 使用一小批训练数据来正确初始化scaler
print("\n拟合标准化器...")
scaler_fit_batch_size = min(50000, trainsample)
X_scaler_fit_batch, _ = generate_data_batch(0, scaler_fit_batch_size, "train")
scaler.fit(X_scaler_fit_batch)
print(f"已使用{scaler_fit_batch_size}个样本拟合标准化器")

# 释放内存
del X_scaler_fit_batch
gc.collect()

# 准备交叉验证
print("\n准备五折交叉验证...")
n_folds = 5
# 计算每折大小
fold_size = trainsample // n_folds
# 每个fold只是索引范围，避免生成大数组
fold_indices = [(i*fold_size, (i+1)*fold_size) for i in range(n_folds)]

# 存储每一折的性能指标
cv_results = {
    'train_accuracy': [], 
    'val_accuracy': [],
    'train_f1': [], 
    'val_f1': [],
    'best_iteration': [],
    'best_threshold': []
}

fold_models = []
fold_thresholds = []

# 准备测试数据 - 分批加载以节省内存
print("\n准备测试集...")
test_batches = []
batch_count = (testsample + BATCH_SIZE - 1) // BATCH_SIZE  # 向上取整

for b in tqdm(range(batch_count)):
    start_idx = b * BATCH_SIZE
    end_idx = min((b + 1) * BATCH_SIZE, testsample)
    X_test_batch, Y_test_batch = generate_data_batch(start_idx, end_idx, "test")
    # 应用特征工程 - 使用已经拟合好的scaler
    X_test_batch_final, _ = add_features_to_batch(X_test_batch, scaler)
    test_batches.append((X_test_batch_final, Y_test_batch))
    # 释放内存
    del X_test_batch
    gc.collect()

# 开始交叉验证训练
print("\n开始五折交叉验证训练...")

for fold in range(n_folds):
    print(f"\n{'='*30} 第 {fold+1}/{n_folds} 折训练 {'='*30}")
    
    # 确定验证集范围
    val_start, val_end = fold_indices[fold]
    
    # 创建训练集和验证集
    # 1. 首先加载验证集数据
    print(f"加载第 {fold+1} 折验证集数据...")
    X_val, y_val = generate_data_batch(val_start, val_end, "train")
    
    # 2. 应用特征工程到验证集 - 使用已经拟合好的 scaler
    X_val_final, _ = add_features_to_batch(X_val, scaler)

    # 3. 准备训练集数据 (使用其他折)
    print(f"准备第 {fold+1} 折训练集数据...")
    train_data = []
    
    # 训练集由其他折组成
    for other_fold in range(n_folds):
        if other_fold != fold:
            other_start, other_end = fold_indices[other_fold]
            batch_size = other_end - other_start
            # 将大的训练集分成更小的批次处理
            sub_batch_size = BATCH_SIZE
            for sub_batch in range(0, batch_size, sub_batch_size):
                sub_start = other_start + sub_batch
                sub_end = min(other_start + sub_batch + sub_batch_size, other_end)
                
                X_train_sub, y_train_sub = generate_data_batch(sub_start, sub_end, "train")
                X_train_sub_final, _ = add_features_to_batch(X_train_sub, scaler)
                
                train_data.append((X_train_sub_final, y_train_sub))
                
                # 释放内存
                del X_train_sub
                gc.collect()
    
    # 4. 创建DMatrix对象 - 更高效的XGBoost数据格式
    print(f"创建第 {fold+1} 折的XGBoost数据结构...")
    
    # 训练XGBoost模型
    model = xgb.XGBClassifier(**xgb_params)
    print(f"第 {fold+1} 折 - 使用验证集训练模型并应用早停策略...")
    
    # 重要修复: 确保eval_set使用验证集数据，而不是训练数据 
    eval_set = [(X_val_final, y_val.ravel())]
    
    # 5. 使用批处理方式训练
    # 第一批训练数据作为初始训练
    X_batch_first, y_batch_first = train_data[0]
    
    model.fit(X_batch_first, y_batch_first.ravel(), 
              eval_set=eval_set,  # 使用验证集进行早停
              verbose=False)
    
    # 使用部分拟合继续训练其他批次
    for i, (X_batch, y_batch) in enumerate(train_data[1:], 1):
        print(f"第 {fold+1} 折 - 训练批次 {i+1}/{len(train_data)}")
        # 注意: 对于增量训练，我们仍然需要eval_set，但可以不使用早停
        # xgb_model参数告诉XGBoost继续训练现有模型
        model.fit(X_batch, y_batch.ravel(), 
                 xgb_model=model.get_booster(),
                 eval_set=eval_set,  # 仍然需要评估集
                 verbose=False)
    
    # 记录最佳迭代次数
    best_iter = model.best_iteration if hasattr(model, 'best_iteration') else model.get_num_boosting_rounds()
    cv_results['best_iteration'].append(best_iter)
    
    # 评估训练集 - 使用小批次避免内存问题
    train_preds = []
    train_true = []
    
    for X_batch, y_batch in train_data:
        batch_pred = model.predict(X_batch)
        train_preds.extend(batch_pred)
        train_true.extend(y_batch.ravel())
        
        # 释放内存
        del X_batch
        gc.collect()
    
    train_acc = accuracy_score(train_true, train_preds)
    train_f1 = f1_score(train_true, train_preds, pos_label=1)
    cv_results['train_accuracy'].append(train_acc)
    cv_results['train_f1'].append(train_f1)
    
    # 评估验证集性能 - 这里已经有X_val_final，无需重新加载
    # 直接使用当前内存中的验证数据进行评估
    y_val_pred = model.predict(X_val_final)
    y_val_proba = model.predict_proba(X_val_final)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, pos_label=1)
    cv_results['val_accuracy'].append(val_acc)
    cv_results['val_f1'].append(val_f1)
    
    print(f"第 {fold+1} 折 - 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
    print(f"第 {fold+1} 折 - 训练F1: {train_f1:.4f}, 验证F1: {val_f1:.4f}")
    
    # 优化阈值(在验证集上)
    print(f"\n第 {fold+1} 折 - 寻找最佳分类阈值...")
    # y_val_proba已在上面计算，无需重新计算
    thresholds = np.arange(0.2, 0.8, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_val_binary = (y_val_proba >= threshold).astype(int)
        rain_f1 = f1_score(y_val, y_val_binary, pos_label=1)
        rain_recall = recall_score(y_val, y_val_binary, pos_label=1)
        no_rain_precision = precision_score(y_val, y_val_binary, pos_label=0)
        
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
    val_opt_acc = accuracy_score(y_val, y_val_optimized)
    val_opt_f1 = f1_score(y_val, y_val_optimized, pos_label=1)
    print(f"第 {fold+1} 折 - 优化阈值后验证集准确率: {val_opt_acc:.4f}, F1: {val_opt_f1:.4f}")
    
    # 现在可以安全地释放验证集内存
    del X_val, X_val_final
    gc.collect()
    
    # 保存此折的模型
    fold_models.append(model)
    
    # 清理内存
    del train_data
    gc.collect()

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
best_fold_idx = np.argmax(cv_results['val_f1'])
best_model = fold_models[best_fold_idx]
best_overall_threshold = fold_thresholds[best_fold_idx]

print(f"\n选择第 {best_fold_idx+1} 折的模型作为最终模型 (验证F1={cv_results['val_f1'][best_fold_idx]:.4f})")
print(f"最终选择的阈值: {best_overall_threshold:.2f}")

# 在测试集上评估最终模型 - 分批处理以节省内存
print("\n在测试集上评估最终模型...")
y_test_all = []
y_test_pred_all = []
y_test_proba_all = []

for X_test_batch, Y_test_batch in test_batches:
    # 获取预测
    y_test_proba_batch = best_model.predict_proba(X_test_batch)[:, 1]
    y_test_pred_batch = (y_test_proba_batch >= best_overall_threshold).astype(int)
    
    # 收集结果
    y_test_all.extend(Y_test_batch.ravel())
    y_test_pred_all.extend(y_test_pred_batch)
    y_test_proba_all.extend(y_test_proba_batch)

# 转换为数组
y_test_all = np.array(y_test_all)
y_test_pred_all = np.array(y_test_pred_all)
y_test_proba_all = np.array(y_test_proba_all)

# 评估最终模型
test_accuracy = accuracy_score(y_test_all, y_test_pred_all)
test_f1 = f1_score(y_test_all, y_test_pred_all, pos_label=1)
test_recall = recall_score(y_test_all, y_test_pred_all, pos_label=1)
test_precision = precision_score(y_test_all, y_test_pred_all, pos_label=1)

print(f"\n测试集最终结果:")
print(f"准确率: {test_accuracy:.4f}")
print(f"F1分数: {test_f1:.4f}")
print(f"召回率: {test_recall:.4f}")
print(f"精确度: {test_precision:.4f}")

print("\n测试集详细分类报告:")
print(classification_report(y_test_all, y_test_pred_all))

# 使用集成方法获得更好的预测结果
print("\n使用所有交叉验证模型进行集成预测...")
ensemble_proba = np.zeros(len(y_test_all))

# 获取所有模型的平均预测概率
for model_idx, model in enumerate(fold_models):
    print(f"计算模型 {model_idx+1}/{n_folds} 的预测结果...")
    model_proba = []
    
    for X_test_batch, _ in test_batches:
        batch_proba = model.predict_proba(X_test_batch)[:, 1]
        model_proba.extend(batch_proba)
    
    ensemble_proba += np.array(model_proba)

ensemble_proba /= n_folds

# 使用平均阈值进行最终预测
avg_threshold = np.mean(fold_thresholds)
ensemble_pred = (ensemble_proba >= avg_threshold).astype(int)

# 评估集成模型
ensemble_accuracy = accuracy_score(y_test_all, ensemble_pred)
ensemble_f1 = f1_score(y_test_all, ensemble_pred, pos_label=1)
ensemble_recall = recall_score(y_test_all, ensemble_pred, pos_label=1)
ensemble_precision = precision_score(y_test_all, ensemble_pred, pos_label=1)

print(f"\n测试集集成模型结果 (阈值={avg_threshold:.2f}):")
print(f"集成准确率: {ensemble_accuracy:.4f}")
print(f"集成F1分数: {ensemble_f1:.4f}")
print(f"集成召回率: {ensemble_recall:.4f}")
print(f"集成精确度: {ensemble_precision:.4f}")

print("\n集成模型分类报告:")
print(classification_report(y_test_all, ensemble_pred))

# 保存最终模型和集成结果
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 保存所有单个模型
for i, model in enumerate(fold_models):
    model_path = f'f:/rainfalldata/models/xgboost_fold{i+1}_{timestamp}.json'
    model.save_model(model_path)

# 保存预测结果
results_df = pd.DataFrame({
    'y_true': y_test_all.ravel(),
    'y_pred_best_model': y_test_pred_all,
    'y_pred_ensemble': ensemble_pred,
    'y_proba_best_model': y_test_proba_all,
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

