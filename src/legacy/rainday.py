import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale, RobustScaler
from torch.nn import MSELoss
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
import sys
import numpy as np
import io
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


DATAFILE = {
    "CMORPH": "D:\降雨融合数据及代码\python\数据\CMORPH_2016_2020.mat",
    "CHIRPS": "D:\降雨融合数据及代码\python\数据\chirps_2016_2020.mat",
    "SM2RAIN": "D:\降雨融合数据及代码\python\数据\sm2rain_2016_2020.mat",
    "IMERG": "D:\降雨融合数据及代码\python\数据\IMERG_2016_2020.mat",
    "GSMAP": "D:\降雨融合数据及代码\python\数据\GSMAP_2016_2020.mat",
    "PERSIANN": "D:\降雨融合数据及代码\python\数据\PERSIANN_2016_2020.mat",
    "CHM": "D:\降雨融合数据及代码\python\数据\CHM_2016_2020.mat",
    "MASK": "D:\降雨融合数据及代码\python\mask\mask.mat",
}

DATAS = {
    "CMORPH":loadmat(DATAFILE["CMORPH"])["data"],
    "CHIRPS":loadmat(DATAFILE["CHIRPS"])["data"],
    "SM2RAIN":loadmat(DATAFILE["SM2RAIN"])["data"],
    "IMERG":loadmat(DATAFILE["IMERG"])["data"],
    "GSMAP":loadmat(DATAFILE["GSMAP"])["data"],
    "PERSIANN":loadmat(DATAFILE["PERSIANN"])["data"],
    "CHM":loadmat(DATAFILE["CHM"])["data"],
    "MASK":np.flipud(np.transpose(loadmat(DATAFILE["MASK"])["mask"], (1, 0))),
}
MASK = DATAS["MASK"]
PRODUCT = DATAS.copy()
PRODUCT.pop("MASK")
PRODUCT.pop("CHM")

volid_point = MASK == 1
volid_point_sum = np.sum(volid_point)
samples = volid_point_sum * DATAS["CHM"].shape[2]
print(f"有效点数: {volid_point_sum}")
X_train = []
Y_train = []
X_test = []
Y_test = []
#时间步长为2天
for i in range(DATAS["CHM"].shape[2] - 1):
    if i < (1827 - 366 - 1):    
        feature = []
        for name, data in PRODUCT.items():
            value = np.zeros(volid_point_sum)
            # np.squeeze() removes single-dimensional entries from array shape
            data_squeeze = np.squeeze(data[:, :, i])
            # np.ravel() flattens the array to 1D
            # ~np.isnan() creates boolean mask where True means not NaN
            # data_squeeze[mask] selects only non-NaN values
            masksqueeze = np.isnan(data_squeeze)
            data_squeeze = data_squeeze[~masksqueeze]
            value[:] = data_squeeze[:]
            feature.append(value)
        for name, data in PRODUCT.items():
            value = np.zeros(volid_point_sum)
            data_squeeze = np.squeeze(data[:, :, i+1])
            masksqueeze = np.isnan(data_squeeze)
            data_squeeze = data_squeeze[~masksqueeze]
            value[:] = data_squeeze[:]
            feature.append(value)
        X_train.extend(np.array(feature).T)
        lables = np.zeros((DATAS["CHM"].shape[0], DATAS["CHM"].shape[1]))
        lables[DATAS["CHM"][:, :, i] > 0] = 1
        lables_squeeze = np.squeeze(lables)
        lables_squeeze = lables_squeeze[~masksqueeze]
        Y_train.extend(lables_squeeze)
    else:
        feature = []
        for name, data in PRODUCT.items():
            value = np.zeros(volid_point_sum)
            data_squeeze = np.squeeze(data[:, :, i])
            masksqueeze = np.isnan(data_squeeze)
            data_squeeze = data_squeeze[~masksqueeze]
            value[:] = data_squeeze[:]
            feature.append(value)
        for name, data in PRODUCT.items():
            value = np.zeros(volid_point_sum)
            data_squeeze = np.squeeze(data[:, :, i+1])
            masksqueeze = np.isnan(data_squeeze)
            data_squeeze = data_squeeze[~masksqueeze]
            value[:] = data_squeeze[:]
            feature.append(value)
        X_test.extend(np.array(feature).T)
        
        lables = np.zeros((DATAS["CHM"].shape[0], DATAS["CHM"].shape[1]))
        lables[DATAS["CHM"][:, :, i] > 0] = 1
        lables_squeeze = np.squeeze(lables)
        lables_squeeze = lables_squeeze[~masksqueeze]
        Y_test.extend(lables_squeeze)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_train = X_train.astype(dtype=np.float32)
Y_train = Y_train.astype(dtype=np.float32)
X_test = X_test.astype(dtype=np.float32)
Y_test = Y_test.astype(dtype=np.float32)
features = X_train.shape[1]
print(f"特征数: {features}")
#形状检查
print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
# Convert to pandas DataFrames for better analysis
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# Check memory usage
print(f"X_train memory usage: {X_train.nbytes / (1024 * 1024):.2f} MB")
print(f"X_test memory usage: {X_test.nbytes / (1024 * 1024):.2f} MB")

# Basic statistics for training data
print("\nX_train statistics:")
print(X_train_df.describe())

# Basic statistics for test data
print("\nX_test statistics:")
print(X_test_df.describe())

# Data types
print("\nX_train data type:", X_train.dtype)
print("X_test data type:", X_test.dtype)

# Check for NaN values
print(f"\nMissing values in X_train: {np.isnan(X_train).sum()}")
print(f"Missing values in X_test: {np.isnan(X_test).sum()}")

#释放内存
del X_train_df, X_test_df
'''
# Distribution visualization (optional, uncomment if needed)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(X_train.flatten(), bins=50, alpha=0.7, range=(0, 50))
plt.title('X_train Distribution')
plt.xlim(0, 50)
plt.subplot(1, 2, 2)
plt.hist(X_test.flatten(), bins=50, alpha=0.7, range=(0, 50))
plt.title('X_test Distribution')
plt.xlim(0, 50)
plt.tight_layout()
plt.show()
'''
#minmax
#X_train = minmax_scale(X_train, feature_range=(0, 1))
#X_test = minmax_scale(X_test, feature_range=(0, 1))
'''
# 归一化后的数据分布
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(X_train.flatten(), bins=50, alpha=0.7, range=(0, 1))
plt.title('X_train Distribution')
plt.xlim(0, 1)
plt.subplot(1, 2, 2)
plt.hist(X_test.flatten(), bins=50, alpha=0.7, range=(0, 1))
plt.title('X_test Distribution')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
'''

'''
# 归一化后的数据分布
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(X_train_scaled.flatten(), bins=50, alpha=0.7, range=(-1, 1))
plt.title('X_train Distribution')
plt.xlim(-1, 1)
plt.subplot(1, 2, 2)
plt.hist(X_test_scaled.flatten(), bins=50, alpha=0.7, range=(-1, 1))
plt.title('X_test Distribution')
plt.xlim(-1, 1)
plt.tight_layout()
plt.show()
'''
#binarize

#bins = [0, 0.1, 1, 5, 10, 20, np.inf]
#lables = ['No rain', 'Light rain', 'Moderate rain', 'Heavy rain', 'Very heavy rain', 'Extreme rain']

negative = (Y_train == 0).sum()
positive = (Y_train == 1).sum()
scale_pos_weight = negative / positive
print(f"scale_pos_weight: {scale_pos_weight}")

#train
xgboost_train_1 = xgb.XGBClassifier(    max_depth=4,  # 减小深度可减少过拟合和过度依赖
    learning_rate=0.05,  # 降低学习率让模型更稳健
    n_estimators=200,  # 增加树的数量以提高稳定性
    subsample=0.8,  # 每棵树使用80%的样本
    colsample_bytree=0.7,  # 每棵树随机选择80%的特征
    colsample_bylevel=0.7,  # 每次分裂随机选择80%的特征
    colsample_bynode=0.7,  # 每个节点随机选择80%的特征
    scale_pos_weight=0.7,
    objective='binary:logistic',
    random_state=42,
    reg_alpha=0.3,  # L1正则化
    reg_lambda=1.5  # L2正则化
    )
feature_names = [
    'CMORPH_YESTERDAY', 'CHIRPS_YESTERDAY', 'SM2RAIN_YESTERDAY',
    'IMERG_YESTERDAY', 'GSMAP_YESTERDAY', 'PERSIANN_YESTERDAY',
    'CMORPH_TODAY', 'CHIRPS_TODAY', 'SM2RAIN_TODAY',
    'IMERG_TODAY', 'GSMAP_TODAY', 'PERSIANN_TODAY'
]

# 创建交叉特征
print("Creating interaction features...")

#今天产品之间的交叉特征
n_product_today = 6
n_today_interaction = n_product_today * (n_product_today - 1) // 2
today_interaction_train = np.zeros((X_train.shape[0], n_today_interaction), dtype=np.float32)
today_interaction_test = np.zeros((X_test.shape[0], n_today_interaction), dtype=np.float32)
idx = 0  # Use a simple incremental index
for i in range(n_product_today):
    for j in range(i+1, n_product_today):
        today_interaction_train[:, idx] = X_train[:, i] * X_train[:, j]
        today_interaction_test[:, idx] = X_test[:, i] * X_test[:, j]
        feature_names.append(f"{feature_names[i]}_{feature_names[j]}")
        idx += 1
#昨天产品之间的交叉特征
n_product_yesterday = 6
n_yesterday_interaction = n_product_yesterday * (n_product_yesterday - 1) // 2
yesterday_interaction_train = np.zeros((X_train.shape[0], n_yesterday_interaction), dtype=np.float32)
yesterday_interaction_test = np.zeros((X_test.shape[0], n_yesterday_interaction), dtype=np.float32)
idx = 0  # Use a simple incremental index
for i in range(n_product_yesterday):
    for j in range(i+1, n_product_yesterday):
        yesterday_interaction_train[:, idx] = X_train[:, i+n_product_today] * X_train[:, j+n_product_today]
        yesterday_interaction_test[:, idx] = X_test[:, i+n_product_today] * X_test[:, j+n_product_today]
        feature_names.append(f"{feature_names[i+n_product_today]}_{feature_names[j+n_product_today]}")
        idx += 1

#各个产品昨天和今天的交叉特征
n_product_today_yesterday = len(PRODUCT)
today_yesterday_interaction_train = np.zeros((X_train.shape[0], n_product_today_yesterday), dtype=np.float32)
today_yesterday_interaction_test = np.zeros((X_test.shape[0], n_product_today_yesterday), dtype=np.float32)
idx = 0  # Use a simple incremental index
for i in range(n_product_today_yesterday):
    today_yesterday_interaction_train[:, idx] = X_train[:, i] * X_train[:, i+n_product_today]
    today_yesterday_interaction_test[:, idx] = X_test[:, i] * X_test[:, i+n_product_today]
    feature_names.append(f"{feature_names[i]}_{feature_names[i+n_product_today]}")
    idx += 1
X_train_interactions = np.hstack((X_train, today_interaction_train, yesterday_interaction_train, today_yesterday_interaction_train))
X_test_interactions = np.hstack((X_test, today_interaction_test, yesterday_interaction_test, today_yesterday_interaction_test))

#添加各天得二值化特征
#定义阈值
threshold = 0.1
X_train_binary = (X_train[:, 0 : 12] > threshold).astype(np.float32)
X_test_binary = (X_test[:, 0 : 12] > threshold).astype(np.float32)
X_train_interactions = np.hstack((X_train_interactions, X_train_binary))
X_test_interactions = np.hstack((X_test_interactions, X_test_binary))

#打印形状
print(f"原始特征: {X_train.shape}, 交叉特征后: {X_train_interactions.shape}")

print(f"原始特征数: {X_train.shape[1]}, 交叉特征后: {X_train_interactions.shape[1]}")

'''
X_train = np.log(X_train + np.min(X_train[X_train > 0]))
X_test = np.log(X_test + np.min(X_test[X_test > 0]))
# standardization
robust = StandardScaler()
X_train_scaled = robust.fit_transform(X_train)
X_test_scaled = robust.transform(X_test)

# 使用SelectKBest选择前20个最重要的特征
selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train_interactions, Y_train.ravel())
X_test_selected = selector.transform(X_test_interactions)

# 获取选择的特征和分数
selected_mask = selector.get_support()
scores = selector.scores_

# 准备显示选择的特征和对应的分数
selected_features = []
selected_scores = []


for feature, mask in zip(feature_names, selected_mask):
    if mask:
        selected_features.append(feature)


selected_scores = scores[selected_mask]

# 按分数排序并显示
feature_score_pairs = list(zip(selected_features, selected_scores))
feature_score_pairs.sort(key=lambda x: x[1], reverse=True)

print("\n按重要性排序的前20个特征:")
for feature, score in feature_score_pairs:
    print(f"{feature}: {score:.4f}")

# 更新特征名称为选择的特征
feature_names = selected_features

# 使用选择的特征继续训练
X_train_scaled = X_train_selected
X_test_scaled = X_test_selected

print(f"最终选择的特征数: {X_train_scaled.shape[1]}")
print(f"更新后 X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")
selector = SelectKBest(f_classif, k=9)
X_train_scaled = selector.fit_transform(X_train_scaled, Y_train.ravel())
X_test_scaled = selector.transform(X_test_scaled)
get_support = selector.get_support()
scores = selector.scores_
print(f"feature_names: {feature_names}")
print(f"get_support: {get_support}")
print(f"scores: {scores}")
print(f"after select X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")
for i in range(len(get_support)):
    if get_support[i] == True:
        print(f"{feature_names[i]} : {scores[i]}")
        
    
xgboost_train_1.fit(X_train_scaled, Y_train.ravel())
Y_pred = xgboost_train_1.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"accuracy : {accuracy}")
feature_importance = xgboost_train_1.feature_importances_


for name, importance in zip(feature_names, feature_importance):
    print(f"{name}:{importance}")

# KNN
KNN = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    p=2,
    metric_params=None,
    n_jobs=None,
    verbose=2
)

KNN.fit(X_train_scaled, Y_train.ravel())
Y_pred = KNN.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"accuracy : {accuracy}")
#评分R2, MAE, MSE,f1, recall, precision


# SVM
svm = SVC(
    C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=1e-3,
    cache_size=200,
    class_weight=None
)
svm.fit(X_train_scaled, Y_train.ravel())
Y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"accuracy : {accuracy}")
#评分R2, MAE, MSE,f1, recall, precision

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf.fit(X_train_scaled, Y_train.ravel())
Y_pred = rf.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"accuracy : {accuracy}")

pipline = Pipeline([
    ('scaler', StandardScaler()),  # 使用标准化作为预处理步骤
    ('cf1', xgboost_train_1)  # 使用XGBoost分类器
    ('cf2', KNN),  # 使用KNN分类器
    ('cf3', rf),  # 使用随机森林分类器
    ('cf4', svm)  # 使用SVM分类器
])
pipline.fit(X_train_scaled, Y_train.ravel())
Y_pred = pipline.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
'''

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import cross_val_score

# 定义用于评估所有分类器的函数
def evaluate_classifier(model, X_test, y_test, model_name):
    # 预测
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    # 如果模型支持概率预测，也计算AUC
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = None
    else:
        auc = None
    
    # 计算各种指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)  # Matthews相关系数，平衡类不平衡问题
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # 输出结果
    print(f"\n===== {model_name} 模型评估 =====")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    print(f"Matthews相关系数: {mcc:.4f}")
    print(f"预测时间: {pred_time:.4f}秒")
    
    print("\n混淆矩阵:")
    print(f"真正例 (TP): {tp}, 假正例 (FP): {fp}")
    print(f"假负例 (FN): {fn}, 真负例 (TN): {tn}")
    
    # 计算正样本的准确率
    positive_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"正样本预测准确率 (重要指标): {positive_accuracy:.4f}")
    
    # 详细的分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'auc': auc,
        'pred_time': pred_time,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'positive_accuracy': positive_accuracy,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# 创建Pipeline函数
print("\n开始创建并评估所有分类器...")

# 创建4个分类器
xgboost_classifier = xgb.XGBClassifier(
    max_depth=4,  
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.7,
    colsample_bylevel=0.7,
    colsample_bynode=0.7,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    random_state=42,
    reg_alpha=0.3,
    reg_lambda=1.5,
    verbose=1
)

knn_classifier = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # 使用距离权重更合理
    algorithm='auto',
    leaf_size=30,  
    metric='minkowski',
    p=2,
    n_jobs=-1,  # 使用所有CPU核心
    )

svm_classifier = SVC(
    C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale',
    probability=True,  # 允许概率输出以计算AUC
    class_weight='balanced',  # 考虑类别不平衡
    random_state=42,
    verbose=1
)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',  # 考虑类别不平衡
    random_state=42,
    n_jobs=-1,  # 使用所有CPU核心
    verbose=1
)

# 创建分类器列表
classifiers = {
    'SVM': svm_classifier,
    'Random Forest': rf_classifier,
    'XGBoost': xgboost_classifier,
    'KNN': knn_classifier,
}

X_train_scaled = X_train_interactions
X_test_scaled = X_test_interactions

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)


# 训练和评估所有分类器
results = {}
for name, classifier in classifiers.items():
    print(f"\n训练 {name} 分类器...")
    start_time = time.time()
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    classifier.fit(X_train_scaled, Y_train.ravel())
    train_time = time.time() - start_time
    print(f"{name} 训练时间: {train_time:.2f} 秒")
    
    # 评估模型
    result = evaluate_classifier(classifier, X_test_scaled, Y_test, name)
    result['train_time'] = train_time
    results[name] = result

# 交叉验证评估
print("\n执行5折交叉验证评估...")
cv_results = {}
for name, classifier in classifiers.items():
    start_time = time.time()
    cv_scores = cross_val_score(classifier, X_train_scaled, Y_train.ravel(), cv=5, scoring='accuracy')
    cv_time = time.time() - start_time
    
    cv_results[name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'time': cv_time
    }
    
    print(f"{name} 交叉验证 - 准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}, 耗时: {cv_time:.2f}秒")

# 比较所有模型的性能
metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
metric_names = ['准确率', '精确率', '召回率', 'F1分数', 'MCC']

# 创建结果比较表格
comparison_df = pd.DataFrame({
    '模型': [name for name in results.keys()],
    '准确率': [results[name]['accuracy'] for name in results.keys()],
    '精确率': [results[name]['precision'] for name in results.keys()],
    '召回率': [results[name]['recall'] for name in results.keys()],
    'F1分数': [results[name]['f1'] for name in results.keys()],
    'MCC': [results[name]['mcc'] for name in results.keys()],
    '训练时间(秒)': [results[name]['train_time'] for name in results.keys()],
    '预测时间(秒)': [results[name]['pred_time'] for name in results.keys()]
})

print("\n模型性能比较表:")
print(comparison_df)

# 可视化比较结果
try:
    plt.figure(figsize=(15, 10))
    
    # 指标对比图
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 3, i+1)
        metric_values = [results[name][metric] for name in results.keys()]
        bars = plt.bar(list(results.keys()), metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # 在柱形上添加数值
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f'{value:.4f}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.title(f'模型{metric_name}比较')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 训练和预测时间对比图
    plt.subplot(2, 3, 6)
    train_times = [results[name]['train_time'] for name in results.keys()]
    pred_times = [results[name]['pred_time'] for name in results.keys()]
    
    x = np.arange(len(results.keys()))
    width = 0.35
    
    plt.bar(x - width/2, train_times, width, label='训练时间')
    plt.bar(x + width/2, pred_times, width, label='预测时间')
    
    plt.xlabel('模型')
    plt.ylabel('时间(秒)')
    plt.title('训练和预测时间比较')
    plt.xticks(x, list(results.keys()))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png')
    plt.close()
    print("\n模型比较结果可视化已保存为 'model_comparison_results.png'")
    
    # 可视化混淆矩阵
    plt.figure(figsize=(15, 10))
    for i, name in enumerate(results.keys()):
        plt.subplot(2, 2, i+1)
        cm = np.array([[results[name]['tn'], results[name]['fp']], 
                       [results[name]['fn'], results[name]['tp']]])
                       
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{name} 混淆矩阵')
        plt.colorbar()
        
        classes = ['无降雨', '有降雨']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # 在矩阵中显示数值
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    print("混淆矩阵可视化已保存为 'confusion_matrices.png'")
    
    # 如果有概率预测，绘制ROC曲线
    plt.figure(figsize=(10, 8))
    from sklearn.metrics import roc_curve
    
    for name in results.keys():
        if results[name]['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(Y_test, results[name]['y_pred_proba'])
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {results[name]["auc"]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title('ROC曲线比较')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('roc_curves.png')
    plt.close()
    print("ROC曲线可视化已保存为 'roc_curves.png'")
    
except Exception as e:
    print(f"绘图时出现错误: {e}")

# 输出最佳模型
best_model = max(results.items(), key=lambda x: x[1]['f1'])
print(f"\n最佳模型(基于F1分数): {best_model[0]}, F1分数: {best_model[1]['f1']:.4f}")

# 查看降雨样本的预测准确率(重要)
for name in results.keys():
    positive_acc = results[name]['positive_accuracy']
    print(f"{name} 降雨样本预测准确率: {positive_acc:.4f}")









