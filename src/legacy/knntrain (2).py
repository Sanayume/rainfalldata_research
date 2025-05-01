import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, minmax_scale, RobustScaler
from torch.nn import MSELoss
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import sys
import numpy as np
import io
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, RFE


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
#对数变换
X_train = np.log(X_train + np.min(X_train[X_train > 0]))
X_test = np.log(X_test + np.min(X_test[X_test > 0]))
#RobustScaler
robust = RobustScaler()
X_train_scaled = robust.fit_transform(X_train)
X_test_scaled = robust.transform(X_test)
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
xgboost_train_1 = xgb.XGBClassifier(
    max_depth=4,  # 减小深度可减少过拟合和过度依赖
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
    reg_lambda=1.5,  # L2正则化
    eval_metric=['error', 'auc']  # 将评估指标移到这里
)
feature_names = [
    'CMORPH_YESTERDAY', 'CHIRPS_YESTERDAY', 'SM2RAIN_YESTERDAY',
    'IMERG_YESTERDAY', 'GSMAP_YESTERDAY', 'PERSIANN_YESTERDAY',
    'CMORPH_TODAY', 'CHIRPS_TODAY', 'SM2RAIN_TODAY',
    'IMERG_TODAY', 'GSMAP_TODAY', 'PERSIANN_TODAY'
]

# 创建交叉特征
print("Creating interaction features...")
X_train_interactions = X_train_scaled.copy()
X_test_interactions = X_test_scaled.copy()
interaction_feature_names = feature_names.copy()

# 添加两两交叉特征
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        # 创建新的交叉特征名称
        new_feature_name = f"{feature_names[i]}*{feature_names[j]}"
        interaction_feature_names.append(new_feature_name)
        
        # 创建交叉特征
        new_feature_train = X_train_scaled[:, i] * X_train_scaled[:, j]
        new_feature_test = X_test_scaled[:, i] * X_test_scaled[:, j]
        
        # 将新特征添加到数据集
        X_train_interactions = np.column_stack((X_train_interactions, new_feature_train))
        X_test_interactions = np.column_stack((X_test_interactions, new_feature_test))
'''
print(f"原始特征数: {X_train_scaled.shape[1]}, 交叉特征后: {X_train_interactions.shape[1]}")
 
# 使用SelectKBest选择前60个最重要的特征
selector = SelectKBest(f_classif, k=60)
X_train_selected = selector.fit_transform(X_train_interactions, Y_train.ravel())
X_test_selected = selector.transform(X_test_interactions)

# 获取选择的特征和分数
selected_mask = selector.get_support()
scores = selector.scores_

# 准备显示选择的特征和对应的分数
selected_features = []
selected_scores = []

print("\n前20个选择的特征及其分数:")
for i in range(len(selected_mask)):
    if selected_mask[i]:
        selected_features.append(interaction_feature_names[i])
        selected_scores.append(scores[i])
        print(f"{interaction_feature_names[i]}: {scores[i]:.4f}")

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
print(f"更新后 X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")'
'''
xgboost_train_1.fit(X_train_scaled, Y_train.ravel())
Y_pred = xgboost_train_1.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"accuracy : {accuracy}")
feature_importance = xgboost_train_1.feature_importances_


for name, importance in zip(feature_names, feature_importance):
    print(f"{name}:{importance}")

# KNN
print("Starting KNN model training process...")

# Parameter search with GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
}

print("Finding optimal KNN parameters...")
grid_search = GridSearchCV(
    KNeighborsRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=2  # Provides detailed output
)

grid_search.fit(X_train_scaled, Y_train.ravel())

# Print best parameters
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")

# Train final model with best parameters
knn = grid_search.best_estimator_
print("Training final KNN model with best parameters...")
knn.fit(X_train_scaled, Y_train.ravel())

# Evaluate on test set
print("Evaluating KNN model on test data...")
Y_pred = knn.predict(X_test_scaled)

# Round predictions for binary classification accuracy
Y_pred_rounded = np.round(Y_pred)
accuracy = accuracy_score(Y_test, Y_pred_rounded)
print(f"KNN Accuracy: {accuracy:.4f}")

# Additional metrics
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f"KNN Mean Squared Error: {mse:.4f}")
print(f"KNN Mean Absolute Error: {mae:.4f}")
print(f"KNN R² Score: {r2:.4f}")

#SVM
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

