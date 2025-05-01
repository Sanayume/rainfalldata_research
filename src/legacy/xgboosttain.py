import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import sys
import xgboost as xgb
import matplotlib.pyplot as plt
import io
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

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
    "MASK":loadmat(DATAFILE["MASK"])["mask"],
    #"MASK":np.flipud(np.transpose(loadmat(DATAFILE["MASK"])["mask"], (1, 0))),
}
MASK = DATAS["MASK"]
PRODUCT = DATAS.copy()
PRODUCT.pop("MASK")
PRODUCT.pop("CHM")
print(f"产品数据: {PRODUCT}")

# 检查数据形状
for key, value in DATAS.items():
    print(f"{key}: {value.shape}")

#数据预处理，X，Y。X输入为一天的六个产品数据，Y基于CHM的是否有雨的分类标签lable

Y = np.zeros((DATAS["CHM"].shape[0], DATAS["CHM"].shape[1], DATAS["CHM"].shape[2]))

nlat, nlon, ntime = DATAS["CHM"].shape
valid_point = MASK == 1
print(f"有效点数: {np.sum(valid_point)}")
n_samples = np.sum(valid_point)*ntime
X = np.zeros((n_samples, len(PRODUCT)))
Y = np.zeros((n_samples, 1))
trainsample = np.sum(valid_point)*(ntime-366)
testsample = np.sum(valid_point)*366
X_train = np.zeros((trainsample, len(PRODUCT)))
Y_train = np.zeros((trainsample, 1))
X_test = np.zeros((testsample, len(PRODUCT)))
Y_test = np.zeros((testsample, 1))
train_idx = 0
test_idx = 0
feature = []
for t in range(ntime):
    for i in range(nlat):
        for j in range(nlon):
            if MASK[i,j] == 1:
                if t < ntime-366:
                    feature = []
                    for product in PRODUCT:
                        value = DATAS[product][i,j,t]
                        #if value > 0:
                        #    value = 1
                        feature.append(value)
                    X_train[train_idx, :] = feature
                    Y_train[train_idx, 0] = DATAS["CHM"][i,j,t] > 0
                    train_idx += 1
                else:
                    feature = []
                    for product in PRODUCT:
                        value = DATAS[product][i,j,t]
                        #if value > 0:
                        #    value = 1
                        feature.append(value)
                    X_test[test_idx, :] = feature
                    Y_test[test_idx, 0] = DATAS["CHM"][i,j,t] > 0
                    test_idx += 1
print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
print(f"trainsample: {trainsample}, testsample: {testsample}, n_samples: {n_samples}")



# Create XGBoost classifier





'''
xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=0.8,  # 给予正类样本更高权重以提升召回率
    objective='binary:logistic',
    random_state=42
)
'''




# 修改XGBoost分类器参数，增加特征采样和样本采样
xgb_model = xgb.XGBClassifier(
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
    reg_lambda=1.5  # L2正则化
)




'''
# 添加特征选择，不直接删除特征，而是通过正则化进行惩罚
xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    reg_alpha=0.1,  # L1正则化
    reg_lambda=1.0,  # L2正则化
    scale_pos_weight=0.8,
    objective='binary:logistic',
    random_state=42
)
'''



# Train the model
print("Training XGBoost model...")
# 在现有代码中，训练前添加特征预处理
from sklearn.preprocessing import StandardScaler

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 添加交互特征
X_train_with_interaction = np.zeros((X_train.shape[0], X_train.shape[1]*2-1))
X_test_with_interaction = np.zeros((X_test.shape[0], X_test.shape[1]*2-1))

# 保留原始特征
X_train_with_interaction[:, :X_train.shape[1]] = X_train_scaled
X_test_with_interaction[:, :X_test.shape[1]] = X_test_scaled

# 添加特征交互项
feature_idx = X_train.shape[1]
for i in range(X_train.shape[1]-1):
    for j in range(i+1, X_train.shape[1]):
        X_train_with_interaction[:, feature_idx] = X_train_scaled[:, i] * X_train_scaled[:, j]
        X_test_with_interaction[:, feature_idx] = X_test_scaled[:, i] * X_test_scaled[:, j]
        feature_idx += 1
        if feature_idx >= X_train_with_interaction.shape[1]:
            break
    if feature_idx >= X_train_with_interaction.shape[1]:
        break

# 使用增强的特征训练模型
xgb_model.fit(X_train_with_interaction, Y_train.ravel())
y_pred = xgb_model.predict(X_test_with_interaction)

# Evaluate the model
accuracy = accuracy_score(Y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

# Feature importance
importance = xgb_model.feature_importances_
product_names = list(PRODUCT.keys())
for name, imp in zip(product_names, importance):
    print(f"{name} importance: {imp:.4f}")
