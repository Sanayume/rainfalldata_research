import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import seaborn as sns
from sklearn.pipeline import Pipeline
import sys
import matplotlib.pyplot as plt
import io

DATAFILE = {
    "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat",
    "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat",
    "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat",
    "IMERG": "IMERGdata/IMERG_2016_2020.mat",
    "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat",
    "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat",
    "CHM": "CHMdata/CHM_2016_2020.mat",
}

MASK = loadmat("mask.mat")["mask"]

DATAS = {
    "CMORPH":loadmat(DATAFILE["CMORPH"])["data"],
    "CHIRPS":loadmat(DATAFILE["CHIRPS"])["data"],
    "SM2RAIN":loadmat(DATAFILE["SM2RAIN"])["data"],
    "IMERG":loadmat(DATAFILE["IMERG"])["data"],
    "GSMAP":loadmat(DATAFILE["GSMAP"])["data"],
    "PERSIANN":loadmat(DATAFILE["PERSIANN"])["data"],
    "CHM":loadmat(DATAFILE["CHM"])["data"],
}

PRODUCT = DATAS.pop("CHM")
print(f"产品数据: {PRODUCT}")

# 检查数据形状
for key, value in DATAS.items():
    print(f"{key}: {value.shape}")

#数据预处理，X，Y。X输入为一天的六个产品数据，Y基于CHM的是否有雨的分类标签lable

Y = np.zeros((DATAS["CHM"].shape[0], DATAS["CHM"].shape[1], DATAS["CHM"].shape[2]))

nlat, nlon, ntime = DATAS["CHM"].shape
valid_point = MASK == 1
n_samples = np.sum(valid_point)*ntime
X = np.zeros((n_samples, len(PRODUCT)))
Y = np.zeros((n_samples, 1))
trainsample = np.sum(valid_point)*ntime-366
testsample = np.sum(valid_point)*366
Xtain = np.zeros((trainsample, len(PRODUCT)))
sample_idx = 0
for t in range(ntime):
    for i in range(nlat):
        for j in range(nlon):
            if MASK[i,j] == 1:
                    for product in PRODUCT:
                        
                        value = product[i,j,t]
                        feature = feature.append(value)
                    X[sample_idx, :] = feature
                    Y[sample_idx, 0] = DATAS["CHM"][i,j,t] > 0
                    sample_idx += 1
print(F"n_samples: {n_samples}")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")