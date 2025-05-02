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
import os

# 导入数据检查模块
from data_check import check_missing_data, plot_missing_data_stats, print_stats_summary, plot_max_missing_day,hybrid_interpolation
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

basepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
# 定义数据路径和常量,6个文件
MASK = os.path.join(basepath, "mask", "mask.mat") #掩膜数据
CHIRPS = os.path.join(basepath, "intermediate", "nationwide", "CHIRPSdata", "chirps_2016_2020.mat") #产品数据 小规模缺失数据 使用插值方法填充
IMERG = os.path.join(basepath, "intermediate", "nationwide", "IMERGdata", "IMERG_2016_2020.mat") #产品数据
CHM = os.path.join(basepath, "intermediate", "nationwide", "CHMdata", "CHM_2016_2020.mat") #真实数据
GSMAP = os.path.join(basepath, "intermediate", "nationwide", "GSMAPdata", "GSMAP_2016_2020.mat") #产品数据
SM2RAIN = os.path.join(basepath, "intermediate", "nationwide", "sm2raindata", "sm2rain_2016_2020.mat") #产品数据 且大面积缺失数据 使用IMERG数据填充
PERSIANN = os.path.join(basepath, "intermediate", "nationwide", "PERSIANNdata", "PERSIANN_2016_2020.mat") #产品数据
CMORPH = os.path.join(basepath, "intermediate", "nationwide", "CMORPHdata", "CMORPH_2016_2020.mat") #产品数据

# 加载数据
mask = loadmat(MASK)['mask']
chirps = loadmat(CHIRPS)['data']
imerg = loadmat(IMERG)['data']
chm = loadmat(CHM)['data']
gsm = loadmat(GSMAP)['data']
sm2rain = loadmat(SM2RAIN)['data']
persiann = loadmat(PERSIANN)['data']
cmorph = loadmat(CMORPH)['data']

#检查全部数据的形状
print(f"CHIRPS: {chirps.shape}")  #ok
print(f"IMERG: {imerg.shape}")    #ok
print(f"CHM: {chm.shape}")    #ok
print(f"GSMAP: {gsm.shape}")  #ok
print(f"SM2RAIN: {sm2rain.shape}")  #ok
print(f"PERSIANN: {persiann.shape}") #ok
print(f"CMORPH: {cmorph.shape}")   #not ok
print(f"MASK: {mask.shape}")

# 创建CHIRPS特殊掩膜
#maskchirps = mask.copy()
#maskchirps[0:16,:] = 0

# 检查所有数据集
datasets = {
    'CHIRPS': (chirps, mask),
    'IMERG': (imerg, mask),
    'CHM': (chm, mask),
    'GSMAP': (gsm, mask),
    'SM2RAIN': (sm2rain, mask),
    'PERSIANN': (persiann, mask),
    'CMORPH': (cmorph, mask)
}

processed_data = {}
data_stats = {}

# 对CHIRPS数据进行插值处理

#datasets['CHIRPS'] = [interpolate_missing_data(datasets.get('CHIRPS')[0], datasets.get('CHIRPS')[1]),maskchirps]
#datasets['CHIRPS'] = [replace_with_reference(datasets.get('CHIRPS')[0], datasets.get('IMERG')[0],mask),mask]

# 对sm2进行混合处理
#processed_data, stats = hybrid_interpolation(datasets.get('SM2RAIN')[0], mask=datasets.get('SM2RAIN')[1], reference_data=datasets.get('IMERG')[0])
#datasets['SM2RAIN'] = (processed_data, datasets.get('SM2RAIN')[1])

#对CMORPH数据进行混合处理
processed_data, stats = hybrid_interpolation(datasets.get('CMORPH')[0], mask=datasets.get('CMORPH')[1], reference_data=datasets.get('IMERG')[0])
datasets['CMORPH'] = (processed_data, datasets.get('CMORPH')[1])

# 保存处理后的数据
savemat(os.path.join(basepath, "intermediate", "nationwide", "CHIRPSdata", "chirps_2016_2020.mat"),{'data':datasets.get('CHIRPS')[0]})
savemat(os.path.join(basepath, "intermediate", "nationwide", "sm2raindata", "sm2rain_2016_2020.mat"),{'data':datasets.get('SM2RAIN')[0]})
savemat(os.path.join(basepath, "intermediate", "nationwide", "CMORPHdata", "CMORPH_2016_2020.mat"),{'data':datasets.get('CMORPH')[0]})

processed_results = {}  # 改用更清晰的变量名
stats_results = {}     # 改用更清晰的变量名

for name, (data, mask_data) in datasets.items():
    processed, stats = check_missing_data(data, mask, name, 
                                        special_mask=mask_data if name == 'CHIRPS' else None)
    processed_results[name] = processed
    stats_results[name] = stats
    
    print_stats_summary(stats, name)
    plot_missing_data_stats(stats, name)
    
    # 添加最大缺失值天的分布图
    plot_max_missing_day(data, mask_data, stats, name)
    print(f"\n{name}数据中缺失值最多的是第{stats['max_nan_day']['index']+1}天，"
          f"缺失值数量为{stats['max_nan_day']['count']}")

# 可选：保存处理后的数据
for name, data in processed_results.items():
    savemat(os.path.join(basepath, "intermediate", "nationwide", f'processed_{name.lower()}.mat'), {'data': data})




