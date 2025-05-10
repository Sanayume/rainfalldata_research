from loaddata import mydata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
mydata = mydata()
_, _, X, Y = mydata.yangtsu()
PRODUCTS = mydata.get_products()
#定义降雨分类阈值, 默认为0.1mm/d
rain_threshold = 0.1 
days = 0

#生成从2016年-2020年5年的时间编码用于plot图的横坐标，三类坐标： 年， 月， 日  
time_d = pd.date_range(start='2016-01-01', end='2020-12-31', freq='D')
time_y = np.array(time_d.year)
time_m = np.array(time_d.month)
time_d = np.arange(X.shape[1])
time = np.stack((time_y, time_m, time_d), axis=1)
years = [2016, 2017, 2018, 2019, 2020]

for i in range(X.shape[0]):
    product = PRODUCTS[i]
    for y in years:
        is_year = time[:, 0] == y
        print(is_year.shape)
        X_data = X[i, is_year, :, :]
        Y_data = Y[is_year, :, :]
        X_is_rain = np.where(X_data > rain_threshold, 1, 0)
        Y_is_rain = np.where(Y_data > rain_threshold, 1, 0)
        print(X_is_rain.shape, Y_is_rain.shape)
        #计算fp, fn, tp, tn
        fp = np.sum((X_is_rain == 1) & (Y_is_rain == 0))
        fn = np.sum((X_is_rain == 0) & (Y_is_rain == 1))
        tp = np.sum((X_is_rain == 1) & (Y_is_rain == 1))
        tn = np.sum((X_is_rain == 0) & (Y_is_rain == 0))
        print(fp.shape, fn.shape, tp.shape, tn.shape)
        #计算y年内每一个位置上的误报漏报命中
        fp_rate = fp / (fp + tn)
        fn_rate = fn / (fn + tp)
        tp_rate = tp / (tp + fn)
        tn_rate = tn / (tn + fp)
        print(fp_rate.shape, fn_rate.shape, tp_rate.shape, tn_rate.shape)

    
    
















