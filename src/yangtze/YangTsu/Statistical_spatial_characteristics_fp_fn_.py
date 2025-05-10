from scipy import stats
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
#定义降雨分类阈值
thresholds = np.arange(0, 1, 0.05)
for i in range(X.shape[0]):
    product = PRODUCTS[i]
    for y in years:
        for threshold in thresholds:
            is_year = time[:, 0] == y
            #print(is_year)
            X_data = X[i, is_year, :, :]
            Y_data = Y[is_year, :, :]
            X_is_rain = np.where(X_data > rain_threshold, 1, 0)
            Y_is_rain = np.where(Y_data > rain_threshold, 1, 0)
            #print(X_is_rain.shape, Y_is_rain.shape)
            #计算fp, fn, tp, tn
            fp = ((X_is_rain == 1) & (Y_is_rain == 0))
            fn = ((X_is_rain == 0) & (Y_is_rain == 1))
            tp = ((X_is_rain == 1) & (Y_is_rain == 1))
            tn = ((X_is_rain == 0) & (Y_is_rain == 0))
            #print(fp.shape, fn.shape, tp.shape, tn.shape)
            fp = np.sum(fp, axis=0)
            fn = np.sum(fn, axis=0)
            tp = np.sum(tp, axis=0)
            tn = np.sum(tn, axis=0)
            #计算y年内每一个位置上的误报漏报命中
            POD = tp / (tp + fn)
            FAR = fp / (fp + tp)
            CSI = tp / (tp + fp + fn)
            #print(POD.shape, FAR.shape, CSI.shape)
            
            # 计算空间平均值和其他统计量
            print(f"产品 {product} 在 {y} 年阈值 {threshold} 的统计量:")
            print(f"  POD 平均值: {np.nanmean(POD):.4f}")
            print(f"  POD 标准差: {np.nanstd(POD):.4f}")
            print(f"  POD 最大值: {np.nanmax(POD):.4f}")
            print(f"  POD 最小值: {np.nanmin(POD):.4f}")
            print(f"  POD 中位数: {np.nanmedian(POD):.4f}")
            print(f"  POD 方差: {np.nanvar(POD):.4f}")
            
            print(f"  FAR 平均值: {np.nanmean(FAR):.4f}")
            print(f"  FAR 标准差: {np.nanstd(FAR):.4f}")
            print(f"  FAR 最大值: {np.nanmax(FAR):.4f}")
            print(f"  FAR 最小值: {np.nanmin(FAR):.4f}")
            print(f"  FAR 中位数: {np.nanmedian(FAR):.4f}")
            print(f"  FAR 方差: {np.nanvar(FAR):.4f}")
            
            print(f"  CSI 平均值: {np.nanmean(CSI):.4f}")
            print(f"  CSI 标准差: {np.nanstd(CSI):.4f}")
            print(f"  CSI 最大值: {np.nanmax(CSI):.4f}")
            print(f"  CSI 最小值: {np.nanmin(CSI):.4f}")
            print(f"  CSI 中位数: {np.nanmedian(CSI):.4f}")
            print(f"  CSI 方差: {np.nanvar(CSI):.4f}")
            
            # 计算偏度和峰度
            POD_flat = POD.flatten()
            FAR_flat = FAR.flatten()
            CSI_flat = CSI.flatten()
            
            pod_skewness = stats.skew(POD_flat, nan_policy='omit')
            pod_kurt = stats.kurtosis(POD_flat, nan_policy='omit')
            print(f"  POD 偏度 (Skewness): {pod_skewness:.4f}")
            print(f"  POD 峰度 (Kurtosis): {pod_kurt:.4f}")
            
            far_skewness = stats.skew(FAR_flat, nan_policy='omit')
            far_kurt = stats.kurtosis(FAR_flat, nan_policy='omit')
            print(f"  FAR 偏度 (Skewness): {far_skewness:.4f}")
            print(f"  FAR 峰度 (Kurtosis): {far_kurt:.4f}")
            
            csi_skewness = stats.skew(CSI_flat, nan_policy='omit')
            csi_kurt = stats.kurtosis(CSI_flat, nan_policy='omit')
            print(f"  CSI 偏度 (Skewness): {csi_skewness:.4f}")
            print(f"  CSI 峰度 (Kurtosis): {csi_kurt:.4f}")
            
            # 如果是最后一个阈值，计算不同产品之间的相关性
            if threshold == thresholds[-1] and i < X.shape[0]-1:
                for j in range(i+1, X.shape[0]):
                    other_product = PRODUCTS[j]
                    # 获取另一个产品的相同年份数据
                    X_other = X[j, is_year, :, :]
                    X_other_is_rain = np.where(X_other > rain_threshold, 1, 0)
                    
                    # 计算另一个产品的POD, FAR, CSI
                    fp_other = ((X_other_is_rain == 1) & (Y_is_rain == 0))
                    fn_other = ((X_other_is_rain == 0) & (Y_is_rain == 1))
                    tp_other = ((X_other_is_rain == 1) & (Y_is_rain == 1))
                    tn_other = ((X_other_is_rain == 0) & (Y_is_rain == 0))
                    
                    fp_other = np.sum(fp_other, axis=0)
                    fn_other = np.sum(fn_other, axis=0)
                    tp_other = np.sum(tp_other, axis=0)
                    tn_other = np.sum(tn_other, axis=0)
                    
                    POD_other = tp_other / (tp_other + fn_other)
                    FAR_other = fp_other / (fp_other + tp_other)
                    CSI_other = tp_other / (tp_other + fp_other + fn_other)
                    
                    # 计算POD, FAR, CSI之间的相关性
                    # POD相关性
                    valid_indices_pod = ~np.isnan(POD_flat) & ~np.isnan(POD_other.flatten())
                    if np.sum(valid_indices_pod) > 1:
                        corr_pod = np.corrcoef(POD_flat[valid_indices_pod], POD_other.flatten()[valid_indices_pod])
                        if corr_pod.shape == (2, 2):
                            print(f"  {product} 和 {other_product} 的POD相关系数: {corr_pod[0, 1]:.4f}")
                    
                    # FAR相关性
                    valid_indices_far = ~np.isnan(FAR_flat) & ~np.isnan(FAR_other.flatten())
                    if np.sum(valid_indices_far) > 1:
                        corr_far = np.corrcoef(FAR_flat[valid_indices_far], FAR_other.flatten()[valid_indices_far])
                        if corr_far.shape == (2, 2):
                            print(f"  {product} 和 {other_product} 的FAR相关系数: {corr_far[0, 1]:.4f}")
                    
                    # CSI相关性
                    valid_indices_csi = ~np.isnan(CSI_flat) & ~np.isnan(CSI_other.flatten())
                    if np.sum(valid_indices_csi) > 1:
                        corr_csi = np.corrcoef(CSI_flat[valid_indices_csi], CSI_other.flatten()[valid_indices_csi])
                        if corr_csi.shape == (2, 2):
                            print(f"  {product} 和 {other_product} 的CSI相关系数: {corr_csi[0, 1]:.4f}")
            
            print("-" * 30)
