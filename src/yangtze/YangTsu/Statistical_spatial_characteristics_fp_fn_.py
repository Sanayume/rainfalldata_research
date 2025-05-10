from scipy import stats
from loaddata import mydata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys # 导入sys模块
import os  # 导入os模块，用于创建输出目录

mydata = mydata()
X, Y = mydata.get_basin_spatial_data(2)
PRODUCTS = mydata.get_products()
days = 0

#生成从2016年-2020年5年的时间编码用于plot图的横坐标，三类坐标： 年， 月， 日  
time_d_pd = pd.date_range(start='2016-01-01', end='2020-12-31', freq='D') # 重命名以避免与下面的time_d冲突
time_y = np.array(time_d_pd.year)
time_m = np.array(time_d_pd.month)
time_d_indices = np.arange(X.shape[1]) # 使用索引作为每日时间标记
time = np.stack((time_y, time_m, time_d_indices), axis=1)
years = [2016, 2017, 2018, 2019, 2020]
#定义降雨分类阈值
thresholds = np.arange(0, 1, 0.05)

# --- 设置输出文件 ---
output_directory = "results/statistical_analysis" # 定义输出目录
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
output_filename = os.path.join(output_directory, "statistical_spatial_characteristics_output.txt")

original_stdout = sys.stdout  # 保存原始stdout

try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        sys.stdout = f  # 重定向stdout到文件

        for i in range(X.shape[0]):
            product = PRODUCTS[i]
            print(f"处理产品: {product}")
            print("=" * 40)
            for y in years:
                print(f"\n年份: {y}")
                print("-" * 30)
                for threshold in thresholds: # 重命名循环变量以避免与外部thresholds冲突

                    is_year = time_y == y # 修正此处的time[:,0] 为 time_y
                    
                    X_data = X[i, is_year, :, :]
                    Y_data = Y[is_year, :, :] # 假设Y已经是对应年份的数据或者会广播

                    if X_data.shape[0] == 0 or Y_data.shape[0] == 0:
                        print(f"  产品 {product} 在 {y} 年数据不足，跳过。")
                        continue

                    X_is_rain = np.where(X_data > threshold, 1, 0)
                    Y_is_rain = np.where(Y_data > threshold, 1, 0)
                    
                    fp = ((X_is_rain == 1) & (Y_is_rain == 0))
                    fn = ((X_is_rain == 0) & (Y_is_rain == 1))
                    tp = ((X_is_rain == 1) & (Y_is_rain == 1))
                    # tn = ((X_is_rain == 0) & (Y_is_rain == 0)) # tn未使用，注释掉
                    
                    fp_sum = np.sum(fp, axis=0) # 重命名变量以避免与原始fp冲突
                    fn_sum = np.sum(fn, axis=0)
                    tp_sum = np.sum(tp, axis=0)
                    # tn_sum = np.sum(tn, axis=0)

                    # --- 安全地计算 POD, FAR, CSI ---
                    # POD = TP / (TP + FN)
                    pod_denominator = tp_sum + fn_sum
                    POD = np.full_like(tp_sum, np.nan, dtype=float)
                    np.divide(tp_sum, pod_denominator, out=POD, where=pod_denominator != 0)

                    # FAR = FP / (TP + FP)
                    far_denominator = tp_sum + fp_sum
                    FAR = np.full_like(fp_sum, np.nan, dtype=float)
                    np.divide(fp_sum, far_denominator, out=FAR, where=far_denominator != 0)
                    FAR[far_denominator == 0] = 0.0 # 如果TP+FP=0, FAR=0

                    # CSI = TP / (TP + FP + FN)
                    csi_denominator = tp_sum + fp_sum + fn_sum
                    CSI = np.full_like(tp_sum, np.nan, dtype=float)
                    np.divide(tp_sum, csi_denominator, out=CSI, where=csi_denominator != 0)
                    
                    print(f"\n  产品 {product} 在 {y} 年 (降雨阈值 {threshold}mm/d) 的空间统计量:")
                    print(f"    POD 平均值: {np.nanmean(POD):.4f}, 标准差: {np.nanstd(POD):.4f}, 中位数: {np.nanmedian(POD):.4f}, 方差: {np.nanvar(POD):.4f}, 最大值: {np.nanmax(POD):.4f}, 最小值: {np.nanmin(POD):.4f}")
                    
                    POD_flat = POD.flatten()
                    pod_skewness = stats.skew(POD_flat[~np.isnan(POD_flat)]) # 使用~np.isnan过滤
                    pod_kurt = stats.kurtosis(POD_flat[~np.isnan(POD_flat)])
                    print(f"    POD 偏度 (Skewness): {pod_skewness:.4f}, 峰度 (Kurtosis): {pod_kurt:.4f}")

                    print(f"    FAR 平均值: {np.nanmean(FAR):.4f}, 标准差: {np.nanstd(FAR):.4f}, 中位数: {np.nanmedian(FAR):.4f}, 方差: {np.nanvar(FAR):.4f}, 最大值: {np.nanmax(FAR):.4f}, 最小值: {np.nanmin(FAR):.4f}")
                    
                    FAR_flat = FAR.flatten()
                    far_skewness = stats.skew(FAR_flat[~np.isnan(FAR_flat)])
                    far_kurt = stats.kurtosis(FAR_flat[~np.isnan(FAR_flat)])
                    print(f"    FAR 偏度 (Skewness): {far_skewness:.4f}, 峰度 (Kurtosis): {far_kurt:.4f}")

                    print(f"    CSI 平均值: {np.nanmean(CSI):.4f}, 标准差: {np.nanstd(CSI):.4f}, 中位数: {np.nanmedian(CSI):.4f}, 方差: {np.nanvar(CSI):.4f}, 最大值: {np.nanmax(CSI):.4f}, 最小值: {np.nanmin(CSI):.4f}")
                    
                    CSI_flat = CSI.flatten()
                    csi_skewness = stats.skew(CSI_flat[~np.isnan(CSI_flat)])
                    csi_kurt = stats.kurtosis(CSI_flat[~np.isnan(CSI_flat)])
                    print(f"    CSI 偏度 (Skewness): {csi_skewness:.4f}, 峰度 (Kurtosis): {csi_kurt:.4f}")
                    
                    # "threshold_val" 相关的逻辑在此脚本中似乎没有实际作用于指标计算，
                    # 主要是用于触发不同产品间的相关性分析。
                    # 如果是最后一个阈值，计算不同产品之间的相关性
                    if threshold == thresholds[-1] and i < X.shape[0]-1:
                        print(f"\n  {product} 与其他产品的相关性分析 (基于 {y} 年, 降雨阈值 {threshold}mm/d):")
                        for j in range(i+1, X.shape[0]):
                            other_product = PRODUCTS[j]
                            X_other_data = X[j, is_year, :, :] # 获取另一产品同年数据
                            
                            if X_other_data.shape[0] == 0:
                                print(f"    产品 {other_product} 在 {y} 年数据不足，无法计算相关性。")
                                continue

                            X_other_is_rain = np.where(X_other_data > threshold, 1, 0)
                            
                            fp_other_sum = np.sum(((X_other_is_rain == 1) & (Y_is_rain == 0)), axis=0)
                            fn_other_sum = np.sum(((X_other_is_rain == 0) & (Y_is_rain == 1)), axis=0)
                            tp_other_sum = np.sum(((X_other_is_rain == 1) & (Y_is_rain == 1)), axis=0)
                            
                            pod_other_denominator = tp_other_sum + fn_other_sum
                            POD_other = np.full_like(tp_other_sum, np.nan, dtype=float)
                            np.divide(tp_other_sum, pod_other_denominator, out=POD_other, where=pod_other_denominator!=0)

                            far_other_denominator = tp_other_sum + fp_other_sum
                            FAR_other = np.full_like(fp_other_sum, np.nan, dtype=float)
                            np.divide(fp_other_sum, far_other_denominator, out=FAR_other, where=far_other_denominator!=0)
                            FAR_other[far_other_denominator==0] = 0.0

                            csi_other_denominator = tp_other_sum + fp_other_sum + fn_other_sum
                            CSI_other = np.full_like(tp_other_sum, np.nan, dtype=float)
                            np.divide(tp_other_sum, csi_other_denominator, out=CSI_other, where=csi_other_denominator!=0)
                            
                            POD_other_flat = POD_other.flatten()
                            FAR_other_flat = FAR_other.flatten()
                            CSI_other_flat = CSI_other.flatten()

                            # POD相关性
                            valid_indices_pod = ~np.isnan(POD_flat) & ~np.isnan(POD_other_flat)
                            if np.sum(valid_indices_pod) > 1: # 需要至少两个有效数据点来计算相关性
                                corr_pod = np.corrcoef(POD_flat[valid_indices_pod], POD_other_flat[valid_indices_pod])
                                if corr_pod.ndim == 2 and corr_pod.shape == (2, 2): # 确保返回的是2x2矩阵
                                    print(f"    {product} vs {other_product} - POD 相关系数: {corr_pod[0, 1]:.4f}")
                                else:
                                    print(f"    {product} vs {other_product} - POD 相关系数: 计算失败或数据不足。")
                            
                            # FAR相关性
                            valid_indices_far = ~np.isnan(FAR_flat) & ~np.isnan(FAR_other_flat)
                            if np.sum(valid_indices_far) > 1:
                                corr_far = np.corrcoef(FAR_flat[valid_indices_far], FAR_other_flat[valid_indices_far])
                                if corr_far.ndim == 2 and corr_far.shape == (2, 2):
                                    print(f"    {product} vs {other_product} - FAR 相关系数: {corr_far[0, 1]:.4f}")
                                else:
                                    print(f"    {product} vs {other_product} - FAR 相关系数: 计算失败或数据不足。")
                            
                            # CSI相关性
                            valid_indices_csi = ~np.isnan(CSI_flat) & ~np.isnan(CSI_other_flat)
                            if np.sum(valid_indices_csi) > 1:
                                corr_csi = np.corrcoef(CSI_flat[valid_indices_csi], CSI_other_flat[valid_indices_csi])
                                if corr_csi.ndim == 2 and corr_csi.shape == (2, 2):
                                    print(f"    {product} vs {other_product} - CSI 相关系数: {corr_csi[0, 1]:.4f}")
                                else:
                                     print(f"    {product} vs {other_product} - CSI 相关系数: 计算失败或数据不足。")
                    print("-" * 30) # 分隔不同阈值/分析的输出
                print("=" * 30) # 分隔不同年份的输出
            print("#" * 50) # 分隔不同产品的输出
        
        print(f"\n所有统计输出已保存到: {output_filename}")

finally:
    sys.stdout = original_stdout # 恢复原始stdout
    print(f"脚本执行完毕。统计输出已保存到: {output_filename}")
    if 'f' in locals() and not f.closed: # 确保文件被关闭
        f.close()
