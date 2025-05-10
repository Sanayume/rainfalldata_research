from loaddata import mydata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

mydata = mydata()
#更改这里的数据来源来指定区域

X, Y = mydata.get_basin_spatial_data(1)
PRODUCTS = mydata.get_products()

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print("-" * 30)

for i in range(X.shape[0]):
    product_name = PRODUCTS[i]
    print(f"Statistics for product: {product_name}")
    
    product_data = X[i, :, :, :] # Shape: (time, lat, lon)
    product_data_flat = product_data.flatten()

    print(f"  平均值: {np.nanmean(product_data):.4f}")
    print(f"  标准差: {np.nanstd(product_data):.4f}")
    print(f"  最大值: {np.nanmax(product_data):.4f}")
    print(f"  最小值: {np.nanmin(product_data):.4f}")
    print(f"  中位数: {np.nanmedian(product_data):.4f}")
    
    mode_result = stats.mode(product_data_flat, nan_policy='omit', keepdims=False)

    # 检查 mode_result.mode 是否包含一个有效的模式值
    # 当找到模式时，mode_result.mode.size 应该是 1 (因为 keepdims=False 且输入是1D，会返回单个模式)
    # 当没有找到模式时 (例如，所有值都是NaN)，mode_result.mode.size 会是 0
    if hasattr(mode_result.mode, 'size') and mode_result.mode.size == 1:
        # 安全地提取模式值和计数值，.item() 适用于0维数组或大小为1的1维数组
        actual_mode_value = mode_result.mode.item()
        actual_count_value = mode_result.count.item()
        print(f"  众数: {actual_mode_value:.4f} (count: {actual_count_value})")
    elif hasattr(mode_result.mode, 'size') and mode_result.mode.size == 0:
        print(f"  众数: No mode found (all NaN or empty after NaN removal)")
    else:
        # 理论上不应该进入这个分支，因为 keepdims=False 对于1D输入应该只返回0个或1个模式
        # 但作为一种保护措施，打印出意外的结果
        print(f"  众数: Unexpected mode result. Mode: {mode_result.mode}, Count: {mode_result.count}")
        
    print(f"  方差: {np.nanvar(product_data):.4f}")
    
    skewness = stats.skew(product_data_flat, nan_policy='omit')
    kurt = stats.kurtosis(product_data_flat, nan_policy='omit') # Fisher's definition (excess kurtosis)
    print(f"  偏度 (Skewness): {skewness:.4f}")
    print(f"  峰度 (Kurtosis): {kurt:.4f}")
    
    # Correlation with Y
    y_flat = Y.flatten()
    valid_indices_xy = ~np.isnan(product_data_flat) & ~np.isnan(y_flat)
    
    if np.sum(valid_indices_xy) > 1: # Need at least 2 valid pairs for correlation
        corr_matrix_xy = np.corrcoef(product_data_flat[valid_indices_xy], y_flat[valid_indices_xy])
        if corr_matrix_xy.shape == (2, 2):
            print(f"  和 Y 之间的相关系数: {corr_matrix_xy[0, 1]:.4f}")
        else:
            print(f"  和 Y 之间的相关系数: Error calculating correlation")
    else:
        print(f"  和 Y 之间的相关系数: Not enough valid data points")

    # Correlations with other products X[j]
    # The original loop was range(i, X.shape[0]).
    # This includes self-correlation and avoids duplicating (i,j) and (j,i) if we only care about the value.
    for j in range(X.shape[0]): # Calculate for all products, including self
        other_product_name = PRODUCTS[j]
        other_product_data = X[j, :, :, :]
        other_product_data_flat = other_product_data.flatten()
        
        valid_indices_xx = ~np.isnan(product_data_flat) & ~np.isnan(other_product_data_flat)
        if np.sum(valid_indices_xx) > 1:
            # For self-correlation, it will be 1.0
            if i == j:
                 print(f"  和 {other_product_name} 之间的相关系数: 1.0000 (self-correlation)")
                 continue

            corr_matrix_xx = np.corrcoef(product_data_flat[valid_indices_xx], other_product_data_flat[valid_indices_xx])
            if corr_matrix_xx.shape == (2, 2):
                 print(f"  和 {other_product_name} 之间的相关系数: {corr_matrix_xx[0, 1]:.4f}")
            else:
                 print(f"  和 {other_product_name} 之间的相关系数: Error calculating correlation")
        else:
            print(f"  和 {other_product_name} 之间的相关系数: Not enough valid data points")
    print("-" * 30)







