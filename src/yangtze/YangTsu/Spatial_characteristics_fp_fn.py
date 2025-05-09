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
time_range_m = pd.date_range(start='2016-01-01', end='2020-12-31', freq='M')
time_range_d = pd.date_range(start='2016-01-01', end='2020-12-31', freq='D')
time_range_y = pd.date_range(start='2016-01-01', end='2020-12-31', freq='Y')
#X.shape = (6, 1827, 144, 256), 6个产品，1827天，144行，256列
#生成一个pd.DataFrame的列表, 用于储存X,并有不同的index, 第一个index是年, 第二个index是月, 第三个index是日, 日的index对齐上X.shape[1]
# Create the MultiIndex using the daily time range.
# The total number of days (1827) in this index matches X.shape[1].
multi_index = pd.MultiIndex.from_arrays(
    [time_range_d.year, time_range_d.month, time_range_d.day],
    names=['year', 'month', 'day']
)

# Initialize an empty list to store the DataFrames
X_dataframes = []

# Iterate over each product in X
# X.shape[0] is the number of products (6)
# X.shape[1] is the number of days (1827)
for product_idx in range(X.shape[0]):
    # Get data for the current product: shape (1827, 144, 256)
    product_data = X[product_idx, :, :, :]
    
    # Create a list of 2D spatial arrays (144, 256) for each day.
    # This list will form a single column in the DataFrame.
    daily_spatial_maps = [product_data[day_idx, :, :] for day_idx in range(product_data.shape[0])]
    
    # Create the DataFrame for the current product.
    # Each row is indexed by (year, month, day) and contains the corresponding 2D spatial map.
    df_product = pd.DataFrame({'spatial_map': daily_spatial_maps}, index=multi_index)
    
    X_dataframes.append(df_product)

# X_dataframes is now a list of pandas DataFrames.
# Each DataFrame contains the time-series data for one product.
# For example, to access the spatial map for the first product (index 0)
# on January 5, 2016, you can use:
# map_example = X_dataframes[0].loc[(2016, 1, 5), 'spatial_map']
# This will return a 2D numpy array of shape (144, 256).

#区域上年总发生误报漏报率
for i in range(X.shape[0]):
    product = PRODUCTS[i]
    for y in time_range_y:
        X_data = X[i, days: np.where(time_range_d.year == y.year)[0], :, :]
        days = np.where(time_range_d.year == y.year)[0]
        X_is_rain = np.where(X_data > rain_threshold, 1, 0)
        Y_is_rain = np.where(Y[days, :, :] > rain_threshold, 1, 0)
        #计算fp, fn, tp, tn
    fp = np.sum((X_is_rain == 1 & Y_is_rain == 0), axis=0)
    fn = np.sum((X_is_rain == 0 & Y_is_rain == 1), axis=0)
    tp = np.sum((X_is_rain == 1 & Y_is_rain == 1), axis=0)
    tn = np.sum((X_is_rain == 0 & Y_is_rain == 0), axis=0)
    print(fp.shape, fn.shape, tp.shape, tn.shape)
    
    
















