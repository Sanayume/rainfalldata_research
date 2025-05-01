import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scale_utils import downscale, upscale
import sys
import io
from tqdm import tqdm  # 添加进度条支持
# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')



imergfolder = 'IMERG'
imergfiles = os.listdir(imergfolder)
print(len(imergfiles))

# 定义年份信息
year_info = [
    (0, 366, 2016),     # 2016年（闰年）
    (366, 731, 2017),   # 2017年
    (731, 1096, 2018),  # 2018年
    (1096, 1461, 2019), # 2019年
    (1461, 1827, 2020)  # 2020年（闰年）
]

# 处理每一年的数据
for start_idx, end_idx, year in year_info:
    print(f"\n处理{year}年数据...")
    print(f"从索引{start_idx}到{end_idx}")
    
    # 读取数据
    datalist = []
    for count in range(start_idx, end_idx):
        print(f"[{count+1}/{1827}] Processing {imergfiles[count]}")
        file = imergfiles[count]
        file = os.path.join(imergfolder, file)
        data = xr.open_dataset(file)
        data = data['precipitationCal'].values
        data = np.transpose(data, (1,2,0))
        data = np.squeeze(data)
        datalist.append(data)

    # 转换为numpy数组
    datalist = np.array(datalist)
    
    # 提取中国区域
    china_data = datalist[:, 2520:3160, 1080:1440]
    china_data = np.transpose(china_data, (1, 2, 0))
    
    # 降尺度处理
    print(f"{year}年：开始0.1°降尺度到0.05°...")
    data_005deg = downscale(china_data, factor=2)
    
    # 加载掩膜
    mask = loadmat('mask.mat')
    mask = mask['mask']
    '''
    temp = np.transpose(mask, (1, 0))
    temp = np.flipud(temp)
    plt.imshow(temp)
    plt.show()
    '''
    expected_shape = (data_005deg.shape[0]//5, data_005deg.shape[1]//5)
    if mask.shape != expected_shape:
        if mask.shape == (expected_shape[1], expected_shape[0]):
            mask = mask.T
    
    # 升尺度处理
    print(f"{year}年：开始0.05°升尺度到0.25°...")
    data_025deg = upscale(data_005deg, factor=5, mask=mask)
    
    # 处理极小值
    data_025deg[data_025deg < 0.01] = 0
    # 显示年度总降水量
    data_025deg = np.transpose(data_025deg, (1, 0, 2))
    data_025deg = np.flipud(data_025deg)
    '''
    sum_data = np.nansum(data_025deg, axis=2)
    plt.figure(figsize=(10, 8))
    plt.imshow(sum_data)
    plt.colorbar(label='Total Precipitation (mm)')
    plt.title(f'Total Precipitation over China Region (0.25°) - {year}')
    plt.show()
    '''
    # 保存结果
    print(f"保存{year}年结果...")
    savemat(f'F:/rainfalldata/IMERGdata/IMERG_{year}.mat', {'data': data_025deg})
    
    # 释放内存
    del datalist, china_data, data_005deg, data_025deg
    print(f"{year}年处理完成!")

print("所有年份数据处理完成!")

# 可选：合并所有年份数据
print("是否需要合并所有年份数据？(y/n)")
if input().lower() == 'y':
    print("开始合并数据...")
    all_data = []
    for _, _, year in year_info:
        data = loadmat(f'F:/rainfalldata/IMERGdata/IMERG_{year}.mat')['data']
        all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=2)
    savemat('F:/rainfalldata/IMERGdata/IMERG_2016_2020.mat', {'data': all_data})
    print("合并完成！")


