import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import sys
import io
from tqdm import tqdm

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 定义数据路径和常量
chirpsfolder = 'CHIRPS'
chirpsfiles = os.listdir(chirpsfolder)
print(f"总文件数：{len(chirpsfiles)}")

# 定义年份信息
year_info = [
    (2016, "2016"),
    (2017, "2017"),
    (2018, "2018"),
    (2019, "2019"),
    (2020, "2020")
]

# 加载掩膜
mask = loadmat("mask.mat")["mask"]
mask = np.flipud(np.transpose(mask, (1,0)))

# 处理每一年的数据
for count, (year, year_str) in enumerate(year_info):
    print(f"\n{'='*20} 处理{year}年数据 {'='*20}")
    
    # 读取数据
    print(f"读取文件：{chirpsfiles[count]}")
    file_path = os.path.join(chirpsfolder, chirpsfiles[count])
    data = xr.open_dataset(file_path)
    data = data['precip'].values
    print(f"原始数据形状: {data.shape}")
    
    # 数据预处理
    data = np.transpose(data, (1,2,0))
    data = np.flipud(data)
    
    # 可选：显示原始数据年降水量
    '''
    annual_precip = np.sum(data, axis=2)
    plt.figure(figsize=(10, 8))
    plt.imshow(annual_precip)
    plt.colorbar(label='Total Precipitation (mm)')
    plt.title(f'Original CHIRPS Precipitation - {year}')
    plt.show()
    '''
    
    # 提取中国区域数据
    china_data = np.zeros((144, 256, data.shape[2]))
    china_data[0:16, :, :] = np.nan
    china_data[16:144, :, :] = data[0:128, 1007:1263, :]
    print(f"中国区域数据形状: {china_data.shape}")
    
    # 应用掩膜和阈值
    china_data[mask == 0] = np.nan
    china_data[china_data < 0.01] = 0
    
    # 可选：显示处理后的年降水量
    '''
    processed_precip = np.nansum(china_data, axis=2)
    plt.figure(figsize=(10, 8))
    plt.imshow(processed_precip)
    plt.colorbar(label='Total Precipitation (mm)')
    plt.title(f'Processed China Region Precipitation - {year}')
    plt.show()
    '''
    
    # 保存数据
    output_file = f'F:/rainfalldata/CHIRPSdata/chirps_{year_str}.mat'
    savemat(output_file, {'data': china_data})
    print(f"✓ {year}年数据已保存至: {output_file}")
    
    # 释放内存
    del data, china_data

print("\n所有年份数据处理完成!")

# 询问是否合并数据
if input("\n是否需要合并所有年份数据？(y/n): ").lower() == 'y':
    print("\n开始合并数据...")
    all_data = []
    for _, year_str in year_info:
        data = loadmat(f'F:/rainfalldata/CHIRPSdata/chirps_{year_str}.mat')['data']
        all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=2)
    savemat('F:/rainfalldata/CHIRPSdata/chirps_2016_2020.mat', {'data': all_data})
    print("✓ 合并完成！")






