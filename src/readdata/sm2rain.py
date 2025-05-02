import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scale_utils import downscale, upscale
import sys
import io
from tqdm import tqdm

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 定义数据路径和常量
sm2rainfolder = 'sm2rain'
sm2rainfiles = os.listdir(sm2rainfolder)
print(f"总文件数：{len(sm2rainfiles)}")

# 定义年份信息
year_info = [
    (2016, "2016"),
    (2017, "2017"),
    (2018, "2018"),
    (2019, "2019"),
    (2020, "2020")
]

# 处理每一年的数据
for count, (year, year_str) in enumerate(year_info):
    print(f"\n处理{year}年数据...")
    
    # 读取数据
    print(f"读取文件：{sm2rainfiles[count]}")
    file_path = os.path.join(sm2rainfolder, sm2rainfiles[count])
    data = xr.open_dataset(file_path)
    data = data['rainfall'].values
    print(f"原始数据形状: {data.shape}")
    
    # 数据处理
    data = np.transpose(data, (1,2,0))
    china_data = data[360:720, 2520:3160, :]
    china_data = np.transpose(china_data, (1,0,2))
    china_data = np.fliplr(china_data)
    print(f"提取中国区域后形状: {china_data.shape}")
    
    # 降尺度处理
    print(f"{year}年：开始0.1°降尺度到0.05°...")
    china_data = downscale(china_data, factor=2)
    print(f"降尺度后形状: {china_data.shape}")
    
    # 加载掩膜
    mask = loadmat('mask.mat')['mask']
    
    # 升尺度处理
    print(f"{year}年：开始0.05°升尺度到0.25°...")
    china_data = upscale(china_data, factor=5, mask=mask)
    print(f"升尺度后形状: {china_data.shape}")
    
    # 处理数据
    china_data[mask == 0] = np.nan
    china_data[china_data < 0.01] = 0
    china_data = np.transpose(china_data, (1,0,2))
    china_data = np.flipud(china_data)
    
    # 可选：显示年度总降水量
    '''
    sum_data = np.nansum(china_data, axis=2)
    plt.figure(figsize=(10, 8))
    plt.imshow(sum_data)
    plt.colorbar(label='Total Precipitation (mm)')
    plt.title(f'Total Precipitation over China Region (0.25°) - {year}')
    plt.show()
    '''
    
    # 保存数据
    output_file = f'F:/rainfalldata/sm2raindata/sm2rain_{year_str}.mat'
    savemat(output_file, {'data': china_data})
    print(f"✓ {year}年数据已保存至: {output_file}")
    
    # 释放内存
    del data, china_data

print("所有年份数据处理完成!")

# 询问是否合并数据
if input("\n是否需要合并所有年份数据？(y/n): ").lower() == 'y':
    print("\n开始合并数据...")
    all_data = []
    for _, year_str in year_info:
        data = loadmat(f'F:/rainfalldata/sm2raindata/sm2rain_{year_str}.mat')['data']
        all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=2)
    savemat('F:/rainfalldata/sm2raindata/sm2rain_2016_2020.mat', {'data': all_data})
    print("✓ 合并完成！")

