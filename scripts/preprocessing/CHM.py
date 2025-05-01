import xarray as xr
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from scale_utils import downscale, upscale
from tqdm import tqdm
import os

# 定义数据文件和路径
DATA_FILE = 'CHM_PRE_0.1dg_19612022.nc'
OUTPUT_PATH = 'CHMdata'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 定义年份信息
year_info = [
    ("2016-01-01", "2016-12-31", 2016),
    ("2017-01-01", "2017-12-31", 2017),
    ("2018-01-01", "2018-12-31", 2018),
    ("2019-01-01", "2019-12-31", 2019),
    ("2020-01-01", "2020-12-31", 2020)
]

# 加载掩膜
mask = loadmat('mask.mat')['mask']

# 可选：显示掩膜
'''
plt.figure(figsize=(10, 8))
plt.imshow(mask)
plt.colorbar(label='Mask Value')
plt.title('CHM Mask')
plt.show()
'''

# 处理每一年的数据
for start_date, end_date, year in year_info:
    print(f"\n{'='*20} 处理{year}年数据 {'='*20}")
    
    # 读取数据
    print(f"读取{start_date}至{end_date}的数据...")
    ds = xr.open_dataset(DATA_FILE)
    data = ds.sel(time=slice(start_date, end_date))
    data = data.variables['pre'].values
    print(f"原始数据形状: {data.shape}")
    
    # 数据预处理
    data = np.transpose(data, (2,1,0))
    data[data == -99.9] = np.nan
    
    # 可选：显示原始年降水量
    '''
    annual_precip = np.nansum(data, axis=2)
    plt.figure(figsize=(10, 8))
    plt.imshow(annual_precip)
    plt.colorbar(label='Total Precipitation (mm)')
    plt.title(f'Original CHM Precipitation - {year}')
    plt.show()
    '''
    
    # 降尺度处理
    print(f"{year}年：开始0.1°降尺度到0.05°...")
    data = downscale(data, factor=2)
    print(f"降尺度后形状: {data.shape}")
    
    # 升尺度处理
    print(f"{year}年：开始0.05°升尺度到0.25°...")
    data = upscale(data, factor=5, mask=mask)
    print(f"升尺度后形状: {data.shape}")
    
    # 可选：显示处理后的年降水量
    data = np.transpose(data, (1,0,2))
    data = np.flipud(data)
    '''
    processed_precip = np.nansum(data, axis=2)
    plt.figure(figsize=(10, 8))
    plt.imshow(processed_precip)
    plt.colorbar(label='Total Precipitation (mm)')
    plt.title(f'Processed CHM Precipitation - {year}')
    plt.show()
    '''
    
    # 保存数据
    output_file = os.path.join(OUTPUT_PATH, f'CHM_{year}.mat')
    savemat(output_file, {'data': data})
    print(f"✓ {year}年数据已保存至: {output_file}")
    
    # 释放内存
    del data, ds

print("\n所有年份数据处理完成!")

# 询问是否合并数据
if input("\n是否需要合并所有年份数据？(y/n): ").lower() == 'y':
    print("\n开始合并数据...")
    all_data = []
    for _, _, year in year_info:
        data = loadmat(os.path.join(OUTPUT_PATH, f'CHM_{year}.mat'))['data']
        all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=2)
    savemat(os.path.join(OUTPUT_PATH, 'CHM_2016_2020.mat'), {'data': all_data})
    print("✓ 合并完成！")
