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
cmorphfolder = 'CMORPH'
cmorphfiles = os.listdir(cmorphfolder)
print(f"总文件数：{len(cmorphfiles)}")

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
    for count in tqdm(range(start_idx, end_idx), desc=f"读取{year}年数据"):
        file = os.path.join(cmorphfolder, cmorphfiles[count])
        data = xr.open_dataset(file)
        data = data['cmorph'].values
        data = np.transpose(data, (1,2,0))
        data = np.squeeze(data)
        datalist.append(data)

    # 数据处理
    datalist = np.array(datalist)
    data = np.transpose(datalist, (1,2,0))
    data = np.flipud(data)
    print(f"处理后数据形状: {data.shape}")

    # 提取中国区域
    china_data = data[24:168, 288:544, :]
    print(f"中国区域数据形状: {china_data.shape}")

    # 应用掩膜
    mask = loadmat('mask.mat')['mask']
    mask = np.flipud(np.transpose(mask, (1,0)))
    china_data[mask == 0] = np.nan
    
    # 处理极小值
    china_data[china_data < 0.01] = 0

    # 保存结果
    savemat(f'F:/rainfalldata/CMORPHdata/CMORPH_{year}.mat', {'data': china_data})
    print(f"✓ {year}年数据已保存")
    
    # 释放内存
    del datalist, data, china_data
    print(f"{year}年处理完成!")

print("所有年份数据处理完成!")

# 询问是否合并数据
if input("\n是否需要合并所有年份数据？(y/n): ").lower() == 'y':
    print("\n开始合并数据...")
    all_data = []
    for _, _, year in year_info:
        data = loadmat(f'F:/rainfalldata/CMORPHdata/CMORPH_{year}.mat')['data']
        all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=2)
    savemat('F:/rainfalldata/CMORPHdata/CMORPH_2016_2020.mat', {'data': all_data})
    print("✓ 合并完成！")




