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
basepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
savepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "intermediate", "nationwide", "PERSIANNdata")
if not os.path.exists(savepath):
    os.makedirs(savepath)
persiannfolder = os.path.join(basepath, "raw", "nationwide", "PERSIANN")
persiannfiles = os.listdir(persiannfolder)
print(f"总文件数：{len(persiannfiles)}")

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
        file = os.path.join(persiannfolder, persiannfiles[count])
        data = xr.open_dataset(file)
        data = data['precipitation'].values
        data = np.transpose(data, (1,2,0))
        data = np.squeeze(data)
        datalist.append(data)

    # 数据处理
    datalist = np.array(datalist)
    data = np.transpose(datalist,(2,1,0))
    china_data = data[24:168,288:544, :]
    
    # 应用掩膜
    mask = loadmat(os.path.join(basepath, "mask", "mask.mat"))['mask']
    #mask = np.flipud(np.transpose(mask, (1,0)))
    china_data[mask == 0] = np.nan
    china_data[china_data < 0.01] = 0

    # 保存数据
    savemat(os.path.join(savepath, f'PERSIANN_{year}.mat'), {'data': china_data})
    print(f"✓ {year}年数据已保存")
    
    # 释放内存
    del china_data, data, datalist

# 合并数据
if input("\n是否需要合并所有年份数据？(y/n): ").lower() == 'y':
    print("\n开始合并数据...")
    all_data = []
    for _, _, year in year_info:
        data = loadmat(os.path.join(savepath, f'PERSIANN_{year}.mat'))['data']
        all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=2)
    savemat(os.path.join(savepath, 'PERSIANN_2016_2020.mat'), {'data': all_data})
    print("✓ 合并完成！")



