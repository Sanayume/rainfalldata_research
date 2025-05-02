import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import sys
import io
from tqdm import tqdm
from scale_utils import downscale, upscale
import pandas as pd

def save_to_excel(data, year, output_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "intermediate", "nationwide", "GSMAPdata")):
    """将数据保存为Excel文件，带有错误处理"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    try:
        nlat, nlon, ntime = data.shape
        reshaped_data = data.reshape(nlat * nlon, ntime)
        rows = [f'Lat{i}_Lon{j}' for i in range(nlat) for j in range(nlon)]
        columns = [f'Day_{i+1}' for i in range(ntime)]
        df = pd.DataFrame(reshaped_data, index=rows, columns=columns)
        
        excel_path = os.path.join(output_path, f'GSMAP_{year}.xlsx')
        if os.path.exists(excel_path):
            os.remove(excel_path)
        df.to_excel(excel_path, engine='openpyxl')
        print(f"✓ Excel文件已保存至: {excel_path}")
    except Exception as e:
        print(f"! 保存Excel失败: {str(e)}")
        try:
            csv_path = os.path.join(output_path, f'GSMAP_{year}.csv')
            df.to_csv(csv_path)
            print(f"✓ 已改为保存CSV文件: {csv_path}")
        except Exception as e:
            print(f"!! 保存CSV也失败了: {str(e)}")

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 定义数据路径和常量
basepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
savepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "intermediate", "nationwide", "GSMAPdata")
if not os.path.exists(savepath):
    os.makedirs(savepath)
gsmapfolder = os.path.join(basepath, "raw", "nationwide", "GSMAP")
gsmapfiles = os.listdir(gsmapfolder)
print(f"总文件数：{len(gsmapfiles)}")

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
    print(f"\n{'='*20} 处理{year}年数据 {'='*20}")
    print(f"从索引{start_idx}到{end_idx}")
    
    # 读取数据
    datalist = []
    for count in tqdm(range(start_idx, end_idx), desc=f"读取{year}年数据"):
        file_path = os.path.join(gsmapfolder, gsmapfiles[count])
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype='float32').reshape(1200, 3600)
            data = data * 24  # 转换单位
            datalist.append(data)
    
    # 数据处理
    data = np.array(datalist)
    data = np.transpose(data, (1,2,0))
    print(f"原始数据形状: {data.shape}")
    
    # 提取中国区域
    china_data = data[60:420, 720:1360, :]
    china_data = np.transpose(china_data, (1,0,2))
    china_data = np.fliplr(china_data)
    print(f"提取中国区域后形状: {china_data.shape}")
    
    # 降尺度处理
    print(f"{year}年：开始0.1°降尺度到0.05°...")
    china_data = downscale(china_data, factor=2)
    print(f"降尺度后形状: {china_data.shape}")
    
    # 加载掩膜并升尺度处理
    mask = loadmat(os.path.join(basepath, "mask", "mask.mat"))['mask']
    print(f"{year}年：开始0.05°升尺度到0.25°...")
    china_data = upscale(china_data, factor=5, mask=mask)
    print(f"升尺度后形状: {china_data.shape}")
    
    # 数据后处理
    china_data = np.flipud(np.transpose(china_data, (1,0,2)))
    china_data[china_data < 0.01] = 0
    
    # 可选：显示年度总降水量
    '''
    annual_precip = np.nansum(china_data, axis=2)
    plt.figure(figsize=(10, 8))
    plt.imshow(annual_precip)
    plt.colorbar(label='Total Precipitation (mm)')
    plt.title(f'Total Precipitation over China Region (0.25°) - {year}')
    plt.show()
    '''
    
    # 保存数据
    savemat(os.path.join(savepath, f'GSMAP_{year}.mat'), {'data': china_data})
    print(f"✓ {year}年数据已保存")
    
    # 可选：保存Excel格式
    # save_to_excel(china_data, year)
    
    # 释放内存
    del data, china_data, datalist

print("\n所有年份数据处理完成!")

# 询问是否合并数据
if input("\n是否需要合并所有年份数据？(y/n): ").lower() == 'y':
    print("\n开始合并数据...")
    all_data = []
    for _, _, year in year_info:
        data = loadmat(os.path.join(savepath, f'GSMAP_{year}.mat'))['data']
        all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=2)
    savemat(os.path.join(savepath, 'GSMAP_2016_2020.mat'), {'data': all_data})
    print("✓ 合并完成！")




