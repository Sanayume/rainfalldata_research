import xarray as xr
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pylab as plt
import os
from glob import glob
import warnings

# 忽略特定警告
warnings.filterwarnings('ignore', 'variable .* has multiple fill values.*')

# 定义目录列表
folders = ['IMERG']

def process_dataset(file, folder):
    """处理单个数据集文件"""
    try:
        ds = xr.open_dataset(file, decode_cf=False)
        
        # 根据数据集类型选择正确的变量名
        if folder == 'CMORPH':
            precip = ds['cmorph']
        elif folder == 'IMERG':
            precip = ds['precipitationCal']
            if 'units' in ds['precipitationCal'].attrs and ds['precipitationCal'].attrs['units'] == 'mm/hr':
                precip = precip * 24
        else:
            precip = ds['precipitation']
            
        # 手动处理填充值
        precip = precip.where((precip != -9999.0) & (precip != -1.0))
        
        # 重命名变量
        if folder == 'CMORPH':
            ds = ds.rename({'cmorph': 'precipitation'})
        elif folder == 'IMERG':
            ds = ds.rename({'precipitationCal': 'precipitation'})
        else:
            ds['precipitation'] = precip
            
        return ds
    except Exception as e:
        print(f"Error processing {file}")
        print(f"Error details: {str(e)}")
        if 'ds' in locals():
            print(f"Available variables: {list(ds.variables.keys())}")
        return None

def process_folder(folder, batch_size=15):
    """分批处理文件夹中的数据"""
    nc_files = sorted(glob(f'D:/data/{folder}/*.nc') + glob(f'D:/data/{folder}/*.nc4'))
    print(f"Processing {folder}, found {len(nc_files)} files")
    
    if not nc_files:
        return
    
    # 分批处理文件
    merged_ds = None
    for i in range(0, len(nc_files), batch_size):
        batch_files = nc_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(nc_files)-1)//batch_size + 1}")
        
        # 处理当前批次的文件
        batch_datasets = []
        for file in batch_files:
            ds = process_dataset(file, folder)
            if ds is not None:
                batch_datasets.append(ds)
        
        if batch_datasets:
            # 合并当前批次
            batch_merged = xr.concat(batch_datasets, dim='time')
            
            # 与之前的结果合并
            if merged_ds is None:
                merged_ds = batch_merged
            else:
                merged_ds = xr.concat([merged_ds, batch_merged], dim='time')
            
            # 关闭数据集释放内存
            for ds in batch_datasets:
                ds.close()
            batch_merged.close()
    
    if merged_ds is not None:
        # 保存最终结果
        output_file = f'D:/data/{folder}_2016_merged.nc'
        merged_ds.to_netcdf(output_file)
        print(f"Saved merged file to {output_file}")
        merged_ds.close()

# 主程序
for folder in folders:
    process_folder(folder)
