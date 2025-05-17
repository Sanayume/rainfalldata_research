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



basepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
savepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "intermediate", "nationwide", "IMERGdata")
if not os.path.exists(savepath):
    os.makedirs(savepath)
imergfolder = os.path.join(basepath, "raw", "nationwide", "IMERG")
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
    
    data = np.transpose(datalist, (1, 2, 0))

    data = np.nansum(data, axis=2)
    data = np.transpose(data, (1, 0))
    data = np.flipud(data)
    plt.imshow(data)
    plt.colorbar()
    plt.show()