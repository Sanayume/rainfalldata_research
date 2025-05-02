import numpy as np
from tqdm import tqdm  # 添加进度条支持，如果没有安装可以通过 pip install tqdm 安装

def downscale(data, factor):
    """
    将数据降尺度（细化分辨率）- 完全匹配MATLAB版本
    """
    if data.ndim == 3:
        h, w, d = data.shape
        # 预分配结果数组，与MATLAB一致
        result = np.zeros((h * factor, w * factor, d))
        
        for k in tqdm(range(d), desc="降尺度处理"):
            for i in range(h):
                for j in range(w):
                    # 对应MATLAB的赋值方式
                    result[i*factor:(i+1)*factor, 
                          j*factor:(j+1)*factor, k] = data[i, j, k]
        return result
    elif data.ndim == 2:
        h, w = data.shape
        # 使用np.repeat快速复制数据，无需循环
        result = np.repeat(np.repeat(data, factor, axis=0), factor, axis=1)
        return result
    else:
        raise ValueError("输入数据必须是二维或三维数组")

def upscale(data, factor, mask=None):
    """
    将数据升尺度（粗化分辨率）- 完全匹配MATLAB版本
    """
    if data.ndim == 3:
        h, w, d = data.shape
        new_h, new_w = h // factor, w // factor
        
        # 预分配结果数组，使用zeros而不是NaN，与MATLAB一致
        result = np.zeros((new_h, new_w, d))
        
        if mask is not None:
            for k in tqdm(range(d), desc="升尺度处理"):
                for i in range(new_h):
                    for j in range(new_w):
                        if mask[i, j] == 1:
                            block = data[i*factor:(i+1)*factor, 
                                       j*factor:(j+1)*factor, k]
                            result[i, j, k] = np.nanmean(block)
                        else:
                            result[i, j, k] = np.nan
                
                # 处理小于0.01的值
                temp = result[:, :, k].copy()
                temp[mask == 0] = np.nan
                temp[temp < 0.01] = 0
                result[:, :, k] = temp
        else:
            # 无掩膜时的快速实现
            for k in tqdm(range(d), desc="升尺度处理"):
                for i in range(new_h):
                    for j in range(new_w):
                        result[i, j, k] = np.nanmean(data[i * factor:(i + 1) * factor, 
                                                     j * factor:(j + 1) * factor, k])
        
        return result
    elif data.ndim == 2:
        h, w = data.shape
        new_h, new_w = h // factor, w // factor
        
        # 预分配结果数组 - 使用NaN填充
        result = np.full((new_h, new_w), np.nan)
        
        # 使用reshape和mean进行快速聚合
        if mask is None:
            # 无掩膜时，直接对重塑后的数组求均值
            for i in range(factor):
                for j in range(factor):
                    sub_data = data[i::factor, j::factor]
                    if i == 0 and j == 0:
                        result = np.zeros_like(sub_data)
                    result += sub_data
            result /= (factor * factor)
            return result
        else:
            # 确保掩膜尺寸正确
            if mask.shape != (new_h, new_w) and mask.shape != (new_w, new_h):
                if mask.shape == (new_w, new_h):
                    mask = mask.T
                else:
                    raise ValueError(f"掩膜形状{mask.shape}与预期结果形状{(new_h, new_w)}不匹配")
            
            # 使用掩膜过滤结果
            for i in range(new_h):
                for j in range(new_w):
                    if i < mask.shape[0] and j < mask.shape[1] and mask[i, j] > 0:
                        block = data[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]
                        result[i, j] = np.nanmean(block)
            
            return result
    else:
        raise ValueError("输入数据必须是二维或三维数组")
