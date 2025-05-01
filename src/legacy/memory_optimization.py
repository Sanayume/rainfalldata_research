import os
import psutil
import gc
import numpy as np
import torch
from functools import wraps
import time

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / (1024 ** 3),  # GB
        'vms': memory_info.vms / (1024 ** 3),  # GB
    }

def get_gpu_memory():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        gpu_memory = {
            'allocated': torch.cuda.memory_allocated() / (1024 ** 3),  # GB
            'reserved': torch.cuda.memory_reserved() / (1024 ** 3),  # GB
            'max_allocated': torch.cuda.max_memory_allocated() / (1024 ** 3),  # GB
        }
    else:
        gpu_memory = {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
    return gpu_memory

def print_memory_stats(message="当前内存状态"):
    """打印内存使用统计信息"""
    cpu_memory = get_memory_usage()
    gpu_memory = get_gpu_memory()
    
    print(f"\n--- {message} ---")
    print(f"CPU 内存: {cpu_memory['rss']:.2f} GB (RSS), {cpu_memory['vms']:.2f} GB (VMS)")
    print(f"GPU 内存: {gpu_memory['allocated']:.2f} GB (已分配), "
          f"{gpu_memory['reserved']:.2f} GB (已预留), "
          f"{gpu_memory['max_allocated']:.2f} GB (最大已分配)")

def memory_monitor(func):
    """装饰器，用于监控函数执行前后的内存使用情况"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print_memory_stats(f"执行 {func.__name__} 前")
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        end_time = time.time()
        print_memory_stats(f"执行 {func.__name__} 后")
        print(f"执行时间: {end_time - start_time:.2f} 秒")
        
        return result
    return wrapper

def optimize_array_storage(array):
    """优化数组存储以减少内存占用"""
    if not isinstance(array, np.ndarray):
        return array
    
    # 如果是浮点型，转换为float32
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    
    # 如果是整数型，尝试使用较小的数据类型
    if np.issubdtype(array.dtype, np.integer):
        min_val = array.min()
        max_val = array.max()
        
        if min_val >= 0:  # 无符号整数
            if max_val <= 255:
                return array.astype(np.uint8)
            elif max_val <= 65535:
                return array.astype(np.uint16)
        else:  # 有符号整数
            if min_val >= -128 and max_val <= 127:
                return array.astype(np.int8)
            elif min_val >= -32768 and max_val <= 32767:
                return array.astype(np.int16)
                
    return array

def batch_process_array(func, array, batch_size=1000, axis=0):
    """批处理大型数组以减少内存使用"""
    if array.shape[axis] <= batch_size:
        return func(array)
    
    # 计算批次数量
    n_batches = (array.shape[axis] + batch_size - 1) // batch_size
    results = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, array.shape[axis])
        
        # 构建索引
        idx = [slice(None)] * array.ndim
        idx[axis] = slice(start_idx, end_idx)
        
        # 处理当前批次
        batch_result = func(array[tuple(idx)])
        results.append(batch_result)
        
        # 垃圾回收
        gc.collect()
    
    # 拼接结果
    if isinstance(results[0], np.ndarray):
        return np.concatenate(results, axis=axis)
    else:
        return results

def preprocess_with_reduced_memory(func):
    """装饰器，用于优化预处理函数的内存使用"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 转换输入参数为float32以减少内存占用
        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray) and np.issubdtype(arg.dtype, np.floating):
                new_args.append(arg.astype(np.float32))
            elif isinstance(arg, list) and all(isinstance(item, np.ndarray) for item in arg):
                new_args.append([item.astype(np.float32) if np.issubdtype(item.dtype, np.floating) else item 
                                for item in arg])
            else:
                new_args.append(arg)
        
        # 调用原始函数
        print_memory_stats(f"开始执行 {func.__name__}")
        start_time = time.time()
        
        result = func(*new_args, **kwargs)
        
        end_time = time.time()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print_memory_stats(f"完成执行 {func.__name__}")
        print(f"执行时间: {end_time - start_time:.2f} 秒")
        
        return result
    return wrapper

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        print("清理GPU内存...")
        torch.cuda.empty_cache()
        print("GPU内存已清理")
    else:
        print("CUDA不可用，无需清理GPU内存")

def test_memory_functions():
    """测试内存监控函数"""
    print_memory_stats("初始状态")
    
    # 创建一个大数组并检查内存使用
    print("\n创建大数组...")
    big_array = np.random.rand(10000, 10000).astype(np.float32)
    print(f"数组大小: {big_array.nbytes / (1024 ** 2):.2f} MB")
    print_memory_stats("创建大数组后")
    
    # 删除大数组并回收内存
    del big_array
    gc.collect()
    print_memory_stats("删除大数组后")
    
    # 测试GPU内存监控（如果可用）
    if torch.cuda.is_available():
        print("\n创建GPU张量...")
        gpu_tensor = torch.rand(5000, 5000, device='cuda')
        print(f"GPU张量大小: {gpu_tensor.element_size() * gpu_tensor.nelement() / (1024 ** 2):.2f} MB")
        print_memory_stats("创建GPU张量后")
        
        del gpu_tensor
        torch.cuda.empty_cache()
        print_memory_stats("删除GPU张量后")

if __name__ == "__main__":
    test_memory_functions()
