"""
GPU工具模块，用于处理设备不匹配和GPU相关操作
"""

import numpy as np
import warnings
import subprocess
import os
import sys
from functools import wraps
import time

# 尝试导入GPU相关库
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not installed. Some GPU functionalities may be limited.")
    TORCH_AVAILABLE = False

def check_gpu_availability():
    """
    检查系统是否有可用GPU以及相关库是否已安装
    
    返回:
        dict: 包含GPU可用性和支持信息的字典
    """
    gpu_info = {
        'cuda_available': False,
        'cupy_available': CUPY_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'nvidia_gpus': [],
        'recommended_device': 'cpu',
        'gpu_memory_info': {}
    }
    
    # 检查CUDA是否可用
    if CUPY_AVAILABLE:
        try:
            gpu_info['cuda_available'] = cp.cuda.is_available()
            if gpu_info['cuda_available']:
                gpu_info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
                gpu_info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
        except Exception as e:
            print(f"检查CuPy CUDA可用性时出错: {str(e)}")
    
    # 或者通过PyTorch检查CUDA
    if not gpu_info['cuda_available'] and TORCH_AVAILABLE:
        try:
            gpu_info['cuda_available'] = torch.cuda.is_available()
            if gpu_info['cuda_available']:
                gpu_info['gpu_count'] = torch.cuda.device_count()
                if hasattr(torch.version, 'cuda'):
                    gpu_info['cuda_version'] = torch.version.cuda
        except Exception as e:
            print(f"检查PyTorch CUDA可用性时出错: {str(e)}")
    
    # 尝试使用nvidia-smi获取GPU信息
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu', 
             '--format=csv,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
        )
        
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split('\n')):
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 5:
                    gpu_name = parts[0]
                    memory_total = parts[1]
                    memory_free = parts[2]
                    memory_used = parts[3]
                    temperature = parts[4]
                    
                    gpu_info['nvidia_gpus'].append({
                        'id': i,
                        'name': gpu_name,
                        'memory_total': memory_total,
                        'memory_free': memory_free,
                        'memory_used': memory_used,
                        'temperature': temperature
                    })
                    
                    gpu_info['gpu_memory_info'][i] = {
                        'total': memory_total,
                        'free': memory_free,
                        'used': memory_used
                    }
    except (subprocess.SubprocessError, FileNotFoundError, TimeoutError) as e:
        print(f"获取NVIDIA GPU信息时出错: {str(e)}")
    
    # 决定推荐设备
    if gpu_info['cuda_available'] and (gpu_info.get('gpu_count', 0) > 0 or len(gpu_info['nvidia_gpus']) > 0):
        # 找到内存最大的GPU
        if gpu_info['nvidia_gpus']:
            max_mem = 0
            best_gpu = 0
            for gpu in gpu_info['nvidia_gpus']:
                try:
                    # 提取内存数值
                    mem_str = gpu['memory_total']
                    mem_val = float(mem_str.split()[0])
                    if mem_val > max_mem:
                        max_mem = mem_val
                        best_gpu = gpu['id']
                except (ValueError, IndexError, KeyError):
                    continue
            
            gpu_info['recommended_device'] = f'cuda:{best_gpu}'
        else:
            gpu_info['recommended_device'] = 'cuda:0'
    
    return gpu_info

def to_device(data, device='cuda', dtype=None):
    """
    将数据转移到指定设备（GPU或CPU）
    
    参数:
        data: 要转移的数据（numpy数组或其他）
        device: 目标设备，'cuda'或'cpu'
        dtype: 可选，数据类型
        
    返回:
        在目标设备上的数据
    """
    if device.startswith('cuda') and CUPY_AVAILABLE:
        try:
            # 首先转换为numpy数组（如果不是）
            if not isinstance(data, (np.ndarray, cp.ndarray)):
                data = np.array(data)
            
            # 然后转换为CuPy数组
            if isinstance(data, np.ndarray):
                if dtype is not None:
                    return cp.array(data, dtype=dtype)
                else:
                    return cp.array(data)
            elif isinstance(data, cp.ndarray):
                # 如果已经是CuPy数组，确保它在正确的设备上
                if dtype is not None:
                    return data.astype(dtype)
                else:
                    return data
        except Exception as e:
            warnings.warn(f"将数据转移到GPU时出错: {str(e)}。将回退到CPU。")
            return data
    
    # 返回CPU上的数据
    if isinstance(data, cp.ndarray):
        try:
            data = data.get()
        except Exception:
            pass
    
    if dtype is not None and isinstance(data, np.ndarray):
        return data.astype(dtype)
    
    return data

def copy_if_needed(data, device='cuda'):
    """
    仅在必要时复制数据到指定设备
    
    参数:
        data: 要转移的数据
        device: 目标设备
    
    返回:
        在目标设备上的数据
    """
    if device.startswith('cuda') and CUPY_AVAILABLE:
        if isinstance(data, np.ndarray):
            return to_device(data, device)
        elif isinstance(data, cp.ndarray):
            return data
    else:
        if isinstance(data, cp.ndarray):
            return to_device(data, 'cpu')
        elif isinstance(data, np.ndarray):
            return data
    
    return data

def gpu_timer(func):
    """
    装饰器，用于测量GPU函数执行时间
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if CUPY_AVAILABLE:
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            result = func(*args, **kwargs)
            end.record()
            end.synchronize()
            msec = cp.cuda.get_elapsed_time(start, end)
            print(f"{func.__name__} 执行时间: {msec:.4f} ms")
            return result
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = (time.time() - start_time) * 1000
            print(f"{func.__name__} 执行时间: {elapsed:.4f} ms")
            return result
    
    return wrapper

def get_optimal_batch_size(array_shape, dtype=np.float32, max_memory_gb=4):
    """
    根据可用GPU内存计算最佳批处理大小
    
    参数:
        array_shape: 数组形状的元组，第一维是样本数
        dtype: 数据类型
        max_memory_gb: 最大允许使用的GPU内存（GB）
        
    返回:
        最佳批处理大小
    """
    if not CUPY_AVAILABLE:
        # 如果没有CuPy，直接返回一个保守的批处理大小
        return min(1000, array_shape[0])
    
    # 计算单个样本的字节数
    element_size = np.dtype(dtype).itemsize
    shape_without_samples = array_shape[1:]
    elements_per_sample = np.prod(shape_without_samples)
    bytes_per_sample = element_size * elements_per_sample
    
    # 估计可用GPU内存
    try:
        gpu_info = check_gpu_availability()
        if gpu_info['nvidia_gpus']:
            # 尝试获取可用内存
            memory_free = gpu_info['nvidia_gpus'][0]['memory_free']
            if isinstance(memory_free, str) and 'MiB' in memory_free:
                available_gb = float(memory_free.split()[0]) / 1024.0
                max_memory_gb = min(max_memory_gb, available_gb * 0.8)  # 使用80%的可用内存
    except Exception:
        # 如果无法获取GPU信息，使用默认值
        pass
    
    # 将GB转换为字节并计算最大样本数
    max_bytes = max_memory_gb * 1024**3
    max_samples = max_bytes / bytes_per_sample
    
    # 确保批处理大小不超过样本总数
    batch_size = min(int(max_samples), array_shape[0])
    
    # 批处理大小至少为1
    return max(1, batch_size)

def batch_process(data, func, batch_size=None, use_gpu=True, *args, **kwargs):
    """
    使用批处理方式处理大型数组
    
    参数:
        data: 要处理的输入数据
        func: 处理函数
        batch_size: 批处理大小，如果为None则自动计算
        use_gpu: 是否使用GPU
        *args, **kwargs: 传递给处理函数的额外参数
        
    返回:
        处理后的数据
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # 如果数据较小，直接处理
    if data.shape[0] < 1000 or (batch_size is not None and data.shape[0] <= batch_size):
        if use_gpu and CUPY_AVAILABLE:
            data_device = to_device(data, 'cuda')
            result = func(data_device, *args, **kwargs)
            return to_device(result, 'cpu')
        else:
            return func(data, *args, **kwargs)
    
    # 计算批处理大小
    if batch_size is None:
        batch_size = get_optimal_batch_size(data.shape)
    
    # 使用批处理
    results = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i+batch_size]
        
        if use_gpu and CUPY_AVAILABLE:
            batch = to_device(batch, 'cuda')
            batch_result = func(batch, *args, **kwargs)
            batch_result = to_device(batch_result, 'cpu')
        else:
            batch_result = func(batch, *args, **kwargs)
            
        results.append(batch_result)
    
    # 组合结果
    try:
        return np.vstack(results)
    except Exception:
        return results

def memory_limit_decorator(max_memory_gb=4):
    """
    装饰器，限制函数使用的最大GPU内存
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if CUPY_AVAILABLE:
                # 设置内存池限制
                try:
                    mempool = cp.get_default_memory_pool()
                    mempool.set_limit(size=max_memory_gb * 1024**3)
                except Exception as e:
                    warnings.warn(f"无法设置GPU内存限制: {str(e)}")
            
            # 调用原始函数
            result = func(*args, **kwargs)
            
            # 清理内存
            if CUPY_AVAILABLE:
                try:
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                except Exception:
                    pass
            
            return result
        return wrapper
    return decorator

def print_gpu_status():
    """打印当前GPU状态信息"""
    gpu_info = check_gpu_availability()
    
    print("\n=== GPU状态信息 ===")
    print(f"CUDA可用: {gpu_info['cuda_available']}")
    print(f"CuPy可用: {gpu_info['cupy_available']}")
    print(f"PyTorch可用: {gpu_info['torch_available']}")
    
    if gpu_info['cuda_available']:
        print(f"CUDA版本: {gpu_info.get('cuda_version', '未知')}")
        print(f"发现的GPU数量: {len(gpu_info['nvidia_gpus'])}")
        
        for i, gpu in enumerate(gpu_info['nvidia_gpus']):
            print(f"\nGPU {i}: {gpu['name']}")
            print(f"  内存总量: {gpu['memory_total']}")
            print(f"  可用内存: {gpu['memory_free']}")
            print(f"  已用内存: {gpu['memory_used']}")
            print(f"  温度: {gpu['temperature']}")
    
    print(f"\n推荐设备: {gpu_info['recommended_device']}")
    print("==================\n")

if __name__ == "__main__":
    # 测试模块功能
    print_gpu_status()
    
    # 测试数据转移功能
    if CUPY_AVAILABLE:
        test_data = np.random.random((1000, 10))
        gpu_data = to_device(test_data, 'cuda')
        print(f"数据已转移到: {type(gpu_data)}")
        cpu_data = to_device(gpu_data, 'cpu')
        print(f"数据已转回: {type(cpu_data)}")
        
        # 测试批处理功能
        @gpu_timer
        def dummy_process(x):
            return x * 2
            
        result = batch_process(np.random.random((5000, 100)), dummy_process, batch_size=1000)
        print(f"批处理结果形状: {result.shape}")
