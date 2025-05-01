"""
GPU内存配置与管理 - 用于优化GPU内存使用并处理OOM错误
"""
import torch
import numpy as np
import gc
import logging
import os
import psutil
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return {
            'total': 0,
            'allocated': 0,
            'reserved': 0,
            'free': 0,
            'available': 0
        }
    
    device = torch.cuda.current_device()
    
    # 总内存
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    # 已分配内存
    allocated_memory = torch.cuda.memory_allocated(device)
    
    # 已保留内存（缓存）
    reserved_memory = torch.cuda.memory_reserved(device)
    
    # 空闲内存（缓存中但未使用）
    free_memory = reserved_memory - allocated_memory
    
    # 可用内存（总内存减去保留内存）
    available_memory = total_memory - reserved_memory
    
    return {
        'total': total_memory / (1024**3),
        'allocated': allocated_memory / (1024**3),
        'reserved': reserved_memory / (1024**3),
        'free': free_memory / (1024**3),
        'available': available_memory / (1024**3)
    }

def log_memory_usage(message=""):
    """记录当前内存使用情况"""
    # GPU内存信息
    gpu_info = get_gpu_memory_info()
    
    # CPU内存信息
    cpu_ram = psutil.virtual_memory()
    cpu_info = {
        'total': cpu_ram.total / (1024**3),
        'available': cpu_ram.available / (1024**3),
        'used': cpu_ram.used / (1024**3),
        'percent': cpu_ram.percent
    }
    
    # 记录日志
    if message:
        logger.info(f"内存使用 - {message}:")
    else:
        logger.info("当前内存使用:")
    
    if torch.cuda.is_available():
        logger.info(f"GPU 总内存: {gpu_info['total']:.2f} GB")
        logger.info(f"GPU 已分配: {gpu_info['allocated']:.2f} GB")
        logger.info(f"GPU 已保留: {gpu_info['reserved']:.2f} GB")
        logger.info(f"GPU 可用: {gpu_info['available']:.2f} GB")
    
    logger.info(f"CPU 总内存: {cpu_info['total']:.2f} GB")
    logger.info(f"CPU 已使用: {cpu_info['used']:.2f} GB ({cpu_info['percent']}%)")
    logger.info(f"CPU 可用: {cpu_info['available']:.2f} GB")

def estimate_tensor_size(shape, dtype=torch.float32):
    """估计张量大小（GB）"""
    # 获取数据类型的字节数
    if dtype == torch.float32 or dtype == np.float32:
        bytes_per_element = 4
    elif dtype == torch.float16 or dtype == np.float16:
        bytes_per_element = 2
    elif dtype == torch.int64 or dtype == np.int64:
        bytes_per_element = 8
    elif dtype == torch.int32 or dtype == np.int32:
        bytes_per_element = 4
    else:
        bytes_per_element = 4  # 默认估计
    
    # 计算元素数量
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    # 计算总字节数并转换为GB
    size_gb = (num_elements * bytes_per_element) / (1024**3)
    
    return size_gb

def optimize_gpu_memory_settings():
    """优化GPU内存设置"""
    if torch.cuda.is_available():
        # 启用TF32以提高性能（适用于安培及以上架构）
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("已启用TF32矩阵乘法")
        
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            logger.info("已启用cudnn TF32")
        
        # 启用cudnn基准测试
        torch.backends.cudnn.benchmark = True
        logger.info("已启用cudnn基准测试")
        
        # 不需要确定性结果时，关闭确定性可提高性能
        torch.backends.cudnn.deterministic = False
        
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 记录优化后的内存信息
        log_memory_usage("优化GPU设置后")
        
        return True
    else:
        logger.warning("未检测到可用的CUDA设备")
        return False

def get_optimal_batch_size(dataset_size, min_batch=32, max_batch=16384):
    """根据可用GPU内存和数据集大小计算最优批次大小"""
    if not torch.cuda.is_available():
        return min_batch
    
    # 获取可用内存
    mem_info = get_gpu_memory_info()
    available_gb = mem_info['available']
    
    # 内存余量系数（保留一些内存给其他操作）
    mem_margin = 0.7
    
    # 每个样本的估计内存需求（根据经验调整）
    # 这个值需要根据实际模型和数据进行估计
    gb_per_sample = 0.00001  # 假设每个样本占用约10KB内存
    
    # 计算可以容纳的最大批次大小
    max_by_memory = int(available_gb * mem_margin / gb_per_sample)
    
    # 取较小值，并确保在min_batch和max_batch之间
    batch_size = min(max_by_memory, dataset_size)
    batch_size = max(min_batch, batch_size)
    batch_size = min(max_batch, batch_size)
    
    # 将批次大小调整为2的幂次方（通常在GPU上效率更高）
    power_of_2 = 2 ** int(np.log2(batch_size))
    if batch_size - power_of_2 > power_of_2 * 2 - batch_size:
        batch_size = power_of_2 * 2
    else:
        batch_size = power_of_2
    
    return batch_size

@contextmanager
def gpu_memory_limit(fraction=0.95):
    """设置GPU内存使用限制的上下文管理器"""
    if torch.cuda.is_available():
        try:
            # 记录当前限制
            old_fraction = torch.cuda.get_per_process_memory_fraction()
            # 设置新的限制
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"GPU内存使用限制设置为{fraction*100}%")
            
            # 让出控制权
            yield
            
            # 恢复原始限制
            torch.cuda.set_per_process_memory_fraction(old_fraction)
            logger.info("恢复原始GPU内存使用限制")
        except:
            # 如果无法设置限制，继续执行而不抛出错误
            yield
    else:
        # 如果没有GPU，直接让出控制权
        yield

def should_use_amp():
    """判断是否应该使用自动混合精度"""
    if not torch.cuda.is_available():
        return False
        
    # 获取当前设备的计算能力
    capability = torch.cuda.get_device_capability(0)
    
    # 转换为浮点数以便比较，例如(7, 5) -> 7.5
    cap_float = capability[0] + capability[1]/10
    
    # Volta及以上架构（计算能力7.0+）支持Tensor Cores
    return cap_float >= 7.0

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    print("="*40)
    print(" GPU内存配置与管理工具测试 ")
    print("="*40)
    
    # 优化GPU内存设置
    optimize_gpu_memory_settings()
    
    # 记录当前内存使用
    log_memory_usage("初始状态")
    
    # 测试张量大小估计
    test_shapes = [
        (1000, 1000),           # 矩阵
        (10, 1000, 1000),       # 批次矩阵
        (100, 224, 224, 3)      # 批次图像
    ]
    
    print("\n张量大小估计:")
    for shape in test_shapes:
        size = estimate_tensor_size(shape)
        print(f"形状 {shape}: {size:.4f} GB")
    
    # 测试最优批次大小计算
    print("\n最优批次大小估计:")
    dataset_sizes = [1000, 10000, 100000, 1000000]
    for size in dataset_sizes:
        batch_size = get_optimal_batch_size(size)
        print(f"数据集大小 {size}: 推荐批次大小 {batch_size}")
    
    # 测试是否应该使用AMP
    print(f"\n是否应该使用自动混合精度: {should_use_amp()}")
    
    if torch.cuda.is_available():
        print(f"\nGPU计算能力: {torch.cuda.get_device_capability(0)}")
    
    print("="*40)
