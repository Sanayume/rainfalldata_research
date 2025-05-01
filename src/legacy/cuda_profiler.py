"""
CUDA性能分析器 - 用于监控和分析GPU使用情况
"""
import torch
import time
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import psutil
import GPUtil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

class CUDAProfiler:
    """CUDA性能分析器，用于监控GPU使用情况"""
    def __init__(self, interval=1.0, log_file="gpu_profile.csv"):
        """
        初始化性能分析器
        
        Args:
            interval: 采样间隔（秒）
            log_file: 日志文件名
        """
        self.interval = interval
        self.log_file = log_file
        self.running = False
        self.thread = None
        self.data = []
        
        # 初始化NVML
        try:
            nvmlInit()
            self.nvml_initialized = True
        except:
            self.nvml_initialized = False
            print("警告: 无法初始化NVML，某些指标将不可用")
    
    def start(self):
        """开始监控"""
        if self.running:
            print("监控器已在运行")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        print(f"GPU性能监控已启动，采样间隔: {self.interval}秒")
    
    def stop(self):
        """停止监控"""
        if not self.running:
            print("监控器未在运行")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        
        # 保存数据
        self._save_data()
        print(f"GPU性能监控已停止，数据保存到 {self.log_file}")
    
    def _monitor(self):
        """监控线程"""
        start_time = time.time()
        
        while self.running:
            # 收集GPU指标
            metrics = self._collect_metrics()
            metrics['timestamp'] = time.time() - start_time
            self.data.append(metrics)
            
            # 等待下一个采样周期
            time.sleep(self.interval)
    
    def _collect_metrics(self):
        """收集GPU指标"""
        metrics = {
            'gpu_utilization': [],
            'memory_used': [],
            'memory_total': [],
            'power_usage': [],
            'temperature': [],
            'cpu_utilization': psutil.cpu_percent(),
            'ram_utilization': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            # 检查每个GPU
            for i in range(torch.cuda.device_count()):
                try:
                    # 使用GPUtil获取利用率信息
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        metrics['gpu_utilization'].append(gpus[i].load * 100)
                        metrics['memory_used'].append(gpus[i].memoryUsed)
                        metrics['memory_total'].append(gpus[i].memoryTotal)
                        metrics['temperature'].append(gpus[i].temperature)
                    
                    # 使用NVML获取更详细的信息
                    if self.nvml_initialized:
                        try:
                            handle = nvmlDeviceGetHandleByIndex(i)
                            info = nvmlDeviceGetMemoryInfo(handle)
                            
                            # 覆盖内存信息，因为NVML通常更准确
                            metrics['memory_used'][-1] = info.used / 1024**2
                            metrics['memory_total'][-1] = info.total / 1024**2
                        except:
                            pass
                    
                    # 获取CUDA内存信息
                    metrics['cuda_allocated'] = torch.cuda.memory_allocated(i) / 1024**2
                    metrics['cuda_reserved'] = torch.cuda.memory_reserved(i) / 1024**2
                    metrics['cuda_max_allocated'] = torch.cuda.max_memory_allocated(i) / 1024**2
                    
                except Exception as e:
                    print(f"收集GPU {i} 指标时出错: {str(e)}")
        
        # 计算平均值
        for key in ['gpu_utilization', 'memory_used', 'memory_total', 'power_usage', 'temperature']:
            if metrics[key]:
                metrics[key] = sum(metrics[key]) / len(metrics[key])
            else:
                metrics[key] = 0
        
        return metrics
    
    def _save_data(self):
        """保存监控数据"""
        if not self.data:
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(self.data)
        
        # 保存为CSV
        df.to_csv(self.log_file, index=False)
        
        # 生成图表
        self._generate_plots(df)
    
    def _generate_plots(self, df):
        """生成性能图表"""
        plt.style.use('ggplot')
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # 1. GPU利用率
        ax = axes[0]
        ax.plot(df['timestamp'], df['gpu_utilization'], 'r-', linewidth=2)
        ax.set_ylabel('GPU利用率 (%)')
        ax.set_title('GPU利用率随时间变化')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # 2. GPU内存使用
        ax = axes[1]
        ax.plot(df['timestamp'], df['memory_used'], 'b-', linewidth=2, label='已用')
        ax.plot(df['timestamp'], df['memory_total'], 'g--', linewidth=1, label='总量')
        ax.set_ylabel('GPU内存 (MB)')
        ax.set_title('GPU内存使用随时间变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. CPU利用率和系统内存
        ax = axes[2]
        ax.plot(df['timestamp'], df['cpu_utilization'], 'm-', linewidth=2, label='CPU利用率')
        ax.plot(df['timestamp'], df['ram_utilization'], 'c-', linewidth=2, label='RAM利用率')
        ax.set_ylabel('利用率 (%)')
        ax.set_xlabel('时间 (秒)')
        ax.set_title('系统资源使用随时间变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # 保存图表
        fig.tight_layout()
        plot_file = os.path.splitext(self.log_file)[0] + '.png'
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)

def profile_model_inference(model, sample_input, num_runs=100, warmup=10, device='cuda'):
    """
    分析模型推理性能
    
    Args:
        model: PyTorch模型
        sample_input: 样本输入
        num_runs: 运行次数
        warmup: 预热次数
        device: 设备
    """
    model.eval()
    model = model.to(device)
    sample_input = sample_input.to(device)
    
    # 预热
    print(f"预热中 ({warmup} 次迭代)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
    
    torch.cuda.synchronize()
    
    # 计时
    print(f"执行性能测试 ({num_runs} 次迭代)...")
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(sample_input)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 计算统计数据
    timings = np.array(timings)
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    
    print(f"\n{'='*50}")
    print(f"推理性能结果 ({device}):")
    print(f"{'='*50}")
    print(f"平均推理时间: {mean_time:.3f} ms")
    print(f"标准差: {std_time:.3f} ms")
    print(f"最小时间: {min_time:.3f} ms")
    print(f"最大时间: {max_time:.3f} ms")
    print(f"{'='*50}")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'all_timings': timings
    }

# 使用示例
if __name__ == "__main__":
    # 创建性能分析器
    profiler = CUDAProfiler(interval=0.5)
    
    try:
        # 开始监控
        profiler.start()
        
        # 模拟一些GPU操作
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # 创建一些随机张量并进行操作
            for i in range(10):
                size = 5000 * (i + 1)
                print(f"测试 {size}x{size} 矩阵乘法...")
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                c = torch.matmul(a, b)
                del a, b, c
                time.sleep(2)
        
        # 等待一段时间
        time.sleep(5)
    
    finally:
        # 停止监控
        profiler.stop()
