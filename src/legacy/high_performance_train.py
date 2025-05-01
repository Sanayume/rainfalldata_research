import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import gc
from scipy.io import loadmat
from paper4 import preprocess_data, calculate_metrics
from lstm_train import RainfallLSTM, CustomLoss
import psutil
from functools import partial
from tqdm import tqdm
import logging
import warnings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 忽略不重要的警告
warnings.filterwarnings('ignore', category=UserWarning)

class OptimizedRainfallDataset(TensorDataset):
    """优化版降水数据集，使用TensorDataset提高性能"""
    def __init__(self, X, y):
        # 检查输入类型，处理不同情况
        if isinstance(X, torch.Tensor):
            self.X = X
        else:
            self.X = torch.FloatTensor(X)
            
        if isinstance(y, torch.Tensor):
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
            
        super().__init__(self.X, self.y)

class TorchMemoryMonitor:
    """CUDA内存监控器"""
    def __init__(self):
        self.peak_memory = 0
    
    def update(self):
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / 1024**3
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
    
    def get_peak(self):
        return self.peak_memory

def get_system_info():
    """获取系统信息"""
    cpu_info = f"CPU核心数: {psutil.cpu_count(logical=False)} (物理), {psutil.cpu_count(logical=True)} (逻辑)"
    ram_info = f"系统内存: {psutil.virtual_memory().total / (1024**3):.2f} GB"
    
    gpu_info = "GPU信息: "
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info += f"\n- GPU {i}: {torch.cuda.get_device_name(i)}"
            gpu_info += f", 显存: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB"
    else:
        gpu_info += "无可用GPU"
    
    return f"{cpu_info}\n{ram_info}\n{gpu_info}"

def get_optimized_dataloader(X, y, batch_size, is_train=True, num_workers=8, pin_memory=True, persistent_workers=True):
    """创建优化的数据加载器"""
    dataset = OptimizedRainfallDataset(X, y)
    
    # 如果数据已经在GPU上，我们需要较少的工作线程
    if isinstance(X, torch.Tensor) and X.is_cuda:
        num_workers = 0  # GPU数据不需要额外的工作线程
        pin_memory = False  # 数据已在GPU上，不需要pin memory
        persistent_workers = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=3 if num_workers > 0 else None,
        # 添加以下参数以提高性能
        drop_last=is_train,  # 训练时丢弃不完整的批次
        generator=torch.Generator().manual_seed(42) if is_train else None,  # 确保可重复性
        multiprocessing_context='spawn' if num_workers > 0 else None  # 更可靠的多进程方式
    )

class OptimizedLSTM(nn.Module):
    """针对RTX 4090优化的LSTM模型"""
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.2):
        super(OptimizedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 使用高性能LSTM实现
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
            proj_size=0  # 投影大小设为0可以略微提高性能
        )
        
        # 使用融合操作的全连接层序列
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*2),  # 增加中间层宽度
            nn.LayerNorm(hidden_size*2),
            nn.SiLU(),  # SiLU (Swish) 通常比ReLU快且效果更好
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # 处理2D和3D输入情况
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # LSTM前向传播
        output, _ = self.lstm(x)
        
        # 只使用最后一个时间步
        last_output = output[:, -1, :]
        
        # 全连接层
        return self.fc_layers(last_output)

def aggressive_training(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    scaler_X, scaler_y, config, device, save_dir='models/', initial_epoch=0
):
    """激进优化的训练函数"""
    # 记录开始时间
    start_time = time.time()
    
    # 初始化混合精度训练
    scaler = GradScaler()
    
    # 内存监控
    memory_monitor = TorchMemoryMonitor()
    
    # 训练记录
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    metrics_history = []
    epoch_times = []
    
    # 创建模型保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置进度条
    epochs_pbar = tqdm(range(config['num_epochs']), desc="训练进度")
    
    # 早停设置
    patience_counter = 0
    
    for epoch in epochs_pbar:
        # 使用实际轮次计算（包括之前的训练）
        actual_epoch = initial_epoch + epoch
        
        # ----- 训练阶段 -----
        model.train()
        epoch_start = time.time()
        total_train_loss = 0
        batch_count = 0
        
        # 使用GPU预热
        if epoch == 0:
            logger.info("GPU预热中...")
            # 更多的预热迭代和更积极的内存预分配
            # 使用模型的输入大小进行预热
            temp_input_size = model.input_size
            for _ in range(20):
                # 创建随机张量并进行一些计算以预热GPU
                dummy_input = torch.randn(config['batch_size'], 
                                         temp_input_size, 
                                         device=device, 
                                         requires_grad=False)
                dummy_output = model(dummy_input)
                loss = dummy_output.mean()
                del dummy_input, dummy_output, loss
                torch.cuda.synchronize()
            
            torch.cuda.empty_cache()
        
        # 创建训练批次的进度条
        train_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for i, (batch_X, batch_y) in enumerate(train_pbar):
            # 转移数据到设备
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # 混合精度训练
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(batch_X)
                batch_loss = criterion(outputs, batch_y) / config['gradient_accumulation']
            
            # 反向传播
            scaler.scale(batch_loss).backward()
            
            # 每n个批次更新一次参数（梯度累积）
            if (i + 1) % config['gradient_accumulation'] == 0 or (i + 1) == len(train_loader):
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 优化器步进
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # 更快地清零梯度
            
            # 累计损失
            total_train_loss += batch_loss.item() * config['gradient_accumulation']
            batch_count += 1
            
            # 更新进度条
            train_pbar.set_postfix({'loss': batch_loss.item()})
            
            # 更新内存监控
            memory_monitor.update()
            
            # 避免内存泄漏
            del batch_X, batch_y, outputs, batch_loss
            
            # 每50批次清理一次缓存，防止长时间积累
            if i % 50 == 0:
                torch.cuda.empty_cache()
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # ----- 验证阶段 -----
        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            # 验证进度条
            val_pbar = tqdm(val_loader, leave=False, desc="验证")
            
            for batch_X, batch_y in val_pbar:
                # 转移数据到设备
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                # 混合精度推理
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(batch_X)
                    val_loss = criterion(outputs, batch_y)
                
                # 累计验证损失
                total_val_loss += val_loss.item()
                val_batch_count += 1
                
                # 收集预测和目标值（使用CPU以节省GPU内存）
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
                # 更新验证进度条
                val_pbar.set_postfix({'val_loss': val_loss.item()})
                
                # 避免内存泄漏
                del batch_X, batch_y, outputs, val_loss
        
        # 计算平均验证损失
        avg_val_loss = total_val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        # 计算处理时间
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # 计算评估指标
        all_preds = np.vstack(all_preds).astype(np.float32)
        all_targets = np.vstack(all_targets).astype(np.float32)
        
        # 反归一化获取原始值
        preds_orig = scaler_y.inverse_transform(all_preds)
        targets_orig = scaler_y.inverse_transform(all_targets)
        
        # 计算性能指标
        metrics = calculate_metrics(targets_orig, preds_orig)
        metrics_history.append(metrics)
        
        # 学习率调整
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # 记录和输出结果
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新进度条描述
        epochs_pbar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'val_loss': f"{avg_val_loss:.4f}",
            'rmse': f"{metrics['RMSE']:.4f}",
            'r2': f"{metrics['R^2']:.4f}",
            'time': f"{epoch_time:.2f}s",
            'lr': f"{current_lr:.6f}"
        })
        
        # 日志记录
        logger.info(
            f"Epoch {actual_epoch+1}/{initial_epoch+config['num_epochs']} - "
            f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, "
            f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R2: {metrics['R^2']:.4f}, "
            f"时间: {epoch_time:.2f}s, 学习率: {current_lr:.6f}"
        )
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # 保存完整模型用于部署
            torch.save({
                'epoch': actual_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': metrics,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'config': config
            }, f'{save_dir}best_model.pth')
            
            # 也保存仅权重版本，更小更快
            torch.save(model.state_dict(), f'{save_dir}best_weights.pth')
            
            logger.info(f"* 第{actual_epoch+1}轮: 保存了新的最佳模型 (损失: {avg_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if config['early_stopping'] and patience_counter >= config['patience']:
            logger.info(f"早停：连续{config['patience']}轮未改善")
            break
        
        # 清理内存
        del all_preds, all_targets, preds_orig, targets_orig
        gc.collect()
        torch.cuda.empty_cache()
    
    # 计算总训练时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 打印训练完成信息
    logger.info("=" * 50)
    logger.info(f"训练完成！总时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")
    logger.info(f"GPU峰值内存使用: {memory_monitor.get_peak():.2f} GB")
    
    # 可视化训练结果
    visualize_training_results(train_losses, val_losses, metrics_history, epoch_times)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics_history,
        'epoch_times': epoch_times,
        'best_val_loss': best_val_loss,
        'total_time': total_time
    }

def visualize_training_results(train_losses, val_losses, metrics_history, epoch_times):
    """可视化训练结果"""
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 损失曲线
    ax = axes[0, 0]
    ax.plot(train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.7)
    ax.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. RMSE曲线
    ax = axes[0, 1]
    rmse_values = [m['RMSE'] for m in metrics_history]
    ax.plot(rmse_values, 'g-', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Squared Error')  # 修复set_title -> set_title
    ax.grid(True, alpha=0.3)
    
    # 3. R² 曲线
    ax = axes[1, 0]
    r2_values = [m['R^2'] for m in metrics_history]
    ax.plot(r2_values, 'c-', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('R2')  # 修改为R2
    ax.set_title('R-squared')
    ax.grid(True, alpha=0.3)
    
    # 4. 每轮训练时间
    ax = axes[1, 1]
    ax.plot(epoch_times, 'm-', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Seconds')
    ax.set_title('Time per Epoch')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_visualization.png', dpi=200)
    plt.close()
    
    # 保存训练数据
    training_data = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'rmse': [m['RMSE'] for m in metrics_history],
        'mae': [m['MAE'] for m in metrics_history],
        'r2': [m['R^2'] for m in metrics_history],
        'nse': [m['NSE'] for m in metrics_history],
        'time_per_epoch': epoch_times
    }
    pd.DataFrame(training_data).to_csv('models/training_history.csv', index=False)

def evaluate_final_model(model, test_loader, criterion, scaler_y, device):
    """评估最终模型性能"""
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    
    logger.info("开始最终评估...")
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc="评估中"):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # 混合精度推理
            if torch.cuda.is_available():
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(batch_X)
                    test_loss += criterion(outputs, batch_y).item()
            else:
                outputs = model(batch_X)
                test_loss += criterion(outputs, batch_y).item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
            del batch_X, batch_y, outputs
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    # 计算平均测试损失
    avg_test_loss = test_loss / len(test_loader)
    
    # 处理预测结果
    all_preds = np.vstack(all_preds).astype(np.float32)
    all_targets = np.vstack(all_targets).astype(np.float32)
    
    # 反归一化
    preds_orig = scaler_y.inverse_transform(all_preds)
    targets_orig = scaler_y.inverse_transform(all_targets)
    
    # 计算指标
    metrics = calculate_metrics(targets_orig, preds_orig)
    
    # 可视化预测结果
    plt.figure(figsize=(12, 10))
    
    # 绘制散点图
    plt.scatter(targets_orig, preds_orig, alpha=0.5, s=5)
    
    # 添加完美预测线
    min_val = min(targets_orig.min(), preds_orig.min())
    max_val = max(targets_orig.max(), preds_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.xlabel('Actual Rainfall')
    plt.ylabel('Predicted Rainfall')
    plt.title('Actual vs Predicted Rainfall')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig('models/prediction_scatter.png', dpi=200)
    plt.close()
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'Actual': targets_orig.flatten(),
        'Predicted': preds_orig.flatten(),
        'Error': (targets_orig - preds_orig).flatten()
    })
    results_df.to_csv('models/prediction_results.csv', index=False)
    
    # 记录评估指标
    logger.info(f"测试损失: {avg_test_loss:.6f}")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.6f}")
    
    return metrics, results_df

def optimize_dataset_for_high_memory(X_train, X_test, y_train, y_test):
    """针对大内存服务器优化数据集"""
    # 对于大内存服务器，我们可以冗余存储一些数据，以优化访问速度
    # 确保数据为连续内存布局
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)
    y_train = np.ascontiguousarray(y_train)
    y_test = np.ascontiguousarray(y_test)
    
    # 提前将部分数据移至CUDA，如果内存允许
    if torch.cuda.is_available():
        # 计算可用GPU内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        current_allocated = torch.cuda.memory_allocated(0)
        current_reserved = torch.cuda.memory_reserved(0)
        available_memory = gpu_memory - current_allocated - current_reserved
        
        # 计算数据集大小
        data_size = (X_train.nbytes + y_train.nbytes + 
                     X_test.nbytes + y_test.nbytes)
        
        logger.info(f"总数据大小: {data_size / (1024**3):.2f} GB")
        logger.info(f"可用GPU内存: {available_memory / (1024**3):.2f} GB")
        
        # 判断是否可以放入GPU内存
        if data_size < 0.7 * available_memory:
            logger.info("将所有数据预加载到CUDA内存")
            X_train_tensor = torch.from_numpy(X_train).cuda(non_blocking=True)
            y_train_tensor = torch.from_numpy(y_train).cuda(non_blocking=True)
            X_test_tensor = torch.from_numpy(X_test).cuda(non_blocking=True)
            y_test_tensor = torch.from_numpy(y_test).cuda(non_blocking=True)
        elif (X_train.nbytes + y_train.nbytes) < 0.6 * available_memory:
            logger.info("将训练数据预加载到CUDA内存")
            X_train_tensor = torch.from_numpy(X_train).cuda(non_blocking=True)
            y_train_tensor = torch.from_numpy(y_train).cuda(non_blocking=True)
            X_test_tensor = torch.from_numpy(X_test)
            y_test_tensor = torch.from_numpy(y_test)
        else:
            logger.info("GPU内存有限，使用CPU数据")
            X_train_tensor = torch.from_numpy(X_train)
            y_train_tensor = torch.from_numpy(y_train)
            X_test_tensor = torch.from_numpy(X_test)
            y_test_tensor = torch.from_numpy(y_test)
    else:
        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train)
        X_test_tensor = torch.from_numpy(X_test)
        y_test_tensor = torch.from_numpy(y_test)
        
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def main():
    """主函数"""
    # 确保目录存在
    os.makedirs('models', exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 检查系统信息
    system_info = get_system_info()
    logger.info("系统信息:")
    logger.info(system_info)
    
    # 配置
    config = {
        'data_path': './',  # 当前目录
        'window_size': 15,
        'batch_size': 1024,  # 大幅增加批次大小
        'hidden_size': 512,  # 增加隐藏层大小
        'num_layers': 3,    # 增加LSTM层数
        'dropout': 0.2,
        'learning_rate': 0.002,  # 增加初始学习率
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'mixed_precision': True,
        'gradient_accumulation': 1,  # 减少梯度累积，因为批次已经很大
        'early_stopping': True,
        'patience': 15,
        'num_workers': 16,  # 增加工作线程数
        'pin_memory': True,
        'use_amp': True,
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 优化PyTorch性能
    if torch.cuda.is_available():
        logger.info("优化CUDA性能设置")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # 关闭确定性模式以获取更好性能
        
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("启用TF32矩阵乘法")
        
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            logger.info("启用cudnn TF32")
        
        try:
            # 更激进的内存使用
            torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的GPU内存
            logger.info("GPU内存使用限制设置为95%")
        except:
            logger.warning("无法设置GPU内存使用限制，将使用默认值")
    
    # 定义数据文件路径 (当前目录)
    DATAFILE = {
        "CMORPH": "./CMORPHdata/CMORPH_2016_2020.mat",
        "CHIRPS": "./CHIRPSdata/chirps_2016_2020.mat",
        "SM2RAIN": "./sm2raindata/sm2rain_2016_2020.mat",
        "IMERG": "./IMERGdata/IMERG_2016_2020.mat",
        "GSMAP": "./GSMAPdata/GSMAP_2016_2020.mat",
        "PERSIANN": "./PERSIANNdata/PERSIANN_2016_2020.mat",
        "CHM": "./CHMdata/CHM_2016_2020.mat",
        "MASK": "./mask.mat",
    }
    
    try:
        # 数据加载
        logger.info("加载参考数据...")
        reference_data = loadmat(DATAFILE["CHM"])['data'].astype(np.float32)
        
        logger.info("加载卫星数据...")
        satellite_data_list = []
        for key in ["CHIRPS", "CMORPH", "GSMAP", "IMERG", "PERSIANN"]:
            data = loadmat(DATAFILE[key])['data'].astype(np.float32)
            satellite_data_list.append(data)
            gc.collect()
        
        logger.info("加载掩码数据...")
        mask = loadmat(DATAFILE["MASK"])['mask']
        
        # 数据预处理
        logger.info("开始数据预处理...")
        preprocess_start = time.time()
        
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, valid_points, feature_importance = preprocess_data(
            reference_data, satellite_data_list, mask, 
            window_size=config['window_size'], 
            add_features=True,
            dtype=np.float32
        )
        
        # 释放原始数据
        del reference_data, satellite_data_list
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"数据预处理完成，耗时: {time.time() - preprocess_start:.2f}秒")
        logger.info(f"数据形状 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 针对大内存优化数据集
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = optimize_dataset_for_high_memory(
            X_train, X_test, y_train, y_test
        )
        
        # 清理NumPy数组
        del X_train, X_test, y_train, y_test
        gc.collect()
        
        # 创建数据加载器
        logger.info("构建数据加载器...")
        
        # 根据数据是否在GPU上调整批次大小
        train_batch_size = config['batch_size']
        val_batch_size = config['batch_size'] * 2
        
        # 如果数据在GPU上，可以使用更大的批次
        if X_train_tensor.is_cuda:
            logger.info("数据在GPU上，使用更大批次和无工作线程")
            train_batch_size = min(8192, len(X_train_tensor))  # 使用更大批次，但不超过数据集大小
            val_batch_size = min(16384, len(X_test_tensor))
            
            # 对于GPU上的数据，禁用工作线程和pin_memory
            workers = 0
            pin_memory = False
        else:
            workers = config['num_workers']
            pin_memory = config['pin_memory']
        
        train_loader = get_optimized_dataloader(
            X_train_tensor, y_train_tensor, 
            batch_size=train_batch_size, 
            is_train=True,
            num_workers=workers,
            pin_memory=pin_memory
        )
        
        val_loader = get_optimized_dataloader(
            X_test_tensor, y_test_tensor, 
            batch_size=val_batch_size,
            is_train=False,
            num_workers=workers,
            pin_memory=pin_memory
        )
        
        # 初始化模型
        logger.info("初始化模型...")
        input_size = X_train_tensor.shape[1] if len(X_train_tensor.shape) == 2 else X_train_tensor.shape[2]
        
        # 使用优化的模型
        model = OptimizedLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总量: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
        # 定义损失函数
        criterion = CustomLoss(alpha=0.7, scale=100.0)
        
        # 使用AdamW优化器，更好的权重衰减处理
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用One Cycle策略代替余弦退火，通常能更快收敛
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader) // config['gradient_accumulation'],
            pct_start=0.3,  # 30%的时间用于预热
            anneal_strategy='cos',
            div_factor=10.0,  # 初始学习率 = max_lr/10
            final_div_factor=100.0  # 最终学习率 = max_lr/1000
        )
        
        # 开始训练
        logger.info("开始模型训练...")
        training_results = aggressive_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            config=config,
            device=device
        )
        
        # 加载最佳模型
        logger.info("加载最佳模型...")
        best_model_path = 'models/best_model.pth'
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"已加载最佳模型 (Epoch: {checkpoint['epoch']+1})")
        else:
            logger.warning("未找到最佳模型文件!")
        
        # 最终评估
        metrics, results_df = evaluate_final_model(
            model=model,
            test_loader=val_loader,
            criterion=criterion,
            scaler_y=scaler_y,
            device=device
        )
        
        # 计算总运行时间
        total_run_time = time.time() - start_time
        hours, remainder = divmod(total_run_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        
        # 导出报告
        with open('models/training_report.txt', 'w') as f:
            f.write("# 降雨预测模型训练报告\n\n")
            f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n\n")
            f.write("## 系统信息\n\n")
            f.write(system_info.replace('\n', '\n- ') + "\n\n")
            f.write("## 模型配置\n\n")
            for k, v in config.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n## 模型参数\n\n")
            f.write(f"- 参数总量: {total_params:,}\n")
            f.write(f"- 可训练参数: {trainable_params:,}\n\n")
            f.write("## 最终评估指标\n\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"- {metric_name}: {metric_value:.6f}\n")
        
        logger.info("训练完成! 详细报告已保存到 models/training_report.txt")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise

if __name__ == "__main__":
    main()
