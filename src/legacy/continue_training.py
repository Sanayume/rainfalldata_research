# 首先导入并立即应用编码修复
import encoding_fix
encoding_fix.setup_environment(safe_logging=True)

"""
继续训练脚本 - 加载已保存的模型并继续训练以获得更好的性能
"""
import os
import sys
import torch
import numpy as np
import time
import logging
import gc
from tqdm import tqdm
from high_performance_train import (
    OptimizedLSTM, get_optimized_dataloader, aggressive_training, 
    optimize_dataset_for_high_memory, visualize_training_results,
    CustomLoss, evaluate_final_model, get_system_info
)
from paper4 import preprocess_data, calculate_metrics
from scipy.io import loadmat
from gpu_memory_config import optimize_gpu_memory_settings, log_memory_usage
from safe_unicode import replace_special_chars

# 配置日志 - 使用更安全的编码设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('continued_training_log.txt', encoding='utf-8'),  # 指定UTF-8编码
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 记录启动信息
logger.info("继续训练脚本启动，已应用编码修复")

def load_checkpoint(model_path, device, strict=True):
    """加载检查点，返回模型和训练状态"""
    logger.info(f"加载检查点: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    try:
        # 添加weights_only参数以避免警告
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 提取训练配置
        config = checkpoint.get('config', {})
        epoch = checkpoint.get('epoch', 0)
        model_state = checkpoint.get('model_state_dict', None)
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        scheduler_state = checkpoint.get('scheduler_state_dict', None)
        scaler_X = checkpoint.get('scaler_X', None)
        scaler_y = checkpoint.get('scaler_y', None)
        train_loss = checkpoint.get('train_loss', float('inf'))
        val_loss = checkpoint.get('val_loss', float('inf'))
        metrics = checkpoint.get('metrics', {})
        
        # 使用replace_special_chars处理可能包含特殊字符的消息
        safe_val_loss = f"{val_loss:.4f}"
        logger.info(f"已加载检查点 (轮次: {epoch+1}, 验证损失: {safe_val_loss})")
        
        # 如果是单纯的权重文件，只返回模型状态
        if isinstance(checkpoint, dict) and len(checkpoint) == 1 and 'state_dict' in checkpoint:
            return {
                'model_state_dict': checkpoint['state_dict'],
                'config': {},
                'epoch': 0,
                'train_loss': float('inf'),
                'val_loss': float('inf'),
                'optimizer_state_dict': None,
                'scheduler_state_dict': None,
                'scaler_X': None,
                'scaler_y': None
            }
        
        return {
            'model_state_dict': model_state,
            'config': config,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'metrics': metrics
        }
    except Exception as e:
        logger.error(f"加载检查点时出错: {type(e).__name__}: {str(e)}")
        raise

def setup_optimizer_scheduler(model, config, optimizer_state=None):
    """设置优化器和学习率调度器"""
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 如果有保存的优化器状态，加载它
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        logger.info("已恢复优化器状态")
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get('T_0', 20),  # 周期长度
        T_mult=config.get('T_mult', 1),  # 周期长度乘数
        eta_min=config.get('min_lr', 1e-6)  # 最小学习率
    )
    
    return optimizer, scheduler

def get_training_data(config):
    """加载和预处理训练数据"""
    # 定义数据文件路径
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
    
    # 加载数据
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
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def main():
    # 确保模型目录存在
    os.makedirs('models/continued', exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 检查系统信息
    system_info = get_system_info()
    logger.info("系统信息:")
    logger.info(system_info)
    
    # 优化GPU内存设置
    optimize_gpu_memory_settings()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载最佳模型
    best_model_path = 'models/continued/best_model.pth'
    checkpoint_data = load_checkpoint(best_model_path, device)
    
    # 提取配置
    saved_config = checkpoint_data['config']
    
    # 更新配置为继续训练的设置
    config = {
        'window_size': saved_config.get('window_size', 15),
        'batch_size': 1024,  # 可以使用相同或更大的批次
        'hidden_size': saved_config.get('hidden_size', 512),
        'num_layers': saved_config.get('num_layers', 3),
        'dropout': saved_config.get('dropout', 0.2),
        'learning_rate': 0.0005,  # 降低学习率继续训练
        'weight_decay': 1e-5,
        'num_epochs': 100,  # 额外训练100轮
        'mixed_precision': True,
        'gradient_accumulation': 1,
        'early_stopping': True,
        'patience': 20,  # 增加耐心值，减少早停概率
        'num_workers': 0,  # 数据已在GPU上时设为0
        'pin_memory': False,  # 数据已在GPU上时设为False
        'use_amp': True,
        # 学习率周期参数
        'T_0': 20,  # 周期长度
        'T_mult': 2,  # 每个周期后长度翻倍
        'min_lr': 1e-6  # 最小学习率
    }
    
    # 决定是否重新加载数据或使用已保存的预处理器
    if checkpoint_data['scaler_X'] is not None and checkpoint_data['scaler_y'] is not None:
        logger.info("从检查点加载预处理器...")
        scaler_X = checkpoint_data['scaler_X']
        scaler_y = checkpoint_data['scaler_y']
        
        # 使用保存的预处理器重新加载和预处理数据
        X_train, X_test, y_train, y_test, _, _ = get_training_data(config)
    else:
        # 如果没有预处理器，从头开始数据处理
        logger.info("未找到预处理器，从头开始数据处理...")
        X_train, X_test, y_train, y_test, scaler_X, scaler_y = get_training_data(config)
    
    # 优化数据集
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = optimize_dataset_for_high_memory(
        X_train, X_test, y_train, y_test
    )
    
    # 清理原始NumPy数组
    del X_train, X_test, y_train, y_test
    gc.collect()
    
    # 创建数据加载器
    logger.info("构建数据加载器...")
    
    # 根据数据是否在GPU上调整批次大小和工作线程
    train_batch_size = config['batch_size']
    val_batch_size = config['batch_size'] * 2
    
    if X_train_tensor.is_cuda:
        logger.info("数据在GPU上，使用更大批次和无工作线程")
        train_batch_size = min(8192, len(X_train_tensor)) 
        val_batch_size = min(16384, len(X_test_tensor))
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
    
    # 创建模型实例
    logger.info("创建模型实例...")
    input_size = X_train_tensor.shape[1] if len(X_train_tensor.shape) == 2 else X_train_tensor.shape[2]
    model = OptimizedLSTM(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint_data['model_state_dict'])
    logger.info("已加载模型权重")
    
    # 设置优化器和调度器
    optimizer, scheduler = setup_optimizer_scheduler(
        model, 
        config, 
        checkpoint_data['optimizer_state_dict']
    )
    
    # 定义损失函数
    criterion = CustomLoss(alpha=0.7, scale=100.0)
    
    # 输出初始模型性能
    logger.info("评估初始模型性能...")
    initial_metrics, _ = evaluate_final_model(
        model=model,
        test_loader=val_loader,
        criterion=criterion,
        scaler_y=scaler_y,
        device=device
    )
    
    # 开始继续训练
    logger.info("开始继续训练...")
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
        device=device,
        save_dir='models/continued/',  # 保存在继续训练子目录
        initial_epoch=checkpoint_data['epoch'] + 1  # 从之前的轮次继续
    )
    
    # 加载最佳模型
    logger.info("加载最佳继续训练模型...")
    best_model_path = 'models/continued/best_model.pth'
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"已加载最佳继续训练模型 (Epoch: {checkpoint['epoch']+1})")
    else:
        logger.warning("未找到最佳继续训练模型文件!")
    
    # 最终评估
    logger.info("最终评估模型性能...")
    final_metrics, results_df = evaluate_final_model(
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
    
    # 输出性能比较
    logger.info("\n" + "="*50)
    logger.info("性能比较 (原始模型 vs 继续训练后)")
    logger.info("="*50)
    
    for metric in ['MSE', 'RMSE', 'MAE', 'R^2', 'NSE']:
        if metric in initial_metrics and metric in final_metrics:
            improvement = final_metrics[metric] - initial_metrics[metric]
            improvement_pct = abs(improvement) / abs(initial_metrics[metric]) * 100
            
            if metric in ['MSE', 'RMSE', 'MAE']:
                better = improvement < 0
                symbol = "↓" if better else "↑"
            else:  # R^2, NSE
                better = improvement > 0
                symbol = "↑" if better else "↓"
            
            logger.info(f"{metric}: {initial_metrics[metric]:.6f} → {final_metrics[metric]:.6f} "
                       f"({symbol}{abs(improvement):.6f}, {improvement_pct:.2f}%)")
    
    logger.info("="*50)
    
    # 导出继续训练报告
    with open('models/continued/training_report.txt', 'w') as f:
        f.write("# 降雨预测模型继续训练报告\n\n")
        f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n\n")
        f.write("## 系统信息\n\n")
        f.write(system_info.replace('\n', '\n- ') + "\n\n")
        f.write("## 模型配置\n\n")
        for k, v in config.items():
            f.write(f"- {k}: {v}\n")
        
        f.write("\n## 性能比较\n\n")
        f.write("| 指标 | 原始模型 | 继续训练后 | 改进 |\n")
        f.write("|------|---------|------------|------|\n")
        
        for metric in ['MSE', 'RMSE', 'MAE', 'R^2', 'NSE']:
            if metric in initial_metrics and metric in final_metrics:
                improvement = final_metrics[metric] - initial_metrics[metric]
                improvement_pct = abs(improvement) / abs(initial_metrics[metric]) * 100
                
                if metric in ['MSE', 'RMSE', 'MAE']:
                    better = improvement < 0
                    symbol = "↓" if better else "↑"
                else:  # R^2, NSE
                    better = improvement > 0
                    symbol = "↑" if better else "↓"
                
                f.write(f"| {metric} | {initial_metrics[metric]:.6f} | {final_metrics[metric]:.6f} | "
                       f"{symbol}{abs(improvement):.6f} ({improvement_pct:.2f}%) |\n")
    
    logger.info("继续训练完成! 详细报告已保存到 models/continued/training_report.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"继续训练过程中出现错误: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise
