"""
训练细调模型 - 使用优化的超参数训练最终模型
"""
import torch
import torch.nn as nn
import numpy as np
import time
import os
import gc
import logging
from tqdm import tqdm
from optimize_hyperparams import FineTunedLSTM
from high_performance_train import (
    OptimizedLSTM, get_optimized_dataloader, 
    aggressive_training, evaluate_final_model, CustomLoss
)
from continue_training import get_training_data, load_checkpoint
from gpu_memory_config import optimize_gpu_memory_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('fine_tuned_train_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 确保目录存在
    os.makedirs('models/fine_tuned', exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 优化GPU设置
    optimize_gpu_memory_settings()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载优化后的配置
    config_path = 'models/fine_tuned/best_config.pth'
    if not os.path.exists(config_path):
        logger.error(f"找不到配置文件: {config_path}")
        logger.error("请先运行optimize_hyperparams.py进行超参数优化")
        return
    
    logger.info(f"加载优化配置: {config_path}")
    best_config_data = torch.load(config_path, map_location=device)
    best_config = best_config_data['config']
    base_model_config = best_config_data['base_model_config']
    
    # 加载原始模型
    model_path = 'models/best_model.pth'
    checkpoint_data = load_checkpoint(model_path, device)
    
    # 数据加载
    logger.info("加载训练数据...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = get_training_data(best_config)
    
    # 创建基础模型
    input_size = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
    base_model = OptimizedLSTM(
        input_size=input_size,
        hidden_size=base_model_config.get('hidden_size', 512),
        num_layers=base_model_config.get('num_layers', 3),
        dropout=checkpoint_data['config'].get('dropout', 0.2)
    ).to(device)
    
    # 加载预训练权重
    base_model.load_state_dict(checkpoint_data['model_state_dict'])
    logger.info("已加载预训练模型权重")
    
    # 创建细调模型
    model = FineTunedLSTM(
        base_model=base_model,
        dropout_fc=best_config['dropout_fc'],
        use_layer_norm=best_config['use_layer_norm'],
        activation=best_config['activation']
    ).to(device)
    
    # 准备数据加载器
    logger.info("准备数据加载器...")
    # 优化数据集
    from high_performance_train import optimize_dataset_for_high_memory
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = optimize_dataset_for_high_memory(
        X_train, X_test, y_train, y_test
    )
    
    # 清理原始NumPy数组
    del X_train, X_test, y_train, y_test
    gc.collect()
    
    # 配置数据加载器
    train_loader = get_optimized_dataloader(
        X_train_tensor, y_train_tensor, 
        batch_size=best_config['batch_size'], 
        is_train=True,
        num_workers=0 if X_train_tensor.is_cuda else 4,
        pin_memory=False if X_train_tensor.is_cuda else True
    )
    
    val_loader = get_optimized_dataloader(
        X_test_tensor, y_test_tensor, 
        batch_size=best_config['batch_size'] * 2, 
        is_train=False,
        num_workers=0 if X_test_tensor.is_cuda else 4,
        pin_memory=False if X_test_tensor.is_cuda else True
    )
    
    # 定义损失函数
    criterion = CustomLoss(alpha=best_config['loss_alpha'], scale=best_config['loss_scale'])
    
    # 定义优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_config['learning_rate'],
        weight_decay=best_config['weight_decay']
    )
    
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # 周期长度
        T_mult=2,  # 每个周期后长度翻倍
        eta_min=best_config['learning_rate'] / 100  # 最小学习率
    )
    
    # 训练配置
    train_config = {
        'num_epochs': best_config['num_epochs'],
        'early_stopping': best_config['early_stopping'],
        'patience': best_config['patience'],
        'batch_size': best_config['batch_size'],
        'gradient_accumulation': 1
    }
    
    # 对比原始模型性能
    logger.info("评估原始模型性能...")
    base_metrics, _ = evaluate_final_model(
        model=base_model,
        test_loader=val_loader,
        criterion=criterion,
        scaler_y=scaler_y,
        device=device
    )
    
    # 开始训练
    logger.info("开始训练细调模型...")
    training_results = aggressive_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        config=train_config,
        device=device,
        save_dir='models/fine_tuned/'
    )
    
    # 加载最佳模型
    best_model_path = 'models/fine_tuned/best_model.pth'
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"已加载最佳细调模型 (Epoch: {checkpoint['epoch']+1})")
    else:
        logger.warning("找不到最佳细调模型文件!")
    
    # 评估最终性能
    logger.info("评估细调模型性能...")
    fine_tuned_metrics, results_df = evaluate_final_model(
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
    logger.info("性能比较 (原始模型 vs 细调模型)")
    logger.info("="*50)
    
    for metric in ['MSE', 'RMSE', 'MAE', 'R^2', 'NSE']:
        if metric in base_metrics and metric in fine_tuned_metrics:
            improvement = fine_tuned_metrics[metric] - base_metrics[metric]
            improvement_pct = abs(improvement) / abs(base_metrics[metric]) * 100
            
            if metric in ['MSE', 'RMSE', 'MAE']:
                better = improvement < 0
                symbol = "↓" if better else "↑"
            else:  # R^2, NSE
                better = improvement > 0
                symbol = "↑" if better else "↓"
            
            logger.info(f"{metric}: {base_metrics[metric]:.6f} → {fine_tuned_metrics[metric]:.6f} "
                       f"({symbol}{abs(improvement):.6f}, {improvement_pct:.2f}%)")
    
    logger.info("="*50)
    
    # 保存训练报告
    with open('models/fine_tuned/training_report.txt', 'w') as f:
        f.write("# 降雨预测模型细调训练报告\n\n")
        f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n\n")
        
        f.write("## 模型配置\n\n")
        for k, v in best_config.items():
            f.write(f"- {k}: {v}\n")
        
        f.write("\n## 性能比较\n\n")
        f.write("| 指标 | 原始模型 | 细调模型 | 改进 |\n")
        f.write("|------|---------|----------|------|\n")
        
        for metric in ['MSE', 'RMSE', 'MAE', 'R^2', 'NSE']:
            if metric in base_metrics and metric in fine_tuned_metrics:
                improvement = fine_tuned_metrics[metric] - base_metrics[metric]
                improvement_pct = abs(improvement) / abs(base_metrics[metric]) * 100
                
                if metric in ['MSE', 'RMSE', 'MAE']:
                    better = improvement < 0
                    symbol = "↓" if better else "↑"
                else:  # R^2, NSE
                    better = improvement > 0
                    symbol = "↑" if better else "↓"
                
                f.write(f"| {metric} | {base_metrics[metric]:.6f} | {fine_tuned_metrics[metric]:.6f} | "
                       f"{symbol}{abs(improvement):.6f} ({improvement_pct:.2f}%) |\n")
    
    logger.info("细调训练完成! 详细报告已保存到 models/fine_tuned/training_report.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise
