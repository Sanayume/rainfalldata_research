"""
超参数再优化脚本 - 基于现有最佳模型进一步优化超参数
"""
import torch
import torch.nn as nn
import numpy as np
import time
import os
import gc
from functools import partial
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import logging
from high_performance_train import (
    OptimizedLSTM, get_optimized_dataloader, 
    optimize_dataset_for_high_memory, CustomLoss
)
from continue_training import get_training_data, load_checkpoint
from paper4 import calculate_metrics
from gpu_memory_config import optimize_gpu_memory_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('fine_tuning_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FineTunedLSTM(nn.Module):
    """用于细调的LSTM模型，保持基本结构但添加一些可调参数"""
    def __init__(self, base_model, dropout_fc=0.2, use_layer_norm=True, activation='silu'):
        super(FineTunedLSTM, self).__init__()
        # 继承基本模型的LSTM层
        self.input_size = base_model.input_size
        self.hidden_size = base_model.hidden_size
        self.num_layers = base_model.num_layers
        self.lstm = base_model.lstm
        
        # 创建自定义的全连接层
        activation_fn = nn.SiLU() if activation == 'silu' else nn.GELU() if activation == 'gelu' else nn.ReLU()
        
        # 使用可选的层规范化
        if use_layer_norm:
            self.fc_layers = nn.Sequential(
                nn.Linear(base_model.hidden_size*2, base_model.hidden_size*2),
                nn.LayerNorm(base_model.hidden_size*2),
                activation_fn,
                nn.Dropout(dropout_fc),
                nn.Linear(base_model.hidden_size*2, base_model.hidden_size),
                nn.LayerNorm(base_model.hidden_size),
                activation_fn,
                nn.Dropout(dropout_fc/2),
                nn.Linear(base_model.hidden_size, 1)
            )
        else:
            self.fc_layers = nn.Sequential(
                nn.Linear(base_model.hidden_size*2, base_model.hidden_size*2),
                activation_fn,
                nn.Dropout(dropout_fc),
                nn.Linear(base_model.hidden_size*2, base_model.hidden_size),
                activation_fn,
                nn.Dropout(dropout_fc/2),
                nn.Linear(base_model.hidden_size, 1)
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

def objective(trial, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, 
              base_model, scaler_y, device):
    """Optuna优化目标函数"""
    # 确保GPU缓存清空
    torch.cuda.empty_cache()
    gc.collect()
    
    # 超参数搜索空间
    config = {
        'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096]),
        'dropout_fc': trial.suggest_float('dropout_fc', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),
        'activation': trial.suggest_categorical('activation', ['silu', 'gelu', 'relu']),
        'loss_alpha': trial.suggest_float('loss_alpha', 0.5, 0.9),
        'loss_scale': trial.suggest_categorical('loss_scale', [10.0, 50.0, 100.0, 200.0]),
    }
    
    # 记录试验配置
    logger.info(f"Trial {trial.number} - 配置: {config}")
    
    # 创建细调模型
    fine_tuned_model = FineTunedLSTM(
        base_model=base_model,
        dropout_fc=config['dropout_fc'],
        use_layer_norm=config['use_layer_norm'],
        activation=config['activation']
    ).to(device)
    
    # 定制损失函数
    criterion = CustomLoss(alpha=config['loss_alpha'], scale=config['loss_scale'])
    
    # 优化器
    optimizer = torch.optim.AdamW(
        fine_tuned_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # 在优化过程中使用0以减少开销
        pin_memory=False  # 数据已在GPU上
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 快速训练 (仅5个epoch)
    fine_tuned_model.train()
    max_epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        # 训练
        fine_tuned_model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = fine_tuned_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fine_tuned_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 清理内存
            del batch_X, batch_y, outputs, loss
        
        # 验证
        fine_tuned_model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = fine_tuned_model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
                del batch_X, batch_y, outputs
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # 提前终止性能不佳的试验
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    # 性能评估
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 反归一化
    preds_orig = scaler_y.inverse_transform(all_preds)
    targets_orig = scaler_y.inverse_transform(all_targets)
    
    # 计算性能指标
    metrics = calculate_metrics(targets_orig, preds_orig)
    rmse = metrics['RMSE']
    r2 = metrics['R^2']
    mae = metrics['MAE']
    
    # 清理内存
    del fine_tuned_model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # 记录结果
    logger.info(f"Trial {trial.number} - RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
    
    # 返回多目标优化评分: 主要是RMSE，但也考虑R2和MAE
    return rmse - 0.1 * r2 + 0.05 * mae

def fine_tune_model():
    """执行模型细调"""
    # 确保输出目录存在
    os.makedirs('models/fine_tuned', exist_ok=True)
    
    # 优化GPU设置
    optimize_gpu_memory_settings()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载最佳模型
    model_path = 'models/best_model.pth'
    checkpoint_data = load_checkpoint(model_path, device)
    
    # 基础配置
    config = {
        'window_size': 15,
        'batch_size': 2048
    }
    
    # 数据加载
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = get_training_data(config)
    
    # 创建基础模型
    input_size = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
    base_model = OptimizedLSTM(
        input_size=input_size,
        hidden_size=checkpoint_data['config'].get('hidden_size', 512),
        num_layers=checkpoint_data['config'].get('num_layers', 3),
        dropout=checkpoint_data['config'].get('dropout', 0.2)
    ).to(device)
    
    # 加载预训练权重
    base_model.load_state_dict(checkpoint_data['model_state_dict'])
    logger.info("已加载预训练模型权重")
    
    # 冻结LSTM层
    for param in base_model.lstm.parameters():
        param.requires_grad = False
    
    logger.info("已冻结LSTM层，只优化全连接层")
    
    # 优化数据集
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = optimize_dataset_for_high_memory(
        X_train, X_test, y_train, y_test
    )
    
    # 清理原始NumPy数组
    del X_train, X_test, y_train, y_test
    gc.collect()
    
    # 创建优化研究
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 设置优化目标
    objective_with_data = partial(
        objective,
        X_train_tensor=X_train_tensor,
        X_test_tensor=X_test_tensor,
        y_train_tensor=y_train_tensor,
        y_test_tensor=y_test_tensor,
        base_model=base_model,
        scaler_y=scaler_y,
        device=device
    )
    
    # 执行优化
    n_trials = 20
    logger.info(f"开始超参数优化，计划执行 {n_trials} 次试验...")
    
    try:
        study.optimize(objective_with_data, n_trials=n_trials)
    except KeyboardInterrupt:
        logger.info("优化被用户中断")
    
    # 获取最佳参数
    best_params = study.best_params
    logger.info(f"最佳参数: {best_params}")
    
    # 使用最佳参数创建和训练最终模型
    best_model = FineTunedLSTM(
        base_model=base_model,
        dropout_fc=best_params['dropout_fc'],
        use_layer_norm=best_params['use_layer_norm'],
        activation=best_params['activation']
    ).to(device)
    
    # 创建最佳配置
    best_config = {
        'window_size': 15,
        'batch_size': best_params['batch_size'],
        'dropout_fc': best_params['dropout_fc'],
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'use_layer_norm': best_params['use_layer_norm'],
        'activation': best_params['activation'],
        'loss_alpha': best_params['loss_alpha'],
        'loss_scale': best_params['loss_scale'],
        'num_epochs': 100,  # 使用更长的训练轮次
        'early_stopping': True,
        'patience': 15,
    }
    
    # 保存最佳模型配置
    torch.save({
        'config': best_config,
        'base_model_config': {
            'hidden_size': base_model.hidden_size,
            'num_layers': base_model.num_layers,
        },
        'study_best_value': study.best_value,
    }, 'models/fine_tuned/best_config.pth')
    
    # 输出优化结果可视化
    try:
        # 绘制优化历史
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig('models/fine_tuned/optimization_history.png')
        plt.close()
        
        # 绘制参数重要性
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig('models/fine_tuned/param_importance.png')
        plt.close()
    except Exception as e:
        logger.warning(f"生成可视化出错: {e}")
    
    # 准备训练最终模型
    logger.info("\n" + "="*50)
    logger.info("最佳超参数配置")
    logger.info("="*50)
    for key, value in best_config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*50)
    
    logger.info("\n要训练最终细调模型，请运行 train_fine_tuned.py 脚本")
    
    # 保存优化研究
    with open('models/fine_tuned/study.pkl', 'wb') as f:
        import pickle
        pickle.dump(study, f)
    
    return study, best_config

if __name__ == "__main__":
    try:
        study, best_config = fine_tune_model()
    except Exception as e:
        logger.error(f"细调过程中出现错误: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise
