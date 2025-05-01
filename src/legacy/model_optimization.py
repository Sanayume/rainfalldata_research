"""
模型优化脚本 - 用于优化模型结构和超参数
"""
import torch
import torch.nn as nn
import numpy as np
import time
import os
import gc
from functools import partial
from high_performance_train import OptimizedLSTM, get_optimized_dataloader, optimize_dataset_for_high_memory
from lstm_train import CustomLoss
from paper4 import preprocess_data, calculate_metrics
from cuda_profiler import CUDAProfiler, profile_model_inference
from scipy.io import loadmat
import logging
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('optimization_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedModelArchitecture(nn.Module):
    """增强型模型架构，包含LSTM、Transformer和CNN组件"""
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.2, 
                 use_transformer=True, use_cnn=True, use_gru=False):
        super(EnhancedModelArchitecture, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_transformer = use_transformer
        self.use_cnn = use_cnn
        self.use_gru = use_gru
        
        # LSTM层
        if not use_gru:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
        else:
            # GRU可能比LSTM更快
            self.lstm = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
        
        # Transformer自注意力层
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size*2,  # *2因为是双向LSTM
                nhead=8,
                dim_feedforward=hidden_size*4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=1
            )
        
        # CNN层用于特征提取
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(hidden_size*2, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Conv1d(hidden_size, hidden_size//2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size//2),
                nn.GELU(),
            )
        
        # 根据使用的组件计算最终特征大小
        final_size = hidden_size*2  # 双向LSTM的输出
        if use_cnn:
            final_size = hidden_size//2
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(final_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_size//2, 1)
        )
    
    def forward(self, x):
        # 处理2D和3D输入
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # Transformer层
        if self.use_transformer:
            # 应用Transformer编码器
            transformer_out = self.transformer_encoder(lstm_out)
            features = transformer_out[:, -1, :]  # 取最后一个时间步
        else:
            features = lstm_out[:, -1, :]  # 最后一个时间步的LSTM输出
        
        # CNN层
        if self.use_cnn:
            # 重塑以适应CNN (batch, channels, sequence_length)
            if len(lstm_out.shape) == 3 and lstm_out.shape[1] > 1:
                cnn_in = lstm_out.transpose(1, 2)
                cnn_out = self.cnn(cnn_in)
                features = cnn_out.mean(dim=2)  # 全局平均池化
            else:
                # 如果序列长度为1，跳过CNN
                pass
        
        # 全连接层
        output = self.fc_layers(features)
        return output

def objective(trial, X_train, X_test, y_train, y_test, scaler_y, device):
    """Optuna优化目标函数"""
    # 确保GPU缓存清空
    torch.cuda.empty_cache()
    gc.collect()
    
    # 超参数搜索空间
    config = {
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        'use_transformer': trial.suggest_categorical('use_transformer', [True, False]),
        'use_cnn': trial.suggest_categorical('use_cnn', [True, False]),
        'use_gru': trial.suggest_categorical('use_gru', [True, False]),
        'gradient_accumulation': 1,
        'num_workers': 4,
        'pin_memory': True,
    }
    
    # 记录试验配置
    logger.info(f"Trial {trial.number} - 配置: {config}")
    
    input_size = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
    
    # 创建模型
    model = EnhancedModelArchitecture(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_transformer=config['use_transformer'],
        use_cnn=config['use_cnn'],
        use_gru=config['use_gru']
    ).to(device)
    
    # 损失函数
    criterion = CustomLoss(alpha=0.7, scale=100.0)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 数据加载器
    if not isinstance(X_train, torch.Tensor):
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
    else:
        # 已经是张量，只需移到正确的设备
        X_train_tensor = X_train.to(device)
        y_train_tensor = y_train.to(device)
        X_test_tensor = X_test.to(device)
        y_test_tensor = y_test.to(device)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # 在优化过程中使用0以减少开销
        pin_memory=False  # 数据已在GPU上
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 快速训练 (仅5个epoch)
    model.train()
    max_epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # 清理内存
            del batch_X, batch_y, outputs, loss
        
        # 验证
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
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
    
    # 性能测试
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 反归一化
    preds_orig = scaler_y.inverse_transform(all_preds)
    targets_orig = scaler_y.inverse_transform(all_targets)
    
    # 计算性能指标
    metrics = calculate_metrics(targets_orig, preds_orig)
    rmse = metrics['RMSE']
    r2 = metrics['R^2']
    
    # 内存使用评估
    cuda_mem_used = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
    
    # 模型大小评估
    model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)  # MB (假设每个参数4字节)
    
    # 推理速度评估
    inference_time = profile_model_inference(
        model, 
        torch.randn(100, input_size, device=device),
        num_runs=50,
        warmup=10,
        device=device
    )['mean']
    
    # 清理内存
    del model, optimizer, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    # 记录结果
    logger.info(f"Trial {trial.number} - RMSE: {rmse:.4f}, R2: {r2:.4f}, "
                f"GPU内存: {cuda_mem_used:.2f} GB, 模型大小: {model_size:.2f} MB, "
                f"推理时间: {inference_time:.2f} ms")
    
    # 返回损失值作为优化目标 (同时考虑精度和速度)
    # 损失值 = RMSE - 0.5*R2 + 0.01*inference_time
    optimization_score = rmse - 0.5*r2 + 0.01*inference_time
    return optimization_score

def optimize_model(n_trials=50):
    """运行模型优化"""
    # 确保目录存在
    os.makedirs('models/optimization', exist_ok=True)
    
    # 加载预处理后的小型数据样本进行超参数优化
    logger.info("加载数据用于超参数优化...")
    
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
    reference_data = loadmat(DATAFILE["CHM"])['data'].astype(np.float32)
    
    satellite_data_list = []
    for key in ["CHIRPS", "CMORPH", "GSMAP", "IMERG", "PERSIANN"]:
        data = loadmat(DATAFILE[key])['data'].astype(np.float32)
        satellite_data_list.append(data)
    
    mask = loadmat(DATAFILE["MASK"])['mask']
    
    # 预处理，但使用较小的窗口大小和样本量
    window_size = 15
    
    # 数据预处理
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, valid_points, _ = preprocess_data(
        reference_data, satellite_data_list, mask, 
        window_size=window_size, 
        add_features=True,
        dtype=np.float32
    )
    
    # 为优化使用子集
    sample_size = min(50000, X_train.shape[0])
    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train[indices]
    
    # 清理原始数据
    del reference_data, satellite_data_list, X_train, y_train
    gc.collect()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建Optuna研究
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler()
    )
    
    # 开始优化
    logger.info(f"开始超参数优化，计划执行 {n_trials} 次试验...")
    
    objective_with_data = partial(
        objective,
        X_train=X_train_sample,
        X_test=X_test,
        y_train=y_train_sample,
        y_test=y_test,
        scaler_y=scaler_y,
        device=device
    )
    
    # 启动性能分析器
    profiler = CUDAProfiler(interval=1.0, log_file="models/optimization/gpu_profile.csv")
    profiler.start()
    
    try:
        study.optimize(objective_with_data, n_trials=n_trials, timeout=3600*8)  # 最多8小时
    except KeyboardInterrupt:
        logger.info("优化被用户中断")
    finally:
        profiler.stop()
    
    # 获取最佳试验
    best_trial = study.best_trial
    
    # 打印和保存结果
    logger.info("优化完成!")
    logger.info(f"最佳试验: {best_trial.number}")
    logger.info(f"最佳值: {best_trial.value}")
    logger.info("最佳超参数:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # 保存优化历史
    df = study.trials_dataframe()
    df.to_csv("models/optimization/optimization_history.csv")
    
    # 可视化优化过程
    plt.figure(figsize=(12, 8))
    
    # 绘制优化历史
    plt.subplot(1, 2, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    
    # 绘制参数重要性
    plt.subplot(1, 2, 2)
    try:
        optuna.visualization.matplotlib.plot_param_importances(study)
    except:
        logger.warning("无法绘制参数重要性图表")
    
    plt.tight_layout()
    plt.savefig("models/optimization/optimization_visualization.png", dpi=200)
    
    # 使用最佳参数创建最终模型
    logger.info("使用最佳参数创建最终模型...")
    
    input_size = X_train_sample.shape[1] if len(X_train_sample.shape) == 2 else X_train_sample.shape[2]
    
    best_model = EnhancedModelArchitecture(
        input_size=input_size,
        hidden_size=best_trial.params['hidden_size'],
        num_layers=best_trial.params['num_layers'],
        dropout=best_trial.params['dropout'],
        use_transformer=best_trial.params['use_transformer'],
        use_cnn=best_trial.params['use_cnn'],
        use_gru=best_trial.params['use_gru']
    )
    
    # 保存最佳模型架构配置
    torch.save({
        'model_config': {
            'input_size': input_size,
            'hidden_size': best_trial.params['hidden_size'],
            'num_layers': best_trial.params['num_layers'],
            'dropout': best_trial.params['dropout'],
            'use_transformer': best_trial.params['use_transformer'],
            'use_cnn': best_trial.params['use_cnn'],
            'use_gru': best_trial.params['use_gru']
        },
        'optimizer_config': {
            'learning_rate': best_trial.params['learning_rate'],
            'weight_decay': best_trial.params['weight_decay'],
            'batch_size': best_trial.params['batch_size']
        },
        'state_dict': best_model.state_dict()
    }, "models/optimization/best_model_architecture.pth")
    
    return study, best_trial.params

if __name__ == "__main__":
    try:
        study, best_params = optimize_model(n_trials=30)
        
        # 输出最佳超参数配置，可以复制到高性能训练脚本中
        print("\n" + "="*50)
        print("最佳超参数配置 (可复制到训练脚本)")
        print("="*50)
        print("config = {")
        for key, value in best_params.items():
            print(f"    '{key}': {value},")
        print("    # 其他配置保持不变")
        print("}")
        print("="*50)
    except Exception as e:
        logger.error(f"优化过程中出现错误: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise
