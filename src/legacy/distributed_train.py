"""
分布式训练脚本 - 用于在多GPU环境下高效训练降水预测模型
"""
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import time
import logging
from high_performance_train import (
    OptimizedLSTM, get_optimized_dataloader, aggressive_training, 
    optimize_dataset_for_high_memory, visualize_training_results
)
from lstm_train import CustomLoss
from paper4 import preprocess_data, calculate_metrics
from scipy.io import loadmat
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('distributed_training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def load_and_preprocess_data(config, rank=0):
    """加载和预处理数据"""
    if rank == 0:
        logger.info("加载参考数据...")
    
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
    
    # 加载参考数据
    reference_data = loadmat(DATAFILE["CHM"])['data'].astype(np.float32)
    
    if rank == 0:
        logger.info("加载卫星数据...")
    
    # 加载卫星数据列表
    satellite_data_list = []
    for key in ["CHIRPS", "CMORPH", "GSMAP", "IMERG", "PERSIANN"]:
        data = loadmat(DATAFILE[key])['data'].astype(np.float32)
        satellite_data_list.append(data)
        gc.collect()
    
    if rank == 0:
        logger.info("加载掩码数据...")
    
    # 加载掩码数据
    mask = loadmat(DATAFILE["MASK"])['mask']
    
    # 数据预处理
    if rank == 0:
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
    
    if rank == 0:
        logger.info(f"数据预处理完成，耗时: {time.time() - preprocess_start:.2f}秒")
        logger.info(f"数据形状 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def run_training(rank, world_size, config):
    """在单个GPU上运行训练"""
    setup(rank, world_size)
    
    # 确保每个进程使用不同的种子
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    
    if rank == 0:
        logger.info(f"世界大小: {world_size}, 当前进程: {rank}")
        logger.info(f"使用GPU: {torch.cuda.get_device_name(rank)}")
    
    # 加载和预处理数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_preprocess_data(config, rank)
    
    # 优化数据集
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = optimize_dataset_for_high_memory(
        X_train, X_test, y_train, y_test
    )
    
    # 释放NumPy数组
    del X_train, X_test, y_train, y_test
    gc.collect()
    
    # 创建分布式数据采样器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # 创建数据加载器
    # 对于GPU上的数据，禁用工作线程和pin_memory
    workers = 0 if X_train_tensor.is_cuda else config['num_workers']
    pin_mem = False if X_train_tensor.is_cuda else True
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=workers > 0
    )
    
    # 验证集不需要分布式采样器
    val_loader = get_optimized_dataloader(
        X_test_tensor, y_test_tensor,
        batch_size=config['batch_size'] * 2,
        is_train=False,
        num_workers=workers,
        pin_memory=pin_mem
    )
    
    # 初始化模型
    if rank == 0:
        logger.info("初始化模型...")
    
    input_size = X_train_tensor.shape[1] if len(X_train_tensor.shape) == 2 else X_train_tensor.shape[2]
    model = OptimizedLSTM(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(rank)  # 将模型移动到当前设备
    
    # 将模型包装为DDP模型
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数总量: {total_params:,}")
    
    # 定义损失函数
    criterion = CustomLoss(alpha=0.7, scale=100.0)
    
    # 定义优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy='cos'
    )
    
    # 开始训练
    if rank == 0:
        logger.info("开始分布式训练...")
    
    # 正常训练流程，但使用DDP模型
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
        device=rank  # 使用当前设备
    )
    
    # 只在主进程做额外的操作
    if rank == 0:
        visualize_training_results(
            training_results['train_losses'], 
            training_results['val_losses'], 
            training_results['metrics'], 
            training_results['epoch_times']
        )
        
        logger.info("分布式训练完成!")
    
    # 清理分布式环境
    cleanup()

def main():
    """主函数"""
    # 创建保存模型的目录
    os.makedirs('models', exist_ok=True)
    
    # 优化配置
    config = {
        'window_size': 15,
        'batch_size': 2048,  # 分布式训练使用更大批次
        'hidden_size': 768,  # 更大的隐藏层
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.002,
        'weight_decay': 1e-5,
        'num_epochs': 50,  # 分布式训练通常需要的轮数减少
        'gradient_accumulation': 1,
        'early_stopping': True,
        'patience': 10,
        'num_workers': 6,  # 每个GPU使用的工作线程数
        'pin_memory': True,
        'use_amp': True,
    }
    
    # 检测可用GPU数量
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        logger.error("没有可用的GPU，退出程序")
        sys.exit(1)
    
    logger.info(f"检测到 {world_size} 个可用GPU")
    
    # 使用多进程并行运行
    try:
        mp.spawn(
            run_training,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        logger.error(f"分布式训练出现错误: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
