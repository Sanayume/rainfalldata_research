import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import pandas as pd
import gc
import time
from paper4 import preprocess_data, calculate_metrics
from lstm_train import RainfallDataset, RainfallLSTM, CustomLoss
from scipy.io import loadmat

# 配置matplotlib支持中文显示
def configure_matplotlib_fonts():
    """配置matplotlib以支持中文显示"""
    try:
        # 尝试使用系统中文字体
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
        
        # 检测可用字体
        available_font = None
        for font_name in font_list:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path) and not font_path.endswith('DejaVuSans.ttf'):
                available_font = font_name
                break
        
        # 如果找到合适的中文字体，则应用配置
        if (available_font):
            plt.rcParams['font.family'] = available_font
            print(f"使用中文字体: {available_font}")
        else:
            # 使用英文替代
            print("未找到支持中文的字体，将使用英文标签")
            plt.rcParams['font.family'] = 'sans-serif'
    except Exception as e:
        print(f"配置中文字体时出错: {e}")
        # 使用英文标签作为后备方案
        plt.rcParams['font.family'] = 'sans-serif'

def check_cuda_environment():
    """检查CUDA环境并返回设备信息"""
    print("-" * 50)
    print("CUDA环境检查")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU训练")
        return torch.device("cpu")
    
    # CUDA可用信息
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    current_device = torch.cuda.current_device()
    print(f"当前GPU: {current_device} - {torch.cuda.get_device_name(current_device)}")
    
    # GPU内存信息
    print(f"GPU总内存: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.2f} GB")
    print(f"当前分配内存: {torch.cuda.memory_allocated(current_device) / 1024**3:.2f} GB")
    print(f"当前缓存内存: {torch.cuda.memory_reserved(current_device) / 1024**3:.2f} GB")
    
    # CUDNN信息
    print(f"cuDNN已启用: {torch.backends.cudnn.enabled}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    
    # 设置为确定性计算(仅调试使用，会降低性能)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # 设置为性能优化模式
    torch.backends.cudnn.benchmark = True
    
    print("-" * 50)
    
    return torch.device("cuda")

def train_with_cuda(model, train_loader, val_loader, criterion, optimizer, 
                    scaler_X, scaler_y, num_epochs=100, early_stop_patience=15,
                    mixed_precision=True, gradient_accumulation_steps=1):
    """
    使用CUDA优化的训练函数
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scaler_X: 特征缩放器
        scaler_y: 目标缩放器
        num_epochs: 训练轮数
        early_stop_patience: 早停耐心值
        mixed_precision: 是否使用混合精度训练
        gradient_accumulation_steps: 梯度累积步数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 使用正确的GradScaler初始化
    scaler = GradScaler() if mixed_precision else None
    
    # 训练记录
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    metrics_history = []
    training_times = []
    
    # 早停设置
    no_improve_epochs = 0
    
    # 确保模型输出目录存在
    model_dir = 'f:/rainfalldata/models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print(f"开始训练 - 设备: {device}, 混合精度: {mixed_precision}, 梯度累积: {gradient_accumulation_steps}步")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for i, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # 混合精度训练 - 修复autocast参数
            if mixed_precision:
                with autocast(device_type='cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y) / gradient_accumulation_steps
                
                # 缩放损失并反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积，每gradient_accumulation_steps步执行一次参数更新
                if (i + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪，防止梯度爆炸
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 参数更新
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # 标准训练流程
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y) / gradient_accumulation_steps
                loss.backward()
                
                # 梯度累积，每gradient_accumulation_steps步执行一次参数更新
                if (i + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps  # 恢复损失原始大小
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                if mixed_precision:
                    with autocast(device_type='cuda'):
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                else:
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
                
                # 收集预测和目标
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
                # 清理不需要的张量
                del outputs
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 计算训练时间
        epoch_time = time.time() - epoch_start_time
        training_times.append(epoch_time)
        
        # 计算评估指标
        all_predictions = np.array(all_predictions, dtype=np.float32)
        all_targets = np.array(all_targets, dtype=np.float32)
        
        # 反归一化以获得实际降雨量值
        pred_orig = scaler_y.inverse_transform(all_predictions)
        target_orig = scaler_y.inverse_transform(all_targets)
        
        # 计算评估指标
        metrics = calculate_metrics(target_orig, pred_orig)
        metrics_history.append(metrics)
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s')
        print(f'训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}')
        print(f'RMSE: {metrics["RMSE"]:.4f}, MAE: {metrics["MAE"]:.4f}, R²: {metrics["R^2"]:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(model_dir, 'best_rainfall_model_cuda.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if mixed_precision else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': metrics,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
            }, checkpoint_path)
            print(f"已保存最佳模型至 {checkpoint_path}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # 早停检查
        if no_improve_epochs >= early_stop_patience:
            print(f"早停机制触发，在{epoch+1}轮后停止训练")
            break
        
        # 手动垃圾回收
        gc.collect()
    
    # 训练结束，展示统计信息
    total_time = sum(training_times)
    avg_epoch_time = total_time / len(training_times)
    
    print("-" * 50)
    print(f"训练完成! 总时间: {total_time:.2f}秒, 平均每轮时间: {avg_epoch_time:.2f}秒")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 保存训练历史和指标图
    save_training_plots(train_losses, val_losses, metrics_history, training_times)
    
    return train_losses, val_losses, metrics_history

def save_training_plots(train_losses, val_losses, metrics_history, training_times):
    """保存训练历史图表"""
    plt.figure(figsize=(15, 12))
    
    # 使用英文替代中文标签，避免字体问题
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制RMSE曲线
    plt.subplot(2, 2, 2)
    plt.plot([m["RMSE"] for m in metrics_history])
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error')
    
    # 绘制R^2曲线
    plt.subplot(2, 2, 3)
    plt.plot([m["R^2"] for m in metrics_history])
    plt.xlabel('Epochs')
    plt.ylabel('R^2')
    plt.title('R-squared')
    
    # 绘制训练时间
    plt.subplot(2, 2, 4)
    plt.plot(training_times)
    plt.xlabel('Epochs')
    plt.ylabel('Seconds')
    plt.title('Time per Epoch')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('f:/rainfalldata/models/cuda_training_metrics.png')
    
    # 保存指标历史为CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv('f:/rainfalldata/models/cuda_metrics_history.csv', index=False)
    
    # 保存完整训练历史
    np.savez('f:/rainfalldata/models/cuda_training_history.npz', 
             train_losses=train_losses, 
             val_losses=val_losses,
             metrics_history=metrics_history,
             training_times=training_times)

def evaluate_model(model, test_loader, criterion, scaler_y):
    """使用CUDA评估模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_loss = 0
    all_predictions = []
    all_targets = []
    
    # 修复autocast参数
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # 使用正确的autocast
            if torch.cuda.is_available():
                with autocast(device_type='cuda'):
                    outputs = model(batch_X)
                    test_loss += criterion(outputs, batch_y).item()
            else:
                outputs = model(batch_X)
                test_loss += criterion(outputs, batch_y).item()
            
            # 收集预测和目标
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    
    # 转换为numpy数组并反归一化
    all_predictions = np.array(all_predictions, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    
    pred_orig = scaler_y.inverse_transform(all_predictions)
    target_orig = scaler_y.inverse_transform(all_targets)
    
    # 计算评估指标
    metrics = calculate_metrics(target_orig, pred_orig)
    
    # 绘制预测vs实际散点图 - 使用英文标签
    plt.figure(figsize=(10, 8))
    plt.scatter(target_orig, pred_orig, alpha=0.5)
    plt.plot([target_orig.min(), target_orig.max()], [target_orig.min(), target_orig.max()], 'r--')
    plt.xlabel('Actual Rainfall')
    plt.ylabel('Predicted Rainfall')
    plt.title('Actual vs Predicted Rainfall')
    plt.savefig('f:/rainfalldata/models/cuda_prediction_scatter.png')
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'Actual': target_orig.flatten(),
        'Predicted': pred_orig.flatten()
    })
    results_df.to_csv('f:/rainfalldata/models/cuda_prediction_results.csv', index=False)
    
    # 打印评估结果
    print("\n模型评估结果:")
    print(f'测试损失: {avg_test_loss:.4f}')
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
    
    return metrics, results_df

def main():
    """主函数 - CUDA优化的训练流程"""
    # 配置matplotlib支持中文
    configure_matplotlib_fonts()
    
    # 检查CUDA环境
    device = check_cuda_environment()
    
    # 数据路径
    DATAFILE = {
        "CMORPH": "f:/rainfalldata/CMORPHdata/CMORPH_2016_2020.mat",
        "CHIRPS": "f:/rainfalldata/CHIRPSdata/chirps_2016_2020.mat",
        "SM2RAIN": "f:/rainfalldata/sm2raindata/sm2rain_2016_2020.mat",
        "IMERG": "f:/rainfalldata/IMERGdata/IMERG_2016_2020.mat",
        "GSMAP": "f:/rainfalldata/GSMAPdata/GSMAP_2016_2020.mat",
        "PERSIANN": "f:/rainfalldata/PERSIANNdata/PERSIANN_2016_2020.mat",
        "CHM": "f:/rainfalldata/CHMdata/CHM_2016_2020.mat",
        "MASK": "f:/rainfalldata/mask.mat",
    }
    
    try:
        # 超参数配置
        config = {
            "window_size": 15,              # 时间窗口大小
            "batch_size": 64,               # 批次大小 (CUDA可以处理更大的批次)
            "hidden_size": 128,             # 隐藏层大小
            "num_layers": 2,                # LSTM层数
            "dropout_rate": 0.3,            # Dropout比率
            "learning_rate": 0.001,         # 学习率
            "num_epochs": 100,              # 最大训练轮数
            "mixed_precision": True,        # 是否使用混合精度训练
            "gradient_accumulation": 2,     # 梯度累积步数
            "prefetch_factor": 2,           # 数据预加载因子 
            "num_workers": 4,               # 数据加载线程数
        }
        
        print("加载数据...")
        # 加载参考数据
        reference_data = loadmat(DATAFILE["CHM"])['data'].astype(np.float32)
        
        # 加载卫星数据
        satellite_data_list = []
        for key in ["CHIRPS", "CMORPH", "GSMAP", "IMERG", "PERSIANN"]:
            satellite_data_list.append(loadmat(DATAFILE[key])['data'].astype(np.float32))
            gc.collect()
        
        # 加载掩码数据
        mask = loadmat(DATAFILE["MASK"])['mask']
        
        print("数据预处理...")
        # 预处理数据
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, valid_points, feature_importance = preprocess_data(
            reference_data, satellite_data_list, mask, 
            window_size=config["window_size"], 
            add_features=True
        )
        
        # 确保数据为float32类型
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # 释放原始数据内存
        del reference_data, satellite_data_list
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"数据形状 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 创建数据集和数据加载器
        print("创建数据加载器...")
        train_dataset = RainfallDataset(X_train, y_train)
        val_dataset = RainfallDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,  # 加速数据传输到GPU
            prefetch_factor=config["prefetch_factor"]
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["batch_size"], 
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True
        )
        
        # 模型初始化
        print("初始化模型...")
        input_size = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
        model = RainfallLSTM(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout_rate"],
            use_attention=True if len(X_train.shape) > 2 and X_train.shape[1] > 1 else False
        )
        
        # 显示模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数总量: {total_params:,}")
        
        # 损失函数和优化器
        criterion = CustomLoss(alpha=0.7, scale=100.0)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config["learning_rate"], 
            weight_decay=1e-5
        )
        
        # 学习率调度器
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer, mode='min', factor=0.5, patience=5, verbose=True
        #)
        # 修改为：
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 开始训练
        print("开始CUDA优化训练...")
        train_losses, val_losses, metrics_history = train_with_cuda(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            num_epochs=config["num_epochs"],
            mixed_precision=config["mixed_precision"],
            gradient_accumulation_steps=config["gradient_accumulation"]
        )
        
        # 加载最佳模型 - 添加安全加载
        print("加载最佳模型...")
        try:
            checkpoint = torch.load('f:/rainfalldata/models/best_rainfall_model_cuda.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("使用当前模型继续...")
        
        # 评估模型
        print("最终模型评估...")
        metrics, results_df = evaluate_model(model, val_loader, criterion, scaler_y)
        
        print("训练完成!")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        # 清理资源
        if 'model' in locals():
            model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        raise

if __name__ == "__main__":
    main()
