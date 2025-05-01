import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from paper4 import preprocess_data, calculate_metrics
from scipy.io import loadmat
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gc

class RainfallDataset(Dataset):
    """降水数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SelfAttention(nn.Module):
    """自注意力机制层"""
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Create Q, K, V projections
        Q = self.query(x)  # (batch_size, seq_len, hidden_size)
        K = self.key(x)    # (batch_size, seq_len, hidden_size)
        V = self.value(x)  # (batch_size, seq_len, hidden_size)
        
        # Scaled dot-product attention
        # (batch_size, seq_len, seq_len)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        
        # Apply softmax
        attention = torch.softmax(attention, dim=-1)
        
        # Multiply by values
        # (batch_size, seq_len, hidden_size)
        output = torch.matmul(attention, V)
        
        return output, attention

class RainfallLSTM(nn.Module):
    """增强版降水预测模型"""
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.3, use_attention=True):
        super(RainfallLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # 注意力层
        if use_attention:
            self.attention = SelfAttention(hidden_size*2)  # *2 是因为双向LSTM
            
            # 使用LayerNorm进行归一化
            self.layer_norm = nn.LayerNorm(hidden_size*2)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # 处理2D和3D输入情况
        if len(x.shape) == 2:
            # 如果是2D输入 (batch_size, features)，转换为3D
            # 假设这是只有1个时间步长的序列
            x = x.unsqueeze(1)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        if self.use_attention and x.shape[1] > 1:  # 只在序列长度>1时使用注意力
            # 应用注意力
            attn_out, _ = self.attention(lstm_out)
            # 残差连接和层归一化
            combined = self.layer_norm(lstm_out + attn_out)
            # 只使用最后一个时间步
            last_output = combined[:, -1, :]
        else:
            # 只使用最后一个时间步
            last_output = lstm_out[:, -1, :]
        
        # 通过全连接层得到预测结果
        output = self.fc(last_output)
        return output

class CustomLoss(nn.Module):
    """自定义损失函数，结合MSE和MAE并放大到更直观的尺度"""
    def __init__(self, alpha=0.7, scale=100.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.scale = scale
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, pred, target):
        # 结合MSE和MAE
        loss = self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)
        # 放大到更直观的尺度
        return loss * self.scale

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scaler_X, scaler_y, num_epochs=100, device='cuda',
                scheduler=None, early_stop_patience=15):
    """训练模型"""
    model = model.to(device)
    best_val_loss = float('inf')
    
    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    metrics_history = []
    
    # 早停参数
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 手动垃圾回收，释放未使用的内存
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                
                # 收集预测和目标
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
                # 清理不需要的张量
                del outputs
                
            # 清理GPU内存
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 计算额外的评估指标
        all_predictions = np.array(all_predictions, dtype=np.float32)
        all_targets = np.array(all_targets, dtype=np.float32)
        
        # 反归一化以获得实际降雨量值
        pred_orig = scaler_y.inverse_transform(all_predictions)
        target_orig = scaler_y.inverse_transform(all_targets)
        
        # 计算评估指标
        metrics = calculate_metrics(target_orig, pred_orig)
        metrics_history.append(metrics)
        
        # 学习率调度器
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'RMSE: {metrics["RMSE"]:.4f}, MAE: {metrics["MAE"]:.4f}, R²: {metrics["R^2"]:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': metrics,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
            }, 'f:/rainfalldata/models/best_rainfall_model.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # 早停
        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # 手动垃圾回收，释放内存
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # 保存损失曲线和评估指标
    plt.figure(figsize=(15, 10))
    
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
    
    # 绘制MAE曲线
    plt.subplot(2, 2, 3)
    plt.plot([m["MAE"] for m in metrics_history])
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    
    # 绘制R^2曲线
    plt.subplot(2, 2, 4)
    plt.plot([m["R^2"] for m in metrics_history])
    plt.xlabel('Epochs')
    plt.ylabel('R^2')
    plt.title('R-squared')
    
    # 创建目录
    if not os.path.exists('f:/rainfalldata/models'):
        os.makedirs('f:/rainfalldata/models')
    
    plt.tight_layout()
    plt.savefig('f:/rainfalldata/models/training_metrics.png')
    plt.show()
    
    # 保存指标历史为CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv('f:/rainfalldata/models/metrics_history.csv', index=False)
    
    # 保存完整训练历史
    np.savez('f:/rainfalldata/models/training_history.npz', 
             train_losses=train_losses, 
             val_losses=val_losses,
             metrics_history=metrics_history)
    
    return train_losses, val_losses, metrics_history

def evaluate_model(model, test_loader, criterion, device, scaler_y):
    """评估模型"""
    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            test_loss += criterion(outputs, batch_y).item()
            
            # 收集预测和目标
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 反归一化
    pred_orig = scaler_y.inverse_transform(all_predictions)
    target_orig = scaler_y.inverse_transform(all_targets)
    
    # 计算评估指标
    metrics = calculate_metrics(target_orig, pred_orig)
    
    # 打印评估结果
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'RMSE: {metrics["RMSE"]:.4f}')
    print(f'MAE: {metrics["MAE"]:.4f}')
    print(f'R²: {metrics["R^2"]:.4f}')
    print(f'NSE: {metrics["NSE"]:.4f}')
    
    # 绘制预测vs实际散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(target_orig, pred_orig, alpha=0.5)
    plt.plot([target_orig.min(), target_orig.max()], [target_orig.min(), target_orig.max()], 'r--')
    plt.xlabel('Actual Rainfall')
    plt.ylabel('Predicted Rainfall')
    plt.title('Actual vs Predicted Rainfall')
    plt.savefig('f:/rainfalldata/models/prediction_scatter.png')
    plt.show()
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'Actual': target_orig.flatten(),
        'Predicted': pred_orig.flatten()
    })
    results_df.to_csv('f:/rainfalldata/models/prediction_results.csv', index=False)
    
    return metrics, results_df

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置PyTorch以减少内存使用
    torch.backends.cudnn.benchmark = True  # 加速卷积操作
    torch.set_float32_matmul_precision('high')  # 使用更高效的矩阵乘法精度
    
    # 创建保存模型的目录
    if not os.path.exists('f:/rainfalldata/models'):
        os.makedirs('f:/rainfalldata/models')
    
    # 定义数据文件路径
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
        # 加载参考数据
        print("加载参考数据...")
        reference_data = loadmat(DATAFILE["CHM"])['data'].astype(np.float32)
        
        # 加载卫星数据列表
        print("加载卫星数据...")
        satellite_data_list = []
        for key in ["CHIRPS", "CMORPH", "GSMAP", "IMERG", "PERSIANN"]:
            data = loadmat(DATAFILE[key])['data'].astype(np.float32)
            satellite_data_list.append(data)
            # 立即清理不需要的变量
            gc.collect()
        
        # 加载掩码数据
        print("加载掩码数据...")
        mask = loadmat(DATAFILE["MASK"])['mask']
        
        # 设置超参数
        window_size = 15    # 减少窗口大小以节省内存
        batch_size = 32     # 减少批次大小以节省显存
        hidden_size = 64    # 减少隐藏层大小以节省显存
        num_layers = 2      # LSTM层数
        dropout_rate = 0.3  # Dropout比率
        learning_rate = 0.001  # 学习率
        num_epochs = 100    # 训练轮数
        
        # 数据预处理
        print("开始数据预处理...")
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, valid_points, feature_importance = preprocess_data(
            reference_data, satellite_data_list, mask, window_size=window_size, add_features=True, 
            dtype=np.float32  # 使用dtype参数确保使用float32
        )
        
        # 确保数据为float32类型
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # 手动释放内存
        del reference_data, satellite_data_list
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # 打印数据形状
        print(f"训练集形状: {X_train.shape}, {y_train.shape}")
        print(f"测试集形状: {X_test.shape}, {y_test.shape}")
        
        # 创建数据加载器
        print("创建数据加载器...")
        train_dataset = RainfallDataset(X_train, y_train)
        val_dataset = RainfallDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 初始化模型
        print("初始化模型...")
        input_size = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
        model = RainfallLSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout_rate,
            use_attention=True if len(X_train.shape) > 2 and X_train.shape[1] > 1 else False
        )
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数总量: {total_params}")
        
        # 自定义损失函数
        criterion = CustomLoss(alpha=0.7, scale=100.0)
        
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练模型
        print("开始训练模型...")
        train_losses, val_losses, metrics_history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            scaler_X, scaler_y, num_epochs=num_epochs, device=device,
            scheduler=scheduler, early_stop_patience=15
        )
        
        # 加载最佳模型
        print("加载最佳模型...")
        checkpoint = torch.load('f:/rainfalldata/models/best_rainfall_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 评估模型
        print("开始评估模型...")
        metrics, results_df = evaluate_model(model, val_loader, criterion, device, scaler_y)
        
        # 打印最终评估结果
        print("\n最终模型评估结果:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")
            
    except Exception as e:
        print(f"程序执行时出现错误: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈，帮助调试
        # 清理资源
        if 'model' in locals() and device == 'cuda':
            model.cpu()
            torch.cuda.empty_cache()
        gc.collect()
        raise

if __name__ == "__main__":
    main()

