import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
# 完全移除SelectKBest和f_regression的导入
import gc

def simple_feature_selection(X, y, k=30):
    """
    简单特征选择函数，使用相关系数的绝对值作为特征重要性指标
    使用float32节省内存
    """
    # 确保输入是float32类型
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    n_samples, n_features = X.shape
    
    # 计算每个特征与目标的相关性 (分批计算以节省内存)
    batch_size = min(1000, n_features)
    n_batches = (n_features + batch_size - 1) // batch_size
    
    feature_scores = np.zeros(n_features, dtype=np.float32)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_features)
        
        X_batch = X[:, start_idx:end_idx]
        
        # 中心化数据 (避免整体中心化，按批次进行)
        X_centered = X_batch - np.mean(X_batch, axis=0, dtype=np.float32)
        y_centered = y - np.mean(y, dtype=np.float32)
        
        # 计算相关系数
        X_std = np.std(X_centered, axis=0, dtype=np.float32) + np.finfo(np.float32).eps
        y_std = np.std(y_centered, dtype=np.float32) + np.finfo(np.float32).eps
        
        # 计算批次的相关系数
        corr = np.zeros(end_idx - start_idx, dtype=np.float32)
        for j in range(end_idx - start_idx):
            # 逐特征计算相关系数，避免一次性大矩阵计算
            corr[j] = np.abs(np.mean(X_centered[:, j] * y_centered, dtype=np.float32) / (X_std[j] * y_std))
        
        feature_scores[start_idx:end_idx] = corr
        
        # 手动清理内存
        del X_batch, X_centered
        gc.collect()
    
    # 获取前k个最重要特征的索引
    selected_indices = np.argsort(-feature_scores)[:k]
    selected_mask = np.zeros(n_features, dtype=bool)
    selected_mask[selected_indices] = True
    
    return selected_mask, feature_scores

def preprocess_data(reference_data, satellite_data_list, mask, window_size=30, add_features=True, dtype=np.float32):
    """
    数据预处理函数
    
    Args:
        reference_data: 参考数据 shape=(144, 256, 366)
        satellite_data_list: 卫星数据列表 [shape=(144, 256, 366)] * 5
        mask: 掩码数据 shape=(144, 256)
        window_size: 滑动窗口大小，默认30天
        add_features: 是否添加高级特征，默认为True
        dtype: 数据类型，默认为np.float32
    """
    # 确保所有输入数据为指定类型，减少内存占用
    reference_data = reference_data.astype(dtype)
    satellite_data_list = [data.astype(dtype) for data in satellite_data_list]
    
    # 1. 整理数据维度
    # 转换为 (days, points, products) 格式
    valid_points = ~np.isnan(reference_data[:,:,0])  # 有效点掩码
    n_points = np.sum(valid_points)
    n_days = reference_data.shape[2]
    
    # 重组卫星数据
    satellite_data = np.stack(satellite_data_list, axis=-1)  # (144, 256, 366, 5)
    
    # 提取有效点的数据
    X = np.zeros((n_days, n_points, 5), dtype=dtype)  # 卫星数据
    y = np.zeros((n_days, n_points), dtype=dtype)     # 参考数据
    
    point_idx = 0
    for i in range(valid_points.shape[0]):
        for j in range(valid_points.shape[1]):
            if valid_points[i,j]:
                X[:,point_idx,:] = satellite_data[i,j,:,:]
                y[:,point_idx] = reference_data[i,j,:]
                point_idx += 1
    
    # 2. 创建滑动窗口样本
    X_windows = []
    y_target = []
    
    for i in range(n_days - window_size):
        X_windows.append(X[i:i+window_size])
        y_target.append(y[i+window_size])
    
    X_windows = np.array(X_windows, dtype=dtype)  # (n_samples, window_size, n_points, 5)
    y_target = np.array(y_target, dtype=dtype)    # (n_samples, n_points)
    
    # 使用分批处理来降低内存使用
    batch_size = 50  # 减小批次大小以降低内存使用
    n_samples = X_windows.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    if add_features and window_size > 1:
        # 分批计算统计特征
        all_features_list = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = X_windows[start_idx:end_idx]
            
            # 计算统计特征，明确指定float32类型
            means = np.mean(batch_data, axis=1, dtype=dtype)  # 均值
            stds = np.std(batch_data, axis=1, dtype=dtype)    # 标准差
            maxs = np.max(batch_data, axis=1)    # 最大值
            mins = np.min(batch_data, axis=1)    # 最小值
            medians = np.median(batch_data, axis=1).astype(dtype)  # 中位数
            q25 = np.percentile(batch_data, 25, axis=1).astype(dtype)  # 25分位数
            q75 = np.percentile(batch_data, 75, axis=1).astype(dtype)  # 75分位数
            
            # 计算趋势特征
            if window_size >= 3:
                # 简单线性趋势 (使用最后3天数据)
                trends = batch_data[:, -1, :, :] - batch_data[:, -3, :, :]
                
                # 变化率
                rates = (batch_data[:, -1, :, :] - batch_data[:, 0, :, :]) / dtype(window_size - 1)
                rates = np.where(np.isnan(rates) | np.isinf(rates), dtype(0), rates)
                
                # 组合所有特征
                batch_features = np.concatenate([
                    batch_data[:, -1, :, :],  # 最后一天的原始特征
                    means, stds, maxs, mins, medians, 
                    q25, q75, trends, rates
                ], axis=2).astype(dtype)  # (batch_size, n_points, 5*10)
            else:
                # 组合所有特征
                batch_features = np.concatenate([
                    batch_data[:, -1, :, :],  # 最后一天的原始特征
                    means, stds, maxs, mins, medians, 
                    q25, q75
                ], axis=2).astype(dtype)  # (batch_size, n_points, 5*8)
                
            all_features_list.append(batch_features)
            
            # 手动清理内存
            del batch_data, means, stds, maxs, mins, medians, q25, q75
            if window_size >= 3:
                del trends, rates
            gc.collect()
        
        # 合并所有批次的特征
        all_features = np.concatenate(all_features_list, axis=0)
        
        # 手动清理内存
        del all_features_list
        gc.collect()
        
        # 重塑数据形状
        X_reshaped = all_features.reshape(n_samples * n_points, -1)
        
        # 创建季节性特征
        day_of_year = np.arange(window_size, n_days, dtype=np.int32) % 366  # 一年中的第几天
        sin_day = np.sin(2 * np.pi * day_of_year / 366).astype(dtype)
        cos_day = np.cos(2 * np.pi * day_of_year / 366).astype(dtype)
        
        # 将季节性特征扩展到与X_reshaped相同的形状
        sin_features = np.repeat(sin_day[:, np.newaxis], n_points, axis=1).flatten()[:, np.newaxis]
        cos_features = np.repeat(cos_day[:, np.newaxis], n_points, axis=1).flatten()[:, np.newaxis]
        
        # 组合所有特征
        X_reshaped = np.hstack([X_reshaped, sin_features, cos_features]).astype(dtype)
        
        # 手动清理内存
        del all_features, sin_day, cos_day, sin_features, cos_features
        gc.collect()
    else:
        # 使用原始数据
        X_reshaped = X_windows.reshape(n_samples * n_points, window_size, 5)
    
    y_reshaped = y_target.reshape(-1, 1)
    
    # 4. 移除包含NaN的样本
    if add_features and window_size > 1:
        valid_samples = ~np.isnan(X_reshaped).any(axis=1) & ~np.isnan(y_reshaped).any(axis=1)
    else:
        valid_samples = ~np.isnan(X_reshaped).any(axis=(1,2)) & ~np.isnan(y_reshaped).any(axis=1)
    
    X_clean = X_reshaped[valid_samples]
    y_clean = y_reshaped[valid_samples]
    
    # 手动清理内存
    del X_reshaped, y_reshaped
    gc.collect()
    
    # 5. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    
    # 手动清理内存
    del X_clean, y_clean
    gc.collect()
    
    # 6. 数据归一化
    if add_features and window_size > 1:
        # 对于扁平化的特征，直接归一化
        scaler_X = RobustScaler(with_centering=False)  # 减少内存使用，不使用中心化
        X_train_scaled = scaler_X.fit_transform(X_train).astype(dtype)
        X_test_scaled = scaler_X.transform(X_test).astype(dtype)
        
        # 特征选择 (只保留最重要的特征) - 使用自定义的特征选择方法
        print("执行特征选择...")
        k_features = min(30, X_train_scaled.shape[1])  # 减少特征数量
        
        # 使用自定义特征选择而不是SelectKBest
        selected_mask, feature_scores = simple_feature_selection(X_train_scaled, y_train.ravel(), k=k_features)
        
        # 应用特征选择
        X_train_selected = X_train_scaled[:, selected_mask]
        X_test_selected = X_test_scaled[:, selected_mask]
        
        # 使用所选特征
        X_train_final = X_train_selected.astype(dtype)
        X_test_final = X_test_selected.astype(dtype)
        
        feature_importance = {
            'selected_mask': selected_mask,
            'feature_scores': feature_scores,
            'original_shape': X_train.shape[1]
        }
        
        # 手动清理内存
        del X_train_scaled, X_test_scaled, X_train_selected, X_test_selected
        gc.collect()
    else:
        # 对于3D数据，需要重塑再归一化
        scaler_X = RobustScaler(with_centering=False)  # 不使用中心化以减少内存使用
        X_train_reshaped = X_train.reshape(-1, 5)
        X_test_reshaped = X_test.reshape(-1, 5)
        
        X_train_scaled = scaler_X.fit_transform(X_train_reshaped).astype(dtype)
        X_test_scaled = scaler_X.transform(X_test_reshaped).astype(dtype)
        
        # 恢复原始形状
        X_train_final = X_train_scaled.reshape(X_train.shape)
        X_test_final = X_test_scaled.reshape(X_test.shape)
        feature_importance = None
        
        # 手动清理内存
        del X_train_reshaped, X_test_reshaped, X_train_scaled, X_test_scaled
        gc.collect()
    
    # 对标签进行归一化
    scaler_y = RobustScaler(with_centering=False)  # 不使用中心化以减少内存使用
    y_train_scaled = scaler_y.fit_transform(y_train).astype(dtype)
    y_test_scaled = scaler_y.transform(y_test).astype(dtype)
    
    # 手动清理内存
    del X_train, X_test, y_train, y_test
    gc.collect()
    
    print(f"Memory usage - X_train: {X_train_final.nbytes / (1024**3):.2f} GB, X_test: {X_test_final.nbytes / (1024**3):.2f} GB")
    
    return (X_train_final, X_test_final, y_train_scaled, y_test_scaled, 
            scaler_X, scaler_y, valid_points, feature_importance)

def calculate_metrics(y_true, y_pred):
    """计算多种评估指标"""
    # 确保输入是float32类型
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    # 均方误差
    mse = np.mean((y_true - y_pred) ** 2, dtype=np.float32)
    # 均方根误差
    rmse = np.sqrt(mse, dtype=np.float32)
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred), dtype=np.float32)
    # R^2分数
    y_mean = np.mean(y_true, dtype=np.float32)
    ss_total = np.sum((y_true - y_mean) ** 2, dtype=np.float32)
    ss_residual = np.sum((y_true - y_pred) ** 2, dtype=np.float32)
    r2 = 1 - (ss_residual / ss_total)
    # Nash-Sutcliffe效率系数 (NSE)
    nse = 1 - (np.sum((y_true - y_pred) ** 2, dtype=np.float32) / 
               np.sum((y_true - np.mean(y_true, dtype=np.float32)) ** 2, dtype=np.float32))
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R^2': float(r2),
        'NSE': float(nse)
    }

# 数据加载和处理示例
if __name__ == "__main__":
    # 定义数据文件路径
    DATAFILE = {
        "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat",
        "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat",
        "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat",
        "IMERG": "IMERGdata/IMERG_2016_2020.mat",
        "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat",
        "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat",
        "CHM": "CHMdata/CHM_2016_2020.mat",
        "MASK": "mask.mat",
    }
    
    # 加载数据
    reference_data = loadmat(DATAFILE["CHM"])['data']
    satellite_data_list = []
    for product in ['CHIRPS', 'CMORPH', 'GSMAP', 'IMERG', 'PERSIANN']:
        data = loadmat(DATAFILE[product])['data']
        satellite_data_list.append(data)
    
    # 加载掩码
    mask = loadmat(DATAFILE["MASK"])["mask"]
    
    # 处理数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, valid_points, feature_importance = preprocess_data(
        reference_data, satellite_data_list, mask, window_size=30, add_features=True
    )
    
    # 打印数据形状
    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")
