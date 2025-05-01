import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import confusion_matrix, mean_squared_error
from lstm_train import RainfallLSTM, RainfallDataset
from paper4 import preprocess_data
from torch.utils.data import DataLoader

def load_best_model(model_path='f:/rainfalldata/models/best_rainfall_model.pth'):
    """加载最佳模型并返回模型及其参数"""
    # 加载检查点
    checkpoint = torch.load(model_path)
    
    # 检查模型输入大小
    input_size = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[1]
    hidden_size = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[0] // 4
    
    # 创建模型
    model = RainfallLSTM(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def visualize_loss_history(history_path='f:/rainfalldata/models/training_history.npz'):
    """可视化训练历史"""
    # 加载训练历史
    history = np.load(history_path, allow_pickle=True)
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    metrics_history = history['metrics_history']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 损失曲线
    axes[0, 0].plot(train_losses, label='训练损失')
    axes[0, 0].plot(val_losses, label='验证损失')
    axes[0, 0].set_title('训练与验证损失')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # RMSE曲线
    rmse_values = [m['RMSE'] for m in metrics_history]
    axes[0, 1].plot(rmse_values)
    axes[0, 1].set_title('均方根误差(RMSE)')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].grid(True)
    
    # MAE曲线
    mae_values = [m['MAE'] for m in metrics_history]
    axes[1, 0].plot(mae_values)
    axes[1, 0].set_title('平均绝对误差(MAE)')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True)
    
    # R^2曲线
    r2_values = [m['R^2'] for m in metrics_history]
    axes[1, 1].plot(r2_values)
    axes[1, 1].set_title('决定系数(R²)')
    axes[1, 1].set_xlabel('轮次')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('f:/rainfalldata/models/detailed_history_plot.png')
    plt.show()

def visualize_predictions_scatter(results_path='f:/rainfalldata/models/prediction_results.csv'):
    """可视化预测结果散点图"""
    # 加载预测结果
    results = pd.read_csv(results_path)
    
    # 创建图表
    plt.figure(figsize=(12, 10))
    
    # 绘制散点图
    sns.scatterplot(x='Actual', y='Predicted', data=results, alpha=0.5)
    
    # 添加1:1线
    max_val = max(results['Actual'].max(), results['Predicted'].max())
    min_val = min(results['Actual'].min(), results['Predicted'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 添加回归线
    sns.regplot(x='Actual', y='Predicted', data=results, 
                scatter=False, ci=None, line_kws={'color': 'green'})
    
    # 计算相关系数
    corr = results['Actual'].corr(results['Predicted'])
    rmse = np.sqrt(mean_squared_error(results['Actual'], results['Predicted']))
    
    # 添加文本信息
    plt.text(0.05, 0.95, f'相关系数: {corr:.4f}\nRMSE: {rmse:.4f}', 
             transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('实际降水量 vs 预测降水量')
    plt.xlabel('实际降水量 (mm/day)')
    plt.ylabel('预测降水量 (mm/day)')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('f:/rainfalldata/models/enhanced_scatter_plot.png')
    plt.show()
    
    # 绘制残差分析
    plt.figure(figsize=(12, 10))
    results['Residual'] = results['Actual'] - results['Predicted']
    
    plt.subplot(2, 1, 1)
    sns.scatterplot(x='Actual', y='Residual', data=results, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('残差 vs 实际值')
    plt.xlabel('实际降水量 (mm/day)')
    plt.ylabel('残差 (mm/day)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    sns.histplot(results['Residual'], kde=True)
    plt.title('残差分布')
    plt.xlabel('残差 (mm/day)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('f:/rainfalldata/models/residual_analysis.png')
    plt.show()
    
    # 绘制分类表现的混淆矩阵 (以0.5mm为分界点判断是否有雨)
    results['Actual_Rain'] = (results['Actual'] > 0.5).astype(int)
    results['Predicted_Rain'] = (results['Predicted'] > 0.5).astype(int)
    
    cm = confusion_matrix(results['Actual_Rain'], results['Predicted_Rain'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['无雨', '有雨'],
                yticklabels=['无雨', '有雨'])
    plt.title('降雨检测混淆矩阵 (阈值: 0.5mm/day)')
    plt.ylabel('实际类别')
    plt.xlabel('预测类别')
    plt.savefig('f:/rainfalldata/models/rain_detection_confusion_matrix.png')
    plt.show()

def generate_spatial_predictions(model, checkpoint, reference_data, satellite_data_list, mask, day_index=-1):
    """生成空间预测图"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 获取标量
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    # 获取有效点掩码
    valid_points = ~np.isnan(reference_data[:,:,0])
    
    # 提取选定日期的数据
    reference_day = reference_data[:,:,day_index]
    
    # 准备卫星数据
    satellite_day_list = [data[:,:,day_index] for data in satellite_data_list]
    satellite_day = np.stack(satellite_day_list, axis=-1)
    
    # 创建输出数组
    predicted_map = np.zeros_like(reference_day)
    predicted_map.fill(np.nan)
    
    # 遍历每个有效点并进行预测
    with torch.no_grad():
        for i in range(valid_points.shape[0]):
            for j in range(valid_points.shape[1]):
                if valid_points[i,j]:
                    # 提取特征
                    features = satellite_day[i,j,:]
                    
                    # 归一化特征
                    features_scaled = scaler_X.transform(features.reshape(1, -1))
                    
                    # 准备模型输入
                    inputs = torch.FloatTensor(features_scaled).to(device)
                    
                    # 预测
                    output = model(inputs)
                    
                    # 反归一化预测结果
                    pred = scaler_y.inverse_transform(output.cpu().numpy())[0, 0]
                    
                    # 存储预测结果
                    predicted_map[i,j] = pred
    
    # 绘制地图
    fig = plt.figure(figsize=(20, 10))
    
    # 设置中国区域的范围
    lat_min, lat_max = 15, 55
    lon_min, lon_max = 70, 140
    
    # 创建制图投影
    proj = ccrs.PlateCarree()
    
    # 创建实际降水量子图
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    # 绘制实际降水量
    im1 = ax1.imshow(reference_day, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max],
                     transform=proj, cmap='Blues', vmin=0, vmax=20)
    ax1.set_title('实际降水量 (mm/day)')
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    
    # 创建预测降水量子图
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    
    # 绘制预测降水量
    im2 = ax2.imshow(predicted_map, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max],
                     transform=proj, cmap='Blues', vmin=0, vmax=20)
    ax2.set_title('预测降水量 (mm/day)')
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(f'f:/rainfalldata/models/spatial_prediction_day_{day_index}.png')
    plt.show()
    
    # 计算预测误差
    error_map = predicted_map - reference_day
    
    # 绘制误差图
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # 绘制误差
    im = ax.imshow(error_map, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max],
                 transform=proj, cmap='RdBu_r', vmin=-10, vmax=10)
    ax.set_title('预测误差 (mm/day)')
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    plt.savefig(f'f:/rainfalldata/models/prediction_error_day_{day_index}.png')
    plt.show()

if __name__ == "__main__":
    # 加载最佳模型
    model, checkpoint = load_best_model()
    print("模型加载成功！")
    
    # 可视化训练历史
    visualize_loss_history()
    
    # 可视化预测结果
    visualize_predictions_scatter()
    
    # 生成空间预测（如果需要）
    generate_spatial = input("是否生成空间预测图？(y/n): ").lower() == 'y'
    if generate_spatial:
        # 加载数据
        reference_data = loadmat('f:/rainfalldata/CHM0-25.mat')['data']
        satellite_data_list = []
        for file in ['f:/rainfalldata/CHIRPS0-25.mat', 'f:/rainfalldata/CMORPH0-25.mat', 
                     'f:/rainfalldata/GSMAP0-25.mat', 'f:/rainfalldata/IMERG0-25.mat', 
                     'f:/rainfalldata/PERSIANN0-25.mat']:
            data = loadmat(file)['data']
            satellite_data_list.append(data)
        
        mask = loadmat('f:/rainfalldata/mask.mat')['mask']
        mask[109:132, 192:204] = 0
        
        # 生成空间预测
        day_index = -1  # 使用最后一天的数据
        generate_spatial_predictions(model, checkpoint, reference_data, satellite_data_list, mask, day_index)
