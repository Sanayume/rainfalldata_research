import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import xgboost as xgb
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

DATAFILE = {
    "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat",
    "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat",
    "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat",
    "IMERG": "IMERGdata/IMERG_2016_2020.mat",
    "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat",
    "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat",
    "CHM": "CHMdata/CHM_2016_2020.mat",
    #"MASK": "mask.mat",
}

DATAS = {
    "CMORPH": loadmat(DATAFILE["CMORPH"])["data"][50:100, 100:150, 0:366],
    "CHIRPS": loadmat(DATAFILE["CHIRPS"])["data"][50:100, 100:150, 0:366],
    "SM2RAIN": loadmat(DATAFILE["SM2RAIN"])["data"][50:100, 100:150, 0:366],
    "IMERG": loadmat(DATAFILE["IMERG"])["data"][50:100, 100:150, 0:366],
    "GSMAP": loadmat(DATAFILE["GSMAP"])["data"][50:100, 100:150, 0:366],
    "PERSIANN": loadmat(DATAFILE["PERSIANN"])["data"][50:100, 100:150, 0:366],
    "CHM": loadmat(DATAFILE["CHM"])["data"][50:100, 100:150, 0:366],
    #"MASK": np.flipud(np.transpose(loadmat(DATAFILE["MASK"])["mask"], (1, 0))),
}
YDATAS = DATAS["CHM"].copy()
y = np.array(YDATAS)
print(f"y shape: {y.shape}") # (50, 50, 366)
y = np.transpose(y, (2, 0, 1))
print(f"y shape: {y.shape}") # (366, 50, 50)
DATAS.pop("CHM")

#print shape
for key, value in DATAS.items():
    print(f"{key}: {value.shape}")

#check is nan
for key, value in DATAS.items():
    if np.isnan(value).any():
        nan_indices = np.where(np.isnan(value))
        nan_sum = np.sum(np.isnan(value))
        print(f"{key} has nan values at indices {nan_indices} with a total of {nan_sum} nan values")
    else:
        print(f"{key} does not have nan values")

#
XDATAS = []
ORIGINAL_FEATURE = []
for key, value in DATAS.items():
    XDATAS.append(value)
    ORIGINAL_FEATURE.append(value)

x = np.array(XDATAS)  #(6, 50, 50, 366)
#tranpose data to (366, 6, 1, 50, 50)
x = np.transpose(x, (3, 0, 1, 2))


#split data into train and test for LSTM, 80% for train, 20% for test, the time is continuous


'''
#preprocess data for LSTM
#standardize data
standardizer = StandardScaler()
train_x = standardizer.fit_transform(train_x.reshape(-1, train_x.shape[2] * train_x.shape[3])).reshape(train_x.shape)
test_x = standardizer.transform(test_x.reshape(-1, test_x.shape[2] * test_x.shape[3])).reshape(test_x.shape)
train_y = y[:int(0.8*366), :, :]    
test_y = y[int(0.8*366):, :, :]
print(f"train_x shape: {train_x.shape}")
print(f"test_x shape: {test_x.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"test_y shape: {test_y.shape}")

# Modify the data reshaping before converting to torch tensors
train_x = train_x.reshape(-1, 6, 1, 50, 50)  # reshape to (batch, channels, depth, height, width)
test_x = test_x.reshape(-1, 6, 1, 50, 50)

# Convert to torch tensors with half precision
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

from torch import nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        # 3D convolution to extract aligned spatial features from rainfall products
        self.conv3d = nn.Conv3d(
            in_channels=6,  # 6 rainfall products
            out_channels=128,  # Increased output features
            kernel_size=(1, 7, 7),  # Larger spatial kernel for better spatial correlation
            stride=1,
            padding=(0, 3, 3),
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(1, 4, 4))  # Larger pooling to reduce spatial dims
        
        # Adaptive pooling to get fixed output size regardless of input spatial dims
        self.adaptive_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Preserves temporal dim (L)
        
        # Flatten spatial and channel dims to get (L, H) output
        self.flatten = nn.Flatten(start_dim=2)  # Flatten everything after temporal dim
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]  # 获取当前批次的大小
        x = self.conv3d(x)
        #print(f"conv3d shape: {x.shape}")
        x = self.relu(x)
        x = self.pool(x)
        #print(f"pool shape: {x.shape}")
        x = self.adaptive_pool(x)
        #print(f"adaptive_pool shape: {x.shape}")
        x = self.flatten(x)
        #print(f"flatten shape: {x.shape}")
        # 使用实际的 batch_size 而不是硬编码的 292
        x = x.reshape(1, batch_size, 128)
        #print(f"reshape shape: {x.shape}")
        x, _ = self.lstm(x)
        #print(f"lstm shape: {x.shape}")
        x = self.fc(x)
        x = x.reshape(batch_size, 50, 50)  # 改用动态的 batch_size
        return x


model = LSTM(input_size=128, hidden_size=128, output_size=50*50)

sgd = torch.optim.SGD(model.parameters(), lr=0.02)
adam = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100
for epoch in range(epochs):
    model.train()
    sgd.zero_grad()
    output = model(train_x)
    loss = nn.MSELoss()(output, train_y)
    loss.backward()
    sgd.step()

    with torch.no_grad():
        model.eval()
        output = model(test_x)
        loss = nn.MSELoss()(output, test_y)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
'''
#分类xgboost
xgb_clf = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.3,
    n_estimators=200,
    objective='binary:logistic', #分类
    booster='gbtree',
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0, #L1正则化
    reg_lambda=1, #L2正则化
    random_state=42,
    n_jobs= -1,  #使用所有CPU核心
    verbose=2 #打印信息
)

#将数据拆分为3中，一种是误报，一种是漏报，一种是正确

#误报  ,产品 > 0, 实际 = 0
#漏报  ,产品 = 0, 实际 > 0
#正确  ,产品 > 0, 实际 > 0
"""
x = np.transpose(x, (1, 0, 2, 3))
print(x.shape)
y = [y,y,y,y,y,y]
y = np.array(y)
print(y.shape)

# 使用括号来正确处理逻辑运算
X_with_false_alarm_mask = np.where((x > 0) & (y == 0))
X_with_miss_mask = np.where((x == 0) & (y > 0))
X_with_correct_mask = np.where((x > 0) & (y > 0))

x_with_false_alarm = x[X_with_false_alarm_mask]
x_with_miss = x[X_with_miss_mask]
x_with_correct = x[X_with_correct_mask]

# 使用更高效的方法找到最优阈值
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 准备数据进行阈值分析
# 将数据扁平化以便于处理
x_flat = x.reshape(x.shape[0], -1)  # (6, n_samples)
y_flat = y.reshape(y.shape[0], -1)  # (6, n_samples)

# 定义产品名称列表
products = ['CMORPH', 'CHIRPS', 'SM2RAIN', 'IMERG', 'GSMAP', 'PERSIANN']
results = []

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
min_fault_rate = 0
# 遍历每个产品进行阈值优化
for prod_idx, product_name in enumerate(products):
    print(f"正在分析 {product_name} 的最优阈值...")
    
    # 初始阈值扫描
    thresholds = np.linspace(0, 10, 50)  # 初始粗扫描使用较少的点
    best_threshold = 0
    min_fault_rate = float('inf')
    
    # 存储每个阈值的评估指标
    threshold_metrics = []
    
    # 粗扫描寻找最佳阈值区域
    for threshold in thresholds:
        # 根据阈值生成预测结果
        predictions = (x_flat[prod_idx] > threshold).astype(int)
        truth = (y_flat[prod_idx] > 0).astype(int)
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
        '''
        # 计算各种指标
        far = fp / (fp + tp) if (fp + tp) > 0 else 0  # 虚警率
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0  # 检测概率
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # 临界成功指数
        accuracy = (tp + tn) / (tp + tn + fp + fn)  # 准确率
        fault_points = fp + fn  # 错误点数（虚警+漏报）
        '''
        fault_rate = (fp + fn) / (tp + tn + fp + fn)
        success_rate = (tp + tn) / (tp + tn + fp + fn)
        # 记录该阈值的结果
        '''
        threshold_metrics.append({
            'threshold': threshold,
            'far': far, 
            'pod': pod,
            'csi': csi,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'fault_points': fault_points,
            'true_points': tp + tn
        })
        '''
        threshold_metrics.append({
            'threshold': threshold,
            'fault_rate': fault_rate,
            'success_rate': success_rate
        })
        

        # 更新最佳阈值（以错误点最少为标准）
        if fault_rate < min_fault_rate:
            min_fault_rate = fault_rate
            best_threshold = threshold
    
    # 在最佳阈值周围进行精细扫描
    fine_thresholds = np.linspace(max(0, best_threshold-0.5), best_threshold+0.5, 20)
    fine_results = []
    
    for threshold in fine_thresholds:
        # 对精细阈值区间重复评估过程
        predictions = (x_flat[prod_idx] > threshold).astype(int)
        truth = (y_flat[prod_idx] > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
        '''
        far = fp / (fp + tp) if (fp + tp) > 0 else 0
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fault_points = fp + fn
        '''
        fault_rate = (fp + fn) / (tp + tn + fp + fn)
        success_rate = (tp + tn) / (tp + tn + fp + fn)
        '''
        fine_results.append({
            'threshold': threshold,
            'far': far, 
            'pod': pod,
            'csi': csi,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'fault_points': fault_points,
            'true_points': tp + tn
        })
        '''
        fine_results.append({
            'threshold': threshold,
            'fault_rate': fault_rate,
            'success_rate': success_rate
        })
    # 根据错误点数排序找出最佳阈值
    fine_results.sort(key=lambda x: x['fault_rate'])
    optimal_result = fine_results[0]
    
    # 将最佳结果添加到总结果列表
    results.append({
        'product': product_name,
        'optimal_threshold': optimal_result['threshold'],
        'metrics': optimal_result
    })
    
    # 绘制阈值性能曲线
    # 为了更平滑的曲线，使用更多点
    plot_thresholds = np.linspace(0, 10, 100)
    '''
    fault_points_list = []
    true_points_list = []
    far_list = []
    pod_list = []
    csi_list = []
    '''
    fault_rate_list = []
    success_rate_list = []
    for threshold in plot_thresholds:
        predictions = (x_flat[prod_idx] > threshold).astype(int)
        truth = (y_flat[prod_idx] > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
        '''
        fault_points = fp + fn
        true_points = tp + tn
        far = fp / (fp + tp) if (fp + tp) > 0 else 0
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        fault_points_list.append(fault_points)
        true_points_list.append(true_points)
        far_list.append(far)
        pod_list.append(pod)
        csi_list.append(csi)
        '''
        fault_rate = (fp + fn) / (tp + tn + fp + fn)
        success_rate = (tp + tn) / (tp + tn + fp + fn)
        fault_rate_list.append(fault_rate)
        success_rate_list.append(success_rate)
    # 绘制当前产品的曲线
    ax = axes[prod_idx]
    ax.plot(plot_thresholds, fault_rate_list, label='错误率')
    ax.plot(plot_thresholds, success_rate_list, label='正确率')
    ax.axvline(x=optimal_result['threshold'], color='r', linestyle='--', 
               label=f'最优阈值: {optimal_result["threshold"]:.3f}')
    ax.set_title(f'{product_name} 阈值分析')
    ax.set_xlabel('阈值')
    ax.set_ylabel('错误率')
    ax.legend()

# 调整布局并显示图表
plt.tight_layout()
plt.savefig('threshold_analysis.png')
plt.show()

# 打印每个产品的最优阈值和对应的评估指标
print("\n===== 最优阈值分析结果 =====")
for result in results:
    print(f"{result['product']}: "
          f"最优阈值 = {result['optimal_threshold']:.3f}, "
          f"错误率 = {result['metrics']['fault_rate']:.3f}, "
          f"正确率 = {result['metrics']['success_rate']:.3f}")

# 为每个产品应用最优阈值
optimal_thresholds = [r['optimal_threshold'] for r in results]
print(f"Optimal thresholds: {optimal_thresholds}")

# 将阈值信息也保存为CSV文件
df_results = pd.DataFrame([{
    'Product': r['product'],
    'Optimal_Threshold': r['optimal_threshold'],
    'Fault_Rate': r['metrics']['fault_rate'],
    'Success_Rate': r['metrics']['success_rate'],
} for r in results])

df_results.to_csv('optimal_thresholds.csv', index=False)
print("阈值分析结果已保存到 'optimal_thresholds.csv'")

#################################################################
# 总体阈值寻优：不分产品，把所有点合并一起进行优化
#################################################################
print("\n===== 总体阈值寻优 =====")
# 将所有产品的数据合并为单个数组
x_all = x_flat.reshape(-1)  # 扁平化所有产品数据
y_all = y_flat.reshape(-1)  # 扁平化所有观测数据

# 进行初步阈值扫描
thresholds = np.linspace(0, 10, 50)
best_threshold = 0
min_fault_rate = float('inf')

# 绘制总体阈值寻优的图表
plt.figure(figsize=(10, 6))
fault_rate_overall = []
success_rate_overall = []

print("正在分析总体最优阈值...")
for threshold in thresholds:
    predictions = (x_all > threshold).astype(int)
    truth = (y_all > 0).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    
    # 计算评估指标
    fault_points = fp + fn
    true_points = tp + tn
    fault_rate = (fp + fn) / (tp + tn + fp + fn)
    success_rate = (tp + tn) / (tp + tn + fp + fn)
    
    fault_rate_overall.append(fault_rate)
    success_rate_overall.append(success_rate)
    
    # 更新最佳阈值
    if fault_rate < min_fault_rate:
        min_fault_rate = fault_rate
        best_threshold = threshold

# 进行精细扫描
fine_thresholds = np.linspace(max(0, best_threshold-0.5), best_threshold+0.5, 20)
fine_results_overall = []

for threshold in fine_thresholds:
    predictions = (x_all > threshold).astype(int)
    truth = (y_all > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    
    fault_rate = (fp + fn) / (tp + tn + fp + fn)
    success_rate = (tp + tn) / (tp + tn + fp + fn)
    
    fine_results_overall.append({
        'threshold': threshold,
        'fault_rate': fault_rate,
        'success_rate': success_rate
    })

# 选择最佳阈值
fine_results_overall.sort(key=lambda x: x['fault_rate'])
optimal_overall = fine_results_overall[0]

# 绘制更详细的阈值性能曲线
plot_thresholds = np.linspace(0, 10, 100)
detailed_metrics = {'fault_rate': [], 'success_rate': []}

for threshold in plot_thresholds:
    predictions = (x_all > threshold).astype(int)
    truth = (y_all > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    
    detailed_metrics['fault_rate'].append((fp + fn) / (tp + tn + fp + fn))
    detailed_metrics['success_rate'].append((tp + tn) / (tp + tn + fp + fn))

# 绘制总体阈值分析图
plt.plot(plot_thresholds, detailed_metrics['fault_rate'], label='错误率')
plt.plot(plot_thresholds, detailed_metrics['success_rate'], label='正确率')
plt.axvline(x=optimal_overall['threshold'], color='r', linestyle='--', 
           label=f'总体最优阈值: {optimal_overall["threshold"]:.3f}')
plt.title('总体数据阈值分析（所有产品合并）')
plt.xlabel('阈值')
plt.ylabel('错误率')
plt.legend()
plt.grid(True)
plt.savefig('overall_threshold_analysis.png')
plt.show()

# 打印总体最优阈值和对应的评估指标
print(f"\n总体最优阈值 = {optimal_overall['threshold']:.3f}")
print(f"错误率 = {optimal_overall['fault_rate']:.3f}")
print(f"正确率 = {optimal_overall['success_rate']:.3f}")

# 将总体阈值结果添加到CSV
overall_result = {
    'Product': 'Overall',
    'Optimal_Threshold': optimal_overall['threshold'],
    'Fault_Rate': optimal_overall['fault_rate'],
    'Success_Rate': optimal_overall['success_rate'],
    'False_Positives': optimal_overall['fp'],
    'False_Negatives': optimal_overall['fn']
}
df_results = pd.concat([df_results, pd.DataFrame([overall_result])], ignore_index=True)
df_results.to_csv('optimal_thresholds_with_overall.csv', index=False)
print("包含总体阈值的分析结果已保存到 'optimal_thresholds_with_overall.csv'")

"""
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


# 重新设置数据的形状，使其适用于分析
# 将y复制6份与x对应
y_repeated = np.repeat(y[np.newaxis, :, :, :], 6, axis=0)  # (6, 366, 50, 50)
x = np.transpose(x, (1, 0, 2, 3))
import matplotlib.pyplot as plt
# 定义函数计算各类错误率
def calculate_error_rates(x_data, y_data, threshold):
    # 根据阈值筛选降雨事件
    selected_mask = np.where(x_data <= threshold, 1, 0)
    selected_x = np.where(x_data <= threshold, x_data, -999)
    total_selected = np.sum(selected_mask)  # 该阈值下被选中的总样本数
    if total_selected == 0:
        return None
    # 计算误报数
    FAR_points = (selected_x > 0) & (y_data == 0)
    MS_points = (selected_x == 0) & (y_data > 0)
    TRUE_points = ((selected_x > 0) & (y_data > 0)) | ((selected_x == 0) & (y_data == 0))
    FAR_sum = np.sum(FAR_points)
    MS_sum = np.sum(MS_points)
    TRUE_sum = np.sum(TRUE_points)
    FAR_rate = FAR_sum / total_selected
    MS_rate = MS_sum / total_selected
    TRUE_rate = TRUE_sum / total_selected
    return {
        'false_alarm_rate': FAR_rate,
        'miss_rate': MS_rate,
        'error_rate': (FAR_rate + MS_rate),
        'correct_rate': TRUE_rate
    }
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 初始化阈值范围用于粗略扫描，扩大到20
thresholds = np.linspace(0, 20, 100)

# 存储所有阈值的错误率
results = []

print("正在进行阈值优化...")
for threshold in thresholds:
    rates = calculate_error_rates(x, y_repeated, threshold)
    results.append({
        'threshold': threshold,
        **rates
    })

# 找出各类错误率最大的阈值
results_df = pd.DataFrame(results)
max_false_alarm_idx = results_df['false_alarm_rate'].idxmax()
max_miss_idx = results_df['miss_rate'].idxmax()
max_error_idx = results_df['error_rate'].idxmax()

best_false_alarm_threshold = results_df.iloc[max_false_alarm_idx]['threshold']
best_miss_threshold = results_df.iloc[max_miss_idx]['threshold']
best_error_threshold = results_df.iloc[max_error_idx]['threshold']

# 在最佳误报率阈值附近进行精细扫描
fine_thresholds_fa = np.linspace(max(0, best_false_alarm_threshold - 0.2), 
                               min(20, best_false_alarm_threshold + 0.2), 50)
fine_results_fa = []
for threshold in fine_thresholds_fa:
    rates = calculate_error_rates(x, y_repeated, threshold)
    fine_results_fa.append({
        'threshold': threshold,
        **rates
    })
fine_df_fa = pd.DataFrame(fine_results_fa)
final_max_fa_idx = fine_df_fa['false_alarm_rate'].idxmax()
final_best_fa_threshold = fine_df_fa.iloc[final_max_fa_idx]['threshold']

# 在最佳漏报率阈值附近进行精细扫描
fine_thresholds_miss = np.linspace(max(0, best_miss_threshold - 0.2), 
                                 min(20, best_miss_threshold + 0.2), 50)
fine_results_miss = []
for threshold in fine_thresholds_miss:
    rates = calculate_error_rates(x, y_repeated, threshold)
    fine_results_miss.append({
        'threshold': threshold,
        **rates
    })
fine_df_miss = pd.DataFrame(fine_results_miss)
final_max_miss_idx = fine_df_miss['miss_rate'].idxmax()
final_best_miss_threshold = fine_df_miss.iloc[final_max_miss_idx]['threshold']

# 在最佳总错误率阈值附近进行精细扫描
fine_thresholds_error = np.linspace(max(0, best_error_threshold - 0.2), 
                                  min(20, best_error_threshold + 0.2), 50)
fine_results_error = []
for threshold in fine_thresholds_error:
    rates = calculate_error_rates(x, y_repeated, threshold)
    fine_results_error.append({
        'threshold': threshold,
        **rates
    })
fine_df_error = pd.DataFrame(fine_results_error)
final_max_error_idx = fine_df_error['error_rate'].idxmax()
final_best_error_threshold = fine_df_error.iloc[final_max_error_idx]['threshold']

# 打印最终结果
print(f"最佳误报率阈值: {final_best_fa_threshold:.4f}, 误报率: {fine_df_fa.iloc[final_max_fa_idx]['false_alarm_rate']:.4f}")
print(f"最佳漏报率阈值: {final_best_miss_threshold:.4f}, 漏报率: {fine_df_miss.iloc[final_max_miss_idx]['miss_rate']:.4f}")
print(f"最佳总错误率阈值: {final_best_error_threshold:.4f}, 错误率: {fine_df_error.iloc[final_max_error_idx]['error_rate']:.4f}")

# 绘制图像
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# 绘制误报率曲线
axes[0].plot(thresholds, results_df['false_alarm_rate'], 'b-', label='粗扫描', linewidth=2)
axes[0].plot(fine_thresholds_fa, fine_df_fa['false_alarm_rate'], 'r--', label='精细扫描', linewidth=2)
axes[0].axvline(x=final_best_fa_threshold, color='g', linestyle='-', label=f'最佳阈值: {final_best_fa_threshold:.4f}')
axes[0].set_title('误报率与阈值的关系', fontsize=14)
axes[0].set_xlabel('阈值', fontsize=12)
axes[0].set_ylabel('误报率', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend(loc='best')

# 绘制漏报率曲线
axes[1].plot(thresholds, results_df['miss_rate'], 'b-', label='粗扫描', linewidth=2)
axes[1].plot(fine_thresholds_miss, fine_df_miss['miss_rate'], 'r--', label='精细扫描', linewidth=2)
axes[1].axvline(x=final_best_miss_threshold, color='g', linestyle='-', label=f'最佳阈值: {final_best_miss_threshold:.4f}')
axes[1].set_title('漏报率与阈值的关系', fontsize=14)
axes[1].set_xlabel('阈值', fontsize=12)
axes[1].set_ylabel('漏报率', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend(loc='best')

# 绘制总错误率曲线
axes[2].plot(thresholds, results_df['error_rate'], 'b-', label='错误率(粗扫描)', linewidth=2)
axes[2].plot(thresholds, results_df['correct_rate'], 'g-', label='正确率', linewidth=2)
axes[2].plot(fine_thresholds_error, fine_df_error['error_rate'], 'r--', label='错误率(精细扫描)', linewidth=2)
axes[2].axvline(x=final_best_error_threshold, color='m', linestyle='-', label=f'最佳阈值: {final_best_error_threshold:.4f}')
axes[2].set_title('总错误率与阈值的关系', fontsize=14)
axes[2].set_xlabel('阈值', fontsize=12)
axes[2].set_ylabel('比率', fontsize=12)
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].legend(loc='best')

plt.tight_layout()
plt.savefig('threshold_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 将结果保存到CSV
results_summary = pd.DataFrame([
    {'错误类型': '误报', '最佳阈值': final_best_fa_threshold, '错误率': fine_df_fa.iloc[final_max_fa_idx]['false_alarm_rate']},
    {'错误类型': '漏报', '最佳阈值': final_best_miss_threshold, '错误率': fine_df_miss.iloc[final_max_miss_idx]['miss_rate']},
    {'错误类型': '总错误', '最佳阈值': final_best_error_threshold, '错误率': fine_df_error.iloc[final_max_error_idx]['error_rate']}
])
results_summary.to_csv('optimal_thresholds_summary.csv', index=False, encoding='utf-8-sig')
print("分析结果已保存到 'threshold_analysis_results.png' 和 'optimal_thresholds_summary.csv'")







    












































































