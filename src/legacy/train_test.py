import xarray as xr          # 用于处理多维数组数据，特别适合处理气象数据
import numpy as np          # 用于数值计算，提供数组操作功能
import torch                # PyTorch深度学习框架，用于构建神经网络
import torch.nn as nn      # 神经网络模块，提供了各种层的实现
import torch.optim as optim # 优化器模块，提供了各种优化算法
from sklearn.model_selection import train_test_split  # 用于将数据集分为训练集和测试集
from sklearn.preprocessing import StandardScaler      # 用于数据标准化，使数据均值为0，方差为1
import matplotlib.pyplot as plt                      # 用于数据可视化
from torch.utils.data import DataLoader, TensorDataset

# 加载数据并进行区域裁剪
# 读取IMERG.npy文件，该文件包含了降水数据
# shape为 [时间, 纬度, 经度]
everyday = np.load('D:\\data\\IMERG.npy')
# 裁剪特定区域: 可以通过修改这些切片参数来选择不同的区域
# 2520:3169 表示纬度范围，360:720 表示经度范围
everyday = everyday[:,3000:3010,1000:1010]

# 创建时序预测的数据集
# X: 输入数据，连续3天的数据
# Y: 预测目标，第4天的数据
X = []
Y = []
# 遍历整个时间序列，留出3天作为最后的预测天数
for i in range(everyday.shape[0] - 3):
    X.append(everyday[i:i+3,:,:])    # 每个样本包含连续3天的数据，维度为[3, 纬度, 经度]
    Y.append(everyday[i+3,:,:])      # 对应的第4天数据，维度为[纬度, 经度]

# 将列表转换为numpy数组，便于后续处理
# X shape: [样本数, 3, 纬度, 经度]
# Y shape: [样本数, 纬度, 经度]
X = np.array(X)
Y = np.array(Y)

# 数据标准化处理
scaler = StandardScaler()
# 将3D数据展平为2D以进行标准化
# -1表示自动计算维度，保证数据总量不变
X_flat = X.reshape(X.shape[0], -1)   
X_scaled = scaler.fit_transform(X_flat)  # 进行标准化转换
X = X_scaled.reshape(X.shape)        # 重新转换回原始维度
Y = scaler.fit_transform(Y.reshape(Y.shape[0], -1)).reshape(Y.shape)  # 对Y进行标准化

# 划分训练集和测试集
# test_size=0.2 表示20%的数据用于测试
# random_state=42 确保可重复性
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# 转换为PyTorch张量，使用GPU训练时可以添加.cuda()
X_train = torch.FloatTensor(X_train)  # 转换为PyTorch浮点张量
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

batch_size = 16  # 批次大小
train_dataset = TensorDataset(X_train, y_train)  # 创建训练数据集
test_dataset = TensorDataset(X_test, y_test)    # 创建测试数据集

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3*10*10, 150),
            nn.BatchNorm1d(150),    # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),          # 添加Dropout
            
            nn.Linear(150, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            
            nn.Linear(100, 10*10)
        )
        
        # 使用更好的初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.layers(x).view(batch_size, 10, 10)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层部分
        self.conv_layers = nn.Sequential(
            # 输入: [batch, 3, 10, 10]
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 第一层卷积
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 第二层卷积
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 第三层卷积
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # 全连接层部分
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 10 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 10 * 10)
        )
        
    def forward(self, x):
        # x shape: [batch, 3, 10, 10]
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x.view(x.size(0), 10, 10)

# 修改输入数据形状
X_train = X_train.view(-1, 3, 10, 10)
X_test = X_test.view(-1, 3, 10, 10)

# 修改优化器和学习率调度器
# 初始化模型和训练参数
model = CNN()  # 创建模型实例
critersion = nn.MSELoss()  # 使用均方误差损失函数
# 使用Adam优化器，可以通过修改learning_rate等参数调整训练效果
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # 学习率调度器

# 训练参数设置
epochs = 150  # 训练轮数，可以根据需要调整
train_losses1= []  # 记录训练损失
train_losses2= []  # 记录训练损失
test_mses1= []    # 记录测试误差
test_mses2 = []    # 记录测试误差

batch_x = np.arange(0, batch_size)  # 创建批次索引

# 训练前的准备
plt.ion()  # 开启交互模式
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,6))
train_losses1, test_mses1 = [], []
train_losses2, test_mses2 = [], []

# 训练循环
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = critersion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_mse =critersion(y_pred, y_test)

    # 记录损失
    train_losses1.append(loss.item())
    test_mses1.append(test_mse.item())

    # 实时更新图形
    ax1.clear()
    ax1.plot(train_losses1, label='Train Loss')
    ax1.plot(test_mses1, label='Test MSE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss/MSE')
    ax1.legend()
    ax1.set_title('Training Progress (by epoch)')

    ax2.clear()
    ax2.plot(train_losses2, label='Batch Train Loss')
    ax2.plot(test_mses2, label='Batch Test MSE')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss/MSE')
    ax2.legend()
    ax2.set_title('Training Progress (by batch)')

    plt.tight_layout()
    plt.pause(0.1)  # 暂停更新图形

    # 打印训练状态
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test MSE: {test_mse.item():.4f}')


plt.ioff()  # 关闭交互模式
plt.show()  # 保持图形显示