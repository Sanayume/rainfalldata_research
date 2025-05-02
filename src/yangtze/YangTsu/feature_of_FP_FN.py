from loaddata import mydata
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data = mydata()

X_raw_spatial, Y_raw_spatial, X_raw_yangtsu, Y_raw_yangtsu = data.yangtsu()
print(f"X_raw_spatial shape: {X_raw_spatial.shape}")
print(f"Y_raw_spatial shape: {Y_raw_spatial.shape}")
print(f"X_raw_yangtsu shape: {X_raw_yangtsu.shape}")
print(f"Y_raw_yangtsu shape: {Y_raw_yangtsu.shape}")
def calculate_POD(X, Y, threshold):
    X = np.where(X <= threshold, X, np.nan)
    mask = np.isnan(X)
    Y = np.where(~mask, Y, np.nan)

    X_is_rain = np.where(X > 0, 1, 0)   
    Y_is_rain = np.where(Y > 0, 1, 0)

    tp = np.sum(X_is_rain * Y_is_rain)
    fp = np.sum(X_is_rain * (1 - Y_is_rain))
    fn = np.sum((1 - X_is_rain) * Y_is_rain)
    tn = np.sum((1 - X_is_rain) * (1 - Y_is_rain))
    print(f"tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}, threshold: {threshold}")
    return tp, fp, fn, tn

thresholds = np.arange(0, 10, 1)
PODs = []
FARs = []
CSIs = []
MistakeRates = []

best_mistake_rate = -1.0
best_threshold_for_mistake = -1.0

for threshold in thresholds:
    tp, fp, fn, tn = calculate_POD(X_raw_yangtsu[0, :, :, :], Y_raw_yangtsu, threshold)
    total_in_subset = tp + fp + fn + tn

    if total_in_subset > 0:
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # FPR
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        mistake_rate = (fp + fn) / total_in_subset # 计算问题率
    else:
        pod = np.nan
        far_fpr = np.nan
        csi = np.nan
        mistake_rate = np.nan # 如果子集为空，则为 NaN

    PODs.append(pod)
    FARs.append(far_fpr)
    CSIs.append(csi)
    MistakeRates.append(mistake_rate)

    # 追踪最大问题率对应的阈值
    if not np.isnan(mistake_rate) and mistake_rate > best_mistake_rate:
        best_mistake_rate = mistake_rate
        best_threshold_for_mistake = threshold

# 输出结果
print(f"\n分析完成。")
print(f"找到的最佳指示性阈值 (最大问题率) T = {best_threshold_for_mistake}")
print(f"在该阈值下，子集(X <= T)内的问题率 P(误报或漏报|X <= T) = {best_mistake_rate:.4f}")

# 绘图 (可以额外绘制 MistakeRates)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, PODs, label="POD (Subset)")
plt.plot(thresholds, FARs, label="FPR (Subset)") # 明确是 FPR
plt.plot(thresholds, CSIs, label="CSI (Subset)")
plt.plot(thresholds, MistakeRates, label="Mistake Rate (Subset)", linestyle='--', color='black') # 绘制问题率
plt.axvline(x=best_threshold_for_mistake, color='r', linestyle=':', label=f'T = {best_threshold_for_mistake} (Max Mistake Rate)') # 标记找到的阈值

plt.xlabel("指示性阈值 T")
plt.ylabel("指标值 (在 X <= T 子集内计算)")
plt.title("指标随指示性阈值 T 的变化")
plt.legend()
plt.grid(True)
plt.show()

# 后续你可以使用 best_threshold_for_mistake 来创建你的二值特征矩阵
# feature_is_risky = np.where(X_product <= best_threshold_for_mistake, 1, 0)

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # 导入 matplotlib 日期处理模块
import numpy as np # 确保 numpy 可用

# 5年 1827天 (2016-01-01 到 2020-12-31)
# 2016 (闰年 366), 2017 (平年 365), 2018 (平年 365), 2019 (平年 365), 2020 (闰年 366)
# 总天数 = 366 + 365 + 365 + 365 + 366 = 1827 天
start_date = datetime.date(2016, 1, 1)
num_days = 1827
# 生成日期列表，用于 x 轴
dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]

# 验证 (可选) - 这部分可以保留或移除，不影响绘图
# print(f"Start date: {start_date}")
# print(f"End date: {dates[-1]}")
# print(f"Number of dates: {len(dates)}")

# 假设 threshold 变量已在前面的代码块中定义 (例如，来自 thresholds 循环的最后一个值)
# 如果不确定，需要在此处显式设置一个值，例如：
# threshold = 9 # 使用上一个循环的最后一个阈值作为示例

PODs = []
FARs = []
CSIs = []
MistakeRates = []
print(f"Calculating daily metrics using threshold = {threshold}...") # 提示正在使用的阈值
for day in range(num_days):
    # 确保 X_raw_yangtsu 和 Y_raw_yangtsu 在此作用域内可用
    # 假设 X_raw_yangtsu 的形状是 (n_products, time, lat, lon)
    # 假设 Y_raw_yangtsu 的形状是 (time, lat, lon)
    # 假设 calculate_POD 函数已定义
    X = X_raw_yangtsu[0, day, :, :] # 使用第一个产品的数据
    Y = Y_raw_yangtsu[day, :, :]

    X_is_rain = np.where(X > 0, 1, 0)
    Y_is_rain = np.where(Y > 0, 1, 0)
    # 调用之前的函数计算 tp, fp, fn, tn
    tp = np.sum(X_is_rain * Y_is_rain)
    fp = np.sum(X_is_rain * (1 - Y_is_rain))
    fn = np.sum((1 - X_is_rain) * Y_is_rain)
    tn = np.sum((1 - X_is_rain) * (1 - Y_is_rain))

    # 计算指标，添加分母为零的检查
    pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    # FAR 通常定义为 fp / (fp + tp)，但这里遵循原始代码 fp / (fp + tn)
    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan

    PODs.append(pod)
    FARs.append(far)
    CSIs.append(csi)

print("Finished calculating daily metrics.")

# --- 使用 matplotlib.dates 绘制图形 ---
print("Plotting daily metrics...")
fig, ax = plt.subplots(figsize=(18, 6)) # 创建图形和坐标轴对象，调整图形大小

ax.plot(dates, PODs, label="POD", linewidth=1)
ax.plot(dates, FARs, label="FAR", linewidth=1)
ax.plot(dates, CSIs, label="CSI", linewidth=1)

# --- 格式化 x 轴 ---
# 设置主刻度定位器为年份
ax.xaxis.set_major_locator(mdates.YearLocator())
# 设置主刻度格式为 'YYYY'
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 设置次刻度定位器为月份
ax.xaxis.set_minor_locator(mdates.MonthLocator())
# 次刻度不显示标签，只显示刻度线（默认行为）

# 自动调整日期标签以避免重叠
fig.autofmt_xdate(rotation=45) # 旋转标签以便更好地显示

# 添加标签、标题和图例
ax.set_xlabel("日期 (年/月)")
ax.set_ylabel("指标值")
ax.set_title(f"每日 POD, FAR, CSI (阈值={threshold})")
ax.legend()
ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray') # 主网格
ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray') # 次网格

plt.tight_layout() # 调整布局防止标签重叠
plt.show()
print("Plot displayed.")

#空间特征
#年平均每个格点的误报命中和临界成功指数

X = X_raw_yangtsu[0, :, :, :]
Y = Y_raw_yangtsu

X_is_rain = np.where(X > 0, 1, 0)
Y_is_rain = np.where(Y > 0, 1, 0)

tp = X_is_rain * Y_is_rain
fp = X_is_rain * (1 - Y_is_rain)
fn = (1 - X_is_rain) * Y_is_rain
tn = (1 - X_is_rain) * (1 - Y_is_rain)

print(f"tp: {tp.shape}")
print(f"fp: {fp.shape}")
print(f"fn: {fn.shape}")
print(f"tn: {tn.shape}")

# 计算评估指标
with np.errstate(divide='ignore', invalid='ignore'):  # 忽略除以零的警告
    POD = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn)!=0)
    FAR = np.divide(fp, (fp + tp), out=np.zeros_like(fp, dtype=float), where=(fp + tp)!=0)  # 修正FAR计算公式
    CSI = np.divide(tp, (tp + fp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fp + fn)!=0)

# 打印形状信息
print(f"POD: {POD.shape}")
print(f"FAR: {FAR.shape}")
print(f"CSI: {CSI.shape}")

# 计算平均值并打印
POD = np.nanmean(POD, axis=0)
FAR = np.nanmean(FAR, axis=0)
CSI = np.nanmean(CSI, axis=0)


# 创建子图布局，使用更专业的配色方案
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# 设置颜色映射和标题
metrics = [POD, FAR, CSI]
titles = ['命中率 (POD)', '虚警率 (FAR)', '临界成功指数 (CSI)']
cmaps = ['Blues', 'Reds', 'Greens']
vmin_max = [(0, 1), (0, 1), (0, 1)]  # 统一色标范围

# 绘制三个指标的空间分布
for i, (metric, title, cmap, (vmin, vmax)) in enumerate(zip(metrics, titles, cmaps, vmin_max)):
    im = axes[i].imshow(metric, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[i].set_title(title, fontsize=14, fontweight='bold')
    axes[i].set_xlabel('经度', fontsize=12)
    if i == 0:
        axes[i].set_ylabel('纬度', fontsize=12)
    
    # 添加色标
    cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    cbar.set_label(title.split(' ')[0], fontsize=10)
    
    # 添加网格线
    axes[i].grid(False)  # 移除默认网格，使图像更清晰

# 添加总标题
plt.suptitle('降水预测评估指标的空间分布', fontsize=16, y=1.05)

# 显示图像
plt.show()

# 额外添加一个组合视图，展示POD和FAR的叠加效果
plt.figure(figsize=(10, 8))

# 创建自定义的颜色映射
plt.imshow(POD, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
plt.imshow(FAR, cmap='Reds', alpha=0.5, vmin=0, vmax=1)

# 添加色标
cbar_pod = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='Blues'), 
                        fraction=0.046, pad=0.04, location='right')
cbar_pod.set_label('命中率 (POD)', fontsize=12)

cbar_far = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='Reds'), 
                        fraction=0.046, pad=0.15, location='right')
cbar_far.set_label('虚警率 (FAR)', fontsize=12)

plt.title('命中率与虚警率的空间分布对比', fontsize=14, fontweight='bold')
plt.xlabel('经度', fontsize=12)
plt.ylabel('纬度', fontsize=12)
plt.grid(False)
plt.show()












