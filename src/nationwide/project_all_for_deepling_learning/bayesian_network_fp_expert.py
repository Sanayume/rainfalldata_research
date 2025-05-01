import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

# 更新pgmpy导入，使用DiscreteBayesianNetwork
try:
    # 尝试使用新版本的pgmpy
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    print("Using pgmpy with DiscreteBayesianNetwork")
except ImportError:
    try:
        # 尝试兼容旧版本
        from pgmpy.models.BayesianModel import BayesianModel
        from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
        from pgmpy.inference import VariableElimination
        DiscreteBayesianNetwork = BayesianModel  # 为旧版本提供兼容
        print("Using pgmpy with BayesianModel")
    except ImportError:
        print("未找到pgmpy库或不支持的版本")
        print("请尝试安装pgmpy: pip install pgmpy")
        print("或者升级: pip install --upgrade pgmpy")
        exit(1)

# --- 配置 ---
PROJECT_DIR = "F:/rainfalldata"
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features_v5.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target_v5.npy")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_v5.txt")
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "bayesian_network_fp_expert_model.pkl")
FP_REDUCTION_PLOT_PATH = os.path.join(PROJECT_DIR, "bayesian_network_fp_reduction.png")

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42

# 由于贝叶斯网络通常难以处理大规模连续数据，我们需要使用更小的子集并考虑离散化
TRAIN_SUBSET_FRACTION = 0.05  # 使用5%的训练数据
# 选择最重要的特征子集，这些特征对减少FP可能更有价值
TOP_FEATURES_FOR_FP = [
    'raw_values_GSMAP',           # GSMAP通常误报率低
    'product_range',              # 产品间差异大可能表示不稳定预测
    'coef_of_variation',         # 变异系数大说明产品不一致
    'rain_product_count',         # 少量产品显示雨而其他不显示时容易误报
    'lag_1_rain_count',           # 前一天无雨但今天多产品显示雨可能是误报
    'window_7_std',               # 近期波动大可能表示不稳定区域
    'spatial_std_GSMAP',          # 空间标准差大说明区域不稳定
    'threshold_proximity'         # 接近阈值的预测更不确定
]

os.makedirs(PROJECT_DIR, exist_ok=True)

# --- 辅助函数：计算性能指标 ---
def calculate_metrics(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    print(f"\n--- {title} Performance ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Accuracy: {accuracy:.4f}, POD: {pod:.4f}, FAR: {far:.4f}, CSI: {csi:.4f}")
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}

# --- 1. 加载数据 ---
print("Loading flattened data (v5)...")
try:
    X_flat = np.load(X_FLAT_PATH)
    Y_flat_raw = np.load(Y_FLAT_PATH)
    print("Successfully loaded full arrays.")
    print(f"Loaded X_flat shape: {X_flat.shape}")
    print(f"Loaded Y_flat_raw shape: {Y_flat_raw.shape}")
except FileNotFoundError:
    print(f"Error: Data files not found. Ensure {X_FLAT_PATH} and {Y_FLAT_PATH} exist.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. 加载特征名称 并 过滤需要的特征 ---
print(f"Loading feature names from {FEATURE_NAMES_PATH}...")
try:
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(feature_names)} feature names.")
    
    # 获取TOP_FEATURES_FOR_FP的索引
    selected_indices = []
    for feature in TOP_FEATURES_FOR_FP:
        found = False
        for i, name in enumerate(feature_names):
            if feature == name:
                selected_indices.append(i)
                found = True
                break
        if not found:
            print(f"Warning: Feature '{feature}' not found in feature names.")
    
    print(f"Selected {len(selected_indices)} features for FP reduction.")
    
except Exception as e:
    print(f"Error loading feature names: {e}. Using default indices.")
    # 如果无法加载特征名称，我们将使用前8个特征（或者可用的）
    selected_indices = list(range(min(8, X_flat.shape[1])))

# --- 3. 预处理和数据分割 ---
print("Preprocessing data...")
y_flat_binary = (Y_flat_raw > RAIN_THRESHOLD).astype(int)
if not isinstance(Y_flat_raw, np.memmap):
    del Y_flat_raw

print("Splitting data into training and testing sets...")
n_samples = X_flat.shape[0]
split_idx = int(n_samples * (1 - TEST_SIZE_RATIO))

X_train_full = X_flat[:split_idx]
y_train_full = y_flat_binary[:split_idx]
X_test = X_flat[split_idx:]
y_test = y_flat_binary[split_idx:]

print(f"Full training set shape: X={X_train_full.shape}, y={y_train_full.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

# 选择特征子集
X_train_full_selected = X_train_full[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

# 创建训练子集
if TRAIN_SUBSET_FRACTION < 1.0:
    print(f"\nUsing a {TRAIN_SUBSET_FRACTION*100:.1f}% subset of the training data...")
    X_train, _, y_train, _ = train_test_split(
        X_train_full_selected, y_train_full,
        train_size=TRAIN_SUBSET_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )
    print(f"Subset training shape: X={X_train.shape}, y={y_train.shape}")
else:
    X_train, y_train = X_train_full_selected, y_train_full

# 释放内存
del X_flat, y_flat_binary, X_train_full, X_train_full_selected

# --- 4. 数据离散化 (贝叶斯网络通常需要离散特征) ---
print("\n--- Discretizing data ---")
n_bins = 5  # 每个特征的箱数
X_train_discrete = np.zeros_like(X_train, dtype=int)
X_test_discrete = np.zeros_like(X_test_selected, dtype=int)

# 为每个特征计算分位数并离散化
for i in range(X_train.shape[1]):
    # 使用训练集上的分位数边界
    bins = np.linspace(0, 100, n_bins+1)[1:-1]  # 分位点（排除0和100）
    quantiles = np.percentile(X_train[:, i], bins)
    
    # 离散化训练集
    X_train_discrete[:, i] = np.digitize(X_train[:, i], quantiles)
    
    # 使用相同的边界离散化测试集
    X_test_discrete[:, i] = np.digitize(X_test_selected[:, i], quantiles)

print(f"Discretized data into {n_bins} bins per feature.")

# --- 5. 准备贝叶斯网络的数据格式 ---
# 创建包含目标变量的DataFrame
feature_cols = [f'F{i}' for i in range(X_train.shape[1])]
train_df = pd.DataFrame(X_train_discrete, columns=feature_cols)
train_df['target'] = y_train

test_df = pd.DataFrame(X_test_discrete, columns=feature_cols)
test_df['target'] = y_test

# --- 6. 构建针对FP的贝叶斯网络结构 ---
print("\n--- Building Bayesian Network structure ---")
# 基于领域知识和假设创建有向图结构
edges = []  # 先创建边列表

# 添加连接目标的边（每个特征都影响目标）
for col in feature_cols:
    edges.append((col, 'target'))

# 添加特征间的边（简化，实际上应基于领域知识或结构学习算法）
# 这里我们假设前面的特征可能影响后面的特征
for i in range(len(feature_cols)-1):
    edges.append((feature_cols[i], feature_cols[i+1]))

# 使用边列表创建离散贝叶斯网络
model = DiscreteBayesianNetwork(edges)

print(f"Created Discrete Bayesian Network with {len(model.nodes())} nodes and {len(model.edges())} edges.")

# --- 7. 训练贝叶斯网络（关注FP减少）---
print("\n--- Training Bayesian Network ---")
start_time = time.time()

# 使用贝叶斯估计器（添加先验知识）
model.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)

end_time = time.time()
print(f"Training complete in {end_time - start_time:.2f} seconds.")

# --- 8. 使用贝叶斯网络进行推断 ---
print("\n--- Making predictions ---")
inference = VariableElimination(model)

# 为了加速，我们在测试集的子集上进行评估
test_subset_size = min(100000, len(test_df))
test_subset_indices = np.random.choice(len(test_df), test_subset_size, replace=False)
test_subset_df = test_df.iloc[test_subset_indices].copy()

# 获取真实标签
y_test_subset = test_subset_df['target'].values

# 使用网络预测概率
print(f"Predicting on {test_subset_size} test samples...")
start_pred = time.time()
y_pred_proba = np.zeros(test_subset_size)

for i in range(test_subset_size):
    if i % 1000 == 0:
        print(f"  Processing sample {i}/{test_subset_size}...")
    
    evidence = {col: test_subset_df.iloc[i][col] for col in feature_cols}
    query_result = inference.query(variables=['target'], evidence=evidence)
    # 获取target=1的概率
    y_pred_proba[i] = query_result.values[1]  # 假设类别1的概率在索引1

end_pred = time.time()
print(f"Prediction complete in {end_pred - start_pred:.2f} seconds.")

# --- 9. 评估不同阈值下的性能，特别关注FP ---
print("\n--- Evaluating performance across thresholds ---")
# 使用多个阈值评估
thresholds_to_evaluate = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
metrics_by_threshold = {}

for threshold in thresholds_to_evaluate:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    metrics = calculate_metrics(y_test_subset, y_pred_threshold, 
                               title=f"Bayesian Network FP Expert (Threshold {threshold:.2f})")
    metrics_by_threshold[threshold] = metrics

# --- 10. 性能对比和可视化 ---
print("\n--- Bayesian Network FP Expert Performance across thresholds ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
threshold_metrics_data = {}

for threshold, metrics in metrics_by_threshold.items():
    threshold_metrics_data[f'BN_FP_{threshold:.2f}'] = {
        metric: metrics.get(metric, float('nan')) for metric in metrics_to_show
    }

threshold_df = pd.DataFrame(threshold_metrics_data).T
threshold_df = threshold_df[metrics_to_show]

float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols: 
    threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols:
    threshold_df[col] = threshold_df[col].map('{:.0f}'.format)

print(threshold_df)

# 绘制FAR-阈值关系图
plt.figure(figsize=(10, 6))
fars = [metrics['far'] for metrics in metrics_by_threshold.values()]
plt.plot(thresholds_to_evaluate, fars, 'o-', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('False Alarm Ratio (FAR)')
plt.title('Bayesian Network FP Expert: FAR vs Threshold')
plt.grid(True)
plt.savefig(FP_REDUCTION_PLOT_PATH)
print(f"FAR reduction plot saved to {FP_REDUCTION_PLOT_PATH}")

# --- 11. 尝试保存模型 ---
try:
    import pickle
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nBayesian Network FP Expert analysis complete.")
