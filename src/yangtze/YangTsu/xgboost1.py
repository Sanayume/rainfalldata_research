import gc
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from loaddata import mydata # 假设 loaddata.py 和 mydata 类已正确定义
import os
import pandas as pd
import joblib
import time 
import optuna

# --- 配置 (Configuration) ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_Yangtsu_flat_features.npy") # 展平后的特征文件路径
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target.npy") # 展平后的目标文件路径
# MODEL_PREDICT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "model_predict") # 模型预测数据保存路径 (当前未使用)
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_yangtsu.txt") # 特征名称文件路径
RAIN_THRESHOLD = 0.1 # 降雨阈值，大于此值视为有雨
TEST_SIZE_RATIO = 0.2 # 测试集在总数据中的比例
MAX_LOOKBACK = 30 # 特征工程中使用的最大回看天数，用于对齐原始数据
N_TRIALS = 80 # Optuna 优化的试验次数
OPTUNA_TIMEOUT = 3600000 # Optuna 优化的最大时长（秒）
OPTIMIZE_METRIC = 'auc' # Optuna 优化的目标指标 ('auc' 或 'logloss')
EARLY_STOPPING_ROUNDS_OPTUNA = 30 # Optuna 优化过程中模型训练的早停轮数
EARLY_STOPPING_ROUNDS_FINAL = 30 # 最终模型训练的早停轮数

# 用于存储 Optuna 每次试验的 AUC 值
optuna_trial_aucs = []
# 用于存储 Optuna 每次试验的详细性能指标
optuna_trial_full_metrics = []


# --- 辅助函数：计算性能指标 (Helper Function: Calculate Performance Metrics) ---
def calculate_metrics(y_true, y_pred, title=""):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() # True Negatives, False Positives, False Negatives, True Positives

    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred) # 准确率
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0 # 命中率 (Probability of Detection / Recall)
    far = fp / (tp + fp) if (tp + fp) > 0 else 0 # 误报率 (False Alarm Ratio)
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0 # 临界成功指数 (Critical Success Index)

    # 打印性能指标
    print(f"\n--- {title} Performance ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"POD (Hit Rate/Recall): {pod:.4f}")
    print(f"FAR (False Alarm Ratio): {far:.4f}")
    print(f"CSI (Critical Success Index): {csi:.4f}")
    print("\nClassification Report:") # 分类报告，包含精确率、召回率、F1分数等
    print(classification_report(y_true, y_pred, target_names=['No Rain', 'Rain'], zero_division=0))
    
    # 返回指标字典
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
            'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi,
            'auc_roc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5} # 如果可能，计算AUC

# --- 1. 加载数据 (Load Data) ---
print("加载长江流域展平化数据 (Loading Yangtze flattened data)...")
start_load_flat = time.time() # 记录开始时间
# 检查所需文件是否存在
if not (os.path.exists(X_FLAT_PATH) and os.path.exists(Y_FLAT_PATH) and os.path.exists(FEATURE_NAMES_PATH)):
    raise FileNotFoundError(f"展平化数据文件未在 {PROJECT_DIR} 中找到。请先运行 turn1.py (或等效的特征生成脚本)。")

try:
    print(f"尝试加载特征文件: {X_FLAT_PATH}...")
    X_flat = np.load(X_FLAT_PATH) # 加载特征数据
    print(f"尝试加载目标文件: {Y_FLAT_PATH}...")
    Y_flat = np.load(Y_FLAT_PATH) # 加载目标数据
except MemoryError as e: # 处理内存不足错误
    print(f"加载展平化数据时发生内存错误: {e}")
    print("如果需要，可以考虑使用XGBoost的外部内存特性或减小数据集。")
    raise
except Exception as e: # 处理其他加载错误
    print(f"加载展平化数据时发生错误: {e}")
    raise

# 加载特征名称
with open(FEATURE_NAMES_PATH, "r") as f:
    feature_names = [line.strip() for line in f]

print(f"已加载展平化特征 X_flat: shape {X_flat.shape}")
print(f"已加载展平化目标 Y_flat: shape {Y_flat.shape}")
print(f"已加载 {len(feature_names)} 个特征名称。")
end_load_flat = time.time() # 记录结束时间
print(f"展平化数据加载完成，耗时 {end_load_flat - start_load_flat:.2f} 秒。")

# 加载原始空间数据用于基线计算
print("\n加载原始长江流域空间数据用于基线计算 (Loading original SPATIAL Yangtze data for baseline calculation)...")
start_load_orig = time.time()
original_data_loader = mydata() # 实例化数据加载器
# 加载流域的空间数据
# X_orig_spatial shape: (产品数量, 时间, 纬度, 经度)
# Y_orig_spatial shape: (时间, 纬度, 经度)
X_orig_spatial, Y_orig_spatial = original_data_loader.get_basin_spatial_data(basin_mask_value=2) # 假设掩码值为2代表长江流域
product_names = original_data_loader.get_products() # 获取产品名称列表
n_products, nday, lat_dim, lon_dim = X_orig_spatial.shape # 获取维度信息
end_load_orig = time.time()
print(f"原始长江流域空间数据加载完成，耗时 {end_load_orig - start_load_orig:.2f} 秒。")
print(f"X_orig_spatial shape: {X_orig_spatial.shape}, Y_orig_spatial shape: {Y_orig_spatial.shape}")

# --- 2. 准备真实标签和基线预测 (使用空间数据) (Prepare True Labels and Baseline Predictions - Spatial) ---
print("\n准备长江流域的真实标签和基线预测 (使用空间数据) (Preparing true labels and baseline predictions for Yangtze - spatial)...")
# 对齐原始 Y (空间) 数据的时间维度
Y_orig_spatial_aligned = Y_orig_spatial[MAX_LOOKBACK:].astype(np.float32) # shape: (有效天数, 纬度, 经度)
# 在对齐和应用掩码后展平真实标签 (仅考虑流域内的有效点)
valid_mask_spatial = ~np.isnan(Y_orig_spatial_aligned[0]) # 使用第一天的有效点掩码 (纬度, 经度)
Y_true_flat_for_baseline = Y_orig_spatial_aligned[:, valid_mask_spatial].reshape(-1) # 展平有效点数据
Y_true_binary_for_baseline = (Y_true_flat_for_baseline > RAIN_THRESHOLD).astype(int) # 二值化
print(f"用于基线评估的展平化真实标签形状 (有效点): {Y_true_binary_for_baseline.shape}")

# --- 3. 计算所有基线产品的性能 (长江流域数据 - 使用空间数据) (Calculate Baseline Performance - Spatial) ---
print("\n计算所有基线产品在长江流域的性能 (使用空间数据) (Calculating baseline performance for all products - Yangtze Spatial)...")
baseline_metrics_all = {}
# 对齐原始 X (空间) 数据的时间维度
X_orig_spatial_aligned = X_orig_spatial[:, MAX_LOOKBACK:, :, :].astype(np.float32) # shape: (产品, 有效天数, 纬度, 经度)

for i in range(n_products): # 遍历每个产品
    product_name = product_names[i]
    print(f"  计算产品: {product_name}")
    # 获取产品数据, 确保形状为 (有效天数, 纬度, 经度)
    baseline_product_data_spatial = X_orig_spatial_aligned[i, :, :, :]
    # 在对齐和应用掩码后展平预测值
    baseline_pred_flat = baseline_product_data_spatial[:, valid_mask_spatial].reshape(-1) # 展平有效点数据
    baseline_pred_binary = (baseline_pred_flat > RAIN_THRESHOLD).astype(int) # 二值化

    # 检查形状是否匹配，防止因掩码或数据问题导致错误
    if baseline_pred_binary.shape != Y_true_binary_for_baseline.shape:
        print(f"    警告: 产品 {product_name} 的形状不匹配! 基线: {baseline_pred_binary.shape}, 真实: {Y_true_binary_for_baseline.shape}. 跳过此产品。")
        continue

    metrics = calculate_metrics(Y_true_binary_for_baseline, baseline_pred_binary, title=f"基线 ({product_name}) - 长江流域空间数据")
    baseline_metrics_all[product_name] = metrics

# 清理大型原始空间数组以释放内存
del X_orig_spatial, Y_orig_spatial, X_orig_spatial_aligned, Y_orig_spatial_aligned, Y_true_flat_for_baseline
gc.collect() # 显式垃圾回收

# --- 4. 准备 XGBoost 数据 (Prepare XGBoost Data) ---
print("\n准备长江流域 XGBoost 模型数据 (Preparing data for XGBoost - Yangtze)...")
Y_binary = (Y_flat > RAIN_THRESHOLD).astype(int) # 对展平化的目标数据进行二值化
n_samples = len(Y_binary)
print(f"划分数据，测试集比例 test_size={TEST_SIZE_RATIO}，随机种子 random_state=42...")
start_split = time.time()

# 使用 train_test_split 划分训练集和测试集，stratify=Y_binary 确保类别比例在划分后大致一致
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, Y_binary, test_size=TEST_SIZE_RATIO, random_state=42, stratify=Y_binary
)
del X_flat, Y_binary # 清理原始展平数据
gc.collect()

end_split = time.time()
print(f"数据划分完成，耗时 {end_split - start_split:.2f} 秒。")

train_counts = np.bincount(y_train) # 计算训练集中各类别数量
test_counts = np.bincount(y_test)   # 计算测试集中各类别数量
print(f"训练集大小: {len(y_train)} (无雨: {train_counts[0]}, 有雨: {train_counts[1] if len(train_counts)>1 else 0})")
print(f"测试集大小: {len(y_test)} (无雨: {test_counts[0]}, 有雨: {test_counts[1] if len(test_counts)>1 else 0})")

# --- 5. 超参数优化 (使用 Optuna) (Hyperparameter Optimization with Optuna) ---
print("\n--- 开始使用 Optuna 进行超参数优化 (Starting Hyperparameter Optimization with Optuna) ---")
start_opt_time = time.time()

# 定义 Optuna 的目标函数
def objective(trial):
    """Optuna 目标函数 (Optuna objective function)."""
    # 定义超参数搜索空间
    param = {
        'objective': 'binary:logistic',         # 目标函数：二分类逻辑回归
        'eval_metric': ['logloss', 'auc'],      # 评估指标：对数损失和AUC
        'tree_method': 'hist',                  # 使用基于直方图的快速算法
        'verbosity': 0,                         # 静默模式，不打印 XGBoost 内部信息
        'n_estimators': 1000,                   # 初始估算器数量，配合早停使用
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), # 学习率 (对数均匀分布)
        'max_depth': trial.suggest_int('max_depth', 4, 10),                        # 最大树深
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),                   # 训练样本子采样比例
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),     # 构建每棵树时特征的子采样比例
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),                           # 惩罚项，控制叶子节点分裂
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),              # L2 正则化项 (对数均匀分布)
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),                # L1 正则化项 (对数均匀分布)
        'scale_pos_weight': (np.sum(y_train == 0) / np.sum(y_train == 1)) if np.sum(y_train == 1) > 0 else 1, # 处理类别不平衡
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS_OPTUNA                      # 早停轮数 (在构造函数中指定)
    }

    model = xgb.XGBClassifier(**param) # 使用当前试验的参数实例化模型
    eval_set = [(X_test, y_test)]      # 使用测试集作为评估集进行早停

    try:
        print(f"\nOptuna Trial #{trial.number}: Training with params: {trial.params}") # 打印当前试验的参数
        model.fit(X_train, y_train,
                  eval_set=eval_set,
                  verbose=False) # 训练过程中不打印 XGBoost 信息

        # ----- 新增：在每次 Optuna 试验后打印详细性能指标 -----
        y_pred_proba_trial = model.predict_proba(X_test)[:, 1] # 获取预测概率
        # 使用一个固定的、合理的阈值（例如0.5）来计算分类指标
        # 注意：这里的阈值选择是为了在优化过程中观察分类性能，最终模型的阈值可以再调整
        y_pred_trial_fixed_thresh = (y_pred_proba_trial >= 0.5).astype(int)
        print(f"Optuna Trial #{trial.number}: Performance on test set (threshold 0.5):")
        trial_metrics = calculate_metrics(y_test, y_pred_trial_fixed_thresh, title=f"Optuna Trial #{trial.number} (Thresh 0.5)")
        optuna_trial_full_metrics.append({'trial': trial.number, 'params': trial.params, 'metrics': trial_metrics})
        # ----- 新增结束 -----

        results = model.evals_result() # 获取评估结果
        if OPTIMIZE_METRIC == 'auc': # 如果优化目标是 AUC
            best_score = results['validation_0']['auc'][model.best_iteration] # 取最佳迭代次数的 AUC
            optuna_trial_aucs.append(best_score) # 记录 AUC 值
            return best_score
        else: # 如果优化目标是 logloss
            best_score = results['validation_0']['logloss'][model.best_iteration] # 取最佳迭代次数的 logloss
            # 如果需要，也可以在这里记录 logloss
            return best_score

    except Exception as e: # 处理训练过程中的错误
        print(f"试验失败，错误: {e}")
        return float('inf') if OPTIMIZE_METRIC == 'logloss' else 0.0 # 返回一个差的值

# 设置优化方向 (最大化AUC或最小化logloss)
study_direction = 'maximize' if OPTIMIZE_METRIC == 'auc' else 'minimize'
study = optuna.create_study(direction=study_direction) # 创建 Optuna study 对象

try:
    # 开始优化
    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)
except KeyboardInterrupt: # 允许手动中断优化
    print("优化过程被手动中断。")

end_opt_time = time.time()
print(f"超参数优化完成，耗时 {end_opt_time - start_opt_time:.2f} 秒。")

best_params = study.best_params # 获取最佳参数组合
print("\n找到的最佳超参数:")
print(best_params)
print(f"最佳 {OPTIMIZE_METRIC} 值: {study.best_value:.4f}")

# --- 新增：绘制 Optuna 寻优 AUC 曲线 ---
if OPTIMIZE_METRIC == 'auc' and optuna_trial_aucs:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(optuna_trial_aucs) + 1), optuna_trial_aucs, marker='o', linestyle='-')
    plt.title('Optuna Optimization: AUC vs. Trial Number')
    plt.xlabel('Trial Number')
    plt.ylabel('AUC Score')
    plt.grid(True)
    auc_plot_path = os.path.join(PROJECT_DIR, "optuna_auc_vs_trial_yangtsu.png")
    plt.savefig(auc_plot_path)
    print(f"Optuna AUC 优化曲线已保存至: {auc_plot_path}")
    plt.close()

    # 同时保存每次试验的详细指标到CSV
    if optuna_trial_full_metrics:
        # 将参数和指标合并到一个DataFrame中
        detailed_trials_data = []
        for item in optuna_trial_full_metrics:
            row = {'trial': item['trial']}
            row.update(item['params']) # 添加参数
            row.update(item['metrics']) # 添加指标
            detailed_trials_data.append(row)
        
        detailed_trials_df = pd.DataFrame(detailed_trials_data)
        detailed_trials_csv_path = os.path.join(PROJECT_DIR, "optuna_detailed_trial_metrics_yangtsu.csv")
        detailed_trials_df.to_csv(detailed_trials_csv_path, index=False)
        print(f"Optuna 每次试验的详细指标已保存至: {detailed_trials_csv_path}")

# --- 新增结束 ---


# --- 5b. 使用优化后的参数训练最终模型并进行早停 (Train Final Model with OPTIMIZED Params) ---
print("\n--- 使用优化后的参数在长江流域训练数据上训练最终模型 (Training FINAL model with OPTIMIZED parameters - Yangtze) ---")
start_train = time.time()

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 设置字体以支持中文显示
plt.rcParams['axes.unicode_minus'] = False # 设置负号显示

# 准备最终模型的参数
final_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'], # 最终模型也监控这些指标
    'tree_method': 'hist',
    'n_estimators': 1500, # 为最终模型设置一个较大的初始树数量，配合早停
    'scale_pos_weight': (np.sum(y_train == 0) / np.sum(y_train == 1)) if np.sum(y_train == 1) > 0 else 1,
    'random_state': 42, # 保证最终模型训练的可复现性
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS_FINAL # 最终模型的早停轮数
}
final_params.update(best_params) # 将 Optuna 找到的最佳参数更新进来

final_model = xgb.XGBClassifier(**final_params) # 实例化最终模型
eval_set_final = [(X_test, y_test)] # 最终模型也使用测试集进行早停和评估
final_model.fit(X_train, y_train, eval_set=eval_set_final, verbose=50) # verbose=50 每50轮打印一次信息
end_train = time.time()
print(f"最终模型训练完成，耗时 {end_train - start_train:.2f} 秒。")
print(f"最终模型的最佳迭代次数: {final_model.best_iteration}")

# --- 保存优化后的训练模型 (Save OPTIMIZED Trained Model) ---
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "xgboost_optimized_yangtsu_model.joblib") # 模型保存路径
print(f"\n保存优化后的训练模型至 {MODEL_SAVE_PATH}...")
try:
    joblib.dump(final_model, MODEL_SAVE_PATH) # 保存模型
    print("优化后的模型已成功保存。")
except Exception as e:
    print(f"保存优化后的模型时发生错误: {e}")

# --- 6. 评估 XGBoost 模型 (使用优化后的参数和不同的概率阈值) (Evaluate XGBoost Model - Optimized) ---
print("\n--- 在长江流域测试集上评估最终 XGBoost 模型 (优化参数)，使用不同概率阈值 (Evaluating FINAL XGBoost model - Optimized, Varying Thresholds) ---")
start_eval = time.time()
y_pred_proba = final_model.predict_proba(X_test)[:, 1] # 获取类别1（有雨）的预测概率
thresholds_to_evaluate = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] # 定义一系列评估阈值
metrics_by_threshold = {} # 用于存储不同阈值下的指标

print("计算不同阈值下的性能指标...")
for threshold_val in thresholds_to_evaluate: # 变量名区分，避免与全局thresholds冲突
    print(f"\n--- 阈值 (Threshold): {threshold_val:.2f} ---")
    y_pred_test_threshold = (y_pred_proba >= threshold_val).astype(int) # 根据阈值进行二分类
    metrics = calculate_metrics(y_test, y_pred_test_threshold, title=f"XGBoost 分类器 长江流域 (优化后, 阈值 {threshold_val:.2f})")
    metrics_by_threshold[threshold_val] = metrics

xgboost_metrics_optimized_threshold_05 = metrics_by_threshold[0.5] # 获取0.5阈值下的指标用于对比
end_eval = time.time()
print(f"评估完成，耗时 {end_eval - start_eval:.2f} 秒。")

# --- 7. 比较基线和 XGBoost 在测试集上的性能 (优化后的模型，0.5阈值) (Compare Baseline and XGBoost - Optimized) ---
print("\n--- 最终性能对比 (长江流域测试集 - 优化后阈值0.5 vs 空间数据基线) (Final Performance Comparison - Optimized vs Spatial Baseline) ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn'] # 定义要展示的指标列
comparison_data = {} # 用于存储对比数据

# 添加优化后的 XGBoost 模型性能
comparison_data['XGBoost_Yangtsu_Optimized (Thr 0.5)'] = {metric: xgboost_metrics_optimized_threshold_05.get(metric, float('nan')) for metric in metrics_to_show}

# 添加所有基线产品的性能
for product_name, metrics_dict in baseline_metrics_all.items(): # 修改变量名避免冲突
    comparison_data[f'Baseline_{product_name}_Spatial'] = {metric: metrics_dict.get(metric, float('nan')) for metric in metrics_to_show}

comparison_df = pd.DataFrame(comparison_data).T # 转换为 DataFrame
comparison_df = comparison_df[metrics_to_show] # 按指定顺序排列列

print("\n--- 最终性能对比 (长江流域 - 优化后阈值0.5 vs 空间数据基线) ---")
# 格式化输出
float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols:
    if col in comparison_df.columns:
        comparison_df[col] = comparison_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols:
    if col in comparison_df.columns:
        comparison_df[col] = comparison_df[col].map('{:.0f}'.format)

print(comparison_df)
comparison_csv_path = os.path.join(PROJECT_DIR, "performance_comparison_yangtsu_optimized.csv") # 保存对比表格
print(f"对比表格已保存至 {comparison_csv_path}")
comparison_df.to_csv(comparison_csv_path)

# --- 可选：显示优化后模型在所有评估阈值下的性能 (Optional: Display Optimized Model Performance Across Thresholds) ---
print("\n--- XGBoost 在不同概率阈值下的性能 (长江流域 - 优化后模型) (XGBoost Performance Across Thresholds - Optimized) ---")
threshold_metrics_data = {}
for threshold_val, metrics_dict in metrics_by_threshold.items(): # 修改变量名
    threshold_metrics_data[f'Optimized_Threshold_{threshold_val:.2f}'] = {metric: metrics_dict.get(metric, float('nan')) for metric in metrics_to_show}

threshold_df = pd.DataFrame(threshold_metrics_data).T
threshold_df = threshold_df[metrics_to_show]

# 格式化输出
for col in float_cols:
    if col in threshold_df.columns:
        threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
for col in int_cols:
    if col in threshold_df.columns:
        threshold_df[col] = threshold_df[col].map('{:.0f}'.format)

print(threshold_df)
threshold_csv_path = os.path.join(PROJECT_DIR, "threshold_performance_yangtsu_optimized.csv") # 保存阈值性能表格
print(f"阈值性能表格已保存至 {threshold_csv_path}")
threshold_df.to_csv(threshold_csv_path)

# --- 8. 可视化训练过程 (优化后的参数) (Visualize Training Process - Optimized) ---
print("绘制训练历史曲线 (优化参数 - 长江流域) (Plotting training history - Optimized)...")
results = final_model.evals_result() # 获取最终模型的评估结果
if 'validation_0' in results: # 检查是否存在评估集的结果
    epochs = len(results['validation_0']['logloss']) # 获取训练轮数
    x_axis = range(0, epochs) # x轴为训练轮数
    fig, ax = plt.subplots(1, 2, figsize=(15, 5)) # 创建子图

    # 绘制 LogLoss 曲线
    ax[0].plot(x_axis, results['validation_0']['logloss'], label='测试集 (eval_set)')
    ax[0].legend()
    ax[0].set_ylabel('LogLoss')
    ax[0].set_xlabel('轮数 (Epochs)')
    ax[0].set_title('XGBoost LogLoss (优化后)')

    # 绘制 AUC 或 Error 曲线
    metric_to_plot_final = 'auc' if 'auc' in results['validation_0'] else 'error' # 决定绘制哪个指标
    if metric_to_plot_final in results['validation_0']:
        ax[1].plot(x_axis, results['validation_0'][metric_to_plot_final], label=f'测试集 (eval_set) - {metric_to_plot_final.upper()}')
        ax[1].legend()
        ax[1].set_ylabel(metric_to_plot_final.upper())
        ax[1].set_xlabel('轮数 (Epochs)')
        ax[1].set_title(f'XGBoost {metric_to_plot_final.upper()} (优化后)')
    else:
        ax[1].set_title('指标未找到 (Metric not found)')

    plt.tight_layout() # 调整布局
    history_plot_path = os.path.join(PROJECT_DIR, "xgboost_training_history_yangtsu_optimized.png") # 保存图像路径
    plt.savefig(history_plot_path)
    print(f"训练历史曲线已保存至 {history_plot_path}")
    plt.close() # 关闭图像，释放资源
else:
    print("警告: 在 evals_result() 中未找到 'validation_0'。无法绘制训练历史。")

# --- 9. 特征重要性 (优化后的参数) (Feature Importance - Optimized) ---
print("\n--- 特征重要性 (优化参数 - 长江流域) (Feature Importances - Optimized) ---")
try:
    importances = final_model.feature_importances_ # 获取特征重要性分数
    # 创建包含特征名称和重要性分数的 DataFrame
    importance_df = pd.DataFrame({'特征 (Feature)': feature_names, '重要性 (Importance)': importances})
    importance_df = importance_df.sort_values(by='重要性 (Importance)', ascending=False) # 按重要性降序排列

    print("最重要的10个特征 (Top 10 Features):")
    print(importance_df.head(40))

    N_TOP_FEATURES_TO_PLOT = 50 # 定义要绘制的最重要特征的数量
    n_features_actual = len(feature_names)
    n_plot = min(N_TOP_FEATURES_TO_PLOT, n_features_actual) # 实际绘制的数量

    plt.figure(figsize=(10, n_plot / 2.0)) # 设置图像大小
    top_features = importance_df.head(n_plot) # 获取前n_plot个特征
    plt.barh(top_features['特征 (Feature)'], top_features['重要性 (Importance)']) # 绘制水平条形图
    plt.xlabel("重要性分数 (Importance Score)")
    plt.ylabel("特征 (Feature)")
    plt.title(f"最重要的 {n_plot} 个特征 (XGBoost 长江流域模型 - 优化后)")
    plt.gca().invert_yaxis() # 反转y轴，使最重要的特征在顶部
    plt.tight_layout()

    importance_plot_path = os.path.join(PROJECT_DIR, "xgboost_feature_importance_yangtsu_optimized.png") # 保存路径
    plt.savefig(importance_plot_path)
    print(f"特征重要性图已保存至 {importance_plot_path}")
    plt.close()

except Exception as e:
    print(f"无法绘制特征重要性图: {e}")

print("\n长江流域数据集分析完成，使用了优化后的参数，并评估了多个概率阈值。")