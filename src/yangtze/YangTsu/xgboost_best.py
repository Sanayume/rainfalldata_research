import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime # For timestamping outputs


# --- 配置 ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"--- 本次运行时间戳 (Current Run Timestamp): {TIMESTAMP} ---")

# 定位项目根目录
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))
    print(f"Warning: __file__ not defined. Assuming project root: {PROJECT_ROOT}")

BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "yangtze")
FEATURES_INPUT_DIR = os.path.join(BASE_RESULTS_DIR, "features") # Input features directory

# 定义输出子目录 (Output subdirectories for V1 optimized results)
# Suffix for output directories and files, reflecting V1 data and optimized params
OUTPUT_SUFFIX = f"v1_opt_xgboost_yangtsu_{TIMESTAMP}"

MODELS_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "models", OUTPUT_SUFFIX)
PLOTS_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "plots", OUTPUT_SUFFIX)
PREDICTIONS_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "predictions", OUTPUT_SUFFIX)
PERFORMANCE_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "performance_reports", OUTPUT_SUFFIX)

# 使用长江流域 V1 数据文件 (Using Yangtze V1 data files)
X_FLAT_PATH = os.path.join(FEATURES_INPUT_DIR, "X_Yangtsu_flat_features.npy") # V1 features
Y_FLAT_PATH = os.path.join(FEATURES_INPUT_DIR, "Y_Yangtsu_flat_target.npy") # V1 target
FEATURE_NAMES_PATH = os.path.join(FEATURES_INPUT_DIR, "feature_names_yangtsu.txt") # V1 feature names

# 更新输出文件路径 (Updated output file paths)
MODEL_PREDICTION_PATH = os.path.join(PREDICTIONS_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_test_proba_predictions.npy")
MODEL_SAVE_PATH = os.path.join(MODELS_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_model.joblib")
IMPORTANCE_PLOT_PATH = os.path.join(PLOTS_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_feature_importance.png")
TRAINING_HISTORY_PLOT_PATH = os.path.join(PLOTS_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_training_history.png")
PERFORMANCE_CSV_PATH = os.path.join(PERFORMANCE_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_threshold_performance.csv")

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 30 # Will be part of the optimized params
N_TOP_FEATURES_TO_PLOT = 50

# --- 创建输出目录 (Create output directories) ---
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_OUTPUT_DIR, exist_ok=True)

print(f"项目根目录 (Project Root): {PROJECT_ROOT}")
print(f"特征输入目录 (Features Input Dir): {FEATURES_INPUT_DIR}")
print(f"所有输出将保存在包含 '{OUTPUT_SUFFIX}' 的子目录中 (All outputs will be saved in subdirectories containing '{OUTPUT_SUFFIX}')")

# --- 辅助函数：计算性能指标 ---
def calculate_metrics(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    print(f"\n--- {title} 性能表现 ---")
    print(f"混淆矩阵 (Confusion Matrix):\n{cm}")
    print(f"  正确负例 (TN): {tn}, 错误正例 (FP): {fp}")
    print(f"  错误负例 (FN): {fn}, 正确正例 (TP): {tp}")
    print(f"准确率 (Accuracy): {accuracy:.4f}, 命中率 (POD): {pod:.4f}, 空报率 (FAR): {far:.4f}, 临界成功指数 (CSI): {csi:.4f}")
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}

# --- 1. 加载数据 ---
print("正在加载长江流域展平数据 (V1)... (Loading flattened Yangtze data (V1)...)")
if not (os.path.exists(X_FLAT_PATH) and os.path.exists(Y_FLAT_PATH) and os.path.exists(FEATURE_NAMES_PATH)):
    print(f"错误：V1 特征文件未在 {FEATURES_INPUT_DIR} 中找到。")
    print(f"寻找路径 (Looking for paths):")
    print(f"X: {X_FLAT_PATH} (存在: {os.path.exists(X_FLAT_PATH)})")
    print(f"Y: {Y_FLAT_PATH} (存在: {os.path.exists(Y_FLAT_PATH)})")
    print(f"Names: {FEATURE_NAMES_PATH} (存在: {os.path.exists(FEATURE_NAMES_PATH)})")
    print("请确保这些文件存在，它们应由适用于长江流域的 turn1.py 生成。")
    exit()
try:
    X_flat = np.load(X_FLAT_PATH) # Should be (nsamples, nfeatures)
    Y_flat_raw = np.load(Y_FLAT_PATH) # Should be (nsamples,)
    print(f"已加载 X_flat 形状 (Loaded X_flat shape): {X_flat.shape}")
    print(f"已加载 Y_flat_raw 形状 (Loaded Y_flat_raw shape): {Y_flat_raw.shape}")
except Exception as e:
    print(f"加载数据时出错 (Error loading data): {e}")
    exit()

# --- 2. 加载特征名称 ---
print(f"正在从 {FEATURE_NAMES_PATH} 加载特征名称... (Loading feature names from {FEATURE_NAMES_PATH}...)")
try:
    with open(FEATURE_NAMES_PATH, 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    print(f"已加载 {len(feature_names)} 个特征名称。(Loaded {len(feature_names)} feature names.)")
    if len(feature_names) != X_flat.shape[1]:
        print(f"警告：特征名称数量 ({len(feature_names)}) 与数据列数 ({X_flat.shape[1]}) 不匹配！")
        feature_names = [f'f{i}' for i in range(X_flat.shape[1])] # Fallback
except Exception as e:
    print(f"加载特征名称时出错: {e}。将使用通用名称。(Error loading feature names: {e}. Using generic names.)")
    feature_names = [f'f{i}' for i in range(X_flat.shape[1])]

# --- 3. 预处理 ---
print("正在预处理数据... (Preprocessing data...)")
y_flat_binary = (Y_flat_raw > RAIN_THRESHOLD).astype(int)
del Y_flat_raw

print("正在将数据分割为训练集和测试集... (Splitting data into training and testing sets...)")
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_flat_binary,
    test_size=TEST_SIZE_RATIO,
    random_state=RANDOM_STATE,
    stratify=y_flat_binary
)
del X_flat, y_flat_binary
import gc
gc.collect()

print(f"训练集形状 (Training set shape): X={X_train.shape}, y={y_train.shape}")
print(f"测试集形状 (Test set shape): X={X_test.shape}, y={y_test.shape}")
if y_train.size > 0 and y_test.size > 0:
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    print(f"训练集分布 (Train distribution): No Rain={train_counts[0] if len(train_counts)>0 else 0}, Rain={train_counts[1]if len(train_counts)>1 else 0}")
    print(f"测试集分布 (Test distribution): No Rain={test_counts[0] if len(test_counts)>0 else 0}, Rain={test_counts[1] if len(test_counts)>1 else 0}")
else:
    print("警告：训练集或测试集为空。(Warning: Training or test set is empty.)")

# --- 4. 定义并训练 XGBoost 模型 (使用V1优化参数) ---
print("正在定义并训练 XGBoost 模型 (长江流域 V1 特征, Optuna 优化参数)...")
print("(Defining and training XGBoost model (Yangtze V1 features, Optuna optimized parameters)...)")

# Optuna 优化的超参数 (来自您之前的提供) - 更新为新的最佳参数
optuna_best_params_v1 = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist',
    # 'scale_pos_weight' will be calculated and added later
    'random_state': RANDOM_STATE, # Uses the global RANDOM_STATE
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS, # Uses the global EARLY_STOPPING_ROUNDS
    'device': 'cuda',
    'n_estimators': 2582,
    'learning_rate': 0.03480882479433697,
    'max_depth': 17,
    'subsample': 0.9273230022264735,
    'colsample_bytree': 0.850202469493706,
    'gamma': 0.1252469721920858,
    'lambda': 4.346099670573116e-05,
    'alpha': 1.3767826511091745e-06,
    'use_label_encoder': False # Standard practice for newer XGBoost versions
}

# 计算 scale_pos_weight 并添加到参数中
num_neg_train = np.sum(y_train == 0)
num_pos_train = np.sum(y_train == 1)
scale_pos_weight = num_neg_train / num_pos_train if num_pos_train > 0 else 1
optuna_best_params_v1['scale_pos_weight'] = scale_pos_weight
print(f"计算得到的 scale_pos_weight (来自训练集): {scale_pos_weight:.4f} (Calculated scale_pos_weight (from training set))")
print(f"最终使用的模型参数 (Final model parameters used): \n{optuna_best_params_v1}")

model = xgb.XGBClassifier(**optuna_best_params_v1)

print("开始模型训练... (Starting model training...)")
start_time = time.time()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train,
          eval_set=eval_set,
          verbose=50)
end_time = time.time()
print(f"训练完成，耗时 {end_time - start_time:.2f} 秒。(Training complete in {end_time - start_time:.2f} seconds.)")

try:
    print(f"最佳迭代次数 (Best iteration): {model.best_iteration}")
    primary_metric = optuna_best_params_v1['eval_metric'][0] # 通常是第一个评估指标
    print(f"最佳分数 (测试集 {primary_metric}) (Best score (test {primary_metric})): {model.best_score:.4f}")
except AttributeError as e:
    print(f"无法直接获取最佳迭代/分数属性: {e} (Could not retrieve best iteration/score attributes directly: {e})")

# --- 4b. 绘制训练历史 (Plotting Training History) ---
print(f"\n--- 正在绘制训练历史 ({OUTPUT_SUFFIX}) (Plotting Training History) ---")
results = model.evals_result()
if hasattr(model, 'best_iteration') and model.best_iteration >= 0 :
    num_actual_rounds = model.best_iteration + 1
else:
    try: # Fallback if best_iteration is not set or is 0 but training happened
        num_actual_rounds = len(results['validation_0'][optuna_best_params_v1['eval_metric'][0]])
        if num_actual_rounds == 0 and model.n_estimators > 0 : # Edge case: no early stopping, full n_estimators
             num_actual_rounds = model.n_estimators
    except KeyError: # If eval_metric isn't in results somehow
        num_actual_rounds = model.n_estimators


x_axis = range(0, num_actual_rounds)
fig_hist, ax_hist = plt.subplots(1, 2, figsize=(15, 5))
fig_hist.suptitle(f'XGBoost Training History ({OUTPUT_SUFFIX})', fontsize=16)

primary_metric_plot = optuna_best_params_v1['eval_metric'][0]
secondary_metric_plot = optuna_best_params_v1['eval_metric'][1] if len(optuna_best_params_v1['eval_metric']) > 1 else primary_metric_plot

ax_hist[0].plot(x_axis, results['validation_0'][primary_metric_plot][:num_actual_rounds], label=f'Train {primary_metric_plot}')
ax_hist[0].plot(x_axis, results['validation_1'][primary_metric_plot][:num_actual_rounds], label=f'Test {primary_metric_plot}')
ax_hist[0].legend()
ax_hist[0].set_ylabel(primary_metric_plot)
ax_hist[0].set_xlabel('Boosting Rounds')
ax_hist[0].set_title(f'XGBoost {primary_metric_plot}')

ax_hist[1].plot(x_axis, results['validation_0'][secondary_metric_plot][:num_actual_rounds], label=f'Train {secondary_metric_plot.upper()}')
ax_hist[1].plot(x_axis, results['validation_1'][secondary_metric_plot][:num_actual_rounds], label=f'Test {secondary_metric_plot.upper()}')
ax_hist[1].legend()
ax_hist[1].set_ylabel(secondary_metric_plot.upper())
ax_hist[1].set_xlabel('Boosting Rounds')
ax_hist[1].set_title(f'XGBoost {secondary_metric_plot.upper()}')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(TRAINING_HISTORY_PLOT_PATH)
print(f"训练历史图已保存至: {TRAINING_HISTORY_PLOT_PATH} (Training history plot saved)")
plt.close(fig_hist)

# --- 5. 特征重要性 ---
print(f"\n--- 特征重要性 ({OUTPUT_SUFFIX}) (Feature Importances) ---")
try:
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("Top 10 特征 (Top 10 Features):")
    print(importance_df.head(10))

    # 保存所有特征重要性到CSV
    ft_imp_csv_path = os.path.join(PERFORMANCE_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_feature_importances_full.csv")
    importance_df.to_csv(ft_imp_csv_path, index=False)
    print(f"完整特征重要性已保存至CSV: {ft_imp_csv_path} (Full feature importances saved to CSV)")


    fig_imp, ax_imp = plt.subplots(figsize=(10, max(6, N_TOP_FEATURES_TO_PLOT / 2.5)))
    top_features_df = importance_df.head(N_TOP_FEATURES_TO_PLOT)
    ax_imp.barh(top_features_df['Feature'], top_features_df['Importance'])
    ax_imp.set_xlabel("Importance Score (Gain)")
    ax_imp.set_ylabel("Feature")
    ax_imp.set_title(f"Top {N_TOP_FEATURES_TO_PLOT} Feature Importances ({OUTPUT_SUFFIX})")
    ax_imp.invert_yaxis()
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH)
    print(f"特征重要性图已保存至: {IMPORTANCE_PLOT_PATH} (Feature importance plot saved)")
    plt.close(fig_imp)

except Exception as plot_e:
    print(f"警告：无法生成特征重要性图 - {plot_e} (Warning: Could not generate feature importance plot)")

# --- 6. 评估模型 ---
print(f"\n--- 在测试集上评估模型 ({OUTPUT_SUFFIX}) (Evaluating Model on Test Set) ---")
y_pred_proba = model.predict_proba(X_test)[:, 1]

np.save(MODEL_PREDICTION_PATH, y_pred_proba)
print(f"预测结果已保存至: {MODEL_PREDICTION_PATH} (Predictions saved)")

thresholds_to_evaluate = np.arange(0.1, 0.71, 0.05) # More granular: 0.1, 0.15, ..., 0.7
metrics_by_threshold = {}
for threshold in thresholds_to_evaluate:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    metrics_val = calculate_metrics(y_test, y_pred_threshold, title=f"XGBoost ({OUTPUT_SUFFIX}, 阈值 {threshold:.2f})")
    metrics_by_threshold[threshold] = metrics_val

print(f"\n--- XGBoost ({OUTPUT_SUFFIX}) 不同阈值下的性能 (测试集) ---")
print(f"(Performance across different thresholds (Test Set) for {OUTPUT_SUFFIX})")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
threshold_metrics_data = {}
for threshold, metrics_val in metrics_by_threshold.items():
    threshold_metrics_data[f'XGB_{OUTPUT_SUFFIX}_Thr_{threshold:.2f}'] = {
        metric_key: metrics_val.get(metric_key, float('nan')) for metric_key in metrics_to_show
    }

threshold_df = pd.DataFrame(threshold_metrics_data).T
if not threshold_df.empty:
    threshold_df = threshold_df[metrics_to_show]
    float_cols = ['accuracy', 'pod', 'far', 'csi']
    for col in float_cols: threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
    int_cols = ['fp', 'fn']
    for col in int_cols: threshold_df[col] = threshold_df[col].map('{:.0f}'.format)
    print(threshold_df)
    print(f"正在将阈值性能表格保存至 {PERFORMANCE_CSV_PATH} (Saving threshold performance table)")
    threshold_df.to_csv(PERFORMANCE_CSV_PATH)
else:
    print("没有指标可供展示或保存。(No metrics to display or save.)")

# --- 7. 保存模型 ---
print(f"\n正在将训练好的模型保存至 {MODEL_SAVE_PATH}... (Saving the trained model)")
try:
    joblib.dump(model, MODEL_SAVE_PATH)
    print("模型保存成功。(Model saved successfully.)")
except Exception as e:
    print(f"保存模型时出错: {e} (Error saving model)")

print("\n脚本执行完毕。(Script finished.)")