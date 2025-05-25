import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold # Added StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime # For timestamping outputs
import sys
import gc # For garbage collection

# --- 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# --- 设置输出格式 ---

# 设置系统输出支持中文 (Terminal output supports Chinese)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception as e:
        print(f"警告：无法重新配置 sys.stdout 编码: {e}")




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
OUTPUT_SUFFIX = f"v1_opt_xgboost_yangtsu_cv_{TIMESTAMP}" # Added _cv to suffix

MODELS_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "models", OUTPUT_SUFFIX)
PLOTS_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "plots", OUTPUT_SUFFIX)
PREDICTIONS_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "predictions", OUTPUT_SUFFIX)
PERFORMANCE_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "performance_reports", OUTPUT_SUFFIX)

# 使用长江流域 V1 数据文件 (Using Yangtze V1 data files)
X_FLAT_PATH = os.path.join(FEATURES_INPUT_DIR, "X_Yangtsu_flat_features.npy") # V1 features
Y_FLAT_PATH = os.path.join(FEATURES_INPUT_DIR, "Y_Yangtsu_flat_target.npy") # V1 target
FEATURE_NAMES_PATH = os.path.join(FEATURES_INPUT_DIR, "feature_names_yangtsu.txt") # V1 feature names

# 更新输出文件路径 (Updated output file paths)
MODEL_PREDICTION_PATH = os.path.join(PREDICTIONS_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_holdout_test_proba_predictions.npy")
MODEL_SAVE_PATH = os.path.join(MODELS_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_final_model.joblib")
IMPORTANCE_PLOT_PATH = os.path.join(PLOTS_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_final_model_feature_importance.png")
TRAINING_HISTORY_PLOT_PATH = os.path.join(PLOTS_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_final_model_training_history.png")
PERFORMANCE_CSV_PATH = os.path.join(PERFORMANCE_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_holdout_test_threshold_performance.csv")
CV_PERFORMANCE_CSV_PATH = os.path.join(PERFORMANCE_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_cv_performance_summary.csv")


RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2 # This will be for the final hold-out test set
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 30
N_TOP_FEATURES_TO_PLOT = 50
N_FOLDS = 5 # Number of folds for cross-validation
CV_EVAL_THRESHOLD = 0.5 # Threshold for evaluating CV folds performance

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
    # Ensure cm is always 2x2 by specifying labels=[0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    if title: # Only print if title is provided
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
gc.collect()

print("正在将数据分割为训练集 (用于CV和最终模型) 和 最终保持测试集... (Splitting data into training set (for CV and final model) and final hold-out test set...)")
X_train_overall, X_holdout_test, y_train_overall, y_holdout_test = train_test_split(
    X_flat, y_flat_binary,
    test_size=TEST_SIZE_RATIO,
    random_state=RANDOM_STATE,
    stratify=y_flat_binary
)
del X_flat, y_flat_binary
gc.collect()

print(f"整体训练集形状 (Overall training set shape): X={X_train_overall.shape}, y={y_train_overall.shape}")
print(f"保持测试集形状 (Hold-out test set shape): X={X_holdout_test.shape}, y={y_holdout_test.shape}")
if y_train_overall.size > 0 and y_holdout_test.size > 0:
    train_counts = np.bincount(y_train_overall)
    test_counts = np.bincount(y_holdout_test)
    print(f"整体训练集分布 (Overall train distribution): No Rain={train_counts[0] if len(train_counts)>0 else 0}, Rain={train_counts[1]if len(train_counts)>1 else 0}")
    print(f"保持测试集分布 (Hold-out test distribution): No Rain={test_counts[0] if len(test_counts)>0 else 0}, Rain={test_counts[1] if len(test_counts)>1 else 0}")
else:
    print("警告：整体训练集或保持测试集为空。(Warning: Overall training or hold-out test set is empty.)")


# Optuna 优化的超参数 (来自您之前的提供) - 更新为新的最佳参数
# These parameters will be used for both CV folds and the final model
optuna_base_params_v1 = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'], # XGBoost uses the first metric for early stopping by default
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
    'device': 'cuda',
    'n_estimators': 2960,
    'learning_rate': 0.026319051020408163,
    'max_depth': 18,
    'subsample': 0.8985668163265306,
    'colsample_bytree': 0.846647612244898,
    'gamma': 0.09964387755102041,
    'lambda': 7.34496612e-06,
    'alpha': 1.1915502e-06,
    'use_label_encoder': False
}

# --- 4. 5折交叉验证 (5-Fold Cross-Validation) ---
print(f"\n--- 开始 {N_FOLDS}-折交叉验证 ({OUTPUT_SUFFIX}) ---")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_metrics_list = [] # Renamed to avoid conflict

for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_train_overall, y_train_overall)):
    print(f"\n--- CV 折叠 (Fold) {fold_num + 1}/{N_FOLDS} ---")
    X_kfold_train, X_kfold_val = X_train_overall[train_idx], X_train_overall[val_idx]
    y_kfold_train, y_kfold_val = y_train_overall[train_idx], y_train_overall[val_idx]

    current_fold_params = optuna_base_params_v1.copy()
    num_neg_fold_train = np.sum(y_kfold_train == 0)
    num_pos_fold_train = np.sum(y_kfold_train == 1)
    scale_pos_weight_fold = num_neg_fold_train / num_pos_fold_train if num_pos_fold_train > 0 else 1
    current_fold_params['scale_pos_weight'] = scale_pos_weight_fold
    
    print(f"折叠 {fold_num+1} 训练数据形状: X={X_kfold_train.shape}, y={y_kfold_train.shape}")
    print(f"折叠 {fold_num+1} 验证数据形状: X={X_kfold_val.shape}, y={y_kfold_val.shape}")
    print(f"折叠 {fold_num+1} 计算得到的 scale_pos_weight: {scale_pos_weight_fold:.4f}")

    model_cv = xgb.XGBClassifier(**current_fold_params)
    
    eval_set_cv = [(X_kfold_train, y_kfold_train), (X_kfold_val, y_kfold_val)]
    model_cv.fit(X_kfold_train, y_kfold_train,
                 eval_set=eval_set_cv,
                 verbose=100 if fold_num == 0 else False) # Print progress for first fold, less verbose for others

    y_pred_proba_val = model_cv.predict_proba(X_kfold_val)[:, 1]
    y_pred_val = (y_pred_proba_val >= CV_EVAL_THRESHOLD).astype(int)
    
    metrics_val_fold = calculate_metrics(y_kfold_val, y_pred_val, title=f"CV 折叠 {fold_num + 1} (阈值 {CV_EVAL_THRESHOLD:.2f})")
    metrics_val_fold['fold'] = fold_num + 1
    try:
        metrics_val_fold['best_iteration'] = model_cv.best_iteration
        primary_metric_cv = current_fold_params['eval_metric'][0] # logloss
        # Get score from the validation set (index 1 in eval_set_cv)
        metrics_val_fold[f'best_val_{primary_metric_cv}'] = model_cv.evals_result()['validation_1'][primary_metric_cv][model_cv.best_iteration]
        if len(current_fold_params['eval_metric']) > 1:
            secondary_metric_cv = current_fold_params['eval_metric'][1] # auc
            metrics_val_fold[f'best_val_{secondary_metric_cv}'] = model_cv.evals_result()['validation_1'][secondary_metric_cv][model_cv.best_iteration]

    except Exception as e:
        print(f"警告: 无法获取折叠 {fold_num+1} 的最佳迭代/分数: {e}")
        metrics_val_fold['best_iteration'] = -1
        metrics_val_fold[f'best_val_{primary_metric_cv}'] = float('nan')
        if len(current_fold_params['eval_metric']) > 1:
             metrics_val_fold[f'best_val_{secondary_metric_cv}'] = float('nan')


    fold_metrics_list.append(metrics_val_fold)
    
    del X_kfold_train, X_kfold_val, y_kfold_train, y_kfold_val, model_cv
    gc.collect()

print(f"\n--- {N_FOLDS}-折交叉验证结果总结 ({OUTPUT_SUFFIX}) ---")
cv_results_df = pd.DataFrame(fold_metrics_list)
print(cv_results_df)
cv_results_df.to_csv(CV_PERFORMANCE_CSV_PATH, index=False)
print(f"CV 性能总结已保存至: {CV_PERFORMANCE_CSV_PATH}")

# Calculate and print average metrics
avg_metrics = {}
print("\n--- 平均交叉验证性能 (Average Cross-Validation Performance) ---")
metric_keys_for_avg = ['accuracy', 'pod', 'far', 'csi', 'tn', 'fp', 'fn', 'tp']
if 'best_val_logloss' in cv_results_df.columns: metric_keys_for_avg.append('best_val_logloss')
if 'best_val_auc' in cv_results_df.columns: metric_keys_for_avg.append('best_val_auc')


for metric_key in metric_keys_for_avg:
    if metric_key in cv_results_df:
        mean_val = cv_results_df[metric_key].mean()
        std_val = cv_results_df[metric_key].std()
        avg_metrics[f'avg_{metric_key}'] = mean_val
        avg_metrics[f'std_{metric_key}'] = std_val
        print(f"平均 CV {metric_key.upper()}: {mean_val:.4f} (+/- {std_val:.4f})")

# --- 5. 定义并训练最终 XGBoost 模型 (在整体训练集上) ---
print("\n--- 正在定义并训练最终 XGBoost 模型 (长江流域 V1 特征, Optuna 优化参数, 在整体训练集上)...")
print("(Defining and training FINAL XGBoost model (Yangtze V1 features, Optuna optimized parameters, on OVERALL training set)...)")

final_model_params = optuna_base_params_v1.copy()
num_neg_train_overall = np.sum(y_train_overall == 0)
num_pos_train_overall = np.sum(y_train_overall == 1)
scale_pos_weight_overall = num_neg_train_overall / num_pos_train_overall if num_pos_train_overall > 0 else 1
final_model_params['scale_pos_weight'] = scale_pos_weight_overall

print(f"为最终模型计算得到的 scale_pos_weight (来自整体训练集): {scale_pos_weight_overall:.4f}")
print(f"最终模型使用的参数: \n{final_model_params}")

model_final = xgb.XGBClassifier(**final_model_params)

print("开始最终模型训练... (Starting final model training...)")
start_time = time.time()
# Evaluate on training data and hold-out test data
eval_set_final = [(X_train_overall, y_train_overall), (X_holdout_test, y_holdout_test)]
model_final.fit(X_train_overall, y_train_overall,
                eval_set=eval_set_final,
                verbose=50) # Print progress every 50 rounds for final model
end_time = time.time()
print(f"最终模型训练完成，耗时 {end_time - start_time:.2f} 秒。(Final model training complete in {end_time - start_time:.2f} seconds.)")

try:
    print(f"最终模型最佳迭代次数 (Best iteration for final model): {model_final.best_iteration}")
    primary_metric_final = final_model_params['eval_metric'][0]
    # Accessing the score on the *second* validation set (holdout_test, index 1)
    best_score_on_holdout = model_final.evals_result()['validation_1'][primary_metric_final][model_final.best_iteration]
    print(f"最终模型最佳分数 (保持测试集 {primary_metric_final}) (Best score for final model (holdout_test {primary_metric_final})): {best_score_on_holdout:.4f}")
    if len(final_model_params['eval_metric']) > 1:
        secondary_metric_final = final_model_params['eval_metric'][1]
        best_score_secondary_on_holdout = model_final.evals_result()['validation_1'][secondary_metric_final][model_final.best_iteration]
        print(f"最终模型最佳分数 (保持测试集 {secondary_metric_final}) (Best score for final model (holdout_test {secondary_metric_final})): {best_score_secondary_on_holdout:.4f}")

except AttributeError as e:
    print(f"无法直接获取最终模型最佳迭代/分数属性: {e} (Could not retrieve best iteration/score attributes directly for final model: {e})")
except KeyError as e:
    print(f"无法从 evals_result() 获取最终模型分数: {e}")


# --- 5b. 绘制最终模型训练历史 (Plotting Final Model Training History) ---
print(f"\n--- 正在绘制最终模型训练历史 ({OUTPUT_SUFFIX}) (Plotting Final Model Training History) ---")
results_final = model_final.evals_result()

num_actual_rounds_final = model_final.n_estimators # Default
if hasattr(model_final, 'best_iteration') and model_final.best_iteration >= 0 :
    num_actual_rounds_final = model_final.best_iteration + 1
else: # Fallback if best_iteration is not set or is 0 but training happened
    try:
        primary_metric_key = final_model_params['eval_metric'][0]
        num_actual_rounds_final = len(results_final['validation_0'][primary_metric_key])
        if num_actual_rounds_final == 0 and model_final.n_estimators > 0 : # Edge case: no early stopping, full n_estimators
             num_actual_rounds_final = model_final.n_estimators
    except (KeyError, IndexError): # If eval_metric isn't in results somehow or list empty
        pass # Keep default num_actual_rounds_final


x_axis_final = range(0, num_actual_rounds_final)
fig_hist_final, ax_hist_final = plt.subplots(1, 2, figsize=(15, 5))
fig_hist_final.suptitle(f'XGBoost Final Model Training History ({OUTPUT_SUFFIX})', fontsize=16)

primary_metric_plot_final = final_model_params['eval_metric'][0]
secondary_metric_plot_final = final_model_params['eval_metric'][1] if len(final_model_params['eval_metric']) > 1 else primary_metric_plot_final

# Plot for primary metric
ax_hist_final[0].plot(x_axis_final, results_final['validation_0'][primary_metric_plot_final][:num_actual_rounds_final], label=f'Train Overall {primary_metric_plot_final}')
ax_hist_final[0].plot(x_axis_final, results_final['validation_1'][primary_metric_plot_final][:num_actual_rounds_final], label=f'Holdout Test {primary_metric_plot_final}')
ax_hist_final[0].legend()
ax_hist_final[0].set_ylabel(primary_metric_plot_final)
ax_hist_final[0].set_xlabel('Boosting Rounds')
ax_hist_final[0].set_title(f'XGBoost {primary_metric_plot_final}')

# Plot for secondary metric
if primary_metric_plot_final != secondary_metric_plot_final: # Avoid plotting the same metric twice if only one eval_metric
    ax_hist_final[1].plot(x_axis_final, results_final['validation_0'][secondary_metric_plot_final][:num_actual_rounds_final], label=f'Train Overall {secondary_metric_plot_final.upper()}')
    ax_hist_final[1].plot(x_axis_final, results_final['validation_1'][secondary_metric_plot_final][:num_actual_rounds_final], label=f'Holdout Test {secondary_metric_plot_final.upper()}')
    ax_hist_final[1].legend()
    ax_hist_final[1].set_ylabel(secondary_metric_plot_final.upper())
    ax_hist_final[1].set_xlabel('Boosting Rounds')
    ax_hist_final[1].set_title(f'XGBoost {secondary_metric_plot_final.upper()}')
else: # If only one metric, make the second plot empty or provide a message
    ax_hist_final[1].text(0.5, 0.5, 'Secondary metric same as primary or not available.', ha='center', va='center')
    ax_hist_final[1].set_title('Secondary Metric')


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(TRAINING_HISTORY_PLOT_PATH)
print(f"最终模型训练历史图已保存至: {TRAINING_HISTORY_PLOT_PATH} (Final model training history plot saved)")
plt.close(fig_hist_final)

# --- 6. 最终模型特征重要性 ---
print(f"\n--- 最终模型特征重要性 ({OUTPUT_SUFFIX}) (Final Model Feature Importances) ---")
try:
    importances_final = model_final.feature_importances_
    importance_df_final = pd.DataFrame({'Feature': feature_names, 'Importance': importances_final})
    importance_df_final = importance_df_final.sort_values(by='Importance', ascending=False)

    print("最终模型 Top 10 特征 (Final Model Top 10 Features):")
    print(importance_df_final.head(10))

    ft_imp_csv_path_final = os.path.join(PERFORMANCE_OUTPUT_DIR, f"xgboost_yangtsu_{OUTPUT_SUFFIX}_final_model_feature_importances_full.csv")
    importance_df_final.to_csv(ft_imp_csv_path_final, index=False)
    print(f"最终模型完整特征重要性已保存至CSV: {ft_imp_csv_path_final} (Final model full feature importances saved to CSV)")

    fig_imp_final, ax_imp_final = plt.subplots(figsize=(10, max(6, N_TOP_FEATURES_TO_PLOT / 2.5)))
    top_features_df_final = importance_df_final.head(N_TOP_FEATURES_TO_PLOT)
    ax_imp_final.barh(top_features_df_final['Feature'], top_features_df_final['Importance'])
    ax_imp_final.set_xlabel("Importance Score (Gain)")
    ax_imp_final.set_ylabel("Feature")
    ax_imp_final.set_title(f"Top {N_TOP_FEATURES_TO_PLOT} Feature Importances (Final Model, {OUTPUT_SUFFIX})")
    ax_imp_final.invert_yaxis()
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH)
    print(f"最终模型特征重要性图已保存至: {IMPORTANCE_PLOT_PATH} (Final model feature importance plot saved)")
    plt.close(fig_imp_final)

except Exception as plot_e:
    print(f"警告：无法生成最终模型特征重要性图 - {plot_e} (Warning: Could not generate final model feature importance plot)")

# --- 7. 在保持测试集上评估最终模型 ---
print(f"\n--- 在保持测试集上评估最终模型 ({OUTPUT_SUFFIX}) (Evaluating Final Model on Hold-out Test Set) ---")
y_pred_proba_holdout = model_final.predict_proba(X_holdout_test)[:, 1]

np.save(MODEL_PREDICTION_PATH, y_pred_proba_holdout)
print(f"最终模型在保持测试集上的预测概率已保存至: {MODEL_PREDICTION_PATH} (Final model predictions on hold-out test set saved)")

thresholds_to_evaluate = np.arange(0.1, 0.71, 0.05)
metrics_by_threshold_holdout = {}
for threshold in thresholds_to_evaluate:
    y_pred_threshold_holdout = (y_pred_proba_holdout >= threshold).astype(int)
    metrics_val_holdout = calculate_metrics(y_holdout_test, y_pred_threshold_holdout, title=f"最终模型 ({OUTPUT_SUFFIX}, 保持测试集, 阈值 {threshold:.2f})")
    metrics_by_threshold_holdout[threshold] = metrics_val_holdout

print(f"\n--- 最终模型 ({OUTPUT_SUFFIX}) 不同阈值下的性能 (保持测试集) ---")
print(f"(Performance across different thresholds (Hold-out Test Set) for final model {OUTPUT_SUFFIX})")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn', 'tp', 'tn'] # Added tp, tn
threshold_metrics_data_holdout = {}

# Populate the dictionary for DataFrame creation
for threshold, metrics_val in metrics_by_threshold_holdout.items():
    row_name = f'XGB_Final_{OUTPUT_SUFFIX}_Thr_{threshold:.2f}'
    threshold_metrics_data_holdout[row_name] = {
        metric_key: metrics_val.get(metric_key, float('nan')) for metric_key in metrics_to_show
    }

threshold_df_holdout = pd.DataFrame.from_dict(threshold_metrics_data_holdout, orient='index')

if not threshold_df_holdout.empty:
    threshold_df_holdout = threshold_df_holdout[metrics_to_show] # Ensure column order
    float_cols = ['accuracy', 'pod', 'far', 'csi']
    for col in float_cols:
        if col in threshold_df_holdout:
             threshold_df_holdout[col] = threshold_df_holdout[col].map('{:.4f}'.format)
    int_cols = ['fp', 'fn', 'tp', 'tn']
    for col in int_cols:
        if col in threshold_df_holdout:
            threshold_df_holdout[col] = threshold_df_holdout[col].map('{:.0f}'.format)
    print(threshold_df_holdout)
    print(f"正在将最终模型在保持测试集上的阈值性能表格保存至 {PERFORMANCE_CSV_PATH} (Saving final model threshold performance table for hold-out set)")
    threshold_df_holdout.to_csv(PERFORMANCE_CSV_PATH)
else:
    print("最终模型在保持测试集上没有指标可供展示或保存。(No metrics for final model on hold-out set to display or save.)")

# --- 8. 保存最终模型 ---
print(f"\n正在将训练好的最终模型保存至 {MODEL_SAVE_PATH}... (Saving the trained final model)")
try:
    joblib.dump(model_final, MODEL_SAVE_PATH)
    print("最终模型保存成功。(Final model saved successfully.)")
except Exception as e:
    print(f"保存最终模型时出错: {e} (Error saving final model)")

print("\n脚本执行完毕。(Script finished.)")