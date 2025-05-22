import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split # 使用 StratifiedKFold 保持类别比例
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from loaddata import mydata # 假设 mydata 在同一目录或可访问
import os
import pandas as pd
import joblib
import time
import optuna

# --- 配置 ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_Yangtsu_flat_features.npy") # v6 特征
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target.npy") # v6 目标
# 输出目录调整，用于存放 K-Fold 相关产出
KFOLD_OUTPUT_DIR = os.path.join(PROJECT_DIR, "kfold_optimization_v6")
os.makedirs(KFOLD_OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_yangtsu.txt") # v6 特征名
RAIN_THRESHOLD = 0.1
# TEST_SIZE_RATIO 将用于从完整数据中分出最终的 Hold-out 测试集
# K-Fold CV 将在 (1 - TEST_SIZE_RATIO) 的数据上进行
TEST_SIZE_RATIO_HOLDOUT = 0.2 # 用于最终评估的独立测试集比例
N_SPLITS_KFold = 5 # 5折交叉验证
MAX_LOOKBACK = 30 # 与 turn1.py 一致

# Optuna 配置
N_TRIALS_OPTUNA = 50 # Optuna 试验次数
OPTUNA_TIMEOUT = 360000 # Optuna 超时时间 (秒)
OPTIMIZE_METRIC = 'auc' # Optuna 优化指标
EARLY_STOPPING_ROUNDS_OPTUNA = 30 # Optuna 内部早停轮数
EARLY_STOPPING_ROUNDS_FINAL_MODEL = 50 # 最终模型/每折模型训练时的早停轮数


# --- 辅助函数：计算性能指标 (保持不变) ---
def calculate_metrics(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

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
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Rain', 'Rain']))
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}

# --- 1. 加载数据 ---
print("--- Step 1: Loading Yangtze v6 Flattened Data ---")
start_load_flat = time.time()
if not (os.path.exists(X_FLAT_PATH) and os.path.exists(Y_FLAT_PATH) and os.path.exists(FEATURE_NAMES_PATH)):
    raise FileNotFoundError(f"Flattened data files for Yangtze v6 dataset not found in {PROJECT_DIR}. Run turn1.py first.")

try:
    print(f"Attempting to load {X_FLAT_PATH}...")
    X_flat_full = np.load(X_FLAT_PATH)
    print(f"Attempting to load {Y_FLAT_PATH}...")
    Y_flat_full_raw = np.load(Y_FLAT_PATH)
except MemoryError as e:
    print(f"MemoryError loading flattened data: {e}. Consider reducing dataset size or using mmap if issues persist.")
    raise
except Exception as e:
    print(f"Error loading flattened data: {e}")
    raise

with open(FEATURE_NAMES_PATH, "r") as f:
    feature_names = [line.strip() for line in f]

print(f"Loaded flattened features X_flat_full: shape {X_flat_full.shape}")
print(f"Loaded flattened target Y_flat_full_raw: shape {Y_flat_full_raw.shape}")
print(f"Loaded {len(feature_names)} feature names.")
end_load_flat = time.time()
print(f"Flattened data loading finished in {end_load_flat - start_load_flat:.2f} seconds.")

# --- 2. 数据预处理与划分 ---
print("\n--- Step 2: Data Preprocessing and Splitting ---")
Y_flat_full_binary = (Y_flat_full_raw > RAIN_THRESHOLD).astype(int)
del Y_flat_full_raw # 释放原始 Y 数据内存

# 首先，划分出一个最终的、与交叉验证和Optuna完全独立的保持测试集 (Hold-out Test Set)
print(f"Splitting full dataset into training/CV pool and a final hold-out test set (test_size={TEST_SIZE_RATIO_HOLDOUT})...")
X_train_cv_pool, X_holdout_test, y_train_cv_pool, y_holdout_test = train_test_split(
    X_flat_full, Y_flat_full_binary,
    test_size=TEST_SIZE_RATIO_HOLDOUT,
    random_state=42,
    stratify=Y_flat_full_binary # 保持类别比例
)
del X_flat_full, Y_flat_full_binary # 释放完整数据集内存

print(f"Training/CV Pool size: {len(y_train_cv_pool)}")
print(f"Hold-out Test Set size: {len(y_holdout_test)}")
train_cv_pool_counts = np.bincount(y_train_cv_pool)
holdout_test_counts = np.bincount(y_holdout_test)
print(f"Training/CV Pool distribution: No Rain={train_cv_pool_counts[0]}, Rain={train_cv_pool_counts[1]}")
print(f"Hold-out Test Set distribution: No Rain={holdout_test_counts[0]}, Rain={holdout_test_counts[1]}")


# --- 3. Optuna 超参数优化 (在 K-Fold CV 的第一折的训练集上进行，或在整个 train_cv_pool 的一部分上进行) ---
print("\n--- Step 3: Hyperparameter Optimization with Optuna ---")
# 为 Optuna 准备数据：可以从 train_cv_pool 中再分出一小部分用于 Optuna 的快速验证，或者直接用 KFold 的第一折
# 这里我们简单地从 train_cv_pool 中再分一个临时训练集和验证集给 Optuna
X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
    X_train_cv_pool, y_train_cv_pool, test_size=0.25, random_state=123, stratify=y_train_cv_pool
)
print(f"Optuna training set size: {len(y_opt_train)}")
print(f"Optuna validation set size: {len(y_opt_val)}")

start_opt_time = time.time()

def optuna_objective(trial):
    """Optuna objective function."""
    param = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', OPTIMIZE_METRIC], # 确保优化指标被评估
        'tree_method': 'hist',
        'verbosity': 0, # 静默模式
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100), # 调整范围和步长
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0), # 调整范围
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True), # L2 正则化，调整范围
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True), # L1 正则化，调整范围
        'scale_pos_weight': (np.sum(y_opt_train == 0) / np.sum(y_opt_train == 1)) if np.sum(y_opt_train == 1) > 0 else 1,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS_OPTUNA
    }

    model_opt = xgb.XGBClassifier(**param)
    # Optuna 内部使用 X_opt_val 进行早停和评估
    eval_set_opt = [(X_opt_val, y_opt_val)]

    try:
        model_opt.fit(X_opt_train, y_opt_train,
                      eval_set=eval_set_opt,
                      verbose=False) # Optuna 内部训练不打印

        results_opt = model_opt.evals_result()
        # 确保 eval_metric 中包含 OPTIMIZE_METRIC
        if OPTIMIZE_METRIC in results_opt['validation_0']:
            best_score_opt = results_opt['validation_0'][OPTIMIZE_METRIC][model_opt.best_iteration]
            return best_score_opt
        else: # 如果 OPTIMIZE_METRIC 不在评估结果中，例如只用了 logloss
            print(f"Warning: {OPTIMIZE_METRIC} not found in Optuna trial eval_results. Using logloss.")
            return results_opt['validation_0']['logloss'][model_opt.best_iteration]


    except Exception as e_opt:
        print(f"Optuna trial failed with error: {e_opt}")
        return float('inf') if OPTIMIZE_METRIC == 'logloss' else 0.0 # 返回差值

study_direction = 'maximize' if OPTIMIZE_METRIC == 'auc' else 'minimize'
study = optuna.create_study(direction=study_direction)
optuna.logging.set_verbosity(optuna.logging.WARNING) # 减少 Optuna 的日志输出

try:
    print(f"Starting Optuna optimization for {N_TRIALS_OPTUNA} trials with timeout {OPTUNA_TIMEOUT}s...")
    study.optimize(optuna_objective, n_trials=N_TRIALS_OPTUNA, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)
except KeyboardInterrupt:
    print("Optuna optimization stopped manually.")

end_opt_time = time.time()
print(f"Hyperparameter optimization finished in {end_opt_time - start_opt_time:.2f} seconds.")

best_hyperparams = study.best_params
print("\nBest hyperparameters found by Optuna:")
print(best_hyperparams)
print(f"Best {OPTIMIZE_METRIC} value during Optuna: {study.best_value:.4f}")

# 清理 Optuna 使用的临时数据
del X_opt_train, X_opt_val, y_opt_train, y_opt_val

# --- 4. K-Fold 交叉验证与折外预测生成 ---
print(f"\n--- Step 4: Performing {N_SPLITS_KFold}-Fold Cross-Validation with Optimized Parameters ---")
kf = StratifiedKFold(n_splits=N_SPLITS_KFold, shuffle=True, random_state=42)

# 初始化用于存储折外预测的数组
oof_preds_L0_v6_Opt = np.zeros(len(y_train_cv_pool))
# 初始化用于存储每折在 Hold-out Test Set 上的预测的列表
holdout_test_preds_from_folds = []
# 初始化用于存储每折模型的列表
fold_models_v6_Opt = []

# 准备最终模型的参数，加入 Optuna 找到的最佳参数
# n_estimators 和 learning_rate 通常是 Optuna 优化的重点，其他可以固定或也加入优化
final_model_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'], # 监控多个指标
    'tree_method': 'hist',
    'scale_pos_weight': (np.sum(y_train_cv_pool == 0) / np.sum(y_train_cv_pool == 1)) if np.sum(y_train_cv_pool == 1) > 0 else 1,
    'random_state': 42,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS_FINAL_MODEL # 用于每折训练
}
final_model_params.update(best_hyperparams) # 合并 Optuna 找到的最佳参数

print(f"Parameters for K-Fold models (after Optuna): {final_model_params}")

start_kfold_time = time.time()
for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train_cv_pool, y_train_cv_pool)):
    print(f"\n--- Fold {fold_num + 1}/{N_SPLITS_KFold} ---")
    X_fold_train, X_fold_val = X_train_cv_pool[train_idx], X_train_cv_pool[val_idx]
    y_fold_train, y_fold_val = y_train_cv_pool[train_idx], y_train_cv_pool[val_idx]

    print(f"  Fold training set size: {len(y_fold_train)}")
    print(f"  Fold validation set size: {len(y_fold_val)}")

    fold_model = xgb.XGBClassifier(**final_model_params)
    eval_set_fold = [(X_fold_val, y_fold_val)] # 使用当前折的验证集进行早停

    print(f"  Fitting model for Fold {fold_num + 1}...")
    fold_model.fit(X_fold_train, y_fold_train,
                   eval_set=eval_set_fold,
                   verbose=100) # 每100轮打印一次

    # 生成折外预测 (Out-of-Fold Predictions)
    oof_preds_L0_v6_Opt[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]
    print(f"  Fold {fold_num + 1} OOF predictions generated.")

    # (可选) 在 Hold-out Test Set 上进行预测
    holdout_test_pred_fold = fold_model.predict_proba(X_holdout_test)[:, 1]
    holdout_test_preds_from_folds.append(holdout_test_pred_fold)
    print(f"  Fold {fold_num + 1} predictions on Hold-out Test Set generated.")

    # 保存当前折的模型
    fold_model_path = os.path.join(KFOLD_OUTPUT_DIR, f"xgboost_v6_opt_fold_{fold_num + 1}.joblib")
    joblib.dump(fold_model, fold_model_path)
    fold_models_v6_Opt.append(fold_model) # 也可以直接将模型对象存入列表
    print(f"  Fold {fold_num + 1} model saved to {fold_model_path}")

end_kfold_time = time.time()
print(f"\nK-Fold Cross-Validation finished in {end_kfold_time - start_kfold_time:.2f} seconds.")

# 保存折外预测结果 (Train_L0_Probs_v6_Opt)
oof_preds_save_path = os.path.join(KFOLD_OUTPUT_DIR, "Train_L0_Probs_v6_Opt.npy")
np.save(oof_preds_save_path, oof_preds_L0_v6_Opt)
print(f"Out-of-Fold predictions for Training/CV Pool saved to: {oof_preds_save_path}")

# 处理 Hold-out Test Set 上的预测：可以简单平均，或用之后在完整数据上训练的模型预测
Test_L0_Probs_v6_Opt_from_folds_mean = np.mean(holdout_test_preds_from_folds, axis=0)
test_preds_mean_save_path = os.path.join(KFOLD_OUTPUT_DIR, "Test_L0_Probs_v6_Opt_from_folds_mean.npy")
np.save(test_preds_mean_save_path, Test_L0_Probs_v6_Opt_from_folds_mean)
print(f"Mean predictions on Hold-out Test Set from K-Folds saved to: {test_preds_mean_save_path}")

# --- 5. (推荐) 用最佳参数在完整的 Training/CV Pool 上训练最终模型 ---
# 这个模型将用于对独立的验证集（如果有）和最终的Hold-out测试集进行标准评估
print("\n--- Step 5: Training Final Model on Full Training/CV Pool with Optimized Parameters ---")
print("This model will be used for predicting on truly unseen data (like a separate validation set or the hold-out test set).")

final_model_full_train = xgb.XGBClassifier(**final_model_params) # 使用相同的优化参数
# 注意：这里的早停应该基于一个独立的验证集，或者不使用早停，直接训练到 n_estimators
# 为简单起见，且我们已经有了 holdout_test，这里我们可以在 holdout_test 上进行早停评估，但这有点“偷看”
# 更标准做法是：Optuna在(train_cv_pool的子集)上调参 -> KFold在train_cv_pool上生成OOF -> 在完整的train_cv_pool上用最优参数训练最终模型，评估用holdout_test
# 如果没有独立的验证集给这里的 final_model_full_train，可以不设置 early_stopping_rounds，或者训练固定轮数
# 或者，我们也可以选择 KFold 中的一个模型（例如性能最好的或平均的）作为“最终模型”代表，或者直接用平均预测

# 我们选择重新训练一个模型，并使用 holdout_test 作为早停的监控集
# 这意味着 holdout_test 的部分信息被用于模型选择（迭代次数），但参数是固定的
# 为了生成 Val_L0_Probs_v6_Opt (如果存在一个独立的验证集)，应该用这个模型去预测它。

# 假设我们没有独立的验证集，我们将用这个模型来对 Hold-out Test Set 做最终评估
eval_set_final_model = [(X_holdout_test, y_holdout_test)] # 早停监控

print(f"Fitting final model on full Training/CV Pool (size: {len(y_train_cv_pool)})...")
start_final_train_time = time.time()
final_model_full_train.fit(X_train_cv_pool, y_train_cv_pool,
                           eval_set=eval_set_final_model,
                           verbose=100)
end_final_train_time = time.time()
print(f"Final model training on full Training/CV Pool finished in {end_final_train_time - start_final_train_time:.2f} seconds.")
print(f"Best iteration for the final model on full training data: {final_model_full_train.best_iteration}")

# 保存这个最终模型
final_model_save_path = os.path.join(KFOLD_OUTPUT_DIR, "xgboost_v6_opt_final_model_on_train_cv_pool.joblib")
joblib.dump(final_model_full_train, final_model_save_path)
print(f"Final model trained on full Training/CV Pool saved to {final_model_save_path}")

# 用这个最终模型对 Hold-out Test Set 进行预测 (Test_L0_Probs_v6_Opt)
Test_L0_Probs_v6_Opt_final_model = final_model_full_train.predict_proba(X_holdout_test)[:, 1]
test_preds_final_model_save_path = os.path.join(KFOLD_OUTPUT_DIR, "Test_L0_Probs_v6_Opt_from_final_model.npy")
np.save(test_preds_final_model_save_path, Test_L0_Probs_v6_Opt_final_model)
print(f"Predictions on Hold-out Test Set from final model saved to: {test_preds_final_model_save_path}")

# Val_L0_Probs_v6_Opt: 如果您有一个独立的验证集 (X_val_independent, y_val_independent)
# Val_L0_Probs_v6_Opt = final_model_full_train.predict_proba(X_val_independent)[:, 1]
# np.save(os.path.join(KFOLD_OUTPUT_DIR, "Val_L0_Probs_v6_Opt.npy"), Val_L0_Probs_v6_Opt)
# print("Predictions on Independent Validation Set saved (if applicable).")


# --- 6. 评估最终模型在 Hold-out Test Set 上的性能 ---
print("\n--- Step 6: Evaluating Final Model (trained on full Training/CV Pool) on Hold-out Test Set ---")
# 使用 final_model_full_train 和 Test_L0_Probs_v6_Opt_final_model 进行评估
thresholds_to_evaluate = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
metrics_by_threshold_final_model = {}
y_pred_proba_final_on_holdout = Test_L0_Probs_v6_Opt_final_model # 使用上面生成的预测

print("Calculating metrics for different thresholds on Hold-out Test Set...")
for threshold_fm in thresholds_to_evaluate:
    print(f"\n--- Hold-out Test Set - Threshold: {threshold_fm:.2f} ---")
    y_pred_test_threshold_fm = (y_pred_proba_final_on_holdout >= threshold_fm).astype(int)
    metrics_fm = calculate_metrics(y_holdout_test, y_pred_test_threshold_fm,
                                   title=f"XGBoost v6 Opt (Final Model) on Hold-out Test (Threshold {threshold_fm:.2f})")
    metrics_by_threshold_final_model[threshold_fm] = metrics_fm

# 准备一个对比表格，包括基线（如果需要的话，但基线通常在完整数据集上评估，这里主要是评估模型本身）
xgboost_metrics_optimized_thr05_holdout = metrics_by_threshold_final_model[0.5]

print("\n--- XGBoost v6 Opt Performance on Hold-out Test Set (across thresholds) ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
holdout_threshold_metrics_data = {}
for threshold, metrics in metrics_by_threshold_final_model.items():
    holdout_threshold_metrics_data[f'v6_Opt_Holdout_Thr_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

holdout_threshold_df = pd.DataFrame(holdout_threshold_metrics_data).T
holdout_threshold_df = holdout_threshold_df[metrics_to_show]

float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols:
    if col in holdout_threshold_df.columns:
        holdout_threshold_df[col] = holdout_threshold_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols:
    if col in holdout_threshold_df.columns:
        holdout_threshold_df[col] = holdout_threshold_df[col].map('{:.0f}'.format)

print(holdout_threshold_df)
holdout_perf_csv_path = os.path.join(KFOLD_OUTPUT_DIR, "performance_v6_opt_final_model_on_holdout.csv")
print(f"Saving final model performance on hold-out set to {holdout_perf_csv_path}")
holdout_threshold_df.to_csv(holdout_perf_csv_path)

# --- 7. 可视化和特征重要性 (使用 final_model_full_train) ---
print("\n--- Step 7: Visualization and Feature Importance (Final Model) ---")
# 绘制训练历史 (evals_result 来自 final_model_full_train)
results_final_model = final_model_full_train.evals_result()
if 'validation_0' in results_final_model: # 'validation_0' 对应 eval_set 中的第一个元素，即 X_holdout_test
    epochs = len(results_final_model['validation_0']['logloss'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(x_axis, results_final_model['validation_0']['logloss'], label='Hold-out Test (eval_set)')
    # 如果 final_model_full_train.fit 中加入了训练集自身的评估，也可以画出来
    # if 'train' in results_final_model:
    #     ax[0].plot(x_axis, results_final_model['train']['logloss'], label='Train')
    ax[0].legend()
    ax[0].set_ylabel('LogLoss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_title('XGBoost LogLoss (v6 Opt - Final Model)')

    metric_to_plot_fm = 'auc' if 'auc' in results_final_model['validation_0'] else 'error'
    if metric_to_plot_fm in results_final_model['validation_0']:
        ax[1].plot(x_axis, results_final_model['validation_0'][metric_to_plot_fm], label=f'Hold-out Test - {metric_to_plot_fm.upper()}')
        ax[1].legend()
        ax[1].set_ylabel(metric_to_plot_fm.upper())
        ax[1].set_xlabel('Epochs')
        ax[1].set_title(f'XGBoost {metric_to_plot_fm.upper()} (v6 Opt - Final Model)')
    else:
        ax[1].set_title('Metric not found for second plot')

    plt.tight_layout()
    history_plot_path_fm = os.path.join(KFOLD_OUTPUT_DIR, "xgboost_v6_opt_final_model_training_history.png")
    plt.savefig(history_plot_path_fm)
    print(f"Final model training history plot saved to {history_plot_path_fm}")
    plt.close()
else:
    print("Warning: 'validation_0' (Hold-out Test eval) not found in final model evals_result(). Cannot plot history.")

# 特征重要性 (来自 final_model_full_train)
print("\nFeature Importances (v6 Opt - Final Model):")
try:
    importances_fm = final_model_full_train.feature_importances_
    importance_df_fm = pd.DataFrame({'Feature': feature_names, 'Importance': importances_fm})
    importance_df_fm = importance_df_fm.sort_values(by='Importance', ascending=False)

    print("Top 10 Features:")
    print(importance_df_fm.head(10))

    N_TOP_FEATURES_TO_PLOT_FM = 50
    n_features_actual_fm = len(feature_names)
    n_plot_fm = min(N_TOP_FEATURES_TO_PLOT_FM, n_features_actual_fm)

    plt.figure(figsize=(10, n_plot_fm / 2.0 if n_plot_fm > 0 else 3)) # 避免除以零
    top_features_fm = importance_df_fm.head(n_plot_fm)
    plt.barh(top_features_fm['Feature'], top_features_fm['Importance'])
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"Top {n_plot_fm} Feature Importances (XGBoost v6 Opt - Final Model)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    importance_plot_path_fm = os.path.join(KFOLD_OUTPUT_DIR, "xgboost_v6_opt_final_model_feature_importance.png")
    plt.savefig(importance_plot_path_fm)
    print(f"Final model feature importance plot saved to {importance_plot_path_fm}")
    plt.close()

except Exception as e_imp_fm:
    print(f"Could not plot final model feature importance: {e_imp_fm}")


# --- (可选) 加载原始数据的基线性能对比 ---
# 这部分与原脚本类似，但注意基线评估的数据范围应与 Hold-out Test Set 对齐（如果要做严格对比）
# 或者，基线可以在完整数据集的对应部分（扣除 lookback 后）进行评估，作为一个参考
# 为保持与原脚本结构类似，这里重新加载并评估基线，但注意其数据范围
print("\n--- (Optional) Baseline Product Performance on Corresponding Hold-out Data Portion ---")
# 重新加载原始空间数据，并确保只取 holdout_test 对应的部分
# 这部分逻辑会比较复杂，因为原始数据是空间格点，而 holdout_test 是从展平数据中抽取的
# 一个简化的做法是：假设 holdout_test 的样本在时间上是连续的，并且是 X_flat_full 的最后一部分。
# 那么我们可以尝试从原始空间数据中也取出最后对应比例的时间段。
# 这仅为近似，严格对齐需要原始数据的索引。

# --- 简单起见，这里我们跳过在精确对齐的 holdout 部分重新计算基线 ---
# --- 之前脚本中您提供的基线数据是在一个可能不同的测试集/数据划分上计算的 ---
# --- 如果需要严格对比，应确保基线评估使用与 holdout_test 完全一致的样本 ---
print("Skipping re-calculation of baseline on exact hold-out portion for brevity.")
print("Refer to initial baseline calculations for general product performance.")


print("\n--- Script Finished ---")
print(f"Key outputs for ensemble learning (using v6 Opt features):")
print(f"  Out-of-Fold predictions for Training/CV Pool: {oof_preds_save_path}")
print(f"  Predictions on Hold-out Test Set (from final model): {test_preds_final_model_save_path}")
print(f"  Final optimized model (trained on Training/CV Pool): {final_model_save_path}")
print(f"  Best hyperparameters from Optuna: {best_hyperparams}")