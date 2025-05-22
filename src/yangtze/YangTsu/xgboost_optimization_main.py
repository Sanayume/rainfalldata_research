# xgboost_optimization_main.py (你的主寻优脚本)
import time
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
# ... (其他 import 保持不变) ...
import optuna
import logging
from datetime import datetime
import os # 确保os被导入
from loaddata import mydata
import matplotlib.pyplot as plt
import pandas as pd

# --- 配置 ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_Yangtsu_flat_features_v6.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target_v6.npy")
MODEL_PREDICT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "model_predict")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_yangtsu_v6.txt")

os.makedirs(MODEL_PREDICT_DATA, exist_ok=True)
LOG_FILE_PATH_MAIN = os.path.join(MODEL_PREDICT_DATA, "xgboost6_opt_log_main.txt") # 主脚本日志


RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
MAX_LOOKBACK = 30
N_TRIALS = 400 # 这是总共期望的试验次数
OPTUNA_TIMEOUT = 3600 * 100 # 10 hours timeout, for example (原为3600000ms)
OPTIMIZE_METRIC = 'auc'
EARLY_STOPPING_ROUNDS_OPTUNA = 30
EARLY_STOPPING_ROUNDS_FINAL = 30

# Optuna 持久化配置
STORAGE_URL = "sqlite:///my_optimization_history.db" # 与 populate_optuna_db.py 中的一致
STUDY_NAME = "xgboost_hyperparam_search_v1"    # 与 populate_optuna_db.py 中的一致

# --- 配置日志记录 (Configure Logging) ---
logger = logging.getLogger("main_opt") # 给主脚本的logger一个不同的名字
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler_main = logging.FileHandler(LOG_FILE_PATH_MAIN, mode='a', encoding='utf-8') # mode='a' for append
stream_handler_main = logging.StreamHandler()
formatter_main = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler_main.setFormatter(formatter_main)
stream_handler_main.setFormatter(formatter_main)
logger.addHandler(file_handler_main)
logger.addHandler(stream_handler_main)

logger.info(f"Log file for main script will be saved to: {LOG_FILE_PATH_MAIN}")
logger.info(f"Optuna study: '{STUDY_NAME}', storage: '{STORAGE_URL}'")


# --- 辅助函数：计算性能指标 (保持不变) ---
def calculate_metrics(y_true, y_pred, title=""):
    # ... (内容不变) ...
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    logger.info(f"\n--- {title} Performance ---")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"  True Negatives (TN): {tn}")
    logger.info(f"  False Positives (FP): {fp}")
    logger.info(f"  False Negatives (FN): {fn}")
    logger.info(f"  True Positives (TP): {tp}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"POD (Hit Rate/Recall): {pod:.4f}")
    logger.info(f"FAR (False Alarm Ratio): {far:.4f}")
    logger.info(f"CSI (Critical Success Index): {csi:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_true, y_pred, target_names=['No Rain', 'Rain']))
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}


# --- 1. 加载数据 ---
logger.info("Loading Yangtze flattened data...")
start_load_flat = time.time()
if not (os.path.exists(X_FLAT_PATH) and os.path.exists(Y_FLAT_PATH) and os.path.exists(FEATURE_NAMES_PATH)):
    logger.error(f"Flattened data files for Yangtze dataset not found in {PROJECT_DIR}. Run turn1.py first.")
    raise FileNotFoundError(f"Flattened data files for Yangtze dataset not found in {PROJECT_DIR}. Run turn1.py first.")

try:
    logger.info(f"Attempting to load {X_FLAT_PATH}...")
    X_flat = np.load(X_FLAT_PATH)
    logger.info(f"Attempting to load {Y_FLAT_PATH}...")
    Y_flat = np.load(Y_FLAT_PATH)
except MemoryError as e:
    logger.error(f"MemoryError loading flattened data: {e}")
    logger.error("Consider using XGBoost's external memory features or reducing the dataset size if needed.")
    raise
except Exception as e:
    logger.error(f"Error loading flattened data: {e}")
    raise

with open(FEATURE_NAMES_PATH, "r") as f:
    feature_names = [line.strip() for line in f]

logger.info(f"Loaded flattened features X_flat: shape {X_flat.shape}")
logger.info(f"Loaded flattened target Y_flat: shape {Y_flat.shape}")
logger.info(f"Loaded {len(feature_names)} feature names.")
end_load_flat = time.time()
logger.info(f"Flattened data loading finished in {end_load_flat - start_load_flat:.2f} seconds.")

logger.info("\nLoading original SPATIAL Yangtze data for baseline calculation...")
start_load_orig = time.time()
original_data_loader = mydata()
# Load SPATIAL data for the basin
# X_orig_spatial shape: (n_products, time, lat, lon)
# Y_orig_spatial shape: (time, lat, lon)
X_orig_spatial, Y_orig_spatial = original_data_loader.get_basin_spatial_data(basin_mask_value=2)
product_names = original_data_loader.get_products()
n_products, nday, lat_dim, lon_dim = X_orig_spatial.shape # Get dimensions
end_load_orig = time.time()
logger.info(f"Original spatial Yangtze data loaded in {end_load_orig - start_load_orig:.2f} seconds.")
logger.info(f"X_orig_spatial shape: {X_orig_spatial.shape}, Y_orig_spatial shape: {Y_orig_spatial.shape}")



# --- 2. 准备真实标签和基线预测 (using spatial data) ---
logger.info("\nPreparing true labels and baseline predictions for Yangtze (using spatial data)...")
# Align original Y (spatial) in time
Y_orig_spatial_aligned = Y_orig_spatial[MAX_LOOKBACK:].astype(np.float32) # (valid_days, lat, lon)
# Flatten true labels AFTER aligning and masking (only consider valid points within the basin)
valid_mask_spatial = ~np.isnan(Y_orig_spatial_aligned[0]) # Use mask from first day (lat, lon)
Y_true_flat_for_baseline = Y_orig_spatial_aligned[:, valid_mask_spatial].reshape(-1) # Flatten only valid points
Y_true_binary_for_baseline = (Y_true_flat_for_baseline > RAIN_THRESHOLD).astype(int)
logger.info(f"Shape of flattened true labels for baseline (valid points): {Y_true_binary_for_baseline.shape}")

# --- 3. 计算所有基线产品的性能 (Yangtze Data - using spatial data) ---
logger.info("\nCalculating baseline performance for all products (Yangtze Data - using spatial data)...")
baseline_metrics_all = {}
# Align original X (spatial) in time
X_orig_spatial_aligned = X_orig_spatial[:, MAX_LOOKBACK:, :, :].astype(np.float32) # (prod, valid_days, lat, lon)

for i in range(n_products):
    product_name = product_names[i]
    logger.info(f"  Calculating for: {product_name}")
    # Get product data, ensure it's (valid_days, lat, lon)
    baseline_product_data_spatial = X_orig_spatial_aligned[i, :, :, :]
    # Flatten predictions AFTER aligning and masking
    baseline_pred_flat = baseline_product_data_spatial[:, valid_mask_spatial].reshape(-1) # Flatten only valid points
    baseline_pred_binary = (baseline_pred_flat > RAIN_THRESHOLD).astype(int)

    if baseline_pred_binary.shape != Y_true_binary_for_baseline.shape:
        logger.warning(f"    WARNING: Shape mismatch for {product_name}! Baseline: {baseline_pred_binary.shape}, True: {Y_true_binary_for_baseline.shape}. Skipping.")
        continue

    metrics = calculate_metrics(Y_true_binary_for_baseline, baseline_pred_binary, title=f"Baseline ({product_name}) - Yangtze Spatial")
    baseline_metrics_all[product_name] = metrics

# Clean up large original spatial arrays
del X_orig_spatial, Y_orig_spatial, X_orig_spatial_aligned, Y_orig_spatial_aligned, Y_true_flat_for_baseline
# --- 4. 准备 XGBoost 数据 (保持不变) ---
logger.info("\nPreparing data for XGBoost (Yangtze)...")
Y_binary = (Y_flat > RAIN_THRESHOLD).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, Y_binary, test_size=TEST_SIZE_RATIO, random_state=42, stratify=Y_binary
)
logger.info(f"Train set size: {len(y_train)}, Test set size: {len(y_test)}")


# --- 5. 超参数优化 (使用 Optuna) ---
logger.info("\n--- Starting Hyperparameter Optimization with Optuna ---")
start_opt_time = time.time()

def objective(trial):
    """Optuna objective function."""
    # 根据之前的寻优结果来调整这些范围
    param = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', OPTIMIZE_METRIC],
        'tree_method': 'hist',
        'verbosity': 0,
        'n_estimators': trial.suggest_int('n_estimators', 2300, 3000),       # 稍微调整并增加上限
        'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.05, log=True), # 缩小范围
        'max_depth': trial.suggest_int('max_depth', 15, 19),                 # 集中在较高深度
        'subsample': trial.suggest_float('subsample', 0.88, 0.98),           # 缩小并集中
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.80, 0.88),# 缩小并集中
        'gamma': trial.suggest_float('gamma', 0.08, 0.20),                  # 大幅缩小
        'lambda': trial.suggest_float('lambda', 1e-6, 5e-4, log=True),      # 大幅缩小上限，更集中
        'alpha': trial.suggest_float('alpha', 1e-8, 5e-5, log=True),       # 大幅缩小上限，更集中
        'scale_pos_weight': (np.sum(y_train == 0) / np.sum(y_train == 1)) if np.sum(y_train == 1) > 0 else 1,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS_OPTUNA,
        'device': 'cuda'
    }
    # 从你的日志看，n_estimators 较高，learning_rate 偏小，max_depth 较大时效果好
    # gamma, lambda, alpha 的最优值似乎也比较小

    model = xgb.XGBClassifier(**param)
    eval_set = [(X_test, y_test)] # 使用测试集进行早停和评估

    try:
        model.fit(X_train, y_train,
                  eval_set=eval_set,
                  verbose=False) # 可以设为 True 或数字来看XGBoost的迭代输出

        results = model.evals_result()
        # 确保 'validation_0' 是你的 eval_set 的名字 (XGBoost 默认)
        if OPTIMIZE_METRIC == 'auc':
            # model.best_score 是基于 n_estimators 和 early_stopping_rounds 的最终分数
            # model.best_iteration 是达到最佳分数的迭代次数
            # evals_result()['validation_0']['auc'] 是一个列表，我们需要用 best_iteration 索引
            best_score = results['validation_0'][OPTIMIZE_METRIC][model.best_iteration]
            return best_score
        else: # logloss
            best_score = results['validation_0'][OPTIMIZE_METRIC][model.best_iteration]
            return best_score # Logloss 是越小越好

    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e} for params {param}")
        # 对于最大化问题返回一个小值，对于最小化问题返回一个大值
        return 0.0 if study_direction == 'maximize' else float('inf')


study_direction = 'maximize' if OPTIMIZE_METRIC == 'auc' else 'minimize'
sampler = optuna.samplers.TPESampler(seed=42) # TPE是默认的，加种子增强可复现性

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_URL,
    direction=study_direction,
    sampler=sampler,
    load_if_exists=True  # 关键：如果存在同名study，则加载
)

# 打印已完成的试验数量
completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
logger.info(f"Study '{STUDY_NAME}' loaded/created. Already completed {completed_trials_count} trials.")

# 计算还需要运行多少 trials
# study.trials 包含所有状态的 trials (COMPLETE, RUNNING, FAIL, PRUNED)
# 我们关心的是目标 N_TRIALS 与当前 study 中所有已尝试（非 WAITING）的 trials 的关系
# 或者更简单，让 Optuna 自己处理，n_trials 参数是指总共要达到的数量
remaining_trials_to_run = N_TRIALS - len(study.trials) # 这只是一个估计，因为可能有失败的
logger.info(f"Target total trials: {N_TRIALS}. Current trials in DB (any state): {len(study.trials)}.")


if len(study.trials) < N_TRIALS:
    logger.info(f"Optimization will continue or start, aiming for a total of {N_TRIALS} trials.")
    try:
        # n_trials 是指总共要运行的试验数量。
        # 如果 study 中已有 m 个试验，Optuna 会再运行 N_TRIALS - m 个试验。
        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT,
                       # callbacks=[lambda study, trial: logger.info(f"Finished trial {trial.number} with value {trial.value}")]
                       )
    except KeyboardInterrupt:
        logger.warning("Optimization stopped manually via Ctrl+C. Progress saved to database.")
    except Exception as e:
        logger.error(f"Optimization run failed due to an error: {e}", exc_info=True)
else:
    logger.info(f"Target number of trials ({N_TRIALS}) already reached or exceeded. No new trials will be run by optimize().")


end_opt_time = time.time()
logger.info(f"Hyperparameter optimization process finished/paused in {end_opt_time - start_opt_time:.2f} seconds.")

# 获取最佳试验
try:
    best_trial = study.best_trial
    logger.info("\nBest trial found:")
    logger.info(f"  Value ({OPTIMIZE_METRIC}): {best_trial.value:.6f}")
    logger.info(f"  Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")
    best_params_from_study = best_trial.params
except ValueError:
    logger.warning("No trials completed yet, or all failed. Cannot determine best trial.")
    best_params_from_study = {} # 提供一个空字典以避免后续错误


# --- 5b. Train Final Model with OPTIMIZED Params & Early Stopping ---
# (使用从study中获取的最佳参数)
logger.info("\n--- Training FINAL model with OPTIMIZED parameters on Yangtze training data ---")
# ... (后续的最终模型训练、评估、可视化代码基本保持不变) ...
# ... 只需要确保 final_params 使用的是 study.best_params ...

if best_params_from_study: # 确保我们有最佳参数
    final_params_main = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'tree_method': 'hist',
        # 'n_estimators': 1500, # 将被 best_params_from_study 中的 n_estimators 覆盖
        'scale_pos_weight': (np.sum(y_train == 0) / np.sum(y_train == 1)) if np.sum(y_train == 1) > 0 else 1,
        'random_state': 42,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS_FINAL,
        'device': 'cuda'
    }
    # 从 study.best_trial.params 获取 n_estimators，因为这是Optuna找到的最佳值
    # 并且它是在 early_stopping_rounds_optuna 下找到的，所以它本身就是个“上限”
    final_params_main.update(best_params_from_study)

    logger.info(f"Final model parameters: {final_params_main}")

    start_train = time.time()
    final_model = xgb.XGBClassifier(**final_params_main)
    eval_set_final = [(X_test, y_test)] # 或者你可以用一个单独的验证集
    final_model.fit(X_train, y_train, eval_set=eval_set_final, verbose=50)
    end_train = time.time()
    logger.info(f"Final model training complete in {end_train - start_train:.2f} seconds.")
    logger.info(f"Best iteration for final model (from XGBoost's early stopping): {final_model.best_iteration}")
    logger.info(f"Best score for final model on eval set: {final_model.best_score}")


    # --- Save the OPTIMIZED trained model ---
    MODEL_SAVE_PATH = os.path.join(MODEL_PREDICT_DATA, "xgboost_optimized_yangtsu_model_from_db.joblib") #改名以区分
    logger.info(f"\nSaving the OPTIMIZED trained model to {MODEL_SAVE_PATH}...")
    joblib.dump(final_model, MODEL_SAVE_PATH)


    # --- 6. 评估 XGBoost 模型 (使用最终模型) ---

    # --- 6. 评估 XGBoost 模型 (Using OPTIMIZED parameters & Varying Thresholds) ---
    logger.info("\n--- Evaluating FINAL XGBoost model (Optimized Params) on the Yangtze test set with varying thresholds ---")
    start_eval = time.time()
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    thresholds_to_evaluate = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    metrics_by_threshold = {}

    logger.info("Calculating metrics for different thresholds...")
    for threshold in thresholds_to_evaluate:
        logger.info(f"\n--- Threshold: {threshold:.2f} ---")
        y_pred_test_threshold = (y_pred_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_test, y_pred_test_threshold, title=f"XGBoost Classifier Yangtze (Optimized, Threshold {threshold:.2f})")
        metrics_by_threshold[threshold] = metrics

    xgboost_metrics_optimized_threshold_05 = metrics_by_threshold[0.5]
    end_eval = time.time()
    logger.info(f"Evaluation finished in {end_eval - start_eval:.2f} seconds.")

    # --- 7. 比较基线和 XGBoost 在测试集上的性能 (Using OPTIMIZED threshold 0.5) ---
    logger.info("\n--- Final Performance Comparison (Yangtze Test Set - Optimized Threshold 0.5 vs Spatial Baseline) ---")
    metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
    comparison_data = {}

    comparison_data['XGBoost_Yangtsu_Optimized (Thr 0.5)'] = {metric: xgboost_metrics_optimized_threshold_05.get(metric, float('nan')) for metric in metrics_to_show}

    for product_name, metrics in baseline_metrics_all.items():
        comparison_data[f'Baseline_{product_name}_Spatial'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df = comparison_df[metrics_to_show]

    logger.info("\n--- Final Performance Comparison (Yangtze - Optimized Threshold 0.5 vs Spatial Baseline) ---")
    float_cols = ['accuracy', 'pod', 'far', 'csi']
    for col in float_cols:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].map('{:.4f}'.format)
    int_cols = ['fp', 'fn']
    for col in int_cols:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].map('{:.0f}'.format)

    logger.info("\n" + comparison_df.to_string())
    comparison_csv_path = os.path.join(PROJECT_DIR, "performance_comparison_yangtsu_v1_optimized.csv")
    logger.info(f"Saving comparison table to {comparison_csv_path}")
    comparison_df.to_csv(comparison_csv_path)

    # --- Optional: Display metrics for all evaluated thresholds (Optimized Model) ---
    logger.info("\n--- XGBoost Performance across different thresholds (Yangtze - Optimized Model) ---")
    threshold_metrics_data = {}
    for threshold, metrics in metrics_by_threshold.items():
        threshold_metrics_data[f'Optimized_Threshold_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

    threshold_df = pd.DataFrame(threshold_metrics_data).T
    threshold_df = threshold_df[metrics_to_show]

    for col in float_cols:
        if col in threshold_df.columns:
            threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
    for col in int_cols:
        if col in threshold_df.columns:
            threshold_df[col] = threshold_df[col].map('{:.0f}'.format)

    logger.info("\n" + threshold_df.to_string())
    threshold_csv_path = os.path.join(PROJECT_DIR, "threshold_performance_yangtsu_v1_optimized.csv")
    logger.info(f"Saving threshold performance table to {threshold_csv_path}")
    threshold_df.to_csv(threshold_csv_path)

    # --- 8. 可视化训练过程 (Using OPTIMIZED parameters) ---
    logger.info("Plotting training history (Optimized Params - Yangtze)...")
    results = final_model.evals_result()
    if 'validation_0' in results:
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].plot(x_axis, results['validation_0']['logloss'], label='Test (eval_set)')
        ax[0].legend()
        ax[0].set_ylabel('LogLoss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_title('XGBoost LogLoss (Optimized)')

        metric_to_plot = 'auc' if 'auc' in results['validation_0'] else 'error'
        if metric_to_plot in results['validation_0']:
            ax[1].plot(x_axis, results['validation_0'][metric_to_plot], label=f'Test (eval_set) - {metric_to_plot.upper()}')
            ax[1].legend()
            ax[1].set_ylabel(metric_to_plot.upper())
            ax[1].set_xlabel('Epochs')
            ax[1].set_title(f'XGBoost {metric_to_plot.upper()} (Optimized)')
        else:
            ax[1].set_title('Metric not found')

        plt.tight_layout()
        history_plot_path = os.path.join(PROJECT_DIR, "xgboost_training_history_yangtsu_v1_optimized.png")
        plt.savefig(history_plot_path)
        logger.info(f"Training history plot saved to {history_plot_path}")
        plt.close()
    else:
        logger.warning("Warning: 'validation_0' not found in evals_result(). Cannot plot history.")

    # --- 9. 特征重要性 (Using OPTIMIZED parameters) ---
    logger.info("\n--- Feature Importances (Optimized Params - Yangtze) ---")
    try:
        importances = final_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        logger.info("Top 10 Features:")
        logger.info("\n" + importance_df.head(10).to_string())

        N_TOP_FEATURES_TO_PLOT = 50
        n_features_actual = len(feature_names)
        n_plot = min(N_TOP_FEATURES_TO_PLOT, n_features_actual)

        plt.figure(figsize=(10, n_plot / 2.0))
        top_features = importance_df.head(n_plot)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.title(f"Top {n_plot} Feature Importances (XGBoost Yangtze Model - Optimized)")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        importance_plot_path = os.path.join(PROJECT_DIR, "xgboost_feature_importance_yangtsu_v1_optimized.png")
        plt.savefig(importance_plot_path)
        logger.info(f"Feature importance plot saved to {importance_plot_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Could not plot feature importance: {e}")

    logger.info("\nAnalysis complete for Yangtze dataset using OPTIMIZED parameters, evaluated multiple thresholds.")
    try:
        if optuna.visualization.is_available():
            fig_opt_history = optuna.visualization.plot_optimization_history(study)
            fig_opt_history.write_image(os.path.join(MODEL_PREDICT_DATA, "optuna_optimization_history.png"))
            logger.info(f"Optuna optimization history plot saved.")

            fig_parallel_coord = optuna.visualization.plot_parallel_coordinate(study)
            fig_parallel_coord.write_image(os.path.join(MODEL_PREDICT_DATA, "optuna_parallel_coordinate.png"))
            logger.info(f"Optuna parallel coordinate plot saved.")

            fig_slice = optuna.visualization.plot_slice(study) # 对每个参数的slice plot
            fig_slice.write_image(os.path.join(MODEL_PREDICT_DATA, "optuna_slice_plot.png"))
            logger.info(f"Optuna slice plot saved.")

            fig_contour = optuna.visualization.plot_contour(study, params=['learning_rate', 'max_depth', 'n_estimators']) #选一些重要的参数
            fig_contour.write_image(os.path.join(MODEL_PREDICT_DATA, "optuna_contour_plot.png"))
            logger.info(f"Optuna contour plot saved.")

            fig_param_importances = optuna.visualization.plot_param_importances(study)
            fig_param_importances.write_image(os.path.join(MODEL_PREDICT_DATA, "optuna_param_importances.png"))
            logger.info(f"Optuna parameter importances plot saved.")

        else:
            logger.warning("Optuna visualization is not available. Please install plotly or matplotlib.")
    except Exception as e:
        logger.error(f"Error generating Optuna visualizations: {e}")

else:
    logger.warning("No best parameters found from Optuna study. Skipping final model training and evaluation.")


logger.info("\nAnalysis complete for Yangtze dataset using OPTIMIZED parameters (from DB), evaluated multiple thresholds.")