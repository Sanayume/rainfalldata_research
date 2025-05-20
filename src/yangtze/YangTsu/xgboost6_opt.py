import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from loaddata import mydata
import os
import pandas as pd
import joblib
import time 
import optuna
import logging
from datetime import datetime
import numpy as np
# --- 设置中文字体 ---
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 配置 ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_Yangtsu_flat_features_v6.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target_v6.npy")
MODEL_PREDICT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "model_predict")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_yangtsu_v6.txt")

# 创建 MODEL_PREDICT_DATA 目录（如果它不存在）
os.makedirs(MODEL_PREDICT_DATA, exist_ok=True)
LOG_FILE_PATH = os.path.join(MODEL_PREDICT_DATA, "xgboost6_opt_log.txt")

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
MAX_LOOKBACK = 30
N_TRIALS = 150
OPTUNA_TIMEOUT = 3600000
OPTIMIZE_METRIC = 'auc'
EARLY_STOPPING_ROUNDS_OPTUNA = 30
EARLY_STOPPING_ROUNDS_FINAL = 30

# --- 配置日志记录 (Configure Logging) ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 清除任何现有的处理器
if logger.hasHandlers():
    logger.handlers.clear()

# 文件处理器
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
# 控制台处理器
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
# 格式化器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
# 添加处理器到记录器
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info(f"Log file will be saved to: {LOG_FILE_PATH}")

# --- 辅助函数：计算性能指标 ---
def calculate_metrics(y_true, y_pred, title=""):
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

# --- 4. 准备 XGBoost 数据 ---
logger.info("\nPreparing data for XGBoost (Yangtze)...")
Y_binary = (Y_flat > RAIN_THRESHOLD).astype(int)
n_samples = len(Y_binary)
logger.info(f"Splitting data with test_size={TEST_SIZE_RATIO} and random_state=42...")
start_split = time.time()

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, Y_binary, test_size=TEST_SIZE_RATIO, random_state=42, stratify=Y_binary
)

end_split = time.time()
logger.info(f"Data splitting finished in {end_split - start_split:.2f} seconds.")

train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)
logger.info(f"Train set size: {len(y_train)} (No Rain: {train_counts[0]}, Rain: {train_counts[1]})")
logger.info(f"Test set size: {len(y_test)} (No Rain: {test_counts[0]}, Rain: {test_counts[1]})")

# --- 5. 超参数优化 (使用 Optuna) ---
logger.info("\n--- Starting Hyperparameter Optimization with Optuna ---")
start_opt_time = time.time()

def objective(trial):
    """Optuna objective function."""
    param = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'tree_method': 'hist',
            'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 500, 2500), # 允许 Optuna 优化 n_estimators
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True), # 扩大范围
            'max_depth': trial.suggest_int('max_depth', 3, 15), # 扩大范围
            'subsample': trial.suggest_float('subsample', 0.5, 1.0), # 扩大范围
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # 扩大范围
            'gamma': trial.suggest_float('gamma', 0.0, 1.0), # 扩大范围
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True), # 扩大范围
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True), # 扩大范围
            'scale_pos_weight': (np.sum(y_train == 0) / np.sum(y_train == 1)) if np.sum(y_train == 1) > 0 else 1,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS_OPTUNA,
            'device' : 'cuda' # 如果可用
        }

    model = xgb.XGBClassifier(**param)
    eval_set = [(X_test, y_test)]

    try:
        model.fit(X_train, y_train,
                  eval_set=eval_set,
                  verbose=False)

        results = model.evals_result()
        if OPTIMIZE_METRIC == 'auc':
            best_score = results['validation_0']['auc'][model.best_iteration]
            return best_score
        else:
            best_score = results['validation_0']['logloss'][model.best_iteration]
            return best_score

    except Exception as e:
        logger.error(f"Trial failed with error: {e}")
        return float('inf') if OPTIMIZE_METRIC == 'logloss' else 0.0

study_direction = 'maximize' if OPTIMIZE_METRIC == 'auc' else 'minimize'
study = optuna.create_study(direction=study_direction)

try:
    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)
except KeyboardInterrupt:
    logger.info("Optimization stopped manually.")

end_opt_time = time.time()
logger.info(f"Hyperparameter optimization finished in {end_opt_time - start_opt_time:.2f} seconds.")

best_params = study.best_params
logger.info("\nBest hyperparameters found:")
logger.info(best_params)
logger.info(f"Best {OPTIMIZE_METRIC} value: {study.best_value:.4f}")

# --- 5b. Train Final Model with OPTIMIZED Params & Early Stopping ---
logger.info("\n--- Training FINAL model with OPTIMIZED parameters on Yangtze training data ---")
start_train = time.time()

final_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist',
    'n_estimators': 1500,
    'scale_pos_weight': (np.sum(y_train == 0) / np.sum(y_train == 1)) if np.sum(y_train == 1) > 0 else 1,
    'random_state': 42,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS_FINAL
}
final_params.update(best_params)

final_model = xgb.XGBClassifier(**final_params)
eval_set = [(X_test, y_test)]
final_model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
end_train = time.time()
logger.info(f"Final model training complete in {end_train - start_train:.2f} seconds.")
logger.info(f"Best iteration for final model: {final_model.best_iteration}")

# --- Save the OPTIMIZED trained model ---
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "xgboost_optimized_yangtsu_model.joblib")
logger.info(f"\nSaving the OPTIMIZED trained model to {MODEL_SAVE_PATH}...")
try:
    joblib.dump(final_model, MODEL_SAVE_PATH)
    logger.info("Optimized model saved successfully.")
except Exception as e:
    logger.error(f"Error saving optimized model: {e}")

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
comparison_csv_path = os.path.join(PROJECT_DIR, "performance_comparison_yangtsu_optimized.csv")
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
threshold_csv_path = os.path.join(PROJECT_DIR, "threshold_performance_yangtsu_optimized.csv")
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
    history_plot_path = os.path.join(PROJECT_DIR, "xgboost_training_history_yangtsu_optimized.png")
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

    importance_plot_path = os.path.join(PROJECT_DIR, "xgboost_feature_importance_yangtsu_optimized.png")
    plt.savefig(importance_plot_path)
    logger.info(f"Feature importance plot saved to {importance_plot_path}")
    plt.close()

except Exception as e:
    logger.error(f"Could not plot feature importance: {e}")

logger.info("\nAnalysis complete for Yangtze dataset using OPTIMIZED parameters, evaluated multiple thresholds.")