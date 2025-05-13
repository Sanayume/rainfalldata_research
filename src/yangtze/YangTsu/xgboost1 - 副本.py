import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from loaddata import mydata
import os
import pandas as pd
import joblib

# --- 配置 ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_Yangtsu_flat_features.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target.npy")
MODEL_PREDICT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "model_predict")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_yangtsu.txt")
RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
MAX_LOOKBACK = 30

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
print("Loading data...")
if not (os.path.exists(X_FLAT_PATH) and os.path.exists(Y_FLAT_PATH) and os.path.exists(FEATURE_NAMES_PATH)):
    raise FileNotFoundError(f"Flattened data files for FULL dataset not found in {PROJECT_DIR}. Run turn1.py first.")

try:
    print(f"Attempting to load {X_FLAT_PATH} (this may take time/memory)...")
    X_flat = np.load(X_FLAT_PATH, mmap_mode='r')
    print(f"Attempting to load {Y_FLAT_PATH}...")
    Y_flat = np.load(Y_FLAT_PATH, mmap_mode='r')
except MemoryError as e:
    print(f"MemoryError loading flattened data: {e}")
    print("The flattened dataset is too large to load into memory directly.")
    print("Consider using XGBoost's external memory features (e.g., DMatrix.from_npy) or reducing the dataset size.")
    raise
except Exception as e:
    print(f"Error loading flattened data: {e}")
    raise

with open(FEATURE_NAMES_PATH, "r") as f:
    feature_names = [line.strip() for line in f]

print(f"Loaded flattened features X_flat: shape {X_flat.shape}")
print(f"Loaded flattened target Y_flat: shape {Y_flat.shape}")
print(f"Loaded {len(feature_names)} feature names.")

print("Loading original data for baseline calculation (using loaddata.py's current slice)...")
original_data = mydata()
X_orig,Y_orig  = original_data.get_basin_spatial_data(2)
try:
    product_names = original_data.features
    if len(product_names) != X_orig.shape[0]:
        print("Warning: Length mismatch between features and X_orig. Using indices.")
        product_names = [f"Product_{i}" for i in range(X_orig.shape[0])]
except AttributeError:
    print("Warning: original_data has no 'features'. Using indices.")
    product_names = [f"Product_{i}" for i in range(X_orig.shape[0])]

# --- 2. 准备真实标签和基线预测 ---
print("Preparing true labels and baseline predictions...")
# Y_orig is already (time, lat, lon) from the updated loaddata.py
Y_orig_squeezed = Y_orig.astype(np.float32)  # Ensure correct dtype
nday, nlat, nlon = Y_orig_squeezed.shape
valid_mask_full = ~np.isnan(Y_orig_squeezed)
valid_mask_aligned = valid_mask_full[MAX_LOOKBACK:]
Y_true_flat_orig = Y_orig_squeezed[MAX_LOOKBACK:][valid_mask_aligned]
Y_true_binary = (Y_true_flat_orig > RAIN_THRESHOLD).astype(int)

# --- 3. 计算所有基线产品的性能 ---
print("Calculating baseline performance for all products...")
baseline_metrics_all = {}
X_orig_transposed = np.transpose(X_orig, (1, 2, 3, 0)).astype(np.float32)

for i in range(X_orig.shape[0]):
    product_name = product_names[i]
    print(f"  Calculating for: {product_name}")
    baseline_product_data = X_orig_transposed[MAX_LOOKBACK:, :, :, i]
    baseline_pred_flat = baseline_product_data[valid_mask_aligned]
    baseline_pred_binary = (baseline_pred_flat > RAIN_THRESHOLD).astype(int)
    metrics = calculate_metrics(Y_true_binary, baseline_pred_binary, title=f"Baseline ({product_name})")
    baseline_metrics_all[product_name] = metrics

# --- 4. 准备 XGBoost 数据 ---
print("Preparing data for XGBoost...")
Y_binary = (Y_flat > RAIN_THRESHOLD).astype(int)
n_samples = len(Y_binary)
split_idx = int(n_samples * (1 - TEST_SIZE_RATIO))

print("Creating train/test splits (this might take time/memory)...")
try:
    X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
    y_train, y_test = Y_binary[:split_idx], Y_binary[split_idx:]
except MemoryError as e:
    print(f"MemoryError creating train/test splits: {e}")
    print("Consider reducing TEST_SIZE_RATIO or using techniques that don't require loading full splits into memory.")
    raise
except Exception as e:
    print(f"Error creating train/test splits: {e}")
    raise

print(f"Train set size: {len(y_train)}")
print(f"Test set size: {len(y_test)}")

# --- 5. 超参数优化 (SKIPPED) ---

# --- 5b. Train Final Model with DEFAULT Params & Early Stopping ---
print("\n--- Training model with DEFAULT parameters on FULL training data (using early stopping) ---")

default_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'error'],
    'use_label_encoder': False,
    'n_estimators': 500,
    'learning_rate': 0.1,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': 42,
    'tree_method': 'hist',
    'early_stopping_rounds': 30
}

final_model = xgb.XGBClassifier(**default_params)

eval_set = [(X_test, y_test)]
print(f"Starting model fitting with default parameters: {default_params}")

final_model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
print("Model training complete.")

# --- Save the trained model ---
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "models", "xgboost_v1_yangtsu.joblib")
print(f"\nSaving the trained model to {MODEL_SAVE_PATH}...")
try:
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# --- 6. 评估 XGBoost 模型 (Using DEFAULT parameters & Varying Thresholds) ---
print("\n--- Evaluating FINAL XGBoost model (Default Params) on the test set with varying thresholds ---")
# Get predicted probabilities for the positive class (Rain)
# predict_proba returns probabilities for [No Rain, Rain]
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

thresholds_to_evaluate = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
metrics_by_threshold = {}

print("Calculating metrics for different thresholds...")
for threshold in thresholds_to_evaluate:
    print(f"\n--- Threshold: {threshold:.2f} ---")
    # Apply threshold to probabilities
    y_pred_test_threshold = (y_pred_proba >= threshold).astype(int)
    # Calculate metrics for this threshold
    metrics = calculate_metrics(y_test, y_pred_test_threshold, title=f"XGBoost Classifier (Threshold {threshold:.2f})")
    metrics_by_threshold[threshold] = metrics

# Select metrics for the default threshold (0.5) for the comparison table later
xgboost_metrics_default_threshold = metrics_by_threshold[0.5]

# --- 7. 比较基线和 XGBoost 在测试集上的性能 (Using DEFAULT threshold 0.5) ---
print("\n--- Final Performance Comparison (Test Set - Full Data, Default Threshold 0.5) ---")
baseline_test_metrics_all = {}
print("Calculating baseline performance for all products on the TEST SET...")
for i in range(X_orig.shape[0]):
    product_name = product_names[i]
    print(f"  Calculating for: {product_name} (Test Set)")
    baseline_product_data = X_orig_transposed[MAX_LOOKBACK:, :, :, i]
    baseline_pred_flat = baseline_product_data[valid_mask_aligned]
    baseline_pred_test = (baseline_pred_flat[split_idx:] > RAIN_THRESHOLD).astype(int)

    metrics = calculate_metrics(y_test, baseline_pred_test, title=f"Baseline ({product_name}) on Test Set")
    baseline_test_metrics_all[product_name] = metrics

metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
comparison_data = {}

# Add XGBoost metrics using the default threshold results
comparison_data['XGBoost_Default_Full (Thr 0.5)'] = {metric: xgboost_metrics_default_threshold.get(metric, float('nan')) for metric in metrics_to_show}

# Add all baseline metrics
for product_name, metrics in baseline_test_metrics_all.items():
    comparison_data[f'Baseline_{product_name}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

comparison_df = pd.DataFrame(comparison_data).T
comparison_df = comparison_df[metrics_to_show]

print("\n--- Final Performance Comparison (Test Set - Default Threshold 0.5) ---")
float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols:
    if col in comparison_df.columns:
        comparison_df[col] = comparison_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols:
     if col in comparison_df.columns:
         comparison_df[col] = comparison_df[col].map('{:.0f}'.format)

print(comparison_df)

# --- Optional: Display metrics for all evaluated thresholds ---
print("\n--- XGBoost Performance across different thresholds ---")
threshold_metrics_data = {}
for threshold, metrics in metrics_by_threshold.items():
    threshold_metrics_data[f'Threshold_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

threshold_df = pd.DataFrame(threshold_metrics_data).T
threshold_df = threshold_df[metrics_to_show]

# Format float columns
for col in float_cols:
    if col in threshold_df.columns:
        threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
# Format integer columns
for col in int_cols:
     if col in threshold_df.columns:
         threshold_df[col] = threshold_df[col].map('{:.0f}'.format)

print(threshold_df)

# --- 8. 可视化训练过程 (Using DEFAULT parameters) ---
print("Plotting training history (Default Params)...")
results = final_model.evals_result()
if 'validation_0' not in results:
     print("Warning: 'validation_0' not found in evals_result(). Cannot plot history.")
else:
    has_train_results = 'validation_0' in results
    if has_train_results:
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].plot(x_axis, results['validation_0']['logloss'], label='Test (eval_set[0])')
        ax[0].legend()
        ax[0].set_ylabel('LogLoss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_title('XGBoost LogLoss')

        ax[1].plot(x_axis, results['validation_0']['error'], label='Test (eval_set[0])')
        ax[1].legend()
        ax[1].set_ylabel('Classification Error (1 - Accuracy)')
        ax[1].set_xlabel('Epochs')
        ax[1].set_title('XGBoost Classification Error')

        plt.tight_layout()
        plt.savefig("xgboost_training_history_default_full.png")
        print("Training history plot saved to xgboost_training_history_default_full.png")
    else:
        print("Training history plotting skipped as only one eval set result found.")

# --- 9. 特征重要性 (Using DEFAULT parameters) ---
try:
    fig_imp, ax_imp = plt.subplots(figsize=(10, max(6, len(feature_names) // 4)))
    xgb.plot_importance(final_model, ax=ax_imp, max_num_features=50, height=0.8)
    ax_imp.set_title('XGBoost Feature Importance (Top 50 - Default Model)')
    plt.tight_layout()
    plt.savefig("xgboost_feature_importance_default_full.png")
    print("Feature importance plot saved to xgboost_feature_importance_default_full.png")
except Exception as e:
    print(f"Could not plot feature importance: {e}")

print("\nAnalysis complete using default parameters on full dataset, evaluated multiple thresholds.")
