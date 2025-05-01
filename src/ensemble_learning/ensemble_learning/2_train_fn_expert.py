import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import joblib
import time
import pandas as pd

# --- 配置 ---
PROJECT_DIR = "F:\\rainfalldata"
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features.npy") # Full features (read-only)
INTERMEDIATE_DATA_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "intermediate_data")
FN_INDICES_PATH = os.path.join(INTERMEDIATE_DATA_DIR, 'indices_fn.npy')
TN_INDICES_PATH = os.path.join(INTERMEDIATE_DATA_DIR, 'indices_tn.npy')
MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "models")
FN_EXPERT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "fn_expert_model.joblib")
VALIDATION_SIZE = 0.15 # Use 15% of FN+TN data for validation/early stopping
RANDOM_STATE = 42

# --- 创建模型保存目录 ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 辅助函数：计算性能指标 ---
def calculate_metrics(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall for class 1 (FN)
    far = fp / (tp + fp) if (tp + fp) > 0 else 0 # How many predicted FN were actually TN
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0 # Jaccard score for class 1 (FN)

    print(f"\n--- {title} Performance ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"  TN: {tn}")
    print(f"  FP: {fp}")
    print(f"  FN: {fn}")
    print(f"  TP: {tp}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"POD (Recall for FN class): {pod:.4f}")
    print(f"FAR (False Alarm Rate): {far:.4f}")
    print(f"CSI (Critical Success Index): {csi:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['TN (No Rain)', 'FN (Actual Rain)']))
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}

# --- 1. 加载所需索引 ---
print("Loading FN and TN indices...")
if not (os.path.exists(FN_INDICES_PATH) and os.path.exists(TN_INDICES_PATH)):
    print("Error: FN or TN index files not found.")
    print(f"Please run 1_generate_base_predictions.py first to generate files in {INTERMEDIATE_DATA_DIR}")
    exit()

try:
    indices_fn = np.load(FN_INDICES_PATH)
    indices_tn = np.load(TN_INDICES_PATH)
except Exception as e:
    print(f"Error loading indices: {e}")
    exit()

print(f"Loaded {len(indices_fn)} FN indices.")
print(f"Loaded {len(indices_tn)} TN indices.")

if len(indices_fn) == 0 or len(indices_tn) == 0:
    print("Error: No FN or TN samples found. Cannot train FN expert.")
    exit()

# --- 2. 准备 FN 专家训练数据 ---
print("Preparing data for FN expert...")

# 合并索引
combined_indices = np.concatenate((indices_fn, indices_tn))
# 创建目标标签 (1 for FN - actual rain, 0 for TN - actual no rain)
labels = np.concatenate((np.ones(len(indices_fn), dtype=int), np.zeros(len(indices_tn), dtype=int)))

# 释放不再需要的原始索引数组内存
del indices_fn, indices_tn

# 加载特征数据 (只加载需要的行)
print("Loading features for FN and TN samples (this might take time/memory)...")
try:
    X_flat = np.load(X_FLAT_PATH, mmap_mode='r')
    # Use advanced indexing to load only the required rows. This might still load significant data into memory.
    # Consider chunking if this causes MemoryError.
    X_expert_data = X_flat[combined_indices]
    # Close the memory map after reading
    if hasattr(X_flat, '_mmap'):
         X_flat._mmap.close()
    del X_flat
except MemoryError as e:
    print(f"MemoryError loading feature data for expert: {e}")
    print("Consider loading data in chunks based on indices.")
    exit()
except Exception as e:
    print(f"Error loading feature data: {e}")
    exit()

print(f"FN expert data shape: X={X_expert_data.shape}, y={labels.shape}")

# --- 3. 划分训练/验证集 ---
print(f"Splitting data into training and validation sets ({1-VALIDATION_SIZE:.0%}/{VALIDATION_SIZE:.0%})...")
X_train, X_val, y_train, y_val = train_test_split(
    X_expert_data, labels,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels # Ensure proportion of FN/TN is similar in both sets
)

# 释放完整专家数据集内存
del X_expert_data, labels, combined_indices

print(f"Training set size: {len(y_train)}")
print(f"Validation set size: {len(y_val)}")

# --- 4. 定义并训练 FN 专家模型 ---
print("Defining and training FN expert model...")

# Calculate scale_pos_weight to handle imbalance between FN and TN
# Weight for the positive class (FN, label=1)
num_tn = np.sum(y_train == 0)
num_fn = np.sum(y_train == 1)
scale_pos_weight_fn = num_tn / num_fn if num_fn > 0 else 1
print(f"Calculated scale_pos_weight for FN class (label 1): {scale_pos_weight_fn:.4f}")

# Define model parameters - focus on recall for class 1 (FN)
# Start with base model params and adjust
fn_expert_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc', 'error'], # Add AUC, good for imbalance
    'use_label_encoder': False,
    'n_estimators': 500, # Can be adjusted
    'learning_rate': 0.1,
    'max_depth': 7,       # Might need adjustment based on validation performance
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist',
    'early_stopping_rounds': 30,
    'scale_pos_weight': scale_pos_weight_fn # Crucial for emphasizing FN detection
}

fn_expert_model = xgb.XGBClassifier(**fn_expert_params)

eval_set = [(X_val, y_val)]
print(f"Starting FN expert model fitting with parameters: {fn_expert_params}")
start_time = time.time()

fn_expert_model.fit(X_train, y_train, eval_set=eval_set, verbose=50)

end_time = time.time()
print(f"FN expert model training complete in {end_time - start_time:.2f} seconds.")

# --- 5. 评估模型 (在验证集上) ---
print("\n--- Evaluating FN expert model on validation set ---")
y_val_pred_proba = fn_expert_model.predict_proba(X_val)[:, 1]

# Evaluate performance across different thresholds
thresholds_to_evaluate = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
expert_metrics_by_threshold = {}

print("Calculating FN expert metrics for different thresholds...")
for threshold in thresholds_to_evaluate:
    print(f"\n--- FN Expert Threshold: {threshold:.2f} ---")
    y_val_pred_threshold = (y_val_pred_proba >= threshold).astype(int)
    metrics = calculate_metrics(y_val, y_val_pred_threshold, title=f"FN Expert (Threshold {threshold:.2f})")
    expert_metrics_by_threshold[threshold] = metrics

# --- 6. 展示最终性能对比 ---
print("\n--- FN Expert Performance across different thresholds (Validation Set) ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
threshold_metrics_data = {}
for threshold, metrics in expert_metrics_by_threshold.items():
    threshold_metrics_data[f'FN_Expert_Thr_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

threshold_df = pd.DataFrame(threshold_metrics_data).T
threshold_df = threshold_df[metrics_to_show]

# Format float columns
float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols:
    if col in threshold_df.columns:
        threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
# Format integer columns
int_cols = ['fp', 'fn']
for col in int_cols:
     if col in threshold_df.columns:
         threshold_df[col] = threshold_df[col].map('{:.0f}'.format)

print(threshold_df)
print("\nFN Expert Evaluation Complete.")

# --- 7. 保存训练好的模型 ---
print(f"\nSaving the trained FN expert model to {FN_EXPERT_MODEL_PATH}...")
try:
    joblib.dump(fn_expert_model, FN_EXPERT_MODEL_PATH)
    print("FN expert model saved successfully.")
except Exception as e:
    print(f"Error saving FN expert model: {e}")

print("\nNext step: Train FP expert (using TN+FP).")

