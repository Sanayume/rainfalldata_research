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
FP_INDICES_PATH = os.path.join(INTERMEDIATE_DATA_DIR, 'indices_fp.npy') # Use FP indices
TN_INDICES_PATH = os.path.join(INTERMEDIATE_DATA_DIR, 'indices_tn.npy')
# --- Add path for base model probabilities on TRAIN set ---
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "predictions")
BASE_PROBS_TRAIN_PATH = os.path.join(PREDICTIONS_DIR, "base_probs_train.npy")
# ---
MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "models")
FP_EXPERT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "fp_expert_model.joblib") # Save FP expert model
VALIDATION_SIZE = 0.15 # Use 15% of FP+TN data for validation/early stopping
RANDOM_STATE = 42

# --- 创建模型保存目录 ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 辅助函数：计算性能指标 ---
def calculate_metrics(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

    print(f"\n--- {title} Performance ---")
    print(f"Confusion Matrix (Expert Task: 1=is_FP, 0=is_TN):\n{cm}")
    print(f"  TN (Correctly identified TN as not FP): {tn}")
    print(f"  FP (Incorrectly identified TN as FP): {fp}")
    print(f"  FN (Incorrectly identified FP as not FP): {fn}")
    print(f"  TP (Correctly identified FP as FP): {tp}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"POD (Recall for FP class): {pod:.4f}")
    print(f"FAR (How often 'predicted FP' was wrong): {far:.4f}")
    print(f"CSI (Jaccard for FP class): {csi:.4f}")
    print("\nClassification Report (Expert Task: 1=is_FP, 0=is_TN):")
    print(classification_report(y_true, y_pred, target_names=['Original TN', 'Original FP']))
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}

# --- 1. 加载所需索引 ---
print("Loading FP and TN indices...")
if not (os.path.exists(FP_INDICES_PATH) and os.path.exists(TN_INDICES_PATH)):
    print("Error: FP or TN index files not found.")
    print(f"Please run 1_generate_base_predictions.py first to generate files in {INTERMEDIATE_DATA_DIR}")
    exit()

try:
    indices_fp = np.load(FP_INDICES_PATH)
    indices_tn = np.load(TN_INDICES_PATH)
except Exception as e:
    print(f"Error loading indices: {e}")
    exit()

print(f"Loaded {len(indices_fp)} FP indices.")
print(f"Loaded {len(indices_tn)} TN indices.")

if len(indices_fp) == 0 or len(indices_tn) == 0:
    print("Error: No FP or TN samples found. Cannot train FP expert.")
    exit()

# --- 2. 准备 FP 专家训练数据 ---
print("Preparing data for FP expert...")

# 合并索引
combined_indices = np.concatenate((indices_fp, indices_tn))
# 创建目标标签 (1 for FP - actual no rain, predicted rain; 0 for TN - actual no rain, predicted no rain)
labels = np.concatenate((np.ones(len(indices_fp), dtype=int), np.zeros(len(indices_tn), dtype=int)))

# 释放不再需要的原始索引数组内存
del indices_fp, indices_tn

# --- Load Base Model Probabilities for Training Set ---
print(f"Loading base model probabilities for training set from {BASE_PROBS_TRAIN_PATH}...")
if not os.path.exists(BASE_PROBS_TRAIN_PATH):
    print("Error: Base model training probabilities file not found.")
    print("Please run 1_generate_base_predictions.py first (ensure it saves train probs).")
    exit()
try:
    # Load the full array of probabilities for the training set
    base_probs_train_full = np.load(BASE_PROBS_TRAIN_PATH)
    # Select probabilities corresponding to the combined TN and FP indices
    base_probs_expert = base_probs_train_full[combined_indices]
    del base_probs_train_full # Free memory
except Exception as e:
    print(f"Error loading or selecting base model probabilities: {e}")
    exit()
print(f"Loaded base probabilities for expert data, shape: {base_probs_expert.shape}")
if len(base_probs_expert) != len(combined_indices):
     print("Error: Mismatch between loaded base probabilities and combined indices count.")
     exit()
# ---

# 加载特征数据 (只加载需要的行)
print("Loading features for FP and TN samples (this might take time/memory)...")
try:
    X_flat = np.load(X_FLAT_PATH, mmap_mode='r')
    X_expert_data_orig = X_flat[combined_indices] # Original features
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

# --- Combine original features with base probability ---
print("Combining original features and base probability...")
# Reshape base_probs_expert to (n_samples, 1) for concatenation
base_probs_expert_reshaped = base_probs_expert.reshape(-1, 1)
# Concatenate along the feature axis (axis=1)
X_expert_data = np.concatenate((X_expert_data_orig, base_probs_expert_reshaped), axis=1)
del X_expert_data_orig, base_probs_expert, base_probs_expert_reshaped # Free memory
# ---

print(f"FP expert data shape (with base prob): X={X_expert_data.shape}, y={labels.shape}")

# --- 3. 划分训练/验证集 ---
print(f"Splitting data into training and validation sets ({1-VALIDATION_SIZE:.0%}/{VALIDATION_SIZE:.0%})...")
X_train, X_val, y_train, y_val = train_test_split(
    X_expert_data, labels,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels # Ensure proportion of FP/TN is similar in both sets
)

# 释放完整专家数据集内存
del X_expert_data, labels, combined_indices

print(f"Training set size: {len(y_train)}")
print(f"Validation set size: {len(y_val)}")

# --- 4. 定义并训练 FP 专家模型 ---
print("Defining and training FP expert model...")

# Calculate scale_pos_weight to handle imbalance between FP and TN
# Weight for the positive class (FP, label=1)
num_tn = np.sum(y_train == 0)
num_fp = np.sum(y_train == 1)
scale_pos_weight_fp = num_tn / num_fp if num_fp > 0 else 1
print(f"Calculated scale_pos_weight for FP class (label 1): {scale_pos_weight_fp:.4f}")

# Define model parameters - focus on precision for class 0 (TN) or recall for class 0
# Start with base model params and adjust
fp_expert_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc', 'error'], # Add AUC
    'use_label_encoder': False,
    'n_estimators': 500, # Can be adjusted
    'learning_rate': 0.1,
    'max_depth': 7,       # Might need adjustment
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist',
    'early_stopping_rounds': 30,
    'scale_pos_weight': scale_pos_weight_fp # Emphasizes the minority class (FP)
    # Consider if a smaller weight (closer to 1) might be better if FP is very small
    # Or adjust threshold later to favor TN prediction (label 0)
}

fp_expert_model = xgb.XGBClassifier(**fp_expert_params)

eval_set = [(X_val, y_val)]
print(f"Starting FP expert model fitting with parameters: {fp_expert_params}")
start_time = time.time()

fp_expert_model.fit(X_train, y_train, eval_set=eval_set, verbose=50)

end_time = time.time()
print(f"FP expert model training complete in {end_time - start_time:.2f} seconds.")

# --- 5. 评估模型 (在验证集上) ---
print("\n--- Evaluating FP expert model on validation set ---")
y_val_pred_proba = fp_expert_model.predict_proba(X_val)[:, 1] # Probability of being FP (label 1)

# Evaluate performance across different thresholds
thresholds_to_evaluate = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
expert_metrics_by_threshold = {}

print("Calculating FP expert metrics for different thresholds...")
for threshold in thresholds_to_evaluate:
    print(f"\n--- FP Expert Threshold: {threshold:.2f} ---")
    y_val_pred_threshold = (y_val_pred_proba >= threshold).astype(int)
    metrics = calculate_metrics(y_val, y_val_pred_threshold, title=f"FP Expert (Threshold {threshold:.2f})")
    expert_metrics_by_threshold[threshold] = metrics

# --- 展示最终性能对比 ---
print("\n--- FP Expert Performance across different thresholds (Validation Set) ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
threshold_metrics_data = {}
for threshold, metrics in expert_metrics_by_threshold.items():
    threshold_metrics_data[f'FP_Expert_Thr_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

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
print("\nFP Expert Evaluation Complete.")

# --- 6. 保存训练好的模型 ---
print(f"\nSaving the trained FP expert model to {FP_EXPERT_MODEL_PATH}...")
try:
    joblib.dump(fp_expert_model, FP_EXPERT_MODEL_PATH)
    print("FP expert model saved successfully.")
except Exception as e:
    print(f"Error saving FP expert model: {e}")

print("\nNext step: Train the meta-learner using predictions from base, FN, and FP experts.")
