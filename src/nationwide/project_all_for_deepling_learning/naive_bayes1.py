import numpy as np
from sklearn.naive_bayes import GaussianNB # Import Gaussian Naive Bayes
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.model_selection import train_test_split # Not needed for default run
import joblib
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
# Optuna is not used for default parameters

# --- 配置 ---
PROJECT_DIR = "F:/rainfalldata"
# Use v5 data files
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features_v5.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target_v5.npy")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_v5.txt") # Still load for context if needed
# Save Naive Bayes v5 model
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "naive_bayes_v5_default_model.joblib")
# IMPORTANCE_PLOT_PATH = os.path.join(PROJECT_DIR, "naive_bayes_v5_default_feature_importance.png") # Naive Bayes doesn't have direct importance plot

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42 # Not used by GaussianNB training, but keep for consistency
# EARLY_STOPPING_ROUNDS = 20 # Not applicable to Naive Bayes
# N_TOP_FEATURES_TO_PLOT = 50 # Not applicable

# --- 创建输出目录 ---
os.makedirs(PROJECT_DIR, exist_ok=True)

# --- 辅助函数：计算性能指标 ---
# ... (Copy the calculate_metrics function) ...
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
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Rain', 'Rain']))
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}

# --- 1. 加载数据 ---
print("Loading flattened data (v5)...")
try:
    # ... (Loading logic same as previous scripts) ...
    print("Attempting to load full arrays into memory...")
    X_flat = np.load(X_FLAT_PATH)
    Y_flat_raw = np.load(Y_FLAT_PATH)
    print("Successfully loaded full arrays.")
    print(f"Loaded X_flat shape: {X_flat.shape}")
    print(f"Loaded Y_flat_raw shape: {Y_flat_raw.shape}")
except FileNotFoundError:
    print(f"Error: Data files not found. Ensure {X_FLAT_PATH} and {Y_FLAT_PATH} exist.")
    print("Run turn5.py first.")
    exit()
except MemoryError:
    print("MemoryError loading full arrays. Falling back to memory mapping...")
    try:
        X_flat = np.load(X_FLAT_PATH, mmap_mode='r')
        Y_flat_raw = np.load(Y_FLAT_PATH, mmap_mode='r')
        print("Loaded using memory mapping.")
        print(f"Loaded X_flat shape: {X_flat.shape}")
        print(f"Loaded Y_flat_raw shape: {Y_flat_raw.shape}")
    except Exception as e_mmap:
        print(f"Error loading data even with mmap: {e_mmap}")
        exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. 加载特征名称 (Optional for context) ---
print(f"Loading feature names from {FEATURE_NAMES_PATH}...")
try:
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(feature_names)} feature names.")
    # No need to check against X_flat.shape[1] if not used for plotting
except Exception as e:
    print(f"Error loading feature names: {e}. Using generic names.")
    feature_names = None # Set to None if loading fails or not needed

# --- 3. 预处理 和 数据分割 ---
print("Preprocessing data...")
y_flat_binary = (Y_flat_raw > RAIN_THRESHOLD).astype(int)
if not isinstance(Y_flat_raw, np.memmap):
    del Y_flat_raw

print("Splitting data into training and testing sets...")
n_samples = X_flat.shape[0]
split_idx = int(n_samples * (1 - TEST_SIZE_RATIO))

# Use full training set directly
X_train = X_flat[:split_idx]
y_train = y_flat_binary[:split_idx]
X_test = X_flat[split_idx:]
y_test = y_flat_binary[split_idx:]

print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

# Clear original large arrays if possible
if hasattr(X_flat, '_mmap'):
    X_flat._mmap.close()
    del X_flat
elif 'X_flat' in locals() and not isinstance(X_flat, np.memmap):
    del X_flat
del y_flat_binary

# --- 4. 定义并训练 Naive Bayes 模型 ---
print("\nDefining and training Gaussian Naive Bayes model (v5 - Default Params)...")

# Instantiate the model with default parameters
model = GaussianNB()

print("Starting model training...")
start_train = time.time()
model.fit(X_train, y_train) # Train on the full training set
end_train = time.time()
print(f"Training complete in {end_train - start_train:.2f} seconds.")

# --- 5. 特征重要性 (Not Applicable) ---
# print("\n--- Feature Importances (Not Applicable for Naive Bayes) ---")

# --- 6. 评估模型 ---
print("\n--- Evaluating Model on Test Set (Naive Bayes v5 - Default Params) ---")
# GaussianNB has predict_proba, get probability for the positive class (class 1)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate across thresholds
thresholds_to_evaluate = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9] # Added more thresholds
metrics_by_threshold = {}
for threshold in thresholds_to_evaluate:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    metrics = calculate_metrics(y_test, y_pred_threshold, title=f"Naive Bayes v5 (Threshold {threshold:.2f})")
    metrics_by_threshold[threshold] = metrics

print("\n--- Naive Bayes v5 (Default) Performance across different thresholds (Test Set) ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
threshold_metrics_data = {}
for threshold, metrics in metrics_by_threshold.items():
    threshold_metrics_data[f'NB_v5_Def_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

threshold_df = pd.DataFrame(threshold_metrics_data).T
threshold_df = threshold_df[metrics_to_show]
float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols: threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols: threshold_df[col] = threshold_df[col].map('{:.0f}'.format)
print(threshold_df)

# --- 7. 保存模型 ---
print(f"\nSaving the trained Naive Bayes model to {MODEL_SAVE_PATH}...")
try:
    joblib.dump(model, MODEL_SAVE_PATH)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nScript finished.")
