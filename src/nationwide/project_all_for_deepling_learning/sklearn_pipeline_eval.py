import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split # Needed for subsetting if memory is an issue, but we'll use the full split first
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Example preprocessor, can be added if needed
import joblib
import os
import time
import pandas as pd

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC # Support Vector Classifier - Might be very slow on large data

# --- 配置 ---
PROJECT_DIR = "F:/rainfalldata"
# Use v5 data files
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features_v5.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target_v5.npy")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_v5.txt") # Optional

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
# Define a subset fraction if full training is too slow/memory intensive for some models (e.g., SVC, KNN)
# Set to 1.0 to use full training data initially
# --- MODIFICATION: Reduce the fraction to avoid MemoryError ---
TRAIN_SUBSET_FRACTION = 0.1 # Try 10% of the training data first

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
    # Suppress classification report for brevity during multi-model eval
    # print("\nClassification Report:")
    # print(classification_report(y_true, y_pred, target_names=['No Rain', 'Rain']))
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

# --- 2. 加载特征名称 (Optional) ---
# ... (Optional loading logic) ...

# --- 3. 预处理 和 数据分割 ---
print("Preprocessing data...")
y_flat_binary = (Y_flat_raw > RAIN_THRESHOLD).astype(int)
if not isinstance(Y_flat_raw, np.memmap):
    del Y_flat_raw

print("Splitting data into training and testing sets...")
n_samples = X_flat.shape[0]
split_idx = int(n_samples * (1 - TEST_SIZE_RATIO))

X_train_full = X_flat[:split_idx]
y_train_full = y_flat_binary[:split_idx]
X_test = X_flat[split_idx:]
y_test = y_flat_binary[split_idx:]

print(f"Full training set shape: X={X_train_full.shape}, y={y_train_full.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

# --- Optional: Create subset for faster evaluation if needed ---
if TRAIN_SUBSET_FRACTION < 1.0:
    print(f"\nUsing a {TRAIN_SUBSET_FRACTION*100:.0f}% subset of the training data for evaluation...")
    X_train, _, y_train, _ = train_test_split(
        X_train_full, y_train_full,
        train_size=TRAIN_SUBSET_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )
    print(f"Subset training shape: X={X_train.shape}, y={y_train.shape}")
else:
    X_train, y_train = X_train_full, y_train_full
    print("\nUsing the full training data for evaluation...")

# Clear original large arrays if possible
if hasattr(X_flat, '_mmap'):
    X_flat._mmap.close()
    del X_flat
elif 'X_flat' in locals() and not isinstance(X_flat, np.memmap):
    del X_flat
del y_flat_binary
# Keep X_train_full, y_train_full only if needed later, otherwise delete
if TRAIN_SUBSET_FRACTION < 1.0:
    del X_train_full, y_train_full

# --- 4. 定义模型和评估 ---
classifiers = {
    "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='liblinear'), # Use liblinear for L1/L2
    "GaussianNB": GaussianNB(),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, n_estimators=100), # Default 100 trees
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100), # Default 100 trees
    # "KNeighbors": KNeighborsClassifier(n_jobs=-1), # KNN can be very slow and memory intensive
    # "SVC": SVC(probability=True, random_state=RANDOM_STATE) # SVC is often O(N^2) or worse, likely too slow
}

results_summary = {}
threshold_to_report = 0.5 # Threshold for summary table

for name, classifier in classifiers.items():
    print(f"\n--- Evaluating: {name} ---")

    # Create pipeline (add StandardScaler if needed, especially for LR, KNN, SVC)
    # pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', classifier)])
    pipeline = Pipeline([('classifier', classifier)]) # No scaling for now

    print("Starting model training...")
    start_train = time.time()
    try:
        pipeline.fit(X_train, y_train)
        end_train = time.time()
        print(f"Training complete in {end_train - start_train:.2f} seconds.")

        print("Starting prediction...")
        start_pred = time.time()
        # Check if the classifier supports predict_proba
        if hasattr(pipeline, "predict_proba"):
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            end_pred = time.time()
            print(f"Prediction complete in {end_pred - start_pred:.2f} seconds.")

            # Evaluate at the specific threshold for summary
            y_pred_threshold = (y_pred_proba >= threshold_to_report).astype(int)
            metrics = calculate_metrics(y_test, y_pred_threshold, title=f"{name} (Threshold {threshold_to_report:.2f})")
            results_summary[name] = metrics

            # Optional: Evaluate across multiple thresholds (can be time-consuming to print all)
            # print(f"\n--- {name} Performance across different thresholds ---")
            # thresholds_to_evaluate = [0.3, 0.4, 0.5, 0.6, 0.7]
            # metrics_by_threshold = {}
            # for threshold in thresholds_to_evaluate:
            #     y_pred_thr = (y_pred_proba >= threshold).astype(int)
            #     metrics_thr = calculate_metrics(y_test, y_pred_thr, title=f"{name} (Threshold {threshold:.2f})")
            #     metrics_by_threshold[threshold] = metrics_thr
            # # ... (code to print threshold table if needed) ...

        else:
            print(f"{name} does not support predict_proba. Evaluating based on predict().")
            y_pred = pipeline.predict(X_test)
            end_pred = time.time()
            print(f"Prediction complete in {end_pred - start_pred:.2f} seconds.")
            # Evaluate using direct predictions (equivalent to threshold 0.5 for many binary classifiers)
            metrics = calculate_metrics(y_test, y_pred, title=f"{name} (Direct Predict)")
            results_summary[name] = metrics

    except MemoryError:
        print(f"MemoryError encountered during training or prediction for {name}. Skipping.")
        results_summary[name] = {'accuracy': np.nan, 'pod': np.nan, 'far': np.nan, 'csi': np.nan, 'fp': np.nan, 'fn': np.nan}
    except Exception as e:
        print(f"An error occurred for {name}: {e}. Skipping.")
        results_summary[name] = {'accuracy': np.nan, 'pod': np.nan, 'far': np.nan, 'csi': np.nan, 'fp': np.nan, 'fn': np.nan}


# --- 5. 结果汇总 ---
print(f"\n--- Model Comparison (Test Set - Threshold {threshold_to_report:.2f} or Direct Predict) ---")
summary_df = pd.DataFrame(results_summary).T
metrics_order = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
summary_df = summary_df[metrics_order] # Ensure consistent column order

# Format the DataFrame
float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols:
    if col in summary_df.columns:
        summary_df[col] = summary_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols:
     if col in summary_df.columns:
        summary_df[col] = summary_df[col].map('{:.0f}'.format)

print(summary_df)

print("\nScript finished.")
