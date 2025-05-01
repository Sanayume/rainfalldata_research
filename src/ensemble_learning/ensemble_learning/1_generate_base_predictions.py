import numpy as np
import xgboost as xgb
import os
import joblib # To save/load the trained model

# --- 配置 ---
PROJECT_DIR = "F:\\rainfalldata"
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features.npy") # Full features
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target.npy") # Full target
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "xgboost_default_full_model.joblib") # Path to save/load the trained model
OUTPUT_DIR_INTERMEDIATE = os.path.join(PROJECT_DIR, "ensemble_learning", "intermediate_data") # Directory for index files
OUTPUT_DIR_PREDICTIONS = os.path.join(PROJECT_DIR, "ensemble_learning", "predictions") # Directory for prediction files
BASE_PREDS_TEST_PATH = os.path.join(OUTPUT_DIR_PREDICTIONS, "base_preds.npy") # Binary 0/1 predictions for test set
BASE_PROBS_TRAIN_PATH = os.path.join(OUTPUT_DIR_PREDICTIONS, "base_probs_train.npy") # Probabilities for train set
BASE_PROBS_TEST_PATH = os.path.join(OUTPUT_DIR_PREDICTIONS, "base_probs_test.npy") # Probabilities for test set

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2 # Must match the split used in xgboost1.py
PREDICTION_THRESHOLD = 0.5 # The threshold used by the base model whose predictions we analyze

# --- 创建输出目录 ---
os.makedirs(OUTPUT_DIR_INTERMEDIATE, exist_ok=True)
os.makedirs(OUTPUT_DIR_PREDICTIONS, exist_ok=True) # Create predictions directory

# --- 1. 加载数据 (内存映射) ---
print("Loading data (memory-mapped)...")
try:
    X_flat = np.load(X_FLAT_PATH, mmap_mode='r')
    Y_flat = np.load(Y_FLAT_PATH, mmap_mode='r')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print(f"X_flat shape: {X_flat.shape}")
print(f"Y_flat shape: {Y_flat.shape}")

# --- 2. 划分训练/测试索引 (与 xgboost1.py 一致) ---
n_samples = len(Y_flat)
split_idx = int(n_samples * (1 - TEST_SIZE_RATIO))

# --- 3. 加载训练好的模型 ---
print(f"Loading trained base model from {MODEL_SAVE_PATH}...")
if not os.path.exists(MODEL_SAVE_PATH):
    print(f"Error: Model file not found at {MODEL_SAVE_PATH}.")
    print("Please run xgboost1.py first to train and save the model.")
    exit()

try:
    base_model = joblib.load(MODEL_SAVE_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Processing Training Set ---
print("\n--- Processing Training Set ---")
# --- 4. 加载训练集数据 ---
print("Accessing training data slice...")
try:
    X_train = X_flat[:split_idx]
    Y_train_raw = Y_flat[:split_idx]
    y_train_true = (Y_train_raw > RAIN_THRESHOLD).astype(int)
except Exception as e:
    print(f"Error accessing training data slice: {e}")
    exit()
print(f"Training data shape: X={X_train.shape}, y={y_train_true.shape}")
n_train_samples = len(y_train_true)

# --- 5. 对训练集进行预测 ---
print("Predicting probabilities on the training set...")
try:
    y_train_pred_proba = base_model.predict_proba(X_train)[:, 1]
except Exception as e:
    print(f"Error during training set prediction: {e}")
    exit()

# --- Save training set probabilities ---
print(f"Saving training set probabilities to {BASE_PROBS_TRAIN_PATH}...")
try:
    np.save(BASE_PROBS_TRAIN_PATH, y_train_pred_proba)
    print("Training set probabilities saved successfully.")
except Exception as e:
    print(f"Error saving training set probabilities: {e}")

print("Applying threshold...")
y_train_pred = (y_train_pred_proba >= PREDICTION_THRESHOLD).astype(int)

# --- 6. 识别 TP, TN, FP, FN 样本索引 (训练集) ---
print("Identifying TP, TN, FP, FN samples (Training Set)...")
is_tp = (y_train_true == 1) & (y_train_pred == 1)
is_tn = (y_train_true == 0) & (y_train_pred == 0)
is_fp = (y_train_true == 0) & (y_train_pred == 1)
is_fn = (y_train_true == 1) & (y_train_pred == 0)
indices_tp = np.where(is_tp)[0]
indices_tn = np.where(is_tn)[0]
indices_fp = np.where(is_fp)[0]
indices_fn = np.where(is_fn)[0]
print(f"  True Positives (TP): {len(indices_tp)}")
print(f"  True Negatives (TN): {len(indices_tn)}")
print(f"  False Positives (FP): {len(indices_fp)}")
print(f"  False Negatives (FN): {len(indices_fn)}")

# --- 7. 保存训练集索引 ---
print("Saving training set indices to files...")
np.save(os.path.join(OUTPUT_DIR_INTERMEDIATE, 'indices_tp.npy'), indices_tp)
np.save(os.path.join(OUTPUT_DIR_INTERMEDIATE, 'indices_tn.npy'), indices_tn)
np.save(os.path.join(OUTPUT_DIR_INTERMEDIATE, 'indices_fp.npy'), indices_fp)
np.save(os.path.join(OUTPUT_DIR_INTERMEDIATE, 'indices_fn.npy'), indices_fn)
print("Training set indices saved successfully.")

# --- Clear training data from memory ---
del X_train, Y_train_raw, y_train_true, y_train_pred_proba, y_train_pred
del is_tp, is_tn, is_fp, is_fn, indices_tp, indices_tn, indices_fp, indices_fn
print("Cleared training set data from memory.")

# --- Processing Test Set ---
print("\n--- Processing Test Set ---")
# --- 8. 加载测试集数据 ---
print("Accessing test data slice...")
try:
    X_test = X_flat[split_idx:]
except MemoryError as e:
    print(f"MemoryError accessing test data slice: {e}")
    print("Consider predicting on test set in chunks if it doesn't fit memory.")
    exit()
except Exception as e:
    print(f"Error accessing test data slice: {e}")
    exit()
print(f"Test data shape: X={X_test.shape}")

# --- 9. 对测试集进行预测 ---
print("Predicting probabilities on the test set...")
try:
    y_test_pred_proba = base_model.predict_proba(X_test)[:, 1]
except MemoryError as e:
    print(f"MemoryError during test set prediction: {e}")
    print("Consider predicting in chunks.")
    exit()
except Exception as e:
    print(f"Error during test set prediction: {e}")
    exit()

# --- Save test set probabilities ---
print(f"Saving test set probabilities to {BASE_PROBS_TEST_PATH}...")
try:
    np.save(BASE_PROBS_TEST_PATH, y_test_pred_proba)
    print("Test set probabilities saved successfully.")
except Exception as e:
    print(f"Error saving test set probabilities: {e}")

print("Applying threshold...")
y_test_pred = (y_test_pred_proba >= PREDICTION_THRESHOLD).astype(int)

# --- 10. 保存测试集二元预测 ---
print(f"Saving test set binary predictions to {BASE_PREDS_TEST_PATH}...")
try:
    np.save(BASE_PREDS_TEST_PATH, y_test_pred)
    print("Test set predictions saved successfully.")
except Exception as e:
    print(f"Error saving test set predictions: {e}")

# --- Cleanup ---
del X_test, y_test_pred_proba, y_test_pred
if hasattr(X_flat, '_mmap'): X_flat._mmap.close()
if hasattr(Y_flat, '_mmap'): Y_flat._mmap.close()
del X_flat, Y_flat
print("Script finished.")

