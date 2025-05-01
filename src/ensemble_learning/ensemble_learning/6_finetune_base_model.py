import numpy as np
import xgboost as xgb
import os
import joblib
import time
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 配置 ---
PROJECT_DIR = "F:\\rainfalldata"
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features.npy") # Full features (read-only)
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target.npy") # Full target (read-only)
INTERMEDIATE_DATA_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "intermediate_data")
FP_INDICES_PATH = os.path.join(INTERMEDIATE_DATA_DIR, 'indices_fp.npy') # FP indices from training set
FN_INDICES_PATH = os.path.join(INTERMEDIATE_DATA_DIR, 'indices_fn.npy') # FN indices from training set

MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "models")
INITIAL_MODEL_PATH = os.path.join(PROJECT_DIR, "xgboost_default_full_model.joblib") # Load the initial base model
FINETUNED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "xgboost_finetuned_model.joblib") # Save the fine-tuned model

# Data split config (must match xgboost1.py and 1_generate_base_predictions.py)
TEST_SIZE_RATIO = 0.2
RAIN_THRESHOLD = 0.1
RANDOM_STATE = 42 # Keep consistent if used elsewhere

# Fine-tuning parameters
FINETUNE_N_ESTIMATORS = 50 # Number of additional boosting rounds for fine-tuning
FINETUNE_LR = 0.02      # Lower learning rate for fine-tuning

# --- 创建模型保存目录 ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 辅助函数：计算性能指标 (Copied from script 5) ---
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

# --- 1. 加载初始模型 ---
print(f"Loading initial base model from {INITIAL_MODEL_PATH}...")
if not os.path.exists(INITIAL_MODEL_PATH):
    print("Error: Initial base model not found. Run xgboost1.py first.")
    exit()
try:
    initial_model = joblib.load(INITIAL_MODEL_PATH)
except Exception as e:
    print(f"Error loading initial model: {e}")
    exit()

# --- 2. 加载 FP 和 FN 索引 ---
print("Loading FP and FN indices from training set...")
if not (os.path.exists(FP_INDICES_PATH) and os.path.exists(FN_INDICES_PATH)):
    print("Error: FP or FN index files not found.")
    print(f"Please run 1_generate_base_predictions.py first to generate files in {INTERMEDIATE_DATA_DIR}")
    exit()
try:
    indices_fp = np.load(FP_INDICES_PATH)
    indices_fn = np.load(FN_INDICES_PATH)
except Exception as e:
    print(f"Error loading indices: {e}")
    exit()
print(f"Loaded {len(indices_fp)} FP indices and {len(indices_fn)} FN indices.")
if len(indices_fp) == 0 and len(indices_fn) == 0:
    print("Error: No FP or FN samples found for fine-tuning.")
    exit()

# --- 3. 加载并提取微调数据 ---
print("Loading features and labels for fine-tuning samples...")
try:
    X_flat = np.load(X_FLAT_PATH, mmap_mode='r')
    Y_flat = np.load(Y_FLAT_PATH, mmap_mode='r')

    # Calculate split index
    n_samples = len(Y_flat)
    split_idx = int(n_samples * (1 - TEST_SIZE_RATIO))

    # Combine FP and FN indices (these are relative to the start of the training set)
    finetune_indices = np.concatenate((indices_fp, indices_fn))
    del indices_fp, indices_fn

    # Extract features using the indices
    X_finetune = X_flat[finetune_indices]

    # Extract corresponding true labels and convert to binary
    y_finetune_raw = Y_flat[finetune_indices]
    y_finetune_true = (y_finetune_raw > RAIN_THRESHOLD).astype(int)
    del y_finetune_raw

    # Close memory maps after reading
    if hasattr(X_flat, '_mmap'): X_flat._mmap.close()
    if hasattr(Y_flat, '_mmap'): Y_flat._mmap.close()
    del X_flat, Y_flat

except MemoryError as e:
    print(f"MemoryError loading or extracting fine-tuning data: {e}")
    exit()
except Exception as e:
    print(f"Error loading or extracting fine-tuning data: {e}")
    exit()

print(f"Fine-tuning data shape: X={X_finetune.shape}, y={y_finetune_true.shape}")

# --- 4. 定义微调参数并进行微调 ---
print("Defining fine-tuning parameters...")

# Get initial parameters but override learning rate and n_estimators for fine-tuning stage
params = initial_model.get_params()
params['learning_rate'] = FINETUNE_LR
params['n_estimators'] = FINETUNE_N_ESTIMATORS # Add this many *more* estimators

# --- Remove early stopping for fine-tuning ---
if 'early_stopping_rounds' in params:
    print(f"Removing 'early_stopping_rounds' ({params['early_stopping_rounds']}) for fine-tuning.")
    params['early_stopping_rounds'] = None
# ---

# Recalculate scale_pos_weight based *only* on the fine-tuning subset
num_neg_finetune = np.sum(y_finetune_true == 0) # Original FP samples
num_pos_finetune = np.sum(y_finetune_true == 1) # Original FN samples
if num_pos_finetune > 0:
    scale_pos_weight_finetune = num_neg_finetune / num_pos_finetune
    print(f"Calculated scale_pos_weight for fine-tuning set: {scale_pos_weight_finetune:.4f}")
    params['scale_pos_weight'] = scale_pos_weight_finetune
else:
    print("Warning: No positive samples (original FN) in fine-tuning set. Using scale_pos_weight=1.")
    params['scale_pos_weight'] = 1

# Create a new classifier instance with updated parameters
finetune_model = xgb.XGBClassifier(**params)

print(f"Starting fine-tuning for {FINETUNE_N_ESTIMATORS} rounds with LR={FINETUNE_LR}...")
start_time = time.time()

# Use xgb_model parameter to continue training from the initial model state
# No eval_set needed here as early stopping is disabled
finetune_model.fit(X_finetune, y_finetune_true, xgb_model=initial_model, verbose=False)

end_time = time.time()
print(f"Fine-tuning complete in {end_time - start_time:.2f} seconds.")

# --- 5. 保存微调后的模型 ---
print(f"Saving fine-tuned model to {FINETUNED_MODEL_PATH}...")
try:
    joblib.dump(finetune_model, FINETUNED_MODEL_PATH)
    print("Fine-tuned model saved successfully.")
except Exception as e:
    print(f"Error saving fine-tuned model: {e}")

# --- 6. 评估微调后的模型 (在原始测试集上) ---
print("\n--- Evaluating Fine-tuned Model on Original Test Set ---")
y_test_true = None
y_test_pred_proba_list = [] # List to store chunk predictions
CHUNK_SIZE_EVAL = 500000 # Adjust chunk size based on available memory

try:
    # --- Load only the true labels for the test set ---
    print("Loading test set true labels...")
    Y_flat_test_mmap = np.load(Y_FLAT_PATH, mmap_mode='r')
    Y_flat_test_raw = Y_flat_test_mmap[split_idx:]
    y_test_true = (Y_flat_test_raw > RAIN_THRESHOLD).astype(int)
    n_test_samples = len(y_test_true)
    print(f"  Test target loaded, shape: {y_test_true.shape}")
    if hasattr(Y_flat_test_mmap, '_mmap'): Y_flat_test_mmap._mmap.close()
    del Y_flat_test_raw, Y_flat_test_mmap
    # ---

    # --- Predict probabilities in chunks ---
    print(f"Predicting probabilities on test set in chunks of size {CHUNK_SIZE_EVAL}...")
    X_flat_test_mmap = np.load(X_FLAT_PATH, mmap_mode='r')
    start_pred_time = time.time()
    processed_samples = 0

    for i_chunk, start_idx_chunk in enumerate(range(0, n_test_samples, CHUNK_SIZE_EVAL)):
        end_idx_chunk = min(start_idx_chunk + CHUNK_SIZE_EVAL, n_test_samples)
        current_chunk_size = end_idx_chunk - start_idx_chunk
        print(f"  Processing chunk {i_chunk + 1}: samples {start_idx_chunk} to {end_idx_chunk-1} (size {current_chunk_size})")

        # Load chunk features from mmap file
        # Need to adjust index relative to the start of the test set slice
        mmap_start_idx = split_idx + start_idx_chunk
        mmap_end_idx = split_idx + end_idx_chunk
        X_chunk = X_flat_test_mmap[mmap_start_idx:mmap_end_idx]
        print(f"    Loaded chunk features, shape: {X_chunk.shape}")

        try:
            # Predict probabilities for the chunk
            chunk_pred_proba = finetune_model.predict_proba(X_chunk)[:, 1]
            y_test_pred_proba_list.append(chunk_pred_proba)
            processed_samples += current_chunk_size
            print(f"    Prediction successful for chunk {i_chunk + 1}.")
        except MemoryError as pred_mem_err:
            print(f"    MEMORY ERROR during prediction for chunk {i_chunk + 1}: {pred_mem_err}")
            print(f"    Chunk shape was: {X_chunk.shape}")
            # Clean up and exit if a chunk fails
            del X_chunk, y_test_pred_proba_list
            if hasattr(X_flat_test_mmap, '_mmap'): X_flat_test_mmap._mmap.close()
            exit()
        except Exception as pred_err:
            print(f"    ERROR during prediction for chunk {i_chunk + 1}: {pred_err}")
            # Clean up and exit if a chunk fails
            del X_chunk, y_test_pred_proba_list
            if hasattr(X_flat_test_mmap, '_mmap'): X_flat_test_mmap._mmap.close()
            exit()
        finally:
             # Clear chunk data in each iteration
             del X_chunk

    end_pred_time = time.time()
    if hasattr(X_flat_test_mmap, '_mmap'): X_flat_test_mmap._mmap.close() # Close mmap after loop
    del X_flat_test_mmap
    print(f"  Finished processing all chunks in {end_pred_time - start_pred_time:.2f} seconds.")
    print(f"  Total samples processed: {processed_samples}")

    # Concatenate predictions from all chunks
    if processed_samples == n_test_samples and y_test_pred_proba_list:
        print("Concatenating chunk predictions...")
        y_test_pred_proba = np.concatenate(y_test_pred_proba_list)
        del y_test_pred_proba_list # Free list memory
        print(f"  Final predicted probabilities shape: {y_test_pred_proba.shape}")
    else:
        print("Error: Not all samples were processed or prediction list is empty. Cannot proceed.")
        exit()
    # --- End chunk prediction ---


    # --- Evaluate performance across different thresholds (using concatenated y_test_pred_proba) ---
    thresholds_to_evaluate = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    finetuned_metrics_by_threshold = {}

    print("Calculating fine-tuned model metrics for different thresholds...")
    start_calc_time = time.time()
    for threshold in thresholds_to_evaluate:
        print(f"\n--- Fine-tuned Model Threshold: {threshold:.2f} ---")
        try:
            y_test_pred_threshold = (y_test_pred_proba >= threshold).astype(int)
            metrics = calculate_metrics(y_test_true, y_test_pred_threshold, title=f"Fine-tuned Model (Threshold {threshold:.2f})")
            finetuned_metrics_by_threshold[threshold] = metrics
        except MemoryError as calc_mem_err:
             print(f"  MEMORY ERROR during metric calculation for threshold {threshold:.2f}: {calc_mem_err}")
             break
        except Exception as calc_err:
             print(f"  ERROR during metric calculation for threshold {threshold:.2f}: {calc_err}")
             break
    end_calc_time = time.time()
    print(f"\nMetric calculation loop finished in {end_calc_time - start_calc_time:.2f} seconds.")

    # --- 展示最终性能对比 ---
    if finetuned_metrics_by_threshold: # Check if any metrics were calculated
        print("\n--- Fine-tuned Model Performance across different thresholds (Test Set) ---")
        metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
        threshold_metrics_data = {}
        for threshold, metrics in finetuned_metrics_by_threshold.items():
            threshold_metrics_data[f'Finetuned_Thr_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

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
        print("\nCompare this table with the original single XGBoost model performance.")
    else:
        print("\nNo metrics were calculated, likely due to errors during evaluation.")


except MemoryError as e:
    print(f"MEMORY ERROR during test set loading or preparation: {e}")
except Exception as e:
    print(f"ERROR during test set evaluation: {e}")
finally:
    # Attempt to clean up memory regardless of errors
    print("Cleaning up evaluation variables...")
    # Set variables to None to help garbage collection and avoid potential Pylance issues with 'del'
    y_test_true = None
    y_test_pred_proba = None
    y_test_pred_proba_list = None
    # Ensure X_chunk is also handled if loop breaks early
    if 'X_chunk' in locals(): del X_chunk


print("\nFine-tuning script finished.")
