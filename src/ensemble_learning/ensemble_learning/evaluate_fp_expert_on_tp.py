import numpy as np
import xgboost as xgb
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# --- 配置 ---
PROJECT_DIR = "F:\\rainfalldata"
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features.npy") # Full features (read-only)
INTERMEDIATE_DATA_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "intermediate_data")
TP_INDICES_PATH = os.path.join(INTERMEDIATE_DATA_DIR, 'indices_tp.npy') # Load TP indices
# --- Add path for base model probabilities on TRAIN set ---
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "predictions")
BASE_PROBS_TRAIN_PATH = os.path.join(PREDICTIONS_DIR, "base_probs_train.npy")
# ---
MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "models")
FP_EXPERT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "fp_expert_model.joblib") # Load FP expert model
OUTPUT_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "analysis") # Directory for analysis results

# --- 创建输出目录 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. 加载 TP 索引 ---
print("Loading TP indices...")
if not os.path.exists(TP_INDICES_PATH):
    print(f"Error: TP index file not found at {TP_INDICES_PATH}")
    print("Please run 1_generate_base_predictions.py first.")
    exit()
try:
    indices_tp = np.load(TP_INDICES_PATH)
except Exception as e:
    print(f"Error loading TP indices: {e}")
    exit()
print(f"Loaded {len(indices_tp)} TP indices.")
if len(indices_tp) == 0:
    print("Error: No TP samples found.")
    exit()

# --- 2. 加载 FP 专家模型 ---
print(f"Loading FP expert model from {FP_EXPERT_MODEL_PATH}...")
if not os.path.exists(FP_EXPERT_MODEL_PATH):
    print(f"Error: FP expert model not found at {FP_EXPERT_MODEL_PATH}")
    print("Please run 3_train_fp_expert.py first.")
    exit()
try:
    fp_expert_model = joblib.load(FP_EXPERT_MODEL_PATH)
except Exception as e:
    print(f"Error loading FP expert model: {e}")
    exit()

# --- 3. 加载 TP 样本特征 (包括 Base_Prob) ---
print("Loading features and base probabilities for TP samples...")
try:
    # Load original features
    X_flat = np.load(X_FLAT_PATH, mmap_mode='r')
    X_tp_data_orig = X_flat[indices_tp] # Original features
    if hasattr(X_flat, '_mmap'):
         X_flat._mmap.close()
    del X_flat

    # Load base probabilities for the training set
    if not os.path.exists(BASE_PROBS_TRAIN_PATH):
        print(f"Error: Base model training probabilities file not found at {BASE_PROBS_TRAIN_PATH}")
        exit()
    base_probs_train_full = np.load(BASE_PROBS_TRAIN_PATH)
    # Select probabilities corresponding to TP indices
    base_probs_tp = base_probs_train_full[indices_tp]
    del base_probs_train_full

    # Combine original features with base probability
    print("Combining original features and base probability for TP samples...")
    base_probs_tp_reshaped = base_probs_tp.reshape(-1, 1)
    X_tp_data = np.concatenate((X_tp_data_orig, base_probs_tp_reshaped), axis=1)
    del X_tp_data_orig, base_probs_tp, base_probs_tp_reshaped

except MemoryError as e:
    print(f"MemoryError loading feature data for TP samples: {e}")
    exit()
except Exception as e:
    print(f"Error loading feature data: {e}")
    exit()
print(f"Loaded combined features for TP samples, shape: {X_tp_data.shape}") # Should be (n_tp_samples, 101)

# --- 4. 使用 FP 专家模型预测 TP 样本 ---
print("Predicting 'is FP' probability for TP samples using FP expert model...")
try:
    # Get the probability of class 1 (which for the FP expert means "is FP")
    # Now X_tp_data has 101 features, matching the model
    tp_pred_proba_is_fp = fp_expert_model.predict_proba(X_tp_data)[:, 1]
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

print("Prediction complete.")
del X_tp_data # Free memory

# --- 5. 分析概率分布 ---
print("\n--- Analysis of 'is FP' Probabilities for Actual TP Samples ---")

# Basic statistics
print(f"Min probability: {np.min(tp_pred_proba_is_fp):.4f}")
print(f"Max probability: {np.max(tp_pred_proba_is_fp):.4f}")
print(f"Mean probability: {np.mean(tp_pred_proba_is_fp):.4f}")
print(f"Median probability: {np.median(tp_pred_proba_is_fp):.4f}")
print(f"Std Dev probability: {np.std(tp_pred_proba_is_fp):.4f}")

# Calculate percentage of TP samples exceeding certain thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\nPercentage of TP samples predicted as 'is FP' above threshold:")
for thresh in thresholds:
    count_above = np.sum(tp_pred_proba_is_fp >= thresh)
    percent_above = (count_above / len(tp_pred_proba_is_fp)) * 100
    print(f"  Threshold >= {thresh:.2f}: {count_above} / {len(tp_pred_proba_is_fp)} ({percent_above:.2f}%)")

# Plot histogram
print("\nGenerating histogram...")
try:
    plt.figure(figsize=(10, 6))
    plt.hist(tp_pred_proba_is_fp, bins=50, density=True, alpha=0.7)
    plt.xlabel("Predicted Probability of 'is FP' (by FP Expert)")
    plt.ylabel("Density")
    plt.title("Distribution of FP Expert's 'is FP' Probability for Actual TP Samples")
    plt.grid(axis='y', alpha=0.5)
    hist_path = os.path.join(OUTPUT_DIR, "fp_expert_prob_on_tp_hist.png")
    plt.savefig(hist_path)
    print(f"Histogram saved to: {hist_path}")
    plt.close()
except Exception as plot_e:
    print(f"Warning: Could not generate histogram plot - {plot_e}")

print("\nAnalysis complete.")
