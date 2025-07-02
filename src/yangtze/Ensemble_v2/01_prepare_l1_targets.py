import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- 配置 ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")

# Level 0 OOF 预测文件路径
L0_OOF_PREDS_PATH = os.path.join(PROJECT_DIR, "kfold_optimization_v6", "Train_L0_Probs_v6_Opt.npy")

# 原始 Y 标签文件路径
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target.npy")

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Ensemble_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAIN_THRESHOLD = 0.1 # 用于将原始Y转换为二分类
PRED_PROB_THRESHOLD = 0.5 # 用于将L0预测概率转换为二分类
TEST_SIZE_RATIO_HOLDOUT = 0.2 # 与L0模型训练时保持一致

# --- 1. 加载数据 ---
print("--- Step 1: Loading Level 0 OOF Predictions and True Labels ---")

# 加载原始Y标签
Y_flat_full_raw = np.load(Y_FLAT_PATH)
Y_flat_full_binary = (Y_flat_full_raw > RAIN_THRESHOLD).astype(int)

# 加载L0模型的OOF预测
L0_oof_preds = np.load(L0_OOF_PREDS_PATH)

# 确保L0_oof_preds的长度与y_train_cv_pool匹配
# 因为L0_oof_preds是针对X_train_cv_pool生成的，所以我们需要重新划分Y_flat_full_binary来获取y_train_cv_pool

# 重新划分数据以获取与L0_oof_preds对应的y_train_cv_pool
# 注意：这里我们不需要X_flat_full，只需要它的长度来正确划分Y
# 创建一个虚拟的X来帮助train_test_split保持一致的划分
virtual_X_full = np.arange(len(Y_flat_full_binary))

_, _, y_train_cv_pool, _ = train_test_split(
    virtual_X_full, Y_flat_full_binary,
    test_size=TEST_SIZE_RATIO_HOLDOUT,
    random_state=42,
    stratify=Y_flat_full_binary
)

print(f"Loaded Level 0 OOF predictions: shape {L0_oof_preds.shape}")
print(f"Loaded corresponding true labels (y_train_cv_pool): shape {y_train_cv_pool.shape}")

# --- 2. 将L0预测概率转换为二分类预测 ---
print(f"--- Step 2: Converting L0 OOF Probabilities to Binary Predictions (Threshold: {PRED_PROB_THRESHOLD}) ---")
L0_oof_binary_preds = (L0_oof_preds >= PRED_PROB_THRESHOLD).astype(int)

# --- 3. 生成TP, FP, FN, TN 标签 ---
print("--- Step 3: Generating TP, FP, FN, TN Labels ---")

is_tp = np.zeros_like(y_train_cv_pool, dtype=int)
is_fp = np.zeros_like(y_train_cv_pool, dtype=int)
is_fn = np.zeros_like(y_train_cv_pool, dtype=int)
is_tn = np.zeros_like(y_train_cv_pool, dtype=int)

# True Positives (TP): Predicted 1, True 1
is_tp[(L0_oof_binary_preds == 1) & (y_train_cv_pool == 1)] = 1

# False Positives (FP): Predicted 1, True 0
is_fp[(L0_oof_binary_preds == 1) & (y_train_cv_pool == 0)] = 1

# False Negatives (FN): Predicted 0, True 1
is_fn[(L0_oof_binary_preds == 0) & (y_train_cv_pool == 1)] = 1

# True Negatives (TN): Predicted 0, True 0
is_tn[(L0_oof_binary_preds == 0) & (y_train_cv_pool == 0)] = 1

print(f"Generated is_tp: {np.sum(is_tp)} samples (shape: {is_tp.shape})")
print(f"Generated is_fp: {np.sum(is_fp)} samples (shape: {is_fp.shape})")
print(f"Generated is_fn: {np.sum(is_fn)} samples (shape: {is_fn.shape})")
print(f"Generated is_tn: {np.sum(is_tn)} samples (shape: {is_tn.shape})")

# --- 4. 保存标签 ---
print("--- Step 4: Saving Labels ---")
np.save(os.path.join(OUTPUT_DIR, "y_is_tp.npy"), is_tp)
np.save(os.path.join(OUTPUT_DIR, "y_is_fp.npy"), is_fp)
np.save(os.path.join(OUTPUT_DIR, "y_is_fn.npy"), is_fn)
np.save(os.path.join(OUTPUT_DIR, "y_is_tn.npy"), is_tn)

print(f"TP, FP, FN, TN labels saved to {OUTPUT_DIR}")
print("\n--- Script Finished ---")
