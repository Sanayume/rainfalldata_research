import numpy as np
import os

# --- 配置路径 ---
# KFold 产出目录，与 xgboost1.py 中定义的一致
KFOLD_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), # 假设此脚本在类似 src/ensemble_learning/ 目录下
    "results", "yangtze", "features", "kfold_optimization_v1"
)
# 原始V1特征和目标文件路径，用于获取 y_train_cv_pool
V1_FEATURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "results", "yangtze", "features"
)
Y_FLAT_FULL_PATH = os.path.join(V1_FEATURES_DIR, "Y_Yangtsu_flat_target.npy") # V1 完整目标

# Level 0 模型的预测概率文件
L0_PROBS_TRAIN_PATH = os.path.join(KFOLD_OUTPUT_DIR, "Train_L0_Probs_V1_Opt.npy")

# 输出文件路径
FN_INDICES_PATH = os.path.join(KFOLD_OUTPUT_DIR, "fn_indices_v1_opt.npy")
FP_INDICES_PATH = os.path.join(KFOLD_OUTPUT_DIR, "fp_indices_v1_opt.npy")
TP_INDICES_PATH = os.path.join(KFOLD_OUTPUT_DIR, "tp_indices_v1_opt.npy")
TN_INDICES_PATH = os.path.join(KFOLD_OUTPUT_DIR, "tn_indices_v1_opt.npy")
Y_TARGET_FN_EXPERT_PATH = os.path.join(KFOLD_OUTPUT_DIR, "y_target_fn_expert_v1_opt.npy")
Y_TARGET_FP_EXPERT_PATH = os.path.join(KFOLD_OUTPUT_DIR, "y_target_fp_expert_v1_opt.npy")
L0_PRED_LABELS_TRAIN_PATH = os.path.join(KFOLD_OUTPUT_DIR, "L0_pred_labels_train_v1_opt.npy")


# 来自 xgboost1.py 的配置
RAIN_THRESHOLD_ORIGINAL_Y = 0.1 # 用于将原始Y转换为二进制
TEST_SIZE_RATIO_HOLDOUT = 0.2 # 与 xgboost1.py 中划分 hold-out 测试集时用的比例一致

# --- 1. 加载数据 ---
print("--- Step 1: Loading Data ---")

# 加载 Level 0 模型对 Training/CV Pool 的折外预测概率
if not os.path.exists(L0_PROBS_TRAIN_PATH):
    raise FileNotFoundError(f"Level 0 OOF predictions not found at: {L0_PROBS_TRAIN_PATH}")
train_l0_probs = np.load(L0_PROBS_TRAIN_PATH)
print(f"Loaded Level 0 OOF probabilities for Training/CV Pool: shape {train_l0_probs.shape}")

# 重新加载或生成 y_train_cv_pool (与 xgboost1.py 中的逻辑一致)
print(f"Loading full Y target from: {Y_FLAT_FULL_PATH}")
if not os.path.exists(Y_FLAT_FULL_PATH):
    raise FileNotFoundError(f"Full Y target file not found at: {Y_FLAT_FULL_PATH}")
Y_flat_full_raw = np.load(Y_FLAT_FULL_PATH)
Y_flat_full_binary = (Y_flat_full_raw > RAIN_THRESHOLD_ORIGINAL_Y).astype(int)
del Y_flat_full_raw

# 模拟 xgboost1.py 中的 train_test_split 来得到 y_train_cv_pool
# 注意：这里的 random_state 和 stratify 必须与 xgboost1.py 中的 train_test_split 完全一致
# 以确保我们得到的是完全相同的 y_train_cv_pool
# 如果 xgboost1.py 没有使用 stratify 分割出 holdout set，这里也不应使用。
# 为简单起见，假设 xgboost1.py 执行了 train_test_split(..., stratify=Y_flat_full_binary)
# 如果 X_flat_full 很大，这里可以只加载 Y_flat_full_binary 进行分割以节省内存
from sklearn.model_selection import train_test_split # 导入以进行分割

# 为了得到与 xgboost1.py 中完全一致的 y_train_cv_pool，我们需要一个虚拟的 X
# 或者，更好的做法是在 xgboost1.py 中直接保存 y_train_cv_pool.npy
# 这里我们假设在 xgboost1.py 中也保存了 y_train_cv_pool
Y_TRAIN_CV_POOL_PATH = os.path.join(KFOLD_OUTPUT_DIR, "y_train_cv_pool_v1_opt.npy") # 假设这个文件被保存了

if os.path.exists(Y_TRAIN_CV_POOL_PATH):
    print(f"Loading y_train_cv_pool from: {Y_TRAIN_CV_POOL_PATH}")
    y_train_cv_true = np.load(Y_TRAIN_CV_POOL_PATH)
else:
    print(f"y_train_cv_pool.npy not found. Re-deriving from full Y and split...")
    # 需要一个与 X_flat_full 行数相同的占位符 X 来执行相同的分割
    # 或者，如果在 xgboost1.py 中已知 X_flat_full 的形状，可以直接用其索引
    # 这是一个潜在的同步问题点，最佳实践是在 xgboost1.py 中保存 y_train_cv_pool
    print("Warning: Re-deriving y_train_cv_pool. Ensure split parameters match xgboost1.py exactly.")
    _, _, y_train_cv_true, _ = train_test_split(
        np.zeros((len(Y_flat_full_binary), 1)), # 虚拟 X
        Y_flat_full_binary,
        test_size=TEST_SIZE_RATIO_HOLDOUT,
        random_state=42, # 与 xgboost1.py 一致
        stratify=Y_flat_full_binary # 与 xgboost1.py 一致
    )
    np.save(Y_TRAIN_CV_POOL_PATH, y_train_cv_true) # 保存以便下次使用
    print(f"Derived and saved y_train_cv_pool to: {Y_TRAIN_CV_POOL_PATH}")

del Y_flat_full_binary
print(f"Loaded true labels for Training/CV Pool (y_train_cv_true): shape {y_train_cv_true.shape}")

if len(train_l0_probs) != len(y_train_cv_true):
    raise ValueError(f"Shape mismatch between L0 probabilities ({train_l0_probs.shape}) and true labels ({y_train_cv_true.shape}) for Training/CV Pool.")

# --- 2. 选择分类阈值并生成预测类别 ---
print("\n--- Step 2: Applying Classification Threshold ---")
# 这个阈值可以基于 xgboost1.py 在 hold-out test set 上的评估结果来选择
# 例如，如果阈值 0.5 或 0.55 表现良好，0.4的时候的临界回归值是最好的
classification_threshold = 0.40 # 可以根据实际情况调整
print(f"Using classification threshold for L0 model: {classification_threshold}")

train_l0_pred_labels = (train_l0_probs >= classification_threshold).astype(int)
np.save(L0_PRED_LABELS_TRAIN_PATH, train_l0_pred_labels)
print(f"Saved L0 predicted labels for Training/CV Pool to: {L0_PRED_LABELS_TRAIN_PATH}")

# --- 3. 识别 FN, FP, TP, TN 样本 ---
print("\n--- Step 3: Identifying FN, FP, TP, TN Samples ---")

# False Negatives (FN): L0 predicts No Rain (0), True is Rain (1)
fn_indices = np.where((train_l0_pred_labels == 0) & (y_train_cv_true == 1))[0]
# False Positives (FP): L0 predicts Rain (1), True is No Rain (0)
fp_indices = np.where((train_l0_pred_labels == 1) & (y_train_cv_true == 0))[0]
# True Positives (TP): L0 predicts Rain (1), True is Rain (1)
tp_indices = np.where((train_l0_pred_labels == 1) & (y_train_cv_true == 1))[0]
# True Negatives (TN): L0 predicts No Rain (0), True is No Rain (0)
tn_indices = np.where((train_l0_pred_labels == 0) & (y_train_cv_true == 0))[0]

print(f"Number of samples in Training/CV Pool: {len(y_train_cv_true)}")
print(f"  Number of True Positives (TP): {len(tp_indices)}")
print(f"  Number of False Positives (FP): {len(fp_indices)}")
print(f"  Number of True Negatives (TN): {len(tn_indices)}")
print(f"  Number of False Negatives (FN): {len(fn_indices)}")

# 验证总数是否匹配
total_classified = len(tp_indices) + len(fp_indices) + len(tn_indices) + len(fn_indices)
if total_classified != len(y_train_cv_true):
    print(f"Warning: Sum of TP,FP,TN,FN ({total_classified}) does not match total samples ({len(y_train_cv_true)})")

# 保存索引（可选，但对于调试和分析有用）
np.save(FN_INDICES_PATH, fn_indices)
np.save(FP_INDICES_PATH, fp_indices)
np.save(TP_INDICES_PATH, tp_indices)
np.save(TN_INDICES_PATH, tn_indices)
print(f"Saved FN indices to: {FN_INDICES_PATH}")
print(f"Saved FP indices to: {FP_INDICES_PATH}")
print(f"Saved TP indices to: {TP_INDICES_PATH}")
print(f"Saved TN indices to: {TN_INDICES_PATH}")


# --- 4. 创建专家模型的目标标签 ---
print("\n--- Step 4: Creating Target Labels for Expert Models ---")

# 目标标签 for FN Expert:
# y_target_fn_expert 将在基础模型预测为“无雨”的样本中，标记出哪些是真正的FN
# 对于基础模型预测为“有雨”的样本，这个标签没有直接意义，可以保持为0
y_target_fn_expert = np.zeros_like(y_train_cv_true, dtype=int)
y_target_fn_expert[fn_indices] = 1 # 标记FN为1
# sanity check: y_target_fn_expert 中为1的样本，其 train_l0_pred_labels 应该都为0
if not np.all(train_l0_pred_labels[y_target_fn_expert == 1] == 0):
    print("Error in y_target_fn_expert logic: Some FN targets don't correspond to L0 'No Rain' predictions.")
print(f"FN Expert target label distribution (for all Training/CV Pool samples): {np.bincount(y_target_fn_expert)}")
np.save(Y_TARGET_FN_EXPERT_PATH, y_target_fn_expert)
print(f"Saved target labels for FN Expert to: {Y_TARGET_FN_EXPERT_PATH}")


# 目标标签 for FP Expert:
# y_target_fp_expert 将在基础模型预测为“有雨”的样本中，标记出哪些是真正的FP
# 对于基础模型预测为“无雨”的样本，这个标签没有直接意义，可以保持为0
y_target_fp_expert = np.zeros_like(y_train_cv_true, dtype=int)
y_target_fp_expert[fp_indices] = 1 # 标记FP为1
# sanity check: y_target_fp_expert 中为1的样本，其 train_l0_pred_labels 应该都为1
if not np.all(train_l0_pred_labels[y_target_fp_expert == 1] == 1):
    print("Error in y_target_fp_expert logic: Some FP targets don't correspond to L0 'Rain' predictions.")
print(f"FP Expert target label distribution (for all Training/CV Pool samples): {np.bincount(y_target_fp_expert)}")
np.save(Y_TARGET_FP_EXPERT_PATH, y_target_fp_expert)
print(f"Saved target labels for FP Expert to: {Y_TARGET_FP_EXPERT_PATH}")

print("\n--- Stage 2 Finished ---")
print("FP/FN identification and target label creation for expert models complete.")