import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import joblib
import time

# --- 配置 ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")

# 原始特征和标签路径
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_Yangtsu_flat_features.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target.npy")

# Level 1 目标标签路径 (由 01_prepare_l1_targets.py 生成)
L1_TARGETS_DIR = os.path.join(os.path.dirname(__file__), "Ensemble_v2")

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Ensemble_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO_HOLDOUT = 0.2
N_SPLITS_KFold = 5

# --- 硬编码的最佳参数 (与L0模型保持一致，或根据需要调整) ---
best_hyperparams_l1_expert = {
    'n_estimators': 2960,
    'learning_rate': 0.026319051020408163,
    'max_depth': 18,
    'subsample': 0.8985668163265306,
    'colsample_bytree': 0.846647612244898,
    'gamma': 0.09964387755102041,
    'lambda': 7.34496612e-06,
    'alpha': 1.1915502e-06,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': 42,
}

# --- 1. 加载原始特征和训练集索引 ---
print("--- Step 1: Loading Original Features and Training Set Indices ---")

X_flat_full = np.load(X_FLAT_PATH, mmap_mode='r') # 使用mmap模式加载X
Y_flat_full_raw = np.load(Y_FLAT_PATH)
Y_flat_full_binary = (Y_flat_full_raw > RAIN_THRESHOLD).astype(int)

# 重新划分数据以获取与L0_oof_preds对应的X_train_cv_pool和y_train_cv_pool
indices = np.arange(len(Y_flat_full_binary))

# 注意：这里我们只获取训练集的索引和标签，X_train_cv_pool本身不完全加载到内存
_, _, y_train_cv_pool, _ = train_test_split(
    indices, Y_flat_full_binary,
    test_size=TEST_SIZE_RATIO_HOLDOUT,
    random_state=42,
    stratify=Y_flat_full_binary
)

# 获取X_train_cv_pool的实际索引
X_train_cv_pool_indices, _, _, _ = train_test_split(
    indices, Y_flat_full_binary,
    test_size=TEST_SIZE_RATIO_HOLDOUT,
    random_state=42,
    stratify=Y_flat_full_binary
)

print(f"Loaded X_flat_full (mmap): shape {X_flat_full.shape}")
print(f"y_train_cv_pool: shape {y_train_cv_pool.shape}")
print(f"X_train_cv_pool_indices: shape {X_train_cv_pool_indices.shape}")

# --- 2. 定义专家模型及其目标 ---
expert_models_info = {
    "tp": {"target_path": os.path.join(L1_TARGETS_DIR, "y_is_tp.npy"), "oof_output_name": "l1_meta_feature_tp.npy"},
    "fp": {"target_path": os.path.join(L1_TARGETS_DIR, "y_is_fp.npy"), "oof_output_name": "l1_meta_feature_fp.npy"},
    "fn": {"target_path": os.path.join(L1_TARGETS_DIR, "y_is_fn.npy"), "oof_output_name": "l1_meta_feature_fn.npy"},
    "tn": {"target_path": os.path.join(L1_TARGETS_DIR, "y_is_tn.npy"), "oof_output_name": "l1_meta_feature_tn.npy"},
}

# --- 3. 循环训练每个专家模型并生成OOF预测 ---
print("\n--- Step 3: Training Level 1 Expert Models and Generating OOF Predictions ---")

for expert_name, info in expert_models_info.items():
    print(f"\n--- Training {expert_name.upper()} Expert Model ---")
    
    # 加载当前专家模型的目标标签
    y_expert_target = np.load(info["target_path"])
    
    # 确保目标标签的长度与y_train_cv_pool一致
    if len(y_expert_target) != len(y_train_cv_pool):
        raise ValueError(f"Length mismatch for {expert_name} target. Expected {len(y_train_cv_pool)}, got {len(y_expert_target)}")

    # 调整 scale_pos_weight 参数，因为TP/FP/FN/TN的类别不平衡可能非常严重
    # 只有当正样本存在时才计算，否则设为1
    pos_count = np.sum(y_expert_target == 1)
    neg_count = np.sum(y_expert_target == 0)
    current_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    # 复制一份参数，并更新 scale_pos_weight
    current_expert_params = best_hyperparams_l1_expert.copy()
    current_expert_params['scale_pos_weight'] = current_scale_pos_weight
    
    print(f"  Current {expert_name.upper()} scale_pos_weight: {current_scale_pos_weight:.2f}")

    kf = StratifiedKFold(n_splits=N_SPLITS_KFold, shuffle=True, random_state=42)
    oof_preds_expert = np.zeros(len(y_train_cv_pool)) # 存储当前专家模型的OOF预测

    for fold_num, (train_idx_local, val_idx_local) in enumerate(kf.split(X_train_cv_pool_indices, y_expert_target)):
        print(f"  --- Fold {fold_num + 1}/{N_SPLITS_KFold} for {expert_name.upper()} ---")
        
        # 获取在完整数据集中的真实索引
        train_idx_global = X_train_cv_pool_indices[train_idx_local]
        val_idx_global = X_train_cv_pool_indices[val_idx_local]

        # 从mmap加载数据
        X_fold_train = X_flat_full[train_idx_global]
        y_fold_train = y_expert_target[train_idx_local]
        X_fold_val = X_flat_full[val_idx_global]
        y_fold_val = y_expert_target[val_idx_local]

        # 训练模型
        model = xgb.XGBClassifier(**current_expert_params)
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=100)

        # 生成折外预测
        oof_preds_expert[val_idx_local] = model.predict_proba(X_fold_val)[:, 1]

    # 保存当前专家模型的OOF预测
    oof_expert_save_path = os.path.join(OUTPUT_DIR, info["oof_output_name"])
    np.save(oof_expert_save_path, oof_preds_expert)
    print(f"  {expert_name.upper()} Expert Model OOF predictions saved to: {oof_expert_save_path}")
    
    # (可选) 保存训练好的专家模型
    expert_model_save_path = os.path.join(OUTPUT_DIR, f"l1_expert_model_{expert_name}.joblib")
    joblib.dump(model, expert_model_save_path)
    print(f"  {expert_name.upper()} Expert Model saved to: {expert_model_save_path}")

print("\n--- Script Finished: All Level 1 Expert Models Trained and OOF Predictions Generated ---")
