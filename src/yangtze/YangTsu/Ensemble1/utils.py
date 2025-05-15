import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time
import pandas as pd # 用于特征重要性

# --- 通用配置 ---
# KFold 产出目录，与 xgboost1.py 和 identify_fp_fn_v1_opt.py 一致
KFOLD_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "results", "yangtze", "features", "kfold_optimization_v1"
)
# V1 特征数据 （Training/CV Pool 和 Hold-out Test Set）
V1_FEATURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "results", "yangtze", "features"
)
X_TRAIN_CV_POOL_PATH = os.path.join(V1_FEATURES_DIR, "X_Yangtsu_flat_features_train_cv_pool_v1_opt.npy") # 假设这个被保存了
X_HOLDOUT_TEST_PATH = os.path.join(V1_FEATURES_DIR, "X_Yangtsu_flat_features_holdout_test_v1_opt.npy") # 假设这个被保存了
Y_HOLDOUT_TEST_PATH = os.path.join(KFOLD_OUTPUT_DIR, "y_holdout_test_v1_opt.npy") # 假设这个被保存了


# Level 0 模型对 Training/CV Pool 的类别预测
L0_PRED_LABELS_TRAIN_PATH = os.path.join(KFOLD_OUTPUT_DIR, "L0_pred_labels_train_v1_opt.npy")
# Level 0 模型对 Hold-out Test Set 的类别预测 (需要用最终的L0模型生成)
L0_PRED_LABELS_TEST_PATH = os.path.join(KFOLD_OUTPUT_DIR, "L0_pred_labels_test_v1_opt.npy")


N_SPLITS_EXPERT_KFold = 5
RANDOM_STATE_EXPERT = 123 # 可以用不同的随机状态
EARLY_STOPPING_ROUNDS_EXPERT = 30

# XGBoost 参数 (可以为专家模型重新进行Optuna调参，或使用一组较好的默认参数)
DEFAULT_EXPERT_XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist',
    'n_estimators': 500, # 示例值，可以调整
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': RANDOM_STATE_EXPERT,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS_EXPERT
}

# --- 辅助函数：计算性能指标 (与之前脚本相同) ---
def calculate_expert_metrics(y_true, y_pred_proba, threshold=0.5, title=""):
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    # 对于专家模型，类别可能是 (例如FN专家): 0=TN (L0无雨预测正确), 1=FN (L0无雨预测错误)
    # tn_expert, fp_expert, fn_expert, tp_expert = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    print(f"\n--- {title} Performance (Threshold: {threshold:.2f}) ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    # 根据具体专家模型调整 target_names
    # 例如 FN Expert: target_names=['L0_is_TN', 'L0_is_FN']
    # 例如 FP Expert: target_names=['L0_is_TP', 'L0_is_FP']
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Class 0 (Correct L0)', 'Class 1 (Error L0)']))
    return {'accuracy': accuracy, 'auc': auc, 'cm': cm}


