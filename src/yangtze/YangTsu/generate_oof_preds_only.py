import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import pandas as pd
import joblib
import time

# --- 配置 ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_Yangtsu_flat_features.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target.npy")
KFOLD_OUTPUT_DIR = os.path.join(PROJECT_DIR, "kfold_optimization_v6")
os.makedirs(KFOLD_OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_yangtsu.txt")
RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO_HOLDOUT = 0.2
N_SPLITS_KFold = 5
EARLY_STOPPING_ROUNDS_FINAL_MODEL = 50

def calculate_metrics(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    print(f'\n--- {title} Performance ---')
    print(f'Confusion Matrix:\n{cm}')
    print(f'  True Negatives (TN): {tn}')
    print(f'  False Positives (FP): {fp}')
    print(f'  False Negatives (FN): {fn}')
    print(f'  True Positives (TP): {tp}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'POD (Hit Rate/Recall): {pod:.4f}')
    print(f'FAR (False Alarm Ratio): {far:.4f}')
    print(f'CSI (Critical Success Index): {csi:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=['No Rain', 'Rain']))
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}

print("--- Step 1: Loading Yangtze v6 Flattened Data ---")
X_flat_full = np.load(X_FLAT_PATH, mmap_mode='r')
Y_flat_full_raw = np.load(Y_FLAT_PATH)
with open(FEATURE_NAMES_PATH, "r") as f:
    feature_names = [line.strip() for line in f]

print("--- Step 2: Data Preprocessing and Splitting ---")
Y_flat_full_binary = (Y_flat_full_raw > RAIN_THRESHOLD).astype(int)
X_train_cv_pool, X_holdout_test, y_train_cv_pool, y_holdout_test = train_test_split(
    X_flat_full, Y_flat_full_binary, test_size=TEST_SIZE_RATIO_HOLDOUT, random_state=42, stratify=Y_flat_full_binary
)

print("\n--- Step 3: Using Pre-defined Best Hyperparameters ---")
print("Skipping Optuna optimization.")
best_hyperparams = {
    'n_estimators': 2960,
    'learning_rate': 0.026319051020408163,
    'max_depth': 18,
    'subsample': 0.8985668163265306,
    'colsample_bytree': 0.846647612244898,
    'gamma': 0.09964387755102041,
    'lambda': 7.34496612e-06,
    'alpha': 1.1915502e-06
}
print("Using the following best hyperparameters from Trial 322:")
print(best_hyperparams)

print(f"\n--- Step 4: Performing {N_SPLITS_KFold}-Fold Cross-Validation with Optimized Parameters ---")
kf = StratifiedKFold(n_splits=N_SPLITS_KFold, shuffle=True, random_state=42)
oof_preds_L0_v6_Opt = np.zeros(len(y_train_cv_pool))

final_model_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist',
    'scale_pos_weight': (np.sum(y_train_cv_pool == 0) / np.sum(y_train_cv_pool == 1)) if np.sum(y_train_cv_pool == 1) > 0 else 1,
    'random_state': 42,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS_FINAL_MODEL,
    'device': 'cuda'
}
final_model_params.update(best_hyperparams)

for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train_cv_pool, y_train_cv_pool)):
    print(f"\n--- 第 {fold_num + 1}/{N_SPLITS_KFold} 折 ---")
    X_fold_train, X_fold_val = X_train_cv_pool[train_idx], X_train_cv_pool[val_idx]
    y_fold_train, y_fold_val = y_train_cv_pool[train_idx], y_train_cv_pool[val_idx]
    
    fold_model = xgb.XGBClassifier(**final_model_params)
    
    # 为了监控过拟合情况，在评估集中同时加入训练集和验证集
    # XGBoost会使用eval_set中的最后一个数据集进行早停
    eval_set_fold = [(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)]
    
    # 训练模型，并周期性地打印进度（例如每200轮）
    fold_model.fit(X_fold_train, y_fold_train, eval_set=eval_set_fold, verbose=20)
    
    # 使用最佳迭代次数的模型来预测OOF样本的概率
    oof_preds_L0_v6_Opt[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]

    # 输出此折在最佳迭代次数下的性能指标
    results = fold_model.evals_result()
    best_iter = fold_model.best_iteration

    # 'validation_0' 对应 eval_set 中的第一个元素 (训练集), 'validation_1' 对应第二个 (验证集)
    train_logloss = results['validation_0']['logloss'][best_iter]
    train_auc = results['validation_0']['auc'][best_iter]
    val_logloss = results['validation_1']['logloss'][best_iter]
    val_auc = results['validation_1']['auc'][best_iter]

    print(f"第 {fold_num + 1} 折 - 最佳迭代次数: {best_iter}")
    print(f"  训练集指标 -> LogLoss: {train_logloss:.4f}, AUC: {train_auc:.4f}")
    print(f"  验证集指标 -> LogLoss: {val_logloss:.4f}, AUC: {val_auc:.4f}")

oof_preds_save_path = os.path.join(KFOLD_OUTPUT_DIR, "Train_L0_Probs_v6_Opt.npy")
np.save(oof_preds_save_path, oof_preds_L0_v6_Opt)
print(f"Out-of-Fold predictions for Training/CV Pool saved to: {oof_preds_save_path}")
