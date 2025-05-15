import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time

# --- 配置路径 ---
KFOLD_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "results", "yangtze", "features", "kfold_optimization_v1"
)
X_TRAIN_CV_POOL_PATH = os.path.join(KFOLD_OUTPUT_DIR, "X_train_cv_pool_v1_opt.npy")

Y_TARGET_FP_EXPERT_PATH = os.path.join(KFOLD_OUTPUT_DIR, "y_target_fp_expert_v1_opt.npy") # <-- 改动
L0_PRED_LABELS_TRAIN_PATH = os.path.join(KFOLD_OUTPUT_DIR, "L0_pred_labels_train_v1_opt.npy")

OOF_FP_EXPERT_PROBS_PATH = os.path.join(KFOLD_OUTPUT_DIR, "oof_fp_expert_probs_v1_opt.npy") # <-- 改动
FP_EXPERT_MODEL_SAVE_PATH = os.path.join(KFOLD_OUTPUT_DIR, "fp_expert_model_v1_opt.joblib") # <-- 改动
FP_EXPERT_FOLDS_DIR = os.path.join(KFOLD_OUTPUT_DIR, "fp_expert_folds_v1_opt") # <-- 改动
os.makedirs(FP_EXPERT_FOLDS_DIR, exist_ok=True)

N_SPLITS_EXPERT_KFold = 5
EXPERT_MODEL_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist',
    'random_state': 789, # 不同的随机种子
    'early_stopping_rounds': 30,
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1
}

# --- 辅助函数：计算性能指标 ---
def calculate_expert_metrics(y_true, y_pred_proba, threshold=0.5, title=""):
    y_pred_labels = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_labels)
    # 对于FP Expert:
    # TP (正确识别FP): 原本是FP，专家也预测是FP
    # FP (错误识别FP): 原本是TP，专家却预测是FP (即错误地认为基础模型误报了)
    # FN (漏识别FP): 原本是FP，专家却预测不是FP (即未能识别出基础模型的误报)
    # TN (正确识别TP): 原本是TP，专家也预测不是FP
    tn, fp, fn, tp = cm.ravel() # tn=Correctly_identified_L0_TP, fp=Mistakenly_flagged_L0_TP_as_FP, fn=Failed_to_identify_L0_FP, tp=Correctly_identified_L0_FP
    accuracy = accuracy_score(y_true, y_pred_labels)
    auc = roc_auc_score(y_true, y_pred_proba)

    print(f"\n--- {title} Performance (Threshold: {threshold}) ---")
    print(f"Confusion Matrix (Expert predicting 'is FP'):\n{cm}")
    print(f"  TN (Correctly identified TP by L0): {tn}")
    print(f"  FP (Mistakenly flagged TP as FP): {fp}")
    print(f"  FN (Failed to identify L0's FP): {fn}")
    print(f"  TP (Correctly identified L0's FP): {tp}")
    print(f"Accuracy of Expert: {accuracy:.4f}")
    print(f"AUC of Expert: {auc:.4f}")
    print("\nClassification Report (Expert predicting 'is FP'):")
    # target_names 0: Not FP (i.e., L0 was TP), 1: Is FP (i.e., L0 was FP)
    print(classification_report(y_true, y_pred_labels, target_names=['Not L0_FP (is L0_TP)', 'Is L0_FP']))
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'auc': auc}


# --- 1. 加载数据 ---
print("--- Step 1: Loading Data for FP Expert Model ---") # <-- 改动
start_load_time = time.time()

if not os.path.exists(X_TRAIN_CV_POOL_PATH):
    raise FileNotFoundError(f"X_train_cv_pool_v1_opt.npy not found at: {X_TRAIN_CV_POOL_PATH}.")
X_train_cv_pool_features = np.load(X_TRAIN_CV_POOL_PATH)
print(f"Loaded X_train_cv_pool features: shape {X_train_cv_pool_features.shape}")

if not os.path.exists(Y_TARGET_FP_EXPERT_PATH): # <-- 改动
    raise FileNotFoundError(f"Target labels for FP expert not found: {Y_TARGET_FP_EXPERT_PATH}")
y_target_fp_expert_full = np.load(Y_TARGET_FP_EXPERT_PATH) # <-- 改动
print(f"Loaded FP expert target labels (full): shape {y_target_fp_expert_full.shape}") # <-- 改动

if not os.path.exists(L0_PRED_LABELS_TRAIN_PATH):
    raise FileNotFoundError(f"L0 predicted labels not found: {L0_PRED_LABELS_TRAIN_PATH}")
train_l0_pred_labels = np.load(L0_PRED_LABELS_TRAIN_PATH)
print(f"Loaded L0 predicted labels for Training/CV Pool: shape {train_l0_pred_labels.shape}")

end_load_time = time.time()
print(f"Data loading finished in {end_load_time - start_load_time:.2f} seconds.")

# --- 2. 筛选训练FP专家模型的样本子集 ---
print("\n--- Step 2: Filtering Samples for FP Expert Training ---") # <-- 改动
indices_l0_predicts_rain = np.where(train_l0_pred_labels == 1)[0] # <-- 改动: 基础模型预测为“有雨”

if len(indices_l0_predicts_rain) == 0:
    print("No samples where L0 model predicted 'Rain'. FP Expert training cannot proceed.")
    np.save(OOF_FP_EXPERT_PROBS_PATH, np.array([]))
    print("FP Expert training skipped.")
    exit()

X_fp_train_subset = X_train_cv_pool_features[indices_l0_predicts_rain] # <-- 改动
y_fp_train_subset_target = y_target_fp_expert_full[indices_l0_predicts_rain] # <-- 改动

print(f"Number of samples where L0 predicted 'Rain': {len(y_fp_train_subset_target)}") # <-- 改动
print(f"  Distribution of targets for FP Expert (1=is_FP, 0=is_TP): {np.bincount(y_fp_train_subset_target)}") # <-- 改动

# --- 3. K-Fold 交叉验证训练FP专家模型并生成折外预测 ---
print(f"\n--- Step 3: K-Fold CV for FP Expert Model ({N_SPLITS_EXPERT_KFold} splits) ---") # <-- 改动
kf_expert = StratifiedKFold(n_splits=N_SPLITS_EXPERT_KFold, shuffle=True, random_state=1011) # 不同的随机种子

oof_fp_expert_probs_subset = np.zeros(len(y_fp_train_subset_target)) # <-- 改动
fp_expert_fold_models = [] # <-- 改动

num_neg_subset = np.sum(y_fp_train_subset_target == 0) # L0的TP (FP Expert眼中的负类)
num_pos_subset = np.sum(y_fp_train_subset_target == 1) # L0的FP (FP Expert眼中的正类)
if num_pos_subset > 0 and num_neg_subset > 0:
    EXPERT_MODEL_PARAMS['scale_pos_weight'] = num_neg_subset / num_pos_subset
    print(f"Calculated scale_pos_weight for FP expert subset: {EXPERT_MODEL_PARAMS['scale_pos_weight']:.4f}")
else:
    EXPERT_MODEL_PARAMS['scale_pos_weight'] = 1
    print(f"Warning: FP expert subset has only one class or is empty. scale_pos_weight set to 1.")
    if num_pos_subset == 0 :
        print("No FP samples in the subset for FP expert training. This is unusual if L0 has FPs.")
    if num_neg_subset == 0:
        print("No TP samples in the subset for FP expert training. All L0 'Rain' predictions were FPs.")


start_kfold_expert_time = time.time()
for fold_num, (train_idx_expert, val_idx_expert) in enumerate(kf_expert.split(X_fp_train_subset, y_fp_train_subset_target)): # <-- 改动
    print(f"\n  --- FP Expert Fold {fold_num + 1}/{N_SPLITS_EXPERT_KFold} ---") # <-- 改动
    X_fold_train_expert, X_fold_val_expert = X_fp_train_subset[train_idx_expert], X_fp_train_subset[val_idx_expert] # <-- 改动
    y_fold_train_expert, y_fold_val_expert = y_fp_train_subset_target[train_idx_expert], y_fp_train_subset_target[val_idx_expert] # <-- 改动

    print(f"    Fold training set size: {len(y_fold_train_expert)}, distribution: {np.bincount(y_fold_train_expert)}")
    print(f"    Fold validation set size: {len(y_fold_val_expert)}, distribution: {np.bincount(y_fold_val_expert)}")

    fp_expert_fold_model = xgb.XGBClassifier(**EXPERT_MODEL_PARAMS) # <-- 改动
    eval_set_expert_fold = [(X_fold_val_expert, y_fold_val_expert)]

    print(f"    Fitting FP Expert model for Fold {fold_num + 1}...") # <-- 改动
    fp_expert_fold_model.fit(X_fold_train_expert, y_fold_train_expert, # <-- 改动
                             eval_set=eval_set_expert_fold,
                             verbose=100)

    oof_fp_expert_probs_subset[val_idx_expert] = fp_expert_fold_model.predict_proba(X_fold_val_expert)[:, 1] # <-- 改动
    print(f"    Fold {fold_num + 1} OOF predictions for FP Expert generated.") # <-- 改动

    calculate_expert_metrics(y_fold_val_expert, oof_fp_expert_probs_subset[val_idx_expert], # <-- 改动
                             title=f"FP Expert Fold {fold_num + 1} Validation") # <-- 改动

    fold_model_path = os.path.join(FP_EXPERT_FOLDS_DIR, f"fp_expert_v1_opt_fold_{fold_num + 1}.joblib") # <-- 改动
    joblib.dump(fp_expert_fold_model, fold_model_path) # <-- 改动
    fp_expert_fold_models.append(fp_expert_fold_model) # <-- 改动
    print(f"    Fold {fold_num + 1} FP Expert model saved to {fold_model_path}") # <-- 改动

end_kfold_expert_time = time.time()
print(f"\nFP Expert K-Fold CV finished in {end_kfold_expert_time - start_kfold_expert_time:.2f} seconds.") # <-- 改动

oof_fp_expert_probs_full = np.full(len(train_l0_pred_labels), 0.0, dtype=float) # 默认为0，或可设为0.5
oof_fp_expert_probs_full[indices_l0_predicts_rain] = oof_fp_expert_probs_subset # <-- 改动
np.save(OOF_FP_EXPERT_PROBS_PATH, oof_fp_expert_probs_full) # <-- 改动
print(f"Out-of-Fold FP Expert probabilities (full Training/CV Pool) saved to: {OOF_FP_EXPERT_PROBS_PATH}") # <-- 改动

# --- 4. 训练最终的FP专家模型 ---
print("\n--- Step 4: Training Final FP Expert Model on Full Subset ---") # <-- 改动
print(f"Fitting final FP Expert model on {len(y_fp_train_subset_target)} samples...") # <-- 改动

final_fp_expert_model_params = EXPERT_MODEL_PARAMS.copy()
if 'early_stopping_rounds' in final_fp_expert_model_params:
    del final_fp_expert_model_params['early_stopping_rounds']

final_fp_expert_model = xgb.XGBClassifier(**final_fp_expert_model_params) # <-- 改动
start_final_fp_expert_time = time.time()
final_fp_expert_model.fit(X_fp_train_subset, y_fp_train_subset_target, verbose=100) # <-- 改动
end_final_fp_expert_time = time.time()
print(f"Final FP Expert model training finished in {end_final_fp_expert_time - start_final_fp_expert_time:.2f} seconds.") # <-- 改动

joblib.dump(final_fp_expert_model, FP_EXPERT_MODEL_SAVE_PATH) # <-- 改动
print(f"Final FP Expert model saved to: {FP_EXPERT_MODEL_SAVE_PATH}") # <-- 改动

print("\n--- FP Expert Model Training Complete ---") # <-- 改动