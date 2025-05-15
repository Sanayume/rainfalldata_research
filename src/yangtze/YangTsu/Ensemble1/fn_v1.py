import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time

# --- 配置路径 ---
# KFold 产出目录，与 xgboost1.py 和 identify_fp_fn_v1_opt.py 一致
KFOLD_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "results", "yangtze", "features", "kfold_optimization_v1"
)
# 原始V1特征 (Training/CV Pool 部分)
# 假设在 xgboost1.py 中保存了 X_train_cv_pool.npy
X_TRAIN_CV_POOL_PATH = os.path.join(KFOLD_OUTPUT_DIR, "X_train_cv_pool_v1_opt.npy") # 需要确保这个文件存在

# 输入文件 (来自 identify_fp_fn_v1_opt.py)
Y_TARGET_FN_EXPERT_PATH = os.path.join(KFOLD_OUTPUT_DIR, "y_target_fn_expert_v1_opt.npy")
L0_PRED_LABELS_TRAIN_PATH = os.path.join(KFOLD_OUTPUT_DIR, "L0_pred_labels_train_v1_opt.npy")

# 输出文件
OOF_FN_EXPERT_PROBS_PATH = os.path.join(KFOLD_OUTPUT_DIR, "oof_fn_expert_probs_v1_opt.npy")
FN_EXPERT_MODEL_SAVE_PATH = os.path.join(KFOLD_OUTPUT_DIR, "fn_expert_model_v1_opt.joblib")
FN_EXPERT_FOLDS_DIR = os.path.join(KFOLD_OUTPUT_DIR, "fn_expert_folds_v1_opt")
os.makedirs(FN_EXPERT_FOLDS_DIR, exist_ok=True)

# 模型训练配置
N_SPLITS_EXPERT_KFold = 5
EXPERT_MODEL_PARAMS = { # 可以使用与基础模型相似的参数，或重新为专家模型调参
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist',
    'random_state': 123, # 不同的随机种子以示区别
    'early_stopping_rounds': 30,
    # 以下参数可以从 xgboost1.py 的 best_hyperparams 获取或重新设置
    'n_estimators': 1000, # 示例值，需要根据早停调整
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1
    # scale_pos_weight 将在下面根据子集重新计算
}

# --- 辅助函数：计算性能指标 (保持不变或简化) ---
def calculate_expert_metrics(y_true, y_pred_proba, threshold=0.5, title=""):
    y_pred_labels = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_labels)
    # 对于FN Expert:
    # TP (正确识别FN): 原本是FN，专家也预测是FN
    # FP (错误识别FN): 原本是TN，专家却预测是FN (即错误地认为基础模型漏报了)
    # FN (漏识别FN): 原本是FN，专家却预测不是FN (即未能识别出基础模型的漏报)
    # TN (正确识别TN): 原本是TN，专家也预测不是FN
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred_labels)
    auc = roc_auc_score(y_true, y_pred_proba)

    print(f"\n--- {title} Performance (Threshold: {threshold}) ---")
    print(f"Confusion Matrix (Expert predicting 'is FN'):\n{cm}")
    print(f"  TN (Correctly identified TN by L0): {tn}")
    print(f"  FP (Mistakenly flagged TN as FN): {fp}")
    print(f"  FN (Failed to identify L0's FN): {fn}")
    print(f"  TP (Correctly identified L0's FN): {tp}")
    print(f"Accuracy of Expert: {accuracy:.4f}")
    print(f"AUC of Expert: {auc:.4f}")
    print("\nClassification Report (Expert predicting 'is FN'):")
    # target_names 0: Not FN (i.e., L0 was TN), 1: Is FN (i.e., L0 was FN)
    print(classification_report(y_true, y_pred_labels, target_names=['Not L0_FN (is L0_TN)', 'Is L0_FN']))
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'auc': auc}


# --- 1. 加载数据 ---
print("--- Step 1: Loading Data for FN Expert Model ---")
start_load_time = time.time()

if not os.path.exists(X_TRAIN_CV_POOL_PATH):
    raise FileNotFoundError(f"X_train_cv_pool_v1_opt.npy not found at: {X_TRAIN_CV_POOL_PATH}. Ensure it was saved by xgboost1.py.")
X_train_cv_pool_features = np.load(X_TRAIN_CV_POOL_PATH)
print(f"Loaded X_train_cv_pool features: shape {X_train_cv_pool_features.shape}")

if not os.path.exists(Y_TARGET_FN_EXPERT_PATH):
    raise FileNotFoundError(f"Target labels for FN expert not found: {Y_TARGET_FN_EXPERT_PATH}")
y_target_fn_expert_full = np.load(Y_TARGET_FN_EXPERT_PATH)
print(f"Loaded FN expert target labels (full): shape {y_target_fn_expert_full.shape}")

if not os.path.exists(L0_PRED_LABELS_TRAIN_PATH):
    raise FileNotFoundError(f"L0 predicted labels not found: {L0_PRED_LABELS_TRAIN_PATH}")
train_l0_pred_labels = np.load(L0_PRED_LABELS_TRAIN_PATH)
print(f"Loaded L0 predicted labels for Training/CV Pool: shape {train_l0_pred_labels.shape}")

end_load_time = time.time()
print(f"Data loading finished in {end_load_time - start_load_time:.2f} seconds.")

# --- 2. 筛选训练FN专家模型的样本子集 ---
# FN专家模型只在基础模型预测为“无雨” (label=0) 的样本上进行训练和预测
# 目标是区分这些“无雨”预测中，哪些是真正的FN (target=1)，哪些是真正的TN (target=0)
print("\n--- Step 2: Filtering Samples for FN Expert Training ---")
indices_l0_predicts_no_rain = np.where(train_l0_pred_labels == 0)[0]

if len(indices_l0_predicts_no_rain) == 0:
    print("No samples where L0 model predicted 'No Rain'. FN Expert training cannot proceed.")
    # 创建空的输出文件或进行其他适当处理
    np.save(OOF_FN_EXPERT_PROBS_PATH, np.array([]))
    # joblib.dump(None, FN_EXPERT_MODEL_SAVE_PATH) # 不能保存None
    print("FN Expert training skipped.")
    exit()

X_fn_train_subset = X_train_cv_pool_features[indices_l0_predicts_no_rain]
# y_target_fn_expert_full 中，只有在 fn_indices 上的值为1
# 当我们筛选 train_l0_pred_labels == 0 的样本时，
# y_target_fn_expert_full 在这些样本上的值，1代表FN，0代表TN
y_fn_train_subset_target = y_target_fn_expert_full[indices_l0_predicts_no_rain]

print(f"Number of samples where L0 predicted 'No Rain': {len(y_fn_train_subset_target)}")
print(f"  Distribution of targets for FN Expert (1=is_FN, 0=is_TN): {np.bincount(y_fn_train_subset_target)}")

# --- 3. K-Fold 交叉验证训练FN专家模型并生成折外预测 ---
print(f"\n--- Step 3: K-Fold CV for FN Expert Model ({N_SPLITS_EXPERT_KFold} splits) ---")
kf_expert = StratifiedKFold(n_splits=N_SPLITS_EXPERT_KFold, shuffle=True, random_state=456) # 不同的随机种子

# 初始化用于存储FN专家模型对筛选出的“无雨”子集的折外预测的数组
# 长度等于 X_fn_train_subset 的样本数
oof_fn_expert_probs_subset = np.zeros(len(y_fn_train_subset_target))
fn_expert_fold_models = []

# 动态计算 scale_pos_weight for the subset
num_neg_subset = np.sum(y_fn_train_subset_target == 0) # L0的TN
num_pos_subset = np.sum(y_fn_train_subset_target == 1) # L0的FN
if num_pos_subset > 0 and num_neg_subset > 0 :
    EXPERT_MODEL_PARAMS['scale_pos_weight'] = num_neg_subset / num_pos_subset
    print(f"Calculated scale_pos_weight for FN expert subset: {EXPERT_MODEL_PARAMS['scale_pos_weight']:.4f}")
else:
    EXPERT_MODEL_PARAMS['scale_pos_weight'] = 1
    print(f"Warning: FN expert subset has only one class or is empty. scale_pos_weight set to 1.")
    if num_pos_subset == 0 :
        print("No FN samples in the subset for FN expert training. This is unusual if L0 has FNs.")
        # 训练可能无意义，但流程继续以生成文件
    if num_neg_subset == 0:
        print("No TN samples in the subset for FN expert training. All L0 'No Rain' predictions were FNs.")


start_kfold_expert_time = time.time()
for fold_num, (train_idx_expert, val_idx_expert) in enumerate(kf_expert.split(X_fn_train_subset, y_fn_train_subset_target)):
    print(f"\n  --- FN Expert Fold {fold_num + 1}/{N_SPLITS_EXPERT_KFold} ---")
    X_fold_train_expert, X_fold_val_expert = X_fn_train_subset[train_idx_expert], X_fn_train_subset[val_idx_expert]
    y_fold_train_expert, y_fold_val_expert = y_fn_train_subset_target[train_idx_expert], y_fn_train_subset_target[val_idx_expert]

    print(f"    Fold training set size: {len(y_fold_train_expert)}, distribution: {np.bincount(y_fold_train_expert)}")
    print(f"    Fold validation set size: {len(y_fold_val_expert)}, distribution: {np.bincount(y_fold_val_expert)}")

    fn_expert_fold_model = xgb.XGBClassifier(**EXPERT_MODEL_PARAMS)
    eval_set_expert_fold = [(X_fold_val_expert, y_fold_val_expert)]

    print(f"    Fitting FN Expert model for Fold {fold_num + 1}...")
    fn_expert_fold_model.fit(X_fold_train_expert, y_fold_train_expert,
                             eval_set=eval_set_expert_fold,
                             verbose=100)

    # 生成折外预测 (预测样本是FN的概率)
    oof_fn_expert_probs_subset[val_idx_expert] = fn_expert_fold_model.predict_proba(X_fold_val_expert)[:, 1]
    print(f"    Fold {fold_num + 1} OOF predictions for FN Expert generated.")

    # 评估当前折在验证集上的性能
    calculate_expert_metrics(y_fold_val_expert, oof_fn_expert_probs_subset[val_idx_expert],
                             title=f"FN Expert Fold {fold_num + 1} Validation")

    # 保存当前折的FN专家模型
    fold_model_path = os.path.join(FN_EXPERT_FOLDS_DIR, f"fn_expert_v1_opt_fold_{fold_num + 1}.joblib")
    joblib.dump(fn_expert_fold_model, fold_model_path)
    fn_expert_fold_models.append(fn_expert_fold_model)
    print(f"    Fold {fold_num + 1} FN Expert model saved to {fold_model_path}")

end_kfold_expert_time = time.time()
print(f"\nFN Expert K-Fold CV finished in {end_kfold_expert_time - start_kfold_expert_time:.2f} seconds.")

# 将子集上的OOF预测映射回完整 Training/CV Pool 的维度
# oof_fn_expert_probs 的长度应该与 train_l0_pred_labels 一致
# 对于基础模型预测为“有雨”的样本，FN专家的预测概率没有意义，可以设为0或一个特定值（如0.5）
oof_fn_expert_probs_full = np.full(len(train_l0_pred_labels), 0.0, dtype=float) # 默认为0，或可设为0.5
oof_fn_expert_probs_full[indices_l0_predicts_no_rain] = oof_fn_expert_probs_subset
np.save(OOF_FN_EXPERT_PROBS_PATH, oof_fn_expert_probs_full)
print(f"Out-of-Fold FN Expert probabilities (full Training/CV Pool) saved to: {OOF_FN_EXPERT_PROBS_PATH}")

# --- 4. 训练最终的FN专家模型 (在整个FN专家训练子集上) ---
print("\n--- Step 4: Training Final FN Expert Model on Full Subset ---")
print(f"Fitting final FN Expert model on {len(y_fn_train_subset_target)} samples...")
# 使用相同的参数，但这次在整个 X_fn_train_subset, y_fn_train_subset_target 上训练
# 早停通常需要一个验证集，这里我们可以在K-Fold时确定一个较优的 n_estimators，或者不使用早停
# 为简单起见，我们重新训练，并假设K-Fold中的 n_estimators 是合理的，或者直接训练到最大轮数
# 或者，选择K-Fold中表现最好的模型，或平均它们的预测
# 这里我们重新训练一个模型，不使用早停，直接用KFold中模型平均的迭代次数或固定一个较大的值

final_fn_expert_model_params = EXPERT_MODEL_PARAMS.copy()
# 可以考虑去掉 early_stopping_rounds 或根据KFold的平均best_iteration设置n_estimators
if 'early_stopping_rounds' in final_fn_expert_model_params:
    del final_fn_expert_model_params['early_stopping_rounds']
    # 或者根据 kf_models 的 best_iteration 平均值来设置 n_estimators
    # avg_best_iteration = int(np.mean([m.best_iteration for m in fn_expert_fold_models if hasattr(m, 'best_iteration')]))
    # final_fn_expert_model_params['n_estimators'] = avg_best_iteration if avg_best_iteration > 0 else EXPERT_MODEL_PARAMS.get('n_estimators', 500)
    # print(f"  Using n_estimators = {final_fn_expert_model_params['n_estimators']} for final FN expert model.")


final_fn_expert_model = xgb.XGBClassifier(**final_fn_expert_model_params)
start_final_fn_expert_time = time.time()
final_fn_expert_model.fit(X_fn_train_subset, y_fn_train_subset_target, verbose=100) # 在完整子集上训练
end_final_fn_expert_time = time.time()
print(f"Final FN Expert model training finished in {end_final_fn_expert_time - start_final_fn_expert_time:.2f} seconds.")

joblib.dump(final_fn_expert_model, FN_EXPERT_MODEL_SAVE_PATH)
print(f"Final FN Expert model saved to: {FN_EXPERT_MODEL_SAVE_PATH}")

print("\n--- FN Expert Model Training Complete ---")