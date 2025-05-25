import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# --- Added Imports ---
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# --- 模型 ---
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB

# --- 配置 ---
PROJECT_DIR_FEATURES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
X_FLAT_PATH = os.path.join(PROJECT_DIR_FEATURES, "X_Yangtsu_flat_features.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR_FEATURES, "Y_Yangtsu_flat_target.npy")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR_FEATURES, "feature_names_yangtsu.txt")

MODELS_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "models_v1_comparison_suite_v2")
if not os.path.exists(MODELS_SAVE_DIR):
    os.makedirs(MODELS_SAVE_DIR)
    print(f"Created directory for saving models: {MODELS_SAVE_DIR}")

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5 # Changed from 3 to 5
SUBSET_FRACTION_KNN_SVM = 0.01 # 为KNN和SVM训练使用10%的数据子集

# --- 辅助函数：计算性能指标 (与之前一致) ---
def calculate_metrics(y_true, y_pred, model_name=""):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)

    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

    print(f"\n--- {model_name} Performance ---")
    print(f"Confusion Matrix:\n{cm}")
    if len(cm.ravel()) == 4:
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP): {tp}")
    else:
        print("  CM could not be unpacked into TN, FP, FN, TP.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"POD (Hit Rate/Recall): {pod:.4f}")
    print(f"FAR (False Alarm Ratio): {far:.4f}")
    print(f"CSI (Critical Success Index): {csi:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Rain', 'Rain'], zero_division=0))
    
    return {'model_name': model_name, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
            'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}

# --- 辅助函数：打印GridSearchCV交叉验证结果 ---
def print_grid_search_cv_results(gscv_object, model_name_str, cv_folds_count):
    print(f"\nDetailed Cross-Validation Results for {model_name_str} (using {cv_folds_count} folds):")
    results_df = pd.DataFrame(gscv_object.cv_results_)
    best_index = gscv_object.best_index_

    print(f"  Best Parameters found: {gscv_object.best_params_}")
    print(f"  Best Mean CV Score ({gscv_object.scoring}): {gscv_object.best_score_:.4f}")

    fold_scores = []
    print("  Scores for each fold for the best estimator:")
    for i in range(cv_folds_count):
        fold_score_column = f'split{i}_test_score'
        if fold_score_column in results_df.columns:
            score = results_df.loc[best_index, fold_score_column]
            fold_scores.append(score)
            print(f"    Fold {i+1}: {score:.4f}")
        else:
            print(f"    Fold {i+1} score column ('{fold_score_column}') not found in cv_results_.")
            
    if fold_scores:
        print(f"  Standard Deviation of CV scores (for best estimator): {np.std(fold_scores):.4f}")
    # print("\n  Top parameter combinations (mean_test_score):") # Uncomment to see more details
    # top_n = min(5, len(results_df))
    # print(results_df[['params', 'mean_test_score', 'std_test_score']].nlargest(top_n, 'mean_test_score'))


# --- 1. 加载数据 ---
print(">>> 1. Loading data...")
if not (os.path.exists(X_FLAT_PATH) and os.path.exists(Y_FLAT_PATH) and os.path.exists(FEATURE_NAMES_PATH)):
    raise FileNotFoundError(f"Data files not found.")
try:
    X_flat = np.load(X_FLAT_PATH, mmap_mode='r')
    Y_flat = np.load(Y_FLAT_PATH, mmap_mode='r')
except Exception as e:
    print(f"Error loading data: {e}")
    raise
with open(FEATURE_NAMES_PATH, "r") as f:
    feature_names = [line.strip() for line in f]
print(f"Loaded X_flat: {X_flat.shape}, Y_flat: {Y_flat.shape}, Features: {len(feature_names)}")

# --- 2. 准备数据 ---
print("\n>>> 2. Preparing data...")
Y_binary = (Y_flat > RAIN_THRESHOLD).astype(int)

if np.isnan(X_flat).any() or np.isinf(X_flat).any():
    print("Warning: NaNs/Infs found. Imputing with column means.")
    col_means = np.nanmean(X_flat, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    inds_nan = np.where(np.isnan(X_flat))
    X_flat[inds_nan] = np.take(col_means, inds_nan[1])
    X_flat = np.nan_to_num(X_flat, posinf=np.finfo(X_flat.dtype).max, neginf=np.finfo(X_flat.dtype).min)

print(f"Target: No Rain (0) count: {np.sum(Y_binary == 0)}, Rain (1) count: {np.sum(Y_binary == 1)}")

n_samples = X_flat.shape[0]
split_idx = int(n_samples * (1 - TEST_SIZE_RATIO))
X_train_val, X_test = X_flat[:split_idx], X_flat[split_idx:]
y_train_val, y_test = Y_binary[:split_idx], Y_binary[split_idx:]

print("\n>>> 2.1 Feature Scaling...")
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_SAVE_DIR, "scaler_v1.joblib"))
print("Scaler saved.")
print(f"X_train_val_scaled shape: {X_train_val_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")


# --- 2.2 创建KNN和SVM训练用的数据子集 ---
print(f"\n>>> 2.2 Creating a {SUBSET_FRACTION_KNN_SVM*100:.0f}% subset of scaled training data for KNN and SVM GridSearchCV...")
X_train_knn_svm_subset, _, y_train_knn_svm_subset, _ = train_test_split(
    X_train_val_scaled,
    y_train_val,
    train_size=SUBSET_FRACTION_KNN_SVM,
    random_state=RANDOM_STATE,
    stratify=y_train_val
)
print(f"Subset for KNN/SVM training: X_subset shape {X_train_knn_svm_subset.shape}, y_subset shape {y_train_knn_svm_subset.shape}")


val_split_ratio_lgbm = 0.1
X_train_lgbm_scaled, X_val_lgbm_scaled, y_train_lgbm, y_val_lgbm = train_test_split(
    X_train_val_scaled, y_train_val, test_size=val_split_ratio_lgbm, random_state=RANDOM_STATE, stratify=y_train_val
)
print(f"Full Scaled Train/Val set (for RF Opt, GNB Opt): X_train_val_scaled shape {X_train_val_scaled.shape}")
print(f"LGBM Scaled Train set: X_train_lgbm_scaled shape {X_train_lgbm_scaled.shape}")
print(f"LGBM Scaled Validation set: X_val_lgbm_scaled shape {X_val_lgbm_scaled.shape}")

del X_flat, Y_flat, Y_binary, X_train_val 
import gc
gc.collect()

all_model_metrics = []

# --- 3. 训练和评估模型 ---

# --- 3.1 K-Nearest Neighbors (KNN) with Hyperparameter Tuning (on subset) ---
print("\n>>> 3.1 Training K-Nearest Neighbors (KNN) model with Hyperparameter Tuning (on subset)...")
param_grid_knn = {
    'n_neighbors': [3, 5, 11], 
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan'] 
}
knn_gscv = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    param_grid_knn,
    cv=CV_FOLDS,
    scoring='accuracy',
    verbose=2, 
    n_jobs=-1
)
try:
    print(f"正在开始KNN的GridSearchCV超参数寻优 (在 {SUBSET_FRACTION_KNN_SVM*100:.0f}% 的数据子集 X_train_knn_svm_subset 上)...")
    print(f"子集形状: X={X_train_knn_svm_subset.shape}, y={y_train_knn_svm_subset.shape}")
    knn_gscv.fit(X_train_knn_svm_subset, y_train_knn_svm_subset) 
    print("KNN GridSearchCV超参数寻优完成。")
    # print(f"最佳KNN参数: {knn_gscv.best_params_}") # Covered by helper
    # print(f"最佳KNN交叉验证得分 (accuracy on subset): {knn_gscv.best_score_:.4f}") # Covered by helper
    print_grid_search_cv_results(knn_gscv, "K-Nearest Neighbors (Tuned on Subset)", CV_FOLDS)


    best_model_knn = knn_gscv.best_estimator_
    print("在完整的X_test_scaled上评估调优后的KNN模型...")
    y_pred_knn = best_model_knn.predict(X_test_scaled) 
    metrics_knn = calculate_metrics(y_test, y_pred_knn, model_name="K-Nearest Neighbors (Tuned on Subset)")
    all_model_metrics.append(metrics_knn)
    joblib.dump(best_model_knn, os.path.join(MODELS_SAVE_DIR, "knn_v1_tuned_subset.joblib"))
    print(f"调优后的KNN模型已保存。")
except Exception as e:
    print(f"K-Nearest Neighbors (Tuned on Subset) 训练过程中发生错误: {e}")
    import traceback
    traceback.print_exc()

# --- 3.2 Support Vector Machine (SVM) with Hyperparameter Tuning (on subset) ---
print("\n>>> 3.2 Training Support Vector Machine (SVM) model with Hyperparameter Tuning (on subset)...")
print("注意: SVM GridSearchCV 即使在子集上也可能比较耗时。")
param_grid_svm = {
    'C': [0.1, 1], 
    'kernel': ['rbf'], 
    'gamma': ['scale', 0.01] 
}
svm_gscv = GridSearchCV(
    SVC(class_weight='balanced', random_state=RANDOM_STATE, probability=True),
    param_grid_svm,
    cv=CV_FOLDS, 
    scoring='accuracy',
    verbose=2, 
    n_jobs=-1
)
try:
    print(f"正在开始SVM的GridSearchCV超参数寻优 (在 {SUBSET_FRACTION_KNN_SVM*100:.0f}% 的数据子集 X_train_knn_svm_subset 上)...")
    print(f"子集形状: X={X_train_knn_svm_subset.shape}, y={y_train_knn_svm_subset.shape}")
    svm_gscv.fit(X_train_knn_svm_subset, y_train_knn_svm_subset) 
    print("SVM GridSearchCV超参数寻优完成。")
    # print(f"最佳SVM参数: {svm_gscv.best_params_}") # Covered by helper
    # print(f"最佳SVM交叉验证得分 (accuracy on subset): {svm_gscv.best_score_:.4f}") # Covered by helper
    print_grid_search_cv_results(svm_gscv, "Support Vector Machine (Tuned on Subset)", CV_FOLDS)

    best_model_svm = svm_gscv.best_estimator_
    print("在完整的X_test_scaled上评估调优后的SVM模型...")
    y_pred_svm = best_model_svm.predict(X_test_scaled) 
    metrics_svm = calculate_metrics(y_test, y_pred_svm, model_name="Support Vector Machine (Tuned on Subset)")
    all_model_metrics.append(metrics_svm)
    joblib.dump(best_model_svm, os.path.join(MODELS_SAVE_DIR, "svm_v1_tuned_subset.joblib"))
    print(f"调优后的SVM模型已保存。")
except Exception as e:
    print(f"Support Vector Machine (Tuned on Subset) 训练过程中发生错误: {e}")
    import traceback
    traceback.print_exc()

# --- 3.3 随机森林 (Random Forest) with Hyperparameter Tuning (on full scaled train/val) ---
print("\n>>> 3.3 Training Random Forest model with Hyperparameter Tuning (on full X_train_val_scaled)...")
param_grid_rf = {
    'n_estimators': [100, 150],
    'max_depth': [15, 20, None],
    'min_samples_split': [10, 20], 
    'min_samples_leaf': [5, 10],   
}
rf_gscv = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, oob_score=True),
    param_grid_rf,
    cv=CV_FOLDS,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)
try:
    print("正在开始Random Forest的GridSearchCV超参数寻优 (在完整的X_train_val_scaled上)...")
    print(f"训练数据形状: X={X_train_val_scaled.shape}, y={y_train_val.shape}")
    rf_gscv.fit(X_train_val_scaled, y_train_val) 
    print("Random Forest GridSearchCV超参数寻优完成。")
    # print(f"最佳RF参数: {rf_gscv.best_params_}") # Covered by helper
    # print(f"最佳RF交叉验证得分 (accuracy): {rf_gscv.best_score_:.4f}") # Covered by helper
    print_grid_search_cv_results(rf_gscv, "Random Forest (Tuned)", CV_FOLDS)
    
    best_model_rf = rf_gscv.best_estimator_
    if hasattr(best_model_rf, 'oob_score_') and best_model_rf.oob_score_:
         print(f"Random Forest (Tuned) OOB Score: {best_model_rf.oob_score_:.4f}")
    else:
        print("Random Forest (Tuned) OOB Score not available.")

    print("在完整的X_test_scaled上评估调优后的RF模型...")
    y_pred_rf = best_model_rf.predict(X_test_scaled)
    metrics_rf = calculate_metrics(y_test, y_pred_rf, model_name="Random Forest (Tuned)")
    all_model_metrics.append(metrics_rf)
    joblib.dump(best_model_rf, os.path.join(MODELS_SAVE_DIR, "random_forest_v1_tuned.joblib"))
    print(f"调优后的RF模型已保存。Top 10 Features:")
    importances_rf = best_model_rf.feature_importances_
    sorted_indices_rf = np.argsort(importances_rf)[::-1]
    for i in range(min(10, len(feature_names))):
        print(f"  {feature_names[sorted_indices_rf[i]]}: {importances_rf[sorted_indices_rf[i]]:.4f}")
except Exception as e:
    print(f"Error during Random Forest (Tuned): {e}")
    import traceback
    traceback.print_exc()

# --- 3.4 LightGBM with Hyperparameter Tuning & Early Stopping (GridSearchCV on full scaled train/val, EarlyStopping on its split) ---
print("\n>>> 3.4 Training LightGBM model with Hyperparameter Tuning & Early Stopping...")
param_grid_lgb = {
    'num_leaves': [31, 41],      
    'learning_rate': [0.01, 0.05],
    'colsample_bytree': [0.7, 0.8],
    'subsample': [0.8], 
}
lgbm_gscv = GridSearchCV(
    lgb.LGBMClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, reg_alpha=0.05, reg_lambda=0.05),
    param_grid_lgb,
    cv=CV_FOLDS,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)
try:
    print("正在开始LightGBM的GridSearchCV超参数寻优 (在完整的X_train_val_scaled上)...")
    print(f"训练数据形状: X={X_train_val_scaled.shape}, y={y_train_val.shape}")
    lgbm_gscv.fit(X_train_val_scaled, y_train_val) 
    print("LightGBM GridSearchCV超参数寻优完成。")
    # print(f"最佳LightGBM参数 (from GSCV): {lgbm_gscv.best_params_}") # Covered by helper
    # print(f"最佳LightGBM交叉验证得分 (accuracy from GSCV): {lgbm_gscv.best_score_:.4f}") # Covered by helper
    print_grid_search_cv_results(lgbm_gscv, "LightGBM (GridSearchCV part)", CV_FOLDS)

    best_params_lgb = lgbm_gscv.best_params_
    best_params_lgb['reg_alpha'] = 0.05
    best_params_lgb['reg_lambda'] = 0.05
    # best_params_lgb['subsample'] = 0.8 # Already in best_params_lgb if it was in param_grid_lgb and chosen

    model_lgb_tuned = lgb.LGBMClassifier(
        **best_params_lgb,
        n_estimators=1000, 
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    print("正在开始最终的LightGBM模型训练 (在X_train_lgbm_scaled上, 早停验证集为X_val_lgbm_scaled)...")
    print(f"LGBM训练集: {X_train_lgbm_scaled.shape}, LGBM验证集: {X_val_lgbm_scaled.shape}")
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=100)]
    model_lgb_tuned.fit(
        X_train_lgbm_scaled, y_train_lgbm, 
        eval_set=[(X_val_lgbm_scaled, y_val_lgbm)],
        callbacks=callbacks
    )
    print("LightGBM (Tuned) training complete.")
    if hasattr(model_lgb_tuned, 'best_iteration_') and model_lgb_tuned.best_iteration_ is not None:
         print(f"Best iteration (n_estimators for tuned model): {model_lgb_tuned.best_iteration_}")
    else:
        print("Best iteration not available.")

    print("在完整的X_test_scaled上评估调优后的LGBM模型...")
    y_pred_lgb = model_lgb_tuned.predict(X_test_scaled)
    metrics_lgb = calculate_metrics(y_test, y_pred_lgb, model_name="LightGBM (Tuned)")
    all_model_metrics.append(metrics_lgb)
    joblib.dump(model_lgb_tuned, os.path.join(MODELS_SAVE_DIR, "lightgbm_v1_tuned.joblib"))
    print(f"调优后的LGBM模型已保存。Top 10 Features:")
    importances_lgb = model_lgb_tuned.feature_importances_
    sorted_indices_lgb = np.argsort(importances_lgb)[::-1]
    for i in range(min(10, len(feature_names))):
        print(f"  {feature_names[sorted_indices_lgb[i]]}: {importances_lgb[sorted_indices_lgb[i]]}")
except Exception as e:
    print(f"Error during LightGBM (Tuned): {e}")
    import traceback
    traceback.print_exc()

# --- 3.5 朴素贝叶斯 (Gaussian Naive Bayes) with Hyperparameter Tuning (on full scaled train/val) ---
print("\n>>> 3.5 Training Gaussian Naive Bayes model with Hyperparameter Tuning (on full X_train_val_scaled)...")
param_grid_gnb = {'var_smoothing': np.logspace(-9, -2, 8)} 
gnb_cv = GridSearchCV(
    GaussianNB(),
    param_grid_gnb,
    cv=CV_FOLDS,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)
try:
    print("正在开始GaussianNB的GridSearchCV超参数寻优 (在完整的X_train_val_scaled上)...")
    print(f"训练数据形状: X={X_train_val_scaled.shape}, y={y_train_val.shape}")
    gnb_cv.fit(X_train_val_scaled, y_train_val) 
    print("GaussianNB GridSearchCV超参数寻优完成。")
    # print(f"Best var_smoothing: {gnb_cv.best_params_['var_smoothing']}") # Covered by helper
    # print(f"Best CV score (accuracy): {gnb_cv.best_score_:.4f}") # Covered by helper
    print_grid_search_cv_results(gnb_cv, "Gaussian Naive Bayes (Tuned)", CV_FOLDS)

    best_model_gnb = gnb_cv.best_estimator_
    print("在完整的X_test_scaled上评估调优后的GNB模型...")
    y_pred_gnb_tuned = best_model_gnb.predict(X_test_scaled)
    metrics_gnb_tuned = calculate_metrics(y_test, y_pred_gnb_tuned, model_name="Gaussian Naive Bayes (Tuned)")
    all_model_metrics.append(metrics_gnb_tuned)
    joblib.dump(best_model_gnb, os.path.join(MODELS_SAVE_DIR, "naive_bayes_v1_tuned.joblib"))
    print(f"调优后的Gaussian Naive Bayes模型已保存。")
except Exception as e:
    print(f"Error during Gaussian Naive Bayes (Tuned): {e}")
    import traceback
    traceback.print_exc()
    print("Falling back to default GaussianNB due to tuning error.")
    model_gnb_default = GaussianNB()
    try:
        model_gnb_default.fit(X_train_val_scaled, y_train_val)
        y_pred_gnb_default = model_gnb_default.predict(X_test_scaled)
        metrics_gnb_default = calculate_metrics(y_test, y_pred_gnb_default, model_name="Gaussian Naive Bayes (Default Fallback)")
        all_model_metrics.append(metrics_gnb_default)
        joblib.dump(model_gnb_default, os.path.join(MODELS_SAVE_DIR, "naive_bayes_v1_default_fallback.joblib"))
        print("Default GaussianNB model (fallback) saved.")
    except Exception as e_fb:
        print(f"Error during fallback default GaussianNB: {e_fb}")

# --- 3.6 贝叶斯网络说明 ---
print("\n>>> 3.6 Bayesian Network (Note)")
print("Naive Bayes (implemented as GaussianNB) is a simple type of Bayesian Network.")
print("Training general Bayesian Networks is complex and out of scope for this script.")

# --- 4. 综合性能对比 ---
print("\n\n>>> 4. Overall Model Performance Comparison <<<")
if all_model_metrics:
    comparison_df = pd.DataFrame(all_model_metrics)
    comparison_df.set_index('model_name', inplace=True)
    
    float_cols = ['accuracy', 'pod', 'far', 'csi']
    for col in float_cols:
        if col in comparison_df.columns: comparison_df[col] = comparison_df[col].map('{:.4f}'.format)
    int_cols = ['tn', 'fp', 'fn', 'tp']
    for col in int_cols:
        if col in comparison_df.columns: comparison_df[col] = comparison_df[col].map('{:.0f}'.format)
            
    print(comparison_df[['accuracy', 'pod', 'far', 'csi', 'tp', 'fn', 'fp', 'tn']])
else:
    print("No model metrics were successfully collected for comparison.")

print("\n--- Script execution finished ---")