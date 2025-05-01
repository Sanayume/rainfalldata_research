import numpy as np
import lightgbm as lgb # Import LightGBM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import optuna

# --- 配置 ---
PROJECT_DIR = "F:/rainfalldata"
# Use v5 data files
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features_v5.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target_v5.npy")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_v5.txt")
# Save LightGBM v5 model and plot
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "lightgbm_v5_optuna_model.joblib")
IMPORTANCE_PLOT_PATH = os.path.join(PROJECT_DIR, "lightgbm_v5_optuna_feature_importance.png")
OPTUNA_STUDY_NAME = "lightgbm_v5_optimization" # Optuna study name

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 20 # Early stopping rounds for LightGBM callback
N_TOP_FEATURES_TO_PLOT = 50

# --- Optuna 配置 ---
N_TRIALS = 50 # Number of optimization trials
HYPERPARAM_SEARCH_TRAIN_FRACTION = 0.2
HYPERPARAM_SEARCH_VAL_SIZE = 0.25
OPTUNA_METRIC = 'auc' # Metric to optimize ('auc' or 'binary_logloss')

# --- 创建输出目录 ---
os.makedirs(PROJECT_DIR, exist_ok=True)

# --- 辅助函数：计算性能指标 ---
# ... (Copy the calculate_metrics function from xgboost3_optuna.py) ...
def calculate_metrics(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    print(f"\n--- {title} Performance ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Accuracy: {accuracy:.4f}, POD: {pod:.4f}, FAR: {far:.4f}, CSI: {csi:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Rain', 'Rain']))
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi}

# --- 1. 加载数据 ---
print("Loading flattened data (v5)...")
try:
    # ... (Loading logic same as xgboost3_optuna.py) ...
    print("Attempting to load full arrays into memory...")
    X_flat = np.load(X_FLAT_PATH)
    Y_flat_raw = np.load(Y_FLAT_PATH)
    print("Successfully loaded full arrays.")
    print(f"Loaded X_flat shape: {X_flat.shape}")
    print(f"Loaded Y_flat_raw shape: {Y_flat_raw.shape}")
except FileNotFoundError:
    print(f"Error: Data files not found. Ensure {X_FLAT_PATH} and {Y_FLAT_PATH} exist.")
    print("Run turn5.py first.")
    exit()
except MemoryError:
    print("MemoryError loading full arrays. Falling back to memory mapping...")
    try:
        X_flat = np.load(X_FLAT_PATH, mmap_mode='r')
        Y_flat_raw = np.load(Y_FLAT_PATH, mmap_mode='r')
        print("Loaded using memory mapping.")
        print(f"Loaded X_flat shape: {X_flat.shape}")
        print(f"Loaded Y_flat_raw shape: {Y_flat_raw.shape}")
    except Exception as e_mmap:
        print(f"Error loading data even with mmap: {e_mmap}")
        exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. 加载特征名称 ---
# ... (Loading logic same as xgboost3_optuna.py) ...
print(f"Loading feature names from {FEATURE_NAMES_PATH}...")
try:
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(feature_names)} feature names.")
    if len(feature_names) != X_flat.shape[1]:
        print(f"Warning: Feature name count ({len(feature_names)}) mismatch with data columns ({X_flat.shape[1]})!")
        feature_names = [f'f{i}' for i in range(X_flat.shape[1])] # Fallback
except Exception as e:
    print(f"Error loading feature names: {e}. Using generic names.")
    feature_names = [f'f{i}' for i in range(X_flat.shape[1])]

# --- 3. 预处理 和 数据分割 ---
# ... (Preprocessing and splitting logic same as xgboost3_optuna.py) ...
print("Preprocessing data...")
y_flat_binary = (Y_flat_raw > RAIN_THRESHOLD).astype(int)
if not isinstance(Y_flat_raw, np.memmap):
    del Y_flat_raw
print("Splitting data into initial training and final testing sets...")
n_samples = X_flat.shape[0]
split_idx = int(n_samples * (1 - TEST_SIZE_RATIO))
X_train_full = X_flat[:split_idx]
y_train_full = y_flat_binary[:split_idx]
X_test = X_flat[split_idx:]
y_test = y_flat_binary[split_idx:]
print(f"Full training set shape: X={X_train_full.shape}, y={y_train_full.shape}")
print(f"Final test set shape: X={X_test.shape}, y={y_test.shape}")
print(f"\nCreating subset for hyperparameter search (using {HYPERPARAM_SEARCH_TRAIN_FRACTION*100:.0f}% of training data)...")
if HYPERPARAM_SEARCH_TRAIN_FRACTION < 1.0:
    X_search, _, y_search, _ = train_test_split(
        X_train_full, y_train_full,
        train_size=HYPERPARAM_SEARCH_TRAIN_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )
else:
    X_search, y_search = X_train_full, y_train_full
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_search, y_search,
    test_size=HYPERPARAM_SEARCH_VAL_SIZE,
    random_state=RANDOM_STATE + 1,
    stratify=y_search
)
print(f"  Search training subset shape: X={X_train_sub.shape}, y={y_train_sub.shape}")
print(f"  Search validation subset shape: X={X_val.shape}, y={y_val.shape}")
if hasattr(X_flat, '_mmap'):
    X_flat._mmap.close()
    del X_flat
elif 'X_flat' in locals() and not isinstance(X_flat, np.memmap):
    del X_flat
del y_flat_binary

# --- 4. Optuna 目标函数 (LightGBM) ---
def objective(trial):
    # Calculate scale_pos_weight on the subset
    num_neg_sub = np.sum(y_train_sub == 0)
    num_pos_sub = np.sum(y_train_sub == 1)
    scale_pos_weight_sub = num_neg_sub / num_pos_sub if num_pos_sub > 0 else 1

    # Define hyperparameters for LightGBM
    param = {
        'objective': 'binary', # Binary classification
        'metric': OPTUNA_METRIC, # Metric for evaluation
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'n_estimators': 1000, # Start with a large number, use early stopping
        'verbose': -1, # Suppress verbose output during training
        'scale_pos_weight': scale_pos_weight_sub, # Handle class imbalance
        # Tunable parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150), # Max leaves in one tree
        'max_depth': trial.suggest_int('max_depth', 3, 12), # Limit tree depth
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0), # Colsample_bytree equivalent
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0), # Subsample equivalent
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7), # Perform bagging every k iterations
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True), # L1 regularization
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True), # L2 regularization
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), # Min data needed in a leaf
    }

    model = lgb.LGBMClassifier(**param)

    eval_set_trial = [(X_val, y_val)]
    callbacks = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]

    model.fit(X_train_sub, y_train_sub,
              eval_set=eval_set_trial,
              callbacks=callbacks)

    # Get the best score achieved on the validation set
    # LGBMClassifier stores best score in best_score_ attribute
    # The score is based on the first metric in eval_metric if multiple are provided
    best_score = model.best_score_['valid_0'][OPTUNA_METRIC]

    return best_score

# --- 5. 运行 Optuna 优化 ---
print(f"\nStarting Optuna hyperparameter search ({N_TRIALS} trials) for LightGBM...")
study_direction = 'maximize' if OPTUNA_METRIC == 'auc' else 'minimize'
study = optuna.create_study(direction=study_direction, study_name=OPTUNA_STUDY_NAME, load_if_exists=True)
start_optuna = time.time()
study.optimize(objective, n_trials=N_TRIALS, timeout=None)
end_optuna = time.time()
print(f"Optuna search finished in {end_optuna - start_optuna:.2f} seconds.")

print("\nBest trial:")
trial = study.best_trial
print(f"  Value ({OPTUNA_METRIC}): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params = trial.params

# --- 6. 使用最佳参数重新训练最终模型 (LightGBM) ---
print("\nRetraining final LightGBM model using best parameters on the full training set...")
# Calculate scale_pos_weight on the full training set
num_neg_full = np.sum(y_train_full == 0)
num_pos_full = np.sum(y_train_full == 1)
scale_pos_weight_full = num_neg_full / num_pos_full if num_pos_full > 0 else 1
print(f"Calculated scale_pos_weight for final model: {scale_pos_weight_full:.4f}")

final_params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'], # Evaluate both in final training
    'boosting_type': 'gbdt',
    'random_state': RANDOM_STATE,
    'scale_pos_weight': scale_pos_weight_full,
    'n_estimators': 1500, # Use a slightly larger number for final training
    'verbose': -1, # Keep verbose off unless debugging
}
# Update with best params found by Optuna
final_params.update(best_params)

final_model = lgb.LGBMClassifier(**final_params)

print("Starting final model training...")
start_final_train = time.time()
# Use test set for early stopping monitoring in the final fit
final_eval_set = [(X_test, y_test)]
final_callbacks = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS * 2, verbose=True)] # Allow more rounds, show stopping info

final_model.fit(X_train_full, y_train_full,
                eval_set=final_eval_set,
                callbacks=final_callbacks)
end_final_train = time.time()
print(f"Final model training complete in {end_final_train - start_final_train:.2f} seconds.")

try:
    print(f"Best iteration (final model): {final_model.best_iteration_}")
    if final_model.best_score_:
        best_logloss = final_model.best_score_['valid_0']['binary_logloss']
        best_auc = final_model.best_score_['valid_0']['auc']
        print(f"Best score on test set (LogLoss): {best_logloss:.4f}")
        print(f"Best score on test set (AUC): {best_auc:.4f}")
except Exception as e:
    print(f"Error retrieving final model best score: {e}")

# --- 7. 特征重要性 (Final Model - LightGBM) ---
print("\n--- Feature Importances (LightGBM v5 Optuna - Final Model) ---")
try:
    importances = final_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    # LightGBM importance is based on split count by default, can also use 'gain'
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("Top 10 Features:")
    print(importance_df.head(10))

    n_features_actual = len(feature_names)
    n_plot = min(N_TOP_FEATURES_TO_PLOT, n_features_actual)

    plt.figure(figsize=(10, n_plot / 2.0))
    top_features = importance_df.head(n_plot)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel("Importance Score (Split Count)")
    plt.ylabel("Feature")
    plt.title(f"Top {n_plot} Feature Importances (LightGBM v5 Optuna)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH)
    print(f"Feature importance plot saved to: {IMPORTANCE_PLOT_PATH}")
    plt.close()

except Exception as plot_e:
    print(f"Warning: Could not generate feature importance plot - {plot_e}")

# --- 8. 评估最终模型 (LightGBM) ---
print("\n--- Evaluating Final Model on Test Set (LightGBM v5 Optuna) ---")
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

# Evaluate across thresholds
thresholds_to_evaluate = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
metrics_by_threshold = {}
for threshold in thresholds_to_evaluate:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    metrics = calculate_metrics(y_test, y_pred_threshold, title=f"LightGBM v5 Optuna (Threshold {threshold:.2f})")
    metrics_by_threshold[threshold] = metrics

print("\n--- LightGBM v5 Optuna Performance across different thresholds (Test Set) ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
threshold_metrics_data = {}
for threshold, metrics in metrics_by_threshold.items():
    threshold_metrics_data[f'LGBM_v5_Opt_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

threshold_df = pd.DataFrame(threshold_metrics_data).T
threshold_df = threshold_df[metrics_to_show]
float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols: threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols: threshold_df[col] = threshold_df[col].map('{:.0f}'.format)
print(threshold_df)

# --- 9. 保存最终模型 (LightGBM) ---
print(f"\nSaving the final trained LightGBM model to {MODEL_SAVE_PATH}...")
try:
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nScript finished.")
