import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import optuna

# --- 配置 ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "nationwide", "features")
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features_v3.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target_v3.npy")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_v3.txt")
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "xgboost_v3_optuna_model.joblib")
IMPORTANCE_PLOT_PATH = os.path.join(PROJECT_DIR, "xgboost_v3_optuna_feature_importance.png")
OPTUNA_STUDY_NAME = "xgboost_v3_optimization"

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 20
N_TOP_FEATURES_TO_PLOT = 50

N_TRIALS = 50
HYPERPARAM_SEARCH_TRAIN_FRACTION = 0.2
HYPERPARAM_SEARCH_VAL_SIZE = 0.25
OPTUNA_METRIC = 'auc'

os.makedirs(PROJECT_DIR, exist_ok=True)

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

print("Loading flattened data (v3)...")
try:
    X_flat = np.load(X_FLAT_PATH)
    Y_flat_raw = np.load(Y_FLAT_PATH)
    print("Successfully loaded full arrays.")
    print(f"Loaded X_flat shape: {X_flat.shape}")
    print(f"Loaded Y_flat_raw shape: {Y_flat_raw.shape}")
except FileNotFoundError:
    print(f"Error: Data files not found. Ensure {X_FLAT_PATH} and {Y_FLAT_PATH} exist.")
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

print(f"Loading feature names from {FEATURE_NAMES_PATH}...")
try:
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(feature_names)} feature names.")
    if len(feature_names) != X_flat.shape[1]:
        print(f"Warning: Feature name count ({len(feature_names)}) mismatch with data columns ({X_flat.shape[1]})!")
        feature_names = [f'f{i}' for i in range(X_flat.shape[1])]
except Exception as e:
    print(f"Error loading feature names: {e}. Using generic names.")
    feature_names = [f'f{i}' for i in range(X_flat.shape[1])]

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

def objective(trial):
    num_neg_sub = np.sum(y_train_sub == 0)
    num_pos_sub = np.sum(y_train_sub == 1)
    scale_pos_weight_sub = num_neg_sub / num_pos_sub if num_pos_sub > 0 else 1

    param = {
        'objective': 'binary:logistic',
        'eval_metric': [OPTUNA_METRIC],
        'use_label_encoder': False,
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'scale_pos_weight': scale_pos_weight_sub,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
    }

    model = xgb.XGBClassifier(**param, n_estimators=1000, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

    eval_set_trial = [(X_val, y_val)]
    model.fit(X_train_sub, y_train_sub,
              eval_set=eval_set_trial,
              verbose=False)

    results = model.evals_result()
    best_score = results['validation_0'][OPTUNA_METRIC][model.best_iteration]

    return best_score

print(f"\nStarting Optuna hyperparameter search ({N_TRIALS} trials)...")
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

print("\nRetraining final model using best parameters on the full training set...")
num_neg_full = np.sum(y_train_full == 0)
num_pos_full = np.sum(y_train_full == 1)
scale_pos_weight_full = num_neg_full / num_pos_full if num_pos_full > 0 else 1
print(f"Calculated scale_pos_weight for final model: {scale_pos_weight_full:.4f}")

final_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'use_label_encoder': False,
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
    'scale_pos_weight': scale_pos_weight_full,
    'n_estimators': 1500,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS * 2
}
final_params.update(best_params)

final_model = xgb.XGBClassifier(**final_params)

print("Starting final model training...")
start_final_train = time.time()
final_eval_set = [(X_test, y_test)]
final_model.fit(X_train_full, y_train_full,
                eval_set=final_eval_set,
                verbose=50)
end_final_train = time.time()
print(f"Final model training complete in {end_final_train - start_final_train:.2f} seconds.")

try:
    print(f"Best iteration (final model): {final_model.best_iteration}")
    results = final_model.evals_result()
    if 'validation_0' in results:
        best_logloss = results['validation_0']['logloss'][final_model.best_iteration]
        best_auc = results['validation_0']['auc'][final_model.best_iteration]
        print(f"Best score on test set (LogLoss): {best_logloss:.4f}")
        print(f"Best score on test set (AUC): {best_auc:.4f}")
except Exception as e:
    print(f"Error retrieving final model best score: {e}")

print("\n--- Feature Importances (v3 Optuna - Final Model) ---")
try:
    importances = final_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("Top 10 Features:")
    print(importance_df.head(10))

    n_features_actual = len(feature_names)
    n_plot = min(N_TOP_FEATURES_TO_PLOT, n_features_actual)

    plt.figure(figsize=(10, n_plot / 2.0))
    top_features = importance_df.head(n_plot)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"Top {n_plot} Feature Importances (XGBoost v3 Optuna)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH)
    print(f"Feature importance plot saved to: {IMPORTANCE_PLOT_PATH}")
    plt.close()

except Exception as plot_e:
    print(f"Warning: Could not generate feature importance plot - {plot_e}")

print("\n--- Evaluating Final Model on Test Set (v3 Optuna) ---")
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

thresholds_to_evaluate = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
metrics_by_threshold = {}
for threshold in thresholds_to_evaluate:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    metrics = calculate_metrics(y_test, y_pred_threshold, title=f"XGBoost v3 Optuna (Threshold {threshold:.2f})")
    metrics_by_threshold[threshold] = metrics

print("\n--- XGBoost v3 Optuna Performance across different thresholds (Test Set) ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
threshold_metrics_data = {}
for threshold, metrics in metrics_by_threshold.items():
    threshold_metrics_data[f'XGB_v3_Opt_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

threshold_df = pd.DataFrame(threshold_metrics_data).T
threshold_df = threshold_df[metrics_to_show]
float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols: threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols: threshold_df[col] = threshold_df[col].map('{:.0f}'.format)
print(threshold_df)

print(f"\nSaving the final trained model to {MODEL_SAVE_PATH}...")
try:
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nScript finished.")