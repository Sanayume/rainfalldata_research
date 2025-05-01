import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import pandas as pd
import optuna

# --- 配置 ---
PROJECT_DIR = "F:/rainfalldata"
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features_v5.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target_v5.npy")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_v5.txt")
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "naive_bayes_v5_optuna_model.joblib")
OPTUNA_STUDY_NAME = "naive_bayes_v5_optimization"

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42

N_TRIALS = 30
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

print("Loading flattened data (v5)...")
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
    var_smoothing = trial.suggest_float('var_smoothing', 1e-10, 1e-3, log=True)
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train_sub, y_train_sub)
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]
    try:
        score = roc_auc_score(y_val, y_pred_proba_val)
    except ValueError:
        print("Warning: ValueError during AUC calculation in Optuna trial. Returning 0.5.")
        score = 0.5
    return score

print(f"\nStarting Optuna hyperparameter search ({N_TRIALS} trials) for Naive Bayes...")
study_direction = 'maximize'
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

print("\nRetraining final Naive Bayes model using best parameters on the full training set...")
final_model = GaussianNB(**best_params)

print("Starting final model training...")
start_final_train = time.time()
final_model.fit(X_train_full, y_train_full)
end_final_train = time.time()
print(f"Final model training complete in {end_final_train - start_final_train:.2f} seconds.")

print("\n--- Evaluating Final Model on Test Set (Naive Bayes v5 - Optuna) ---")
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

thresholds_to_evaluate = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
metrics_by_threshold = {}
for threshold in thresholds_to_evaluate:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    metrics = calculate_metrics(y_test, y_pred_threshold, title=f"Naive Bayes v5 Optuna (Threshold {threshold:.2f})")
    metrics_by_threshold[threshold] = metrics

print("\n--- Naive Bayes v5 (Optuna) Performance across different thresholds (Test Set) ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
threshold_metrics_data = {}
for threshold, metrics in metrics_by_threshold.items():
    threshold_metrics_data[f'NB_v5_Opt_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

threshold_df = pd.DataFrame(threshold_metrics_data).T
threshold_df = threshold_df[metrics_to_show]
float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols: threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols: threshold_df[col] = threshold_df[col].map('{:.0f}'.format)
print(threshold_df)

print(f"\nSaving the final trained Naive Bayes model to {MODEL_SAVE_PATH}...")
try:
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nScript finished.")