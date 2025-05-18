import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

# --- 配置 ---
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features") # Changed to Yangtze directory
# Use Yangtze v4 data files
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_Yangtsu_flat_features_v4.npy") # Changed filename
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target_v4.npy") # Changed filename
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_yangtsu_v4.txt") # Changed filename
MODEL_PREDICTION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "predictions", "xgboost_yangtsu_v4_predictions.npy")
# Save Yangtze v4 model and plot
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "xgboost_yangtsu_v4_model.joblib") # Changed filename
IMPORTANCE_PLOT_PATH = os.path.join(PROJECT_DIR, "xgboost_yangtsu_v4_feature_importance.png") # Changed filename
PERFORMANCE_CSV_PATH = os.path.join(PROJECT_DIR, "threshold_performance_yangtsu_v4.csv") # Changed filename

RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 30
N_TOP_FEATURES_TO_PLOT = 50

# --- 创建输出目录 ---
os.makedirs(PROJECT_DIR, exist_ok=True)

# --- 辅助函数：计算性能指标 ---
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
print("Loading flattened Yangtze data (v4)...")
try:
    # Load directly as Yangtze subset should be smaller
    X_flat = np.load(X_FLAT_PATH)
    Y_flat_raw = np.load(Y_FLAT_PATH)
    print(f"Loaded X_flat shape: {X_flat.shape}")
    print(f"Loaded Y_flat_raw shape: {Y_flat_raw.shape}")
except FileNotFoundError:
    print(f"Error: Data files not found. Ensure {X_FLAT_PATH} and {Y_FLAT_PATH} exist.")
    print("Run turn2.py for Yangtze first.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. 加载特征名称 ---
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

# --- 3. 预处理 ---
print("Preprocessing data...")
y_flat_binary = (Y_flat_raw > RAIN_THRESHOLD).astype(int)
del Y_flat_raw # Free memory

print("Splitting data into training and testing sets...")
# Use train_test_split for better shuffling
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_flat_binary,
    test_size=TEST_SIZE_RATIO,
    random_state=RANDOM_STATE,
    stratify=y_flat_binary # Stratify based on binary labels
)
del X_flat, y_flat_binary # Free memory

print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)
print(f"Train distribution: No Rain={train_counts[0]}, Rain={train_counts[1]}")
print(f"Test distribution: No Rain={test_counts[0]}, Rain={test_counts[1]}")


# --- 4. 定义并训练 XGBoost 模型 ---
print("Defining and training XGBoost model (Yangtze v4)...")
# Calculate scale_pos_weight based on the FULL training set
num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1
print(f"Calculated scale_pos_weight (from full train set): {scale_pos_weight:.4f}")

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'use_label_encoder': False,
    'n_estimators': 1000, # Keep high, rely on early stopping
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'gamma': 0.2,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist', # Use hist for efficiency
    'scale_pos_weight': scale_pos_weight,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS
}

model = xgb.XGBClassifier(**params)

print("Starting model training...")
start_time = time.time()
# Use the full training set, test set for evaluation
eval_set = [(X_train, y_train), (X_test, y_test)] # Evaluate on both train and test
model.fit(X_train, y_train,
          eval_set=eval_set,
          verbose=50) # Print progress every 50 rounds
end_time = time.time()
print(f"Training complete in {end_time - start_time:.2f} seconds.")
try:
    print(f"Best iteration: {model.best_iteration}")
    if hasattr(model, 'best_score'):
         print(f"Best score (test {model.evals_result()['validation_1'][params['eval_metric'][0]][model.best_iteration]:.4f})")
    else:
         results = model.evals_result()
         if 'validation_1' in results and params['eval_metric'][0] in results['validation_1']:
             best_score_val = results['validation_1'][params['eval_metric'][0]][model.best_iteration]
             print(f"Best score (test {params['eval_metric'][0]}): {best_score_val:.4f}")

except AttributeError:
    print("Could not retrieve best iteration/score attributes directly.")

# --- 5. 特征重要性 ---
print("\n--- Feature Importances (Yangtze v4) ---")
try:
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("Top 10 Features:")
    print(importance_df.head(10))

    plt.figure(figsize=(10, N_TOP_FEATURES_TO_PLOT / 2.0))
    top_features = importance_df.head(N_TOP_FEATURES_TO_PLOT)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"Top {N_TOP_FEATURES_TO_PLOT} Feature Importances (XGBoost Yangtze v4)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH)
    print(f"Feature importance plot saved to: {IMPORTANCE_PLOT_PATH}")
    plt.close()

except Exception as plot_e:
    print(f"Warning: Could not generate feature importance plot - {plot_e}")

# --- 6. 评估模型 ---
print("\n--- Evaluating Model on Test Set (Yangtze v4) ---")
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Save predictions
np.save(MODEL_PREDICTION_PATH, y_pred_proba)
print(f"Predictions saved to: {MODEL_PREDICTION_PATH}")

# Evaluate across thresholds
thresholds_to_evaluate = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
metrics_by_threshold = {}
for threshold in thresholds_to_evaluate:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    metrics = calculate_metrics(y_test, y_pred_threshold, title=f"XGBoost Yangtze v4 (Threshold {threshold:.2f})")
    metrics_by_threshold[threshold] = metrics

print("\n--- XGBoost Yangtze v4 Performance across different thresholds (Test Set) ---")
metrics_to_show = ['accuracy', 'pod', 'far', 'csi', 'fp', 'fn']
threshold_metrics_data = {}
for threshold, metrics in metrics_by_threshold.items():
    threshold_metrics_data[f'XGB_Yangtsu_v4_Thr_{threshold:.2f}'] = {metric: metrics.get(metric, float('nan')) for metric in metrics_to_show}

threshold_df = pd.DataFrame(threshold_metrics_data).T
threshold_df = threshold_df[metrics_to_show]
float_cols = ['accuracy', 'pod', 'far', 'csi']
for col in float_cols: threshold_df[col] = threshold_df[col].map('{:.4f}'.format)
int_cols = ['fp', 'fn']
for col in int_cols: threshold_df[col] = threshold_df[col].map('{:.0f}'.format)
print(threshold_df)
# Save threshold performance table
print(f"Saving threshold performance table to {PERFORMANCE_CSV_PATH}")
threshold_df.to_csv(PERFORMANCE_CSV_PATH)


# --- 7. 保存模型 ---
print(f"\nSaving the trained model to {MODEL_SAVE_PATH}...")
try:
    joblib.dump(model, MODEL_SAVE_PATH)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nScript finished.")
