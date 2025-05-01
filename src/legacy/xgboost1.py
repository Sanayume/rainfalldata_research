import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas

# --- 配置 ---
PROJECT_DIR = "F:/rainfalldata"
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target.npy")
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "xgboost_default_full_model.joblib")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names.txt")  # Path to feature names file
RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 30
N_TOP_FEATURES_TO_PLOT = 50  # Number of top features to show in the importance plot

# --- 创建输出目录 ---
os.makedirs(PROJECT_DIR, exist_ok=True)

# --- 辅助函数：计算性能指标 ---
def calculate_metrics(y_true, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nAccuracy Score:")
    print(f"{accuracy_score(y_true, y_pred):.4f}")

# --- 1. 加载数据 ---
print("Loading flattened data...")
X_flat = np.load(X_FLAT_PATH)
Y_flat_raw = np.load(Y_FLAT_PATH)

# --- 2. 加载特征名称 ---
print(f"Loading feature names from {FEATURE_NAMES_PATH}...")
try:
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(feature_names)} feature names.")
    if len(feature_names) != X_flat.shape[1]:
        print(f"Warning: Number of feature names ({len(feature_names)}) does not match data columns ({X_flat.shape[1]})!")
except Exception as e:
    print(f"Error loading feature names: {e}. Using generic names.")
    feature_names = [f'f{i}' for i in range(X_flat.shape[1])]

# --- 3. 预处理 ---
print("Preprocessing data...")
Y_flat = (Y_flat_raw > RAIN_THRESHOLD).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_flat, Y_flat, test_size=TEST_SIZE_RATIO, random_state=RANDOM_STATE)

# --- 4. 定义并训练 XGBoost 模型 ---
print("Defining and training XGBoost model...")
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "scale_pos_weight": scale_pos_weight,
    "random_state": RANDOM_STATE
}
model = xgb.XGBClassifier(**params)

print("Starting model training...")
start_time = time.time()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
end_time = time.time()
print(f"Training complete in {end_time - start_time:.2f} seconds.")

# --- 5. 特征重要性 ---
print("\n--- Feature Importances ---")
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
    plt.title(f"Top {N_TOP_FEATURES_TO_PLOT} Feature Importances (XGBoost)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    importance_plot_path = os.path.join(PROJECT_DIR, "xgboost_feature_importance.png")
    plt.savefig(importance_plot_path)
    print(f"Feature importance plot saved to: {importance_plot_path}")
    plt.close()

except Exception as plot_e:
    print(f"Warning: Could not generate feature importance plot - {plot_e}")

# --- 6. 评估模型 ---
print("\nEvaluating model on test set...")
y_pred = model.predict(X_test)
calculate_metrics(y_test, y_pred)

# --- 7. 保存模型 ---
print(f"Saving model to {MODEL_SAVE_PATH}...")
joblib.dump(model, MODEL_SAVE_PATH)
print("Model saved.")

print("\nScript finished.")