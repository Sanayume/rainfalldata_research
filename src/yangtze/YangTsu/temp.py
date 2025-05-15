import numpy as np
from sklearn.model_selection import train_test_split
import os
import time

# --- 配置 ---
# V1 特征和目标文件路径
V1_FEATURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))), # 假设此脚本在类似 src/temp/ 目录下
    "results", "yangtze", "features"
)
X_FLAT_FULL_PATH = os.path.join(V1_FEATURES_DIR, "X_Yangtsu_flat_features.npy")
Y_FLAT_FULL_PATH = os.path.join(V1_FEATURES_DIR, "Y_Yangtsu_flat_target.npy")

# 输出目录 (与 xgboost1.py 修改版中的 KFOLD_OUTPUT_DIR 一致)
KFOLD_OUTPUT_DIR = os.path.join(V1_FEATURES_DIR, "kfold_optimization_v1")
os.makedirs(KFOLD_OUTPUT_DIR, exist_ok=True)

# 输出文件名
X_TRAIN_CV_POOL_SAVE_PATH = os.path.join(KFOLD_OUTPUT_DIR, "X_train_cv_pool_v1_opt.npy")
Y_TRAIN_CV_POOL_SAVE_PATH = os.path.join(KFOLD_OUTPUT_DIR, "y_train_cv_pool_v1_opt.npy")
X_HOLDOUT_TEST_SAVE_PATH = os.path.join(KFOLD_OUTPUT_DIR, "X_holdout_test_v1_opt.npy")
Y_HOLDOUT_TEST_SAVE_PATH = os.path.join(KFOLD_OUTPUT_DIR, "y_holdout_test_v1_opt.npy")


# 与 xgboost1.py 中用于划分保持测试集 (Hold-out Test Set) 的参数保持完全一致
RAIN_THRESHOLD_ORIGINAL_Y = 0.1 # 用于将原始Y转换为二进制
TEST_SIZE_RATIO_HOLDOUT = 0.2
RANDOM_STATE_SPLIT = 42 # 确保与 xgboost1.py 中的 random_state 一致

def main():
    print("--- Temporary Script to Save Train/CV Pool and Hold-out Test Data ---")

    # --- 1. 加载完整数据 ---
    print("\n--- Step 1: Loading Full V1 Flattened Data ---")
    start_load_flat = time.time()
    if not (os.path.exists(X_FLAT_FULL_PATH) and os.path.exists(Y_FLAT_FULL_PATH)):
        raise FileNotFoundError(f"Full V1 flattened data files not found in {V1_FEATURES_DIR}.")

    try:
        print(f"Attempting to load {X_FLAT_FULL_PATH}...")
        X_flat_full = np.load(X_FLAT_FULL_PATH)
        print(f"Attempting to load {Y_FLAT_FULL_PATH}...")
        Y_flat_full_raw = np.load(Y_FLAT_FULL_PATH)
    except Exception as e:
        print(f"Error loading flattened data: {e}")
        raise

    print(f"Loaded X_flat_full: shape {X_flat_full.shape}")
    print(f"Loaded Y_flat_full_raw: shape {Y_flat_full_raw.shape}")
    end_load_flat = time.time()
    print(f"Full data loading finished in {end_load_flat - start_load_flat:.2f} seconds.")

    # --- 2. 数据预处理与划分 (与 xgboost1.py 一致) ---
    print("\n--- Step 2: Data Preprocessing and Splitting ---")
    Y_flat_full_binary = (Y_flat_full_raw > RAIN_THRESHOLD_ORIGINAL_Y).astype(int)
    del Y_flat_full_raw

    print(f"Splitting full dataset using test_size={TEST_SIZE_RATIO_HOLDOUT} and random_state={RANDOM_STATE_SPLIT}...")
    # 确保这里的 stratify 参数与 xgboost1.py 中一致！
    # 从您的日志看，xgboost1.py 中使用了 stratify=Y_flat_full_binary
    X_train_cv_pool, X_holdout_test, y_train_cv_pool, y_holdout_test = train_test_split(
        X_flat_full, Y_flat_full_binary,
        test_size=TEST_SIZE_RATIO_HOLDOUT,
        random_state=RANDOM_STATE_SPLIT,
        stratify=Y_flat_full_binary # 保持与 xgboost1.py 一致
    )
    del X_flat_full, Y_flat_full_binary

    print(f"Training/CV Pool shapes: X={X_train_cv_pool.shape}, y={y_train_cv_pool.shape}")
    print(f"Hold-out Test Set shapes: X={X_holdout_test.shape}, y={y_holdout_test.shape}")
    train_cv_pool_counts = np.bincount(y_train_cv_pool)
    holdout_test_counts = np.bincount(y_holdout_test)
    print(f"Training/CV Pool distribution: No Rain={train_cv_pool_counts[0]}, Rain={train_cv_pool_counts[1]}")
    print(f"Hold-out Test Set distribution: No Rain={holdout_test_counts[0]}, Rain={holdout_test_counts[1]}")

    # --- 3. 保存所需文件 ---
    print("\n--- Step 3: Saving Split Data ---")

    print(f"Saving X_train_cv_pool to: {X_TRAIN_CV_POOL_SAVE_PATH}")
    np.save(X_TRAIN_CV_POOL_SAVE_PATH, X_train_cv_pool)
    print("X_train_cv_pool saved successfully.")

    print(f"Saving y_train_cv_pool to: {Y_TRAIN_CV_POOL_SAVE_PATH}")
    np.save(Y_TRAIN_CV_POOL_SAVE_PATH, y_train_cv_pool)
    print("y_train_cv_pool saved successfully.")

    print(f"Saving X_holdout_test to: {X_HOLDOUT_TEST_SAVE_PATH}")
    np.save(X_HOLDOUT_TEST_SAVE_PATH, X_holdout_test)
    print("X_holdout_test saved successfully.")

    print(f"Saving y_holdout_test to: {Y_HOLDOUT_TEST_SAVE_PATH}")
    np.save(Y_HOLDOUT_TEST_SAVE_PATH, y_holdout_test)
    print("y_holdout_test saved successfully.")

    print("\n--- Temporary Script Finished ---")
    print("Required data for expert model training should now be saved.")

if __name__ == "__main__":
    main()