import numpy as np
import xgboost as xgb
import os
import joblib

# --- 配置 ---
PROJECT_DIR = "F:\\rainfalldata"
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features.npy") # Full features (read-only)
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_flat_target.npy") # Full target (read-only)
# --- Add path for base model's probabilities on TEST set ---
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "predictions")
BASE_PROBS_TEST_PATH = os.path.join(PREDICTIONS_DIR, "base_probs_test.npy") # Probabilities for test set

# Model paths
FP_EXPERT_MODEL_PATH = os.path.join(PROJECT_DIR, "ensemble_learning", "models", "fp_expert_model.joblib") # Load the NEW FP expert

# Output directory for meta-features
META_FEATURE_DIR = os.path.join(PROJECT_DIR, "ensemble_learning", "meta_features")
META_X_PATH = os.path.join(META_FEATURE_DIR, "X_meta.npy")
META_Y_PATH = os.path.join(META_FEATURE_DIR, "y_meta.npy") # True labels for the meta set

# Data split config
TEST_SIZE_RATIO = 0.2
RAIN_THRESHOLD = 0.1

# --- 创建输出目录 ---
os.makedirs(META_FEATURE_DIR, exist_ok=True)

# --- 1. 加载数据标识符 ---
print("Loading data identifiers...")
try:
    # Get shapes
    print("  Getting X shape...")
    x_memmap = np.load(X_FLAT_PATH, mmap_mode='r')
    x_shape = x_memmap.shape
    del x_memmap

    print("  Getting Y shape...")
    y_memmap = np.load(Y_FLAT_PATH, mmap_mode='r')
    y_shape = y_memmap.shape
    del y_memmap

    # --- Load base probabilities (test) shape ---
    print("  Getting Base Probs Test shape...")
    base_probs_test_memmap = np.load(BASE_PROBS_TEST_PATH, mmap_mode='r')
    base_probs_test_shape = base_probs_test_memmap.shape
    del base_probs_test_memmap
    # ---

    print(f"Full X shape: {x_shape}")
    print(f"Full Y shape: {y_shape}")
    print(f"Full Base Probs Test shape: {base_probs_test_shape}")

except Exception as e:
    print(f"Error getting data shapes: {e}")
    exit()

# --- 2. 划分并加载测试集到内存 ---
n_samples = y_shape[0]
split_idx = int(n_samples * (1 - TEST_SIZE_RATIO))
n_test_samples = n_samples - split_idx

print(f"Loading FULL test data slice (samples {split_idx} to {n_samples}) into memory...")
try:
    # Load X_test, Y_test_raw, base_probs_test into memory
    X_test = np.load(X_FLAT_PATH, mmap_mode='r')[split_idx:]
    Y_test_raw = np.load(Y_FLAT_PATH, mmap_mode='r')[split_idx:]
    # --- Load base_probs_test (NO SLICING) ---
    base_probs_test = np.load(BASE_PROBS_TEST_PATH)
    # ---
    print("Test data slices (X, Y, Base_Probs) loaded into memory.")

    # ... (Validation of X_test) ...
    print("Validating loaded X_test slice for NaN/Inf...")
    if np.isnan(X_test).any() or np.isinf(X_test).any():
         print("    ERROR: Loaded X_test slice contains NaN or Inf!")
         exit()
    else:
        print("    X_test slice validation passed (in-memory).")

    y_test_true = (Y_test_raw > RAIN_THRESHOLD).astype(int)
    del Y_test_raw

except MemoryError:
    print(f"MemoryError: Not enough RAM to load the full test slice...")
    exit()
except Exception as e:
    print(f"Error loading test data slice directly into memory: {e}")
    exit()

print(f"Test data shape (in memory): X={X_test.shape}, y={y_test_true.shape}, base_probs={base_probs_test.shape}")
if len(base_probs_test) != n_test_samples:
    print(f"Error: Mismatch between y_test_true ({n_test_samples}) and base_probs_test ({len(base_probs_test)}) sample count!")
    exit()


# --- 3. 加载训练好的 FP 专家模型 ---
print("Loading trained FP expert model...")
try:
    fp_expert_model = joblib.load(FP_EXPERT_MODEL_PATH)
    print("  FP expert model loaded.")
except Exception as e:
    print(f"Error loading FP expert model: {e}")
    exit()

# --- 4. 生成元特征 (Base_Prob, FP_Expert_Prob) ---
print("Generating meta-features (Base_Prob, FP_Expert_Prob)...")

CHUNK_SIZE = 10000
# --- Update number of meta features ---
n_meta_features = 2 # Base_Prob, FP_Expert_Prob
# ---

# Pre-allocate array for meta features
X_meta = np.zeros((n_test_samples, n_meta_features), dtype=np.float32) # Adjust shape

# Predict in chunks
for i_chunk, start_idx in enumerate(range(0, n_test_samples, CHUNK_SIZE)):
    end_idx = min(start_idx + CHUNK_SIZE, n_test_samples)
    print(f"  Processing chunk {i_chunk + 1}: samples {start_idx} to {end_idx-1}")

    # Slicing from the in-memory arrays
    X_chunk = X_test[start_idx:end_idx]
    base_prob_chunk = base_probs_test[start_idx:end_idx]
    print(f"    X_chunk shape: {X_chunk.shape}, dtype: {X_chunk.dtype}")
    print(f"    base_prob_chunk shape: {base_prob_chunk.shape}, dtype: {base_prob_chunk.dtype}")

    # Prepare input for FP expert (101 features)
    base_prob_chunk_reshaped = base_prob_chunk.reshape(-1, 1)
    X_fp_expert_input_chunk = np.concatenate((X_chunk, base_prob_chunk_reshaped), axis=1)
    print(f"    FP expert input shape: {X_fp_expert_input_chunk.shape}")

    # Predict with FP expert
    chunk_meta_features = [0.0] * n_meta_features
    try:
        # ... (Validation of X_chunk) ...
        print(f"    Validating X_chunk for NaN/Inf...")
        if np.isnan(X_chunk).any() or np.isinf(X_chunk).any():
             print(f"    ERROR: Chunk {i_chunk + 1} contains invalid values!")
             exit()
        else:
             print(f"    X_chunk validation passed.")

        # --- Prediction logic ---
        print(f"    Predicting with FP Expert Model...")
        # Predict probability of being FP (class 1)
        pred_fp_expert_chunk = fp_expert_model.predict_proba(X_fp_expert_input_chunk)[:, 1]
        chunk_meta_features[1] = pred_fp_expert_chunk # Index 1: FP_Expert_Prob

        # Base probability is already loaded
        chunk_meta_features[0] = base_prob_chunk # Index 0: Base_Prob
        # --- End Prediction ---

        # Stack predictions
        print(f"    Stacking predictions for chunk {i_chunk + 1}...")
        # Order: Base Prob, FP Prob
        X_meta[start_idx:end_idx, :] = np.stack(chunk_meta_features, axis=1)
        print(f"    Stored predictions for chunk {i_chunk + 1}.")

    except BaseException as e: # Catch errors
        print(f"ERROR during processing chunk {i_chunk + 1}: {type(e).__name__} - {e}")
        # Print shapes for debugging if it's a shape mismatch
        if 'shape' in str(e).lower():
            print(f"      Input shape to FP expert: {X_fp_expert_input_chunk.shape}")
        exit()

    # Clear chunk data
    print(f"    Clearing chunk {i_chunk + 1} data...")
    del X_chunk, base_prob_chunk, X_fp_expert_input_chunk, chunk_meta_features, pred_fp_expert_chunk
    print(f"    Finished processing chunk {i_chunk + 1}.")

# --- 5. 保存元特征和对应的真实标签 ---
print(f"\nMeta-feature matrix shape: {X_meta.shape}") # Should now be (n_test_samples, 2)
print("Saving meta-features and labels...")
try:
    np.save(META_X_PATH, X_meta)
    np.save(META_Y_PATH, y_test_true)
    print(f"Meta-features saved to: {META_X_PATH}")
    print(f"Meta labels saved to: {META_Y_PATH}")
except Exception as e:
    print(f"Error saving meta-features or labels: {e}")

print("\nNext step: Train the meta-learner using the generated meta-features.")
