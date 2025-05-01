import numpy as np
import joblib
import xgboost as xgb # Import xgboost to ensure its libraries are loaded
import os

# --- 配置 ---
PROJECT_DIR = "F:\\rainfalldata"
# --- Choose which model to test ---
# MODEL_PATH = os.path.join(PROJECT_DIR, "xgboost_default_full_model.joblib")
MODEL_PATH = os.path.join(PROJECT_DIR, "ensemble_learning", "models", "fn_expert_model.joblib")
# MODEL_PATH = os.path.join(PROJECT_DIR, "ensemble_learning", "models", "fp_expert_model.joblib")

N_FEATURES = 100 # Number of features the model expects
N_SAMPLES_TEST = 10 # Number of dummy samples to predict

print(f"--- Testing Model Prediction ---")
print(f"Python version: {__import__('sys').version}")
print(f"NumPy version: {np.__version__}")
print(f"Joblib version: {joblib.__version__}")
print(f"XGBoost version: {xgb.__version__}")
print(f"Loading model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found.")
    exit()

try:
    print("Attempting to load model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    print(f"Model type: {type(model)}")

    # Create dummy data
    print(f"Creating dummy data with shape ({N_SAMPLES_TEST}, {N_FEATURES})...")
    dummy_data = np.random.rand(N_SAMPLES_TEST, N_FEATURES).astype(np.float32)
    print("Dummy data created.")

    # Attempt prediction
    print("Attempting to call predict_proba...")
    pred_proba = model.predict_proba(dummy_data)
    print("predict_proba call successful.")
    print(f"Prediction probabilities shape: {pred_proba.shape}") # Should be (N_SAMPLES_TEST, 2)

    print("\n--- Test Successful ---")

except BaseException as e:
    print(f"\n--- ERROR during test ---")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

