import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# --- 配置 ---
PROJECT_DIR = "F:/rainfalldata"
# Use v1 data files
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features.npy") # v1 data
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names.txt") # v1 feature names
CORRELATION_PLOT_PATH = os.path.join(PROJECT_DIR, "feature_correlation_v1_analysis.png")
CORRELATION_MATRIX_PATH = os.path.join(PROJECT_DIR, "feature_correlation_v1_matrix.csv") # Save matrix

N_SAMPLES_FOR_ANALYSIS = 1000000 # Analyze 1 million samples (adjust based on memory)
RANDOM_STATE = 42

# --- 1. 加载特征名称 ---
print("Loading feature names (v1)...")
try:
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names_v1 = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(feature_names_v1)} v1 feature names.")
except Exception as e:
    print(f"Error loading feature names: {e}")
    exit()

# --- 2. 定义关键特征组 ---
# Select Top ~30 features from v1 importance results (based on previous analysis/plots)
top_v1_features = [
    'sin_day', 'cos_day', 'spatial_mean_GSMAP', 'spatial_mean_PERSIANN',
    'lag_1_values_GSMAP', 'lag_1_values_PERSIANN', 'spatial_mean_CMORPH',
    'spatial_mean_CHIRPS', 'lag_1_mean', 'lag_2_mean', 'lag_3_mean',
    'window_15_range', 'window_7_range', 'product_mean', 'product_std',
    'spatial_max_GSMAP', 'spatial_max_PERSIANN', 'spatial_std_GSMAP',
    'spatial_std_PERSIANN', 'lag_2_values_GSMAP', 'lag_2_values_PERSIANN',
    'spatial_center_diff_GSMAP', 'spatial_center_diff_PERSIANN',
    'coef_of_variation', 'threshold_proximity', 'rain_product_count',
    'lag_1_std', 'lag_2_std', 'lag_3_std', 'window_15_mean'
]

# Filter list to ensure they exist (should be fine for v1)
features_to_analyze = [f for f in top_v1_features if f in feature_names_v1]
if len(features_to_analyze) != len(top_v1_features):
    print("Warning: Some selected top_v1_features were not found in feature_names.txt")

print(f"\nTotal unique features for v1 correlation analysis ({len(features_to_analyze)}): {features_to_analyze}")

# Get indices of these features in the v1 list
try:
    feature_indices = [feature_names_v1.index(f) for f in features_to_analyze]
except ValueError as e:
    print(f"Error: One of the selected features not found in v1 feature list: {e}")
    exit()

# --- 3. 加载数据子集 ---
print(f"\nLoading data subset ({N_SAMPLES_FOR_ANALYSIS} samples from v1 data)...")
start_load = time.time()
try:
    X_flat_mmap = np.load(X_FLAT_PATH, mmap_mode='r')
    n_total_samples = X_flat_mmap.shape[0]
    n_features_total = X_flat_mmap.shape[1]

    if n_features_total != len(feature_names_v1):
         print(f"FATAL: Mismatch between feature count in {X_FLAT_PATH} ({n_features_total}) and {FEATURE_NAMES_PATH} ({len(feature_names_v1)})!")
         exit()

    if n_total_samples < N_SAMPLES_FOR_ANALYSIS:
        print(f"Warning: Requested sample size ({N_SAMPLES_FOR_ANALYSIS}) > total samples ({n_total_samples}). Using all samples.")
        N_SAMPLES_FOR_ANALYSIS = n_total_samples
        sample_indices = np.arange(n_total_samples)
    else:
        np.random.seed(RANDOM_STATE)
        sample_indices = np.random.choice(n_total_samples, N_SAMPLES_FOR_ANALYSIS, replace=False)
        sample_indices.sort() # Sorting might improve read performance slightly

    # Load selected rows first, then select columns
    print("Reading selected rows...")
    X_subset_rows = X_flat_mmap[sample_indices]
    print("Selecting feature columns...")
    X_subset = X_subset_rows[:, feature_indices]

    # Close mmap
    if hasattr(X_flat_mmap, '_mmap'):
        X_flat_mmap._mmap.close()
    del X_flat_mmap, X_subset_rows # Free memory

    print(f"Loaded data subset shape: {X_subset.shape}")
    end_load = time.time()
    print(f"Data loading finished in {end_load - start_load:.2f} seconds.")

except Exception as e:
    print(f"Error loading data subset: {e}")
    exit()

# --- 4. 计算和分析相关性 ---
print("\nCalculating correlation matrix (v1 features)...")
start_corr = time.time()
df_subset = pd.DataFrame(X_subset, columns=features_to_analyze)
del X_subset # Free memory

# Calculate correlation matrix
correlation_matrix = df_subset.corr()
end_corr = time.time()
print(f"Correlation calculation finished in {end_corr - start_corr:.2f} seconds.")

# Save the full correlation matrix
try:
    print(f"Saving correlation matrix to {CORRELATION_MATRIX_PATH}...")
    correlation_matrix.to_csv(CORRELATION_MATRIX_PATH)
except Exception as e:
    print(f"Error saving correlation matrix: {e}")

# Analyze correlations within Top V1 features
print("\n--- Correlations within Top V1 Features ---")
# Find and print high correlations (e.g., absolute value > 0.7)
# Exclude self-correlation (diagonal)
corr_matrix_no_diag = correlation_matrix.mask(np.equal(*np.indices(correlation_matrix.shape)))
high_corr_pairs = corr_matrix_no_diag[corr_matrix_no_diag.abs() > 0.7].stack().reset_index()
high_corr_pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']
# Remove duplicate pairs (e.g., (A, B) and (B, A)) by sorting names
high_corr_pairs['sorted_pair'] = high_corr_pairs.apply(lambda row: tuple(sorted((row['Feature_1'], row['Feature_2']))), axis=1)
high_corr_pairs = high_corr_pairs.drop_duplicates(subset='sorted_pair').drop(columns='sorted_pair')


if not high_corr_pairs.empty:
    print("High Correlation Pairs (|Correlation| > 0.7):")
    print(high_corr_pairs.to_string())
else:
    print("No high correlations (|Correlation| > 0.7) found between the selected top v1 features.")

# --- 5. 可视化 ---
print("\nGenerating correlation heatmap (v1 features)...")
try:
    plt.figure(figsize=(max(15, len(features_to_analyze)*0.4), max(12, len(features_to_analyze)*0.4))) # Adjust size dynamically
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5) # Annot=False for large matrices
    plt.title(f'Feature Correlation Matrix (V1 Analysis - {N_SAMPLES_FOR_ANALYSIS} samples)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CORRELATION_PLOT_PATH)
    print(f"Correlation heatmap saved to: {CORRELATION_PLOT_PATH}")
    plt.close()
except Exception as e:
    print(f"Error generating heatmap: {e}")

print("\nAnalysis finished.")
