import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# --- 配置 ---
PROJECT_DIR = "F:/rainfalldata"
# Use v2 data for analysis as it contains the superset of features
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_flat_features_v2.npy")
FEATURE_NAMES_V1_PATH = os.path.join(PROJECT_DIR, "feature_names.txt")
FEATURE_NAMES_V2_PATH = os.path.join(PROJECT_DIR, "feature_names_v2.txt")
CORRELATION_PLOT_PATH = os.path.join(PROJECT_DIR, "feature_correlation_v1_vs_v2.png")
CORRELATION_MATRIX_PATH = os.path.join(PROJECT_DIR, "feature_correlation_v1_vs_v2_matrix.csv")

N_SAMPLES_FOR_ANALYSIS = 1000000 # Analyze 1 million samples (adjust based on memory)
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.7 # Threshold for highlighting high correlations

# --- 1. 加载特征名称并分类 ---
print("Loading and comparing feature names...")
try:
    with open(FEATURE_NAMES_V1_PATH, 'r') as f:
        feature_names_v1 = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(feature_names_v1)} unique v1 feature names.")

    with open(FEATURE_NAMES_V2_PATH, 'r') as f:
        feature_names_v2_list = [line.strip() for line in f if line.strip()] # Keep order for indexing
        feature_names_v2_set = set(feature_names_v2_list)
    print(f"Loaded {len(feature_names_v2_set)} unique v2 feature names.")

    common_features = sorted(list(feature_names_v1.intersection(feature_names_v2_set)))
    v1_only_features = sorted(list(feature_names_v1 - feature_names_v2_set)) # Should be empty if v2 is superset
    v2_only_features = sorted(list(feature_names_v2_set - feature_names_v1))

    print(f"\nCommon features: {len(common_features)}")
    print(f"V1 only features: {len(v1_only_features)}")
    if v1_only_features: print(f"  V1 Only: {v1_only_features}")
    print(f"V2 only features: {len(v2_only_features)}")
    # print(f"  V2 Only: {v2_only_features}") # Optionally print the long list

except Exception as e:
    print(f"Error loading or comparing feature names: {e}")
    exit()

# --- 2. 定义关键特征组进行详细分析 ---
# Select common features that dropped in importance from v1 to v2
dropped_common_features = sorted([
    f for f in common_features if f in [
        'sin_day', 'cos_day', 'spatial_mean_3x3_GSMAP', 'spatial_mean_3x3_PERSIANN',
        'spatial_mean_3x3_CMORPH', 'spatial_mean_3x3_CHIRPS', 'lag_1_mean', 'lag_2_mean',
        'lag_3_mean', 'product_mean', 'product_std', 'spatial_center_diff_3x3_GSMAP',
        'spatial_center_diff_3x3_PERSIANN', 'threshold_proximity'
        # Add more if needed based on v1/v2 importance plots
    ]
])

# Select new v2 features that gained importance
important_new_v2_features = sorted([
    f for f in v2_only_features if f in [
        'spatial_mean_5x5_GSMAP', 'spatial_mean_5x5_PERSIANN', 'spatial_max_5x5_GSMAP',
        'spatial_max_5x5_PERSIANN', 'gradient_magnitude_GSMAP', 'gradient_magnitude_PERSIANN',
        'spatial_mean_5x5_IMERG', 'spatial_max_5x5_IMERG', 'spatial_mean_5x5_CMORPH',
        'spatial_mean_5x5_CHIRPS', 'spatial_max_5x5_CMORPH', 'spatial_max_5x5_CHIRPS',
        'gradient_magnitude_mean', 'lag_1_coef_of_variation', 'lag_1_rain_count',
        'lag_2_rain_count', 'lag_3_rain_count', 'window_7_mean_GSMAP', 'window_7_std_PERSIANN',
        'spatial_center_diff_5x5_GSMAP', 'spatial_center_diff_5x5_PERSIANN',
        'intensity_bins_count_0', 'intensity_bins_count_1', 'intensity_bins_count_2', 'intensity_bins_count_>=3'
        # Add more based on v2 importance plot
    ]
])

print(f"\nSelected 'Dropped Common' Features ({len(dropped_common_features)}): {dropped_common_features}")
print(f"Selected 'Important New V2' Features ({len(important_new_v2_features)}): {important_new_v2_features}")

# Combine lists for analysis
features_to_analyze = sorted(list(set(dropped_common_features + important_new_v2_features)))
print(f"\nTotal unique features for correlation analysis ({len(features_to_analyze)}): {features_to_analyze}")

# Get indices of these features in the v2 list
try:
    feature_indices = [feature_names_v2_list.index(f) for f in features_to_analyze]
except ValueError as e:
    print(f"Error: One of the selected features not found in v2 feature list: {e}")
    # This might happen if the manual lists above contain typos or features not actually present
    print("Please double-check the feature lists against feature_names_v2.txt")
    exit()

# --- 3. 加载数据子集 (Using v2 data) ---
print(f"\nLoading data subset ({N_SAMPLES_FOR_ANALYSIS} samples from v2 data)...")
start_load = time.time()
try:
    X_flat_mmap = np.load(X_FLAT_PATH, mmap_mode='r')
    n_total_samples = X_flat_mmap.shape[0]
    n_features_total = X_flat_mmap.shape[1]

    if n_features_total != len(feature_names_v2_list):
         print(f"FATAL: Mismatch between feature count in {X_FLAT_PATH} ({n_features_total}) and {FEATURE_NAMES_V2_PATH} ({len(feature_names_v2_list)})!")
         exit()

    if n_total_samples < N_SAMPLES_FOR_ANALYSIS:
        print(f"Warning: Requested sample size ({N_SAMPLES_FOR_ANALYSIS}) > total samples ({n_total_samples}). Using all samples.")
        N_SAMPLES_FOR_ANALYSIS = n_total_samples
        sample_indices = np.arange(n_total_samples)
    else:
        np.random.seed(RANDOM_STATE)
        sample_indices = np.random.choice(n_total_samples, N_SAMPLES_FOR_ANALYSIS, replace=False)
        sample_indices.sort()

    print("Reading selected rows...")
    X_subset_rows = X_flat_mmap[sample_indices]
    print("Selecting feature columns...")
    X_subset = X_subset_rows[:, feature_indices]

    if hasattr(X_flat_mmap, '_mmap'):
        X_flat_mmap._mmap.close()
    del X_flat_mmap, X_subset_rows

    print(f"Loaded data subset shape: {X_subset.shape}")
    end_load = time.time()
    print(f"Data loading finished in {end_load - start_load:.2f} seconds.")

except Exception as e:
    print(f"Error loading data subset: {e}")
    exit()

# --- 4. 计算和分析相关性 ---
print("\nCalculating correlation matrix...")
start_corr = time.time()
df_subset = pd.DataFrame(X_subset, columns=features_to_analyze)
del X_subset

correlation_matrix = df_subset.corr()
end_corr = time.time()
print(f"Correlation calculation finished in {end_corr - start_corr:.2f} seconds.")

try:
    print(f"Saving correlation matrix to {CORRELATION_MATRIX_PATH}...")
    correlation_matrix.to_csv(CORRELATION_MATRIX_PATH)
except Exception as e:
    print(f"Error saving correlation matrix: {e}")

print(f"\n--- Correlations between 'Dropped Common' and 'Important New V2' Features (Threshold: |{CORRELATION_THRESHOLD}|) ---")
# Ensure both lists are non-empty before slicing
if dropped_common_features and important_new_v2_features:
    # Check if all features are present in the correlation matrix columns/index
    valid_dropped = [f for f in dropped_common_features if f in correlation_matrix.index]
    valid_new = [f for f in important_new_v2_features if f in correlation_matrix.columns]

    if valid_dropped and valid_new:
        corr_submatrix = correlation_matrix.loc[valid_dropped, valid_new]

        high_corr_pairs = corr_submatrix[corr_submatrix.abs() > CORRELATION_THRESHOLD].stack().reset_index()
        high_corr_pairs.columns = ['Dropped_Common_Feature', 'Important_New_V2_Feature', 'Correlation']

        if not high_corr_pairs.empty:
            print(f"High Correlation Pairs (|Correlation| > {CORRELATION_THRESHOLD}):")
            print(high_corr_pairs.to_string())
        else:
            print(f"No high correlations (|Correlation| > {CORRELATION_THRESHOLD}) found between the selected groups.")
    else:
        print("Warning: Could not find all selected features in the calculated correlation matrix.")
else:
    print("Warning: One or both feature groups ('Dropped Common', 'Important New V2') are empty. Skipping detailed correlation analysis.")


# --- 5. 可视化 ---
print("\nGenerating correlation heatmap...")
try:
    plt.figure(figsize=(max(15, len(features_to_analyze)*0.45), max(12, len(features_to_analyze)*0.45)))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
    plt.title(f'Feature Correlation Matrix (V1 vs V2 Analysis - {N_SAMPLES_FOR_ANALYSIS} samples)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CORRELATION_PLOT_PATH)
    print(f"Correlation heatmap saved to: {CORRELATION_PLOT_PATH}")
    plt.close()
except Exception as e:
    print(f"Error generating heatmap: {e}")

print("\nAnalysis finished.")
