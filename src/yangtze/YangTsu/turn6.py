from loaddata import mydata # Assuming loaddata.py is in the same directory
import numpy as np
import os
import time

# --- Constants ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
X_flat_filename = os.path.join(OUTPUT_DIR, "X_Yangtsu_flat_features_v6.npy")
Y_flat_filename = os.path.join(OUTPUT_DIR, "Y_Yangtsu_flat_target_v6.npy")
feature_names_filename = os.path.join(OUTPUT_DIR, "feature_names_yangtsu_v6.txt")
MAX_LOOKBACK = 30 # Based on longest window/lag used
EPSILON = 1e-6 # Small constant to prevent division by zero
RAIN_THR = 0.1 # Threshold for counting rain products

# Helper function for safe division
def safe_divide(numerator, denominator, default=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / (denominator + EPSILON)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

# --- 1. Data Loading ---
start_time = time.time()
print("Loading Yangtze point data...")
ALL_DATA = mydata()
# Load Yangtze POINT data using the new method
# X_raw shape: (n_products, time, n_points), Y_raw shape: (time, n_points)
# Use basin_mask_value=2 for Yangtze
# MODIFIED: Changed get_basin_spatial_data to get_basin_point_data for discrete point data
X_raw, Y_raw = ALL_DATA.get_basin_point_data(basin_mask_value=2)
product_names = ALL_DATA.get_products() # Use getter
n_products, nday, n_points = X_raw.shape # Get dimensions
print(f"Data loaded. X_raw shape: {X_raw.shape}, Y_raw shape: {Y_raw.shape}")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# --- 2. Data Preparation ---
print("\nPreparing data...")
start_prep_time = time.time()
# Transpose X to (time, n_points, n_products)
X = np.transpose(X_raw, (1, 2, 0)).astype(np.float32) # (time, n_points, n_products)
del X_raw # Free memory
Y = Y_raw.astype(np.float32) # (time, n_points)
del Y_raw # Free memory
print(f"Transposed X shape: {X.shape}") # Should be (1827, n_points, 6)

# --- 3. Handle Time Dependency ---
print("\nHandling time dependency...")
# nday, n_points, n_products already defined
# Define the valid time range after truncation
valid_time_slice = slice(MAX_LOOKBACK, nday)
n_valid_days = nday - MAX_LOOKBACK

# Align X and Y with the valid time range
X_aligned = X[valid_time_slice] # Shape: (n_valid_days, n_points, n_products)
Y_aligned = Y[valid_time_slice] # Shape: (n_valid_days, n_points)
print(f"Aligned X shape: {X_aligned.shape}")
print(f"Aligned Y shape: {Y_aligned.shape}")
print(f"Data preparation time: {time.time() - start_prep_time:.2f} seconds")

# --- 4. Feature Engineering (Mirroring national turn1.py structure) ---
print("\nStarting feature engineering...")
start_feat_time = time.time()
features_dict = {} # Use dictionary approach

# --- 4.1 Basic Features ---
print("  Calculating basic features...")
features_dict['raw_values'] = X_aligned # Shape: (n_valid_days, n_points, 6)

# --- 4.2 Multi-product Consistency/Difference Features ---
print("  Calculating multi-product stats (current time)...")
# Calculate stats across the product dimension (axis=2)
features_dict['product_mean'] = np.nanmean(X_aligned, axis=2, keepdims=True).astype(np.float32)
features_dict['product_std'] = np.nanstd(X_aligned, axis=2, keepdims=True).astype(np.float32)
features_dict['product_median'] = np.nanmedian(X_aligned, axis=2, keepdims=True).astype(np.float32)
product_max = np.nanmax(X_aligned, axis=2, keepdims=True).astype(np.float32)
product_min = np.nanmin(X_aligned, axis=2, keepdims=True).astype(np.float32)
features_dict['product_max'] = product_max
features_dict['product_min'] = product_min
features_dict['product_range'] = (product_max - product_min).astype(np.float32)
X_rain = (np.nan_to_num(X_aligned, nan=0.0) > RAIN_THR)
features_dict['rain_product_count'] = np.sum(X_rain, axis=2, keepdims=True).astype(np.float32)

print("  Calculating periodicity features...")
days_in_year = 365.25
day_index_original = np.arange(nday, dtype=np.float32)
day_of_year = day_index_original % days_in_year
sin_time = np.sin(2 * np.pi * day_of_year / days_in_year).astype(np.float32)
cos_time = np.cos(2 * np.pi * day_of_year / days_in_year).astype(np.float32)
# Slice and expand: (n_valid_days,) -> (n_valid_days, 1, 1) -> broadcast to (n_valid_days, n_points, 1)
sin_time_aligned = sin_time[valid_time_slice, np.newaxis, np.newaxis] * np.ones((1, n_points, 1), dtype=np.float32)
cos_time_aligned = cos_time[valid_time_slice, np.newaxis, np.newaxis] * np.ones((1, n_points, 1), dtype=np.float32)
features_dict['sin_day'] = sin_time_aligned
features_dict['cos_day'] = cos_time_aligned

# Season One-Hot
month = (day_of_year // 30.4375).astype(int) % 12 + 1
season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
season = np.array([season_map[m] for m in month])
seasons_onehot = np.zeros((nday, 4), dtype=np.float32)
seasons_onehot[np.arange(nday), season] = 1
# Slice and expand: (n_valid_days, 4) -> (n_valid_days, 1, 4) -> broadcast to (n_valid_days, n_points, 4)
season_aligned = seasons_onehot[valid_time_slice, np.newaxis, :] * np.ones((1, n_points, 1), dtype=np.float32)
features_dict['season_onehot'] = season_aligned

# 4.3.2 Lag Features (t-1, t-2, t-3)
print("  Calculating lag features...")
lag_data_cache = {}
lag_mean_cache = {}
lag_std_cache = {}
for lag in [1, 2, 3]:
    print(f"    Lag {lag}...")
    lag_slice = slice(MAX_LOOKBACK - lag, nday - lag)
    lag_data = X[lag_slice] # Shape: (n_valid_days, n_points, n_products)
    lag_data_cache[lag] = lag_data
    features_dict[f'lag_{lag}_values'] = lag_data

    lag_mean = np.nanmean(lag_data, axis=2, keepdims=True).astype(np.float32)
    lag_std = np.nanstd(lag_data, axis=2, keepdims=True).astype(np.float32)
    lag_mean_cache[lag] = lag_mean
    lag_std_cache[lag] = lag_std

    features_dict[f'lag_{lag}_mean'] = lag_mean
    features_dict[f'lag_{lag}_std'] = lag_std

# 4.3.3 Difference Features (t - (t-1))
print("  Calculating difference features (t - t-1)...")
prev_data = lag_data_cache[1] # t-1
features_dict['diff_1_values'] = (X_aligned - prev_data).astype(np.float32)
features_dict['diff_1_mean'] = features_dict['product_mean'] - lag_mean_cache[1]
features_dict['diff_1_std'] = features_dict['product_std'] - lag_std_cache[1]
del lag_data_cache, prev_data # Clean up cache

# 4.3.4 Moving Window Features (on product mean)
print("  Calculating moving window features (on product mean)...")
# Calculate product mean over full time first
product_mean_full = np.nanmean(X, axis=2, keepdims=True).astype(np.float32) # Shape: (nday, n_points, 1)
for window in [3, 7, 15]:
    print(f"    Window {window}...")
    window_mean = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)
    window_std = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)
    window_max = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)
    window_min = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)

    # Iterate through each valid day
    for i in range(n_valid_days):
        current_original_idx = i + MAX_LOOKBACK
        # Slice the product mean data for the window period
        window_data_mean_prod = product_mean_full[current_original_idx - window : current_original_idx] # Shape: (window, n_points, 1)

        # Calculate stats over the time dimension (axis=0)
        window_mean[i] = np.nanmean(window_data_mean_prod, axis=0)
        window_std[i] = np.nanstd(window_data_mean_prod, axis=0)
        window_max[i] = np.nanmax(window_data_mean_prod, axis=0)
        window_min[i] = np.nanmin(window_data_mean_prod, axis=0)

    features_dict[f'window_{window}_mean'] = window_mean.astype(np.float32)
    features_dict[f'window_{window}_std'] = window_std.astype(np.float32)
    features_dict[f'window_{window}_max'] = window_max.astype(np.float32)
    features_dict[f'window_{window}_min'] = window_min.astype(np.float32)
    features_dict[f'window_{window}_range'] = (window_max - window_min).astype(np.float32)
del product_mean_full # Clean up

# --- 4.4 Spatial Context Features (3x3 Neighborhood) ---
# NOTE: This calculation is kept to mirror the national script, but results
# will be mostly NaN for scattered points. Flattening handles NaNs later.
# For discrete point data (as now loaded by get_basin_point_data),
# a 3x3 neighborhood is not well-defined without actual coordinates and methods like k-NN.
# Thus, these features are filled with NaNs to maintain structural consistency with gridded datasets.
print("  Calculating spatial context features (3x3 neighborhood)...")
spatial_mean = np.full((n_valid_days, n_points, n_products), np.nan, dtype=np.float32)
spatial_std = np.full((n_valid_days, n_points, n_products), np.nan, dtype=np.float32)
spatial_max = np.full((n_valid_days, n_points, n_products), np.nan, dtype=np.float32)

print("    INFO: Spatial 3x3 features are filled with NaNs for discrete point data, as neighborhood is not directly applicable.")
# The original national code iterated through lat/lon (i, j). Here, the second dimension is 'n_points' (discrete points).
# We cannot directly apply the neighborhood logic. We will fill with NaNs.
# If you need spatial features for points, coordinates and a different method (like KNN) are required.

features_dict['spatial_mean'] = spatial_mean # Filled with NaNs
features_dict['spatial_std'] = spatial_std   # Filled with NaNs
features_dict['spatial_max'] = spatial_max   # Filled with NaNs
# Difference will also be NaN
features_dict['spatial_center_diff'] = (features_dict['raw_values'] - spatial_mean).astype(np.float32)


# --- 4.5 Low Intensity Signal Features ---
print("  Calculating low intensity features...")
# 4.5.1 Threshold Proximity
features_dict['threshold_proximity'] = np.abs(features_dict['product_mean'] - RAIN_THR).astype(np.float32)

# 4.5.2 Coefficient of Variation
cv = safe_divide(features_dict['product_std'], features_dict['product_mean'])
features_dict['coef_of_variation'] = cv.astype(np.float32)

# 4.5.3 Conditional Uncertainty
low_intensity_std = np.where(
    features_dict['product_mean'] < 1.0,
    features_dict['product_std'],
    0.0
).astype(np.float32)
features_dict['low_intensity_std'] = low_intensity_std

# 4.5.4 Intensity Bins (based on product mean)
mean_values = features_dict['product_mean']
intensity_bins = np.zeros((n_valid_days, n_points, 4), dtype=np.float32)
intensity_bins[:, :, 0] = (mean_values <= 0.1).squeeze(axis=-1)
intensity_bins[:, :, 1] = ((mean_values > 0.1) & (mean_values <= 0.5)).squeeze(axis=-1)
intensity_bins[:, :, 2] = ((mean_values > 0.5) & (mean_values <= 1.0)).squeeze(axis=-1)
intensity_bins[:, :, 3] = (mean_values > 1.0).squeeze(axis=-1)
features_dict['intensity_bins'] = intensity_bins # Keep original name

# --- 4.6 Interaction Features ---
print("  Calculating interaction features...")
features_dict['std_season_interaction'] = (features_dict['product_std'] * np.abs(features_dict['sin_day'])).astype(np.float32)
features_dict['low_intense_high_uncertain'] = (low_intensity_std * features_dict['coef_of_variation']).astype(np.float32)
features_dict['rain_count_std_interaction'] = (features_dict['rain_product_count'] * features_dict['product_std']).astype(np.float32)

# Clean up intermediate vars
del cv, low_intensity_std, lag_mean_cache, lag_std_cache

end_feat_time = time.time()
print(f"Feature engineering finished in {end_feat_time - start_feat_time:.2f} seconds.")


# --- 5. Concatenate Features and Generate Names ---
print("\nConcatenating features and generating names...")
start_concat_time = time.time()

# Build feature matrix by concatenating arrays in features_dict
features_list_final = []
feature_names = []

# Use the same naming logic as national turn1.py
for name, feat_array in features_dict.items():
    # Check if feat_array is valid before accessing shape
    if feat_array is None or not isinstance(feat_array, np.ndarray):
        print(f"Warning: Feature '{name}' is None or not a numpy array. Skipping.")
        continue
    try:
        n_cols = feat_array.shape[2] # Axis 2 is the feature dimension for (time, points, features)
        features_list_final.append(feat_array) # Add array to list for concatenation
    except IndexError:
         print(f"Warning: Feature '{name}' has unexpected shape {feat_array.shape}. Skipping.")
         continue


    # Generate names based on dict key and shape
    if n_cols == 1:
        feature_names.append(name)
    # Adjust naming for features that originally had product dimension (now axis=2)
    elif name == 'raw_values' or \
         (name.startswith('lag_') and '_values' in name) or \
         name == 'diff_1_values' or \
         name == 'spatial_mean' or \
         name == 'spatial_std' or \
         name == 'spatial_max' or \
         name == 'spatial_center_diff':
        # Check if n_cols matches n_products
        if n_cols == n_products:
            for i in range(n_cols): feature_names.append(f"{name}_{product_names[i]}")
        else:
            print(f"Warning: Mismatch in columns for {name}. Expected {n_products}, got {n_cols}. Using index.")
            for i in range(n_cols): feature_names.append(f"{name}_{i}")
    elif name == 'season_onehot':
        for i in range(n_cols): feature_names.append(f"{name}_{i}")
    elif name == 'intensity_bins': # Match the name used in national script
        bin_labels = ['<=0.1', '0.1-0.5', '0.5-1.0', '>1.0']
        if n_cols == len(bin_labels):
             for i in range(n_cols): feature_names.append(f"{name}_{bin_labels[i]}")
        else:
            print(f"Warning: Mismatch in columns for {name}. Expected {len(bin_labels)}, got {n_cols}. Using index.")
            for i in range(n_cols): feature_names.append(f"{name}_{i}")
    else: # Fallback for single-column features already handled or other unexpected multi-column
        if n_cols != 1: # Only print warning for unexpected multi-column
             print(f"Warning: Unhandled multi-column feature naming for '{name}'. Using index.")
             for i in range(n_cols): feature_names.append(f"{name}_{i}")
        # else: single column already appended

# Concatenate along the feature axis (axis=2)
if not features_list_final:
    raise ValueError("No features were generated or added to the list for concatenation.")
X_features = np.concatenate(features_list_final, axis=2).astype(np.float32)
del features_list_final, features_dict # Free memory

total_features = X_features.shape[2]
print(f"Concatenated features shape: {X_features.shape}") # (n_valid_days, n_points, n_total_features)
print(f"Total calculated feature columns: {total_features}")
print(f"Length of feature_names list: {len(feature_names)}")
if len(feature_names) != total_features:
     print(f"FATAL: Mismatch between calculated total features ({total_features}) and feature name list length ({len(feature_names)})!")
     exit() # Exit if mismatch

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save feature names
with open(feature_names_filename, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")
print(f"Feature names saved to {feature_names_filename}")
print(f"Concatenation and naming time: {time.time() - start_concat_time:.2f} seconds")

# --- 6. Flatten Data for Model Input ---
print("\nFlattening data...")
start_flat_time = time.time()
# Reshape X: (n_valid_days, n_points, n_total_features) -> (n_valid_days * n_points, n_total_features)
n_samples = n_valid_days * n_points
X_flat = X_features.reshape(n_samples, total_features)
del X_features # Free memory

# Reshape Y: (n_valid_days, n_points) -> (n_valid_days * n_points,)
Y_flat = Y_aligned.reshape(n_samples)
del Y_aligned # Free memory

print(f"Flattened X shape: {X_flat.shape}")
print(f"Flattened Y shape: {Y_flat.shape}")

# Handle potential NaNs introduced during feature calculation (especially spatial)
X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
if np.isnan(Y_flat).any():
    print("Warning: NaNs found in flattened Y target data! This should not happen.")
    # Optional: Add handling like removing rows if necessary

print(f"Flattening time: {time.time() - start_flat_time:.2f} seconds")

# --- 7. Save Data ---
print("\nSaving flattened data...")
print("X_flat shape:", X_flat.shape)
print("Y_flat shape:", Y_flat.shape)
start_save_time = time.time()
np.save(X_flat_filename, X_flat)
np.save(Y_flat_filename, Y_flat)
print(f"Flattened features saved to: {X_flat_filename}")
print(f"Flattened target saved to: {Y_flat_filename}")
print(f"Saving time: {time.time() - start_save_time:.2f} seconds")

print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
print("Data processing complete.")

