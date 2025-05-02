from loaddata import mydata # Assuming loaddata.py is in the same directory or accessible
import numpy as np
import os
import time

# --- 1. Data Loading and Initial Preparation ---
print("--- Step 1: Data Loading (Yangtze) ---")
start_load = time.time()
ALL_DATA = mydata()
# Load Yangtze data
X_raw, Y_raw, _, _ = ALL_DATA.yangtsu() # Shape: (prod, time, points), (time, points)
product_names = ALL_DATA.features
n_products, nday, n_points = X_raw.shape
print(f"Initial X_raw shape: {X_raw.shape}")
print(f"Initial Y_raw shape: {Y_raw.shape}")
print(f"Product names: {product_names}")
end_load = time.time()
print(f"Data loading finished in {end_load - start_load:.2f} seconds.")

# --- 2. Data Integration & Reshaping ---
print("\n--- Step 2: Reshaping ---")
# Transpose X to (time, n_points, n_products)
X = np.transpose(X_raw, (1, 2, 0)).astype(np.float32)
del X_raw # Free memory
Y = Y_raw.astype(np.float32) # Ensure correct dtype
del Y_raw # Free memory
print(f"Transposed X shape: {X.shape}") # (1827, n_points, 6)
print(f"Y shape: {Y.shape}")   # (1827, n_points)

# --- 3. Handle Time Dependency ---
print("\n--- Step 3: Time Alignment ---")
max_lookback = 30 # Keep 30
# nday, n_points, n_products defined earlier

valid_time_range = slice(max_lookback, nday)
n_valid_days = nday - max_lookback

# No need for valid_mask calculation based on Y NaNs for Yangtze data
print(f"Max lookback: {max_lookback} days")
print(f"Valid time range: Day index {max_lookback} to {nday-1}")

# Align X and Y
X_aligned = X[valid_time_range] # Shape: (n_valid_days, n_points, n_products)
Y_aligned = Y[valid_time_range] # Shape: (n_valid_days, n_points)
print(f"Aligned X shape: {X_aligned.shape}")
print(f"Aligned Y shape: {Y_aligned.shape}")

# --- 4. Feature Engineering (Simplified - V3 - Adapted for Points) ---
print("\n--- Step 4: Feature Engineering (Yangtze V3 - Simplified) ---")
start_feat = time.time()
features_dict = {}
epsilon = 1e-6
RAIN_THR = 0.1 # Threshold for counting rain products

# Helper function for safe division
def safe_divide(numerator, denominator, default=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / (denominator + epsilon)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

# --- 4.1 Basic Features ---
print("  Calculating basic features...")
features_dict['raw_values'] = X_aligned

# --- 4.2 Multi-product Consistency/Difference Features (Simplified) ---
print("  Calculating simplified multi-product stats (current time)...")
# Keep only rain product count
X_rain = (np.nan_to_num(X_aligned, nan=0.0) > RAIN_THR)
features_dict['rain_product_count'] = np.sum(X_rain, axis=2, keepdims=True).astype(np.float32)
del X_rain

# --- 4.3 Temporal Evolution Features (Simplified) ---
# 4.3.1 Periodicity (Simplified)
print("  Calculating periodicity features (Season only)...")
days_in_year = 365.25
day_index_original = np.arange(nday, dtype=np.float32)
day_of_year = day_index_original % days_in_year
# Keep Season One-Hot
month = (day_of_year // 30.4375).astype(int) % 12 + 1
season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
season = np.array([season_map[m] for m in month])
seasons_onehot = np.zeros((nday, 4), dtype=np.float32)
seasons_onehot[np.arange(nday), season] = 1
# Slice and expand: (n_valid_days, 4) -> (n_valid_days, 1, 4) -> broadcast to (n_valid_days, n_points, 4)
season_aligned = seasons_onehot[valid_time_range, np.newaxis, :] * np.ones((1, n_points, 1), dtype=np.float32)
features_dict['season_onehot'] = season_aligned
del day_index_original, day_of_year, month, season, seasons_onehot, season_aligned # Clean up

# 4.3.2 Lag Features (Simplified)
print("  Calculating simplified lag features...")
lag_data_cache = {} # Cache raw lag data
lag_std_cache = {} # Cache lag stds
lag_rain_count_cache = {} # Cache lag rain counts

for lag in [1, 2, 3]:
    print(f"    Lag {lag}...")
    lag_slice = slice(max_lookback - lag, nday - lag)
    lag_data = X[lag_slice] # Shape: (n_valid_days, n_points, n_products)
    lag_data_cache[lag] = lag_data
    features_dict[f'lag_{lag}_values'] = lag_data # Raw values

    # Calculate and cache lag statistics (Std and Rain Count only)
    lag_std = np.nanstd(lag_data, axis=2, keepdims=True).astype(np.float32)
    lag_rain = (np.nan_to_num(lag_data, nan=0.0) > RAIN_THR)
    lag_rain_count = np.sum(lag_rain, axis=2, keepdims=True).astype(np.float32)

    lag_std_cache[lag] = lag_std
    lag_rain_count_cache[lag] = lag_rain_count

    features_dict[f'lag_{lag}_std'] = lag_std # Std
    features_dict[f'lag_{lag}_rain_count'] = lag_rain_count # Lag rain product count

# 4.3.3 Difference Features (Simplified)
print("  Calculating difference features (Values only)...")
prev_data = lag_data_cache[1] # t-1 (already sliced)
features_dict['diff_1_values'] = (X_aligned - prev_data).astype(np.float32)
# --- Delete raw lag data cache ---
print("    Deleting raw lag data cache...")
del lag_data_cache, prev_data

# 4.3.4 Moving Window Features (Keep as in v2, based on product mean)
print("  Calculating moving window features (on product mean)...")
# Calculate product mean over full time first
product_mean_full = np.nanmean(X, axis=2, keepdims=True).astype(np.float32) # Shape: (nday, n_points, 1)
for window in [3, 7, 15]:
    print(f"    Window {window} (on product mean)...")
    window_mean = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)
    window_std = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)
    window_max = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)
    window_min = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)
    for i in range(n_valid_days):
        current_original_idx = i + max_lookback
        window_data_mean_prod = product_mean_full[current_original_idx - window : current_original_idx]
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

# Per-product moving window stats (Keep from v2)
window = 7
print(f"    Window {window} (per product - GSMAP, PERSIANN)...")
gsmap_idx = product_names.index("GSMAP")
persiann_idx = product_names.index("PERSIANN")
window_mean_gsmap = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)
window_std_persiann = np.zeros((n_valid_days, n_points, 1), dtype=np.float32)
for i in range(n_valid_days):
    current_original_idx = i + max_lookback
    window_data_X = X[current_original_idx - window : current_original_idx] # Shape: (window, n_points, n_products)
    mean_gsmap_result = np.nanmean(window_data_X[:, :, gsmap_idx], axis=0)
    std_persiann_result = np.nanstd(window_data_X[:, :, persiann_idx], axis=0)
    window_mean_gsmap[i] = mean_gsmap_result[..., np.newaxis]
    window_std_persiann[i] = std_persiann_result[..., np.newaxis]
features_dict[f'window_{window}_mean_GSMAP'] = window_mean_gsmap.astype(np.float32)
features_dict[f'window_{window}_std_PERSIANN'] = window_std_persiann.astype(np.float32)
del window_mean_gsmap, window_std_persiann, mean_gsmap_result, std_persiann_result # Clean up

# --- 4.4 Spatial Context Features (Simplified - 5x5 and Gradients only - Adapted for Points) ---
print("  Calculating simplified spatial features (5x5 and Gradients - Adapted for Points)...")
# 4.4.1 3x3 Neighborhood -> REMOVED

# 4.4.2 5x5 Neighborhood (NaN Placeholder)
print("    5x5 neighborhood (NaN placeholder)...")
spatial_mean_5x5 = np.full((n_valid_days, n_points, n_products), np.nan, dtype=np.float32)
spatial_std_5x5 = np.full((n_valid_days, n_points, n_products), np.nan, dtype=np.float32)
spatial_max_5x5 = np.full((n_valid_days, n_points, n_products), np.nan, dtype=np.float32)
print("    WARNING: Spatial 5x5 calculation assumes grid structure and will produce NaNs for point data.")
features_dict['spatial_mean_5x5'] = spatial_mean_5x5
features_dict['spatial_std_5x5'] = spatial_std_5x5
features_dict['spatial_max_5x5'] = spatial_max_5x5
# Keep spatial_center_diff_5x5 (will be NaN)
features_dict['spatial_center_diff_5x5'] = (features_dict['raw_values'] - spatial_mean_5x5).astype(np.float32)
del spatial_mean_5x5 # Can delete mean after diff calculation

# 4.4.3 Spatial Gradients (NaN Placeholder)
print("    Spatial gradients (NaN placeholder)...")
print("    WARNING: Spatial gradient calculation assumes grid structure and will produce NaNs for point data.")
features_dict['gradient_magnitude_GSMAP'] = np.full((n_valid_days, n_points, 1), np.nan, dtype=np.float32)
features_dict['gradient_magnitude_PERSIANN'] = np.full((n_valid_days, n_points, 1), np.nan, dtype=np.float32)
features_dict['gradient_magnitude_mean'] = np.full((n_valid_days, n_points, 1), np.nan, dtype=np.float32)

# --- Delete X_aligned after spatial calculations ---
print("    Deleting X_aligned...")
del X_aligned # Delete alias

# --- 4.5 Low Intensity Signal Features (Simplified) ---
print("  Calculating simplified low intensity features...")
# 4.5.1 Threshold Proximity -> REMOVED
# 4.5.2 Coefficient of Variation -> REMOVED
# --- Delete lag stats caches ---
print("    Deleting lag stats caches...")
del lag_std_cache, lag_rain_count_cache
# 4.5.3 Conditional Uncertainty -> REMOVED
# 4.5.4 Intensity Bins (Keep count-based bins from v2)
count_values = features_dict['rain_product_count']
intensity_bins_count = np.zeros((n_valid_days, n_points, 4), dtype=np.float32) # 4 bins: 0, 1, 2, 3+
intensity_bins_count[:, :, 0] = (count_values == 0).squeeze(axis=-1)
intensity_bins_count[:, :, 1] = (count_values == 1).squeeze(axis=-1)
intensity_bins_count[:, :, 2] = (count_values == 2).squeeze(axis=-1)
intensity_bins_count[:, :, 3] = (count_values >= 3).squeeze(axis=-1)
features_dict['intensity_bins_count'] = intensity_bins_count
del count_values, intensity_bins_count # Clean up

# --- 4.6 Interaction Features -> REMOVED ---
print("  Skipping interaction features...")

# --- Delete remaining intermediate vars before flattening ---
# Most intermediate vars removed or deleted earlier
# Delete the large spatial diff array now that it's used (or NaN)
if 'spatial_center_diff_5x5' in features_dict:
     print("    Deleting spatial_center_diff_5x5...")
     del features_dict['spatial_center_diff_5x5']

end_feat = time.time()
print(f"Feature engineering finished in {end_feat - start_feat:.2f} seconds.")


# --- 5. Concatenate Features and Generate Names ---
print("\n--- Step 5: Concatenating Features & Naming (Yangtze v3) ---")
start_concat = time.time()
# Build feature matrix by concatenating arrays in features_dict
features_list_final = []
feature_names = []

# Use the same naming logic as national turn3.py
for name, feat_array in features_dict.items():
    if feat_array is None or not isinstance(feat_array, np.ndarray): continue
    try:
        n_cols = feat_array.shape[2] # Axis 2 is feature dimension
        features_list_final.append(feat_array)
    except IndexError:
         print(f"Warning: Feature '{name}' has unexpected shape {feat_array.shape}. Skipping.")
         continue

    base_name = name
    if n_cols == 1:
        feature_names.append(base_name)
    # Adjust naming for features that originally had product dimension
    elif base_name == 'raw_values' or \
         (base_name.startswith('lag_') and '_values' in base_name) or \
         base_name == 'diff_1_values' or \
         base_name.startswith('spatial_std_5x5') or \
         base_name.startswith('spatial_max_5x5'): # Mean was deleted, diff kept separate name
        if n_cols == n_products:
            for i in range(n_cols): feature_names.append(f"{base_name}_{product_names[i]}")
        else:
            print(f"Warning: Mismatch in columns for {base_name}. Expected {n_products}, got {n_cols}. Using index.")
            for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    elif base_name == 'season_onehot':
        for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    elif base_name == 'intensity_bins_count':
        bin_labels = ['0', '1', '2', '>=3']
        if n_cols == len(bin_labels):
             for i in range(n_cols): feature_names.append(f"{base_name}_{bin_labels[i]}")
        else:
             print(f"Warning: Mismatch in columns for {base_name}. Expected {len(bin_labels)}, got {n_cols}. Using index.")
             for i in range(n_cols): feature_names.append(f"{base_name}_{i}")
    else: # Fallback
        if n_cols != 1:
             print(f"Warning: Unhandled multi-column feature naming for '{base_name}'. Using index.")
             for i in range(n_cols): feature_names.append(f"{base_name}_{i}")

# Concatenate along the feature axis (axis=2)
if not features_list_final:
    raise ValueError("No features were generated or added to the list for concatenation.")
X_features = np.concatenate(features_list_final, axis=2).astype(np.float32)
del features_list_final, features_dict # Free memory

total_features = X_features.shape[2]
print(f"Concatenated features shape: {X_features.shape}")
print(f"Total calculated feature columns: {total_features}")
print(f"Length of feature_names list: {len(feature_names)}")
if len(feature_names) != total_features:
     print(f"FATAL: Mismatch between calculated total features ({total_features}) and feature name list length ({len(feature_names)})!")
     exit()

# Define output directory and filenames explicitly for Yangtze v3
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
X_flat_filename = os.path.join(OUTPUT_DIR, "X_Yangtsu_flat_features_v3.npy") # Add _v3
Y_flat_filename = os.path.join(OUTPUT_DIR, "Y_Yangtsu_flat_target_v3.npy") # Add _v3
feature_names_filename = os.path.join(OUTPUT_DIR, "feature_names_yangtsu_v3.txt") # Add _v3

# Save feature names
print(f"Saving feature names to {feature_names_filename}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(feature_names_filename, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")
end_concat = time.time()
print(f"Concatenation and naming finished in {end_concat - start_concat:.2f} seconds.")

# --- 6. Flatten Data for Model Input ---
print("\n--- Step 6: Flattening Data (Yangtze v3) ---")
start_flat = time.time()
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

end_flat = time.time()
print(f"Flattening finished in {end_flat - start_flat:.2f} seconds.")

# --- 7. Save Data ---
print("\n--- Step 7: Saving Flattened Data (Yangtze v3) ---")
start_save = time.time()
np.save(X_flat_filename, X_flat)
np.save(Y_flat_filename, Y_flat)
print(f"Flattened features saved to: {X_flat_filename}")
print(f"Flattened target saved to: {Y_flat_filename}")
end_save = time.time()
print(f"Saving finished in {end_save - start_save:.2f} seconds.")

print(f"\nTotal processing time: {time.time() - start_load:.2f} seconds")
print("Data processing complete.")
