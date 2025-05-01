from loaddata import mydata # Assuming loaddata.py is in the same directory or accessible
import numpy as np
import os
import time

# --- 1. Data Loading and Initial Preparation ---
print("--- Step 1: Data Loading ---")
start_load = time.time()
ALL_DATA = mydata()
X = np.array(ALL_DATA.X) # Shape: (6, 1827, 144, 256)
Y = np.array(ALL_DATA.Y) # Shape: (1827, 144, 256) - Already squeezed in loaddata
product_names = ALL_DATA.features # ["CMORPH", "CHIRPS", ...]
print(f"Initial X shape: {X.shape}")
print(f"Initial Y shape: {Y.shape}")
print(f"Product names: {product_names}")
end_load = time.time()
print(f"Data loading finished in {end_load - start_load:.2f} seconds.")

# --- 2. Data Integration & Reshaping ---
print("\n--- Step 2: Reshaping ---")
X = np.transpose(X, (1, 2, 3, 0)).astype(np.float32) # (time, lat, lon, product)
Y = Y.astype(np.float32) # Ensure correct dtype
print(f"Transposed X shape: {X.shape}") # (1827, 144, 256, 6)
print(f"Y shape: {Y.shape}")   # (1827, 144, 256)

# --- 3. Apply Mask & Handle Time Dependency ---
print("\n--- Step 3: Masking & Time Alignment ---")
# Determine max_lookback based on features (window=15, lag=3, spatial 5x5 needs border=2)
max_lookback = 30 # Keep 30, sufficient for window=15, lag=3
nday, nlat, nlon, nproduct = X.shape

valid_time_range = slice(max_lookback, nday)
n_valid_days = nday - max_lookback

valid_mask_full = ~np.isnan(Y)
valid_mask = valid_mask_full[valid_time_range]

n_valid_samples = np.sum(valid_mask)
print(f"Max lookback: {max_lookback} days")
print(f"Valid time range: Day index {max_lookback} to {nday-1}")
print(f"Shape of valid mask (aligned): {valid_mask.shape}")
print(f"Total valid samples for training/evaluation: {n_valid_samples}")

Y_aligned = Y[valid_time_range]
print(f"Aligned Y shape: {Y_aligned.shape}")

# --- 4. Feature Engineering (Enhanced) ---
print("\n--- Step 4: Feature Engineering ---")
start_feat = time.time()
features_dict = {}
epsilon = 1e-6
RAIN_THR = 0.1 # Threshold for counting rain products

# Helper function for safe division
def safe_divide(numerator, denominator, default=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / (denominator + epsilon)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

# --- 4.1 Basic Features (from turn1) ---
print("  Calculating basic features...")
X_valid_time = X[valid_time_range]
features_dict['raw_values'] = X_valid_time

# --- 4.2 Multi-product Consistency/Difference Features (from turn1) ---
print("  Calculating multi-product stats (current time)...")
features_dict['product_mean'] = np.nanmean(X_valid_time, axis=3, keepdims=True).astype(np.float32)
features_dict['product_std'] = np.nanstd(X_valid_time, axis=3, keepdims=True).astype(np.float32)
features_dict['product_median'] = np.nanmedian(X_valid_time, axis=3, keepdims=True).astype(np.float32)
product_max = np.nanmax(X_valid_time, axis=3, keepdims=True).astype(np.float32)
product_min = np.nanmin(X_valid_time, axis=3, keepdims=True).astype(np.float32)
features_dict['product_max'] = product_max
features_dict['product_min'] = product_min
features_dict['product_range'] = (product_max - product_min).astype(np.float32)
X_rain = (np.nan_to_num(X_valid_time, nan=0.0) > RAIN_THR)
features_dict['rain_product_count'] = np.sum(X_rain, axis=3, keepdims=True).astype(np.float32)

# --- 4.3 Temporal Evolution Features ---
# 4.3.1 Periodicity (from turn1)
print("  Calculating periodicity features...")
days_in_year = 365.25
day_index_original = np.arange(nday, dtype=np.float32)
day_of_year = day_index_original % days_in_year
sin_time = np.sin(2 * np.pi * day_of_year / days_in_year).astype(np.float32)
cos_time = np.cos(2 * np.pi * day_of_year / days_in_year).astype(np.float32)
sin_time_expanded = np.reshape(sin_time, (nday, 1, 1, 1)) * np.ones((1, nlat, nlon, 1), dtype=np.float32)
cos_time_expanded = np.reshape(cos_time, (nday, 1, 1, 1)) * np.ones((1, nlat, nlon, 1), dtype=np.float32)
features_dict['sin_day'] = sin_time_expanded[valid_time_range]
features_dict['cos_day'] = cos_time_expanded[valid_time_range]
# Season One-Hot (Optional, keep from turn1)
month = (day_of_year // 30.4375).astype(int) % 12 + 1
season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
season = np.array([season_map[m] for m in month])
seasons_onehot = np.zeros((nday, 4), dtype=np.float32)
seasons_onehot[np.arange(nday), season] = 1
season_expanded = np.reshape(seasons_onehot, (nday, 1, 1, 4)) * np.ones((1, nlat, nlon, 1), dtype=np.float32)
features_dict['season_onehot'] = season_expanded[valid_time_range]

# 4.3.2 Lag Features (Deepened)
print("  Calculating deepened lag features...")
lag_data_cache = {} # Cache raw lag data to avoid re-slicing
lag_mean_cache = {} # Cache lag means
lag_std_cache = {} # Cache lag stds
lag_rain_count_cache = {} # Cache lag rain counts

for lag in [1, 2, 3]:
    print(f"    Lag {lag}...")
    lag_slice = slice(max_lookback - lag, nday - lag)
    lag_data = X[lag_slice]
    lag_data_cache[lag] = lag_data
    features_dict[f'lag_{lag}_values'] = lag_data # Raw values (from turn1)

    # Calculate and cache lag statistics
    lag_mean = np.nanmean(lag_data, axis=3, keepdims=True).astype(np.float32)
    lag_std = np.nanstd(lag_data, axis=3, keepdims=True).astype(np.float32)
    lag_rain = (np.nan_to_num(lag_data, nan=0.0) > RAIN_THR)
    lag_rain_count = np.sum(lag_rain, axis=3, keepdims=True).astype(np.float32)

    lag_mean_cache[lag] = lag_mean
    lag_std_cache[lag] = lag_std
    lag_rain_count_cache[lag] = lag_rain_count

    features_dict[f'lag_{lag}_mean'] = lag_mean # Mean (from turn1)
    features_dict[f'lag_{lag}_std'] = lag_std # Std (from turn1)
    features_dict[f'lag_{lag}_rain_count'] = lag_rain_count # NEW: Lag rain product count

# NEW: Lag differences (mean)
if 1 in lag_mean_cache and 2 in lag_mean_cache:
    features_dict['lag_1_2_mean_diff'] = (lag_mean_cache[1] - lag_mean_cache[2]).astype(np.float32)
if 2 in lag_mean_cache and 3 in lag_mean_cache:
    features_dict['lag_2_3_mean_diff'] = (lag_mean_cache[2] - lag_mean_cache[3]).astype(np.float32)

# 4.3.3 Difference Features (t - (t-1)) (from turn1)
print("  Calculating difference features (t - t-1)...")
current_data = X_valid_time # t
prev_data = lag_data_cache[1] # t-1 (already sliced)
features_dict['diff_1_values'] = (current_data - prev_data).astype(np.float32)
features_dict['diff_1_mean'] = features_dict['product_mean'] - lag_mean_cache[1]
features_dict['diff_1_std'] = features_dict['product_std'] - lag_std_cache[1]
# --- Delete raw lag data cache ---
print("    Deleting raw lag data cache...")
del lag_data_cache, prev_data

# 4.3.4 Moving Window Features (Deepened)
print("  Calculating deepened moving window features...")
# Keep window features on product mean (from turn1)
for window in [3, 7, 15]:
    print(f"    Window {window} (on product mean)...")
    window_mean = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    window_std = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    window_max = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    window_min = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    for i in range(n_valid_days):
        current_original_idx = i + max_lookback
        window_data_X = X[current_original_idx - window : current_original_idx]
        window_data_mean_prod = np.nanmean(window_data_X, axis=3, keepdims=True)
        window_mean[i] = np.nanmean(window_data_mean_prod, axis=0)
        window_std[i] = np.nanstd(window_data_mean_prod, axis=0)
        window_max[i] = np.nanmax(window_data_mean_prod, axis=0)
        window_min[i] = np.nanmin(window_data_mean_prod, axis=0)
    features_dict[f'window_{window}_mean'] = window_mean.astype(np.float32)
    features_dict[f'window_{window}_std'] = window_std.astype(np.float32)
    features_dict[f'window_{window}_max'] = window_max.astype(np.float32)
    features_dict[f'window_{window}_min'] = window_min.astype(np.float32)
    features_dict[f'window_{window}_range'] = (window_max - window_min).astype(np.float32)

# NEW: Per-product moving window stats (example for window=7, GSMAP, PERSIANN)
window = 7
print(f"    Window {window} (per product - GSMAP, PERSIANN)...")
gsmap_idx = product_names.index("GSMAP")
persiann_idx = product_names.index("PERSIANN")
window_mean_gsmap = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
window_std_persiann = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
for i in range(n_valid_days):
    current_original_idx = i + max_lookback
    window_data_X = X[current_original_idx - window : current_original_idx]
    # Calculate mean over time axis (axis=0), result shape (nlat, nlon)
    mean_gsmap_result = np.nanmean(window_data_X[:, :, :, gsmap_idx], axis=0)
    # Calculate std over time axis (axis=0), result shape (nlat, nlon)
    std_persiann_result = np.nanstd(window_data_X[:, :, :, persiann_idx], axis=0)
    # Assign to target array, adding the feature dimension
    window_mean_gsmap[i] = mean_gsmap_result[..., np.newaxis]
    window_std_persiann[i] = std_persiann_result[..., np.newaxis]
features_dict[f'window_{window}_mean_GSMAP'] = window_mean_gsmap.astype(np.float32)
features_dict[f'window_{window}_std_PERSIANN'] = window_std_persiann.astype(np.float32)


# --- 4.4 Spatial Context Features (Deepened) ---
print("  Calculating deepened spatial features...")
# 4.4.1 3x3 Neighborhood (from turn1)
spatial_mean_3x3 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_std_3x3 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_max_3x3 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
print("    3x3 neighborhood...")
for i in range(1, nlat - 1):
    for j in range(1, nlon - 1):
        neighborhood = X_valid_time[:, i-1:i+2, j-1:j+2, :]
        spatial_mean_3x3[:, i, j, :] = np.nanmean(neighborhood, axis=(1, 2))
        spatial_std_3x3[:, i, j, :] = np.nanstd(neighborhood, axis=(1, 2))
        spatial_max_3x3[:, i, j, :] = np.nanmax(neighborhood, axis=(1, 2))
features_dict['spatial_mean_3x3'] = spatial_mean_3x3.astype(np.float32)
features_dict['spatial_std_3x3'] = spatial_std_3x3.astype(np.float32)
features_dict['spatial_max_3x3'] = spatial_max_3x3.astype(np.float32)
features_dict['spatial_center_diff_3x3'] = (features_dict['raw_values'] - spatial_mean_3x3).astype(np.float32)

# 4.4.2 NEW: 5x5 Neighborhood
spatial_mean_5x5 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_std_5x5 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_max_5x5 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
print("    5x5 neighborhood...")
for i in range(2, nlat - 2): # Need border of 2
    for j in range(2, nlon - 2):
        neighborhood = X_valid_time[:, i-2:i+3, j-2:j+3, :]
        spatial_mean_5x5[:, i, j, :] = np.nanmean(neighborhood, axis=(1, 2))
        spatial_std_5x5[:, i, j, :] = np.nanstd(neighborhood, axis=(1, 2))
        spatial_max_5x5[:, i, j, :] = np.nanmax(neighborhood, axis=(1, 2))
features_dict['spatial_mean_5x5'] = spatial_mean_5x5.astype(np.float32)
features_dict['spatial_std_5x5'] = spatial_std_5x5.astype(np.float32)
features_dict['spatial_max_5x5'] = spatial_max_5x5.astype(np.float32)
features_dict['spatial_center_diff_5x5'] = (features_dict['raw_values'] - spatial_mean_5x5).astype(np.float32)

# 4.4.3 NEW: Spatial Gradients (Example for GSMAP, PERSIANN, product_mean)
print("    Spatial gradients...")
grad_lat_gsmap, grad_lon_gsmap = np.gradient(X_valid_time[:, :, :, gsmap_idx], axis=(1, 2))
features_dict['gradient_magnitude_GSMAP'] = np.sqrt(grad_lat_gsmap**2 + grad_lon_gsmap**2).astype(np.float32)[..., np.newaxis]

grad_lat_persiann, grad_lon_persiann = np.gradient(X_valid_time[:, :, :, persiann_idx], axis=(1, 2))
features_dict['gradient_magnitude_PERSIANN'] = np.sqrt(grad_lat_persiann**2 + grad_lon_persiann**2).astype(np.float32)[..., np.newaxis]

# Gradient on product mean (squeeze needed as gradient input must be 3D)
grad_lat_mean, grad_lon_mean = np.gradient(np.squeeze(features_dict['product_mean'], axis=-1), axis=(1, 2))
features_dict['gradient_magnitude_mean'] = np.sqrt(grad_lat_mean**2 + grad_lon_mean**2).astype(np.float32)[..., np.newaxis]

# --- Delete X_valid_time after spatial calculations ---
print("    Deleting X_valid_time...")
del X_valid_time, current_data # Also delete alias

# --- 4.5 Low Intensity Signal Features (Enhanced) ---
print("  Calculating enhanced low intensity features...")
# 4.5.1 Threshold Proximity (from turn1)
features_dict['threshold_proximity'] = np.abs(features_dict['product_mean'] - RAIN_THR).astype(np.float32)

# 4.5.2 Coefficient of Variation (from turn1) & Lagged version
cv = safe_divide(features_dict['product_std'], features_dict['product_mean'])
features_dict['coef_of_variation'] = cv.astype(np.float32)
# NEW: Lagged CV
cv_lag1 = safe_divide(lag_std_cache[1], lag_mean_cache[1])
features_dict['lag_1_coef_of_variation'] = cv_lag1.astype(np.float32)
# --- Delete lag stats caches ---
print("    Deleting lag stats caches...")
del lag_mean_cache, lag_std_cache, lag_rain_count_cache, cv_lag1

# 4.5.3 Conditional Uncertainty (from turn1)
low_intensity_std = np.where(
    features_dict['product_mean'] < 1.0,
    features_dict['product_std'],
    0.0
).astype(np.float32)
features_dict['low_intensity_std'] = low_intensity_std

# 4.5.4 Intensity Bins (based on product mean - keep original 4 bins from turn1)
mean_values = features_dict['product_mean']
intensity_bins = np.zeros((n_valid_days, nlat, nlon, 4), dtype=np.float32)
intensity_bins[:, :, :, 0] = (mean_values <= 0.1).squeeze(axis=-1)
intensity_bins[:, :, :, 1] = ((mean_values > 0.1) & (mean_values <= 0.5)).squeeze(axis=-1)
intensity_bins[:, :, :, 2] = ((mean_values > 0.5) & (mean_values <= 1.0)).squeeze(axis=-1)
intensity_bins[:, :, :, 3] = (mean_values > 1.0).squeeze(axis=-1)
features_dict['intensity_bins_mean'] = intensity_bins # Rename slightly

# NEW: Intensity Bins based on Rain Product Count
count_values = features_dict['rain_product_count']
intensity_bins_count = np.zeros((n_valid_days, nlat, nlon, 4), dtype=np.float32) # 4 bins: 0, 1, 2, 3+
intensity_bins_count[:, :, :, 0] = (count_values == 0).squeeze(axis=-1)
intensity_bins_count[:, :, :, 1] = (count_values == 1).squeeze(axis=-1)
intensity_bins_count[:, :, :, 2] = (count_values == 2).squeeze(axis=-1)
intensity_bins_count[:, :, :, 3] = (count_values >= 3).squeeze(axis=-1)
features_dict['intensity_bins_count'] = intensity_bins_count

# --- 4.6 Interaction Features (Enhanced) ---
print("  Calculating enhanced interaction features...")
# Keep interactions from turn1
features_dict['std_season_interaction'] = (features_dict['product_std'] * np.abs(features_dict['sin_day'])).astype(np.float32)
features_dict['low_intense_high_uncertain'] = (low_intensity_std * features_dict['coef_of_variation']).astype(np.float32) # cv still exists here
features_dict['rain_count_std_interaction'] = (features_dict['rain_product_count'] * features_dict['product_std']).astype(np.float32)

# NEW Interactions - OPTIMIZED
# Use 3x3 spatial diff for interaction
print("    Calculating std_x_spatial_diff interaction (optimized)...")
# --- Manual nanmean implementation along axis=3 ---
input_array = features_dict['spatial_center_diff_3x3']
# Calculate sum, ignoring NaNs
sum_val = np.nansum(input_array, axis=3, keepdims=True, dtype=np.float32)
# Calculate count of non-NaNs using int32
mask = np.isnan(input_array)
count_val = np.sum(~mask, axis=3, keepdims=True, dtype=np.int32)
del mask # Free mask memory
# Calculate mean using safe_divide
mean_spatial_center_diff_3x3 = safe_divide(sum_val, count_val)
del sum_val, count_val # Free intermediate sum/count memory
# --- End manual nanmean ---

# Calculate interaction directly using absolute value of the mean diff
features_dict['std_x_spatial_diff'] = (features_dict['product_std'] * np.abs(mean_spatial_center_diff_3x3)).astype(np.float32)
del mean_spatial_center_diff_3x3 # Delete intermediate mean

print("    Calculating std_x_diff_1_mean interaction...")
features_dict['std_x_diff_1_mean'] = (features_dict['product_std'] * np.abs(features_dict['diff_1_mean'])).astype(np.float32)

# --- Delete remaining intermediate vars before flattening ---
print("    Deleting remaining intermediate feature variables...")
del cv, low_intensity_std
# Delete the large spatial diff array now that it's used
if 'spatial_center_diff_3x3' in features_dict:
    print("    Deleting spatial_center_diff_3x3...")
    del features_dict['spatial_center_diff_3x3']
if 'spatial_center_diff_5x5' in features_dict: # Also delete the 5x5 one if no longer needed
     print("    Deleting spatial_center_diff_5x5...")
     del features_dict['spatial_center_diff_5x5']


end_feat = time.time()
print(f"Feature engineering finished in {end_feat - start_feat:.2f} seconds.")


# --- 5. Flattening Data for Model Input ---
print("\n--- Step 5: Flattening Data ---")
start_flat = time.time()
# Calculate total number of features
total_features = sum(feat.shape[3] for feat in features_dict.values())
print(f"Total number of calculated feature columns: {total_features}")

# Prepare feature names list
feature_names = []
# Use product names loaded earlier
for name, feat in features_dict.items():
    n_cols = feat.shape[3]
    base_name = name # Keep original dict key name

    # Adjust naming based on feature type
    if n_cols == 1:
        feature_names.append(base_name)
    elif base_name == 'raw_values' or base_name.startswith('lag_') and '_values' in base_name or \
         base_name.startswith('diff_1_values') or base_name.startswith('spatial_') and '_3x3' in base_name or \
         base_name.startswith('spatial_') and '_5x5' in base_name:
         # Append product name/index for multi-column features related to products
        for i in range(n_cols):
            feature_names.append(f"{base_name}_{product_names[i]}")
    elif base_name == 'season_onehot':
         for i in range(n_cols):
             feature_names.append(f"{base_name}_{i}")
    elif base_name == 'intensity_bins_mean':
         bin_labels = ['<=0.1', '0.1-0.5', '0.5-1.0', '>1.0']
         for i in range(n_cols):
             feature_names.append(f"{base_name}_{bin_labels[i]}")
    elif base_name == 'intensity_bins_count':
         bin_labels = ['0', '1', '2', '>=3']
         for i in range(n_cols):
             feature_names.append(f"{base_name}_{bin_labels[i]}")
    else:
        # Fallback for other multi-column features (should ideally be handled above)
        print(f"Warning: Unhandled multi-column feature naming for '{base_name}'. Using index.")
        for i in range(n_cols):
            feature_names.append(f"{base_name}_{i}")

print(f"Length of feature_names list: {len(feature_names)}")
if len(feature_names) != total_features:
     print(f"FATAL: Mismatch between calculated total features ({total_features}) and feature name list length ({len(feature_names)})!")
     exit() # Exit if mismatch

# Define output directory and filenames explicitly for v2
OUTPUT_DIR = "F:/rainfalldata"
X_flat_filename = os.path.join(OUTPUT_DIR, "X_flat_features_v2.npy")
Y_flat_filename = os.path.join(OUTPUT_DIR, "Y_flat_target_v2.npy")
feature_names_filename = os.path.join(OUTPUT_DIR, "feature_names_v2.txt")

# Save feature names
print(f"Saving feature names to {feature_names_filename}")
with open(feature_names_filename, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")

# Use memory-mapped files
X_flat_shape = (int(n_valid_samples), int(total_features))
Y_flat_shape = (int(n_valid_samples),)

print(f"Creating memory-mapped file: {X_flat_filename} with shape {X_flat_shape}")
os.makedirs(os.path.dirname(X_flat_filename), exist_ok=True)
X_flat_mmap = np.lib.format.open_memmap(X_flat_filename, mode='w+', dtype=np.float32, shape=X_flat_shape)

print(f"Creating memory-mapped file: {Y_flat_filename} with shape {Y_flat_shape}")
os.makedirs(os.path.dirname(Y_flat_filename), exist_ok=True)
Y_flat_mmap = np.lib.format.open_memmap(Y_flat_filename, mode='w+', dtype=np.float32, shape=Y_flat_shape)

print("Starting flattening process (day by day)...")
current_mmap_idx = 0

# Iterate through each valid day
for t in range(n_valid_days):
    day_mask = valid_mask[t]
    day_valid_indices = np.where(day_mask)
    n_day_valid = len(day_valid_indices[0])
    if n_day_valid == 0: continue

    day_X_flat = np.zeros((n_day_valid, total_features), dtype=np.float32)
    col_idx = 0
    for name, feat_array in features_dict.items():
        n_cols = feat_array.shape[3]
        feat_day_valid = feat_array[t][day_valid_indices]
        if n_cols == 1 and feat_day_valid.ndim == 1:
             feat_day_valid = feat_day_valid[:, np.newaxis]
        # Handle NaNs introduced by spatial features at borders/masked areas
        # Replace NaN with 0 for simplicity. Consider imputation if needed.
        feat_day_valid = np.nan_to_num(feat_day_valid, nan=0.0, posinf=0.0, neginf=0.0)
        day_X_flat[:, col_idx : col_idx + n_cols] = feat_day_valid
        col_idx += n_cols

    day_Y_flat = Y_aligned[t][day_valid_indices]

    write_start_idx = current_mmap_idx
    write_end_idx = current_mmap_idx + n_day_valid

    if write_end_idx > X_flat_shape[0]:
        print(f"Warning: Exceeding expected number of valid samples at day {t}. Adjusting.")
        write_end_idx = X_flat_shape[0]
        n_day_valid_adjusted = X_flat_shape[0] - write_start_idx
        if n_day_valid_adjusted <= 0: continue
        day_X_flat = day_X_flat[:n_day_valid_adjusted]
        day_Y_flat = day_Y_flat[:n_day_valid_adjusted]
        n_day_valid = n_day_valid_adjusted

    X_flat_mmap[write_start_idx:write_end_idx] = day_X_flat
    Y_flat_mmap[write_start_idx:write_end_idx] = day_Y_flat
    current_mmap_idx = write_end_idx

    if (t + 1) % 50 == 0 or t == n_valid_days - 1:
        progress_percent = (t + 1) / n_valid_days * 100
        print(f"  Processed day {t+max_lookback}/{nday-1}. Samples written: {current_mmap_idx}/{X_flat_shape[0]} ({progress_percent:.1f}%)")

X_flat_mmap.flush()
Y_flat_mmap.flush()
del X_flat_mmap
del Y_flat_mmap

end_flat = time.time()
print(f"\nFlattening process complete in {end_flat - start_flat:.2f} seconds.")
if current_mmap_idx != X_flat_shape[0]:
     print(f"Warning: Final written sample count ({current_mmap_idx}) does not match calculated count ({X_flat_shape[0]}).")
else:
    print(f"Successfully wrote {current_mmap_idx} samples.")

print(f"Flattened features saved to: {X_flat_filename}")
print(f"Flattened target saved to: {Y_flat_filename}")
print("Data processing complete.")

