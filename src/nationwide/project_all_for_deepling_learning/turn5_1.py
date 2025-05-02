from loaddata import mydata # Assuming loaddata.py is in the same directory or accessible
import numpy as np
import os
import time
from scipy.ndimage import generic_filter # For median filter

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
max_lookback = 30 # Keep 30, sufficient for window=15, lag=3, spatial 5x5 needs border=2
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

# --- 4. Feature Engineering ---
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

# --- 4.1 Basic Features ---
print("  Calculating basic features...")
X_valid_time = X[valid_time_range]
features_dict['raw_values'] = X_valid_time

# --- 4.2 Multi-product Consistency/Difference Features ---
print("  Calculating multi-product stats (current time)...")
X_rain = (np.nan_to_num(X_valid_time, nan=0.0) > RAIN_THR)
features_dict['rain_product_count'] = np.sum(X_rain, axis=3, keepdims=True).astype(np.float32)
del X_rain

# --- 4.3 Temporal Evolution Features ---
print("  Calculating periodicity features (Season only)...")
days_in_year = 365.25
day_index_original = np.arange(nday, dtype=np.float32)
day_of_year = day_index_original % days_in_year
month = (day_of_year // 30.4375).astype(int) % 12 + 1
season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
season = np.array([season_map[m] for m in month])
seasons_onehot = np.zeros((nday, 4), dtype=np.float32)
seasons_onehot[np.arange(nday), season] = 1
season_expanded = np.reshape(seasons_onehot, (nday, 1, 1, 4)) * np.ones((1, nlat, nlon, 1), dtype=np.float32)
features_dict['season_onehot'] = season_expanded[valid_time_range]
del day_index_original, day_of_year, month, season, seasons_onehot, season_expanded

print("  Calculating lag features...")
lag_data_cache = {}
for lag in [1, 2, 3]:
    print(f"    Lag {lag}...")
    lag_slice = slice(max_lookback - lag, nday - lag)
    lag_data = X[lag_slice]
    lag_data_cache[lag] = lag_data
    features_dict[f'lag_{lag}_values'] = lag_data

print("  Calculating difference features...")
current_data = X_valid_time
prev_data = lag_data_cache[1]
features_dict['diff_1_values'] = (current_data - prev_data).astype(np.float32)
del lag_data_cache, prev_data

print("  Calculating moving window features...")
product_mean_local = np.nanmean(X_valid_time, axis=3, keepdims=True).astype(np.float32)
for window in [3, 7, 15]:
    print(f"    Window {window}...")
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
del product_mean_local

print("  Calculating spatial features...")
spatial_mean_5x5 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_std_5x5 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_max_5x5 = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
for i in range(2, nlat - 2):
    for j in range(2, nlon - 2):
        neighborhood = X_valid_time[:, i-2:i+3, j-2:j+3, :]
        spatial_mean_5x5[:, i, j, :] = np.nanmean(neighborhood, axis=(1, 2))
        spatial_std_5x5[:, i, j, :] = np.nanstd(neighborhood, axis=(1, 2))
        spatial_max_5x5[:, i, j, :] = np.nanmax(neighborhood, axis=(1, 2))
features_dict['spatial_mean_5x5'] = spatial_mean_5x5.astype(np.float32)
features_dict['spatial_std_5x5'] = spatial_std_5x5.astype(np.float32)
features_dict['spatial_max_5x5'] = spatial_max_5x5.astype(np.float32)
del spatial_mean_5x5

end_feat = time.time()
print(f"Feature engineering finished in {end_feat - start_feat:.2f} seconds.")

# --- 5. Flattening Data for Model Input ---
print("\n--- Step 5: Flattening Data ---")
start_flat = time.time()
total_features = sum(feat.shape[3] for feat in features_dict.values())
print(f"Total number of calculated feature columns: {total_features}")

feature_names = []
for name, feat in features_dict.items():
    n_cols = feat.shape[3]
    base_name = name
    if n_cols == 1:
        feature_names.append(base_name)
    elif base_name == 'raw_values' or base_name.startswith('lag_') or base_name.startswith('diff_1_values') or base_name.startswith('spatial_'):
        for i in range(n_cols):
            feature_names.append(f"{base_name}_{product_names[i]}")
    elif base_name == 'season_onehot':
        for i in range(n_cols):
            feature_names.append(f"{base_name}_{i}")
    else:
        for i in range(n_cols):
            feature_names.append(f"{base_name}_{i}")

print(f"Length of feature_names list: {len(feature_names)}")
if len(feature_names) != total_features:
    print(f"FATAL: Mismatch between calculated total features ({total_features}) and feature name list length ({len(feature_names)})!")
    exit()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "nationwide", "features")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
X_flat_filename = os.path.join(OUTPUT_DIR, "X_flat_features_v5_1.npy")
Y_flat_filename = os.path.join(OUTPUT_DIR, "Y_flat_target_v5_1.npy")
feature_names_filename = os.path.join(OUTPUT_DIR, "feature_names_v5_1.txt")

print(f"Saving feature names to {feature_names_filename}")
with open(feature_names_filename, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")

X_flat_shape = (int(n_valid_samples), int(total_features))
Y_flat_shape = (int(n_valid_samples),)

print(f"Creating memory-mapped file: {X_flat_filename} with shape {X_flat_shape}")
os.makedirs(os.path.dirname(X_flat_filename), exist_ok=True)
X_flat_mmap = np.lib.format.open_memmap(X_flat_filename, mode='w+', dtype=np.float32, shape=X_flat_shape)

print(f"Creating memory-mapped file: {Y_flat_filename} with shape {Y_flat_shape}")
os.makedirs(os.path.dirname(Y_flat_filename), exist_ok=True)
Y_flat_mmap = np.lib.format.open_memmap(Y_flat_filename, mode='w+', dtype=np.float32, shape=Y_flat_shape)

print("Starting flattening process...")
current_mmap_idx = 0
for t in range(n_valid_days):
    day_mask = valid_mask[t]
    day_valid_indices = np.where(day_mask)
    n_day_valid = len(day_valid_indices[0])
    if n_day_valid == 0: continue

    day_X_flat = np.zeros((n_day_valid, total_features), dtype=np.float32)
    col_idx = 0
    for name, feat in features_dict.items():
        n_cols = feat.shape[3]
        feat_day_valid = feat[t][day_valid_indices]
        if n_cols == 1 and feat_day_valid.ndim == 1:
            feat_day_valid = feat_day_valid[:, np.newaxis]
        feat_day_valid = np.nan_to_num(feat_day_valid, nan=0.0, posinf=0.0, neginf=0.0)
        day_X_flat[:, col_idx : col_idx + n_cols] = feat_day_valid
        col_idx += n_cols

    day_Y_flat = Y_aligned[t][day_valid_indices]

    write_start_idx = current_mmap_idx
    write_end_idx = current_mmap_idx + n_day_valid

    if write_end_idx > X_flat_shape[0]:
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
del features_dict

end_flat = time.time()
print(f"\nFlattening process complete in {end_flat - start_flat:.2f} seconds.")
if current_mmap_idx != X_flat_shape[0]:
    print(f"Warning: Final written sample count ({current_mmap_idx}) does not match calculated count ({X_flat_shape[0]}).")
else:
    print(f"Successfully wrote {current_mmap_idx} samples.")

print(f"Flattened features saved to: {X_flat_filename}")
print(f"Flattened target saved to: {Y_flat_filename}")
print("Data processing complete.")