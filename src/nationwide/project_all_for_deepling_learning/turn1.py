from loaddata import mydata
import numpy as np
import os # Import os

# --- 1. Data Loading and Initial Preparation ---
# Assumes mydata() loads pre-masked X and Y data where invalid points are NaN
ALL_DATA = mydata()
X = np.array(ALL_DATA.X) # Shape: (6, 1827, 144, 256) - 6 products, time, lat, lon
Y = np.array(ALL_DATA.Y) # Shape: (1, 1827, 144, 256) - 1 observation, time, lat, lon
# features list from ALL_DATA might be useful for product names if needed
# product_names = ALL_DATA.features # Example if available

print("Initial X shape:", X.shape)
print("Initial Y shape:", Y.shape)

# --- 2. Data Integration & Reshaping ---
# Transpose X to (time, lat, lon, product) for easier feature engineering per time-space point
X = np.transpose(X, (1, 2, 3, 0)).astype(np.float32)
# Squeeze Y is no longer needed as loaddata.py already provides shape (time, lat, lon)
Y = Y.astype(np.float32) # Ensure correct dtype

print("Transposed X shape:", X.shape) # (1827, 144, 256, 6)
print("Squeezed Y shape:", Y.shape)   # Should remain (1827, 144, 256)

# --- 3. Apply Mask (Implicit via NaN) & Handle Time Dependency ---
# Determine max_lookback based on features requiring past data (lags, moving windows)
max_lookback = 30 # Based on longest window/lag used (e.g., 15-day window, lag 3)
nday, nlat, nlon, nproduct = X.shape

# Define the valid time range after truncation
valid_time_range = slice(max_lookback, nday)
n_valid_days = nday - max_lookback # Number of days included in the final dataset

# Derive the mask directly from Y (ground truth) for the valid time range
# Assumes NaN in Y indicates points to be excluded (outside region or invalid)
valid_mask_full = ~np.isnan(Y) # Mask for all days
valid_mask = valid_mask_full[valid_time_range] # Mask aligned with the truncated feature/target data

n_valid_samples = np.sum(valid_mask)
print(f"Max lookback: {max_lookback} days")
print(f"Valid time range: Day index {max_lookback} to {nday-1}")
print(f"Shape of valid mask (aligned): {valid_mask.shape}") # (n_valid_days, nlat, nlon)
print(f"Total valid samples for training/evaluation: {n_valid_samples}")

# Align Y target data with the valid time range
Y_aligned = Y[valid_time_range]
print(f"Aligned Y shape: {Y_aligned.shape}") # (n_valid_days, nlat, nlon)

# --- 4. Feature Engineering ---
# Dictionary to store feature arrays (all aligned to valid_time_range)
features_dict = {}
epsilon = 1e-6 # Small constant to prevent division by zero

# --- 4.1 Basic Features ---
# Raw precipitation values for the 6 products
features_dict['raw_values'] = X[valid_time_range] # Shape: (n_valid_days, nlat, nlon, 6)

# --- 4.2 Multi-product Consistency/Difference Features ---
# Calculate stats across the product dimension (axis=3)
# Use np.nan* functions to handle potential NaNs within product data
X_valid_time = X[valid_time_range] # Use data already sliced for valid time
features_dict['product_mean'] = np.nanmean(X_valid_time, axis=3, keepdims=True).astype(np.float32)
features_dict['product_std'] = np.nanstd(X_valid_time, axis=3, keepdims=True).astype(np.float32)
features_dict['product_median'] = np.nanmedian(X_valid_time, axis=3, keepdims=True).astype(np.float32)
product_max = np.nanmax(X_valid_time, axis=3, keepdims=True).astype(np.float32)
product_min = np.nanmin(X_valid_time, axis=3, keepdims=True).astype(np.float32)
features_dict['product_max'] = product_max
features_dict['product_min'] = product_min
features_dict['product_range'] = (product_max - product_min).astype(np.float32)

# Count of products predicting rain (> 0.1 mm threshold)
# Handle NaNs by treating them as non-rain before counting
X_rain = (np.nan_to_num(X_valid_time, nan=0.0) > 0.1)
features_dict['rain_product_count'] = np.sum(X_rain, axis=3, keepdims=True).astype(np.float32)

# --- 4.3 Temporal Evolution Features ---
# 4.3.1 Periodicity
days_in_year = 365.25
# Day index relative to the start of the original data (before truncation)
day_index_original = np.arange(nday, dtype=np.float32)
day_of_year = day_index_original % days_in_year

sin_time = np.sin(2 * np.pi * day_of_year / days_in_year).astype(np.float32)
cos_time = np.cos(2 * np.pi * day_of_year / days_in_year).astype(np.float32)

# Expand and truncate to match feature dimensions
sin_time_expanded = np.reshape(sin_time, (nday, 1, 1, 1)) * np.ones((1, nlat, nlon, 1), dtype=np.float32)
cos_time_expanded = np.reshape(cos_time, (nday, 1, 1, 1)) * np.ones((1, nlat, nlon, 1), dtype=np.float32)
features_dict['sin_day'] = sin_time_expanded[valid_time_range]
features_dict['cos_day'] = cos_time_expanded[valid_time_range]

# Optional: Season One-Hot Encoding
month = (day_of_year // 30.4375).astype(int) % 12 + 1 # Approximate month (1-12)
season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0} # 0=Win, 1=Spr, 2=Sum, 3=Aut
season = np.array([season_map[m] for m in month])
seasons_onehot = np.zeros((nday, 4), dtype=np.float32)
seasons_onehot[np.arange(nday), season] = 1
season_expanded = np.reshape(seasons_onehot, (nday, 1, 1, 4)) * np.ones((1, nlat, nlon, 1), dtype=np.float32)
features_dict['season_onehot'] = season_expanded[valid_time_range] # Shape: (n_valid_days, nlat, nlon, 4)

# 4.3.2 Lag Features (t-1, t-2, t-3)
for lag in [1, 2, 3]:
    # Ensure indices stay within bounds [0, nday)
    lag_slice = slice(max_lookback - lag, nday - lag)
    lag_data = X[lag_slice]
    features_dict[f'lag_{lag}_values'] = lag_data # Raw values, Shape: (n_valid_days, nlat, nlon, 6)
    # Consistency metrics for lagged data
    features_dict[f'lag_{lag}_mean'] = np.nanmean(lag_data, axis=3, keepdims=True).astype(np.float32)
    features_dict[f'lag_{lag}_std'] = np.nanstd(lag_data, axis=3, keepdims=True).astype(np.float32)

# 4.3.3 Difference Features (t - (t-1))
current_data = X[valid_time_range] # t
prev_data = X[max_lookback-1 : nday-1] # t-1
features_dict['diff_1_values'] = (current_data - prev_data).astype(np.float32) # Raw diff, Shape: (n_valid_days, nlat, nlon, 6)

# Difference for consistency metrics
current_mean = features_dict['product_mean'] # Already calculated for time t
prev_mean = np.nanmean(prev_data, axis=3, keepdims=True).astype(np.float32)
features_dict['diff_1_mean'] = current_mean - prev_mean

current_std = features_dict['product_std'] # Already calculated for time t
prev_std = np.nanstd(prev_data, axis=3, keepdims=True).astype(np.float32)
features_dict['diff_1_std'] = current_std - prev_std

# 4.3.4 Moving Window Features (on product mean)
# Calculate window stats based on the mean across products for simplicity and efficiency
for window in [3, 7, 15]:
    # Pre-allocate arrays for the results
    window_mean = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    window_std = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    window_max = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)
    window_min = np.zeros((n_valid_days, nlat, nlon, 1), dtype=np.float32)

    # Iterate through each valid day to calculate window features ending on that day
    for i in range(n_valid_days):
        # Index in the original X array corresponding to the current valid day
        current_original_idx = i + max_lookback
        # Slice the original X data for the window period [current_original_idx - window, current_original_idx)
        window_data_X = X[current_original_idx - window : current_original_idx] # Shape: (window, nlat, nlon, nproduct)

        # Calculate mean across products first for each day in the window
        window_data_mean_prod = np.nanmean(window_data_X, axis=3, keepdims=True) # Shape: (window, nlat, nlon, 1)

        # Calculate stats over the time dimension (axis=0) of the product means
        window_mean[i] = np.nanmean(window_data_mean_prod, axis=0).astype(np.float32)
        window_std[i] = np.nanstd(window_data_mean_prod, axis=0).astype(np.float32)
        window_max[i] = np.nanmax(window_data_mean_prod, axis=0).astype(np.float32)
        window_min[i] = np.nanmin(window_data_mean_prod, axis=0).astype(np.float32)

    features_dict[f'window_{window}_mean'] = window_mean
    features_dict[f'window_{window}_std'] = window_std
    features_dict[f'window_{window}_max'] = window_max
    features_dict[f'window_{window}_min'] = window_min
    features_dict[f'window_{window}_range'] = (window_max - window_min).astype(np.float32)

# --- 4.4 Spatial Context Features (3x3 Neighborhood) ---
# Calculate spatial stats per product
spatial_mean = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_std = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)
spatial_max = np.full((n_valid_days, nlat, nlon, nproduct), np.nan, dtype=np.float32)

# Iterate through spatial dimensions (excluding borders where 3x3 is not possible)
# Use the X data already sliced for the valid time range
X_valid_time = X[valid_time_range]
for i in range(1, nlat - 1):
    for j in range(1, nlon - 1):
        # Extract 3x3 neighborhood for all valid days and all products
        neighborhood = X_valid_time[:, i-1:i+2, j-1:j+2, :] # Shape: (n_valid_days, 3, 3, nproduct)
        # Calculate stats over the spatial axes (1, 2)
        spatial_mean[:, i, j, :] = np.nanmean(neighborhood, axis=(1, 2)).astype(np.float32)
        spatial_std[:, i, j, :] = np.nanstd(neighborhood, axis=(1, 2)).astype(np.float32)
        spatial_max[:, i, j, :] = np.nanmax(neighborhood, axis=(1, 2)).astype(np.float32)

# Handle borders (e.g., copy neighbor values or leave as NaN - leaving NaN is safer)
# Note: Flattening later will only take non-NaN Y points, implicitly handling spatial NaNs if Y is NaN there.

features_dict['spatial_mean'] = spatial_mean # Shape: (n_valid_days, nlat, nlon, 6)
features_dict['spatial_std'] = spatial_std   # Shape: (n_valid_days, nlat, nlon, 6)
features_dict['spatial_max'] = spatial_max   # Shape: (n_valid_days, nlat, nlon, 6)
# Difference between center point raw value and spatial mean for that product
features_dict['spatial_center_diff'] = (features_dict['raw_values'] - spatial_mean).astype(np.float32)

# --- 4.5 Low Intensity Signal Features ---
# 4.5.1 Threshold Proximity (based on product mean)
features_dict['threshold_proximity'] = np.abs(features_dict['product_mean'] - 0.1).astype(np.float32)

# 4.5.2 Coefficient of Variation (std / mean)
# Use product_mean and product_std calculated earlier
cv = (features_dict['product_std'] / (features_dict['product_mean'] + epsilon)).astype(np.float32)
# Handle cases where mean is near zero -> CV can be large or infinite/NaN
cv = np.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0) # Replace problematic values with 0
features_dict['coef_of_variation'] = cv

# 4.5.3 Conditional Uncertainty (std if mean < 1.0)
low_intensity_std = np.where(
    features_dict['product_mean'] < 1.0,
    features_dict['product_std'],
    0.0
).astype(np.float32)
features_dict['low_intensity_std'] = low_intensity_std

# 4.5.4 Intensity Bins (based on product mean) - Original 4 bins
mean_values = features_dict['product_mean'] # Shape: (n_valid_days, nlat, nlon, 1)
intensity_bins = np.zeros((n_valid_days, nlat, nlon, 4), dtype=np.float32) # Original 4 bins
# Bin 0: mean <= 0.1
intensity_bins[:, :, :, 0] = (mean_values <= 0.1).squeeze(axis=-1).astype(np.float32)
# Bin 1: 0.1 < mean <= 0.5
intensity_bins[:, :, :, 1] = ((mean_values > 0.1) & (mean_values <= 0.5)).squeeze(axis=-1).astype(np.float32)
# Bin 2: 0.5 < mean <= 1.0
intensity_bins[:, :, :, 2] = ((mean_values > 0.5) & (mean_values <= 1.0)).squeeze(axis=-1).astype(np.float32)
# Bin 3: mean > 1.0
intensity_bins[:, :, :, 3] = (mean_values > 1.0).squeeze(axis=-1).astype(np.float32)
features_dict['intensity_bins'] = intensity_bins # Add original bins back

# --- 4.6 Interaction Features (Optional but potentially useful) ---
# Example: Product standard deviation * seasonality (using |sin_day|)
features_dict['std_season_interaction'] = (features_dict['product_std'] * np.abs(features_dict['sin_day'])).astype(np.float32)
# Example: Low intensity * high uncertainty (using conditional std * CV)
features_dict['low_intense_high_uncertain'] = (low_intensity_std * features_dict['coef_of_variation']).astype(np.float32)
# Example: Rain count * product std dev
features_dict['rain_count_std_interaction'] = (features_dict['rain_product_count'] * features_dict['product_std']).astype(np.float32)

# --- 5. Flattening Data for Model Input ---
# Calculate total number of features
total_features = sum(feat.shape[3] for feat in features_dict.values())
print(f"\nTotal number of calculated feature columns: {total_features}")

# Prepare feature names list
feature_names = []
# Get product names from loaddata if possible, otherwise use indices
try:
    product_names = ALL_DATA.features # ["CMORPH", "CHIRPS", ...]
    if len(product_names) != nproduct:
        print("Warning: Length of product_names from loaddata does not match nproduct. Using indices.")
        product_names = [str(i) for i in range(nproduct)]
except AttributeError:
    print("Warning: ALL_DATA object does not have 'features' attribute. Using indices for product names.")
    product_names = [str(i) for i in range(nproduct)]

for name, feat in features_dict.items():
    n_cols = feat.shape[3]
    if n_cols == 1:
        feature_names.append(name)
    elif name.endswith('_values') or name.startswith('spatial_') or name.startswith('lag_') and '_values' in name:
         # Append product name/index for multi-column features related to products
        for i in range(n_cols):
            feature_names.append(f"{name}_{product_names[i]}") # Use product name
    elif name == 'season_onehot':
         for i in range(n_cols):
             feature_names.append(f"{name}_{i}") # Keep index for one-hot
    elif name == 'intensity_bins':
         bin_labels = ['<=0.1', '0.1-0.5', '0.5-1.0', '>1.0']
         for i in range(n_cols):
             feature_names.append(f"{name}_{bin_labels[i]}")
    else:
        # Fallback for other multi-column features (should ideally be handled above)
        for i in range(n_cols):
            feature_names.append(f"{name}_{i}")

print(f"Length of feature_names list: {len(feature_names)}")
if len(feature_names) != total_features:
     print(f"Warning: Mismatch between calculated total features ({total_features}) and feature name list length ({len(feature_names)})!")

# Define output directory and filenames explicitly
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "nationwide", "features")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
X_flat_filename = os.path.join(OUTPUT_DIR, "X_flat_features.npy")
Y_flat_filename = os.path.join(OUTPUT_DIR, "Y_flat_target.npy")
feature_names_filename = os.path.join(OUTPUT_DIR, "feature_names.txt") # Ensure this is also consistent

# Save feature names
with open(feature_names_filename, "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")
print(f"Feature names saved to {feature_names_filename}")

# Use memory-mapped files for potentially large flattened data
# Ensure shape uses standard Python integers
X_flat_shape = (int(n_valid_samples), int(total_features)) # Convert to int
Y_flat_shape = (int(n_valid_samples),) # Convert to int

print(f"Creating memory-mapped file: {X_flat_filename} with shape {X_flat_shape}")
# Ensure the directory exists before creating the file
os.makedirs(os.path.dirname(X_flat_filename), exist_ok=True)
X_flat_mmap = np.lib.format.open_memmap(X_flat_filename, mode='w+', dtype=np.float32, shape=X_flat_shape)

print(f"Creating memory-mapped file: {Y_flat_filename} with shape {Y_flat_shape}")
os.makedirs(os.path.dirname(Y_flat_filename), exist_ok=True)
Y_flat_mmap = np.lib.format.open_memmap(Y_flat_filename, mode='w+', dtype=np.float32, shape=Y_flat_shape)

print("\nStarting flattening process (day by day)...")
current_mmap_idx = 0 # Index for writing into the memory-mapped files

# Iterate through each valid day
for t in range(n_valid_days):
    # Get the mask for the current day (lat, lon)
    day_mask = valid_mask[t] # Shape: (nlat, nlon)
    # Find the (lat, lon) indices where the mask is True for this day
    day_valid_indices = np.where(day_mask) # Tuple of (lat_indices, lon_indices)

    n_day_valid = len(day_valid_indices[0])
    if n_day_valid == 0:
        continue # Skip days with no valid samples

    # Allocate temporary array for this day's flattened features
    day_X_flat = np.zeros((n_day_valid, total_features), dtype=np.float32)

    # Extract and concatenate features for the valid points of the current day
    col_idx = 0
    for name, feat_array in features_dict.items():
        n_cols = feat_array.shape[3]
        # Extract data for the current time step 't' and valid spatial indices
        # feat_array[t] has shape (nlat, nlon, n_cols)
        # day_valid_indices selects the valid points -> shape (n_day_valid, n_cols)
        feat_day_valid = feat_array[t][day_valid_indices]
        # Ensure it has the correct shape if n_cols=1 (might become 1D)
        if n_cols == 1 and feat_day_valid.ndim == 1:
             feat_day_valid = feat_day_valid[:, np.newaxis]

        # Handle potential NaNs introduced by spatial features at borders/masked areas
        # If a feature value is NaN for a point where Y is valid, replace it (e.g., with 0 or mean)
        # Using 0 for simplicity here, consider imputation if needed.
        feat_day_valid = np.nan_to_num(feat_day_valid, nan=0.0)

        day_X_flat[:, col_idx : col_idx + n_cols] = feat_day_valid
        col_idx += n_cols

    # Extract target values for the valid points of the current day
    day_Y_flat = Y_aligned[t][day_valid_indices] # Shape: (n_day_valid,)

    # Write this day's flattened data to the memory-mapped files
    write_start_idx = current_mmap_idx
    write_end_idx = current_mmap_idx + n_day_valid

    # Boundary check (should not happen if n_valid_samples was calculated correctly)
    if write_end_idx > X_flat_shape[0]: # Use the integer shape
        print(f"Warning: Exceeding expected number of valid samples at day {t}. Adjusting.")
        write_end_idx = X_flat_shape[0]
        n_day_valid_adjusted = X_flat_shape[0] - write_start_idx
        if n_day_valid_adjusted <= 0:
            print(f"  Skipping day {t+max_lookback} due to boundary adjustment.")
            continue
        day_X_flat = day_X_flat[:n_day_valid_adjusted]
        day_Y_flat = day_Y_flat[:n_day_valid_adjusted]
        # Update n_day_valid for correct index update
        n_day_valid = n_day_valid_adjusted

    X_flat_mmap[write_start_idx:write_end_idx] = day_X_flat
    Y_flat_mmap[write_start_idx:write_end_idx] = day_Y_flat

    current_mmap_idx = write_end_idx # Update index based on actual written count

    # Print progress periodically
    if (t + 1) % 50 == 0 or t == n_valid_days - 1:
        progress_percent = (t + 1) / n_valid_days * 100
        # Use integer shape for total samples in print statement
        print(f"  Processed day {t+max_lookback}/{nday-1}. Samples written: {current_mmap_idx}/{X_flat_shape[0]} ({progress_percent:.1f}%)")

# Ensure memory-mapped files are flushed and closed properly
X_flat_mmap.flush()
Y_flat_mmap.flush()
del X_flat_mmap
del Y_flat_mmap

print("\nFlattening process complete.")
# Final check on sample count
if current_mmap_idx != X_flat_shape[0]: # Use integer shape for check
     print(f"Warning: Final written sample count ({current_mmap_idx}) does not match calculated count ({X_flat_shape[0]}). Check mask/flattening logic.")
else:
    print(f"Successfully wrote {current_mmap_idx} samples.")

print(f"Flattened features saved to: {X_flat_filename}")
print(f"Flattened target saved to: {Y_flat_filename}")
print("Data processing complete.")


