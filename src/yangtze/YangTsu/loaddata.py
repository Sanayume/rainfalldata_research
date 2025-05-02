import numpy as np
from scipy.io import loadmat
import os
from typing import Tuple, List, Optional

class mydata:
    def __init__(self, time_slice: slice = slice(0, 1827)):
        """
        Initializes the data loader with file paths and product names.
        Data is loaded lazily upon request.

        Args:
            time_slice: A slice object to select the time dimension upon loading.
                        Defaults to slice(0, 1827).
        """
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "intermediate", "nationwide")
        mask_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "processed", "nationwide", "masks")

        self._DATAFILES = {
            "CMORPH": os.path.join(base_path, "CMORPHdata", "CMORPH_2016_2020.mat"),
            "CHIRPS": os.path.join(base_path, "CHIRPSdata", "chirps_2016_2020.mat"),
            "SM2RAIN": os.path.join(base_path, "SM2RAINDATA", "sm2rain_2016_2020.mat"),
            "IMERG": os.path.join(base_path, "IMERGdata", "IMERG_2016_2020.mat"),
            "GSMAP": os.path.join(base_path, "GSMAPdata", "GSMAP_2016_2020.mat"),
            "PERSIANN": os.path.join(base_path, "PERSIANNdata", "PERSIANN_2016_2020.mat"),
            "CHM": os.path.join(base_path, "CHMdata", "CHM_2016_2020.mat"), # Target
            "MASK": os.path.join(mask_base_path, "combined_china_basin_mask.mat"),
        }
        self.PRODUCTS = ["CMORPH", "CHIRPS", "SM2RAIN", "IMERG", "GSMAP", "PERSIANN"]
        self._time_slice = time_slice

        # Internal cache for loaded data
        self._mask_data: Optional[np.ndarray] = None
        self._X_data: Optional[np.ndarray] = None # Shape: (n_products, time, lat, lon)
        self._Y_data: Optional[np.ndarray] = None # Shape: (time, lat, lon)

    def _load_mask(self) -> np.ndarray:
        """Loads the mask data if not already loaded."""
        if self._mask_data is None:
            print("Loading mask data...")
            mask_path = self._DATAFILES["MASK"]
            if not os.path.exists(mask_path):
                 raise FileNotFoundError(f"Mask file not found: {mask_path}")
            mask_mat = loadmat(mask_path)
            if 'data' in mask_mat:
                self._mask_data = mask_mat['data'].astype(np.int16) # Use int16 for mask
            elif 'mask' in mask_mat:
                self._mask_data = mask_mat['mask'].astype(np.int16)
            else:
                raise KeyError("Mask variable not found in MASK file. Expected 'data' or 'mask'.")
            print(f"  Finished loading mask. Shape: {self._mask_data.shape}")
        return self._mask_data

    def _load_product_data(self) -> np.ndarray:
        """Loads all product data (X) if not already loaded."""
        if self._X_data is None:
            print("Loading all product data (X)...")
            x_list = []
            for key in self.PRODUCTS:
                print(f"  Loading {key}...")
                filepath = self._DATAFILES[key]
                if not os.path.exists(filepath):
                     raise FileNotFoundError(f"Data file not found: {filepath}")
                # Load .mat, extract 'data', apply time slice, transpose (lat, lon, time) -> (time, lat, lon)
                data = loadmat(filepath)['data'][:, :, self._time_slice]
                x_list.append(np.transpose(data, (2, 0, 1)))
            # Stack along the first dimension (products)
            self._X_data = np.array(x_list, dtype=np.float32) # Shape: (n_products, time, lat, lon)
            print(f"  Finished loading X. Shape: {self._X_data.shape}")
        return self._X_data

    def _load_target_data(self) -> np.ndarray:
        """Loads the target data (Y) if not already loaded."""
        if self._Y_data is None:
            print("Loading target data (Y)...")
            filepath = self._DATAFILES["CHM"]
            if not os.path.exists(filepath):
                 raise FileNotFoundError(f"Target file not found: {filepath}")
            y_data = loadmat(filepath)['data'][:, :, self._time_slice]
            # Transpose (lat, lon, time) -> (time, lat, lon) and squeeze if needed
            self._Y_data = np.squeeze(np.transpose(y_data, (2, 0, 1))).astype(np.float32)
            print(f"  Finished loading Y. Shape: {self._Y_data.shape}")
        return self._Y_data

    def get_mask(self) -> np.ndarray:
        """Returns the loaded mask data."""
        return self._load_mask()

    def get_products(self) -> List[str]:
        """Returns the list of product names."""
        return self.PRODUCTS

    def get_basin_spatial_data(self, basin_mask_value: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads national data (if needed) and returns the spatial data
        masked for a specific basin. Data outside the basin will be NaN.

        Args:
            basin_mask_value: The integer value representing the basin in the mask file.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X_spatial (np.ndarray): Product data masked spatially.
                                          Shape: (n_products, time, lat, lon)
                - Y_spatial (np.ndarray): Target data masked spatially.
                                          Shape: (time, lat, lon)
        """
        print(f"Extracting spatial data for basin mask value {basin_mask_value}...")
        mask = self._load_mask()
        X = self._load_product_data()
        Y = self._load_target_data()

        # Create boolean masks matching the data dimensions
        mask_bool = (mask == basin_mask_value) # (lat, lon)
        if not np.any(mask_bool):
             raise ValueError(f"No points found for mask value {basin_mask_value} in the loaded mask.")

        # Ensure Y has 3 dimensions (time, lat, lon) before broadcasting
        if len(Y.shape) == 2: # Should not happen if loaded correctly, but check
             raise ValueError("Y data has unexpected shape after loading.")

        # Broadcast mask to match Y dimensions (time, lat, lon)
        mask_3d = np.broadcast_to(mask_bool, Y.shape)
        # Broadcast mask to match X dimensions (n_products, time, lat, lon)
        mask_4d = np.broadcast_to(mask_bool, X.shape)

        # Apply mask using np.where, keeping NaNs outside the basin
        X_spatial = np.where(mask_4d, X, np.nan).astype(np.float32)
        Y_spatial = np.where(mask_3d, Y, np.nan).astype(np.float32)

        print("  Finished extracting spatial basin data.")
        print(f"  X_spatial shape: {X_spatial.shape}")
        print(f"  Y_spatial shape: {Y_spatial.shape}")
        return X_spatial, Y_spatial

    def get_basin_point_data(self, basin_mask_value: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads national data (if needed) and returns the data for points
        within a specific basin, flattening the spatial dimensions.

        Args:
            basin_mask_value: The integer value representing the basin in the mask file.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X_points (np.ndarray): Product data for basin points.
                                         Shape: (n_products, time, n_basin_points)
                - Y_points (np.ndarray): Target data for basin points.
                                         Shape: (time, n_basin_points)
        """
        print(f"Extracting point data for basin mask value {basin_mask_value}...")
        mask = self._load_mask()
        X = self._load_product_data()
        Y = self._load_target_data()

        # Find indices (lat, lon) for the basin
        basin_indices = np.where(mask == basin_mask_value)
        if not basin_indices[0].size:
             raise ValueError(f"No points found for mask value {basin_mask_value} in the loaded mask.")

        n_basin_points = len(basin_indices[0])
        print(f"  Found {n_basin_points} points in the basin.")

        # Extract Y data for these points: (time, lat, lon) -> (time, n_points)
        Y_points = Y[:, basin_indices[0], basin_indices[1]].astype(np.float32)

        # Extract X data for these points: (n_products, time, lat, lon) -> (n_products, time, n_points)
        X_points = X[:, :, basin_indices[0], basin_indices[1]].astype(np.float32)

        print("  Finished extracting point basin data.")
        print(f"  X_points shape: {X_points.shape}")
        print(f"  Y_points shape: {Y_points.shape}")
        return X_points, Y_points

    def yangtsu(self):
        """
        DEPRECATED: Use get_basin_spatial_data(2) or get_basin_point_data(2) instead.
        Kept for backward compatibility reference, but raises a warning.
        """
        import warnings
        warnings.warn("The 'yangtsu' method is deprecated. Use get_basin_spatial_data(2) for spatial masked data or get_basin_point_data(2) for point data.", DeprecationWarning, stacklevel=2)
        # Mimic old behavior by returning both spatial and point data (less efficient)
        X_points, Y_points = self.get_basin_point_data(basin_mask_value=2)
        X_spatial, Y_spatial = self.get_basin_spatial_data(basin_mask_value=2)
        # The original returned X_points, Y_points, X_spatial, Y_spatial
        return X_points, Y_points, X_spatial, Y_spatial





