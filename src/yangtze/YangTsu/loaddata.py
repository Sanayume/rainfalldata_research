import numpy as np
from scipy.io import loadmat
import os

class mydata:
    def __init__(self):
        # 使用 os.path.join 确保路径分隔符正确，并使用绝对路径
        base_path = "F:/rainfalldata"
        self.DATAFILE ={
            "CMORPH": os.path.join(base_path, "CMORPHdata", "CMORPH_2016_2020.mat"), # (144, 256, 1827)
            "CHIRPS": os.path.join(base_path, "CHIRPSdata", "chirps_2016_2020.mat"), # (144, 256, 1827)
            "SM2RAIN": os.path.join(base_path, "SM2RAINDATA", "sm2rain_2016_2020.mat"), # (144, 256, 1827)
            "IMERG": os.path.join(base_path, "IMERGdata", "IMERG_2016_2020.mat"), # (144, 256, 1827)
            "GSMAP": os.path.join(base_path, "GSMAPdata", "GSMAP_2016_2020.mat"), # (144, 256, 1827)
            "PERSIANN": os.path.join(base_path, "PERSIANNdata", "PERSIANN_2016_2020.mat"), # (144, 256, 1827)
            "CHM": os.path.join(base_path, "CHMdata", "CHM_2016_2020.mat"), # (144, 256, 1827) - Target
            "MASK": os.path.join(base_path, "combined_china_basin_mask.mat"), # (144, 256)
        }
        # 加载并处理 X 数据
        self.X = []
        print("Loading full spatial data for X...")
        for key in ["CMORPH", "CHIRPS", "SM2RAIN", "IMERG", "GSMAP", "PERSIANN"]:
            print(f"  Loading {key}...")
            # 加载 .mat 文件，提取 'data' 字段，应用时间切片，然后转置
            # 原始 shape: (lat, lon, time) -> 时间切片后 -> 转置后 shape: (time, lat, lon)
            data = loadmat(self.DATAFILE[key])['data'][:, :, slice(0, 1827)] # No spatial slice
            self.X.append(np.transpose(data, (2, 0, 1)))
            # 每个元素的 shape 应该是 (1827, 144, 256)
        self.X = np.array(self.X, dtype=np.float32) # Convert list to numpy array (n_products, time, lat, lon)
        print(f"  Finished loading X. Shape: {self.X.shape}") # (6, 1827, 144, 256)

        self.features = ["CMORPH", "CHIRPS", "SM2RAIN", "IMERG", "GSMAP", "PERSIANN"]

        # 加载并处理 Y 数据 (目标)
        print("Loading full spatial data for Y...")
        y_data = loadmat(self.DATAFILE["CHM"])['data'][:, :, slice(0, 1827)] # No spatial slice
        self.Y = np.transpose(y_data, (2, 0, 1)).astype(np.float32) # Shape: (1827, 144, 256)
        # Squeeze into (time, lat, lon)
        self.Y = np.squeeze(self.Y)
        print(f"  Finished loading Y. Shape: {self.Y.shape}")


        # 加载 Mask (保持不变，因为 mask 本身就是全区域的)
        print("Loading mask...")
        # 假设 mask 文件中的变量名是 'data'，如果不是请修改
        mask_data = loadmat(self.DATAFILE["MASK"])
        # 检查 'data' 是否在 .mat 文件中，如果不在，尝试常见的 'mask'
        if 'data' in mask_data:
            self.MASK = mask_data['data'] # Shape: (144, 256)
        elif 'mask' in mask_data:
             self.MASK = mask_data['mask'] # Shape: (144, 256)
        else:
            raise KeyError("Mask variable not found in MASK file. Expected 'data' or 'mask'.")
        print(f"  Finished loading mask. Shape: {self.MASK.shape}")

    def get_x(self):
        # print("type(self.X):", type(self.X), "shape:", self.X.shape)
        return self.X
    def get_y(self):
        # print("type(self.Y):", type(self.Y), "shape:", self.Y.shape)
        return self.Y
    def get_mask(self):
        # print("type(self.MASK):", type(self.MASK), "shape:", self.MASK.shape)
        return self.MASK

    def yangtsu(self):
        """
        Extracts data for the Yangtze River Basin (MASK == 2).

        Returns:
            tuple: (X_yangtsu, Y_yangtsu)
                X_yangtsu (np.ndarray): Feature data for Yangtze points.
                                        Shape: (n_products, time, n_yangtsu_points)
                Y_yangtsu (np.ndarray): Target data for Yangtze points.
                                        Shape: (time, n_yangtsu_points)
        """
        print("Extracting Yangtze River Basin data (MASK == 2)...")
        # Find the indices (lat, lon) where MASK == 2
        yangtsu_indices = np.where(self.MASK == 2)
        if not yangtsu_indices[0].size:
             raise ValueError("No points found for MASK == 2 in the loaded mask.")

        n_yangtsu_points = len(yangtsu_indices[0])
        n_products, n_time, _, _ = self.X.shape
        print(f"  Found {n_yangtsu_points} points in the Yangtze basin.")

        # Extract Y data for these points across all time steps
        # self.Y shape: (time, lat, lon)
        # Y_yangtsu shape should be (time, n_yangtsu_points)
        Y_yangtsu = self.Y[:, yangtsu_indices[0], yangtsu_indices[1]].astype(np.float32)

        # Extract X data for these points across all products and time steps
        # self.X shape: (n_products, time, lat, lon)
        # X_yangtsu shape should be (n_products, time, n_yangtsu_points)
        X_yangtsu = self.X[:, :, yangtsu_indices[0], yangtsu_indices[1]].astype(np.float32)

        print(f"  Finished extracting Yangtze data.")
        print(f"  X_yangtsu shape: {X_yangtsu.shape}") # (6, 1827, n_yangtsu_points)
        print(f"  Y_yangtsu shape: {Y_yangtsu.shape}") # (1827, n_yangtsu_points)

        return X_yangtsu, Y_yangtsu





